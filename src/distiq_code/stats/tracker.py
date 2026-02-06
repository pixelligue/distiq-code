"""Statistics tracking for optimization metrics."""

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel


class RequestStats(BaseModel):
    """Statistics for a single request."""

    timestamp: float
    model: str

    # Token usage
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int

    # Cache
    cache_hit: bool
    cache_similarity: float = 0.0

    # Latency
    latency_ms: float

    # Compression
    compression_enabled: bool
    compression_ratio: float = 1.0


class AggregateStats(BaseModel):
    """Aggregated statistics."""

    total_requests: int
    cache_hits: int
    cache_hit_rate: float

    total_tokens_original: int
    total_tokens_compressed: int
    total_tokens_saved: int
    average_compression_ratio: float

    average_latency_ms: float


class StatsTracker:
    """
    Track optimization statistics.

    Metrics:
    - Token savings (compression)
    - Cache hit rate
    - Latency
    - Cost savings (estimated)
    """

    def __init__(self, stats_file: Path):
        """
        Initialize stats tracker.

        Args:
            stats_file: JSON file to store stats
        """
        self.stats_file = stats_file
        self.requests: list[RequestStats] = []

        # Load existing stats
        self._load_stats()

    def record_request(
        self,
        model: str,
        original_tokens: int,
        compressed_tokens: int,
        cache_hit: bool,
        cache_similarity: float,
        latency_ms: float,
        compression_enabled: bool,
        compression_ratio: float,
    ) -> None:
        """
        Record a request.

        Args:
            model: Model name
            original_tokens: Original token count
            compressed_tokens: Compressed token count
            cache_hit: Was this a cache hit?
            cache_similarity: Cache similarity score
            latency_ms: Request latency in milliseconds
            compression_enabled: Was compression enabled?
            compression_ratio: Compression ratio
        """
        tokens_saved = original_tokens - compressed_tokens if compression_enabled else 0

        stats = RequestStats(
            timestamp=time.time(),
            model=model,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            cache_hit=cache_hit,
            cache_similarity=cache_similarity,
            latency_ms=latency_ms,
            compression_enabled=compression_enabled,
            compression_ratio=compression_ratio,
        )

        self.requests.append(stats)

        # Save every 10 requests
        if len(self.requests) % 10 == 0:
            self._save_stats()

        logger.debug(
            f"Stats recorded: tokens_saved={tokens_saved}, "
            f"cache_hit={cache_hit}, latency={latency_ms:.0f}ms"
        )

    def get_aggregate_stats(
        self,
        since_hours: int | None = None,
    ) -> AggregateStats:
        """
        Get aggregated statistics.

        Args:
            since_hours: Only include requests from last N hours (None = all time)

        Returns:
            Aggregated stats
        """
        # Filter by time
        if since_hours is not None:
            cutoff_timestamp = time.time() - (since_hours * 3600)
            requests = [r for r in self.requests if r.timestamp >= cutoff_timestamp]
        else:
            requests = self.requests

        if not requests:
            return AggregateStats(
                total_requests=0,
                cache_hits=0,
                cache_hit_rate=0.0,
                total_tokens_original=0,
                total_tokens_compressed=0,
                total_tokens_saved=0,
                average_compression_ratio=1.0,
                average_latency_ms=0.0,
            )

        # Calculate aggregates
        total_requests = len(requests)
        cache_hits = sum(1 for r in requests if r.cache_hit)
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0

        total_tokens_original = sum(r.original_tokens for r in requests)
        total_tokens_compressed = sum(r.compressed_tokens for r in requests)
        total_tokens_saved = sum(r.tokens_saved for r in requests)

        avg_compression_ratio = (
            sum(r.compression_ratio for r in requests) / total_requests
            if total_requests > 0 else 1.0
        )

        avg_latency = (
            sum(r.latency_ms for r in requests) / total_requests
            if total_requests > 0 else 0.0
        )

        return AggregateStats(
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_hit_rate=cache_hit_rate,
            total_tokens_original=total_tokens_original,
            total_tokens_compressed=total_tokens_compressed,
            total_tokens_saved=total_tokens_saved,
            average_compression_ratio=avg_compression_ratio,
            average_latency_ms=avg_latency,
        )

    def get_cost_savings(
        self,
        since_hours: int | None = None,
        cost_per_1m_tokens: float = 15.0,  # Claude Opus pricing
    ) -> dict[str, Any]:
        """
        Estimate cost savings.

        Args:
            since_hours: Time window
            cost_per_1m_tokens: Cost per 1M tokens (default: Claude Opus $15/1M)

        Returns:
            Cost savings breakdown
        """
        stats = self.get_aggregate_stats(since_hours)

        # Cost of original tokens
        original_cost = (stats.total_tokens_original / 1_000_000) * cost_per_1m_tokens

        # Cost of compressed tokens
        compressed_cost = (stats.total_tokens_compressed / 1_000_000) * cost_per_1m_tokens

        # Savings
        savings = original_cost - compressed_cost
        savings_percent = (savings / original_cost * 100) if original_cost > 0 else 0.0

        return {
            "original_cost_usd": round(original_cost, 2),
            "compressed_cost_usd": round(compressed_cost, 2),
            "savings_usd": round(savings, 2),
            "savings_percent": round(savings_percent, 1),
            "tokens_saved": stats.total_tokens_saved,
        }

    def clear(self) -> None:
        """Clear all statistics."""
        self.requests.clear()
        self._save_stats()
        logger.info("Stats cleared")

    def _save_stats(self) -> None:
        """Save stats to disk."""
        try:
            stats_data = [r.model_dump() for r in self.requests]
            self.stats_file.write_text(json.dumps(stats_data, indent=2))
            logger.debug(f"Stats saved ({len(self.requests)} requests)")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def _load_stats(self) -> None:
        """Load stats from disk."""
        if not self.stats_file.exists():
            logger.info("No existing stats found")
            return

        try:
            stats_data = json.loads(self.stats_file.read_text())
            self.requests = [RequestStats(**s) for s in stats_data]
            logger.success(f"Loaded stats: {len(self.requests)} requests")
        except Exception as e:
            logger.error(f"Failed to load stats: {e}")
            self.requests.clear()
