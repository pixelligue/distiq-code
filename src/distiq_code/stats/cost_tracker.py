"""
Cost Tracker for Multi-Provider System

Tracks real costs per provider, model, and request type.
Supports alerts on budget limits and detailed breakdowns.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel


class RequestType(str, Enum):
    """Type of request for cost categorization."""
    PLANNING = "planning"
    EXECUTION = "execution"
    CACHE_HIT = "cache_hit"
    TOOL_USE = "tool_use"
    OTHER = "other"


class CostEntry(BaseModel):
    """Single cost entry."""
    
    timestamp: float
    provider: str
    model: str
    request_type: str
    
    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    
    # Cost in USD
    cost_usd: float = 0.0
    
    # Optional savings (e.g., from routing)
    savings_usd: float = 0.0
    
    # Latency
    latency_ms: float = 0.0
    
    # Metadata
    metadata: dict = {}


class CostBreakdown(BaseModel):
    """Cost breakdown by provider/model."""
    
    provider: str
    model: str
    total_requests: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    avg_cost_per_request: float = 0.0


class CostSummary(BaseModel):
    """Summary of costs for a period."""
    
    period_start: float
    period_end: float
    total_requests: int = 0
    total_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    
    # Breakdown by provider
    by_provider: dict[str, float] = {}
    
    # Breakdown by model
    by_model: dict[str, float] = {}
    
    # Breakdown by request type
    by_type: dict[str, float] = {}
    
    # Budget status
    budget_limit_usd: float | None = None
    budget_remaining_usd: float | None = None
    budget_percent_used: float | None = None


class CostTracker:
    """
    Track costs across all providers and models.
    
    Features:
    - Real cost tracking per request
    - Breakdown by provider, model, request type
    - Budget limits and alerts
    - Savings calculation (vs baseline)
    """
    
    def __init__(
        self,
        data_file: Path | None = None,
        monthly_budget_usd: float | None = None,
    ):
        """
        Initialize cost tracker.
        
        Args:
            data_file: File to persist cost data
            monthly_budget_usd: Optional monthly budget limit
        """
        self.data_file = data_file
        self.monthly_budget_usd = monthly_budget_usd
        self.entries: list[CostEntry] = []
        
        # Callbacks for alerts
        self._alert_callbacks: list = []
        
        # Load existing data
        if data_file:
            self._load_data()
    
    def record(
        self,
        provider: str,
        model: str,
        request_type: RequestType | str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        cost_usd: float = 0.0,
        savings_usd: float = 0.0,
        latency_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> CostEntry:
        """
        Record a cost entry.
        
        Args:
            provider: Provider name (anthropic, deepseek, openai_compatible)
            model: Model name
            request_type: Type of request
            input_tokens: Input token count
            output_tokens: Output token count
            cached_tokens: Cached token count
            cost_usd: Actual cost in USD
            savings_usd: Savings vs baseline (e.g., if routed to cheaper model)
            latency_ms: Request latency
            metadata: Additional metadata
            
        Returns:
            Created cost entry
        """
        if isinstance(request_type, str):
            request_type = request_type
        else:
            request_type = request_type.value
        
        entry = CostEntry(
            timestamp=time.time(),
            provider=provider,
            model=model,
            request_type=request_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            savings_usd=savings_usd,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        
        self.entries.append(entry)
        
        # Check budget alerts
        self._check_budget_alerts()
        
        # Auto-save every 10 entries
        if len(self.entries) % 10 == 0 and self.data_file:
            self._save_data()
        
        logger.debug(
            f"Cost recorded: {provider}/{model} ${cost_usd:.4f} "
            f"({input_tokens}+{output_tokens} tokens)"
        )
        
        return entry
    
    def get_summary(
        self,
        since_hours: float | None = None,
        since_timestamp: float | None = None,
    ) -> CostSummary:
        """
        Get cost summary for a period.
        
        Args:
            since_hours: Include entries from last N hours
            since_timestamp: Include entries since this timestamp
            
        Returns:
            Cost summary
        """
        now = time.time()
        
        if since_timestamp:
            start_time = since_timestamp
        elif since_hours:
            start_time = now - (since_hours * 3600)
        else:
            start_time = 0
        
        # Filter entries
        entries = [e for e in self.entries if e.timestamp >= start_time]
        
        if not entries:
            return CostSummary(
                period_start=start_time,
                period_end=now,
            )
        
        # Calculate totals
        total_cost = sum(e.cost_usd for e in entries)
        total_savings = sum(e.savings_usd for e in entries)
        
        # Breakdown by provider
        by_provider: dict[str, float] = {}
        for entry in entries:
            by_provider[entry.provider] = by_provider.get(entry.provider, 0) + entry.cost_usd
        
        # Breakdown by model
        by_model: dict[str, float] = {}
        for entry in entries:
            by_model[entry.model] = by_model.get(entry.model, 0) + entry.cost_usd
        
        # Breakdown by type
        by_type: dict[str, float] = {}
        for entry in entries:
            by_type[entry.request_type] = by_type.get(entry.request_type, 0) + entry.cost_usd
        
        # Budget status
        budget_remaining = None
        budget_percent = None
        if self.monthly_budget_usd:
            budget_remaining = self.monthly_budget_usd - total_cost
            budget_percent = (total_cost / self.monthly_budget_usd) * 100
        
        return CostSummary(
            period_start=start_time,
            period_end=now,
            total_requests=len(entries),
            total_cost_usd=total_cost,
            total_savings_usd=total_savings,
            by_provider=by_provider,
            by_model=by_model,
            by_type=by_type,
            budget_limit_usd=self.monthly_budget_usd,
            budget_remaining_usd=budget_remaining,
            budget_percent_used=budget_percent,
        )
    
    def get_breakdown_by_provider(
        self,
        since_hours: float | None = None,
    ) -> list[CostBreakdown]:
        """Get detailed breakdown by provider and model."""
        now = time.time()
        start_time = now - (since_hours * 3600) if since_hours else 0
        
        entries = [e for e in self.entries if e.timestamp >= start_time]
        
        # Group by provider+model
        groups: dict[tuple[str, str], list[CostEntry]] = {}
        for entry in entries:
            key = (entry.provider, entry.model)
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)
        
        # Calculate breakdown for each group
        breakdowns = []
        for (provider, model), group_entries in groups.items():
            total_cost = sum(e.cost_usd for e in group_entries)
            breakdowns.append(CostBreakdown(
                provider=provider,
                model=model,
                total_requests=len(group_entries),
                total_cost_usd=total_cost,
                total_input_tokens=sum(e.input_tokens for e in group_entries),
                total_output_tokens=sum(e.output_tokens for e in group_entries),
                total_cached_tokens=sum(e.cached_tokens for e in group_entries),
                avg_cost_per_request=total_cost / len(group_entries) if group_entries else 0,
            ))
        
        # Sort by cost descending
        breakdowns.sort(key=lambda x: x.total_cost_usd, reverse=True)
        return breakdowns
    
    def get_monthly_cost(self) -> float:
        """Get total cost for current month."""
        now = datetime.now()
        month_start = datetime(now.year, now.month, 1).timestamp()
        
        return sum(
            e.cost_usd for e in self.entries 
            if e.timestamp >= month_start
        )
    
    def add_alert_callback(self, callback) -> None:
        """Add callback for budget alerts."""
        self._alert_callbacks.append(callback)
    
    def _check_budget_alerts(self) -> None:
        """Check and trigger budget alerts."""
        if not self.monthly_budget_usd:
            return
        
        monthly_cost = self.get_monthly_cost()
        percent_used = (monthly_cost / self.monthly_budget_usd) * 100
        
        # Alert thresholds
        thresholds = [50, 75, 90, 100]
        
        for threshold in thresholds:
            if percent_used >= threshold:
                for callback in self._alert_callbacks:
                    try:
                        callback(
                            threshold=threshold,
                            current_cost=monthly_cost,
                            budget=self.monthly_budget_usd,
                            percent_used=percent_used,
                        )
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
    
    def clear(self) -> None:
        """Clear all cost data."""
        self.entries.clear()
        if self.data_file:
            self._save_data()
        logger.info("Cost data cleared")
    
    def _save_data(self) -> None:
        """Save data to file."""
        if not self.data_file:
            return
        
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            data = [e.model_dump() for e in self.entries]
            self.data_file.write_text(json.dumps(data, indent=2))
            logger.debug(f"Cost data saved ({len(self.entries)} entries)")
        except Exception as e:
            logger.error(f"Failed to save cost data: {e}")
    
    def _load_data(self) -> None:
        """Load data from file."""
        if not self.data_file or not self.data_file.exists():
            return
        
        try:
            data = json.loads(self.data_file.read_text())
            self.entries = [CostEntry(**e) for e in data]
            logger.info(f"Loaded cost data: {len(self.entries)} entries")
        except Exception as e:
            logger.error(f"Failed to load cost data: {e}")
            self.entries = []


# Global instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker(
    data_dir: Path | None = None,
    monthly_budget_usd: float | None = None,
) -> CostTracker:
    """
    Get or create global cost tracker.
    
    Args:
        data_dir: Directory for cost data file
        monthly_budget_usd: Optional monthly budget
        
    Returns:
        CostTracker instance
    """
    global _cost_tracker
    
    if _cost_tracker is None:
        if data_dir is None:
            from distiq_code.config import settings
            data_dir = settings.config_dir
        
        data_file = data_dir / "cost_data.json"
        _cost_tracker = CostTracker(
            data_file=data_file,
            monthly_budget_usd=monthly_budget_usd,
        )
    
    return _cost_tracker
