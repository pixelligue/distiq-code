"""FAISS-based semantic caching for LLM responses."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class CacheEntry(BaseModel):
    """Cached LLM response."""

    query_hash: str
    query_text: str
    response: str
    model: str
    timestamp: float
    hit_count: int = 0


class CacheStats(BaseModel):
    """Cache statistics."""

    hit: bool
    similarity: float
    tokens_saved: int = 0


class SemanticCache:
    """
    Semantic cache using FAISS + sentence-transformers.

    Instead of exact match, finds similar queries using embeddings.
    Example:
        Query 1: "How to create a React component?"
        Query 2: "How do I make a React component?" → Cache hit!
    """

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        similarity_threshold: float = 0.85,
        max_cache_size: int = 10000,
        ttl_seconds: int = 86400 * 7,  # 7 days
    ):
        """
        Initialize semantic cache.

        Args:
            cache_dir: Directory to store cache
            model_name: Sentence transformer model
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
            max_cache_size: Maximum cached entries
            ttl_seconds: Time-to-live for cached entries
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.success(f"Embedding model loaded (dim={self.embedding_dim})")

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.cache_entries: list[CacheEntry] = []

        # Load existing cache
        self._load_cache()

    def get(
        self,
        query: str,
        model: str,
    ) -> tuple[str | None, CacheStats]:
        """
        Get cached response for similar query.

        Args:
            query: User query text
            model: Model name

        Returns:
            Cached response (if found) and stats
        """
        # Encode query
        query_embedding = self._encode_query(query)

        # Search in FAISS
        if self.index.ntotal == 0:
            # Empty cache
            return None, CacheStats(hit=False, similarity=0.0)

        # Search for most similar
        similarities, indices = self.index.search(query_embedding, k=1)
        similarity = float(similarities[0][0])
        index = int(indices[0][0])

        logger.debug(f"Cache search: similarity={similarity:.3f}, threshold={self.similarity_threshold}")

        # Check similarity threshold
        if similarity < self.similarity_threshold:
            return None, CacheStats(hit=False, similarity=similarity)

        # Get cached entry
        entry = self.cache_entries[index]

        # Check model match
        if entry.model != model:
            logger.debug(f"Model mismatch: cached={entry.model}, requested={model}")
            return None, CacheStats(hit=False, similarity=similarity)

        # Check TTL
        age_seconds = time.time() - entry.timestamp
        if age_seconds > self.ttl_seconds:
            logger.debug(f"Cache entry expired (age={age_seconds:.0f}s)")
            return None, CacheStats(hit=False, similarity=similarity)

        # Cache hit!
        entry.hit_count += 1
        logger.info(
            f"Cache HIT (similarity={similarity:.2f}, hits={entry.hit_count}): "
            f"{query[:50]}..."
        )

        # Estimate tokens saved (rough: 1 token = 4 chars)
        tokens_saved = len(entry.response) // 4

        return entry.response, CacheStats(
            hit=True,
            similarity=similarity,
            tokens_saved=tokens_saved,
        )

    def set(
        self,
        query: str,
        response: str,
        model: str,
    ) -> None:
        """
        Cache response for query.

        Args:
            query: User query text
            response: LLM response
            model: Model name
        """
        # Check cache size limit
        if len(self.cache_entries) >= self.max_cache_size:
            logger.warning(f"Cache full ({self.max_cache_size} entries), evicting oldest")
            self._evict_oldest()

        # Create cache entry
        query_hash = self._hash_query(query, model)
        entry = CacheEntry(
            query_hash=query_hash,
            query_text=query[:200],  # Store truncated for debugging
            response=response,
            model=model,
            timestamp=time.time(),
        )

        # Add to cache
        self.cache_entries.append(entry)

        # Add embedding to FAISS
        query_embedding = self._encode_query(query)
        self.index.add(query_embedding)

        logger.debug(f"Cached response for: {query[:50]}...")

        # Persist to disk
        self._save_cache()

    def clear(self) -> None:
        """Clear all cache."""
        self.cache_entries.clear()
        self.index.reset()
        self._save_cache()
        logger.info("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache_entries)
        total_hits = sum(e.hit_count for e in self.cache_entries)

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "index_size": self.index.ntotal,
            "similarity_threshold": self.similarity_threshold,
        }

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector."""
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        # Reshape for FAISS
        return embedding.reshape(1, -1).astype("float32")

    def _hash_query(self, query: str, model: str) -> str:
        """Generate hash for query+model."""
        content = f"{model}:{query}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.cache_entries:
            return

        # Find oldest entry
        oldest_idx = min(
            range(len(self.cache_entries)),
            key=lambda i: self.cache_entries[i].timestamp,
        )

        # Remove from cache
        self.cache_entries.pop(oldest_idx)

        # Rebuild FAISS index (no direct removal API)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from cache entries."""
        self.index.reset()

        if not self.cache_entries:
            return

        # Re-encode all queries
        embeddings = []
        for entry in self.cache_entries:
            emb = self._encode_query(entry.query_text)
            embeddings.append(emb)

        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.index.add(embeddings_array)

    def _save_cache(self) -> None:
        """Save cache to disk."""
        cache_file = self.cache_dir / "cache.json"
        index_file = self.cache_dir / "index.faiss"

        # Save entries
        entries_data = [e.model_dump() for e in self.cache_entries]
        cache_file.write_text(json.dumps(entries_data, indent=2))

        # Save FAISS index
        faiss.write_index(self.index, str(index_file))

        logger.debug(f"Cache saved ({len(self.cache_entries)} entries)")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        index_file = self.cache_dir / "index.faiss"

        if not cache_file.exists() or not index_file.exists():
            logger.info("No existing cache found")
            return

        try:
            # Load entries
            entries_data = json.loads(cache_file.read_text())
            self.cache_entries = [CacheEntry(**e) for e in entries_data]

            # Load FAISS index
            self.index = faiss.read_index(str(index_file))

            logger.success(f"Loaded cache: {len(self.cache_entries)} entries")

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache_entries.clear()
            self.index.reset()
