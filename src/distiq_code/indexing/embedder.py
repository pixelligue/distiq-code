"""
Code Embedding Model

Generates embeddings for code chunks using local models.
Uses sentence-transformers with code-optimized models.

Recommended models:
- nomic-ai/nomic-embed-text-v1.5 (default, good balance)
- Salesforce/SFR-Embedding-Code (best quality, larger)
- BAAI/bge-small-en-v1.5 (fast, smaller)
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


class CodeEmbedder:
    """
    Generate embeddings for code chunks.
    
    Uses local sentence-transformers models for privacy and speed.
    Embeddings are cached to avoid recomputation.
    """
    
    # Default model - good balance of quality and speed
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    
    # Alternative models
    MODELS = {
        "nomic": "nomic-ai/nomic-embed-text-v1.5",        # 768 dim, good quality
        "bge-small": "BAAI/bge-small-en-v1.5",             # 384 dim, fast
        "bge-base": "BAAI/bge-base-en-v1.5",               # 768 dim, balanced
        "sfr-code": "Salesforce/SFR-Embedding-Code",       # Best for code
        "mpnet": "sentence-transformers/all-mpnet-base-v2", # 768 dim, general
    }
    
    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | None = None,
        device: str = "cpu",
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: Model name or alias from MODELS
            cache_dir: Directory to cache embeddings
            device: Device to run model on (cpu, cuda, mps)
        """
        # Resolve model name
        if model_name is None:
            model_name = self.DEFAULT_MODEL
        elif model_name in self.MODELS:
            model_name = self.MODELS[model_name]
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        
        # Lazy load model
        self._model = None
        self._embedding_dim = None
        
        # In-memory cache
        self._cache: dict[str, np.ndarray] = {}
        
        # Load disk cache
        if cache_dir:
            self._load_cache()
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,  # Required for nomic-ai models
            )
            
            # Get embedding dimension
            test_embedding = self._model.encode(["test"])
            self._embedding_dim = test_embedding.shape[1]
            
            logger.info(
                f"Model loaded: {self.model_name} "
                f"(dim={self._embedding_dim}, device={self.device})"
            )
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate embedding
        self._load_model()
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        
        # Cache
        self._cache[cache_key] = embedding
        
        return embedding
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check which texts are already cached
        cached_embeddings = {}
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                cached_embeddings[i] = self._cache[cache_key]
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            self._load_model()
            new_embeddings = self._model.encode(
                texts_to_embed,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            
            # Cache new embeddings
            for i, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = embedding
                cached_embeddings[text_indices[i]] = embedding
        
        # Combine in correct order
        result = np.array([cached_embeddings[i] for i in range(len(texts))])
        
        return result
    
    def embed_code_chunk(self, chunk: Any) -> np.ndarray:
        """
        Embed a code chunk with context.
        
        Adds prefix for better code understanding.
        
        Args:
            chunk: CodeChunk object
            
        Returns:
            Embedding vector
        """
        # Build text with context
        parts = []
        
        # Add type prefix
        parts.append(f"[{chunk.chunk_type.upper()}]")
        
        # Add name
        if chunk.name:
            parts.append(f"{chunk.name}:")
        
        # Add signature
        if chunk.signature:
            parts.append(chunk.signature)
        
        # Add docstring
        if chunk.docstring:
            parts.append(chunk.docstring)
        
        # Add content (truncated if too long)
        content = chunk.content
        if len(content) > 2000:
            content = content[:2000] + "..."
        parts.append(content)
        
        text = "\n".join(parts)
        return self.embed(text)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        return float(np.dot(embedding1, embedding2))
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Include model name in cache key
        key_content = f"{self.model_name}:{text}"
        return hashlib.sha256(key_content.encode()).hexdigest()[:16]
    
    def _load_cache(self):
        """Load embeddings cache from disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "embeddings_cache.npz"
        if not cache_file.exists():
            return
        
        try:
            data = np.load(cache_file, allow_pickle=True)
            self._cache = dict(data["cache"].item())
            logger.info(f"Loaded {len(self._cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
    
    def save_cache(self):
        """Save embeddings cache to disk."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "embeddings_cache.npz"
        
        try:
            np.savez_compressed(cache_file, cache=self._cache)
            logger.debug(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        if self.cache_dir:
            cache_file = self.cache_dir / "embeddings_cache.npz"
            if cache_file.exists():
                cache_file.unlink()
        logger.info("Embedding cache cleared")
