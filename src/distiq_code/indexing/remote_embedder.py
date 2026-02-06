"""
Remote Embeddings Client

Supports multiple providers with automatic fallback:
1. Voyage AI (200M free tokens!) - Best quality
2. Jina AI (10M free tokens) - Good quality
3. Local (nomic-embed) - Offline fallback

Zero download required - everything runs via API.
"""

import os
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
from loguru import logger


@dataclass
class EmbeddingResponse:
    """Response from embedding API."""
    embeddings: list[np.ndarray]
    model: str
    tokens_used: int
    provider: str


class VoyageEmbedder:
    """
    Voyage AI Embeddings.
    
    Free tier: 200M tokens!
    Models: voyage-code-3 (best for code), voyage-4-lite (cheap)
    """
    
    BASE_URL = "https://api.voyageai.com/v1/embeddings"
    
    # Models optimized for code
    MODELS = {
        "code": "voyage-code-3",      # Best for code, 1024 dim
        "lite": "voyage-4-lite",       # Cheapest, 1024 dim
        "default": "voyage-4",         # Balanced, 1024 dim
    }
    
    def __init__(self, api_key: str | None = None, model: str = "code"):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.model = self.MODELS.get(model, model)
        self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._client
    
    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not set")
        
        response = await self.client.post(
            self.BASE_URL,
            json={
                "input": texts,
                "model": self.model,
                "input_type": "document",
            }
        )
        response.raise_for_status()
        
        data = response.json()
        
        embeddings = [
            np.array(item["embedding"], dtype=np.float32)
            for item in data["data"]
        ]
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            provider="voyage",
        )
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class JinaEmbedder:
    """
    Jina AI Embeddings.
    
    Free tier: 10M tokens
    Models: jina-embeddings-v3 (best), jina-embeddings-v2-base-code
    """
    
    BASE_URL = "https://api.jina.ai/v1/embeddings"
    
    MODELS = {
        "code": "jina-embeddings-v2-base-code",  # For code, 768 dim
        "default": "jina-embeddings-v3",          # General, 1024 dim
    }
    
    def __init__(self, api_key: str | None = None, model: str = "code"):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model = self.MODELS.get(model, model)
        self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._client
    
    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        if not self.api_key:
            raise ValueError("JINA_API_KEY not set")
        
        response = await self.client.post(
            self.BASE_URL,
            json={
                "input": texts,
                "model": self.model,
            }
        )
        response.raise_for_status()
        
        data = response.json()
        
        embeddings = [
            np.array(item["embedding"], dtype=np.float32)
            for item in data["data"]
        ]
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            provider="jina",
        )
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class SmartEmbedder:
    """
    Smart embedder with automatic fallback.
    
    Priority:
    1. Voyage AI (if VOYAGE_API_KEY set) - 200M free!
    2. Jina AI (if JINA_API_KEY set) - 10M free
    3. Local model (if installed) - 0 cost, but downloads 547MB
    
    Usage:
        embedder = SmartEmbedder()
        result = await embedder.embed(["code snippet 1", "code snippet 2"])
    """
    
    def __init__(
        self,
        prefer_local: bool = False,
        model_preference: str = "code",
    ):
        self.prefer_local = prefer_local
        self.model_preference = model_preference
        
        self._voyage = None
        self._jina = None
        self._local = None
        
        self._active_provider = None
    
    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """
        Generate embeddings with automatic fallback.
        """
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                model="none",
                tokens_used=0,
                provider="none",
            )
        
        # Try providers in order
        providers = self._get_provider_order()
        
        last_error = None
        for provider_name, provider_func in providers:
            try:
                logger.debug(f"Trying {provider_name} embeddings...")
                result = await provider_func(texts)
                self._active_provider = provider_name
                logger.debug(f"Using {provider_name}: {len(result.embeddings)} embeddings")
                return result
            except Exception as e:
                logger.debug(f"{provider_name} failed: {e}")
                last_error = e
                continue
        
        raise RuntimeError(f"All embedding providers failed. Last error: {last_error}")
    
    def _get_provider_order(self) -> list[tuple[str, Any]]:
        """Get providers in priority order."""
        providers = []
        
        # Check available providers
        voyage_key = os.getenv("VOYAGE_API_KEY")
        jina_key = os.getenv("JINA_API_KEY")
        
        if self.prefer_local:
            providers.append(("local", self._embed_local))
        
        if voyage_key:
            providers.append(("voyage", self._embed_voyage))
        
        if jina_key:
            providers.append(("jina", self._embed_jina))
        
        if not self.prefer_local:
            providers.append(("local", self._embed_local))
        
        return providers
    
    async def _embed_voyage(self, texts: list[str]) -> EmbeddingResponse:
        """Embed with Voyage AI."""
        if self._voyage is None:
            self._voyage = VoyageEmbedder(model=self.model_preference)
        return await self._voyage.embed(texts)
    
    async def _embed_jina(self, texts: list[str]) -> EmbeddingResponse:
        """Embed with Jina AI."""
        if self._jina is None:
            self._jina = JinaEmbedder(model=self.model_preference)
        return await self._jina.embed(texts)
    
    async def _embed_local(self, texts: list[str]) -> EmbeddingResponse:
        """Embed with local model."""
        try:
            from distiq_code.indexing.embedder import CodeEmbedder
            
            if self._local is None:
                self._local = CodeEmbedder()
            
            embeddings = self._local.embed_batch(texts, show_progress=False)
            
            return EmbeddingResponse(
                embeddings=[e for e in embeddings],
                model=self._local.model_name,
                tokens_used=sum(len(t.split()) for t in texts) * 2,  # estimate
                provider="local",
            )
        except ImportError as e:
            raise RuntimeError(f"Local embeddings not available: {e}")
    
    async def close(self):
        """Close all clients."""
        if self._voyage:
            await self._voyage.close()
        if self._jina:
            await self._jina.close()
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension based on model."""
        if self._active_provider == "voyage":
            return 1024
        elif self._active_provider == "jina":
            return 768 if self.model_preference == "code" else 1024
        else:
            return 768  # local nomic model


# Convenience function
async def smart_embed(
    texts: list[str],
    prefer_local: bool = False,
) -> EmbeddingResponse:
    """
    Generate embeddings using best available provider.
    
    Args:
        texts: List of texts to embed
        prefer_local: Prefer local model over API
        
    Returns:
        EmbeddingResponse with embeddings and metadata
    """
    embedder = SmartEmbedder(prefer_local=prefer_local)
    try:
        return await embedder.embed(texts)
    finally:
        await embedder.close()
