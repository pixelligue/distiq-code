"""Configuration management with Pydantic Settings."""

import sys
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_safe_home() -> Path:
    """Get home directory with ASCII-safe path on Windows.

    FAISS C++ library cannot handle Unicode (Cyrillic) paths on Windows.
    We use the Windows short path (8.3 format) to avoid this issue.
    """
    home = Path.home()

    if sys.platform != "win32":
        return home

    home_str = str(home)
    if home_str.isascii():
        return home

    # Get Windows 8.3 short path (e.g. C:\Users\CD86~1 for C:\Users\Алексей)
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        ctypes.windll.kernel32.GetShortPathNameW(home_str, buf, 260)
        if buf.value:
            return Path(buf.value)
    except Exception:
        pass

    return home


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    proxy_host: str = Field(default="127.0.0.1", description="Proxy server host")
    proxy_port: int = Field(default=11434, description="Proxy server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level"
    )

    # Compression
    compression_enabled: bool = Field(default=True, description="Enable prompt compression")
    compression_target_tokens: int = Field(
        default=500, ge=100, description="Target token count after compression"
    )
    compression_model: str = Field(
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        description="LLMLingua-2 model (BERT-base: 440MB, 3-6x faster)",
    )

    # Caching
    cache_enabled: bool = Field(default=True, description="Enable semantic caching")
    cache_ttl_hours: int = Field(default=168, ge=1, description="Cache TTL in hours (default: 7 days)")
    cache_similarity_threshold: float = Field(
        default=0.85, ge=0.7, le=1.0, description="Similarity threshold for cache hits"
    )
    cache_max_size: int = Field(
        default=10000, ge=100, description="Maximum cached entries"
    )
    cache_embedding_model: str = Field(
        default="google/embeddinggemma-300m",
        description="Sentence transformer model for embeddings (Matryoshka-enabled)",
    )

    # Smart Routing
    smart_routing: bool = Field(default=True, description="Auto-select model based on query complexity")

    # ML Routing (embedding-based, uses same model as cache)
    ml_routing_enabled: bool = Field(
        default=True, description="Use embedding router when available (fallback: regex)"
    )

    # Anthropic Prompt Caching
    prompt_caching_enabled: bool = Field(
        default=True, description="Inject cache_control breakpoints for Anthropic prompt caching (90% input savings)"
    )

    # Tool Compression (inspired by ClaudeSlim)
    tool_compression_enabled: bool = Field(
        default=False, description="Compress tool definitions (names + params) to save 500-1000 tokens per request"
    )

    # Anthropic API forwarding
    anthropic_api_base: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic API base URL for forwarding",
    )

    # API Keys (loaded from .env)
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")
    deepseek_api_key: str | None = Field(default=None, description="DeepSeek API key")
    jina_api_key: str | None = Field(default=None, description="Jina AI API key (search + embeddings)")
    voyage_api_key: str | None = Field(default=None, description="Voyage AI API key (embeddings)")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key (web search)")


    # Paths
    @property
    def config_dir(self) -> Path:
        """Get config directory path (ASCII-safe for FAISS on Windows)."""
        home = _get_safe_home()
        config_dir = home / ".distiq-code"
        config_dir.mkdir(exist_ok=True)
        return config_dir

    @property
    def cache_dir_path(self) -> Path:
        """Get semantic cache directory."""
        cache_dir = self.config_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    @property
    def stats_db_file(self) -> Path:
        """Get stats database path."""
        return self.config_dir / "stats.db"

    @property
    def models_cache_dir(self) -> Path:
        """Get models cache directory (for LLMLingua, sentence-transformers)."""
        cache_dir = self.config_dir / "models"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir


def get_available_features() -> dict[str, bool]:
    """Detect which optional features are available based on installed packages."""
    features = {
        "routing": True,  # regex routing always available
        "ml": False,
        "cache": False,
        "compression": False,
        "ml_routing": False,
    }

    try:
        import sentence_transformers  # noqa: F401
        import faiss  # noqa: F401
        features["ml"] = True
        features["cache"] = True
        features["ml_routing"] = True
    except ImportError:
        pass

    try:
        import llmlingua  # noqa: F401
        features["compression"] = True
    except ImportError:
        pass

    return features


# Global settings instance
settings = Settings()
