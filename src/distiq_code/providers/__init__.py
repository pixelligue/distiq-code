"""
Multi-Provider System for Distiq Code

Supports three connection methods:
1. Claude Code CLI (subscription) - via ClaudeCliProvider
2. OpenAI-compatible API - OpenRouter, DeepSeek, Ollama, local models
3. Anthropic API - direct API key access

Usage:
    from distiq_code.providers import get_provider, ProviderType
    
    # Auto-detect best available
    provider = get_provider()
    
    # Specific provider
    provider = get_provider(ProviderType.ANTHROPIC_API)
    provider = get_provider(ProviderType.OPENAI_COMPATIBLE, base_url="https://openrouter.ai/api/v1")
    provider = get_provider(ProviderType.CLAUDE_CLI)
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseProvider

__all__ = [
    "ProviderType",
    "get_provider",
    "get_available_providers",
]


class ProviderType(str, Enum):
    """Available provider types."""
    
    # Claude Code subscription (via CLI)
    CLAUDE_CLI = "claude_cli"
    
    # Anthropic API (direct, needs API key)
    ANTHROPIC_API = "anthropic_api"
    
    # OpenAI-compatible API (OpenRouter, DeepSeek, Ollama, etc.)
    OPENAI_COMPATIBLE = "openai_compatible"
    
    # DeepSeek (specialized, uses OpenAI format but with caching)
    DEEPSEEK = "deepseek"


# Provider instances cache (singleton)
_providers: dict[str, "BaseProvider"] = {}


def get_provider(
    provider_type: ProviderType | str | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> "BaseProvider":
    """
    Get or create a provider instance.
    
    Args:
        provider_type: Type of provider. If None, auto-detects best available.
        base_url: Base URL for OpenAI-compatible providers
        api_key: API key (reads from env if not provided)
        model: Default model for this provider
        
    Returns:
        Provider instance
        
    Raises:
        ValueError: If provider type is unknown
        RuntimeError: If provider is not available (missing deps/keys)
    """
    import os
    
    # Convert string to enum
    if isinstance(provider_type, str):
        provider_type = ProviderType(provider_type)
    
    # Auto-detect if not specified
    if provider_type is None:
        provider_type = _detect_best_provider()
    
    # Create cache key
    cache_key = f"{provider_type.value}:{base_url or 'default'}"
    
    # Return cached if exists
    if cache_key in _providers:
        return _providers[cache_key]
    
    # Create new provider
    provider: "BaseProvider"
    
    if provider_type == ProviderType.CLAUDE_CLI:
        from distiq_code.auth.cli_provider import ClaudeCliProvider
        # Wrap in our base interface
        provider = ClaudeCliProviderAdapter()
        
    elif provider_type == ProviderType.ANTHROPIC_API:
        from .anthropic import AnthropicProvider
        provider = AnthropicProvider(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        
    elif provider_type == ProviderType.DEEPSEEK:
        from .deepseek import DeepSeekProvider
        provider = DeepSeekProvider(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        )
        
    elif provider_type == ProviderType.OPENAI_COMPATIBLE:
        from .openai_compat import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            default_model=model,
        )
        
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    # Cache and return
    _providers[cache_key] = provider
    return provider


def _detect_best_provider() -> ProviderType:
    """Auto-detect the best available provider."""
    import os
    import shutil
    
    # Priority 1: Claude CLI (subscription - free usage)
    if shutil.which("claude"):
        return ProviderType.CLAUDE_CLI
    
    # Priority 2: DeepSeek (cheapest API)
    if os.environ.get("DEEPSEEK_API_KEY"):
        return ProviderType.DEEPSEEK
    
    # Priority 3: Anthropic API
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderType.ANTHROPIC_API
    
    # Priority 4: OpenRouter (has many models)
    if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return ProviderType.OPENAI_COMPATIBLE
    
    # Default: try Claude CLI anyway
    return ProviderType.CLAUDE_CLI


def get_available_providers() -> list[dict]:
    """
    Get list of available providers with their status.
    
    Returns:
        List of dicts with provider info
    """
    import os
    import shutil
    
    providers = []
    
    # Claude CLI
    claude_available = bool(shutil.which("claude"))
    providers.append({
        "type": ProviderType.CLAUDE_CLI,
        "name": "Claude Code (Subscription)",
        "available": claude_available,
        "reason": "CLI found" if claude_available else "Install: npm i -g @anthropic-ai/claude-code",
        "cost": "Included in subscription",
    })
    
    # Anthropic API
    anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    providers.append({
        "type": ProviderType.ANTHROPIC_API,
        "name": "Anthropic API",
        "available": anthropic_key,
        "reason": "API key found" if anthropic_key else "Set ANTHROPIC_API_KEY",
        "cost": "$3/MTok (Sonnet), $15/MTok (Opus)",
    })
    
    # DeepSeek
    deepseek_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    providers.append({
        "type": ProviderType.DEEPSEEK,
        "name": "DeepSeek",
        "available": deepseek_key,
        "reason": "API key found" if deepseek_key else "Set DEEPSEEK_API_KEY",
        "cost": "$0.28/MTok (cheapest!)",
    })
    
    # OpenRouter
    openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    providers.append({
        "type": ProviderType.OPENAI_COMPATIBLE,
        "name": "OpenRouter / OpenAI Compatible",
        "available": openrouter_key,
        "reason": "API key found" if openrouter_key else "Set OPENROUTER_API_KEY",
        "cost": "Varies by model (+5% fee)",
    })
    
    return providers


class ClaudeCliProviderAdapter:
    """Adapter to make ClaudeCliProvider compatible with BaseProvider interface."""
    
    def __init__(self):
        from distiq_code.auth.cli_provider import ClaudeCliProvider
        self._cli = ClaudeCliProvider()
        self.name = "claude_cli"
        
    async def generate(
        self,
        messages: list[dict],
        model: str = "sonnet",
        **kwargs,
    ) -> dict:
        """Generate completion using Claude CLI."""
        response = await self._cli.complete(messages, model=model, stream=False)
        
        # Return in standard format
        return {
            "content": response,
            "model": model,
            "provider": self.name,
            "usage": {
                "input_tokens": 0,  # CLI doesn't report
                "output_tokens": 0,
                "cached_tokens": 0,
            },
            "cost_usd": 0.0,  # Included in subscription
        }
    
    async def stream(
        self,
        messages: list[dict],
        model: str = "sonnet",
        **kwargs,
    ):
        """Stream completion using Claude CLI."""
        stream = await self._cli.complete(messages, model=model, stream=True)
        async for chunk in stream:
            yield chunk
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Claude CLI is included in subscription - no additional cost."""
        return 0.0
