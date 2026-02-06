"""
OpenAI-Compatible Provider

Supports any API that follows the OpenAI chat completions format:
- OpenRouter
- DeepSeek
- Ollama
- LM Studio
- vLLM
- Any OpenAI-compatible server

Usage:
    provider = OpenAICompatibleProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key="your-key",
    )
    
    response = await provider.generate(
        messages=[{"role": "user", "content": "Hello"}],
        model="deepseek/deepseek-chat",
    )
"""

import os
import time
from typing import AsyncGenerator, Any

import httpx
from loguru import logger

from .base import BaseProvider, ProviderPricing, ProviderResponse


# Pricing for common models ($ per 1M tokens)
# Update these based on current pricing!
OPENROUTER_PRICING = {
    # DeepSeek
    "deepseek/deepseek-chat": ProviderPricing(0.28, 0.42, 0.028),
    "deepseek/deepseek-coder": ProviderPricing(0.28, 0.42, 0.028),
    
    # Anthropic via OpenRouter (+5% fee included)
    "anthropic/claude-3.5-sonnet": ProviderPricing(3.15, 15.75, 0.315),
    "anthropic/claude-3-haiku": ProviderPricing(0.26, 1.31, 0.026),
    "anthropic/claude-3-opus": ProviderPricing(15.75, 78.75, 1.575),
    
    # OpenAI via OpenRouter
    "openai/gpt-4o": ProviderPricing(2.63, 10.50, 0.263),
    "openai/gpt-4o-mini": ProviderPricing(0.16, 0.63, 0.016),
    
    # Qwen
    "qwen/qwen-2.5-coder-32b-instruct": ProviderPricing(0.18, 0.18, 0.018),
    
    # Llama
    "meta-llama/llama-3.1-70b-instruct": ProviderPricing(0.52, 0.75, 0.052),
    "meta-llama/llama-3.1-8b-instruct": ProviderPricing(0.055, 0.055, 0.0055),
    
    # Mistral
    "mistralai/mistral-large": ProviderPricing(2.10, 6.30, 0.21),
    "mistralai/codestral-latest": ProviderPricing(0.30, 0.90, 0.03),
}

# Default pricing for unknown models
DEFAULT_PRICING = ProviderPricing(1.0, 2.0, 0.1)


class OpenAICompatibleProvider(BaseProvider):
    """
    Provider for any OpenAI-compatible API.
    
    Supports:
    - OpenRouter (https://openrouter.ai/api/v1)
    - DeepSeek (https://api.deepseek.com)
    - Ollama (http://localhost:11434/v1)
    - Any OpenAI-compatible server
    """
    
    name = "openai_compatible"
    
    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize provider.
        
        Args:
            base_url: API base URL
            api_key: API key (or from OPENROUTER_API_KEY / OPENAI_API_KEY env)
            default_model: Default model to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.default_model = default_model
        self.timeout = timeout
        
        # HTTP client with HTTP/2 support
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True,
        )
        
        logger.debug(f"Initialized OpenAI-compatible provider: {self.base_url}")
    
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> ProviderResponse:
        """Generate completion."""
        model = model or self.default_model or "deepseek/deepseek-chat"
        
        # Prepare messages
        api_messages = self._prepare_messages(messages, system)
        
        # Build request
        payload = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs,
        }
        
        # Add OpenRouter-specific headers if needed
        headers = self._get_headers()
        
        # Send request
        start_time = time.time()
        
        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            raise
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("prompt_cache_hit_tokens", 0)  # DeepSeek-specific
        
        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model, cached_tokens)
        
        return ProviderResponse(
            content=content,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream completion."""
        model = model or self.default_model or "deepseek/deepseek-chat"
        
        # Prepare messages
        api_messages = self._prepare_messages(messages, system)
        
        # Build request
        payload = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }
        
        headers = self._get_headers()
        
        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                    
                data_str = line[6:]  # Remove "data: " prefix
                
                if data_str == "[DONE]":
                    break
                
                try:
                    import json
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost in USD."""
        pricing = self.get_pricing(model)
        return pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)
    
    def get_pricing(self, model: str) -> ProviderPricing:
        """Get pricing for model."""
        # Try exact match
        if model in OPENROUTER_PRICING:
            return OPENROUTER_PRICING[model]
        
        # Try without provider prefix
        if "/" in model:
            short = model.split("/")[-1]
            for key, pricing in OPENROUTER_PRICING.items():
                if short in key:
                    return pricing
        
        return DEFAULT_PRICING
    
    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> list[dict]:
        """Prepare messages for API call."""
        api_messages = []
        
        # Add system message if provided
        if system:
            api_messages.append({"role": "system", "content": system})
        
        # Add user messages
        for msg in messages:
            api_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        
        return api_messages
    
    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # OpenRouter-specific headers
        if "openrouter" in self.base_url.lower():
            headers["HTTP-Referer"] = "https://distiq.dev"
            headers["X-Title"] = "Distiq Code"
        
        return headers
    
    async def health_check(self) -> bool:
        """Check if API is available."""
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()
