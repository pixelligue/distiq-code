"""
DeepSeek Provider

Specialized provider for DeepSeek API - the cheapest option for code generation.
Uses OpenAI-compatible format but with DeepSeek-specific optimizations.

Pricing (February 2026):
- Input: $0.28/MTok
- Output: $0.42/MTok  
- Cache hit: $0.028/MTok (10x cheaper!)

Usage:
    provider = DeepSeekProvider(api_key="your-key")
    
    response = await provider.generate(
        messages=[{"role": "user", "content": "Write a function"}],
        model="deepseek-chat",
    )
"""

import os
import time
from typing import AsyncGenerator, Any

import httpx
from loguru import logger

from .base import BaseProvider, ProviderPricing, ProviderResponse


# DeepSeek pricing ($ per 1M tokens) - February 2026
DEEPSEEK_PRICING = {
    "deepseek-chat": ProviderPricing(0.28, 0.42, 0.028),
    "deepseek-coder": ProviderPricing(0.28, 0.42, 0.028),
    "deepseek-reasoner": ProviderPricing(0.55, 2.19, 0.055),  # V3.1
}


class DeepSeekProvider(BaseProvider):
    """
    Provider for DeepSeek API.
    
    Features:
    - Ultra-cheap pricing ($0.28/MTok vs $3/MTok for Sonnet)
    - Native caching support (10x cheaper on cache hits)
    - High quality code generation (~92% of Claude Sonnet)
    - OpenAI-compatible API format
    """
    
    name = "deepseek"
    
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key (or from DEEPSEEK_API_KEY env)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY env var.")
        
        self.base_url = "https://api.deepseek.com"
        self.timeout = timeout
        
        # HTTP client
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True,
        )
        
        logger.debug("Initialized DeepSeek provider")
    
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str = "deepseek-chat",
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate completion with DeepSeek API.
        
        Args:
            messages: List of messages
            model: Model name (deepseek-chat, deepseek-coder)
            system: System prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional API parameters
        """
        # Prepare messages
        api_messages = self._prepare_messages(messages, system)
        
        # Build request payload
        payload = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs,
        }
        
        # Send request
        start_time = time.time()
        
        headers = self._get_headers()
        
        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"DeepSeek API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # DeepSeek-specific: cache hit tokens
        cached_tokens = usage.get("prompt_cache_hit_tokens", 0)
        cache_miss_tokens = usage.get("prompt_cache_miss_tokens", 0)
        
        # Calculate cost with cache savings
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
        model: str = "deepseek-chat",
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream completion with DeepSeek API."""
        # Prepare messages
        api_messages = self._prepare_messages(messages, system)
        
        # Build request payload
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
                
                data_str = line[6:]
                
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
        """
        Calculate cost in USD.
        
        DeepSeek has special pricing for cached tokens:
        - Cache hit: $0.028/MTok (10x cheaper than regular input)
        - Cache miss: $0.28/MTok (regular input price)
        """
        pricing = self.get_pricing(model)
        return pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)
    
    def get_pricing(self, model: str) -> ProviderPricing:
        """Get pricing for model."""
        if model in DEEPSEEK_PRICING:
            return DEEPSEEK_PRICING[model]
        
        # Default to deepseek-chat pricing
        return DEEPSEEK_PRICING["deepseek-chat"]
    
    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> list[dict]:
        """Prepare messages for DeepSeek API."""
        api_messages = []
        
        # Add system message if provided
        if system:
            api_messages.append({
                "role": "system",
                "content": system,
            })
        
        # Add user messages
        for msg in messages:
            api_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        
        return api_messages
    
    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    async def health_check(self) -> bool:
        """Check if API is available."""
        try:
            response = await self._client.get(
                "/models",
                headers=self._get_headers(),
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()
    
    @staticmethod
    def estimate_savings_vs_claude(
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        """
        Estimate cost savings vs Claude Sonnet.
        
        Returns:
            Dict with deepseek_cost, claude_cost, savings, savings_percent
        """
        # DeepSeek pricing
        ds_cost = (input_tokens * 0.28 + output_tokens * 0.42) / 1_000_000
        
        # Claude Sonnet pricing
        claude_cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
        
        savings = claude_cost - ds_cost
        savings_percent = (savings / claude_cost * 100) if claude_cost > 0 else 0
        
        return {
            "deepseek_cost": ds_cost,
            "claude_cost": claude_cost,
            "savings": savings,
            "savings_percent": savings_percent,
        }
