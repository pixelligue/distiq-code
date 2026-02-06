"""
Anthropic API Provider

Direct access to Claude models via Anthropic API.
Supports prompt caching for significant cost savings.

Usage:
    provider = AnthropicProvider(api_key="your-key")
    
    response = await provider.generate(
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-sonnet-4-20250514",
        enable_caching=True,
    )
"""

import os
import time
from typing import AsyncGenerator, Any

import httpx
from loguru import logger

from .base import BaseProvider, ProviderPricing, ProviderResponse


# Anthropic pricing ($ per 1M tokens) - February 2026
# https://www.anthropic.com/pricing
ANTHROPIC_PRICING = {
    # Claude 4.5 (latest)
    "claude-sonnet-4-20250514": ProviderPricing(3.0, 15.0, 0.30),
    "claude-4-sonnet-20250514": ProviderPricing(3.0, 15.0, 0.30),  # alias
    "claude-opus-4-20250514": ProviderPricing(15.0, 75.0, 1.50),
    "claude-4-opus-20250514": ProviderPricing(15.0, 75.0, 1.50),  # alias
    
    # Claude 3.5
    "claude-3-5-sonnet-20241022": ProviderPricing(3.0, 15.0, 0.30),
    "claude-3-5-haiku-20241022": ProviderPricing(1.0, 5.0, 0.10),
    
    # Claude 3
    "claude-3-opus-20240229": ProviderPricing(15.0, 75.0, 1.50),
    "claude-3-sonnet-20240229": ProviderPricing(3.0, 15.0, 0.30),
    "claude-3-haiku-20240307": ProviderPricing(0.25, 1.25, 0.025),
    
    # Aliases for convenience
    "sonnet": ProviderPricing(3.0, 15.0, 0.30),
    "opus": ProviderPricing(15.0, 75.0, 1.50),
    "haiku": ProviderPricing(1.0, 5.0, 0.10),
}

# Model aliases to full names
MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
    "haiku": "claude-3-5-haiku-20241022",
}


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic Claude API.
    
    Features:
    - Direct API access (requires API key)
    - Prompt caching support for 90% input cost reduction
    - Streaming support
    - All Claude models
    """
    
    name = "anthropic"
    
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")
        
        self.base_url = "https://api.anthropic.com"
        self.timeout = timeout
        
        # HTTP client
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True,
        )
        
        logger.debug("Initialized Anthropic provider")
    
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str = "sonnet",
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate completion with Anthropic API.
        
        Args:
            messages: List of messages
            model: Model name (sonnet, opus, haiku, or full model ID)
            system: System prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            enable_caching: Enable prompt caching for system prompt
            **kwargs: Additional API parameters
        """
        # Resolve model alias
        model_id = MODEL_ALIASES.get(model, model)
        
        # Prepare messages
        api_messages = self._prepare_messages(messages)
        
        # Build request payload
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add system prompt with caching
        if system:
            if enable_caching:
                # Use cache_control for system prompt
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system
        
        # Send request
        start_time = time.time()
        
        headers = self._get_headers()
        
        try:
            response = await self._client.post(
                "/v1/messages",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Anthropic API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
        
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        # Cache tokens (Anthropic returns these separately)
        cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
        cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
        cached_tokens = cache_read_input_tokens
        
        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model_id, cached_tokens)
        
        return ProviderResponse(
            content=content,
            model=model_id,
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
        model: str = "sonnet",
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream completion with Anthropic API."""
        # Resolve model alias
        model_id = MODEL_ALIASES.get(model, model)
        
        # Prepare messages
        api_messages = self._prepare_messages(messages)
        
        # Build request payload
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        # Add system prompt with caching
        if system:
            if enable_caching:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system
        
        headers = self._get_headers()
        
        async with self._client.stream(
            "POST",
            "/v1/messages",
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
                    
                    event_type = data.get("type")
                    
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text
                                
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
        if model in ANTHROPIC_PRICING:
            return ANTHROPIC_PRICING[model]
        
        # Try to find partial match
        model_lower = model.lower()
        if "opus" in model_lower:
            return ANTHROPIC_PRICING["opus"]
        elif "sonnet" in model_lower:
            return ANTHROPIC_PRICING["sonnet"]
        elif "haiku" in model_lower:
            return ANTHROPIC_PRICING["haiku"]
        
        # Default to Sonnet pricing
        return ANTHROPIC_PRICING["sonnet"]
    
    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict]:
        """Prepare messages for Anthropic API format."""
        api_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Skip system messages (handled separately)
            if role == "system":
                continue
            
            # Anthropic uses "user" and "assistant" roles
            if role not in ("user", "assistant"):
                role = "user"
            
            api_messages.append({
                "role": role,
                "content": content,
            })
        
        return api_messages
    
    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2024-01-01",
            "anthropic-beta": "prompt-caching-2024-07-31",  # Enable caching
        }
    
    async def health_check(self) -> bool:
        """Check if API is available."""
        try:
            # Simple test request
            response = await self._client.get(
                "/v1/messages",
                headers=self._get_headers(),
            )
            # Even 405 (method not allowed) means API is reachable
            return response.status_code in (200, 405)
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()
