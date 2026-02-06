"""
Base Provider Interface

All providers must implement this interface for unified usage across the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any


@dataclass
class ProviderPricing:
    """Pricing information for a provider/model combination."""
    
    input_cost_per_mtok: float  # $ per 1M input tokens
    output_cost_per_mtok: float  # $ per 1M output tokens
    cached_input_cost_per_mtok: float = 0.0  # $ per 1M cached input tokens
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate total cost in USD."""
        input_cost = (input_tokens - cached_tokens) * self.input_cost_per_mtok / 1_000_000
        cached_cost = cached_tokens * self.cached_input_cost_per_mtok / 1_000_000
        output_cost = output_tokens * self.output_cost_per_mtok / 1_000_000
        return input_cost + cached_cost + output_cost


@dataclass
class ProviderResponse:
    """Standardized response from any provider."""
    
    content: str
    model: str
    provider: str
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    
    # Cost tracking
    cost_usd: float = 0.0
    
    # Timing
    latency_ms: float = 0.0
    
    # Raw response for debugging
    raw_response: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
            },
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Implementations must support:
    - generate(): Single-shot completion
    - stream(): Streaming completion
    - calculate_cost(): Cost estimation
    """
    
    name: str = "base"
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate a completion.
        
        Args:
            messages: List of messages in OpenAI format:
                [{"role": "user", "content": "..."}]
            model: Model name/ID
            system: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options
            
        Returns:
            ProviderResponse with content, usage, and cost
        """
        ...
    
    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a completion.
        
        Args:
            Same as generate()
            
        Yields:
            Text chunks as they arrive
        """
        ...
    
    @abstractmethod
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float:
        """
        Calculate cost in USD for a request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model used
            cached_tokens: Number of tokens served from cache
            
        Returns:
            Cost in USD
        """
        ...
    
    def get_pricing(self, model: str) -> ProviderPricing:
        """Get pricing for a specific model. Override in subclasses."""
        return ProviderPricing(
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
            cached_input_cost_per_mtok=0.0,
        )
    
    async def health_check(self) -> bool:
        """Check if provider is available. Override for custom checks."""
        return True
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
