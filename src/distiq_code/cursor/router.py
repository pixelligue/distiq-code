"""Smart routing logic for Cursor API requests.

Classifies request complexity and routes to appropriate model.
"""

import re
from typing import Optional

from loguru import logger


class CursorRouter:
    """
    Smart router for Cursor API requests.

    Routes requests to cheapest model that can handle the task:
    - Simple questions → cursor-small / gpt-4o-mini (free)
    - Code reading, tool-use → cursor-small (cheap)
    - Code generation → original model (claude/gpt-4)
    """

    # Model pricing (per 1M tokens input/output, Feb 2026)
    MODEL_COSTS = {
        # Anthropic Claude models
        "claude-4.6-opus": 15.0,  # $5/$25
        "claude-4.5-opus": 15.0,  # $5/$25
        "claude-4.5-sonnet": 9.0,  # $3/$15
        "claude-4-sonnet": 9.0,  # $3/$15
        "claude-4-sonnet-1m": 14.25,  # $6/$22.50 (1M context)
        "claude-4.5-haiku": 3.0,  # $1/$5

        # Google Gemini models
        "gemini-3-pro": 7.0,  # $2/$12
        "gemini-3-flash": 1.75,  # $0.50/$3
        "gemini-2.5-flash": 1.4,  # $0.30/$2.50
        "gemini-2.5-flash-thinking": 1.4,

        # OpenAI models
        "gpt-5.2": 7.875,  # $1.75/$14
        "gpt-5.1": 11.25,  # $2.50/$20
        "gpt-5.1-codex": 0.75,  # $0.25/$1.25 (coding optimized)
        "gpt-5-codex": 0.75,
        "gpt-4.1": 5.0,  # Legacy
        "gpt-4o": 5.0,
        "gpt-4": 30.0,

        # Cursor/Other models
        "composer-1": 5.625,  # $1.25/$10
        "grok-code": 0.85,  # $0.20/$1.50

        # FREE models (no API cost on Cursor)
        "cursor-small": 0.0,
        "deepseek-v3": 0.0,
        "deepseek-v3.1": 0.0,
        "gpt-4o-mini": 0.0,  # 500 req/day limit on Free plan
        "auto": 3.625,  # Average: $1.25/$6
    }

    # Free models available on Cursor (no credit usage)
    FREE_MODELS = [
        "cursor-small",
        "deepseek-v3",
        "deepseek-v3.1",
        "gpt-4o-mini",
        "gemini-2.5-flash",  # Ultra cheap, nearly free
    ]

    # Best models for different tasks (updated Feb 2026)
    BEST_MODELS = {
        "simple": "cursor-small",  # Free, good for Q&A
        "medium": "deepseek-v3.1",  # Free, fast, good for code reading
        "complex": "claude-4.5-sonnet",  # Best quality/price for code generation
        "reasoning": "gpt-5.2",  # For complex logic
        "speed": "gemini-3-flash",  # Ultra fast responses
    }

    # Simple question patterns
    SIMPLE_PATTERNS = [
        r"^(what|how|why|when|where|who|explain|tell me|describe)",
        r"(what is|what are|how does|how do)",
        r"^(yes|no|ok|thanks|hello|hi)\b",
        r"^\w+\?$",  # Single word question
    ]

    # Code generation patterns
    CODE_GEN_PATTERNS = [
        r"(write|create|implement|build|generate|add|refactor)",
        r"(function|class|component|module|method)",
        r"(fix|debug|solve|correct)",
        r"(test|unit test|integration test)",
    ]

    def __init__(self):
        """Initialize router."""
        self.simple_regex = re.compile("|".join(self.SIMPLE_PATTERNS), re.IGNORECASE)
        self.codegen_regex = re.compile("|".join(self.CODE_GEN_PATTERNS), re.IGNORECASE)

    def classify_request(self, messages: list[str]) -> str:
        """
        Classify request complexity.

        Args:
            messages: List of user message contents

        Returns:
            Complexity level: "simple", "medium", "complex"
        """
        if not messages:
            return "medium"

        # Use last message for classification
        last_message = messages[-1].lower()

        # Check for simple patterns
        if self.simple_regex.search(last_message):
            return "simple"

        # Check for code generation patterns
        if self.codegen_regex.search(last_message):
            return "complex"

        # Medium complexity (tool-use, code reading, etc.)
        return "medium"

    def route_model(
        self,
        original_model: str,
        complexity: str,
        enable_routing: bool = True,
    ) -> tuple[str, str]:
        """
        Route to appropriate model based on complexity.

        Args:
            original_model: Model requested by Cursor
            complexity: Request complexity (simple/medium/complex)
            enable_routing: If False, return original model

        Returns:
            Tuple of (routed_model, reason)
        """
        if not enable_routing:
            return original_model, "routing disabled"

        # Never downgrade if already using free/cheap model
        if original_model in self.FREE_MODELS:
            return original_model, "already free"

        original_cost = self.MODEL_COSTS.get(original_model, 999.0)
        if original_cost < 2.0:  # Already cheap
            return original_model, "already cheap"

        # Routing rules (updated Feb 2026)
        if complexity == "simple":
            # Simple questions → cursor-small (free, unlimited)
            return "cursor-small", "simple question"

        elif complexity == "medium":
            # Medium tasks (tool-use, code reading) → deepseek-v3.1 (free, fast)
            # Alternative: cursor-small if deepseek unavailable
            return "deepseek-v3.1", "medium task (code reading)"

        elif complexity == "complex":
            # Complex tasks (code generation) → best quality/price ratio
            # Route expensive models to claude-4.5-sonnet or gpt-5.1-codex

            if original_model in ("claude-4.6-opus", "claude-4.5-opus", "gpt-5.1", "gpt-5.2", "gpt-4"):
                # Super expensive → claude-4.5-sonnet (best for code)
                return "claude-4.5-sonnet", "complex task (downgrade to sonnet)"

            elif original_model in ("gemini-3-pro", "composer-1"):
                # Medium-expensive → gpt-5.1-codex (optimized for code, cheaper)
                return "gpt-5.1-codex", "complex task (codex optimized)"

            else:
                # Already reasonable price, keep it
                return original_model, "complex task (keep original)"

        return original_model, "unknown complexity"

    def calculate_savings(
        self,
        original_model: str,
        routed_model: str,
        estimated_tokens: int = 1000,
    ) -> float:
        """
        Calculate cost savings from routing.

        Args:
            original_model: Original model
            routed_model: Routed model
            estimated_tokens: Estimated tokens (default 1000)

        Returns:
            Savings in USD
        """
        original_cost = self.MODEL_COSTS.get(original_model, 0.0)
        routed_cost = self.MODEL_COSTS.get(routed_model, 0.0)

        # Cost per 1M tokens → cost per request
        original_request_cost = (original_cost / 1_000_000) * estimated_tokens
        routed_request_cost = (routed_cost / 1_000_000) * estimated_tokens

        savings = original_request_cost - routed_request_cost

        return max(savings, 0.0)  # Never negative

    def format_routing_log(
        self,
        original_model: str,
        routed_model: str,
        reason: str,
        savings: float,
    ) -> str:
        """
        Format routing decision for logging.

        Args:
            original_model: Original model
            routed_model: Routed model
            reason: Routing reason
            savings: Cost savings (USD)

        Returns:
            Formatted log message
        """
        if original_model == routed_model:
            return f"Model: {original_model} (no routing)"

        return (
            f"Model: {original_model} -> {routed_model} "
            f"({reason}) | "
            f"Saved ~${savings:.4f}"
        )
