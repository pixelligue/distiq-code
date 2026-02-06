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

    # Model pricing (approximate, per 1M tokens)
    MODEL_COSTS = {
        "claude-4.5-opus-high": 15.0,
        "claude-4.5-opus": 15.0,
        "claude-4.5-sonnet": 3.0,
        "claude-4-sonnet": 3.0,
        "gpt-4o": 5.0,
        "gpt-4": 30.0,
        "gpt-4o-mini": 0.15,
        "cursor-small": 0.0,  # Free on Cursor
        "deepseek-v3": 0.0,  # Free on Cursor
        "gemini-2.5-flash": 0.0,  # Free on Cursor
    }

    # Free models available on Cursor
    FREE_MODELS = ["cursor-small", "deepseek-v3", "gemini-2.5-flash", "gpt-4o-mini"]

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

        # Never downgrade if already using free model
        if original_model in self.FREE_MODELS:
            return original_model, "already free"

        # Routing rules
        if complexity == "simple":
            # Simple questions → cursor-small (free)
            return "cursor-small", "simple question"

        elif complexity == "medium":
            # Medium tasks (tool-use, code reading) → cursor-small
            return "cursor-small", "medium task (tool-use)"

        elif complexity == "complex":
            # Complex tasks (code generation) → keep original
            # But if it's super expensive (opus/gpt-4), downgrade to sonnet
            if original_model in ("claude-4.5-opus-high", "claude-4.5-opus", "gpt-4"):
                return "claude-4.5-sonnet", "complex task (downgrade to sonnet)"
            else:
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
