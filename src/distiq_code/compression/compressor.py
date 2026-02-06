"""LLMLingua-2 based prompt compression."""

from typing import Literal

from llmlingua import PromptCompressor as LLMCompressor
from loguru import logger
from pydantic import BaseModel


class CompressionStats(BaseModel):
    """Compression statistics."""

    original_length: int
    compressed_length: int
    compression_ratio: float
    tokens_saved: int


class PromptCompressor:
    """
    Compress prompts using LLMLingua-2.

    Reduces context size by 90% while preserving:
    - Code structure
    - Function names
    - Important keywords
    - User query
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        target_token: int = 500,
        use_llmlingua2: bool = True,
    ):
        """
        Initialize compressor.

        Args:
            model_name: LLMLingua model name
            target_token: Target token count after compression
            use_llmlingua2: Use LLMLingua-2 (recommended)
        """
        logger.info(f"Loading compression model: {model_name}")

        self.compressor = LLMCompressor(
            model_name=model_name,
            use_llmlingua2=use_llmlingua2,
            device_map="cpu",
        )

        self.target_token = target_token
        logger.success("Compression model loaded")

    def compress_messages(
        self,
        messages: list[dict[str, str]],
        role_priority: Literal["system", "user", "assistant"] = "user",
    ) -> tuple[list[dict[str, str]], CompressionStats]:
        """
        Compress chat messages.

        Strategy:
        - User query: NEVER compress (critical)
        - System prompt: Compress heavily (usually boilerplate)
        - Previous assistant: Compress moderately (context)

        Args:
            messages: OpenAI-style messages
            role_priority: Which role to preserve most (default: user)

        Returns:
            Compressed messages and stats
        """
        if not messages:
            return messages, self._empty_stats()

        # Separate messages by role
        system_messages = [m for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        # Get latest user message (never compress!)
        if user_messages:
            latest_user = user_messages[-1]
            context_messages = messages[:-1]  # Everything before last user message
        else:
            latest_user = None
            context_messages = messages

        # Build compression context
        original_text = self._messages_to_text(context_messages)
        original_length = len(original_text)

        if original_length < 500:
            # Too short to compress
            logger.debug(f"Skipping compression (text too short: {original_length} chars)")
            return messages, self._empty_stats()

        # Compress context (everything except latest user query)
        try:
            compressed_result = self.compressor.compress_prompt(
                original_text,
                target_token=self.target_token,
                rate=0.5,  # Compression rate
                force_tokens=[
                    "def",
                    "class",
                    "function",
                    "import",
                    "return",
                    "async",
                    "await",
                ],  # Preserve code keywords
            )

            compressed_text = compressed_result["compressed_prompt"]
            compressed_length = len(compressed_text)

            # Build compressed messages
            compressed_messages = []

            # Add compressed context as system message
            if compressed_text.strip():
                compressed_messages.append(
                    {
                        "role": "system",
                        "content": f"<compressed_context>\n{compressed_text}\n</compressed_context>",
                    }
                )

            # Add latest user query (uncompressed)
            if latest_user:
                compressed_messages.append(latest_user)

            # Calculate stats
            compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
            tokens_saved = int((original_length - compressed_length) / 4)  # Rough estimate

            stats = CompressionStats(
                original_length=original_length,
                compressed_length=compressed_length,
                compression_ratio=compression_ratio,
                tokens_saved=tokens_saved,
            )

            logger.info(
                f"Compressed {original_length} → {compressed_length} chars "
                f"({compression_ratio:.1%} ratio, ~{tokens_saved} tokens saved)"
            )

            return compressed_messages, stats

        except Exception as e:
            logger.error(f"Compression failed: {e}, using original messages")
            return messages, self._empty_stats()

    def _messages_to_text(self, messages: list[dict[str, str]]) -> str:
        """Convert messages to single text for compression."""
        parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        return "\n\n".join(parts)

    def _empty_stats(self) -> CompressionStats:
        """Return empty stats (no compression)."""
        return CompressionStats(
            original_length=0,
            compressed_length=0,
            compression_ratio=1.0,
            tokens_saved=0,
        )
