"""
History Summarizer

Automatically compresses conversation history to save tokens while
preserving important context. Uses a cheap model (Haiku) to 
generate summaries.

Key strategies:
1. Summarize old messages (keep last 5 detailed)
2. Extract key decisions and code changes
3. Compress to ~500 tokens from 5000+
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class SummarizedHistory:
    """Result of history summarization."""
    
    # Original message count
    original_count: int
    original_tokens: int
    
    # Summarized result
    summary: str
    summary_tokens: int
    
    # Recent messages kept in full
    recent_messages: list[dict]
    recent_count: int
    
    # Savings
    tokens_saved: int
    compression_ratio: float


class HistorySummarizer:
    """
    Summarize conversation history to reduce token usage.
    
    Strategy:
    - Keep last N messages in full detail
    - Summarize older messages into a context block
    - Extract: key decisions, code changes, file edits
    """
    
    # How many recent messages to keep in full
    KEEP_RECENT = 5
    
    # Chars per token estimate
    CHARS_PER_TOKEN = 4
    
    # Summary prompt template
    SUMMARY_PROMPT = """Summarize this conversation history into a brief context block.

Focus on:
- Key decisions made
- Files created or modified
- Important code changes
- Current task/goal

Keep the summary under 500 tokens. Be concise.

Conversation:
{history}

Summary:"""

    def __init__(
        self,
        keep_recent: int = 5,
        provider: Any = None,
    ):
        """
        Initialize summarizer.
        
        Args:
            keep_recent: Number of recent messages to keep in full
            provider: LLM provider for generating summaries
        """
        self.keep_recent = keep_recent
        self._provider = provider
    
    async def summarize(
        self,
        messages: list[dict],
        force: bool = False,
    ) -> SummarizedHistory:
        """
        Summarize conversation history.
        
        Args:
            messages: Full conversation history
            force: Force summarization even if not needed
            
        Returns:
            SummarizedHistory with compressed history
        """
        original_count = len(messages)
        original_tokens = self._estimate_tokens(messages)
        
        # Check if summarization is needed
        if not force and original_count <= self.keep_recent + 2:
            logger.debug("History too short for summarization")
            return SummarizedHistory(
                original_count=original_count,
                original_tokens=original_tokens,
                summary="",
                summary_tokens=0,
                recent_messages=messages.copy(),
                recent_count=original_count,
                tokens_saved=0,
                compression_ratio=1.0,
            )
        
        # Split into old and recent
        old_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]
        
        # Generate summary of old messages
        summary = await self._generate_summary(old_messages)
        summary_tokens = len(summary) // self.CHARS_PER_TOKEN
        
        # Calculate savings
        old_tokens = self._estimate_tokens(old_messages)
        tokens_saved = old_tokens - summary_tokens
        
        recent_tokens = self._estimate_tokens(recent_messages)
        total_new_tokens = summary_tokens + recent_tokens
        compression_ratio = original_tokens / total_new_tokens if total_new_tokens > 0 else 1.0
        
        logger.info(
            f"Summarized history: {original_count} messages → {len(recent_messages)} + summary, "
            f"saved {tokens_saved} tokens ({compression_ratio:.1f}x compression)"
        )
        
        return SummarizedHistory(
            original_count=original_count,
            original_tokens=original_tokens,
            summary=summary,
            summary_tokens=summary_tokens,
            recent_messages=recent_messages,
            recent_count=len(recent_messages),
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
        )
    
    async def _generate_summary(self, messages: list[dict]) -> str:
        """Generate summary using LLM."""
        # Format messages for summary
        history_text = self._format_messages(messages)
        
        # If no provider, use extractive summary
        if self._provider is None:
            return self._extractive_summary(messages)
        
        # Generate with LLM
        prompt = self.SUMMARY_PROMPT.format(history=history_text)
        
        try:
            response = await self._provider.generate(
                messages=[{"role": "user", "content": prompt}],
                model="haiku",  # Use cheap model
                max_tokens=600,
            )
            return response.content
        except Exception as e:
            logger.warning(f"LLM summary failed, using extractive: {e}")
            return self._extractive_summary(messages)
    
    def _extractive_summary(self, messages: list[dict]) -> str:
        """
        Create summary without LLM using extraction.
        
        Extracts:
        - File names mentioned
        - Code blocks (first lines only)
        - Key action words
        """
        import re
        
        summary_parts = ["## Previous Context\n"]
        
        files_mentioned = set()
        actions = []
        code_snippets = []
        
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue
            
            # Extract file paths
            file_patterns = [
                r'`([^`]+\.\w{2,4})`',  # `file.py`
                r'"([^"]+\.\w{2,4})"',   # "file.py"
                r'(\S+\.\w{2,4}):',      # file.py:
            ]
            for pattern in file_patterns:
                for match in re.findall(pattern, content):
                    if len(match) < 50:  # Reasonable file path
                        files_mentioned.add(match)
            
            # Extract action keywords
            action_words = [
                "created", "modified", "fixed", "added", "removed",
                "updated", "implemented", "refactored", "changed",
            ]
            for word in action_words:
                if word in content.lower():
                    # Get sentence containing the word
                    sentences = content.split(".")
                    for sentence in sentences:
                        if word in sentence.lower() and len(sentence) < 200:
                            actions.append(sentence.strip())
                            break
            
            # Extract first line of code blocks
            code_blocks = re.findall(r'```\w*\n(.+?)(?:\n|```)', content, re.DOTALL)
            for block in code_blocks[:3]:
                first_line = block.split("\n")[0].strip()
                if first_line and len(first_line) < 100:
                    code_snippets.append(first_line)
        
        # Build summary
        if files_mentioned:
            summary_parts.append(f"**Files touched:** {', '.join(sorted(files_mentioned)[:10])}\n")
        
        if actions:
            summary_parts.append("**Actions taken:**")
            for action in actions[:5]:
                summary_parts.append(f"- {action}")
            summary_parts.append("")
        
        if code_snippets:
            summary_parts.append("**Key code:**")
            for snippet in code_snippets[:3]:
                summary_parts.append(f"- `{snippet}`")
        
        return "\n".join(summary_parts)
    
    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for summary prompt."""
        parts = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Truncate very long messages
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            parts.append(f"[{role}]: {content}")
        
        return "\n\n".join(parts)
    
    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for messages."""
        total_chars = sum(
            len(msg.get("content", ""))
            for msg in messages
        )
        return total_chars // self.CHARS_PER_TOKEN
    
    def should_summarize(self, messages: list[dict]) -> bool:
        """Check if history should be summarized."""
        return len(messages) > self.keep_recent + 2
    
    def build_messages_with_summary(
        self,
        result: SummarizedHistory,
        new_message: dict | None = None,
    ) -> list[dict]:
        """
        Build message list with summary injected.
        
        Args:
            result: Summarization result
            new_message: Optional new user message to append
            
        Returns:
            Message list ready for LLM
        """
        messages = []
        
        # Add summary as first message if present
        if result.summary:
            messages.append({
                "role": "user",
                "content": f"[Previous conversation summary]\n\n{result.summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "I understand the context. Let's continue.",
            })
        
        # Add recent messages
        messages.extend(result.recent_messages)
        
        # Add new message if provided
        if new_message:
            messages.append(new_message)
        
        return messages


# Convenience function
async def compress_history(
    messages: list[dict],
    provider: Any = None,
) -> list[dict]:
    """
    Compress conversation history if needed.
    
    Args:
        messages: Full conversation history
        provider: Optional LLM provider for smart summarization
        
    Returns:
        Compressed message list
    """
    summarizer = HistorySummarizer(provider=provider)
    
    if not summarizer.should_summarize(messages):
        return messages
    
    result = await summarizer.summarize(messages)
    return summarizer.build_messages_with_summary(result)
