"""Claude CLI subprocess provider for OAuth authentication."""

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Literal

from loguru import logger
from pydantic import BaseModel

ModelAlias = Literal["opus", "sonnet", "haiku"]


class ClaudeMessage(BaseModel):
    """Claude CLI output message."""

    role: Literal["assistant"]
    content: str


class ClaudeStreamChunk(BaseModel):
    """Streaming chunk from Claude CLI."""

    type: Literal["content_delta", "message_complete"]
    delta: str | None = None


class StreamResponse:
    """
    Async-iterable wrapper around Claude CLI streaming.

    Yields text chunks (str) and captures usage data from the `result` event.
    After iteration completes, real token/cost/latency data is available.
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.cache_creation_tokens: int = 0
        self.cost_usd: float = 0.0
        self.duration_ms: int = 0
        self.duration_api_ms: int = 0
        self._gen: AsyncGenerator[str | "ToolUseEvent", None] | None = None

    def __aiter__(self):
        return self._gen.__aiter__()

    async def __anext__(self) -> "str | ToolUseEvent":
        if self._gen is None:
            raise StopAsyncIteration
        return await self._gen.__anext__()


@dataclass
class ToolUseEvent:
    """Emitted when Claude CLI invokes a tool."""

    name: str  # "Read", "Bash", "Glob", etc.
    input: dict  # tool input parameters (complete)


class ClaudeCliProvider:
    """
    Provider for Claude CLI subprocess integration.

    Uses the official Claude CLI with OAuth authentication from Claude Max/Pro subscription.
    Requires: claude CLI installed and authenticated via `claude auth login`
    """

    def __init__(self):
        """Initialize Claude CLI provider."""
        self.cli_path = self._find_claude_cli()

    def _find_claude_cli(self) -> str:
        """
        Find claude CLI executable.

        Returns:
            Path to claude executable

        Raises:
            RuntimeError: If claude CLI not found
        """
        # Use shutil.which for cross-platform, Unicode-safe path resolution
        # (subprocess "where" on Windows mangles Cyrillic paths due to cp866/cp1251 mismatch)
        cli_path = shutil.which("claude")
        if cli_path:
            logger.debug(f"Found claude CLI at: {cli_path}")
            return cli_path

        raise RuntimeError(
            "Claude CLI not found. Install via: npm install -g claude-code-cli"
        )

    def _convert_openai_to_claude_prompt(
        self, messages: list[dict[str, str]]
    ) -> str:
        """
        Convert OpenAI messages to single Claude CLI prompt.

        Claude CLI in --print mode expects a single prompt string, not multi-turn conversation.
        We merge all messages into one readable format with XML tags for context.

        Args:
            messages: OpenAI-style messages [{"role": "system|user|assistant", "content": "..."}]

        Returns:
            Single prompt string for Claude CLI
        """
        prompt_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # System messages wrapped in <system> tags
                prompt_parts.append(f"<system>\n{content}\n</system>")
            elif role == "user":
                # User messages direct
                prompt_parts.append(content)
            elif role == "assistant":
                # Previous assistant responses for context
                prompt_parts.append(f"<previous_response>\n{content}\n</previous_response>")

        return "\n\n".join(prompt_parts)

    def _map_model_name(self, openai_model: str) -> ModelAlias:
        """
        Map OpenAI model name to Claude CLI alias.

        Args:
            openai_model: OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)

        Returns:
            Claude model alias (opus, sonnet, haiku)
        """
        model_lower = openai_model.lower()

        # Direct Claude names
        if "opus" in model_lower:
            return "opus"
        if "sonnet" in model_lower:
            return "sonnet"
        if "haiku" in model_lower:
            return "haiku"

        # Map OpenAI models to Claude equivalents
        if "gpt-4" in model_lower or "o1" in model_lower:
            return "opus"  # Flagship models
        if "gpt-3.5" in model_lower:
            return "sonnet"  # Mid-tier

        # Default to opus (best quality)
        logger.warning(f"Unknown model {openai_model}, defaulting to opus")
        return "opus"

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str = "opus",
        stream: bool = False,
        allowed_tools: list[str] | None = None,
    ) -> "str | StreamResponse":
        """
        Send chat completion request via Claude CLI.

        Args:
            messages: OpenAI-style messages
            model: Model name (will be mapped to opus/sonnet/haiku)
            stream: Enable streaming response
            allowed_tools: Tools the agent can use (e.g. ["Read", "Glob", "Grep", "Bash"])

        Returns:
            Complete response string, or StreamResponse (async iterable with usage data)
        """
        # Convert format
        prompt = self._convert_openai_to_claude_prompt(messages)
        claude_model = self._map_model_name(model)

        # Build CLI command (prompt passed via stdin to avoid Windows cmd line length limit)
        args = [
            self.cli_path,
            "--print",  # Non-interactive mode
            "--output-format",
            "stream-json",  # JSON streaming
            "--verbose",  # Required for stream-json
            "--include-partial-messages",  # Enable streaming chunks
            "--model",
            claude_model,
            "--no-session-persistence",  # Don't save session
        ]

        if allowed_tools:
            args.extend(["--allowedTools", ",".join(allowed_tools)])
            # In --print mode, CLI can't interactively ask for tool permissions,
            # so bypass checks. Tools are already restricted by --allowedTools.
            args.extend(["--permission-mode", "bypassPermissions"])

        logger.debug(f"Spawning claude CLI: {' '.join(args[:7])}...")

        if stream:
            return self._stream_response(args, prompt)
        else:
            return await self._complete_response(args, prompt)

    # 2 MB buffer for StreamReader — tool results (e.g. Read) can exceed the 64 KB default
    _STREAM_LIMIT = 2 * 1024 * 1024

    async def _complete_response(self, args: list[str], prompt: str) -> str:
        """
        Get complete response from Claude CLI.

        Args:
            args: CLI command arguments
            prompt: Prompt text to send via stdin

        Returns:
            Complete assistant response
        """
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            limit=self._STREAM_LIMIT,
        )

        # Send prompt via stdin (avoids Windows ~8191 char command line limit)
        stdout, stderr = await process.communicate(input=prompt.encode("utf-8"))

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"Claude CLI error: {error_msg}")
            raise RuntimeError(f"Claude CLI failed: {error_msg}")

        # Parse stream-json output — extract full result text
        for line in stdout.decode().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "result":
                    return data.get("result", "")
            except json.JSONDecodeError:
                continue

        logger.error("No result found in Claude CLI output")
        return ""

    def _stream_response(self, args: list[str], prompt: str) -> StreamResponse:
        """
        Create a StreamResponse that yields text chunks and captures usage data.

        Args:
            args: CLI command arguments
            prompt: Prompt text to send via stdin

        Returns:
            StreamResponse — async iterable of text chunks with usage metadata
        """
        sr = StreamResponse()
        sr._gen = self._generate_chunks(args, prompt, sr)
        return sr

    async def _generate_chunks(
        self, args: list[str], prompt: str, sr: StreamResponse
    ) -> AsyncGenerator[str | ToolUseEvent, None]:
        """
        Async generator that yields text deltas / ToolUseEvents and populates StreamResponse usage data.

        Args:
            args: CLI command arguments
            prompt: Prompt text to send via stdin
            sr: StreamResponse to populate with result event data

        Yields:
            Text chunks (str) or ToolUseEvent when the agent invokes a tool
        """
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            limit=self._STREAM_LIMIT,
        )

        # Send prompt via stdin then close (avoids Windows cmd line limit)
        if process.stdin:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()

        # Stream stdout line by line — extract text deltas and result event
        if process.stdout:
            async for line in process.stdout:
                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)

                    if data.get("type") == "stream_event":
                        event = data.get("event", {})
                        if event.get("type") == "content_block_delta":
                            text = event.get("delta", {}).get("text", "")
                            if text:
                                yield text

                    elif data.get("type") == "assistant":
                        # Full assistant message — extract completed tool_use blocks
                        for block in data.get("message", {}).get("content", []):
                            if block.get("type") == "tool_use":
                                yield ToolUseEvent(
                                    name=block["name"],
                                    input=block.get("input", {}),
                                )

                    elif data.get("type") == "result":
                        # Capture real usage data from Claude CLI
                        usage = data.get("usage", {})
                        sr.input_tokens = usage.get("input_tokens", 0)
                        sr.output_tokens = usage.get("output_tokens", 0)
                        sr.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                        sr.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
                        sr.cost_usd = data.get("total_cost_usd", 0.0)
                        sr.duration_ms = data.get("duration_ms", 0)
                        sr.duration_api_ms = data.get("duration_api_ms", 0)

                except json.JSONDecodeError:
                    continue

        # Wait for completion
        await process.wait()

        if process.returncode != 0:
            stderr_output = ""
            if process.stderr:
                stderr_output = (await process.stderr.read()).decode()
            logger.error(f"Claude CLI error: {stderr_output}")
