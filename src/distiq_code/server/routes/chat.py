"""OpenAI-compatible chat completion endpoints."""

import json
import time
import uuid
from typing import Any, AsyncGenerator, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from distiq_code.auth.cli_provider import ClaudeCliProvider
from distiq_code.config import settings
from distiq_code.stats import StatsTracker

router = APIRouter()

# Global instances (lazy initialization)
_compressor = None
_compressor_unavailable = False
_cache = None
_cache_unavailable = False
_stats: StatsTracker | None = None


def get_compressor():
    """Get or initialize compressor. Returns None if deps missing."""
    global _compressor, _compressor_unavailable
    if _compressor_unavailable:
        return None
    if _compressor is None:
        try:
            from distiq_code.compression import PromptCompressor
            logger.info("Initializing prompt compressor...")
            _compressor = PromptCompressor(
                model_name=settings.compression_model,
                target_token=settings.compression_target_tokens,
            )
        except (ImportError, Exception) as e:
            logger.info(f"Prompt compressor unavailable: {e}")
            _compressor_unavailable = True
            return None
    return _compressor


def get_cache():
    """Get or initialize semantic cache. Returns None if deps missing."""
    global _cache, _cache_unavailable
    if _cache_unavailable:
        return None
    if _cache is None:
        try:
            from distiq_code.cache import SemanticCache
            logger.info("Initializing semantic cache...")
            _cache = SemanticCache(
                cache_dir=settings.cache_dir_path,
                model_name=settings.cache_embedding_model,
                similarity_threshold=settings.cache_similarity_threshold,
                max_cache_size=settings.cache_max_size,
                ttl_seconds=settings.cache_ttl_hours * 3600,
            )
        except (ImportError, Exception) as e:
            logger.info(f"Semantic cache unavailable: {e}")
            _cache_unavailable = True
            return None
    return _cache


def get_stats() -> StatsTracker:
    """Get or initialize stats tracker."""
    global _stats
    if _stats is None:
        logger.info("Initializing stats tracker...")
        _stats = StatsTracker(stats_file=settings.stats_db_file)
    return _stats


# OpenAI-compatible schemas
class Message(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str = Field(default="gpt-4", description="Model to use")
    messages: list[Message] = Field(..., description="Chat messages")
    stream: bool = Field(default=False, description="Enable streaming")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int] | None = None


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint with optimization.

    Features:
    - Prompt compression (LLMLingua-2)
    - Semantic caching (FAISS)
    - Claude CLI OAuth authentication
    - Statistics tracking
    """
    start_time = time.time()

    logger.info(
        f"Chat completion request: model={request.model}, "
        f"messages={len(request.messages)}, stream={request.stream}"
    )

    # Convert messages to dict format
    messages_dict = [msg.model_dump() for msg in request.messages]

    # Build query for caching (last user message)
    user_messages = [m for m in messages_dict if m["role"] == "user"]
    query_text = user_messages[-1]["content"] if user_messages else ""

    # Original token estimate (rough: 1 token = 4 chars)
    original_text = "\n".join(m["content"] for m in messages_dict)
    original_tokens = len(original_text) // 4

    # Phase 2.1: Check cache
    cache_hit = False
    cache_similarity = 0.0
    cached_response = None

    if settings.cache_enabled and query_text:
        cache = get_cache()
        if cache is not None:
            cached_response, cache_stats = cache.get(query_text, request.model)

        if cached_response:
            cache_hit = True
            cache_similarity = cache_stats.similarity
            logger.info(
                f"Cache HIT (similarity={cache_similarity:.2f}): "
                f"Saved {cache_stats.tokens_saved} tokens"
            )

            # Record stats
            latency_ms = (time.time() - start_time) * 1000
            stats = get_stats()
            stats.record_request(
                model=request.model,
                original_tokens=original_tokens,
                compressed_tokens=0,  # From cache, no compression needed
                cache_hit=True,
                cache_similarity=cache_similarity,
                latency_ms=latency_ms,
                compression_enabled=False,
                compression_ratio=1.0,
            )

            # Return cached response
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cached_response,
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": 0,  # From cache
                    "completion_tokens": len(cached_response) // 4,
                    "total_tokens": len(cached_response) // 4,
                },
            )

    # Phase 2.2: Compress prompt
    compressed_tokens = original_tokens
    compression_ratio = 1.0

    if settings.compression_enabled and len(messages_dict) > 1:
        compressor = get_compressor()
        if compressor is not None:
            messages_dict, compression_stats = compressor.compress_messages(messages_dict)

            compressed_tokens = compression_stats.compressed_length // 4
            compression_ratio = compression_stats.compression_ratio

            logger.info(
                f"Compressed: {original_tokens} → {compressed_tokens} tokens "
                f"({compression_ratio:.1%} ratio, saved ~{compression_stats.tokens_saved} tokens)"
        )

    # Initialize Claude CLI provider
    try:
        provider = ClaudeCliProvider()
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Claude CLI not available: {str(e)}. "
            "Install via: npm install -g claude-code-cli && claude auth login",
        )

    # Forward to Claude CLI
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(
                provider,
                messages_dict,
                request.model,
                query_text,
                original_tokens,
                compressed_tokens,
                compression_ratio,
                start_time,
            ),
            media_type="text/event-stream",
        )
    else:
        return await forward_chat_completion(
            provider,
            messages_dict,
            request.model,
            query_text,
            original_tokens,
            compressed_tokens,
            compression_ratio,
            start_time,
        )


async def forward_chat_completion(
    provider: ClaudeCliProvider,
    messages: list[dict[str, str]],
    model: str,
    query_text: str,
    original_tokens: int,
    compressed_tokens: int,
    compression_ratio: float,
    start_time: float,
) -> ChatCompletionResponse:
    """
    Forward non-streaming request to Claude CLI.

    Args:
        provider: Claude CLI provider instance
        messages: OpenAI-style messages (possibly compressed)
        model: Model name
        query_text: Original query for caching
        original_tokens: Original token count
        compressed_tokens: Compressed token count
        compression_ratio: Compression ratio
        start_time: Request start time

    Returns:
        OpenAI-compatible response
    """
    try:
        # Get complete response from Claude CLI
        content = await provider.complete(messages, model, stream=False)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Cache response
        if settings.cache_enabled and query_text:
            cache = get_cache()
            if cache is not None:
                cache.set(query_text, content, model)
                logger.debug("Response cached")

        # Record stats
        stats = get_stats()
        stats.record_request(
            model=model,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            cache_hit=False,
            cache_similarity=0.0,
            latency_ms=latency_ms,
            compression_enabled=settings.compression_enabled,
            compression_ratio=compression_ratio,
        )

        # Build OpenAI-compatible response
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_at = int(time.time())

        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created_at,
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": compressed_tokens,
                "completion_tokens": len(content) // 4,  # Rough estimate
                "total_tokens": compressed_tokens + len(content) // 4,
            },
        )

    except RuntimeError as e:
        logger.error(f"Claude CLI error: {e}")
        raise HTTPException(status_code=502, detail=f"Claude CLI error: {str(e)}")


async def stream_chat_completion(
    provider: ClaudeCliProvider,
    messages: list[dict[str, str]],
    model: str,
    query_text: str,
    original_tokens: int,
    compressed_tokens: int,
    compression_ratio: float,
    start_time: float,
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion from Claude CLI.

    Args:
        provider: Claude CLI provider instance
        messages: OpenAI-style messages (possibly compressed)
        model: Model name
        query_text: Original query for caching
        original_tokens: Original token count
        compressed_tokens: Compressed token count
        compression_ratio: Compression ratio
        start_time: Request start time

    Yields:
        OpenAI-compatible SSE chunks
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_at = int(time.time())

    # Accumulate response for caching
    full_response = []

    try:
        # Get streaming response from Claude CLI
        response_stream = await provider.complete(messages, model, stream=True)

        # Stream OpenAI-compatible chunks
        async for chunk in response_stream:
            full_response.append(chunk)

            # Build SSE chunk
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send [DONE] marker
        yield "data: [DONE]\n\n"

        # After streaming complete: cache response & record stats
        complete_response = "".join(full_response)
        latency_ms = (time.time() - start_time) * 1000

        # Cache response
        if settings.cache_enabled and query_text and complete_response:
            cache = get_cache()
            if cache is not None:
                cache.set(query_text, complete_response, model)
                logger.debug("Streamed response cached")

        # Record stats
        stats = get_stats()
        stats.record_request(
            model=model,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            cache_hit=False,
            cache_similarity=0.0,
            latency_ms=latency_ms,
            compression_enabled=settings.compression_enabled,
            compression_ratio=compression_ratio,
        )

    except RuntimeError as e:
        logger.error(f"Claude CLI stream error: {e}")
        error_data = {
            "error": {
                "message": str(e),
                "type": "claude_cli_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
