"""Anthropic Messages API proxy endpoint.

Transparent proxy for Claude Code: forwards requests to api.anthropic.com
with smart routing and semantic caching.

Usage:
    ANTHROPIC_BASE_URL=http://localhost:11434 claude
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from distiq_code.config import settings
from distiq_code.routing import classify_and_route, estimate_savings

router = APIRouter()

# ---------------------------------------------------------------------------
# Model tier mapping
# ---------------------------------------------------------------------------

TIER_TO_MODEL = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}


def _model_id_to_tier(model_id: str) -> str:
    """Map a full model ID to a tier name.

    >>> _model_id_to_tier("claude-opus-4-6")
    'opus'
    """
    model_lower = model_id.lower()
    for tier in ("opus", "sonnet", "haiku"):
        if tier in model_lower:
            return tier
    return "sonnet"  # safe default


def _tier_to_model_id(tier: str, original_model: str) -> str:
    """Convert tier name back to a concrete model ID.

    If the tier matches the original model's tier, return original unchanged.
    """
    if _model_id_to_tier(original_model) == tier:
        return original_model
    return TIER_TO_MODEL.get(tier, original_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL
)


def _extract_latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract text from the latest user message.

    Strips <system-reminder> tags that Claude Code injects into user messages,
    and skips tool_result blocks.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            cleaned = _SYSTEM_REMINDER_RE.sub("", content).strip()
            if cleaned:
                return cleaned
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        cleaned = _SYSTEM_REMINDER_RE.sub("", text).strip()
                        if cleaned:
                            parts.append(cleaned)
                    elif block.get("type") == "tool_result":
                        continue
            if parts:
                return " ".join(parts)
    return ""


def _has_tool_messages(body: dict[str, Any]) -> bool:
    """Check if messages contain actual tool_use/tool_result blocks.

    This is different from having tools *defined* in the request.
    Claude Code always sends tools list, but only interactive
    tool exchanges should skip caching.
    """
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "tool_use",
                    "tool_result",
                ):
                    return True
    return False


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    """Format a single SSE event as bytes."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _build_cached_response(
    text: str, model: str, input_tokens: int
) -> dict[str, Any]:
    """Build a synthetic Anthropic Messages response from cached text."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": max(1, len(text) // 4),
        },
    }


async def _build_cached_stream(
    text: str, model: str, input_tokens: int
):
    """Yield SSE events that reconstruct a cached response as a stream."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # message_start
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    # content_block_start
    yield _sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    # content_block_delta — send text in one chunk (fast)
    yield _sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    })

    # content_block_stop
    yield _sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    })

    output_tokens = max(1, len(text) // 4)

    # message_delta
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    # message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Lazy service accessors (reuse from chat.py pattern)
# ---------------------------------------------------------------------------

_cache = None
_cache_unavailable = False
_stats = None


def _get_cache():
    global _cache, _cache_unavailable
    if _cache_unavailable:
        return None
    if _cache is None:
        try:
            from distiq_code.cache import SemanticCache
            logger.info("Initializing semantic cache (messages endpoint)...")
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


def _get_stats():
    global _stats
    if _stats is None:
        from distiq_code.stats import StatsTracker
        logger.info("Initializing stats tracker (messages endpoint)...")
        _stats = StatsTracker(stats_file=settings.stats_db_file)
    return _stats


# ---------------------------------------------------------------------------
# Thinking cleanup for model downgrades
# ---------------------------------------------------------------------------

def _strip_thinking(body: dict[str, Any]) -> None:
    """Remove thinking-related params when routing to a cheaper model."""
    body.pop("thinking", None)
    body.pop("temperature", None)
    body.pop("context_management", None)  # may contain clear_thinking strategy

    # Also strip thinking blocks from messages (if any thinking content exists)
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") == "thinking")
            ]


# ---------------------------------------------------------------------------
# Anthropic Prompt Caching
# ---------------------------------------------------------------------------

_CACHE_CONTROL = {"type": "ephemeral"}


def _has_existing_cache_control(body: dict[str, Any]) -> bool:
    """Check if the request already contains cache_control breakpoints.

    Claude Code (and other clients) may inject their own cache_control.
    Adding more would exceed the 4-breakpoint limit or break TTL ordering.
    """
    # Check tools
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            return True
    # Check system
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                return True
    # Check messages
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False


def _inject_cache_control(body: dict[str, Any]) -> None:
    """Inject cache_control breakpoints for Anthropic prompt caching.

    Places ephemeral breakpoints on:
    1. Last tool definition (tools are static per session)
    2. Last system prompt block (system prompt is static per session)
    3. Last content block of the last user message (conversation prefix)

    Anthropic caches everything up to the last breakpoint for 5 minutes.
    Cached prefix tokens get a 90% discount on input cost.

    Skips injection if the request already has cache_control (e.g. from Claude Code).
    """
    if _has_existing_cache_control(body):
        return

    # 1. Tools — last tool gets breakpoint
    tools = body.get("tools")
    if tools and isinstance(tools, list) and len(tools) > 0:
        tools[-1]["cache_control"] = _CACHE_CONTROL

    # 2. System prompt — normalize to list, mark last block
    system = body.get("system")
    if isinstance(system, str) and system:
        body["system"] = [
            {"type": "text", "text": system, "cache_control": _CACHE_CONTROL}
        ]
    elif isinstance(system, list) and len(system) > 0:
        system[-1]["cache_control"] = _CACHE_CONTROL

    # 3. Last user message — mark last content block
    messages = body.get("messages")
    if messages and isinstance(messages, list):
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content:
                msg["content"] = [
                    {"type": "text", "text": content, "cache_control": _CACHE_CONTROL}
                ]
            elif isinstance(content, list) and len(content) > 0:
                content[-1]["cache_control"] = _CACHE_CONTROL
            break  # only the last user message


# ---------------------------------------------------------------------------
# Headers to forward
# ---------------------------------------------------------------------------

_FORWARD_HEADERS = {
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "anthropic-organization",
}


def _collect_forward_headers(
    request: Request, strip_thinking: bool = False
) -> dict[str, str]:
    """Collect headers from the incoming request to forward upstream."""
    headers = {}
    for key, value in request.headers.items():
        if key.lower() in _FORWARD_HEADERS:
            if key.lower() == "anthropic-beta" and strip_thinking:
                # Remove any thinking-related beta flags
                betas = [
                    b.strip() for b in value.split(",")
                    if "thinking" not in b.strip().lower()
                ]
                if betas:
                    headers[key] = ",".join(betas)
                # If no betas left, skip the header entirely
            else:
                headers[key] = value
    # Ensure content-type
    headers["content-type"] = "application/json"
    return headers


# ---------------------------------------------------------------------------
# Pretty request summary
# ---------------------------------------------------------------------------

def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _cost_usd(tier: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate request cost in USD."""
    from distiq_code.routing import MODEL_INPUT_PRICE, MODEL_OUTPUT_PRICE
    inp = input_tokens * MODEL_INPUT_PRICE.get(tier, 15.0) / 1_000_000
    out = output_tokens * MODEL_OUTPUT_PRICE.get(tier, 75.0) / 1_000_000
    return inp + out


# Cumulative session savings tracker
_session_total_saved = 0.0
_session_total_cost = 0.0


def _log_summary(
    *,
    original_model: str,
    routed_model: str,
    input_tokens: int,
    output_tokens: int,
    stop_reason: str | None,
    latency_ms: float,
    cache_hit: bool = False,
    cache_similarity: float = 0.0,
    tokens_saved: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Print a compact, readable summary line for each request."""
    global _session_total_saved, _session_total_cost
    parts: list[str] = []

    if cache_hit:
        # Calculate what this would have cost on opus
        opus_cost = _cost_usd("opus", tokens_saved, tokens_saved)
        _session_total_saved += opus_cost
        parts.append(f"CACHE HIT (sim={cache_similarity:.2f})")
        parts.append(f"saved {_fmt_tokens(tokens_saved)} tokens (~${opus_cost:.4f})")
        parts.append(f"session saved: ${_session_total_saved:.3f}")
        parts.append(f"{latency_ms:.0f}ms")
        logger.opt(colors=True).info(
            "<green>[proxy]</green> " + " | ".join(parts)
        )
        return

    # Model info
    tier = _model_id_to_tier(routed_model)
    orig_tier = _model_id_to_tier(original_model)

    if routed_model != original_model:
        parts.append(f"{tier} (from {orig_tier})")
    else:
        parts.append(tier)

    # Tokens
    parts.append(f"{_fmt_tokens(input_tokens)} in / {_fmt_tokens(output_tokens)} out")

    # Prompt caching info
    if cache_read_tokens or cache_creation_tokens:
        parts.append(f"cache: {_fmt_tokens(cache_read_tokens)} read / {_fmt_tokens(cache_creation_tokens)} write")

    # Cost
    actual_cost = _cost_usd(tier, input_tokens, output_tokens)
    # Adjust for prompt caching savings (read=90% off, creation=25% surcharge)
    if cache_read_tokens or cache_creation_tokens:
        from distiq_code.routing import MODEL_INPUT_PRICE
        price_per_token = MODEL_INPUT_PRICE.get(tier, 15.0) / 1_000_000
        actual_cost -= cache_read_tokens * price_per_token * 0.9   # 90% discount
        actual_cost += cache_creation_tokens * price_per_token * 0.25  # 25% surcharge
        actual_cost = max(0.0, actual_cost)
    _session_total_cost += actual_cost
    parts.append(f"${actual_cost:.4f}")

    # Savings from routing + prompt caching
    full_cost = _cost_usd(orig_tier, input_tokens, output_tokens)
    saved = full_cost - actual_cost
    if saved > 0.0001:
        _session_total_saved += saved
        parts.append(f"saved ${saved:.4f}")

    # Stop reason (only if unusual)
    if stop_reason and stop_reason not in ("end_turn",):
        parts.append(stop_reason)

    # Latency
    if latency_ms >= 1000:
        parts.append(f"{latency_ms / 1000:.1f}s")
    else:
        parts.append(f"{latency_ms:.0f}ms")

    # Session total
    if _session_total_saved > 0:
        parts.append(f"session saved: ${_session_total_saved:.3f}")

    logger.opt(colors=True).info(
        "<cyan>[proxy]</cyan> " + " | ".join(parts)
    )


# ---------------------------------------------------------------------------
# Streaming proxy
# ---------------------------------------------------------------------------

async def _stream_proxy(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    user_text: str,
    original_model: str,
    routed_model: str,
    is_tool_use: bool,
    start_time: float,
):
    """Stream response from Anthropic API, forwarding bytes and collecting stats."""
    async with client.stream(
        "POST",
        url,
        headers=headers,
        json=body,
        timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
    ) as response:
        # Collect metadata from streamed events for caching/stats
        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0
        stop_reason = None
        buffer = ""

        async for raw_bytes in response.aiter_bytes():
            # Forward raw bytes immediately (byte-perfect)
            yield raw_bytes

            # Parse SSE events from buffer for metadata extraction
            buffer += raw_bytes.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                # Extract data line
                for line in event_str.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except (json.JSONDecodeError, ValueError):
                            continue

                        msg_type = data.get("type", "")

                        if msg_type == "message_start":
                            usage = (
                                data.get("message", {}).get("usage", {})
                            )
                            input_tokens = usage.get("input_tokens", 0)
                            cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                            cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

                        elif msg_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_parts.append(delta.get("text", ""))

                        elif msg_type == "message_delta":
                            delta = data.get("delta", {})
                            stop_reason = delta.get("stop_reason")
                            usage = data.get("usage", {})
                            output_tokens = usage.get(
                                "output_tokens", output_tokens
                            )

    # After stream complete: cache + stats
    full_text = "".join(text_parts)
    latency_ms = (time.time() - start_time) * 1000

    # Cache only if: end_turn, no tool_use, has text, caching enabled
    if (
        settings.cache_enabled
        and user_text
        and full_text
        and stop_reason == "end_turn"
        and not is_tool_use
    ):
        try:
            cache = _get_cache()
            if cache is not None:
                cache.set(user_text, full_text, routed_model)
                logger.debug(f"Cached streamed response ({len(full_text)} chars)")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    # Record stats
    try:
        stats = _get_stats()
        stats.record_request(
            model=routed_model,
            original_tokens=input_tokens,
            compressed_tokens=input_tokens,  # no compression in messages endpoint
            cache_hit=False,
            cache_similarity=0.0,
            latency_ms=latency_ms,
            compression_enabled=False,
            compression_ratio=1.0,
        )
    except Exception as e:
        logger.warning(f"Failed to record stats: {e}")

    _log_summary(
        original_model=original_model,
        routed_model=routed_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
        latency_ms=latency_ms,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


# ---------------------------------------------------------------------------
# Non-streaming proxy
# ---------------------------------------------------------------------------

async def _forward_proxy(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    user_text: str,
    original_model: str,
    routed_model: str,
    is_tool_use: bool,
    start_time: float,
) -> JSONResponse:
    """Forward non-streaming request and return JSONResponse."""
    resp = await client.post(
        url,
        headers=headers,
        json=body,
        timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
    )

    latency_ms = (time.time() - start_time) * 1000

    # Parse response for caching/stats
    try:
        data = resp.json()
    except Exception:
        # Can't parse — just forward as-is
        return JSONResponse(
            content=resp.text, status_code=resp.status_code,
            media_type="application/json",
        )

    # Extract text and usage
    content_blocks = data.get("content", [])
    text_parts = [
        b.get("text", "")
        for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    full_text = "".join(text_parts)

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_read_tokens = usage.get("cache_read_input_tokens", 0)
    cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
    stop_reason = data.get("stop_reason")

    # Cache
    if (
        settings.cache_enabled
        and user_text
        and full_text
        and stop_reason == "end_turn"
        and not is_tool_use
    ):
        try:
            cache = _get_cache()
            if cache is not None:
                cache.set(user_text, full_text, routed_model)
                logger.debug(f"Cached response ({len(full_text)} chars)")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    # Stats
    try:
        stats = _get_stats()
        stats.record_request(
            model=routed_model,
            original_tokens=input_tokens,
            compressed_tokens=input_tokens,
            cache_hit=False,
            cache_similarity=0.0,
            latency_ms=latency_ms,
            compression_enabled=False,
            compression_ratio=1.0,
        )
    except Exception as e:
        logger.warning(f"Failed to record stats: {e}")

    _log_summary(
        original_model=original_model,
        routed_model=routed_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
        latency_ms=latency_ms,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )

    # Forward response headers that matter
    resp_headers = {}
    for h in ("request-id", "x-request-id"):
        if h in resp.headers:
            resp_headers[h] = resp.headers[h]

    return JSONResponse(
        content=data,
        status_code=resp.status_code,
        headers=resp_headers,
    )


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post("/messages")
async def messages_proxy(request: Request):
    """Anthropic Messages API proxy with smart routing and caching.

    Transparently forwards all headers (x-api-key, authorization, anthropic-version,
    anthropic-beta) so it works with both API keys and OAuth tokens.
    """
    start_time = time.time()

    # Parse body as raw dict (Anthropic schema is too complex for Pydantic)
    body: dict[str, Any] = await request.json()

    original_model: str = body.get("model", "")
    messages: list[dict[str, Any]] = body.get("messages", [])
    is_stream: bool = body.get("stream", False)
    has_tool_msgs = _has_tool_messages(body)

    logger.debug(
        f"[messages] model={original_model} msgs={len(messages)} "
        f"stream={is_stream} tool_msgs={has_tool_msgs}"
    )

    # Extract user text for routing + caching
    user_text = _extract_latest_user_text(messages)

    # --- Smart Routing ---
    routed_model = original_model
    original_tier = _model_id_to_tier(original_model)

    # Tool-use forced routing: agentic loops (Read, Glob, Grep, Bash, MCP)
    # always go to Sonnet. Opus is wasted on tool execution — Sonnet handles
    # file reads, searches, and command runs just as well at 5x less cost.
    if has_tool_msgs and original_tier == "opus":
        tier = "sonnet"
        new_model = _tier_to_model_id(tier, original_model)
        logger.info(
            f"[routing] {original_model} -> {new_model} "
            f"(tool-use loop, forced sonnet)"
        )
        routed_model = new_model
        _strip_thinking(body)

    # Text-based routing for non-tool messages
    elif settings.smart_routing and user_text:
        tier, complexity = classify_and_route(
            user_text, len(messages), original_tier
        )
        if tier:
            # Only downgrade, never upgrade (e.g. haiku stays haiku)
            _TIER_RANK = {"haiku": 0, "sonnet": 1, "opus": 2}
            orig_rank = _TIER_RANK.get(original_tier, 2)
            new_rank = _TIER_RANK.get(tier, 1)

            if new_rank >= orig_rank:
                tier = None  # not a downgrade, skip
            elif tier == "haiku" and original_tier == "opus":
                # Claude Code system prompts too complex for haiku
                tier = "sonnet"

            new_model = _tier_to_model_id(tier, original_model) if tier else original_model
            if new_model != original_model:
                logger.debug(
                    f"[routing] {original_model} -> {new_model} "
                    f"(tier={tier}, complexity={complexity})"
                )
                routed_model = new_model

                # Strip features not supported by cheaper models
                _strip_thinking(body)

    # --- Cache Check ---
    # Skip cache for interactive tool exchanges
    if (
        settings.cache_enabled
        and user_text
        and not has_tool_msgs
    ):
        try:
            cache = _get_cache()
            if cache is None:
                raise ValueError("cache unavailable")
            cached_text, cache_stats = cache.get(user_text, routed_model)

            if cached_text:
                latency_ms = (time.time() - start_time) * 1000
                _log_summary(
                    original_model=original_model,
                    routed_model=routed_model,
                    input_tokens=0,
                    output_tokens=0,
                    stop_reason="end_turn",
                    latency_ms=latency_ms,
                    cache_hit=True,
                    cache_similarity=cache_stats.similarity,
                    tokens_saved=cache_stats.tokens_saved,
                )

                # Record stats
                try:
                    stats = _get_stats()
                    stats.record_request(
                        model=routed_model,
                        original_tokens=cache_stats.tokens_saved,
                        compressed_tokens=0,
                        cache_hit=True,
                        cache_similarity=cache_stats.similarity,
                        latency_ms=latency_ms,
                        compression_enabled=False,
                        compression_ratio=1.0,
                    )
                except Exception:
                    pass

                # Rough input token estimate
                input_tokens = sum(
                    len(str(m.get("content", ""))) // 4
                    for m in messages
                )

                if is_stream:
                    return StreamingResponse(
                        _build_cached_stream(
                            cached_text, routed_model, input_tokens
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    return JSONResponse(
                        content=_build_cached_response(
                            cached_text, routed_model, input_tokens
                        )
                    )
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

    # --- Forward to Anthropic ---
    http_client: httpx.AsyncClient = request.app.state.http_client

    # Update model in body if routing changed it
    body["model"] = routed_model

    # Inject Anthropic prompt caching breakpoints
    if settings.prompt_caching_enabled:
        _inject_cache_control(body)

    upstream_url = f"{settings.anthropic_api_base}/v1/messages"
    was_routed = routed_model != original_model
    forward_headers = _collect_forward_headers(request, strip_thinking=was_routed)

    if is_stream:
        return StreamingResponse(
            _stream_proxy(
                http_client,
                upstream_url,
                forward_headers,
                body,
                user_text,
                original_model,
                routed_model,
                has_tool_msgs,
                start_time,
            ),
            media_type="text/event-stream",
        )
    else:
        return await _forward_proxy(
            http_client,
            upstream_url,
            forward_headers,
            body,
            user_text,
            original_model,
            routed_model,
            has_tool_msgs,
            start_time,
        )
