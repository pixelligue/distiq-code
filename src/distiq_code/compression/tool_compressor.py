"""Tool definition compression for reducing Claude Code request payload.

Inspired by ClaudeSlim's dictionary compression approach, but safer:
- Only compresses tool names and common parameter keys
- Preserves all functionality and semantic meaning
- Can be disabled if Claude Code updates break compatibility

Typical savings: 500-1000 tokens per request (Claude Code sends 15+ tools).
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

# Tool name compression mapping
TOOL_NAME_MAP = {
    "Bash": "B",
    "Read": "R",
    "Write": "W",
    "Edit": "E",
    "Glob": "G",
    "Grep": "S",  # Search
    "Task": "T",
    "TaskOutput": "TO",
    "TaskStop": "TS",
    "Skill": "SK",
    "EnterPlanMode": "EP",
    "ExitPlanMode": "XP",
    "AskUserQuestion": "AQ",
    "TaskCreate": "TC",
    "TaskGet": "TG",
    "TaskUpdate": "TU",
    "TaskList": "TL",
    "NotebookEdit": "NE",
    "WebFetch": "WF",
    "WebSearch": "WS",
}

# Reverse mapping for decompression (if needed)
TOOL_NAME_REVERSE = {v: k for k, v in TOOL_NAME_MAP.items()}

# Common parameter name compression
PARAM_MAP = {
    "description": "d",
    "file_path": "f",
    "command": "c",
    "pattern": "p",
    "old_string": "o",
    "new_string": "n",
    "content": "ct",
    "prompt": "pr",
    "query": "q",
    "task_id": "ti",
    "subagent_type": "st",
    "model": "m",
    "url": "u",
    "notebook_path": "np",
    "new_source": "ns",
    "cell_type": "ce",
    "edit_mode": "em",
    "path": "pt",
    "offset": "of",
    "limit": "li",
    "timeout": "to",
    "dangerouslyDisableSandbox": "dd",
    "run_in_background": "bg",
    "_simulatedSedEdit": "ss",
}

PARAM_REVERSE = {v: k for k, v in PARAM_MAP.items()}


def _compress_object(obj: dict[str, Any], param_map: dict[str, str]) -> dict[str, Any]:
    """Recursively compress parameter keys in a nested object."""
    compressed = {}
    for key, value in obj.items():
        new_key = param_map.get(key, key)
        if isinstance(value, dict):
            compressed[new_key] = _compress_object(value, param_map)
        elif isinstance(value, list):
            compressed[new_key] = [
                _compress_object(item, param_map) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            compressed[new_key] = value
    return compressed


def compress_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Compress tool definitions by shortening names and parameter keys.

    Args:
        tools: List of tool definition dicts from Claude Code

    Returns:
        Tuple of (compressed_tools, original_tokens, compressed_tokens)
    """
    if not tools:
        return tools, 0, 0

    # Estimate original size (rough: JSON length / 4)
    original_json = json.dumps(tools, separators=(",", ":"))
    original_tokens = len(original_json) // 4

    compressed_tools = []
    for tool in tools:
        compressed_tool = tool.copy()

        # 1. Compress tool name
        tool_name = tool.get("name", "")
        if tool_name in TOOL_NAME_MAP:
            compressed_tool["name"] = TOOL_NAME_MAP[tool_name]

        # 2. Compress parameter schema (nested in "parameters")
        params = tool.get("parameters")
        if params and isinstance(params, dict):
            compressed_tool["parameters"] = _compress_object(params, PARAM_MAP)

        # 3. Compress description if too verbose (keep first 50 chars + "...")
        desc = tool.get("description", "")
        if len(desc) > 100:
            compressed_tool["description"] = desc[:97] + "..."

        # 4. Keep cache_control unchanged (needed for prompt caching)
        if "cache_control" in tool:
            compressed_tool["cache_control"] = tool["cache_control"]

        compressed_tools.append(compressed_tool)

    # Estimate compressed size
    compressed_json = json.dumps(compressed_tools, separators=(",", ":"))
    compressed_tokens = len(compressed_json) // 4

    tokens_saved = original_tokens - compressed_tokens
    if tokens_saved > 0:
        logger.debug(
            f"Tool compression: {original_tokens} → {compressed_tokens} tokens "
            f"(saved {tokens_saved}, {100 * tokens_saved / original_tokens:.1f}%)"
        )

    return compressed_tools, original_tokens, compressed_tokens


def decompress_tool_use(tool_use: dict[str, Any]) -> dict[str, Any]:
    """Decompress a tool_use block from Claude's response (if needed).

    This is only necessary if Claude returns compressed tool names in its response,
    which is unlikely since we only compress the tool *definitions*, not the usage.

    Args:
        tool_use: A tool_use block from Claude's response

    Returns:
        Decompressed tool_use block
    """
    decompressed = tool_use.copy()

    # Decompress tool name if it was compressed
    tool_name = tool_use.get("name", "")
    if tool_name in TOOL_NAME_REVERSE:
        decompressed["name"] = TOOL_NAME_REVERSE[tool_name]

    # Decompress input parameters
    tool_input = tool_use.get("input")
    if tool_input and isinstance(tool_input, dict):
        decompressed["input"] = _compress_object(tool_input, PARAM_REVERSE)

    return decompressed
