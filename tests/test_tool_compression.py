"""Tests for tool compression functionality."""

import json

from distiq_code.compression.tool_compressor import (
    compress_tools,
    decompress_tool_use,
)


def test_tool_name_compression():
    """Test that tool names are compressed correctly."""
    tools = [
        {
            "name": "Bash",
            "description": "Execute bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"}
                },
                "required": ["command"],
            },
        },
        {
            "name": "Read",
            "description": "Read file from filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"}
                },
                "required": ["file_path"],
            },
        },
    ]

    compressed, orig_tokens, comp_tokens = compress_tools(tools)

    # Check tool names compressed
    assert compressed[0]["name"] == "B"
    assert compressed[1]["name"] == "R"

    # Check parameter names compressed
    assert "c" in compressed[0]["parameters"]["properties"]
    assert "f" in compressed[1]["parameters"]["properties"]

    # Check token savings
    assert comp_tokens < orig_tokens
    assert orig_tokens > 0


def test_long_description_truncation():
    """Test that long descriptions are truncated."""
    tools = [
        {
            "name": "WebFetch",
            "description": (
                "This is a very long description that should be truncated "
                "because it exceeds 100 characters and wastes tokens unnecessarily "
                "when sent to the API repeatedly."
            ),
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    compressed, _, _ = compress_tools(tools)

    # Description should be truncated to ~100 chars
    assert len(compressed[0]["description"]) <= 100
    assert compressed[0]["description"].endswith("...")


def test_cache_control_preserved():
    """Test that cache_control breakpoints are preserved."""
    tools = [
        {
            "name": "Bash",
            "description": "Execute bash command",
            "parameters": {"type": "object", "properties": {}},
            "cache_control": {"type": "ephemeral"},
        }
    ]

    compressed, _, _ = compress_tools(tools)

    # cache_control must be preserved for Anthropic prompt caching
    assert "cache_control" in compressed[0]
    assert compressed[0]["cache_control"]["type"] == "ephemeral"


def test_nested_parameter_compression():
    """Test that nested objects in parameters are compressed recursively."""
    tools = [
        {
            "name": "Task",
            "description": "Launch task",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {"type": "string"},
                    "_simulatedSedEdit": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "new_source": {"type": "string"},
                        },
                    },
                },
            },
        }
    ]

    compressed, _, _ = compress_tools(tools)

    # Top-level params compressed
    props = compressed[0]["parameters"]["properties"]
    assert "pr" in props  # prompt
    assert "d" in props  # description

    # Nested object params compressed
    nested = props["ss"]  # _simulatedSedEdit
    assert "f" in nested["properties"]  # file_path
    assert "ns" in nested["properties"]  # new_source


def test_decompress_tool_use():
    """Test decompressing a tool_use block from Claude's response."""
    tool_use = {
        "type": "tool_use",
        "id": "toolu_123",
        "name": "B",  # Compressed Bash
        "input": {
            "c": "ls -la",  # Compressed command
            "d": "List files",  # Compressed description
        },
    }

    decompressed = decompress_tool_use(tool_use)

    assert decompressed["name"] == "Bash"
    assert decompressed["input"]["command"] == "ls -la"
    assert decompressed["input"]["description"] == "List files"


def test_empty_tools_list():
    """Test that empty tools list is handled gracefully."""
    compressed, orig, comp = compress_tools([])
    assert compressed == []
    assert orig == 0
    assert comp == 0


def test_tool_without_parameters():
    """Test tool with no parameters field."""
    tools = [
        {
            "name": "TaskList",
            "description": "List all tasks",
            # No parameters field
        }
    ]

    compressed, _, _ = compress_tools(tools)

    # Should not crash, name should be compressed
    assert compressed[0]["name"] == "TL"


def test_unknown_tool_name_preserved():
    """Test that unknown tool names are preserved as-is."""
    tools = [
        {
            "name": "CustomTool",
            "description": "Some custom tool",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    compressed, _, _ = compress_tools(tools)

    # Unknown tool name should remain unchanged
    assert compressed[0]["name"] == "CustomTool"


def test_actual_claude_code_tools():
    """Test with a realistic set of Claude Code tools."""
    # Simplified example of what Claude Code sends
    tools = [
        {
            "name": "Bash",
            "description": "Execute bash command with timeout",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                    "timeout": {"type": "number"},
                    "run_in_background": {"type": "boolean"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "Read",
            "description": "Read file from filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": "number"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "Edit",
            "description": "Perform exact string replacement in file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    ]

    compressed, orig_tokens, comp_tokens = compress_tools(tools)

    # Should save tokens (10-15% is realistic for already compact tools)
    savings_pct = 100 * (orig_tokens - comp_tokens) / orig_tokens
    assert savings_pct > 10  # At least 10% savings
    assert comp_tokens < orig_tokens  # Must save something

    # Verify structure is intact
    assert len(compressed) == 3
    assert compressed[0]["name"] == "B"
    assert compressed[1]["name"] == "R"
    assert compressed[2]["name"] == "E"

    # Verify parameters compressed
    assert "c" in compressed[0]["parameters"]["properties"]  # command
    assert "f" in compressed[1]["parameters"]["properties"]  # file_path
    assert "o" in compressed[2]["parameters"]["properties"]  # old_string
    assert "n" in compressed[2]["parameters"]["properties"]  # new_string
