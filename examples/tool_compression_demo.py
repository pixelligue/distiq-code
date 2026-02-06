"""Demo: Tool compression savings visualization.

Shows how tool compression reduces token count for Claude Code requests.
"""

from distiq_code.compression.tool_compressor import compress_tools

# Realistic Claude Code tools payload
CLAUDE_CODE_TOOLS = [
    {
        "name": "Bash",
        "description": "Execute bash command with optional timeout",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"},
                "description": {"type": "string", "description": "Human-readable description"},
                "timeout": {"type": "number", "description": "Timeout in milliseconds"},
                "run_in_background": {"type": "boolean", "description": "Run in background"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read file from local filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to file"},
                "offset": {"type": "number", "description": "Line number to start"},
                "limit": {"type": "number", "description": "Number of lines to read"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write a file to the local filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": "Perform exact string replacement in file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
                "old_string": {"type": "string", "description": "String to replace"},
                "new_string": {"type": "string", "description": "Replacement string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching glob pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {"type": "string", "description": "Base directory"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "description": "Search path"},
                "output_mode": {"type": "string", "description": "Output mode"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Task",
        "description": "Launch specialized agent for complex tasks",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Task prompt"},
                "description": {"type": "string", "description": "Short description"},
                "subagent_type": {"type": "string", "description": "Agent type"},
            },
            "required": ["prompt", "description", "subagent_type"],
        },
    },
]


def main():
    print("=" * 70)
    print("Tool Compression Demo: Claude Code Tools")
    print("=" * 70)

    print(f"\nOriginal tools count: {len(CLAUDE_CODE_TOOLS)}")

    compressed, orig_tokens, comp_tokens = compress_tools(CLAUDE_CODE_TOOLS)

    savings = orig_tokens - comp_tokens
    savings_pct = 100 * savings / orig_tokens if orig_tokens > 0 else 0

    print(f"\nToken counts:")
    print(f"  Original:   {orig_tokens:>5} tokens")
    print(f"  Compressed: {comp_tokens:>5} tokens")
    print(f"  Saved:      {savings:>5} tokens ({savings_pct:.1f}%)")

    print("\n" + "=" * 70)
    print("Example: Bash tool transformation")
    print("=" * 70)

    original_bash = CLAUDE_CODE_TOOLS[0]
    compressed_bash = compressed[0]

    print("\nOriginal:")
    print(f'  Name: "{original_bash["name"]}"')
    print(f"  Parameters: {list(original_bash['parameters']['properties'].keys())}")

    print("\nCompressed:")
    print(f'  Name: "{compressed_bash["name"]}"')
    print(f"  Parameters: {list(compressed_bash['parameters']['properties'].keys())}")

    print("\n" + "=" * 70)
    print("Cost Impact (rough estimate)")
    print("=" * 70)

    # Claude Code sends ~50 requests per session
    # Each request includes full tools list
    total_requests = 50
    total_savings = savings * total_requests
    cost_per_token = 15.0 / 1_000_000  # Opus input pricing

    print(f"\nAssumptions:")
    print(f"  Requests per session: {total_requests}")
    print(f"  Cost per token: ${cost_per_token * 1_000_000:.2f}/M tokens (Opus)")

    print(f"\nSession savings:")
    print(f"  Tokens saved: {total_savings:,}")
    print(f"  Cost saved: ${total_savings * cost_per_token:.4f}")

    print("\n" + "=" * 70)
    print("Recommendation")
    print("=" * 70)

    if savings_pct > 15:
        print("\n[OK] Good savings! Consider enabling TOOL_COMPRESSION_ENABLED=true")
    elif savings_pct > 10:
        print("\n[WARN] Modest savings. Test in your environment first.")
    else:
        print("\n[INFO] Minimal savings. Focus on routing + caching instead.")

    print("\nNote: Main savings come from smart routing (5x) and semantic caching (100%).")
    print("Tool compression is a bonus optimization, not the primary strategy.")
    print()


if __name__ == "__main__":
    main()
