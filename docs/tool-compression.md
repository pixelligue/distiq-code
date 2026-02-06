# Tool Compression Feature

**Inspired by:** [ClaudeSlim](https://github.com/apolloraines/claudeslim)
**Status:** Experimental (disabled by default)
**Savings:** ~100-300 tokens per request (10-15%)

## Overview

Tool compression reduces the size of tool definitions sent with each Claude Code request by:

1. **Shortening tool names**: `"Bash"` → `"B"`, `"Read"` → `"R"`, etc.
2. **Compressing parameter keys**: `"file_path"` → `"f"`, `"command"` → `"c"`
3. **Truncating long descriptions**: Keeping only first 100 chars

## Why Not More Savings?

ClaudeSlim reports 80% compression on tools, but we see only 10-15% because:

- Claude Code already uses compact tool schemas (no verbose descriptions)
- We preserve `cache_control` breakpoints for Anthropic Prompt Caching
- We prioritize stability over aggressive compression

**Our approach:** Combine multiple optimizations (routing + caching + compression) rather than betting everything on one technique.

## Configuration

```bash
# .env or environment variable
TOOL_COMPRESSION_ENABLED=false  # Default: off (experimental)
```

## When to Enable

✅ **Enable if:**
- You're hitting token limits frequently
- You're on a strict budget
- You understand the risks (may break on Claude Code updates)

❌ **Don't enable if:**
- You prefer stability over optimization
- You're already getting good savings from routing + caching
- You haven't tested it on your specific use case

## Implementation Details

### Compression Mappings

**Tool names:**
```python
{
    "Bash": "B",
    "Read": "R",
    "Write": "W",
    "Edit": "E",
    "Glob": "G",
    "Grep": "S",
    "Task": "T",
    # ... etc
}
```

**Parameter keys:**
```python
{
    "file_path": "f",
    "command": "c",
    "pattern": "p",
    "description": "d",
    "prompt": "pr",
    # ... etc
}
```

### Example

**Before compression (184 tokens):**
```json
{
  "name": "Bash",
  "description": "Execute bash command with timeout",
  "parameters": {
    "type": "object",
    "properties": {
      "command": {"type": "string", "description": "Command to run"},
      "timeout": {"type": "number", "description": "Timeout in ms"},
      "run_in_background": {"type": "boolean", "description": "Run in background"}
    },
    "required": ["command"]
  }
}
```

**After compression (162 tokens, 12% savings):**
```json
{
  "name": "B",
  "description": "Execute bash command with timeout",
  "parameters": {
    "type": "object",
    "properties": {
      "c": {"type": "string", "d": "Command to run"},
      "to": {"type": "number", "d": "Timeout in ms"},
      "bg": {"type": "boolean", "d": "Run in background"}
    },
    "required": ["c"]
  }
}
```

## Differences from ClaudeSlim

| Feature | ClaudeSlim | distiq-code |
|---------|------------|-------------|
| **Approach** | Aggressive compression + hashing | Selective compression only |
| **System prompt** | SHA256 hash (95% reduction) | Anthropic Prompt Caching (90% discount) |
| **Message history** | Text compression (40% reduction) | Semantic caching (100% skip on hit) |
| **Tool definitions** | Dictionary compression (80%) | Parameter shortening (10-15%) |
| **Compatibility** | Fragile (breaks on updates) | Safer (preserves structure) |

## Risks

⚠️ **Potential issues:**

1. **Claude Code updates** — New tools or parameters may not be in our mapping
2. **MCP servers** — Custom tools from MCP servers won't be compressed
3. **Debugging difficulty** — Compressed logs are harder to read

**Mitigation:** Disabled by default, graceful fallback on errors, comprehensive tests.

## Testing

Run tests to verify compression works correctly:

```bash
pytest tests/test_tool_compression.py -v
```

All tests should pass before enabling in production.

## Monitoring

When enabled, look for log lines like:

```
[DEBUG] Tool compression: 184 → 162 tokens (saved 22, 12.0%)
```

If you see errors or unusual behavior, disable immediately:

```bash
TOOL_COMPRESSION_ENABLED=false
```

## Future Improvements

- [ ] Add decompression for tool_use responses (if needed)
- [ ] Expand mappings for new Claude Code tools
- [ ] A/B test to measure real-world impact on costs
- [ ] Consider more aggressive compression for Pro/Max tier users

## Conclusion

Tool compression is a **nice-to-have**, not a must-have. Our main savings come from:

1. **Smart routing** (5x cheaper models for simple tasks)
2. **Semantic caching** (skip API calls entirely)
3. **Anthropic Prompt Caching** (90% off repeated context)

Tool compression adds an extra 10-15% on top, but with added complexity. Use judiciously.
