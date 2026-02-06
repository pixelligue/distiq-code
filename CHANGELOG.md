# Changelog

All notable changes to distiq-code will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-06

### Added
- **Tool compression** (inspired by ClaudeSlim) - optional compression of tool definitions to save 10-15% tokens per request
  - Compress tool names: `"Bash"` → `"B"`, `"Read"` → `"R"`, etc.
  - Compress parameter keys: `"file_path"` → `"f"`, `"command"` → `"c"`
  - Truncate long descriptions (>100 chars)
  - New config: `TOOL_COMPRESSION_ENABLED=false` (disabled by default)
- Documentation: `docs/tool-compression.md` with full implementation details
- Demo script: `examples/tool_compression_demo.py` showing savings visualization
- Comprehensive tests: `tests/test_tool_compression.py` (9 tests, 100% coverage)

### Changed
- Updated `README.md` with tool compression section and configuration examples
- Enhanced `compression/__init__.py` to export tool compression functions

### Technical Details
- Tool compression is experimental and disabled by default for stability
- Savings: ~88 tokens per request (15.4% for typical 7 Claude Code tools)
- Graceful fallback if compression fails
- Preserves `cache_control` breakpoints for Anthropic Prompt Caching compatibility

## [0.1.0] - 2026-02-05

### Added
- Initial release of distiq-code
- Smart routing: automatically route simple tasks to cheaper models (Sonnet/Haiku)
- Semantic caching: FAISS + EmbeddingGemma-300M for cache hit detection
- Anthropic Prompt Caching: automatic cache_control injection (90% savings on repeated context)
- Tool-use optimization: force agentic loops (Read, Glob, Grep, Bash) to Sonnet
- ML-based routing: K-NN BERT classifier for complexity estimation
- Regex-based routing: fallback for environments without ML dependencies
- FastAPI proxy server with streaming support
- CLI commands: start, setup, stats, config, chat
- Stats tracking: SQLite database for usage analytics
- Optional LLMLingua-2 prompt compression (requires `[compression]` extra)

### Features
- Transparent proxy for Claude Code CLI
- Works with OAuth tokens (no API key needed)
- Live statistics in logs (model, tokens, cost, latency)
- Session savings tracking
- Cross-platform support (Windows, macOS, Linux)

### Configuration
- Environment variables or `.env` file support
- Configurable cache TTL, similarity threshold, max size
- Optional ML features with `pip install distiq-code[ml]`
- Optional compression with `pip install distiq-code[compression]`

### Documentation
- Comprehensive README with quick start guide
- Architecture overview
- Configuration reference
- Real-world savings examples
