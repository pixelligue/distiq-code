# distiq-code

> Save 3-5x on your Claude Code subscription with smart routing and caching.

**distiq-code** is a local proxy that sits between Claude Code and the Anthropic API. It automatically routes simple tasks to cheaper models and caches repeated queries — so your $200/month Claude Max subscription lasts much longer.

```
Claude Code → localhost:11434 → [Route / Cache] → api.anthropic.com
```

## How It Works

**Smart Routing** — Most Claude Code requests don't need Opus. distiq-code analyzes each prompt and routes it to the cheapest model that can handle it:

| Task | Model | Cost |
|------|-------|------|
| Architecture, system design | Opus | $15/1M tokens |
| Code generation, debugging | Sonnet | $3/1M (5x cheaper) |
| Simple questions, explanations | Haiku | $0.25/1M (60x cheaper) |

**Semantic Caching** — Similar questions get instant answers from local FAISS cache instead of hitting the API again.

**Anthropic Prompt Caching** — Automatically injects `cache_control` breakpoints so repeated context (tools, system prompt) gets a 90% discount from Anthropic.

**Live Stats** — See exactly what's happening on every request:
```
[proxy] sonnet (from opus) | 6.1K in / 795 out | $0.0303 | saved $0.1210 | 20.3s | session saved: $0.134
[proxy] CACHE HIT (sim=0.95) | saved 342 tokens (~$0.0308) | session saved: $0.165 | 36ms
```

## Quick Start

### 1. Install

```bash
pip install distiq-code
```

Or from source:

```bash
git clone https://github.com/distiq-ai/distiq-code.git
cd distiq-code
pip install -e .
```

Optional ML features (semantic cache + BERT routing):

```bash
pip install distiq-code[ml]          # FAISS + sentence-transformers
pip install distiq-code[compression] # LLMLingua-2 prompt compression
pip install distiq-code[all]         # Everything
```

### 2. Setup

```bash
distiq-code setup
```

This will:
- Check that Claude CLI is installed
- Download ML models (~400 MB, if `[ml]` installed)
- Configure `ANTHROPIC_BASE_URL` for Claude Code

### 3. Start Proxy

```bash
distiq-code start
```

### 4. Use Claude Code

In another terminal:

```bash
claude
```

That's it. All requests now go through the proxy automatically.

## Features

### Smart Model Routing

Two routing backends:

- **ML Router** (default with `[ml]`) — BERT-based K-NN classifier over 75 reference examples. ~5ms per query.
- **Regex Router** (fallback) — Pattern matching for RU + EN queries. Zero dependencies.

Routing only **downgrades** — if Claude Code requests Opus but the task is simple, it gets routed to Sonnet. Never upgrades.

### Semantic Caching

Uses FAISS + sentence-transformers (`all-mpnet-base-v2`) to find similar queries:

```
Query 1: "How to create a React component?"
Query 2: "How do I make a React component?" → Cache hit (sim=0.94)
```

- Configurable similarity threshold (default: 0.85)
- 7-day TTL
- Up to 10,000 cached entries
- Tool-use conversations are never cached (stale results)

### Anthropic Prompt Caching

Automatically adds `cache_control` breakpoints to:
1. Last tool definition
2. System prompt
3. Last user message

Cached prefix tokens get 90% off input cost. No configuration needed.

### Prompt Compression (Optional)

With `pip install distiq-code[compression]`:

- LLMLingua-2 compresses conversation context by up to 5x
- Latest user query is never compressed
- Code keywords (`def`, `class`, `import`, etc.) are preserved

## CLI Commands

```bash
distiq-code start    # Start the proxy server
distiq-code setup    # One-time setup (models + env)
distiq-code stats    # Show usage statistics
distiq-code config   # Show current configuration
distiq-code chat     # Interactive chat REPL (optional)
distiq-code version  # Show version
```

### Stats

```bash
distiq-code stats --period week

# Output:
# Requests: 347
# Cache hits: 72%
# Tokens saved: 1,084,000
# Cost saved: $12.40
```

## Configuration

All settings via environment variables or `.env` file:

```bash
# Server
PROXY_HOST=127.0.0.1
PROXY_PORT=11434

# Routing
SMART_ROUTING=true          # Enable smart model routing
ML_ROUTING_ENABLED=true     # Use BERT router (requires [ml])

# Caching
CACHE_ENABLED=true
CACHE_TTL_HOURS=168                  # 7 days
CACHE_SIMILARITY_THRESHOLD=0.85

# Anthropic Prompt Caching
PROMPT_CACHING_ENABLED=true          # Inject cache_control breakpoints

# Compression
COMPRESSION_ENABLED=true
COMPRESSION_TARGET_TOKENS=500

# Debug
DEBUG=false
LOG_LEVEL=INFO
```

## Architecture

```
src/distiq_code/
├── cli.py                 # Typer CLI + chat REPL
├── config.py              # Pydantic Settings
├── routing.py             # Smart routing (regex + embedding)
├── embedding_router.py    # K-NN embedding router
├── clipboard.py           # Screenshot paste support
├── auth/
│   └── cli_provider.py    # Claude CLI subprocess (OAuth)
├── cache/
│   └── semantic_cache.py  # FAISS + sentence-transformers
├── compression/
│   └── compressor.py      # LLMLingua-2
├── stats/
│   └── tracker.py         # Metrics + cost tracking
└── server/
    ├── app.py             # FastAPI factory
    ├── main.py            # Uvicorn entry point
    └── routes/
        ├── messages.py    # /v1/messages (Anthropic proxy)
        ├── chat.py        # /v1/chat/completions (OpenAI)
        └── health.py      # /health, /ready
```

## Requirements

- Python 3.11+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Made by [Distiq](https://distiq.ru)**
