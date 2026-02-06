# Development Guide

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/distiq-ai/distiq-code.git
cd distiq-code

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install in editable mode (core only)
pip install -e .

# With ML features
pip install -e ".[all]"

# With dev tools
pip install -e ".[dev]"
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work fine)
```

### 3. Run

```bash
# One-time setup (downloads ML models)
distiq-code setup

# Start proxy
distiq-code start

# In another terminal:
ANTHROPIC_BASE_URL=http://localhost:11434 claude
```

## Project Structure

```
distiq-code/
├── src/distiq_code/
│   ├── __init__.py              # Package metadata (__version__)
│   ├── cli.py                   # Typer CLI (start, setup, stats, chat)
│   ├── config.py                # Pydantic Settings + feature detection
│   ├── routing.py               # Smart routing (regex + ML, EN + RU)
│   ├── embedding_router.py      # K-NN embedding router (75 examples)
│   ├── clipboard.py             # Clipboard image detection
│   ├── auth/
│   │   └── cli_provider.py      # Claude CLI subprocess (OAuth, streaming)
│   ├── cache/
│   │   └── semantic_cache.py    # FAISS + sentence-transformers cache
│   ├── compression/
│   │   └── compressor.py        # LLMLingua-2 prompt compression
│   ├── stats/
│   │   └── tracker.py           # JSON metrics tracking
│   └── server/
│       ├── app.py               # FastAPI app factory + lifespan
│       ├── main.py              # Uvicorn entry point
│       └── routes/
│           ├── messages.py      # /v1/messages — Anthropic API proxy
│           ├── chat.py          # /v1/chat/completions — OpenAI compat
│           └── health.py        # /health, /ready
├── tests/                       # Tests
├── pyproject.toml               # Dependencies & build config
├── .env.example                 # Environment template
├── Dockerfile                   # Multi-stage Docker build
└── README.md                    # User documentation
```

## How the Proxy Works

```
Claude Code
  → POST /v1/messages
  → localhost:11434
  → [1. Smart Routing]    — classify prompt → pick cheapest model
  → [2. Cache Check]      — FAISS similarity search → instant response if hit
  → [3. Prompt Caching]   — inject cache_control breakpoints for Anthropic
  → [4. Forward]          — httpx HTTP/2 streaming to api.anthropic.com
  → [5. Cache Store]      — save response for future hits
  → [6. Stats]            — record tokens, cost, latency
  ← Stream passthrough ←
```

### Key Design Decisions

- **Transparent proxy** — forwards all headers (x-api-key, authorization, anthropic-beta) untouched
- **Only downgrades** — routing never upgrades model tier (haiku stays haiku)
- **Thinking cleanup** — when downgrading from Opus, strips `thinking` and `context_management` params
- **Tool-use skip** — conversations with tool_use/tool_result blocks are never cached (stale results)
- **Byte-perfect streaming** — raw bytes forwarded immediately, parsed in parallel for stats

## Testing the Proxy

### Health Check

```bash
curl http://localhost:11434/health
```

### Manual API Call

```bash
curl -X POST http://localhost:11434/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-opus-4-6",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Code Quality

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Tests
pytest
```

## Optional Dependencies

The project uses optional dependency groups to keep the base install lightweight:

| Group | Packages | Purpose |
|-------|----------|---------|
| `ml` | sentence-transformers, faiss-cpu, Pillow | Semantic cache (EmbeddingGemma-300M), ML routing, clipboard |
| `compression` | llmlingua | LLMLingua-2 prompt compression |
| `all` | ml + compression | Everything |
| `dev` | pytest, black, ruff, mypy | Development tools |

Features gracefully degrade when optional deps are missing:
- No `[ml]` → regex routing, no cache
- No `[compression]` → no prompt compression

## Troubleshooting

### "Claude CLI not found"

```bash
npm install -g @anthropic-ai/claude-code
claude auth login
```

### "Port 11434 already in use"

Change port in `.env`: `PROXY_PORT=11435`

### FAISS Unicode path error (Windows)

The config automatically uses Windows 8.3 short paths for non-ASCII home directories (e.g. Cyrillic usernames).

### ML models download slow

Models are ~400 MB total. First `distiq-code setup` downloads them. Subsequent starts use cached models.
