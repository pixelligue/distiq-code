"""CLI entry point using Typer."""

import asyncio
import re
import sys
import time

import typer
from rich.console import Console
from rich.panel import Panel
from distiq_code import __version__

app = typer.Typer(
    name="distiq-code",
    help="🚀 Get 10x more from your AI coding subscriptions",
    add_completion=False,
)

# Force UTF-8 output on Windows to support emoji in Rich panels
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold blue]distiq-code[/bold blue] version {__version__}")


@app.command()
def setup():
    """
    One-time setup: check deps, download ML models (if installed), configure proxy.

    After setup:
        distiq-code start    # Start proxy (keep running)
        claude               # Use Claude Code through proxy
    """
    import io
    import logging
    import os
    import subprocess
    import warnings
    from pathlib import Path

    from distiq_code.config import settings, get_available_features

    features = get_available_features()
    has_ml = features["ml"]

    steps_total = 3 if has_ml else 2

    setup_items = "  1. Check Claude CLI\n"
    if has_ml:
        setup_items += "  2. Download ML models for smart routing & caching (~400 MB)\n"
        setup_items += f"  3. Configure ANTHROPIC_BASE_URL for Claude Code"
    else:
        setup_items += "  2. Configure ANTHROPIC_BASE_URL for Claude Code\n"
        setup_items += "\n[dim]ML features not installed. For cache + ML routing:[/dim]\n"
        setup_items += "  [cyan]pip install distiq-code[ml][/cyan]"

    console.print(Panel.fit(
        "[bold green]distiq-code Setup[/bold green]\n\n"
        "This will:\n" + setup_items + "\n\n"
        "[dim]One-time setup[/dim]",
        title="Setup",
    ))

    # --- Step 1: Check Claude CLI ---
    console.print(f"\n[bold]Step 1/{steps_total}:[/bold] Claude CLI")
    try:
        result = subprocess.run(
            ["where" if sys.platform == "win32" else "which", "claude"],
            capture_output=True, check=False,
        )
        if result.returncode == 0:
            console.print("  [green]Claude CLI installed[/green]")
        else:
            console.print("  [yellow]Claude CLI not found[/yellow]")
            console.print("  Install: [bold cyan]npm install -g @anthropic-ai/claude-code[/bold cyan]")
            console.print("  Then:    [bold cyan]claude auth login[/bold cyan]")
    except Exception:
        console.print("  [yellow]Could not check Claude CLI[/yellow]")

    # --- Step 2: Download ML models (only if ML deps installed) ---
    if has_ml:
        console.print(f"\n[bold]Step 2/{steps_total}:[/bold] ML models")

        # Suppress noisy logging/tqdm during model download
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TQDM_DISABLE"] = "1"
        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)

        stderr_orig = sys.stderr
        sys.stderr = io.StringIO()

        try:
            # Download and test sentence-transformers
            with console.status(
                "  [dim]Downloading sentence-transformers model...[/dim]",
                spinner="dots",
            ):
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(settings.cache_embedding_model)
                test_vec = model.encode("test query", normalize_embeddings=True)
                dim = len(test_vec)

            console.print(f"  [green]Embedding model ready (dim={dim})[/green]")

            # Test FAISS
            with console.status("  [dim]Testing FAISS vector index...[/dim]", spinner="dots"):
                import faiss
                import numpy as np

                index = faiss.IndexFlatIP(dim)
                vec = test_vec.reshape(1, -1).astype("float32")
                index.add(vec)
                assert index.ntotal == 1

            console.print("  [green]FAISS working[/green]")

            # Pre-build embedding router index (K-NN over reference examples)
            with console.status("  [dim]Building routing index...[/dim]", spinner="dots"):
                from distiq_code.embedding_router import EmbeddingRouter

                emb_router = EmbeddingRouter(model=model)
                test_tier, test_complexity, test_conf = emb_router.route(
                    "write a function"
                )

            console.print(
                f"  [green]Routing index ready "
                f"(test: '{test_complexity}' conf={test_conf:.0%})[/green]"
            )

        except Exception as e:
            sys.stderr = stderr_orig
            logging.disable(logging.NOTSET)
            console.print(f"  [red]Failed: {e}[/red]")
            raise typer.Exit(1)

        sys.stderr = stderr_orig
        logging.disable(logging.NOTSET)
    else:
        console.print("\n  [dim]Skipping ML models (not installed)[/dim]")

    # --- Configure ANTHROPIC_BASE_URL ---
    console.print(f"\n[bold]Step {steps_total}/{steps_total}:[/bold] Configure proxy")

    proxy_url = f"http://{settings.proxy_host}:{settings.proxy_port}"

    configured = False
    if sys.platform == "win32":
        # Windows: set persistent user environment variable via setx
        try:
            result = subprocess.run(
                ["setx", "ANTHROPIC_BASE_URL", proxy_url],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                os.environ["ANTHROPIC_BASE_URL"] = proxy_url
                console.print(f"  [green]ANTHROPIC_BASE_URL={proxy_url}[/green]")
                console.print(
                    "  [dim]Saved as permanent user env var "
                    "(restart terminal to apply)[/dim]"
                )
                configured = True
        except Exception:
            pass
    else:
        # Linux/macOS: add to shell profile
        shell = os.environ.get("SHELL", "")
        rc_file = (
            Path.home() / ".zshrc"
            if "zsh" in shell
            else Path.home() / ".bashrc"
        )
        export_line = f'export ANTHROPIC_BASE_URL="{proxy_url}"'
        try:
            existing = rc_file.read_text() if rc_file.exists() else ""
            if "ANTHROPIC_BASE_URL" not in existing:
                with open(rc_file, "a") as f:
                    f.write(f"\n# distiq-code proxy\n{export_line}\n")
                console.print(f"  [green]Added to {rc_file}[/green]")
                configured = True
            else:
                console.print(f"  [green]Already in {rc_file}[/green]")
                configured = True
        except Exception:
            pass

    if not configured:
        console.print(f"  [yellow]Auto-configure failed. Set manually:[/yellow]")
        if sys.platform == "win32":
            console.print(
                f'  [cyan]$env:ANTHROPIC_BASE_URL = "{proxy_url}"[/cyan]'
            )
        else:
            console.print(
                f'  [cyan]export ANTHROPIC_BASE_URL="{proxy_url}"[/cyan]'
            )

    # --- Done! ---
    console.print()
    if sys.platform == "win32":
        use_cmd = (
            f'  [cyan]$env:ANTHROPIC_BASE_URL = "{proxy_url}"; claude[/cyan]'
        )
    else:
        use_cmd = f"  [cyan]ANTHROPIC_BASE_URL={proxy_url} claude[/cyan]"

    console.print(Panel.fit(
        "[bold green]Setup complete![/bold green]\n\n"
        "[bold]Usage:[/bold]\n"
        "  1. [cyan]distiq-code start[/cyan]   — start proxy (keep running)\n"
        "  2. Open new terminal:\n"
        + ("     [cyan]claude[/cyan]              — works automatically\n"
           if configured else
           use_cmd + "\n")
        + "\n"
        "[bold]What happens:[/bold]\n"
        "  Claude Code -> proxy -> smart routing -> Anthropic API\n"
        "  Opus requests -> Sonnet (5x cheaper) when possible\n"
        "  Repeated questions -> instant cache hit\n\n"
        "[dim]To remove: "
        + ("setx ANTHROPIC_BASE_URL \"\"" if sys.platform == "win32"
           else "remove ANTHROPIC_BASE_URL from shell profile")
        + "[/dim]",
        title="Ready!",
    ))


@app.command()
def start():
    """
    Start the proxy server.

    Run 'distiq-code setup' first to download models.
    Then use Claude Code normally — all requests go through the proxy.
    """
    import os

    from distiq_code.config import settings, get_available_features
    from distiq_code.server.main import run_server

    features = get_available_features()
    proxy_url = f"http://{settings.proxy_host}:{settings.proxy_port}"

    # Check if ANTHROPIC_BASE_URL is configured
    env_val = os.environ.get("ANTHROPIC_BASE_URL", "")
    env_ok = proxy_url in env_val

    if sys.platform == "win32":
        manual_cmd = f'$env:ANTHROPIC_BASE_URL = "{proxy_url}"; claude'
    else:
        manual_cmd = f"ANTHROPIC_BASE_URL={proxy_url} claude"

    use_section = (
        "[bold]In another terminal:[/bold]\n"
        "  [cyan]claude[/cyan]\n"
        if env_ok else
        "[bold]In another terminal:[/bold]\n"
        f"  [cyan]{manual_cmd}[/cyan]\n"
        "  [dim](run 'distiq-code setup' to set permanently)[/dim]\n"
    )

    # Build features display
    routing_str = "ml (BERT)" if features["ml_routing"] else "regex"
    cache_str = "on" if features["cache"] and settings.cache_enabled else "off"
    compression_str = "on" if features["compression"] and settings.compression_enabled else "off"
    if not features["cache"]:
        cache_str += " [dim](pip install distiq-code\\[ml])[/dim]"
    if not features["compression"]:
        compression_str += " [dim](pip install distiq-code\\[compression])[/dim]"

    console.print(Panel.fit(
        f"[bold green]distiq-code proxy[/bold green]\n\n"
        f"Listening:    [bold]{proxy_url}[/bold]\n"
        f"Target:       [dim]{settings.anthropic_api_base}[/dim]\n"
        f"Routing:      [cyan]{routing_str}[/cyan]\n"
        f"Caching:      [cyan]{cache_str}[/cyan]\n"
        f"Compression:  [cyan]{compression_str}[/cyan]\n\n"
        + use_section + "\n"
        "[dim]Ctrl+C to stop[/dim]",
        title="Proxy Server",
    ))

    try:
        run_server()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


@app.command()
def stats(
    period: str = typer.Option("week", help="Time period (day, week, month, all)"),
    clear: bool = typer.Option(False, help="Clear all statistics"),
):
    """
    Show usage statistics and optimization metrics.

    Example:
        distiq-code stats --period week
        distiq-code stats --clear
    """
    from distiq_code.config import settings
    from distiq_code.stats import StatsTracker

    tracker = StatsTracker(stats_file=settings.stats_db_file)

    if clear:
        if typer.confirm("Are you sure you want to clear all statistics?", default=False):
            tracker.clear()
            console.print("[green]✅ Statistics cleared[/green]")
        return

    # Parse period
    period_hours = {
        "day": 24,
        "week": 24 * 7,
        "month": 24 * 30,
        "all": None,
    }

    since_hours = period_hours.get(period.lower())
    if since_hours is None and period.lower() != "all":
        console.print(f"[red]❌ Unknown period: {period}[/red]")
        console.print("Valid periods: day, week, month, all")
        raise typer.Exit(1)

    # Get stats
    aggregate = tracker.get_aggregate_stats(since_hours=since_hours)
    cost_savings = tracker.get_cost_savings(since_hours=since_hours)

    # Display stats
    console.print(Panel.fit(
        f"[bold green]📊 Optimization Statistics[/bold green]\n\n"
        f"[bold]Period:[/bold] {period.title()}\n\n"
        f"[bold cyan]Requests:[/bold cyan]\n"
        f"  Total: {aggregate.total_requests:,}\n"
        f"  Cache hits: {aggregate.cache_hits:,} ({aggregate.cache_hit_rate:.1%})\n"
        f"  Avg latency: {aggregate.average_latency_ms:.0f}ms\n\n"
        f"[bold cyan]Token Optimization:[/bold cyan]\n"
        f"  Original tokens: {aggregate.total_tokens_original:,}\n"
        f"  Compressed tokens: {aggregate.total_tokens_compressed:,}\n"
        f"  Tokens saved: [bold green]{aggregate.total_tokens_saved:,}[/bold green]\n"
        f"  Avg compression: {aggregate.average_compression_ratio:.1%}\n\n"
        f"[bold cyan]Cost Savings:[/bold cyan]\n"
        f"  Original cost: ${cost_savings['original_cost_usd']}\n"
        f"  Compressed cost: ${cost_savings['compressed_cost_usd']}\n"
        f"  [bold green]Saved: ${cost_savings['savings_usd']} ({cost_savings['savings_percent']}%)[/bold green]\n\n"
        f"[dim]Based on Claude Opus pricing ($15/1M tokens)[/dim]",
        title=f"📈 distiq-code Stats - {period.title()}",
    ))


@app.command()
def config():
    """Show current configuration."""
    from distiq_code.config import settings

    console.print("[bold]⚙️  Configuration[/bold]\n")
    console.print(f"Config dir: {settings.config_dir}")
    console.print(f"Compression: {'enabled' if settings.compression_enabled else 'disabled'}")
    console.print(f"Caching: {'enabled' if settings.cache_enabled else 'disabled'}")
    console.print(f"Debug: {settings.debug}")


@app.command()
def chat(
    model: str = typer.Option("sonnet", help="Model: opus, sonnet, haiku"),
):
    """Interactive chat with optimization (compression + caching)."""
    _suppress_logging()
    asyncio.run(_chat_loop(model))


def _suppress_logging():
    """Kill all logging/warnings before heavy imports."""
    import logging
    import os
    import warnings

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_DISABLE"] = "1"

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    from loguru import logger
    logger.disable("distiq_code")
    logger.disable("sentence_transformers")
    logger.disable("transformers")
    logger.disable("faiss")
    logger.disable("httpx")


def _format_tokens(tokens: int) -> str:
    """Format token count: 42.3K, 1.2M, 500."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def _print_header(
    c: Console, model: str, compression: bool, cache: bool, routing_type: str = "regex"
):
    """Print compact chat header."""
    parts = ["[bold]distiq-code[/bold]", f"[cyan]{model}[/cyan]"]
    if compression:
        parts.append("[green]compression[/green]")
    if cache:
        parts.append("[green]cache[/green]")
    parts.append(f"[magenta]{routing_type}-routing[/magenta]")
    c.print(f" {' [dim]·[/dim] '.join(parts)}")
    c.print(" [dim]/help for commands · screenshot auto-attaches on Enter · Ctrl+C to exit[/dim]")
    c.print()


CHAT_TOOLS = ["Read", "Glob", "Grep", "Bash"]

IMAGE_TAG_RE = re.compile(r"\[image:\s*(.+?)\]")


def _try_paste_image() -> str | None:
    """Try to grab an image from clipboard and return [image: path] tag, or None."""
    try:
        from distiq_code.clipboard import check_clipboard_image

        path = check_clipboard_image()
        if path:
            return f"[image: {path}]"
    except Exception:
        pass
    return None


def _format_tool_event(event) -> str:
    """Format a ToolUseEvent into a short human-readable description."""
    name = event.name
    inp = event.input
    if name == "Read":
        path = inp.get("file_path", "")
        # Show only filename (handle both / and \ separators)
        filename = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        return f"Reading {filename}..."
    elif name == "Bash":
        cmd = inp.get("command", "")[:50]
        return f"$ {cmd}..."
    elif name in ("Glob", "Grep"):
        return f"Searching: {inp.get('pattern', '')}..."
    else:
        return f"{name}..."


async def _chat_loop(model: str) -> None:
    """Async REPL for interactive chat."""
    import io

    from distiq_code.config import settings

    # Clean start
    console.clear()
    console.print()

    # Init provider
    try:
        from distiq_code.auth.cli_provider import ClaudeCliProvider
        provider = ClaudeCliProvider()
    except RuntimeError:
        console.print(" [red]Claude CLI not found. Run: distiq-code setup[/red]")
        return

    # Init cache/compressor/ML-router with spinner, suppress stderr (tqdm, warnings)
    cache = None
    compressor = None
    routing_type = "regex"

    with console.status(" [dim]Loading models...[/dim]", spinner="dots"):
        stderr_orig = sys.stderr
        sys.stderr = io.StringIO()
        try:
            if settings.cache_enabled:
                try:
                    from distiq_code.server.routes.chat import get_cache
                    cache = get_cache()
                except Exception:
                    pass
            if settings.compression_enabled:
                try:
                    from distiq_code.server.routes.chat import get_compressor
                    compressor = get_compressor()
                except Exception:
                    pass
            # Load embedding router (reuse model from cache if available)
            if settings.ml_routing_enabled:
                try:
                    from distiq_code.routing import load_embedding_router
                    cache_model = getattr(cache, "embedding_model", None) if cache else None
                    if load_embedding_router(model=cache_model):
                        routing_type = "ml"
                except Exception:
                    pass
        finally:
            sys.stderr = stderr_orig

    _print_header(console, model, settings.compression_enabled, settings.cache_enabled, routing_type)

    from distiq_code.stats import StatsTracker
    stats_tracker = StatsTracker(stats_file=settings.stats_db_file)

    # Session state
    messages: list[dict[str, str]] = []
    session_requests = 0
    session_tokens_saved = 0
    session_cache_hits = 0
    session_total_tokens = 0

    # Real usage counters (from Claude CLI result event)
    session_input_tokens = 0
    session_output_tokens = 0
    session_cost_usd = 0.0
    session_routing_saved = 0.0  # Money saved by smart routing

    # Track last clipboard image to detect new screenshots
    from distiq_code.clipboard import get_clipboard_hash, check_clipboard_image
    last_clip_hash = get_clipboard_hash()

    while True:
        try:
            user_input = console.input("[bold cyan]>[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        user_input = user_input.strip()

        # Check if clipboard has a NEW image since last input
        current_hash = get_clipboard_hash()
        if current_hash is not None and current_hash != last_clip_hash:
            last_clip_hash = current_hash
            path = check_clipboard_image()
            if path:
                tag = f"[image: {path}]"
                if user_input:
                    user_input = f"{tag} {user_input}"
                else:
                    user_input = tag
                console.print(f" [green]image attached:[/green] {path}")

        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                break

            elif cmd == "/clear":
                messages.clear()
                console.clear()
                console.print()
                _print_header(console, model, settings.compression_enabled, settings.cache_enabled, routing_type)
                continue

            elif cmd == "/help":
                console.print(" [dim]/quit     Exit[/dim]")
                console.print(" [dim]/clear    Clear conversation[/dim]")
                console.print(" [dim]/stats    Session statistics[/dim]")
                console.print(" [dim]/model    Switch model (opus, sonnet, haiku)[/dim]")
                console.print(" [dim]/routing  Toggle smart model routing on/off[/dim]")
                console.print(" [dim]/paste    Paste image from clipboard[/dim]")
                console.print(" [dim]Auto: copy screenshot → notification → auto-attaches to next msg[/dim]")
                console.print()
                continue

            elif cmd == "/paste":
                tag = _try_paste_image()
                if tag:
                    console.print(f" [dim]{tag}[/dim]")
                    extra = ""
                    if len(cmd_parts) > 1:
                        extra = cmd_parts[1].strip()
                    user_input = f"{tag} {extra}".strip() if extra else tag
                    # Fall through to message sending below
                else:
                    console.print(" [dim]No image in clipboard[/dim]")
                    console.print()
                    continue

            elif cmd == "/stats":
                hit_rate = f"{session_cache_hits / session_requests:.0%}" if session_requests else "0%"
                console.print(
                    f" [dim]Requests: {session_requests} · "
                    f"In: {_format_tokens(session_input_tokens)} · "
                    f"Out: {_format_tokens(session_output_tokens)} · "
                    f"Cache: {session_cache_hits} ({hit_rate}) · "
                    f"${session_cost_usd:.2f}[/dim]"
                )
                console.print()
                continue

            elif cmd == "/model":
                if len(cmd_parts) > 1 and cmd_parts[1].strip() in ("opus", "sonnet", "haiku"):
                    model = cmd_parts[1].strip()
                    console.print(f" [dim]Switched to {model}[/dim]")
                else:
                    console.print(" [dim]Usage: /model opus|sonnet|haiku[/dim]")
                console.print()
                continue

            elif cmd == "/routing":
                settings.smart_routing = not settings.smart_routing
                state = "on" if settings.smart_routing else "off"
                from distiq_code.routing import is_ml_routing_active
                backend = "ml (BERT)" if is_ml_routing_active() else "regex"
                console.print(f" [dim]Smart routing: {state} ({backend})[/dim]")
                console.print()
                continue

            else:
                console.print(f" [dim]Unknown: {cmd}. Type /help[/dim]")
                console.print()
                continue

        # --- Process image tags ---
        # If user_input contains [image: path], instruct Claude to read the file
        image_paths = IMAGE_TAG_RE.findall(user_input)
        has_image = bool(image_paths)
        if has_image:
            clean_text = IMAGE_TAG_RE.sub("", user_input).strip()
            read_instructions = "\n".join(
                f"Read and analyze the image at: {p}" for p in image_paths
            )
            if clean_text:
                user_input = f"{read_instructions}\n\n{clean_text}"
            else:
                user_input = f"{read_instructions}\n\nDescribe what you see in the image."

        # --- Send message ---
        messages.append({"role": "user", "content": user_input})

        # Smart model routing
        routed_model = model
        complexity = ""
        if settings.smart_routing:
            from distiq_code.routing import classify_and_route
            routed_model, complexity = classify_and_route(user_input, len(messages), model)

        start_time = time.time()
        original_tokens = sum(len(m["content"]) for m in messages) // 4

        # Check cache first (skip for image messages — each image is unique)
        if cache and not has_image:
            try:
                cached_response, cache_stats = cache.get(user_input, routed_model)
                if cached_response:
                    latency_ms = (time.time() - start_time) * 1000
                    messages.append({"role": "assistant", "content": cached_response})

                    console.print()
                    console.print(cached_response, highlight=False, markup=False)
                    console.print()
                    console.print(f" [dim]⚡ cached · similarity {cache_stats.similarity:.0%} · {latency_ms:.0f}ms[/dim]")
                    console.print()

                    session_requests += 1
                    session_cache_hits += 1
                    session_tokens_saved += original_tokens
                    session_total_tokens += original_tokens

                    stats_tracker.record_request(
                        model=routed_model,
                        original_tokens=original_tokens,
                        compressed_tokens=0,
                        cache_hit=True,
                        cache_similarity=cache_stats.similarity,
                        latency_ms=latency_ms,
                        compression_enabled=False,
                        compression_ratio=1.0,
                    )
                    continue
            except Exception:
                pass

        # Compress if multi-turn
        messages_to_send = [m.copy() for m in messages]
        compressed_tokens = original_tokens
        compression_ratio = 1.0

        if compressor and len(messages_to_send) > 1:
            try:
                messages_to_send, comp_stats = compressor.compress_messages(messages_to_send)
                compressed_tokens = comp_stats.compressed_length // 4
                compression_ratio = comp_stats.compression_ratio
            except Exception:
                pass

        # Stream response with thinking spinner
        from distiq_code.auth.cli_provider import StreamResponse, ToolUseEvent
        from rich.markdown import Markdown

        full_response: list[str] = []
        status = console.status(" [dim]Thinking...[/dim]", spinner="dots")
        status.start()

        try:
            stream = await provider.complete(
                messages_to_send, routed_model, stream=True, allowed_tools=CHAT_TOOLS
            )
            async for chunk in stream:
                if isinstance(chunk, ToolUseEvent):
                    tool_desc = _format_tool_event(chunk)
                    status.update(f" [dim]{tool_desc}[/dim]")
                elif isinstance(chunk, str):
                    full_response.append(chunk)

            status.stop()
            console.print()
            # Render with Rich Markdown (tables, headers, code blocks)
            response_text = "".join(full_response)
            if response_text.strip():
                console.print(Markdown(response_text))
        except RuntimeError as e:
            status.stop()
            console.print(f"\n [red]Error: {e}[/red]")
            console.print()
            messages.pop()
            continue

        if not response_text:
            messages.pop()
            continue

        latency_ms = (time.time() - start_time) * 1000

        # Read real usage data from StreamResponse (if available)
        from distiq_code.auth.cli_provider import StreamResponse
        input_tok = 0
        output_tok = 0
        cost = 0.0
        api_ms = 0
        if isinstance(stream, StreamResponse):
            input_tok = stream.input_tokens
            output_tok = stream.output_tokens
            cost = stream.cost_usd
            api_ms = stream.duration_api_ms

        # Accumulate real session counters
        session_input_tokens += input_tok
        session_output_tokens += output_tok
        session_cost_usd += cost

        # Calculate routing savings (how much saved vs default model)
        if complexity and routed_model != model:
            from distiq_code.routing import estimate_savings
            routing_saved = estimate_savings(routed_model, model, input_tok, output_tok)
            session_routing_saved += routing_saved

        # Update state
        messages.append({"role": "assistant", "content": response_text})

        if cache and not has_image:
            try:
                cache.set(user_input, response_text, routed_model)
            except Exception:
                pass

        tokens_saved = original_tokens - compressed_tokens if compressor else 0
        session_requests += 1
        session_tokens_saved += tokens_saved
        session_total_tokens += compressed_tokens

        stats_tracker.record_request(
            model=routed_model,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            cache_hit=False,
            cache_similarity=0.0,
            latency_ms=latency_ms,
            compression_enabled=compressor is not None,
            compression_ratio=compression_ratio,
        )

        # Status line — real data from Claude CLI when available, fallback to estimates
        routing_str = f" (auto: {complexity})" if complexity else ""
        routing_saved_str = ""
        if complexity and routed_model != model and (input_tok or output_tok):
            from distiq_code.routing import estimate_savings
            saved_usd = estimate_savings(routed_model, model, input_tok, output_tok)
            if saved_usd > 0.0001:
                routing_saved_str = f" · [green]saved ${saved_usd:.4f}[/green]"

        if input_tok or output_tok:
            comp_str = f" · comp {_format_tokens(tokens_saved)}" if tokens_saved > 0 else ""
            console.print(
                f" [dim]{_format_tokens(input_tok)} in / {_format_tokens(output_tok)} out{comp_str} · "
                f"${cost:.4f} · {routed_model}{routing_str} · {latency_ms / 1000:.1f}s[/dim]"
                f"{routing_saved_str}"
            )
        else:
            comp_str = f" · comp {_format_tokens(tokens_saved)}" if tokens_saved > 0 else ""
            console.print(
                f" [dim]{_format_tokens(compressed_tokens)} tokens{comp_str} · "
                f"{routed_model}{routing_str} · {latency_ms / 1000:.1f}s[/dim]"
                f"{routing_saved_str}"
            )
        console.print()

    # Session summary on exit
    if session_requests > 0:
        total_tok = session_input_tokens + session_output_tokens
        savings_str = ""
        if session_routing_saved > 0.001:
            savings_str = f" · [green]routing saved ${session_routing_saved:.2f}[/green]"
        if total_tok > 0:
            console.print(
                f" [dim]Session: {session_requests} requests · "
                f"{session_cache_hits} cached · "
                f"{_format_tokens(total_tok)} tokens · "
                f"${session_cost_usd:.2f}[/dim]"
                f"{savings_str}"
            )
        else:
            console.print(
                f" [dim]Session: {session_requests} requests · "
                f"{session_cache_hits} cached · "
                f"{_format_tokens(session_tokens_saved)} saved[/dim]"
                f"{savings_str}"
            )
    console.print(" [dim]Bye![/dim]")
    console.print()


@app.command()
def cursor_test(
    prompt: str = typer.Option("Hello, test message", help="Test prompt to send"),
    model: str = typer.Option("cursor-small", help="Model: cursor-small, claude-4.5-opus-high, etc."),
):
    """
    Test Cursor API connection.

    Extracts tokens from Cursor database and sends a test request.
    """
    asyncio.run(_cursor_test(prompt, model))


async def _cursor_test(prompt: str, model: str):
    """Test Cursor API connectivity."""
    from distiq_code.cursor import CursorClient

    console.print("[bold blue]Testing Cursor API connection...[/bold blue]\n")

    try:
        # Initialize client (auto-extracts tokens)
        console.print("📂 Extracting tokens from Cursor database...")
        client = CursorClient()

        console.print(f"✅ Authenticated as: [cyan]{client.tokens.email}[/cyan]")
        console.print(f"✅ Membership: [cyan]{client.tokens.membership_type or 'unknown'}[/cyan]")
        console.print()

        # Send test request
        console.print(f"📤 Sending request to model: [cyan]{model}[/cyan]")
        console.print(f"💬 Prompt: [dim]{prompt}[/dim]\n")

        console.print("🤖 Response:\n", style="bold")

        response_text = ""
        async for chunk in client.chat(prompt=prompt, model=model):
            console.print(chunk, end="")
            response_text += chunk

        console.print("\n")
        console.print(f"✅ [green]Success![/green] Received {len(response_text)} characters")

        await client.close()

    except FileNotFoundError as e:
        console.print(f"❌ [red]Error:[/red] {e}")
        console.print("\n[yellow]Make sure:[/yellow]")
        console.print("  1. Cursor IDE is installed")
        console.print("  2. You've logged in to Cursor")

    except Exception as e:
        console.print(f"❌ [red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())


@app.command()
def cursor_proxy(
    port: int = typer.Option(443, help="Port to run proxy on (443 for HTTPS)"),
    passthrough: bool = typer.Option(True, help="Passthrough mode (no modifications)"),
):
    """
    Start Cursor MITM proxy server.

    Setup:
        1. Run this command AS ADMINISTRATOR: distiq-code cursor-proxy
        2. Add to hosts file (use setup_hosts.bat):
           127.0.0.1 api2.cursor.sh
        3. Restart Cursor IDE
        4. All requests will go through our proxy

    To restore:
        - Run restore_hosts.bat
        - Restart Cursor IDE
    """
    import os
    import uvicorn
    from pathlib import Path
    from distiq_code.cursor.proxy import create_mitm_app

    console.print(f"[bold blue]Starting Cursor MITM Proxy on port {port}...[/bold blue]\n")

    # Check if running as admin (required for port 443)
    if port < 1024:
        import ctypes
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        except:
            is_admin = os.geteuid() == 0 if hasattr(os, 'geteuid') else False

        if not is_admin:
            console.print("[red]ERROR: Port 443 requires Administrator privileges![/red]")
            console.print("Please run as Administrator (right-click → Run as Administrator)\n")
            raise typer.Exit(1)

    console.print("[yellow]Setup Instructions:[/yellow]")
    console.print("1. Add this line to your hosts file:")
    console.print("   [cyan]127.0.0.1 api2.cursor.sh[/cyan]")
    console.print("\n   Quick setup: Run [cyan]setup_hosts.bat[/cyan] as Administrator")
    console.print("\n2. Restart Cursor IDE")
    console.print("3. Cursor traffic will flow through this proxy\n")

    console.print(f"[green]Proxy mode:[/green] {'Passthrough (no modifications)' if passthrough else 'Smart routing'}\n")

    # Check for SSL certificate
    cert_path = Path("cursor_cert.pem")
    key_path = Path("cursor_key.pem")

    if not cert_path.exists() or not key_path.exists():
        console.print("[yellow]SSL certificate not found. Generating...[/yellow]")
        from distiq_code.cursor.ssl_gen import generate_ssl_cert
        generate_ssl_cert()
        console.print("[green]SSL certificate generated![/green]\n")

    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Create FastAPI app
    enable_routing = not passthrough
    app_instance = create_mitm_app(enable_routing=enable_routing)

    # Run with uvicorn (HTTPS)
    uvicorn.run(
        app_instance,
        host="127.0.0.1",
        port=port,
        ssl_keyfile=str(key_path),
        ssl_certfile=str(cert_path),
        log_level="info",
    )


if __name__ == "__main__":
    app()
