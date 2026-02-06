"""FastAPI application with lifespan management."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from distiq_code import __version__
from distiq_code.config import settings
from distiq_code.server.routes import health, messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Configure log level
    import sys
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    from distiq_code.config import get_available_features

    # Startup
    logger.info("Starting distiq-code proxy server")
    logger.info(f"Version: {__version__}")
    logger.info(f"Host: {settings.proxy_host}:{settings.proxy_port}")
    logger.info(f"Anthropic proxy: {settings.anthropic_api_base}")

    features = get_available_features()
    active = [k for k, v in features.items() if v]
    logger.info(f"Features: {', '.join(active)}")

    # Shared httpx client for Anthropic API forwarding
    app.state.http_client = httpx.AsyncClient(
        http2=True,
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )

    yield

    # Shutdown
    logger.info("Shutting down distiq-code proxy server")
    await app.state.http_client.aclose()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="distiq-code",
        description="Optimize AI coding assistant token usage",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for local proxy
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(messages.router, prefix="/v1", tags=["Anthropic Compatible"])

    # OpenAI-compatible chat router (requires ML deps for compression/cache)
    try:
        from distiq_code.server.routes import chat
        app.include_router(chat.router, prefix="/v1", tags=["OpenAI Compatible"])
    except ImportError:
        pass

    return app
