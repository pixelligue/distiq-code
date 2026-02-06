"""Uvicorn server entry point."""

import uvicorn
from loguru import logger

from distiq_code.config import settings
from distiq_code.server.app import create_app


def run_server():
    """Run FastAPI server with Uvicorn."""
    app = create_app()

    logger.info(f"Starting server on {settings.proxy_host}:{settings.proxy_port}")

    uvicorn.run(
        app,
        host=settings.proxy_host,
        port=settings.proxy_port,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
