"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from distiq_code import __version__
from distiq_code.config import settings, get_available_features

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    features: dict[str, bool]
    compression_enabled: bool
    cache_enabled: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns basic service status, configuration, and available features.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        features=get_available_features(),
        compression_enabled=settings.compression_enabled,
        cache_enabled=settings.cache_enabled,
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Returns 200 if service is ready to accept requests.
    """
    return {"status": "ready"}
