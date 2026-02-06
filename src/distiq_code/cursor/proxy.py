"""Cursor MITM proxy server.

Intercepts requests from Cursor IDE via hosts file redirect,
optionally modifies them (smart routing), and forwards to real Cursor API.
"""

import asyncio
import gzip
import struct
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from distiq_code.config import settings
from distiq_code.cursor.protobuf_parser import ProtobufParser
from distiq_code.cursor.router import CursorRouter


class CursorMITMProxy:
    """
    MITM proxy for Cursor IDE.

    Setup:
        1. Add to hosts file: 127.0.0.1 api2.cursor.sh
        2. Start proxy on localhost:443 (or use nginx reverse proxy)
        3. Cursor IDE requests go to our proxy
        4. We forward to real Cursor API

    Features:
        - Transparent passthrough (Cursor works normally)
        - Request/response logging
        - Model substitution (Opus → Sonnet for simple tasks)
        - Statistics tracking
    """

    REAL_API_BASE = "https://api2.cursor.sh"

    def __init__(self, enable_routing: bool = False):
        """
        Initialize MITM proxy.

        Args:
            enable_routing: Enable smart model routing (default: False for passthrough)
        """
        # HTTP/2 client for forwarding to real API
        self.client = httpx.AsyncClient(http2=True, timeout=120.0)

        # Smart router
        self.router = CursorRouter()
        self.enable_routing = enable_routing

        logger.info(f"Cursor MITM proxy initialized (routing: {enable_routing})")

    async def forward_request(
        self,
        method: str,
        path: str,
        headers: dict,
        body: bytes,
        modify_model: bool = False,
    ) -> tuple[int, dict, bytes]:
        """
        Forward request to real Cursor API.

        Args:
            method: HTTP method (POST, GET, etc.)
            path: Request path
            headers: Request headers
            body: Request body (binary)
            modify_model: If True, apply smart routing (change model to cheaper one)

        Returns:
            Tuple of (status_code, response_headers, response_body)
        """
        # Build real API URL
        url = f"{self.REAL_API_BASE}{path}"

        # Clean headers (remove host, content-length - httpx will set them)
        forward_headers = {
            k: v for k, v in headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        # Optional: Modify request body (model substitution)
        if modify_model and path == "/aiserver.v1.ChatService/StreamUnifiedChatWithTools":
            body = self._modify_model_in_request(body)

        # Log request
        logger.info(
            f"Forwarding {method} {path} "
            f"(body: {len(body)} bytes, modified: {modify_model})"
        )

        # Forward to real API
        response = await self.client.request(
            method=method,
            url=url,
            headers=forward_headers,
            content=body,
        )

        # Extract response
        status_code = response.status_code
        response_headers = dict(response.headers)
        response_body = response.content

        logger.info(
            f"Received response {status_code} "
            f"(body: {len(response_body)} bytes)"
        )

        return status_code, response_headers, response_body

    async def forward_streaming_request(
        self,
        method: str,
        path: str,
        headers: dict,
        body: bytes,
        modify_model: bool = False,
    ):
        """
        Forward streaming request to real Cursor API.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            body: Request body
            modify_model: If True, apply smart routing

        Yields:
            Response chunks (bytes)
        """
        # Build real API URL
        url = f"{self.REAL_API_BASE}{path}"

        # Clean headers
        forward_headers = {
            k: v for k, v in headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        # Optional: Modify request body
        if modify_model and path == "/aiserver.v1.ChatService/StreamUnifiedChatWithTools":
            body = self._modify_model_in_request(body)

        logger.info(
            f"Forwarding streaming {method} {path} "
            f"(body: {len(body)} bytes, modified: {modify_model})"
        )

        # Stream from real API
        async with self.client.stream(
            method=method,
            url=url,
            headers=forward_headers,
            content=body,
        ) as response:
            logger.info(f"Streaming response {response.status_code}")

            # Forward status and headers first
            # (handled by FastAPI StreamingResponse)

            # Stream body chunks
            async for chunk in response.aiter_bytes():
                yield chunk

    def _modify_model_in_request(self, body: bytes) -> bytes:
        """
        Modify model in request body (smart routing).

        Parses protobuf envelope, extracts model field, and substitutes
        with cheaper model if appropriate.

        Args:
            body: Original request body

        Returns:
            Modified request body with new model
        """
        try:
            # Parse envelope
            is_compressed, protobuf_payload = ProtobufParser.parse_envelope(body)

            # Extract original model
            original_model = ProtobufParser.extract_model(protobuf_payload)
            if not original_model:
                logger.warning("Could not extract model from request")
                return body

            # Extract messages for classification
            messages = ProtobufParser.extract_messages(protobuf_payload)

            # Classify request
            complexity = self.router.classify_request(messages)

            # Route to appropriate model
            routed_model, reason = self.router.route_model(
                original_model=original_model,
                complexity=complexity,
                enable_routing=self.enable_routing,
            )

            # Calculate savings
            savings = self.router.calculate_savings(original_model, routed_model)

            # Log routing decision
            routing_log = self.router.format_routing_log(
                original_model, routed_model, reason, savings
            )
            logger.info(f"[ROUTING] {routing_log}")

            # If model changed, modify protobuf
            if routed_model != original_model:
                logger.info(f"Replacing model: {original_model} → {routed_model}")
                modified_payload = ProtobufParser.replace_model(protobuf_payload, routed_model)

                # Re-wrap with envelope
                modified_body = ProtobufParser.wrap_envelope(modified_payload, compress=is_compressed)
                return modified_body

            # No change needed
            return body

        except Exception as e:
            logger.error(f"Failed to modify request: {e}", exc_info=True)
            return body

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# FastAPI app factory
def create_mitm_app(enable_routing: bool = False) -> FastAPI:
    """
    Create FastAPI app for MITM proxy.

    Args:
        enable_routing: Enable smart model routing

    Returns:
        FastAPI application
    """
    app = FastAPI(title="Cursor MITM Proxy", version="0.1.0")

    # Global proxy instance
    proxy = CursorMITMProxy(enable_routing=enable_routing)

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        await proxy.close()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def proxy_request(request: Request, path: str):
        """
        Proxy all requests to real Cursor API.

        Args:
            request: FastAPI request
            path: Request path

        Returns:
            Response from real API
        """
        method = request.method
        headers = dict(request.headers)
        body = await request.body()

        # Determine if we should modify (smart routing)
        modify_model = proxy.enable_routing

        # Check if streaming response expected
        # Cursor API streaming endpoints always use POST to /aiserver.v1.ChatService/*
        is_streaming = (
            method == "POST" and
            "ChatService" in path
        )

        if is_streaming:
            # Stream response
            return StreamingResponse(
                proxy.forward_streaming_request(
                    method=method,
                    path=f"/{path}",
                    headers=headers,
                    body=body,
                    modify_model=modify_model,
                ),
                media_type="application/connect+proto",
            )
        else:
            # Regular response
            status_code, response_headers, response_body = await proxy.forward_request(
                method=method,
                path=f"/{path}",
                headers=headers,
                body=body,
                modify_model=modify_model,
            )

            # Clean response headers (remove problematic ones)
            clean_headers = {
                k: v for k, v in response_headers.items()
                if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")
            }

            return Response(
                content=response_body,
                status_code=status_code,
                headers=clean_headers,
            )

    return app
