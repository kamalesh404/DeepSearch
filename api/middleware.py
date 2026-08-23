"""ASGI middleware: rate limiting, request logging, CORS, API key validation."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("deepsearch.api")

API_KEYS = {"demo-key-123", "local-dev"}
RATE_LIMIT = 60  # requests per window per client


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-client limiter keyed by API key or client IP."""

    def __init__(self, app, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.window = window_seconds
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        identity = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
        now = time.monotonic()
        bucket = self.hits[identity]
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            return JSONResponse({"error": "rate_limited", "retry_after": self.window}, status_code=429)
        bucket.append(now)
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests lacking a valid ``X-API-Key`` header on protected paths."""

    EXEMPT_PATHS = {"/docs", "/openapi.json", "/health", "/redoc"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        key = request.headers.get("X-API-Key")
        if key not in API_KEYS:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return await call_next(request)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Structured access log with latency for observability."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("%s %s -> %d (%.1f ms)", request.method,
                    request.url.path, response.status_code, elapsed_ms)
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response


def install_middleware(app: FastAPI) -> None:
    """Attach all middleware layers to the application, outermost first."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    app.add_middleware(RequestLogMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RateLimitMiddleware)
