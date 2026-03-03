"""
In-memory sliding window rate limiter.

Tracks requests per user (or per IP if no auth) using a sliding window.
Returns HTTP 429 when the limit is exceeded.
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter middleware.

    Parameters
    ----------
    app : FastAPI app
    requests_per_minute : int
        Maximum requests allowed per minute per client.
    enabled : bool
        Whether rate limiting is active.
    """

    def __init__(self, app, *, requests_per_minute: int = 60, enabled: bool = False):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.enabled = enabled
        # {client_key: [timestamp, ...]}
        self._windows: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        """Identify the client: use X-API-Key if present, else IP."""
        api_key = request.headers.get("x-api-key", "")
        if api_key:
            return f"key:{api_key[:16]}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    def _prune(self, timestamps: list[float], now: float) -> list[float]:
        """Remove timestamps older than 60 seconds."""
        cutoff = now - 60.0
        return [t for t in timestamps if t > cutoff]

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        now = time.monotonic()
        key = self._client_key(request)

        # Prune old entries
        self._windows[key] = self._prune(self._windows[key], now)
        remaining = self.rpm - len(self._windows[key])

        if remaining <= 0:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"X-RateLimit-Remaining": "0", "Retry-After": "60"},
            )

        # Record this request
        self._windows[key].append(now)
        remaining -= 1

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        return response
