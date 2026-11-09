import time
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.config.settings import settings


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Sliding-window IP-based rate limiter."""
    def __init__(self, app):
        super().__init__(app)
        self.requests: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next):
        # Exclude static endpoints and health checks from rate limiting
        if request.url.path.startswith("/health") or request.url.path.startswith("/docs") or request.url.path == "/":
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        limit = settings.RATE_LIMIT_REQUESTS

        # Clean old timestamps
        history = self.requests.get(client_ip, [])
        history = [ts for ts in history if now - ts < window]

        if len(history) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit of {limit} requests per {window}s exceeded. Please slow down.",
                    "retry_after_seconds": int(window - (now - history[0])) if history else window
                }
            )

        history.append(now)
        self.requests[client_ip] = history

        return await call_next(request)
