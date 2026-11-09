import logging
import traceback
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("ecommerce_rag")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Centralized exception handling middleware producing standard JSON error envelopes."""
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            logger.error(f"Unhandled exception on {request.method} {request.url.path} [ReqID: {req_id}]: {exc}\n{traceback.format_exc()}")

            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred while processing your request.",
                    "detail": str(exc),
                    "request_id": req_id
                }
            )
