"""Set request boundaries before routing or parsing untrusted data."""

import json
import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from apps.api.adapters import Authenticator, RateLimiter
from apps.api.contracts import APIError, ErrorDetail, ErrorResponse


def error_response(
    error: APIError, request_id: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(code=error.code, message=error.message, request_id=request_id)
    )
    return JSONResponse(payload.model_dump(), status_code=error.status, headers=headers)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, separators=(",", ":")))


class RequestBoundary:
    def __init__(
        self,
        app: ASGIApp,
        *,
        auth: Authenticator,
        limiter: RateLimiter,
        logger: logging.Logger,
        upload_bytes: int,
        max_active: int = 8,
    ) -> None:
        self.app, self.auth, self.limiter = app, auth, limiter
        self.logger, self.upload_bytes, self.max_active = (
            logger,
            upload_bytes,
            max_active,
        )
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        path = scope["path"]
        root = scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        protected = path.startswith("/api/v1/") and path not in {
            "/api/v1/health",
            "/api/v1/ready",
            "/api/v1/openapi.json",
        }
        incoming = Headers(scope=scope)
        status = 500
        started = False
        acquired = False
        start = perf_counter()

        async def send_response(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Content-Type-Options"] = "nosniff"
                if path.startswith("/api/v1/"):
                    headers["Cache-Control"] = "no-store"
            await send(message)

        async def reject(
            error: APIError, headers: dict[str, str] | None = None
        ) -> None:
            await error_response(error, request_id, headers)(
                scope, receive, send_response
            )

        try:
            if protected:
                owner = self.auth.authenticate(incoming.get("authorization"))
                if owner is None:
                    await reject(
                        APIError(
                            401, "unauthorized", "A valid bearer token is required."
                        ),
                        {"WWW-Authenticate": "Bearer"},
                    )
                    return
                state["owner"] = owner
                retry_after = self.limiter.retry_after(owner)
                if retry_after is not None:
                    await reject(
                        APIError(429, "rate_limited", "Request limit exceeded."),
                        {"Retry-After": str(retry_after)},
                    )
                    return
                if self.active >= self.max_active:
                    await reject(APIError(503, "server_busy", "Research API is busy."))
                    return
                self.active += 1
                acquired = True

            limit = self.upload_bytes if path == "/api/v1/scans" else 65_536
            used = 0
            if protected and scope["method"] == "POST":
                lengths = incoming.getlist("content-length")
                if len(lengths) > 1 or (
                    lengths
                    and (
                        not lengths[0].isascii()
                        or not lengths[0].isdecimal()
                        or len(lengths[0]) > 20
                    )
                ):
                    await reject(
                        APIError(400, "invalid_length", "Invalid Content-Length.")
                    )
                    return
                if lengths and int(lengths[0]) > limit:
                    await reject(
                        APIError(
                            413,
                            "body_too_large",
                            "Request body exceeds its byte limit.",
                        )
                    )
                    return

            async def bounded_receive() -> Message:
                nonlocal used
                message = await receive()
                if message["type"] == "http.request" and protected:
                    used += len(message.get("body", b""))
                    if used > limit:
                        raise HTTPException(status_code=413)
                return message

            await self.app(scope, bounded_receive, send_response)
        except APIError as exc:
            if started:
                raise
            await reject(exc)
        except Exception as exc:
            if started:
                raise
            log_event(
                self.logger,
                "request_failure",
                request_id=request_id,
                exception=type(exc).__name__,
            )
            await reject(
                APIError(
                    500,
                    "internal_error",
                    "Research service could not complete the request.",
                )
            )
        finally:
            if acquired:
                self.active -= 1
            route = scope.get("route")
            log_event(
                self.logger,
                "request_complete",
                request_id=request_id,
                method=scope["method"],
                route=getattr(route, "path", None),
                status=status,
                duration_ms=round((perf_counter() - start) * 1000, 2),
            )
