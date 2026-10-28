"""One error shape for API v2, and the handlers that give it.

Every ``/api`` failure is ``{"error": {"code", "message", "details"}}``.
A route raises :class:`ApiError`; FastAPI's validation errors become a 400
with the first field named, and anything unhandled becomes a 500 whose
message says nothing about the failure. The legacy ``/v1`` routes keep
FastAPI's own 404 and 422 bodies: these handlers are installed by the
console and leave a request outside ``/api`` alone.

:class:`ApiMiddleware` also puts ``Cache-Control: no-store`` on every
``/api`` response, the successes and the errors alike.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# The message of a failure the gateway did not expect. It names neither the
# traceback nor any value of the request.
INTERNAL_ERROR: Final = "Internal error; see the gateway log."


class ApiError(Exception):
    """A refusal with the status, code and message API v2 documents.

    ``details`` is a JSON object, ``{}`` by default. A route may raise it
    from any depth; the console's handler turns it into the one error
    shape.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = {} if details is None else details


def error_body(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The body of every API v2 error."""
    return {"error": {"code": code, "message": message, "details": details or {}}}


def error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, details),
        headers={"Cache-Control": "no-store"},
    )


def api_error_response(exc: ApiError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


def is_api_path(path: str) -> bool:
    """Whether *path* is ``/api`` or below it, also with leading slashes."""
    path = "/" + path.lstrip("/")
    return path == "/api" or path.startswith("/api/")


def validation_message(exc: RequestValidationError) -> str:
    """``field: problem`` for the first invalid value, without the value."""
    errors = exc.errors()
    if not errors:
        return "request: Invalid request"
    error = errors[0]
    location = error.get("loc") or ()
    names = [str(part) for part in location if isinstance(part, str)]
    field = ".".join(names) or "request"
    return f"{field}: {error.get('msg', 'Invalid request')}"


class ApiMiddleware:
    """Add ``no-store`` to ``/api`` responses and catch what a route did not.

    A request outside ``/api`` passes through untouched. Inside it, an
    exception that no handler caught becomes the 500 shape, unless the
    response has already started, when it is re-raised.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_api_path(str(scope.get("path", ""))):
            await self.app(scope, receive, send)
            return
        started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = list(message.get("headers", ()))
                if not any(key.lower() == b"cache-control" for key, _ in headers):
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            logger.exception("Unhandled API error")
            if started:
                raise
            response = error_response(500, "internal_error", INTERNAL_ERROR)
            await response(scope, receive, send_with_headers)


async def _api_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, ApiError)
    return api_error_response(exc)


async def _http_exception_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    if not is_api_path(request.url.path):
        return await http_exception_handler(request, exc)
    if exc.status_code == 404:
        return error_response(404, "not_found", "Not found.")
    if exc.status_code == 405:
        return error_response(405, "method_not_allowed", "Method not allowed.")
    detail = exc.detail if isinstance(exc.detail, str) else "Invalid request"
    return error_response(exc.status_code, "invalid_request", detail)


async def _validation_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)
    if not is_api_path(request.url.path):
        return await request_validation_exception_handler(request, exc)
    return error_response(400, "invalid_request", validation_message(exc))


def install(app: FastAPI) -> None:
    """Install the API v2 error handlers and middleware on *app*."""
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_middleware(ApiMiddleware)
