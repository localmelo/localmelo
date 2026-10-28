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

import json
import logging
from collections.abc import Mapping
from typing import Any, Final
from urllib.parse import unquote_to_bytes

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from localmelo.support.gateway.guard import (
    _SECURITY_HEADERS,
    _TOO_LARGE_ERROR,
    _UNSUPPORTED_MEDIA_TYPE_ERROR,
    API_CSP,
    BODY_BYTES_LIMIT,
    _content_length_too_large,
    is_json_content_type,
)

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
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    merged = {"Cache-Control": "no-store"}
    if headers:
        merged.update(headers)
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, details),
        headers=merged,
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


def _utf8_problem(value: Any, path: str) -> str | None:
    """``field: problem`` for the first string with a lone surrogate, else ``None``."""
    if isinstance(value, str):
        if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
            return f"{path or 'request'}: the text must be valid UTF-8"
        return None
    if isinstance(value, Mapping):
        for key, inner in value.items():
            if isinstance(key, str) and any(0xD800 <= ord(ch) <= 0xDFFF for ch in key):
                return f"{path or 'request'}: the text must be valid UTF-8"
            found = _utf8_problem(inner, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
        return None
    if isinstance(value, list | tuple):
        for index, inner in enumerate(value):
            found = _utf8_problem(inner, f"{path}.{index}" if path else str(index))
            if found is not None:
                return found
    return None


def _query_is_valid_utf8(scope: Scope) -> bool:
    """Whether the raw query string decodes as UTF-8 after percent-decoding."""
    raw = scope.get("query_string", b"")
    if not raw:
        return True
    try:
        decoded = unquote_to_bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return False
    return _utf8_problem(decoded, "query") is None


class ApiMiddleware:
    """Add ``no-store`` and security headers to ``/api`` responses and validate bodies.

    A request outside ``/api`` passes through untouched. Inside it, body size
    (1 MiB), ``Content-Type: application/json`` on non-empty bodies, UTF-8
    encoding and lone surrogates are checked before any route runs, and an
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
                present = {key.lower() for key, _ in headers}
                if b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                if b"content-security-policy" not in present:
                    headers.append((b"content-security-policy", API_CSP))
                for name, header_value in _SECURITY_HEADERS:
                    if name not in present:
                        headers.append((name, header_value))
                message = {**message, "headers": headers}
            await send(message)

        if not _query_is_valid_utf8(scope):
            response = error_response(
                400, "invalid_request", "Query string must be valid UTF-8."
            )
            await response(scope, receive, send_with_headers)
            return

        req_headers: dict[bytes, list[str]] = {}
        for name, value in scope.get("headers", ()):
            req_headers.setdefault(name.lower(), []).append(value.decode("latin-1"))

        content_lengths = req_headers.get(b"content-length", [])
        if _content_length_too_large(content_lengths):
            response = error_response(413, "too_large", _TOO_LARGE_ERROR)
            await response(scope, receive, send_with_headers)
            return
        if any(
            raw.strip().isdigit() and int(raw.strip()) > 0 for raw in content_lengths
        ) and not is_json_content_type(req_headers.get(b"content-type", [])):
            response = error_response(
                415, "unsupported_media_type", _UNSUPPORTED_MEDIA_TYPE_ERROR
            )
            await response(scope, receive, send_with_headers)
            return

        chunks: list[bytes] = []
        total = 0
        disconnected = False
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                break
            chunk = message.get("body", b"")
            if chunk:
                total += len(chunk)
                if total > BODY_BYTES_LIMIT:
                    response = error_response(413, "too_large", _TOO_LARGE_ERROR)
                    await response(scope, receive, send_with_headers)
                    return
                chunks.append(chunk)
            if not message.get("more_body", False):
                break

        full_body = b"".join(chunks)
        if total > 0:
            if not is_json_content_type(req_headers.get(b"content-type", [])):
                response = error_response(
                    415, "unsupported_media_type", _UNSUPPORTED_MEDIA_TYPE_ERROR
                )
                await response(scope, receive, send_with_headers)
                return
            try:
                text = full_body.decode("utf-8")
            except UnicodeDecodeError:
                response = error_response(
                    400, "invalid_request", "Request body must be valid UTF-8."
                )
                await response(scope, receive, send_with_headers)
                return
            try:
                parsed = json.loads(text)
            except ValueError:
                response = error_response(
                    400, "invalid_request", "Malformed JSON request body."
                )
                await response(scope, receive, send_with_headers)
                return
            problem = _utf8_problem(parsed, "")
            if problem is not None:
                response = error_response(400, "invalid_request", problem)
                await response(scope, receive, send_with_headers)
                return

        first_sent = False

        async def replay_receive() -> Message:
            nonlocal first_sent
            if not first_sent:
                first_sent = True
                if disconnected:
                    return {"type": "http.disconnect"}
                return {
                    "type": "http.request",
                    "body": full_body,
                    "more_body": False,
                }
            return await receive()

        try:
            await self.app(scope, replay_receive, send_with_headers)
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
    assert isinstance(exc, StarletteHTTPException)
    if not is_api_path(request.url.path):
        return await http_exception_handler(request, exc)
    if exc.status_code == 404:
        return error_response(404, "not_found", "Not found.")
    if exc.status_code == 405:
        return error_response(
            405, "method_not_allowed", "Method not allowed.", headers=exc.headers
        )
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
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_middleware(ApiMiddleware)
