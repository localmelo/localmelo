"""One error shape for API v2, and the handlers that give it.

Every ``/api`` failure is ``{"error": {"code", "message", "details"}}``.
A route raises :class:`ApiError`; FastAPI's validation errors become a 400
with the first field named; a store error a route lets through is answered by
its kind (a chat, task or chore that is not there is a 404, a refused change a
409) and any other store failure is a 503 ``store_unavailable``, or a 404 when
the chat in its path is gone (:meth:`Console.store_api_error`); a busy session
becomes a 409 ``session_busy``; and anything unhandled becomes a 500 whose
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

from localmelo.support.gateway.chore_store import (
    ChoreLimitError,
    ChoreNotFoundError,
    ChoreStaleError,
)
from localmelo.support.gateway.guard import (
    _SECURITY_HEADERS,
    API_CSP,
    is_api_path,
    is_upload_request,
)
from localmelo.support.gateway.i18n import said_text, t
from localmelo.support.gateway.task_store import (
    TaskNotFoundError,
    TaskNotPausedError,
    TaskStoreError,
)
from localmelo.support.sessions import display
from localmelo.support.sessions.store import (
    SessionBusyError,
    SessionStoreError,
    StoreOwnedError,
    UnknownSessionError,
)

logger = logging.getLogger(__name__)

# What an API v2 message says when another process holds the store's owner
# lock. It never names the other process: spec 2.2 keeps lease holder
# strings out of API messages. The terminal keeps the store's own text.
STORE_OWNED_MESSAGE: Final = (
    "Another melo gateway is using the console's session file. "
    "Stop it to use the console here."
)


def store_message(exc: SessionStoreError) -> str:
    """A store failure as an API v2 message, with the home folder written ``~``."""
    if isinstance(exc, StoreOwnedError):
        return STORE_OWNED_MESSAGE
    return display.home_labelled(str(exc))


def store_unavailable(exc: SessionStoreError) -> ApiError:
    """The 503 of a store call that failed: one fixed message, the cause in the log.

    The failure's own text can name a folder or repeat the database's words, so
    it is logged and never answered.
    """
    logger.warning("A session store call failed: %s", store_message(exc))
    if isinstance(exc, StoreOwnedError):
        return ApiError(503, "store_unavailable", STORE_OWNED_MESSAGE)
    return ApiError(503, "store_unavailable", t("errors.storeUnavailable"))


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
    """``field: problem`` for the first string, a key included, with a lone surrogate, else ``None``.

    SQLite stores text, and a lone surrogate (``"\\ud800"``, which a JSON body
    can carry and Python accepts) encodes to nothing: written, it fails the
    turn after the route answered. A key is named by the object it is in.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return f"{path or 'request'}: the text must be valid UTF-8"
        return None
    if isinstance(value, Mapping):
        for key, inner in value.items():
            found = _utf8_problem(key, path) or _utf8_problem(
                inner, f"{path}.{key}" if path else str(key)
            )
            if found is not None:
                return found
        return None
    if isinstance(value, list | tuple):
        for index, inner in enumerate(value):
            found = _utf8_problem(inner, f"{path}.{index}" if path else str(index))
            if found is not None:
                return found
    return None


def _body_problem(body: bytes) -> str | None:
    """Why a JSON body is a 400, or ``None``: not UTF-8, not JSON, or a lone surrogate.

    A body nested deeper than the stack allows (to parse or to walk) is not JSON
    this gateway reads.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return "Request body must be valid UTF-8."
    try:
        return _utf8_problem(json.loads(text), "")
    except (ValueError, RecursionError):
        return "Malformed JSON request body."


def _query_is_valid_utf8(scope: Scope) -> bool:
    """Whether the raw query string decodes as UTF-8 after percent-decoding."""
    try:
        unquote_to_bytes(scope.get("query_string", b"")).decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


class ApiMiddleware:
    """Add ``no-store`` and security headers to ``/api`` responses and validate bodies.

    A request outside ``/api`` passes through untouched. Inside it, the query
    string and a JSON body (not a picture's bytes) must be UTF-8, the body must
    parse as JSON, and none of its strings may hold a lone surrogate; all of
    that is judged before any route runs. An exception that no handler caught
    becomes the 500 shape, unless the response has already started, when it is
    re-raised.

    The size and the Content-Type of a body are
    :class:`~localmelo.support.gateway.guard.LoopbackGuard`'s, which wraps the
    console in the gateway and has refused a body that is too large or of
    another type by now.
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

        chunks: list[bytes] = []
        disconnected = False
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        full_body = b"".join(chunks)
        # A picture's bytes go to one route and are not JSON.
        if full_body and not is_upload_request(scope):
            problem = _body_problem(full_body)
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


def error_message(exc: Exception) -> str:
    """What a refusal says: the words it keeps, in the request's language, else its English message."""
    words = getattr(exc, "words", None)
    return said_text(words) if words else str(exc)


def _kind_error(exc: SessionStoreError) -> ApiError | None:
    """The answer for a store error that is no store failure, or ``None`` for one.

    A chat, task or chore that is not there, or a change the stored state does
    not allow, is the request's doing and says so; a route that knows better
    answers first with its own :class:`ApiError`.
    """
    if isinstance(exc, UnknownSessionError):
        return ApiError(404, "not_found", t("errors.sessionDeleted"))
    if isinstance(exc, TaskNotFoundError | ChoreNotFoundError):
        return ApiError(404, "not_found", error_message(exc))
    if isinstance(exc, ChoreStaleError):
        return ApiError(409, "stale_revision", error_message(exc))
    if isinstance(exc, ChoreLimitError):
        return ApiError(409, "chore_limit", error_message(exc))
    if isinstance(exc, TaskNotPausedError):
        return ApiError(409, "not_paused", error_message(exc))
    if isinstance(exc, TaskStoreError):
        return ApiError(409, "invalid_transition", error_message(exc))
    return None


async def _store_error_handler(request: Request, exc: Exception) -> Response:
    """Any store error a route lets through, as API v2 maps it.

    A chat, task or chore that is not there, or a change the task store
    refuses, is answered by its kind (:func:`_kind_error`). Any other failure
    is the store's: the chat in the path, if any, decides between 404 and 503
    (:meth:`Console.store_api_error`), and a route whose chat id is not a path
    parameter maps its own.
    """
    assert isinstance(exc, SessionStoreError)
    if not is_api_path(request.url.path):
        raise exc
    error = _kind_error(exc)
    if error is None:
        console = request.app.state.console
        error = await console.store_api_error(
            exc, request.path_params.get("session_id")
        )
    return api_error_response(error)


async def _busy_handler(request: Request, exc: Exception) -> Response:
    if not is_api_path(request.url.path):
        raise exc
    return error_response(409, "session_busy", t("errors.turnRunning"))


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
    app.add_exception_handler(SessionStoreError, _store_error_handler)
    app.add_exception_handler(SessionBusyError, _busy_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_middleware(ApiMiddleware)
