"""Refuse requests that are not addressed to the local gateway.

The gateway runs tasks with tools and has no authentication. On a
loopback address only programs on this machine can reach it, but a
browser on this machine sends requests for any web page: through DNS
rebinding, a page on another domain becomes same-origin with the
gateway, any page can send it simple cross-origin requests, and any
page can show the gateway's pages in a frame. :class:`LoopbackGuard`
refuses the requests before they reach any route, and forbids framing.
"""

from __future__ import annotations

import ipaddress
import socket

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Names of a loopback address, compared ignoring case.
_LOOPBACK_NAMES = frozenset({"localhost", "localhost."})

# The names of the gateway in a Host header or an origin, IPv6 in brackets.
_GATEWAY_NAMES = ("127.0.0.1", "localhost", "[::1]")

# Requests that a page on another origin must not send: those that can
# change something, and the preflight that asks to send them.
_ORIGIN_CHECKED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE", "OPTIONS"})

# Sec-Fetch-Site values of a request from the gateway's own page, or from
# the user (typing the URL, a bookmark).
_OWN_FETCH_SITES = frozenset({"same-origin", "none"})

# Close code of a refused websocket handshake: policy violation.
_WS_POLICY_VIOLATION = 1008

# Largest request body the gateway accepts, in bytes (1 MiB).
BODY_BYTES_LIMIT = 1_048_576
_TOO_LARGE_ERROR = f"Request body too large (max {BODY_BYTES_LIMIT} bytes)."
_UNSUPPORTED_MEDIA_TYPE_ERROR = "Content-Type must be application/json."

# The headers of an API v2 refusal: the console's middleware never sees it,
# because the guard is outermost, so it sets no-store itself.
_API_REFUSAL_HEADERS = {"cache-control": "no-store"}

# Headers of every HTTP response: no page may show the gateway's pages in
# a frame, where a page on another site could trick the user into
# clicking them, and browsers get the console's baseline security headers.
# The CSP rule is added to a response's own policy when it has one, so a
# blob's ``default-src 'none'; sandbox`` stays one header.
_SECURITY_HEADERS = (
    (b"x-frame-options", b"DENY"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
)
_FRAMING_CSP = b"frame-ancestors 'none'"


def is_loopback(host: str) -> bool:
    """Whether the listen address *host* is a loopback address.

    Loopback addresses are localhost, ignoring case and a final dot, and
    the IP addresses in 127.0.0.0/8 and ::1, in any spelling that binding
    accepts (127.1, 2130706433, 0:0:0:0:0:0:0:1, ::ffff:127.0.0.1, IPv6
    with or without brackets). Other host names are not looked up, so they
    are not loopback addresses here.
    """
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if host.lower() in _LOOPBACK_NAMES:
        return True
    if not host:
        # An empty host listens on every address.
        return False
    try:
        # Parse the address as binding does, without looking up a name.
        infos = socket.getaddrinfo(
            host, None, type=socket.SOCK_STREAM, flags=socket.AI_NUMERICHOST
        )
    except (OSError, UnicodeError, ValueError):
        return False
    return bool(infos) and all(_is_loopback_ip(str(info[4][0])) for info in infos)


def _is_loopback_ip(address: str) -> bool:
    """Whether the IP *address* that getaddrinfo gave is a loopback address."""
    ip = ipaddress.ip_address(address.partition("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        # ::ffff:127.0.0.1 is 127.0.0.1; not every Python release says so.
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def _forbid_framing(send: Send) -> Send:
    """Wrap *send* so that the response carries the headers that forbid framing."""

    async def send_forbidding_framing(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", ()))
            # A response may set its own CSP, as a blob does. Appending a
            # second policy would make the browser intersect two headers;
            # one header with both directives says the same and keeps the
            # documented "every response has one CSP" contract.
            policies = [
                value
                for name, value in headers
                if name.lower() == b"content-security-policy"
            ]
            headers = [
                (name, value)
                for name, value in headers
                if name.lower() != b"content-security-policy"
            ]
            value = b"; ".join([*policies, _FRAMING_CSP]) if policies else _FRAMING_CSP
            headers.append((b"content-security-policy", value))
            present = {name.lower() for name, _ in headers}
            for name, header_value in _SECURITY_HEADERS:
                if name not in present:
                    headers.append((name, header_value))
            message = {**message, "headers": headers}
        await send(message)

    return send_forbidding_framing


def _is_api_path(scope: Scope) -> bool:
    """Whether the request is ``/api`` or below it, also with leading slashes."""
    path = "/" + str(scope.get("path", "")).lstrip("/")
    return path == "/api" or path.startswith("/api/")


def _one_of(values: list[str], allowed: frozenset[str]) -> bool:
    """Whether a header's *values* are a single value in *allowed*, ignoring case."""
    return len(values) == 1 and values[0].lower() in allowed


def is_json_content_type(values: list[str]) -> bool:
    """Whether *values* is a single ``application/json`` Content-Type header."""
    if len(values) != 1:
        return False
    parts = [part.strip() for part in values[0].split(";")]
    if not parts or parts[0].lower() != "application/json":
        return False
    for param in parts[1:]:
        if not param:
            return False
        key, sep, val = param.partition("=")
        if not sep:
            return False
        if (
            key.strip().lower() == "charset"
            and val.strip().strip('"').lower() != "utf-8"
        ):
            return False
    return True


def _content_length_too_large(values: list[str]) -> bool:
    """Whether any ``Content-Length`` header exceeds :data:`BODY_BYTES_LIMIT`."""
    for raw in values:
        text = raw.strip()
        if not text.isdigit():
            continue
        if int(text) > BODY_BYTES_LIMIT:
            return True
    return False


class LoopbackGuard:
    """ASGI middleware that refuses requests not addressed to the local gateway.

    For each HTTP request and websocket handshake, in order:

    1. The Host header must be ``127.0.0.1:<port>``, ``localhost:<port>``
       or ``[::1]:<port>`` (on port 80 also without the port). A missing,
       empty or repeated Host, or any other value, gets 421 with the code
       ``misdirected_request``. This refuses DNS rebinding.
    2. A POST, PUT, PATCH, DELETE or OPTIONS request, or a websocket
       handshake, whose Origin header is present must come from
       ``http://127.0.0.1:<port>``, ``http://localhost:<port>`` or
       ``http://[::1]:<port>``; otherwise it gets 403 with the code
       ``foreign_origin`` (``Origin: null`` included). A request without
       Origin, such as one from curl or a script, passes.
    3. A request under ``/v1/`` or ``/api/`` whose Sec-Fetch-Site header is
       present and is neither ``same-origin`` nor ``none`` gets 403
       ``foreign_origin``. A link from another site to the web UI at ``/``
       still opens it, and so does a navigation to ``/api/health`` typed into
       the address bar, whose ``Sec-Fetch-Site`` is ``none``.

    Host names compare ignoring case. A refused legacy request gets the
    gateway's error shape plus its code, ``{"error": "...", "code": "..."}``;
    a request under ``/api`` gets API v2's nested envelope instead, with
    ``Cache-Control: no-store``. A refused websocket handshake is closed.
    Other scopes, such as lifespan, pass through.

    Every HTTP response, a refusal or not, carries ``X-Frame-Options:
    DENY`` and ``Content-Security-Policy: frame-ancestors 'none'``, so no
    page can show the gateway's pages in a frame.
    """

    def __init__(self, app: ASGIApp, *, port: int) -> None:
        self.app = app
        hosts = {f"{name}:{port}" for name in _GATEWAY_NAMES}
        if port == 80:
            hosts.update(_GATEWAY_NAMES)
        self._hosts = frozenset(hosts)
        self._origins = frozenset(f"http://{host}" for host in hosts)
        self._host_error = (
            f"Host header must be 127.0.0.1:{port}, localhost:{port} "
            f"or [::1]:{port}"
        )
        self._origin_error = (
            f"Cross-origin request refused: only pages at "
            f"http://127.0.0.1:{port}, http://localhost:{port} or "
            f"http://[::1]:{port} may send it"
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http":
            send = _forbid_framing(send)

        refusal = self._refusal(scope)
        if refusal is None and scope["type"] == "http":
            receive, refusal = await self._read_body(scope, receive)
        if refusal is None:
            await self.app(scope, receive, send)
            return

        status_code, code, error = refusal
        if scope["type"] == "websocket":
            # Closing before the handshake is accepted refuses it.
            await send(
                {
                    "type": "websocket.close",
                    "code": _WS_POLICY_VIOLATION,
                    "reason": code,
                }
            )
            return
        if _is_api_path(scope):
            # API v2 has one error shape; the legacy routes keep theirs.
            content: dict[str, object] = {
                "error": {"code": code, "message": error, "details": {}}
            }
            response = JSONResponse(
                status_code=status_code,
                content=content,
                headers=_API_REFUSAL_HEADERS,
            )
        else:
            response = JSONResponse(
                status_code=status_code, content={"error": error, "code": code}
            )
        await response(scope, receive, send)

    async def _read_body(
        self, scope: Scope, receive: Receive
    ) -> tuple[Receive, tuple[int, str, str] | None]:
        """Read up to :data:`BODY_BYTES_LIMIT` bytes and check ``/api`` Content-Type."""
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
                    return receive, (413, "too_large", _TOO_LARGE_ERROR)
                chunks.append(chunk)
            if not message.get("more_body", False):
                break

        if total > 0 and _is_api_path(scope):
            content_types = [
                value.decode("latin-1")
                for name, value in scope.get("headers", ())
                if name.lower() == b"content-type"
            ]
            if not is_json_content_type(content_types):
                return receive, (
                    415,
                    "unsupported_media_type",
                    _UNSUPPORTED_MEDIA_TYPE_ERROR,
                )

        full_body = b"".join(chunks)
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

        return replay_receive, None

    def _refusal(self, scope: Scope) -> tuple[int, str, str] | None:
        """Status code, code and error of a request to refuse, or None."""
        headers: dict[bytes, list[str]] = {}
        for name, value in scope.get("headers", ()):
            headers.setdefault(name.lower(), []).append(value.decode("latin-1"))

        if not _one_of(headers.get(b"host", []), self._hosts):
            return 421, "misdirected_request", self._host_error

        origins = headers.get(b"origin")
        # The server passes the method on as sent, in any case.
        origin_checked = (
            scope["type"] == "websocket"
            or str(scope.get("method", "")).upper() in _ORIGIN_CHECKED_METHODS
        )
        if (
            origin_checked
            and origins is not None
            and not _one_of(origins, self._origins)
        ):
            return 403, "foreign_origin", self._origin_error

        # A path may start with several slashes, as in //v1/health.
        path = "/" + str(scope.get("path", "")).lstrip("/")
        if path.startswith("/v1/") or path == "/api" or path.startswith("/api/"):
            for site in headers.get(b"sec-fetch-site", []):
                if site.lower() not in _OWN_FETCH_SITES:
                    return 403, "foreign_origin", self._origin_error

        if scope["type"] == "http":
            content_lengths = headers.get(b"content-length", [])
            if _content_length_too_large(content_lengths):
                return 413, "too_large", _TOO_LARGE_ERROR
            if (
                _is_api_path(scope)
                and any(
                    raw.strip().isdigit() and int(raw.strip()) > 0
                    for raw in content_lengths
                )
                and not is_json_content_type(headers.get(b"content-type", []))
            ):
                return (
                    415,
                    "unsupported_media_type",
                    _UNSUPPORTED_MEDIA_TYPE_ERROR,
                )

        return None
