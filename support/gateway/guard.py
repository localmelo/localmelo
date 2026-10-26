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

# Headers of every HTTP response: no page may show the gateway's pages in
# a frame, where a page on another site could trick the user into
# clicking them.
_NO_FRAMING_HEADERS = (
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
)


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
            headers = [*message.get("headers", ()), *_NO_FRAMING_HEADERS]
            message = {**message, "headers": headers}
        await send(message)

    return send_forbidding_framing


def _one_of(values: list[str], allowed: frozenset[str]) -> bool:
    """Whether a header's *values* are a single value in *allowed*, ignoring case."""
    return len(values) == 1 and values[0].lower() in allowed


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
    3. A request under ``/v1/`` whose Sec-Fetch-Site header is present and
       is neither ``same-origin`` nor ``none`` gets 403 ``foreign_origin``.
       A link from another site to the web UI at ``/`` still opens it.

    Host names compare ignoring case. A refused request gets the gateway's
    error shape plus its code, ``{"error": "...", "code": "..."}``; a
    refused websocket handshake is closed. Other scopes, such as lifespan,
    pass through.

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
        response = JSONResponse(
            status_code=status_code, content={"error": error, "code": code}
        )
        await response(scope, receive, send)

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
        if path.startswith("/v1/"):
            for site in headers.get(b"sec-fetch-site", []):
                if site.lower() not in _OWN_FETCH_SITES:
                    return 403, "foreign_origin", self._origin_error

        return None
