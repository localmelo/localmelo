"""Refuse requests that are not addressed to the local gateway.

The gateway runs tasks with tools and has no authentication. On a
loopback address only programs on this machine can reach it, but a
browser on this machine sends requests for any web page: through DNS
rebinding, a page on another domain becomes same-origin with the
gateway, and any page can send it simple cross-origin requests.
:class:`LoopbackGuard` refuses both before a request reaches any route.
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

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


def is_loopback(host: str) -> bool:
    """Whether the listen address *host* is 127.0.0.1, localhost or ::1."""
    return host.lower() in _LOOPBACK_HOSTS


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
        origin_checked = (
            scope["type"] == "websocket"
            or scope.get("method") in _ORIGIN_CHECKED_METHODS
        )
        if (
            origin_checked
            and origins is not None
            and not _one_of(origins, self._origins)
        ):
            return 403, "foreign_origin", self._origin_error

        if scope.get("path", "").startswith("/v1/"):
            for site in headers.get(b"sec-fetch-site", []):
                if site.lower() not in _OWN_FETCH_SITES:
                    return 403, "foreign_origin", self._origin_error

        return None
