"""Apps on this Mac that a page in background Chrome may reach: the listed sign-in helpers, and nothing else.

Some sign-in pages talk to an app on this computer. WeChat's quick login is
one: its sign-in frame (``https://open.weixin.qq.com``) asks the WeChat app
at ``https://localhost.weixin.qq.com`` on a few fixed ports, a name that leads
to this computer, and shows "quick login" only when the app answers.

Chrome's Local Network Access check stops a public page from reaching this
computer or its network unless the person allows it, and Chrome in the
background cannot ask. melo keeps that check on, for every page, frame,
worker and WebSocket, and grants no permission (a grant would be for a whole
site and every port). Instead melo answers the sign-in helper's requests
itself:

- Chrome pauses (DevTools ``Fetch``, for the whole browser) only the requests
  to a listed helper's addresses, and the requests whose address names this
  computer or its network as written (:meth:`LocalRule.patterns`).
- A request is **relayed** (:func:`relay`) only when its scheme, host and
  port are a listed helper's, the page that sent it is one of that helper's
  own pages (the ``Origin`` header, which Chrome sets and a page cannot), and
  the port is not one of melo's own (its gateway and console, and every model
  server in the config: :func:`localmelo.support.local_ports.own_ports`).
  melo connects to the address the list pins (never looking the name up),
  checks the certificate for the helper's name, follows no redirect, and gives
  Chrome the answer. Chrome itself never connects to this computer for it, so
  its check has nothing to stop; a redirect in the answer is a new request
  that Chrome checks again.
- Every other paused request whose address is this computer or its network
  is **blocked** before it is sent, top-level navigations included (Chrome's
  check leaves those alone), and told to the step (:func:`told`). A name
  that only resolves to this computer is not paused: Chrome's own check
  blocks it, and does not say so.
- Anything else goes on unchanged.

There is no setting for the list: it is :data:`SIGNIN_HELPERS`, in code.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ssl
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from localmelo.support import config, local_ports
from localmelo.support.web.guard import (
    LOCAL_NAMES,
    LOCAL_SUFFIXES,
    AddressError,
    NotPublicError,
    check_url,
)

# How long the app may take to answer a request that waits for the person (``SigninHelper.waits``).
PERSON_S = 180.0


@dataclass(frozen=True)
class SigninHelper:
    """A sign-in app's local service that its own sign-in page may reach.

    ``host`` and ``ports`` are where the page sends its requests (with
    ``scheme``); ``address`` is where melo connects instead of looking the
    name up; ``pages`` are the origins of the pages that may use it.
    ``waits`` are the paths the app holds open until the person answers its
    own window; their answer may take ``wait_s`` seconds, any other one
    :data:`RELAY_S`.
    """

    name: str
    scheme: Literal["http", "https"]
    host: str
    ports: tuple[int, ...]
    address: str
    pages: tuple[str, ...]
    waits: tuple[str, ...] = ()
    wait_s: float = PERSON_S


# The sign-in helpers a page may reach, and why each is here.
SIGNIN_HELPERS: tuple[SigninHelper, ...] = (
    # WeChat's quick login: the frame from open.weixin.qq.com looks for the
    # WeChat app's local service on these ports (a public name that leads to
    # 127.0.0.1), and shows the signed-in account's "quick login" button only
    # when the app answers. Safari lets it through; Chrome asks the person,
    # which Chrome in the background cannot.
    SigninHelper(
        name="WeChat",
        scheme="https",
        host="localhost.weixin.qq.com",
        ports=(14013, 14014, 14015, 13013, 13014, 13015),
        address="127.0.0.1",
        pages=("https://open.weixin.qq.com",),
        # The quick-login click: WeChat answers when the person presses Allow
        # or Decline in its window, and closes the window if the request ends.
        waits=("/api/authorize",),
    ),
)

# A relayed request's body and its answer, at most; the time it may take (connecting
# within the first part); how many at once.
MAX_BODY_BYTES = 1024 * 1024
MAX_ANSWER_BYTES = 1024 * 1024
RELAY_S = 10.0
CONNECT_S = 5.0
MAX_RELAYS = 8
METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
# Headers of the connection, not of the request or the answer: never passed on.
_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
)
# A cookie never goes to the app; Chrome has none at this stage, and melo adds none.
_NOT_SENT = _HOP | {"host", "accept-encoding", "cookie"}
# The only headers of the app's answer that reach the page: never a cookie. A
# redirect's ``location`` comes back too: Chrome checks where it leads, again.
_RETURNED = frozenset(
    {"content-type", "cache-control", "expires", "vary", "date", "location"}
)
# The addresses of this computer and its network that a URL can name as written.
_IPV4_STARTS = ("127.", "0.", "10.", "172.", "192.168.", "169.254.", "100.")
_SCHEME_PORTS = {"http": 80, "https": 443}
BLOCKED = "The page tried to reach an app on this Mac (port {port}); melo blocked it."


@dataclass(frozen=True)
class Verdict:
    """What melo does with a paused request: ``relay``, ``block`` or ``pass`` (go on)."""

    kind: Literal["relay", "block", "pass"]
    host: str = ""
    port: int = 0
    helper: SigninHelper | None = None


@dataclass(frozen=True)
class LocalAttempt:
    """A request melo blocked: where it went, the page that sent it, and the helper it named, if any.

    ``page`` is the request's ``Origin`` (``""`` when it had none). This is
    what a card that offers to allow an app would show.
    """

    host: str
    port: int
    page: str
    helper: str | None


@dataclass(frozen=True)
class Relayed:
    """A helper's answer, for Chrome: the status, its phrase, the headers and the body."""

    status: int
    reason: str
    headers: list[tuple[str, str]]
    body: bytes


class RelayError(Exception):
    """The helper gave no usable answer; ``reason`` is how Chrome fails the request (``Network.ErrorReason``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_PASS = Verdict("pass")


def _melos_own_ports() -> Mapping[int, str]:
    return local_ports.own_ports(config.load(), running=True)


class LocalRule:
    """Which paused requests melo relays, blocks or lets go on (see the module docstring).

    *helpers* are the sign-in helpers (default :data:`SIGNIN_HELPERS`). *own*
    gives melo's own ports, read at each request to a helper; when it fails,
    no helper request is relayed.
    """

    def __init__(
        self,
        helpers: Iterable[SigninHelper] = SIGNIN_HELPERS,
        own: Callable[[], Mapping[int, str]] = _melos_own_ports,
    ) -> None:
        self.helpers = tuple(helpers)
        self._own = own

    def patterns(self) -> list[dict[str, str]]:
        """The requests Chrome pauses (``Fetch.enable``): every helper address, every local address written as such, and every navigation.

        A pattern may match more (``*`` matches any text, a query string
        too); :meth:`decide` lets those go on. Every navigation (a tab's or a
        frame's document) is paused, since Chrome does not check where a page
        sends its tab, and an address with a user part (``x@127.0.0.1``) or
        a name of one word matches no other pattern.
        """
        found: list[str] = []
        for helper in self.helpers:
            for port in helper.ports:
                shown = "" if port == _SCHEME_PORTS[helper.scheme] else f":{port}"
                found.append(f"{helper.scheme}://{helper.host}{shown}/*")
        for name in sorted(LOCAL_NAMES):
            for end in ("", "."):
                found += [f"*://{name}{end}/*", f"*://{name}{end}:*"]
        for suffix in LOCAL_SUFFIXES:
            for end in ("", "."):
                found += [f"*://*{suffix}{end}/*", f"*://*{suffix}{end}:*"]
        found += [f"*://{start}*" for start in _IPV4_STARTS]
        found.append("*://[*")  # an IPv6 address; the public ones go on
        paused = [
            {"urlPattern": pattern, "requestStage": "Request"} for pattern in found
        ]
        paused.append(
            {"urlPattern": "*", "resourceType": "Document", "requestStage": "Request"}
        )
        return paused

    def decide(self, url: str, origin: str) -> Verdict:
        """What to do with a request to *url* sent by a page of *origin* (``""`` when it had none)."""
        try:
            parts = urlsplit(url)
            scheme = parts.scheme.lower()
            host = parts.hostname or ""
            port = parts.port or _SCHEME_PORTS.get(scheme, 0)
        except ValueError:
            return _PASS
        if scheme not in _SCHEME_PORTS or not host:
            return _PASS
        for helper in self.helpers:
            if (scheme, host) == (helper.scheme, helper.host) and port in helper.ports:
                if origin in helper.pages and not self._is_own(port):
                    return Verdict("relay", host, port, helper)
                return Verdict("block", host, port, helper)
        # The address as written, without a user part, which a browser would drop.
        shown = f"[{host}]" if ":" in host else host
        try:
            check_url(f"{scheme}://{shown}:{port}/")
        except NotPublicError:
            return Verdict("block", host, port)
        except AddressError:
            return _PASS
        return _PASS

    def _is_own(self, port: int) -> bool:
        try:
            return port in self._own()
        except Exception:  # melo's own ports unknown: never relay
            return True


def told(attempts: Iterable[LocalAttempt]) -> list[str]:
    """The sentences for the step about *attempts*: one per port, in order."""
    ports = dict.fromkeys(attempt.port for attempt in attempts)
    return [BLOCKED.format(port=port) for port in ports]


def _body(request: Mapping[str, Any]) -> bytes:
    """The body of a paused request; :class:`RelayError` when it is too big or Chrome did not give it."""
    entries = request.get("postDataEntries")
    data = b""
    if isinstance(entries, list) and entries:
        chunks = []
        for entry in entries:
            raw = entry.get("bytes") if isinstance(entry, dict) else None
            if not isinstance(raw, str):
                raise RelayError("Failed")
            try:
                chunks.append(base64.b64decode(raw, validate=True))
            except (binascii.Error, ValueError):
                raise RelayError("Failed") from None
        data = b"".join(chunks)
    elif isinstance(request.get("postData"), str):
        data = str(request["postData"]).encode("utf-8")
    elif request.get("hasPostData"):
        raise RelayError("Failed")  # a body melo cannot see is never sent without it
    if len(data) > MAX_BODY_BYTES:
        raise RelayError("Failed")
    return data


async def relay(
    helper: SigninHelper,
    port: int,
    request: Mapping[str, Any],
    *,
    verify: ssl.SSLContext | bool = True,
) -> Relayed:
    """Send the paused *request* to *helper* at *port* on its pinned address; its answer.

    The name is never looked up: melo connects to ``helper.address``, sends
    the helper's name in ``Host`` and, for https, checks the certificate for
    that name (*verify*: the trusted authorities). A redirect is handed back,
    not followed. Connecting may take :data:`CONNECT_S`; the answer
    :data:`RELAY_S`, or the helper's ``wait_s`` on a path it ``waits`` on. :class:`RelayError` when there is no usable answer.
    """
    method = str(request.get("method") or "GET").upper()
    if method not in METHODS:
        raise RelayError("Failed")
    parts = urlsplit(str(request.get("url") or ""))
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    body = _body(request)
    headers: list[tuple[str, str]] = []
    sent = request.get("headers")
    if isinstance(sent, dict):
        for name, value in sent.items():
            named = isinstance(name, str) and isinstance(value, str)
            if named and name.lower() not in _NOT_SENT:
                headers.append((name, value))
    shown = "" if port == _SCHEME_PORTS[helper.scheme] else f":{port}"
    headers += [("Host", f"{helper.host}{shown}"), ("Accept-Encoding", "identity")]
    address = f"[{helper.address}]" if ":" in helper.address else helper.address
    pinned = f"{helper.scheme}://{address}:{port}{path}"
    extensions = {"sni_hostname": helper.host} if helper.scheme == "https" else {}
    waits = any((parts.path or "/").startswith(waited) for waited in helper.waits)
    limit = helper.wait_s if waits else RELAY_S
    try:
        async with asyncio.timeout(CONNECT_S + limit):
            async with httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                verify=verify,
                timeout=httpx.Timeout(limit, connect=CONNECT_S),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            ) as client:
                built = client.build_request(
                    method, pinned, headers=headers, content=body, extensions=extensions
                )
                answer = await client.send(built, stream=True)
                try:
                    data = bytearray()
                    async for chunk in answer.aiter_bytes():  # gzip and deflate undone
                        data += chunk
                        if len(data) > MAX_ANSWER_BYTES:
                            raise RelayError("Failed")
                finally:
                    await answer.aclose()
    except (TimeoutError, httpx.TimeoutException):
        raise RelayError("TimedOut") from None
    except httpx.ConnectError as exc:
        raise RelayError("ConnectionRefused" if _refused(exc) else "Failed") from None
    except httpx.HTTPError:
        raise RelayError("Failed") from None
    if not 200 <= answer.status_code <= 599:
        raise RelayError("Failed")
    returned = [
        (name, value)
        for name, value in answer.headers.multi_items()
        if name.lower() in _RETURNED or name.lower().startswith("access-control-")
    ]
    return Relayed(answer.status_code, answer.reason_phrase, returned, bytes(data))


def _refused(exc: BaseException) -> bool:
    """Whether nothing listened (the app is closed), as opposed to a TLS or other failure."""
    seen: BaseException | None = exc
    for _ in range(8):
        if seen is None:
            return False
        if isinstance(seen, ConnectionRefusedError):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def fulfilled(ident: str, answer: Relayed) -> dict[str, Any]:
    """The ``Fetch.fulfillRequest`` parameters that give Chrome *answer* for the request *ident*."""
    params: dict[str, Any] = {"requestId": ident, "responseCode": answer.status}
    if answer.reason:
        params["responsePhrase"] = answer.reason
    params["responseHeaders"] = [
        {"name": name, "value": value} for name, value in answer.headers
    ]
    params["body"] = base64.b64encode(answer.body).decode("ascii")
    return params


def origin_of(headers: object) -> str:
    """The ``Origin`` header of a paused request's *headers*, or ``""``."""
    if isinstance(headers, dict):
        for name, value in headers.items():
            if isinstance(name, str) and name.lower() == "origin":
                return value if isinstance(value, str) else ""
    return ""


__all__ = [
    "BLOCKED",
    "MAX_ANSWER_BYTES",
    "MAX_BODY_BYTES",
    "MAX_RELAYS",
    "SIGNIN_HELPERS",
    "LocalAttempt",
    "LocalRule",
    "RelayError",
    "Relayed",
    "SigninHelper",
    "Verdict",
    "fulfilled",
    "origin_of",
    "relay",
    "told",
]
