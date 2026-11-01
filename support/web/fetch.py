"""Download one public page: a GET with fixed limits, checked addresses and checked redirects.

:class:`WebFetcher` makes each request to an address that
:func:`~localmelo.support.web.guard.checked_addresses` approved: the URL it
sends to names that address, and the ``Host`` header and the TLS server name
carry the host, so the certificate is still checked for the host and a DNS
answer that changes after the check is never used. Each redirect is checked
the same way before it is followed, at most :data:`MAX_REDIRECTS` times.

Nothing of the person's goes with a request: a new client per request with
no cookies (cookies a server sets are dropped), no ``Authorization`` header,
no proxy or ``.netrc`` from the environment, and a fixed User-Agent that
names localmelo. The body is read up to :data:`MAX_BYTES` (after gzip or
deflate is undone, so a small compressed body cannot grow past it) and
within :data:`TOTAL_S` for the whole fetch.
"""

from __future__ import annotations

import asyncio
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from localmelo.support.web.guard import (
    AddressError,
    AllowHook,
    IPAddress,
    Resolver,
    Target,
    check_url,
    checked_addresses,
    system_resolver,
)

USER_AGENT = "localmelo-web-fetch/1.0 (+https://github.com/localmelo/localmelo)"
ACCEPT = (
    "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.9,"
    "text/*;q=0.8,*/*;q=0.1"
)
MAX_REDIRECTS = 5
MAX_BYTES = 2_000_000
CONNECT_S = 8.0
READ_S = 15.0
TOTAL_S = 30.0
DNS_S = 5.0
# Addresses of one host tried in turn when a connection fails.
MAX_ADDRESSES = 3
REDIRECTS = frozenset({301, 302, 303, 307, 308})


class FetchError(Exception):
    """A fetch that did not give a page; the message is for the model."""


@dataclass
class Response:
    """What a page answered: the final address, status, type and (bounded) body."""

    url: str
    status: int
    reason: str
    content_type: str
    charset: str
    body: bytes
    cut: bool = False
    redirects: list[str] = field(default_factory=list)
    location: str = ""


def _media_type(header: str) -> tuple[str, str]:
    """The media type (lower case) and the charset of a ``Content-Type`` header."""
    media, _, params = header.partition(";")
    charset = ""
    for param in params.split(";"):
        name, _, value = param.partition("=")
        if name.strip().lower() == "charset":
            charset = value.strip().strip("\"'").lower()
    return media.strip().lower(), charset


class _Decoder:
    """Undoes ``Content-Encoding`` (identity, gzip, deflate), never past *limit* bytes."""

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self._zlib: zlib._Decompress | None = None
        if encoding in ("gzip", "x-gzip"):
            self._zlib = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            self._zlib = zlib.decompressobj()
            self._raw_tried = False
        elif encoding not in ("", "identity"):
            raise FetchError(
                f"The server sent the page compressed as {encoding!r}, which melo does not read."
            )

    def feed(self, data: bytes, limit: int) -> tuple[bytes, bool]:
        """The decoded bytes of *data*, at most *limit*; True when more would follow."""
        if self._zlib is None:
            return data[:limit], len(data) > limit
        try:
            out = self._zlib.decompress(data, limit)
        except zlib.error:
            if self.encoding == "deflate" and not self._raw_tried:
                # Some servers send raw deflate without the zlib header.
                self._raw_tried = True
                self._zlib = zlib.decompressobj(-zlib.MAX_WBITS)
                return self.feed(data, limit)
            raise FetchError("The page's compressed data is damaged.") from None
        return out, bool(self._zlib.unconsumed_tail)


class WebFetcher:
    """GETs public pages under the limits above. See the module docstring.

    *resolver* and *allow* are for tests and screenshots, which reach a page
    server on this computer: *resolver* answers for a test host name, and
    *allow* lets that one host's local address through. The checks of the
    address, of every answer and of every redirect stay the same.
    """

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        allow: AllowHook | None = None,
        max_bytes: int = MAX_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        connect_s: float = CONNECT_S,
        read_s: float = READ_S,
        total_s: float = TOTAL_S,
        verify: bool = True,
    ) -> None:
        self._resolver = resolver or system_resolver
        self._allow = allow
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self._timeout = httpx.Timeout(read_s, connect=connect_s)
        self.total_s = total_s
        self._verify = verify

    async def fetch(
        self, url: str, accept: Callable[[str], bool] | None = None
    ) -> Response:
        """The page at *url* after redirects; :class:`AddressError` or :class:`FetchError` with why not.

        *accept*, when given, says which media types have their body read; a
        page of another type is returned with an empty body, so the caller can
        say what it was without downloading it. A page with no type is read.
        """
        try:
            async with asyncio.timeout(self.total_s):
                return await self._follow(url, accept)
        except TimeoutError:
            raise FetchError(
                f"The page did not finish loading within {self.total_s:g} seconds."
            ) from None

    async def _follow(self, url: str, accept: Callable[[str], bool] | None) -> Response:
        current = url
        redirects: list[str] = []
        while True:
            target = check_url(current)
            addresses = await checked_addresses(
                target, self._resolver, allow=self._allow, timeout_s=DNS_S
            )
            response = await self._get(target, addresses, accept)
            location = response.location if response.status in REDIRECTS else ""
            if not location:
                response.redirects = redirects
                return response
            if len(redirects) >= self.max_redirects:
                raise FetchError(
                    f"The page redirected more than {self.max_redirects} times; melo stopped."
                )
            redirects.append(target.url)
            current = urljoin(target.url, location)

    async def _get(
        self,
        target: Target,
        addresses: list[tuple[IPAddress, int]],
        accept: Callable[[str], bool] | None,
    ) -> Response:
        """One GET of *target* at the first of *addresses* that connects."""
        last: Exception | None = None
        for ip, port in addresses[:MAX_ADDRESSES]:
            try:
                return await self._get_at(target, ip, port, accept)
            except httpx.ConnectError as exc:
                last = exc
        raise FetchError(f"Could not connect to {target.host}: {last}.")

    async def _get_at(
        self,
        target: Target,
        ip: IPAddress,
        port: int,
        accept: Callable[[str], bool] | None,
    ) -> Response:
        pinned = httpx.URL(target.url).copy_with(host=str(ip), port=port)
        headers = {
            "Host": target.host_header,
            "User-Agent": USER_AGENT,
            "Accept": ACCEPT,
            "Accept-Encoding": "gzip, deflate",
            "Connection": "close",
        }
        # The certificate is checked for the host, not for the address.
        extensions = {} if target.ip is not None else {"sni_hostname": target.host}
        try:
            async with httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                timeout=self._timeout,
                verify=self._verify,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            ) as client:
                request = client.build_request(
                    "GET", pinned, headers=headers, extensions=extensions
                )
                response = await client.send(request, stream=True)
                try:
                    return await self._read(target, response, accept)
                finally:
                    await response.aclose()
        except httpx.ConnectError:
            raise
        except httpx.TimeoutException:
            raise FetchError(f"{target.host} did not answer in time.") from None
        except httpx.HTTPError as exc:
            raise FetchError(f"The request to {target.host} failed: {exc}.") from None

    async def _read(
        self,
        target: Target,
        response: httpx.Response,
        accept: Callable[[str], bool] | None,
    ) -> Response:
        status = response.status_code
        media, charset = _media_type(response.headers.get("content-type", ""))
        reason = response.reason_phrase or ""
        location = response.headers.get("location", "").strip()
        if status >= 300 or (accept is not None and media and not accept(media)):
            return Response(
                target.url, status, reason, media, charset, b"", location=location
            )
        decoder = _Decoder(response.headers.get("content-encoding", "").strip().lower())
        chunks: list[bytes] = []
        size = raw = 0
        cut = False
        async for chunk in response.aiter_raw():
            raw += len(chunk)
            # One byte past the limit says whether the page goes on.
            data, more = decoder.feed(chunk, self.max_bytes + 1 - size)
            chunks.append(data)
            size += len(data)
            if more or size > self.max_bytes or raw > 2 * self.max_bytes:
                cut = True
                break
        body = b"".join(chunks)[: self.max_bytes]
        return Response(target.url, status, reason, media, charset, body, cut)


__all__ = [
    "MAX_BYTES",
    "MAX_REDIRECTS",
    "USER_AGENT",
    "FetchError",
    "AddressError",
    "Response",
    "WebFetcher",
]
