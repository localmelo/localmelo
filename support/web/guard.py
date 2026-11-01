"""Which addresses ``web_fetch`` may reach: public web servers, never this computer or its network.

Two checks, both before any connection is made:

1. :func:`check_url` reads the address itself. Only ``http`` and ``https``,
   no user name or password in it, no control characters, spaces or
   backslashes, at most :data:`MAX_URL_CHARS` characters. The host is put
   in its ASCII form first (so full-width digits and dots cannot hide a
   number), then refused when it is ``localhost``, a name under a local
   suffix (:data:`LOCAL_SUFFIXES`), a name with no dot (a local network
   name), or a number written in any form but the plain four-part one
   (``2130706433``, ``0x7f.1``, ``0177.0.0.1``, ``127.1``).
2. :func:`address_problem` reads each address the host resolves to (and an
   address written in the URL): every one must be public. Loopback,
   private (RFC 1918), shared carrier (100.64.0.0/10), link-local
   (169.254.0.0/16, where cloud metadata lives), multicast, reserved and
   documentation ranges are refused, and so are IPv6 unique-local and
   link-local addresses and any IPv6 address that carries an IPv4 one
   (mapped, compatible, NAT64, 6to4, Teredo).

The fetcher (:mod:`localmelo.support.web.fetch`) resolves the name itself,
checks every answer, and connects to an address it checked, so a DNS answer
that changes between the check and the connection is never used. It checks
again on every redirect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
# (host, port) -> the (address, port) pairs the host resolves to.
Resolver = Callable[[str, int], Awaitable[list[tuple[str, int]]]]
# (host, address) -> True to let a non-public address through. Tests and the
# screenshot tool pass one to reach a local page server; nothing else does.
AllowHook = Callable[[str, IPAddress], bool]

MAX_URL_CHARS = 2048
SCHEMES = {"http": 80, "https": 443}
# Names that only ever mean this computer or the local network.
LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain"})
LOCAL_SUFFIXES = (
    ".localhost",
    ".local",
    ".localdomain",
    ".internal",
    ".home.arpa",
    ".home",
    ".lan",
    ".corp",
)
_HOST_CHARS = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*")
_NUMERIC_LABEL = re.compile(r"[0-9]+|0x[0-9a-f]*")

_V4 = ipaddress.IPv4Network
_V6 = ipaddress.IPv6Network
# Each refused range, with what it is in the words the model reads.
BLOCKED: tuple[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, str], ...] = (
    (_V4("0.0.0.0/8"), "an unspecified address"),
    (_V4("127.0.0.0/8"), "this computer (loopback)"),
    (_V4("10.0.0.0/8"), "a private network address"),
    (_V4("172.16.0.0/12"), "a private network address"),
    (_V4("192.168.0.0/16"), "a private network address"),
    (_V4("100.64.0.0/10"), "a shared carrier-network address"),
    (_V4("169.254.0.0/16"), "a link-local or cloud metadata address"),
    (_V4("192.0.0.0/24"), "a special-purpose address"),
    (_V4("192.0.2.0/24"), "a documentation address"),
    (_V4("192.88.99.0/24"), "a special-purpose address"),
    (_V4("198.18.0.0/15"), "a benchmarking address"),
    (_V4("198.51.100.0/24"), "a documentation address"),
    (_V4("203.0.113.0/24"), "a documentation address"),
    (_V4("224.0.0.0/4"), "a multicast address"),
    (_V4("240.0.0.0/4"), "a reserved address"),
    (_V6("::/128"), "an unspecified address"),
    (_V6("::1/128"), "this computer (loopback)"),
    (_V6("::ffff:0:0/96"), "an IPv4 address written as IPv6"),
    (_V6("::/96"), "an IPv4 address written as IPv6"),
    (_V6("64:ff9b::/96"), "an IPv4 address written as IPv6"),
    (_V6("64:ff9b:1::/48"), "an IPv4 address written as IPv6"),
    (_V6("100::/64"), "a special-purpose address"),
    (_V6("2001::/23"), "a special-purpose address"),
    (_V6("2001:db8::/32"), "a documentation address"),
    (_V6("2002::/16"), "an IPv4 address written as IPv6"),
    (_V6("3fff::/20"), "a documentation address"),
    (_V6("5f00::/16"), "a special-purpose address"),
    (_V6("fc00::/7"), "a unique-local (private) IPv6 address"),
    (_V6("fe80::/10"), "a link-local address"),
    (_V6("fec0::/10"), "a site-local address"),
    (_V6("ff00::/8"), "a multicast address"),
)


class AddressError(Exception):
    """An address ``web_fetch`` does not reach; the message is for the model."""


@dataclass(frozen=True)
class Target:
    """A checked address: its parts, and its host in ASCII without a final dot.

    ``ip`` is set when the URL names an address rather than a host name.
    ``url`` is the address as it is requested (no fragment).
    """

    url: str
    scheme: str
    host: str
    port: int
    ip: IPAddress | None

    @property
    def default_port(self) -> bool:
        return self.port == SCHEMES[self.scheme]

    @property
    def host_header(self) -> str:
        """The ``Host`` header: the host as written, with a port that is not the scheme's."""
        shown = f"[{self.host}]" if ":" in self.host else self.host
        return shown if self.default_port else f"{shown}:{self.port}"


def address_problem(ip: IPAddress) -> str | None:
    """What *ip* is when ``web_fetch`` must not reach it; ``None`` for a public address."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.scope_id:
        return "a link-local address"
    for network, words in BLOCKED:
        if ip.version == network.version and ip in network:
            return words
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    ):
        return "not a public address"
    return None


def _ascii_host(host: str) -> str:
    """*host* in ASCII (IDNA), lower case, without one final dot; :class:`AddressError` if it has none."""
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise AddressError("The address has a host name melo cannot read.") from None
    if ascii_host.endswith("."):
        ascii_host = ascii_host[:-1]
    if not ascii_host or ascii_host.endswith("."):
        raise AddressError("The address has no host name.")
    return ascii_host


def _name_problem(host: str) -> str | None:
    """Why the host name *host* (ASCII, no final dot) is refused before it is resolved."""
    if host in LOCAL_NAMES or host.endswith(LOCAL_SUFFIXES):
        return "it is a name for this computer or the local network"
    if len(host) > 253 or not _HOST_CHARS.fullmatch(host):
        return "it is not a valid host name"
    if any(len(label) > 63 for label in host.split(".")):
        return "it is not a valid host name"
    if "." not in host:
        return "a name without a dot can only be a name on the local network"
    return None


def _not_public(host: str, what: str) -> AddressError:
    return AddressError(
        f"melo does not read {host}: {what}. web_fetch reads public web pages only, "
        "never this computer or its local network."
    )


def _split_netloc(netloc: str) -> tuple[str, str | None]:
    """The host and the port text of *netloc* (no user info), as written."""
    if netloc.startswith("["):
        end = netloc.find("]")
        if end == -1:
            raise AddressError("The address has an unclosed [ in its host.")
        host, rest = netloc[1:end], netloc[end + 1 :]
        if rest and not rest.startswith(":"):
            raise AddressError("The address has text after its IPv6 host.")
        return host, rest[1:] if rest else None
    host, colon, port = netloc.rpartition(":")
    return (host, port) if colon else (netloc, None)


def check_url(url: str) -> Target:
    """The checked :class:`Target` of *url*; :class:`AddressError` with why it is not read."""
    text = url.strip() if isinstance(url, str) else ""
    if not text:
        raise AddressError("Give an http or https address.")
    if len(text) > MAX_URL_CHARS:
        raise AddressError(f"The address is longer than {MAX_URL_CHARS} characters.")
    if any(ord(char) <= 0x20 or 0x7F <= ord(char) <= 0x9F for char in text):
        raise AddressError("The address has spaces or control characters in it.")
    if "\\" in text:
        raise AddressError("The address has a backslash in it.")
    try:
        parts = urlsplit(text)
    except ValueError:
        raise AddressError("That is not a valid address.") from None
    scheme = parts.scheme.lower()
    if scheme not in SCHEMES:
        shown = f", not {scheme}: ones" if scheme else ""
        raise AddressError(f"web_fetch reads http and https addresses only{shown}.")
    netloc = parts.netloc
    if not netloc:
        raise AddressError("The address has no host name.")
    if "@" in netloc:
        raise AddressError(
            "The address has a user name or password in it; web_fetch never logs in."
        )
    raw_host, port_text = _split_netloc(netloc)
    if port_text is None or port_text == "":
        port = SCHEMES[scheme]
    elif port_text.isascii() and port_text.isdigit() and 0 < int(port_text) < 65536:
        port = int(port_text)
    else:
        raise AddressError("The address has an invalid port.")
    if "%" in raw_host:
        raise AddressError("The address has a % in its host.")
    ip: IPAddress | None = None
    if ":" in raw_host:
        try:
            ip = ipaddress.IPv6Address(raw_host)
        except ValueError:
            raise AddressError("The address has an invalid IPv6 host.") from None
        host = str(ip)
    else:
        host = _ascii_host(raw_host)
        if _NUMERIC_LABEL.fullmatch(host.rsplit(".", 1)[-1]):
            # A host that ends in a number is an IPv4 address to a resolver,
            # in whatever form; only the plain dotted one is read.
            try:
                ip = ipaddress.IPv4Address(host)
            except ValueError:
                ip = None
            if ip is None or str(ip) != host:
                raise _not_public(
                    host,
                    "it is an address written in an unusual form (one number, "
                    "octal, hex or a short form)",
                )
        else:
            problem = _name_problem(host)
            if problem:
                raise _not_public(host, problem)
    if ip is not None:
        problem = address_problem(ip)
        if problem:
            raise _not_public(host, f"it is {problem}")
    shown = f"[{host}]" if ":" in host else host
    port_part = "" if port == SCHEMES[scheme] else f":{port}"
    path = parts.path or "/"
    query = f"?{parts.query}" if parts.query else ""
    return Target(f"{scheme}://{shown}{port_part}{path}{query}", scheme, host, port, ip)


async def system_resolver(host: str, port: int) -> list[tuple[str, int]]:
    """The addresses the system's resolver gives for *host*, with their ports."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )
    return [(str(info[4][0]), int(info[4][1])) for info in infos]


async def checked_addresses(
    target: Target,
    resolver: Resolver,
    *,
    allow: AllowHook | None = None,
    timeout_s: float = 5.0,
) -> list[tuple[IPAddress, int]]:
    """Every address *target* resolves to, each checked; :class:`AddressError` if any is not public.

    One refused answer refuses them all, so a name that answers with both a
    public and a private address is never read. *allow* is asked about a
    non-public address only (tests and screenshots); without it, nothing
    non-public passes.
    """
    if target.ip is not None:
        found = [(str(target.ip), target.port)]
    else:
        try:
            found = await asyncio.wait_for(
                resolver(target.host, target.port), timeout_s
            )
        except TimeoutError:
            raise AddressError(f"Looking up {target.host} took too long.") from None
        except (OSError, UnicodeError, ValueError):
            raise AddressError(f"{target.host} was not found.") from None
    checked: list[tuple[IPAddress, int]] = []
    for address, port in found:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise AddressError(
                f"{target.host} resolved to {address!r}, which is not an address."
            ) from None
        problem = address_problem(ip)
        if problem and not (allow is not None and allow(target.host, ip)):
            raise _not_public(target.host, f"it resolves to {ip}, {problem}")
        if (ip, port) not in checked:
            checked.append((ip, port))
    if not checked:
        raise AddressError(f"{target.host} was not found.")
    return checked


__all__ = [
    "BLOCKED",
    "LOCAL_SUFFIXES",
    "MAX_URL_CHARS",
    "AllowHook",
    "IPAddress",
    "AddressError",
    "Resolver",
    "Target",
    "address_problem",
    "check_url",
    "checked_addresses",
    "system_resolver",
]
