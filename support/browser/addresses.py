"""Which addresses melo's browser opens: public ones, never this computer, its network or melo's console.

The checks are ``web_fetch``'s (:mod:`localmelo.support.web.guard`), run on
the host and port of an address before its site is asked about and before
the page loads, and again where a page lands:

1. The host as written: ``localhost`` and names under a local suffix, a
   name with no dot, a number in any form but the plain dotted one
   (``2130706433``, ``0x7f.1``, ``127.1``), and loopback, private, shared
   carrier, link-local, unspecified and other non-public addresses, IPv4
   ones written as IPv6 (``[::ffff:127.0.0.1]``) included.
2. Every address the name resolves to: one that is not public, or that is
   one of this computer's own or on its network, refuses the name. A name
   that is not found is not opened either: what it would reach cannot be
   checked.
3. melo's own console, where the saved ``[gateway]`` host and port and the
   running gateway (:mod:`localmelo.support.gateway_endpoint`) say it
   listens, even at a public address.

An address with a user part (``user@host``) is not opened: browsers and
Python split ``a\\@b`` differently, so the host checked could be another
than the host loaded.

The browser looks a name up again when it loads the page, so an answer that
changes in between is not seen here; the console refuses a request whose
``Host`` is not its own (see :mod:`localmelo.support.gateway.guard`).
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from localmelo.support import config, gateway_endpoint
from localmelo.support.browser.driver import BrowserError
from localmelo.support.web.guard import (
    AddressError,
    NotPublicError,
    check_url,
    checked_addresses,
    system_resolver,
)

REFUSED = "melo's browser does not open local or private addresses"
HINT = "For a task's result, use task_get."
CONSOLE = "it is melo's own console"


def refusal(host: str, what: str) -> str:
    """The sentence for the model when *host* is refused because of *what* it is."""
    shown = f"[{host}]" if ":" in host else host
    return f"{REFUSED}, so it did not open {shown}: {what}. {HINT}"


def _site(host: str, port: object) -> tuple[str, int] | None:
    """*host* and *port* as :func:`check_url` writes a target's; None when unusable."""
    try:
        number = int(str(port))
    except ValueError:
        return None
    text = str(host).strip().lower()
    text = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    text = text[:-1] if text.endswith(".") else text
    if not text or not 0 < number < 65536:
        return None
    try:
        return str(ipaddress.ip_address(text)), number
    except ValueError:
        pass
    try:
        return text.encode("idna").decode("ascii"), number
    except UnicodeError:
        return text, number


def console_sites() -> set[tuple[str, int]]:
    """Where melo's console listens: the saved ``[gateway]`` host and port, and the running gateway's."""
    found: set[tuple[str, int] | None] = set()
    try:
        gateway = config.load().gateway
        found.add(_site(gateway.host, gateway.port))
    except Exception:  # an unreadable config still leaves the other checks
        pass
    running = gateway_endpoint.read()
    if running is not None:
        found.add(_site(running.host, running.port))
    return {site for site in found if site is not None}


async def require_public(url: str) -> None:
    """Return when *url* may be opened; :class:`BrowserError` with the sentence for the model if not.

    See the module docstring. Only the scheme, host and port are read: the
    path and query are the site's own business.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        raise BrowserError("That is not an http or https address.") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise BrowserError("That is not an http or https address.")
    if "@" in parts.netloc:
        raise BrowserError(
            "melo's browser does not open an address with a user name or password in it."
        )
    try:
        target = check_url(f"{parts.scheme}://{parts.netloc}/")
        console = console_sites()
        if (target.host, target.port) in console:
            raise NotPublicError(target.host, CONSOLE)
        found = await checked_addresses(target, system_resolver)
        if any((str(ip), port) in console for ip, port in found):
            raise NotPublicError(target.host, CONSOLE)
    except NotPublicError as exc:
        raise BrowserError(refusal(exc.host, exc.what)) from None
    except AddressError as exc:
        raise BrowserError(f"melo did not open that address. {exc}") from None


__all__ = ["CONSOLE", "HINT", "REFUSED", "console_sites", "refusal", "require_public"]
