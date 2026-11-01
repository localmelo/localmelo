"""Which sites melo may open: http and https pages, each site asked once per chat.

A site is an origin (scheme, host and a port that is not the scheme's own).
Before melo loads a page, and again where a redirect lands, the site is put to
the run's approval policy as a request named ``browser:<origin>``. The policy
keeps "allow for this chat" as a grant of that name, so a site asks once per
chat, ``/new`` and ``/approvals forget`` take the grants back, and the console
shows the question as its approval card. There is no grant that outlives the
chat. Without a policy (``--yolo``, a library caller) nobody is asked.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from localmelo.melo.contracts.approvals import TOOL_ASK, denial_text
from localmelo.support.browser.driver import BrowserError

PREFIX = "browser:"
# The longest address a site question shows (web_fetch's): longer ones are cut.
URL_SHOWN = 300
_PORTS = {"http": 80, "https": 443}


def origin_of(url: str) -> str | None:
    """``scheme://host[:port]`` of an http(s) *url*; None for anything else."""
    try:
        parts = urlsplit(url.strip())
        scheme, host, port = parts.scheme.lower(), parts.hostname, parts.port
    except ValueError:
        return None
    if scheme not in _PORTS or not host or not re.fullmatch(r"[\w.\-\[\]:%]+", host):
        return None
    shown = f"[{host}]" if ":" in host else host
    return f"{scheme}://{shown}" + (
        f":{port}" if port not in (None, _PORTS[scheme]) else ""
    )


def require_origin(url: str) -> str:
    """The origin of *url*; :class:`BrowserError` when melo does not open it at all."""
    origin = origin_of(url)
    if origin is not None:
        return origin
    scheme = urlsplit(url.strip()).scheme.lower()
    if scheme in ("file", "about", "data", "javascript", "blob") or "-" in scheme:
        raise BrowserError(
            f"melo does not open {scheme}: pages (the browser's own and local ones)."
        )
    raise BrowserError("That is not an http or https address.")


def host_of(origin: str) -> str:
    return origin.split("://", 1)[-1]


def shown_url(url: str) -> str:
    """*url* as a question shows it: at most :data:`URL_SHOWN` characters."""
    url = url.strip()
    return url if len(url) <= URL_SHOWN else url[: URL_SHOWN - 1] + "…"


async def require_site(
    origin: str, *, via: str = "browser", url: str | None = None
) -> None:
    """Ask about *origin*; :class:`BrowserError` with the sentence for the model if refused.

    A "no" is final for the call: the person's answer is not asked twice.
    *via* ``"web_fetch"`` asks the same question (the same grant name, so one
    "allow for this chat" covers both tools) worded "read <url>", with
    ``via`` and the address it requests (*url*, cut to :data:`URL_SHOWN`
    characters) in its arguments, so a surface can show the address and say
    no browser opens, and auto mode's review reads the query too.
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return
    host = host_of(origin)
    reading = via == "web_fetch"
    arguments = {"site": host, "origin": origin, "access": "read"}
    preview = f"open {host} in Safari and read it"
    if reading:
        arguments["via"] = "web_fetch"
        preview = f"read {host}"
        if url:
            arguments["url"] = shown_url(url)
            preview = f"read {arguments['url']}"
    decision = await ask(PREFIX + origin, arguments, preview)
    if not decision.allow:
        verb = "read" if reading else "open"
        raise BrowserError(f"melo did not {verb} {host}. " + denial_text(decision))
