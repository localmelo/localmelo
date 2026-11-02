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

from localmelo.melo.contracts.approvals import (
    TOOL_ASK,
    ApprovalDecision,
    denial_text,
)
from localmelo.support.browser.driver import BrowserError

PREFIX = "browser:"
# The longest address a site question shows (web_fetch's): longer ones are cut.
URL_SHOWN = 300
_PORTS = {"http": 80, "https": 443}


def origin_of(url: str) -> str | None:
    """``scheme://host[:port]`` of an http(s) *url*; None for anything else.

    The host as web_fetch reads it: in ASCII (IDNA), lower case and without
    one final dot, so one site has one origin however its address is spelt.
    """
    try:
        parts = urlsplit(url.strip())
        scheme, host, port = parts.scheme.lower(), parts.hostname or "", parts.port
        if not host.isascii():
            host = host.encode("idna").decode("ascii")
    except (ValueError, UnicodeError):
        return None
    host = host.lower()
    host = host[:-1] if host.endswith(".") else host
    # No % in a host, as web_fetch's check: a browser decodes `coll%65ct.example`
    # to `collect.example`, so an encoded host would dodge a site refused earlier.
    if scheme not in _PORTS or not host or not re.fullmatch(r"[\w.\-\[\]:]+", host):
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
    try:
        scheme = urlsplit(url.strip()).scheme.lower()
    except ValueError:  # `＠` that normalizes to `@`, an unclosed or odd [ ] host
        raise BrowserError("That is not a valid http or https address.") from None
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


def _question(
    origin: str, via: str, url: str | None
) -> tuple[str, dict[str, str], str]:
    """The site question about *origin*: its name, arguments and preview."""
    host = host_of(origin)
    arguments = {"site": host, "origin": origin, "access": "read"}
    preview = f"open {host} in Safari and read it"
    if via == "web_fetch":
        arguments["via"] = "web_fetch"
        preview = f"read {host}"
        if url:
            arguments["url"] = shown_url(url)
            preview = f"read {arguments['url']}"
    return PREFIX + origin, arguments, preview


def _refusal(origin: str, via: str, decision: ApprovalDecision) -> str:
    verb = "read" if via == "web_fetch" else "open"
    return f"melo did not {verb} {host_of(origin)}. " + denial_text(decision)


def refused_earlier(
    origin: str, *, via: str = "browser", url: str | None = None
) -> str | None:
    """Why *origin* is refused at once, a site refused earlier in this task (#431), or None.

    The sentence for the model, as :func:`require_site` raises it. A tool
    that would read a site without asking (one the user's words name)
    checks this first: a block stands over a site the request names.
    """
    check = getattr(TOOL_ASK.get(), "refused_earlier", None)
    if not callable(check):
        return None
    decision = check(*_question(origin, via, url))
    return None if decision is None else _refusal(origin, via, decision)


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
    decision = await ask(*_question(origin, via, url))
    if not decision.allow:
        raise BrowserError(_refusal(origin, via, decision))
