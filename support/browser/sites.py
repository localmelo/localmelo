"""Which sites melo may open: http and https pages, each site asked once per chat.

Each question names an origin (scheme, host and a port that is not the
scheme's own). Before melo loads a page, and again where a redirect lands, the
origin is put to the run's approval policy as a request named
``browser:<origin>``. The policy keeps "allow for this chat" as a grant of that
name, so a site asks once per chat, ``/new`` and ``/approvals forget`` take the
grants back, and the console shows the question as its approval card. There is
no grant that outlives the chat. Without a policy (``--yolo``, a library caller)
nobody is asked.

A site is the hosts of one registrable domain under one scheme (#577,
:func:`~localmelo.melo.contracts.review.host_site`): what the person keeps for
``browser:`` and ``browser-act:`` covers them all, and a refusal does too. What
cannot be read that way (an address by number, a local name, a public suffix,
an unlisted suffix, a port that is not the scheme's own) stays its exact origin.
Every other answer stays on the exact origin it was asked about: an Allow once,
typing, a form, a file, a click at a point, a page's box and a take-over. A page
approved earlier is read again unasked until its site is refused: the refusal
takes the tabs' approval back (:class:`SiteRefusedError`), and only the person's
newer answer stands over it (:func:`require_not_refused`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Literal
from urllib.parse import urlsplit

from localmelo.melo.contracts.approvals import (
    TOOL_ASK,
    WEB_PORTS,
    ApprovalDecision,
    denial_text,
    is_refusal,
)
from localmelo.support.browser.apps import scheme_of
from localmelo.support.browser.driver import BrowserError

PREFIX = "browser:"
# Acting on a page asks apart from reading it: a click or a choice on a site is
# one grant per site and chat, so the answer to "open this site" never lets melo
# click. Typing is a question of its own that shows all the text and is asked
# every time, and so is sending a form: neither is ever kept for the chat.
ACT_PREFIX = "browser-act:"
TYPE_PREFIX = "browser-type:"
SUBMIT_PREFIX = "browser-submit:"
POINT_PREFIX = "browser-point:"
# Attaching a file to a page's upload field is a send of that file to the site:
# asked every time, never kept (``browser-attach:<site>``).
ATTACH_PREFIX = "browser-attach:"
# A page asks to open an address of another app's kind (``chat://...``):
# ``handoff:<scheme>``, asked per chat. A ``confirm`` box a page puts up while
# melo acts: ``browser-dialog:<site>``, asked every time.
HANDOFF_PREFIX = "handoff:"
DIALOG_PREFIX = "browser-dialog:"
# melo needs the person on a page (a sign-in, a code, a QR code, a human check):
# ``takeover:<site>``, a card that offers to take the page over in the Browser
# panel. Asked when needed, never kept for the chat, only the person answers it.
TAKEOVER_PREFIX = "takeover:"
# Every ``browser-`` question, whenever a new one is added, and the two others.
QUESTIONS = (PREFIX, "browser-", HANDOFF_PREFIX, TAKEOVER_PREFIX)
TAKEOVER_NEEDS = {
    "sign-in": "a sign-in",
    "password": "a password field",
    "code": "a one-time code",
    "card": "a payment card field",
    "qr": "a QR code to scan",
    "captcha": "a check that you are human",
}
# The browser melo browses in for the running call, as questions name it:
# Safari's automation windows unless the call's provider says otherwise.
BROWSER: ContextVar[str] = ContextVar("localmelo_browser_name", default="Safari")
# The longest address a site question shows (web_fetch's): longer ones are cut.
URL_SHOWN = 300
# The most form fields a send question lists.
FORM_SHOWN = 12


class SiteRefusedError(BrowserError):
    """A question about a site was refused: nobody's silence or a withdrawn card, a no (#577)."""


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
    if scheme not in WEB_PORTS or not host or not re.fullmatch(r"[\w.\-\[\]:]+", host):
        return None
    shown = f"[{host}]" if ":" in host else host
    return f"{scheme}://{shown}" + (
        f":{port}" if port not in (None, WEB_PORTS[scheme]) else ""
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


def _in() -> str:
    """Where melo acts, for a question's preview: ``in Safari``, ``in Chrome``."""
    return f"in {BROWSER.get()}"


def _browser(arguments: dict[str, object]) -> dict[str, object]:
    """*arguments* with the browser named, when it is not Safari (the surfaces word the card for it)."""
    name = BROWSER.get()
    if name != "Safari":
        arguments["browser"] = name.lower()
    return arguments


def shown_url(url: str) -> str:
    """*url* as a question shows it: at most :data:`URL_SHOWN` characters."""
    url = url.strip()
    return url if len(url) <= URL_SHOWN else url[: URL_SHOWN - 1] + "…"


def _question(
    origin: str, via: str, url: str | None
) -> tuple[str, dict[str, str], str]:
    """The site question about *origin*: its name, arguments and preview."""
    host = host_of(origin)
    arguments: dict[str, str] = {"site": host, "origin": origin, "access": "read"}
    preview = f"open {host} {_in()} and read it"
    if via == "web_fetch":
        arguments["via"] = "web_fetch"
        preview = f"read {host}"
        if url:
            arguments["url"] = shown_url(url)
            preview = f"read {arguments['url']}"
    else:
        name = BROWSER.get()
        if name != "Safari":
            arguments["browser"] = name.lower()
    return PREFIX + origin, arguments, preview


def _refusal(origin: str, via: str, decision: ApprovalDecision) -> str:
    verb = "read" if via == "web_fetch" else "open"
    said = f"melo did not {verb} {host_of(origin)}. " + denial_text(decision)
    return f"{said} {decision.notice}" if decision.notice else said


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


def refused_unasked(
    origin: str, *, via: str = "browser", url: str | None = None
) -> str | None:
    """As :func:`refused_earlier`, for what passes with no question: a site refused in this run too (#577).

    A refusal covers the whole site and every page approved before it. Where
    :func:`require_site` asks next, the person decides instead, so that their
    newest word stands.
    """
    check = getattr(TOOL_ASK.get(), "refused_unasked", None)
    if not callable(check):
        return None
    decision = check(*_question(origin, via, url))
    return None if decision is None else _refusal(origin, via, decision)


def require_not_refused(origin: str) -> None:
    """:class:`BrowserError` when a page on *origin* approved earlier stands refused now (:func:`refused_unasked`)."""
    refused = refused_unasked(origin)
    if refused is not None:
        raise BrowserError(refused)


async def require_site(
    origin: str, *, via: str = "browser", url: str | None = None
) -> None:
    """Ask about *origin*; :class:`BrowserError` with the sentence for the model if refused.

    A "no" is final for the call: the person's answer is not asked twice. It
    raises :class:`SiteRefusedError`, an answer nobody gave the plain error.
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
        error = SiteRefusedError if is_refusal(decision) else BrowserError
        raise error(_refusal(origin, via, decision))


async def require_act(origin: str, what: str, on: str = "") -> None:
    """Ask about clicking or choosing on *origin*'s page; :class:`BrowserError` if refused.

    The question has no ``origin`` argument on purpose: a refusal of a click
    must not refuse the site for reading too (the earlier-blocks rule keys on
    it). *what* is ``click`` or ``choose an option``; *on* is what it is done
    on, in words (``a button "Check in"``), so auto mode's review judges the
    step by its target (#465). One answer for the site covers the chat's later
    clicks and choices; typing asks apart (:func:`require_type`).
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return
    host = host_of(origin)
    arguments = {"site": host, "access": "act", "how": what}
    if on:
        arguments["on"] = on
    decision = await ask(
        ACT_PREFIX + origin,
        _browser(dict(arguments)),
        f"{what} {on} on {host} {_in()}" if on else f"{what} on {host} {_in()}",
    )
    if not decision.allow:
        raise BrowserError(
            f"melo did not click or choose on {host}. " + denial_text(decision)
        )


async def require_type(
    origin: str, text: str, on: str = "", search: bool = False
) -> None:
    """Ask about typing *text* on *origin*'s page; :class:`BrowserError` if refused.

    Asked for every text, whatever was allowed before, and never kept for the
    chat. The question carries all of the text, uncut (the caller keeps it to
    :data:`~localmelo.support.browser.driver.MAX_TYPED_CHARS` characters it can
    show): it is the model's own words, a person or a reviewer must see what
    goes into the page. *on* is the box in words (``a textbox "Comment"``),
    so the review sees where it goes (#465). Like a click's question it has
    no ``origin`` argument. *search*: the box is a search box
    (:func:`~localmelo.support.browser.fields.searches`), which the question
    says as ``field: "search"``: with the person's words asking to search, a
    query of any length is no message.
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return
    host = host_of(origin)
    arguments = {"site": host, "access": "type", "how": "type", "text": text}
    if on:
        arguments["on"] = on
    if search:
        arguments["field"] = "search"
    into = f" into {on}" if on else ""
    decision = await ask(
        TYPE_PREFIX + origin,
        _browser(dict(arguments)),
        f"type {len(text)} character(s){into} on {host} {_in()}",
    )
    if not decision.allow:
        raise BrowserError(f"melo did not type on {host}. " + denial_text(decision))


async def require_submit(
    origin: str,
    target: str,
    how: str,
    form: Sequence[tuple[str, str]] = (),
    search: bool = False,
) -> None:
    """Ask about sending a form from *origin*'s page to *target*; :class:`BrowserError` if refused.

    Reviewed by itself, every time: the question carries where the form goes
    and how (``click`` on its button, or ``enter`` in a field) and what it
    holds (*form*: each field's name and value, with nothing secret in it), and
    an answer is never kept for the chat. *target* is an absolute address,
    shown cut to :data:`URL_SHOWN` characters, or ``""`` when the page decides
    where the text goes (Enter in a box that is in no form). *search*: every
    text box of the form is a search box, which the question says as
    ``field: "search"`` (see :func:`require_type`).
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return
    host = host_of(origin)
    to = shown_url(target) if target else ""
    arguments: dict[str, object] = {
        "site": host,
        "access": "submit",
        "to": to,
        "how": how,
        "fields": [[name, value] for name, value in form][:FORM_SHOWN],
    }
    if search:
        arguments["field"] = "search"
    decision = await ask(
        SUBMIT_PREFIX + origin,
        arguments,
        (
            f"send a form on {host} to {to}"
            if to
            else f"send what is typed on {host} (the page decides where it goes)"
        ),
    )
    if not decision.allow:
        raise BrowserError(
            f"melo did not send the form from {host}. " + denial_text(decision)
        )


def attach_source(path: str) -> str | None:
    """How the run vouches for the file *path* (a canonical path) as a file to attach: ``folder`` or ``named``; None when it does not (#551).

    The run's question handle answers (``attachable``): the file lies in the
    chat's folder, or the person's own words name this file. Without a run that
    answers, nothing is vouched for, and the file is asked about as any other.
    """
    check = getattr(TOOL_ASK.get(), "attachable", None)
    found = check(path) if callable(check) else None
    return found if found in ("folder", "named") else None


async def require_attach(
    origin: str, name: str, folder: str, size: int, source: str
) -> None:
    """Ask about attaching the file *name* (*size* bytes, in *folder*, from *source*) to a field of *origin*'s page; :class:`BrowserError` if refused.

    Asked every time and never kept for the chat. The question carries the
    file's name, its folder as the person reads it and why melo may use it
    (``folder``, ``named``, ``work``, or ``other`` for a file nobody vouched
    for), and nothing of what is in it. Auto mode decides it by code alone
    (:func:`~localmelo.melo.agent._step_review.attach_review`): no card when the
    person's own words ask for the send of a file the run vouches for, a card
    naming the file and the site otherwise. Without a policy nobody is asked, as
    for a form, except about a file nobody vouched for: that one is refused,
    since nobody can say yes to it.
    """
    ask = TOOL_ASK.get()
    host = host_of(origin)
    if ask is None:
        if source == "other":
            raise BrowserError(
                f"melo sends {name} to {host} only when the user says yes to that "
                "file, and nobody can be asked here. Ask the user for the file's path."
            )
        return
    arguments = {
        "site": host,
        "access": "attach",
        "name": name,
        "folder": folder,
        "size": size,
        "source": source,
    }
    decision = await ask(
        ATTACH_PREFIX + origin,
        _browser(dict(arguments)),
        f"attach {name} from {folder} to a field on {host} {_in()}",
    )
    if not decision.allow:
        raise BrowserError(
            f"melo did not attach {name} to {host}. " + denial_text(decision)
        )


async def require_point(
    origin: str, what: str, how: str = "click", *, known: bool = True
) -> None:
    """Ask about a click (or typing, *how*) on *origin*'s page that lands on *what*; :class:`BrowserError` if refused.

    For a point that lands on a password, code or card field, on a control that
    agrees to terms, consent or a sign-in with another account, or inside a box
    from another site whose target melo cannot tell (*known* false: the
    question says ``target: unknown``, and auto mode asks the person, #465).
    Asked every time and never kept for the chat: the person sees what it is
    on. A box melo knows by a name (its title, a control listed in it) is
    clicked on the click's own question instead (:func:`require_act`).
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return
    host = host_of(origin)
    arguments = {"site": host, "access": "point", "how": how, "on": what}
    if not known:
        arguments["target"] = "unknown"
    decision = await ask(
        POINT_PREFIX + origin,
        _browser(dict(arguments)),
        f"{how} {what} on {host} {_in()}",
    )
    if not decision.allow:
        raise BrowserError(
            f"melo did not {how} there on {host}. " + denial_text(decision)
        )


def is_browser_question(tool_name: str) -> bool:
    """Whether *tool_name* is one of these questions: its call can hold typed words and a page's own."""
    return tool_name.startswith(QUESTIONS)


async def require_handoff(origin: str, url: str, app: str) -> bool:
    """Ask whether melo may open *url*, an address of another app's kind a page on *origin* asked for; :class:`BrowserError` if refused.

    *app* is the app this Mac opens it with. The question is
    ``handoff:<scheme>``: the person's "allow for this chat" is a grant of that
    kind of address, held where the person's grants hold (in the main chat,
    every chat) until it is revoked; in auto mode the step is reviewed high
    and asked on the card every time. It has no ``origin`` argument,
    so a refusal does not refuse the site for reading. Returns whether
    anybody was asked (False without a policy).
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return False
    scheme = scheme_of(url)[:30]
    host = host_of(origin) if origin else "a page"
    decision = await ask(
        HANDOFF_PREFIX + scheme,
        _browser(
            {
                "site": host,
                "access": "handoff",
                "app": app,
                "scheme": scheme,
                "url": shown_url(url),
            }
        ),
        f"open {app} for a page on {host} ({scheme}: address)",
    )
    if not decision.allow:
        raise BrowserError(f"melo did not open {app}. " + denial_text(decision))
    return True


async def require_dialog(origin: str, message: str) -> None:
    """Ask whether melo may answer OK to a ``confirm`` box a page on *origin* put up while it acted; :class:`BrowserError` if refused.

    *message* is the page's own words in the box (cut short): the person
    sees what they agree to. Asked every time and never kept for the chat;
    without a policy nobody is asked and the answer is OK, as the act was.
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return
    host = host_of(origin) if origin else "a page"
    decision = await ask(
        DIALOG_PREFIX + (origin or "page"),
        _browser(
            {"site": host, "access": "dialog", "how": "confirm", "text": message[:300]}
        ),
        f"answer OK to a question the page on {host} asks {_in()}",
    )
    if not decision.allow:
        raise BrowserError(f"melo answered Cancel on {host}. " + denial_text(decision))


async def require_takeover(
    origin: str, need: str, tab: str, *, form: bool = False
) -> Literal["yes", "no", "words"]:
    """Ask the person to take over the page on *origin*: "yes", "no", or "words" when their message ended the card.

    *need* says what melo cannot do there itself (a key of :data:`TAKEOVER_NEEDS`)
    and *tab* is the chat's tab; *form* says the page has a form the person began,
    which a real window would clear (the card offers the app first). The question
    is ``takeover:<origin>``: its yes is "Take over", which hands them the tab;
    nothing of it is kept for the chat. Without a policy nobody is asked ("no").
    A message ended the card is not a "no" to be remembered: melo may ask again.
    """
    ask = TOOL_ASK.get()
    if ask is None:
        return "no"
    host = host_of(origin) if origin else "a page"
    what = TAKEOVER_NEEDS.get(need, TAKEOVER_NEEDS["sign-in"])
    asked: dict[str, object] = {"site": host, "access": "takeover", "need": need}
    asked["tab"] = tab
    if form:
        asked["form"] = True
    decision = await ask(
        TAKEOVER_PREFIX + (origin or "page"),
        _browser(asked),
        f"take over {host}: melo needs you for {what}",
    )
    if decision.allow:
        return "yes"
    return "words" if decision.origin == "words" else "no"
