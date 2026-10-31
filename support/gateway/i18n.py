"""The console's words in English and Simplified Chinese.

One catalog per language, ``static/i18n/<language>.json``, which the page
(``static/i18n.mjs``), this module and the macOS app (its strings are built
from the ``app.*`` keys) all read. Keys are flat and grouped by screen
(``running.more``); a value is a string with ``{name}`` parameters, or, for a
count, ``{"one": ..., "other": ...}`` picked by the parameter ``n``. English is
the source language: a key a catalog lacks reads in English.

The language is chosen once, by :func:`pick_language`: the ``[console]
language`` setting when it names a language, else the first of the reader's
preferred languages that the console has (``zh``, ``zh-CN`` or ``zh-Hans`` is
Simplified Chinese; Traditional Chinese tags are skipped), else English.

The catalogs, :func:`catalog` and :func:`t_in` live in
:mod:`localmelo.support.words`, which needs no gateway extra; this module adds
the request's language (and, with it, :func:`t`).

The server answers a request in the language the console page asks for, with
the ``X-LocalMelo-Language`` header or its ``localmelo_language`` cookie;
every other client (the terminal, scripts) gets English. While a request is
handled, :func:`t` uses its language.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from http.cookies import CookieError, SimpleCookie
from typing import Any, Final

from starlette.types import ASGIApp, Receive, Scope, Send

from localmelo.support.words import (
    CATALOGS,
    LANGUAGES,
    Entry,
    catalog,
    format_entry,
    plural_form,
    said,
    said_in,
    t_in,
)

SETTINGS: Final = ("auto", *LANGUAGES)
COOKIE: Final = "localmelo_language"
HEADER: Final = b"x-localmelo-language"

_TRADITIONAL: Final = frozenset({"hant", "tw", "hk", "mo"})
_current: ContextVar[str] = ContextVar("localmelo_language", default="en")


def _language_of(tag: str) -> str | None:
    """The console language a BCP 47 *tag* asks for, or ``None``."""
    parts = tag.strip().replace("_", "-").lower().split("-")
    if parts[0] == "en":
        return "en"
    if parts[0] != "zh":
        return None
    if "hans" in parts[1:]:
        return "zh-Hans"
    return None if _TRADITIONAL & set(parts[1:]) else "zh-Hans"


def pick_language(setting: str, preferred: Iterable[str]) -> str:
    """The language to show: *setting* when it names one, else the first preferred one the console has."""
    if setting in LANGUAGES:
        return setting
    for tag in preferred:
        found = _language_of(str(tag))
        if found is not None:
            return found
    return "en"


def accept_languages(header: str) -> list[str]:
    """The tags of an ``Accept-Language`` header, most wanted first."""
    ranked: list[tuple[float, int, str]] = []
    for index, part in enumerate(header.split(",")):
        tag, _, rest = part.strip().partition(";")
        quality = 1.0
        if rest.strip().startswith("q="):
            try:
                quality = float(rest.strip()[2:])
            except ValueError:
                quality = 0.0
        if tag and tag != "*" and quality > 0:
            ranked.append((-quality, index, tag))
    return [tag for _, _, tag in sorted(ranked)]


def t(key: str, /, **params: Any) -> str:
    """The words of *key* in the language of the request being handled."""
    return t_in(_current.get(), key, **params)


def current() -> str:
    """The language of the request being handled: ``en`` outside one."""
    return _current.get()


def said_text(words: Mapping[str, Any], language: str | None = None) -> str:
    """The words of a :func:`said` in *language* (the request's when omitted)."""
    return said_in(language or _current.get(), words)


@contextmanager
def using(language: str) -> Iterator[None]:
    """Write in *language* inside the block, as a stored record that must read the same to every reader does."""
    token = _current.set(language if language in LANGUAGES else "en")
    try:
        yield
    finally:
        _current.reset(token)


def _cookie(headers: Iterable[tuple[bytes, bytes]]) -> str:
    for name, value in headers:
        if name.lower() != b"cookie":
            continue
        jar: SimpleCookie = SimpleCookie()
        try:
            jar.load(value.decode("latin-1"))
        except CookieError:
            continue
        if COOKIE in jar:
            return jar[COOKIE].value
    return ""


def asked_language(headers: Iterable[tuple[bytes, bytes]]) -> str:
    """The language a request asks for, by header or cookie; ``""`` for any other client."""
    pairs = list(headers)
    asked = next(
        (value.decode("latin-1") for name, value in pairs if name.lower() == HEADER),
        "",
    )
    asked = asked.strip() or _cookie(pairs)
    return asked if asked in LANGUAGES else ""


def request_language(headers: Iterable[tuple[bytes, bytes]]) -> str:
    """What the console page asked for, by header or cookie; English for any other client."""
    return asked_language(headers) or "en"


def page_language(setting: str, headers: Iterable[tuple[bytes, bytes]]) -> str:
    """The first guess for the console page: the setting, the page's cookie, then ``Accept-Language``.

    The page checks it with the languages the browser (or the macOS app)
    prefers and switches when they differ.
    """
    pairs = list(headers)
    if setting in LANGUAGES:
        return setting
    asked = _cookie(pairs)
    if asked in LANGUAGES:
        return asked
    accept = ",".join(
        value.decode("latin-1")
        for name, value in pairs
        if name.lower() == b"accept-language"
    )
    return pick_language("auto", accept_languages(accept))


class LanguageMiddleware:
    """Sets the language of each HTTP request for :func:`t`.

    *remember*, when given, hears each language a request asks for: the console
    keeps the last one, to write in it what nobody asked for (a check-in).
    """

    def __init__(
        self, app: ASGIApp, remember: Callable[[str], None] | None = None
    ) -> None:
        self.app = app
        self.remember = remember

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", ())
        asked = asked_language(headers)
        if asked and self.remember is not None:
            self.remember(asked)
        token = _current.set(asked or "en")
        try:
            await self.app(scope, receive, send)
        finally:
            _current.reset(token)


__all__ = [
    "CATALOGS",
    "COOKIE",
    "LANGUAGES",
    "SETTINGS",
    "LanguageMiddleware",
    "Entry",
    "accept_languages",
    "asked_language",
    "catalog",
    "current",
    "format_entry",
    "page_language",
    "pick_language",
    "plural_form",
    "request_language",
    "said",
    "said_text",
    "t",
    "t_in",
    "using",
]
