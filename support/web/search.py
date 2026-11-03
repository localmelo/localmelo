"""Searching the public web with words: ordered keyless ways, one answer.

:class:`WebSearcher` tries the *ways* in :data:`DEFAULT_ORDER` (or the order it
is given) until one gives results that read as an answer to the query, and says
which way that was and why the others did not answer. A way is a result page
read from an engine by ``web_fetch``'s fetcher (a plain GET: no cookies, no
login, melo's own User-Agent, the same address checks and redirect checks), or
``browser``: the same kind of page loaded in the browser window the person
switched on (``[browser] driver``), where a plain GET is turned away. Which
ways come first was decided by measuring them (see ``scripts/search_probe.py``):
a plain GET is fast but an engine can answer a client it does not know with
results that parse well and are about something else, so every answer is also
checked against the query (:func:`relevant`) and an answer that fails is not
taken for one.

Nothing here asks the person: whether a search may go out is the tool's job
(:mod:`localmelo.support.web.search_tool`). Engines are asked politely: at most
one request per way per search, :data:`GAP_S` seconds apart for one engine, and
a way that was turned away is not asked again in the same search.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import quote

from localmelo.support.browser.driver import BrowserError
from localmelo.support.web import search_pages as pages
from localmelo.support.web.fetch import FetchError, WebFetcher
from localmelo.support.web.guard import AddressError, check_url
from localmelo.support.web.search_pages import Hit
from localmelo.support.web.tool import readable

if TYPE_CHECKING:
    from localmelo.support.browser.driver import BrowserDriver

# Seconds between two requests to one engine.
GAP_S = 2.0
# The most hits a search returns.
MAX_COUNT = 8
# A query is words; a longer one is text pasted from somewhere.
MAX_QUERY_CHARS = 200
# The browser way loads this page: DuckDuckGo's page without scripts.
BROWSER_URL = "https://html.duckduckgo.com/html/?q={q}"
BROWSER = "browser"
BROWSER_ENGINE = "DuckDuckGo in Safari"
# The longest the browser way waits for a page, in seconds.
BROWSER_S = 25.0
BROWSER_OWN = frozenset({"duckduckgo.com"})
# Statuses that turn a client away rather than say a page is missing.
_STATUS_BLOCKED = frozenset({202, 401, 403, 407, 429, 451})

BLOCK_MARKERS = (
    "captcha",
    "unusual traffic",
    "are you a robot",
    "bots use duckduckgo",
    "verify you are human",
    "verifying your request",
    "access denied",
    "just a moment",
    "bot activity",
    "验证码",
    "安全验证",
    "百度安全",
    "异常流量",
)


# Why a way did not answer (closed sentences: nothing a server wrote reaches the model).
BLOCKED = "asked for a check"
UNREAD = "gave a page melo could not read results from"
OFF_TOPIC = "gave results that did not match the words"


@dataclass(frozen=True)
class Way:
    """A result page melo can read with a plain GET."""

    key: str
    engine: str
    url: str  # with {q}
    read: Callable[[str], list[Hit]]


# Where a search service that needs a key goes later: one more entry here (its key
# name in `way_names()` is then valid in `[web] search_order`), a `read` for its
# answer, and its key taken from the keychain by the code that stores keys, never
# from this module or the config file. The searcher, the checks and the tool need
# no change.
WAYS: dict[str, Way] = {
    way.key: way
    for way in (
        Way(
            "bing-rss",
            "Bing",
            "https://www.bing.com/search?format=rss&q={q}",
            pages.parse_bing_rss,
        ),
        Way(
            "baidu",
            "Baidu",
            "https://www.baidu.com/s?wd={q}",
            pages.parse_baidu,
        ),
    )
}
# The ways in the order they are tried, as measured: Bing's feed first (a fifth of a
# second, and on topic often enough to try), then the browser window when it is
# switched on (the best answers, under a second), then Baidu.
DEFAULT_ORDER: tuple[str, ...] = ("bing-rss", BROWSER, "baidu")


def way_names() -> tuple[str, ...]:
    """Every name ``[web] search_order`` may hold."""
    return (*WAYS, BROWSER)


# ── is it an answer? ──

_STOP = frozenset(
    {"the", "and", "for", "with", "what", "how", "why", "who", "when", "where"}
    | {"does", "can", "you", "are", "was", "this", "that", "from", "into", "your"}
    | {"have", "has", "not", "but", "all", "any", "about"}
)
_LATIN = re.compile(r"[A-Za-z0-9_.+#'-]+")
_CJK = re.compile(r"[㐀-鿿]+")


def _tokens(query: str) -> list[str]:
    """What a result should share with *query*: Latin words and CJK character pairs."""
    found = [
        word.lower()
        for word in _LATIN.findall(query)
        if len(word) >= 3 and word.lower() not in _STOP
    ]
    for run in _CJK.findall(query):
        found += [run[i : i + 2] for i in range(len(run) - 1)] or [run]
    return found


def relevant(query: str, hits: Sequence[Hit]) -> bool:
    """Whether at least two of the first five *hits* mention a third of the query's tokens.

    An engine can answer a client it does not like with results that parse
    fine and are about something else (a company's home page for the first word
    of the query, an unrelated feed), so results alone are not an answer.
    """
    wanted = _tokens(query)
    if not wanted:
        return True
    close = 0
    for hit in hits[:5]:
        text = f"{hit.title} {hit.snippet}".lower()
        if sum(1 for token in wanted if token in text) / len(wanted) >= 0.34:
            close += 1
    return close >= 2


def looks_blocked(text: str) -> bool:
    """Whether a page that gave no results asks for a check instead (a captcha, a wait)."""
    low = text[:200_000].lower()
    return any(marker in low for marker in BLOCK_MARKERS)


def public(hits: Sequence[Hit]) -> list[Hit]:
    """*hits* whose address ``web_fetch`` could reach (a public ``http(s)`` address)."""
    kept: list[Hit] = []
    for hit in hits:
        try:
            check_url(hit.url)
        except AddressError:
            continue
        kept.append(hit)
    return kept


# ── the answer ──


@dataclass
class Answer:
    """What a search found, from which way, and why the ways before it did not answer."""

    way: str
    engine: str
    hits: list[Hit]
    # (engine, reason) for each way that did not answer, in the order tried.
    skipped: list[tuple[str, str]] = field(default_factory=list)
    # True when no way gave results that read as an answer and these are the
    # nearest: the caller says so.
    unsure: bool = False


class SearchError(Exception):
    """No way gave any result; the message says why each did not, for the model."""


Sleep = Callable[[float], Awaitable[None]]


class WebSearcher:
    """Searches with the ways of *order*, one at a time. See the module docstring.

    *browser* is the driver of the browser window the person switched on
    (None: the ``browser`` way is skipped). *fetcher*, *sleep* and *clock* are
    for tests.
    """

    def __init__(
        self,
        *,
        order: Sequence[str] = DEFAULT_ORDER,
        fetcher: WebFetcher | None = None,
        browser: BrowserDriver | None = None,
        gap_s: float = GAP_S,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.order = tuple(order) or DEFAULT_ORDER
        self._fetcher = fetcher or WebFetcher()
        self._browser = browser
        self._gap_s = gap_s
        self._sleep = sleep
        self._clock = clock
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def search(self, query: str, count: int = 6) -> Answer:
        """The first answer to *query*, at most *count* results; :class:`SearchError` if none."""
        count = max(1, min(MAX_COUNT, count))
        skipped: list[tuple[str, str]] = []
        nearest: Answer | None = None
        for key in self.order:
            way = WAYS.get(key)
            engine = way.engine if way else BROWSER_ENGINE
            if way is None and self._browser is None:
                continue  # the browser is not switched on: nothing to say about it
            hits, why = await (
                self._ask(way, query) if way else self._ask_browser(query)
            )
            if why:
                skipped.append((engine, why))
                continue
            if relevant(query, hits):
                return Answer(key, engine, hits[:count], skipped)
            skipped.append((engine, OFF_TOPIC))
            if nearest is None:
                nearest = Answer(key, engine, hits[:count], [])
        if nearest is not None:
            nearest.skipped = [item for item in skipped if item[0] != nearest.engine]
            nearest.unsure = True
            return nearest
        raise SearchError(
            "No search way gave results. "
            + " ".join(f"{engine} {why}." for engine, why in skipped)
        )

    # ── one way ──

    async def _gap(self, engine: str) -> None:
        """Wait until *engine* was last asked :attr:`_gap_s` ago."""
        lock = self._locks.setdefault(engine, asyncio.Lock())
        async with lock:
            wait = self._gap_s - (self._clock() - self._last.get(engine, -1e9))
            if wait > 0:
                await self._sleep(wait)
            self._last[engine] = self._clock()

    async def _ask(self, way: Way, query: str) -> tuple[list[Hit], str]:
        await self._gap(way.engine)
        try:
            page = await self._fetcher.fetch(
                way.url.format(q=quote(query)), accept=readable
            )
        except (AddressError, FetchError):
            return [], "could not be reached"
        if page.status in _STATUS_BLOCKED:
            return [], f"asked for a check (HTTP {page.status})"
        if page.status >= 300:
            return [], f"answered HTTP {page.status}"
        text = page.body.decode(page.charset or "utf-8", "replace")
        hits = public(way.read(text))
        if hits:
            return hits, ""
        return [], BLOCKED if looks_blocked(text) else UNREAD

    async def _ask_browser(self, query: str) -> tuple[list[Hit], str]:
        driver = self._browser
        assert driver is not None
        await self._gap(BROWSER_ENGINE)
        tab: str | None = None
        try:
            async with asyncio.timeout(BROWSER_S):
                url = BROWSER_URL.format(q=quote(query))
                tab = (await driver.open(url, new_tab=True)).id
                page = await driver.page(tab)
        except (BrowserError, TimeoutError):
            return [], "could not be reached"
        finally:
            if tab is not None:
                with contextlib.suppress(Exception):
                    await driver.close_tab(tab)
        links = [(e.label, e.href) for e in page.elements if e.kind == "link"]
        hits = public(pages.read_links(page.text, links, BROWSER_OWN))
        if hits:
            return hits, ""
        return [], BLOCKED if looks_blocked(f"{page.title}\n{page.text}") else UNREAD


__all__ = [
    "BROWSER",
    "DEFAULT_ORDER",
    "GAP_S",
    "MAX_COUNT",
    "MAX_QUERY_CHARS",
    "WAYS",
    "Answer",
    "SearchError",
    "WebSearcher",
    "looks_blocked",
    "relevant",
    "way_names",
]
