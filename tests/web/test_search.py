"""WebSearcher: the order of ways, the relevance check, fallback and politeness, with no network."""

from __future__ import annotations

import asyncio
import dataclasses
import socket
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import TOOL_ASK, ApprovalDecision
from localmelo.support.browser.driver import BrowserError, TabInfo
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.web import search as searching
from localmelo.support.web.fetch import FetchError, Response
from localmelo.tests.browser.fakes import FakeDns, FakeDriver, link, page
from localmelo.tests.web import search_fixtures as fx

QUERY = "garden tomato blight"


def _response(body: str, status: int = 200) -> Response:
    return Response(
        url="https://x.example/",
        status=status,
        reason="",
        content_type="text/html",
        charset="utf-8",
        body=body.encode(),
    )


class FakeFetcher:
    """Answers by engine host; records every address asked."""

    def __init__(self, **answers: Any) -> None:
        self.answers: dict[str, Response | Exception] = answers
        self.asked: list[str] = []

    async def fetch(self, url: str, **_: Any) -> Response:
        self.asked.append(url)
        for host, answer in self.answers.items():
            if host in url:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(f"unexpected address {url}")


ENGINE = "https://html.duckduckgo.com/html/?q=garden%20tomato%20blight"
HOME = "https://www.example.com/home"
LOCAL = "http://127.0.0.1:8765/console"
ELSEWHERE = "https://elsewhere.example/landing"
SECRET = "the console's private text"


@pytest.fixture(autouse=True)
def dns(monkeypatch: pytest.MonkeyPatch) -> FakeDns:
    """Name lookups answer from a table, never the network (a public address unless a test says)."""
    fake = FakeDns()
    monkeypatch.setattr(socket, "getaddrinfo", fake.getaddrinfo)
    return fake


def _results(text: str = fx.BROWSER_TEXT, **more: Any) -> Any:
    """The engine's result page, as the browser reads it."""
    elements = [link(label, href) for label, href in fx.BROWSER_LINKS]
    return page(
        ENGINE, text, *elements, title="garden tomato blight at DuckDuckGo", **more
    )


class Window(FakeDriver):
    """The Safari window: the engine's page for every query, and what a failing, slow or hijacked browser does.

    ``moved`` is the page a script has moved the engine's tab to, whatever the
    tab's address says.
    """

    def __init__(
        self,
        *pages: Any,
        fail: Exception | None = None,
        hang: bool = False,
        moved: Any = None,
        **more: Any,
    ) -> None:
        super().__init__(*pages, **more)
        self.fail = fail
        self.hang = hang
        self.moved = moved

    def _engine(self, url: str) -> bool:
        return url.startswith(ENGINE.split("?")[0])

    async def open(self, url: str, *, new_tab: bool, tab: str | None = None) -> TabInfo:
        if self.hang:
            await asyncio.sleep(3600)
        if self.fail:
            raise self.fail
        if self._engine(url) and ENGINE in self.redirects:
            self.redirects[url] = self.redirects[ENGINE]
        return await super().open(url, new_tab=new_tab, tab=tab)

    async def page(self, tab: str | None, whole: bool = False) -> Any:
        self.calls.append(("page", tab or ""))
        url = self._url(tab)
        if self._engine(url) and self.moved is not None:
            return self.moved
        if self._engine(url) and ENGINE in self.pages:
            return dataclasses.replace(self.pages[ENGINE], url=url)
        return self.pages[url]

    def read(self) -> list[str]:
        return [tab for call, tab in self.calls if call == "page"]


class Asks:
    """The run's approval policy as the person's answers: says yes, keeps the questions."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def __call__(
        self, name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        self.asked.append(name)
        return ApprovalDecision(True)


def _window(*pages: Any, **more: Any) -> tuple[Window, BrowserToolProvider]:
    driver = Window(*pages, **more)
    return driver, BrowserToolProvider(driver)


class Clock:
    """A clock the fake sleep moves, so the gap is checked without waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _searcher(
    fetcher: FakeFetcher,
    *,
    browser: BrowserToolProvider | None = None,
    order: tuple[str, ...] = searching.DEFAULT_ORDER,
    clock: Clock | None = None,
) -> searching.WebSearcher:
    clock = clock or Clock()
    return searching.WebSearcher(
        order=order,
        fetcher=fetcher,  # type: ignore[arg-type]
        browser=browser,  # type: ignore[arg-type]
        sleep=clock.sleep,
        clock=clock,
    )


# ── the order, and the first way that answers ──


async def test_the_first_way_that_gives_results_that_fit_answers_alone() -> None:
    fetcher = FakeFetcher(**{"bing.com": _response(fx.BING_RSS)})
    answer = await _searcher(fetcher).search(QUERY)
    assert (answer.way, answer.engine, answer.skipped, answer.unsure) == (
        "bing-rss",
        "Bing",
        [],
        False,
    )
    assert len(answer.hits) == 3 and len(fetcher.asked) == 1
    assert (
        "format=rss" in fetcher.asked[0]
        and "garden%20tomato%20blight" in fetcher.asked[0]
    )


async def test_count_cuts_the_results_and_is_held_to_the_most() -> None:
    fetcher = FakeFetcher(**{"bing.com": _response(fx.BING_RSS)})
    assert len((await _searcher(fetcher).search(QUERY, 2)).hits) == 2
    assert len((await _searcher(fetcher).search(QUERY, 99)).hits) == 3
    assert len((await _searcher(fetcher).search(QUERY, 0)).hits) == 1


def test_the_default_order_was_measured_not_guessed() -> None:
    assert searching.DEFAULT_ORDER == ("bing-rss", "browser", "baidu")
    assert set(searching.way_names()) == {"bing-rss", "baidu", "browser"}


# ── fallback says which way answered and why the others did not ──


async def test_a_blocked_way_falls_back_and_the_answer_says_so() -> None:
    fetcher = FakeFetcher(
        **{"bing.com": _response("", 429), "baidu.com": _response(fx.BAIDU)}
    )
    answer = await _searcher(fetcher, order=("bing-rss", "baidu")).search(
        "番茄 晚疫病 怎么防治"
    )
    assert (answer.way, answer.engine) == ("baidu", "Baidu")
    assert answer.skipped == [("Bing", "asked for a check (HTTP 429)")]


async def test_a_page_that_asks_for_a_check_is_blocked_not_unreadable() -> None:
    fetcher = FakeFetcher(
        **{
            "bing.com": _response("<html>Please complete the CAPTCHA</html>"),
            "baidu.com": _response(fx.BAIDU),
        }
    )
    answer = await _searcher(fetcher, order=("bing-rss", "baidu")).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.skipped == [("Bing", searching.BLOCKED)]


async def test_a_page_with_no_results_and_no_check_is_unreadable() -> None:
    fetcher = FakeFetcher(
        **{"bing.com": _response(fx.BING_RSS_EMPTY), "baidu.com": _response(fx.BAIDU)}
    )
    answer = await _searcher(fetcher, order=("bing-rss", "baidu")).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.skipped == [("Bing", searching.UNREAD)]


async def test_a_way_that_cannot_be_reached_falls_back() -> None:
    fetcher = FakeFetcher(
        **{"bing.com": FetchError("timed out"), "baidu.com": _response(fx.BAIDU)}
    )
    answer = await _searcher(fetcher, order=("bing-rss", "baidu")).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.skipped == [("Bing", "could not be reached")]


async def test_results_about_something_else_are_not_an_answer() -> None:
    # Bing answers an unknown client with results that parse and do not fit.
    fetcher = FakeFetcher(
        **{"bing.com": _response(fx.BING_RSS), "baidu.com": _response(fx.BAIDU)}
    )
    answer = await _searcher(fetcher, order=("bing-rss", "baidu")).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.way == "baidu"
    assert answer.skipped == [("Bing", searching.OFF_TOPIC)]
    assert answer.unsure is False


async def test_when_every_way_is_off_topic_the_nearest_is_returned_as_unsure() -> None:
    fetcher = FakeFetcher(
        **{"bing.com": _response(fx.BING_RSS), "baidu.com": _response(fx.BAIDU)}
    )
    answer = await _searcher(fetcher, order=("bing-rss", "baidu")).search(
        "quantum chromodynamics lattice"
    )
    assert answer.unsure is True and answer.way == "bing-rss"
    assert answer.skipped == [("Baidu", searching.OFF_TOPIC)]


async def test_when_no_way_gives_anything_the_error_names_each_reason() -> None:
    fetcher = FakeFetcher(
        **{"bing.com": _response("", 403), "baidu.com": FetchError("down")}
    )
    with pytest.raises(searching.SearchError) as err:
        await _searcher(fetcher, order=("bing-rss", "baidu")).search(QUERY)
    text = str(err.value)
    assert "Bing asked for a check (HTTP 403)." in text
    assert "Baidu could not be reached." in text


async def test_an_address_the_fetcher_would_refuse_is_dropped_from_the_results() -> (
    None
):
    fetcher = FakeFetcher(
        **{
            "bing.com": _response(
                fx.BING_RSS.replace(
                    "https://extension.example.org/blight", "http://127.0.0.1:8080/x"
                )
            )
        }
    )
    answer = await _searcher(fetcher).search(QUERY)
    assert all("127.0.0.1" not in hit.url for hit in answer.hits)
    assert len(answer.hits) == 2


# ── the browser way ──


def _bing_blocked() -> FakeFetcher:
    return FakeFetcher(
        **{"bing.com": _response("", 429), "baidu.com": _response(fx.BAIDU)}
    )


async def test_the_browser_way_reads_the_window_and_closes_its_tab() -> None:
    driver, browser = _window(_results())
    fetcher = FakeFetcher(**{"bing.com": _response("", 202)})
    answer = await _searcher(fetcher, browser=browser).search(QUERY)
    assert (answer.way, answer.engine) == ("browser", "DuckDuckGo in Safari")
    assert answer.skipped == [("Bing", "asked for a check (HTTP 202)")]
    assert [hit.url for hit in answer.hits] == [
        "https://www.example.com/garden/tomato-blight",
        "https://extension.example.org/blight",
        "https://shop.example.com/sprays?id=7",
    ]  # the address on this computer is dropped
    assert [url for call, url in driver.calls if call == "open"] == [ENGINE]
    assert driver.tabs_ == {}, "the tab is closed"
    assert "Water the soil" in answer.hits[0].snippet


async def test_the_browser_way_is_left_out_when_the_browser_is_off() -> None:
    answer = await _searcher(_bing_blocked()).search("番茄 晚疫病 怎么防治")
    assert answer.way == "baidu"
    # Nothing is said about a way that was never on.
    assert answer.skipped == [("Bing", "asked for a check (HTTP 429)")]


async def test_a_challenge_page_in_the_window_is_blocked_and_the_tab_still_closes() -> (
    None
):
    driver, browser = _window(
        page(ENGINE, fx.CHALLENGE_TEXT, title="DuckDuckGo"),
    )
    answer = await _searcher(_bing_blocked(), browser=browser).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.way == "baidu"
    assert ("DuckDuckGo in Safari", searching.BLOCKED) in answer.skipped
    assert driver.tabs_ == {}


@pytest.mark.parametrize(
    "make",
    [
        lambda: Window(fail=BrowserError("window closed")),
        lambda: Window(hang=True),
    ],
)
async def test_a_window_that_fails_or_hangs_falls_back(
    make: Callable[[], Window], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(searching, "BROWSER_S", 0.05)
    browser = BrowserToolProvider(make())
    answer = await _searcher(_bing_blocked(), browser=browser).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.way == "baidu"
    assert ("DuckDuckGo in Safari", "could not be reached") in answer.skipped


# The browser tools' own checks: the engine's page is read as any page is.


@pytest.mark.parametrize(
    ("what", "redirects", "moved"),
    [
        ("redirects to this computer", {ENGINE: LOCAL}, None),
        ("redirects to another site", {ENGINE: ELSEWHERE}, None),
        ("moves itself to this computer", {}, page(LOCAL, SECRET)),
        ("moves itself to another site", {}, page(ELSEWHERE, SECRET)),
    ],
)
async def test_a_page_that_leaves_the_engine_is_blocked_unread_and_nobody_is_asked(
    what: str, redirects: dict[str, str], moved: Any
) -> None:
    driver, browser = _window(
        _results(), page(ELSEWHERE, SECRET), redirects=redirects, moved=moved
    )
    asks = Asks()
    token = TOOL_ASK.set(asks)
    try:
        answer = await _searcher(_bing_blocked(), browser=browser).search(
            "番茄 晚疫病 怎么防治"
        )
    finally:
        TOOL_ASK.reset(token)
    assert answer.way == "baidu", what
    assert ("DuckDuckGo in Safari", searching.PAGE_REFUSED) in answer.skipped
    assert all(SECRET not in hit.snippet for hit in answer.hits)
    assert (
        asks.asked == []
    ), "the person is not asked about a site a search page went to"
    assert ("blank", "t1") in driver.calls, "the refused page is left for an empty one"
    assert driver.tabs_ == {}, "and its tab is closed"
    if redirects:
        assert driver.read() == [], "a page that was redirected is not even read"


async def test_a_page_with_a_frame_on_this_computer_is_blocked() -> None:
    driver, browser = _window(_results(frames=(LOCAL,)))
    answer = await _searcher(_bing_blocked(), browser=browser).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.way == "baidu"
    assert ("DuckDuckGo in Safari", searching.PAGE_REFUSED) in answer.skipped
    assert ("blank", "t1") in driver.calls and driver.tabs_ == {}


async def test_an_engine_name_that_resolves_to_this_computer_is_blocked_unloaded(
    dns: FakeDns,
) -> None:
    dns.answers["html.duckduckgo.com"] = ["127.0.0.1"]
    driver, browser = _window(_results())
    answer = await _searcher(_bing_blocked(), browser=browser).search(
        "番茄 晚疫病 怎么防治"
    )
    assert answer.way == "baidu"
    assert [call for call, _ in driver.calls if call in ("open", "page")] == []
    assert any(why == "could not be reached" for _, why in answer.skipped)


async def test_the_engine_site_is_not_asked_about_since_the_switch_allows_it() -> None:
    driver, browser = _window(_results())
    asks = Asks()
    token = TOOL_ASK.set(asks)
    try:
        answer = await _searcher(_bing_blocked(), browser=browser).search(QUERY)
    finally:
        TOOL_ASK.reset(token)
    assert answer.way == "browser" and asks.asked == []


async def test_the_chats_own_tabs_are_left_as_they_were() -> None:
    driver, browser = _window(_results(), page(HOME, "Welcome home."))
    tools = {t.definition.name: t for t in await browser.start()}
    await tools["browser_open"].function(url=HOME)  # the chat's tab, t1
    await _searcher(_bing_blocked(), browser=browser).search(QUERY)
    assert list(driver.tabs_) == ["t1"], "only the search tab was closed"
    again = await tools["browser_read"].function()  # still the current tab
    assert "Welcome home." in str(again)


async def test_a_second_search_reads_a_second_page_in_a_tab_of_its_own() -> None:
    driver, browser = _window(_results())
    searcher = _searcher(_bing_blocked(), browser=browser)
    await searcher.search(QUERY)
    await searcher.search(QUERY)
    opens = [url for call, url in driver.calls if call == "open"]
    assert opens == [ENGINE, ENGINE] and driver.tabs_ == {}


# ── politeness ──


async def test_one_engine_is_asked_at_most_once_per_search_and_two_seconds_apart() -> (
    None
):
    clock = Clock()
    fetcher = FakeFetcher(**{"bing.com": _response(fx.BING_RSS)})
    searcher = _searcher(fetcher, clock=clock)
    await searcher.search(QUERY)
    await searcher.search(QUERY)
    assert len(fetcher.asked) == 2
    assert clock.slept and clock.slept[0] == pytest.approx(searching.GAP_S)


async def test_different_engines_do_not_wait_for_each_other() -> None:
    clock = Clock()
    fetcher = FakeFetcher(
        **{"bing.com": _response("", 429), "baidu.com": _response(fx.BAIDU)}
    )
    await _searcher(fetcher, order=("bing-rss", "baidu"), clock=clock).search(
        "番茄 晚疫病 怎么防治"
    )
    assert clock.slept == []


# ── the check ──


def test_relevance_needs_two_of_the_first_five_to_fit() -> None:
    hits = [
        searching.Hit("Tomato blight guide", "https://a.example.com", "garden notes"),
        searching.Hit("Blight in the garden", "https://b.example.com", ""),
        searching.Hit("Insurance quotes", "https://c.example.com", "cheap"),
    ]
    assert searching.relevant(QUERY, hits)
    assert not searching.relevant(QUERY, hits[:1] + hits[2:])
    assert not searching.relevant(QUERY, [])


def test_relevance_counts_chinese_by_character_pairs() -> None:
    fits = [searching.Hit("番茄晚疫病防治", "https://a.example.com", "")] * 2
    assert searching.relevant("番茄晚疫病", fits)
    assert not searching.relevant(
        "番茄晚疫病", [searching.Hit("天气预报", "https://a.example.com", "")] * 2
    )


def test_a_query_of_only_small_words_is_not_held_to_the_check() -> None:
    assert searching.relevant("how to", [])


def test_block_markers_read_in_both_languages() -> None:
    assert searching.looks_blocked("Our systems have detected unusual traffic")
    assert searching.looks_blocked("请完成安全验证")
    assert not searching.looks_blocked("Tomatoes like sun")
