"""``[tools] web_fetch_allow_local``: the browser opens the listed test servers, and only as melo's own step.

A listed server is reached by melo's own navigation (``browser_open``) and by
a page already on a listed server; a request of any other page to it stays
refused, as when nothing is listed. The provider's rows run on a stand-in
driver (melo's own address checks, all a driver like Safari has); the
request rows run through the rule background Chrome applies to each request
it pauses (``LocalRule.decide``), with no real browser.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.contracts.tools import TOOL_MARKS
from localmelo.melo.executor.models import ToolOutput
from localmelo.support import config, web_local
from localmelo.support.browser import chrome, local_apps
from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.support.browser.driver import TabInfo
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.web.guard import LocalAllow
from localmelo.tests.browser.fakes import FakeDriver, button, link, page
from localmelo.tests.browser.test_local_addresses import REFUSAL, Asks, asking, run
from localmelo.tests.browser.test_local_apps import MODEL_PORT, Calls, _browser, rule

A, B, UNLISTED, GATEWAY = 18601, 18602, 18603, 18605
FIXTURE = f"http://127.0.0.1:{A}/form"
OTHER = f"http://127.0.0.1:{B}/next"
PUBLIC = "https://example.com/"
NOTE = web_local.NOTE


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def _cfg(*ports: int) -> config.Config:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    cfg.vllm.chat_model = "m"
    cfg.tools.web_fetch_allow_local = [f"127.0.0.1:{port}" for port in ports]
    return cfg


def _listed(*ports: int) -> LocalAllow:
    allow = web_local.allow_of(_cfg(*ports))
    assert allow is not None
    return allow


def _driver(*pages: Any, **redirects: str) -> FakeDriver:
    return FakeDriver(
        page(FIXTURE, "Fixture A.", link("Next", OTHER), button("Go"), title="A"),
        page(FIXTURE + "/done", "Fixture A, done.", title="A"),
        page(OTHER, "Fixture B.", title="B"),
        page(PUBLIC, "Public.", link("Fixture", FIXTURE), button("Go"), title="P"),
        page(FIXTURE + "/out", "Fixture A, out.", link("Public", PUBLIC), title="A"),
        *pages,
        redirects=redirects,
    )


async def _tools(driver: FakeDriver, allow: LocalAllow | None) -> dict[str, Any]:
    provider = BrowserToolProvider(driver, listed=allow)
    return {tool.definition.name: tool for tool in await provider.start()}


async def _step(driver: FakeDriver, *steps: tuple[str, dict[str, Any]]) -> ToolOutput:
    """Run *steps* on the tools with A and B listed; the last step's result."""
    tools = await _tools(driver, _listed(A, B))
    out = ToolOutput(text="")
    with asking(Asks()):
        for name, args in steps:
            out = await run(tools, name, **args)
    return out


def _open(url: str) -> tuple[str, dict[str, Any]]:
    return ("browser_open", {"url": url})


# ── Who asks: melo's own steps and the pages on a listed server (both drivers) ──

# (who asks, the driver, the steps, allowed)
ROWS: list[tuple[str, Callable[[], FakeDriver], list[Any], bool]] = [
    ("melo's navigation", _driver, [_open(FIXTURE)], True),
    (
        "a redirect within the listed site melo opened",
        lambda: _driver(**{FIXTURE: FIXTURE + "/done"}),
        [_open(FIXTURE)],
        True,
    ),
    (
        "a redirect of a public page melo opened",
        lambda: _driver(**{PUBLIC: FIXTURE}),
        [_open(PUBLIC)],
        False,
    ),
    (
        "a redirect from the listed site melo opened to another listed one",
        lambda: _driver(**{FIXTURE: OTHER}),
        [_open(FIXTURE)],
        False,
    ),
    (
        "a frame of a listed page",
        lambda: _driver(page(FIXTURE + "/framed", "Framed.", frames=(OTHER,))),
        [_open(FIXTURE + "/framed")],
        True,
    ),
    (
        "a frame of a public page",
        lambda: _driver(page(PUBLIC + "framed", "Framed.", frames=(FIXTURE,))),
        [_open(PUBLIC + "framed")],
        False,
    ),
    (
        "a link a listed page holds",
        _driver,
        [_open(FIXTURE), ("browser_follow", {"ref": "e1"})],
        True,
    ),
    (
        "a link a public page holds",
        _driver,
        [_open(PUBLIC), ("browser_follow", {"ref": "e1"})],
        False,
    ),
    (
        "a click on a public page's link",
        _driver,
        [_open(PUBLIC), ("browser_click", {"ref": "e1"})],
        False,
    ),
    (
        "a listed page that moves itself to another listed server",
        lambda: _moves(OTHER),
        [_open(FIXTURE), ("browser_click", {"ref": "e2"})],
        True,
    ),
    (
        "a public page that moves itself to a listed server",
        lambda: _moves(FIXTURE),
        [_open(PUBLIC), ("browser_click", {"ref": "e2"})],
        False,
    ),
    (
        "a public page reached from a listed one, moving itself to a listed server",
        lambda: _moves(OTHER),
        [
            _open(FIXTURE + "/out"),
            ("browser_click", {"ref": "e1"}),
            ("browser_click", {"ref": "e3"}),  # refs count across the chat
        ],
        False,
    ),
    (
        "a new tab a listed page opens",
        lambda: _pops(OTHER),
        [_open(FIXTURE), ("browser_click", {"ref": "e2"})],
        False,
    ),
]


def _moves(to: str) -> FakeDriver:
    driver = _driver()
    driver.then["Go"] = to
    return driver


def _pops(to: str) -> FakeDriver:
    driver = _driver()
    driver.popups["Go"] = to
    return driver


@pytest.mark.parametrize(
    ("who", "make", "steps", "allowed"), ROWS, ids=[row[0] for row in ROWS]
)
async def test_who_reaches_a_listed_server_in_the_browser_tools(
    who: str, make: Callable[[], FakeDriver], steps: list[Any], allowed: bool
) -> None:
    out = await _step(make(), *steps)
    if allowed:
        assert not out.is_error and NOTE in out.text, out.text
    else:
        assert out.is_error and REFUSAL in out.text, out.text
        assert "Fixture" not in out.text, "nothing of a listed page is read"


# (who asks, the request's sender header, melo's navigation of the main frame to, allowed)
REQUESTS: list[tuple[str, dict[str, str], tuple[str, int] | None, bool]] = [
    ("melo's navigation", {}, ("127.0.0.1", A), True),
    (
        "a redirect of melo's navigation to another listed site",
        {},
        ("127.0.0.1", B),
        False,
    ),
    ("a frame or sub-request of a listed page", {"Referer": OTHER}, None, True),
    (
        "a listed page's request that carries its origin",
        {"Origin": f"http://127.0.0.1:{B}"},
        None,
        True,
    ),
    ("a sub-request of a public page", {"Referer": PUBLIC}, None, False),
    ("a request with no sender", {}, None, False),
    (
        "a redirected request whose origin is null",
        {"Origin": "null", "Referer": OTHER},
        None,
        False,
    ),
    (
        "a page on a name for a listed address",
        {"Referer": f"http://localhost:{B}/"},
        None,
        False,
    ),
]


@pytest.mark.parametrize(
    ("who", "headers", "opening", "allowed"), REQUESTS, ids=[r[0] for r in REQUESTS]
)
def test_who_reaches_a_listed_server_in_background_chrome(
    who: str, headers: dict[str, str], opening: tuple[str, int] | None, allowed: bool
) -> None:
    verdict = rule().decide(
        FIXTURE,
        headers.get("Origin", ""),
        (),
        _listed(A, B),
        sender=local_apps.sender_of(headers),
        opening=opening,
    )
    assert verdict.kind == ("pass" if allowed else "block"), verdict


async def _pause(
    browser: chrome._Chrome, url: str, kind: str, headers: dict[str, str], ident: str
) -> None:
    browser._dispatch(
        {
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": ident,
                "frameId": "PAGE1",
                "resourceType": kind,
                "request": {"url": url, "method": "GET", "headers": headers},
            },
        }
    )
    while browser.jobs:
        await asyncio.gather(*list(browser.jobs), return_exceptions=True)


async def test_background_chrome_lets_melos_own_navigation_through_and_tells_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver marks the listed site it opens for that load only; a request of another page is blocked and told."""
    calls = Calls()
    browser = _browser(rule(), calls)
    driver = ChromeBackgroundDriver(command=["unused"], local=rule())
    driver._browser, driver._mine = browser, {"PAGE1"}
    driver.allow_local(None, _listed(A))
    assert set(browser.listed) == {"PAGE1"}

    async def target(tab: str | None) -> str:
        return "PAGE1"

    async def here(target: str) -> TabInfo:
        return TabInfo("PAGE1", FIXTURE)

    async def load(session: str, start: Callable[[], Awaitable[Any]]) -> None:
        await _pause(browser, FIXTURE, "Document", {}, "nav")  # Chrome pauses the load

    monkeypatch.setattr(driver, "_target", target)
    monkeypatch.setattr(driver, "_focus", target)
    monkeypatch.setattr(driver, "_here", here)
    monkeypatch.setattr(driver, "_go", load)
    await driver.open(FIXTURE, new_tab=False, tab="PAGE1")
    assert browser.opening == {}, "the mark ends with the load"
    # The same address asked by the page itself after the load, and by a public page.
    await _pause(browser, FIXTURE, "Document", {}, "self")
    await _pause(browser, FIXTURE, "XHR", {"Referer": PUBLIC}, "public")
    assert [(m, p.get("errorReason")) for m, p in calls.answered()] == [
        ("Fetch.continueRequest", None),
        ("Fetch.failRequest", "BlockedByClient"),
        ("Fetch.failRequest", "BlockedByClient"),
    ]
    assert [a.port for a in driver.local_attempts()] == [A]  # told, once


async def test_a_page_let_through_as_a_server_the_person_wrote_reaches_no_listed_one() -> (
    None
):
    """A written server opens from any page, as before; that gives its page no listed server."""
    from localmelo.tests.browser.test_acting import asking as asking_with
    from localmelo.tests.web.test_written_local import WroteForBrowser

    tools = await _tools(_driver(), _listed(A, B))
    with asking_with(WroteForBrowser(("127.0.0.1", A))):
        assert not (await run(tools, "browser_open", url=PUBLIC)).is_error
        written = await run(tools, "browser_follow", ref="e1")  # PUBLIC's link to A
        assert not written.is_error and "Fixture A." in written.text, written.text
        out = await run(tools, "browser_follow", ref="e3")  # A's link to B
    assert out.is_error and REFUSAL in out.text and "Fixture B" not in out.text


# ── Nothing listed, melo's own ports, names, the mark ──


async def test_with_nothing_listed_a_local_address_is_refused_as_before() -> None:
    assert web_local.allow_of(_cfg()) is None
    for allow in (None, web_local.allow_of(_cfg())):
        driver = _driver()
        tools = await _tools(driver, allow)
        asks = Asks()
        with asking(asks):
            out = await run(tools, "browser_open", url=FIXTURE)
        assert out.is_error and REFUSAL in out.text and NOTE not in out.text
        assert not asks.asked and not [c for c in driver.calls if c[0] == "open"]
    assert rule().decide(FIXTURE, "", (), None, sender=OTHER).kind == "block"


async def test_melos_own_port_is_refused_even_when_listed() -> None:
    """The session's live config: ``--port`` sets the gateway port in memory only."""
    live = _cfg(A, GATEWAY)
    live.gateway.port = GATEWAY
    allow = web_local.allow_of(live)
    assert allow is not None and allow.listed(FIXTURE) is not None
    driver = _driver(page(f"http://127.0.0.1:{GATEWAY}/", "melo's console."))
    tools = await _tools(driver, allow)
    with asking(Asks()):
        out = await run(tools, "browser_open", url=f"http://127.0.0.1:{GATEWAY}/")
    assert out.is_error and REFUSAL in out.text
    assert not [c for c in driver.calls if c[0] == "open"]
    # Background Chrome refuses them too, and the ports its own rule knows (the saved file's).
    for port, listed in ((GATEWAY, allow), (MODEL_PORT, _listed(A, MODEL_PORT))):
        url = f"http://127.0.0.1:{port}/"
        assert rule().decide(url, "", (), listed, sender=FIXTURE).kind == "block"


@pytest.mark.parametrize(
    "url",
    [
        f"http://localhost:{A}/form",
        f"http://127.1:{A}/form",
        f"http://[::ffff:127.0.0.1]:{A}/form",
        f"http://2130706433:{A}/form",
        f"http://127.0.0.1:{UNLISTED}/",
    ],
)
async def test_only_the_listed_address_itself_is_opened(url: str) -> None:
    driver = _driver(page(url, "Fixture by another name."))
    tools = await _tools(driver, _listed(A))
    with asking(Asks()):
        out = await run(tools, "browser_open", url=url)
    assert out.is_error and REFUSAL in out.text
    assert not [c for c in driver.calls if c[0] == "open"]


async def test_a_step_on_a_listed_server_is_marked_and_another_is_not() -> None:
    """Opening it, and a part of the reading kept from it; a public page is not marked."""
    tools = await _tools(_driver(), _listed(A))
    for url, marked in ((PUBLIC, False), (FIXTURE, True)):
        for name, args in (
            ("browser_open", {"url": url}),
            ("browser_read", {"part": 1}),
        ):
            marks: dict[str, Any] = {}
            token = TOOL_MARKS.set(marks)
            try:
                with asking(Asks()):
                    out = await run(tools, name, **args)
            finally:
                TOOL_MARKS.reset(token)
            assert not out.is_error, out.text
            assert (marks.get("allowed_local") is True) is marked, name
            assert (NOTE in out.text) is marked, name


async def test_a_session_gives_the_browser_the_list_of_its_live_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read when the session's tools are made, from its own config, not the saved file."""
    from localmelo.support import browser as browser_package
    from localmelo.support.sessions.agents import build_session_agent

    config.save(_cfg())  # the file lists nothing
    live = _cfg(A, GATEWAY)
    live.gateway.port = GATEWAY
    given: list[Any] = []
    made = browser_package.provider_for

    def spy(driver: str, **kwargs: Any) -> Any:
        given.append(kwargs["listed"])
        return made(driver, **kwargs)

    monkeypatch.setattr(browser_package, "provider_for", spy)
    agent = await build_session_agent(
        cfg=live,
        base_url=None,
        model=None,
        workspace=str(Path.cwd()),
        options=AgentOptions(mode="interactive"),
        browser_driver="chrome",
    )
    await agent.close()
    [allow] = given
    assert allow.listed(FIXTURE) == ("127.0.0.1", A)
    assert allow.listed(f"http://127.0.0.1:{GATEWAY}/") is None


async def test_no_tool_argument_carries_a_list() -> None:
    """The list is config only: the provider takes it once, and no browser tool's arguments name it."""
    for tool in await BrowserToolProvider(FakeDriver()).start():
        for name in tool.definition.parameters.get("properties", {}):
            assert not any(word in name for word in ("allow", "local", "listed")), name
