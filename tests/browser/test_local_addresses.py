"""The browser never opens this computer, its network or melo's own console.

Attacker-side: each address below is one a model was led to, or a page sent
the browser to. The fake browser loads anything, as a real one would, and a
local page shows a secret, so a test fails if the secret is ever read.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import TOOL_ASK, ApprovalDecision
from localmelo.melo.contracts.tools import ProvidedTool
from localmelo.melo.executor.models import ToolOutput
from localmelo.support import config, gateway_endpoint
from localmelo.support.browser import sites
from localmelo.support.browser.driver import BrowserError, RawPage, Shot, TabInfo
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.sites import origin_of
from localmelo.tests.browser.fakes import FakeDns, FakeDriver, link, page

HOME = "https://example.com/"
CONSOLE = "http://127.0.0.1:8765/#task/9c28dd0ce883"
SECRET = "Task 9c28dd0ce883 answer: the secret"
REFUSAL = "melo's browser does not open local or private addresses"
HINT = "For a task's result, use task_get."


class Anywhere(FakeDriver):
    """A browser that loads every address: one it has no page for is the console.

    A wait polls whatever the tab shows at that moment, as a real browser
    does: ``visit`` is an address the page goes to while it is watched, and
    ``returns`` sends it back before the wait ends.
    """

    visit = ""
    returns = False

    def shown(self, url: str) -> RawPage:
        return self.pages.get(url) or page(url, SECRET, title="localmelo console")

    async def page(self, tab: str | None) -> RawPage:
        self.calls.append(("page", tab or ""))
        return self.shown(self._url(tab))

    async def wait(
        self,
        tab: str | None,
        until: str,
        text: str,
        seconds: float,
        *,
        origin: str = "",
    ) -> bool:
        self.calls.append(("wait", origin))
        history = self.tabs_[str(self.current if tab in (None, "current") else tab)]
        polled = [history[-1]] + ([self.visit] if self.visit else [])
        if self.visit:
            history.append(self.visit)
        found = False
        for url in polled:
            if origin and origin_of(url) != origin:
                continue  # the condition holds only on the page's own site
            found = found or text in self.shown(url).text
        if self.visit and self.returns:
            history.pop()
        return found


class Asks:
    """The run's approval policy: says yes to every site and keeps what was asked."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def __call__(
        self, name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        self.asked.append(name)
        return ApprovalDecision(True, remember=True)


@contextlib.contextmanager
def asking(asks: Asks) -> Iterator[None]:
    token = TOOL_ASK.set(asks)
    try:
        yield
    finally:
        TOOL_ASK.reset(token)


def world(**redirects: str) -> Anywhere:
    return Anywhere(
        page(HOME, "Welcome.", link("Console", CONSOLE), link("News", HOME + "news")),
        page(HOME + "news", "Today's news."),
        redirects=redirects,
    )


async def tools_of(driver: FakeDriver) -> dict[str, ProvidedTool]:
    return {t.definition.name: t for t in await BrowserToolProvider(driver).start()}


async def run(tools: dict[str, ProvidedTool], name: str, **args: Any) -> ToolOutput:
    result = await tools[name].function(**args)
    return result if isinstance(result, ToolOutput) else ToolOutput(text=result)


def opened(driver: FakeDriver) -> list[str]:
    return [url for call, url in driver.calls if call == "open"]


def assert_refused(out: ToolOutput) -> None:
    assert out.is_error, out.text
    assert REFUSAL in out.text and HINT in out.text, out.text
    assert SECRET not in out.text


LOCAL = [
    CONSOLE,
    "http://127.0.0.1/",
    "http://127.8.9.10:8401/",
    "http://localhost:8765/#task/x",
    "http://LOCALHOST./",
    "http://console.localhost/",
    "http://[::1]:8765/",
    "http://[::]/",
    "http://0.0.0.0:8765/",
    "http://10.0.0.8/",
    "http://172.16.4.2/",
    "http://192.168.1.1/admin",
    "http://100.64.0.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[fd00::1]/",
    "http://[fe80::1]/",
    "http://[::ffff:127.0.0.1]:8765/",
    "http://[::ffff:7f00:1]/",
    "http://[::ffff:192.168.1.1]/",
    "http://2130706433:8765/",
    "http://0x7f000001/",
    "http://0x7f.1/",
    "http://0177.0.0.1/",
    "http://127.1/",
    "http://0/",
    "http://１２７．０．０．１:8765/",
    "http://intranet/",
    "http://printer.local/",
    "http://router.lan/",
]


@pytest.mark.parametrize("url", LOCAL)
async def test_a_local_or_private_address_is_refused_before_anything(
    url: str, dns: FakeDns
) -> None:
    # Refused before the site question and before the page loads: nothing to
    # approve, nothing loaded, and the model is pointed at task_get.
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url=url)
    assert_refused(out)
    assert not asks.asked and not opened(driver)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1\\@example.com:8765/",
        "http://example.com@127.0.0.1:8765/",
        "http://user:pw@example.com/",
    ],
)
async def test_an_address_with_a_user_part_is_not_opened(url: str) -> None:
    # A browser reads `127.0.0.1\@example.com` as the host 127.0.0.1, which
    # Python reads as a user part before example.com: melo opens neither.
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url=url)
    assert out.is_error and SECRET not in out.text
    assert not asks.asked and not opened(driver)


@pytest.mark.parametrize(
    "answers",
    [
        ["127.0.0.1"],
        ["::1"],
        ["10.1.2.3"],
        ["169.254.169.254"],
        ["::ffff:127.0.0.1"],
        ["93.184.216.34", "192.168.0.10"],
    ],
)
async def test_a_name_that_resolves_to_a_local_address_is_refused(
    answers: list[str], dns: FakeDns
) -> None:
    dns.answers["console.example"] = answers
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url="http://console.example:8765/")
    assert_refused(out)
    assert "console.example" in out.text
    assert not asks.asked and not opened(driver)


async def test_a_name_that_is_not_found_is_not_opened(dns: FakeDns) -> None:
    # melo checks every address a name has before the browser loads it; a
    # name with no address cannot be checked, so it is not opened.
    dns.answers["gone.example"] = None
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url="https://gone.example/")
    assert out.is_error and "gone.example was not found" in out.text
    assert not asks.asked and not opened(driver)


async def test_a_redirect_to_this_computer_is_blanked_and_refused() -> None:
    asks, driver = Asks(), world(**{"https://short.example/x": CONSOLE})
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url="https://short.example/x")
    assert_refused(out)
    assert asks.asked == ["browser:https://short.example"]
    assert ("blank", "t1") in driver.calls and ("page", "t1") not in driver.calls


async def test_a_page_that_moves_itself_to_this_computer_is_not_read() -> None:
    # The address the tab reports is public; the page a script left it on is not.
    driver = world()
    driver.pages["https://jump.example/"] = page(CONSOLE, SECRET)
    tools = await tools_of(driver)
    out = await run(tools, "browser_open", url="https://jump.example/")
    assert_refused(out)
    assert ("blank", "t1") in driver.calls


async def test_a_link_to_this_computer_is_not_followed() -> None:
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_follow", ref="e1")  # Console
    assert_refused(out)
    assert opened(driver) == [HOME] and asks.asked == ["browser:https://example.com"]


@pytest.mark.parametrize("tab", [None, "current", "t1"])
@pytest.mark.parametrize("tool", ["browser_screenshot", "browser_wait"])
async def test_a_page_that_moved_to_this_computer_is_not_pictured_or_watched(
    tool: str, tab: str | None
) -> None:
    # A page read as public can send its tab to the console afterwards; a
    # picture or a wait on that tab would show it, whichever way the tab is named.
    driver = world()
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    driver.tabs_["t1"].append(CONSOLE)
    args = {} if tab is None else {"tab": tab}
    if tool == "browser_wait":
        args.update(until="text", text="the secret")
    out = await run(tools, tool, **args)
    assert_refused(out)
    assert not out.images and ("blank", "t1") in driver.calls
    assert not [c for c in driver.calls if c[0] == "wait"]


async def test_a_wait_after_a_new_chat_watches_no_tab() -> None:
    # /new takes back every approval; the tab left on the console is nobody's.
    driver = world()
    provider = BrowserToolProvider(driver)
    tools = {t.definition.name: t for t in await provider.start()}
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    driver.tabs_["t1"].append(CONSOLE)
    provider.forget()
    out = await run(tools, "browser_wait", until="text", text="the secret")
    assert out.is_error and "No page is open" in out.text
    assert not [c for c in driver.calls if c[0] == "wait"]


async def test_a_page_that_moves_while_it_is_watched_answers_nothing() -> None:
    driver = world()
    driver.visit = CONSOLE
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    out = await run(tools, "browser_wait", until="text", text="the secret")
    assert_refused(out)
    assert ("wait", "https://example.com") in driver.calls
    assert ("blank", "t1") in driver.calls


@pytest.mark.parametrize("text", ["the secret", "nothing like this"])
async def test_a_page_that_visits_this_computer_during_a_wait_answers_nothing(
    text: str,
) -> None:
    # Gone to the console and back before the wait ends: the answer must not
    # depend on what the console showed.
    driver = world()
    driver.visit, driver.returns = CONSOLE, True
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    out = await run(tools, "browser_wait", until="text", text=text, seconds=1)
    assert not out.is_error and out.text.startswith("Timed out after 1 s"), out.text


class Blind(Anywhere):
    """A browser that cannot say which page a tab shows."""

    async def tabs(self) -> list[Any]:
        raise BrowserError("Safari's automation server has no list_tabs tool.")

    async def location(self, tab: str | None) -> str:
        raise BrowserError("Safari did not answer evaluate_javascript in 15 s.")


class Unsure(Anywhere):
    """A browser that lists its tabs by title only and answers with no address."""

    async def tabs(self) -> list[Any]:
        return [TabInfo(tab, "", "") for tab in self.tabs_]

    async def location(self, tab: str | None) -> str:
        return ""


@pytest.mark.parametrize("make", [Blind, Unsure])
@pytest.mark.parametrize("tool", ["browser_screenshot", "browser_wait"])
async def test_a_tab_whose_page_cannot_be_checked_is_not_pictured_or_watched(
    make: type[Anywhere], tool: str
) -> None:
    driver = make(page(HOME, "Welcome."))
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    driver.tabs_["t1"].append(CONSOLE)
    args = {"until": "text", "text": "the secret"} if tool == "browser_wait" else {}
    out = await run(tools, tool, **args)
    assert out.is_error and not out.images and SECRET not in out.text
    assert not [c for c in driver.calls if c[0] == "wait"]


class MovesWhilePictured(Anywhere):
    async def screenshot(self, tab: str | None) -> Shot:
        self.tabs_["t1"].append(CONSOLE)  # the page moved while the picture was taken
        return await super().screenshot(tab)


async def test_a_picture_taken_while_the_page_moved_is_dropped() -> None:
    driver = MovesWhilePictured(page(HOME, "Welcome."))
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    out = await run(tools, "browser_screenshot")
    assert_refused(out)
    assert not out.images


async def test_the_tab_list_shows_only_pages_melo_checked() -> None:
    driver = world()
    driver.pages[CONSOLE] = page(CONSOLE, SECRET, title="localmelo console " + SECRET)
    provider = BrowserToolProvider(driver)
    tools = {t.definition.name: t for t in await provider.start()}
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=HOME + "news")
    driver.tabs_["t1"].append(CONSOLE)  # moved itself after it was read
    listing = (await run(tools, "browser_tabs", action="list")).text
    assert "  t1 (not checked)" in listing and "127.0.0.1" not in listing
    assert "console" not in listing and SECRET not in listing
    assert '* t2 https://example.com/news "T"' in listing
    provider.forget()
    listing = (await run(tools, "browser_tabs", action="list")).text
    assert "t1 (not checked)" in listing and "t2 (not checked)" in listing
    assert "example.com" not in listing


@pytest.mark.parametrize(
    "url", ["http://evil.com\uff20127.0.0.1/", "http://[::ffff:127.0.0.1]./"]
)
async def test_an_address_python_cannot_split_is_refused_in_plain_words(
    url: str,
) -> None:
    # `＠` becomes `@` under the normalization a browser applies; Python refuses
    # to split such an address and must not hand the model its own error.
    with pytest.raises(BrowserError, match="That is not a valid http or https address"):
        sites.require_origin(url)
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url=url)
    assert out.is_error and "not a valid http or https address" in out.text
    assert "NFKC" not in out.text and "netloc" not in out.text
    assert not asks.asked and not opened(driver)


def _console_at(host: str, port: int) -> None:
    path = config.config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(f'[gateway]\nhost = "{host}"\nport = {port}\n')


async def test_the_configured_console_is_refused_even_at_a_public_address(
    dns: FakeDns,
) -> None:
    _console_at("melo.example.net", 8765)
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(
            tools, "browser_open", url="http://MELO.example.net:8765/#task/x"
        )
        other_port = await run(
            tools, "browser_open", url="http://melo.example.net:8080/"
        )
    assert_refused(out)
    assert "console" in out.text
    assert not other_port.is_error
    assert opened(driver) == ["http://melo.example.net:8080/"]


async def test_a_name_for_the_consoles_public_address_is_refused(dns: FakeDns) -> None:
    _console_at("93.184.216.34", 8765)
    dns.answers["alias.example"] = ["93.184.216.34"]
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url="http://alias.example:8765/")
    assert_refused(out)
    assert not asks.asked and not opened(driver)


async def test_the_running_console_is_refused_where_it_listens() -> None:
    gateway_endpoint.write(
        gateway_endpoint.Endpoint(
            pid=os.getpid(),
            host="melo-run.example.net",
            port=9100,
            instance="",
            id="g1",
            started=time.time(),
        )
    )
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url="http://melo-run.example.net:9100/")
    assert_refused(out)
    assert not asks.asked and not opened(driver)


async def test_a_public_site_still_opens_after_its_name_is_checked(
    dns: FakeDns,
) -> None:
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url=HOME)
        news = await run(tools, "browser_follow", ref="e2")
    assert not out.is_error and "Welcome." in out.text
    assert not news.is_error and "Today's news." in news.text
    assert asks.asked == ["browser:https://example.com"]
    # The name is looked up, and every answer checked, before the page loads.
    assert "example.com" in dns.asked
