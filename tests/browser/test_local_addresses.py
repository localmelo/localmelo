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
from localmelo.support.browser.driver import RawPage
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.fakes import FakeDns, FakeDriver, link, page

HOME = "https://example.com/"
CONSOLE = "http://127.0.0.1:8765/#task/9c28dd0ce883"
SECRET = "Task 9c28dd0ce883 answer: the secret"
REFUSAL = "melo's browser does not open local or private addresses"
HINT = "For a task's result, use task_get."


class Anywhere(FakeDriver):
    """A browser that loads every address: one it has no page for is the console."""

    async def page(self, tab: str | None) -> RawPage:
        self.calls.append(("page", tab or ""))
        url = self._url(tab)
        return self.pages.get(url) or page(url, SECRET, title="localmelo console")


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


async def test_a_page_that_moved_to_this_computer_is_not_pictured_or_watched() -> None:
    # A page read as public can send its tab to the console afterwards; a
    # picture or a wait on that tab would show it.
    driver = world()
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    driver.tabs_["t1"].append(CONSOLE)
    shot = await run(tools, "browser_screenshot")
    assert_refused(shot)
    assert not shot.images
    driver.tabs_["t1"].append(CONSOLE)
    waited = await run(tools, "browser_wait", until="text", text="secret")
    assert_refused(waited)
    assert not [c for c in driver.calls if c[0] == "wait"]


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
