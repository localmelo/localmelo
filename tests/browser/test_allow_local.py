"""``[tools] web_fetch_allow_local`` is web_fetch's alone: the browser never opens a listed address.

The setting lets ``web_fetch`` read a test server on this computer. The browser
tools are a second way to reach the same address, and they must not follow the
setting. The first test lists a real server, shows that ``web_fetch`` reads it
(so the setting is in force) and that the browser still refuses it before the
site question and before any page loads. The other two show why: the browser
never asks the list and takes no hook that could carry it.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

import pytest

from localmelo.support import config, web_local
from localmelo.support.browser import addresses
from localmelo.support.browser.addresses import require_public
from localmelo.support.browser.driver import BrowserError
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.web.guard import LocalAllow
from localmelo.support.web.tool import WebFetchProvider
from localmelo.tests.browser.fakes import link, page
from localmelo.tests.browser.test_local_addresses import (
    HOME,
    REFUSAL,
    Anywhere,
    Asks,
    asking,
    assert_refused,
    opened,
    run,
    tools_of,
    world,
)
from localmelo.tests.web import pages

KEY = "tools.web_fetch_allow_local"


@pytest.fixture
def server() -> Iterator[dict[str, Any]]:
    with pages.serve() as state:
        yield state


def _list(entries: list[str], how: str = "file") -> None:
    """Put *entries* in the setting, in the config file or as a ``-X`` override."""
    if how == "x":
        config.set_overrides({KEY: entries})
        return
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    cfg.vllm.chat_model = "m"
    cfg.tools.web_fetch_allow_local = entries
    config.save(cfg)


def _url(state: dict[str, Any], path: str = "/page") -> str:
    return f"http://127.0.0.1:{state['port']}{path}"


def _paths(state: dict[str, Any]) -> list[str]:
    return [r["path"] for r in state["requests"]]  # type: ignore[index]


async def _web_fetch_reads(url: str) -> None:
    """The control: the setting is in force, so web_fetch reads *url* once the person says yes."""
    [fetch] = await WebFetchProvider().start()
    with asking(Asks()):
        text = await fetch.function(url=url)
    assert "# Tomatoes" in str(text), text


@pytest.mark.parametrize("how", ["file", "x"])
async def test_a_listed_server_is_read_by_web_fetch_and_refused_by_the_browser(
    server: dict[str, Any], how: str
) -> None:
    url = _url(server)
    _list([f"127.0.0.1:{server['port']}"], how)
    await _web_fetch_reads(url)
    served = _paths(server)
    assert served == ["/page"]

    # The browser's own check refuses it, in the words for the model.
    with pytest.raises(BrowserError) as caught:
        await require_public(url)
    assert REFUSAL in str(caught.value) and "loopback" in str(caught.value)

    # And so does every tool, before the site question and before a load.
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url=url)
        current = await run(tools, "browser_open", url=url, tab="current")
    assert_refused(out)
    assert_refused(current)
    assert not asks.asked and not opened(driver)
    assert _paths(server) == served, "the browser never reached the server"


async def test_the_browser_never_asks_the_local_allow_list(
    server: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spy on ``LocalAllow.allows``: web_fetch asks it, the browser tools never do."""
    url = _url(server)
    _list([f"127.0.0.1:{server['port']}"])
    asked: list[tuple[str, int]] = []
    allows = LocalAllow.allows

    def spy(self: LocalAllow, host: str, ip: Any, port: int) -> bool:
        asked.append((host, port))
        return allows(self, host, ip, port)

    monkeypatch.setattr(LocalAllow, "allows", spy)
    await _web_fetch_reads(url)
    assert asked, "web_fetch asks the list"
    asked.clear()

    def no_fetcher(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the browser built a web_fetch fetcher")

    monkeypatch.setattr(web_local, "fetcher", no_fetcher)
    driver = Anywhere(page(HOME, "Welcome.", link("Test server", url)))
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    assert_refused(await run(tools, "browser_follow", ref="e1"))
    assert_refused(await run(tools, "browser_open", url=url))
    assert_refused(await run(tools, "browser_open", url="http://[::1]:1/"))
    assert asked == []


def test_the_browser_tools_take_no_hook() -> None:
    parameters = inspect.signature(BrowserToolProvider).parameters
    assert set(parameters) <= {
        "driver",
        "keep_tabs",
        "idle_close_s",
        "settle_s",
        "desktop",
        "opener",
        # The live view only watches; it opens and reads nothing.
        "live",
        "live_fps",
    }
    assert not [name for name in parameters if "allow" in name or "local" in name]
    for function in (require_public, addresses.console_sites):
        assert not [
            name
            for name in inspect.signature(function).parameters
            if "allow" in name or "local" in name or "protect" in name
        ]
