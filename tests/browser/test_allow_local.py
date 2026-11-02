"""``[tools] web_fetch_allow_local`` is web_fetch's alone: the browser never opens a listed address.

The setting lets ``web_fetch`` read a test server on this computer. The browser
tools are a second way to reach the same address, and they must not follow the
setting. Each test lists a real server, shows that ``web_fetch`` reads it
(so the setting is in force), and then shows that the browser still refuses it
before the site question, before any page loads, and for every way a tab can
get there: an address, a link, a redirect, a script, a tab that moved.
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
    SECRET,
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


@pytest.mark.parametrize(
    ("entry", "url"),
    [
        ("127.0.0.1:8123", "http://127.0.0.1:8123/"),
        ("127.0.0.1:8123", "https://127.0.0.1:8123/x?y=1#z"),
        ("127.0.0.1:8123", "http://127.0.0.1:8123"),
        ("[::1]:8123", "http://[::1]:8123/"),
        ("[::1]:8123", "http://[::1]:8123/a/b"),
    ],
)
async def test_every_listed_form_is_refused_by_the_browser(
    entry: str, url: str
) -> None:
    _list([entry])
    with pytest.raises(BrowserError, match="does not open local or private"):
        await require_public(url)
    asks, driver = Asks(), world()
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url=url)
    assert_refused(out)
    assert not asks.asked and not opened(driver)


async def test_a_name_for_a_listed_address_is_refused_by_the_browser(
    server: dict[str, Any],
) -> None:
    # Not listed as a name, and not let through as one either way.
    _list([f"127.0.0.1:{server['port']}"])
    for host in ("localhost", "127.1", "2130706433", "[::ffff:127.0.0.1]"):
        url = f"http://{host}:{server['port']}/page"
        with pytest.raises(BrowserError, match="does not open local or private"):
            await require_public(url)
    assert _paths(server) == []


async def test_a_link_to_a_listed_server_is_not_followed(
    server: dict[str, Any],
) -> None:
    url = _url(server)
    _list([f"127.0.0.1:{server['port']}"])
    driver = Anywhere(page(HOME, "Welcome.", link("Test server", url)))
    asks = Asks()
    tools = await tools_of(driver)
    with asking(asks):
        assert not (await run(tools, "browser_open", url=HOME)).is_error
        out = await run(tools, "browser_follow", ref="e1")
    assert_refused(out)
    assert opened(driver) == [HOME] and asks.asked == ["browser:https://example.com"]
    assert _paths(server) == []


async def test_a_redirect_to_a_listed_server_is_blanked_and_refused(
    server: dict[str, Any],
) -> None:
    url = _url(server)
    _list([f"127.0.0.1:{server['port']}"])
    asks, driver = Asks(), world(**{"https://short.example/x": url})
    tools = await tools_of(driver)
    with asking(asks):
        out = await run(tools, "browser_open", url="https://short.example/x")
    assert_refused(out)
    assert ("blank", "t1") in driver.calls and ("page", "t1") not in driver.calls
    assert _paths(server) == []


async def test_a_page_that_moves_itself_to_a_listed_server_is_not_read(
    server: dict[str, Any],
) -> None:
    url = _url(server)
    _list([f"127.0.0.1:{server['port']}"])
    driver = world()
    driver.pages["https://jump.example/"] = page(url, SECRET)
    tools = await tools_of(driver)
    out = await run(tools, "browser_open", url="https://jump.example/")
    assert_refused(out)
    assert ("blank", "t1") in driver.calls


@pytest.mark.parametrize("tool", ["browser_screenshot", "browser_wait"])
async def test_a_tab_that_moved_to_a_listed_server_is_not_pictured_or_watched(
    server: dict[str, Any], tool: str
) -> None:
    url = _url(server)
    _list([f"127.0.0.1:{server['port']}"])
    driver = world()
    tools = await tools_of(driver)
    assert not (await run(tools, "browser_open", url=HOME)).is_error
    driver.tabs_["t1"].append(url)
    args = {"until": "text", "text": "the secret"} if tool == "browser_wait" else {}
    out = await run(tools, tool, **args)
    assert_refused(out)
    assert not out.images and ("blank", "t1") in driver.calls
    assert not [c for c in driver.calls if c[0] == "wait"]


async def test_the_browsers_checks_carry_no_hook(
    server: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``require_public`` calls the guard with an address and a resolver, never a hook."""
    _list([f"127.0.0.1:{server['port']}"])
    seen: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    check_url, checked_addresses = addresses.check_url, addresses.checked_addresses

    def spy_check_url(*args: Any, **kwargs: Any) -> Any:
        seen.append(("check_url", args, kwargs))
        return check_url(*args, **kwargs)

    async def spy_checked(*args: Any, **kwargs: Any) -> Any:
        seen.append(("checked_addresses", args, kwargs))
        return await checked_addresses(*args, **kwargs)

    monkeypatch.setattr(addresses, "check_url", spy_check_url)
    monkeypatch.setattr(addresses, "checked_addresses", spy_checked)

    with pytest.raises(BrowserError):
        await require_public(_url(server))
    await require_public("https://example.com/")  # a name: both checks run

    assert [name for name, _, _ in seen] == [
        "check_url",
        "check_url",
        "checked_addresses",
    ]
    for name, args, kwargs in seen:
        assert kwargs == {}, f"{name} was given {kwargs}"
        assert len(args) == (1 if name == "check_url" else 2), f"{name} {args}"


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
    assert set(parameters) == {"driver"}
    for function in (require_public, addresses.console_sites):
        assert not [
            name
            for name in inspect.signature(function).parameters
            if "allow" in name or "local" in name or "protect" in name
        ]
