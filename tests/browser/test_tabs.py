"""melo's own tabs: one tab per chat by default, short ids every tool takes, tidy endings."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.fakes import FakeDriver, button, link, page, safari_id
from localmelo.tests.browser.test_provider import run, tools_of

HOME = "https://example.com/"
NEWS = "https://example.com/news"
OTHER = "https://other.org/"


def world(**kwargs: Any) -> FakeDriver:
    return FakeDriver(
        page(HOME, "Welcome.", link("News", NEWS), title="Home"),
        page(NEWS, "Today's news.", link("Home", HOME), title="News"),
        page(OTHER, "Elsewhere.", title="Other"),
        **kwargs,
    )


def listing(text: str) -> list[str]:
    return [line for line in text.splitlines() if line[:2] in ("* ", "  ")]


async def provider_of(
    driver: FakeDriver, **kwargs: Any
) -> tuple[BrowserToolProvider, dict[str, Any]]:
    provider = BrowserToolProvider(driver, **kwargs)
    return provider, {t.definition.name: t for t in await provider.start()}


# ── One tab per chat ──


async def test_open_reuses_the_tab_melo_opened_unless_asked_for_a_new_one() -> None:
    driver = world()
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_open", url=NEWS)
    assert not out.is_error and "Today's news." in out.text
    assert list(driver.tabs_) == ["t1"] and driver.tabs_["t1"] == [HOME, NEWS]
    assert driver.open_in == ["t1"]  # the second open asked for the first tab
    await run(tools, "browser_open", url=OTHER, tab="new")
    assert list(driver.tabs_) == ["t1", "t2"]
    # "current" is the tab opened last, and a tab id names one.
    await run(tools, "browser_open", url=HOME, tab="current")
    assert driver.tabs_["t2"][-1] == HOME
    await run(tools, "browser_open", url=NEWS, tab="t1")
    assert driver.tabs_["t1"][-1] == NEWS and len(driver.tabs_) == 2


async def test_ten_opens_in_one_chat_leave_one_tab() -> None:
    # What the person saw: ten pages guessed in ten new tabs.
    driver = FakeDriver(*[page(f"{HOME}p{n}", f"Page {n}.") for n in range(10)])
    tools = await tools_of(driver)
    for n in range(10):
        assert not (await run(tools, "browser_open", url=f"{HOME}p{n}")).is_error
    assert list(driver.tabs_) == ["t1"] and len(driver.tabs_["t1"]) == 10


async def test_a_lost_tab_is_replaced_by_a_new_one_and_open_goes_on() -> None:
    driver = world()
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    driver.lose("t1")
    out = await run(tools, "browser_open", url=NEWS)
    assert not out.is_error and "Today's news." in out.text
    assert list(driver.tabs_) == ["t2"]


# ── Tab ids that work ──


async def test_the_tab_list_shows_short_ids_never_the_helpers_long_ones() -> None:
    driver = world(ids=safari_id)
    tools = await tools_of(driver)
    first = await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    text = (await run(tools, "browser_tabs", action="list")).text
    assert listing(text) == [
        '  t1 https://example.com/ "Home"',
        '* t2 https://other.org/ "Other"',
    ]
    for shown in (text, first.text):
        assert "page-" not in shown and "…" not in shown
    assert " tab=t1 " in first.text


@pytest.mark.parametrize("tool", ["browser_read", "browser_find", "browser_wait"])
async def test_every_tool_that_takes_a_tab_takes_a_short_id(tool: str) -> None:
    driver = world(ids=safari_id)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    args = {
        "browser_read": {},
        "browser_find": {"query": "news"},
        "browser_wait": {"until": "load"},
    }[tool]
    out = await run(tools, tool, tab="t1", **args)
    assert not out.is_error, out.text
    assert "page-" not in out.text
    # The call went to the helper with the helper's own id.
    assert any(safari_id(1) in arg for _, arg in driver.calls) or tool == "browser_find"


async def test_screenshot_switch_and_close_take_short_ids() -> None:
    driver = world(ids=safari_id)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    assert (await run(tools, "browser_screenshot", tab="t1")).images
    switched = await run(tools, "browser_tabs", action="switch", tab="t1")
    assert "* t1 " in switched.text
    closed = await run(tools, "browser_tabs", action="close", tab="t2")
    assert not closed.is_error and "t2" not in closed.text
    assert ("close_tab", safari_id(2)) in driver.calls
    assert list(driver.tabs_) == [safari_id(1)]


@pytest.mark.parametrize(
    "tab",
    [
        "page-635DB8F8-27ED-4CA5-81A2-7EAEE1DF2B…",  # cut off, as the list once showed it
        safari_id(1),  # the helper's own id
        "t9",
        "t",
        "T1x",
        "1",
        "../t1",
        "",
    ],
)
@pytest.mark.parametrize("action", ["switch", "close"])
async def test_a_cut_off_or_unknown_tab_id_is_never_accepted(
    tab: str, action: str
) -> None:
    driver = world(ids=safari_id)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    before = list(driver.calls)
    out = await run(tools, "browser_tabs", action=action, tab=tab)
    assert out.is_error
    assert driver.calls == before, "nothing reached the browser"
    assert list(driver.tabs_) == [safari_id(1)]
    if tab:
        assert "is not a tab of this chat" in out.text and "t1" in out.text


@pytest.mark.parametrize("tool", ["browser_read", "browser_screenshot"])
async def test_other_tools_refuse_a_helper_id_too(tool: str) -> None:
    driver = world(ids=safari_id)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, tool, tab=safari_id(1))
    assert out.is_error and "is not a tab of this chat" in out.text


async def test_a_tab_the_helper_lost_leaves_the_list_and_the_list_says_so() -> None:
    driver = world(ids=safari_id)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    driver.lose(safari_id(1))
    text = (await run(tools, "browser_tabs", action="list")).text
    assert "t1 is gone: the browser no longer has it." in text
    assert listing(text) == ['* t2 https://other.org/ "Other"']
    # It is not listed again, and its id is not handed out again.
    again = (await run(tools, "browser_tabs", action="list")).text
    assert "gone" not in again
    await run(tools, "browser_open", url=HOME, tab="new")
    ids = [
        r.lstrip("* ").split()[0]
        for r in listing((await run(tools, "browser_tabs", action="list")).text)
    ]
    assert ids == ["t2", "t3"]


async def test_closing_a_tab_the_helper_lost_succeeds_and_says_so() -> None:
    driver = world(ids=safari_id)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    driver.lose(safari_id(1))
    out = await run(tools, "browser_tabs", action="close", tab="t1")
    assert not out.is_error and "t1 was already gone" in out.text
    assert listing(out.text) == ['* t2 https://other.org/ "Other"']


async def test_switching_to_a_lost_tab_says_it_is_gone() -> None:
    driver = world()
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    driver.lose("t1")
    out = await run(tools, "browser_tabs", action="switch", tab="t1")
    assert out.is_error and "t1 is gone" in out.text


async def test_a_tab_melo_did_not_open_is_not_listed_or_closed() -> None:
    # The window may hold a tab of the person's own: melo lists none of them,
    # says how many there are, and cannot be made to close one.
    driver = world(ids=safari_id)
    driver.tabs_["persons-tab"] = [HOME]
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=OTHER)
    text = (await run(tools, "browser_tabs", action="list")).text
    assert listing(text) == ['* t1 https://other.org/ "Other"']
    assert "1 other tab in the window: not melo's, left alone." in text
    assert "persons-tab" not in text
    out = await run(tools, "browser_tabs", action="close", tab="persons-tab")
    assert out.is_error and "persons-tab" in driver.tabs_


# ── Endings ──


async def test_tabs_melo_opened_are_closed_when_the_chat_ends() -> None:
    driver = world(ids=safari_id)
    driver.tabs_["persons-tab"] = [HOME]
    provider, tools = await provider_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    await provider.close()
    assert list(driver.tabs_) == ["persons-tab"], "only the person's own tab is left"
    assert driver.calls[-1] == ("close", "")


# ── Tabs that appeared are not melo's to close ──


def popup_world() -> FakeDriver:
    driver = FakeDriver(
        page(HOME, "Welcome.", button("Open"), title="Home"),
        page(NEWS, "Today's news.", link("Home", HOME), title="News"),
    )
    driver.popups["Open"] = NEWS
    return driver


async def test_a_tab_that_appeared_in_an_action_is_left_open_when_the_chat_ends() -> (
    None
):
    driver = popup_world()
    provider, tools = await provider_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert "It opened t2." in out.text and "Today's news." in out.text
    # t2 was read: melo may use it while the chat lasts.
    assert (await run(tools, "browser_read", tab="t2")).text.startswith("<<page")
    await provider.close()
    assert list(driver.tabs_) == ["t2"], "only the tab browser_open made was closed"
    assert ("close_tab", "t2") not in driver.calls


async def test_a_new_chat_and_the_idle_tidy_leave_appeared_tabs_open_and_forgotten() -> (
    None
):
    driver = popup_world()
    provider, tools = await provider_of(driver, idle_close_s=0.05)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_click", ref="e1")
    await provider.end_chat()
    assert list(driver.tabs_) == ["t1", "t2"], "t1 is kept blank, t2 left alone"
    assert driver.tabs_["t1"][-1] == "about:blank" and driver.tabs_["t2"] == [NEWS]
    # Let go of: it is no tab of this chat any more.
    gone = await run(tools, "browser_read", tab="t2")
    assert gone.is_error and "not a tab of this chat" in gone.text
    await provider.close()


async def test_a_plain_open_never_reuses_a_tab_that_appeared() -> None:
    driver = popup_world()
    driver.pages[OTHER] = page(OTHER, "Elsewhere.", title="Other")
    provider, tools = await provider_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_click", ref="e1")  # t2 appeared and is now current
    await run(tools, "browser_open", url=OTHER)
    assert driver.tabs_["t2"] == [NEWS], "the popup was not navigated away"
    assert driver.tabs_["t1"][-1] == OTHER, "the tab melo made is the one reused"
    # A tab the model names itself is its own call.
    await run(tools, "browser_open", url=HOME, tab="t2")
    assert driver.tabs_["t2"][-1] == HOME
    await provider.close()


async def test_keep_tabs_leaves_the_tabs_open() -> None:
    driver = world()
    provider, tools = await provider_of(driver, keep_tabs=True, idle_close_s=0.01)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    await asyncio.sleep(0.1)
    await provider.end_chat()
    await provider.close()
    assert list(driver.tabs_) == ["t1", "t2"]
    assert not [c for c in driver.calls if c[0] in ("close_tab", "blank")]


async def test_a_close_that_fails_still_closes_the_driver() -> None:
    driver = world()
    provider, tools = await provider_of(driver)
    await run(tools, "browser_open", url=HOME)
    driver.lose("t1")  # close_tab now raises
    await provider.close()
    assert driver.calls[-1] == ("close", "")


async def test_idle_tabs_are_closed_except_one_left_blank() -> None:
    driver = world()
    provider, tools = await provider_of(driver, idle_close_s=0.05)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    await run(tools, "browser_open", url=NEWS, tab="new")
    assert list(driver.tabs_) == ["t1", "t2", "t3"]
    await asyncio.sleep(0.3)
    assert list(driver.tabs_) == ["t3"], "the tab used last is kept"
    assert ("blank", "t3") in driver.calls
    # Nothing in it is approved; the next open reuses it.
    assert (await run(tools, "browser_read")).is_error
    out = await run(tools, "browser_open", url=HOME)
    assert not out.is_error and driver.tabs_["t3"][-1] == HOME
    assert list(driver.tabs_) == ["t3"]


async def test_a_busy_chat_is_not_tidied() -> None:
    driver = world()
    provider, tools = await provider_of(driver, idle_close_s=0.2)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    for _ in range(4):  # a call every 0.1 s keeps the timer from ever ending
        await asyncio.sleep(0.1)
        assert not (await run(tools, "browser_read")).is_error
    assert list(driver.tabs_) == ["t1", "t2"]
    await provider.close()


async def test_a_new_chat_tidies_the_old_chats_tabs_and_approves_nothing() -> None:
    driver = world()
    provider, tools = await provider_of(driver)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_open", url=OTHER, tab="new")
    await provider.end_chat()
    assert list(driver.tabs_) == ["t2"] and not provider._origins
    assert (await run(tools, "browser_read")).is_error
    await provider.close()


def test_keep_tabs_is_a_setting_that_is_off_by_default() -> None:
    from localmelo.support import config

    cfg = config.Config()
    assert cfg.browser.keep_tabs is False
    assert cfg.validate() == [] or "keep_tabs" not in " ".join(cfg.validate())
    cfg.browser.keep_tabs = "yes"  # type: ignore[assignment]
    assert any("keep_tabs" in problem for problem in cfg.validate())
