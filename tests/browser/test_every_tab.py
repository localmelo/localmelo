"""After an action, every tab it touched is checked before any is read.

A click can open several tabs, move the page it was made on, move it a moment
later, or fail after it has already moved it. Whatever a page does, nothing
from a local or private address may reach the model: the tabs are checked by
the browser's own report of their addresses first, a tab that cannot be made
blank is closed, and a tab whose address cannot be told is not read.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.support.browser.driver import Act
from localmelo.support.browser.fields import form_pairs
from localmelo.tests.browser.fakes import FakeDriver, button, page
from localmelo.tests.browser.test_acting import (
    DONE,
    HOME,
    OTHER,
    Asks,
    acts,
    asking,
    textbox,
    world,
)
from localmelo.tests.browser.test_local_addresses import (
    CONSOLE,
    SECRET,
    assert_refused,
)
from localmelo.tests.browser.test_provider import run, tools_of


def reads(driver: FakeDriver) -> list[str]:
    """The tabs whose page the model's tools read, in order."""
    return [tab for call, tab in driver.calls if call == "page"]


async def opened_home(driver: FakeDriver) -> Any:
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    return tools


# ── Every tab a click opens ──


async def test_one_local_tab_among_three_refuses_all_and_none_is_read() -> None:
    driver = world(button("Open"))
    driver.popups["Open"] = [DONE, CONSOLE, OTHER]
    tools = await opened_home(driver)
    before = reads(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert_refused(out)
    assert reads(driver) == before, "no tab was read after the click"
    # The local one is blanked in place and left open: melo did not make the tab.
    assert ("blank", "t3") in driver.calls and ("close_tab", "t3") not in driver.calls
    assert driver.tabs_["t3"] == ["about:blank"]
    listing = (await run(tools, "browser_tabs", action="list")).text
    assert SECRET not in listing and "127.0.0.1" not in listing


async def test_the_opener_moved_to_a_local_address_refuses_a_public_popup_too() -> None:
    driver = world(button("Open"))
    driver.popups["Open"] = DONE
    tools = await opened_home(driver)
    opening = driver.act

    async def act(tab: str | None, a: Act) -> str:
        note = await opening(tab, a)
        driver.tabs_["t1"].append(CONSOLE)  # the page moved itself as it opened one
        return note

    driver.act = act  # type: ignore[assignment]
    before = reads(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert_refused(out)
    assert reads(driver) == before
    assert ("blank", "t1") in driver.calls
    assert SECRET not in (await run(tools, "browser_read")).text


async def test_a_page_that_moves_itself_a_moment_after_the_action_is_seen() -> None:
    driver = world(button("Check in"))
    driver.then["Check in"] = CONSOLE
    tools = await opened_home(driver)
    before = reads(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert_refused(out)
    assert ("blank", "t1") in driver.calls
    assert reads(driver) == before, "the page it moved to was never read"


async def test_a_tab_that_cannot_be_blanked_is_closed() -> None:
    driver = world(button("Check in"))
    driver.lands["Check in"] = CONSOLE
    driver.blank_fails = True
    tools = await opened_home(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert_refused(out)
    assert ("close_tab", "t1") in driver.calls
    assert "t1" not in (await run(tools, "browser_tabs", action="list")).text


async def test_a_popup_that_cannot_be_blanked_is_let_go_of_not_closed() -> None:
    driver = world(button("Open"))
    driver.popups["Open"] = CONSOLE
    driver.blank_fails = True
    tools = await opened_home(driver)
    assert_refused(await run(tools, "browser_click", ref="e1"))
    # melo did not make the tab, so it does not close it; it just stops using it.
    assert list(driver.tabs_) == ["t1", "t2"]
    assert ("close_tab", "t2") not in driver.calls
    listing = (await run(tools, "browser_tabs", action="list")).text
    assert "t2" not in listing and "127.0.0.1" not in listing
    assert (await run(tools, "browser_read", tab="t2")).is_error


# ── Back never returns to a page that was refused ──


async def test_back_never_returns_to_a_refused_address() -> None:
    driver = world(button("Check in"))
    driver.lands["Check in"] = CONSOLE  # the page sends the tab to a local address
    tools = await opened_home(driver)
    assert_refused(await run(tools, "browser_click", ref="e1"))
    # The refused page was swapped for an empty one in the history, so nothing
    # after the home page remembers the local address.
    assert not any(CONSOLE in entry for entry in driver.tabs_["t1"])
    before = reads(driver)
    out = await run(tools, "browser_back")
    assert CONSOLE not in out.text and SECRET not in out.text
    assert not any(CONSOLE in entry for entry in driver.tabs_["t1"])
    assert reads(driver) == before + ["t1"], "the page it went back to was read"
    assert "Welcome" in out.text


async def test_back_onto_a_refused_address_blanks_it_and_reads_nothing() -> None:
    # A history the helper could not keep clean (its blank was a new entry):
    # Back lands on the local address, which is checked before anything is read.
    driver = world(button("Check in"))
    tools = await opened_home(driver)
    driver.tabs_["t1"][:] = [CONSOLE, HOME]
    before = reads(driver)
    out = await run(tools, "browser_back")
    assert_refused(out)
    assert reads(driver) == before
    assert ("blank", "t1") in driver.calls


# ── An action that fails ──


async def test_a_failed_action_still_checks_where_the_tab_went() -> None:
    driver = world(button("Check in"))
    driver.fails["Check in"] = CONSOLE  # it moved the tab, then raised
    tools = await opened_home(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert_refused(out)
    assert ("blank", "t1") in driver.calls


async def test_a_failed_action_drops_the_refs_of_a_page_it_may_have_changed() -> None:
    driver = world(button("Check in"))
    driver.fails["Check in"] = HOME  # nothing visible moved
    tools = await opened_home(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "did not take that action" in out.text
    again = await run(tools, "browser_click", ref="e1")
    assert again.is_error and len(acts(driver)) == 1, "the old ref is not used again"


# ── A tab whose address cannot be told ──


async def test_a_tab_whose_address_is_unknown_is_not_read_or_acted_in() -> None:
    driver = world(button("Check in"))
    tools = await opened_home(driver)
    driver.location_empty.add("t1")
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "could not tell which page" in out.text
    assert not driver.acts
    driver.location_empty.clear()
    # The reading went with it: a ref of the page that could not be told is not used.
    again = await run(tools, "browser_click", ref="e1")
    assert again.is_error and not driver.acts


async def test_an_address_that_vanishes_after_the_action_is_not_read() -> None:
    driver = world(button("Check in"))
    tools = await opened_home(driver)
    acting = driver.act

    async def act(tab: str | None, a: Act) -> str:
        note = await acting(tab, a)
        driver.location_empty.add("t1")
        return note

    driver.act = act  # type: ignore[assignment]
    before = reads(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "could not tell which page" in out.text
    assert reads(driver) == before


# ── A window that cannot be listed ──


async def test_a_window_that_cannot_be_listed_is_not_acted_in() -> None:
    driver = world(button("Check in"))
    tools = await opened_home(driver)
    driver.tabs_fail = True
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "could not list the browser's tabs" in out.text
    assert not driver.acts


async def test_a_window_that_cannot_be_listed_after_the_action_is_not_read() -> None:
    driver = world(button("Check in"))
    tools = await opened_home(driver)
    acting = driver.act

    async def act(tab: str | None, a: Act) -> str:
        note = await acting(tab, a)
        driver.tabs_fail = True
        return note

    driver.act = act  # type: ignore[assignment]
    before = reads(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "could not list the browser's tabs after that" in out.text
    assert reads(driver) == before


# ── Frames ──


async def test_a_page_whose_readable_frame_is_local_is_refused() -> None:
    driver = world(button("Check in"))
    driver.pages[HOME] = page(HOME, "Welcome.", button("Check in"), frames=(CONSOLE,))
    tools = await tools_of(driver)
    out = await run(tools, "browser_open", url=HOME)
    assert_refused(out)
    assert ("blank", "t1") in driver.calls
    assert SECRET not in (await run(tools, "browser_read")).text


async def test_a_frame_on_a_public_address_or_a_blank_one_is_no_problem() -> None:
    driver = world(button("Check in"))
    driver.pages[HOME] = page(
        HOME, "Welcome.", button("Check in"), frames=(OTHER, "about:blank", "")
    )
    tools = await tools_of(driver)
    out = await run(tools, "browser_open", url=HOME)
    assert not out.is_error


# ── Who is a tab: only what melo made or saw appear ──


async def test_a_tab_melo_did_not_make_is_not_listed_as_its_own() -> None:
    driver = world(button("Check in"))
    driver.tabs_["person-tab"] = [OTHER]  # the person's own tab in the window
    tools = await opened_home(driver)
    listing = (await run(tools, "browser_tabs", action="list")).text
    assert "person-tab" not in listing and OTHER not in listing
    assert "1 other tab in the window: not melo's, left alone." in listing
    out = await run(tools, "browser_tabs", action="close", tab="person-tab")
    assert out.is_error and "person-tab" in driver.tabs_


async def test_a_tab_that_appears_while_melo_acts_is_adopted_but_not_one_before() -> (
    None
):
    driver = world(button("Open"))
    driver.tabs_["person-tab"] = [OTHER]
    driver.popups["Open"] = DONE
    tools = await opened_home(driver)
    out = await run(tools, "browser_click", ref="e1")
    assert not out.is_error and "It opened t2." in out.text
    assert "person-tab" in driver.tabs_
    listing = (await run(tools, "browser_tabs", action="list")).text
    assert "person-tab" not in listing and "t2" in listing


# ── What a form shows ──


async def test_the_question_about_a_send_shows_its_fields_with_secrets_left_out() -> (
    None
):
    asks = Asks()
    fields = form_pairs(
        [
            ["email", "me@example.org"],
            ["password", "hunter2"],
            ["csrf", "x" * 40],
            ["note", "hello", "h"],
        ]
    )
    driver = world(button("Send", submit=OTHER, form=fields))
    driver.lands["Send"] = OTHER
    tools = await opened_home(driver)
    with asking(asks):
        out = await run(tools, "browser_click", ref="e1")
    assert not out.is_error
    [(_, arguments, _)] = [a for a in asks.asked if a[0].startswith("browser-submit:")]
    assert arguments["fields"] == [
        ["email", "me@example.org"],
        ["password", "(hidden)"],
        ["csrf", "(hidden)"],
        ["note", "(hidden)"],
    ]
    assert "hunter2" not in repr(asks.asked)


async def test_enter_outside_a_form_is_a_send_to_an_address_nobody_knows() -> None:
    asks = Asks()
    driver = world(textbox("Search"))
    driver.lands["enter:Search"] = DONE
    tools = await opened_home(driver)
    with asking(asks):
        out = await run(tools, "browser_type", ref="e1", text="cats", submit=True)
    assert not out.is_error
    [(_, arguments, preview)] = [
        a for a in asks.asked if a[0].startswith("browser-submit:")
    ]
    assert arguments["to"] == "" and arguments["how"] == "enter"
    assert arguments["fields"] == [["typed", "cats"]]
    assert (
        preview == "send what is typed on example.com (the page decides where it goes)"
    )


async def test_typed_text_is_in_the_question_for_the_site() -> None:
    asks = Asks()
    driver = world(textbox("Search"))
    tools = await opened_home(driver)
    with asking(asks):
        await run(tools, "browser_type", ref="e1", text="my own words")
    [(_, arguments, _)] = [a for a in asks.asked if a[0].startswith("browser-type:")]
    assert arguments["how"] == "type" and arguments["text"] == "my own words"
    assert not [a for a in asks.asked if a[0].startswith("browser-act:")]


@pytest.mark.parametrize("ref_kind", ["click", "select"])
async def test_a_click_or_choice_names_its_kind_in_the_site_question(
    ref_kind: str,
) -> None:
    from localmelo.support.browser.driver import Element

    asks = Asks()
    elements = {
        "click": button("Check in"),
        "select": Element("select", "Country", options=("France", "Spain")),
    }
    driver = world(elements[ref_kind])
    tools = await opened_home(driver)
    with asking(asks):
        if ref_kind == "click":
            await run(tools, "browser_click", ref="e1")
        else:
            await run(tools, "browser_select", ref="e1", option="Spain")
    [(_, arguments, _)] = [a for a in asks.asked if a[0].startswith("browser-act:")]
    assert (
        arguments["how"] == {"click": "click", "select": "choose an option"}[ref_kind]
    )
    assert "text" not in arguments
