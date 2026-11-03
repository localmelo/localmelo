"""Frames: a box from another site is reached through Safari's own reading, and each act in it asks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from localmelo.support.browser.driver import Act, Element, StaleElementError
from localmelo.tests.browser.fakes import button
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world
from localmelo.tests.browser.test_native_input import native, opened
from localmelo.tests.browser.test_provider import run, tools_of


async def test_a_box_from_another_site_is_named_with_the_way_to_reach_it() -> None:
    driver = world(button("Go"))
    driver.pages[HOME] = dataclasses.replace(
        driver.pages[HOME],
        frames=("https://accounts.example.org/login",),
        unreached=("https://accounts.example.org/login",),
    )
    tools = await tools_of(driver)
    out = await run(tools, "browser_open", url=HOME)
    assert (
        "[a box from accounts.example.org that melo can't reach: browser_screenshot, "
        "then browser_click_point, can click in it]"
    ) in out.text
    assert out.text.index("can't reach") < out.text.index("<<end page")


async def test_at_most_four_boxes_from_other_sites_are_named() -> None:
    driver = world(button("Go"))
    boxes = tuple(f"https://ads{n}.example.net/slot" for n in range(7))
    driver.pages[HOME] = dataclasses.replace(
        driver.pages[HOME], frames=boxes, unreached=boxes
    )
    tools = await tools_of(driver)
    out = await run(tools, "browser_open", url=HOME)
    assert out.text.count("that melo can't reach") == 4


async def test_a_named_control_in_a_box_from_another_site_is_clicked_like_any_other() -> (
    None
):
    """melo knows it by its name: the click's own question names it, box and all (#465)."""
    go = Element(
        "button", "Continue", frame="x2_1_92", site="https://accounts.example.org"
    )
    pw = Element(
        "textbox",
        "",
        sensitive="password",
        frame="x2_1_90",
        site="https://accounts.example.org",
    )
    driver = world(button("Post"), go, pw)
    asks = Asks()
    tools = await tools_of(driver)
    with asking(asks):
        opened = await run(tools, "browser_open", url=HOME)
        assert (
            '[e2 button "Continue" in a box from accounts.example.org]' in opened.text
        )
        assert (
            "[e3 password field: melo never types here in a box from accounts.example.org]"
            in opened.text
        )
        out = await run(tools, "browser_click", ref="e2")
    assert not out.is_error
    assert not [n for n, _, _ in asks.asked if n.startswith("browser-point:")]
    [(name, arguments, preview)] = [
        q for q in asks.asked if q[0].startswith("browser-act:")
    ]
    on = 'a button "Continue" inside a box from accounts.example.org'
    assert (name, arguments["on"]) == ("browser-act:https://example.com", on)
    assert preview == f"click {on} on example.com in Safari"
    refusing = world(button("Post"), go)
    tools = await tools_of(refusing)
    with asking(Asks(deny=("browser-act:",))):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_click", ref="e2")
    assert out.is_error and "did not click or choose" in out.text
    assert not refusing.acts


async def test_the_controls_of_a_box_from_another_site_are_listed_and_clicked_by_the_browser(
    tmp_path: Path,
) -> None:
    driver, raw, log = await opened("", tmp_path, "https://example.com/box")
    try:
        boxed = [e for e in raw.elements if e.site]
        assert [(e.kind, e.label, e.sensitive, e.frame) for e in boxed] == [
            ("textbox", "Email", "", "x2_1_89"),
            ("textbox", "", "password", "x2_1_90"),
            ("textbox", "", "password", "x2_1_91"),  # no type named: counted as secret
            ("button", "Continue", "", "x2_1_92"),
        ]
        assert {e.site for e in boxed} == {"https://accounts.example.org"}
        assert raw.unreached == ()  # reached, so not named as a box melo can't reach
        go = boxed[-1]
        done = await driver.act(None, Act("click", raw.url, go, doc=raw.doc))
        assert done.via == "browser"
        assert native(log) == [
            {"node": "2_1_92", "purpose": "Click the checked element", "type": "click"}
        ]
        gone = dataclasses.replace(go, label="Other", frame="x2_1_99", fp="uid:2_1_99")
        with pytest.raises(StaleElementError):
            await driver.act(None, Act("click", raw.url, gone, doc=raw.doc))
    finally:
        await driver.close()
