"""A click at a point of the last picture: read leniently, checked in the page, asked where melo cannot check, clicked by Safari."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import NEVER_GRANTED, SELF_ASKING_TOOLS
from localmelo.support.browser.driver import (
    Act,
    BrowserError,
    Element,
    PointHit,
    StaleElementError,
    View,
)
from localmelo.support.browser.provider import PAGE_NOTE, TOOLS, parse_point
from localmelo.tests.browser.fakes import button, link, page
from localmelo.tests.browser.test_acting import HOME, OTHER, Asks, asking, world
from localmelo.tests.browser.test_local_addresses import CONSOLE, assert_refused
from localmelo.tests.browser.test_native_input import DAILY, native, opened
from localmelo.tests.browser.test_provider import run, tools_of


def marks(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    lines = log.read_text("utf-8").splitlines()
    return [json.loads(line[5:]) for line in lines if line.startswith("mark ")]


@pytest.mark.parametrize(
    ("x", "y", "rest", "want"),
    [
        (500, 300, {}, (500, 300)),
        ("500", "300", {}, (500, 300)),
        ("[500, 300]", None, {}, (500, 300)),
        (None, None, {"point": [500, 300]}, (500, 300)),
        (None, None, {"point_2d": [500, 300]}, (500, 300)),
        (None, None, {"point": {"x": 500, "y": 300}}, (500, 300)),
        (None, None, {"coordinate": "(500, 300"}, (500, 300)),  # cut off
        (0.5, 0.25, {}, (500, 250)),  # fractions of the picture
        (1, 1, {}, (1, 1)),
    ],
)
def test_a_point_is_read_however_the_model_wrote_it(
    x: Any, y: Any, rest: dict[str, Any], want: tuple[float, float]
) -> None:
    assert parse_point(x, y, rest) == want


@pytest.mark.parametrize(
    ("x", "y", "rest"),
    [(None, None, {}), ("left", "top", {}), (1500, 10, {}), (-5, 10, {})],
)
def test_a_point_that_cannot_be_read_says_how_to_write_it(
    x: Any, y: Any, rest: dict[str, Any]
) -> None:
    with pytest.raises(BrowserError, match="0 to 1000|0-1000|from 0 to 1000"):
        parse_point(x, y, rest)


def point_world(hit: PointHit, *more: Element) -> Any:
    driver = world(*more)
    driver.shot_view = View("", "d1", 1000.0, 500.0)
    driver.hits[(500, 150)] = hit  # (500, 300) of 1000 on a view 500 high
    return driver


async def pointed(driver: Any, asks: Asks | None = None, **where: Any) -> Any:
    tools = await tools_of(driver)
    with asking(asks or Asks()):
        await run(tools, "browser_open", url=HOME)
        shot = await run(tools, "browser_screenshot")
        assert shot.images
        return await run(
            tools, "browser_click_point", **(where or {"x": 500, "y": 300})
        )


async def test_a_point_on_a_drawing_is_clicked_and_looked_at_again() -> None:
    canvas = Element("button", "", at=-1, fp="canvas|1")
    driver = point_world(PointHit("canvas", canvas))
    out = await pointed(driver)
    assert not out.is_error
    head, said = out.text.split("\n")[:2]
    assert head.startswith("No effect: a point click at (500, 300) changed nothing")
    assert said == "Clicked the point (500, 300) (via point)."
    # What the point was on is page words: inside a block, never on the first lines.
    assert "the point was on: a drawing or picture" in out.text.split("<<page", 1)[1]
    assert driver.acts[0].point == (500, 150) and driver.acts[0].view is not None
    # It looks again: a new picture, with the point drawn on it.
    assert out.images and driver.marks[-1] == (driver.acts[0].view, 500, 150)


async def test_a_point_from_a_reading_needs_no_picture() -> None:
    canvas = Element("button", "", at=-1, fp="canvas|1")
    driver = point_world(PointHit("canvas", canvas))
    tools = await tools_of(driver)
    with asking(Asks()):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_click_point", x=500, y=300)
    # The reading's view is the one coordinate space: the point lands as on a picture.
    assert not out.is_error and driver.acts[0].point == (500, 150)


async def test_a_point_on_nothing_is_refused() -> None:
    driver = point_world(PointHit("none", tag="p", words=True))
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click_point", x=500, y=300)
    assert out.is_error and not driver.acts
    # What is there is told; a point on the 0-1000 scale is never doubted.
    assert "nothing to click at that point (what is there is text (p))" in out.text
    assert "pixel" not in out.text
    bare = point_world(PointHit("none"))
    tools = await tools_of(bare)
    await run(tools, "browser_open", url=HOME)
    await run(tools, "browser_screenshot")
    blank = await run(tools, "browser_click_point", x=500, y=300)
    assert "(it is text or background)" in blank.text


def test_a_number_beyond_the_scale_says_where_it_would_be_in_pixels() -> None:
    view = View("", "d1", 1280.0, 713.0)
    with pytest.raises(BrowserError) as fits:
        parse_point(1100, 534, {}, view)
    assert "As pixels of the 1280x713 picture, (1100, 534) is (859, 749)" in str(
        fits.value
    )
    with pytest.raises(BrowserError) as far:
        parse_point(1500, 534, {}, view)
    assert "pixel" in str(far.value) and "As pixels" not in str(far.value)


@pytest.mark.parametrize(
    ("hit", "said"),
    [
        (
            PointHit("control", Element("textbox", "", sensitive="password", at=2)),
            "click a password field",
        ),
        (
            PointHit(
                "control",
                Element("checkbox", "I agree to the terms", at=3),
                consent=True,
            ),
            'click a checkbox "I agree to the terms", which agrees to terms, consent or a sign-in',
        ),
        (
            PointHit(
                "frame",
                Element("button", "", fp="iframe|1"),
                site="https://accounts.example.org/x",
            ),
            "click inside a box from accounts.example.org that melo cannot check",
        ),
    ],
)
async def test_a_point_on_a_secret_consent_or_a_box_from_elsewhere_asks_every_time(
    hit: PointHit, said: str
) -> None:
    asks = Asks()
    driver = point_world(hit)
    out = await pointed(driver, asks)
    assert not out.is_error
    asked = [(n, p) for n, _, p in asks.asked if n.startswith("browser-point:")]
    assert asked == [
        ("browser-point:https://example.com", f"{said} on example.com in Safari")
    ]
    assert any(n.startswith("browser-point:") for n in NEVER_GRANTED)
    # Refused: nothing is clicked.
    refusing = point_world(hit)
    out = await pointed(refusing, Asks(deny=("browser-point:",)))
    assert out.is_error and "did not click there" in out.text and not refusing.acts


async def test_a_point_in_a_box_melo_cannot_tell_says_its_target_is_unknown() -> None:
    """Auto mode asks the person about it, as nobody can judge it on the goal (#465)."""
    box = PointHit(
        "frame",
        Element("button", "", fp="iframe|1"),
        site="https://login.example.net/x",
    )
    asks = Asks()
    out = await pointed(point_world(box), asks)
    assert not out.is_error
    [(_, arguments, _)] = [q for q in asks.asked if q[0].startswith("browser-point:")]
    assert arguments["target"] == "unknown"


async def test_a_point_in_a_box_melo_knows_by_its_title_is_clicked_like_any_other() -> (
    None
):
    """The box's title says what it is: the click's own question names it (#465)."""
    box = PointHit(
        "frame",
        Element("button", "Quick login", fp="iframe|1"),
        site="https://login.example.net/x",
    )
    asks = Asks()
    driver = point_world(box)
    out = await pointed(driver, asks)
    assert not out.is_error and driver.acts
    assert not [n for n in asks.names if n.startswith("browser-point:")]
    [(name, arguments, preview)] = [
        q for q in asks.asked if q[0].startswith("browser-act:")
    ]
    on = 'inside a box "Quick login" from login.example.net'
    assert (name, arguments["on"]) == ("browser-act:https://example.com", on)
    assert preview == f"click {on} on example.com in Safari"
    # A refusal of the click leaves the box alone.
    refusing = point_world(box)
    out = await pointed(refusing, Asks(deny=("browser-act:",)))
    assert out.is_error and not refusing.acts


async def test_a_point_in_a_box_from_elsewhere_never_says_nothing_changed() -> None:
    """melo cannot see into that box: what the click did shows in the new picture only."""
    box = PointHit(
        "frame",
        Element("button", "", fp="iframe|1"),
        site="https://accounts.example.org/x",
    )
    driver = point_world(box)
    out = await pointed(driver, Asks())
    assert not out.is_error and out.images
    first = out.text.split("\n", 1)[0]
    assert "Nothing on the page changed" not in first
    assert "inside the box shows only in the new picture" in first


async def test_a_point_on_a_link_or_a_send_keeps_the_link_and_send_checks() -> None:
    away = PointHit("control", link("Away", OTHER + "x"))
    asks = Asks()
    driver = point_world(away)
    driver.pages[OTHER + "x"] = page(OTHER + "x", "There.")
    out = await pointed(driver, asks)
    assert not out.is_error and "browser:https://other.org" in asks.names
    sending = PointHit(
        "control", button("Pay", submit="https://example.com/pay", form=(("a", "1"),))
    )
    asks = Asks(deny=("browser-submit:",))
    driver = point_world(sending)
    out = await pointed(driver, asks)
    assert out.is_error and "did not send the form" in out.text and not driver.acts
    local = PointHit("control", link("Console", CONSOLE))
    driver = point_world(local)
    assert_refused(await pointed(driver))
    assert not driver.acts


async def test_the_point_tool_asks_for_itself_and_the_note_names_the_order() -> None:
    assert "browser_click_point" in SELF_ASKING_TOOLS
    tool = next(t for t in TOOLS if t.name == "browser_click_point")
    assert tool.parameters["required"] == ["x", "y"]
    assert tool.annotations == {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    }
    assert "the same in a picture, a reading and a step's result" in PAGE_NOTE


async def test_a_point_on_a_label_clicks_its_field_where_the_field_is() -> None:
    box = Element("checkbox", "Remember me", at=4)
    driver = point_world(PointHit("control", box, at=(120.0, 160.0)))
    out = await pointed(driver)
    assert not out.is_error and driver.acts[0].point == (120.0, 160.0)


async def test_a_point_on_a_drawing_is_clicked_by_the_browser_at_that_point(
    tmp_path: Path,
) -> None:
    driver, raw, log = await opened("", tmp_path)
    try:
        shot = await driver.screenshot(None)
        assert shot.view == View(DAILY, "doc-1", 1000, 500, 0, 0)
        hit = await driver.hit(None, shot.view, 300, 100)
        assert hit.surface == "canvas" and hit.element is not None
        done = await driver.act(
            None,
            Act(
                "click",
                shot.view.url,
                hit.element,
                doc=shot.view.doc,
                point=(300, 100),
                view=shot.view,
            ),
        )
        assert done.via == "point"
        assert native(log) == [
            {
                "point": {"x": 300, "y": 100},
                "purpose": "Click the checked point",
                "type": "click",
            }
        ]
        # The next picture carries the point: a ring there, taken away after.
        await driver.screenshot(None, (shot.view, 300, 100))
        drawn = marks(log)
        assert [m["on"] for m in drawn] == [True, False]
        assert drawn[0]["x"] == 300 and drawn[0]["id"].startswith("melo-ring-")
    finally:
        await driver.close()


async def test_nothing_at_a_point_and_a_box_from_another_site_are_named(
    tmp_path: Path,
) -> None:
    driver, raw, _ = await opened("", tmp_path)
    try:
        shot = await driver.screenshot(None)
        assert shot.view is not None
        assert (await driver.hit(None, shot.view, 10, 10)).surface == "none"
        box = await driver.hit(None, shot.view, 500, 250)
        assert box.surface == "frame" and box.site == "https://accounts.example.org/box"
        moved = View("https://example.com/news", "doc-1", 1000, 500)
        with pytest.raises(StaleElementError):
            await driver.hit(None, moved, 300, 100)
    finally:
        await driver.close()
