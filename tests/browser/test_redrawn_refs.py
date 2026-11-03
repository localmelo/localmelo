"""A ref the page redrew: found again only when one element is like it and belongs where it did.

Like means the same kind, words, destination, form and field; where it
belongs is its row or the heading above it and the text next to it
(``Element.context``, which the page script reads). A control wins over a
look-alike.
"""

from __future__ import annotations

import dataclasses

import pytest

from localmelo.support.browser.driver import Element, FieldAttrs
from localmelo.tests.browser.fakes import button, link, page
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world
from localmelo.tests.browser.test_provider import run, tools_of

MOODS = '["button", "", "", "", "", "h:Pick a mood", ""]'


def at(label: str, where: str = MOODS, **more: object) -> Element:
    """A button that belongs somewhere (its row, heading or nearby text)."""
    return button(label, context=where, **more)  # type: ignore[arg-type]


def row(name: str) -> str:
    return f'["button", "", "", "", "", "r:{name} Delete", "{name}"]'


async def test_a_redrawn_ref_is_used_when_one_element_is_like_it() -> None:
    asks = Asks()
    driver = world(at("Happy"), at("Check in"))
    driver.redraw["Happy"] = page(
        HOME, "Welcome.", at("Sad"), at("Happy"), at("Check in")
    )
    tools = await tools_of(driver)
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_click", ref="e1")
    assert not out.is_error
    assert out.text.startswith(
        "Clicked e4 (via browser). The page redrew e1; e4 is the one element like it now."
    )
    assert [a.element.label for a in driver.acts] == ["Happy"]
    # The site was asked about once: the act on the redrawn element needs no new answer.
    assert asks.names.count("browser-act:https://example.com") == 1


async def test_a_redrawn_ref_with_two_like_it_stays_stale() -> None:
    driver = world(at("Delete"))
    driver.redraw["Delete"] = page(HOME, "List.", at("Delete"), at("Delete"))
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "no longer on the page" in out.text
    assert not driver.acts


async def test_a_control_wins_over_a_look_alike_when_names_collide() -> None:
    driver = world(at("Submit"))
    driver.redraw["Submit"] = page(
        HOME,
        "Form.",
        dataclasses.replace(at("Submit"), plain=True),
        at("Submit"),
    )
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert not out.is_error and out.text.startswith("Clicked e3 (via browser).")
    assert driver.acts[0].element.plain is False


@pytest.mark.parametrize(
    "redrawn",
    [
        at("Pay", submit="https://example.com/elsewhere"),  # sends elsewhere now
        dataclasses.replace(link("Pay", "https://example.com/other"), context=MOODS),
        at("Pay", submit="https://example.com/pay", form=(("amount", "900"),)),
    ],
)
async def test_a_redrawn_element_that_sends_or_leads_elsewhere_is_never_used(
    redrawn: Element,
) -> None:
    item = at("Pay", submit="https://example.com/pay", form=(("amount", "9"),))
    driver = world(item)
    driver.redraw["Pay"] = page(HOME, "Pay.", redrawn)
    tools = await tools_of(driver)
    with asking(Asks()):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and not driver.acts


async def test_a_redrawn_text_box_must_be_the_same_field() -> None:
    box = Element(
        "textbox", "Name", attrs=FieldAttrs(type="text", words="name"), context=MOODS
    )
    driver = world(box)
    changed = dataclasses.replace(box, attrs=FieldAttrs(type="text", words="pin"))
    driver.redraw["Name"] = page(HOME, "Form.", changed)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_type", ref="e1", text="Alex")
    assert out.is_error and not driver.acts


async def test_the_delete_of_another_record_is_never_taken_for_the_one_that_went() -> (
    None
):
    """Record A's row is gone when melo acts: record B's "Delete" is left, and stays untouched."""
    form = (("id", "(hidden)"),)
    a = at("Delete", row("Record A"), submit=HOME + "delete", form=form)
    b = at("Delete", row("Record B"), submit=HOME + "delete", form=form)
    for before in ((a, b), (a,)):  # with B listed too, and with A alone on the page
        driver = world(*before)
        driver.redraw["Delete"] = page(HOME, "Records.", b)
        tools = await tools_of(driver)
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            out = await run(tools, "browser_click", ref="e1")
        assert out.is_error and "no longer on the page" in out.text, before
        assert not driver.acts, before


async def test_a_redrawn_element_that_belongs_nowhere_stays_stale() -> None:
    driver = world(button("Happy"))  # no row, heading or text next to it
    driver.redraw["Happy"] = page(HOME, "Welcome.", button("Happy"))
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and not driver.acts


async def test_a_redrawn_element_under_another_heading_stays_stale() -> None:
    driver = world(at("Archive"))
    elsewhere = MOODS.replace("Pick a mood", "Shared with me")
    driver.redraw["Archive"] = page(HOME, "Welcome.", at("Archive", elsewhere))
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and not driver.acts
