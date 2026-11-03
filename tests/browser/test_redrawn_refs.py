"""A ref the page redrew: found again only when one element is like it, a control before a look-alike."""

from __future__ import annotations

import dataclasses

import pytest

from localmelo.support.browser.driver import Element, FieldAttrs
from localmelo.tests.browser.fakes import button, link, page
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world
from localmelo.tests.browser.test_provider import run, tools_of


async def test_a_redrawn_ref_is_used_when_one_element_is_like_it() -> None:
    asks = Asks()
    driver = world(button("Happy"), button("Check in"))
    driver.redraw["Happy"] = page(
        HOME, "Welcome.", button("Sad"), button("Happy"), button("Check in")
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
    driver = world(button("Delete"))
    driver.redraw["Delete"] = page(HOME, "List.", button("Delete"), button("Delete"))
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "no longer on the page" in out.text
    assert not driver.acts


async def test_a_control_wins_over_a_look_alike_when_names_collide() -> None:
    driver = world(button("Submit"))
    driver.redraw["Submit"] = page(
        HOME,
        "Form.",
        dataclasses.replace(button("Submit"), plain=True),
        button("Submit"),
    )
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert not out.is_error and out.text.startswith("Clicked e3 (via browser).")
    assert driver.acts[0].element.plain is False


@pytest.mark.parametrize(
    "redrawn",
    [
        button("Pay", submit="https://example.com/elsewhere"),  # sends elsewhere now
        link("Pay", "https://example.com/other"),  # another kind
        button("Pay", submit="https://example.com/pay", form=(("amount", "900"),)),
    ],
)
async def test_a_redrawn_element_that_sends_or_leads_elsewhere_is_never_used(
    redrawn: Element,
) -> None:
    item = button("Pay", submit="https://example.com/pay", form=(("amount", "9"),))
    driver = world(item)
    driver.redraw["Pay"] = page(HOME, "Pay.", redrawn)
    tools = await tools_of(driver)
    with asking(Asks()):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and not driver.acts


async def test_a_redrawn_text_box_must_be_the_same_field() -> None:
    box = Element("textbox", "Name", attrs=FieldAttrs(type="text", words="name"))
    driver = world(box)
    changed = dataclasses.replace(box, attrs=FieldAttrs(type="text", words="pin"))
    driver.redraw["Name"] = page(HOME, "Form.", changed)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_type", ref="e1", text="Alex")
    assert out.is_error and not driver.acts
