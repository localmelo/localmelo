"""A ref the page redrew: found again only when one element is like it and belongs where it did.

Like means the same kind, words, destination, form and field; where it
belongs is its row or the heading above it and the text next to it
(``Element.context``, which the page script reads). A control wins over a
look-alike.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.browser.driver import Element, FieldAttrs, StaleElementError
from localmelo.support.browser.pagescript import PAGE_JS, _element
from localmelo.tests.browser.fakes import FakeDriver, button, link, page
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


def reload(driver: FakeDriver, shown: Any = None) -> None:
    """The tab's page loads again at the same address: another document (``d2``)."""
    driver.docs[driver.current or ""] = "d2"
    driver.pages[HOME] = dataclasses.replace(shown or driver.pages[HOME], doc="d2")


async def test_a_ref_from_before_a_reload_is_used_when_one_element_is_like_it() -> None:
    """The page loaded again at the same address: the one button like the ref, where it was, is used, and the result says so."""
    asks = Asks()
    driver = world(at("Happy"), at("Check in"))
    tools = await tools_of(driver)
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        reload(driver)
        out = await run(tools, "browser_click", ref="e1")
    assert not out.is_error
    assert out.text.startswith(
        "Clicked e3 (via browser). The page reloaded since e1 was read; e3 is the "
        "one element like it now."
    )
    assert (
        "- the page reloaded since e1 was read; e3 is the one button 'Happy' on it now"
        in out.text
    )
    assert [a.element.label for a in driver.acts] == ["Happy"]
    assert driver.acts[0].doc == "d2"


async def test_a_reload_that_shows_another_record_never_gets_the_records_action() -> (
    None
):
    """Record A's "Delete" (a script button, no form): the reload shows record B, with the same button."""
    driver = world(at("Delete", where=row("Record A")))
    tools = await tools_of(driver)
    with asking(Asks()):
        await run(tools, "browser_open", url=HOME)
        reload(driver, page(HOME, "Records.", at("Delete", where=row("Record B"))))
        out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "reloaded or replaced" in out.text and not driver.acts


@pytest.mark.parametrize("after", ["two like it", "a form send", "nothing says where"])
async def test_a_ref_from_before_a_reload_otherwise_stays_stale(after: str) -> None:
    item = {
        "two like it": at("Happy"),
        "a form send": at("Happy", submit=HOME + "mood"),
        "nothing says where": button("Happy"),
    }[after]
    driver = world(item)
    tools = await tools_of(driver)
    with asking(Asks()):
        await run(tools, "browser_open", url=HOME)
        if after == "two like it":
            reload(driver, page(HOME, "Moods.", at("Happy"), at("Happy")))
        else:
            reload(driver)  # a send was not asked about; a context-less one is unknown
        out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "reloaded or replaced" in out.text and not driver.acts


async def test_enter_in_a_box_is_never_carried_over_to_a_reloaded_page() -> None:
    """Enter outside a form sends what the page decides: after a reload it stays stale."""
    box = Element(
        "textbox", "Line", attrs=FieldAttrs(type="text", words="line"), context=MOODS
    )
    driver = world(box)
    tools = await tools_of(driver)
    with asking(Asks()):
        await run(tools, "browser_open", url=HOME)
        reload(driver)
        out = await run(tools, "browser_type", ref="e1", text="hello", submit=True)
    assert out.is_error and "reloaded or replaced" in out.text and not driver.acts


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


JS = Path(__file__).with_name("js")
needs_jsdom = pytest.mark.skipif(
    shutil.which("node") is None or not (JS / "node_modules" / "jsdom").is_dir(),
    reason="needs node and jsdom (npm ci --prefix tests/browser/js)",
)


def read_in_js(*pages: str) -> list[tuple[Element, ...]]:
    """The elements melo's page script reads on each page, run in jsdom."""
    ran = subprocess.run(
        ["node", str(JS / "read_pages.mjs")],
        input=json.dumps({"script": PAGE_JS, "pages": list(pages)}),
        cwd=JS,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 0, ran.stderr
    return [
        tuple(e for e in map(_element, reading["elements"]) if e)
        for reading in json.loads(ran.stdout)
    ]


def card(title: str, record: str) -> str:
    """A record's card: its title is a link, its Share sends a hidden id to one address."""
    return (
        f'<div class="card"><a href="/records/{record}">{title}</a>'
        '<form action="/share" method="post">'
        f'<input type="hidden" name="id" value="{record}"><button>Share</button>'
        "</form></div>"
    )


def records(*cards: str) -> str:
    return "<main><h2>Shared with me</h2>" + "".join(cards) + "</main>"


@needs_jsdom
async def test_a_card_whose_linked_title_changed_is_another_record() -> None:
    """Record A goes and record B is drawn in its place, in the same document."""
    a, b, a_again = read_in_js(
        records(card("Trip notes", "a")),
        records(card("Tax papers", "b")),
        records(card("Trip notes", "a")),
    )
    # The link titles are what tells the two apart: same address, same hidden field.
    assert a[1].submit == b[1].submit and a[1].form == b[1].form
    assert a[1].context != b[1].context
    for redrawn, recovered in ((b, False), (a_again, True)):
        driver = world(*a)
        driver.redraw["Share"] = page(HOME, "Records.", *redrawn)
        tools = await tools_of(driver)
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            out = await run(tools, "browser_click", ref="e2")
        if recovered:  # the same record, redrawn: found again
            assert not out.is_error and "is the one element like it now" in out.text
            assert [x.element.label for x in driver.acts] == ["Share"]
        else:
            assert out.is_error and not driver.acts, out.text


async def test_a_ref_is_never_carried_over_to_a_reloaded_page() -> None:
    """A reload at the same address is another document: stale, even when the one there looks the same."""
    first = at("Share", submit=HOME + "share")
    driver = world(first)
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    driver.docs["t1"] = "d2"  # reloaded, and now shows another record's card
    reloaded = page(HOME, "Records.", at("Share", submit=HOME + "share"))
    driver.pages[HOME] = dataclasses.replace(reloaded, doc="d2")
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and "reloaded" in out.text and not driver.acts


class ReloadsWhenRead(FakeDriver):
    """The element is gone as it was read, and by the time melo reads again the page reloaded."""

    async def act(self, tab: str | None, act: Any) -> Any:
        try:
            return await super().act(tab, act)
        except StaleElementError:
            self.docs["t1"] = "d2"
            raise


async def test_a_redraw_whose_new_reading_is_another_document_stays_stale() -> None:
    driver = ReloadsWhenRead(page(HOME, "Records.", at("Share")))
    driver.redraw["Share"] = page(HOME, "Records.", at("Share"))
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    out = await run(tools, "browser_click", ref="e1")
    assert out.is_error and not driver.acts
