"""A search is no message: the browser says when a box, or every box of a form, is a search box (#547).

The question about typing or sending carries ``field: "search"`` then, and the
reviewer reads it only with the person's own words asking to search. A form
with one box that is no search box is a message like any other.
"""

from __future__ import annotations

import pytest

from localmelo.support.browser.driver import Element, FieldAttrs
from localmelo.tests.browser.fakes import button
from localmelo.tests.browser.test_acting import DONE, HOME, Asks, asking, world
from localmelo.tests.browser.test_provider import run, tools_of

FORM = (("q", ""), ("note", ""))


def box(label: str, words: str = "", kind: str = "", form: tuple = ()) -> Element:
    return Element(
        "textbox", label, attrs=FieldAttrs(type=kind, words=words), form=form
    )


ROWS = {
    "type=search": (box("Find", kind="search"), True),
    "search in its name": (box("Find", words="twotabsearchtextbox"), True),
    "search in its placeholder": (box("Find", words="Search the shop"), True),
    "a box in no form that says nothing": (box("Comment", words="comment"), False),
    "research is no search": (box("Notes", words="research notes"), False),
    "a search box beside a comment box": (
        box("Find", kind="search", form=FORM),
        False,
    ),
    "a search form of one box": (
        box("Find", kind="search", form=(("q", ""),)),
        True,
    ),
}


@pytest.mark.parametrize("row", list(ROWS), ids=list(ROWS))
async def test_typing_says_search_for_a_search_box_only(row: str) -> None:
    field, says = ROWS[row]
    if field.form == FORM:
        # The other box of the form is a plain one.
        driver = world(field, box("Comment", words="comment", form=FORM))
    else:
        driver = world(field)
    asks = Asks()
    tools = await tools_of(driver)
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        await run(tools, "browser_type", ref="e1", text="cheap flights to Denver")
    typed = [a for name, a, _ in asks.asked if name.startswith("browser-type:")]
    assert len(typed) == 1
    assert (typed[0].get("field") == "search") is says


async def test_a_form_of_search_boxes_is_sent_as_a_search_and_one_with_a_comment_box_is_not() -> (
    None
):
    only = (("q", ""),)
    both = (("q", ""), ("note", ""))
    for form, boxes, says in (
        (only, [box("Find", kind="search", form=only)], True),
        (
            both,
            [box("Find", kind="search", form=both), box("Note", form=both)],
            False,
        ),
    ):
        driver = world(*boxes, button("Go", submit=DONE, form=form))
        asks = Asks()
        tools = await tools_of(driver)
        with asking(asks):
            await run(tools, "browser_open", url=HOME)
            await run(tools, "browser_click", ref=f"e{len(boxes) + 1}")
        sent = [a for name, a, _ in asks.asked if name.startswith("browser-submit:")]
        assert len(sent) == 1
        assert (sent[0].get("field") == "search") is says
