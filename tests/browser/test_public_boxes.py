"""A form is not sent with a box ticked that shares or publishes what the person does (#547).

A page may tick "make my activity public" or "share to social media" before
anyone looks. The send is refused with the box named, melo unticks it, and the
form goes. Only the person's own words that speak of sharing leave it ticked; the
page's words are never read for that.
"""

from __future__ import annotations

import pytest

from localmelo.support.browser.driver import Element
from localmelo.tests.browser.fakes import button
from localmelo.tests.browser.test_acting import DONE, HOME, Asks, acts, asking, world
from localmelo.tests.browser.test_provider import run, tools_of

FORM = (("topic", "Router"), ("make_public", "on"))
ELSEWHERE = (("search", "cats"),)


class Reads(Asks):
    """The run's policy when the reviewer reads the person's words: do they speak of sharing?"""

    def __init__(self, shares: bool | None) -> None:
        super().__init__()
        self.shares = shares

    def asks_to_share(self) -> bool | None:
        return self.shares


def box(
    label: str, *, checked: bool = True, form: tuple[tuple[str, str], ...] = FORM
) -> Element:
    return Element("checkbox", label, checked=checked, form=form)


def post(form: tuple[tuple[str, str], ...] = FORM) -> Element:
    return button("Post", submit=DONE, form=form)


ROWS = {
    "a public box the page ticked": (
        box("Make my activity public"),
        Reads(False),
        True,
    ),
    "a share box the page ticked": (box("Share to social media"), Reads(False), True),
    "a Chinese label": (box("公开我的动态"), Reads(False), True),
    "the person asked to share": (box("Make my activity public"), Reads(True), False),
    "the box is unticked": (
        box("Make my activity public", checked=False),
        Reads(False),
        False,
    ),
    "another box": (box("Remember me"), Reads(False), False),
    "a box of another form": (
        box("Make my activity public", form=ELSEWHERE),
        Reads(False),
        False,
    ),
    "nothing reads the person's words": (box("Make my activity public"), Asks(), False),
}


@pytest.mark.parametrize("row", list(ROWS), ids=list(ROWS))
async def test_a_form_is_not_sent_with_a_public_box_ticked_the_person_did_not_ask_for(
    row: str,
) -> None:
    ticked, asks, refused = ROWS[row]
    driver = world(ticked, post())
    tools = await tools_of(driver)
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_click", ref="e2")
    sent = any(name.startswith("browser-submit:") for name in asks.names)
    assert out.is_error is refused, out.text
    assert sent is not refused
    if refused:
        # The box is named, so melo can untick it; nothing was clicked or sent.
        assert 'e1 "' in out.text and "untick" in out.text
        assert not acts(driver)
