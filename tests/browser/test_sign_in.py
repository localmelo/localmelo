"""The browser types the password the person gave, and only that one, only there (#547).

The run's policy (``TOOL_ASK``) says whether a handle is a password the person
gave in their newest message and whether their words name the site; the
browser tool asks it and types the value it gets. The value is typed into the
page and shown nowhere: not in a question, not in a step result.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.support.browser.driver import Element
from localmelo.tests.browser.fakes import button, page
from localmelo.tests.browser.test_acting import (
    HOME,
    Anywhere,
    Asks,
    asking,
    textbox,
    world,
)
from localmelo.tests.browser.test_provider import run, tools_of
from localmelo.tests.browser.test_takeover import LOGIN, browser, started

VALUE = "hunter2-Secret"


class Gave(Asks):
    """The run's policy when the person gave a password: what the reviewer answers."""

    def __init__(
        self,
        *,
        sites: tuple[str, ...] = ("example.com",),
        live: bool = True,
        **more: Any,
    ) -> None:
        super().__init__(**more)
        self.sites, self.live = sites, live

    def sign_in_value(self, handle: str, site: str) -> str | None:
        ok = self.live and handle == "[password-1]" and site in self.sites
        return VALUE if ok else None

    def signs_in(self, site: str) -> bool:
        return self.live and site in self.sites


def password_box(**more: Any) -> Element:
    return Element("textbox", "Password", sensitive="password", **more)


async def test_the_password_the_person_gave_is_typed_into_a_password_box_and_shown_nowhere() -> (
    None
):
    driver = world(textbox("Email"), password_box(), button("Check in"))
    tools = await tools_of(driver)
    asks = Gave()
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_type", ref="e2", text="[password-1]")
    assert not out.is_error, out.text
    assert [act.text for act in driver.acts] == [VALUE]
    assert "Typed the password the user gave" in out.text and VALUE not in out.text
    assert VALUE not in repr(asks.asked), "no question carries the value"
    assert "[password-1]" in repr(asks.asked), "the question shows the handle"


REFUSED: dict[str, tuple[Element, str, Gave | Asks | None]] = {
    "the value itself, not a handle": (password_box(), VALUE, Gave()),
    "a handle the person never gave": (password_box(), "[password-2]", Gave()),
    "a site the person did not name": (password_box(), "[password-1]", Gave(sites=())),
    "an earlier message's handle": (password_box(), "[password-1]", Gave(live=False)),
    "a run that cannot say": (password_box(), "[password-1]", Asks()),
    "no run at all": (password_box(), "[password-1]", None),
    "a one-time code box": (
        Element("textbox", "Code", sensitive="code"),
        "[password-1]",
        Gave(),
    ),
    # The page is the named site's, the box in it is another site's frame.
    "a box in another site's frame": (
        password_box(site="https://accounts.example.org/x"),
        "[password-1]",
        Gave(),
    ),
    "a box whose site cannot be read": (
        password_box(site="not a site"),
        "[password-1]",
        Gave(),
    ),
}


@pytest.mark.parametrize("case", list(REFUSED))
async def test_nothing_else_is_typed_into_a_secret_box(case: str) -> None:
    element, text, policy = REFUSED[case]
    driver = world(element, button("Check in"))
    tools = await tools_of(driver)
    with asking(policy):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_type", ref="e1", text=text)
    assert out.is_error and VALUE not in out.text
    assert not driver.acts, case


async def test_a_handle_goes_only_into_a_password_box_and_only_on_https() -> None:
    plain = world(textbox("Search"), button("Check in"))
    tools = await tools_of(plain)
    with asking(Gave()):
        await run(tools, "browser_open", url=HOME)
        out = await run(tools, "browser_type", ref="e1", text="[password-1]")
    assert out.is_error and "goes only into a password box" in out.text
    assert not plain.acts

    http = Anywhere(
        page("http://example.com/", "Welcome.", password_box(), title="Home")
    )
    tools = await tools_of(http)
    with asking(Gave()):
        await run(tools, "browser_open", url="http://example.com/")
        out = await run(tools, "browser_type", ref="e1", text="[password-1]")
    assert out.is_error and not http.acts


async def test_melo_signs_in_before_it_offers_the_page_and_offers_it_when_that_fails() -> (
    None
):
    driver = browser()
    provider, tools, _ = await started(driver)
    # Every card is refused: the first read of the sign-in page asks nobody
    # when the person gave a password; the next read, after melo tried, does.
    asks = Gave(deny=("takeover:",))
    with asking(asks):
        first = await run(tools, "browser_open", url=LOGIN)
        assert not [n for n in asks.names if n.startswith("takeover:")]
        assert "did not take" not in first.text
        await run(tools, "browser_read")
    assert [n for n in asks.names if n.startswith("takeover:")] == [
        "takeover:https://example.com"
    ]
    await provider.close()
    # Without a password given, the page is offered at once, as before.
    provider, tools, _ = await started(browser())
    other = Asks(deny=("takeover:",))
    with asking(other):
        await run(tools, "browser_open", url=LOGIN)
    assert [n for n in other.names if n.startswith("takeover:")]
    await provider.close()
