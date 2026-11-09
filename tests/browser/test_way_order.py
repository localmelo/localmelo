"""The ways in order: every element click tries the browser's own input first, and one miss does not stick to the site."""

from __future__ import annotations

from localmelo.tests.browser.fakes import button
from localmelo.tests.browser.test_acting import HOME, world
from localmelo.tests.browser.test_provider import run, tools_of


async def test_a_miss_of_the_browsers_input_on_one_element_does_not_stick_to_the_site() -> (
    None
):
    driver = world(button("Happy"), button("Sad"))
    driver.refuse_native = True
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    first = await run(tools, "browser_click", ref="e1")
    assert first.text.startswith("No change seen on the page in 2 s after a script")
    assert "the browser's own input did not reach it" in first.text
    # The next click on the site is tried with the browser's own input again.
    second = await run(tools, "browser_click", ref="e2")
    assert "the browser's own input did not reach it" in second.text
    assert "earlier in this chat" not in second.text
