"""The ways in order: a site where Safari's own input did not work starts with the page script."""

from __future__ import annotations

from localmelo.tests.browser.fakes import button, page
from localmelo.tests.browser.test_acting import HOME, OTHER, Asks, asking, world
from localmelo.tests.browser.test_provider import run, tools_of


async def test_a_site_where_the_browsers_input_failed_starts_with_the_page_script() -> (
    None
):
    driver = world(button("Happy"), button("Sad"))
    driver.refuse_native = True
    tools = await tools_of(driver)
    await run(tools, "browser_open", url=HOME)
    first = await run(tools, "browser_click", ref="e1")
    assert first.text.startswith(
        "Clicked e1 (via page-script: the browser's own input did not reach it)."
    )
    second = await run(tools, "browser_click", ref="e4")
    assert driver.acts[1].native is False
    assert second.text.startswith(
        "Clicked e4 (via page-script: the browser's own input did not reach this "
        "site's elements earlier in this chat)."
    )
    # Another site still starts with the browser's own input.
    driver.pages[OTHER] = page(OTHER, "Other.", button("Go"))
    with asking(Asks()):
        await run(tools, "browser_open", url=OTHER)
        await run(tools, "browser_click", ref="e7")
    assert driver.acts[2].native is True
