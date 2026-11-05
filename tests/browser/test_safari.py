"""The safari-automation driver against a fake ``safaridriver --mcp`` (real subprocess)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.browser.driver import (
    Act,
    BrowserError,
    BrowserSetupError,
    Element,
    StaleElementError,
)
from localmelo.support.browser.pagescript import same_page
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.safari import (
    SETUP_MESSAGE,
    TOOL_NAMES,
    SafariAutomationDriver,
)
from localmelo.support.mcp import StdioMcpClient

FAKE = str(Path(__file__).with_name("fake_safari_server.py"))


def driver_for(mode: str = "", **env: str) -> SafariAutomationDriver:
    def factory(config: Any) -> StdioMcpClient:
        config.env = {"FAKE_SAFARI_MODE": mode, **env}
        return StdioMcpClient(config, shutdown_grace_s=0.5)

    return SafariAutomationDriver(
        command=sys.executable,
        args=("-I", FAKE),
        client_factory=factory,  # type: ignore[arg-type]
        startup_timeout_s=10,
        call_timeout_s=10,
    )


@pytest.fixture
async def driver() -> Any:
    safari = driver_for()
    await safari.start()
    yield safari
    await safari.close()


async def test_the_handshake_lists_tools_and_a_page_is_read(
    driver: SafariAutomationDriver,
) -> None:
    assert set(TOOL_NAMES.values()) <= set(driver.tool_names)
    info = await driver.open("https://example.com/", new_tab=True)
    assert (info.url, info.title) == ("https://example.com/", "Example Home")
    raw = await driver.page(None)
    assert raw.title == "Example Home" and "中文。" in raw.text
    # An unknown kind is dropped; the password field is marked and has no value.
    assert [(e.kind, e.label, e.sensitive) for e in raw.elements] == [
        ("link", "News", ""),
        ("textbox", "Search", ""),
        ("textbox", "", "password"),
        ("button", "Go", ""),
    ]
    assert raw.elements[0].href == "https://example.com/news"


async def test_tabs_new_switch_back_close_and_blank(
    driver: SafariAutomationDriver,
) -> None:
    first = await driver.open("https://example.com/", new_tab=True)
    second = await driver.open("https://example.com/news", new_tab=True)
    assert second.id != first.id and (await driver.page(None)).title == "News"
    # The window starts with a tab melo did not open; it is never taken for melo's.
    assert [t.id for t in await driver.tabs()] == ["page-1", first.id, second.id]
    assert first.id != "page-1" and second.id != "page-1"
    assert (await driver.switch(first.id)).url == "https://example.com/"
    await driver.open("https://example.com/news", new_tab=False)
    assert (await driver.back(None)).url == "https://example.com/"
    await driver.blank(None)
    assert (await driver.page(None)).url == "about:blank"
    await driver.close_tab(second.id)
    assert [t.id for t in await driver.tabs()] == ["page-1", first.id]


async def test_wait_polls_until_the_page_has_the_text(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/", new_tab=True)
    assert await driver.wait(None, "text", "Welcome", 2)
    assert await driver.wait(None, "gone", "Nowhere", 2)
    assert not await driver.wait(None, "text", "Nowhere", 0.5)
    assert await driver.wait(None, "load", "", 2)


@pytest.mark.parametrize("until", ["text", "gone"])
async def test_a_page_with_no_body_yet_is_not_loaded_and_the_wait_goes_on(
    until: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bot check between two documents has no body: the wait keeps looking, it does not fail."""
    monkeypatch.setattr("localmelo.support.browser.pagescript.POLL_S", 0.01)
    safari = driver_for("no_body")
    await safari.start()
    try:
        await safari.open("https://example.com/", new_tab=True)
        word = "Welcome" if until == "text" else "Nowhere"
        assert await safari.wait(None, until, word, 5)
    finally:
        await safari.close()


async def test_where_a_tab_is_is_read_in_the_tab_and_a_wait_holds_to_its_site(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/", new_tab=True)
    assert await driver.location(None) == "https://example.com/"
    assert await driver.wait(None, "text", "Welcome", 2, origin="https://example.com")
    assert await driver.wait(None, "load", "", 2, origin="https://example.com")
    # On another site the condition never counts, whatever the page says.
    assert not await driver.wait(None, "text", "Welcome", 0.5, origin="http://a.net")
    assert not await driver.wait(None, "load", "", 0.5, origin="http://a.net")
    # ``holds`` is one look at the same condition, with no waiting.
    assert await driver.holds(None, "load", "")
    assert await driver.holds(None, "text", "Welcome", origin="https://example.com")
    assert not await driver.holds(None, "text", "Nowhere")
    assert await driver.holds(None, "gone", "Nowhere")
    assert not await driver.holds(None, "load", "", origin="http://a.net")


async def test_slow_loading_pages_are_waited_for() -> None:
    slow = driver_for("slow_load")
    try:
        await slow.start()
        info = await slow.open("https://example.com/", new_tab=True)
        assert info.title == "Example Home"
    finally:
        await slow.close()


async def test_a_screenshot_is_the_png_safari_saved_where_melo_asked(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/", new_tab=True)
    shot = await driver.screenshot(None)
    assert shot.mime_type == "image/png" and shot.data.startswith("iVBORw0KGgo")
    folder = driver._shots
    assert folder is not None and os.listdir(folder) == []  # the file is gone
    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700
    await driver.close()
    assert not os.path.exists(folder) and driver._shots is None  # and so is the folder


async def test_an_image_part_is_still_used_when_a_server_sends_one() -> None:
    inline = driver_for("inline_image")
    try:
        await inline.start()
        await inline.open("https://example.com/", new_tab=True)
        assert (await inline.screenshot(None)).data.startswith("iVBORw0KGgo")
    finally:
        await inline.close()


async def test_the_file_is_deleted_when_the_call_fails() -> None:
    failing = driver_for("shot_error")
    try:
        await failing.start()
        await failing.open("https://example.com/", new_tab=True)
        with pytest.raises(BrowserError, match="Could not finish the screenshot"):
            await failing.screenshot(None)
        assert failing._shots is not None and os.listdir(failing._shots) == []
    finally:
        await failing.close()


@pytest.mark.parametrize("mode", ["decoy", "symlink"])
async def test_a_path_in_the_reply_or_a_link_in_its_place_is_never_read(
    mode: str, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"\x89PNG\r\n\x1a\nnot for the model")
    other = driver_for(mode, FAKE_SAFARI_DECOY=str(secret))
    try:
        await other.start()
        await other.open("https://example.com/", new_tab=True)
        with pytest.raises(BrowserError, match="saved no picture"):
            await other.screenshot(None)
        assert other._shots is not None and os.listdir(other._shots) == []
    finally:
        await other.close()
    assert secret.read_bytes().endswith(
        b"not for the model"
    )  # untouched, never deleted


async def test_no_picture_while_a_password_field_has_focus() -> None:
    focused = driver_for("focus_secret")
    try:
        await focused.start()
        await focused.open("https://example.com/", new_tab=True)
        with pytest.raises(BrowserError, match="has focus"):
            await focused.screenshot(None)
    finally:
        await focused.close()


async def test_other_argument_names_and_script_styles_are_found_from_the_listing() -> (
    None
):
    for mode in ("alt_names", "return_style"):
        other = driver_for(mode)
        try:
            await other.start()
            info = await other.open("https://example.com/", new_tab=True)
            assert info.title == "Example Home", mode
            assert (await other.page(None)).title == "Example Home", mode
        finally:
            await other.close()


@pytest.mark.parametrize("mode", ["automation_off", "exit_off"])
async def test_automation_off_becomes_the_setup_message(mode: str) -> None:
    off = driver_for(mode)
    try:
        with pytest.raises(BrowserSetupError) as caught:
            await off.start()
            await off.open("https://example.com/", new_tab=True)
    finally:
        await off.close()
    text = str(caught.value)
    assert text == SETUP_MESSAGE
    assert "Settings > Developer" in text and "Allow remote automation" in text


async def test_a_server_without_the_needed_tools_is_named() -> None:
    bare = driver_for("no_script")
    with pytest.raises(BrowserError, match="evaluate_javascript"):
        await bare.start()


async def test_a_missing_helper_is_an_error_with_the_place_to_look() -> None:
    gone = SafariAutomationDriver(
        command="/nonexistent/safaridriver", startup_timeout_s=2
    )
    with pytest.raises(BrowserError, match="did not start"):
        await gone.start()


async def test_the_tools_work_end_to_end_over_the_fake_server() -> None:
    provider = BrowserToolProvider(driver_for())
    tools = {t.definition.name: t for t in await provider.start()}
    try:
        out = await tools["browser_open"].function(url="https://example.com/")
        assert isinstance(out, str) and "Welcome to the example." in out
        assert '[e1 link "News" -> /news]' in out
        assert (
            "[e3 password field: melo never types here]" in out and "hunter2" not in out
        )
        followed = await tools["browser_follow"].function(ref="e1")
        assert isinstance(followed, str) and "Today's news." in followed
        shot = await tools["browser_screenshot"].function()
        assert isinstance(shot, ToolOutput) and shot.images[0].mime_type == "image/png"
    finally:
        await provider.close()


async def test_a_setup_problem_in_a_tool_call_is_the_tools_result() -> None:
    provider = BrowserToolProvider(driver_for("automation_off"))
    tools = {t.definition.name: t for t in await provider.start()}
    try:
        out = await tools["browser_open"].function(url="https://example.com/")
    finally:
        await provider.close()
    assert isinstance(out, ToolOutput) and out.is_error and out.text == SETUP_MESSAGE


def test_the_probe_reports_each_step_against_the_fake(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from localmelo.support.browser import probe

    monkeypatch.setenv("LOCALMELO_SAFARIDRIVER", f"{sys.executable} -I {FAKE}")
    assert asyncio.run(probe.probe("https://example.com/")) == 0
    out = capsys.readouterr().out
    assert "tools (17):" in out and "  navigate_to_url(tab_uuid, url)" in out
    assert "opened: tab=" in out and "title='Example Home'" in out
    assert "page: " in out and "link 'News' https://example.com/news" in out
    assert "screenshot: image/png" in out and out.rstrip().endswith("ok")


def test_the_probe_says_what_to_turn_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from localmelo.support.browser import probe

    monkeypatch.setenv("LOCALMELO_SAFARIDRIVER", f"{sys.executable} -I {FAKE} exit_off")
    assert asyncio.run(probe.probe("https://example.com/")) == 1
    assert "FAILED: Safari is not set up for melo yet." in capsys.readouterr().out


# ── Tabs, acting and the page scripts ──


async def test_a_tab_is_named_when_a_page_is_opened_in_it(
    driver: SafariAutomationDriver,
) -> None:
    first = await driver.open("https://example.com/", new_tab=True)
    second = await driver.open("https://example.com/news", new_tab=True)
    await driver.open("https://example.com/form", new_tab=False, tab=first.id)
    assert (await driver.location(first.id)) == "https://example.com/form"
    assert (await driver.location(second.id)) == "https://example.com/news"


async def test_the_page_script_reads_buttons_fields_choices_and_what_it_skipped(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/form", new_tab=True)
    raw = await driver.page(None)
    assert raw.skipped == 2
    check, nick, country = raw.elements
    assert (check.kind, check.label, check.also, check.at) == (
        "button",
        "Check in",
        "Daily check-in",
        0,
    )
    assert nick.submit == "https://example.com/save" and nick.at == 1
    assert nick.attrs is not None and nick.attrs.words == "nick Nickname"
    assert nick.attrs.maxlength == 40 and nick.attrs.masked is False
    assert country.options == ("France", "Spain") and country.at == 2


async def test_clicking_typing_and_choosing_over_the_fake_server(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/form", new_tab=True)
    raw = await driver.page(None)
    url = raw.url
    check, nick, country = raw.elements
    await driver.act(None, Act("type", url, nick, text="Alex", doc=raw.doc))
    await driver.act(None, Act("select", url, country, text="Spain", doc=raw.doc))
    await driver.act(None, Act("click", url, check, doc=raw.doc))
    assert await driver.location(None) == "https://example.com/done"
    assert (await driver.page(None)).title == "Done"


async def test_an_act_on_a_page_that_is_not_the_one_read_is_refused(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/form", new_tab=True)
    raw = await driver.page(None)
    await driver.open("https://example.com/news", new_tab=False)
    with pytest.raises(StaleElementError, match="its address changed"):
        await driver.act(None, Act("click", raw.url, raw.elements[0], doc=raw.doc))
    assert await driver.location(None) == "https://example.com/news"


async def test_an_act_on_an_element_that_is_not_the_one_read_is_refused(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/form", new_tab=True)
    raw = await driver.page(None)
    other = Element("button", "Check out", at=0)
    with pytest.raises(StaleElementError, match="no longer on the page"):
        await driver.act(None, Act("click", raw.url, other, doc=raw.doc))
    assert await driver.location(None) == "https://example.com/form"


@pytest.mark.parametrize(
    ("mode", "problem"),
    [
        ("act_reloaded", "reloaded or replaced"),
        ("act_moved", "no longer on the page as it was read"),
    ],
)
async def test_a_reloaded_page_or_a_moved_element_is_refused(
    mode: str, problem: str
) -> None:
    safari = driver_for(mode)
    await safari.start()
    try:
        await safari.open("https://example.com/form", new_tab=True)
        raw = await safari.page(None)
        assert raw.doc == "doc-1" and raw.elements[0].fp == "button|check in|0"
        with pytest.raises(StaleElementError, match=problem):
            await safari.act(None, Act("click", raw.url, raw.elements[0], doc=raw.doc))
        assert await safari.location(None) == "https://example.com/form"
    finally:
        await safari.close()


async def test_a_page_read_names_its_document_and_an_act_for_another_is_refused(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/form", new_tab=True)
    raw = await driver.page(None)
    with pytest.raises(StaleElementError, match="reloaded or replaced"):
        await driver.act(None, Act("click", raw.url, raw.elements[0], doc="doc-0"))


async def test_an_option_that_is_not_there_is_not_chosen(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/form", new_tab=True)
    raw = await driver.page(None)
    with pytest.raises(BrowserError, match="no single option"):
        await driver.act(
            None, Act("select", raw.url, raw.elements[2], text="Mars", doc=raw.doc)
        )


async def test_the_act_script_text_is_valid_and_carries_its_spec_as_data() -> None:
    from localmelo.support.browser.pagescript import act_script

    spec = {
        "act": "type",
        "url": "https://a/",
        "at": 3,
        "text": "</script>\u2028x $uid(4)",
    }
    script = act_script(spec)
    assert "\u2028" not in script
    start = script.rindex('})({"') + 3
    carried, end = json.JSONDecoder().raw_decode(script, start)
    assert carried == spec and script[end:] == ")"
    # A server macro such as $uid() never sees a page's words: every $ is escaped.
    assert "$uid(4)" not in script and "\\u0024uid(4)" in script


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_page_scripts_are_valid_javascript(tmp_path: Path) -> None:
    from localmelo.support.browser.pagescript import FOCUS_JS, PAGE_JS, act_script

    for name, source in (
        ("page", PAGE_JS),
        ("focus", FOCUS_JS),
        ("act", act_script({"act": "click", "url": "u", "at": 0})),
    ):
        file = tmp_path / f"{name}.js"
        file.write_text(f"(function () {{ return {source}; }});", encoding="utf-8")
        checked = subprocess.run(
            ["node", "--check", str(file)], capture_output=True, text=True, timeout=30
        )
        assert checked.returncode == 0, f"{name}: {checked.stderr}"


async def test_the_tools_use_short_ids_over_the_real_driver_with_long_helper_ids() -> (
    None
):
    # The bug the person saw: the helper's ids are 41 characters, the tab list
    # cut them with an ellipsis, and every close failed.
    provider = BrowserToolProvider(driver_for("long_ids"))
    tools = {t.definition.name: t for t in await provider.start()}
    try:
        await tools["browser_open"].function(url="https://example.com/")
        await tools["browser_open"].function(url="https://example.com/news", tab="new")
        listing = await tools["browser_tabs"].function(action="list")
        assert isinstance(listing, str)
        assert (
            "t1 https://example.com/ " in listing
            and "* t2 https://example.com/news" in listing
        )
        assert "page-" not in listing and "…" not in listing
        closed = await tools["browser_tabs"].function(action="close", tab="t1")
        assert isinstance(closed, str) and "t1" not in closed and "t2" in closed
        assert not isinstance(closed, ToolOutput)
        again = await tools["browser_tabs"].function(action="close", tab="t1")
        assert isinstance(again, ToolOutput) and again.is_error
        assert "not a tab of this chat" in again.text
    finally:
        await provider.close()


async def test_a_button_is_found_clicked_and_the_new_page_read_over_the_fake_server() -> (
    None
):
    provider = BrowserToolProvider(driver_for())
    tools = {t.definition.name: t for t in await provider.start()}
    try:
        await tools["browser_open"].function(url="https://example.com/form")
        found = await tools["browser_find"].function(query="daily check-in")
        # The find read the page again: its refs follow the first reading's.
        assert isinstance(found, str) and '[e4 button "Check in"]' in found
        assert "2 more things on the page were skipped" in found
        clicked = await tools["browser_click"].function(ref="e4")
        assert isinstance(clicked, str)
        assert (
            clicked.startswith("Clicked e4 (via browser).\n<<page id=s3")
            and "Checked in." in clicked
        )
    finally:
        await provider.close()


# ── Which tab is which: by the handle the browser gave it, never by place ──


@pytest.mark.parametrize("mode", ["", "no_active", "no_url", "focus_stays"])
async def test_a_new_tab_is_the_one_that_appeared_whatever_the_list_says(
    mode: str,
) -> None:
    safari = driver_for(mode)
    await safari.start()
    try:
        first = await safari.open("https://example.com/", new_tab=True)
        second = await safari.open("https://example.com/news", new_tab=True)
        assert second.id != first.id
        assert (second.url, second.title) == ("https://example.com/news", "News")
        # The tab melo named is the one read, wherever the browser's focus is.
        assert (await safari.page(second.id)).title == "News"
        assert (await safari.page(first.id)).title == "Example Home"
        assert await safari.location(first.id) == "https://example.com/"
        assert await safari.location(second.id) == "https://example.com/news"
    finally:
        await safari.close()


async def test_a_list_without_handles_gives_no_tab_to_name() -> None:
    safari = driver_for("no_ids")
    await safari.start()
    try:
        assert await safari.tabs() == []
        with pytest.raises(BrowserError, match="could not tell which tab"):
            await safari.open("https://example.com/", new_tab=True)
    finally:
        await safari.close()


# ── Where a page is when it is read ──


async def test_a_page_that_moves_between_the_address_read_and_the_script_is_refused() -> (
    None
):
    safari = driver_for("page_moves")
    await safari.start()
    try:
        await safari.open("https://example.com/", new_tab=True)
        # The tab list said the home page; by the time the script ran the page
        # had gone to another, which nobody checked.
        with pytest.raises(BrowserError, match="moved while melo was reading"):
            await safari.page(None)
        # Read again, with nothing moving, it is the page that is there.
        assert (await safari.page(None)).url == "https://example.com/news"
    finally:
        await safari.close()


@pytest.mark.parametrize(
    ("listed", "script", "same"),
    [
        ("https://example.com/", "https://example.com/", True),
        ("https://EXAMPLE.com", "https://example.com/", True),
        ("https://example.com:443/a?b=1", "https://example.com/a?b=1", True),
        ("https://example.com/a#top", "https://example.com/a#end", True),
        ("https://example.com/a", "https://example.com/b", False),
        ("https://example.com/a?b=1", "https://example.com/a?b=2", False),
        ("https://example.com/", "https://example.com:8443/", False),
        ("https://example.com/", "http://example.com/", False),
        ("https://example.com/", "https://127.0.0.1/", False),
        ("https://example.com/", None, False),
        ("https://example.com/", "https://[::1", False),
    ],
)
def test_two_addresses_are_one_page_by_site_path_and_query(
    listed: str, script: object, same: bool
) -> None:
    assert same_page(listed, script) is same


# ── Leaving a refused page ──


async def test_a_blanked_page_is_replaced_in_the_history_not_followed_by_a_blank() -> (
    None
):
    safari = driver_for()
    await safari.start()
    try:
        info = await safari.open("https://example.com/", new_tab=True)
        await safari.open("https://example.com/news", new_tab=False, tab=info.id)
        await safari.blank(info.id)
        assert await safari.location(info.id) == "about:blank"
        # Back goes to the page before the refused one, never to the refused one.
        assert (await safari.back(info.id)).url == "https://example.com/"
    finally:
        await safari.close()


async def test_a_blank_that_does_not_take_falls_back_to_the_browsers_own_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("localmelo.support.browser.pagescript.POLL_S", 0.0)
    safari = driver_for("replace_fails")
    await safari.start()
    try:
        info = await safari.open("https://example.com/", new_tab=True)
        await safari.blank(info.id)
        assert await safari.location(info.id) == "about:blank"
    finally:
        await safari.close()
