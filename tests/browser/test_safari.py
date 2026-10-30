"""The safari-automation driver against a fake ``safaridriver --mcp`` (real subprocess)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.executor.models import ToolOutput
from localmelo.support.browser.driver import BrowserError, BrowserSetupError
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
    info = await driver.open("https://example.com/", new_tab=False)
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
    first = await driver.open("https://example.com/", new_tab=False)
    second = await driver.open("https://example.com/news", new_tab=True)
    assert second.id != first.id and (await driver.page(None)).title == "News"
    assert [t.id for t in await driver.tabs()] == [first.id, second.id]
    assert (await driver.switch(first.id)).url == "https://example.com/"
    await driver.open("https://example.com/news", new_tab=False)
    assert (await driver.back(None)).url == "https://example.com/"
    await driver.blank(None)
    assert (await driver.page(None)).url == "about:blank"
    await driver.close_tab(second.id)
    assert [t.id for t in await driver.tabs()] == [first.id]


async def test_wait_polls_until_the_page_has_the_text(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/", new_tab=False)
    assert await driver.wait(None, "text", "Welcome", 2)
    assert await driver.wait(None, "gone", "Nowhere", 2)
    assert not await driver.wait(None, "text", "Nowhere", 0.5)
    assert await driver.wait(None, "load", "", 2)


async def test_slow_loading_pages_are_waited_for() -> None:
    slow = driver_for("slow_load")
    try:
        await slow.start()
        info = await slow.open("https://example.com/", new_tab=False)
        assert info.title == "Example Home"
    finally:
        await slow.close()


async def test_a_screenshot_is_the_png_safari_saved_where_melo_asked(
    driver: SafariAutomationDriver,
) -> None:
    await driver.open("https://example.com/", new_tab=False)
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
        await inline.open("https://example.com/", new_tab=False)
        assert (await inline.screenshot(None)).data.startswith("iVBORw0KGgo")
    finally:
        await inline.close()


async def test_the_file_is_deleted_when_the_call_fails() -> None:
    failing = driver_for("shot_error")
    try:
        await failing.start()
        await failing.open("https://example.com/", new_tab=False)
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
        await other.open("https://example.com/", new_tab=False)
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
        await focused.open("https://example.com/", new_tab=False)
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
            info = await other.open("https://example.com/", new_tab=False)
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
