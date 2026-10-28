"""The picker keeps the ordinary draft and history out of its filter."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localmelo.support.sessions.store import SessionStore
from localmelo.tests.cli.ui_helpers import (
    BACKSPACE,
    CTRL_D,
    ENTER,
    ESC,
    UP,
    running_app,
)


async def add_session(
    store: SessionStore, title: str, workspace: str, *, kind: str = "chat"
) -> str:
    info = await store.create_session(workspace=workspace, kind=kind)
    turn = await store.begin_turn(info.id, title, holder="test")
    await store.finish_turn(
        info.id,
        turn.turn,
        holder="test",
        status="completed",
        cause="run",
        result=None,
        answer="ok",
    )
    return info.id


async def test_filter_scope_selection_and_draft_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "s.db", writer="cli")
    try:
        local = await add_session(store, "Fix 中文", "/project")
        await add_session(store, "Elsewhere", "/other")
        await add_session(store, "Hidden task", "/project", kind="task")
        empty = await store.create_session(workspace="/project")
        await store.rename_session(empty.id, "Empty")
        async with running_app() as driver:
            driver.ui.buffer.text = "unsent draft"
            task = asyncio.create_task(driver.ui.choose_session(store, "/project"))
            await driver.wait(lambda: "Fix 中文" in driver.text())
            assert "Elsewhere" not in driver.text()
            assert "Hidden task" not in driver.text()
            assert "Empty" not in driver.text()
            driver.type("中文界" + BACKSPACE)
            await driver.wait(
                lambda: driver.ui.picker is not None
                and driver.ui.picker.buffer.text == "中文"
            )
            await driver.wait(lambda: "1 session" in driver.text())
            assert "1 turns" not in driver.text() and "1 sessions" not in driver.text()
            driver.type("\t")
            await driver.wait(lambda: "all folders" in driver.text())
            driver.type(ENTER)
            selected = await asyncio.wait_for(task, 3)
            assert selected is not None and selected.id == local
            assert driver.ui.buffer.text == "unsent draft"
            driver.type(UP)  # back in the message input, not the old filter
            await driver.settle()
    finally:
        await store.aclose()


@pytest.mark.parametrize("close", [ESC, CTRL_D, "\x03"])
async def test_picker_close_never_exits_or_sends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "s.db", writer="cli")
    try:
        async with running_app(columns=40) as driver:
            task = asyncio.create_task(driver.ui.choose_session(store, "/project"))
            await driver.wait(lambda: "Resume a session" in driver.text())
            driver.type(close)
            assert await asyncio.wait_for(task, 3) is None
            assert driver.ui.app.is_running
            assert await store.list_sessions() == []
    finally:
        await store.aclose()
