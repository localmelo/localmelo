"""The picker keeps the ordinary draft and history out of its filter."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localmelo.support.cli.picker import ModelPicker, ModelRow
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.cli.ui_helpers import (
    BACKSPACE,
    CTRL_D,
    ENTER,
    ESC,
    UP,
    Driver,
    running_app,
)

DOWN = "\x1b[B"


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


# ── The model picker shares the list ──

MODELS = [
    ModelRow("stub-model"),
    ModelRow("stub-model-large", loaded=True),
    ModelRow("stub-model-small"),
]
WHERE = "MLC-LLM · http://127.0.0.1:9/v1"


def _choose(driver: Driver, current: str = "stub-model") -> asyncio.Task[str | None]:
    return asyncio.create_task(
        driver.ui.choose_model(MODELS, where=WHERE, current=current)
    )


async def test_the_model_list_starts_on_the_current_model_and_filters_by_typing() -> (
    None
):
    async with running_app() as driver:
        driver.ui.buffer.text = "unsent draft"
        task = _choose(driver)
        await driver.wait(lambda: "stub-model-small" in driver.text())
        text = driver.text()
        assert f"Switch model · {WHERE}" in text
        assert "❯ ✓ stub-model\n" in text  # the cursor starts where the model is
        assert "    stub-model-large  (loaded)\n" in text
        assert "3 models · enter switches and saves · esc keeps stub-model" in text
        driver.type("LARGE")  # a filter ignores case
        await driver.wait(lambda: "1 model · " in driver.text())
        assert "stub-model-small" not in driver.text()
        assert "❯   stub-model-large  (loaded)" in driver.text()
        driver.type(ENTER)
        assert await asyncio.wait_for(task, 3) == "stub-model-large"
        assert driver.ui.buffer.text == "unsent draft"  # the filter was its own


async def test_the_arrow_keys_move_the_cursor_and_enter_chooses() -> None:
    async with running_app() as driver:
        task = _choose(driver)
        await driver.wait(lambda: "❯ ✓ stub-model\n" in driver.text())
        driver.type(DOWN + DOWN + DOWN)  # past the last row stays on it
        await driver.wait(lambda: "❯   stub-model-small" in driver.text())
        driver.type(UP)
        await driver.wait(lambda: "❯   stub-model-large" in driver.text())
        driver.type(ENTER)
        assert await asyncio.wait_for(task, 3) == "stub-model-large"


async def test_a_filter_that_matches_nothing_says_so_and_enter_does_nothing() -> None:
    async with running_app() as driver:
        task = _choose(driver)
        await driver.wait(lambda: "stub-model-small" in driver.text())
        driver.type("zzz")
        await driver.wait(lambda: "No model matches 'zzz'." in driver.text())
        driver.type(ENTER)
        await driver.settle()
        assert not task.done()
        driver.type(BACKSPACE * 3)
        await driver.wait(lambda: "3 models" in driver.text())
        driver.type(ESC)
        assert await asyncio.wait_for(task, 3) is None


@pytest.mark.parametrize("close", [ESC, CTRL_D, "\x03"])
async def test_closing_the_model_list_chooses_nothing(close: str) -> None:
    async with running_app(columns=40) as driver:
        task = _choose(driver)
        await driver.wait(lambda: "Switch model" in driver.text())
        driver.type(close)
        assert await asyncio.wait_for(task, 3) is None
        assert driver.ui.app.is_running and driver.ui.picker is None


async def test_with_no_default_nothing_is_chosen_until_a_key_moves_to_a_row() -> None:
    async with running_app() as driver:
        picker = ModelPicker(
            MODELS,
            where=WHERE,
            start=None,
            title="Choose a model",
            width=driver.ui._width,
            invalidate=driver.ui.invalidate,
            symbols=UNICODE,
        )

        async def pick() -> ModelRow | None:
            return await driver.ui.choose(picker)

        task = asyncio.create_task(pick())
        await driver.wait(lambda: "↑↓ choose one, then enter" in driver.text())
        assert "❯" not in driver.text().split("Filter")[1].split("─")[0]
        driver.type(ENTER)
        await driver.settle()
        assert not task.done()
        driver.type(DOWN)
        await driver.wait(lambda: "enter chooses · esc cancels" in driver.text())
        driver.type(ENTER)
        assert (await asyncio.wait_for(task, 3)) == MODELS[0]


async def test_the_model_list_fits_44_columns_and_the_ascii_symbols() -> None:
    async with running_app(columns=44) as driver:
        driver.ui.symbols = ASCII
        task = _choose(driver)
        await driver.wait(lambda: "stub-model-small" in driver.text())
        rows = [row for row in driver.screen() if row.strip()]
        start = next(i for i, row in enumerate(rows) if "Switch model" in row)
        picker = rows[start : start + 6]
        assert all(cells(row) <= 44 for row in picker), picker
        assert all(row.isascii() for row in picker), picker
        assert picker[2].startswith(" > + stub-model") and "(loaded)" in picker[3]
        assert picker[-1].strip() == "3 models - enter switches - esc cancels"
        driver.type(ESC)
        await asyncio.wait_for(task, 3)


async def test_model_ids_are_cleaned_and_a_long_one_is_cut_before_its_tags() -> None:
    picker = _model_picker_without_a_screen(
        [
            ModelRow("evil\x1b]0;PWNED\x1b\\model"),
            ModelRow("a-very-long-model-name-that-does-not-fit-4bit", loaded=True),
        ],
        width=30,
    )
    picker.rows = list(picker.models)
    drawn = picker._rows()
    assert "\x1b" not in drawn and "PWNED" not in drawn
    long_row = drawn.split("\n")[1]
    assert cells(long_row) <= 30 and long_row.endswith("  (loaded)")
    assert "…  (loaded)" in long_row


def _model_picker_without_a_screen(models: list[ModelRow], width: int) -> ModelPicker:
    return ModelPicker(
        models,
        where=WHERE,
        current="stub-model",
        width=lambda: width,
        invalidate=lambda: None,
        symbols=UNICODE,
    )
