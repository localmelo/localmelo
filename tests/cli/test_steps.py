"""A turn's steps: the status bar's brief figure, the run zone, and the ctrl+o list."""

from __future__ import annotations

import pytest

from localmelo.support.cli.app import StatusBar, status_fragments
from localmelo.support.cli.render import ToolOutput
from localmelo.support.cli.steps import (
    StepRow,
    StepsPicker,
    call_status,
    learned_total,
    step_figure,
)
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE
from localmelo.tests.cli.test_render import Turn
from localmelo.tests.cli.ui_helpers import ESC, FakeController, running_app

CTRL_O = "\x0f"


# ── The figure ──


def test_a_plan_gives_its_total_a_learned_one_is_marked_and_none_gives_the_step() -> (
    None
):
    assert step_figure(2, planned=8) == "2/8"
    assert step_figure(2, learned=8) == "2/~8"
    assert step_figure(2) == "step 2"
    # A plan wins over what was learned.
    assert step_figure(2, planned=5, learned=8) == "2/5"


def test_a_total_the_run_has_passed_is_dropped() -> None:
    assert step_figure(9, learned=8) == "step 9"
    assert step_figure(9, planned=8) == "step 9"
    assert step_figure(0, learned=8) == ""


def test_the_learned_total_comes_from_the_bar_then_the_last_run_then_the_most() -> None:
    assert learned_total({"expected_steps": 7, "past": {"last_steps": 5}}) == 7
    assert learned_total({"past": {"last_steps": 5, "steps_hi": 9}}) == 5
    assert learned_total({"past": {"steps_hi": 9}}) == 9
    assert learned_total({"step": 2}) is None
    assert learned_total(None) is None
    # A task that is new has nothing learned, whatever the payload holds.
    new = {"basis": {"match": "new"}, "past": {"last_steps": 5}}
    assert learned_total(new) is None


@pytest.mark.parametrize(
    ("status", "exit_code", "row"),
    [
        ("success", None, "ok"),
        ("success", 0, "ok"),
        ("success", 1, "failed"),
        ("error", None, "failed"),
        ("blocked", None, "refused"),
        ("cancelled", None, "stopped"),
    ],
)
def test_a_call_status_becomes_a_row_status(
    status: str, exit_code: int | None, row: str
) -> None:
    assert call_status(status, exit_code) == row


# ── The run zone ──


def _bar(width: int, run: str = "2/~8 · ctrl+o steps", mode: str = "") -> str:
    bar = StatusBar(
        "qwen3-8b",
        "~/demo-project",
        "",
        "Add a --json flag",
        mode,
        segments=("83% cached", "context 8% used"),
        run=run,
    )
    line = "".join(part[1] for part in status_fragments(bar, width, "…"))
    assert cells(line) == width, (width, line)
    return line


def test_the_zones_keep_their_place_and_order() -> None:
    assert _bar(100).rstrip() == (
        "  qwen3-8b · ~/demo-project │ 2/~8 · ctrl+o steps"
        " │ 83% cached · context 8% used"
    )
    # Idle, the run zone holds the chat's title.
    assert _bar(100, run="").rstrip() == (
        "  qwen3-8b · ~/demo-project │ Add a --json flag"
        " │ 83% cached · context 8% used"
    )


def test_resources_go_first_then_the_folder_is_cut_and_the_run_stays() -> None:
    # The context goes whole, then the cache, then the folder is cut.
    assert _bar(70).rstrip().endswith("│ 83% cached")
    assert "~/demo-project │ 2/~8 · ctrl+o steps" in _bar(70)
    assert _bar(50).rstrip() == "  qwen3-8b · ~/demo-project │ 2/~8 · ctrl+o steps"
    assert _bar(40) == "  qwen3-8b · ~/de… │ 2/~8 · ctrl+o steps"
    # Then the folder goes, the key's word, the model, the key: the figure stays.
    assert _bar(30).rstrip() == "  qwen3-8b │ 2/~8 · ctrl+o"
    assert _bar(16).rstrip() == "  2/~8 · ctrl+o"
    assert _bar(10).rstrip() == "  2/~8"


@pytest.mark.parametrize("width", range(8, 110, 3))
def test_the_bar_with_a_run_never_overflows_or_shows_a_bare_ellipsis(
    width: int,
) -> None:
    for mode in ("", "auto"):
        line = _bar(width, mode=mode)
        assert " … " not in line and not line.strip().startswith("…"), line
        if width >= 12:
            assert "2/~8" in line


def test_the_zones_are_drawn_in_their_roles() -> None:
    bar = StatusBar(
        "qwen3-8b", "~/demo-project", "", "", "", segments=("83% cached",), run="2/~8"
    )
    parts = status_fragments(bar, 80, "…")
    styles = {text: style for style, text, *_ in parts}
    assert styles["qwen3-8b"] == "class:status_model"
    assert styles["~/demo-project"] == "class:status_path"
    assert styles[" │ "] == "class:status_sep"
    assert styles["2/~8"] == "class:status_run"
    assert styles["83% cached"] == "class:status"
    running = StatusBar("m", "~/p", "", "", "", run="2/~8 · ctrl+o steps")
    parts = status_fragments(running, 80, "…")
    assert ("class:key", "ctrl+o") in parts and ("class:faint", " steps") in parts


def test_ascii_and_no_color_get_a_plain_separator() -> None:
    bar = StatusBar("m", "~/p", "", "chat", "", run="")
    line = "".join(
        part[1] for part in status_fragments(bar, 40, "...", " - ", f" {ASCII.gutter} ")
    )
    assert line.startswith("  m - ~/p | chat")


# ── The list ──


def _picker(rows: list[StepRow], width: int = 70) -> StepsPicker:
    return StepsPicker(
        lambda: rows,
        title="Steps of this turn",
        width=lambda: width,
        invalidate=lambda: None,
        symbols=UNICODE,
    )


async def test_the_list_shows_each_call_and_the_steps_to_come() -> None:
    picker = _picker(
        [
            StepRow("ok", "file_read", "app.py", 0.4, "read"),
            StepRow("failed", "shell_exec", "pytest", 2.0, "run"),
            StepRow("running", "file_write", "todo.md", 3.0, "write"),
            StepRow("next", "Run the tests"),
        ]
    )
    text = picker._rows()
    assert "✓ → file_read  app.py · 0.4s" in text
    assert "✗ $ shell_exec  pytest · 2.0s" in text
    assert "± file_write  todo.md · running 0:03" in text
    assert "◌ Run the tests · to come" in text
    assert text.endswith("3 steps, 1 to come · ↑↓ move · esc close")
    # The highlight starts on the newest call, not on a step to come.
    assert picker.selected == 2
    marks = dict(
        (text, style) for style, text, *_ in picker._row_fragments(picker.rows[1])
    )
    assert marks["✗"] == "class:error" and marks["$ shell_exec"] == "class:kind_run"


async def test_a_long_target_is_cut_to_the_width() -> None:
    picker = _picker([StepRow("ok", "file_read", "a/" * 60, 0.1, "read")], width=40)
    first = picker._rows().split("\n")[0]
    assert cells(first) <= 40 and "…" in first and first.endswith("· 0.1s")


def test_the_renderer_lists_the_turns_calls_and_the_items_to_come() -> None:
    turn = Turn()
    turn.tool("file_read", "app.py", call_id="c1")
    turn.tool("shell_exec", "pytest", call_id="c2", exit_code=1)
    turn.event(
        "run.progress",
        step=3,
        items=[
            {"label": "Read app.py", "state": "done"},
            {"label": "Run the tests", "state": "pending"},
        ],
    )
    rows = turn.renderer.steps()
    assert [(row.status, row.tool, row.kind) for row in rows] == [
        ("ok", "file_read", "read"),
        ("failed", "shell_exec", "run"),
        ("next", "Run the tests", ""),
    ]
    # An item that is not a mapping is skipped; after the turn none is to come.
    turn.event("run.progress", step=3, items=["odd", {"state": "pending"}])
    assert [row.status for row in turn.renderer.steps()] == ["ok", "failed"]
    turn.renderer.live = None
    turn.event("run.progress", step=3, items=[{"label": "x", "state": "pending"}])
    assert [row.status for row in turn.renderer.steps()] == ["ok", "failed"]
    # The next turn starts its own list.
    turn.renderer.begin_turn("again")
    assert [row.status for row in turn.renderer.steps()] == []


async def test_ctrl_o_opens_the_steps_and_esc_closes_them() -> None:
    controller = FakeController(state="running")
    async with running_app(controller) as app:
        app.ui.renderer.tool_outputs.append(
            ToolOutput("1.1", "file_read", "app.py", "success", "", 0.2)
        )
        app.ui.renderer._turn_outputs = 0
        app.type(CTRL_O)
        await app.wait(lambda: "Steps of the last turn" in app.text())
        assert "✓ → file_read  app.py · 0.2s" in app.text()
        assert "Ask melo" not in app.text()
        app.type(ESC)
        await app.wait(lambda: "Steps of the last turn" not in app.text())
        assert app.ui.picker is None and controller.stops == 0  # esc stopped nothing


async def test_ctrl_o_with_no_steps_says_so() -> None:
    async with running_app() as app:
        app.type(CTRL_O)
        await app.wait(lambda: "No steps yet" in app.text())
        assert app.ui.picker is None


@pytest.mark.parametrize("width", range(10, 40))
def test_a_control_character_in_the_model_is_never_drawn_raw(width: int) -> None:
    bar = StatusBar(
        "evil\x1b]0;x\x07model", "~/p", "", "", "", run="2/~8 · ctrl+o steps"
    )
    line = "".join(part[1] for part in status_fragments(bar, width, "…"))
    assert "\x1b" not in line and "\x07" not in line and cells(line) == width
