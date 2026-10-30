"""The terminal UI's keys and input box, driven through a pipe.

Keys go in as a terminal sends them; the screen is read back with pyte.
The session is a fake that records what the app asks of it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory

from localmelo.support.cli.app import StatusBar, open_history, status_fragments
from localmelo.support.cli.completion import SlashCompleter
from localmelo.support.cli.render import LiveState
from localmelo.support.cli.text import cells
from localmelo.tests.cli.ui_helpers import (
    BACKSPACE,
    CTRL_C,
    CTRL_D,
    CTRL_J,
    ENTER,
    ESC,
    UP,
    Driver,
    FakeController,
    running_app,
)


def _box(driver: Driver) -> list[str]:
    """The rows between the input box's two rules."""
    screen = driver.screen()
    rules = [i for i, line in enumerate(screen) if line.startswith("──────")]
    assert len(rules) >= 2, "\n".join(screen)
    return screen[rules[0] + 1 : rules[1]]


async def test_explicit_attachment_selection_preserves_draft(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("not read by completion")

    class Controller(FakeController):
        agent = SimpleNamespace(workspace_root=str(tmp_path))
        attachment_line = ""
        paths: list[str] = []

        def attach_path(self, path: str) -> str | None:
            self.paths.append(path)
            self.attachment_line = f"Attached: {path}"
            return None

    controller = Controller()
    async with running_app(controller, completer=SlashCompleter()) as app:
        app.type("look @fi")
        await app.wait(lambda: app.ui.buffer.text == "look @fi")
        app.type("\t" + ENTER)
        await app.wait(lambda: controller.paths == ["file.txt"])
        assert app.ui.buffer.text == "look @file.txt"
        assert controller.entered == []
        await app.wait(lambda: "Attached: file.txt" in app.text())
        assert app.ui._attachments_fragments() == [
            ("class:muted", "  Attached: "),
            ("", "file.txt"),
        ]
        app.type(ENTER)
        await app.wait(lambda: controller.entered == ["look @file.txt"])
        app.type("literal @file.txt" + ENTER)
        await app.wait(lambda: len(controller.entered) == 2)
        assert controller.paths == ["file.txt"]


async def test_enter_sends_and_blank_drafts_are_not_sent() -> None:
    controller = FakeController()
    async with running_app(controller) as app:
        app.type("   " + ENTER)
        await app.settle()
        assert controller.entered == [] and app.ui.buffer.text == "   "
        app.type(BACKSPACE * 3 + "hello" + ENTER)
        await app.wait(lambda: controller.entered == ["hello"])
        assert app.ui.buffer.text == ""


@pytest.mark.parametrize("newline", [ESC + ENTER, CTRL_J, "\\" + ENTER])
async def test_new_line_keys_add_a_line(newline: str) -> None:
    controller = FakeController()
    async with running_app(controller) as app:
        app.type("first" + newline)
        await app.settle(0.5)
        app.type("second" + ENTER)
        await app.wait(lambda: controller.entered == ["first\nsecond"])


async def test_backspace_deletes_whole_wide_characters() -> None:
    async with running_app() as app:
        app.type("你好世界" + BACKSPACE * 2)
        await app.wait(lambda: app.ui.buffer.text == "你好")
        app.type("世界abc" + BACKSPACE * 4)
        await app.wait(lambda: app.ui.buffer.text == "你好世")
        assert cells(app.ui.buffer.text) == 6
        await app.wait(lambda: _box(app)[0].rstrip() == "› 你好世")


async def test_a_character_split_across_writes_arrives_whole() -> None:
    async with running_app() as app:
        for byte in "你".encode():
            app.pipe.send_bytes(bytes([byte]))
            await app.settle(0.05)
        await app.wait(lambda: app.ui.buffer.text == "你")


async def test_wrapped_wide_text_shows_whole_and_shrinks_without_blank_rows() -> None:
    async with running_app(columns=21) as app:
        app.type("你" * 30)
        await app.wait(lambda: app.ui.buffer.text == "你" * 30)
        await app.settle()
        box = _box(app)
        assert box == [
            "› " + "你" * 9,
            "  " + "你" * 9,
            "  " + "你" * 9,
            "  " + "你" * 3,
        ]
        app.type(BACKSPACE * 10)
        await app.wait(lambda: len(app.ui.buffer.text) == 20)
        await app.settle()
        box = _box(app)
        assert box == ["› " + "你" * 9, "  " + "你" * 9, "  " + "你" * 2]
        assert all(row.strip() for row in box)
        # The app itself shrank: no blank rows of it are left under the
        # status bar (the inline renderer never shrinks on its own).
        screen = app.screen()
        top = next(i for i, row in enumerate(screen) if row.startswith("─"))
        bar = max(i for i, row in enumerate(screen) if row.strip())
        last = app.ui.app.renderer._last_screen
        assert last is not None and last.height == bar - top + 1


async def test_a_bracketed_paste_inserts_lines_and_sends_nothing() -> None:
    controller = FakeController()
    async with running_app(controller) as app:
        app.type("\x1b[200~第一行\n第二行\x1b[201~")
        await app.wait(lambda: app.ui.buffer.text == "第一行\n第二行")
        await app.settle()
        assert controller.entered == []
        assert _box(app)[:2] == ["› 第一行", "  第二行"]


async def test_enter_during_a_turn_queues_and_the_line_shows() -> None:
    controller = FakeController(state="running")
    async with running_app(controller) as app:
        app.type("Then add a test" + ENTER)
        await app.wait(lambda: controller.queued == "Then add a test")
        await app.wait(
            lambda: "↳ Queued for the next turn: Then add a test" in app.text()
        )
        app.type("another" + ENTER)
        await app.wait(lambda: "One message is already queued" in app.text())
        assert app.ui.buffer.text == "another"


async def test_up_on_an_empty_draft_takes_the_queued_message_back() -> None:
    controller = FakeController(state="running", queued="fix it")
    async with running_app(controller) as app:
        app.type(UP)
        await app.wait(lambda: app.ui.buffer.text == "fix it")
        assert controller.queued is None


async def test_esc_stops_a_turn_and_does_nothing_when_idle() -> None:
    controller = FakeController()
    async with running_app(controller) as app:
        app.type(ESC)
        await app.settle(0.6)
        assert controller.stops == 0
        controller.state = "running"
        app.type(ESC)
        await app.wait(lambda: controller.stops == 1, timeout_s=3)
        app.type(ESC)
        await app.settle(0.6)
        assert controller.stops == 1  # stopping: a second Esc does nothing


async def test_esc_with_the_menu_open_closes_it_and_the_turn_runs_on() -> None:
    controller = FakeController(state="running")
    completer = WordCompleter(["/help", "/new"], sentence=True)
    async with running_app(controller, completer=completer) as app:
        app.type("/he")
        await app.wait(lambda: app.ui.buffer.complete_state is not None)
        app.type(ESC)
        await app.wait(lambda: app.ui.buffer.complete_state is None, timeout_s=3)
        await app.settle(0.6)
        assert controller.stops == 0
        assert app.ui.buffer.text == "/he"


async def test_enter_on_a_partial_command_takes_the_first_completion() -> None:
    controller = FakeController()
    completer = WordCompleter(["/help", "/new"], sentence=True)
    async with running_app(controller, completer=completer) as app:
        app.type("/he")
        await app.wait(lambda: app.ui.buffer.complete_state is not None)
        app.type(ENTER)
        await app.wait(lambda: controller.entered == ["/help"])


async def test_ctrl_c_clears_the_draft_then_the_queue_then_stops() -> None:
    controller = FakeController(state="running", queued="later")
    async with running_app(controller) as app:
        app.type("draft" + CTRL_C)
        await app.wait(
            lambda: app.ui.buffer.text == "" and "Draft cleared" in app.text()
        )
        app.type(CTRL_C)
        await app.wait(lambda: controller.queued is None)
        assert controller.stops == 0
        app.type(CTRL_C)
        await app.wait(lambda: controller.stops == 1)
        await app.wait(lambda: "press ctrl+c again to quit" in app.text())
        app.type(CTRL_C)  # stopping, within 2 s: quits
        await app.wait(lambda: not app.ui.app.is_running)


async def test_ctrl_c_twice_quits_when_idle_but_clearing_a_draft_does_not_arm_it() -> (
    None
):
    async with running_app() as app:
        app.type("draft" + CTRL_C)
        await app.wait(lambda: app.ui.buffer.text == "")
        app.type(CTRL_C)
        await app.wait(lambda: "Press ctrl+c again to quit" in app.text())
        assert app.ui.app.is_running
        app.type(CTRL_C)
        await app.wait(lambda: not app.ui.app.is_running)


async def test_ctrl_c_after_two_seconds_asks_again() -> None:
    now = [100.0]
    async with running_app(clock=lambda: now[0]) as app:
        app.type(CTRL_C)
        await app.wait(lambda: "Press ctrl+c again to quit" in app.text())
        now[0] += 2.5
        app.type(CTRL_C)
        await app.settle()
        assert app.ui.app.is_running


async def test_ctrl_d_quits_when_idle_deletes_with_text_and_hints_in_a_turn() -> None:
    controller = FakeController(state="running")
    async with running_app(controller) as app:
        app.type("ab\x1b[D" + CTRL_D)  # left, then delete the b
        await app.wait(lambda: app.ui.buffer.text == "a")
        app.type(BACKSPACE + CTRL_D)
        await app.wait(lambda: "esc stops it, /exit quits" in app.text())
        assert app.ui.app.is_running
        controller.state = "idle"
        app.type(CTRL_D)
        await app.wait(lambda: not app.ui.app.is_running)


async def test_a_hint_keeps_the_draft() -> None:
    controller = FakeController(state="stopping")
    async with running_app(controller) as app:
        app.type("next" + ENTER)
        await app.wait(lambda: "enter sends once it has stopped" in app.text())
        assert app.ui.buffer.text == "next"


async def test_history_recalls_what_was_sent(tmp_path: Path) -> None:
    history = open_history(str(tmp_path / "cli_history"))
    controller = FakeController()
    async with running_app(controller, history=history) as app:
        app.type("first message" + ENTER)
        app.type("second" + ENTER)
        await app.wait(lambda: len(controller.entered) == 2)
        # The buffer reloads the history when it draws next after Enter.
        await app.settle()
        app.type(UP + UP)
        await app.wait(lambda: app.ui.buffer.text == "first message")
        app.type(CTRL_C)
        await app.wait(lambda: app.ui.buffer.text == "")
        app.type("\x12first")  # ctrl+r
        await app.wait(lambda: "first" in app.text() and "I-search" in app.text())
        app.type(ENTER)
        await app.wait(lambda: app.ui.buffer.text == "first message")
    assert list(FileHistory(str(tmp_path / "cli_history")).load_history_strings()) == [
        "second",
        "first message",
    ]


async def test_a_question_takes_its_default_and_keeps_the_draft() -> None:
    async with running_app() as app:
        app.type("my draft")
        await app.wait(lambda: app.ui.buffer.text == "my draft")
        asking = asyncio.ensure_future(
            app.ui.ask("Model server URL", "http://127.0.0.1:8000/v1")
        )
        await app.wait(
            lambda: "Model server URL [http://127.0.0.1:8000/v1]:" in app.text()
        )
        assert app.ui.buffer.text == ""
        app.type(ENTER)
        await app.wait(asking.done)
        assert asking.result() == "http://127.0.0.1:8000/v1"
        assert app.ui.buffer.text == "my draft"
        asking = asyncio.ensure_future(app.ui.ask("Model", "1"))
        await app.wait(lambda: "Model [1]:" in app.text())
        app.type("qwen" + ESC)
        await app.wait(asking.done, timeout_s=3)
        assert asking.result() is None


async def test_the_status_bar_keeps_the_mode_in_the_last_column() -> None:
    controller = FakeController(title="修复失败的测试")
    async with running_app(controller, columns=60) as app:
        await app.settle()
        bar = app.screen()[-24:]
        line = next(row for row in reversed(app.screen()) if row.strip())
        assert line.endswith("ask") and cells(line) == 60, bar
        assert "修复失败的测试" in line


async def test_the_status_bar_cuts_the_title_first_when_narrow() -> None:
    controller = FakeController(title="Fix the failing test in the parser")
    async with running_app(controller, columns=40) as app:
        await app.settle()
        line = next(row for row in reversed(app.screen()) if row.strip())
        assert line.startswith("  stub-model · ~/demo-project")
        assert line.endswith("ask") and cells(line) == 40


async def test_the_live_area_shows_the_spinner_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALMELO_SPINNER_FRAME", "3")
    now = [50.0]
    controller = FakeController(state="running")
    async with running_app(controller, clock=lambda: now[0]) as app:
        app.ui.renderer.live = LiveState(
            activity="Thinking", started=44.0, step=4, output_tokens=412
        )
        app.ui.invalidate()
        await app.wait(
            lambda: "✶ Thinking… · 6s · ↓ 412 tokens · esc to stop" in app.text()
        )
        assert "step 4" not in app.text()
        # Developer mode adds the run's facts to the row.
        app.ui.renderer.developer = True
        app.ui.invalidate()
        await app.wait(
            lambda: "✶ Thinking… · step 4 · 6s · ↓ 412 tokens · esc to stop"
            in app.text()
        )


def test_the_history_file_is_private_and_kept_under_a_size(tmp_path: Path) -> None:
    path = tmp_path / "logs-and-history" / "cli_history"
    open_history(str(path))
    assert path.stat().st_mode & 0o777 == 0o600
    with path.open("a", encoding="utf-8") as file:
        for number in range(1_500):
            file.write(f"\n# t\n+entry {number} " + "x" * 800 + "\n")
    assert path.stat().st_size > 1_000_000
    history = open_history(str(path))
    entries = list(history.load_history_strings())
    assert len(entries) == 1_000
    assert entries[0].startswith("entry 1499 ") and entries[-1].startswith("entry 500 ")
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.stat().st_size < 1_000_000


async def test_ctrl_r_shows_a_long_match_whole() -> None:
    from prompt_toolkit.history import InMemoryHistory

    history = InMemoryHistory()
    entry = "fix the parser\nthen run the tests\nand update the changelog"
    history.append_string(entry)
    async with running_app(history=history, columns=40) as app:
        await app.settle()
        app.type("\x12parser")  # ctrl+r
        await app.wait(lambda: "I-search" in app.text())
        await app.settle()
        box = _box(app)
        assert "  then run the tests" in box and "  and update the changelog" in box
        app.type(ENTER)
        await app.wait(lambda: app.ui.buffer.text == entry)


@pytest.mark.parametrize(("columns", "count"), [(21, 120), (31, 201), (41, 441)])
async def test_a_tall_chinese_draft_scrolls_to_its_cursor(
    columns: int, count: int
) -> None:
    text = "一二三四五六七八九十" * (count // 10) + "终"
    async with running_app(columns=columns, rows=30) as app:
        app.ui.buffer.text = text
        app.ui.buffer.cursor_position = len(text)
        await app.settle()
        box = _box(app)
        assert len(box) == 10
        assert "终" in box[-1] or "终" in box[-2], box


def _bar_line(width: int, mode: str = "ask") -> str:
    bar = StatusBar("stub-model", "~/demo-project", "ctx 120", "fail please", mode)
    return "".join(fragment[1] for fragment in status_fragments(bar, width, "…"))


def test_a_dropped_context_gives_its_room_back_to_the_workspace() -> None:
    assert _bar_line(30) == "  stub-model · ~/demo-pro… ask"


def test_a_bar_without_a_mode_gives_its_room_to_the_rest() -> None:
    # The cells and the gap kept for "ask" are the folder's: it is not cut at 41 columns.
    assert _bar_line(41, "") == "  stub-model · ~/demo-project · ctx 120  "
    assert "~/demo-proj… · ctx 120" in _bar_line(41)


@pytest.mark.parametrize("width", [16, 20, 25, 30, 37, 39, 41, 60])
def test_a_bar_without_a_mode_never_overflows(width: int) -> None:
    line = _bar_line(width, "")
    assert cells(line) == width and not line.strip().startswith("…"), line


@pytest.mark.parametrize("width", [16, 20, 22, 25, 27, 30, 40])
def test_the_status_bar_never_overflows_or_shows_a_bare_ellipsis(width: int) -> None:
    for mode in ("ask", "⚠ YOLO: no approvals"):
        line = _bar_line(width, mode)
        assert cells(line) == width, (width, mode, line)
        assert "YOLO" in line or mode == "ask"
        assert (
            "……" not in line and " … " not in line and not line.strip().startswith("…")
        )


async def test_the_placeholder_is_cut_with_an_ellipsis_when_narrow() -> None:
    async with running_app(columns=20) as app:
        await app.settle()
        assert _box(app) == ["› Ask melo · / for…"]


def test_a_closed_terminal_closes_the_input() -> None:
    # prompt_toolkit reads EIO as a resize: the app would wait forever.
    import os
    import tty

    from localmelo.support.cli.app import terminal_input

    master, slave = os.openpty()
    tty.setraw(slave)  # as while the app runs
    closes: list[bool] = []
    with os.fdopen(slave, "r", encoding="utf-8", closefd=True) as stdin:
        reading = terminal_input(stdin, on_close=lambda: closes.append(True))
        os.write(master, b"hi")
        assert reading.read_keys()[0].data == "h"
        assert closes == []
        os.close(master)
        assert reading.read_keys() == []
        assert reading.read_keys() == []
        assert reading.closed and closes == [True]  # told once


# ── Enter takes the completion picked with Tab or the arrows ──

LISTED = ["stub-model", "qwen3-8b", "llama-3.1"]


@pytest.mark.parametrize(
    ("typed", "keys", "entered"),
    [
        ("/he", "\t", "/help"),
        ("/model q", "\t", "/model qwen3-8b"),
        ("/model ll", "\x1b[B", "/model llama-3.1"),
        ("/", "\x1b[B\x1b[B", "/backend"),
    ],
)
async def test_enter_takes_the_picked_completion(
    typed: str, keys: str, entered: str
) -> None:
    controller = FakeController()
    completer = SlashCompleter(lambda: LISTED)
    async with running_app(controller, completer=completer) as app:
        app.type(typed)
        await app.wait(lambda: app.ui.buffer.complete_state is not None)
        app.type(keys)
        await app.wait(lambda: app.ui.buffer.text == entered)
        app.type(ENTER)
        await app.wait(lambda: controller.entered == [entered])


async def test_a_question_does_not_walk_the_message_history() -> None:
    from prompt_toolkit.history import InMemoryHistory

    history = InMemoryHistory()
    for entry in ("List the files here", "/backend"):
        history.append_string(entry)
    async with running_app(history=history) as app:
        await app.settle()
        asking = asyncio.ensure_future(app.ui.ask("Model server URL", "http://h/v1"))
        await app.wait(lambda: "Model server URL [http://h/v1]:" in app.text())
        for key in (UP, UP, "\x10", "\x1b[B", "\x0e"):  # up, ctrl+p, down, ctrl+n
            app.type(key)
            await app.settle(0.1)
            assert app.ui.buffer.text == "", key
        app.type(ENTER)
        await app.wait(asking.done)
        assert asking.result() == "http://h/v1"
        app.type(UP)  # at the input box, history again
        await app.wait(lambda: app.ui.buffer.text == "/backend")


async def test_ctrl_z_does_nothing_in_the_input_box() -> None:
    controller = FakeController()
    async with running_app(controller) as app:
        app.type("abc\x1a")
        await app.settle()
        assert app.ui.buffer.text == "abc"
        app.type(ENTER)
        await app.wait(lambda: controller.entered == ["abc"])


def test_control_characters_never_push_the_mode_off_the_status_bar() -> None:
    bar = StatusBar("stub-model", "~/demo-project", "ctx 120", "abc\x1a\x1b[2J", "ask")
    line = "".join(fragment[1] for fragment in status_fragments(bar, 70, "…"))
    assert line.endswith("ask") and cells(line) == 70, line
    assert "abc^Z^[[2J" in line and "\x1a" not in line


async def test_control_characters_in_the_queued_line_and_the_hint_are_spelled() -> None:
    controller = FakeController(state="running", queued="one\x1atwo")
    async with running_app(controller, columns=40) as app:
        app.ui.hint("hint\x07here")
        await app.settle()
        assert "Queued for the next turn: one^Ztwo" in app.text()
        assert "hint^Ghere" in app.text()


# ── The input box fits the terminal's rows ──

PASTE = "\x1b[200~" + "\n".join(f"line {n}" for n in range(1, 16)) + "\x1b[201~"


async def test_a_tall_draft_on_a_short_terminal_keeps_the_ui() -> None:
    async with running_app(columns=60, rows=12) as app:
        app.type(PASTE)
        await app.wait(lambda: app.ui.buffer.text.endswith("line 15"))
        await app.settle()
        screen = app.screen()
        assert "Window too small" not in "\n".join(screen)
        rows = [row for row in screen if row.strip()]
        assert "stub-model" in rows[-1]  # the status bar
        assert any("line 15" in row for row in rows)  # the cursor's row


async def test_a_tall_draft_during_a_turn_keeps_the_live_rows() -> None:
    now = [50.0]
    controller = FakeController(state="running", queued="queued one")
    async with running_app(
        controller, columns=80, rows=15, clock=lambda: now[0]
    ) as app:
        app.ui.renderer.live = LiveState(
            activity="Running",
            started=40.0,
            step=2,
            tool="shell_exec",
            preview="pytest -q",
            tool_started=47.0,
            tool_limit_s=600,
        )
        app.type(PASTE)
        await app.wait(lambda: app.ui.buffer.text.endswith("line 15"))
        await app.settle()
        text = app.text()
        assert "Window too small" not in text
        assert "esc to stop" in text and "of 10:00" in text  # both live rows
        assert "Queued for the next turn: queued one" in text
        assert any("line 15" in row for row in app.screen())


@pytest.mark.parametrize(
    ("typed", "completed"),
    [
        ("/model q\t", "/model qwen3-8b"),
        ("/he\t", "/help"),
        ("/mo\t", "/model"),
        ("/\t", "/model"),
        ("/\t\t", "/backend"),
    ],
)
async def test_tab_completes_at_once_before_the_menu_shows(
    typed: str, completed: str
) -> None:
    controller = FakeController()
    completer = SlashCompleter(lambda: LISTED)
    async with running_app(controller, completer=completer) as app:
        app.type(typed)  # Tab in the same keystrokes as the text
        await app.wait(lambda: app.ui.buffer.text == completed)
        app.type(ENTER)
        await app.wait(lambda: controller.entered == [completed])


async def test_printing_above_the_app_keeps_the_terminal_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In cooked mode the terminal echoes keys typed meanwhile and turns
    # Enter into Ctrl+J, a new line: a message typed then was not sent.
    from rich.text import Text

    from localmelo.support.cli.output import TerminalOut

    async with running_app() as app:
        cooked: list[bool] = []
        real = app.ui.app.input.cooked_mode

        def cooked_mode() -> Any:
            cooked.append(True)
            return real()

        monkeypatch.setattr(app.ui.app.input, "cooked_mode", cooked_mode)
        out = TerminalOut(app.ui.app.output, no_color=True)
        out.print(Text("printed above"))
        await app.wait(lambda: "printed above" in app.text())
        await app.wait(lambda: "Ask melo" in app.text())
        assert cooked == []
        screen = app.screen()
        assert screen.index("printed above") < screen.index(
            "› Ask melo · / for commands"
        )


ROW = ("cache 83% idle 3m", "ctx 12k/32k", "step 4")


def _row_line(width: int, segments: tuple[str, ...] = ROW) -> str:
    bar = StatusBar(
        "stub-model", "~/demo-project", "", "fail please", "ask", segments=segments
    )
    line = "".join(fragment[1] for fragment in status_fragments(bar, width, "…"))
    assert len(line) == width  # nothing wraps, and the mode stays at the edge
    return line


def test_the_status_row_shows_every_segment_at_120_columns() -> None:
    line = _row_line(120)
    assert line.index("cache 83% idle 3m") < line.index("ctx 12k/32k")
    assert line.index("ctx 12k/32k") < line.index("step 4") < line.index("fail please")
    assert line.endswith(" ask")


def test_the_status_row_drops_the_title_then_segments_right_to_left() -> None:
    shown_at = {
        width: [part for part in (*ROW, "fail please") if part in _row_line(width)]
        for width in (120, 80, 70, 60, 50)
    }
    assert shown_at[120] == [*ROW, "fail please"]
    assert shown_at[80] == list(ROW)
    assert shown_at[70] == list(ROW[:2])
    assert shown_at[60] == list(ROW[:1])
    assert shown_at[50] == []
    assert "stub-model · ~/demo-project" in _row_line(50)


@pytest.mark.parametrize("width", [8, 10, 16, 20, 25, 30, 40, 60])
def test_the_status_row_never_wraps_at_any_width(width: int) -> None:
    line = _row_line(width)
    if "cache" in line:
        assert "stub-model" in line and "~/demo" in line


def test_without_segments_the_bar_is_the_old_one() -> None:
    bar = StatusBar("stub-model", "~/demo-project", "ctx 120", "fail please", "ask")
    left = "  stub-model · ~/demo-project · ctx 120 · fail please"
    expected = left + " " * (120 - len(left) - len("ask")) + "ask"
    assert "".join(f[1] for f in status_fragments(bar, 120, "…")) == expected
