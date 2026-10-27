"""The progress line's text and its display, on synthetic events.

:func:`format_line` is pure: every level and state is checked on
synthetic ``run.progress`` payloads, level 2 and the off-template states
included, since a real run cannot reach level 2 while
``tracker.LEVEL2_ENABLED`` is off. :class:`ProgressLine` is checked on a
stream in memory with a clock the test sets.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from unittest import mock

import pytest

from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.procedures.tracker import PROGRESS_KEYS
from localmelo.support import progress_line as pl
from localmelo.support.progress_line import (
    DIM,
    ERASE,
    RESET,
    ProgressLine,
    RunningTool,
    format_line,
)

WIDE = 400

LIMITS = {
    "max_steps": 30,
    "steps_per_attempt": 10,
    "max_attempts": 5,
    "timeout_s": None,
}

ONE_PAST_RUN = {
    "runs": 1,
    "steps_lo": 6,
    "steps_median": 6.0,
    "steps_hi": 6,
    "wall_lo_s": 130.0,
    "wall_hi_s": 130.0,
    "last_steps": 6,
    "last_wall_s": 130.0,
}

THREE_PAST_RUNS = {
    "runs": 3,
    "steps_lo": 5,
    "steps_median": 6.0,
    "steps_hi": 8,
    "wall_lo_s": 130.0,
    "wall_hi_s": 170.0,
    "last_steps": 6,
    "last_wall_s": 150.0,
}

ITEMS: list[dict[str, Any]] = [
    {"label": "Read test_parser.py", "state": "done", "repeat": None},
    {"label": "Edit parser.py + pytest", "state": "current", "repeat": 2},
    {"label": "git commit", "state": "pending", "repeat": None},
]


def payload(**data: Any) -> dict[str, Any]:
    """A run.progress payload with every key: level 0, step 0, until *data*."""
    base: dict[str, Any] = {
        "level": 0,
        "state": "running",
        "step": 0,
        "attempt": 0,
        "limits": dict(LIMITS),
        "phase": None,
        "cycle": 0,
        "tests": None,
        "items": None,
        "fraction": None,
        "expected_steps": None,
        "past": None,
        "off_reason": None,
        "basis": None,
    }
    assert set(base) == set(PROGRESS_KEYS)
    base.update(data)
    return base


def level0(**data: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "step": 4,
        "attempt": 1,
        "phase": "Verify",
        "cycle": 1,
        "tests": {"passed": 3, "failed": 5, "total": 8},
    }
    fields.update(data)
    return payload(**fields)


def level1(**data: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "level": 1,
        "step": 4,
        "attempt": 1,
        "phase": "Write",
        "cycle": 1,
        "items": [dict(item) for item in ITEMS],
        "past": dict(ONE_PAST_RUN),
    }
    fields.update(data)
    return payload(**fields)


def level2(**data: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "level": 2,
        "step": 4,
        "attempt": 1,
        "phase": "Verify",
        "cycle": 1,
        "tests": {"passed": 7, "failed": 1, "total": 8},
        "items": [
            {"label": "Read test_parser.py", "state": "done", "repeat": None},
            {"label": "Edit parser.py", "state": "current", "repeat": None},
            {"label": "pytest", "state": "pending", "repeat": None},
        ],
        "fraction": 0.5714,
        "expected_steps": 7,
        "past": {
            "runs": 2,
            "steps_lo": 6,
            "steps_median": 7.0,
            "steps_hi": 8,
            "wall_lo_s": 130.0,
            "wall_hi_s": 170.0,
            "last_steps": 8,
            "last_wall_s": 170.0,
        },
    }
    fields.update(data)
    return payload(**fields)


def line(
    data: Mapping[str, Any] | None,
    *,
    tool: RunningTool | None = None,
    elapsed: float = 63.0,
    width: int = WIDE,
    color: bool = False,
) -> str:
    """format_line at 100 s on the clock, the run *elapsed* seconds old."""
    return format_line(
        data, tool, 100.0, width, run_started=100.0 - elapsed, color=color
    )


PYTEST_41S = RunningTool(label="pytest", started=100.0 - 41)


# ── Text helpers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (9.9, "9s"), (59.99, "59s"), (63, "1m03s"), (3600, "1h00m")]
    + [(3720, "1h02m"), (-5, "0s")],
)
def test_run_elapsed(seconds: float, text: str) -> None:
    assert pl.run_elapsed(seconds) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0:00"), (41.7, "0:41"), (723, "12:03"), (3723, "1:02:03")],
)
def test_tool_elapsed(seconds: float, text: str) -> None:
    assert pl.tool_elapsed(seconds) == text


@pytest.mark.parametrize(
    ("lo", "hi", "text"),
    [
        (40, 55, "40-55s"),
        (45, 45, "45s"),
        (40.2, 54.1, "40-55s"),  # the range holds both ends
        (130, 170, "2-3 min"),
        (120, 120, "2 min"),
        (125, 130, "2-3 min"),
        (45, 130, "45s-3 min"),
        (170, 130, "2-3 min"),  # either order
    ],
)
def test_wall_range_holds_both_ends(lo: float, hi: float, text: str) -> None:
    assert pl.wall_range(lo, hi) == text


def test_display_width_counts_wide_characters_twice() -> None:
    assert pl.display_width("parser.py") == 9
    assert pl.display_width("测试.py") == 7
    assert pl.display_width("ｆｕｌｌ") == 8  # full-width forms
    assert pl.display_width("é") == 1  # a combining accent takes no cell


@pytest.mark.parametrize("width", range(0, 12))
def test_cut_to_width_never_overflows(width: int) -> None:
    text = "Read 测试文件.py"
    cut = pl.cut_to_width(text, width)
    assert pl.display_width(cut) <= width
    if pl.display_width(text) > width and width > 0:
        assert cut.endswith("…")


def test_printable_replaces_control_and_format_characters() -> None:
    assert pl.printable("Read \x1b[2Jx.py") == "Read ?[2Jx.py"
    assert pl.printable("Edit ‮gnp.exe") == "Edit ?gnp.exe"  # bidi override
    assert pl.printable("a\nb\tc") == "a b c"


# ── Level 0: facts ──────────────────────────────────────────────────────────


def test_level_0_shows_the_facts() -> None:
    assert line(level0(), tool=PYTEST_41S) == (
        "step 4 · Verify (cycle 1) · pytest 0:41 · tests 3/8 · 1m03s"
    )


def test_attempt_1_is_not_shown() -> None:
    assert "attempt" not in line(level0(attempt=1))


def test_a_later_attempt_follows_the_step() -> None:
    assert line(level0(attempt=2)).startswith("step 4 · attempt 2 · Verify")


def test_no_cycle_before_the_first() -> None:
    text = line(level0(phase="Explore", cycle=0))
    assert "Explore · " in text
    assert "cycle" not in text


def test_no_phase_before_the_first_tool_call() -> None:
    assert line(level0(phase=None, cycle=0, tests=None)) == "step 4 · 1m03s"


def test_tests_only_when_counted() -> None:
    assert "tests" not in line(level0(tests=None))
    assert "tests 3/8" in line(level0())


def test_the_tool_timer_only_while_a_tool_runs() -> None:
    assert "pytest" not in line(level0())
    assert "pytest 0:41" in line(level0(), tool=PYTEST_41S)


def test_before_the_first_event_the_time_only() -> None:
    assert line(None, elapsed=0) == "starting · 0s"
    assert line({}, elapsed=3) == "starting · 3s"
    assert line(payload(), elapsed=3) == "starting · 3s"


def test_never_step_of_the_budget() -> None:
    text = line(level0(step=4, limits={**LIMITS, "max_steps": 30}))
    assert "/30" not in text
    assert "of" not in text


@pytest.mark.parametrize("level", [0, 1])
def test_never_a_percent_or_a_bar_below_level_2(level: int) -> None:
    # A malformed payload with a fraction still gets no percent.
    data = level1(level=level, fraction=0.57, expected_steps=7)
    text = line(data)
    assert "%" not in text
    assert "▓" not in text and "░" not in text
    assert "of ~" not in text


# ── Level 1: past facts and named steps ─────────────────────────────────────


def test_level_1_after_one_past_run() -> None:
    assert line(level1()) == (
        "step 4 · Write (cycle 1) · 1m03s · last time 6 steps, 2m10s: "
        "Read test_parser.py ✓ · Edit parser.py + pytest (×2) ◐ · git commit"
    )


def test_level_1_after_several_past_runs() -> None:
    assert line(level1(past=dict(THREE_PAST_RUNS))) == (
        "step 4 · Write (cycle 1) · 1m03s · past 5-8 steps, 2-3 min · "
        "last time: Read test_parser.py ✓ · Edit parser.py + pytest (×2) ◐ · "
        "git commit"
    )


def test_level_1_past_runs_of_one_length() -> None:
    past = {**THREE_PAST_RUNS, "steps_lo": 6, "steps_hi": 6}
    assert "past 6 steps, 2-3 min" in line(level1(past=past))


def test_level_1_one_step_is_singular() -> None:
    past = {**ONE_PAST_RUN, "last_steps": 1, "last_wall_s": 9.0}
    assert "last time 1 step, 9s:" in line(level1(past=past))


def test_level_1_items_marks() -> None:
    items = [
        {"label": "a", "state": "done", "repeat": None},
        {"label": "b", "state": "skipped", "repeat": 1},
        {"label": "c", "state": "current", "repeat": 3},
        {"label": "d", "state": "pending", "repeat": None},
    ]
    assert line(level1(items=items)).endswith(": a ✓ · b – · c (×3) ◐ · d")


def test_level_1_without_items_or_past() -> None:
    assert line(level1(items=None)) == (
        "step 4 · Write (cycle 1) · 1m03s · last time 6 steps, 2m10s"
    )
    assert line(level1(items=[], past=None)) == "step 4 · Write (cycle 1) · 1m03s"
    assert line(level1(past=None)) == (
        "step 4 · Write (cycle 1) · 1m03s · last time: Read test_parser.py ✓ · "
        "Edit parser.py + pytest (×2) ◐ · git commit"
    )


def test_level_1_before_the_first_step_shows_what_to_expect() -> None:
    items = [{**item, "state": "pending"} for item in ITEMS]
    data = level1(step=0, attempt=0, phase=None, cycle=0, items=items)
    assert line(data, elapsed=0) == (
        "starting · 0s · last time 6 steps, 2m10s: "
        "Read test_parser.py · Edit parser.py + pytest (×2) · git commit"
    )


def test_item_labels_cannot_write_escape_sequences() -> None:
    items = [{"label": "Read \x1b]0;pwned\x07x.py", "state": "done", "repeat": None}]
    text = line(level1(items=items))
    assert "\x1b" not in text and "\x07" not in text


def test_every_payload_is_read_afresh() -> None:
    # Facts at level 1, then a payload without them: nothing is kept.
    assert "last time" in line(level1())
    assert line(level0(tests=None, phase=None)) == "step 4 · 1m03s"


# ── Level 2 and off the template (synthetic: level 2 is off) ────────────────


def test_level_2_shows_the_bar() -> None:
    assert line(level2()) == (
        "▓▓▓▓▓▓░░░░ 57% · step 4 of ~7 · Verify (cycle 1) · tests 7/8 · 1m03s · "
        "past runs 2-3 min · Read test_parser.py ✓ · Edit parser.py ◐ · pytest ○"
    )


def test_level_2_running_never_shows_more_than_95() -> None:
    for fraction in (0.95, 0.99, 1.0, 3.0):
        text = line(level2(fraction=fraction))
        assert text.startswith("▓▓▓▓▓▓▓▓▓░ 95% · ")
        assert "100%" not in text


def test_level_2_longer_than_usual_freezes_the_bar() -> None:
    data = level2(
        state="off_template",
        off_reason="longer",
        step=9,
        fraction=0.89,
        expected_steps=None,
    )
    assert line(data).startswith(
        "▓▓▓▓▓▓▓▓▓░ paused at 89% · step 9, longer than usual (past 6-8) · Verify"
    )


@pytest.mark.parametrize("reason", ["retrying", "stuck"])
def test_level_2_retrying_and_stuck(reason: str) -> None:
    data = level2(
        state="off_template",
        off_reason=reason,
        step=9,
        attempt=2,
        fraction=0.8,
        expected_steps=None,
    )
    assert line(data).startswith(
        f"▓▓▓▓▓▓▓▓░░ paused at 80% · step 9, {reason} · attempt 2 · Verify"
    )


def test_level_2_ended_completed_is_100() -> None:
    data = level2(state="ended", fraction=1.0, expected_steps=None, step=7)
    assert line(data).startswith("▓▓▓▓▓▓▓▓▓▓ 100% · step 7 · ")


def test_level_2_ended_otherwise_keeps_its_value() -> None:
    data = level2(state="ended", fraction=0.57, expected_steps=None, step=7)
    text = line(data)
    assert text.startswith("▓▓▓▓▓▓░░░░ stopped at 57% · step 7 · ")
    assert "100%" not in text


def test_level_2_before_the_first_step() -> None:
    data = level2(
        step=0, attempt=0, fraction=0.0, expected_steps=None, phase=None, tests=None
    )
    assert line(data, elapsed=0).startswith("░░░░░░░░░░ 0% · starting · 0s")


# ── Width ───────────────────────────────────────────────────────────────────


def test_parts_are_left_out_in_order_as_the_width_shrinks() -> None:
    data = level1(
        attempt=2,
        tests={"passed": 3, "failed": 5, "total": 8},
        past=dict(THREE_PAST_RUNS),
    )
    stages = [
        "step 4 · attempt 2 · Write (cycle 1) · tests 3/8 · 1m03s · "
        "past 5-8 steps, 2-3 min · last time: Read test_parser.py ✓ · "
        "Edit parser.py + pytest (×2) ◐ · git commit",
        # past times first
        "step 4 · attempt 2 · Write (cycle 1) · tests 3/8 · 1m03s · "
        "past 5-8 steps · last time: Read test_parser.py ✓ · "
        "Edit parser.py + pytest (×2) ◐ · git commit",
        # then tests
        "step 4 · attempt 2 · Write (cycle 1) · 1m03s · past 5-8 steps · "
        "last time: Read test_parser.py ✓ · Edit parser.py + pytest (×2) ◐ · "
        "git commit",
        # then the items
        "step 4 · attempt 2 · Write (cycle 1) · 1m03s · past 5-8 steps",
        # then the attempt
        "step 4 · Write (cycle 1) · 1m03s · past 5-8 steps",
    ]
    for stage in stages:
        assert line(data, width=pl.display_width(stage)) == stage
    # Narrower still: cut, with an ellipsis.
    last = stages[-1]
    for width in (len(last) - 1, 20, 8, 1):
        text = line(data, width=width)
        assert pl.display_width(text) <= width
        assert text.endswith("…")
        assert text[:-1] == last[: len(text) - 1]


def test_one_past_run_keeps_its_steps_last() -> None:
    narrow = "step 4 · Write (cycle 1) · 1m03s · last time 6 steps"
    assert line(level1(), width=len(narrow)) == narrow


def test_a_wide_file_name_does_not_overflow() -> None:
    tool = RunningTool(label="Read 测试数据文件名很长很长的文件.py", started=99.0)
    for width in (20, 31, 40, 57):
        text = line(level0(), tool=tool, width=width)
        assert pl.display_width(text) <= width, (width, text)


def test_the_level_2_bar_is_cut_last() -> None:
    text = line(level2(), width=24)
    assert text.startswith("▓▓▓▓▓▓░░░░ 57% · step 4")
    assert pl.display_width(text) <= 24


# ── Color ───────────────────────────────────────────────────────────────────


def test_color_dims_the_line_without_using_its_width() -> None:
    plain = line(level0(), tool=PYTEST_41S, width=30)
    dim = line(level0(), tool=PYTEST_41S, width=30, color=True)
    assert dim == f"{DIM}{plain}{RESET}"
    assert "\x1b" not in plain


@pytest.mark.parametrize(
    ("environ", "color"),
    [({}, True), ({"NO_COLOR": "1"}, False), ({"NO_COLOR": ""}, True)],
)
def test_no_color(environ: dict[str, str], color: bool) -> None:
    assert pl.wants_color(environ) is color


# ── When to draw ────────────────────────────────────────────────────────────


class _Tty(io.StringIO):
    def __init__(self, tty: bool = True) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty

    def fileno(self) -> int:
        return 2


TERM = {"TERM": "xterm-256color"}


@pytest.mark.parametrize(
    ("mode", "query", "output_format", "stream", "environ", "drawn"),
    [
        (None, "q", "text", _Tty(), TERM, True),
        ("auto", "q", "text", _Tty(), TERM, True),
        ("auto", "q", "text", _Tty(), {}, True),  # TERM unset
        ("off", "q", "text", _Tty(), TERM, False),
        ("auto", "q", "json", _Tty(), TERM, False),
        ("auto", "", "text", _Tty(), TERM, False),
        ("auto", "q", "text", _Tty(tty=False), TERM, False),
        ("auto", "q", "text", _Tty(), {"TERM": "dumb"}, False),
        ("auto", "q", "text", object(), TERM, False),  # no isatty at all
    ],
)
def test_wants_line(
    mode: str | None,
    query: str,
    output_format: str,
    stream: Any,
    environ: dict[str, str],
    drawn: bool,
) -> None:
    assert (
        pl.wants_line(
            mode,
            query=query,
            output_format=output_format,
            stream=stream,
            environ=environ,
        )
        is drawn
    )


def test_terminal_columns_falls_back_to_80() -> None:
    assert pl.terminal_columns(io.StringIO()) == pl.FALLBACK_COLUMNS
    zero = os.terminal_size((0, 0))  # a new pseudo-terminal may say 0 x 0
    with mock.patch.object(pl.os, "get_terminal_size", return_value=zero):
        assert pl.terminal_columns(_Tty()) == pl.FALLBACK_COLUMNS
    wide = os.terminal_size((132, 40))
    with mock.patch.object(pl.os, "get_terminal_size", return_value=wide):
        assert pl.terminal_columns(_Tty()) == 132


# ── The display ─────────────────────────────────────────────────────────────


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Stream(io.StringIO):
    """Keeps every write; ``fail`` makes writes raise OSError."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []
        self.fail = False
        self.lock = threading.Lock()

    def write(self, text: str) -> int:
        if self.fail:
            raise OSError(5, "Input/output error")
        with self.lock:
            self.writes.append(text)
        return super().write(text)

    def draws(self) -> list[str]:
        """The lines drawn so far, in order."""
        with self.lock:
            text = "".join(self.writes)
        return [part for part in text.split(ERASE)[1:] if part]


def _event(type_: str, seq: int = 1, **data: Any) -> AgentEvent:
    return AgentEvent(type=type_, run_id="r", seq=seq, ts=0.0, data=data)


@pytest.fixture
def display() -> Iterator[tuple[ProgressLine, Stream, Clock]]:
    stream, clock = Stream(), Clock()
    line_ = ProgressLine(stream, interval_s=3600, clock=clock, columns=lambda: 200)
    try:
        yield line_, stream, clock
    finally:
        line_.close()


def _wait_for(check: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_a_run_drawn_from_start_to_finish(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, clock = display
    line_.start()
    line_.observe(_event("run.started", 1))
    line_.observe(_event("run.progress", 2, **payload()))
    clock.now += 2
    line_.observe(_event("step.started", 3, step=1, attempt=1))
    line_.observe(_event("run.progress", 4, **payload(step=1, attempt=1)))
    line_.observe(
        _event(
            "tool.started",
            5,
            step=1,
            call_id="c1",
            name="shell_exec",
            arguments={"command": "pytest -q"},
            preview="pytest -q",
        )
    )
    clock.now += 41
    line_.observe(_event("tool.finished", 6, step=1, call_id="c1", name="shell_exec"))
    line_.observe(
        _event(
            "run.progress",
            7,
            **payload(
                step=1,
                attempt=1,
                phase="Verify",
                tests={"passed": 3, "failed": 5, "total": 8},
            ),
        )
    )
    line_.observe(_event("run.finished", 8, result={}))
    assert stream.draws() == [
        "starting · 0s",
        "step 1 · 2s",
        "step 1 · pytest 0:00 · 2s",
        "step 1 · 43s",
        "step 1 · Verify · tests 3/8 · 43s",
    ]
    text = stream.getvalue()
    assert text.endswith(ERASE)  # erased at the end
    assert "\n" not in text  # never a newline


def test_the_tool_timer_and_the_run_timer_advance_between_events() -> None:
    stream, clock = Stream(), Clock()
    line_ = ProgressLine(stream, interval_s=0.01, clock=clock, columns=lambda: 200)
    line_.start()
    try:
        line_.observe(
            _event("tool.started", 1, call_id="c", name="python_exec", arguments={})
        )
        clock.now += 5
        assert _wait_for(lambda: stream.draws()[-1] == "starting · Python 0:05 · 5s")
        clock.now += 60
        assert _wait_for(lambda: stream.draws()[-1] == "starting · Python 1:05 · 1m05s")
    finally:
        line_.close()
    assert line_._thread is not None and not line_._thread.is_alive()


def test_nothing_is_drawn_after_the_end(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, clock = display
    line_.start()
    line_.observe(_event("run.finished", 1, result={}))
    after = stream.getvalue()
    line_.observe(_event("run.progress", 2, **payload(step=3)))
    line_.close()  # idempotent
    line_.write_above("a warning")  # still written, with no line below it
    assert stream.getvalue() == after + "a warning\n"
    assert after == ERASE + "starting · 0s" + ERASE


def test_close_before_start_writes_nothing() -> None:
    stream = Stream()
    line_ = ProgressLine(stream, columns=lambda: 80)
    line_.close()
    line_.start()
    assert stream.getvalue() == ""
    assert not line_.active


def test_a_payload_never_outlives_the_next(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, _ = display
    line_.start()
    line_.observe(_event("run.progress", 1, **level1()))
    line_.observe(_event("run.progress", 2, **payload(step=5)))
    draws = stream.draws()
    assert "last time 6 steps" in draws[-2]
    assert draws[-1] == "step 5 · 0s"


def test_a_tool_that_finished_leaves_no_timer(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, _ = display
    line_.start()
    line_.observe(
        _event("tool.started", 1, call_id="a", name="file_read", arguments={})
    )
    line_.observe(_event("tool.finished", 2, call_id="other"))  # not this call
    assert stream.draws()[-1] == "starting · Read 0:00 · 0s"
    line_.observe(_event("step.started", 3, step=2, attempt=1))
    assert stream.draws()[-1] == "step 2 · 0s"


def test_the_label_names_a_file_without_its_folder(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, _ = display
    line_.start()
    line_.observe(
        _event(
            "tool.started",
            1,
            call_id="a",
            name="file_write",
            arguments={"path": "/home/me/secret-project/parser.py", "content": "x"},
        )
    )
    assert stream.draws()[-1] == "starting · Edit parser.py 0:00 · 0s"


def test_an_unchanged_line_is_not_drawn_again(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, _ = display
    line_.start()
    for seq in range(5):
        line_.observe(_event("run.progress", seq, **payload()))
    assert stream.draws() == ["starting · 0s"]


def test_the_width_is_read_on_every_draw() -> None:
    stream, columns = Stream(), [200]
    line_ = ProgressLine(stream, interval_s=3600, columns=lambda: columns[0])
    line_.start()
    try:
        line_.observe(_event("run.progress", 1, **level1()))
        assert "git commit" in stream.draws()[-1]
        columns[0] = 41
        line_.observe(_event("run.progress", 2, **level1(step=5)))
        last = stream.draws()[-1]
        assert pl.display_width(last) <= 40  # one cell short of the width
    finally:
        line_.close()


def test_events_it_cannot_read_never_raise(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, _ = display
    line_.start()

    class Broken(dict[str, Any]):
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("broken")

    line_.observe(AgentEvent("tool.started", "r", 1, 0.0, Broken()))
    line_.observe(_event("run.progress", 2, level="x", step=None, items="nope"))
    line_.observe(_event("run.progress", 3, past={"runs": "many"}, level=1))
    assert stream.draws()[0] == "starting · 0s"


def test_a_terminal_that_fails_stops_the_drawing(
    display: tuple[ProgressLine, Stream, Clock],
) -> None:
    line_, stream, _ = display
    line_.start()
    stream.fail = True
    line_.observe(_event("run.progress", 1, **payload(step=2)))  # does not raise
    assert not line_.active
    stream.fail = False
    line_.observe(_event("run.progress", 2, **payload(step=3)))
    assert stream.draws() == ["starting · 0s"]


# ── Log records above the line ──────────────────────────────────────────────


@pytest.fixture
def quiet_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("localmelo.tests.progress_line")
    level = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield logger
    finally:
        logger.setLevel(level)


def test_a_warning_goes_above_the_line(
    display: tuple[ProgressLine, Stream, Clock], quiet_logger: logging.Logger
) -> None:
    line_, stream, _ = display
    root_handlers = list(logging.getLogger().handlers)
    line_.start()
    added = [h for h in logging.getLogger().handlers if h not in root_handlers]
    assert len(added) == 1 and added[0].level == logging.WARNING
    line_.observe(_event("run.progress", 1, **payload(step=2)))
    quiet_logger.info("not shown")
    quiet_logger.warning("Could not save the record: %s", "locked")
    assert stream.getvalue().endswith(
        ERASE
        + "step 2 · 0s"
        + ERASE
        + "Could not save the record: locked\n"
        + ERASE
        + "step 2 · 0s"
    )
    assert "not shown" not in stream.getvalue()
    line_.close()
    assert logging.getLogger().handlers == root_handlers


def test_a_warning_keeps_its_traceback(
    display: tuple[ProgressLine, Stream, Clock], quiet_logger: logging.Logger
) -> None:
    line_, stream, _ = display
    line_.start()
    try:
        raise ValueError("bad value")
    except ValueError:
        quiet_logger.error("Run crashed", exc_info=True)
    text = stream.getvalue()
    assert "Run crashed\nTraceback (most recent call last):" in text
    assert "ValueError: bad value\n" in text
    assert text.endswith(ERASE + "starting · 0s")


def test_no_handler_without_a_line() -> None:
    root_handlers = list(logging.getLogger().handlers)
    ProgressLine(Stream(), columns=lambda: 80)  # never started
    assert logging.getLogger().handlers == root_handlers
