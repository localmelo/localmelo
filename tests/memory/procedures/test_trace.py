"""The trace of a run: calls, segments, and the rebuild from segments."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from localmelo.melo.memory.procedures.classify import FINISHING, TestSummary
from localmelo.melo.memory.procedures.trace import (
    Segment,
    Trace,
    TraceCall,
    first_steps,
    trace_from_segments,
)


class Clock:
    """A clock that returns the times it is given, one per call."""

    def __init__(self, *times: float) -> None:
        self._times: Iterator[float] = iter(times)

    def __call__(self) -> float:
        return next(self._times)


def read(path: str) -> tuple[str, dict[str, str]]:
    return "file_read", {"path": path}


def run_tests_trace() -> Trace:
    trace = Trace(clock=Clock(*(float(i) for i in range(100))))
    trace.step(1, 1)
    trace.tool(*read("/p/test_parser.py"), "success", "def test(): ...")
    trace.tool(*read("/p/parser.py"), "success", "code")
    trace.step(2, 1)
    trace.tool("file_write", {"path": "/p/parser.py", "content": "x"}, "success", "")
    trace.step(3, 1)
    trace.tool("shell_exec", {"command": "pytest -q"}, "success", "1 failed, 7 passed")
    trace.step(4, 1)
    trace.tool("file_write", {"path": "/p/parser.py", "content": "y"}, "success", "")
    trace.step(5, 1)
    trace.tool("shell_exec", {"command": "pytest -q"}, "success", "8 passed in 0.1s")
    trace.step(6, 1)
    return trace


def test_tool_classifies_labels_and_reads_the_summary() -> None:
    trace = run_tests_trace()
    assert [call.cls for call in trace.calls] == ["R", "R", "W", "T", "W", "T"]
    assert [call.step for call in trace.calls] == [1, 1, 2, 3, 4, 5]
    assert trace.calls[0].label == "Read test_parser.py"
    assert trace.calls[3].tests == TestSummary(7, 1, 0)
    assert not trace.calls[3].passed
    assert trace.calls[5].passed
    assert trace.tests == TestSummary(8, 0, 0)
    assert trace.first_pass_step == 5
    assert trace.phase == FINISHING
    assert trace.cycles == 2
    assert trace.step_number == 6
    assert trace.first_step_of("W") == 2
    assert trace.first_step_of("X") is None


def test_no_output_is_kept() -> None:
    trace = Trace(clock=lambda: 0.0)
    trace.step(1, 1)
    trace.tool(
        "shell_exec", {"command": "pytest"}, "success", "SECRET " * 100 + "1 passed"
    )
    trace.tool("shell_exec", {"command": "cat f"}, "success", "SECRET text")
    assert "SECRET" not in repr(trace)


def test_a_test_summary_outside_the_tail_is_not_seen() -> None:
    trace = Trace(clock=lambda: 0.0)
    trace.step(1, 1)
    trace.tool("shell_exec", {"command": "pytest"}, "success", "5 passed\n" + "." * 500)
    assert trace.tests is None
    assert trace.first_pass_step is None


def test_a_failed_call_is_never_a_pass() -> None:
    trace = Trace(clock=lambda: 0.0)
    trace.step(1, 1)
    call = trace.tool(
        "shell_exec", {"command": "pytest"}, "success", "[exit code 1]\n3 passed"
    )
    assert call.failed and not call.passed
    call = trace.tool("shell_exec", {"command": "pytest"}, "error", "3 passed")
    assert call.failed and not call.passed


def test_segments_group_consecutive_calls_of_one_class_and_target() -> None:
    segments = run_tests_trace().segments()
    assert [(s.cls, s.target, s.steps, s.calls) for s in segments] == [
        ("R", "test_parser.py", (1, 1), 1),
        ("R", "parser.py", (1, 1), 1),
        ("W", "parser.py", (2, 2), 1),
        ("T", "pytest", (3, 3), 1),
        ("W", "parser.py", (4, 4), 1),
        ("T", "pytest", (5, 5), 1),
    ]
    assert [s.passed for s in segments] == [False] * 5 + [True]


def test_segment_wall_time_runs_from_its_first_step_to_its_last_call() -> None:
    trace = Trace(clock=Clock(0.0, 1.0, 2.5, 4.0, 10.0))
    trace.step(1, 1)  # 0.0
    trace.tool(*read("/a"), "success", "")  # 1.0
    trace.step(2, 1)  # 2.5
    trace.tool(*read("/a"), "success", "")  # 4.0
    trace.tool("python_exec", {"code": "1"}, "success", "")  # 10.0
    first, second = trace.segments()
    assert first.steps == (1, 2) and first.calls == 2
    assert first.wall_s == pytest.approx(4.0)
    assert second.wall_s == pytest.approx(7.5)


def test_reflections_and_stuck_record_their_step() -> None:
    trace = Trace(clock=lambda: 0.0)
    trace.step(1, 1)
    trace.step(2, 1)
    trace.reflection()
    trace.step(3, 2)
    trace.stuck()
    trace.stuck()
    assert trace.reflections == [2]
    assert trace.stuck_at == 3
    assert trace.attempt == 2


def test_segment_round_trip_and_validation() -> None:
    segment = Segment(
        "W", "Edit a.py", "a.py", (2, 4), calls=3, attempt=2, wall_s=1.25, passed=False
    )
    data = segment.as_dict()
    assert data == {
        "cls": "W",
        "label": "Edit a.py",
        "target": "a.py",
        "steps": [2, 4],
        "calls": 3,
        "attempt": 2,
        "wall_s": 1.25,
        "pass": False,
    }
    assert Segment.from_dict(data) == segment
    # Defaults for the keys the spec's example leaves out.
    minimal = Segment.from_dict({"cls": "R", "label": "x", "steps": [1, 1]})
    assert (minimal.calls, minimal.attempt, minimal.target) == (1, 1, None)
    for bad in (
        {"cls": "Q", "steps": [1, 1]},
        {"cls": "R"},
        {"cls": "R", "steps": [3, 2]},
        {"cls": "R", "steps": [0, 1]},
        {"cls": "R", "steps": [1, 1], "calls": 0},
        {"cls": "R", "steps": "ab"},
    ):
        with pytest.raises(ValueError):
            Segment.from_dict(bad)


def test_trace_from_segments_spreads_calls_over_their_steps() -> None:
    segments = [
        Segment("R", "", None, (1, 2), calls=2),
        Segment("W", "", None, (3, 3), calls=3),
        Segment("T", "", None, (4, 6), calls=2, passed=True),
    ]
    trace = trace_from_segments(segments, 7, first_pass_step=5)
    assert [(c.cls, c.step) for c in trace.calls] == [
        ("R", 1),
        ("R", 2),
        ("W", 3),
        ("W", 3),
        ("W", 3),
        ("T", 4),
        ("T", 5),
    ]
    assert [c.passed for c in trace.calls][-2:] == [False, True]
    assert trace.first_pass_step == 5
    assert trace.step_number == 7


def test_trace_from_segments_places_reflections_where_the_attempt_rises() -> None:
    segments = [
        Segment("R", "", None, (1, 2), calls=2, attempt=1),
        Segment("W", "", None, (3, 3), attempt=2),
        Segment("T", "", None, (4, 4), attempt=2, passed=True),
    ]
    trace = trace_from_segments(segments, 4)
    assert trace.reflections == [2]
    assert [c.attempt for c in trace.calls] == [1, 1, 2, 2]
    # The pass goes on the segment's last call without a first_pass_step.
    assert trace.first_pass_step == 4


def test_trace_from_segments_keeps_calls_beyond_the_last_step() -> None:
    trace = trace_from_segments([Segment("R", "", None, (1, 5), calls=2)], 2)
    assert len(trace.calls) == 2
    assert trace.step_number == 2


def test_first_steps() -> None:
    segments = [
        Segment("R", "", None, (1, 1)),
        Segment("W", "", None, (2, 3), calls=2),
        Segment("T", "", None, (4, 4)),
        Segment("W", "", None, (5, 5)),
        Segment("T", "", None, (6, 7), calls=2, passed=True),
    ]
    assert first_steps(segments) == {"W": 2, "T": 4, "P": 7}
    assert first_steps([]) == {}


def test_add_accepts_a_classified_call() -> None:
    trace = Trace(clock=lambda: 0.0)
    trace.step(1, 1)
    trace.add(
        TraceCall(
            step=1, attempt=1, cls="T", label="pytest", target="pytest", passed=True
        )
    )
    assert trace.first_pass_step == 1
    assert trace.phase == FINISHING
