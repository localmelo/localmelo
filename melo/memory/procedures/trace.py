"""The trace of one run: its steps, tool calls, reflections and stuck notices.

A :class:`Trace` lives in RAM for one run and builds no event. The agent
appends to it where it emits the matching events: :meth:`Trace.step`
after ``step.started``, :meth:`Trace.tool` after ``tool.finished``,
:meth:`Trace.reflection` after ``reflection.started`` and
:meth:`Trace.stuck` on a ``stuck`` notice. Each call is classified when it
is added; of its output, only a test call's last
:data:`~.classify.OUTPUT_TAIL` characters are read, for the test summary,
and none is kept.

:meth:`Trace.segments` groups the calls into segments, the episode's
record of what the run did, and :func:`trace_from_segments` rebuilds an
approximate trace from them for an offline replay.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from localmelo.melo.memory.procedures.classify import (
    CLASSES,
    PASS_ANCHOR,
    TEST,
    WRITE,
    PhaseState,
    TestSummary,
    call_failed,
    classify_call,
    clean_label,
    is_pass,
    label_for,
    output_tail,
    parse_test_summary,
    target_for,
)


@dataclass(frozen=True)
class TraceCall:
    """One tool call: its step, class, label, target and result.

    ``t`` is when it finished, in seconds since the trace started.
    """

    step: int
    attempt: int
    cls: str
    label: str
    target: str | None
    passed: bool = False
    failed: bool = False
    tests: TestSummary | None = None
    t: float = 0.0


@dataclass(frozen=True)
class Segment:
    """A maximal run of consecutive calls with the same class and target.

    ``steps`` are the first and last planning steps its calls span,
    ``calls`` how many calls it holds, ``attempt`` the attempt of its first
    call, ``wall_s`` the seconds from the start of its first call's step to
    the end of its last call, and ``passed`` whether one of its calls is a
    pass.
    """

    cls: str
    label: str
    target: str | None
    steps: tuple[int, int]
    calls: int = 1
    attempt: int = 1
    wall_s: float = 0.0
    passed: bool = False

    def as_dict(self) -> dict[str, Any]:
        """The segment as an entry of an episode's ``segments_json``."""
        return {
            "cls": self.cls,
            "label": self.label,
            "target": self.target,
            "steps": [self.steps[0], self.steps[1]],
            "calls": self.calls,
            "attempt": self.attempt,
            "wall_s": round(self.wall_s, 3),
            "pass": self.passed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Segment:
        """A segment from a ``segments_json`` entry; ``ValueError`` if malformed."""
        try:
            klass = str(data["cls"])
            if klass not in CLASSES:
                raise ValueError(f"unknown class {klass!r}")
            first, last = (int(step) for step in data["steps"])
            target = data.get("target")
            calls = int(data.get("calls", 1))
            attempt = int(data.get("attempt", 1))
            wall_s = float(data.get("wall_s", 0.0))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed segment: {exc}") from None
        if not (1 <= first <= last) or calls < 1 or attempt < 1:
            raise ValueError("malformed segment: bad steps, calls or attempt")
        if not math.isfinite(wall_s):
            wall_s = 0.0
        return cls(
            cls=klass,
            label=clean_label(str(data.get("label", ""))),
            target=None if target is None else clean_label(str(target)),
            steps=(first, last),
            calls=calls,
            attempt=attempt,
            wall_s=max(0.0, wall_s),
            passed=bool(data.get("pass", False)),
        )


@dataclass
class Trace:
    """The steps and tool calls of one run, in order. See the module docstring."""

    clock: Callable[[], float] = time.monotonic
    calls: list[TraceCall] = field(default_factory=list)
    # The step at which each reflection started, and the stuck notice's.
    reflections: list[int] = field(default_factory=list)
    stuck_at: int | None = None
    step_number: int = 0
    attempt: int = 0
    tests: TestSummary | None = None
    first_pass_step: int | None = None
    phases: PhaseState = field(default_factory=PhaseState)
    # When each step started, in seconds since the trace started.
    step_starts: dict[int, float] = field(default_factory=dict)
    _origin: float | None = None

    def _now(self) -> float:
        now = self.clock()
        if self._origin is None:
            self._origin = now
        return now - self._origin

    def step(self, k: int, attempt: int) -> None:
        """Step *k* of attempt *attempt* started."""
        self.step_number = k
        self.attempt = attempt
        self.step_starts.setdefault(k, self._now())

    def tool(self, name: str, arguments: Any, status: str, output: str) -> TraceCall:
        """A call to *name* with *arguments* finished with *status* and *output*.

        The call is classified and labelled; a test call's summary is read
        from its output's tail. The output itself is not kept.
        """
        cls = classify_call(name, arguments)
        label = label_for(name, arguments)
        failed = call_failed(status, output)
        summary = parse_test_summary(output_tail(output)) if cls == TEST else None
        call = TraceCall(
            step=self.step_number,
            attempt=self.attempt,
            cls=cls,
            label=label,
            target=target_for(name, arguments, label),
            passed=is_pass(cls, summary, failed),
            failed=failed,
            tests=summary,
            t=self._now(),
        )
        self.add(call)
        return call

    def add(self, call: TraceCall) -> None:
        """Append a classified *call*: the phases, tests and first pass follow it."""
        self.calls.append(call)
        self.phases.add(call.cls, call.passed)
        if call.tests is not None:
            self.tests = call.tests
        if call.passed and self.first_pass_step is None:
            self.first_pass_step = call.step

    def reflection(self) -> None:
        """A reflection started, after the current step."""
        self.reflections.append(self.step_number)

    def stuck(self) -> None:
        """A ``stuck`` notice arrived during the current step."""
        if self.stuck_at is None:
            self.stuck_at = self.step_number

    @property
    def phase(self) -> str | None:
        return self.phases.phase

    @property
    def cycles(self) -> int:
        return self.phases.cycle

    def first_step_of(self, cls: str) -> int | None:
        """The step of the first call of class *cls*, or ``None``."""
        return next((call.step for call in self.calls if call.cls == cls), None)

    def segments(self) -> list[Segment]:
        """The calls grouped into segments, in order."""
        segments: list[Segment] = []
        group: list[TraceCall] = []
        for call in self.calls:
            if group and (call.cls, call.target) != (group[0].cls, group[0].target):
                segments.append(self._segment(group))
                group = []
            group.append(call)
        if group:
            segments.append(self._segment(group))
        return segments

    def _segment(self, group: Sequence[TraceCall]) -> Segment:
        first, last = group[0], group[-1]
        start = self.step_starts.get(first.step, 0.0)
        return Segment(
            cls=first.cls,
            label=first.label,
            target=first.target,
            steps=(max(1, first.step), max(1, last.step)),
            calls=len(group),
            attempt=max(1, first.attempt),
            wall_s=max(0.0, last.t - start),
            passed=any(call.passed for call in group),
        )


def trace_from_segments(
    segments: Iterable[Segment],
    steps: int,
    *,
    first_pass_step: int | None = None,
) -> Trace:
    """An approximate trace of a run of *steps* steps with *segments*.

    Steps 1 to *steps* start in order. A segment's calls are spread evenly
    over the steps it spans, each at the segment's attempt, and a
    reflection is placed before the first step of a segment whose attempt
    is higher than any before it. A segment's pass goes on its call at
    *first_pass_step* when it has one there, else on its last call. Calls
    beyond the last step come after it. No time is known: every time is 0.
    """
    calls: list[TraceCall] = []
    reflect_before: set[int] = set()
    attempt_at: dict[int, int] = {}
    highest = 1
    for segment in segments:
        first, last = segment.steps
        span = last - first + 1
        if segment.attempt > highest:
            reflect_before.add(first)
            highest = segment.attempt
        at = [
            first + min(span - 1, i * span // segment.calls)
            for i in range(segment.calls)
        ]
        pass_index = -1
        if segment.passed:
            pass_index = next(
                (i for i, step in enumerate(at) if step == first_pass_step),
                segment.calls - 1,
            )
        for index, step in enumerate(at):
            attempt_at.setdefault(step, segment.attempt)
            calls.append(
                TraceCall(
                    step=step,
                    attempt=segment.attempt,
                    cls=segment.cls,
                    label=segment.label,
                    target=segment.target,
                    passed=index == pass_index,
                )
            )
    trace = Trace(clock=lambda: 0.0)
    attempt = 1
    index = 0
    for k in range(1, steps + 1):
        if k in reflect_before:
            trace.reflection()
        attempt = max(attempt, attempt_at.get(k, attempt))
        trace.step(k, attempt)
        while index < len(calls) and calls[index].step <= k:
            trace.add(calls[index])
            index += 1
    for call in calls[index:]:
        trace.add(call)
    return trace


def first_steps(segments: Sequence[Segment]) -> dict[str, int]:
    """The first step of a W call, a T call and a pass (``P``) in *segments*."""
    found: dict[str, int] = {}
    for segment in segments:
        if segment.cls in (WRITE, TEST):
            found.setdefault(segment.cls, segment.steps[0])
        if segment.passed:
            found.setdefault(PASS_ANCHOR, segment.steps[1])
    return found
