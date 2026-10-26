"""The tracker: where a running task stands on its template.

:func:`track` is a pure, total reducer from a template view, the run's
trace and its limits to a :class:`ProgressState`, the data of a
``run.progress`` event. It never raises on any trace. What it may show
depends on the view's level:

- level 0, facts only: step, attempt, phase, cycle, tests;
- level 1, also past facts and the template's named steps (``items``);
- level 2, also a determinate bar: ``fraction`` and ``expected_steps``.

The fraction changes only when a step starts: after ``step.started`` for
step k it is the denominator rule's value (section 6.3) for k, never
lower than before and never above the cap while running; it is 1.0 only
on the end of a completed run. Off the template (a run longer than every
run in the window, a reflection the window does not expect, or a stuck
notice) the bar freezes for the rest of the run.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from localmelo.melo.memory.procedures.classify import PASS_ANCHOR, TEST, WRITE
from localmelo.melo.memory.procedures.template import (
    COMPLETED,
    HYBRID_JUMP,
    SkeletonItem,
    TemplateView,
)
from localmelo.melo.memory.procedures.trace import Trace, TraceCall

logger = logging.getLogger(__name__)

RUNNING: Final = "running"
OFF_TEMPLATE: Final = "off_template"
ENDED: Final = "ended"

LONGER: Final = "longer"
RETRYING: Final = "retrying"
STUCK: Final = "stuck"

ITEM_DONE: Final = "done"
ITEM_CURRENT: Final = "current"
ITEM_PENDING: Final = "pending"
ITEM_SKIPPED: Final = "skipped"

# The keys of a run.progress event, in order.
PROGRESS_KEYS: Final = (
    "level",
    "state",
    "step",
    "attempt",
    "limits",
    "phase",
    "cycle",
    "tests",
    "items",
    "fraction",
    "expected_steps",
    "past",
    "off_reason",
    "basis",
)


@dataclass(frozen=True)
class Limits:
    """A run's budgets; ``None`` where the run has none."""

    max_steps: int | None = None
    steps_per_attempt: int | None = None
    max_attempts: int | None = None
    timeout_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "steps_per_attempt": self.steps_per_attempt,
            "max_attempts": self.max_attempts,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class ProgressState:
    """The data of one ``run.progress`` event: every key, null where unknown."""

    level: int = 0
    state: str = RUNNING
    step: int = 0
    attempt: int = 0
    limits: dict[str, Any] | None = None
    phase: str | None = None
    cycle: int = 0
    tests: dict[str, int] | None = None
    items: list[dict[str, Any]] | None = None
    fraction: float | None = None
    expected_steps: int | None = None
    past: dict[str, Any] | None = None
    off_reason: str | None = None
    basis: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = {key: getattr(self, key) for key in PROGRESS_KEYS}
        if data["limits"] is None:
            data["limits"] = Limits().as_dict()
        return data


def raw_fraction(k: int, steps: Sequence[int], median: float) -> float:
    """Section 6.3's fraction at step *k* for a window of *steps*.

    k ÷ median while k < median; else k ÷ (k + the median of s − k over
    the window's runs longer than k), or k ÷ (k + 1) when none is.
    """
    if k <= 0:
        return 0.0
    if k < median:
        return k / median
    longer = [s - k for s in steps if s > k]
    remaining = statistics.median(longer) if longer else 1
    return k / (k + remaining)


def expected_steps(k: int, steps: Sequence[int], median: float) -> int:
    """k plus the median remaining steps: the step count the bar expects."""
    if k < median:
        return max(1, math.ceil(median - 1e-9))
    longer = [s - k for s in steps if s > k]
    remaining = statistics.median(longer) if longer else 1
    return k + math.ceil(remaining - 1e-9)


class _Milestones:
    """The milestone bar of the bench's checklist arm, in planning steps.

    Anchors W, T and P (first write, first test, first pass) split the bar
    at the window's median fraction of each; the bar moves linearly within
    a segment over its median length in steps, and never reaches the next
    anchor's fraction before that anchor is reached.
    """

    def __init__(self, view: TemplateView) -> None:
        anchors = view.stats.anchors
        cap = view.params.cap
        self.names = [name for name, _, _ in anchors]
        self.bounds = [0.0, *[step for _, step, _ in anchors], view.stats.median]
        fractions = [0.0, *[fraction for _, _, fraction in anchors], cap]
        for i in range(1, len(fractions)):
            fractions[i] = max(fractions[i], fractions[i - 1])
        self.fractions = fractions
        self.reached = 0  # anchors ticked so far
        self.since = 0  # the step at which the last one ticked

    def value(self, k: int, hits: dict[str, int | None]) -> float:
        for index, name in enumerate(self.names, start=1):
            hit = hits.get(name)
            if hit is not None and hit < k and index > self.reached:
                self.reached, self.since = index, hit
        j = self.reached
        lo = self.fractions[j]
        hi = self.fractions[j + 1] if j + 1 < len(self.fractions) else lo
        length = (
            max(self.bounds[j + 1] - self.bounds[j], 1.0)
            if j + 1 < len(self.bounds)
            else 1.0
        )
        value = lo + (hi - lo) * min((k - self.since) / length, 1.0)
        if j + 1 < len(self.fractions) - 1:
            value = min(value, hi - 1e-9)
        return value


def _first_before(calls: Sequence[TraceCall], k: int) -> dict[str, int | None]:
    """The steps of the first W, T and pass among the calls of steps before *k*."""
    hits: dict[str, int | None] = {WRITE: None, TEST: None, PASS_ANCHOR: None}
    for call in calls:
        if call.step >= k:
            break
        if call.cls in (WRITE, TEST) and hits[call.cls] is None:
            hits[call.cls] = call.step
        if call.passed and hits[PASS_ANCHOR] is None:
            hits[PASS_ANCHOR] = call.step
    return hits


def bar_series(view: TemplateView, trace: Trace) -> tuple[list[float], str | None]:
    """The fraction after each step.started of *trace*, and the off reason.

    Entry k − 1 is the fraction shown after step k started, from what
    happened before that event. The off reason, if any, is the first
    thing that took the run off the template, up to now.
    """
    stats = view.stats
    params = view.params
    window = stats.steps
    longest = max(window) if window else 0
    retry_expected = stats.attempts_share >= 0.5
    reflections = sorted(trace.reflections)
    stuck_at = trace.stuck_at
    milestones = _Milestones(view) if params.denominator == "milestone" else None
    shown = 0.0
    off: str | None = None
    series: list[float] = []
    for k in range(1, trace.step_number + 1):
        if off is None:
            if not retry_expected and any(0 < at < k for at in reflections):
                off = RETRYING
            elif stuck_at is not None and stuck_at < k:
                off = STUCK
            elif k > longest:
                off = LONGER
        if off is None:
            if params.denominator == "median":
                raw = k / stats.median if stats.median > 0 else 0.0
            elif milestones is not None:
                raw = milestones.value(k, _first_before(trace.calls, k))
            else:
                raw = raw_fraction(k, window, stats.median)
                if params.denominator == "hybrid":
                    passed = _first_before(trace.calls, k)[PASS_ANCHOR]
                    if passed is not None:
                        raw = max(raw, HYBRID_JUMP)
            shown = min(params.cap, max(shown, raw))
        series.append(shown)
    if off is None:
        step = trace.step_number
        if not retry_expected and any(at >= max(step, 1) for at in reflections):
            off = RETRYING
        elif stuck_at is not None:
            off = STUCK
    return series, off


def _items(
    skeleton: Sequence[SkeletonItem], calls: Sequence[TraceCall], completed: bool
) -> list[dict[str, Any]]:
    """The skeleton's items with their states after *calls* (forward only)."""
    states = [ITEM_PENDING] * len(skeleton)
    current: int | None = None
    for call in calls:
        start = 0 if current is None else current
        for i in range(start, len(skeleton)):
            if not skeleton[i].matches(call.cls, call.target):
                continue
            if i != current:
                if current is not None:
                    states[current] = ITEM_DONE
                for between in range(start if current is None else current + 1, i):
                    if states[between] == ITEM_PENDING:
                        states[between] = ITEM_SKIPPED
                states[i] = ITEM_CURRENT
                current = i
            break
    if completed and current is not None:
        states[current] = ITEM_DONE
    return [
        {"label": item.label, "state": state, "repeat": item.repeat}
        for item, state in zip(skeleton, states, strict=True)
    ]


def _past(view: TemplateView) -> dict[str, Any]:
    stats = view.stats
    return {
        "runs": stats.n,
        "steps_lo": stats.lo if stats.n else None,
        "steps_median": stats.median if stats.n else None,
        "steps_hi": stats.hi if stats.n else None,
        "wall_lo_s": stats.wall_s[0] if stats.n else None,
        "wall_hi_s": stats.wall_s[2] if stats.n else None,
        "last_steps": stats.last_steps,
        "last_wall_s": stats.last_wall_s,
    }


def track(
    view: TemplateView | None,
    trace: Trace,
    limits: Limits | None = None,
    *,
    ended: str | None = None,
) -> ProgressState:
    """The progress of a run with *trace* on *view*'s template.

    *ended* is the run's status on the event that comes right before
    ``run.finished``, else ``None``. Total: any exception inside gives the
    facts-only state of level 0, logged at DEBUG.
    """
    try:
        return _track(view, trace, limits, ended)
    except Exception as exc:  # the tracker never fails a run
        logger.debug("Progress tracking failed: %s", exc)
        return _facts(trace, limits, ended)


def _facts(trace: Trace, limits: Limits | None, ended: str | None) -> ProgressState:
    try:
        tests = None if trace.tests is None else trace.tests.as_dict()
        return ProgressState(
            state=ENDED if ended is not None else RUNNING,
            step=int(trace.step_number),
            attempt=int(trace.attempt),
            limits=(limits or Limits()).as_dict(),
            phase=trace.phase,
            cycle=int(trace.cycles),
            tests=tests,
        )
    except Exception:
        return ProgressState(state=ENDED if ended is not None else RUNNING)


def _track(
    view: TemplateView | None,
    trace: Trace,
    limits: Limits | None,
    ended: str | None,
) -> ProgressState:
    facts = _facts(trace, limits, ended)
    if view is None or view.level <= 0:
        return facts
    completed = ended == COMPLETED
    level = min(2, view.level)
    items = _items(view.skeleton, trace.calls, completed)
    fraction: float | None = None
    expected: int | None = None
    off: str | None = None
    if level == 2:
        series, off = bar_series(view, trace)
        fraction = series[-1] if series else 0.0
        if completed:
            fraction = 1.0
        elif off is None and ended is None:
            expected = expected_steps(
                trace.step_number, view.stats.steps, view.stats.median
            )
    state = facts.state
    if state == RUNNING and off is not None:
        state = OFF_TEMPLATE
    return ProgressState(
        level=level,
        state=state,
        step=facts.step,
        attempt=facts.attempt,
        limits=facts.limits,
        phase=facts.phase,
        cycle=facts.cycle,
        tests=facts.tests,
        items=items,
        fraction=fraction,
        expected_steps=expected,
        past=_past(view),
        off_reason=off,
        basis=view.basis(),
    )


def shadow_bar_error(view: TemplateView, trace: Trace, steps: int) -> float | None:
    """The bar error *view*'s bar would have had on a run of *steps* steps.

    The mean over k = 1..steps of |shown_k − k ÷ steps|, shown_k the
    fraction after step k started (see :func:`bar_series`), whatever the
    view's level. ``None`` for a run without steps.
    """
    if steps <= 0:
        return None
    series, _ = bar_series(view, trace)
    series = (series + [series[-1] if series else 0.0] * steps)[:steps]
    return statistics.fmean(abs(shown - k / steps) for k, shown in enumerate(series, 1))
