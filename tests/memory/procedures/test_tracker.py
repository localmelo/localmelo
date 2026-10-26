"""The tracker: levels, the fraction, off-template, items and totality.

The property tests draw random traces and windows from ``random`` with
fixed seeds, so every run checks the same cases.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from localmelo.melo.memory.procedures.classify import CLASSES, VERIFY
from localmelo.melo.memory.procedures.template import (
    DEFAULT_PARAMS,
    Params,
    SkeletonItem,
    Stats,
    TemplateView,
)
from localmelo.melo.memory.procedures.trace import Trace, TraceCall
from localmelo.melo.memory.procedures.tracker import (
    PROGRESS_KEYS,
    Limits,
    ProgressState,
    bar_series,
    expected_steps,
    raw_fraction,
    shadow_bar_error,
    track,
)

CAP = DEFAULT_PARAMS.cap


def stats(*steps: int, attempts_share: float = 0.0, **kw: object) -> Stats:
    ordered = sorted(steps)
    n = len(ordered)
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    return Stats(
        n=n,
        steps=tuple(steps),
        median=float(median),
        lo=min(steps),
        hi=max(steps),
        spread=(max(steps) - min(steps)) / median,
        wall_s=(60.0, 90.0, 130.0),
        attempts_share=attempts_share,
        gate=True,
        last_steps=steps[-1],
        last_wall_s=130.0,
        **kw,  # type: ignore[arg-type]
    )


def view(
    *steps: int,
    level: int = 2,
    skeleton: tuple[SkeletonItem, ...] = (),
    params: Params = DEFAULT_PARAMS,
    attempts_share: float = 0.0,
    anchors: tuple[tuple[str, float, float], ...] = (),
) -> TemplateView:
    return TemplateView(
        level=level,
        reason="" if level == 2 else "gate",
        match="exact",
        routine_id="r",
        template_id="r:t",
        version=3,
        model="m",
        stats=stats(*steps, attempts_share=attempts_share, anchors=anchors),
        skeleton=skeleton,
        params=params,
    )


def run(steps: int, calls: dict[int, list[TraceCall]] | None = None) -> Trace:
    trace = Trace(clock=lambda: 0.0)
    for k in range(1, steps + 1):
        trace.step(k, 1)
        for call in (calls or {}).get(k, []):
            trace.add(call)
    return trace


def call(
    step: int, cls: str, target: str | None = None, passed: bool = False
) -> TraceCall:
    return TraceCall(
        step=step,
        attempt=1,
        cls=cls,
        label=f"{cls} {target}",
        target=target,
        passed=passed,
    )


# ── the fraction (section 6.3) ──────────────────────────────────────────────


def test_raw_fraction() -> None:
    window = [6, 8]
    assert raw_fraction(0, window, 7.0) == 0.0
    assert raw_fraction(4, window, 7.0) == pytest.approx(4 / 7)
    # k >= median: k / (k + median of the remaining steps of longer runs).
    assert raw_fraction(7, window, 7.0) == pytest.approx(7 / 8)
    # No run is longer: k / (k + 1).
    assert raw_fraction(8, window, 7.0) == pytest.approx(8 / 9)
    assert raw_fraction(5, [4, 5, 10], 5.0) == pytest.approx(5 / 10)


def test_expected_steps() -> None:
    assert expected_steps(4, [6, 8], 7.0) == 7
    assert expected_steps(0, [6, 8], 7.0) == 7
    assert expected_steps(2, [5, 6], 5.5) == 6
    assert expected_steps(7, [6, 8], 7.0) == 8
    assert expected_steps(8, [6, 8], 7.0) == 9


def test_the_test_task_example_of_section_9() -> None:
    v = view(6, 8)
    state = track(v, run(4))
    assert state.level == 2
    assert state.fraction == pytest.approx(4 / 7)  # 57%
    assert state.expected_steps == 7  # "step 4 of ~7"
    assert state.past == {
        "runs": 2,
        "steps_lo": 6,
        "steps_median": 7.0,
        "steps_hi": 8,
        "wall_lo_s": 60.0,
        "wall_hi_s": 130.0,
        "last_steps": 8,
        "last_wall_s": 130.0,
    }
    paused = track(v, run(9))
    assert paused.state == "off_template"
    assert paused.off_reason == "longer"
    assert paused.fraction == pytest.approx(8 / 9)  # "paused at 89%"
    assert paused.expected_steps is None
    done = track(v, run(9), ended="completed")
    assert done.state == "ended" and done.fraction == 1.0


def test_the_median_denominator() -> None:
    v = view(4, 6, params=replace(DEFAULT_PARAMS, denominator="median"))
    series, off = bar_series(v, run(7))
    assert series[:4] == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert series[4:] == [CAP] * 3  # capped at 0.95, then frozen
    assert off == "longer"


def test_the_hybrid_jumps_to_0_9_after_the_first_pass() -> None:
    v = view(10, 12, params=replace(DEFAULT_PARAMS, denominator="hybrid"))
    series, _ = bar_series(v, run(5, {3: [call(3, "T", passed=True)]}))
    assert series[2] == pytest.approx(3 / 11)  # the pass is in step 3...
    assert series[3] == pytest.approx(0.9)  # ...so step 4 starts at 0.9
    assert series[4] == pytest.approx(0.9)


def test_the_milestone_denominator() -> None:
    anchors = (("W", 3.0, 0.4), ("T", 4.0, 0.6), ("P", 6.0, 0.8))
    params = replace(DEFAULT_PARAMS, denominator="milestone")
    v = view(8, 8, params=params, anchors=anchors)
    trace = run(
        8,
        {2: [call(2, "W")], 3: [call(3, "T")], 5: [call(5, "T", passed=True)]},
    )
    series, _ = bar_series(v, trace)
    # Before W ticks, the bar rises toward 0.4 over 3 steps, never reaching it.
    assert series[0] == pytest.approx(0.4 / 3)
    assert series[1] == pytest.approx(0.8 / 3)
    # W (step 2) and T (step 3) have ticked by step 4, one step after T:
    # halfway from T's 0.6 to P's 0.8 over the 2 steps between them.
    assert series[3] == pytest.approx(0.6 + 0.2 * (1 / 2))
    # P (step 5) has ticked by step 6: one of the 2 steps from P to the
    # median (8) toward the cap.
    assert series[5] == pytest.approx(0.8 + 0.15 * (1 / 2))
    assert all(b >= a for a, b in zip(series, series[1:], strict=False))
    assert max(series) <= CAP


# ── off-template ────────────────────────────────────────────────────────────


def test_off_template_is_sticky_and_freezes_the_bar() -> None:
    v = view(4, 5)
    series, off = bar_series(v, run(9))
    assert off == "longer"
    assert series[5:] == [series[4]] * 4
    assert track(v, run(9)).fraction == series[4]


def test_a_reflection_takes_the_run_off_unless_the_window_retries() -> None:
    trace = run(3)
    trace.reflection()
    state = track(view(8, 9), trace)
    assert (state.state, state.off_reason) == ("off_template", "retrying")
    trace.step(4, 2)
    series, off = bar_series(view(8, 9), trace)
    assert off == "retrying" and series[3] == series[2]
    expected = track(view(8, 9, attempts_share=0.5), trace)
    assert expected.state == "running" and expected.off_reason is None


def test_a_stuck_notice_takes_the_run_off() -> None:
    trace = run(2)
    trace.stuck()
    state = track(view(8, 9), trace)
    assert (state.state, state.off_reason) == ("off_template", "stuck")
    trace.step(3, 1)
    series, _ = bar_series(view(8, 9), trace)
    assert series[2] == series[1]


def test_off_template_needs_a_bar() -> None:
    state = track(view(4, 5, level=1), run(9))
    assert state.state == "running" and state.off_reason is None
    assert state.fraction is None


# ── levels and keys ─────────────────────────────────────────────────────────


def test_every_key_on_every_state() -> None:
    limits = Limits(max_steps=30, steps_per_attempt=10, max_attempts=3, timeout_s=None)
    for v in (None, view(6, 8, level=1), view(6, 8)):
        for ended in (None, "completed", "failed"):
            data = track(v, run(3), limits, ended=ended).as_dict()
            assert tuple(data) == PROGRESS_KEYS
    data = track(None, run(3), limits).as_dict()
    assert data["limits"] == {
        "max_steps": 30,
        "steps_per_attempt": 10,
        "max_attempts": 3,
        "timeout_s": None,
    }
    assert ProgressState().as_dict()["limits"] == Limits().as_dict()


def test_level_0_shows_facts_only() -> None:
    trace = run(2, {1: [call(1, "R")], 2: [call(2, "T")]})
    trace.tests = None
    state = track(None, trace)
    assert state.level == 0
    assert (state.step, state.attempt, state.phase, state.cycle) == (2, 1, VERIFY, 0)
    assert state.items is None and state.fraction is None and state.past is None
    assert state.basis is None and state.expected_steps is None
    assert track(view(6, 8, level=0), trace).level == 0


def test_level_1_shows_past_facts_and_items_but_no_percent() -> None:
    skeleton = (SkeletonItem(("R",), "a", "Read a"),)
    state = track(view(6, level=1, skeleton=skeleton), run(2, {1: [call(1, "R", "a")]}))
    assert state.level == 1
    assert state.fraction is None and state.expected_steps is None
    assert state.items == [{"label": "Read a", "state": "current", "repeat": None}]
    assert state.past is not None and state.past["runs"] == 1
    assert state.basis == {
        "routine_id": "r",
        "template_id": "r:t",
        "version": 3,
        "match": "exact",
        "model": "m",
    }


def test_before_the_first_step() -> None:
    state = track(view(6, 8), run(0))
    assert (state.step, state.attempt, state.fraction, state.phase) == (0, 0, 0.0, None)
    assert state.expected_steps == 7


def test_the_end_of_a_run_that_did_not_complete_keeps_the_last_value() -> None:
    v = view(6, 8)
    running = track(v, run(5))
    for status in ("failed", "budget_exhausted", "crashed", "cancelled"):
        ended = track(v, run(5), ended=status)
        assert ended.fraction == running.fraction
        assert ended.state == "ended"


# ── items ───────────────────────────────────────────────────────────────────

SKELETON = (
    SkeletonItem(("R",), "test.py", "Read test.py"),
    SkeletonItem(("R",), "parser.py", "Read parser.py"),
    SkeletonItem(("W", "T"), None, "Edit parser.py + pytest", 2),
    SkeletonItem(("X",), "git", "git"),
)


def items(calls: list[TraceCall], completed: bool = False) -> list[str]:
    trace = run(len(calls), {i + 1: [c] for i, c in enumerate(calls)})
    state = track(
        view(6, 8, skeleton=SKELETON), trace, ended="completed" if completed else None
    )
    assert state.items is not None
    return [item["state"] for item in state.items]


def test_items_before_the_first_match() -> None:
    assert items([]) == ["pending"] * 4
    assert items([call(1, "T", "other")]) == [
        "skipped",
        "skipped",
        "current",
        "pending",
    ]


def test_items_tick_forward() -> None:
    assert items([call(1, "R", "test.py"), call(2, "R", "parser.py")]) == [
        "done",
        "current",
        "pending",
        "pending",
    ]
    # Going back to an earlier file never moves the list back.
    assert items(
        [call(1, "R", "test.py"), call(2, "R", "parser.py"), call(3, "R", "test.py")]
    ) == ["done", "current", "pending", "pending"]


def test_a_collapsed_item_stays_current_while_its_classes_match() -> None:
    calls = [call(1, "R", "test.py"), call(2, "W", "parser.py"), call(3, "T", "pytest")]
    calls.append(call(4, "W", "parser.py"))
    assert items(calls) == ["done", "skipped", "current", "pending"]


def test_a_match_far_ahead_skips_the_pending_items_between() -> None:
    assert items([call(1, "R", "test.py"), call(2, "X", "git")]) == [
        "done",
        "skipped",
        "skipped",
        "current",
    ]


def test_a_completed_end_marks_the_current_item_done() -> None:
    assert items([call(1, "R", "test.py")], completed=True) == [
        "done",
        "pending",
        "pending",
        "pending",
    ]


def test_items_never_return_to_pending_on_random_calls() -> None:
    rng = random.Random(7)
    order = {"pending": 0, "skipped": 1, "current": 1, "done": 2}
    for _ in range(200):
        calls: list[TraceCall] = []
        previous = ["pending"] * len(SKELETON)
        for k in range(1, 15):
            calls.append(
                call(
                    k,
                    rng.choice(CLASSES),
                    rng.choice(["test.py", "parser.py", "git", None]),
                )
            )
            now = items(list(calls))
            assert all(
                order[after] >= order[before] or before == "current"
                for before, after in zip(previous, now, strict=True)
            )
            assert all(
                not (before in ("done", "skipped") and after != before)
                for before, after in zip(previous, now, strict=True)
            )
            assert now.count("current") <= 1
            previous = now


# ── properties ──────────────────────────────────────────────────────────────


def random_view(rng: random.Random) -> TemplateView:
    window = [rng.randint(1, 25) for _ in range(rng.randint(1, 8))]
    params = replace(
        DEFAULT_PARAMS,
        denominator=rng.choice(["median", "dynamic", "milestone", "hybrid"]),
    )
    anchors = tuple(
        sorted(
            (
                (name, float(rng.randint(1, 20)), rng.random())
                for name in rng.sample(["W", "T", "P"], rng.randint(0, 3))
            ),
            key=lambda a: a[1],
        )
    )
    return view(
        *window,
        params=params,
        attempts_share=rng.choice([0.0, 0.5, 1.0]),
        anchors=anchors,
    )


def random_trace(rng: random.Random) -> Trace:
    trace = Trace(clock=lambda: 0.0)
    for k in range(1, rng.randint(0, 40) + 1):
        trace.step(k, 1 + len(trace.reflections))
        for _ in range(rng.randint(0, 3)):
            trace.add(
                call(
                    k,
                    rng.choice(CLASSES),
                    rng.choice(["a", "b", None]),
                    rng.random() < 0.1,
                )
            )
        if rng.random() < 0.05:
            trace.reflection()
        if rng.random() < 0.02:
            trace.stuck()
    return trace


def test_the_fraction_never_moves_back_nor_passes_the_cap_while_running() -> None:
    rng = random.Random(1)
    for _ in range(500):
        v = random_view(rng)
        trace = random_trace(rng)
        series, _ = bar_series(v, trace)
        assert all(b >= a for a, b in zip(series, series[1:], strict=False))
        assert all(0.0 <= s <= CAP for s in series)
        state = track(v, trace)
        assert state.fraction is not None and state.fraction <= CAP


def test_the_fraction_is_1_only_on_a_completed_end() -> None:
    rng = random.Random(2)
    statuses = ["completed", "failed", "budget_exhausted", "stuck", "crashed", None]
    for _ in range(300):
        v = random_view(rng)
        trace = random_trace(rng)
        for status in statuses:
            fraction = track(v, trace, ended=status).fraction
            assert fraction is not None
            assert (fraction == 1.0) == (status == "completed")


def test_the_shown_values_through_a_run_never_move_back() -> None:
    rng = random.Random(3)
    for _ in range(100):
        v = random_view(rng)
        full = random_trace(rng)
        trace = Trace(clock=lambda: 0.0)
        shown: list[float] = []
        index = 0
        for k in range(1, full.step_number + 1):
            trace.step(k, 1)
            shown.append(track(v, trace).fraction or 0.0)
            while index < len(full.calls) and full.calls[index].step == k:
                trace.add(full.calls[index])
                index += 1
                shown.append(track(v, trace).fraction or 0.0)
        shown.append(track(v, trace, ended="completed").fraction or 0.0)
        assert all(b >= a for a, b in zip(shown, shown[1:], strict=False))


def test_the_tracker_is_total() -> None:
    rng = random.Random(4)
    for _ in range(300):
        v = random_view(rng) if rng.random() < 0.8 else None
        state = track(
            v, random_trace(rng), Limits(), ended=rng.choice([None, "completed"])
        )
        assert isinstance(state, ProgressState)
    # Broken inputs give the facts, never an exception.
    broken = replace(view(6, 8), stats=Stats(n=1, steps=(), median=0.0))
    assert track(broken, run(3)).step == 3
    nonsense = Trace(clock=lambda: 0.0)
    nonsense.step_number = -5
    nonsense.calls.append(call(-1, "?"))
    assert track(view(6, 8), nonsense).level in (0, 2)
    assert track(object(), run(2)).level == 0  # type: ignore[arg-type]


def test_shadow_bar_error() -> None:
    v = view(4, 4)
    # 1/4, 2/4, 3/4, then 4/5 at the median: only the last step is off.
    assert shadow_bar_error(v, run(4), 4) == pytest.approx((1.0 - 0.8) / 4)
    longer = shadow_bar_error(v, run(8), 8)
    assert longer is not None and longer > 0
    assert shadow_bar_error(v, run(0), 0) is None
    # The trace may be shorter than the run: the last value holds.
    assert shadow_bar_error(v, run(2), 4) == pytest.approx((0.25 + 0.5) / 4)
