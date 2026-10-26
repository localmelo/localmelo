"""Template statistics, the exploration drop, the skeleton, the gate and levels."""

from __future__ import annotations

from dataclasses import replace

import pytest

from localmelo.melo.memory.procedures.template import (
    DEFAULT_PARAMS,
    Episode,
    Params,
    SkeletonItem,
    Stats,
    Template,
    build_version,
    collapse,
    compute_stats,
    decide_level,
    edit_distance,
    medoid,
    model_demoted,
    statistics_window,
    view_of,
)
from localmelo.melo.memory.procedures.trace import Segment


def seg(
    cls: str, target: str | None, first: int, last: int | None = None, **kw: object
) -> Segment:
    label = f"{cls} {target}" if target else cls
    return Segment(cls, label, target, (first, last or first), **kw)  # type: ignore[arg-type]


def ep(
    task_id: str,
    steps: int,
    status: str = "completed",
    *,
    attempts: int = 1,
    duration_s: float = 10.0,
    first_pass_step: int | None = None,
    segments: tuple[Segment, ...] = (),
) -> Episode:
    return Episode(
        task_id=task_id,
        routine_id="r",
        template_id="r:t",
        model="m",
        status=status,
        steps=steps,
        created_at=float(int(task_id[1:]) if task_id[1:].isdigit() else 0),
        attempts=attempts,
        duration_s=duration_s,
        first_pass_step=first_pass_step,
        segments=segments,
    )


# ── Params ──────────────────────────────────────────────────────────────────


def test_params_defaults_and_round_trip() -> None:
    params = Params()
    assert params.as_dict() == {
        "window": 8,
        "stats": "completed",
        "drop": "conditional",
        "min_episodes": 2,
        "spread": 0.5,
        "cap": 0.95,
        "denominator": "dynamic",
        "demotion": 0.18,
        "demotion_window": 10,
        "demotion_min": 5,
    }
    assert Params.from_dict({**params.as_dict(), "unknown": 1}) == params
    assert params.statistics_statuses == {"completed"}
    assert Params(stats="normal").statistics_statuses == {
        "completed",
        "budget_exhausted",
        "stuck",
        "gate_stopped",
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"stats": "all"},
        {"drop": "maybe"},
        {"denominator": "mean"},
        {"window": 0},
        {"spread": 0},
        {"cap": 1.0},
        {"cap": float("nan")},
        {"demotion": -1},
    ],
)
def test_params_refuse_bad_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Params(**bad)  # type: ignore[arg-type]


# ── the statistics window and the exploration drop ──────────────────────────


def ids(episodes: list[Episode]) -> list[str]:
    return [e.task_id for e in episodes]


def test_a_slow_first_run_leaves_once_two_later_runs_completed() -> None:
    window = [ep("t1", 9), ep("t2", 5)]
    assert statistics_window(window, "t1", DEFAULT_PARAMS) == (window, False)
    window.append(ep("t3", 5))
    kept, dropped = statistics_window(window, "t1", DEFAULT_PARAMS)
    assert dropped and ids(kept) == ["t2", "t3"]


def test_a_first_run_no_slower_than_every_later_run_stays() -> None:
    window = [ep("t1", 6), ep("t2", 5), ep("t3", 6)]
    assert statistics_window(window, "t1", DEFAULT_PARAMS) == (window, False)


def test_unconditional_and_off() -> None:
    window = [ep("t1", 4), ep("t2", 5), ep("t3", 6)]
    unconditional = replace(DEFAULT_PARAMS, drop="unconditional")
    assert ids(statistics_window(window, "t1", unconditional)[0]) == ["t2", "t3"]
    slow_first = [ep("t1", 9), ep("t2", 5), ep("t3", 5)]
    off = replace(DEFAULT_PARAMS, drop="off")
    assert statistics_window(slow_first, "t1", off) == (slow_first, False)


def test_only_the_templates_first_run_is_exploration() -> None:
    # The window's oldest run is not the template's first: it stays.
    window = [ep("t5", 9), ep("t6", 5), ep("t7", 5)]
    assert statistics_window(window, "t1", DEFAULT_PARAMS) == (window, False)
    assert statistics_window(window, None, DEFAULT_PARAMS) == (window, False)


def test_later_runs_must_be_completed_to_count() -> None:
    params = replace(DEFAULT_PARAMS, stats="normal")
    window = [ep("t1", 9), ep("t2", 5, "gate_stopped"), ep("t3", 5)]
    assert statistics_window(window, "t1", params) == (window, False)


def test_a_first_run_that_did_not_complete_stays_under_the_conditional_drop() -> None:
    params = replace(DEFAULT_PARAMS, stats="normal")
    window = [ep("t1", 9, "budget_exhausted"), ep("t2", 5), ep("t3", 5)]
    assert statistics_window(window, "t1", params) == (window, False)


# ── compute_stats ───────────────────────────────────────────────────────────


def test_compute_stats() -> None:
    window = [
        ep("t1", 6, duration_s=100, first_pass_step=5),
        ep("t2", 8, duration_s=160, attempts=2, first_pass_step=6),
        ep("t3", 7, duration_s=130),
    ]
    stats = compute_stats(window, "t1", DEFAULT_PARAMS)
    assert stats.n == 3
    assert stats.steps == (6, 8, 7)
    assert (stats.median, stats.lo, stats.hi) == (7.0, 6, 8)
    assert stats.spread == pytest.approx(2 / 7)
    assert stats.cv == pytest.approx(0.1166, abs=1e-4)
    assert stats.wall_s == (100.0, 130.0, 160.0)
    assert stats.attempts_share == pytest.approx(1 / 3)
    assert stats.tail_after_pass == 1.5
    assert stats.gate and stats.gate_reason == ""
    assert (stats.last_steps, stats.last_wall_s) == (7, 130.0)


def test_gate_reasons() -> None:
    one = compute_stats([ep("t1", 6)], "t1", DEFAULT_PARAMS)
    assert not one.gate and one.gate_reason == "1 of 2 runs"
    wide = compute_stats([ep("t1", 9), ep("t2", 5)], "t1", DEFAULT_PARAMS)
    assert not wide.gate and wide.gate_reason == "spread 0.57 above 0.5"
    normal = replace(DEFAULT_PARAMS, stats="normal")
    failed = compute_stats([ep("t1", 6, "stuck"), ep("t2", 6, "stuck")], None, normal)
    assert not failed.gate and failed.gate_reason == "no completed run"
    empty = compute_stats([], None, DEFAULT_PARAMS)
    assert not empty.gate and empty.n == 0


def test_the_spread_gate_is_inclusive() -> None:
    stats = compute_stats([ep("t1", 4), ep("t2", 6)], None, DEFAULT_PARAMS)
    assert stats.spread == pytest.approx(0.4)
    assert stats.gate
    assert not compute_stats([ep("t1", 4), ep("t2", 6)], None, Params(spread=0.39)).gate
    exact = compute_stats([ep("t1", 6), ep("t2", 10)], None, DEFAULT_PARAMS)
    assert exact.spread == 0.5 and exact.gate


def test_runs_without_steps_are_left_out() -> None:
    stats = compute_stats([ep("t1", 0), ep("t2", 5), ep("t3", 5)], "t1", DEFAULT_PARAMS)
    assert stats.steps == (5, 5)


def test_the_daily_routine_example_of_section_9() -> None:
    days = [ep("t1", 9), ep("t2", 5)]
    day3 = compute_stats(days, "t1", DEFAULT_PARAMS)
    assert not day3.gate and day3.gate_reason.startswith("spread 0.57")
    day4 = compute_stats([*days, ep("t3", 5)], "t1", DEFAULT_PARAMS)
    assert day4.gate and day4.steps == (5, 5)
    off = compute_stats([*days, ep("t3", 5)], "t1", replace(DEFAULT_PARAMS, drop="off"))
    assert not off.gate


def test_anchors_need_most_runs() -> None:
    def run(
        task_id: str, steps: int, w: int | None, t: int | None, p: int | None
    ) -> Episode:
        segments: list[Segment] = []
        if w:
            segments.append(seg("W", "a", w))
        if t:
            segments.append(seg("T", "pytest", t, passed=p is not None))
        return ep(task_id, steps, first_pass_step=p, segments=tuple(segments))

    window = [run("t1", 4, 2, 3, 3), run("t2", 6, 2, 4, None), run("t3", 8, 4, 5, 7)]
    stats = compute_stats(window, None, replace(DEFAULT_PARAMS, drop="off"))
    names = [name for name, _, _ in stats.anchors]
    assert names == ["W", "T", "P"]
    w = stats.anchors[0]
    assert w[1] == 2.0 and w[2] == pytest.approx(0.5)
    # A pass in 2 of 3 runs is enough (0.6); in 1 of 3 it is not.
    window[2] = run("t3", 8, 4, 5, None)
    stats = compute_stats(window, None, replace(DEFAULT_PARAMS, drop="off"))
    assert [name for name, _, _ in stats.anchors] == ["W", "T"]


def test_stats_round_trip() -> None:
    stats = compute_stats(
        [ep("t1", 6, first_pass_step=4), ep("t2", 8)], None, DEFAULT_PARAMS
    )
    assert Stats.from_dict(stats.as_dict()) == replace(
        stats, spread=round(stats.spread, 6), cv=round(stats.cv, 6)
    )


# ── the skeleton ────────────────────────────────────────────────────────────


def test_edit_distance() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], [1, 2]) == 2
    assert edit_distance([("R", "a")], [("R", "a")]) == 0


def test_medoid_least_summed_distance_and_latest_on_ties() -> None:
    a = ep("t1", 4, segments=(seg("R", "a", 1), seg("W", "a", 2)))
    b = ep("t2", 4, segments=(seg("R", "a", 1), seg("W", "a", 2), seg("T", "p", 3)))
    c = ep(
        "t3",
        4,
        segments=(
            seg("R", "a", 1),
            seg("W", "a", 2),
            seg("T", "p", 3),
            seg("R", "b", 4),
        ),
    )
    assert medoid([a, b, c]) is b
    twin = ep("t4", 4, segments=a.segments)
    assert medoid([a, twin]) is twin
    assert medoid([]) is None


def test_collapse_a_repeated_pattern() -> None:
    segments = [
        seg("R", "test.py", 1),
        seg("W", "parser.py", 2),
        seg("T", "pytest", 3),
        seg("W", "parser.py", 4),
        seg("T", "pytest", 5),
        seg("R", "notes", 6),
    ]
    items = collapse(segments)
    assert items == (
        SkeletonItem(("R",), "test.py", "R test.py"),
        SkeletonItem(("W", "T"), None, "W parser.py + T pytest", repeat=2),
        SkeletonItem(("R",), "notes", "R notes"),
    )
    assert collapse([]) == ()


def test_collapse_prefers_the_longest_coverage() -> None:
    segments = [seg("W", "a", 1), seg("T", "p", 2)] * 3
    (item,) = collapse(segments)
    assert item.repeat == 3 and item.cls == ("W", "T")


def test_skeleton_item_matching() -> None:
    single = SkeletonItem(("W",), "parser.py", "Edit parser.py")
    assert single.matches("W", "parser.py")
    assert not single.matches("W", "other.py")
    assert not single.matches("R", "parser.py")
    pattern = SkeletonItem(("W", "T"), None, "Edit + pytest", 2)
    assert pattern.matches("T", "anything") and not pattern.matches("R", None)
    assert SkeletonItem.from_dict(pattern.as_dict()) == pattern
    assert SkeletonItem.from_dict({"cls": "R", "label": "x"}).cls == ("R",)


# ── versions ────────────────────────────────────────────────────────────────


def version(
    window: list[Episode], parent: Template | None = None, **kw: object
) -> Template:
    return build_version(
        template_id="r:t",
        routine_id="r",
        model="m",
        options={"max_steps": 30},
        window=window,
        first_task_id=window[0].task_id if window else None,
        parent=parent,
        params=DEFAULT_PARAMS,
        harness_version="1.0",
        created_at=5.0,
        **kw,  # type: ignore[arg-type]
    )


def test_build_version() -> None:
    a = ep("t1", 4, segments=(seg("R", "a", 1), seg("W", "a", 2)))
    b = ep("t2", 5, segments=(seg("R", "a", 1), seg("W", "a", 2), seg("T", "p", 3)))
    first = version([a])
    assert (first.version, first.parent_version, first.status) == (1, None, "active")
    assert (first.source, first.admitted_by) == ("online", "harness")
    assert first.source_ids == ("t1",)
    assert first.medoid == "t1"
    second = version([a, b], parent=first)
    assert (second.version, second.parent_version) == (2, 1)
    assert second.source_ids == ("t1", "t2")
    assert second.stats.steps == (4, 5)


def test_a_sleep_skeleton_is_copied_and_only_statistics_refresh() -> None:
    a = ep("t1", 4, segments=(seg("R", "a", 1),))
    sleep_items = (SkeletonItem(("W",), "x", "Edit x"),)
    parent = replace(version([a]), skeleton=sleep_items, source="sleep", medoid=None)
    child = version([a, ep("t2", 6, segments=(seg("T", "p", 1),))], parent=parent)
    assert child.skeleton == sleep_items
    assert child.skeleton_source == "sleep"
    assert child.stats.n == 2
    grandchild = version([a, ep("t3", 5)], parent=child)
    assert grandchild.skeleton == sleep_items


def test_the_skeleton_comes_from_completed_runs_only() -> None:
    params = replace(DEFAULT_PARAMS, stats="normal")
    done = ep("t1", 4, segments=(seg("R", "a", 1),))
    failed = ep("t2", 4, "stuck", segments=(seg("X", None, 1),))
    built = build_version(
        template_id="r:t",
        routine_id="r",
        model="m",
        options={},
        window=[done, failed, failed],
        first_task_id="t1",
        parent=None,
        params=params,
        harness_version="1.0",
        created_at=0.0,
    )
    assert built.medoid == "t1"


# ── levels ──────────────────────────────────────────────────────────────────


def test_decide_level() -> None:
    passing = version([ep("t1", 6), ep("t2", 7)])
    assert passing.stats.gate
    assert decide_level(passing, match="exact", model="m", model_scores=[]) == (2, "")
    assert decide_level(passing, match="routine", model="m", model_scores=[])[0] == 2
    assert decide_level(passing, match="similar", model="m", model_scores=[]) == (
        1,
        "similar task",
    )
    assert decide_level(passing, match="exact", model="", model_scores=[]) == (
        1,
        "no model id",
    )
    assert (
        decide_level(passing, match="exact", model="default", model_scores=[])[0] == 1
    )
    bad = [0.3] * 5
    assert decide_level(passing, match="exact", model="m", model_scores=bad) == (
        1,
        "model demoted",
    )
    held = version([ep("t1", 6)])
    assert decide_level(held, match="exact", model="m", model_scores=[]) == (
        1,
        "1 of 2 runs",
    )


def test_model_demotion() -> None:
    params = DEFAULT_PARAMS
    assert not model_demoted([0.5] * 4, params)  # fewer than 5 scored
    assert model_demoted([0.2] * 5, params)
    assert not model_demoted([0.18] * 5, params)
    # Only the latest 10 count: a model recovers.
    assert not model_demoted([0.9] * 10 + [0.1] * 10, params)


def test_view_of() -> None:
    template = version([ep("t1", 6), ep("t2", 7)])
    view = view_of(template, match="exact", model="m", model_scores=[], title=None)
    assert view.level == 2
    assert view.basis() == {
        "routine_id": "r",
        "template_id": "r:t",
        "version": 1,
        "match": "exact",
        "model": "m",
    }
