"""Effective levels: aging, inheritance, cycles, the WIP cap and the one P0.

Requirements 11, 15 and 26: user levels never age and aging stops at P2;
inheritance passes only along confirmed edges, capped at P1 and at the
holder's origin, with P0 only from a user's P0 through principals; cycle
edges pass nothing; automatic P1 beyond the cap is refused in a fixed
order; only one user-set P0 holds, and a computed P0 is P1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from localmelo.support.scheduler import Dep, Params, Task, decide
from localmelo.support.scheduler.levels import Levels, compute_levels
from localmelo.support.scheduler.model import us
from localmelo.tests.scheduler.helpers import WALL0, clock, reasons, state, task

DAY = us(86400.0)


def lv(tasks: Sequence[Task], deps: Sequence[Dep] = (), **kw: Any) -> Levels:
    return compute_levels(state(tasks, deps=deps), clock(0.0), Params(**kw))


def test_a_user_level_is_never_aged() -> None:
    t = task("a", level=3, level_source="user", wait_held_us=10 * DAY)
    assert lv([t]).eff["a"] == 3
    assert lv([t]).aged == ()


def test_a_p3_task_ages_to_p2_after_24_h_of_runnable_wait_and_never_reaches_p1() -> (
    None
):
    assert lv([task("a", level=3, wait_held_us=DAY - 1)]).eff["a"] == 3
    reached = lv([task("a", level=3, wait_held_us=DAY)])
    assert reached.eff["a"] == 2 and reached.aged == ("a",)
    sticky = lv([task("a", level=3, aged=True)])
    assert sticky.eff["a"] == 2 and sticky.aged == ()
    assert lv([task("a", level=3, wait_held_us=100 * DAY)]).eff["a"] == 2
    assert lv([task("a", level=4, wait_held_us=100 * DAY)]).eff["a"] == 4


def test_inheritance_passes_only_along_confirmed_edges_while_the_waiter_waits() -> None:
    def holder_level(kind: Any, confirmed: bool, phase: Any) -> int:
        w = task("w", level=1, level_source="user", phase=phase)
        return lv([w, task("h", level=3)], [Dep("w", "h", kind, confirmed)]).eff["h"]

    assert holder_level("result", True, "waiting") == 1
    assert holder_level("lease", True, "parked") == 1  # a parked waiter still waits
    assert holder_level("result", True, "running") == 3  # withdrawn when it wakes
    assert holder_level("result", False, "waiting") == 3
    assert holder_level("start_after", True, "queued") == 1
    assert holder_level("start_after", False, "queued") == 3
    paused = task("w", level=1, phase="waiting", paused=True)
    assert (
        lv([paused, task("h", level=3)], [Dep("w", "h", "result", True)]).eff["h"] == 3
    )


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("user", 1),
        ("user_rule", 1),
        ("front", 2),
        ("harness", 3),
        ("agent", 3),
        ("channel", 3),
        ("sleep", 3),
    ],
)
def test_inheritance_is_capped_at_p1_and_at_the_origin_ceiling(
    origin: Any, expected: int
) -> None:
    w = task("w", level=1, level_source="user")
    h = task("h", level=4, origin=origin)
    assert lv([w, h], [Dep("w", "h", confirmed=True)]).eff["h"] == expected


def test_a_user_set_p0_passes_p0_along_a_chain_of_principal_tasks_only() -> None:
    tasks = [
        task("a", level=0, level_source="user"),
        task("b", level=3, origin="user"),
        task("c", level=3, origin="user_rule"),
        task("d", level=4, origin="agent"),
        task("x", level=3, origin="front"),
        task("y", level=4, origin="user"),
    ]
    edges = [Dep("a", "b"), Dep("b", "c"), Dep("c", "d"), Dep("a", "x"), Dep("x", "y")]
    out = lv(tasks, [Dep(e.waiter, e.holder, confirmed=True) for e in edges])
    assert [out.eff[k] for k in "abcdxy"] == [0, 0, 0, 3, 2, 2]
    assert not [
        a for a in out.asks if a[1] == "p0_conflict"
    ]  # inherited P0 is not a second P0


def test_an_inheritor_takes_the_waiters_deadline() -> None:
    w = task("w", level=1, level_source="user", deadline_wall_us=WALL0 + 100)
    later = task("h", level=3, deadline_wall_us=WALL0 + 500)
    out = lv([w, later], [Dep("w", "h", confirmed=True)])
    assert out.d_eff["h"] == WALL0 + 100
    sooner = task("h", level=3, deadline_wall_us=WALL0 + 50)
    assert lv([w, sooner], [Dep("w", "h", confirmed=True)]).d_eff["h"] == WALL0 + 50
    blocked = task("w", level=1, blocked_now=True, created_wall_us=WALL0)
    out = lv([blocked, task("h", level=3)], [Dep("w", "h", confirmed=True)])
    assert out.d_eff["h"] == WALL0 + us(7200.0)


def test_a_dependency_cycle_passes_nothing_blocks_its_members_and_raises_an_alarm() -> (
    None
):
    tasks = [
        task("a", level=1, level_source="user"),
        task("b", level=3),
        task("c", level=3),
    ]
    deps = [Dep("a", "b", confirmed=True), Dep("b", "a", confirmed=True), Dep("c", "c")]
    out = lv(tasks, deps)
    assert out.eff["b"] == 3
    assert sorted(out.cycle) == ["a", "b", "c"]
    assert {("a", "cycle"), ("b", "cycle"), ("c", "cycle")} <= set(out.asks)
    assert "dependency_cycle:a,b" in out.alarms and "dependency_cycle:c" in out.alarms
    d = decide(state(tasks, deps=deps), clock(0.0), Params())
    assert all(reasons(d)[k] == ("Dependency", "") for k in "abc")
    # An unconfirmed result edge that closes a loop changes no level and asks nothing.
    waiting = [
        task("a", level=1, level_source="user", phase="waiting"),
        task("b", level=3),
    ]
    loop = [Dep("a", "b", "result", True), Dep("b", "a", "result", False)]
    out = lv(waiting, loop)
    assert out.eff["b"] == 1 and out.cycle == frozenset()
    assert not [a for a in out.asks if a[1] == "cycle"]


def test_automatic_p1_beyond_the_cap_is_refused_in_a_fixed_order_and_asked() -> None:
    holders = [task(f"h{i}", level=3, created_mono_us=i) for i in range(1, 5)]
    tasks = [task("w", level=1, level_source="user"), *holders]
    deps = [Dep("w", h.task_id, confirmed=True) for h in holders]
    deps.append(Dep("h1", "ghost"))  # h1 cannot run: it sorts last
    out = lv(tasks, deps, p1_cap=3)
    assert {k: out.eff[k] for k in ("h1", "h2", "h3", "h4")} == {
        "h1": 2,
        "h2": 1,
        "h3": 1,
        "h4": 2,
    }
    assert ("h1", "p1_cap") in out.asks and ("h4", "p1_cap") in out.asks
    users = [task(f"u{i}", level=1, level_source="user") for i in range(4)]
    over = lv(users, p1_cap=3)
    assert all(over.eff[u.task_id] == 1 for u in users)
    assert "p1_over_cap" in over.alarms


def test_a_second_p0_counts_as_p1_with_an_ask() -> None:
    old = task("old", level=0, level_source="user", created_mono_us=1)
    new = task("new", level=0, level_source="user", created_mono_us=2)
    out = lv([old, new])
    assert (out.eff["old"], out.eff["new"]) == (1, 0)
    assert ("old", "p0_conflict") in out.asks


def test_a_computed_or_rule_p0_counts_as_p1_and_never_displaces_the_users_p0() -> None:
    tasks = [
        task("user", level=0, level_source="user", created_mono_us=1),
        task("comp", level=0, level_source="computed", created_mono_us=5),
        task("rule", level=0, level_source="user_rule", created_mono_us=6),
    ]
    out = lv(tasks)
    assert (out.eff["user"], out.eff["comp"], out.eff["rule"]) == (0, 1, 1)
    assert {"p0_not_user:comp", "p0_not_user:rule"} <= set(out.alarms)
    assert not [a for a in out.asks if a[1] == "p0_conflict"]


def test_the_aging_and_inheritance_switches_turn_each_off() -> None:
    old = task("a", level=3, wait_held_us=10 * DAY)
    assert lv([old], aging=False).eff["a"] == 3
    w = task("w", level=1, level_source="user", deadline_wall_us=WALL0 + 9)
    deps = [Dep("w", "h", confirmed=True), Dep("x", "x")]
    out = lv([w, task("h", level=3), task("x", level=3)], deps, inheritance=False)
    assert out.eff["h"] == 3 and out.d_eff["h"] is None
    assert ("x", "cycle") in out.asks
    assert out.unmet["w"] == ("h",)
