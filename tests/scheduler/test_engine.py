"""The engine replays workloads through the policy and finishes them (acceptance c).

Requirements 2, 41 and 46-48: under bounded load every task finishes in
every arm, with parks, the idle window and dependency blocks exercised;
mode off queues every call at the server and still books the accounting;
the lane deadline answers by code; a park evicts the prefix and needs a
new start; a P0 starts within one lower call; events at one instant run
in rank order, stale completions are discarded and the horizon is
exclusive.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from localmelo.support.scheduler import (
    ARMS,
    CallEnded,
    CallSubmitted,
    Machine,
    State,
    params_for_arm,
)
from localmelo.support.scheduler.model import us
from localmelo.support.scheduler.sim import ServerParams, Workload, engine, run
from localmelo.support.scheduler.sim.engine import Mode
from localmelo.support.scheduler.sim.trace import TaskTemplate, TemplateCall
from localmelo.support.scheduler.sim.workloads import WALL_START_US
from localmelo.tests.scheduler.helpers import (
    bounded_workload,
    flat_template,
    offered_load,
    wtask,
)


def workload(tasks: Any, horizon_s: float = 3600.0, **kw: Any) -> Workload:
    return Workload("test", 0, us(horizon_s), WALL_START_US, tuple(tasks), **kw)


@pytest.mark.parametrize("arm", ARMS)
def test_every_task_finishes_under_bounded_load(arm: Mode) -> None:
    parks = 0
    for seed in range(1, 6):
        wl = bounded_workload(seed)
        assert 0.55 <= offered_load(wl) <= 0.65
        r = run(wl, arm, params_for_arm(arm, t_park_s=120.0))
        assert [t.task_id for t in r.tasks if t.finish_us is None] == []
        parks += sum(t.parks for t in r.tasks)
        by_id = {t.task_id: t for t in r.tasks}
        (p4,) = [t for t in r.tasks if t.gold_level == 4]
        assert (p4.first_admit_us or 0) >= us(7200.0 + 600.0)  # the idle window
        for dep in wl.deps:  # a waiter starts only after its holder is done
            assert (by_id[dep.waiter].first_admit_us or 0) >= (
                by_id[dep.holder].finish_us or 0
            )
    assert parks >= 1


def test_off_mode_sends_every_call_straight_to_the_server_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tpl = flat_template("ten-seconds", 3, 660)
    wl = workload(wtask(f"t{i}", 0.0, 2 + i % 2, tpl) for i in range(4))
    seen: list[tuple[Any, State]] = []
    real = engine.apply_dispatch

    def spy(st: State, now: Any, d: Any, p: Any) -> State:
        out = real(st, now, d, p)
        seen.append((d, out))
        return out

    monkeypatch.setattr(engine, "apply_dispatch", spy)
    r = run(wl, "off", server=ServerParams(n_srv=2))
    for d, st in seen:
        assert d.decision.reasons == () and d.decision.parks == ()
        assert all(c.admitted_mono_us is not None for c in st.calls)
        assert all(t.phase != "held" for t in st.tasks)
    assert all(t.level_eff == t.level for t in seen[-1][1].tasks)
    assert all(t.finish_us is not None for t in r.tasks)
    assert all(t.service_s == pytest.approx(30.0) for t in r.tasks)  # 3 × 10 s
    assert all(t.wait_own_s == t.wait_total_s for t in r.tasks)
    assert sum(t.wait_total_s for t in r.tasks) > 0  # 4 tasks on 2 slots queue
    # A call still in the server's queue at the horizon has waited too.
    long = flat_template("1000s", 1, 66000)
    stuck = workload([wtask("x", 0.0, 2, long), wtask("y", 1.0, 2, long)], 600.0)
    y = run(stuck, "off", server=ServerParams(n_srv=1)).tasks[1]
    assert (y.first_admit_us, y.wait_total_s, y.wait_own_s) == (None, 599.0, 599.0)


def test_the_lane_deadline_answers_by_code_after_45_s() -> None:
    msgs = ((us(10.0), "answer"), (us(100.0), "create"))
    # A 1000 s task call takes the one slot that a foreign client leaves.
    wl = workload(
        [wtask("x", 0.0, 2, flat_template("1000s", 1, 66000))],
        300.0,
        front_messages=msgs,
    )
    busy = ServerParams(n_srv=2, foreign_calls=1)
    for r in (
        run(wl, "las2", server=busy),  # stuck in the server's queue
        run(  # held by the gate under respect, then withdrawn
            replace(wl, machine_events=((0, Machine(lock="foreign")),)),
            "las2",
            params_for_arm("las2", reservation_policy="respect"),
        ),
    ):
        assert [(x.by_code, x.end_us - x.send_us) for x in r.replies if x.end_us] == [
            (True, us(45.0)),
            (True, us(45.0)),
        ]
    free = run(wl, "las2")
    assert [x.by_code for x in free.replies] == [False, False]
    # Mode off has no gate, so nothing answers by code: the reply waits.
    off = run(wl, "off", server=busy)
    assert [(x.by_code, x.end_us) for x in off.replies] == [(False, None)] * 2
    with pytest.raises(ValueError, match="foreign_calls"):
        run(wl, "las2", server=ServerParams(foreign_calls=8))
    create = free.replies[1]
    assert create.first_event_us is not None and create.end_us is not None
    assert create.send_us < create.first_event_us < create.end_us < us(100.0 + 45.0)


def test_a_park_evicts_the_tasks_prefix_and_needs_a_new_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = tuple(
        TemplateCall(
            "plan", 1, 2000 + 100 * i, 1320, 2000 + 100 * (i - 1) if i else 0, 0.0, "ok"
        )
        for i in range(4)
    )
    wl = workload(wtask(f"t{i}", 0.0, 2, TaskTemplate("warm", calls)) for i in range(3))
    restarts: list[str] = []
    real = engine.dispatch

    def spy(st: State, now: Any, p: Any, **kw: Any) -> Any:
        d = real(st, now, p, **kw)
        phase = {t.task_id: t.phase for t in st.tasks}
        owner = {c.call_id: c.task_id or "" for c in st.calls}
        restarts.extend(
            owner[c] for c in d.decision.admit if phase.get(owner[c]) == "parked"
        )
        return d

    monkeypatch.setattr(engine, "dispatch", spy)
    r = run(wl, "fifo", params_for_arm("fifo", n_a=1, w_max=3, t_park_s=30.0))
    parked = [t for t in r.tasks if t.parks]
    assert parked and all(t.finish_us is not None for t in r.tasks)
    assert all(t.reprefill_tokens > 0 for t in parked)  # the prefix was evicted
    assert all(t.reprefill_tokens == 0 for t in r.tasks if not t.parks)
    assert sorted(set(restarts)) == sorted(t.task_id for t in parked)


@pytest.mark.parametrize("arm", ["class", "las2"])
def test_a_p0_starts_within_one_lower_call_when_w_max_is_full(arm: Mode) -> None:
    lower = flat_template("p3", 30, 660)  # 10 s alone a call, no tool time
    tasks = [wtask(f"p3-{i}", 0.0, 3, lower) for i in range(4)]
    tasks.append(wtask("p0", 100.0, 0, flat_template("p0", 2, 660)))
    p = params_for_arm(arm, n_a=4)  # all four P3 calls in flight
    r = run(workload(tasks), arm, p)
    p0 = next(t for t in r.tasks if t.task_id == "p0")
    theta = ServerParams().theta_alpha
    one_call = us(10.0 * 4 / 4**theta)  # the wall time of a P3 call at n = 4
    assert p0.first_admit_us is not None
    assert p0.first_admit_us - p0.arrival_us <= one_call + us(p.tick_s)
    assert sum(t.parks for t in r.tasks) >= 1


def test_events_at_one_instant_run_completions_then_submissions_then_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps: list[tuple[int, list[Any]]] = []
    real = engine.advance

    def spy(st: State, now: Any, events: Any, p: Any) -> State:
        steps.append((now.mono_us, list(events)))
        return real(st, now, events, p)

    monkeypatch.setattr(engine, "advance", spy)
    # "a" ends at exactly 15 s, when "b" arrives and a tick is due. "z"
    # starts at 40 s with 15 s of work, "c" at 45 s with 10 s: both have 10 s
    # left at 45 s, so they end at one instant, z first although "c/0" sorts
    # before "z/0". The horizon is z's second end.
    fifteen = flat_template("15s", 1, 990)
    t_zc = us(45.0) + math.ceil(10.0 * 2 / 2**0.4 * 1e6)
    wl = workload(
        [
            wtask("a", 0.0, 2, fifteen),
            wtask("b", 15.0, 2, fifteen),
            wtask("z", 40.0, 2, flat_template("pair", 2, 990)),
            wtask("c", 45.0, 2, flat_template("one", 1, 660)),
        ]
    )
    r = run(replace(wl, horizon_us=t_zc + us(15.0)), "fifo")
    kinds = [type(e).__name__ for t, evs in steps if t == us(15.0) for e in evs]
    assert kinds.index("CallEnded") < kinds.index("CallSubmitted") < kinds.index("Tick")
    at_zc = [evs for t, evs in steps if t == t_zc]
    ended = [e.call_id for e in at_zc[0] if isinstance(e, CallEnded)]
    assert ended == ["z/0", "c/0"]  # equal ends pop start-first, in one step
    assert [type(e) for e in at_zc[1]] == [CallSubmitted]  # a further step, dt = 0
    assert max(t for t, _ in steps) < t_zc + us(15.0)  # the horizon is exclusive
    assert next(x for x in r.tasks if x.task_id == "z").finish_us is None

    # A stale completion is discarded: "e" slows down when "f" starts.
    steps.clear()
    ten = flat_template("10s", 1, 660)
    run(workload([wtask("e", 0.0, 2, ten), wtask("f", 5.0, 2, ten)], 60.0), "fifo")
    end_e = [t for t, evs in steps for e in evs if isinstance(e, CallEnded)][0]
    assert end_e == us(5.0) + math.ceil(5.0 * 2 / 2**0.4 * 1e6)

    # A tick shorter than 1 µs still advances the clock (1 µs a tick).
    steps.clear()
    tiny = run(
        workload([wtask("t", 0.0, 2, ten)], 0.001),
        "fifo",
        params_for_arm("fifo", tick_s=4e-7),
    )
    assert tiny.decisions == len(steps) == 1000
