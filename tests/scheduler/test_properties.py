"""decide() and dispatch() keep their invariants on generated states (acceptance e, a, b).

Requirements 2, 13-18, 20, 23, 24 and 41, over states with zeros, NULLs,
NaN, oversized integers, cycles and orphans from ``gen.random_state``:
decide() never raises and keeps one reason per waiting task, the limit,
the own-state blocks, the foreign lock and its numeric floors; a blocked
started task never takes a resident slot; the lane is admitted whatever
decide() does; floor credit stays within its cap; decide() is pure; every
arm admits the same never-held calls.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

from localmelo.support.scheduler import (
    ARMS,
    Call,
    CallEnded,
    CallSubmitted,
    Clock,
    Decision,
    Dep,
    Dispatch,
    Params,
    State,
    Tick,
    advance,
    apply_dispatch,
    decide,
    decision_json,
    dispatch,
    params_for_arm,
)
from localmelo.support.scheduler.accounting import _credit_cap_us
from localmelo.support.scheduler.model import (
    CHI_HAT_FLOOR_US,
    TERMINAL_PHASES,
    clean_state,
    is_holdable,
    is_lane,
    is_planning,
)
from localmelo.support.scheduler.policy import _order_keys
from localmelo.tests.scheduler.gen import random_state


def boom(st: State, now: Clock, p: Params) -> Decision:
    raise ArithmeticError


def _active(st: State, d: Decision) -> set[str]:
    """Requirement 27's active tasks, admissions of *d* included (an independent oracle)."""
    admitted = set(d.admit) | set(d.admit_unheld)
    out = set()
    for t in st.tasks:
        mine = [c for c in st.calls if c.task_id == t.task_id]
        if (
            any(c.admitted_mono_us is not None or is_lane(c) for c in mine)
            or any(c.call_id in admitted for c in mine)
            or (
                t.phase == "running"
                and not any(is_planning(c) and c.admitted_mono_us is None for c in mine)
            )
        ):
            out.add(t.task_id)
    return out


def _blocked(st: State, task_id: str, level: int) -> bool:
    t = next(t for t in st.tasks if t.task_id == task_id)
    done = {x.task_id for x in st.tasks if x.phase == "done"}
    unmet = any(
        d.waiter == task_id and d.kind == "start_after" and d.holder not in done
        for d in st.deps
    )
    return (
        level == 5
        or t.paused
        or t.phase in ("waiting", "interrupted", *TERMINAL_PHASES)
        or unmet
    )


def test_decide_is_total_on_generated_states() -> None:
    for seed in range(300):
        st, p = random_state(random.Random(seed))
        now = st.clock
        d = decide(st, now, p)
        decision_json(Dispatch((), (), d))
        for _, key in _order_keys(st, now, p):
            assert not any(isinstance(x, float) and math.isnan(x) for x in key), seed
        assert all(c >= CHI_HAT_FLOOR_US for c in d.chi_hat_us)
        sched = {t.task_id for t in st.tasks if t.phase not in TERMINAL_PHASES}
        named = [tid for tid, _, _ in d.reasons]
        assert sorted(named) == sorted(sched - _active(st, d)), seed
        calls = {c.call_id: c for c in st.calls}
        admitted = [*d.admit_unheld, *d.admit]
        assert len(set(admitted)) == len(admitted)
        assert all(calls[c].admitted_mono_us is None for c in admitted)
        before = sum(
            1 for c in st.calls if c.admitted_mono_us is not None and not is_lane(c)
        )
        if d.admit:
            assert before + len(admitted) <= d.limit_nonlane, seed
        levels = dict(d.levels)
        for cid in d.admit:
            c = calls[cid]
            assert is_holdable(c) and c.task_id in levels
            assert not _blocked(st, c.task_id, levels[c.task_id]), seed
        if st.machine.lock == "foreign":
            assert not [c for c in admitted if is_planning(calls[c])], seed


def _starts(st: State, p: Params) -> int:
    """Admitted calls of starting (queued or parked) tasks."""
    phases = {t.task_id: t.phase for t in st.tasks}
    calls = {c.call_id: c for c in st.calls}
    d = decide(st, st.clock, p)
    return sum(
        1 for c in d.admit if phases.get(calls[c].task_id or "") in ("queued", "parked")
    )


def test_a_started_task_blocked_by_its_own_state_never_keeps_a_task_from_starting() -> (
    None
):
    exercised = 0
    for seed in range(400):
        st, p = random_state(random.Random(seed))
        # Strict class order without inheritance, so the change moves no other task's order.
        p = replace(p, arm="class", floors=False, inheritance=False)
        touched = {d.waiter for d in st.deps} | {d.holder for d in st.deps}
        started = [t for t in st.tasks if t.phase in ("running", "held")]
        queued = [
            t for t in st.tasks if t.phase == "queued" and t.task_id not in touched
        ]
        if not started or not queued:
            continue
        exercised += 1
        s, q = started[0], queued[0]
        before = _starts(st, p)
        dep = replace(st, deps=(*st.deps, Dep(s.task_id, q.task_id)))
        paused = replace(
            st, tasks=tuple(replace(t, paused=t is s or t.paused) for t in st.tasks)
        )
        assert _starts(dep, p) >= before, seed
        assert _starts(paused, p) >= before, seed
    assert exercised >= 50


def test_dispatch_admits_or_answers_the_lane_on_generated_states_when_decide_raises() -> (
    None
):
    for seed in range(300):
        st, p = random_state(random.Random(seed))
        out = dispatch(st, st.clock, p, decide_fn=boom)
        lane = [
            c for c in clean_state(st)[0].calls if is_lane(c)
        ]  # odd submit times as 0
        flying = [c for c in lane if c.admitted_mono_us is not None]
        waiting = sorted(
            (c.submit_mono_us, c.call_id) for c in lane if c.admitted_mono_us is None
        )
        assert out.decision.fallback
        if flying or not waiting:
            assert out.lane_admit == () and out.lane_answer == ()
            continue
        oldest = waiting[0][1]
        in_flight = sum(1 for c in st.calls if c.admitted_mono_us is not None)
        respect = st.machine.lock == "foreign" and p.reservation_policy == "respect"
        if out.lane_admit:
            assert out.lane_admit == (oldest,) and out.lane_answer == ()
        elif out.lane_answer:
            assert out.lane_answer[0][0] == oldest and len(out.lane_answer) == 1
        else:
            assert respect or in_flight >= p.n_srv, seed


def test_floor_credit_stays_within_its_cap_on_generated_event_sequences() -> None:
    for seed in range(50):
        rng = random.Random(1000 + seed)
        st, p = random_state(rng)
        now = st.clock
        for i in range(20):
            now = Clock(now.mono_us + rng.randint(-5, 60) * 1_000_000, now.wall_us + 1)
            events: list[Any] = [Tick()]
            flying = [c for c in st.calls if c.admitted_mono_us is not None]
            for c in rng.sample(flying, min(len(flying), rng.randint(0, 3))):
                charge = rng.choice((0, 5_000_000, 600_000_000, 10**12))
                events.append(
                    CallEnded(c.call_id, "ok", charge, None, rng.randint(0, 900))
                )
            if st.tasks:
                tid = rng.choice(st.tasks).task_id
                events.append(
                    CallSubmitted(_new_call(f"n{seed}-{i}", tid, now.mono_us, rng))
                )
            st = advance(st, now, events, p)
            out = dispatch(st, now, p)
            st = apply_dispatch(st, now, out, p)
            for k in (0, 1):
                assert 0 <= st.credit_us[k] <= _credit_cap_us(st.chi_hat_us[k], p), seed


def _new_call(call_id: str, task_id: str, mono: int, rng: random.Random) -> Call:
    kind: Any = rng.choice(("plan", "plan", "reflect", "embed"))
    return Call(
        call_id, task_id, kind=kind, submit_mono_us=mono, attempt=rng.choice((1, 1, 2))
    )


def test_decide_is_a_pure_function() -> None:
    for seed in range(100):
        st, p = random_state(random.Random(seed))
        before = repr(st)
        first = decide(st, st.clock, p)
        again, _ = random_state(random.Random(seed))
        assert decide(again, again.clock, p) == first
        assert decision_json(Dispatch((), (), first)) == decision_json(
            Dispatch((), (), decide(st, st.clock, p))
        )
        assert repr(st) == before


def test_every_arm_admits_the_same_never_held_calls() -> None:
    for seed in range(200):
        st, p = random_state(random.Random(seed))
        unheld = set()
        for arm in ARMS:
            q = replace(p, arm=arm, floors=False, yield_marks=arm != "fifo")
            unheld.add(decide(st, st.clock, q).admit_unheld)
        assert len(unheld) == 1, seed
        assert params_for_arm("fifo").arm == "fifo"
