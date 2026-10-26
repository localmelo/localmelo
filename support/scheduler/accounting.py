"""Accounting: how events and decisions change the state over time.

``decide()`` only reads; every accumulation lives here (spec decision D4),
so time and events change the state in one visible place and replays are
deterministic. ``advance()`` books runnable waits, applies events,
charges finished calls to their root task, moves floor credit and runs
LAS-2 promotion; ``apply_dispatch()`` records a decision at the same
instant. Integer microseconds keep every sum exact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

from localmelo.support.scheduler.model import (
    ACCOUNTING_FIELDS,
    CHI_HAT_FLOOR_US,
    RECENT_ADMITS_MAX,
    Call,
    CallEnded,
    CallSubmitted,
    Clock,
    DepsSet,
    Dispatch,
    Event,
    MachineChanged,
    State,
    Task,
    TaskAdded,
    TaskUpdated,
    Typing,
    is_holdable,
    is_lane,
    is_planning,
    is_terminal,
    root_of,
    us,
)
from localmelo.support.scheduler.params import Params

# Task fields the caller may change through TaskUpdated.
_CALLER_FIELDS = (
    frozenset(f.name for f in fields(Task)) - ACCOUNTING_FIELDS - {"task_id"}
)

# Reasons that mean waiting only for order or capacity: the only runnable
# wait (decision D5), so gate, memory and own-state time never ages a task.
_RUNNABLE = (("Priority", ""), ("Resources", ""))


def _credit_cap_us(chi_hat_us: int, p: Params) -> int:
    """The floor credit cap, ``credit_cap_factor · χ̂``, rounded down to the µs once."""
    return math.floor(p.credit_cap_factor * max(chi_hat_us, CHI_HAT_FLOOR_US))


def _sorted_calls(calls: dict[str, Call]) -> tuple[Call, ...]:
    return tuple(calls[k] for k in sorted(calls))


def _sorted_tasks(tasks: dict[str, Task]) -> tuple[Task, ...]:
    return tuple(tasks[k] for k in sorted(tasks))


def advance(state: State, now: Clock, events: Sequence[Event], params: Params) -> State:
    """The state at *now*: waits booked, *events* applied, LAS-2 promotion run.

    The wait increment is capped at two ticks and never negative, so a Mac
    that slept wakes into no mass aging (design 3.3, clocks). Raises
    ``ValueError`` only for malformed events: a duplicate task or call, a
    ``TaskUpdated`` of an unknown task or of a field the caller does not
    own, or a submitted call that is already admitted.
    """
    p = params
    dt = now.mono_us - state.clock.mono_us
    anomalies = state.clock_anomalies
    if dt < 0:
        dt, anomalies = 0, anomalies + 1
    dt = min(dt, 2 * us(p.tick_s))
    tasks = {t.task_id: t for t in state.tasks}
    calls = {c.call_id: c for c in state.calls}

    # Runnable waits, from the last decision's reasons.
    waiting = {
        c.task_id
        for c in state.calls
        if c.admitted_mono_us is None and is_holdable(c) and c.task_id
    }
    roots: dict[str, None] = {}
    for t in state.tasks:
        runnable = (t.last_reason, t.last_detail) in _RUNNABLE
        if dt > 0 and runnable and t.task_id in waiting and not is_terminal(t.phase):
            tasks[t.task_id] = replace(
                t, wait_total_us=t.wait_total_us + dt, wait_held_us=t.wait_held_us + dt
            )
            roots[root_of(t, tasks).task_id] = None
    for r in roots:  # once per root per advance, however many tasks wait
        tasks[r] = replace(tasks[r], wait_promo_us=tasks[r].wait_promo_us + dt)

    deps, machine = state.deps, state.machine
    typing, front_end = state.last_typing_mono_us, state.last_front_end_mono_us
    credit = list(state.credit_us)
    for ev in events:
        if isinstance(ev, TaskAdded):
            if ev.task.task_id in tasks:
                raise ValueError(f"TaskAdded: duplicate task_id {ev.task.task_id!r}")
            tasks[ev.task.task_id] = ev.task
        elif isinstance(ev, TaskUpdated):
            cur = tasks.get(ev.task_id)
            if cur is None:
                raise ValueError(f"TaskUpdated: unknown task_id {ev.task_id!r}")
            changes: dict[str, Any] = dict(ev.changes)
            for name in changes:
                if name not in _CALLER_FIELDS:
                    raise ValueError(f"TaskUpdated: {name!r} is not a caller field")
            new = replace(cur, **changes)
            if (new.level, new.level_source) != (cur.level, cur.level_source):
                new = replace(new, aged=False)
            tasks[ev.task_id] = new
        elif isinstance(ev, DepsSet):
            deps = tuple(sorted(ev.deps, key=lambda d: (d.waiter, d.holder, d.kind)))
        elif isinstance(ev, CallSubmitted):
            if ev.call.admitted_mono_us is not None:
                raise ValueError("CallSubmitted: admitted_mono_us must be None")
            if ev.call.call_id in calls:
                raise ValueError(
                    f"CallSubmitted: duplicate call_id {ev.call.call_id!r}"
                )
            calls[ev.call.call_id] = ev.call
        elif isinstance(ev, CallEnded):
            c = calls.pop(ev.call_id, None)
            if c is None or c.admitted_mono_us is None:
                continue  # unknown, repeated or withdrawn: no charge, no credit
            charge = max(0, ev.charge_us)
            own = tasks.get(c.task_id or "")
            if own is not None:
                root = root_of(own, tasks)
                tasks[root.task_id] = replace(
                    root,
                    service_us=root.service_us + charge,
                    promo_service_us=root.promo_service_us + charge,
                    calls_done=root.calls_done + 1,
                    completion_tokens_done=root.completion_tokens_done
                    + max(0, ev.completion_tokens or 0),
                )
                own = tasks[own.task_id]
                tasks[own.task_id] = replace(
                    own,
                    last_prompt_tokens=(
                        own.last_prompt_tokens
                        if ev.prompt_tokens is None
                        else ev.prompt_tokens
                    ),
                    last_call_end_mono_us=now.mono_us,
                    last_activity_wall_us=now.wall_us,
                )
            if is_lane(c):
                front_end = now.mono_us
                continue  # lane calls add no credit
            for i, (cls, phi) in enumerate(((2, p.phi2), (3, p.phi3))):
                if c.floor_class == cls:
                    credit[i] = max(0, credit[i] - charge)
                elif (
                    c.level_at_admit is not None
                    and 1 <= c.level_at_admit < cls
                    and state.floor_open[i]
                ):
                    cap = _credit_cap_us(state.chi_hat_us[i], p)
                    credit[i] = min(credit[i] + round(phi * charge), cap)
        elif isinstance(ev, Typing):
            typing = now.mono_us
        elif isinstance(ev, MachineChanged):
            machine = ev.machine

    # LAS-2 promotion on roots, only at or above Q (spec decision D7).
    for tid in sorted(tasks):
        t = tasks[tid]
        if root_of(t, tasks).task_id != tid or t.promo_service_us < us(p.las_q_s):
            continue
        if t.wait_promo_us >= p.rho_las * t.promo_service_us:
            tasks[tid] = replace(t, promo_service_us=0, wait_promo_us=0)

    return replace(
        state,
        clock=now,
        tasks=_sorted_tasks(tasks),
        deps=deps,
        calls=_sorted_calls(calls),
        machine=machine,
        last_typing_mono_us=typing,
        last_front_end_mono_us=front_end,
        credit_us=(credit[0], credit[1]),
        clock_anomalies=anomalies,
    )


def apply_dispatch(
    state: State, now: Clock, dispatch: Dispatch, params: Params
) -> State:
    """The state after *dispatch* at *now*: admissions, levels, reasons and credit.

    Unknown or already admitted call ids in the decision are ignored.
    """
    d = dispatch.decision
    calls = {c.call_id: c for c in state.calls}
    tasks = {t.task_id: t for t in state.tasks}
    for cid in dispatch.lane_admit:
        if cid in calls:
            calls[cid] = replace(calls[cid], admitted_mono_us=now.mono_us)
    for cid, _ in dispatch.lane_answer:
        calls.pop(cid, None)
    levels = dict(d.levels)
    floor = dict(d.floor)
    recent = list(state.recent_admits)
    for cid in d.admit_unheld + d.admit:
        c = calls.get(cid)
        if c is None or c.admitted_mono_us is not None:
            continue
        lvl = levels.get(c.task_id) if c.task_id else None
        calls[cid] = replace(
            c,
            admitted_mono_us=now.mono_us,
            level_at_admit=lvl,
            floor_class=floor.get(cid),
        )
        t = tasks.get(c.task_id or "")
        if t is None or not is_planning(c):
            continue
        if is_holdable(c):
            start = now.mono_us if t.started_mono_us is None else t.started_mono_us
            t = replace(t, started_mono_us=start, wait_held_us=0)
        tasks[t.task_id] = replace(t, phase="running")
        recent.append(t.task_id)

    reasons = {tid: (r, det) for tid, r, det in d.reasons}
    aged = set(d.aged)
    for tid in sorted(tasks):
        t = tasks[tid]
        reason, detail = reasons.get(tid, ("", ""))
        new = (
            levels.get(tid, t.level_eff),
            t.aged or tid in aged,
            reason,
            detail,
            "held" if t.phase == "running" and tid in reasons else t.phase,
        )
        if new != (t.level_eff, t.aged, t.last_reason, t.last_detail, t.phase):
            tasks[tid] = replace(
                t,
                level_eff=new[0],
                aged=new[1],
                last_reason=new[2],
                last_detail=new[3],
                phase=new[4],
            )

    credit = tuple(
        (
            min(state.credit_us[i], _credit_cap_us(d.chi_hat_us[i], params))
            if d.class_runnable[i]
            else 0
        )
        for i in (0, 1)
    )
    busy = any(is_lane(c) for c in calls.values()) or any(
        levels.get(c.task_id or "", 5) <= 3
        and (
            c.admitted_mono_us is not None
            or (is_holdable(c) and reasons.get(c.task_id or "") in _RUNNABLE)
        )
        for c in calls.values()
        if not is_lane(c)
    )
    return replace(
        state,
        tasks=_sorted_tasks(tasks),
        calls=_sorted_calls(calls),
        recent_admits=tuple(recent[-RECENT_ADMITS_MAX:]),
        credit_us=(credit[0], credit[1]),
        floor_open=d.floor_open,
        chi_hat_us=d.chi_hat_us,
        battery_band=d.battery_band,
        busy_above_p4_mono_us=now.mono_us if busy else state.busy_above_p4_mono_us,
    )
