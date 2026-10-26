"""Builders for scheduler tests, with the defaults of spec section 3.

Times are given in seconds and stored in microseconds; ``state()`` sorts
its records as ``State`` requires, and ``step()`` runs one trigger the
way the simulator and the live service do: ``advance()``, ``dispatch()``,
``apply_dispatch()``. ``bounded_workload()`` is the workload of
acceptance (c), and ``offered_load()`` its ρ.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from localmelo.support.scheduler import (
    Call,
    Clock,
    Decision,
    Dep,
    Dispatch,
    Machine,
    Params,
    State,
    Task,
    advance,
    apply_dispatch,
    decide,
    dispatch,
)
from localmelo.support.scheduler.model import Event, us
from localmelo.support.scheduler.policy import _limit
from localmelo.support.scheduler.sim import Workload
from localmelo.support.scheduler.sim.trace import TaskTemplate, TemplateCall
from localmelo.support.scheduler.sim.workloads import (
    WALL_START_US,
    WorkloadTask,
    alone_s,
    synthetic_templates,
)

# Wall-clock instant of mono 0 in every test (2026-10-26, in µs).
WALL0 = 1_793_000_000_000_000


def clock(mono_s: float = 0.0) -> Clock:
    """The instant *mono_s* seconds after mono 0, on both clocks."""
    return Clock(us(mono_s), WALL0 + us(mono_s))


def task(task_id: str, **kw: Any) -> Task:
    """A task with the defaults of Requirement 13 and the given fields."""
    return Task(task_id=task_id, **kw)


def call(call_id: str, task_id: str | None = None, **kw: Any) -> Call:
    """A task call (holdable plan call by default)."""
    return Call(call_id=call_id, task_id=task_id, **kw)


def lane(call_id: str = "lane", **kw: Any) -> Call:
    """A lane call, from the front's session."""
    return Call(call_id=call_id, session_kind="front", **kw)


def machine(**kw: Any) -> Machine:
    """A machine snapshot with the given fields."""
    return Machine(**kw)


def state(
    tasks: Sequence[Task] = (),
    calls: Sequence[Call] = (),
    deps: Sequence[Dep] = (),
    **kw: Any,
) -> State:
    """A state with its records sorted as Requirement 18 requires."""
    return State(
        tasks=tuple(sorted(tasks, key=lambda t: t.task_id)),
        calls=tuple(sorted(calls, key=lambda c: c.call_id)),
        deps=tuple(sorted(deps, key=lambda d: (d.waiter, d.holder, d.kind))),
        **kw,
    )


def step(
    st: State,
    now: Clock,
    params: Params,
    events: Sequence[Event] = (),
    decide_fn: Callable[[State, Clock, Params], Decision] = decide,
) -> tuple[State, Dispatch]:
    """One trigger: advance, dispatch, apply; returns the new state and the dispatch."""
    st = advance(st, now, events, params)
    d = dispatch(st, now, params, decide_fn=decide_fn)
    return apply_dispatch(st, now, d, params), d


def reasons(d: Decision) -> dict[str, tuple[str, str]]:
    """The decision's reasons by task id."""
    return {tid: (r, det) for tid, r, det in d.reasons}


def find(st: State, task_id: str) -> Task:
    """The task *task_id* of *st*."""
    return next(t for t in st.tasks if t.task_id == task_id)


# ── Simulated workloads ──

# The window over which ρ is measured: bounded_workload's arrivals.
ARRIVAL_WINDOW_S = 3600.0

# Levels of bounded_workload's tasks other than its P4, by arrival:
# 2 P1, 12 P2 and 5 P3.
_BOUNDED_LEVELS = (2, 2, 1, 2, 3, 2, 2, 3, 2, 2, 1, 3, 2, 2, 3, 2, 2, 3, 2)


def offered_load(wl: Workload, p: Params | None = None) -> float:
    """ρ: the tasks' alone service over what k = min(limit, n_a) permits serve in an hour.

    Requirement 41, read as c-server utilisation with c = k permits of
    one alone-second per second each: the reserve and ``n_a`` bound the
    background to k calls in flight.
    """
    p = p or Params()
    k = min(_limit(p)[0], p.n_a)
    return sum(alone_s(t.template) for t in wl.tasks) / (ARRIVAL_WINDOW_S * k)


def wtask(
    task_id: str, arrival_s: float, level: int, template: TaskTemplate, **kw: Any
) -> WorkloadTask:
    """A workload task with origin user (agent for P4) and no deadline by default."""
    values: dict[str, Any] = {
        "kind": "task",
        "origin": "agent" if level == 4 else "user",
        "deadline_wall_us": None,
        "blocked_now": False,
        "size": "",
    }
    values.update(kw)
    return WorkloadTask(
        task_id=task_id,
        arrival_us=us(arrival_s),
        gold_level=level,
        level=level,
        template=template,
        **values,
    )


def flat_template(
    name: str, calls: int, completion: int, gap_s: float = 0.0
) -> TaskTemplate:
    """*calls* calls with no prompt and *completion* tokens each: D/66 s alone apiece."""
    one = TemplateCall("plan", 1, 0, completion, 0, gap_s, "ok")
    return TaskTemplate(name, (replace(one, gap_before_s=0.0),) + (one,) * (calls - 1))


def bounded_workload(seed: int) -> Workload:
    """The workload of acceptance (c) (Requirements 41 and 61): ρ = 0.60, 3 h.

    20 tasks arrive at U(0, 3600) s, task i (by arrival) on template i;
    the one with the least alone service is the P4 (origin agent). Every
    completion count is scaled by one factor f, in closed form, so that
    ``offered_load()`` is 0.60 before rounding; ``ValueError`` if no
    positive f exists.
    """
    arrivals = sorted(
        random.Random(seed).uniform(0.0, ARRIVAL_WINDOW_S) for _ in range(20)
    )
    tpl = synthetic_templates(seed)[:20]
    p4 = min(range(20), key=lambda i: (alone_s(tpl[i]), i))
    rest = iter(_BOUNDED_LEVELS)
    levels = [4 if i == p4 else next(rest) for i in range(20)]
    decode = sum(c.completion_tokens / 66.0 for t in tpl for c in t.calls)
    prefill = sum(alone_s(t) for t in tpl) - decode
    k = min(_limit(Params())[0], Params().n_a)
    f = (0.60 * ARRIVAL_WINDOW_S * k - prefill) / decode
    if not f > 0:
        raise ValueError("no positive completion factor reaches rho = 0.60")
    tasks = []
    for i, t in enumerate(tpl):
        scaled = tuple(
            replace(c, completion_tokens=round(c.completion_tokens * f))
            for c in t.calls
        )
        at = arrivals[i]
        deadline = WALL_START_US + us(at + 7200.0) if levels[i] == 1 else None
        tasks.append(
            wtask(
                f"bounded-{seed}-{i:03d}",
                at,
                levels[i],
                replace(t, calls=scaled),
                deadline_wall_us=deadline,
            )
        )
    later = [i for i in range(20) if levels[i] in (2, 3)]
    deps = tuple(
        Dep(tasks[later[w]].task_id, tasks[later[h]].task_id, confirmed=True)
        for w, h in ((4, 1), (9, 6), (14, 11))
    )
    return Workload(
        "bounded",
        seed,
        us(10800.0),
        WALL_START_US,
        tuple(tasks),
        deps=deps,
        front_messages=tuple((us(300.0 * k), "answer") for k in range(1, 24)),
        presence=((us(7200.0), us(10800.0)),),
    )
