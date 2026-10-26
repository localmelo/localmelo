"""Builders for scheduler tests, with the defaults of spec section 3.

Times are given in seconds and stored in microseconds; ``state()`` sorts
its records as ``State`` requires, and ``step()`` runs one trigger the
way the simulator and the live service do: ``advance()``, ``dispatch()``,
``apply_dispatch()``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
