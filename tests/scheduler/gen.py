"""Random scheduler states for the property tests (spec Requirement 61).

``random_state(rng)`` draws a state that is valid by type but odd by
value: zeros, NULLs, NaN and inf, oversized integers, out-of-range
levels, dangling ids, self-loops and cycles, orphan calls. Its machine
and own-state draws close almost every call, so ``plain_state(rng)``
keeps its levels, edges and parameters on an open machine with
schedulable tasks, where ``decide()`` admits calls. No hypothesis: the
tests loop over fixed seeds of ``random.Random``.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

from localmelo.support.scheduler import Call, Clock, Dep, Machine, Params, State, Task
from localmelo.support.scheduler.model import MAX_INT, MAX_TOKENS
from localmelo.support.scheduler.params import params_for_arm

_LEVELS = (-1, 0, 1, 2, 3, 4, 5, 9)
_PHASES: tuple[Any, ...] = (
    "queued",
    "running",
    "held",
    "waiting",
    "parked",
    "interrupted",
    "done",
    "failed",
    "cancelled",
    "expired",
)
_ORIGINS: tuple[Any, ...] = (
    "user",
    "user_rule",
    "front",
    "harness",
    "agent",
    "channel",
    "sleep",
)
_REASONS = (
    ("", ""),
    ("Priority", ""),
    ("Resources", ""),
    ("Battery", ""),
    ("Memory", "kv"),
)


def _float(rng: random.Random, lo: float, hi: float) -> float | None:
    r = rng.random()
    if r < 0.05:
        return rng.choice((math.nan, math.inf, -math.inf))
    return None if r < 0.4 else rng.uniform(lo, hi)


def _int(rng: random.Random, hi: int, optional: bool, cap: int = MAX_INT) -> int | None:
    r = rng.random()
    if r < 0.02:
        return cap + rng.randint(1, 9)
    if r < 0.04:
        return -rng.randint(1, 9)
    if optional and r < 0.4:
        return None
    return 0 if r < 0.55 else rng.randint(0, hi)


def _req(v: int | None) -> int:
    return 0 if v is None else v


def _task(rng: random.Random, tid: str, ids: list[str], now: Clock) -> Task:
    mono, wall = now.mono_us, now.wall_us
    return Task(
        task_id=tid,
        root_id=rng.choice(["", tid, "ghost", *ids]),
        kind=rng.choice(("task", "chat", "front_bg")),
        origin=rng.choice(_ORIGINS),
        level=rng.choice(_LEVELS),
        level_source=rng.choice(("user", "user_rule", "computed")),
        phase=rng.choice(_PHASES),
        wait_reason=rng.choice(("", "input", "approval", "dependency", "lease")),
        paused=rng.random() < 0.15,
        user_rank=_float(rng, -3.0, 3.0),
        deadline_wall_us=_int(rng, wall + 10**10, True),
        blocked_now=rng.random() < 0.3,
        size=rng.choice(("", "small", "medium", "large")),
        remaining_s_est=_float(rng, -10.0, 5000.0),
        tools_left_s_est=_float(rng, 0.0, 600.0),
        budget_exhausted=rng.random() < 0.1,
        created_mono_us=_req(_int(rng, mono, False)),
        created_wall_us=_req(_int(rng, wall, False)),
        mention_mono_us=_int(rng, mono + 10**8, True),
        started_mono_us=_int(rng, mono, True),
        last_activity_wall_us=_int(rng, wall, True),
        service_us=_req(_int(rng, 10**10, False)),
        promo_service_us=_req(_int(rng, 10**9, False)),
        wait_promo_us=_req(_int(rng, 10**10, False)),
        wait_total_us=_req(_int(rng, 10**11, False)),
        wait_held_us=_req(_int(rng, 10**11, False)),
        calls_done=_req(_int(rng, 50, False)),
        completion_tokens_done=_req(_int(rng, 20000, False, MAX_TOKENS)),
        last_prompt_tokens=_int(rng, 9000, True, MAX_TOKENS),
        last_call_end_mono_us=_int(rng, mono, True),
        aged=rng.random() < 0.2,
        level_eff=_int(rng, 5, True),
        last_reason=(reason := rng.choice(_REASONS))[0],
        last_detail=reason[1],
    )


def _call(rng: random.Random, cid: str, ids: list[str], now: Clock) -> Call:
    kind = rng.choice(("lane", "lane_chat", "holdable", "unheld", "retry", "orphan"))
    task_id: str | None = rng.choice(ids) if ids else None
    session: Any = "task"
    ck: Any = rng.choice(("plan", "reflect"))
    attempt = 1
    if kind == "lane":
        session, task_id = "front", None
    elif kind == "lane_chat":
        session = "front"
    elif kind == "unheld":
        ck = rng.choice(("embed", "retrieve", "estimate"))
        attempt = rng.choice((1, 2))
    elif kind == "retry":
        attempt = rng.choice((2, 3))
    elif kind == "orphan":
        task_id = rng.choice((None, "ghost"))
    return Call(
        call_id=cid,
        task_id=task_id,
        session_kind=session,
        kind=ck,
        attempt=attempt,
        submit_mono_us=_req(_int(rng, now.mono_us, False)),
        prompt_tokens=_int(rng, 9000, True, MAX_TOKENS),
        completion_p99_tokens=_int(rng, 3000, True, MAX_TOKENS),
        admitted_mono_us=_int(rng, now.mono_us, False) if rng.random() < 0.4 else None,
        floor_class=rng.choice((None, None, 2, 3)),
        level_at_admit=rng.choice((None, 0, 1, 2, 3, 4)),
    )


def _deps(rng: random.Random, ids: list[str]) -> list[Dep]:
    names = [*ids, "ghost"]
    kinds: tuple[Any, ...] = ("start_after", "result", "lease")
    out = [
        Dep(rng.choice(names), rng.choice(names), rng.choice(kinds), rng.random() < 0.6)
        for _ in range(rng.randint(0, 6))
    ]
    if len(ids) >= 3 and rng.random() < 0.5:  # a self-loop, a 2-cycle and a 3-cycle
        a, b, c = rng.sample(ids, 3)
        conf = rng.random() < 0.7
        out += [Dep(a, a, rng.choice(kinds), conf), Dep(a, b, rng.choice(kinds), conf)]
        out += [Dep(b, a, rng.choice(kinds), conf), Dep(b, c, "start_after", conf)]
        out += [Dep(c, a, rng.choice(kinds), conf)]
    return out


def _params(rng: random.Random) -> Params:
    arm = rng.choice(("fifo", "class", "las2"))
    small = rng.random() < 0.5
    over: dict[str, Any] = {
        "n_srv": rng.randint(1, 8),
        "w_max": rng.randint(1, 4),
        "p1_cap": rng.randint(0, 3),
        "kv_bytes_per_token": rng.choice((0.0, 131072.0)),
        "m_kv_bytes": 2e9 if small else math.inf,
        "m_hot_bytes": rng.choice((math.inf, 1e9)),
        "phi2": rng.choice((0.0, 0.2, 1.0)),
        "phi3": rng.choice((0.0, 0.2, 1.0)),
        "n_a": rng.choice((1, 3)),
        "n_fg": rng.choice((0, 1, 3)),
        "r_f": rng.choice((0, 1, 8)),
        "h_u_s": rng.choice((1e-6, 14400.0)),
        "kappa_s": rng.choice((1e-9, 1.5)),
        "u_max": rng.choice((1.0, 4.0)),
        "battery_gate": rng.choice(("size_and_class", "size_only", "class_only")),
        "reservation_policy": rng.choice(("ask", "respect", "front_only")),
        "p0_meaning": rng.choice(("task", "conversation")),
        "w_floor_s": rng.choice((1.0, 900.0)),
        "t_park_s": rng.choice((1.0, 1800.0)),
        "aging_s": rng.choice((1.0, 86400.0)),
    }
    for name in (
        "aging",
        "inheritance",
        "idle_gate",
        "battery_gate_enforce",
        "lock_lines",
    ):
        over[name] = rng.random() < 0.5
    if arm != "fifo":
        over["floors"] = rng.random() < 0.5
        over["yield_marks"] = rng.random() < 0.5
    return params_for_arm(arm, **over)


def random_state(rng: random.Random) -> tuple[State, Params]:
    """A random state and parameters; ``state.clock`` is the decision's ``now``."""
    mono = rng.randint(0, 10**11)
    now = Clock(mono, 1_793_000_000_000_000 + mono)
    ids = [f"t{i:02d}" for i in range(rng.randint(0, 12))]
    tasks = [_task(rng, tid, ids, now) for tid in ids]
    calls = [_call(rng, f"c{i:02d}", ids, now) for i in range(rng.randint(0, 10))]
    m = Machine(
        lock=rng.choice(("none", "self", "foreign")),
        lock_mem_gib=rng.choice((0.0, 8.0, math.nan)),
        on_ac=rng.random() < 0.5,
        battery_pct=rng.choice((None, 5.0, 45.0, 55.0, 85.0, math.nan)),
        low_power=rng.random() < 0.2,
        thermal=rng.choice(("nominal", "fair", "serious", "critical")),
        memory_pressure=rng.choice(("normal", "warning", "critical")),
        server_up=rng.random() < 0.85,
        model=rng.choice(("loaded", "loaded", "draining", "unloaded", "loading")),
        away_since_mono_us=_int(rng, mono, True),
    )
    st = State(
        clock=now,
        tasks=tuple(tasks),
        deps=tuple(sorted(_deps(rng, ids), key=lambda d: (d.waiter, d.holder, d.kind))),
        calls=tuple(calls),
        machine=m,
        battery_band=rng.choice((0, 1, 2, 4, 9)),
        last_typing_mono_us=_int(rng, mono, True),
        last_front_end_mono_us=_int(rng, mono, True),
        credit_us=(_req(_int(rng, 10**8, False)), _req(_int(rng, 10**8, False))),
        floor_open=(rng.random() < 0.5, rng.random() < 0.5),
        chi_hat_us=(_req(_int(rng, 10**8, False)), _req(_int(rng, 10**8, False))),
        busy_above_p4_mono_us=_int(rng, mono, True),
        recent_admits=tuple(rng.choice(ids) for _ in range(rng.randint(0, 16)) if ids),
    )
    return st, _params(rng)


def plain_state(rng: random.Random) -> tuple[State, Params]:
    """A random state on an open machine whose tasks can be scheduled.

    Levels, edges, counters and parameters come from ``random_state()``;
    each task gets a started or startable phase and at most one call,
    pending or in flight, plus at most one lane call.
    """
    st, p = random_state(rng)
    tasks = tuple(
        replace(
            t,
            phase=rng.choice(("queued", "running", "held", "parked")),
            paused=False,
            wait_reason="",
            budget_exhausted=False,
        )
        for t in st.tasks
    )
    calls: list[Call] = []
    for i, t in enumerate(tasks):
        r = rng.random()
        if r < 0.6:
            submit = rng.randint(0, st.clock.mono_us)
            calls.append(Call(f"p{i:02d}", t.task_id, submit_mono_us=submit))
        elif r < 0.85:
            calls.append(
                Call(
                    f"p{i:02d}",
                    t.task_id,
                    admitted_mono_us=0,
                    level_at_admit=rng.choice((1, 2, 3)),
                    floor_class=rng.choice((None, 2, 3)),
                )
            )
    if rng.random() < 0.5:
        calls.append(Call("lane", None, session_kind="front"))
    machine = Machine(away_since_mono_us=rng.choice((None, 0)))
    st = replace(st, tasks=tasks, calls=tuple(calls), machine=machine, battery_band=4)
    return st, p
