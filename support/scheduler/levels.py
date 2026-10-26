"""Effective levels: the stored level, aging, inheritance, cycles and the WIP cap.

Step 2 of ``decide()`` (design 2.4): ``L_eff = min(L_base, L_inh, L_age,
L_dl)``. Only a user's 0 is P0 (design 2.6). Inheritance is a monotone
min-fixpoint along confirmed edges, so a task that blocks a more urgent
one runs at that task's level, capped by its origin; cycle edges pass
nothing, so content the front drew can never lower a level. ``L_dl``
(deadline promotion) is S6c's and is +inf here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from localmelo.support.scheduler.model import Clock, Dep, State, Task, is_terminal, us
from localmelo.support.scheduler.params import Params

# Lowest level inheritance may give a holder, by its origin; principals
# (user, user_rule) may take P0 from a user's P0 and P1 otherwise.
_CEILING = {"front": 2, "harness": 3, "agent": 3, "channel": 3, "sleep": 3}

# Waiter phases in which a result or lease edge passes a level (decision D14).
_WAITING = ("waiting", "parked")


@dataclass(frozen=True)
class Levels:
    """The levels of every schedulable task, by id, and what computing them found.

    ``base`` is ``L_base`` after the P0 rules, ``start`` is ``min(L_base,
    L_age)``, ``eff`` is ``L_eff``, ``d_eff`` the effective deadline
    (``None``: none). ``unmet`` lists each waiter's holders along unmet
    ``start_after`` edges.
    """

    base: dict[str, int]
    start: dict[str, int]
    eff: dict[str, int]
    d_eff: dict[str, int | None]
    late: frozenset[str]
    aged: tuple[str, ...]
    cycle: frozenset[str]
    unmet: dict[str, tuple[str, ...]]
    asks: tuple[tuple[str, str], ...]
    alarms: tuple[str, ...]


def unmet_start_after(state: State) -> dict[str, tuple[str, ...]]:
    """Holders of each schedulable waiter's ``start_after`` edges whose holder is not done.

    Confirmed or not, such an edge blocks (design 2.4); a holder id that
    names no task is never done.
    """
    by_id = {t.task_id: t for t in state.tasks}
    out: dict[str, tuple[str, ...]] = {}
    for d in state.deps:
        w = by_id.get(d.waiter)
        if d.kind != "start_after" or w is None or is_terminal(w.phase):
            continue
        h = by_id.get(d.holder)
        if h is None or h.phase != "done":
            out[d.waiter] = out.get(d.waiter, ()) + (d.holder,)
    return out


def _deadline(t: Task, p: Params) -> int | None:
    if t.deadline_wall_us is not None:
        return t.deadline_wall_us
    if t.blocked_now:
        return t.created_wall_us + us(p.implicit_deadline_s)
    return None


def _reach(edges: list[Dep]) -> dict[str, set[str]]:
    """Nodes reachable from each node along one or more edges (waiter to holder)."""
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.waiter, []).append(e.holder)
    reach: dict[str, set[str]] = {}
    for v in sorted(adj):
        seen: set[str] = set()
        stack = sorted(adj[v], reverse=True)
        while stack:
            u = stack.pop()
            if u not in seen:
                seen.add(u)
                stack.extend(sorted(adj.get(u, ()), reverse=True))
        reach[v] = seen
    return reach


def compute_levels(state: State, now: Clock, p: Params) -> Levels:
    """``L_eff`` and deadlines of every schedulable task of a cleaned *state*.

    Never raises: cycles end the fixpoint, which also stops after
    ``6n + n² + 1`` rounds with the alarm ``fixpoint_cap``.
    """
    by_id = {t.task_id: t for t in state.tasks}
    sched = [t for t in state.tasks if not is_terminal(t.phase)]
    ids = {t.task_id for t in sched}
    alarms: list[str] = []
    asks: list[tuple[str, str]] = []

    base: dict[str, int] = {}
    for t in sched:
        lv = t.level
        if lv == 0 and t.level_source != "user":
            lv = 1  # P0 is never computed (design 2.6)
            alarms.append(f"p0_not_user:{t.task_id}")
        elif lv == 0 and p.p0_meaning == "conversation":
            lv = 1
            alarms.append(f"p0_conversation:{t.task_id}")
        base[t.task_id] = lv
    # One P0: the newest user setting wins (design 2.5).
    zeros = sorted(
        (t for t in sched if base[t.task_id] == 0),
        key=lambda t: (t.created_mono_us, t.task_id),
    )
    for t in zeros[:-1]:
        base[t.task_id] = 1
        asks.append((t.task_id, "p0_conflict"))

    start: dict[str, int] = {}
    aged: list[str] = []
    for t in sched:
        ages = (
            p.aging
            and base[t.task_id] == 3
            and t.level_source != "user"
            and (t.aged or t.wait_held_us >= us(p.aging_s))
        )
        start[t.task_id] = 2 if ages else base[t.task_id]
        if ages and not t.aged:
            aged.append(t.task_id)

    def passes(e: Dep) -> bool:  # every passing condition but the cycle one
        if not (e.confirmed and e.waiter in ids and e.holder in ids):
            return False
        w = by_id[e.waiter]
        return (
            base[e.waiter] != 5
            and base[e.holder] != 5
            and not w.paused
            and (e.kind == "start_after" or w.phase in _WAITING)
        )

    acting = [
        e
        for e in state.deps
        if (e.kind == "start_after" and e.waiter in ids and e.holder in ids)
        or passes(e)
    ]
    reach = _reach(acting)
    members = sorted(v for v in reach if v in reach[v])
    grouped: set[str] = set()
    for v in members:
        if v not in grouped:
            comp = [u for u in members if u in reach[v] and v in reach[u]]
            grouped.update(comp)
            alarms.append("dependency_cycle:" + ",".join(comp))
    asks.extend((v, "cycle") for v in members)
    passing = [
        e for e in acting if passes(e) and e.waiter not in reach.get(e.holder, ())
    ]

    eff = dict(start)
    p0 = {k: base[k] == 0 for k in start}
    d_eff = {t.task_id: _deadline(t, p) for t in sched}
    if p.inheritance and passing:
        n = len(sched)
        for rounds in range(1, 6 * n + n * n + 2):
            changed = False
            for e in passing:
                w, h = e.waiter, e.holder
                cand = 0 if eff[w] == 0 and p0[w] else max(1, eff[w])
                origin = by_id[h].origin
                ceiling = _CEILING.get(origin, 0 if cand == 0 else 1)
                new = max(cand, ceiling)
                if new < eff[h]:
                    eff[h], p0[h], changed = new, new == 0, True
                dw, dh = d_eff[w], d_eff[h]
                if dw is not None and (dh is None or dw < dh):
                    d_eff[h], changed = dw, True
            if not changed:
                break
            if rounds == 6 * n + n * n + 1:
                alarms.append("fixpoint_cap")

    # WIP cap on automatic promotion to P1 (design 2.6): filters the
    # fixpoint's result without recomputing it (decision D15).
    unmet = unmet_start_after(state)
    n1 = sum(1 for k in start if start[k] == 1)
    promoted = sorted(
        (t for t in sched if eff[t.task_id] == 1 and start[t.task_id] > 1),
        key=lambda t: (
            t.task_id in unmet,
            _inf(d_eff[t.task_id]),
            _inf(t.user_rank),
            t.created_mono_us,
            t.task_id,
        ),
    )
    for t in promoted[max(0, p.p1_cap - n1) :]:
        eff[t.task_id] = min(start[t.task_id], 2)
        asks.append((t.task_id, "p1_cap"))
    if n1 > p.p1_cap:
        alarms.append("p1_over_cap")

    late = frozenset(k for k, d in d_eff.items() if d is not None and now.wall_us >= d)
    asks.extend((k, "late") for k in sorted(late))
    return Levels(
        base,
        start,
        eff,
        d_eff,
        late,
        tuple(aged),
        frozenset(members),
        unmet,
        tuple(asks),
        tuple(alarms),
    )


def _inf(v: float | None) -> float:
    """``None`` as +inf, for sort keys."""
    return math.inf if v is None else v
