"""Synthetic workloads W1, W2, W2c, W3 and W5 (design 8.2), drawn before a run.

A workload is a plain value: tasks with templates and arrival times,
front messages, away intervals and machine events. Every random draw
happens here, from one ``random.Random`` per stream seeded through
sha256, never ``hash()`` or the module-level functions, so a workload is
identical for a seed on every machine and ``run()`` draws nothing
(spec Requirement 50). W4 (lock windows, battery curves) is built by
S2b and S7 on ``machine_events``; W6 and W7 are S5's text workloads.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from localmelo.support.scheduler.costs import RATES_OMLX_OQ5, Rates, alone_service_s
from localmelo.support.scheduler.model import Dep, Machine, Origin, us
from localmelo.support.scheduler.sim.trace import TaskTemplate, TemplateCall

# The front's calls (design 3.10): prompt, new tokens per turn (the rest
# is the cached prefix), completion, and the typing lead before a message.
FRONT_PROMPT_TOKENS = 3235
FRONT_NEW_TOKENS = 400
FRONT_COMPLETION_TOKENS = 250
TYPING_LEAD_S = 20.0

# The 20 seeds of a sweep (design 8.4).
SEEDS = tuple(range(1, 21))

# The wall clock at mono 0: 2026-10-26 08:00 UTC, in µs.
WALL_START_US = 1_793_001_600_000_000

# The most calls a task makes (MAX_AGENT_STEPS).
MAX_CALLS = 30

# An hour and a minute, in seconds.
_H = 3600.0
_MIN = 60.0

WorkloadName = Literal["W1", "W2", "W2c", "W3", "W5"]


@dataclass(frozen=True)
class WorkloadTask:
    """A task of a workload; ``level`` is its initial level (the gold level in S1)."""

    task_id: str
    kind: Literal["task", "chat"]
    arrival_us: int
    origin: Origin
    gold_level: int
    level: int
    deadline_wall_us: int | None
    blocked_now: bool
    size: Literal["", "small", "medium", "large"]
    template: TaskTemplate


@dataclass(frozen=True)
class Workload:
    """What one run replays.

    ``front_messages`` holds ``(at_us, kind)`` with kind ``"answer"`` or
    ``"create"``; ``presence`` the away intervals ``(start_us, end_us)``;
    ``machine_events`` snapshots ``(at_us, Machine)`` whose
    ``away_since_mono_us`` the engine sets from ``presence``.
    """

    name: str
    seed: int
    horizon_us: int
    wall_start_us: int
    tasks: tuple[WorkloadTask, ...]
    deps: tuple[Dep, ...] = ()
    front_messages: tuple[tuple[int, str], ...] = ()
    presence: tuple[tuple[int, int], ...] = ()
    machine_events: tuple[tuple[int, Machine], ...] = ()


def _substream_seed(seed: int, stream: str) -> int:
    """The first 8 bytes of ``sha256(f"{seed}:{stream}")``, big-endian."""
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed: int, stream: str) -> random.Random:
    return random.Random(_substream_seed(seed, stream))


def alone_s(template: TaskTemplate, rates: Rates = RATES_OMLX_OQ5) -> float:
    """A template's model service alone at its warm cache, in seconds."""
    total, prev = 0.0, 0
    for c in template.calls:
        warm = c.warm_cached_tokens
        warm = min(c.prompt_tokens, prev) if warm is None else warm
        total += alone_service_s(c.prompt_tokens, warm, c.completion_tokens, rates)
        prev = c.prompt_tokens
    return total


def synthetic_templates(
    seed: int, n: int = 40, tool_pool: Sequence[float] | None = None
) -> tuple[TaskTemplate, ...]:
    """*n* templates with heavy-tailed sizes (design 8.2).

    ``1 + int(2·Pareto(1.2))`` calls, at most 30; prompts grow by 400
    tokens a call, each call's warm cache is the previous prompt; tool
    gaps U(0.05, 30) s, or U(30, 300) s one time in ten, or drawn from
    *tool_pool* (sorted tool seconds from ``history.db``) when given.
    """
    rng = _rng(seed, "templates")
    out: list[TaskTemplate] = []
    for j in range(n):
        k = min(MAX_CALLS, 1 + int(2 * rng.paretovariate(1.2)))
        calls: list[TemplateCall] = []
        prev = 0
        for i in range(k):
            prompt = 1700 + 400 * i + rng.randint(0, 300)
            completion = rng.randint(80, 600)
            if i == 0:
                gap = 0.0
            elif tool_pool:
                gap = rng.choice(tool_pool)
            elif rng.random() < 0.1:
                gap = rng.uniform(30.0, 300.0)
            else:
                gap = rng.uniform(0.05, 30.0)
            kind: Literal["plan", "reflect"] = "reflect" if i % 10 == 9 else "plan"
            calls.append(TemplateCall(kind, 1, prompt, completion, prev, gap, "ok"))
            prev = prompt
        out.append(TaskTemplate(f"synthetic-{seed}-{j:02d}", tuple(calls)))
    return tuple(out)


def _task(
    name: str,
    seed: int,
    i: int,
    arrival_s: float,
    level: int,
    template: TaskTemplate,
    deadline_s: float | None = None,
    kind: Literal["task", "chat"] = "task",
) -> WorkloadTask:
    """Task *i*; origin ``agent`` for P4, else ``user``; a deadline in seconds from mono 0."""
    deadline = None if deadline_s is None else WALL_START_US + us(deadline_s)
    return WorkloadTask(
        f"{name}-{seed}-{i:03d}",
        kind,
        us(arrival_s),
        "agent" if level == 4 else "user",
        level,
        level,
        deadline,
        False,
        "",
        template,
    )


def _messages(rng: random.Random, n: int) -> list[float]:
    """*n* front messages one a minute, each moved by U(−20, 20) s."""
    return [_MIN * (j + 1) + rng.uniform(-20.0, 20.0) for j in range(n)]


def _chain(templates: Sequence[TaskTemplate], start: int, min_s: float) -> TaskTemplate:
    """Templates from *start* on, joined until their alone service reaches *min_s*."""
    if not any(alone_s(t) > 0 for t in templates):
        raise ValueError("templates: no template has any service")
    parts: list[TaskTemplate] = []
    while sum(alone_s(t) for t in parts) < min_s:
        parts.append(templates[(start + len(parts)) % len(templates)])
    calls = tuple(c for t in parts for c in t.calls)
    return TaskTemplate("+".join(t.name for t in parts), calls)


def make_workload(
    name: WorkloadName, seed: int, *, templates: Sequence[TaskTemplate] | None = None
) -> Workload:
    """The workload *name* for *seed* (design 8.2); task i uses template i mod n."""
    tpl = tuple(templates) if templates is not None else synthetic_templates(seed)
    if not tpl:
        raise ValueError("templates: none given")
    arr, lev, dl = _rng(seed, "arrivals"), _rng(seed, "levels"), _rng(seed, "deadlines")
    pres, front = _rng(seed, "presence"), _rng(seed, "front")
    tasks: list[WorkloadTask] = []
    msgs: list[tuple[int, str]] = []
    away: tuple[tuple[int, int], ...] = ()
    if name == "W1":
        horizon = 12 * _H
        a0 = pres.uniform(2 * _H, 5 * _H)
        away = ((us(a0), us(a0 + _H)),)
        t = front.uniform(2 * _MIN, 10 * _MIN)
        while t < 8 * _H:
            if not a0 <= t < a0 + _H:
                msgs.append((us(t), "answer"))
            t += front.uniform(2 * _MIN, 10 * _MIN)
        for i, at in enumerate(sorted(arr.uniform(0.0, 7 * _H) for _ in range(24))):
            level = lev.choices((2, 1, 3, 4), weights=(70, 15, 10, 5))[0]
            if level != 4 and a0 <= at < a0 + _H:
                at = a0 + _H  # the user creates it on return
            deadline = None
            if level == 1:
                deadline = at + dl.uniform(1 * _H, 4 * _H)
            if level != 4:
                msgs.append((us(at), "create"))
            tasks.append(_task(name, seed, i, at, level, tpl[i % len(tpl)], deadline))
    elif name in ("W2", "W2c"):
        horizon = 3 * _H if name == "W2" else _H
        msgs = [(us(t), "answer") for t in _messages(front, 20)]
        n = 6 if name == "W2" else 3 + arr.randint(0, 3)
        kind: Literal["task", "chat"] = "task" if name == "W2" else "chat"
        for i, at in enumerate(sorted(arr.uniform(0.0, 120.0) for _ in range(n))):
            tasks.append(_task(name, seed, i, at, 2, tpl[i % len(tpl)], kind=kind))
    elif name == "W3":
        horizon = 6 * _H
        p1 = sorted(arr.uniform(0.0, 10 * _MIN) for _ in range(4))
        rest = sorted(arr.uniform(0.0, 30 * _MIN) for _ in range(12))
        levels = (2, 2, 3) * 4  # fixed: every third later arrival is P3
        spec: list[tuple[float, int, float | None]] = [
            (at, 1, at + m * _MIN) for at, m in zip(p1, (60, 75, 90, 105), strict=True)
        ]
        spec += [(at, lv, None) for at, lv in zip(rest, levels, strict=True)]
        for i, (at, lv, d) in enumerate(sorted(spec, key=lambda s: s[0])):
            tasks.append(_task(name, seed, i, at, lv, tpl[i % len(tpl)], d))
    elif name == "W5":
        horizon = 12 * _H
        arrivals = sorted(arr.uniform(0.0, 2 * _H) for _ in range(32))
        long = set(arr.sample(range(32), 2))
        for i, at in enumerate(arrivals):
            t_i = _chain(tpl, i, _H) if i in long else tpl[i % len(tpl)]
            tasks.append(_task(name, seed, i, at, 2, t_i))
    else:
        raise ValueError(f"name: {name!r} is not a generated workload")
    return Workload(
        name,
        seed,
        us(horizon),
        WALL_START_US,
        tuple(tasks),
        front_messages=tuple(sorted(msgs)),
        presence=away,
    )
