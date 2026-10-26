"""Metrics of one run (design 8.3): front replies, weighted completion, starvation.

``compute_metrics()`` reads only a ``SimResult``, so a stored result can
be scored again. Unfinished tasks count at the workload's fixed horizon,
never at the arm's last completion, so an arm that finishes late cannot
stretch its own horizon (decision D25). Percentiles use the nearest rank
and sums ``math.fsum`` over sorted values; ``to_json()`` rounds floats to
9 decimals, which goldens compare across machines (spec Requirement 50).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from localmelo.support.scheduler.sim.engine import SimResult
from localmelo.support.scheduler.sim.trace import dumps

# Weights of the gold levels P0-P4 in the weighted completion time (design 2.1).
LEVEL_WEIGHTS = (64.0, 16.0, 4.0, 1.0, 0.25)

# Targets of the start delay for P1, P2 and P3, in seconds (design 2.1).
START_TARGETS_S = (300.0, 3600.0, 86400.0)

# Three percentiles, as (p50, p95, p99), in seconds; None without samples.
Pcts = tuple[float | None, float | None, float | None]


def percentile(values: Sequence[float], p: float) -> float | None:
    """The nearest-rank *p*-th percentile; ``None`` for no values."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p * len(s) / 100) - 1)]


def _pcts(values_us: Sequence[int]) -> Pcts:
    s = [v / 1e6 for v in values_us]
    return (percentile(s, 50), percentile(s, 95), percentile(s, 99))


def mean(values: Sequence[float]) -> float | None:
    """The mean by ``math.fsum`` over sorted values; ``None`` for no values."""
    return math.fsum(sorted(values)) / len(values) if values else None


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 9)
    if isinstance(v, list | tuple):
        return [_round(x) for x in v]
    return v


@dataclass(frozen=True)
class Metrics:
    """The metrics of design 8.3 for one run; times in seconds unless named ``_us``."""

    workload: str
    seed: int
    mode: str
    reply_whole_s: Pcts
    reply_first_s: Pcts
    reply_answer_s: Pcts
    reply_create_s: Pcts
    by_code_share: float | None
    reply_whole_us: tuple[int, ...]
    reply_first_us: tuple[int, ...]
    wct_s: float | None
    turnaround_mean_s: float | None
    turnaround_p95_s: float | None
    finished_by_class: tuple[int, int, int, int, int]
    tasks_by_class: tuple[int, int, int, int, int]
    completion_by_class: tuple[float | None, ...]
    slowdown_mean: float | None
    slowdown_p95: float | None
    deadline_miss_rate: float | None
    tardiness_mean_s: float | None
    throughput_per_h: tuple[float, ...]
    busy_share: float
    cache_hit_share: float | None
    reprefill_per_finished: float | None
    model_loads_per_h: float
    starvation_max: float | None
    starvation_total_max: float | None
    start_delay_p95_s: tuple[float | None, ...]
    start_delay_met: tuple[bool | None, ...]
    parks_per_h: float
    yields_per_h: float
    foreign_lock_task_calls: int

    def to_json(self) -> str:
        """Canonical JSON with floats rounded to 9 decimals."""
        values = {k: _round(v) for k, v in asdict(self).items()}
        return dumps(values)


def compute_metrics(result: SimResult) -> Metrics:
    """The metrics of *result* (design 8.3; spec Requirement 51)."""
    r = result
    horizon, hours = r.horizon_us, r.horizon_us / 3.6e9
    s0 = float(json.loads(r.params_json)["s0_s"])
    done = [
        (
            x.kind,
            x.end_us - x.send_us,
            (x.first_event_us or x.end_us) - x.send_us,
            x.by_code,
        )
        for x in r.replies
        if x.end_us is not None
    ]
    whole = sorted(w for _, w, _, _ in done)
    first = sorted(f for _, _, f, _ in done)

    def kind(k: str) -> Pcts:
        return _pcts([w for kd, w, _, _ in done if kd == k])

    tasks = [t for t in r.tasks if 0 <= t.gold_level <= 4]
    ends = {
        t.task_id: t.finish_us if t.finish_us is not None else horizon for t in r.tasks
    }
    turn = {t.task_id: (ends[t.task_id] - t.arrival_us) / 1e6 for t in r.tasks}
    weights = [LEVEL_WEIGHTS[t.gold_level] for t in tasks]
    wsum = math.fsum(sorted(weights))
    wct = None
    if wsum > 0:
        terms = sorted(LEVEL_WEIGHTS[t.gold_level] * turn[t.task_id] for t in tasks)
        wct = math.fsum(terms) / wsum
    fin = [0] * 5
    by = [0] * 5
    for t in tasks:
        by[t.gold_level] += 1
        fin[t.gold_level] += int(t.finish_us is not None)
    slow = [turn[t.task_id] / max(t.alone_s, s0) for t in r.tasks]
    dl = [t for t in r.tasks if t.deadline_wall_us is not None]
    late = [
        max(0.0, (r.wall_start_us + ends[t.task_id] - (t.deadline_wall_us or 0)) / 1e6)
        for t in dl
    ]
    missed = [t.finish_us is None or x > 0 for t, x in zip(dl, late, strict=True)]
    prompt = sum(t.prompt_tokens for t in r.tasks)
    n_fin = sum(fin)
    starve = [(t.wait_own_s + s0) / (t.service_s + s0) for t in r.tasks]
    starve_total = [(t.wait_total_s + s0) / (t.service_s + s0) for t in r.tasks]
    delays: list[list[float]] = [[] for _ in range(5)]
    for t in tasks:
        start = t.first_admit_us if t.first_admit_us is not None else horizon
        delays[t.gold_level].append((start - t.arrival_us) / 1e6)
    p95_delay = tuple(percentile(d, 95) for d in delays)
    met = tuple(
        None if p95_delay[c] is None else (p95_delay[c] or 0.0) <= target
        for c, target in zip((1, 2, 3), START_TARGETS_S, strict=True)
    )
    return Metrics(
        workload=r.workload,
        seed=r.seed,
        mode=r.mode,
        reply_whole_s=_pcts(whole),
        reply_first_s=_pcts(first),
        reply_answer_s=kind("answer"),
        reply_create_s=kind("create"),
        by_code_share=sum(b for _, _, _, b in done) / len(done) if done else None,
        reply_whole_us=tuple(whole),
        reply_first_us=tuple(first),
        wct_s=wct,
        turnaround_mean_s=mean([turn[t.task_id] for t in tasks]),
        turnaround_p95_s=percentile([turn[t.task_id] for t in tasks], 95),
        finished_by_class=(fin[0], fin[1], fin[2], fin[3], fin[4]),
        tasks_by_class=(by[0], by[1], by[2], by[3], by[4]),
        completion_by_class=tuple(
            f / n if n else None for f, n in zip(fin, by, strict=True)
        ),
        slowdown_mean=mean(slow),
        slowdown_p95=percentile(slow, 95),
        deadline_miss_rate=sum(missed) / len(dl) if dl else None,
        tardiness_mean_s=mean(late),
        throughput_per_h=tuple(f / hours if hours else 0.0 for f in fin),
        busy_share=r.busy_us / horizon if horizon else 0.0,
        cache_hit_share=(
            sum(t.cached_tokens for t in r.tasks) / prompt if prompt else None
        ),
        reprefill_per_finished=(
            sum(t.reprefill_tokens for t in r.tasks) / n_fin if n_fin else None
        ),
        model_loads_per_h=0.0,  # one model: no loads in S1
        starvation_max=max(starve, default=None),
        starvation_total_max=max(starve_total, default=None),
        start_delay_p95_s=p95_delay,
        start_delay_met=met,
        parks_per_h=sum(t.parks for t in r.tasks) / hours if hours else 0.0,
        yields_per_h=r.yields / hours if hours else 0.0,
        foreign_lock_task_calls=r.foreign_lock_task_calls,
    )
