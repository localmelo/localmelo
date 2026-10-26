"""The decision rule of design 8.4, reported only: guardrails, the WCT margin, S2's rule.

A sweep runs workloads, modes and seeds; ``compare()`` judges each
candidate arm against a baseline with the guardrails a simulator can
evaluate, the primary WCT margin with a paired bootstrap over seeds, and
the tie-breaks. Every report says ``validated: false`` and gives a
verdict, never "adopt": no default changes because of a simulation until
a live window passes the 15% check (design 8.1). Each bootstrap test
draws from its own ``random.Random`` with one fixed seed, so a report
does not depend on which other tests ran.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

from localmelo.support.scheduler.params import Params
from localmelo.support.scheduler.sim.engine import Mode, run
from localmelo.support.scheduler.sim.metrics import (
    Metrics,
    compute_metrics,
    mean,
    percentile,
)
from localmelo.support.scheduler.sim.server import ServerParams
from localmelo.support.scheduler.sim.trace import dumps
from localmelo.support.scheduler.sim.workloads import SEEDS, WorkloadName, make_workload

# Bootstrap resamples and the seed of every test's own generator (decision D36).
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261026

# The primary margin: mean relative WCT reduction, over at least 20 seeds.
MARGIN = 0.10
MIN_SEEDS = 20

# Guardrails (design 8.4): completion no more than 5 points below the
# baseline, starvation ratio at most 8, whole-reply p95 ratio at most 1.10
# and first-event p95 at most 10 s at the 97.5th percentile, from pools of
# at least 200 replies.
COMPLETION_DROP = 0.05
STARVATION_MAX = 8.0
REPLY_RATIO_MAX = 1.10
FIRST_EVENT_MAX_S = 10.0
MIN_REPLIES = 200

# S2's rule: front whole-reply p95 25% below off, throughput within 5%.
S2_REPLY_CUT = 0.25
S2_THROUGHPUT_DROP = 0.05

# Workloads of the primary metric and of S2's rule.
PRIMARY = ("W1", "W3", "W5")
S2_WORKLOADS = ("W2", "W2c")

# Guardrails of design 8.4 that a simulator cannot evaluate.
NOT_EVALUATED = ("t_compat", "task_success", "injection", "restarts")

Results = Mapping[tuple[str, str, int], Metrics]


@dataclass(frozen=True)
class CandidateReport:
    """One candidate's verdict and its evidence.

    ``guardrails`` holds ``(guardrail, workload, "pass" | "fail" |
    "insufficient_data")``; ``primary`` ``(workload, mean, ci_low,
    ci_high, seeds, met)``; ``tie_breaks`` ``(name, candidate, baseline)``.
    """

    candidate: str
    verdict: str
    guardrails: tuple[tuple[str, str, str], ...]
    primary: tuple[tuple[str, float | None, float | None, float | None, int, bool], ...]
    s2_rule: tuple[tuple[str, bool], ...]
    tie_breaks: tuple[tuple[str, float | None, float | None], ...]
    not_evaluated: tuple[str, ...] = NOT_EVALUATED


@dataclass(frozen=True)
class ComparisonReport:
    """The comparison of each candidate with the baseline; never validated in S1."""

    baseline: str
    candidates: tuple[CandidateReport, ...]
    validated: bool = False

    def to_json(self) -> str:
        return dumps(asdict(self))


def run_sweep(
    workloads: Sequence[str],
    modes: Sequence[str],
    seeds: Sequence[int] = SEEDS,
    params: Mapping[str, Params] | None = None,
    server: ServerParams | None = None,
) -> dict[tuple[str, str, int], Metrics]:
    """Metrics of every (workload, mode, seed), run in sorted order; CPU-heavy at full size."""
    out: dict[tuple[str, str, int], Metrics] = {}
    for w, mode, seed in sorted(itertools.product(workloads, modes, seeds)):
        wl = make_workload(cast(WorkloadName, w), seed)
        p = (params or {}).get(mode)
        out[(w, mode, seed)] = compute_metrics(run(wl, cast(Mode, mode), p, server))
    return out


def _runs(results: Results, w: str, mode: str) -> list[Metrics]:
    return [results[k] for k in sorted(results) if k[0] == w and k[1] == mode]


def _q(values: list[float], p: float) -> float:
    return percentile(values, p) or 0.0


def _p95_ratio(cand: list[int], base: list[int], rng: random.Random) -> float:
    """The 97.5th percentile of p95(candidate) / p95(baseline) over resamples."""
    ratios = []
    for _ in range(BOOTSTRAP_B):
        c = _q([float(x) for x in rng.choices(cand, k=len(cand))], 95)
        b = _q([float(x) for x in rng.choices(base, k=len(base))], 95)
        ratios.append(c / b if b > 0 else (1.0 if c == 0 else math.inf))
    return _q(ratios, 97.5)


def _guardrails(
    results: Results, w: str, cand: str, base: str
) -> list[tuple[str, str, str]]:
    cr, br = _runs(results, w, cand), _runs(results, w, base)
    out = []
    ok = True
    for k in range(5):
        nc, nb = sum(m.tasks_by_class[k] for m in cr), sum(
            m.tasks_by_class[k] for m in br
        )
        if nc and nb:
            rc = sum(m.finished_by_class[k] for m in cr) / nc
            rb = sum(m.finished_by_class[k] for m in br) / nb
            ok = ok and rc >= rb - COMPLETION_DROP
    out.append(("completion", w, "pass" if ok else "fail"))
    worst = max((m.starvation_max or 0.0 for m in cr), default=0.0)
    out.append(("starvation", w, "pass" if worst <= STARVATION_MAX else "fail"))
    foreign = sum(m.foreign_lock_task_calls for m in cr)
    out.append(("foreign_lock", w, "pass" if foreign == 0 else "fail"))
    whole_c = [x for m in cr for x in m.reply_whole_us]
    whole_b = [x for m in br for x in m.reply_whole_us]
    first_c = [x for m in cr for x in m.reply_first_us]
    if not (whole_c or whole_b):
        return out  # no front turns in this workload
    if min(len(whole_c), len(whole_b), len(first_c)) < MIN_REPLIES:
        return out + [
            (g, w, "insufficient_data") for g in ("reply_first", "reply_whole")
        ]
    # Tests run in sorted order of (workload, candidate, metric), each fresh.
    rng = random.Random(BOOTSTRAP_SEED)
    firsts = [
        _q([float(x) for x in rng.choices(first_c, k=len(first_c))], 95) / 1e6
        for _ in range(BOOTSTRAP_B)
    ]
    first_ok = _q(firsts, 97.5) <= FIRST_EVENT_MAX_S
    out.append(("reply_first", w, "pass" if first_ok else "fail"))
    ratio = _p95_ratio(whole_c, whole_b, random.Random(BOOTSTRAP_SEED))
    out.append(("reply_whole", w, "pass" if ratio <= REPLY_RATIO_MAX else "fail"))
    return out


def _primary(
    results: Results, w: str, cand: str, base: str
) -> tuple[str, float | None, float | None, float | None, int, bool]:
    """The mean relative WCT reduction over paired seeds and its bootstrap 95% CI."""
    by_seed = {k[2]: m for k, m in results.items() if k[0] == w and k[1] == base}
    red = []
    for k in sorted(results):
        if k[0] == w and k[1] == cand and k[2] in by_seed:
            wc, wb = results[k].wct_s, by_seed[k[2]].wct_s
            if wc is not None and wb:
                red.append(1.0 - wc / wb)
    if not red:
        return (w, None, None, None, 0, False)
    mean = math.fsum(red) / len(red)
    rng = random.Random(BOOTSTRAP_SEED)
    means = [
        math.fsum(rng.choices(red, k=len(red))) / len(red) for _ in range(BOOTSTRAP_B)
    ]
    lo, hi = _q(means, 2.5), _q(means, 97.5)
    met = len(red) >= MIN_SEEDS and mean >= MARGIN and lo > 0
    return (w, mean, lo, hi, len(red), met)


def _s2_rule(results: Results, w: str, cand: str) -> tuple[str, bool] | None:
    cr, off = _runs(results, w, cand), _runs(results, w, "off")
    if not (cr and off):
        return None
    pc = percentile([float(x) for m in cr for x in m.reply_whole_us], 95)
    po = percentile([float(x) for m in off for x in m.reply_whole_us], 95)
    tc = sum(sum(m.finished_by_class) for m in cr)
    to = sum(sum(m.finished_by_class) for m in off)
    faster = pc is not None and po is not None and pc <= (1 - S2_REPLY_CUT) * po
    return (w, faster and tc >= (1 - S2_THROUGHPUT_DROP) * to)


def _mean(values: list[float | None]) -> float | None:
    return mean([v for v in values if v is not None])


def compare(
    results: Results,
    baseline: str = "fifo",
    candidates: Sequence[str] = ("class", "las2"),
) -> ComparisonReport:
    """Each candidate against *baseline* (design 8.4); a report, never a decision."""
    workloads = sorted({k[0] for k in results})
    reports = []
    for cand in candidates:
        both = [
            w
            for w in workloads
            if _runs(results, w, cand) and _runs(results, w, baseline)
        ]
        guards = [g for w in both for g in _guardrails(results, w, cand, baseline)]
        primary = [_primary(results, w, cand, baseline) for w in both if w in PRIMARY]
        s2 = [r for w in S2_WORKLOADS if (r := _s2_rule(results, w, cand))]
        mine = [m for k, m in sorted(results.items()) if k[1] == cand]
        base = [m for k, m in sorted(results.items()) if k[1] == baseline]
        ties = [
            (name, _mean([f(m) for m in mine]), _mean([f(m) for m in base]))
            for name, f in (
                ("deadline_miss_rate", lambda m: m.deadline_miss_rate),
                ("front_p95_s", lambda m: m.reply_whole_s[1]),
                ("reprefill_per_finished", lambda m: m.reprefill_per_finished),
            )
        ]
        states = {g[2] for g in guards}
        if "fail" in states:
            verdict = "fails_guardrail"
        elif (
            "insufficient_data" in states
            or not primary
            or any(p[4] < MIN_SEEDS for p in primary)
        ):
            verdict = "insufficient_data"
        else:
            verdict = "meets_margin" if all(p[5] for p in primary) else "below_margin"
        reports.append(
            CandidateReport(
                cand, verdict, tuple(guards), tuple(primary), tuple(s2), tuple(ties)
            )
        )
    return ComparisonReport(baseline, tuple(reports))
