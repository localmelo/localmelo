"""The decision rule reports guardrails, the WCT margin and S2's rule, never "adopt".

Requirement 52 on synthetic ``Metrics`` (no simulation): a guardrail
failure fails the candidate; the margin needs a 10% mean reduction with
a CI above zero over 20 seeds; every report is unvalidated; S2's rule
compares reply p95 and throughput with mode off; the front guardrails
bootstrap pooled replies with a fresh seeded generator per test, so a
candidate's report does not depend on the others compared.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from localmelo.support.scheduler.sim import compare
from localmelo.support.scheduler.sim.metrics import Metrics

VERDICTS = {"fails_guardrail", "below_margin", "insufficient_data", "meets_margin"}


def metrics(**kw: Any) -> Metrics:
    """Metrics of a run with two finished P2 tasks and no front turns."""
    values: dict[str, Any] = {
        "workload": "W1",
        "seed": 1,
        "mode": "fifo",
        "reply_whole_s": (None, None, None),
        "reply_first_s": (None, None, None),
        "reply_answer_s": (None, None, None),
        "reply_create_s": (None, None, None),
        "by_code_share": None,
        "reply_whole_us": (),
        "reply_first_us": (),
        "wct_s": 100.0,
        "turnaround_mean_s": 100.0,
        "turnaround_p95_s": 100.0,
        "finished_by_class": (0, 0, 2, 0, 0),
        "tasks_by_class": (0, 0, 2, 0, 0),
        "completion_by_class": (None, None, 1.0, None, None),
        "slowdown_mean": 1.0,
        "slowdown_p95": 1.0,
        "deadline_miss_rate": None,
        "tardiness_mean_s": None,
        "throughput_per_h": (0.0, 0.0, 1.0, 0.0, 0.0),
        "busy_share": 0.5,
        "cache_hit_share": None,
        "reprefill_per_finished": 0.0,
        "model_loads_per_h": 0.0,
        "starvation_max": 1.0,
        "starvation_total_max": 1.0,
        "start_delay_p95_s": (None,) * 5,
        "start_delay_met": (None,) * 3,
        "parks_per_h": 0.0,
        "yields_per_h": 0.0,
        "foreign_lock_task_calls": 0,
    }
    values.update(kw)
    return Metrics(**values)


def sweep(seeds: int = 20, /, **by_mode: Any) -> dict[tuple[str, str, int], Metrics]:
    """W1 results per mode, the same fields for every seed unless given per seed."""
    out = {}
    for mode, kw in by_mode.items():
        for seed in range(1, seeds + 1):
            fields = {k: v(seed) if callable(v) else v for k, v in kw.items()}
            out[("W1", mode, seed)] = metrics(mode=mode, seed=seed, **fields)
    return out


def verdict(results: Any, cand: str = "las2", **kw: Any) -> str:
    return compare(results, candidates=(cand,), **kw).candidates[0].verdict


def test_a_candidate_failing_a_guardrail_fails() -> None:
    good = {"wct_s": 50.0}
    assert verdict(sweep(fifo={}, las2=good)) == "meets_margin"
    for bad in (
        {"starvation_max": 8.5},
        {"foreign_lock_task_calls": 1},
        {"finished_by_class": (0, 0, 1, 0, 0)},  # 50% against 100%
    ):
        assert verdict(sweep(fifo={}, las2={**good, **bad})) == "fails_guardrail"
    # Exactly 5 points below passes (32/80 = 40% against 28/80 = 35%).
    p1 = {"tasks_by_class": (0, 80, 0, 0, 0)}
    edge = sweep(
        fifo={**p1, "finished_by_class": (0, 32, 0, 0, 0)},
        las2={**good, **p1, "finished_by_class": (0, 28, 0, 0, 0)},
    )
    assert verdict(edge) == "meets_margin"
    over = {**good, **p1, "finished_by_class": (0, 27, 0, 0, 0)}
    assert verdict({**edge, **sweep(las2=over)}) == "fails_guardrail"
    report = compare(sweep(fifo={}, las2={**good, "starvation_max": 9.0}))
    las2 = report.candidates[1]
    assert ("starvation", "W1", "fail") in las2.guardrails
    assert las2.not_evaluated == ("t_compat", "task_success", "injection", "restarts")


def test_the_wct_margin_needs_ten_percent_and_a_ci_above_zero_over_twenty_seeds() -> (
    None
):
    assert verdict(sweep(fifo={}, las2={"wct_s": 89.0})) == "meets_margin"
    assert verdict(sweep(fifo={}, las2={"wct_s": 91.0})) == "below_margin"
    assert verdict(sweep(19, fifo={}, las2={"wct_s": 50.0})) == "insufficient_data"
    # A mean reduction above 10% whose CI reaches below zero: three seeds
    # gain a lot, the others lose a little.
    noisy = {"wct_s": lambda s: 1.0 if s <= 3 else 105.0}
    report = compare(sweep(fifo={}, las2=noisy), candidates=("las2",))
    (row,) = report.candidates[0].primary
    workload, mean, low, high, seeds, met = row
    assert (workload, seeds, met) == ("W1", 20, False)
    assert mean is not None and mean >= 0.10 and low is not None and low <= 0
    assert report.candidates[0].verdict == "below_margin"


def test_every_report_is_unvalidated_and_never_adopts() -> None:
    results = sweep(fifo={}, las2={"wct_s": 50.0}, **{"class": {"wct_s": 120.0}})
    report = compare(results)
    assert report.validated is False
    text = report.to_json()
    assert json.loads(text)["validated"] is False and "adopt" not in text
    assert {c.verdict for c in report.candidates} <= VERDICTS
    assert [c.candidate for c in report.candidates] == ["class", "las2"]
    ties = dict((name, (c, b)) for name, c, b in report.candidates[1].tie_breaks)
    assert list(ties) == ["deadline_miss_rate", "front_p95_s", "reprefill_per_finished"]


def test_s2s_rule_compares_reply_p95_with_off() -> None:
    def replies(scale: float) -> dict[str, Any]:
        whole = tuple(int(scale * 1_000_000 * (1 + i % 10)) for i in range(20))
        return {"reply_whole_us": whole, "reply_first_us": whole}

    def rule(scale: float, finished: int = 2) -> list[tuple[str, bool]]:
        res = {}
        for mode, s in (("off", 1.0), ("las2", scale), ("fifo", 1.0)):
            done = (0, 0, finished if mode == "las2" else 2, 0, 0)
            for seed in (1, 2):
                res[("W2", mode, seed)] = metrics(
                    workload="W2",
                    mode=mode,
                    seed=seed,
                    finished_by_class=done,
                    **replies(s),
                )
        return list(compare(res, candidates=("las2",)).candidates[0].s2_rule)

    assert rule(0.7) == [("W2", True)]
    assert rule(0.8) == [("W2", False)]  # only 20% faster
    assert rule(0.7, finished=1) == [("W2", False)]  # throughput down 50%
    assert compare(sweep(fifo={}, las2={})).candidates[1].s2_rule == ()


def test_front_guardrails_bootstrap_pooled_replies_with_a_fresh_seeded_rng_per_test() -> (
    None
):
    def front(scale: float, first_s: float = 2.0, n: int = 10) -> dict[str, Any]:
        whole = tuple(int(scale * 1e6 * (5 + (i * 7) % 11)) for i in range(n))
        first = tuple(int(first_s * 1e6) for _ in range(n))
        return {"reply_whole_us": whole, "reply_first_us": first, "wct_s": 50.0}

    def guards(results: Any, cand: str = "las2") -> dict[str, str]:
        rep = next(c for c in compare(results).candidates if c.candidate == cand)
        return {g: state for g, _, state in rep.guardrails}

    ok = sweep(fifo=front(1.0), las2=front(1.0))  # 20 seeds × 10 = 200 replies
    assert guards(ok)["reply_whole"] == "pass" and guards(ok)["reply_first"] == "pass"
    slow = sweep(fifo=front(1.0), las2=front(1.3))
    assert guards(slow)["reply_whole"] == "fail"
    late = sweep(fifo=front(1.0), las2=front(1.0, first_s=12.0))
    assert guards(late)["reply_first"] == "fail"
    few = sweep(fifo=front(1.0, n=9), las2=front(1.0, n=9))  # 180 replies
    assert guards(few)["reply_whole"] == "insufficient_data"
    assert verdict(few) == "insufficient_data"
    # Completion is pooled over seeds: one seed's miss is 2.5 points overall.
    pooled = sweep(
        fifo={},
        las2={
            "wct_s": 50.0,
            "finished_by_class": lambda s: (0, 0, 1 if s == 1 else 2, 0, 0),
        },
    )
    assert guards(pooled)["completion"] == "pass"
    # The same report whatever other candidates are compared.
    extra = {
        (w, "class", s): replace(m, mode="class")
        for (w, mode, s), m in ok.items()
        if mode == "las2"
    }
    both = {**slow, **extra}
    alone = compare(slow, candidates=("las2",)).candidates[0]
    assert next(c for c in compare(both).candidates if c.candidate == "las2") == alone
