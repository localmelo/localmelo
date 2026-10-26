"""The cost model charges time alone, falls back in order, and predicts the cache.

Requirements 42 and 43: an ok call is charged its alone service from
tokens, a failed one its server timings or its wall share; no charge is
negative or NaN; the cache proxy predicts a hit only for a recent call of
the same session with few other sessions in between.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from localmelo.support.scheduler import Params
from localmelo.support.scheduler.costs import (
    RATES_OMLX_OQ5,
    Rates,
    alone_service_s,
    charge_s,
    fit_rates,
    others_since,
    predict_cached,
)

R = Rates(1000.0, 100.0)


def test_an_ok_call_is_charged_its_time_alone() -> None:
    expected = (3000 - 1000) / 1000.0 + 200 / 100.0
    for mean_inflight in (1.0, 4.0, 8.0):
        assert charge_s("ok", 3000, 1000, 200, 9.0, 9.0, 99.0, mean_inflight, R) == (
            expected,
            "tokens",
        )
    assert Rates() == RATES_OMLX_OQ5


def test_cached_tokens_are_clamped_to_the_prompt() -> None:
    assert alone_service_s(1000, -50, 0, R) == 1.0
    assert alone_service_s(1000, 5000, 100, R) == 1.0
    assert alone_service_s(1000, None, 0, R) == 1.0


def test_a_failed_call_is_charged_from_server_timings_then_its_wall_share() -> None:
    assert charge_s("failed", 3000, 0, 200, 1.5, 2.5, 30.0, 3.0, R) == (4.0, "server")
    assert charge_s("timeout", None, None, None, None, 2.5, 30.0, 3.0, R) == (
        10.0,
        "wall_share",
    )
    assert charge_s("aborted", None, None, None, -1.0, 2.0, 6.0, 0.5, R) == (
        6.0,
        "wall_share",
    )
    assert charge_s("ok", 100, 0, None, 1.0, 1.0, None, None, R) == (2.0, "server")
    assert charge_s("failed", None, None, None, None, None, None, None, R) == (
        0.0,
        "none",
    )


@pytest.mark.parametrize(
    "args",
    [
        ("ok", 0, 0, 0, None, None, None, None),
        ("ok", None, None, None, None, None, None, None),
        ("failed", None, None, None, math.nan, math.inf, math.nan, math.nan),
        ("failed", None, None, None, None, None, 5.0, math.nan),
        ("ok", -5, -5, -5, None, None, None, None),
    ],
)
def test_zero_or_missing_tokens_cost_nothing_and_never_nan(
    args: tuple[Any, ...],
) -> None:
    outcome, p, h, d, server_p, server_g, wall, mean = args
    value, _ = charge_s(outcome, p, h, d, server_p, server_g, wall, mean, R)
    assert not math.isnan(value) and value >= 0
    assert alone_service_s(0, 0, 0, R) == 0.0
    assert alone_service_s(None, None, None, R) == 0.0


def test_the_cache_proxy_predicts_a_hit_only_within_60_s_and_h_c_other_tasks() -> None:
    p = Params(shared_prefix_tokens=100)
    s = 1_000_000
    assert predict_cached(3000, 2800, 0, 59 * s, 2, p) == 2800  # hit: min(P, last)
    assert predict_cached(2000, 2800, 0, 59 * s, 2, p) == 2000
    assert predict_cached(3000, 2800, 0, 60 * s, 2, p) == 100  # window over: miss
    assert predict_cached(3000, 2800, 0, 10 * s, 3, p) == 100  # too many others
    assert predict_cached(3000, None, 0, 10 * s, 0, p) == 100  # no previous prompt
    assert predict_cached(3000, 2800, None, 10 * s, 0, p) == 100
    assert predict_cached(3000, 2800, 20 * s, 10 * s, 0, p) == 100  # end in the future
    assert predict_cached(50, None, None, 0, 0, p) == 50  # never above P
    assert others_since(("a", "b", "a", "c", "c", "d"), "a") == 2
    assert others_since(("a", "b"), "b") == 0
    assert others_since(("a", "b", "a"), "z") == 3
    assert others_since((), "z") == 1


def test_fit_rates_needs_five_samples_and_uses_server_timings() -> None:
    row = (2000, 1000, 300, 1.0, 3.0)
    assert fit_rates([row] * 4) is None
    rates = fit_rates([row] * 5)
    assert rates == Rates(1000.0, 100.0)
    assert fit_rates([(2000, None, 300, 0.0, 3.0)] * 5) is None  # zero prompt time
    assert fit_rates([(2000, 2000, 300, 1.0, 3.0)] * 5) is None  # no prefill: rate 0
