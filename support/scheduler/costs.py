"""The cost model: what a call costs alone, what it is charged, the cache proxy.

A task is charged the seconds its call would take alone on the server
(time-alone), so its charge does not depend on its batch-mates (design
3.2). Before R1 reports cached tokens, the cache is predicted from the
task's own last call (design 3.2, the magnitude check). Every function is
total: ``None`` counts as zero or unknown as stated, and no result is
negative or NaN.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from localmelo.support.scheduler.model import us
from localmelo.support.scheduler.params import Params


@dataclass(frozen=True)
class Rates:
    """Prefill and decode rates in tokens per second, both finite and positive."""

    prefill_tok_s: float = 829.0
    decode_tok_s: float = 66.0

    def __post_init__(self) -> None:
        for name in ("prefill_tok_s", "decode_tok_s"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name}: must be finite and positive")


# F3's fits for oQ5 on oMLX and for rec; valid until S0 measures new ones
# (design 3.2).
RATES_OMLX_OQ5 = Rates(829.0, 66.0)
RATES_REC_FLINT = Rates(1730.0, 164.0)


def alone_service_s(
    prompt: int | None, cached: int | None, completion: int | None, rates: Rates
) -> float:
    """χ = (P − H)/r_p + D/r_d; ``None`` is 0, P and D floor at 0, 0 ≤ H ≤ P."""
    p = max(0, prompt or 0)
    h = min(max(0, cached or 0), p)
    d = max(0, completion or 0)
    return (p - h) / rates.prefill_tok_s + d / rates.decode_tok_s


def _non_negative(v: float | None) -> bool:
    return v is not None and math.isfinite(v) and v >= 0


def charge_s(
    outcome: str,
    prompt: int | None,
    cached: int | None,
    completion: int | None,
    server_prompt_s: float | None,
    server_generation_s: float | None,
    wall_s: float | None,
    mean_inflight: float | None,
    rates: Rates,
) -> tuple[float, str]:
    """The charge of a finished call in seconds and its source (decision D33).

    Every outcome is charged: tokens for an ``ok`` call, else the server's
    timings, else the wall time divided by the mean calls in flight (the
    call's share of a processor-shared server), else nothing.
    """
    if outcome == "ok" and prompt is not None and completion is not None:
        return alone_service_s(prompt, cached, completion, rates), "tokens"
    if (
        server_prompt_s is not None
        and server_generation_s is not None
        and _non_negative(server_prompt_s)
        and _non_negative(server_generation_s)
    ):
        return server_prompt_s + server_generation_s, "server"
    if wall_s is not None and _non_negative(wall_s):
        share = mean_inflight if _non_negative(mean_inflight) else 1.0
        return wall_s / max(1.0, share or 1.0), "wall_share"
    return 0.0, "none"


def others_since(recent_admits: Sequence[str], task_id: str) -> int:
    """Distinct ids admitted after *task_id*'s last admission.

    When *task_id* is not in the window, the distinct ids plus one: at
    least as many as the window shows, so the proxy predicts a miss.
    """
    ids = list(recent_admits)
    if task_id in ids:
        last = len(ids) - 1 - ids[::-1].index(task_id)
        return len(dict.fromkeys(ids[last + 1 :]))
    return len(dict.fromkeys(ids)) + 1


def predict_cached(
    prompt: int,
    last_prompt: int | None,
    last_end_mono_us: int | None,
    now_mono_us: int,
    others: int,
    params: Params,
) -> int:
    """Ĥ, the cached tokens predicted for the task's next call, within 0..P.

    A hit needs a previous prompt, a previous end within
    ``cache_window_s``, and at most ``h_c`` other sessions served since;
    a hit caches ``min(P, last_prompt)`` (decision D19), a miss only the
    shared prefix.
    """
    p = max(0, prompt)
    hit = (
        last_prompt is not None
        and last_end_mono_us is not None
        and 0 <= now_mono_us - last_end_mono_us < us(params.cache_window_s)
        and others <= params.h_c
    )
    if hit and last_prompt is not None:
        return min(p, max(0, last_prompt))
    return min(p, params.shared_prefix_tokens)


def delta_s(prompt: int, cached_hat: int, rates: Rates) -> float:
    """Δ = max(0, P − Ĥ)/r_p, the cold-prefill seconds (design 3.7)."""
    return max(0, prompt - cached_hat) / rates.prefill_tok_s


def chi_hat_s(
    prompt: int, cached_hat: int, completion_hat: float, rates: Rates
) -> float:
    """χ̂, the estimated charge of a waiting call."""
    p = max(0, prompt)
    h = min(max(0, cached_hat), p)
    return (p - h) / rates.prefill_tok_s + max(0.0, completion_hat) / rates.decode_tok_s


def fit_rates(
    samples: Sequence[tuple[int, int | None, int, float, float]], min_samples: int = 5
) -> Rates | None:
    """Rates from ``(P, H, D, prompt_eval_s, generation_s)`` of non-overlapping ok calls.

    ``None`` with fewer than *min_samples* samples, a zero or non-finite
    sum, or a rate that is not positive.
    """
    rows = sorted((p, h or 0, d, pe, ge) for p, h, d, pe, ge in samples)
    if len(rows) < min_samples:
        return None
    prompt_s = math.fsum(r[3] for r in rows)
    gen_s = math.fsum(r[4] for r in rows)
    if (
        not (math.isfinite(prompt_s) and math.isfinite(gen_s))
        or prompt_s <= 0
        or gen_s <= 0
    ):
        return None
    r_p = math.fsum(r[0] - r[1] for r in rows) / prompt_s
    r_d = math.fsum(r[2] for r in rows) / gen_s
    if not (math.isfinite(r_p) and math.isfinite(r_d)) or r_p <= 0 or r_d <= 0:
        return None
    return Rates(r_p, r_d)
