"""Parameters of the scheduling policy, validated, and the presets per arm.

Every value is Proposed in the design and every row is an arm (design
3.11), so the parameters are one frozen record that a simulation or the
live service passes to ``decide()``. Validation runs once at construction,
which lets the policy divide by rates and durations without checks. The
switches ``aging``, ``inheritance``, ``idle_gate`` and
``battery_gate_enforce`` let later phases turn mechanisms on live by
changing a value, not this code (spec decision D49).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from localmelo.support.scheduler.model import Arm

# The policy arms, in the order fifo ⊂ class ⊂ las2 (spec Requirement 44).
ARMS: tuple[Arm, ...] = ("fifo", "class", "las2")

# Preset per arm: floors, yield marks and resident-slot claims, focus boost.
_PRESETS: dict[str, dict[str, Any]] = {
    "fifo": {"floors": False, "yield_marks": False, "focus_boost_beta": 0.0},
    "class": {"floors": False, "yield_marks": True, "focus_boost_beta": 0.0},
    "las2": {"floors": True, "yield_marks": True, "focus_boost_beta": 1.0},
}

# Allowed values of the Literal fields.
_CHOICES: dict[str, tuple[str, ...]] = {
    "arm": ARMS,
    "battery_gate": ("size_and_class", "size_only", "class_only"),
    "reservation_policy": ("ask", "respect", "front_only"),
    "p0_meaning": ("task", "conversation"),
}

# Floats that may be +inf (memory not measured yet; decision D20).
_MAY_BE_INF = frozenset({"m_kv_bytes", "m_hot_bytes"})

# Floats that are not durations but end in _s by the design's name (κ_s).
_NOT_DURATIONS = frozenset({"kappa_s"})

# Integers and floats that must be at least 1 or 0, by field.
_AT_LEAST: dict[str, float] = {
    "n_srv": 1,
    "w_max": 1,
    "n_a": 1,
    "r_f": 0,
    "n_fg": 0,
    "h_c": 0,
    "p1_cap": 0,
    "shared_prefix_tokens": 0,
    "prompt_prior_tokens": 0,
    "completion_prior_tokens": 0,
    "completion_p99_tokens": 0,
    "credit_cap_factor": 1.0,
    "u_max": 1.0,
    "h_y": 1.0,
    "kv_bytes_per_token": 0.0,
    "kappa_s": 0.0,
    "focus_boost_beta": 0.0,
    "rho_las": 0.0,
    "m_kv_bytes": 0.0,
    "m_hot_bytes": 0.0,
    "m_front_bytes": 0.0,
    "idle_battery_pct": 0.0,
}


@dataclass(frozen=True)
class Params:
    """The policy's parameters; ``__post_init__`` raises ``ValueError`` naming a bad field."""

    arm: Arm = "las2"
    floors: bool = True
    yield_marks: bool = True
    aging: bool = True
    inheritance: bool = True
    idle_gate: bool = True
    battery_gate_enforce: bool = True
    battery_gate: Literal["size_and_class", "size_only", "class_only"] = (
        "size_and_class"
    )
    focus_boost_beta: float = 1.0
    tau_f_s: float = 600.0
    n_srv: int = 8
    r_f: int = 1
    n_a: int = 3
    n_fg: int = 1
    tau_focus_s: float = 30.0
    w_max: int = 4
    lane_deadline_s: float = 45.0
    tick_s: float = 15.0
    phi2: float = 0.2
    phi3: float = 0.1
    w_floor_s: float = 900.0
    credit_cap_factor: float = 2.0
    las_q_s: float = 60.0
    rho_las: float = 4.0
    s0_s: float = 20.0
    kappa_s: float = 1.5
    h_u_s: float = 14400.0
    u_max: float = 4.0
    implicit_deadline_s: float = 7200.0
    aging_s: float = 86400.0
    p4_expiry_s: float = 604800.0
    h_y: float = 1.5
    t_park_s: float = 1800.0
    p1_cap: int = 3
    prefill_tok_s: float = 829.0
    decode_tok_s: float = 66.0
    cache_window_s: float = 60.0
    h_c: int = 2
    shared_prefix_tokens: int = 0
    prompt_prior_tokens: int = 3235
    completion_prior_tokens: int = 400
    completion_p99_tokens: int = 2000
    remaining_prior_s: tuple[float, float, float] = (300.0, 1200.0, 3600.0)
    battery_large_s: float = 1200.0
    kv_bytes_per_token: float = 0.0
    m_kv_bytes: float = math.inf
    m_hot_bytes: float = math.inf
    m_front_bytes: float = 0.0
    idle_away_s: float = 600.0
    idle_quiet_s: float = 300.0
    idle_battery_pct: float = 80.0
    reservation_policy: Literal["ask", "respect", "front_only"] = "ask"
    lock_lines: bool = True
    cloud_offload: bool = False
    p0_meaning: Literal["task", "conversation"] = "task"

    def __post_init__(self) -> None:
        for f in fields(self):
            _check(f.name, getattr(self, f.name))
        if self.arm == "fifo" and (self.floors or self.yield_marks):
            name = "floors" if self.floors else "yield_marks"
            raise ValueError(f"{name}: not available under the fifo arm")
        if self.cloud_offload:
            raise ValueError("cloud_offload: not supported before a later phase")


def _check(name: str, v: Any) -> None:
    """Raise ``ValueError`` naming *name* when the value *v* is not allowed."""
    if name in _CHOICES:
        if v not in _CHOICES[name]:
            raise ValueError(f"{name}: {v!r} is not one of {_CHOICES[name]}")
        return
    if isinstance(v, bool):
        return
    if name == "remaining_prior_s":
        if len(v) != 3 or not all(math.isfinite(x) and x > 0 for x in v):
            raise ValueError(f"{name}: needs three positive finite seconds")
        return
    if isinstance(v, float) and (
        math.isnan(v) or (math.isinf(v) and name not in _MAY_BE_INF)
    ):
        raise ValueError(f"{name}: must be finite")
    if name.endswith(("_s", "_tok_s")) and name not in _NOT_DURATIONS and not v > 0:
        raise ValueError(f"{name}: must be positive")
    if name in _AT_LEAST and not v >= _AT_LEAST[name]:
        raise ValueError(f"{name}: must be at least {_AT_LEAST[name]}")
    if name in ("phi2", "phi3") and not 0.0 <= v <= 1.0:
        raise ValueError(f"{name}: must be within [0, 1]")


def params_for_arm(arm: str, **overrides: Any) -> Params:
    """The preset of *arm* (spec Requirement 44) with *overrides*, validated."""
    if arm not in _PRESETS:
        raise ValueError(f"arm: {arm!r} is not one of {ARMS}")
    return replace(Params(), **{"arm": arm, **_PRESETS[arm], **overrides})
