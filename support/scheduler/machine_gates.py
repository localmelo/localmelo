"""The machine gates: which task levels may run on a machine snapshot.

Step 1 of ``decide()`` (design 3.3; the gate table of design 3.8). The
caller's monitors debounce the thermal state, memory pressure and Low
Power Mode; this module keeps only the battery band's hysteresis, in the
state, so the battery gate does not flap. It is public so that S2b can
record what the gates would do on a snapshot before it enforces them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from localmelo.support.scheduler.model import Clock, Machine, Reason, us
from localmelo.support.scheduler.params import Params

# Battery percentages that lower the band on entry (design 3.8): 4 (all
# levels) at 60% or more, 2 from 50%, 1 from 20%, 0 below.
_BAND_ENTRY = ((60.0, 4), (50.0, 2), (20.0, 1))

# A band of 1 or 2 is released to 4 at 80%; a band of 0 rises to 1 at 50%.
_RELEASE_PCT = 80.0
_RELEASE_FROM_ZERO_PCT = 50.0

# A gate's answer for one task: the reason and its detail, or None.
Block = Callable[[int, bool, bool], tuple[Reason, str] | None]


@dataclass(frozen=True)
class Gates:
    """The new battery band, whether the idle window holds, and the gate check.

    ``block(level, starting, large)`` gives the first gate that stops a
    task of *level* (its ``L_eff``), or ``None``.
    """

    battery_band: int
    idle: bool
    block: Block


def battery_band(m: Machine, previous: int) -> int:
    """The highest level the battery allows, with hysteresis over *previous*.

    On AC or with an unknown battery, 4.
    """
    pct = m.battery_pct
    if m.on_ac or pct is None:
        return 4
    entered = next((b for lo, b in _BAND_ENTRY if pct >= lo), 0)
    band = min(previous, entered)
    if band == 0 and pct >= _RELEASE_FROM_ZERO_PCT:
        band = 1
    if band in (1, 2) and pct >= _RELEASE_PCT:
        band = 4
    return band


def idle_window(
    m: Machine, busy_above_p4_mono_us: int | None, now: Clock, p: Params
) -> bool:
    """Whether P4 may run: the user away long enough and the machine quiet."""
    if m.away_since_mono_us is None:
        return False
    away = now.mono_us - m.away_since_mono_us >= us(p.idle_away_s)
    power = m.on_ac or (
        m.battery_pct is not None and m.battery_pct >= p.idle_battery_pct
    )
    quiet = busy_above_p4_mono_us is None or now.mono_us - busy_above_p4_mono_us >= us(
        p.idle_quiet_s
    )
    return away and power and m.lock == "none" and m.thermal == "nominal" and quiet


def hard_gate(m: Machine) -> tuple[Reason, str] | None:
    """Checks 1 and 2 of the table, which hold every task call, retries included.

    A foreign lock (no task model call, design 3.8), then a down server,
    then a model that is not loaded.
    """
    if m.lock == "foreign":
        return ("Reservation", "lock")
    if not m.server_up:
        return ("Resources", "server_down")
    if m.model != "loaded":
        return ("Resources", "model_not_loaded")
    return None


def evaluate_gates(
    m: Machine,
    previous_band: int,
    busy_above_p4_mono_us: int | None,
    now: Clock,
    p: Params,
) -> Gates:
    """The gate table over the snapshot *m* (spec Requirement 25).

    Checks run in the table's order and the first match is the reason:
    lock, server, battery band (only with ``battery_gate_enforce``), Low
    Power Mode, thermal state, memory pressure, then the idle gate for P4.
    """
    band = battery_band(m, previous_band)
    idle = idle_window(m, busy_above_p4_mono_us, now, p)
    by_class = p.battery_gate in ("class_only", "size_and_class")
    by_size = p.battery_gate in ("size_only", "size_and_class")

    hard = hard_gate(m)

    def block(level: int, starting: bool, large: bool) -> tuple[Reason, str] | None:
        if hard is not None:
            return hard
        if p.battery_gate_enforce:
            if by_class and level > band:
                return ("Battery", "")
            if by_size and starting and band <= 2 and large:
                return ("Battery", "size")
            if by_size and starting and band <= 1:
                return ("Battery", "start")
        if m.low_power and level > 2:
            return ("Battery", "low_power")
        if (m.thermal == "serious" and level > 1) or m.thermal == "critical":
            return ("Thermal", "")
        if m.memory_pressure in ("warning", "critical"):
            if starting:
                return ("Memory", "no_new_worker")
            if level >= (3 if m.memory_pressure == "warning" else 2):
                return ("Memory", "pressure")
        if level == 4 and not idle and p.idle_gate:
            return ("IdleOnly", "")
        return None

    return Gates(band, idle, block)
