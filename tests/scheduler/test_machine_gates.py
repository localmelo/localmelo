"""The machine gates follow the design's gate table over a snapshot.

Requirements 16 and 25: each row allows the levels of design 3.8 and gives
its reason; the battery band keeps hysteresis in the state; the start
rule follows the ``battery_gate`` arm; a self lock passes; the idle window
needs every one of its conditions.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.support.scheduler import Machine, Params, evaluate_gates
from localmelo.support.scheduler.machine_gates import battery_band
from localmelo.tests.scheduler.helpers import clock

NOW = clock(10_000.0)


def _blocks(m: Machine, starting: bool = False, p: Params | None = None) -> list[Any]:
    g = evaluate_gates(m, 4, None, NOW, p or Params(idle_gate=False))
    return [g.block(level, starting, False) for level in range(5)]


@pytest.mark.parametrize(
    ("m", "starting", "expected"),
    [
        (Machine(lock="foreign"), False, [("Reservation", "lock")] * 5),
        (Machine(server_up=False), False, [("Resources", "server_down")] * 5),
        (Machine(model="loading"), False, [("Resources", "model_not_loaded")] * 5),
        (Machine(model="draining"), False, [("Resources", "model_not_loaded")] * 5),
        (
            Machine(low_power=True),
            False,
            [None, None, None] + [("Battery", "low_power")] * 2,
        ),
        (Machine(thermal="serious"), False, [None, None] + [("Thermal", "")] * 3),
        (Machine(thermal="critical"), False, [("Thermal", "")] * 5),
        (Machine(thermal="fair"), False, [None] * 5),
        (
            Machine(memory_pressure="warning"),
            False,
            [None] * 3 + [("Memory", "pressure")] * 2,
        ),
        (
            Machine(memory_pressure="critical"),
            False,
            [None] * 2 + [("Memory", "pressure")] * 3,
        ),
        (Machine(memory_pressure="warning"), True, [("Memory", "no_new_worker")] * 5),
        (Machine(memory_pressure="critical"), True, [("Memory", "no_new_worker")] * 5),
        (
            Machine(on_ac=False, battery_pct=55.0),
            False,
            [None] * 3 + [("Battery", "")] * 2,
        ),
        (
            Machine(on_ac=False, battery_pct=45.0),
            False,
            [None] * 2 + [("Battery", "")] * 3,
        ),
        (Machine(on_ac=False, battery_pct=10.0), False, [None] + [("Battery", "")] * 4),
    ],
)
def test_each_gate_row_allows_the_levels_of_the_design_table(
    m: Machine, starting: bool, expected: list[Any]
) -> None:
    got = _blocks(m, starting)
    if starting and m.memory_pressure == "critical":
        assert got[1] == ("Memory", "no_new_worker")  # a starting P1 too
    if m.battery_pct is not None and m.battery_pct < 20:
        g = evaluate_gates(m, 4, None, NOW, Params())
        assert g.battery_band == 0
    assert got == expected
    idle_gate = evaluate_gates(Machine(), 4, None, NOW, Params())
    assert idle_gate.block(4, False, False) == ("IdleOnly", "")
    assert idle_gate.block(3, False, False) is None


def test_the_battery_band_releases_only_at_80_or_ac_and_from_below_20_at_50() -> None:
    def walk(start: int, pcts: list[float]) -> list[int]:
        out, band = [], start
        for pct in pcts:
            band = battery_band(Machine(on_ac=False, battery_pct=pct), band)
            out.append(band)
        return out

    assert walk(4, [59.0, 70.0, 80.0]) == [2, 2, 4]
    assert walk(4, [19.0, 45.0, 50.0]) == [0, 0, 1]
    assert walk(1, [79.9, 80.0]) == [1, 4]
    assert battery_band(Machine(on_ac=True, battery_pct=5.0), 0) == 4
    assert battery_band(Machine(on_ac=False, battery_pct=None), 0) == 4


@pytest.mark.parametrize("rule", ["size_and_class", "size_only", "class_only"])
def test_the_battery_start_rule_blocks_large_tasks_below_60_and_every_start_below_50(
    rule: str,
) -> None:
    p = Params(battery_gate=rule, idle_gate=False)  # type: ignore[arg-type]
    sizes = rule != "class_only"
    classes = rule != "size_only"
    at_55 = evaluate_gates(Machine(on_ac=False, battery_pct=55.0), 4, None, NOW, p)
    assert at_55.battery_band == 2
    assert at_55.block(1, True, True) == (("Battery", "size") if sizes else None)
    assert at_55.block(1, True, False) is None
    assert at_55.block(1, False, True) is None  # a running task is not starting
    assert at_55.block(3, False, False) == (("Battery", "") if classes else None)
    at_45 = evaluate_gates(Machine(on_ac=False, battery_pct=45.0), 4, None, NOW, p)
    assert at_45.block(0, True, False) == (("Battery", "start") if sizes else None)
    off = Params(battery_gate=rule, battery_gate_enforce=False, idle_gate=False)  # type: ignore[arg-type]
    g = evaluate_gates(Machine(on_ac=False, battery_pct=45.0), 4, None, NOW, off)
    assert g.battery_band == 1
    assert all(g.block(level, True, True) is None for level in range(5))


def test_a_self_lock_is_not_foreign() -> None:
    assert _blocks(Machine(lock="self")) == [None] * 5
    assert _blocks(Machine(lock="none")) == [None] * 5
    assert _blocks(Machine(lock="foreign"))[0] == ("Reservation", "lock")


def test_the_idle_window_needs_every_condition() -> None:
    p = Params()
    away = NOW.mono_us - 600_000_000
    base: dict[str, Any] = {"away_since_mono_us": away}
    quiet = NOW.mono_us - 300_000_000
    assert evaluate_gates(Machine(**base), 4, quiet, NOW, p).idle
    assert (
        evaluate_gates(Machine(**base), 4, None, NOW, p).block(4, False, False) is None
    )
    variants: list[tuple[dict[str, Any], int | None]] = [
        ({"away_since_mono_us": None}, quiet),
        ({"away_since_mono_us": away + 1}, quiet),
        ({"on_ac": False, "battery_pct": 79.0}, quiet),
        ({"on_ac": False, "battery_pct": None}, quiet),
        ({"lock": "self"}, quiet),
        ({"thermal": "fair"}, quiet),
        ({}, quiet + 1),
    ]
    for change, busy in variants:
        g = evaluate_gates(Machine(**{**base, **change}), 4, busy, NOW, p)
        assert not g.idle, change
    assert evaluate_gates(
        Machine(**base, on_ac=False, battery_pct=80.0), 4, None, NOW, p
    ).idle
