"""Credit floors bound the wait of P2 and P3 without idling a slot (acceptance b).

Requirements 2, 32, 39 and 40: credit accrues only while the class is
admissible and has waited ``w_floor_s``; it is capped at twice χ̂, resets
when the class has nothing runnable and never goes negative; lane and P0
calls add none; a floor admission goes ahead of P1 but not of the lane,
P0 or an urgent P1; one floor call per class is in flight; floors only
reorder; a full bucket after a crunch pays for about two calls; in a
simulated crunch every P3 call starts within the floor bound under las2,
and never under class.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from localmelo.support.scheduler import (
    Call,
    CallEnded,
    CallSubmitted,
    Decision,
    Machine,
    State,
    Task,
    advance,
    apply_dispatch,
    decide,
    dispatch,
    params_for_arm,
)
from localmelo.support.scheduler.costs import RATES_OMLX_OQ5, alone_service_s
from localmelo.support.scheduler.model import us
from localmelo.support.scheduler.sim import ServerParams, Workload, engine, run
from localmelo.support.scheduler.sim.trace import TaskTemplate
from localmelo.support.scheduler.sim.workloads import WALL_START_US, synthetic_templates
from localmelo.tests.scheduler.helpers import (
    call,
    clock,
    lane,
    state,
    step,
    task,
    wtask,
)

NOW = clock(10_000.0)
P = params_for_arm("las2", w_max=10)
WAITED = us(900.0)


def base(**kw: Any) -> tuple[list[Task], list[Call]]:
    b = task("b", phase="running", started_mono_us=0)
    w = task("w", level=3, wait_held_us=kw.pop("wait", WAITED))
    return [b, w], [
        call("c-b", "b", admitted_mono_us=0, level_at_admit=2),
        call("c-w", "w"),
    ]


def decide_on(
    tasks: Sequence[Task], calls: Sequence[Call], m: Machine | None = None, **kw: Any
) -> Decision:
    st = state(tasks, calls, machine=m or Machine(), clock=NOW, **kw.pop("st", {}))
    return decide(st, NOW, params_for_arm("las2", w_max=10, **kw))


def test_credit_accrues_only_while_the_class_is_admissible_and_has_waited_w_floor() -> (
    None
):
    tasks, calls = base()
    soon = NOW.wall_us + us(600.0)
    urgent = task("u", level=1, level_source="user", deadline_wall_us=soon)
    assert decide_on(tasks, calls).floor_open == (False, True)
    assert (
        decide_on(tasks, calls, Machine(on_ac=False, battery_pct=45.0)).floor_open[1]
        is False
    )
    assert (
        decide_on(tasks, calls, kv_bytes_per_token=1.0, m_kv_bytes=6000.0).floor_open[1]
        is False
    )
    assert (
        decide_on([*tasks, urgent], [*calls, call("c-u", "u")]).floor_open[1] is False
    )
    assert decide_on(*base(wait=WAITED - 1)).floor_open[1] is False
    running = task(
        "u", level=1, level_source="user", deadline_wall_us=soon, phase="running"
    )
    in_flight = call("c-u", "u", admitted_mono_us=0)
    assert decide_on([*tasks, running], [*calls, in_flight]).floor_open[1] is True

    def credit_after(m: Machine) -> int:
        st, _ = step(state(tasks, calls, machine=m, clock=NOW), NOW, P)
        st = advance(st, NOW, [CallEnded("c-b", "ok", us(10.0))], P)
        return st.credit_us[1]

    assert credit_after(Machine()) == us(1.0)  # phi3 · 10 s
    assert credit_after(Machine(on_ac=False, battery_pct=45.0)) == 0


def test_credit_is_capped_at_twice_chi_hat_and_resets_when_nothing_is_runnable() -> (
    None
):
    tasks, calls = base()
    st = state(
        tasks, calls, clock=NOW, floor_open=(True, True), chi_hat_us=(0, us(10.0))
    )
    st = advance(st, NOW, [CallEnded("c-b", "ok", us(3600.0))], P)
    assert st.credit_us == (0, us(20.0))
    d = dispatch(st, NOW, P)
    st = apply_dispatch(st, NOW, d, P)
    assert st.credit_us[1] == math.floor(2.0 * d.decision.chi_hat_us[1])
    nothing = dispatch(state([task("w", level=3)], credit_us=(9, 9), clock=NOW), NOW, P)
    assert nothing.decision.class_runnable == (False, False)
    after = apply_dispatch(state(credit_us=(9, 9)), NOW, nothing, P)
    assert after.credit_us == (0, 0)


def test_lane_and_p0_calls_add_no_credit() -> None:
    calls = [
        lane("l", admitted_mono_us=0),
        call("p0", "z", admitted_mono_us=0, level_at_admit=0),
        call("p1", "z", admitted_mono_us=0, level_at_admit=1),
    ]
    st = state(
        [task("z")], calls, floor_open=(True, True), chi_hat_us=(us(50.0), us(50.0))
    )
    ended = advance(
        st, NOW, [CallEnded("l", "ok", us(10.0)), CallEnded("p0", "ok", us(10.0))], P
    )
    assert ended.credit_us == (0, 0)
    p1 = advance(st, NOW, [CallEnded("p1", "ok", us(10.0))], P)
    assert p1.credit_us == (us(2.0), us(1.0))


def floor_state(*extra: tuple[Task, Call], credit: int = us(100.0)) -> State:
    w = task("w", level=3, wait_held_us=WAITED)
    p1 = task("p1", level=1, level_source="user")
    tasks = [w, p1, *(t for t, _ in extra)]
    calls = [call("c-w", "w"), call("c-p1", "p1"), *(c for _, c in extra)]
    return state(tasks, calls, credit_us=(0, credit), clock=NOW)


def test_a_floor_admission_goes_ahead_of_p1_but_never_ahead_of_the_lane_p0_or_an_urgent_p1() -> (
    None
):
    p = params_for_arm("las2", n_srv=3, w_max=10)  # limit 1
    d = decide(floor_state(), NOW, p)
    assert d.admit == ("c-w",) and d.floor == (("c-w", 3),)
    p0 = (task("z", level=0, level_source="user"), call("c-z", "z"))
    assert decide(floor_state(p0), NOW, p).admit == ("c-z",)
    soon = NOW.wall_us + us(600.0)
    urgent = (
        task("u", level=1, level_source="user", deadline_wall_us=soon),
        call("c-u", "u"),
    )
    d = decide(floor_state(urgent), NOW, p)
    assert d.admit == ("c-u",) and d.floor == ()
    st = floor_state()
    with_lane = state(st.tasks, [*st.calls, lane()], credit_us=st.credit_us)
    out = dispatch(with_lane, NOW, params_for_arm("las2", n_srv=1, w_max=10))
    assert out.lane_admit == ("lane",) and out.decision.admit == ("c-w",)


def test_only_one_floor_call_per_class_is_in_flight() -> None:
    p = params_for_arm("las2", n_srv=4, w_max=10)  # limit 2
    first = (
        task("w0", level=3, phase="running"),
        call("c-w0", "w0", admitted_mono_us=0, floor_class=3),
    )
    d = decide(floor_state(first), NOW, p)
    assert d.floor == () and d.admit == ("c-p1",)


def test_credit_never_goes_negative() -> None:
    st = state(
        [task("w", level=3)], [call("f", "w", admitted_mono_us=0, floor_class=3)]
    )
    st = state(st.tasks, st.calls, credit_us=(0, us(5.0)), chi_hat_us=(0, us(10.0)))
    assert advance(st, NOW, [CallEnded("f", "ok", us(9.0))], P).credit_us == (0, 0)


def test_floors_only_reorder_and_never_idle_a_slot() -> None:
    for n_srv in (3, 4, 8):
        on = decide(floor_state(), NOW, params_for_arm("las2", n_srv=n_srv, w_max=10))
        off = decide(
            floor_state(),
            NOW,
            params_for_arm("las2", n_srv=n_srv, w_max=10, floors=False),
        )
        assert len(on.admit) == len(off.admit) and set(on.admit) >= set(off.admit) - {
            "c-p1"
        }
    # A floor call still passes cap_bg; the slot then goes to the P1, not idle.
    bg = (task("b", phase="running"), call("c-b", "b", admitted_mono_us=0))
    d = decide(floor_state(bg), NOW, params_for_arm("las2", n_srv=8, n_a=1, w_max=10))
    assert d.admit == ("c-p1",) and d.floor == ()


def _crunch(charge_of: Any) -> tuple[list[str], int]:
    """Floor admissions of P3 while P2 keeps cap_bg full and adds no credit."""
    p = params_for_arm("las2", w_max=10)
    tasks = [task(f"b{i}", phase="running" if i < 3 else "queued") for i in range(5)]
    tasks += [task(w, level=3, wait_held_us=WAITED) for w in ("w1", "w2")]
    calls = [
        call(f"b{i}-0", f"b{i}", admitted_mono_us=0, level_at_admit=2) for i in range(3)
    ]
    calls += [call(f"{t.task_id}-0", t.task_id) for t in tasks[3:]]
    chi = decide(state(tasks, calls, clock=clock(0.0)), clock(0.0), p).chi_hat_us
    st = state(
        tasks, calls, clock=clock(0.0), credit_us=(0, 2 * chi[1]), chi_hat_us=chi
    )
    admitted: list[str] = []
    for i in range(1, 30):
        events: list[Any] = []
        flying = [c for c in st.calls if c.admitted_mono_us is not None]
        ends = [c for c in flying if c.task_id and c.task_id.startswith("w")]
        ends += [c for c in flying if c.task_id and c.task_id.startswith("b")][:1]
        for c in ends:
            charge = 0 if c.task_id.startswith("b") else charge_of(chi[1])  # type: ignore[union-attr]
            events += [CallEnded(c.call_id, "ok", charge, None, 400)]
            events += [
                CallSubmitted(call(f"{c.task_id}-{i}", c.task_id, submit_mono_us=us(i)))
            ]
        st, out = step(st, clock(float(i)), p, events)
        assert out.decision.chi_hat_us[1] == chi[1]
        admitted += [cid for cid, cls in out.decision.floor if cls == 3]
        assert sum(1 for c in st.calls if c.floor_class == 3) <= 1  # one at a time
    return admitted, chi[1]


def test_no_credit_burst_after_a_crunch() -> None:
    assert len(_crunch(lambda chi: chi)[0]) == 2
    for parts in (2, 3, 5):
        admitted, chi = _crunch(lambda chi, n=parts: chi // n)
        assert 2 < len(admitted) <= chi // (chi // parts) + 1  # ⌊χ̂ / χ_min⌋ + 1


def _gapless_chain(start: int, min_s: float) -> TaskTemplate:
    """Synthetic templates joined until *min_s* of alone service, with no tool time."""
    tpl = synthetic_templates(1)
    calls: list[Any] = []
    i = start
    while (
        sum(
            alone_service_s(c.prompt_tokens, 0, c.completion_tokens, RATES_OMLX_OQ5)
            for c in calls
        )
        < min_s
    ):
        calls += [replace(c, gap_before_s=0.0) for c in tpl[i % len(tpl)].calls]
        i += 1
    return TaskTemplate(f"chain-{start}", tuple(calls))


def test_the_simulated_floor_bound_holds_in_a_crunch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p2 = [_gapless_chain(4 * i, 7200.0) for i in range(4)]
    tasks = [wtask(f"p2-{i}", 0.0, 2, t) for i, t in enumerate(p2)]
    tasks.append(wtask("p3", 0.0, 3, _gapless_chain(20, 7200.0)))
    wl = Workload("crunch", 1, us(5400.0), WALL_START_US, tuple(tasks))
    over = {"n_a": 3, "w_max": 5}
    plain = run(wl, "class", params_for_arm("class", **over))
    assert next(t for t in plain.tasks if t.task_id == "p3").first_admit_us is None

    # Every P3 call: (submit, admit or the horizon, χ̂_3 of the last decision).
    p3_calls: dict[str, tuple[int, int, int]] = {}
    real = engine.dispatch

    def spy(st: State, now: Any, p: Any, **kw: Any) -> Any:
        d = real(st, now, p, **kw)
        chi = d.decision.chi_hat_us[1]
        for c in st.calls:
            pending = c.admitted_mono_us is None and c.call_id not in d.decision.admit
            if c.task_id == "p3" and c.admitted_mono_us is None:
                end = wl.horizon_us if pending else now.mono_us
                p3_calls[c.call_id] = (c.submit_mono_us, end, chi)
        return d

    monkeypatch.setattr(engine, "dispatch", spy)
    p = params_for_arm("las2", **over)
    run(wl, "las2", p)
    theta3 = 3 ** ServerParams().theta_alpha
    longest = max(
        alone_service_s(c.prompt_tokens, 0, c.completion_tokens, RATES_OMLX_OQ5)
        for t in p2
        for c in t.calls
    )
    wall = us(longest * 3 / theta3)  # the longest P2 call at K = 3, cold
    admitted = [x for x in p3_calls.values() if x[1] < wl.horizon_us]
    assert len(admitted) >= 2 and len(p3_calls) == len(admitted) + 1
    for submit, admit, chi in p3_calls.values():  # a call still waiting included
        bound = us(p.w_floor_s) + us(chi / 1e6 / (p.phi3 * theta3)) + 2 * wall
        assert admit - submit <= bound + us(p.tick_s)
