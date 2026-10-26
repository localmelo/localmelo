"""The lane is admitted, or answered by code, whatever decide() does (acceptance a).

Requirements 2, 14, 21, 22, 29, 38 and 60: a lane call never waits behind
a task call; degraded modes answer by code; ``reservation_policy`` decides
under a foreign lock; at most one lane call is in flight; the fallback
keeps the gates and the reserve; failures of the gate code, the lane check
and the fallback each leave the lane open; a focused chat moves into the
lane.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.support.scheduler import (
    CallEnded,
    CallSubmitted,
    Clock,
    Decision,
    Machine,
    Params,
    State,
    dispatch,
    params_for_arm,
    policy,
)
from localmelo.support.scheduler.model import us
from localmelo.tests.scheduler.helpers import (
    call,
    clock,
    find,
    lane,
    reasons,
    state,
    step,
    task,
)

NOW = clock(100.0)


def boom(st: State, now: Clock, p: Params) -> Decision:
    raise ZeroDivisionError


def raising(*args: Any, **kw: Any) -> Any:
    raise KeyError("x")


def test_a_lane_call_is_admitted_when_decide_raises() -> None:
    st = state([task("t")], [call("c", "t"), lane()])
    out = dispatch(st, NOW, Params(), decide_fn=boom)
    assert out.lane_admit == ("lane",)
    assert (
        out.decision.fallback
        and "decide_failed:ZeroDivisionError" in out.decision.alarms
    )
    assert out.decision.admit == ("c",)


def test_a_lane_call_is_admitted_before_any_task_call() -> None:
    running = task("r", phase="running")
    p0 = task("p0", level=0, level_source="user")
    st = state(
        [running, p0], [call("busy", "r", admitted_mono_us=0), call("c0", "p0"), lane()]
    )
    out = dispatch(st, NOW, Params(n_srv=2))
    assert out.lane_admit == ("lane",) and out.decision.admit == ()
    assert reasons(out.decision)["p0"] == ("Resources", "")


@pytest.mark.parametrize(
    ("m", "kw", "cause"),
    [
        (Machine(lock="foreign"), {}, "reservation"),
        (Machine(server_up=False), {}, "server_down"),
        (Machine(model="loading"), {}, "model_not_loaded"),
        (Machine(), {"kv_bytes_per_token": 1.0, "m_kv_bytes": 100.0}, "memory"),
    ],
)
def test_a_lane_call_is_answered_by_code_in_each_degraded_mode(
    m: Machine, kw: dict[str, Any], cause: str
) -> None:
    out = dispatch(state(calls=[lane()], machine=m), NOW, Params(**kw))
    assert out.lane_admit == () and out.lane_answer == (("lane", cause),)
    hot = Machine(thermal="critical", low_power=True, memory_pressure="critical")
    critical = dispatch(
        state(calls=[lane()], machine=hot), NOW, Params(battery_gate_enforce=True)
    )
    assert critical.lane_admit == ("lane",)


def test_respect_waits_and_front_only_admits_under_a_foreign_lock() -> None:
    st = state([task("t")], [lane(), call("c", "t")], machine=Machine(lock="foreign"))
    respect = dispatch(st, NOW, Params(reservation_policy="respect"))
    assert respect.lane_admit == () and respect.lane_answer == ()
    front_only = dispatch(st, NOW, Params(reservation_policy="front_only"))
    assert front_only.lane_admit == ("lane",)
    assert respect.decision.admit == () and front_only.decision.admit == ()


def test_the_lane_never_has_two_calls_in_flight() -> None:
    st = state(calls=[lane("l1", admitted_mono_us=0), lane("l2", submit_mono_us=1)])
    assert dispatch(st, NOW, Params()).lane_admit == ()
    assert policy._lane_by_count(st, Params()) == ()
    two = state(calls=[lane("l2", submit_mono_us=5), lane("l1", submit_mono_us=9)])
    assert dispatch(two, NOW, Params()).lane_admit == ("l2",)


def test_with_one_slot_the_lane_waits_for_at_most_the_call_in_flight() -> None:
    p = Params(n_srv=1)
    st = state(
        [task("t", phase="running")], [call("c", "t", admitted_mono_us=0), lane()]
    )
    st, out = step(st, NOW, p)
    assert out.lane_admit == () and out.lane_answer == ()
    st, out = step(st, clock(101.0), p, [CallEnded("c", "ok", us(1.0))])
    assert out.lane_admit == ("lane",)


def test_the_fallback_keeps_the_gates_and_the_reserve() -> None:
    tasks = [task(f"u{i}", level=1, level_source="user") for i in range(3)]
    calls = [call(f"c{i}", f"u{i}", submit_mono_us=i) for i in range(3)]
    locked = state(tasks, calls, machine=Machine(lock="foreign"))
    d = dispatch(locked, NOW, Params(n_srv=3), decide_fn=boom).decision
    assert d.admit == () and set(reasons(d).values()) == {("Reservation", "lock")}
    free = dispatch(state(tasks, calls), NOW, Params(n_srv=3), decide_fn=boom).decision
    assert free.admit == ("c0",) and free.fallback
    assert reasons(free) == {"u1": ("Resources", ""), "u2": ("Priority", "")}


def test_a_failing_gate_computation_closes_task_admission_but_not_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "evaluate_gates", raising)
    tasks = [task("t"), task("r", phase="running")]
    st = state(tasks, [call("c", "t"), call("e", "r", kind="embed"), lane()])
    out = dispatch(st, NOW, Params())
    assert out.lane_admit == ("lane",)
    d = out.decision
    assert d.admit == () and d.admit_unheld == ("e",) and d.fallback
    assert reasons(d) == {"t": ("Resources", "fallback")}
    assert {"gates_failed", "decide_failed:KeyError"} <= set(d.alarms)


def test_a_failing_lane_check_still_admits_the_lane_by_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "admit_lane", raising)
    st = state([task("t")], [call("c", "t"), lane()], machine=Machine(lock="foreign"))
    out = dispatch(st, NOW, Params())
    assert out.lane_admit == ("lane",)
    assert "lane_check_failed" in out.decision.alarms
    assert out.decision.admit == ()  # task calls still hold for the lock


def test_a_failing_fallback_admits_only_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "fallback_fcfs", raising)
    tasks = [task("t"), task("r", phase="running"), task("bad", level=-1)]
    st = state(tasks, [call("c", "t"), call("e", "r", kind="embed"), lane()])
    out = dispatch(st, NOW, Params(n_srv=1), decide_fn=boom)
    d = out.decision
    assert out.lane_admit == ("lane",)
    assert (d.admit, d.admit_unheld, d.floor, d.yields, d.parks, d.expire) == ((),) * 6
    assert reasons(d) == {
        "bad": ("Resources", "fallback"),
        "t": ("Resources", "fallback"),
    }
    assert dict(d.levels) == {"bad": 2, "r": 2, "t": 2}
    assert d.alarms == ("decide_failed:ZeroDivisionError", "fallback_failed:KeyError")
    assert d.limit_nonlane == 1 and d.fallback


def test_a_focus_switch_moves_a_held_chat_call_into_the_lane() -> None:
    p = params_for_arm("las2", w_max=1, n_fg=0)
    chat = task("chat", kind="chat", phase="held", started_mono_us=0)
    q = task("q", level=1, level_source="user")
    calls = [call("c1", "chat", session_kind="chat"), call("cq", "q")]
    focused: dict[str, Any] = {
        "last_typing_mono_us": NOW.mono_us,
        "floor_open": (True, True),
    }
    st = state([chat, q], calls, clock=NOW, **focused)
    st, out = step(st, NOW, p)
    assert out.decision.admit == ()  # the chat is held by the focus cap, q by w_max
    assert reasons(out.decision) == {"chat": ("Priority", ""), "q": ("Resources", "")}
    moved = call("c2", "chat", session_kind="front", submit_mono_us=us(101.0))
    switch: list[Any] = [CallEnded("c1", "aborted", 0), CallSubmitted(moved)]
    st, out = step(st, clock(101.0), p, switch)
    assert out.lane_admit == ("c2",)
    assert "chat" not in reasons(out.decision)
    assert out.decision.admit == ("cq",)  # the chat left running_tasks
    st, _ = step(st, clock(110.0), p, [CallEnded("c2", "ok", us(5.0))])
    assert find(st, "chat").service_us == us(5.0) and find(st, "chat").calls_done == 1
    assert st.credit_us == (0, 0)
