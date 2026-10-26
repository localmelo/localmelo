"""decide() fills in strict class order behind the reserve, the caps and the gates.

Requirements 10, 12, 14, 27, 28, 30, 31, 33-37, 44, 45, 59 and 60: the
reserve and the focus cap, never-held calls and retries, the memory check,
``w_max`` and resident-slot claims, the order of each arm, the idle gate,
the foreign lock, one reason per task by precedence, a work-conserving
fill, yield and park marks, expiry and parameter validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from localmelo.support.scheduler import (
    ARMS,
    Call,
    Decision,
    Dep,
    Machine,
    Params,
    Task,
    decide,
    dispatch,
    params_for_arm,
)
from localmelo.support.scheduler.model import us
from localmelo.support.scheduler.policy import _order_keys
from localmelo.tests.scheduler.helpers import (
    WALL0,
    call,
    clock,
    find,
    lane,
    reasons,
    state,
    step,
    task,
)

NOW = clock(10_000.0)
LONG_AGO = NOW.mono_us - us(5000.0)


def run(
    tasks: Sequence[Task],
    calls: Sequence[Call] = (),
    deps: Sequence[Dep] = (),
    m: Machine | None = None,
    arm: str = "class",
    **kw: Any,
) -> Decision:
    st = state(tasks, calls, deps, machine=m or Machine(), clock=NOW)
    return decide(st, NOW, params_for_arm(arm, **kw))


def order(
    tasks: Sequence[Task], calls: Sequence[Call], arm: str, m: Machine | None = None
) -> list[str]:
    st = state(tasks, calls, machine=m or Machine(), clock=NOW)
    return [tid for tid, _ in _order_keys(st, NOW, params_for_arm(arm))]


def queued(
    tid: str, level: int = 2, submit: float = 0.0, **kw: Any
) -> tuple[Task, Call]:
    source = "user" if level <= 1 else "computed"
    t = task(tid, level=level, level_source=kw.pop("level_source", source), **kw)
    return t, call(f"c-{tid}", tid, submit_mono_us=us(submit))


def busy(tid: str, level: int = 2, **kw: Any) -> tuple[Task, Call]:
    """A running task with a call in flight."""
    kw.setdefault("started_mono_us", 0)
    t = task(tid, level=level, phase="running", **kw)
    return t, call(f"c-{tid}", tid, admitted_mono_us=0)


def split(*pairs: tuple[Task, Call]) -> tuple[list[Task], list[Call]]:
    return [t for t, _ in pairs], [c for _, c in pairs]


@pytest.mark.parametrize("arm", ["class", "las2"])
def test_strict_class_order_admits_p0_then_p1_then_p2(arm: str) -> None:
    tasks, calls = split(queued("p2", 2, 0), queued("p1", 1, 1), queued("p0", 0, 2))
    d = run(tasks, calls, arm=arm, n_srv=4, w_max=10)
    assert d.limit_nonlane == 2
    assert d.admit == ("c-p0", "c-p1")
    assert reasons(d) == {"p2": ("Resources", "")}


@pytest.mark.parametrize(
    ("n_srv", "limit", "clamped"),
    [(8, 6, False), (3, 1, False), (2, 1, False), (1, 1, True)],
)
def test_the_reserve_keeps_one_slot_for_the_lane(
    n_srv: int, limit: int, clamped: bool
) -> None:
    tasks, calls = split(*(queued(f"t{i}", 1, i) for i in range(8)))
    d = run(tasks, calls, n_srv=n_srv, w_max=10)
    assert d.limit_nonlane == limit and len(d.admit) == limit
    assert ("reserve_clamped" in d.alarms) == clamped


def test_the_focus_cap_limits_p2_to_p4_but_not_p0_or_p1() -> None:
    pairs = [queued(f"b{i}", 2, i) for i in range(3)] + [
        queued(f"u{i}", 1, 9) for i in range(2)
    ]
    tasks, calls = split(*pairs)

    def admitted(**kw: Any) -> int:
        st = state(tasks, calls, clock=NOW, **kw.pop("st", {}))
        return len(decide(st, NOW, params_for_arm("class", w_max=10, **kw)).admit)

    edge = NOW.mono_us - us(30.0)
    assert admitted() == 5
    assert admitted(st={"last_typing_mono_us": edge}) == 3  # focus at exactly tau
    assert admitted(st={"last_typing_mono_us": edge - 1}) == 5
    assert admitted(st={"last_front_end_mono_us": edge}) == 3
    lane_busy = split(*pairs, (task("x", phase="done"), lane(admitted_mono_us=0)))
    d = decide(state(*lane_busy, clock=NOW), NOW, params_for_arm("class", w_max=10))
    assert d.focus and len(d.admit) == 3
    assert admitted(st={"last_typing_mono_us": edge}, n_fg=0) == 2


def test_never_held_calls_are_admitted_at_once_and_count_against_slots() -> None:
    a = task("a", phase="running", started_mono_us=0)
    tasks = [a, queued("b", 1)[0]]
    calls = [
        call("e1", "a", kind="embed"),
        call("e2", "a", kind="retrieve"),
        queued("b", 1)[1],
    ]
    d = run(tasks, calls, n_srv=3)
    assert d.admit_unheld == ("e1", "e2") and d.admit == ()
    assert reasons(d) == {"b": ("Resources", "")}
    locked = run(tasks, calls, m=Machine(lock="foreign"), n_srv=3)
    assert locked.admit_unheld == ("e1", "e2") and locked.admit == ()
    assert reasons(locked)["b"] == ("Reservation", "lock")
    # A never-held call of no task is admitted too; one naming an unknown
    # task is an orphan.
    bare = Call("e3", None, session_kind="front_bg", kind="embed")
    ghost = Call("e4", "ghost", kind="embed")
    d = run([], [bare, ghost])
    assert d.admit_unheld == ("e3",) and d.alarms == ("orphan_call:e4",)


@pytest.mark.parametrize(
    ("m", "kw", "expected"),
    [
        (Machine(), {}, None),
        (Machine(lock="foreign"), {}, ("Reservation", "lock")),
        (Machine(server_up=False), {}, ("Resources", "server_down")),
        (Machine(model="loading"), {}, ("Resources", "model_not_loaded")),
        (
            Machine(),
            {"kv_bytes_per_token": 1.0, "m_kv_bytes": 1000.0},
            ("Memory", "kv"),
        ),
    ],
)
def test_a_retry_rides_on_its_first_admission_but_waits_for_the_lock_the_server_and_memory(
    m: Machine, kw: dict[str, Any], expected: tuple[str, str] | None
) -> None:
    other_t, other_c = busy("other", 3, last_prompt_tokens=0)
    other_c = Call(
        "c-other", "other", prompt_tokens=0, completion_p99_tokens=0, admitted_mono_us=0
    )
    held = task("a", level=3, phase="held", started_mono_us=0)
    retry = call("r", "a", attempt=2, prompt_tokens=10)
    d = run([other_t, held], [other_c, retry], m=m, n_srv=3, n_a=1, **kw)
    if expected is None:  # beyond the limit and the background cap
        assert d.admit_unheld == ("r",) and "a" not in reasons(d)
    else:
        assert d.admit_unheld == () and reasons(d)["a"] == expected
    paused = run(
        [other_t, task("a", phase="held", paused=True)], [other_c, retry], m=m, **kw
    )
    assert ("r" in paused.admit_unheld) == (expected is None)
    assert reasons(paused).get("a") == (None if expected is None else ("Paused", ""))
    # A retry resumes no task: a waiting task stays waiting, its block kept.
    waiting = task("a", phase="waiting", wait_reason="approval", started_mono_us=0)
    st = state([other_t, waiting], [other_c, retry], machine=m, clock=NOW)
    st, out = step(st, NOW, params_for_arm("class", n_srv=3, n_a=1, **kw))
    assert ("r" in out.decision.admit_unheld) == (expected is None)
    assert find(st, "a").phase == "waiting"


def test_the_memory_check_holds_a_call_that_does_not_fit() -> None:
    def calls() -> list[Call]:
        return [
            Call(f"c-{k}", k, prompt_tokens=3000, completion_p99_tokens=2000)
            for k in "ab"
        ]

    tasks = [task("a"), task("b", created_mono_us=1)]
    kw: dict[str, Any] = {"kv_bytes_per_token": 1.0, "m_kv_bytes": 9000.0}
    d = run(tasks, calls(), **kw)
    assert d.admit == ("c-a",) and reasons(d) == {"b": ("Memory", "kv")}
    assert run(tasks, calls(), **{**kw, "m_kv_bytes": 10000.0}).admit == ("c-a", "c-b")
    gib = Machine(lock="self", lock_mem_gib=1.0)
    big = {**kw, "m_kv_bytes": 2.0**30 + 9000.0}
    assert run(tasks, calls(), m=gib, **big).admit == ("c-a",)
    assert run(tasks, calls(), m=gib, lock_lines=False, **big).admit == ("c-a", "c-b")


def test_w_max_bounds_running_tasks_and_waiting_tasks_do_not_count() -> None:
    r1 = task("r1", phase="running", started_mono_us=0)
    r2 = task("r2", phase="running", started_mono_us=0)
    q, qc = queued("q")
    d = run([r1, r2, q], [qc], w_max=2)
    assert d.admit == () and reasons(d) == {"q": ("Resources", "")}
    waiting = task("r2", phase="waiting", wait_reason="input")
    assert run([r1, waiting, q], [qc], w_max=2).admit == ("c-q",)
    assert run([r1, task("r2", phase="held", paused=True), q], [qc], w_max=2).admit == (
        "c-q",
    )
    blocked = run([r1, r2, q], [qc], deps=[Dep("r2", "q")], w_max=2)
    assert blocked.admit == ("c-q",)  # r2 waits on q: it holds no resident slot


@pytest.mark.parametrize("arm", ["class", "las2", "fifo"])
def test_a_p0_or_p1_refused_by_w_max_claims_one_lower_resident_slot(arm: str) -> None:
    s1 = task("s1", level=3, phase="running", started_mono_us=0)
    s2 = task("s2", level=3, phase="held", started_mono_us=1)
    s2c = call("c-s2", "s2", submit_mono_us=0)
    z, zc = queued("z", 0, 5)
    d = run([s1, s2, z], [s2c, zc], arm=arm, w_max=2)
    if arm == "fifo":
        assert d.parks == () and d.admit == ("c-s2",)
    else:
        assert d.parks == (("s2", "resident_slot"),)
        assert d.admit == () and reasons(d)["s2"] == ("Priority", "")
    p2 = run([s1, s2, task("z", level=2)], [s2c, zc], arm=arm, w_max=2)
    assert p2.parks == () and p2.admit == ("c-s2",)


def test_fifo_orders_by_submit_time_and_ignores_class() -> None:
    tasks, calls = split(queued("hi", 1, 5), queued("lo", 3, 1), queued("mid", 2, 3))
    assert order(tasks, calls, "fifo") == ["lo", "mid", "hi"]
    d = run(tasks, calls, arm="fifo", n_srv=3)
    assert d.admit == ("c-lo",) and reasons(d)["hi"] == ("Priority", "")


def test_class_orders_first_come_within_every_class() -> None:
    tasks = [
        task("x", created_mono_us=1),
        task("y", phase="held", started_mono_us=100),
        task("z", phase="held", started_mono_us=50),
        task("r", user_rank=0.5, created_mono_us=9),
        task(
            "p",
            level=1,
            level_source="user",
            started_mono_us=200,
            deadline_wall_us=WALL0,
        ),
        task("q", level=1, level_source="user", started_mono_us=100, phase="held"),
    ]
    calls = [call(f"c-{t.task_id}", t.task_id) for t in tasks]
    assert order(tasks, calls, "class") == ["q", "p", "r", "z", "y", "x"]


def test_las2_prefers_a_task_below_q_and_otherwise_ranks_by_score() -> None:
    below = task("below", phase="held", started_mono_us=9, promo_service_us=0)
    above = task("above", phase="held", started_mono_us=1, promo_service_us=us(60.0))
    calls = [call("c-below", "below"), call("c-above", "above")]
    assert order([below, above], calls, "las2") == ["below", "above"]
    assert order([below, above], calls, "class") == ["above", "below"]
    near = task("near", started_mono_us=9, deadline_wall_us=NOW.wall_us + us(600.0))
    mentioned = task("mentioned", started_mono_us=8, mention_mono_us=NOW.mono_us)
    plain = task("plain", started_mono_us=1)
    calls = [call(f"c-{k}", k) for k in ("near", "mentioned", "plain")]
    assert order([near, mentioned, plain], calls, "las2") == [
        "near",
        "mentioned",
        "plain",
    ]


def test_las2_runs_p1_earliest_deadline_first_with_null_last_and_late_behind() -> None:
    def p1(tid: str, deadline_s: float | None) -> Task:
        d = None if deadline_s is None else NOW.wall_us + us(deadline_s)
        return task(tid, level=1, level_source="user", deadline_wall_us=d)

    tasks = [
        p1("late", -1.0),
        p1("none", None),
        p1("later", 7200.0),
        p1("soon", 3600.0),
    ]
    calls = [call(f"c-{t.task_id}", t.task_id) for t in tasks]
    assert order(tasks, calls, "las2") == ["soon", "later", "none", "late"]


def test_p4_runs_only_in_an_idle_window_and_oldest_first() -> None:
    tasks, calls = split(
        queued("a", 4, created_mono_us=2),
        queued("b", 4, created_mono_us=1),
        queued("first", 4, created_mono_us=3, user_rank=0.0),
    )
    present = run(tasks, calls, arm="las2")
    assert present.admit == () and set(reasons(present).values()) == {("IdleOnly", "")}
    away = Machine(away_since_mono_us=LONG_AGO)
    assert order(tasks, calls, "las2", away) == ["first", "b", "a"]
    assert run(tasks, calls, m=away, arm="las2", n_srv=3).admit == ("c-first",)
    t2, c2 = queued("p2", 2, 9)
    d = run([*tasks, t2], [*calls, c2], arm="las2", n_srv=3, idle_gate=False)
    assert d.admit == ("c-p2",) and reasons(d)["first"] == ("Resources", "")


@pytest.mark.parametrize("arm", ARMS)
def test_a_foreign_lock_admits_no_task_call_under_every_arm(arm: str) -> None:
    tasks, calls = split(queued("p0", 0), queued("p2", 2), busy("r", 1))
    tasks.append(task("h", phase="held", started_mono_us=0))
    calls += [
        call("retry", "h", attempt=2, kind="reflect"),
        call("emb", "r", kind="embed"),
    ]
    calls = [c for c in calls if c.call_id != "c-r"]
    st = state(tasks, calls, machine=Machine(lock="foreign"), clock=NOW)

    def boom(*args: Any) -> Decision:
        raise RuntimeError

    for decide_fn in (decide, boom):
        d = dispatch(st, NOW, params_for_arm(arm), decide_fn=decide_fn).decision
        assert d.admit == () and d.admit_unheld == ("emb",)
        assert reasons(d)["h"] == ("Reservation", "lock")


@pytest.mark.parametrize(
    ("kw", "m", "expected"),
    [
        (
            {"paused": True, "phase": "interrupted", "level": 3},
            Machine(on_ac=False, battery_pct=45.0),
            ("Paused", ""),
        ),
        (
            {"phase": "interrupted", "budget_exhausted": True},
            Machine(),
            ("Input", "interrupted"),
        ),
        (
            {"phase": "waiting", "wait_reason": "approval", "dep": True},
            Machine(),
            ("Approval", ""),
        ),
        ({"dep": True, "budget_exhausted": True}, Machine(), ("Dependency", "")),
        ({"budget_exhausted": True}, Machine(lock="foreign"), ("Budget", "")),
        ({}, Machine(lock="foreign", server_up=False), ("Reservation", "lock")),
        (
            {},
            Machine(server_up=False, on_ac=False, battery_pct=10.0),
            ("Resources", "server_down"),
        ),
        (
            {"level": 3},
            Machine(on_ac=False, battery_pct=45.0, thermal="serious"),
            ("Battery", ""),
        ),
        (
            {"level": 3},
            Machine(thermal="serious", memory_pressure="warning"),
            ("Thermal", ""),
        ),
        (
            {"level": 4, "phase": "held"},
            Machine(memory_pressure="warning"),
            ("Memory", "pressure"),
        ),
        ({"level": 4}, Machine(), ("IdleOnly", "")),
        (
            {"level": 3},
            Machine(),
            ("Resources", ""),
        ),  # capacity: head took the last slot
    ],
)
def test_every_task_not_active_gets_exactly_one_reason_by_precedence(
    kw: dict[str, Any], m: Machine, expected: tuple[str, str]
) -> None:
    deps = [Dep("t", "other-q")] if kw.pop("dep", False) else []
    t, c = queued("t", **kw)
    head, head_c = queued("head", 2, created_mono_us=-1)  # ahead of t in class order
    full_t, full_c = busy("full", 1)
    d = run([t, head, full_t, task("other-q")], [c, head_c, full_c], deps, m=m, n_srv=4)
    assert reasons(d)["t"] == expected
    assert [r[0] for r in d.reasons].count("t") == 1
    both = run(
        [task("x", phase="running")],
        [call("in", "x", admitted_mono_us=0), call("next", "x")],
    )
    assert both.reasons == () and both.admit == ()


def test_the_fill_is_work_conserving() -> None:
    pairs = [queued(f"u{i}", 1, i) for i in range(2)] + [
        queued(f"b{i}", 2, 5 + i) for i in range(4)
    ]
    tasks, calls = split(*pairs)
    d = run(tasks, calls, w_max=10)
    assert len(d.admit) == 5  # one slot left: the fourth P2 is held by cap_bg alone
    assert reasons(d) == {"b3": ("Resources", "")}
    assert len(run(tasks, calls, w_max=10, n_a=6).admit) == 6
    assert len(run(tasks, calls, w_max=3, n_a=6).admit) == 3
    mem: dict[str, Any] = {"kv_bytes_per_token": 1.0, "m_kv_bytes": 4.0 * (3235 + 2000)}
    assert len(run(tasks, calls, w_max=10, n_a=6, **mem).admit) == 4


def test_yield_marks_the_worst_lower_task_once_per_waiting_high_call_with_h_y() -> None:
    actives = [
        busy("a3", 3),
        busy("b2", 2, started_mono_us=10),
        busy("c2", 2, started_mono_us=20),
    ]
    tasks, calls = split(*actives, queued("p", 1, 1), queued("q", 1, 2))
    d = run(tasks, calls, n_srv=5, n_a=3)
    assert d.limit_nonlane == 3 and d.yields == ("a3", "c2")
    one = run(tasks[:-1], calls[:-1], n_srv=5)
    assert one.yields == ("a3",)
    assert run(tasks, calls, n_srv=5, arm="fifo").yields == ()
    # las2: a P2 task yields only if score · h_y ≤ the best P2 candidate's score.
    q = us(60.0)
    z = task("z", phase="running", started_mono_us=0, promo_service_us=0)
    y = task("y", promo_service_us=q)
    base = [z, y, task("p", level=1, level_source="user")]
    calls = [call("c-z", "z", admitted_mono_us=0), call("c-y", "y"), call("c-p", "p")]
    assert run(base, calls, arm="las2", n_srv=1).yields == ()
    flipped = [task("z", phase="running", promo_service_us=q), task("y"), base[2]]
    assert run(flipped, calls, arm="las2", n_srv=1).yields == ("z",)


def test_park_marks_after_t_park_and_under_memory_pressure() -> None:
    long = us(1800.0)
    full_t, full_c = busy("full", 1)
    h = task("h", phase="held", started_mono_us=0, wait_held_us=long)
    short = task("s", phase="held", started_mono_us=0, wait_held_us=long - 1)
    calls = [full_c, call("c-h", "h"), call("c-s", "s")]
    d = run([full_t, h, short], calls, n_srv=3)
    assert d.parks == (("h", "t_park"),)
    pressured = [
        task("p3", level=3, phase="running"),
        task("p2", level=2, phase="held"),
    ]
    warn = run(pressured, m=Machine(memory_pressure="warning"))
    assert warn.parks == (("p3", "memory_pressure"),)
    crit = run(pressured, m=Machine(memory_pressure="critical"))
    assert crit.parks == (("p2", "memory_pressure"), ("p3", "memory_pressure"))
    hot: dict[str, Any] = {"kv_bytes_per_token": 1.0, "m_hot_bytes": 1.0}
    both = run([full_t, h], [full_c, call("c-h", "h")], n_srv=3, **hot)
    assert both.parks == (("h", "working_set"),)  # also past T_park: one entry
    warn_ws = run(pressured, m=Machine(memory_pressure="warning"), **hot)
    assert warn_ws.parks == (("p3", "memory_pressure"),)


def test_a_working_set_over_m_hot_parks_the_running_task_with_the_highest_level_and_worst_key() -> (
    None
):
    tasks = [
        task("a", level=2, phase="held", started_mono_us=1, last_prompt_tokens=4000),
        task("b", level=3, phase="held", started_mono_us=2, last_prompt_tokens=4000),
        task("c", level=3, phase="held", started_mono_us=3, last_prompt_tokens=4000),
    ]
    calls = [call(f"c-{t.task_id}", t.task_id) for t in tasks]
    hot: dict[str, Any] = {"kv_bytes_per_token": 1.0, "m_hot_bytes": 10000.0}
    d = run(tasks, calls, **hot)
    assert d.parks == (("c", "working_set"),)
    assert d.admit == ("c-a", "c-b") and reasons(d) == {"c": ("Memory", "kv")}
    assert run(tasks, calls, kv_bytes_per_token=1.0, m_hot_bytes=12000.0).parks == ()
    # Once parked, it does not start again while that would push the set over.
    parked = [*tasks[:2], task("c", level=3, phase="parked", last_prompt_tokens=4000)]
    again = run(parked, calls, **hot)
    assert again.parks == () and reasons(again) == {"c": ("Memory", "working_set")}


def test_a_p4_task_expires_after_7_days_without_activity() -> None:
    week = us(604800.0)
    old = task("old", level=4, last_activity_wall_us=NOW.wall_us - week)
    fresh = task("fresh", level=4, last_activity_wall_us=NOW.wall_us - week + 1)
    created = task("created", level=4, created_wall_us=NOW.wall_us - week)
    p3 = task("p3", level=3, created_wall_us=0)
    assert run([old, fresh, created, p3]).expire == ("created", "old")
    # Not while it runs or is admitted, nor while a waiter needs it (inherited).
    window = Machine(away_since_mono_us=NOW.mono_us - us(3600.0))
    now_admitted = run([old], [call("c-old", "old")], m=window)
    assert now_admitted.admit == ("c-old",) and now_admitted.expire == ()
    waiter = task("w", level=1, level_source="user")
    needed = run([old, waiter], deps=[Dep("w", "old", confirmed=True)])
    assert dict(needed.levels)["old"] == 1 and needed.expire == ()


def test_p0_meaning_conversation_counts_a_p0_task_as_p1() -> None:
    t, c = queued("t", 0)
    assert dict(run([t], [c]).levels) == {"t": 0}
    d = run([t], [c], p0_meaning="conversation")
    assert dict(d.levels) == {"t": 1} and "p0_conversation:t" in d.alarms


@pytest.mark.parametrize(
    ("make", "field"),
    [
        (lambda: Params(cloud_offload=True), "cloud_offload"),
        (lambda: Params(n_a=0), "n_a"),
        (lambda: Params(credit_cap_factor=0.5), "credit_cap_factor"),
        (lambda: Params(arm="fifo"), "floors"),
        (lambda: params_for_arm("fifo", yield_marks=True), "yield_marks"),
        (lambda: params_for_arm("fifo", floors=True), "floors"),
        (lambda: Params(tick_s=0.0), "tick_s"),
        (lambda: Params(prefill_tok_s=float("nan")), "prefill_tok_s"),
        (lambda: Params(phi2=1.5), "phi2"),
        (lambda: Params(m_front_bytes=float("inf")), "m_front_bytes"),
        (lambda: params_for_arm("gittins"), "arm"),
        (lambda: Params(n_srv=False), "n_srv"),  # would close the lane for good
        (lambda: params_for_arm("las2", n_srv=8.5), "n_srv"),
        (lambda: params_for_arm("las2", p1_cap=1.5), "p1_cap"),  # every decide() raised
        (lambda: Params(n_a=False), "n_a"),
        (lambda: params_for_arm("las2", aging=1), "aging"),
        (lambda: params_for_arm("las2", remaining_prior_s=None), "remaining_prior_s"),
    ],
)
def test_cloud_offload_and_bad_params_are_refused(make: Any, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make()
    assert params_for_arm("fifo").floors is False and Params().arm == "las2"
