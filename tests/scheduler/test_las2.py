"""advance() books runnable waits, charges roots and runs LAS-2 promotion.

Requirements 19, 37, 39 and 43: promotion needs ρ times the service since
the last one and is tested only at or above Q; only order and capacity
waits are runnable; each wait increment is capped at two ticks; a tree
shares its root's LAS-2 state; malformed events raise and a repeated end
is ignored.
"""

from __future__ import annotations

import pytest

from localmelo.support.scheduler import (
    CallEnded,
    Params,
    TaskAdded,
    TaskUpdated,
    advance,
)
from localmelo.support.scheduler.model import us
from localmelo.support.scheduler.policy import _order_keys
from localmelo.tests.scheduler.helpers import call, clock, find, state, task

P = Params()
Q = us(60.0)


def test_promotion_waits_for_rho_times_the_service_since_the_last_promotion() -> None:
    def after(wait_s: float) -> tuple[int, int]:
        t = task("a", promo_service_us=us(100.0), wait_promo_us=us(wait_s))
        out = find(advance(state([t]), clock(0.0), [], P), "a")
        return out.promo_service_us, out.wait_promo_us

    assert after(399.999) == (us(100.0), us(399.999))
    assert after(400.0) == (0, 0)


def test_promotion_resets_both_counters_and_is_tested_only_at_or_above_q() -> None:
    below = task("a", promo_service_us=Q - 1, wait_promo_us=us(10_000.0))
    out = find(advance(state([below]), clock(0.0), [], P), "a")
    assert (out.promo_service_us, out.wait_promo_us) == (Q - 1, us(10_000.0))
    at_q = task("a", promo_service_us=Q, wait_promo_us=4 * Q, service_us=us(500.0))
    out = find(advance(state([at_q]), clock(0.0), [], P), "a")
    assert (out.promo_service_us, out.wait_promo_us, out.service_us) == (
        0,
        0,
        us(500.0),
    )
    # A root_id that names a task with a root of its own: the task charged
    # (one hop up) is promoted too, so its tree is never stuck at gamma 1.
    chain = [task("c"), task("b", root_id="c", promo_service_us=Q, wait_promo_us=4 * Q)]
    out = find(advance(state(chain), clock(0.0), [], P), "b")
    assert (out.promo_service_us, out.wait_promo_us) == (0, 0)


@pytest.mark.parametrize(
    ("reason", "detail", "runnable"),
    [
        ("Priority", "", True),
        ("Resources", "", True),
        ("Resources", "server_down", False),
        ("Battery", "", False),
        ("Budget", "", False),
        ("Dependency", "", False),
        ("IdleOnly", "", False),
        ("Memory", "kv", False),
        ("", "", False),
    ],
)
def test_waits_accrue_only_while_runnable(
    reason: str, detail: str, runnable: bool
) -> None:
    t = task("a", last_reason=reason, last_detail=detail)
    st = state([t], [call("c", "a")])
    out = find(advance(st, clock(10.0), [], P), "a")
    added = us(10.0) if runnable else 0
    assert (out.wait_total_us, out.wait_held_us, out.wait_promo_us) == (added,) * 3
    no_call = find(advance(state([t]), clock(10.0), [], P), "a")
    assert no_call.wait_held_us == 0


def test_a_wait_increment_is_capped_at_two_ticks_and_never_negative() -> None:
    t = task("a", last_reason="Priority")
    st = state([t], [call("c", "a")], clock=clock(100.0))
    jumped = advance(st, clock(100.0 + 36_000.0), [], P)
    assert find(jumped, "a").wait_held_us == us(30.0)
    back = advance(st, clock(50.0), [], P)
    assert find(back, "a").wait_held_us == 0 and back.clock_anomalies == 1
    assert back.clock == clock(50.0)


def test_a_child_task_is_ordered_by_its_roots_attained_service() -> None:
    root = task("root", phase="running", promo_service_us=Q, started_mono_us=5)
    child = task("child", root_id="root", kind="front_bg", started_mono_us=9)
    other = task("other", started_mono_us=9, created_mono_us=1)
    calls = [call("cc", "child"), call("co", "other")]
    st = state([root, child, other], calls)
    keys = _order_keys(st, clock(0.0), P)
    assert [k for k, _ in keys] == ["other", "child"]  # the child's γ is its root's
    # Waits of two children reach the root's wait_promo_us once per advance.
    kids = [
        task("k1", root_id="root", last_reason="Priority"),
        task("k2", root_id="root", last_reason="Resources"),
    ]
    st = state(
        [task("root", phase="running"), *kids], [call("c1", "k1"), call("c2", "k2")]
    )
    st = advance(st, clock(10.0), [], P)
    assert find(st, "root").wait_promo_us == us(10.0)
    assert find(st, "k1").wait_promo_us == 0
    # A child's charge goes to the root; promotion runs on the root only.
    st = state([root, child], [call("cc", "child", admitted_mono_us=0)])
    st = advance(st, clock(0.0), [CallEnded("cc", "ok", Q, 3000, 200)], P)
    r, c = find(st, "root"), find(st, "child")
    assert (
        r.service_us,
        r.promo_service_us,
        r.calls_done,
        r.completion_tokens_done,
    ) == (
        Q,
        2 * Q,
        1,
        200,
    )
    assert (c.service_us, c.calls_done, c.last_prompt_tokens) == (0, 0, 3000)
    promoted = state([task("root", promo_service_us=Q, wait_promo_us=4 * Q), child])
    after = advance(promoted, clock(0.0), [], P)
    assert find(after, "root").promo_service_us == 0


def test_advance_rejects_malformed_events_and_ignores_a_repeated_end() -> None:
    st = state([task("a", aged=True)], [call("pending", "a")], credit_us=(5, 5))
    with pytest.raises(ValueError, match="'a'"):
        advance(st, clock(0.0), [TaskAdded(task("a"))], P)
    with pytest.raises(ValueError, match="task_id"):
        advance(st, clock(0.0), [TaskUpdated("a", (("task_id", "b"),))], P)
    with pytest.raises(ValueError, match="service_us"):
        advance(st, clock(0.0), [TaskUpdated("a", (("service_us", 9),))], P)
    with pytest.raises(ValueError, match="ghost"):
        advance(st, clock(0.0), [TaskUpdated("ghost", (("paused", True),))], P)
    relevel = advance(st, clock(0.0), [TaskUpdated("a", (("level", 3),))], P)
    assert find(relevel, "a").level == 3 and not find(relevel, "a").aged
    floor_open = state(
        [task("a")], [call("pending", "a", level_at_admit=1)], floor_open=(True, True)
    )
    for ev in (
        CallEnded("unknown", "ok", us(9.0)),
        CallEnded("pending", "aborted", us(9.0)),
    ):
        out = advance(floor_open, clock(0.0), [ev], P)
        assert find(out, "a") == find(floor_open, "a") and out.credit_us == (0, 0)
    gone = advance(floor_open, clock(0.0), [CallEnded("pending", "aborted", 0)], P)
    assert gone.calls == ()
