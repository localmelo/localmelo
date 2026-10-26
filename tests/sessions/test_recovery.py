"""Recovery of turns whose process is gone, and the store's write failures.

In-process: a store that is closed without finishing its turn stands in for
a process that died (``test_restart_process.py`` crosses a real process
boundary).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.schema import answer_text
from localmelo.support.sessions.store import (
    FAILED_ERROR,
    LEASE_S,
    RECOVERED_APPROVAL_REASON,
    STALE_ERROR,
    NewSession,
    SessionBusyError,
    SessionStore,
    SessionStoreError,
    TurnInfo,
    process_holder,
)

from .helpers import Clock, approval_event, check_invariants, core_event, run_result

HOLDER = process_holder("gateway")  # this host and pid: a lease that looks live


async def running_turn(
    path: Path, events: list[dict[str, Any]] | None = None, *, key: str | None = None
) -> TurnInfo:
    """A gateway turn left running by a store that then went away."""
    store = SessionStore(path, writer="gateway")
    turn = await store.begin_turn(
        None, "hello", holder=HOLDER, client_request_id=key, new_session=NewSession()
    )
    if events:
        await store.append_events(turn.session_id, turn.turn, events, holder=HOLDER)
    await store.aclose()
    return turn


async def test_a_gateway_open_recovers_every_running_turn_at_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    turn = await running_turn(path, [core_event("run.started", run_id="task-1")])
    clock = Clock()
    store = SessionStore(path, writer="gateway", clock=clock)
    await store.open()  # the lease looks live, and LEASE_S has not passed
    (recovered,) = await store.list_turns(turn.session_id)
    assert (recovered.status, recovered.cause) == ("crashed", "crash")
    assert recovered.run_id == "task-1"
    assert recovered.answer == answer_text("crashed", "")
    assert recovered.result is None
    events = await store.list_events(turn.session_id)
    assert events[-1].event["type"] == "turn.finished"
    assert events[-1].event["data"] == {
        "turn": 1,
        "turn_id": turn.id,
        "status": "crashed",
        "error": STALE_ERROR,
        "cause": "crash",
    }
    check_invariants(path, "gateway")
    # A turn this store runs survives an explicit recovery and a wake from sleep.
    mine = await store.begin_turn(turn.session_id, "again", holder=HOLDER)
    clock.advance(10 * LEASE_S)
    assert await store.recover_stale_turns() == 0
    with pytest.raises(SessionBusyError):
        await store.begin_turn(turn.session_id, "third", holder=HOLDER)
    assert (await store.list_turns(turn.session_id))[-1].status == "running"
    assert mine.turn == 2
    await store.aclose()


async def test_recovery_withdraws_the_approvals_nobody_answered(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    turn = await running_turn(
        path,
        [
            approval_event("approval.requested", "r-answered"),
            approval_event("approval.resolved", "r-answered"),
            approval_event("approval.requested", "r-open", call_id="call_2"),
        ],
    )
    store = SessionStore(path, writer="gateway")
    events = await store.list_events(turn.session_id)
    types = [e.event["type"] for e in events]
    assert types[-2:] == ["approval.resolved", "turn.finished"]
    withdrawn = events[-2].event["data"]
    assert withdrawn == {
        "request_id": "r-open",
        "call_id": "call_2",
        "tool_name": "shell_exec",
        "outcome": "withdrawn",
        "origin": "withdrawn",
        "reason": RECOVERED_APPROVAL_REASON,
        "remember": False,
    }
    resolved = [e for e in events if e.event["type"] == "approval.resolved"]
    assert [e.event["data"]["request_id"] for e in resolved] == ["r-answered", "r-open"]
    check_invariants(path, "gateway")
    await store.aclose()


@pytest.mark.parametrize(
    ("status", "cause"), [("completed", "run"), ("cancelled", "crash")]
)
async def test_a_stored_run_finished_gives_the_outcome(
    tmp_path: Path, status: str, cause: str
) -> None:
    path = tmp_path / "s.db"
    result = run_result(
        status=status, result="the answer" if status == "completed" else ""
    )
    turn = await running_turn(
        path,
        [core_event("run.started"), core_event("run.finished", seq=2, result=result)],
    )
    store = SessionStore(path, writer="gateway")
    (recovered,) = await store.list_turns(turn.session_id)
    assert (recovered.status, recovered.cause) == (status, cause)
    assert recovered.result == result
    assert recovered.answer == answer_text(status, result["result"])
    finished = (await store.list_events(turn.session_id))[-1].event["data"]
    assert (finished["status"], finished["cause"]) == (status, cause)
    check_invariants(path, "gateway")
    await store.aclose()


async def test_a_turn_without_run_started_keeps_an_empty_run_id(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    turn = await running_turn(path)
    store = SessionStore(path, writer="gateway")
    (recovered,) = await store.list_turns(turn.session_id)
    assert recovered.run_id == "" and recovered.status == "crashed"
    await store.aclose()


async def test_recovery_is_idempotent(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "c.db"
    store = SessionStore(path, writer="cli", clock=clock)
    remote = "otherhost:77:abababab"
    one = await store.begin_turn(None, "a", holder=remote, new_session=NewSession())
    two = await store.begin_turn(None, "b", holder=remote, new_session=NewSession())
    check_invariants(path, "cli")
    clock.advance(LEASE_S + 1)
    assert await store.recover_stale_turns(one.session_id) == 1
    check_invariants(path, "cli")
    assert await store.recover_stale_turns() == 1
    check_invariants(path, "cli")
    events = [len(await store.list_events(t.session_id)) for t in (one, two)]
    assert await store.recover_stale_turns() == 0
    assert [len(await store.list_events(t.session_id)) for t in (one, two)] == events
    await store.aclose()
    reopened = SessionStore(path, writer="cli", clock=clock)
    await reopened.open()
    assert [len(await reopened.list_events(t.session_id)) for t in (one, two)] == events
    await reopened.aclose()


async def test_recoverys_events_follow_the_old_ids(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    turn = await running_turn(path, [core_event("run.started")], key="k1")
    seen = SessionStore(path, writer="gateway")
    # What a client saw before the crash: read through the file, not the store.
    with sqlite3.connect(path) as conn:
        last_seen = conn.execute("SELECT MAX(id) FROM events").fetchone()[0]
    replay = await seen.list_events(turn.session_id, after_id=last_seen)
    assert [e.event["type"] for e in replay] == ["turn.finished"]
    assert replay[0].event["data"]["cause"] == "crash"
    assert replay[0].id is not None and replay[0].id > last_seen
    submission = await seen.get_submission(turn.session_id, "k1")
    assert submission is not None and submission.status == "crashed"
    await seen.aclose()


@pytest.mark.parametrize("write", ["append_events", "finish_turn"])
async def test_a_failed_write_of_a_running_turn_fails_the_gateway_store_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: str
) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    turn = await store.begin_turn(
        None, "hello", holder=HOLDER, client_request_id="k1", new_session=NewSession()
    )
    other = await store.create_session()
    error = sqlite3.OperationalError("disk I/O error")

    def fail(self: SessionStore, label: str) -> None:
        if label == write:
            raise error

    monkeypatch.setattr(SessionStore, "_before_commit", fail)
    with pytest.raises(sqlite3.OperationalError):
        if write == "append_events":
            await store.append_events(
                turn.session_id, 1, [core_event("run.started")], holder=HOLDER
            )
        else:
            await store.finish_turn(
                turn.session_id,
                1,
                holder=HOLDER,
                status="completed",
                cause="run",
                result=None,
                answer="x",
            )
    monkeypatch.undo()
    assert store.failed is error
    refused = [
        store.begin_turn(other.id, "x", holder=HOLDER),
        store.create_session(),
        store.withdraw_submission(other.id, "k9", user_text="x"),
        store.append_session_event(other.id, "task.progress", {}),
        store.rename_session(other.id, "x"),
        store.update_session(other.id, model="m"),
        store.delete_session(other.id),
        store.undo_last_turn(other.id),
        store.apply_retention(),
    ]
    for call in refused:
        with pytest.raises(SessionStoreError) as raised:
            await call
        assert str(raised.value) == FAILED_ERROR
    # Reads keep working, and the running turn's own writes are still taken.
    assert (await store.list_turns(turn.session_id))[0].status == "running"
    assert (await store.get_submission(turn.session_id, "k1")) is not None
    assert await store.append_events(
        turn.session_id, 1, [core_event("step.started")], holder=HOLDER
    )
    await store.aclose()
    # The restart repairs the turn.
    restarted = SessionStore(path, writer="gateway")
    (recovered,) = await restarted.list_turns(turn.session_id)
    assert (recovered.status, recovered.cause) == ("crashed", "crash")
    assert restarted.failed is None
    check_invariants(path, "gateway")
    await restarted.aclose()


async def test_a_failed_begin_turn_leaves_the_store_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    session = await store.create_session()

    def fail(self: SessionStore, label: str) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SessionStore, "_before_commit", fail)
    with pytest.raises(sqlite3.OperationalError):
        await store.begin_turn(session.id, "x", holder=HOLDER)
    monkeypatch.undo()
    assert store.failed is None
    assert (await store.begin_turn(session.id, "x", holder=HOLDER)).turn == 1
    await store.aclose()


async def test_a_cli_failure_does_not_fail_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "c.db", writer="cli")
    turn = await store.begin_turn(None, "x", holder=HOLDER, new_session=NewSession())

    def fail(self: SessionStore, label: str) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SessionStore, "_before_commit", fail)
    with pytest.raises(sqlite3.OperationalError):
        await store.append_events(
            turn.session_id, 1, [core_event("run.started")], holder=HOLDER
        )
    monkeypatch.undo()
    assert store.failed is None
    assert (await store.create_session()).kind == "chat"
    await store.aclose()


async def test_a_failed_recovery_fails_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.db"
    await running_turn(path)

    def fail(self: SessionStore, label: str) -> None:
        if label == "recover":
            raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SessionStore, "_before_commit", fail)
    store = SessionStore(path, writer="gateway")
    with pytest.raises(SessionStoreError, match="disk I/O error"):
        await store.open()
    monkeypatch.undo()
    await store.open()  # retried, and the turn is recovered now
    check_invariants(path, "gateway")
    await store.aclose()
