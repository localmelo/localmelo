"""Committed worker evidence uses durable events from the active turn only."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import localmelo.support.gateway.worker_evidence as evidence_module
from localmelo.support.gateway.worker_evidence import EvidenceError, committed_evidence
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo

HOLDER = "test-worker"


def event(kind: str, data: dict[str, Any], *, run_id: str = "run-1") -> dict[str, Any]:
    return {"v": 1, "type": kind, "run_id": run_id, "seq": 1, "ts": 1.0, "data": data}


class Runner:
    def __init__(self, turn: TurnInfo) -> None:
        self.current_turn: TurnInfo | None = turn
        self.session_id = turn.session_id
        self.run_ended = False
        self.on_flush: Any = None

    async def flush(self) -> None:
        if self.on_flush is not None:
            self.on_flush()


def as_runner(runner: Runner) -> SessionRunner:
    return cast(SessionRunner, runner)


@pytest.fixture
async def setup(tmp_path: Path) -> AsyncIterator[tuple[SessionStore, Runner, TurnInfo]]:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    turn = await store.begin_turn(None, "test", holder=HOLDER, new_session=NewSession())
    try:
        yield store, Runner(turn), turn
    finally:
        await store.aclose()


async def append(store: SessionStore, turn: TurnInfo, *events: dict[str, Any]) -> None:
    await store.append_events(turn.session_id, turn.turn, events, holder=HOLDER)


def run() -> dict[str, Any]:
    return event(
        "run.started",
        {"query": "q", "max_steps": 4, "workspace_root": None, "approval": True},
    )


def step(number: int, attempt: int, *, run_id: str = "run-1") -> dict[str, Any]:
    return event("step.started", {"step": number, "attempt": attempt}, run_id=run_id)


def reflection(attempt: int) -> dict[str, Any]:
    return event("reflection.started", {"attempt": attempt, "failure_type": "failure"})


def approval(request_id: str) -> dict[str, Any]:
    return event(
        "approval.requested",
        {
            "request_id": request_id,
            "call_id": "call",
            "tool_name": "tool",
            "arguments": {},
            "preview": "",
            "workspace_root": "",
        },
        run_id="",
    )


async def test_plan_reflection_and_paging(
    setup: tuple[SessionStore, Runner, TurnInfo], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, runner, turn = setup
    monkeypatch.setattr(evidence_module, "_PAGE_SIZE", 2)
    await append(store, turn, run(), step(1, 1), step(2, 2))
    found = await committed_evidence(
        as_runner(runner), store, turn_id=turn.id, site="plan"
    )
    assert (found.step, found.attempt, found.site) == (2, 2, "plan")
    await append(store, turn, reflection(2))
    found = await committed_evidence(
        as_runner(runner), store, turn_id=turn.id, site="reflect"
    )
    assert (found.step, found.attempt) == (2, 2)
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")


async def test_approval_request_identity_and_stale_site(
    setup: tuple[SessionStore, Runner, TurnInfo],
) -> None:
    store, runner, turn = setup
    await append(store, turn, run(), step(1, 1), approval("a"), approval("b"))
    result = await committed_evidence(
        as_runner(runner), store, turn_id=turn.id, site="approval", request_id="b"
    )
    assert (result.step, result.attempt, result.request_id) == (1, 1, "b")
    with pytest.raises(EvidenceError):
        await committed_evidence(
            as_runner(runner),
            store,
            turn_id=turn.id,
            site="approval",
            request_id="missing",
        )
    await append(store, turn, step(2, 1))
    with pytest.raises(EvidenceError):
        await committed_evidence(
            as_runner(runner), store, turn_id=turn.id, site="approval", request_id="a"
        )


async def test_missing_write_and_failed_store(
    setup: tuple[SessionStore, Runner, TurnInfo], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, runner, turn = setup
    await append(store, turn, run())
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")

    def fail(label: str) -> None:
        if label == "append_events":
            raise sqlite3.OperationalError("injected")

    monkeypatch.setattr(store, "_before_commit", fail)
    with pytest.raises(sqlite3.OperationalError):
        await append(store, turn, step(1, 1))
    assert store.failed is not None
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")


async def test_turn_change_after_flush_or_page(
    setup: tuple[SessionStore, Runner, TurnInfo], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, runner, turn = setup
    await append(store, turn, run(), step(1, 1))
    runner.on_flush = lambda: setattr(runner, "current_turn", replace(turn, id="new"))
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")
    runner.current_turn, runner.on_flush = turn, None
    original = store.list_events

    async def changed(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        runner.current_turn = replace(turn, id="new")
        return result

    monkeypatch.setattr(store, "list_events", changed)
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")


async def test_ambiguous_steps_fail_closed(
    setup: tuple[SessionStore, Runner, TurnInfo],
) -> None:
    store, runner, turn = setup
    await append(store, turn, run(), step(1, 1), step(1, 1))
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")


async def test_other_run_cannot_supply_coordinates(
    setup: tuple[SessionStore, Runner, TurnInfo],
) -> None:
    store, runner, turn = setup
    await append(store, turn, run(), step(1, 1, run_id="newer-run"))
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")


async def test_malformed_persisted_step_is_rejected(
    setup: tuple[SessionStore, Runner, TurnInfo],
) -> None:
    store, runner, turn = setup
    await append(store, turn, run(), step(1, 1))
    broken = event("step.started", {"step": True, "attempt": 1})
    async with store.component_transaction("break_event") as conn:
        await conn.execute(
            "UPDATE events SET event = ? WHERE session_id = ? AND type = 'step.started'",
            (json.dumps(broken), turn.session_id),
        )
    with pytest.raises(EvidenceError):
        await committed_evidence(as_runner(runner), store, turn_id=turn.id, site="plan")


async def test_prior_turn_events_do_not_supply_missing_coordinates(
    setup: tuple[SessionStore, Runner, TurnInfo],
) -> None:
    from localmelo.tests.sessions.helpers import run_result

    store, runner, turn = setup
    await append(store, turn, run(), step(1, 1))
    await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="done",
    )
    newer = await store.begin_turn(turn.session_id, "next", holder=HOLDER)
    runner.current_turn = newer
    await append(store, newer, run())
    with pytest.raises(EvidenceError):
        await committed_evidence(
            as_runner(runner), store, turn_id=newer.id, site="plan"
        )
