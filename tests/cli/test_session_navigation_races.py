"""Regression tests for the independent session-state review."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionStore
from localmelo.support.sessions.sync import session_mark
from localmelo.tests.sessions.helpers import core_event

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import scripted, seed


async def test_foreign_finish_between_precheck_and_lease_is_loaded(
    setup: Setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    other = SessionStore(tmp_path / "sessions.db", writer="cli")
    foreign = await other.begin_turn(sid, "interleaved foreign request", holder="other")
    begin = setup.store.begin_turn

    async def finishing(*args: Any, **kwargs: Any) -> Any:
        await other.finish_turn(
            sid,
            foreign.turn,
            holder="other",
            status="completed",
            cause="run",
            result=None,
            answer="interleaved foreign answer",
        )
        return await begin(*args, **kwargs)

    monkeypatch.setattr(setup.store, "begin_turn", finishing)
    try:
        setup.session.enter("next request")
        await setup.idle()
        text = "\n".join(m.content for m in scripted(setup).requests[-1])
        assert "interleaved foreign request" in text
        assert "interleaved foreign answer" in text
    finally:
        await other.aclose()


async def test_retry_cancel_after_undo_restores_the_actual_message(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup, "keep this cancelled retry")
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    assert setup.session.runner is not None
    undo = setup.session.runner.undo
    committed, release = asyncio.Event(), asyncio.Event()

    async def slow_undo() -> Any:
        turn = await undo()
        committed.set()
        await release.wait()
        return turn

    monkeypatch.setattr(setup.session.runner, "undo", slow_undo)
    setup.session.enter("/retry")
    await asyncio.wait_for(committed.wait(), 3)
    assert setup.session.stop()
    release.set()
    await setup.idle()
    assert setup.app.buffer.text == "keep this cancelled retry"
    assert not scripted(setup).requests


async def test_cancelled_resume_does_not_move_target_workspace(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = await seed(setup, "original")
    target = await seed(setup, "target", workspace="/other")
    setup.session.enter(f"/resume {first}")
    await setup.idle()
    before = await setup.store.get_session(target)
    loading = asyncio.Event()

    async def blocked_resume(self: SessionRunner, **kwargs: Any) -> int:
        loading.set()
        await asyncio.Event().wait()
        return 0

    monkeypatch.setattr(SessionRunner, "resume", blocked_resume)
    setup.session.enter(f"/resume {target}")
    await asyncio.wait_for(loading.wait(), 3)
    assert setup.session.stop()
    await setup.idle()
    after = await setup.store.get_session(target)
    assert before is not None and after is not None
    assert (after.workspace, after.model) == (before.workspace, before.model)
    assert setup.session.runner is not None and setup.session.runner.session_id == first


async def test_mark_uses_counters_without_decoding_turn_payloads(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup)

    async def unbounded_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("mark must not load user texts, answers or JSON results")

    monkeypatch.setattr(setup.store, "list_turns", unbounded_read)
    mark = await session_mark(setup.store, sid, holder="mine")
    assert mark.foreign_newest == 1 and mark.undone == 0


async def test_numeric_expand_uses_the_latest_turn_that_had_calls(setup: Setup) -> None:
    info = await setup.store.create_session(
        workspace=setup.session.agent.workspace_root
    )
    first = await setup.store.begin_turn(info.id, "tool turn", holder="other")
    await setup.store.append_events(
        info.id,
        first.turn,
        [core_event("tool.finished", output="last tool output")],
        holder="other",
    )
    await setup.store.finish_turn(
        info.id,
        first.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer="done",
    )
    second = await setup.store.begin_turn(info.id, "answer only", holder="other")
    await setup.store.finish_turn(
        info.id,
        second.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer="no tools",
    )
    setup.session.enter(f"/resume {info.id}")
    await setup.idle()
    await setup.session.expand("1")
    assert "last tool output" in setup.out.text()
