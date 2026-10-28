"""The CLI's read-side contracts: typed text, deletion and cross-window marks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    SessionStoreError,
    UnknownSessionError,
)
from localmelo.support.sessions.sync import session_mark

from .helpers import ScriptedLLM, answer, started_agent


@pytest.fixture
async def store(tmp_path: Path) -> Any:
    store = SessionStore(tmp_path / "s.db", writer="cli")
    try:
        yield store
    finally:
        await store.aclose()


async def test_typed_text_is_exposed_by_every_turn_projection(
    store: SessionStore,
) -> None:
    turn = await store.begin_turn(
        None,
        "typed\n\n<attachment>body</attachment>",
        holder="mine",
        title="typed",
        new_session=NewSession(),
    )
    assert turn.typed_text == "typed"
    rows = await store.list_turns(turn.session_id)
    assert rows[0].typed_text == "typed"


async def test_mark_ignores_own_turns_but_tracks_foreign_and_undo(
    store: SessionStore,
) -> None:
    info = await store.create_session()
    initial = await session_mark(store, info.id, holder="mine")
    for holder in ("mine", "other"):
        turn = await store.begin_turn(info.id, "hello", holder=holder)
        await store.finish_turn(
            info.id,
            turn.turn,
            holder=holder,
            status="completed",
            cause="run",
            result=None,
            answer="reply",
        )
        mark = await session_mark(store, info.id, holder="mine")
        if holder == "mine":
            assert mark == initial
    assert mark.foreign_newest == 2
    assert mark.undone == 0
    await store.undo_last_turn(info.id)
    assert (await session_mark(store, info.id, holder="mine")).undone == 1


async def test_deleted_session_is_typed_but_list_turns_stays_empty(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    info = await store.create_session()
    agent = await started_agent(tmp_path / "ws", ScriptedLLM([answer("ok")]))
    runner = SessionRunner(agent, store, info.id)
    try:
        await store.delete_session(info.id)
        assert await store.list_turns(info.id) == []
        for operation in (
            runner.resume(),
            session_mark(store, info.id, holder="mine"),
            store.begin_turn(info.id, "hello", holder="mine"),
        ):
            with pytest.raises(UnknownSessionError) as exc:
                await operation
            assert isinstance(exc.value, SessionStoreError)
    finally:
        await runner.close()
        await agent.close()
