"""Client request ids: acceptance, repeats, conflicts, withdrawals, the create scope."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    MAX_WITHDRAWN_PER_SCOPE,
    NewSession,
    SessionStore,
    SessionStoreError,
    TurnInfo,
)
from localmelo.support.sessions.submissions import (
    CLIENT_REQUEST_ID_RE,
    WITHDRAWN_ERROR,
    SubmissionConflict,
    SubmissionRepeated,
    SubmissionWithdrawn,
    check_submission,
)

from .helpers import (
    Clock,
    ScriptedLLM,
    check_invariants,
    events_of,
    rows,
    started_agent,
)

HOLDER = "testhost:1:aaaaaaaa"


def store_at(
    path: Path, writer: str = "gateway", clock: Clock | None = None
) -> SessionStore:
    return SessionStore(path, writer=writer, clock=clock or Clock())  # type: ignore[arg-type]


async def finish(store: SessionStore, turn: TurnInfo) -> None:
    await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="ok",
    )


async def test_a_repeat_returns_the_stored_turn_even_while_it_runs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = store_at(path)
    session = await store.create_session()
    turn = await store.begin_turn(
        session.id, "hello", holder=HOLDER, client_request_id="k1"
    )
    before = events_of(path, session.id)
    with pytest.raises(SubmissionRepeated) as repeated:
        await store.begin_turn(
            session.id, "hello", holder=HOLDER, client_request_id="k1"
        )
    assert repeated.value.turn == turn
    assert repeated.value.turn.status == "running"  # never SessionBusyError
    with pytest.raises(SubmissionConflict) as conflict:
        await store.begin_turn(
            session.id, "other", holder=HOLDER, client_request_id="k1"
        )
    assert conflict.value.submission.state == "accepted"
    assert rows(path, "SELECT COUNT(*) FROM turns") == [(1,)]
    assert events_of(path, session.id) == before
    # Also once the turn has ended.
    await finish(store, turn)
    before = events_of(path, session.id)
    with pytest.raises(SubmissionRepeated) as repeated:
        await store.begin_turn(
            session.id, "hello", holder=HOLDER, client_request_id="k1"
        )
    assert repeated.value.turn.id == turn.id
    assert repeated.value.turn.status == "completed"
    assert rows(path, "SELECT COUNT(*) FROM turns") == [(1,)]
    assert events_of(path, session.id) == before
    await store.aclose()


async def test_a_withdrawal_refuses_the_later_submit_and_changes_nothing_else(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = store_at(path)
    session = await store.create_session()
    first = await store.begin_turn(session.id, "first", holder=HOLDER)
    await finish(store, first)
    events_before = await store.list_events(session.id)

    async def resumed(store: SessionStore) -> list[tuple[str, str]]:
        """The window that the pairs resume loads."""
        agent = await started_agent(tmp_path / "ws", ScriptedLLM())
        try:
            runner = SessionRunner(agent, store, session.id)
            assert await runner.resume() == 1
            return [(m.role, m.content) for m in agent.conversation()]
        finally:
            await agent.close()

    pairs_before = await resumed(store)
    withdrawn = await store.withdraw_submission(session.id, "k1", user_text="second")
    assert withdrawn.state == "withdrawn"
    assert withdrawn.error == WITHDRAWN_ERROR
    assert withdrawn.turn is None and withdrawn.turn_id is None
    assert withdrawn.session_id == session.id and withdrawn.scope == session.id

    async def unchanged(store: SessionStore) -> None:
        with pytest.raises(SubmissionWithdrawn) as refused:
            await store.begin_turn(
                session.id, "second", holder=HOLDER, client_request_id="k1"
            )
        assert refused.value.submission.state == "withdrawn"
        assert rows(path, "SELECT MAX(turn) FROM turns") == [(1,)]
        info = await store.get_session(session.id)
        assert info is not None and info.turns == 1
        assert await store.list_events(session.id) == events_before
        assert [t.user_text for t in await store.list_turns(session.id)] == ["first"]
        assert await resumed(store) == pairs_before

    await unchanged(store)
    await store.aclose()
    again = store_at(path)  # also after a restart
    await unchanged(again)
    check_invariants(path, "gateway")
    await again.aclose()


async def test_withdrawing_an_accepted_or_withdrawn_key(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    store = store_at(path, clock=clock)
    session = await store.create_session()
    turn = await store.begin_turn(
        session.id, "hello", holder=HOLDER, client_request_id="k1"
    )
    accepted = await store.withdraw_submission(session.id, "k1", user_text="hello")
    assert accepted.state == "accepted" and accepted.turn_id == turn.id
    assert accepted.status == "running"
    with pytest.raises(SubmissionConflict):
        await store.withdraw_submission(session.id, "k1", user_text="other")
    first = await store.withdraw_submission(session.id, "k2", user_text="x")
    stored = rows(path, "SELECT * FROM submissions ORDER BY rowid")
    clock.advance(5)
    second = await store.withdraw_submission(session.id, "k2", user_text="x")
    assert first == second  # a no-op
    assert rows(path, "SELECT * FROM submissions ORDER BY rowid") == stored
    with pytest.raises(SubmissionConflict):
        await store.withdraw_submission(session.id, "k2", user_text="y")
    await store.aclose()


async def test_acceptance_and_withdrawal_race_and_exactly_one_wins(
    tmp_path: Path,
) -> None:
    for attempt in range(6):
        path = tmp_path / f"s{attempt}.db"
        store = store_at(path)
        session = await store.create_session()
        begin = store.begin_turn(
            session.id, "msg", holder=HOLDER, client_request_id="k"
        )
        withdraw = store.withdraw_submission(session.id, "k", user_text="msg")
        if attempt % 2:  # either may reach the store first
            withdrawn, begun = await asyncio.gather(
                withdraw, begin, return_exceptions=True
            )
        else:
            begun, withdrawn = await asyncio.gather(
                begin, withdraw, return_exceptions=True
            )
        if isinstance(begun, TurnInfo):
            # The turn won: the cancel saw it accepted.
            assert not isinstance(withdrawn, BaseException)
            assert withdrawn.state == "accepted" and withdrawn.turn_id == begun.id
        else:
            assert isinstance(begun, SubmissionWithdrawn)
            assert not isinstance(withdrawn, BaseException)
            assert withdrawn.state == "withdrawn"
            assert begun.submission == withdrawn  # the winner's row
            assert rows(path, "SELECT COUNT(*) FROM turns") == [(0,)]
        check_invariants(path, "gateway")
        await store.aclose()


async def test_two_submits_of_one_key_give_one_turn(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = store_at(path)
    session = await store.create_session()
    results = await asyncio.gather(
        *(
            store.begin_turn(session.id, "msg", holder=HOLDER, client_request_id="k")
            for _ in range(3)
        ),
        return_exceptions=True,
    )
    turns = [r for r in results if isinstance(r, TurnInfo)]
    repeats = [r for r in results if isinstance(r, SubmissionRepeated)]
    assert len(turns) == 1 and len(repeats) == 2
    assert all(r.turn.id == turns[0].id for r in repeats)
    await store.aclose()
    # Two terminals, one file.
    cli_path = tmp_path / "c.db"
    first, second = store_at(cli_path, "cli"), store_at(cli_path, "cli")
    shared = await first.create_session()
    pair = await asyncio.gather(
        first.begin_turn(shared.id, "msg", holder=HOLDER, client_request_id="k"),
        second.begin_turn(shared.id, "msg", holder=HOLDER, client_request_id="k"),
        return_exceptions=True,
    )
    assert sum(isinstance(r, TurnInfo) for r in pair) == 1
    assert sum(isinstance(r, SubmissionRepeated) for r in pair) == 1
    assert rows(cli_path, "SELECT COUNT(*) FROM turns") == [(1,)]
    await first.aclose()
    await second.aclose()


async def test_get_submission_reads_unknown_accepted_and_withdrawn(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path / "s.db")
    session = await store.create_session()
    assert await store.get_submission(session.id, "k1") is None
    turn = await store.begin_turn(
        session.id, "hello", holder=HOLDER, client_request_id="k1"
    )
    found = await store.get_submission(session.id, "k1")
    assert found is not None
    assert (found.state, found.turn_id, found.status, found.cause) == (
        "accepted",
        turn.id,
        "running",
        "",
    )
    assert found.turn == turn and found.turn.client_request_id == "k1"  # type: ignore[union-attr]
    await finish(store, turn)
    found = await store.get_submission(session.id, "k1")
    assert found is not None and (found.status, found.cause) == ("completed", "run")
    await store.withdraw_submission(session.id, "k2", user_text="x")
    withdrawn = await store.get_submission(session.id, "k2")
    assert withdrawn is not None and withdrawn.state == "withdrawn"
    assert (
        withdrawn.turn is None and withdrawn.status is None and withdrawn.cause is None
    )
    with pytest.raises(SessionStoreError, match="Unknown session: gone"):
        await store.get_submission("gone", "k1")
    await store.aclose()


@pytest.mark.parametrize("key", ["", "x" * 129, "-leading", "with space", "é"])
async def test_a_malformed_key_is_refused(tmp_path: Path, key: str) -> None:
    store = store_at(tmp_path / "s.db")
    session = await store.create_session()
    message = "client_request_id must be 1 to 128 characters"
    with pytest.raises(ValueError, match=message):
        await store.get_submission(session.id, key)
    with pytest.raises(ValueError, match=message):
        await store.withdraw_submission(session.id, key, user_text="x")
    with pytest.raises(ValueError, match=message):
        await store.begin_turn(session.id, "x", holder=HOLDER, client_request_id=key)
    await store.aclose()


async def test_random_uuids_and_128_characters_pass(tmp_path: Path) -> None:
    store = store_at(tmp_path / "s.db")
    session = await store.create_session()
    for key in (str(uuid.uuid4()), "req-" + uuid.uuid4().hex, "a" * 128, "A1._:-z"):
        assert CLIENT_REQUEST_ID_RE.fullmatch(key)
        assert await store.get_submission(session.id, key) is None
    await store.aclose()


async def test_check_submission_applies_begin_turns_rules(tmp_path: Path) -> None:
    store = store_at(tmp_path / "s.db")
    session = await store.create_session()
    check_submission(None, "anything")
    await store.begin_turn(session.id, "hello", holder=HOLDER, client_request_id="k1")
    accepted = await store.get_submission(session.id, "k1")
    assert accepted is not None
    with pytest.raises(SubmissionRepeated) as repeated:
        check_submission(accepted, "hello")
    assert repeated.value.turn == accepted.turn
    with pytest.raises(SubmissionConflict):
        check_submission(accepted, "other")
    withdrawn = await store.withdraw_submission(session.id, "k2", user_text="draft")
    with pytest.raises(SubmissionConflict):
        check_submission(withdrawn, "other")
    with pytest.raises(SubmissionWithdrawn):
        check_submission(withdrawn, "draft")
    await store.aclose()


async def test_a_created_sessions_key_is_found_through_its_own_scope(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = store_at(path)
    turn = await store.begin_turn(
        None, "hello", holder=HOLDER, client_request_id="k", new_session=NewSession()
    )
    sid = turn.session_id
    found = await store.get_submission(sid, "k")
    assert found is not None and found.state == "accepted" and found.turn_id == turn.id
    assert found.scope == ""
    before = events_of(path, sid)
    accepted = await store.withdraw_submission(sid, "k", user_text="hello")
    assert accepted.state == "accepted" and accepted.turn_id == turn.id
    with pytest.raises(SubmissionRepeated) as repeated:
        await store.begin_turn(sid, "hello", holder=HOLDER, client_request_id="k")
    assert repeated.value.turn.id == turn.id
    assert rows(path, "SELECT COUNT(*) FROM turns") == [(1,)]
    assert rows(path, "SELECT COUNT(*) FROM submissions") == [(1,)]
    assert events_of(path, sid) == before
    await store.aclose()


async def test_a_key_in_another_session_is_independent(tmp_path: Path) -> None:
    store = store_at(tmp_path / "s.db")
    one, two = await store.create_session(), await store.create_session()
    first = await store.begin_turn(one.id, "a", holder=HOLDER, client_request_id="k")
    second = await store.begin_turn(two.id, "b", holder=HOLDER, client_request_id="k")
    assert first.session_id != second.session_id
    assert (await store.get_submission(one.id, "k")).turn_id == first.id  # type: ignore[union-attr]
    assert (await store.get_submission(two.id, "k")).turn_id == second.id  # type: ignore[union-attr]
    await store.aclose()


async def test_the_withdrawal_cap_prunes_the_oldest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import localmelo.support.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "MAX_WITHDRAWN_PER_SCOPE", 5)
    assert MAX_WITHDRAWN_PER_SCOPE == 1_000
    clock = Clock()
    path = tmp_path / "s.db"
    store = store_at(path, clock=clock)
    session = await store.create_session()
    for index in range(8):
        clock.advance(1)
        await store.withdraw_submission(session.id, f"k{index}", user_text="x")
    kept = rows(
        path,
        "SELECT client_request_id FROM submissions WHERE scope = ? ORDER BY created_at",
        session.id,
    )
    assert [row[0] for row in kept] == [f"k{index}" for index in range(3, 8)]
    # Another scope keeps its own.
    await store.withdraw_submission(None, "c0", user_text="x")
    assert rows(path, "SELECT COUNT(*) FROM submissions WHERE scope = ''") == [(1,)]
    # The prune is in the withdrawal's transaction: when it fails, so does
    # the withdrawal.
    conn = store._conn
    assert conn is not None
    execute = conn.execute

    def failing(sql: str, *args: Any, **kwargs: Any) -> Any:
        if sql.startswith("DELETE FROM submissions WHERE rowid IN"):
            raise sqlite3.OperationalError("disk I/O error")
        return execute(sql, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(conn, "execute", failing)
        with pytest.raises(sqlite3.OperationalError):
            await store.withdraw_submission(session.id, "k8", user_text="x")
    assert (
        rows(
            path,
            "SELECT client_request_id FROM submissions WHERE scope = ? "
            "ORDER BY created_at",
            session.id,
        )
        == kept
    )
    await store.aclose()


async def test_the_create_scope(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = store_at(path)
    turn = await store.begin_turn(
        None, "hello", holder=HOLDER, client_request_id="k", new_session=NewSession()
    )
    # A repeat returns the turn of the session the first create made.
    with pytest.raises(SubmissionRepeated) as repeated:
        await store.begin_turn(
            None,
            "hello",
            holder=HOLDER,
            client_request_id="k",
            new_session=NewSession(),
        )
    assert repeated.value.turn.session_id == turn.session_id
    assert rows(path, "SELECT COUNT(*) FROM sessions") == [(1,)]
    found = await store.get_submission(None, "k")
    assert found is not None and found.session_id == turn.session_id
    # Deleting that session makes a later create with the key a new create.
    await finish(store, turn)
    assert await store.delete_session(turn.session_id)
    assert await store.get_submission(None, "k") is None
    again = await store.begin_turn(
        None, "hello", holder=HOLDER, client_request_id="k", new_session=NewSession()
    )
    assert again.session_id != turn.session_id
    # A withdrawn create refuses the later create, and writes no session.
    withdrawn = await store.withdraw_submission(None, "w", user_text="draft")
    assert withdrawn.session_id is None and withdrawn.scope == ""
    with pytest.raises(SubmissionWithdrawn):
        await store.begin_turn(
            None,
            "draft",
            holder=HOLDER,
            client_request_id="w",
            new_session=NewSession(),
        )
    assert rows(path, "SELECT COUNT(*) FROM sessions") == [(1,)]
    check_invariants(path, "gateway")
    await store.aclose()
