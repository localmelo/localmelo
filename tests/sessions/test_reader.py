"""SessionReader: a read-only change feed of finished and undone turns."""

from __future__ import annotations

import io
import json
import os
import sqlite3
from pathlib import Path

import pytest

from localmelo.support.sessions.reader import (
    READER_VERSION,
    SessionReader,
    export_turns,
)
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    SessionStoreError,
)

from .helpers import Clock, core_event, run_result

HOLDER = "testhost:1:aaaaaaaa"


async def turn_of(
    store: SessionStore,
    session_id: str | None,
    text: str,
    *,
    kind: str = "chat",
    finish: bool = True,
) -> str:
    """A turn with a run.started, finished unless *finish* is false; its session."""
    if session_id is None:
        turn = await store.begin_turn(
            None,
            text,
            holder=HOLDER,
            title=f"typed {text}",
            new_session=NewSession(kind=kind),
        )
    else:
        turn = await store.begin_turn(
            session_id, text, holder=HOLDER, title=f"typed {text}"
        )
    await store.append_events(
        turn.session_id,
        turn.turn,
        [core_event("run.started", run_id=f"run-{text}")],
        holder=HOLDER,
    )
    if finish:
        await store.finish_turn(
            turn.session_id,
            turn.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(result=f"answer {text}"),
            answer=f"answer {text}",
        )
    return turn.session_id


async def test_finished_records_in_cursor_order_with_their_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    sid = await turn_of(store, None, "one")
    await turn_of(store, sid, "two")
    task = await turn_of(store, None, "job", kind="task")
    reader = SessionReader(path)
    changes = await reader.changes()
    assert [(c.change, c.user_text) for c in changes] == [
        ("finished", "one"),
        ("finished", "two"),
        ("finished", "job"),
    ]
    first = changes[0]
    assert (first.writer, first.session_id, first.session_kind, first.turn) == (
        "gateway",
        sid,
        "chat",
        1,
    )
    assert (first.run_id, first.typed_text, first.status, first.cause) == (
        "run-one",
        "typed one",
        "completed",
        "run",
    )
    assert first.answer == "answer one" and first.note == "" and not first.undone
    assert first.result is not None and first.result["result"] == "answer one"
    assert first.session_title == "typed one" and first.events is None
    ids = [c.event_id for c in changes]
    assert ids == sorted(ids)
    assert [c.session_id for c in await reader.changes(kinds=("task",))] == [task]
    with_events = await reader.changes(with_events=True, limit=1)
    assert with_events[0].events is not None
    assert [e["type"] for e in with_events[0].events] == [
        "turn.started",
        "session.title",
        "run.started",
        "turn.finished",
    ]
    for bad in (0, 5001):
        with pytest.raises(ValueError):
            await reader.changes(limit=bad)
    await reader.aclose()
    await store.aclose()


async def test_an_undo_after_the_fact_and_paging(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="cli")
    sid = await turn_of(store, None, "one")
    await turn_of(store, sid, "two")
    reader = SessionReader(path)
    page = await reader.changes(limit=1)
    assert [c.user_text for c in page] == ["one"]
    cursor = page[-1].event_id
    page = await reader.changes(after_event_id=cursor)
    assert [c.user_text for c in page] == ["two"]
    cursor = page[-1].event_id
    await store.undo_last_turn(sid)
    later = await reader.changes(after_event_id=cursor)
    assert [(c.change, c.user_text, c.undone, c.writer) for c in later] == [
        ("undone", "two", True, "cli")
    ]
    # Read from the start again, the finished record shows the turn as now.
    assert [c.undone for c in await reader.changes()] == [False, True, True]
    await reader.aclose()
    await store.aclose()


async def test_running_turns_wait_for_their_end(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    sid = await turn_of(store, None, "running", finish=False)
    reader = SessionReader(path)
    assert await reader.changes() == []
    await store.finish_turn(
        sid,
        1,
        holder=HOLDER,
        status="cancelled",
        cause="user_stop",
        result=None,
        answer="x",
    )
    (change,) = await reader.changes()
    assert (change.status, change.cause) == ("cancelled", "user_stop")
    await reader.aclose()
    await store.aclose()


async def test_it_reads_the_gateways_file_while_the_gateway_owns_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    sid = await turn_of(store, None, "one")
    await turn_of(store, sid, "two", finish=False)  # a turn runs
    assert store._conn is not None
    before = list(await store._conn.execute_fetchall("PRAGMA data_version"))
    reader = SessionReader(path)
    assert await reader.writer() == "gateway"
    assert [c.user_text for c in await reader.changes(with_events=True)] == ["one"]
    after = list(await store._conn.execute_fetchall("PRAGMA data_version"))
    assert after == before  # the reader wrote nothing
    await reader.aclose()
    await store.aclose()


async def test_it_never_writes_a_closed_file(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="cli")
    await turn_of(store, None, "one")
    await store.aclose()
    for suffix in ("-wal", "-shm"):
        assert not Path(f"{path}{suffix}").exists()
    content, stat = path.read_bytes(), path.stat()
    reader = SessionReader(path)
    assert await reader.writer() == "cli"
    assert len(await reader.changes(with_events=True)) == 1
    await reader.aclose()
    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    with pytest.raises(SessionStoreError, match="closed"):
        await reader.changes()


async def test_the_cursor_survives_a_clock_that_steps_back(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    store = SessionStore(path, writer="gateway", clock=clock)
    sid = await turn_of(store, None, "one")
    reader = SessionReader(path)
    (first,) = await reader.changes()
    clock.advance(-3600)  # the clock steps backwards between turns
    await turn_of(store, sid, "two")
    (second,) = await reader.changes(after_event_id=first.event_id)
    assert second.user_text == "two" and second.finished_at < first.finished_at
    assert second.event_id > first.event_id
    await reader.aclose()
    await store.aclose()


async def test_a_newer_schema_raises_and_an_empty_file_has_no_records(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.db"
    empty.touch()
    reader = SessionReader(empty)
    assert await reader.writer() is None
    assert await reader.changes() == []
    await reader.aclose()
    newer = tmp_path / "newer.db"
    store = SessionStore(newer, writer="gateway")
    await store.create_session()
    await store.aclose()
    conn = sqlite3.connect(newer)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()
    reader = SessionReader(newer)
    with pytest.raises(SessionStoreError, match="has schema version 4"):
        await reader.changes()
    await reader.aclose()
    junk = tmp_path / "junk.db"
    junk.write_bytes(os.urandom(2048))
    reader = SessionReader(junk)
    with pytest.raises(SessionStoreError, match="is not a localmelo sessions file"):
        await reader.writer()
    await reader.aclose()
    missing = SessionReader(tmp_path / "missing.db")
    with pytest.raises(SessionStoreError):
        await missing.changes()
    assert not (tmp_path / "missing.db").exists()


async def test_every_version_0_file_has_no_records(tmp_path: Path) -> None:
    history = tmp_path / "history.db"  # version 0 with a table of its own
    conn = sqlite3.connect(history)
    conn.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    before = history.read_bytes()
    reader = SessionReader(history)
    assert await reader.writer() is None
    assert await reader.changes() == []
    await reader.aclose()
    assert history.read_bytes() == before
    unmarked = tmp_path / "unmarked.db"  # version 1 without a writer marker
    conn = sqlite3.connect(unmarked)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    reader = SessionReader(unmarked)
    with pytest.raises(SessionStoreError) as raised:
        await reader.changes()
    assert str(raised.value) == f"{unmarked} has no valid writer marker"
    await reader.aclose()


async def test_export_turns_writes_one_line_per_change(tmp_path: Path) -> None:
    gateway_path, cli_path = tmp_path / "g.db", tmp_path / "c.db"
    gateway = SessionStore(gateway_path, writer="gateway")
    terminal = SessionStore(cli_path, writer="cli")
    sid = await turn_of(gateway, None, "one")
    await turn_of(gateway, sid, "two")
    await turn_of(terminal, None, "three")
    out = io.StringIO()
    cursors = await export_turns([gateway_path, str(cli_path)], out)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["run_id"] for line in lines] == ["run-one", "run-two", "run-three"]
    assert all(line["kind"] == "localmelo.session_turn" for line in lines)
    assert all(
        line["v"] == READER_VERSION == 1 and line["note"] == "" for line in lines
    )
    assert set(cursors) == {str(gateway_path), str(cli_path)}
    assert cursors[str(gateway_path)] == max(
        line["event_id"] for line in lines if line["writer"] == "gateway"
    )
    # From the cursors, only what is new.
    await turn_of(terminal, None, "four")
    out = io.StringIO()
    again = await export_turns([gateway_path, str(cli_path)], out, after=cursors)
    assert [json.loads(line)["user_text"] for line in out.getvalue().splitlines()] == [
        "four"
    ]
    assert again[str(gateway_path)] == cursors[str(gateway_path)]
    await gateway.aclose()
    await terminal.aclose()


async def test_a_tasks_question_is_marked_by_its_origin_and_older_files_read_as_the_persons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    sid = await turn_of(store, None, "one")
    asked = await store.begin_turn(
        sid, "[task t1 · Fix asks · data]", holder=HOLDER, origin="ask_owner"
    )
    await store.finish_turn(
        sid, asked.turn, holder=HOLDER, status="completed", cause="run",
        result=run_result(result="B"), answer="B",
    )  # fmt: skip
    await store.aclose()
    out = io.StringIO()
    await export_turns([path], out)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [(line["user_text"], line["origin"]) for line in lines] == [
        ("one", ""),
        ("[task t1 · Fix asks · data]", "ask_owner"),
    ]
    # A file a schema-2 localmelo wrote has no origin column: its turns are the person's.
    old = tmp_path / "old.db"
    with sqlite3.connect(path) as conn, sqlite3.connect(old) as copy:
        conn.backup(copy)
    with sqlite3.connect(old) as conn:
        conn.execute("ALTER TABLE turns DROP COLUMN origin")
        conn.execute("PRAGMA user_version = 2")
    reader = SessionReader(old)
    assert [c.origin for c in await reader.changes()] == ["", ""]
    await reader.aclose()
