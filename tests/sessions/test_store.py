"""The session store: schema, files, writers, sessions, turns, events, leases.

Every test uses a temporary file and HOME, and an injected clock where time
matters; nothing opens ``~/.cache/localmelo`` or the network.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import localmelo.melo.memory._sqlite as core_sqlite
import localmelo.support.sessions.store as store_mod
from localmelo.melo.schema import ImagePart, Message, answer_text, split_thinking
from localmelo.support.sessions.store import (
    LEASE_S,
    RESERVED_EVENT_PREFIXES,
    SESSION_EVENT_FIELDS,
    WRITER_BATCH_MAX,
    NewSession,
    SessionBusyError,
    SessionStore,
    SessionStoreError,
    StoredEvent,
    cli_host_id,
    default_path,
    title_from_text,
)
from localmelo.support.sessions.submissions import SubmissionRepeated
from localmelo.support.sessions.window import decode_window

from .helpers import (
    DAY,
    Clock,
    approval_event,
    check_invariants,
    core_event,
    rows,
    run_result,
)

HOLDER = "testhost:1:aaaaaaaa"


def gateway(path: Path, clock: Clock | None = None) -> SessionStore:
    return SessionStore(path, writer="gateway", clock=clock or Clock())


def cli(path: Path, clock: Clock | None = None) -> SessionStore:
    return SessionStore(path, writer="cli", clock=clock or Clock())


async def finished_turn(
    store: SessionStore,
    session_id: str | None = None,
    text: str = "hello",
    *,
    holder: str = HOLDER,
    answer: str = "done",
    window: list[Message] | None = None,
) -> tuple[str, int]:
    """Begin and finish one turn; return its session id and number."""
    if session_id is None:
        turn = await store.begin_turn(
            None, text, holder=holder, new_session=NewSession()
        )
    else:
        turn = await store.begin_turn(session_id, text, holder=holder)
    await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=holder,
        status="completed",
        cause="run",
        result=run_result(result=answer),
        answer=answer,
        window=window,
    )
    return turn.session_id, turn.turn


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


# ── Schema ──────────────────────────────────────────────────────────────────


async def test_a_new_file_gets_schema_version_1(tmp_path: Path) -> None:
    path = tmp_path / "s" / "sessions.db"
    store = gateway(path)
    await store.open()
    await store.aclose()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert rows(path, "PRAGMA user_version") == [(1,)]
    assert rows(path, "SELECT value FROM store_info WHERE key = 'writer'") == [
        ("gateway",)
    ]
    assert rows(path, "SELECT COUNT(*) FROM store_info WHERE key = 'created_at'") == [
        (1,)
    ]
    assert rows(path, "SELECT * FROM schema_components") == []
    names = {
        row[0]
        for row in rows(
            path, "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        )
    }
    for name in (
        "store_info",
        "schema_components",
        "sessions",
        "turns",
        "submissions",
        "events",
        "blobs",
        "windows",
        "turns_running",
        "submissions_turn",
        "submissions_withdrawn",
        "sessions_kind",
        "sessions_updated",
        "sessions_workspace",
        "events_turn",
        "events_session",
    ):
        assert name in names, name
    turn_columns = {row[1]: row for row in rows(path, "PRAGMA table_info(turns)")}
    assert turn_columns["note"][4] == "''"
    assert turn_columns["typed_text"][4] == "''"
    session_columns = {row[1]: row for row in rows(path, "PRAGMA table_info(sessions)")}
    assert session_columns["kind"][4] == "'chat'"
    window_columns = {row[1]: row for row in rows(path, "PRAGMA table_info(windows)")}
    assert window_columns["thinking_since"][3] == 0  # nullable


def _sessions_file(path: Path, writer: str = "gateway", version: int = 1) -> None:
    """A sessions file made by hand, with a chosen marker and version."""
    conn = sqlite3.connect(path)
    for statement in store_mod._SCHEMA:
        conn.execute(statement)
    if writer:
        conn.execute("INSERT INTO store_info VALUES ('writer', ?)", (writer,))
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("newer", "has schema version 2; this localmelo reads version 1"),
        ("no_marker", "has no valid writer marker"),
        ("other_marker", "has no valid writer marker"),
        ("history_db", "is not a localmelo sessions file"),
    ],
)
async def test_a_refused_file_stays_unchanged(
    tmp_path: Path, case: str, message: str
) -> None:
    path = tmp_path / "sessions.db"
    if case == "newer":
        _sessions_file(path, version=2)
    elif case == "no_marker":
        _sessions_file(path, writer="")
    elif case == "other_marker":
        _sessions_file(path, writer="other")
    else:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()
    before = file_bytes(path)
    for writer in ("gateway", "cli"):
        store = SessionStore(path, writer=writer)  # type: ignore[arg-type]
        with pytest.raises(SessionStoreError) as raised:
            await store.open()
        assert str(raised.value) == f"{path} {message}"
        await store.aclose()
    assert file_bytes(path) == before
    assert not Path(f"{path}.owner").exists()


# A writer that dies with its WAL frames not yet checkpointed into the file.
LEAVE_WAL = textwrap.dedent("""
    import asyncio
    import os
    import sys

    from localmelo.support.sessions.store import NewSession, SessionStore


    async def main() -> None:
        store = SessionStore(sys.argv[1], writer=sys.argv[2])
        await store.begin_turn(
            None, "hello", holder="h:1:aaaaaaaa", new_session=NewSession()
        )
        os._exit(0)


    asyncio.run(main())
    """)


@pytest.mark.parametrize("writer", ["gateway", "cli"])
async def test_a_refused_wal_file_keeps_its_bytes_and_its_wal(
    tmp_path: Path, writer: str
) -> None:
    path = tmp_path / "sessions.db"
    other = "cli" if writer == "gateway" else "gateway"
    made = subprocess.run(
        [sys.executable, "-I", "-c", LEAVE_WAL, str(path), other],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert made.returncode == 0, made.stderr
    wal = Path(f"{path}-wal")
    assert wal.stat().st_size > 0  # frames the refusing open must not checkpoint
    before, wal_before = file_bytes(path), file_bytes(wal)
    store = SessionStore(path, writer=writer)  # type: ignore[arg-type]
    with pytest.raises(SessionStoreError, match="belongs to the"):
        await store.open()
    await store.aclose()
    assert file_bytes(path) == before
    assert file_bytes(wal) == wal_before
    if writer == "gateway":
        assert not Path(f"{path}.owner").exists()


async def test_a_foreign_rollback_journal_file_is_left_byte_for_byte(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE notes (body TEXT)")
    conn.execute("INSERT INTO notes VALUES ('keep me')")
    conn.commit()
    conn.close()
    before = file_bytes(path)
    for writer in ("gateway", "cli"):
        store = SessionStore(path, writer=writer)  # type: ignore[arg-type]
        with pytest.raises(SessionStoreError) as raised:
            await store.open()
        assert str(raised.value) == f"{path} is not a localmelo sessions file"
    assert file_bytes(path) == before
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
    assert not Path(f"{path}.owner").exists()


async def test_a_file_that_is_not_a_database_is_refused_and_can_be_retried(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    path.write_bytes(os.urandom(4096))
    store = cli(path)
    with pytest.raises(SessionStoreError) as raised:
        await store.open()
    assert str(raised.value) == f"{path} is not a localmelo sessions file"
    assert isinstance(raised.value.__cause__, sqlite3.DatabaseError)
    path.unlink()
    await store.open()  # the next call tries again
    assert (await store.create_session()).kind == "chat"
    await store.aclose()


async def test_an_unusable_directory_raises_and_can_be_retried(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should be")
    path = blocker / "sessions.db"
    store = gateway(path)
    with pytest.raises(SessionStoreError) as raised:
        await store.open()
    assert str(raised.value).startswith(f"Cannot open {path}: ")
    assert isinstance(raised.value.__cause__, OSError)
    assert store._owner_fd is None
    blocker.unlink()
    await store.open()
    await store.aclose()


async def test_a_permission_error_raises_chained_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As a directory without write permission refuses the new file, also
    # for a test that runs as root.
    path = tmp_path / "sessions.db"
    store = cli(path)
    with monkeypatch.context() as m:
        real = os.open

        def denied(target: Any, *args: Any, **kwargs: Any) -> int:
            if os.fspath(target) == str(path):
                raise PermissionError(13, "Permission denied", str(target))
            return real(target, *args, **kwargs)

        m.setattr(os, "open", denied)
        with pytest.raises(SessionStoreError) as raised:
            await store.open()
    assert str(raised.value).startswith(f"Cannot open {path}: ")
    assert isinstance(raised.value.__cause__, PermissionError)
    assert not path.exists() and store._conn is None
    await store.open()  # the next call tries again
    await store.aclose()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into any directory")
async def test_a_directory_without_write_permission_raises(tmp_path: Path) -> None:
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        store = cli(folder / "sessions.db")
        with pytest.raises(SessionStoreError) as raised:
            await store.open()
        assert isinstance(raised.value.__cause__, OSError)
    finally:
        folder.chmod(0o700)
    await store.open()
    await store.aclose()


async def test_two_terminals_create_one_new_file_at_the_same_moment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sessions.db"
    both_checked = asyncio.Event()
    seen: list[bool] = []
    original = SessionStore._check_file

    async def check_then_wait(self: SessionStore, conn: Any) -> bool:
        new = await original(self, conn)
        seen.append(new)
        if len(seen) == 2:
            both_checked.set()
        await both_checked.wait()
        return new

    monkeypatch.setattr(SessionStore, "_check_file", check_then_wait)
    first, second = cli(path), cli(path)
    await asyncio.gather(first.open(), second.open())
    # Both found the file new; only one created the schema, the other found
    # version 1 inside its transaction and checked the marker instead.
    assert seen[:2] == [True, True]
    assert rows(path, "SELECT key, value FROM store_info WHERE key = 'writer'") == [
        ("writer", "cli")
    ]
    one = await first.create_session()
    assert (await second.get_session(one.id)) == one
    await first.aclose()
    await second.aclose()


async def test_aclose_is_final_and_leaves_a_running_turn_running(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    store = gateway(path)
    turn = await store.begin_turn(
        None, "hello", holder=HOLDER, new_session=NewSession()
    )
    await store.aclose()
    for call in (
        store.open(),
        store.finish_turn(
            turn.session_id,
            turn.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(),
            answer="done",
        ),
    ):
        with pytest.raises(SessionStoreError) as raised:
            await call
        assert str(raised.value) == "The session store is closed"
    assert store._conn is None
    assert rows(path, "SELECT status FROM turns") == [("running",)]
    await store.aclose()  # a second close does nothing
    other = gateway(path)  # the owner lock is free, and not taken again
    await other.open()
    await other.aclose()


async def test_an_aclose_during_open_stops_it_before_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sessions.db"
    made = gateway(path)
    await made.begin_turn(None, "hello", holder=HOLDER, new_session=NewSession())
    await made.aclose()  # a turn left running
    reached, gate = asyncio.Event(), asyncio.Event()
    check = SessionStore._check_file

    async def slow_check(self: SessionStore, conn: Any) -> bool:
        reached.set()
        await gate.wait()
        return await check(self, conn)

    monkeypatch.setattr(SessionStore, "_check_file", slow_check)
    store = gateway(path)
    opening = asyncio.create_task(store.open())
    await reached.wait()
    closing = asyncio.create_task(store.aclose())
    await asyncio.sleep(0.02)
    gate.set()
    with pytest.raises(SessionStoreError) as raised:
        await opening
    assert str(raised.value) == "The session store is closed"
    await closing
    # The open stopped at its next step: no recovery ran after the close.
    assert rows(path, "SELECT status FROM turns") == [("running",)]
    assert store._conn is None and store._owner_fd is None


@pytest.mark.parametrize("stage", ["connecting", "checking"])
async def test_a_cancelled_aclose_during_open_is_still_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    path = tmp_path / "sessions.db"
    reached, gate = asyncio.Event(), asyncio.Event()
    if stage == "connecting":
        connect = store_mod.aiosqlite.connect

        async def slow_connect(*args: Any, **kwargs: Any) -> Any:
            reached.set()
            await gate.wait()
            return await connect(*args, **kwargs)

        monkeypatch.setattr(store_mod.aiosqlite, "connect", slow_connect)
    else:
        check = SessionStore._check_file

        async def slow_check(self: SessionStore, conn: Any) -> bool:
            reached.set()
            await gate.wait()
            return await check(self, conn)

        monkeypatch.setattr(SessionStore, "_check_file", slow_check)
    store = gateway(path)
    opening = asyncio.create_task(store.open())
    await reached.wait()
    closing = asyncio.create_task(store.aclose())
    await asyncio.sleep(0.02)  # aclose waits for the open's lock
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    gate.set()
    with pytest.raises(SessionStoreError) as raised:
        await opening
    assert str(raised.value) == "The session store is closed"
    assert store._conn is None and store._owner_fd is None
    with pytest.raises(SessionStoreError, match="The session store is closed"):
        await store.open()
    other = gateway(path)  # nothing holds the owner lock
    await other.open()
    await other.aclose()


async def test_a_close_cancelled_at_shutdown_is_done_by_the_next_aclose(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    store = gateway(path)
    await store.open()
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")  # a write elsewhere: the store waits
    writing = asyncio.create_task(store.create_session())  # holds the lock
    await asyncio.sleep(0.2)
    closing = asyncio.create_task(store.aclose())
    await asyncio.sleep(0.05)
    background = store._closing
    assert background is not None and not background.done()
    for task in (writing, background, closing):  # as a loop's shutdown does
        task.cancel()
    await asyncio.gather(writing, background, closing, return_exceptions=True)
    blocker.execute("ROLLBACK")
    blocker.close()
    await store.aclose()  # closes now, and raises nothing
    assert store._conn is None and store._owner_fd is None
    other = gateway(path)
    await other.open()  # the owner lock is free
    await other.aclose()


@pytest.mark.parametrize("step", ["create", "apply_retention", "recover"])
async def test_an_open_that_fails_partway_leaves_the_file_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    path = tmp_path / "sessions.db"
    if step != "create":
        made = gateway(path)
        await made.begin_turn(None, "hello", holder=HOLDER, new_session=NewSession())
        await made.aclose()  # a turn left running, as by a process that died
    before = file_bytes(path) if path.exists() else None

    def disk_full(self: SessionStore, label: str) -> None:
        if label == step:
            raise sqlite3.OperationalError("database or disk is full")

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "_before_commit", disk_full)
        store = gateway(path)
        with pytest.raises(SessionStoreError) as raised:
            await store.open()
        assert str(raised.value) == f"Cannot open {path}: database or disk is full"
        assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
        assert store._conn is None and store._owner_fd is None
        if before is None:  # still a new file: no schema, no version
            assert rows(path, "PRAGMA user_version") == [(0,)]
            assert rows(path, "SELECT name FROM sqlite_master") == []
        else:
            assert file_bytes(path) == before
            assert rows(path, "SELECT status FROM turns") == [("running",)]
    await store.open()  # the retry sees the file as it was
    if before is not None:
        assert rows(path, "SELECT status, cause FROM turns") == [("crashed", "crash")]
    assert (await store.create_session()).kind == "chat"
    await store.aclose()


async def test_an_unknown_schema_component_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = gateway(path)
    await store.open()
    await store.aclose()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO schema_components VALUES ('scheduler', 7)")
    conn.commit()
    conn.close()
    again = gateway(path)
    await again.open()
    assert (await again.create_session()).kind == "chat"
    await again.aclose()


def test_the_session_event_keys_are_pinned() -> None:
    assert dict(SESSION_EVENT_FIELDS) == {
        "turn.started": {"turn", "turn_id", "user_text", "client_request_id"},
        "turn.finished": {"turn", "turn_id", "status", "error", "cause"},
        "turn.undone": {"turn", "turn_id"},
        "session.title": {"title"},
        "session.updated": {"model", "workspace", "meta"},
        "approval.requested": {
            "request_id",
            "call_id",
            "tool_name",
            "arguments",
            "preview",
            "workspace_root",
        },
        "approval.resolved": {
            "request_id",
            "call_id",
            "tool_name",
            "outcome",
            "origin",
            "reason",
            "remember",
        },
    }
    assert RESERVED_EVENT_PREFIXES == ("task.", "sched.", "attention.")


# ── Files and modes ─────────────────────────────────────────────────────────


def test_each_writer_has_its_default_path(home: Path) -> None:
    assert default_path("gateway") == str(
        home / ".cache" / "localmelo" / "gateway" / "sessions.db"
    )
    assert default_path("cli") == str(
        home / ".cache" / "localmelo" / "cli" / "sessions.db"
    )
    assert SessionStore(writer="cli").path == default_path("cli")
    with pytest.raises(ValueError):
        SessionStore(writer="web")  # type: ignore[arg-type]


async def test_missing_directories_are_created_private(home: Path) -> None:
    store = SessionStore(writer="gateway")
    assert not (home / ".cache").exists()  # nothing touches the disk yet
    await store.open()
    await store.aclose()
    for folder in (
        home / ".cache",
        home / ".cache" / "localmelo",
        home / ".cache" / "localmelo" / "gateway",
    ):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700, folder
    assert stat.S_IMODE(Path(default_path("gateway")).stat().st_mode) == 0o600


async def test_a_loose_default_directory_is_tightened(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = home / ".cache" / "localmelo" / "cli"
    folder.mkdir(parents=True)
    folder.chmod(0o755)
    caplog.set_level(logging.INFO, logger="localmelo.support.sessions.store")
    store = SessionStore(writer="cli")
    await store.open()
    await store.aclose()
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert any("0700" in record.getMessage() for record in caplog.records)


async def test_a_chosen_folder_keeps_its_mode_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = tmp_path / "shared"
    folder.mkdir()
    folder.chmod(0o755)
    caplog.set_level(logging.WARNING, logger="localmelo.support.sessions.store")
    store = cli(folder / "sessions.db")
    await store.open()
    await store.aclose()
    assert stat.S_IMODE(folder.stat().st_mode) == 0o755
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(folder) in warnings[0].getMessage()


@pytest.mark.parametrize(("writer", "synchronous"), [("gateway", 2), ("cli", 1)])
async def test_synchronous_and_journal_size_limit(
    tmp_path: Path, writer: str, synchronous: int
) -> None:
    store = SessionStore(tmp_path / "s.db", writer=writer)  # type: ignore[arg-type]
    await store.open()
    assert store._conn is not None
    assert list(await store._conn.execute_fetchall("PRAGMA synchronous")) == [
        (synchronous,)
    ]
    assert list(await store._conn.execute_fetchall("PRAGMA journal_size_limit")) == [
        (67108864,)
    ]
    assert list(await store._conn.execute_fetchall("PRAGMA journal_mode")) == [("wal",)]
    assert list(await store._conn.execute_fetchall("PRAGMA foreign_keys")) == [(1,)]
    await store.aclose()


# ── Writers ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("made_by", "opened_by", "owner"),
    [("gateway", "cli", "gateway"), ("cli", "gateway", "terminal melo")],
)
async def test_the_other_writers_file_is_refused(
    tmp_path: Path, made_by: str, opened_by: str, owner: str
) -> None:
    path = tmp_path / "sessions.db"
    maker = SessionStore(path, writer=made_by)  # type: ignore[arg-type]
    await maker.create_session()
    await maker.aclose()
    before = file_bytes(path)
    store = SessionStore(path, writer=opened_by)  # type: ignore[arg-type]
    with pytest.raises(SessionStoreError) as raised:
        await store.open()
    assert str(raised.value) == (
        f"{path} belongs to the {owner} (writer '{made_by}'); open it with "
        f"writer='{made_by}' or use the default path"
    )
    assert file_bytes(path) == before
    if made_by == "cli":
        assert not Path(f"{path}.owner").exists()


# ── Sessions ────────────────────────────────────────────────────────────────


async def test_sessions_are_created_listed_and_filtered(tmp_path: Path) -> None:
    clock = Clock()
    store = cli(tmp_path / "s.db", clock)
    first = await store.create_session(model="m1", workspace="/w/a", meta={"level": 2})
    clock.advance(1)
    second = await store.create_session(model="m2", workspace="/w/b")
    await store.rename_session(first.id, "50% off")
    clock.advance(1)
    await store.rename_session(second.id, "500 off")
    assert (await store.get_session(first.id)) is not None
    assert (await store.get_session("nope")) is None
    assert first.meta == {"level": 2} and first.kind == "chat" and not first.busy
    listed = await store.list_sessions()
    assert [info.id for info in listed] == [second.id, first.id]
    assert [i.id for i in await store.list_sessions(workspace="/w/a")] == [first.id]
    assert [i.id for i in await store.list_sessions(query="50%")] == [first.id]
    assert [i.id for i in await store.list_sessions(query="0_ off")] == []
    assert len(await store.list_sessions(limit=1)) == 1
    with pytest.raises(ValueError):
        await store.list_sessions(limit=0)
    await store.aclose()


async def test_a_query_matches_any_user_text(tmp_path: Path) -> None:
    store = cli(tmp_path / "s.db")
    session_id, _ = await finished_turn(store, text="find the needle here")
    await finished_turn(store, text="something else")
    assert [i.id for i in await store.list_sessions(query="needle")] == [session_id]
    await store.aclose()


async def test_kinds_keep_task_sessions_out_of_chat_lists(tmp_path: Path) -> None:
    store = gateway(tmp_path / "s.db")
    chat = await store.create_session()
    task = await store.create_session(kind="task")
    assert [i.id for i in await store.list_sessions()] == [chat.id]
    assert {i.id for i in await store.list_sessions(kinds=None)} == {chat.id, task.id}
    assert [i.id for i in await store.list_sessions(kinds=("task",))] == [task.id]
    assert [i.id for i in await store.find_sessions(task.id)] == [task.id]
    assert await store.find_sessions(task.id[:6]) == []
    assert [i.id for i in await store.find_sessions(task.id[:6], kinds=None)] == [
        task.id
    ]
    await store.rename_session(task.id, "Nightly job")
    assert await store.find_sessions("nightly job") == []  # nor by its title
    assert [i.id for i in await store.find_sessions("nightly job", kinds=None)] == [
        task.id
    ]
    with pytest.raises(ValueError):
        await store.create_session(kind="bogus")
    with pytest.raises(ValueError):
        await store.list_sessions(kinds=("bogus",))
    turn = await store.begin_turn(
        None, "hi", holder=HOLDER, new_session=NewSession(kind="front_bg")
    )
    assert (await store.get_session(turn.session_id)).kind == "front_bg"  # type: ignore[union-attr]
    await store.aclose()


async def test_find_sessions_prefers_id_then_prefix_then_title(tmp_path: Path) -> None:
    clock = Clock()
    store = cli(tmp_path / "s.db", clock)
    older = await store.create_session()
    clock.advance(1)
    newer = await store.create_session()
    await store.rename_session(older.id, "Daily Notes")
    clock.advance(1)
    await store.rename_session(newer.id, "daily notes")
    assert [i.id for i in await store.find_sessions(older.id)] == [older.id]
    assert [i.id for i in await store.find_sessions(newer.id[:5])] == [newer.id]
    assert await store.find_sessions(newer.id[:3]) == []  # a prefix needs 4 characters
    titled = await store.find_sessions("DAILY notes")
    assert [i.id for i in titled] == [newer.id, older.id]  # newest first
    # A title that looks like an id prefix still matches when no id does.
    await store.rename_session(older.id, "abcdef")
    assert [i.id for i in await store.find_sessions("ABCDEF")] == [older.id]
    assert await store.find_sessions("  ") == []
    await store.aclose()


async def test_an_id_prefix_matches_exactly_and_with_case(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = cli(path)
    made = await store.create_session()
    await store.aclose()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE sessions SET id = 'abcd_f123456' WHERE id = ?", (made.id,))
    conn.commit()
    conn.close()
    store = cli(path)
    assert [i.id for i in await store.find_sessions("abcd_f")] == ["abcd_f123456"]
    assert await store.find_sessions("ABCD_F") == []  # ids are lowercase hex
    await store.rename_session("abcd_f123456", "ABCD")
    titled = await store.find_sessions("ABCD")  # no id starts so: the title
    assert [i.id for i in titled] == ["abcd_f123456"]
    await store.aclose()


async def test_rename_validates_beats_the_automatic_title_and_writes_an_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = cli(path)
    info = await store.create_session()
    for bad in ("", "   ", "x" * 201):
        with pytest.raises(ValueError, match="title must be 1 to 200 characters"):
            await store.rename_session(info.id, bad)
    assert await store.rename_session(info.id, "  Mine  ")
    assert not await store.rename_session("nope", "x")
    await finished_turn(store, info.id, "a first message")
    assert (await store.get_session(info.id)).title == "Mine"  # type: ignore[union-attr]
    events = await store.list_events(info.id)
    assert events[0].event["type"] == "session.title"
    assert events[0].turn is None
    assert events[0].event["data"] == {"title": "Mine"}
    assert all(e.event["type"] != "session.title" for e in events[1:])
    await store.aclose()


async def test_the_first_title_comes_from_the_typed_text(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = cli(path)
    composed = "Summarize this\n\n<file name='notes.txt'>\n" + "line\n" * 50 + "</file>"
    turn = await store.begin_turn(
        None, composed, holder=HOLDER, title="Summarize this", new_session=NewSession()
    )
    info = await store.get_session(turn.session_id)
    assert info is not None and info.title == "Summarize this"
    assert rows(path, "SELECT typed_text FROM turns") == [("Summarize this",)]
    await store.finish_turn(
        turn.session_id,
        1,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="ok",
    )
    await store.begin_turn(turn.session_id, "next", holder=HOLDER, title="Other")
    assert (await store.get_session(turn.session_id)).title == "Summarize this"  # type: ignore[union-attr]
    assert rows(path, "SELECT typed_text FROM turns WHERE turn = 2") == [("Other",)]
    blank = await store.begin_turn(
        None,
        "  the   user\ntext  ",
        holder=HOLDER,
        title="  ",
        new_session=NewSession(),
    )
    assert (await store.get_session(blank.session_id)).title == "the user text"  # type: ignore[union-attr]
    assert rows(
        path, "SELECT typed_text FROM turns WHERE session_id = ?", blank.session_id
    ) == [("",)]
    await store.aclose()


def test_title_from_text_collapses_whitespace_and_keeps_60_characters() -> None:
    assert title_from_text("  a\n\tb  c ") == "a b c"
    long = title_from_text("word " * 30)
    assert len(long) == 60 and long.endswith("…")
    assert long[:59] == ("word " * 30).strip()[:59]


async def test_turn_counts_leave_out_running_and_undone_turns(tmp_path: Path) -> None:
    store = cli(tmp_path / "s.db")
    empty = await store.create_session()
    session_id, _ = await finished_turn(store)

    async def count() -> int:
        info = await store.get_session(session_id)
        assert info is not None
        # Every read computes it alike.
        listed = {i.id: i.turns for i in await store.list_sessions(kinds=None)}
        found = await store.find_sessions(session_id)
        assert listed[session_id] == found[0].turns == info.turns
        return info.turns

    assert await count() == 1
    await store.begin_turn(session_id, "second", holder=HOLDER)
    assert await count() == 1
    await store.finish_turn(
        session_id,
        2,
        holder=HOLDER,
        status="failed",
        cause="run",
        result=None,
        answer="x",
    )
    assert await count() == 2
    await store.undo_last_turn(session_id)
    assert await count() == 1
    await store.withdraw_submission(session_id, "k-1", user_text="never sent")
    assert await count() == 1
    assert [i.id for i in await store.list_sessions(min_turns=1)] == [session_id]
    assert {i.id for i in await store.list_sessions()} == {session_id, empty.id}
    await store.aclose()


async def test_update_session_changes_only_the_given_fields(tmp_path: Path) -> None:
    clock = Clock()
    store = cli(tmp_path / "s.db", clock)
    info = await store.create_session(model="m1", workspace="/w", meta={"a": 1})
    clock.advance(5)
    updated = await store.update_session(info.id, model="m2")
    assert (updated.model, updated.workspace, updated.meta) == ("m2", "/w", {"a": 1})
    assert updated.updated_at == info.updated_at + 5
    updated = await store.update_session(info.id, meta={"level": 3})
    assert updated.meta == {"level": 3} and updated.model == "m2"
    events = await store.list_events(info.id)
    assert [e.event["type"] for e in events] == ["session.updated", "session.updated"]
    assert events[-1].event["data"] == {
        "model": "m2",
        "workspace": "/w",
        "meta": {"level": 3},
    }
    assert events[-1].turn is None
    with pytest.raises(SessionStoreError, match="Unknown session: nope"):
        await store.update_session("nope", model="x")
    await store.aclose()


async def test_delete_cascades_to_everything_of_the_session(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = gateway(path)
    image = ImagePart.from_bytes(b"png bytes", "image/png")
    turn = await store.begin_turn(
        None, "hello", holder=HOLDER, client_request_id="k1", new_session=NewSession()
    )
    sid = turn.session_id
    await store.append_events(
        sid, 1, [core_event("run.started")], holder=HOLDER, images=[image]
    )
    await store.finish_turn(
        sid,
        1,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="done",
        window=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="done"),
        ],
    )
    await store.withdraw_submission(sid, "k2", user_text="other")
    keep = await store.create_session()
    assert await store.delete_session(sid)
    assert not await store.delete_session(sid)
    assert rows(path, "SELECT COUNT(*) FROM sessions WHERE id = ?", sid) == [(0,)]
    for table in ("turns", "events", "blobs", "submissions", "windows"):
        assert rows(
            path, f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", sid
        ) == [(0,)], table
    assert (await store.get_session(keep.id)) is not None
    await store.aclose()


async def test_delete_is_refused_while_a_turn_runs(tmp_path: Path) -> None:
    clock = Clock()
    store = cli(tmp_path / "s.db", clock)
    holder = f"{cli_host_id()}:{os.getpid()}:bbbbbbbb"  # this process: alive
    turn = await store.begin_turn(None, "hi", holder=holder, new_session=NewSession())
    with pytest.raises(SessionBusyError):
        await store.delete_session(turn.session_id)
    await store.aclose()


async def test_the_gateway_store_counts_its_running_turn_busy_after_a_sleep(
    tmp_path: Path,
) -> None:
    clock = Clock()
    store = gateway(tmp_path / "s.db", clock)
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    clock.advance(10 * LEASE_S)  # the Mac slept: the lease lapsed by wall clock
    info = await store.get_session(turn.session_id)
    assert info is not None and info.busy
    with pytest.raises(SessionBusyError):
        await store.delete_session(turn.session_id)
    with pytest.raises(SessionBusyError, match="is already running a turn"):
        await store.begin_turn(turn.session_id, "again", holder=HOLDER)
    await store.aclose()


# ── Turns and events ────────────────────────────────────────────────────────


async def test_turns_are_numbered_across_two_cli_stores(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    first, second = cli(path), cli(path)
    session_id, _ = await finished_turn(first)
    for store, number in ((second, 2), (first, 3), (second, 4)):
        assert (await finished_turn(store, session_id))[1] == number
    # Two terminals send at once: one turn starts, the other is busy.
    host = cli_host_id()
    holders = (f"{host}:{os.getpid()}:11111111", f"{host}:{os.getpid()}:22222222")
    results = await asyncio.gather(
        first.begin_turn(session_id, "a", holder=holders[0]),
        second.begin_turn(session_id, "b", holder=holders[1]),
        return_exceptions=True,
    )
    started = [r for r in results if not isinstance(r, BaseException)]
    busy = [r for r in results if isinstance(r, SessionBusyError)]
    assert len(started) == 1 and len(busy) == 1
    assert started[0].turn == 5
    assert "is running a turn in another process" in str(busy[0])
    await first.aclose()
    await second.aclose()


async def test_one_running_turn_per_session(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = cli(path)
    holder = f"{cli_host_id()}:{os.getpid()}:cccccccc"
    turn = await store.begin_turn(None, "hi", holder=holder, new_session=NewSession())
    with pytest.raises(SessionBusyError, match="is already running a turn"):
        await store.begin_turn(turn.session_id, "again", holder=holder)
    conn = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO turns (session_id, turn, id, user_text, holder, started_at) "
            "VALUES (?, 2, 'x', 'y', 'h', 0)",
            (turn.session_id,),
        )
    conn.close()
    await store.aclose()


async def test_a_new_session_and_its_turn_commit_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.db"
    store = cli(path)
    turn = await store.begin_turn(
        None, "hello", holder=HOLDER, client_request_id="k1", new_session=NewSession()
    )
    assert rows(path, "SELECT COUNT(*) FROM sessions") == [(1,)]
    with pytest.raises(ValueError, match="text must not be blank"):
        await store.begin_turn(None, "  ", holder=HOLDER, new_session=NewSession())
    with pytest.raises(SubmissionRepeated) as repeated:
        await store.begin_turn(
            None,
            "hello",
            holder=HOLDER,
            client_request_id="k1",
            new_session=NewSession(),
        )
    assert repeated.value.turn.id == turn.id

    def fail(self: SessionStore, label: str) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as m:
        m.setattr(SessionStore, "_before_commit", fail)
        with pytest.raises(sqlite3.OperationalError):
            await store.begin_turn(None, "x", holder=HOLDER, new_session=NewSession())
    assert rows(path, "SELECT COUNT(*) FROM sessions") == [(1,)]
    assert store.failed is None
    await store.aclose()


async def test_begin_turn_checks_its_arguments(tmp_path: Path) -> None:
    store = cli(tmp_path / "s.db")
    info = await store.create_session()
    with pytest.raises(ValueError):
        await store.begin_turn(None, "x", holder=HOLDER)
    with pytest.raises(ValueError):
        await store.begin_turn(info.id, "x", holder=HOLDER, new_session=NewSession())
    with pytest.raises(ValueError, match="text must be at most 100000 characters"):
        await store.begin_turn(info.id, "x" * 100_001, holder=HOLDER)
    with pytest.raises(ValueError, match="text must be valid UTF-8"):
        await store.begin_turn(info.id, "x\ud800", holder=HOLDER)
    with pytest.raises(ValueError, match="title must be valid UTF-8"):
        await store.begin_turn(info.id, "x", holder=HOLDER, title="y\ud800")
    with pytest.raises(ValueError, match="client_request_id must be 1 to 128"):
        await store.begin_turn(info.id, "x", holder=HOLDER, client_request_id="-x")
    with pytest.raises(SessionStoreError, match="Unknown session: nope"):
        await store.begin_turn("nope", "x", holder=HOLDER)
    assert (await store.begin_turn(info.id, "x" * 100_000, holder=HOLDER)).turn == 1
    await store.aclose()


async def test_latest_event_id_is_the_newest_stored_event(tmp_path: Path) -> None:
    store = gateway(tmp_path / "s.db")
    session_id, _ = await finished_turn(store)
    events = await store.list_events(session_id)
    assert await store.latest_event_id(session_id) == events[-1].id
    await store.rename_session(session_id, "named")
    renamed = await store.list_events(session_id)
    assert await store.latest_event_id(session_id) == renamed[-1].id
    assert await store.latest_event_id("0123456789ab") == 0
    await store.aclose()


async def test_event_ids_grow_and_page_by_turn(tmp_path: Path) -> None:
    store = gateway(tmp_path / "s.db")
    session_id, _ = await finished_turn(store)
    for _ in range(3):
        await finished_turn(store, session_id)
    await store.rename_session(session_id, "named")
    events = await store.list_events(session_id)
    ids = [e.id or 0 for e in events]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert [e.id for e in await store.list_events(session_id, after_id=ids[2])] == ids[
        3:
    ]
    assert len(await store.list_events(session_id, limit=2)) == 2
    for bad in (0, 1001):
        with pytest.raises(ValueError):
            await store.list_events(session_id, limit=bad)
    # Paging (UR-B).
    assert [t.turn for t in await store.list_turns(session_id)] == [1, 2, 3, 4]
    assert [t.turn for t in await store.list_turns(session_id, limit=2)] == [3, 4]
    assert [t.turn for t in await store.list_turns(session_id, limit=2, before=3)] == [
        1,
        2,
    ]
    first = await store.list_events(session_id, turns=(3, 3), limit=1)
    assert first[0].event["type"] == "turn.started"
    assert first[0].event["data"]["turn"] == 3
    paged = await store.list_events(session_id, turns=(1, 4))
    assert all(e.turn is not None for e in paged)
    assert any(e.turn is None for e in events)  # the rename's session.title
    await store.aclose()


async def test_append_events_sets_the_run_id_and_batches_at_most_200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = gateway(tmp_path / "s.db")
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    commits: list[str] = []
    monkeypatch.setattr(
        SessionStore, "_before_commit", lambda self, label: commits.append(label)
    )
    with pytest.raises(ValueError):
        await store.append_events(
            turn.session_id,
            1,
            [core_event("step.started", seq=i) for i in range(WRITER_BATCH_MAX + 1)],
            holder=HOLDER,
        )
    batch = [core_event("run.started", run_id="task-9")] + [
        core_event("step.started", run_id="task-9", seq=i + 2)
        for i in range(WRITER_BATCH_MAX - 1)
    ]
    stored = await store.append_events(turn.session_id, 1, batch, holder=HOLDER)
    assert len(stored) == WRITER_BATCH_MAX
    assert commits == ["append_events"]
    assert [s.event["seq"] for s in stored] == list(range(1, WRITER_BATCH_MAX + 1))
    info = (await store.list_turns(turn.session_id))[0]
    assert info.run_id == "task-9"
    await store.aclose()


async def test_append_events_refuses_events_it_must_not_store(tmp_path: Path) -> None:
    store = cli(tmp_path / "s.db")
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    bad: list[dict[str, Any]] = [
        {"type": "turn.finished", "data": {}},
        {"type": "foo.bar", "data": {}},
        {"type": "sched.held", "data": {}},  # a CLI file holds no scheduler events
        {**core_event("step.started"), "data": {"step": 1}},  # a key missing
        core_event("step.started", step=float("nan")),
    ]
    for event in bad:
        with pytest.raises(ValueError):
            await store.append_events(turn.session_id, 1, [event], holder=HOLDER)
    assert [e.event["type"] for e in await store.list_events(turn.session_id)] == [
        "turn.started",
        "session.title",
    ]
    await store.aclose()


async def test_listeners_run_after_the_commit(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = gateway(path)
    seen: list[tuple[int, bool]] = []

    def listener(event: StoredEvent) -> None:
        assert event.id is not None
        visible = rows(path, "SELECT COUNT(*) FROM events WHERE id = ?", event.id)
        seen.append((event.id, visible == [(1,)]))

    def broken(event: StoredEvent) -> None:
        raise RuntimeError("a broken listener")

    store.add_listener(broken)
    store.add_listener(listener)
    session_id, _ = await finished_turn(store)
    assert seen and all(visible for _, visible in seen)
    assert [i for i, _ in seen] == [e.id for e in await store.list_events(session_id)]
    await store.aclose()


async def test_a_read_waits_for_a_write_and_never_sees_a_rolled_back_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = gateway(tmp_path / "s.db")
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    inserted = asyncio.Event()
    original = SessionStore._insert_event

    async def insert_then_wait(self: SessionStore, *args: Any) -> None:
        await original(self, *args)
        if args[-1]["type"] == "step.started":
            inserted.set()
            await asyncio.Event().wait()  # inside the transaction, until cancelled

    monkeypatch.setattr(SessionStore, "_insert_event", insert_then_wait)
    writer = asyncio.create_task(
        store.append_events(
            turn.session_id, 1, [core_event("step.started")], holder=HOLDER
        )
    )
    await inserted.wait()
    reader = asyncio.create_task(store.list_events(turn.session_id))
    await asyncio.sleep(0.05)
    assert not reader.done()  # it waits for the store's lock
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer
    events = await reader
    assert [e.event["type"] for e in events] == ["turn.started", "session.title"]
    assert store._conn is not None and not store._conn.in_transaction
    assert store.failed is None  # a cancellation is not a write failure
    await store.aclose()


async def test_blobs_are_stored_once_per_digest(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = gateway(path)
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    image = ImagePart.from_bytes(b"\x89PNG data", "image/png")
    for seq in (1, 2):
        await store.append_events(
            turn.session_id,
            1,
            [core_event("step.started", seq=seq)],
            holder=HOLDER,
            images=[image, image.as_reference()],
        )
    assert rows(path, "SELECT COUNT(*) FROM blobs") == [(1,)]
    assert await store.get_blob(turn.session_id, image.sha256) == (
        "image/png",
        b"\x89PNG data",
    )
    assert await store.get_blob(turn.session_id, "0" * 64) is None
    await store.aclose()


async def test_a_turns_end_withdraws_the_approvals_left_open(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = cli(path)
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    await store.append_events(
        turn.session_id,
        1,
        [
            approval_event("approval.requested", "answered"),
            approval_event("approval.resolved", "answered"),
            approval_event("approval.requested", "lost"),
        ],
        holder=HOLDER,
    )
    # The batch that held the second approval.resolved was lost; T-8 commits.
    await store.finish_turn(
        turn.session_id,
        1,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="done",
    )
    events = [e.event for e in await store.list_events(turn.session_id)]
    assert [e["type"] for e in events][-2:] == ["approval.resolved", "turn.finished"]
    withdrawn = events[-2]["data"]
    assert withdrawn["request_id"] == "lost"
    assert (withdrawn["outcome"], withdrawn["origin"]) == ("withdrawn", "withdrawn")
    assert withdrawn["reason"] == store_mod.ENDED_APPROVAL_REASON
    assert (withdrawn["call_id"], withdrawn["remember"]) == ("call_1", False)
    check_invariants(path, "cli")  # S-3: every request answered exactly once
    await store.aclose()


async def test_an_unknown_session_reads_as_one_everywhere(tmp_path: Path) -> None:
    store = gateway(tmp_path / "s.db")
    with pytest.raises(SessionStoreError) as raised:
        await store.finish_turn(
            "nope",
            1,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer="",
        )
    assert str(raised.value) == "Unknown session: nope"
    with pytest.raises(SessionStoreError) as raised:
        await store.append_events(
            "nope", 1, [core_event("step.started")], holder=HOLDER
        )
    assert str(raised.value) == "Unknown session: nope"
    assert store.failed is None  # a caller's mistake, not a lost write
    await store.aclose()


@pytest.mark.parametrize("writer", ["gateway", "cli"])
async def test_the_busy_messages_name_the_holder(tmp_path: Path, writer: str) -> None:
    store = SessionStore(tmp_path / "s.db", writer=writer)  # type: ignore[arg-type]
    holder = f"{cli_host_id()}:{os.getpid()}:bbbbbbbb"  # this process: alive
    turn = await store.begin_turn(None, "hi", holder=holder, new_session=NewSession())
    with pytest.raises(SessionBusyError) as raised:
        await store.begin_turn(turn.session_id, "again", holder=holder)
    assert str(raised.value) == f"Session {turn.session_id} is already running a turn"
    with pytest.raises(SessionBusyError) as raised:
        await store.begin_turn(turn.session_id, "mine", holder=HOLDER)
    assert str(raised.value) == (
        f"Session {turn.session_id} is running a turn in another process ({holder})"
    )
    await store.aclose()


# ── Out-of-turn events ──────────────────────────────────────────────────────


async def test_out_of_turn_events_on_the_gateway_store(tmp_path: Path) -> None:
    clock = Clock()
    store = gateway(tmp_path / "s.db", clock)
    session_id, _ = await finished_turn(store)
    before = await store.get_session(session_id)
    heard: list[StoredEvent] = []
    store.add_listener(heard.append)
    clock.advance(10)
    stored = await store.append_session_event(
        session_id, "task.progress", {"done": 3, "of": float("inf")}
    )
    assert stored.turn is None
    assert stored.event["data"] == {"done": 3, "of": "inf"}
    assert stored.event["seq"] is None and stored.event["run_id"] == ""
    assert heard == [stored]
    after = await store.get_session(session_id)
    assert after is not None and before is not None
    assert after.updated_at == before.updated_at
    for bad in ("turn.started", "run.finished", "foo.bar", "task", "task.X"):
        with pytest.raises(ValueError):
            await store.append_session_event(session_id, bad, {})
    assert stored not in await store.list_events(session_id, turns=(1, 1))
    assert stored in await store.list_events(session_id)
    with pytest.raises(SessionStoreError, match="Unknown session"):
        await store.append_session_event("nope", "task.progress", {})
    await store.aclose()
    terminal = cli(tmp_path / "c.db")
    info = await terminal.create_session()
    with pytest.raises(
        SessionStoreError, match="Out-of-turn events need the gateway's"
    ):
        await terminal.append_session_event(info.id, "task.progress", {})
    await terminal.aclose()


# ── Leases (CLI file) ───────────────────────────────────────────────────────


async def test_a_live_lease_of_another_holder_is_busy(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "s.db"
    first, second = cli(path, clock), cli(path, clock)
    remote = "otherhost:4242:dddddddd"  # another host: its lease_until decides
    turn = await first.begin_turn(None, "hi", holder=remote, new_session=NewSession())
    clock.advance(LEASE_S - 1)
    with pytest.raises(SessionBusyError, match=r"another process \(otherhost:4242"):
        await second.begin_turn(turn.session_id, "mine", holder=HOLDER)
    info = await second.get_session(turn.session_id)
    assert info is not None and info.busy
    await first.aclose()
    await second.aclose()


async def test_an_expired_lease_is_taken_over_and_its_turn_recovered(
    tmp_path: Path,
) -> None:
    clock = Clock()
    path = tmp_path / "s.db"
    first, second = cli(path, clock), cli(path, clock)
    remote = "otherhost:4242:dddddddd"
    turn = await first.begin_turn(None, "hi", holder=remote, new_session=NewSession())
    clock.advance(LEASE_S + 1)
    info = await second.get_session(turn.session_id)
    assert info is not None and not info.busy
    new = await second.begin_turn(turn.session_id, "mine", holder=HOLDER)
    assert new.turn == 2
    # Another host's lease that expired: a takeover (revision 48).
    old = (await second.list_turns(turn.session_id))[0]
    assert (old.status, old.cause) == ("cancelled", "takeover")
    assert old.answer == answer_text("cancelled", "") and old.result is None
    finished = [
        e.event for e in await second.list_events(turn.session_id) if e.turn == 1
    ][-1]
    assert finished["type"] == "turn.finished"
    assert finished["data"]["error"] == store_mod.TAKEOVER_ERROR
    assert finished["data"]["error"] == "The session was taken over by another process"
    # The old holder lost the lease; refreshing it says so.
    assert not await first.refresh_lease(turn.session_id, holder=remote)
    assert await first.refresh_lease(turn.session_id, holder=HOLDER)
    await first.aclose()
    await second.aclose()


async def test_a_stored_run_finished_decides_also_under_a_takeover(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    first = cli(path, clock)
    remote = "otherhost:4242:dddddddd"
    turn = await first.begin_turn(None, "hi", holder=remote, new_session=NewSession())
    await first.append_events(
        turn.session_id,
        1,
        [
            core_event("run.started"),
            core_event("run.finished", seq=2, result=run_result(result="done")),
        ],
        holder=remote,
    )
    await first.aclose()
    clock.advance(LEASE_S + 1)  # the other host's lease expired
    second = cli(path, clock)
    await second.open()
    (recovered,) = await second.list_turns(turn.session_id)
    assert (recovered.status, recovered.cause) == ("completed", "run")
    assert recovered.answer == "done" and recovered.result == run_result(result="done")
    check_invariants(path, "cli")
    await second.aclose()


async def test_a_dead_pid_on_this_host_is_taken_over_at_once(tmp_path: Path) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    holder = f"{cli_host_id()}:{dead.pid}:eeeeeeee"
    store = cli(tmp_path / "s.db")
    turn = await store.begin_turn(None, "hi", holder=holder, new_session=NewSession())
    info = await store.get_session(turn.session_id)
    assert info is not None and not info.busy  # lease_until is still ahead
    assert (await store.begin_turn(turn.session_id, "mine", holder=HOLDER)).turn == 2
    # The old holder ran on this host: its process died, so a crash.
    old = (await store.list_turns(turn.session_id))[0]
    assert (old.status, old.cause) == ("crashed", "crash")
    assert old.answer == answer_text("crashed", "")
    finished = [
        e.event for e in await store.list_events(turn.session_id) if e.turn == 1
    ][-1]
    assert finished["type"] == "turn.finished"
    assert finished["data"]["cause"] == "crash"
    assert finished["data"]["error"] == store_mod.STALE_ERROR
    check_invariants(tmp_path / "s.db", "cli")
    await store.aclose()


async def test_a_live_pid_on_this_host_keeps_its_lease_past_lease_until(
    tmp_path: Path,
) -> None:
    clock = Clock()
    store = cli(tmp_path / "s.db", clock)
    holder = f"{cli_host_id()}:{os.getpid()}:ffffffff"
    turn = await store.begin_turn(None, "hi", holder=holder, new_session=NewSession())
    clock.advance(10 * LEASE_S)  # a long lid close
    with pytest.raises(SessionBusyError):
        await store.begin_turn(turn.session_id, "mine", holder=HOLDER)
    assert await store.recover_stale_turns() == 0
    await store.aclose()


async def test_refresh_lease_after_recovery_and_after_a_lapse(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "s.db"
    store = cli(path, clock)
    remote = "otherhost:4242:dddddddd"
    turn = await store.begin_turn(None, "hi", holder=remote, new_session=NewSession())
    clock.advance(LEASE_S + 5)  # lapsed, and nobody took it over
    assert await store.refresh_lease(turn.session_id, holder=remote)
    assert rows(path, "SELECT lease_until FROM sessions") == [(clock.now + LEASE_S,)]
    clock.advance(LEASE_S + 5)
    assert await store.recover_stale_turns(turn.session_id) == 1
    assert not await store.refresh_lease(turn.session_id, holder=remote)
    await store.aclose()


async def test_a_lease_stays_live_while_a_turn_waits_three_leases(
    tmp_path: Path,
) -> None:
    clock = Clock()
    path = tmp_path / "s.db"
    store, other = cli(path, clock), cli(path, clock)
    remote = "otherhost:4242:dddddddd"  # lease_until alone decides its liveness
    turn = await store.begin_turn(None, "wait", holder=remote, new_session=NewSession())
    approved = asyncio.Event()
    run = asyncio.create_task(approved.wait())  # the run, waiting on a person
    for _ in range(9):  # 3 × LEASE_S, refreshed every LEASE_S / 3
        clock.advance(LEASE_S / 3)
        assert await store.refresh_lease(turn.session_id, holder=remote)
        info = await other.get_session(turn.session_id)
        assert info is not None and info.busy
        with pytest.raises(SessionBusyError):
            await other.begin_turn(turn.session_id, "mine", holder=HOLDER)
        assert await other.recover_stale_turns() == 0
    approved.set()
    await run
    finished = await store.finish_turn(
        turn.session_id,
        1,
        holder=remote,
        status="completed",
        cause="run",
        result=None,
        answer="ok",
    )
    assert finished.status == "completed"
    await store.aclose()
    await other.aclose()


# ── Cancelled writes ────────────────────────────────────────────────────────


@pytest.mark.parametrize("write", ["refresh_lease", "begin_turn"])
async def test_a_cancelled_write_leaves_no_transaction_open(
    tmp_path: Path, write: str
) -> None:
    path = tmp_path / "s.db"
    store = cli(path)
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    if write == "begin_turn":
        await store.finish_turn(
            turn.session_id,
            1,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer="ok",
        )
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")  # holds the write lock
    task: asyncio.Task[Any]
    if write == "refresh_lease":
        task = asyncio.create_task(store.refresh_lease(turn.session_id, holder=HOLDER))
    else:
        task = asyncio.create_task(
            store.begin_turn(turn.session_id, "x", holder=HOLDER)
        )
    await asyncio.sleep(0.2)  # waiting in busy_timeout
    task.cancel()
    await asyncio.sleep(0.05)
    blocker.execute("ROLLBACK")
    blocker.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    await store.get_session(turn.session_id)  # queued behind the rollback
    assert store._conn is not None and not store._conn.in_transaction
    if write == "begin_turn":
        assert [t.turn for t in await store.list_turns(turn.session_id)] == [1]
        assert (await store.begin_turn(turn.session_id, "y", holder=HOLDER)).turn == 2
    else:
        assert await store.refresh_lease(turn.session_id, holder=HOLDER)
    await store.aclose()


async def test_a_failed_rollback_never_hides_a_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="localmelo.support.sessions.store")
    path = tmp_path / "s.db"
    store = cli(path)
    session_id, _ = await finished_turn(store)
    conn = store._conn
    assert conn is not None
    rollback = conn.rollback
    failures = 0

    async def failing_rollback() -> None:
        nonlocal failures
        failures += 1
        await rollback()
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as m:
        m.setattr(conn, "rollback", failing_rollback)
        blocker = sqlite3.connect(path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")  # holds the write lock
        task = asyncio.create_task(store.begin_turn(session_id, "x", holder=HOLDER))
        await asyncio.sleep(0.2)  # waiting in busy_timeout
        task.cancel()
        await asyncio.sleep(0.05)
        blocker.execute("ROLLBACK")
        blocker.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert failures == 1
        assert any("disk I/O error" in r.getMessage() for r in caplog.records)
    assert (await store.begin_turn(session_id, "y", holder=HOLDER)).turn == 2
    await store.aclose()


# ── Retention (UR-M) ────────────────────────────────────────────────────────


async def _old_and_new_rows(store: SessionStore, clock: Clock) -> dict[str, Any]:
    """Rows written 31 days ago and 1 day ago, with thinking and <think> blocks."""
    block = "<think>\nprivate plan\n</think>\n\nThe answer."
    made: dict[str, Any] = {}
    for age in ("old", "new"):
        ts = clock.now
        turn = await store.begin_turn(
            None, f"{age} q", holder=HOLDER, new_session=NewSession()
        )
        sid = turn.session_id
        await store.append_events(
            sid,
            1,
            [
                core_event(
                    "assistant.message", ts=ts, seq=1, content=block, thinking="hmm"
                ),
                core_event(
                    "assistant.message", ts=ts, seq=2, content="plain", thinking=""
                ),
                core_event(
                    "run.finished", ts=ts, seq=3, result=run_result(result=block)
                ),
            ],
            holder=HOLDER,
        )
        await store.finish_turn(
            sid,
            1,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(result=block),
            answer=block,
            window=[
                Message(role="user", content="q", timestamp=ts),
                Message(role="assistant", content="a", timestamp=ts),
            ],
        )
        plain = await store.begin_turn(sid, "plain q", holder=HOLDER)
        await store.finish_turn(
            sid,
            plain.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=run_result(result="no block"),
            answer="no block",
        )
        await store.withdraw_submission(sid, f"w-{age}", user_text="never")
        accepted = await store.begin_turn(
            sid, "keyed", holder=HOLDER, client_request_id=f"a-{age}"
        )
        await store.finish_turn(
            sid,
            accepted.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer="k",
        )
        made[age] = sid
        clock.advance(30 * DAY)
    clock.advance(-29 * DAY)  # now: the old rows are 31 days old, the new ones 1
    return made


def _check_retained(path: Path, made: dict[str, Any]) -> None:
    for age, sid in made.items():
        events = rows(
            path,
            "SELECT type, event FROM events WHERE session_id = ? AND type IN "
            "('assistant.message', 'run.finished') ORDER BY id",
            sid,
        )
        first, second, finished = (json.loads(e[1]) for e in events)
        answer = rows(
            path,
            "SELECT answer, result FROM turns WHERE session_id = ? AND turn = 1",
            sid,
        )[0]
        if age == "old":
            assert first["data"]["thinking"] == ""
            assert first["data"]["content"] == "The answer."
            assert set(first["data"]) == set(second["data"])  # other keys stay
            assert first["data"]["usage"] == {} and first["seq"] == 1
            assert finished["data"]["result"]["result"] == "The answer."
            assert answer[0] == answer_text("completed", "The answer.")
            assert json.loads(answer[1])["result"] == "The answer."
            assert rows(
                path, "SELECT COUNT(*) FROM windows WHERE session_id = ?", sid
            ) == [(0,)]
            assert rows(
                path,
                "SELECT COUNT(*) FROM submissions WHERE session_id = ? AND state = 'withdrawn'",
                sid,
            ) == [(0,)]
        else:  # newer copies are unchanged
            block = "<think>\nprivate plan\n</think>\n\nThe answer."
            assert first["data"]["thinking"] == "hmm"
            assert first["data"]["content"] == block
            assert finished["data"]["result"]["result"] == block
            assert answer[0] == block
            assert json.loads(answer[1])["result"] == block
            (data,) = rows(path, "SELECT data FROM windows WHERE session_id = ?", sid)
            decoded = decode_window(data[0])
            assert decoded is not None
            assert [(m.role, m.content) for m in decoded] == [
                ("user", "q"),
                ("assistant", "a"),
            ]
            assert rows(
                path,
                "SELECT client_request_id FROM submissions WHERE session_id = ? "
                "AND state = 'withdrawn'",
                sid,
            ) == [("w-new",)]
        assert second["data"]["content"] == "plain"
        assert rows(
            path, "SELECT answer FROM turns WHERE session_id = ? AND turn = 2", sid
        ) == [("no block",)]
        assert rows(
            path,
            "SELECT client_request_id FROM submissions WHERE session_id = ? "
            "AND state = 'accepted'",
            sid,
        ) == [(f"a-{age}",)]


async def test_retention_at_open_clears_old_thinking(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    store = gateway(path, clock)
    made = await _old_and_new_rows(store, clock)
    await store.aclose()
    reopened = gateway(path, clock)
    await reopened.open()
    await reopened.aclose()
    _check_retained(path, made)


def _first_thinking(path: Path, sid: str) -> str:
    """The thinking of a session's first assistant.message event, as stored."""
    (event,) = rows(
        path,
        "SELECT event FROM events WHERE session_id = ? "
        "AND type = 'assistant.message' ORDER BY id LIMIT 1",
        sid,
    )[0]
    return str(json.loads(event)["data"]["thinking"])


async def test_the_retention_variable_sets_the_stores_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One value for history.db, its model calls and sessions.db (R1): the
    # store's default is the core's, LOCALMELO_THINKING_RETENTION_DAYS
    # included.
    assert store_mod.THINKING_RETENTION_DAYS is core_sqlite.THINKING_RETENTION_DAYS
    path = tmp_path / "s.db"
    clock = Clock()
    store = gateway(path, clock)
    assert store._retention_s == 30 * DAY
    made = await _old_and_new_rows(store, clock)  # 31 days and 1 day old
    await store.aclose()

    # 40 days: the 31-day-old thinking stays.
    monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "40")
    longer = gateway(path, clock)
    assert longer._retention_s == 40 * DAY
    await longer.open()
    await longer.aclose()
    assert _first_thinking(path, made["old"]) == "hmm"

    # Half a day: the 1-day-old thinking goes too.
    monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "0.5")
    shorter = gateway(path, clock)
    assert shorter._retention_s == 0.5 * DAY
    await shorter.open()
    await shorter.aclose()
    assert [_first_thinking(path, sid) for sid in made.values()] == ["", ""]

    # An explicit value still wins; an invalid variable names itself.
    explicit = SessionStore(
        path, writer="gateway", clock=clock, thinking_retention_days=30
    )
    assert explicit._retention_s == 30 * DAY
    monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "abc")
    with pytest.raises(ValueError, match="LOCALMELO_THINKING_RETENTION_DAYS"):
        gateway(path, clock)


async def test_apply_retention_on_an_open_store_returns_its_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    store = gateway(path, clock)
    made = await _old_and_new_rows(store, clock)
    # Old: 1 assistant.message, 1 run.finished, 1 turn, 1 window, 1 withdrawal.
    assert await store.apply_retention() == 5
    _check_retained(path, made)
    assert await store.apply_retention() == 0
    await store.aclose()


async def test_apply_retention_counts_only_the_rows_it_changes(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    store = gateway(path, clock)
    turn = await store.begin_turn(None, "q", holder=HOLDER, new_session=NewSession())
    # "think>" without a block: the LIKE prefilter finds them, nothing changes.
    text = "I think> this is fine"
    await store.append_events(
        turn.session_id,
        1,
        [
            core_event("assistant.message", content=text, thinking=""),
            core_event("run.finished", seq=2, result=run_result(result=text)),
        ],
        holder=HOLDER,
    )
    await store.finish_turn(
        turn.session_id,
        1,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(result=text),
        answer=text,
    )
    before = rows(path, "SELECT id, event FROM events ORDER BY id")
    clock.advance(31 * DAY)
    assert await store.apply_retention() == 0
    assert rows(path, "SELECT id, event FROM events ORDER BY id") == before
    await store.aclose()


async def test_retention_clears_checkpointed_thinking_by_message_age(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    clock = Clock()
    store = gateway(path, clock)
    now = clock.now
    old = Message(
        role="assistant",
        content="<think>old plan</think>old answer",
        thinking="old thoughts",
        timestamp=now - 31 * DAY,
    )
    new = Message(
        role="assistant",
        content="<think>new plan</think>new answer",
        thinking="new thoughts",
        timestamp=now - DAY,
    )
    window = [
        Message(role="user", content="q1", timestamp=now - 31 * DAY),
        old,
        Message(role="user", content="q2", timestamp=now - DAY),
        new,
    ]
    sid, _ = await finished_turn(store, window=window)
    stored = await store.get_window(sid)
    assert stored is not None
    decoded = decode_window(stored.data)
    assert decoded is not None and len(decoded) == 4
    assert (decoded[1].thinking, decoded[1].content) == ("", "old answer")
    assert (decoded[3].thinking, decoded[3].content) == (
        "new thoughts",
        "<think>new plan</think>new answer",
    )
    assert rows(path, "SELECT thinking_since FROM windows") == [(now - DAY,)]
    # Later, the newer message's thinking expires too, in a row a recent turn wrote.
    clock.advance(29.5 * DAY)
    assert await store.apply_retention() == 1
    stored = await store.get_window(sid)
    assert stored is not None
    decoded = decode_window(stored.data)
    assert decoded is not None and len(decoded) == 4
    assert (decoded[3].thinking, decoded[3].content) == ("", "new answer")
    assert rows(path, "SELECT thinking_since FROM windows") == [(None,)]
    await store.aclose()


# ── Redaction ───────────────────────────────────────────────────────────────

SECRET = "sk-" + "a1b2c3d4e5" * 3
CODE_TEXT = f"Your key is {SECRET} and the verification code is 482913."


async def test_the_final_answers_stored_copies_are_redacted(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    store = gateway(path)
    turn = await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())
    result = run_result(result=CODE_TEXT)
    event = core_event("run.finished", result=result)
    stored = await store.append_events(turn.session_id, 1, [event], holder=HOLDER)
    expected = "Your key is [redacted] and the verification code is [redacted]."
    assert stored[0].event["data"]["result"]["result"] == expected
    on_disk = json.loads(
        rows(path, "SELECT event FROM events WHERE type = 'run.finished'")[0][0]
    )
    assert on_disk["data"]["result"]["result"] == expected
    info = await store.finish_turn(
        turn.session_id,
        1,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=result,
        answer=CODE_TEXT,
    )
    assert info.answer == expected
    assert info.result is not None and info.result["result"] == expected
    assert rows(path, "SELECT answer FROM turns") == [(expected,)]
    # The caller's mappings keep the text.
    assert result["result"] == CODE_TEXT
    assert event["data"]["result"]["result"] == CODE_TEXT
    await store.aclose()


# ── Windows ─────────────────────────────────────────────────────────────────


def _window(text: str) -> list[Message]:
    return [Message(role="user", content=text), Message(role="assistant", content="ok")]


async def test_window_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = gateway(tmp_path / "s.db")
    sid, _ = await finished_turn(store, window=_window("one"))
    first = await store.get_window(sid)
    assert first is not None and first.turn == 1 and first.version == 1
    await finished_turn(store, sid, window=_window("two"))
    second = await store.get_window(sid)
    assert second is not None and second.turn == 2
    assert decode_window(second.data)[0].content == "two"  # type: ignore[index]
    # A late finish_turn stores none.
    turn = await store.begin_turn(sid, "three", holder=HOLDER)
    late = await store.finish_turn(
        sid,
        turn.turn,
        holder="someone:else:x",
        status="completed",
        cause="run",
        result=None,
        answer="x",
        window=_window("late"),
    )
    assert late.status == "running"
    assert (await store.get_window(sid)) == second
    # One over WINDOW_MAX_CHARS keeps the older row.
    with monkeypatch.context() as m:
        m.setattr(store_mod, "WINDOW_MAX_CHARS", 50)
        await store.finish_turn(
            sid,
            turn.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer="x",
            window=_window("x" * 100),
        )
        assert (await store.get_window(sid)) == second
    # Undo deletes the row that covers the undone turn, and keeps an older one.
    undone = await store.undo_last_turn(sid)
    assert undone is not None and undone.turn == 3 and undone.undone
    assert (await store.get_window(sid)) == second
    undone = await store.undo_last_turn(sid)
    assert undone is not None and undone.turn == 2
    assert (await store.get_window(sid)) is None
    events = await store.list_events(sid)
    assert [e.event["type"] for e in events][-2:] == ["turn.undone", "turn.undone"]
    assert events[-1].turn is None and events[-1].event["data"]["turn"] == 2
    await store.aclose()


async def test_undo_refuses_while_a_turn_runs_and_returns_none_without_turns(
    tmp_path: Path,
) -> None:
    store = gateway(tmp_path / "s.db")
    info = await store.create_session()
    assert await store.undo_last_turn(info.id) is None
    await store.begin_turn(info.id, "x", holder=HOLDER)
    with pytest.raises(SessionBusyError):
        await store.undo_last_turn(info.id)
    with pytest.raises(SessionStoreError, match="Unknown session"):
        await store.undo_last_turn("nope")
    await store.aclose()


def test_split_thinking_rule_matches_the_store() -> None:
    # The store strips blocks by split_thinking's rule, lone closing tags too.
    from localmelo.support.sessions.window import without_think_block

    for text in ("<think>a</think>b", "reasoning</think>answer", "  <think>cut off"):
        assert without_think_block(text) == split_thinking(text)[1]
    assert without_think_block("no block <think> in the middle") == (
        "no block <think> in the middle"
    )
