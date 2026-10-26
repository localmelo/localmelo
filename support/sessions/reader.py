"""A read-only change feed of the turns in a sessions file, for sleep.

Sleep gets data, not features: :class:`SessionReader` reports one record
per ``turn.finished`` or ``turn.undone`` event, in event-id order, each
joined to its turn and session as they read now, and :func:`export_turns`
writes them as JSON lines. The reader keeps its cursor, the event id, on
its own side; it writes nothing into the file and opens it read-only, so it
reads either writer's file, also while the gateway owns it.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, TextIO, cast

import aiosqlite

from localmelo.support.sessions.store import (
    SCHEMA_VERSION,
    SessionStoreError,
    Writer,
)

READER_VERSION: Final = 1
# The kind of every exported record.
RECORD_KIND: Final = "localmelo.session_turn"
_MAX_LIMIT: Final = 5_000


@dataclass(frozen=True)
class TurnChange:
    """A turn that finished, or was undone, as the file reads now.

    ``event_id`` is the id of the ``turn.finished`` or ``turn.undone``
    event: the cursor. ``run_id`` is ``history.db``'s ``tasks.task_id``
    (``""`` when the run never started); ``answer`` is the stored copy,
    redacted and, after retention, without its ``<think>`` block.
    ``events`` holds the turn's stored events, in id order, with
    ``with_events``.
    """

    event_id: int
    change: Literal["finished", "undone"]
    writer: Writer
    session_id: str
    session_kind: str
    session_title: str
    workspace: str | None
    model: str
    turn: int
    turn_id: str
    run_id: str
    user_text: str
    typed_text: str
    status: str
    cause: str
    answer: str
    result: dict[str, Any] | None
    note: str
    undone: bool
    started_at: float
    finished_at: float
    events: tuple[dict[str, Any], ...] | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.events is not None:
            data["events"] = list(self.events)
        return {"kind": RECORD_KIND, "v": READER_VERSION, **data}


class SessionReader:
    """Reads a sessions file's change feed, read-only.

    It opens ``file:<path>?mode=ro`` with ``PRAGMA query_only``, takes no
    owner lock and no lease, and runs no migration, retention or recovery.
    Each call reads in one short read transaction and keeps no snapshot
    between calls, so it never holds up the writer's checkpoints.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        self._lock = asyncio.Lock()
        self._conn: aiosqlite.Connection | None = None
        self._closed = False

    @property
    def path(self) -> str:
        return self._path

    async def _ready(self) -> aiosqlite.Connection:
        if self._closed:
            raise SessionStoreError("The session reader is closed")
        if self._conn is not None:
            return self._conn
        uri = f"{Path(self._path).as_uri()}?mode=ro"
        try:
            conn = await aiosqlite.connect(uri, uri=True, isolation_level=None)
        except Exception as exc:  # sqlite3.Error, or OSError underneath
            raise SessionStoreError(f"Cannot open {self._path}: {exc}") from exc
        try:
            await conn.execute("PRAGMA query_only = ON")
            await conn.execute("PRAGMA busy_timeout = 5000")
        except BaseException:
            await conn.close()
            raise
        self._conn = conn
        return conn

    async def _check(self, conn: aiosqlite.Connection) -> Writer | None:
        """The file's writer; ``None`` for a version-0 file, which has no records.

        A version-0 file is a sessions file whose schema is not created yet,
        or another SQLite file; neither holds a turn. A newer version raises.
        """
        try:
            version = int(
                list(await conn.execute_fetchall("PRAGMA user_version"))[0][0]
            )
            tables = {
                row[0]
                for row in await conn.execute_fetchall(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        except Exception as exc:
            raise SessionStoreError(
                f"{self._path} is not a localmelo sessions file"
            ) from exc
        if version > SCHEMA_VERSION:
            raise SessionStoreError(
                f"{self._path} has schema version {version}; this localmelo reads "
                f"version {SCHEMA_VERSION}"
            )
        if version == 0:
            return None
        if version < 0:
            raise SessionStoreError(f"{self._path} is not a localmelo sessions file")
        rows = []
        if "store_info" in tables:
            rows = list(
                await conn.execute_fetchall(
                    "SELECT value FROM store_info WHERE key = 'writer'"
                )
            )
        if not rows or rows[0][0] not in ("gateway", "cli"):
            raise SessionStoreError(f"{self._path} has no valid writer marker")
        return cast(Writer, rows[0][0])

    async def writer(self) -> Writer | None:
        """The file's writer marker; ``None`` for a version-0 file (no records)."""
        async with self._lock:
            return await self._check(await self._ready())

    async def changes(
        self,
        *,
        after_event_id: int = 0,
        limit: int = 500,
        kinds: Collection[str] | None = None,
        with_events: bool = False,
    ) -> list[TurnChange]:
        """The changes after *after_event_id*, in event-id order, at most *limit*.

        *limit* is 1 to 5,000. *kinds* keeps the sessions of those kinds.
        A turn undone after its ``finished`` record shows up again as an
        ``undone`` record, so a reader that applies the feed in order is
        never wrong. Running turns have no record until their end.
        """
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _MAX_LIMIT
        ):
            raise ValueError(
                f"limit must be an int from 1 to {_MAX_LIMIT}, got {limit!r}"
            )
        if isinstance(kinds, str):
            raise ValueError("kinds must be a collection of session kinds, not a str")
        kind_sql = ""
        kind_args: tuple[str, ...] = ()
        if kinds is not None:
            chosen = tuple(kinds)
            kind_sql = (
                f"AND s.kind IN ({', '.join('?' * len(chosen))})" if chosen else "AND 0"
            )
            kind_args = chosen
        async with self._lock:
            conn = await self._ready()
            writer = await self._check(conn)
            if writer is None:
                return []
            await conn.execute("BEGIN")
            try:
                rows = list(
                    await conn.execute_fetchall(
                        "SELECT e.id, e.type, e.event, s.kind, s.title, s.workspace, "
                        "s.model, t.session_id, t.turn, t.id, t.run_id, t.user_text, "
                        "t.typed_text, t.status, t.cause, t.answer, t.result, t.note, "
                        "t.undone, t.started_at, t.finished_at FROM events e "
                        "JOIN turns t ON t.id = json_extract(e.event, '$.data.turn_id') "
                        "JOIN sessions s ON s.id = t.session_id "
                        "WHERE e.id > ? AND e.type IN ('turn.finished', 'turn.undone') "
                        f"{kind_sql} ORDER BY e.id LIMIT ?",
                        (after_event_id, *kind_args, limit),
                    )
                )
                changes = []
                for row in rows:
                    events = None
                    if with_events:
                        events = tuple(
                            json.loads(item[0])
                            for item in await conn.execute_fetchall(
                                "SELECT event FROM events WHERE session_id = ? "
                                "AND turn = ? ORDER BY id",
                                (row[7], row[8]),
                            )
                        )
                    finished_at = row[20]
                    if finished_at is None:
                        finished_at = float(json.loads(row[2]).get("ts") or 0.0)
                    changes.append(
                        TurnChange(
                            event_id=row[0],
                            change=(
                                "finished" if row[1] == "turn.finished" else "undone"
                            ),
                            writer=writer,
                            session_id=row[7],
                            session_kind=row[3],
                            session_title=row[4],
                            workspace=row[5],
                            model=row[6],
                            turn=row[8],
                            turn_id=row[9],
                            run_id=row[10],
                            user_text=row[11],
                            typed_text=row[12],
                            status=row[13],
                            cause=row[14],
                            answer=row[15],
                            result=None if row[16] is None else json.loads(row[16]),
                            note=row[17],
                            undone=bool(row[18]),
                            started_at=row[19],
                            finished_at=finished_at,
                            events=events,
                        )
                    )
            finally:
                await conn.rollback()  # a read transaction: nothing to commit
            return changes

    async def aclose(self) -> None:
        """Close the connection; idempotent, and final."""
        self._closed = True
        async with self._lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                await conn.close()


async def export_turns(
    paths: Sequence[str | os.PathLike[str]],
    out: TextIO,
    *,
    after: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Write every change after each path's cursor to *out*, one JSON line each.

    *after* maps a path, as given, to the event id it was exported up to.
    Returns each path's new cursor; the caller keeps the cursors.
    """
    cursors: dict[str, int] = {}
    for path in paths:
        key = os.fspath(path)
        cursor = (after or {}).get(key, 0)
        reader = SessionReader(path)
        try:
            while True:
                page = await reader.changes(after_event_id=cursor, limit=_MAX_LIMIT)
                for change in page:
                    out.write(json.dumps(change.to_dict(), ensure_ascii=False) + "\n")
                    cursor = change.event_id
                if len(page) < _MAX_LIMIT:
                    break
        finally:
            await reader.aclose()
        cursors[key] = cursor
    return cursors
