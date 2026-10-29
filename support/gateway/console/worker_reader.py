"""Read worker sessions from the CLI file without opening it for writing."""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from localmelo.support.sessions.store import (
    _SESSION_COLUMNS,
    SessionInfo,
    SessionStore,
    SessionStoreError,
    StoredEvent,
    TurnInfo,
    _escape_like,
    _rows,
)
from localmelo.support.sessions.worker import redact_worker

from .terminal_redaction import (
    redact_event,
    redact_info,
    redact_turn,
    sensitive_path,
)

# The stored kinds this reader serves: one-shot runs, and the terminal's chats.
READ_KINDS = ("worker", "chat")


@dataclass(frozen=True)
class WorkerRecord:
    info: SessionInfo
    started_at: float
    status: str | None
    steps: Any
    tokens: int | None
    duration_s: float | None


def _total_tokens(meta: dict[str, Any], result: dict[str, Any]) -> int | None:
    usage = meta.get("usage", meta.get("tokens", result.get("usage")))
    value = usage.get("total_tokens") if isinstance(usage, dict) else usage
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def encode_cursor(record: WorkerRecord) -> str:
    payload = json.dumps([record.started_at, record.info.id], separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(value: str | None) -> tuple[float, str] | None:
    if value is None:
        return None
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
        data = json.loads(decoded)
        if (
            not isinstance(data, list)
            or len(data) != 2
            or isinstance(data[0], bool)
            or not isinstance(data[0], (int, float))
            or not math.isfinite(data[0])
            or not isinstance(data[1], str)
            or not data[1]
            or len(data[1]) > 128
        ):
            raise ValueError
        return float(data[0]), data[1]
    except (ValueError, TypeError, OverflowError, binascii.Error, UnicodeDecodeError):
        raise ValueError("Invalid worker cursor.") from None


class WorkerReader(SessionStore):
    """Read-only CLI connection, with keyset paging over worker start times.

    It serves worker runs as they are stored and terminal chats (``kind=chat``)
    redacted on every read: their events are stored unredacted.
    """

    def __init__(self, path: str | None = None) -> None:
        super().__init__(path, writer="cli")

    async def _open_steps(self) -> None:
        uri = f"{Path(self.path).as_uri()}?mode=ro"
        conn = await aiosqlite.connect(uri, uri=True, isolation_level=None)
        self._conn = conn
        await conn.execute("PRAGMA query_only = ON")
        await conn.execute("PRAGMA busy_timeout = 5000")
        if await self._check_file(conn):
            raise SessionStoreError(f"{self.path} has no sessions schema")

    def _check_writable(self) -> None:
        raise SessionStoreError("The worker session reader is read-only")

    def _running_is_live(
        self, lease_holder: str | None, lease_until: float | None, now: float
    ) -> bool:
        # Do not call cli_host_id(): it can create a host-id file on a read.
        return (
            lease_holder is not None and lease_until is not None and lease_until >= now
        )

    async def workers(
        self,
        *,
        limit: int,
        query: str,
        status: str | None = None,
        source: str | None = None,
        parent_run: str | None = None,
        after: tuple[float, str] | None = None,
        session_id: str | None = None,
        kind: str = "worker",
    ) -> list[WorkerRecord]:
        """Rows of *kind*, each with the turn that stands for it.

        That turn is the first of a run and the latest of a chat, so a chat's
        ``status``, ``started_at``, ``steps`` and ``duration_s`` are its
        newest turn's, and the list orders chats by their last activity.
        """
        if kind not in READ_KINDS:
            raise ValueError(f"kind must be one of {READ_KINDS}, got {kind!r}")
        if not os.path.exists(self.path):
            return []
        clauses = ["s.kind = ?"]
        args: list[Any] = [kind]
        if session_id is not None:
            clauses.append("s.id = ?")
            args.append(session_id)
        if query:
            pattern = f"%{_escape_like(query)}%"
            clauses.append(
                "(s.title LIKE ? ESCAPE '\\' OR EXISTS (SELECT 1 FROM turns q WHERE q.session_id = s.id AND q.user_text LIKE ? ESCAPE '\\'))"
            )
            args.extend((pattern, pattern))
        if status is not None:
            clauses.append("COALESCE(json_extract(s.meta, '$.status'), t.status) = ?")
            args.append(status)
        for key, value in (("source", source), ("parent_run", parent_run)):
            if value is not None:
                clauses.append(f"json_extract(s.meta, '$.{key}') = ?")
                args.append(value)
        if after is not None:
            clauses.append("(t.started_at < ? OR (t.started_at = ? AND s.id < ?))")
            args.extend((after[0], after[0], after[1]))
        args.append(limit)
        sql = (
            f"SELECT {_SESSION_COLUMNS}, t.started_at, t.status, t.result "
            "FROM sessions s JOIN turns t ON t.session_id = s.id "
            f"AND t.turn = (SELECT {'MIN' if kind == 'worker' else 'MAX'}(first.turn) "
            "FROM turns first WHERE first.session_id = s.id) "
            f"WHERE {' AND '.join(clauses)} ORDER BY t.started_at DESC, s.id DESC LIMIT ?"
        )
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(conn, sql, args)
            now = self._clock()
            records = []
            for row in rows:
                info = self._session_info(row, now)
                if kind == "chat":
                    info = redact_info(info)
                result = json.loads(row[14]) if row[14] else {}
                if not isinstance(result, dict):
                    result = {}
                meta = info.meta
                duration = result.get("duration_s")
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (float, int))
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    duration = None
                records.append(
                    WorkerRecord(
                        info=info,
                        started_at=float(row[12]),
                        status=(
                            meta.get("status")
                            if meta.get("status") is not None
                            else row[13]
                        ),
                        steps=meta.get("steps", result.get("steps")),
                        tokens=_total_tokens(meta, result),
                        duration_s=duration,
                    )
                )
            return records

    async def running(self) -> list[tuple[str, str, str]]:
        """``(kind, id, title)`` of every worker run and terminal chat running now.

        One query over the index of running turns; a turn counts only under a
        live lease, so a run whose process died is not listed. Newest first.
        A terminal chat's title is redacted, as on every other read.
        """
        if not os.path.exists(self.path):
            return []
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                "SELECT s.kind, s.id, s.title, s.lease_holder, s.lease_until "
                "FROM turns t JOIN sessions s ON s.id = t.session_id "
                "WHERE t.status = 'running' AND s.kind IN ('worker', 'chat') "
                "ORDER BY t.started_at DESC, s.id DESC",
            )
            now = self._clock()
        return [
            (kind, sid, redact_worker(title) if kind == "chat" else title)
            for kind, sid, title, holder, until in rows
            if self._running_is_live(holder, until, now)
        ]

    async def get_session(self, session_id: str) -> SessionInfo | None:
        if not os.path.exists(self.path):
            return None
        info = await super().get_session(session_id)
        if info is None or info.kind not in READ_KINDS:
            return None
        return redact_info(info) if info.kind == "chat" else info

    async def _chat_workspace(self, session_id: str) -> tuple[bool, str | None]:
        """Whether the session is a terminal chat, and its workspace."""
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                "SELECT kind, workspace FROM sessions WHERE id = ?",
                (session_id,),
            )
        return (bool(rows) and rows[0][0] == "chat"), (rows[0][1] if rows else None)

    async def list_turns(
        self, session_id: str, *, limit: int | None = None, before: int | None = None
    ) -> list[TurnInfo]:
        turns = await super().list_turns(session_id, limit=limit, before=before)
        chat, _ = await self._chat_workspace(session_id)
        return [redact_turn(turn) for turn in turns] if chat else turns

    async def list_events(
        self,
        session_id: str,
        *,
        after_id: int = 0,
        limit: int = 1000,
        turns: tuple[int, int] | None = None,
    ) -> list[StoredEvent]:
        events = await super().list_events(
            session_id, after_id=after_id, limit=limit, turns=turns
        )
        chat, workspace = await self._chat_workspace(session_id)
        if not chat:
            return events
        # A call's output is hidden when its call read a secret file. The
        # call's start is usually in the same page; when it is in an earlier
        # one (a followed stream), it is read back from the file.
        secret_reads: dict[str, str] = {}
        started_here: set[str] = set()
        redacted = []
        for stored in events:
            etype = stored.event.get("type")
            data = stored.event.get("data")
            call_id = str(data.get("call_id", "")) if isinstance(data, dict) else ""
            if etype == "tool.started":
                started_here.add(call_id)
                path = sensitive_path(stored.event, workspace)
                if path:
                    secret_reads[call_id] = path
                else:
                    secret_reads.pop(call_id, None)
            elif etype == "tool.finished" and call_id not in started_here:
                started = await self._started_before(stored, call_id)
                path = sensitive_path(started, workspace) if started else None
                if path:
                    secret_reads[call_id] = path
            redacted.append(redact_event(stored, secret_reads))
        return redacted

    async def _started_before(
        self, finished: StoredEvent, call_id: str
    ) -> Mapping[str, Any] | None:
        """The ``tool.started`` event of *call_id* closest before *finished*."""
        if finished.id is None:
            return None
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(
                conn,
                "SELECT event FROM events WHERE session_id = ? AND turn IS ? AND id < ? "
                "AND json_extract(event, '$.type') = 'tool.started' "
                "AND json_extract(event, '$.data.call_id') = ? ORDER BY id DESC LIMIT 1",
                (finished.session_id, finished.turn, finished.id, call_id),
            )
        started = json.loads(rows[0][0]) if rows else None
        return started if isinstance(started, dict) else None
