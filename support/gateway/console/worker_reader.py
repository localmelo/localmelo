"""Read worker sessions from the CLI file without opening it for writing."""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from localmelo.support.sessions.store import (
    _SESSION_COLUMNS,
    SessionInfo,
    SessionStore,
    SessionStoreError,
    _escape_like,
    _rows,
)


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
    """Read-only CLI connection, with keyset paging over worker start times."""

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
    ) -> list[WorkerRecord]:
        if not os.path.exists(self.path):
            return []
        clauses = ["s.kind = 'worker'"]
        args: list[Any] = []
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
            "AND t.turn = (SELECT MIN(first.turn) FROM turns first WHERE first.session_id = s.id) "
            f"WHERE {' AND '.join(clauses)} ORDER BY t.started_at DESC, s.id DESC LIMIT ?"
        )
        async with self._lock:
            conn = await self._ready()
            rows = await _rows(conn, sql, args)
            now = self._clock()
            records = []
            for row in rows:
                info = self._session_info(row, now)
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

    async def get_session(self, session_id: str) -> SessionInfo | None:
        if not os.path.exists(self.path):
            return None
        info = await super().get_session(session_id)
        return info if info is not None and info.kind == "worker" else None
