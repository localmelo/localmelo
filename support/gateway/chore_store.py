"""Chores: saved requests with a schedule, in the gateway's session file.

A chore is a request melo runs on a schedule (:mod:`.chore_schedule`): its
title, the request text, a level, an on/off switch, when it last ran and when
it runs next. The rows live in the scheduler component of ``sessions.db``
(version 6): :data:`CHORE_SCHEMA` is created by
:meth:`~localmelo.support.gateway.task_store.TaskStore.open`, for a new file
and for an older one alike, so no existing row changes. Each run is a task
with the origin ``user_rule``; :class:`ChoreStore` only keeps the rows.

Every change bumps ``revision``; a write that names a revision refuses
(:class:`ChoreStaleError`) when the row has moved on, so two windows never
overwrite each other unseen. ``chore_meta`` holds the check-in's switch.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import tzinfo
from typing import Any, Final

from localmelo.support.gateway.chore_schedule import JITTER_S, Schedule
from localmelo.support.gateway.task_policy import MAX_OBJECTIVE_CHARS, MAX_TITLE_CHARS
from localmelo.support.sessions.store import SessionStore, SessionStoreError

CHORE_SCHEMA: Final = (
    """CREATE TABLE IF NOT EXISTS chores (
  chore_id TEXT PRIMARY KEY,
  front_id TEXT,
  title TEXT NOT NULL,
  request TEXT NOT NULL,
  schedule TEXT NOT NULL,
  level INTEGER NOT NULL DEFAULT 2,
  enabled INTEGER NOT NULL DEFAULT 1,
  revision INTEGER NOT NULL DEFAULT 1,
  next_run REAL,
  last_run REAL,
  last_task_id TEXT NOT NULL DEFAULT '',
  reported_task_id TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS chore_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)

# Chores one file holds: the tick reads every enabled one each 30 s.
MAX_CHORES: Final = 50
_COLUMNS: Final = (
    "chore_id, front_id, title, request, schedule, level, enabled, revision, "
    "next_run, last_run, last_task_id, reported_task_id, created_at, updated_at"
)


class ChoreError(SessionStoreError):
    """A chore operation cannot do what was asked."""


class ChoreNotFoundError(ChoreError):
    """No chore has this id."""


class ChoreStaleError(ChoreError):
    """The chore changed since the revision the caller read."""


class ChoreLimitError(ChoreError):
    """The file already holds :data:`MAX_CHORES` chores."""


@dataclass(frozen=True)
class Chore:
    chore_id: str
    front_id: str | None
    title: str
    request: str
    schedule: Schedule
    level: int
    enabled: bool
    revision: int
    next_run: float | None
    last_run: float | None
    last_task_id: str
    reported_task_id: str
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row: Any) -> Chore:
        values = list(row)
        values[4] = Schedule.from_json(json.loads(values[4]))
        values[6] = bool(values[6])
        return cls(*values)


def _text(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value.strip()) > limit:
        raise ValueError(f"{name} must be at most {limit} characters")
    return value.strip()


def _level(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
        raise ValueError("level: Today, Normal, Whenever or If idle")
    return value


class ChoreStore:
    """The chores of one gateway :class:`SessionStore`; its tables come from the task store's open."""

    def __init__(
        self,
        store: SessionStore,
        *,
        clock: Callable[[], float] = time.time,
        tz: tzinfo | None = None,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock
        self._tz = tz
        self._jitter = jitter or (lambda: random.uniform(0.0, JITTER_S))

    def plan(self, schedule: Schedule, after: float) -> float:
        """The next run after *after*: the schedule's next time and a small jitter, never early."""
        return schedule.next_after(after, self._tz) + self._jitter()

    async def list_chores(self, front_id: str | None = None) -> list[Chore]:
        """Every chore, oldest first, or one chat's own."""
        where, params = (
            ("", ()) if front_id is None else (" WHERE front_id = ?", (front_id,))
        )
        async with self._store.component_transaction("chore_list") as conn:
            cursor = await conn.execute(
                f"SELECT {_COLUMNS} FROM chores{where} ORDER BY created_at, rowid",
                params,
            )
            return [Chore.from_row(row) for row in await cursor.fetchall()]

    async def get(self, chore_id: str) -> Chore:
        async with self._store.component_transaction("chore_get") as conn:
            return await self._get(conn, chore_id)

    async def create(
        self,
        *,
        title: str,
        request: str,
        schedule: Schedule,
        level: int = 2,
        front_id: str | None = None,
    ) -> Chore:
        title = _text(title, "title", MAX_TITLE_CHARS)
        request = _text(request, "request", MAX_OBJECTIVE_CHARS)
        level = _level(level)
        now = self._clock()
        chore_id = uuid.uuid4().hex[:12]
        async with self._store.component_transaction("chore_create") as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM chores")
            row = await cursor.fetchone()
            if row is not None and row[0] >= MAX_CHORES:
                raise ChoreLimitError(f"At most {MAX_CHORES} chores. Delete one first.")
            await conn.execute(
                f"INSERT INTO chores ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, NULL, '', '', ?, ?)",
                (chore_id, front_id, title, request, json.dumps(schedule.to_json()), level,
                 self.plan(schedule, now), now, now),
            )  # fmt: skip
            return await self._get(conn, chore_id)

    async def update(
        self,
        chore_id: str,
        *,
        revision: int,
        title: str | None = None,
        request: str | None = None,
        schedule: Schedule | None = None,
        level: int | None = None,
        enabled: bool | None = None,
    ) -> Chore:
        """Change a chore if its revision is *revision*; a new time or a switch back on plans from now."""
        now = self._clock()
        async with self._store.component_transaction("chore_update") as conn:
            chore = await self._get(conn, chore_id)
            if chore.revision != revision:
                raise ChoreStaleError("This chore changed. Reload it and try again.")
            new = dataclasses.replace(
                chore,
                title=(
                    chore.title
                    if title is None
                    else _text(title, "title", MAX_TITLE_CHARS)
                ),
                request=(
                    chore.request
                    if request is None
                    else _text(request, "request", MAX_OBJECTIVE_CHARS)
                ),
                schedule=chore.schedule if schedule is None else schedule,
                level=chore.level if level is None else _level(level),
                enabled=chore.enabled if enabled is None else enabled,
            )
            next_run = chore.next_run
            if not new.enabled:
                next_run = None
            elif schedule is not None or not chore.enabled or next_run is None:
                next_run = self.plan(new.schedule, now)
            await conn.execute(
                "UPDATE chores SET title = ?, request = ?, schedule = ?, level = ?, enabled = ?, "
                "next_run = ?, revision = revision + 1, updated_at = ? WHERE chore_id = ?",
                (new.title, new.request, json.dumps(new.schedule.to_json()), new.level,
                 int(new.enabled), next_run, now, chore_id),
            )  # fmt: skip
            return await self._get(conn, chore_id)

    async def delete(self, chore_id: str, *, revision: int | None = None) -> None:
        async with self._store.component_transaction("chore_delete") as conn:
            chore = await self._get(conn, chore_id)
            if revision is not None and chore.revision != revision:
                raise ChoreStaleError("This chore changed. Reload it and try again.")
            await conn.execute("DELETE FROM chores WHERE chore_id = ?", (chore_id,))

    async def due(self, now: float) -> list[Chore]:
        """The enabled chores whose next run has come, oldest first."""
        async with self._store.component_transaction("chore_due") as conn:
            cursor = await conn.execute(
                f"SELECT {_COLUMNS} FROM chores WHERE enabled = 1 AND next_run IS NOT NULL "
                "AND next_run <= ? ORDER BY next_run, rowid",
                (now,),
            )
            return [Chore.from_row(row) for row in await cursor.fetchall()]

    async def advance(
        self,
        chore: Chore,
        *,
        next_run: float | None,
        task_id: str = "",
        ran_at: float | None = None,
    ) -> bool:
        """Move *chore* to its next run; with *task_id*, record the run that just started.

        ``False`` when the row changed or went meanwhile: the caller's view was stale.
        """
        async with self._store.component_transaction("chore_advance") as conn:
            cursor = await conn.execute(
                "UPDATE chores SET next_run = ?, last_run = COALESCE(?, last_run), "
                "last_task_id = CASE WHEN ? = '' THEN last_task_id ELSE ? END, "
                "revision = revision + 1, updated_at = ? WHERE chore_id = ? AND revision = ?",
                (next_run, ran_at, task_id, task_id, self._clock(), chore.chore_id, chore.revision),
            )  # fmt: skip
            return cursor.rowcount == 1

    async def mark_reported(self, reported: dict[str, str]) -> None:
        """Record that the check-in told the person about each chore's run (``chore_id -> task_id``)."""
        async with self._store.component_transaction("chore_reported") as conn:
            for chore_id, task_id in reported.items():
                await conn.execute(
                    "UPDATE chores SET reported_task_id = ? WHERE chore_id = ?",
                    (task_id, chore_id),
                )

    async def checkin_on(self) -> bool:
        """Whether the 30-minute check-in is on (it is, until the person switches it off)."""
        async with self._store.component_transaction("chore_meta") as conn:
            cursor = await conn.execute(
                "SELECT value FROM chore_meta WHERE key = 'checkin_on'"
            )
            row = await cursor.fetchone()
        return row is None or row[0] != "0"

    async def set_checkin(self, on: bool) -> None:
        async with self._store.component_transaction("chore_meta") as conn:
            await conn.execute(
                "INSERT INTO chore_meta (key, value) VALUES ('checkin_on', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("1" if on else "0",),
            )

    @staticmethod
    async def _get(conn: Any, chore_id: str) -> Chore:
        cursor = await conn.execute(
            f"SELECT {_COLUMNS} FROM chores WHERE chore_id = ?", (chore_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise ChoreNotFoundError(f"No chore has id {chore_id}")
        return Chore.from_row(row)


__all__ = [
    "CHORE_SCHEMA",
    "MAX_CHORES",
    "Chore",
    "ChoreError",
    "ChoreLimitError",
    "ChoreNotFoundError",
    "ChoreStaleError",
    "ChoreStore",
]
