"""The scheduler's task store: component version 1 of the gateway's file.

The seven tables of design 4.2 arrive as the ``scheduler`` component of the
gateway's ``sessions.db``, never of the CLI's file. :meth:`TaskStore.open`
reads ``schema_components`` and, when the component is absent, creates the
tables and the version row in one ``BEGIN IMMEDIATE`` transaction on the
session store's own connection; a newer component is refused, the way a
newer ``user_version`` is. ``PRAGMA user_version`` stays the session
schema's, so a terminal ``melo`` that opens no component keeps reading the
same file.

Rows are written through :meth:`SessionStore.component_transaction`, so a
task, the ``kind="task"`` session it names and its first
``priority_events``/``sched_log`` rows commit together or not at all.
``UNIQUE(origin, request_key)`` makes a repeated create idempotent: the same
request returns its row with ``repeated=True`` and a reused key with a
different spec raises :class:`TaskConflictError`.

Retention is the session store's thinking retention (30 days by default):
:meth:`TaskStore.apply_retention` clears ``sched_log`` rows and
``idle_windows`` older than that cutoff; it is a hook of the session store's
sweep and also runs when the component opens. Console routes and task workers
use the store for task persistence.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, get_args

from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.scheduler.model import TERMINAL_PHASES, Origin
from localmelo.support.sessions.store import SessionStore, SessionStoreError

COMPONENT: Final = "scheduler"
SCHEDULER_VERSION: Final = 1

TASK_ORIGINS: Final = frozenset(get_args(Origin))
_TERMINAL_STATES: Final = TERMINAL_PHASES

# One statement each; the migration runs them all inside one transaction.
_SCHEMA: Final = (
    """CREATE TABLE IF NOT EXISTS tasks (
  task_id     TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  front_id    TEXT,
  root_id     TEXT NOT NULL,
  origin      TEXT NOT NULL,
  request_key TEXT,
  title TEXT NOT NULL, spec TEXT NOT NULL,
  state TEXT NOT NULL, wait_reason TEXT NOT NULL DEFAULT '',
  hold_site TEXT NOT NULL DEFAULT '',
  level INTEGER NOT NULL, level_source TEXT NOT NULL,
  level_pinned INTEGER NOT NULL DEFAULT 0, level_eff INTEGER NOT NULL,
  peak_level INTEGER NOT NULL, reason TEXT NOT NULL DEFAULT '',
  user_rank REAL, features TEXT NOT NULL DEFAULT '{}',
  deadline REAL, budget_s REAL,
  service_s REAL NOT NULL DEFAULT 0, wait_s REAL NOT NULL DEFAULT 0,
  tool_s REAL NOT NULL DEFAULT 0,
  counters TEXT NOT NULL DEFAULT '{}',
  crashes INTEGER NOT NULL DEFAULT 0, pgids TEXT NOT NULL DEFAULT '[]',
  card TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL, started_at REAL, finished_at REAL,
  updated_at REAL NOT NULL, expires_at REAL,
  UNIQUE (origin, request_key))""",
    """CREATE TABLE IF NOT EXISTS task_deps (
  task_id TEXT NOT NULL, waits_on TEXT NOT NULL,
  created_by TEXT NOT NULL, confirmed INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (task_id, waits_on))""",
    """CREATE TABLE IF NOT EXISTS priority_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
  ts REAL NOT NULL, actor TEXT NOT NULL, from_level INTEGER, to_level INTEGER NOT NULL,
  pinned INTEGER NOT NULL, reason TEXT NOT NULL DEFAULT '', quote TEXT NOT NULL DEFAULT '')""",
    """CREATE TABLE IF NOT EXISTS questions (
  qid TEXT PRIMARY KEY, task_id TEXT NOT NULL, text TEXT NOT NULL,
  options TEXT NOT NULL DEFAULT '[]', default_answer TEXT NOT NULL DEFAULT '',
  blocking INTEGER NOT NULL, created_at REAL NOT NULL, surfaced_at REAL,
  answered_at REAL, answer TEXT, answered_by TEXT)""",
    """CREATE TABLE IF NOT EXISTS sched_rules (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, match TEXT NOT NULL,
  level INTEGER NOT NULL, enabled INTEGER NOT NULL, created_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS sched_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
  kind TEXT NOT NULL, task_id TEXT, data TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS idle_windows (
  start REAL NOT NULL, end REAL, power TEXT, battery_pct REAL,
  thermal TEXT, lock_state TEXT, user_present INTEGER)""",
    "CREATE INDEX IF NOT EXISTS tasks_state_level ON tasks(state, level_eff)",
    "CREATE INDEX IF NOT EXISTS priority_events_task ON priority_events(task_id, ts)",
    "CREATE INDEX IF NOT EXISTS sched_log_ts ON sched_log(ts)",
    "CREATE INDEX IF NOT EXISTS idle_windows_start ON idle_windows(start)",
)

_TASK_COLUMNS: Final = (
    "task_id",
    "front_id",
    "root_id",
    "origin",
    "request_key",
    "title",
    "spec",
    "state",
    "wait_reason",
    "hold_site",
    "level",
    "level_source",
    "level_pinned",
    "level_eff",
    "peak_level",
    "reason",
    "user_rank",
    "features",
    "deadline",
    "budget_s",
    "service_s",
    "wait_s",
    "tool_s",
    "counters",
    "crashes",
    "pgids",
    "card",
    "created_at",
    "started_at",
    "finished_at",
    "updated_at",
    "expires_at",
)
_SELECT_TASK: Final = "SELECT " + ", ".join(_TASK_COLUMNS) + " FROM tasks"


class TaskStoreError(SessionStoreError):
    """A task store operation cannot do what was asked."""


class TaskConflictError(TaskStoreError):
    """A request key was reused with a different payload."""


class TaskNotFoundError(TaskStoreError):
    """No task has this id."""


class TaskNotPausedError(TaskStoreError):
    """A resume was requested for a task that is not paused."""


@dataclass(frozen=True)
class TaskRecord:
    """One ``tasks`` row, with its JSON columns decoded."""

    task_id: str
    front_id: str | None
    root_id: str
    origin: str
    request_key: str | None
    title: str
    spec: Mapping[str, Any]
    state: str
    wait_reason: str
    hold_site: str
    level: int
    level_source: str
    level_pinned: bool
    level_eff: int
    peak_level: int
    reason: str
    user_rank: float | None
    features: Mapping[str, Any]
    deadline: float | None
    budget_s: float | None
    service_s: float
    wait_s: float
    tool_s: float
    counters: Mapping[str, Any]
    crashes: int
    pgids: Sequence[str]
    card: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    updated_at: float
    expires_at: float | None

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> TaskRecord:
        values = dict(zip(_TASK_COLUMNS, row, strict=True))
        return cls(
            **{**values, **_json_columns(values)},  # type: ignore[arg-type]
        )


def _json_columns(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": json.loads(values["spec"]),
        "features": json.loads(values["features"]),
        "counters": json.loads(values["counters"]),
        "pgids": tuple(json.loads(values["pgids"])),
        "level_pinned": bool(values["level_pinned"]),
    }


class TaskStore:
    """The scheduler's component over one gateway :class:`SessionStore`."""

    def __init__(
        self,
        store: SessionStore,
        *,
        clock: Any = time.time,
    ) -> None:
        if store.writer != "gateway":
            raise ValueError("the scheduler's task store needs the gateway's store")
        self._store = store
        self._clock = clock
        self._version: int | None = None
        self._retention_hooked = False

    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def version(self) -> int | None:
        """The component version :meth:`open` last saw."""
        return self._version

    async def open(self) -> int:
        """Create the ``scheduler`` component at version 1, or check it.

        Idempotent. On a gateway file whose component is absent, the seven
        tables and the version row are written in one transaction; a newer
        version raises :class:`SessionStoreError` and writes nothing.
        """
        self._version = await self._store.ensure_component(
            COMPONENT, SCHEDULER_VERSION, _SCHEMA
        )
        if not self._retention_hooked:
            self._store.add_retention_hook(self.apply_retention)
            self._retention_hooked = True
        # A gateway that was down past the retention window prunes its own
        # trace and idle rows as it opens, beside the session store's sweep.
        await self.apply_retention()
        return self._version

    # ── Creation ──

    async def find_existing_request(
        self,
        spec: TaskSpec,
        *,
        origin: str,
        front_id: str | None = None,
        request_key: str | None = None,
        task_id: str | None = None,
        meta: Mapping[str, Any] | None = None,
        deadline: float | None = None,
    ) -> TaskRecord | None:
        """Find an identical keyed request before validating its workspace.

        A reused key with a different request raises ``TaskConflictError``.
        An absent key or match returns None. This is only a preflight lookup:
        ``create_task`` repeats the check atomically when inserting.
        """
        _validate_create_request(spec, origin, front_id, request_key, task_id, meta)
        if request_key is None:
            return None
        async with self._store.component_transaction("task_find_request") as conn:
            return await self._find_request(
                conn, spec, origin, front_id, request_key, deadline
            )

    @staticmethod
    async def _find_request(
        conn: Any,
        spec: TaskSpec,
        origin: str,
        front_id: str | None,
        request_key: str,
        deadline: float | None,
    ) -> TaskRecord | None:
        row = await _first(
            conn,
            f"{_SELECT_TASK} WHERE origin = ? AND request_key = ?",
            (origin, request_key),
        )
        if row is None:
            return None
        record = TaskRecord.from_row(row)
        if not _same_request(record, spec, origin, front_id, request_key, deadline):
            raise TaskConflictError(
                "This request key was already used for a different task"
            )
        return record

    async def create_task(
        self,
        spec: TaskSpec,
        *,
        origin: str,
        workspace: str,
        front_id: str | None = None,
        request_key: str | None = None,
        task_id: str | None = None,
        meta: Mapping[str, Any] | None = None,
        deadline: float | None = None,
    ) -> tuple[TaskRecord, bool]:
        """Create the task and its ``kind="task"`` session in one transaction.

        Returns ``(record, repeated)``: with *request_key*, the same
        ``(origin, request_key)`` and the same payload returns the stored
        row with ``repeated=True``; a different payload raises
        :class:`TaskConflictError`. The session shares the task's id and stores
        *workspace*, the real path already validated by trusted code; the
        original workspace request remains in ``tasks.spec``.
        """
        if not isinstance(workspace, str) or not os.path.isabs(workspace):
            raise ValueError("workspace must be a validated absolute path")
        meta_text, body = _validate_create_request(
            spec, origin, front_id, request_key, task_id, meta
        )
        body["deadline"] = deadline
        now = self._clock()
        async with self._store.component_transaction("task_create") as conn:
            if request_key is not None:
                record = await self._find_request(
                    conn, spec, origin, front_id, request_key, deadline
                )
                if record is not None:
                    return record, True
            identifier = task_id or _new_id()
            if await self._get(conn, identifier) is not None or await _session_exists(
                conn, identifier
            ):
                raise TaskConflictError(
                    f"A session with id {identifier} already exists"
                )
            if spec.level == 0 and spec.level_source == "user":
                await self._demote_other_p0(conn, identifier, origin, now)
            stored_spec = {
                **body,
                "task_id": identifier,
                "origin": origin,
                "request_key": request_key,
                "front_id": front_id,
            }
            await conn.execute(
                "INSERT INTO sessions (id, kind, title, created_at, updated_at, "
                "model, workspace, meta) VALUES (?, 'task', ?, ?, ?, '', ?, ?)",
                (
                    identifier,
                    spec.title,
                    now,
                    now,
                    workspace,
                    meta_text,
                ),
            )
            await conn.execute(
                "INSERT INTO tasks (task_id, front_id, root_id, origin, "
                "request_key, title, spec, state, wait_reason, hold_site, level, "
                "level_source, level_pinned, level_eff, peak_level, reason, "
                "user_rank, deadline, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', '', '', ?, ?, ?, ?, ?, ?, "
                "NULL, ?, ?, ?)",
                (
                    identifier,
                    front_id,
                    identifier,
                    origin,
                    request_key,
                    spec.title,
                    json.dumps(stored_spec, sort_keys=True),
                    spec.level,
                    spec.level_source,
                    int(_pinned(spec.level_source)),
                    spec.level,
                    spec.level,
                    spec.reason,
                    deadline,
                    now,
                    now,
                ),
            )
            await self._event(
                conn,
                identifier,
                from_level=None,
                to_level=spec.level,
                actor=origin,
                reason=spec.reason,
                quote="",
                pinned=_pinned(spec.level_source),
                ts=now,
            )
            await self._log(
                conn,
                "task.created",
                identifier,
                {
                    "origin": origin,
                    "request_key": request_key,
                    "level": spec.level,
                    "level_source": spec.level_source,
                },
                ts=now,
            )
            created = await self._get(conn, identifier)
        assert created is not None
        return created, False

    # ── Level and state changes ──

    async def set_level(
        self,
        task_id: str,
        level: int,
        *,
        source: str = "user",
        actor: str = "user",
        reason: str = "",
        quote: str = "",
        force: bool = False,
    ) -> TaskRecord:
        """Set a task's level, write its ``priority_events`` and ``sched_log``.

        A user's P0 is single: setting it on a second task returns the prior
        P0 to P1 with no ``user_rank`` in the same transaction (design 2.5).
        A level the user (or a user rule) pinned is never replaced by a
        computed one unless *force* is set; the chip's ``auto`` is the
        explicit force.
        """
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 5:
            raise ValueError("level must be an integer from 0 (P0) to 5 (Later)")
        if source not in ("user", "user_rule", "computed"):
            raise ValueError("source must be 'user', 'user_rule' or 'computed'")
        if level == 0 and source != "user":
            raise ValueError("P0 is the user's setting; level 0 needs source 'user'")
        now = self._clock()
        async with self._store.component_transaction("task_set_level") as conn:
            current = await self._get(conn, task_id)
            if current is None:
                raise TaskNotFoundError(f"No task has id {task_id}")
            if current.state in _TERMINAL_STATES:
                raise TaskStoreError(f"Task {task_id} is finished ({current.state})")
            if current.level_pinned and source == "computed" and not force:
                raise TaskStoreError(
                    f"Task {task_id}'s level is set by the user; use force to "
                    "hand it back to the scheduler"
                )
            if level == 0 and source == "user":
                await self._demote_other_p0(conn, task_id, actor, now)
            await self._write_level(conn, current, level, source, None, reason, now)
            if current.level != level:
                await self._event(
                    conn,
                    task_id,
                    from_level=current.level,
                    to_level=level,
                    actor=actor,
                    reason=reason,
                    quote=quote,
                    pinned=_pinned(source),
                    ts=now,
                )
            await self._log(
                conn,
                "task.level",
                task_id,
                {
                    "from": current.level,
                    "to": level,
                    "source": source,
                    "actor": actor,
                    "reason": reason,
                },
                ts=now,
            )
            record = await self._get(conn, task_id)
        assert record is not None
        return record

    async def set_priority(
        self,
        task_id: str,
        choice: int | str,
        *,
        actor: str = "user",
        reason: str = "",
        quote: str = "",
    ) -> TaskRecord:
        """The chip's three kinds of choice: ``P0``-``P5``/``0``-``5``, ``auto``, ``first``.

        ``auto`` hands the task back to the system at P2, the computed
        default; ``first`` keeps the level and sets ``user_rank`` so the task
        sorts first among its class (earlier "first" choices sort earlier).
        """
        if isinstance(choice, str):
            text = choice.strip().lower()
            if text == "auto":
                return await self.set_level(
                    task_id,
                    2,
                    source="computed",
                    actor=actor,
                    reason=reason,
                    quote=quote,
                    force=True,
                )
            if text == "first":
                return await self._set_first(
                    task_id, actor=actor, reason=reason, quote=quote
                )
            if len(text) == 2 and text[0] == "p" and text[1].isdigit():
                choice = int(text[1])
        if isinstance(choice, bool) or not isinstance(choice, int):
            raise ValueError("priority must be P0-P5, auto or first")
        return await self.set_level(
            task_id, choice, source="user", actor=actor, reason=reason, quote=quote
        )

    async def get_task(self, task_id: str) -> TaskRecord:
        async with self._store.component_transaction("task_get") as conn:
            record = await self._get(conn, task_id)
            if record is None:
                raise TaskNotFoundError(f"No task has id {task_id}")
            return record

    async def list_tasks(self) -> list[TaskRecord]:
        async with self._store.component_transaction("task_list") as conn:
            rows = await _rows(
                conn, f"{_SELECT_TASK} ORDER BY created_at DESC, task_id"
            )
            return [TaskRecord.from_row(row) for row in rows]

    async def priority_events(self, task_id: str) -> list[dict[str, Any]]:
        async with self._store.component_transaction("task_events") as conn:
            rows = await _rows(
                conn,
                "SELECT ts, actor, from_level, to_level, pinned, "
                "reason, quote FROM priority_events WHERE task_id = ? ORDER BY id",
                (task_id,),
            )
            keys = (
                "ts",
                "actor",
                "from_level",
                "to_level",
                "pinned",
                "reason",
                "quote",
            )
            return [dict(zip(keys, row, strict=True)) for row in rows]

    async def change(
        self,
        task_id: str,
        *,
        priority: object = None,
        state: str | None = None,
        actor: str = "user",
        failure_reason: str | None = None,
    ) -> TaskRecord:
        """Commit priority and state together, including P0 demotions and audit rows.

        A failure reason is stored in wait_reason for a failed task.
        """
        choice = priority.strip().lower() if isinstance(priority, str) else priority
        if (
            isinstance(choice, str)
            and len(choice) == 2
            and choice[0] == "p"
            and choice[1] in "012345"
        ):
            choice = int(choice[1])
        if (
            choice is not None
            and choice not in ("auto", "first")
            and (
                isinstance(choice, bool)
                or not isinstance(choice, int)
                or not 0 <= choice <= 5
            )
        ):
            raise ValueError("priority must be P0-P5, auto or first")
        if state is not None and state not in {
            "queued",
            "running",
            "held",
            "parked",
            "done",
            "failed",
            "cancelled",
            "expired",
        }:
            raise ValueError("invalid task state")
        if failure_reason is not None and (
            not isinstance(failure_reason, str) or state != "failed"
        ):
            raise ValueError(
                "failure_reason requires a failed state and must be a string"
            )
        now = self._clock()
        async with self._store.component_transaction("task_change") as conn:
            current = await self._get(conn, task_id)
            if current is None:
                raise TaskNotFoundError(f"No task has id {task_id}")
            if current.state in _TERMINAL_STATES:
                raise TaskStoreError(f"Task {task_id} is finished ({current.state})")
            if choice is not None:
                if choice == "first":
                    await conn.execute(
                        "UPDATE tasks SET user_rank = ?, updated_at = ? WHERE task_id = ?",
                        (now, now, task_id),
                    )
                else:
                    level = 2 if choice == "auto" else choice
                    source = "computed" if choice == "auto" else "user"
                    if level == 0:
                        await self._demote_other_p0(conn, task_id, actor, now)
                    reason = current.reason if current.level == level else ""
                    await self._write_level(
                        conn, current, level, source, None, reason, now
                    )
                    if current.level != level:
                        await self._event(
                            conn,
                            task_id,
                            from_level=current.level,
                            to_level=level,
                            actor=actor,
                            reason="",
                            quote="",
                            pinned=_pinned(source),
                            ts=now,
                        )
                await self._log(
                    conn,
                    (
                        "task.level.noop"
                        if choice == 5
                        and current.level == 5
                        and current.reason == "pause"
                        else "task.level"
                    ),
                    task_id,
                    {"choice": choice, "actor": actor},
                    ts=now,
                )
            if state is not None:
                await conn.execute(
                    "UPDATE tasks SET state = ?, started_at = CASE WHEN ? = 'running' THEN COALESCE(started_at, ?) ELSE started_at END, "
                    "finished_at = CASE WHEN ? IN ('done','failed','cancelled','expired') THEN ? ELSE NULL END, "
                    "wait_reason = ?, hold_site = '', updated_at = ? WHERE task_id = ?",
                    (state, state, now, state, now, failure_reason or "", now, task_id),
                )
                await self._log(
                    conn,
                    "task.state",
                    task_id,
                    {"from": current.state, "to": state, "actor": actor},
                    ts=now,
                )
            await self._touch_session(conn, task_id, now)
            result = await self._get(conn, task_id)
        assert result is not None
        return result

    async def pause(self, task_id: str, *, actor: str = "user") -> TaskRecord:
        """Park at P5, recording the previous priority in the audit log."""
        now = self._clock()
        async with self._store.component_transaction("task_pause") as conn:
            current = await self._get(conn, task_id)
            if current is None:
                raise TaskNotFoundError(f"No task has id {task_id}")
            if current.state in _TERMINAL_STATES:
                raise TaskStoreError(f"Task {task_id} is finished ({current.state})")
            if current.state == "parked" and current.level == 5:
                return current
            await self._write_level(conn, current, 5, "user", None, "pause", now)
            await self._event(
                conn,
                task_id,
                from_level=current.level,
                to_level=5,
                actor=actor,
                reason="pause",
                quote="",
                pinned=True,
                ts=now,
            )
            await conn.execute(
                "UPDATE tasks SET state = 'parked', wait_reason = '', hold_site = '' WHERE task_id = ?",
                (task_id,),
            )
            await self._log(
                conn,
                "task.pause",
                task_id,
                {
                    "from_level": current.level,
                    "from_source": current.level_source,
                    "reason": current.reason,
                    "rank": current.user_rank,
                },
                ts=now,
            )
            result = await self._get(conn, task_id)
        assert result is not None
        return result

    async def resume(self, task_id: str, *, actor: str = "user") -> TaskRecord:
        """Restore a pause priority; a parked P5 without a pause falls back to P2."""
        now = self._clock()
        async with self._store.component_transaction("task_resume") as conn:
            current = await self._get(conn, task_id)
            if current is None:
                raise TaskNotFoundError(f"No task has id {task_id}")
            if current.state in _TERMINAL_STATES:
                raise TaskStoreError(f"Task {task_id} is finished ({current.state})")
            if current.state != "parked" or current.level != 5:
                raise TaskNotPausedError(f"Task {task_id} is not paused")
            row = (
                await _first(
                    conn,
                    "SELECT id, data FROM sched_log WHERE task_id = ? AND kind = 'task.pause' ORDER BY id DESC LIMIT 1",
                    (task_id,),
                )
                if current.reason == "pause"
                else None
            )
            later = (
                await _first(
                    conn,
                    "SELECT 1 FROM sched_log WHERE task_id = ? AND id > ? AND kind IN ('task.level', 'task.first', 'task.resume') LIMIT 1",
                    (task_id, row[0]),
                )
                if row
                else None
            )
            saved = json.loads(row[1]) if row and not later else None
            level = saved["from_level"] if saved else (current.level if later else 2)
            source = (
                saved["from_source"]
                if saved
                else (current.level_source if later else "computed")
            )
            reason = saved["reason"] if saved else (current.reason if later else "")
            rank = saved["rank"] if saved else (current.user_rank if later else None)
            if level == 0 and source == "user":
                await self._demote_other_p0(conn, task_id, actor, now)
            await self._write_level(conn, current, level, source, rank, reason, now)
            if current.level != level:
                await self._event(
                    conn,
                    task_id,
                    from_level=current.level,
                    to_level=level,
                    actor=actor,
                    reason="resume",
                    quote="",
                    pinned=_pinned(source),
                    ts=now,
                )
            await conn.execute(
                "UPDATE tasks SET state = ?, wait_reason = '', hold_site = '' WHERE task_id = ?",
                ("parked" if level == 5 else "queued", task_id),
            )
            await self._log(
                conn, "task.resume", task_id, {"level": level, "source": source}, ts=now
            )
            result = await self._get(conn, task_id)
        assert result is not None
        return result

    async def fail_nonterminal_on_restart(self) -> int:
        """Fail unrecovered tasks together when a new gateway starts."""
        now = self._clock()
        reason = "gateway restarted; recovery arrives in S3b"
        async with self._store.component_transaction("task_restart") as conn:
            rows = await _rows(
                conn,
                "SELECT task_id, state FROM tasks WHERE state NOT IN (?, ?, ?, ?)",
                tuple(sorted(_TERMINAL_STATES)),
            )
            for task_id, state in rows:
                await conn.execute(
                    "UPDATE tasks SET state = 'failed', wait_reason = ?, hold_site = '', "
                    "finished_at = ?, updated_at = ? WHERE task_id = ?",
                    (reason, now, now, task_id),
                )
                await self._touch_session(conn, task_id, now)
                await self._log(
                    conn,
                    "task.state",
                    task_id,
                    {
                        "from": state,
                        "to": "failed",
                        "actor": "scheduler",
                        "reason": reason,
                    },
                    ts=now,
                )
        return len(rows)

    # ── Retention ──

    async def apply_retention(self) -> int:
        """Delete ``sched_log`` and ``idle_windows`` older than retention."""
        cutoff = self._store.retention_cutoff()
        changed = 0
        async with self._store.component_transaction("task_retention") as conn:
            cursor = await conn.execute("DELETE FROM sched_log WHERE ts < ?", (cutoff,))
            changed += max(cursor.rowcount, 0)
            cursor = await conn.execute(
                "DELETE FROM idle_windows WHERE COALESCE(end, start) < ?", (cutoff,)
            )
            changed += max(cursor.rowcount, 0)
        return changed

    # ── Internals ──

    async def _get(self, conn: Any, task_id: str) -> TaskRecord | None:
        row = await _first(conn, f"{_SELECT_TASK} WHERE task_id = ?", (task_id,))
        return TaskRecord.from_row(row) if row is not None else None

    async def _demote_other_p0(
        self, conn: Any, task_id: str, actor: str, now: float
    ) -> None:
        rows = await _rows(
            conn,
            "SELECT task_id, level, level_pinned FROM tasks WHERE level = 0 "
            "AND level_source = 'user' AND task_id <> ? AND state NOT IN "
            "(?, ?, ?, ?) ORDER BY created_at",
            (task_id, *sorted(_TERMINAL_STATES)),
        )
        for other_id, old_level, pinned in rows:
            await conn.execute(
                "UPDATE tasks SET level = 1, level_eff = 1, "
                "peak_level = MIN(peak_level, 1), user_rank = NULL, "
                "reason = 'p0_conflict', updated_at = ? WHERE task_id = ?",
                (now, other_id),
            )
            await self._touch_session(conn, other_id, now)
            await self._event(
                conn,
                other_id,
                from_level=old_level,
                to_level=1,
                actor="scheduler",
                reason="p0_conflict",
                quote="",
                pinned=bool(pinned),
                ts=now,
            )
            await self._log(
                conn,
                "task.p0_conflict",
                other_id,
                {"from": old_level, "to": 1, "winner": task_id, "actor": actor},
                ts=now,
            )

    async def _write_level(
        self,
        conn: Any,
        current: TaskRecord,
        level: int,
        source: str,
        user_rank: float | None,
        reason: str,
        now: float,
    ) -> None:
        await conn.execute(
            "UPDATE tasks SET level = ?, level_source = ?, level_pinned = ?, "
            "level_eff = ?, peak_level = MIN(peak_level, ?), user_rank = ?, "
            "reason = ?, updated_at = ? WHERE task_id = ?",
            (
                level,
                source,
                int(source in ("user", "user_rule")),
                level,
                level,
                user_rank,
                reason,
                now,
                current.task_id,
            ),
        )
        await self._touch_session(conn, current.task_id, now)

    async def _set_first(
        self, task_id: str, *, actor: str, reason: str, quote: str
    ) -> TaskRecord:
        now = self._clock()
        # Seconds, so an earlier "first" is a smaller value and sorts first
        # within its class (levels.py sorts ascending; None is +inf).
        rank = now
        async with self._store.component_transaction("task_set_first") as conn:
            current = await self._get(conn, task_id)
            if current is None:
                raise TaskNotFoundError(f"No task has id {task_id}")
            if current.state in _TERMINAL_STATES:
                raise TaskStoreError(f"Task {task_id} is finished ({current.state})")
            await conn.execute(
                "UPDATE tasks SET user_rank = ?, updated_at = ? WHERE task_id = ?",
                (rank, now, task_id),
            )
            await self._touch_session(conn, task_id, now)
            await self._log(
                conn,
                "task.first",
                task_id,
                {"rank": rank, "actor": actor, "reason": reason, "quote": quote},
                ts=now,
            )
            record = await self._get(conn, task_id)
        assert record is not None
        return record

    async def _touch_session(self, conn: Any, task_id: str, now: float) -> None:
        await conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, task_id)
        )

    async def _event(
        self,
        conn: Any,
        task_id: str,
        *,
        from_level: int | None,
        to_level: int,
        actor: str,
        reason: str,
        quote: str,
        pinned: bool,
        ts: float,
    ) -> None:
        await conn.execute(
            "INSERT INTO priority_events (task_id, ts, actor, from_level, "
            "to_level, pinned, reason, quote) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, ts, actor, from_level, to_level, int(pinned), reason, quote),
        )

    async def _log(
        self,
        conn: Any,
        kind: str,
        task_id: str | None,
        data: Mapping[str, Any],
        *,
        ts: float,
    ) -> None:
        await conn.execute(
            "INSERT INTO sched_log (ts, kind, task_id, data) VALUES (?, ?, ?, ?)",
            (ts, kind, task_id, json.dumps(dict(data), sort_keys=True)),
        )


def _validate_create_request(
    spec: TaskSpec,
    origin: str,
    front_id: str | None,
    request_key: str | None,
    task_id: str | None,
    meta: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if origin not in TASK_ORIGINS:
        raise ValueError(f"unknown origin {origin!r}")
    if request_key is not None and not isinstance(request_key, str):
        raise ValueError("request_key must be a string or None")
    if request_key == "":
        raise ValueError("request_key must not be empty")
    if front_id is not None and not isinstance(front_id, str):
        raise ValueError("front_id must be a string or None")
    if task_id is not None and not isinstance(task_id, str):
        raise ValueError("task_id must be a string or None")
    return json.dumps(dict(meta or {}), sort_keys=True), spec.to_dict()


def _pinned(source: str) -> bool:
    """Whether a level source pins the level against computed changes."""
    return source in ("user", "user_rule")


def _same_request(
    record: TaskRecord,
    spec: TaskSpec,
    origin: str,
    front_id: str | None,
    request_key: str,
    deadline: float | None,
) -> bool:
    stored = {key: value for key, value in record.spec.items() if key != "task_id"}
    stored.setdefault("deadline", None)
    expected = {
        **spec.to_dict(),
        "deadline": deadline,
        "origin": origin,
        "request_key": request_key,
        "front_id": front_id,
    }
    return record.front_id == front_id and stored == json.loads(
        json.dumps(expected, sort_keys=True)
    )


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


async def _rows(conn: Any, sql: str, params: Sequence[Any] = ()) -> list[Any]:
    cursor = await conn.execute(sql, tuple(params))
    return list(await cursor.fetchall())


async def _first(conn: Any, sql: str, params: Sequence[Any] = ()) -> Any:
    cursor = await conn.execute(sql, tuple(params))
    return await cursor.fetchone()


async def _session_exists(conn: Any, session_id: str) -> bool:
    row = await _first(conn, "SELECT 1 FROM sessions WHERE id = ?", (session_id,))
    return row is not None


__all__ = [
    "COMPONENT",
    "SCHEDULER_VERSION",
    "TASK_ORIGINS",
    "TaskConflictError",
    "TaskNotFoundError",
    "TaskNotPausedError",
    "TaskRecord",
    "TaskStore",
    "TaskStoreError",
]
