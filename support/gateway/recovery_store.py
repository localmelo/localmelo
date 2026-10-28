"""Dormant, durable primitives for scheduler restart recovery.

No worker uses these primitives until creation and every launch path are wired
through the same fence. Missing provenance is unknown, not never started.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from localmelo.support.gateway.task_store import (
    TaskConflictError,
    TaskNotFoundError,
    TaskStoreError,
    compare_park_identity,
)
from localmelo.support.scheduler.model import TERMINAL_PHASES
from localmelo.support.sessions.store import SessionStore


@dataclass(frozen=True)
class RecoveryProvenance:
    task_id: str
    first_start: str
    execution_id: str | None
    unsafe_resume: bool
    reason: str


@dataclass(frozen=True)
class RecoveryLaunch:
    provenance: RecoveryProvenance
    acquired: bool


async def _one(conn: Any, sql: str, args: tuple[Any, ...]) -> Any:
    return await (await conn.execute(sql, args)).fetchone()


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _guard(generation: int | None, operation_id: str | None) -> None:
    if generation is not None and (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("invalid expected generation")
    if operation_id is not None:
        _text(operation_id, "expected_operation_id")
        if generation is None:
            raise ValueError("operation identity requires a generation")


def _request(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


class RecoveryStore:
    """Recovery state retained with a task, independently of event retention."""

    def __init__(self, store: SessionStore) -> None:
        if store.writer != "gateway":
            raise ValueError("recovery store needs the gateway's store")
        self.store = store

    async def provenance(self, task_id: str) -> RecoveryProvenance:
        async with self.store.component_transaction("recovery_get") as conn:
            if not await _one(
                conn, "SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)
            ):
                raise TaskNotFoundError(f"No task with id {task_id}")
            row = await _one(
                conn,
                "SELECT first_start, execution_id, unsafe_resume, reason "
                "FROM recovery_provenance WHERE task_id = ?",
                (task_id,),
            )
            if row is None:
                return RecoveryProvenance(task_id, "unknown", None, True, "")
            return RecoveryProvenance(task_id, row[0], row[1], bool(row[2]), row[3])

    async def receipt(
        self, task_id: str, operation_id: str
    ) -> tuple[dict[str, Any], bool] | None:
        _text(operation_id, "operation_id")
        async with self.store.component_transaction("recovery_receipt") as conn:
            return await self._receipt(conn, task_id, operation_id)

    async def active_intent(self, task_id: str) -> tuple[str, dict[str, Any]] | None:
        """Find the durable operation that must be reconciled before another."""
        async with self.store.component_transaction("recovery_active") as conn:
            row = await _one(
                conn,
                "SELECT operation_id, request FROM recovery_operations "
                "WHERE task_id = ? AND kind = 'recovery' AND phase = 'prepared'",
                (task_id,),
            )
            return (row[0], json.loads(row[1])) if row else None

    @staticmethod
    async def _receipt(
        conn: Any, task_id: str, operation_id: str
    ) -> tuple[dict[str, Any], bool] | None:
        row = await _one(
            conn,
            "SELECT kind, request, phase, generation, pending_operation_id, task_state "
            "FROM recovery_operations WHERE task_id = ? AND operation_id = ?",
            (task_id, operation_id),
        )
        if row is None:
            return None
        task = await _one(conn, "SELECT state FROM tasks WHERE task_id = ?", (task_id,))
        identity = await _one(
            conn,
            "SELECT generation, operation_id FROM park_priority WHERE task_id = ?",
            (task_id,),
        )
        current = bool(
            row[2] == "committed"
            and task
            and task[0] == row[5]
            and (tuple(identity) if identity else (None, None)) == (row[3], row[4])
        )
        return {
            "kind": row[0],
            "request": json.loads(row[1]),
            "phase": row[2],
            "generation": row[3],
            "pending_operation_id": row[4],
            "state": row[5],
        }, current

    async def start(
        self,
        task_id: str,
        *,
        operation_id: str,
        execution_id: str,
        expected_generation: int | None,
        expected_operation_id: str | None,
        expected_execution_id: str | None = None,
    ) -> RecoveryLaunch:
        """Fence one launch; retries do not grant another execution permit.

        A fresh acquisition is not gate admission: the caller must arbitrate
        Stop and provider entry synchronously through its worker coordinator.
        """
        _text(operation_id, "operation_id")
        _text(execution_id, "execution_id")
        if expected_execution_id is not None:
            _text(expected_execution_id, "expected_execution_id")
        _guard(expected_generation, expected_operation_id)
        acquired = False
        data = {
            "execution_id": execution_id,
            "prior_execution": expected_execution_id,
            "generation": expected_generation,
            "previous": expected_operation_id,
        }
        encoded = _request(data)
        async with self.store.component_event_transaction("recovery_start") as (
            conn,
            events,
        ):
            previous = await self._receipt(conn, task_id, operation_id)
            if previous:
                receipt, current = previous
                if (
                    receipt["kind"] != "start"
                    or receipt["request"] != data
                    or not current
                ):
                    raise TaskConflictError("start operation reused or superseded")
            else:
                task = await _one(
                    conn,
                    "SELECT state, level, card FROM tasks WHERE task_id = ?",
                    (task_id,),
                )
                if task is None:
                    raise TaskNotFoundError(f"No task with id {task_id}")
                provenance = await _one(
                    conn,
                    "SELECT first_start, execution_id, unsafe_resume FROM recovery_provenance WHERE task_id = ?",
                    (task_id,),
                )
                pending = await _one(
                    conn,
                    "SELECT pause_pending FROM park_priority WHERE task_id = ?",
                    (task_id,),
                )
                first = expected_execution_id is None
                if (
                    task[0] != "queued"
                    or task[1] >= 5
                    or not provenance
                    or (first and provenance[0] != "never_started")
                    or (
                        not first
                        and (
                            provenance[0] != "started"
                            or provenance[1] != expected_execution_id
                            or provenance[2]
                            or not task[2].strip()
                            or execution_id == expected_execution_id
                        )
                    )
                    or (pending and pending[0])
                ):
                    raise TaskConflictError("task is not eligible for this launch")
                used = await _one(
                    conn,
                    "SELECT 1 FROM recovery_operations WHERE task_id = ? AND kind = 'start' "
                    "AND json_extract(request, '$.execution_id') = ?",
                    (task_id, execution_id),
                )
                if used:
                    raise TaskConflictError("execution identity was already used")
                generation = await compare_park_identity(
                    conn,
                    task_id,
                    expected_generation=expected_generation,
                    expected_operation_id=expected_operation_id,
                    operation_id=expected_operation_id,
                    create=True,
                )
                await conn.execute(
                    "UPDATE recovery_provenance SET first_start = 'started', execution_id = ?, unsafe_resume = 1 WHERE task_id = ?",
                    (execution_id, task_id),
                )
                await conn.execute(
                    "UPDATE tasks SET state = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE task_id = ?",
                    (time.time(), time.time(), task_id),
                )
                await conn.execute(
                    "INSERT INTO recovery_operations VALUES (?, ?, 'start', ?, 'committed', ?, ?, 'running')",
                    (task_id, operation_id, encoded, generation, expected_operation_id),
                )
                await self.store.append_component_event(
                    conn,
                    events,
                    task_id,
                    "sched.launch_started",
                    {
                        "operation_id": operation_id,
                        "execution_id": execution_id,
                        "generation": generation,
                        "state": "running",
                    },
                )
                acquired = True
            row = await _one(
                conn,
                "SELECT first_start, execution_id, unsafe_resume, reason "
                "FROM recovery_provenance WHERE task_id = ?",
                (task_id,),
            )
            assert row is not None
            result = RecoveryProvenance(task_id, row[0], row[1], bool(row[2]), row[3])
        return RecoveryLaunch(result, acquired)

    async def prepare(
        self,
        task_id: str,
        *,
        operation_id: str,
        execution_id: str,
        expected_generation: int | None,
        expected_operation_id: str | None,
        outcome: str,
        card: str,
        reason: str,
        crash: bool,
        unsafe_resume: bool,
    ) -> bool:
        """Persist an intent; only the new writer may begin its external work."""
        _text(operation_id, "operation_id")
        _text(execution_id, "execution_id")
        _guard(expected_generation, expected_operation_id)
        if outcome not in {"interrupted", "queued", *TERMINAL_PHASES}:
            raise ValueError("invalid recovery outcome")
        if not isinstance(card, str) or len(card) > 2000 or not isinstance(reason, str):
            raise ValueError("invalid recovery card or reason")
        if not isinstance(crash, bool) or not isinstance(unsafe_resume, bool):
            raise ValueError("recovery flags must be boolean")
        if crash and not unsafe_resume:
            raise ValueError("a crashed execution requires manual safety review")
        data = {
            "execution_id": execution_id,
            "generation": expected_generation,
            "previous": expected_operation_id,
            "outcome": outcome,
            "card": card,
            "reason": reason,
            "crash": crash,
            "unsafe_resume": unsafe_resume,
        }
        async with self.store.component_transaction("recovery_prepare") as conn:
            existing = await self._receipt(conn, task_id, operation_id)
            if existing:
                if existing[0]["kind"] != "recovery" or existing[0]["request"] != data:
                    raise TaskConflictError("recovery operation ID reused")
                if existing[0]["phase"] == "committed" and not existing[1]:
                    raise TaskConflictError("recovery operation superseded")
                return False
            pending = await _one(
                conn,
                "SELECT operation_id FROM recovery_operations "
                "WHERE task_id = ? AND kind = 'recovery' AND phase = 'prepared'",
                (task_id,),
            )
            if pending is not None:
                raise TaskConflictError("another recovery intent must be reconciled")
            task = await _one(
                conn, "SELECT state FROM tasks WHERE task_id = ?", (task_id,)
            )
            if task is None:
                raise TaskNotFoundError(f"No task with id {task_id}")
            if task[0] in TERMINAL_PHASES:
                raise TaskStoreError("task is terminal")
            provenance = await _one(
                conn,
                "SELECT first_start, execution_id FROM recovery_provenance WHERE task_id = ?",
                (task_id,),
            )
            if (
                provenance
                and provenance[0] == "started"
                and provenance[1] != execution_id
            ):
                raise TaskConflictError("recovery execution is stale")
            if outcome == "queued" and (
                not provenance
                or provenance[0] != "never_started"
                or crash
                or unsafe_resume
            ):
                raise TaskConflictError("only proved unstarted tasks can be queued")
            identity = await _one(
                conn,
                "SELECT generation, operation_id FROM park_priority WHERE task_id = ?",
                (task_id,),
            )
            if (tuple(identity) if identity else (None, None)) != (
                expected_generation,
                expected_operation_id,
            ):
                raise TaskConflictError("stale task identity")
            await conn.execute(
                "INSERT INTO recovery_operations VALUES (?, ?, 'recovery', ?, 'prepared', NULL, NULL, NULL)",
                (task_id, operation_id, _request(data)),
            )
        return True

    async def commit(self, task_id: str, *, operation_id: str) -> None:
        """Commit a prepared task outcome, receipt and crash count together."""
        _text(operation_id, "operation_id")
        async with self.store.component_event_transaction("recovery_commit") as (
            conn,
            events,
        ):
            found = await self._receipt(conn, task_id, operation_id)
            if found is None or found[0]["kind"] != "recovery":
                raise TaskConflictError("unknown recovery operation")
            receipt, current = found
            if receipt["phase"] == "committed":
                if not current:
                    raise TaskConflictError("recovery operation superseded")
                return
            data = receipt["request"]
            task = await _one(
                conn, "SELECT state FROM tasks WHERE task_id = ?", (task_id,)
            )
            if task is None:
                raise TaskNotFoundError(f"No task with id {task_id}")
            if task[0] in TERMINAL_PHASES:
                raise TaskStoreError("task is terminal")
            terminal = data["outcome"] in TERMINAL_PHASES
            clear_pending = terminal or data["outcome"] == "interrupted"
            pending_id = None if clear_pending else data["previous"]
            generation = await compare_park_identity(
                conn,
                task_id,
                expected_generation=data["generation"],
                expected_operation_id=data["previous"],
                operation_id=pending_id,
                create=True,
            )
            if clear_pending:
                await conn.execute(
                    "UPDATE park_priority SET pause_pending = 0, snapshot_active = CASE WHEN ? THEN 0 ELSE snapshot_active END WHERE task_id = ?",
                    (int(terminal), task_id),
                )
            if data["crash"]:
                inserted = await conn.execute(
                    "INSERT OR IGNORE INTO recovery_crashes VALUES (?, ?)",
                    (task_id, data["execution_id"]),
                )
                if inserted.rowcount:
                    await conn.execute(
                        "UPDATE tasks SET crashes = crashes + 1 WHERE task_id = ?",
                        (task_id,),
                    )
            await conn.execute(
                "INSERT INTO recovery_provenance (task_id, first_start, execution_id, unsafe_resume, reason) VALUES (?, 'unknown', ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET execution_id = excluded.execution_id, unsafe_resume = excluded.unsafe_resume, reason = excluded.reason",
                (
                    task_id,
                    data["execution_id"],
                    int(data["unsafe_resume"]),
                    data["reason"],
                ),
            )
            now = time.time()
            await conn.execute(
                "UPDATE tasks SET state = ?, card = ?, wait_reason = ?, hold_site = '', "
                "updated_at = ?, finished_at = CASE WHEN ? THEN ? ELSE finished_at END WHERE task_id = ?",
                (
                    data["outcome"],
                    data["card"],
                    data["reason"],
                    now,
                    int(terminal),
                    now,
                    task_id,
                ),
            )
            await conn.execute(
                "UPDATE recovery_operations SET phase = 'committed', generation = ?, pending_operation_id = ?, task_state = ? WHERE task_id = ? AND operation_id = ?",
                (generation, pending_id, data["outcome"], task_id, operation_id),
            )
            await self.store.append_component_event(
                conn,
                events,
                task_id,
                "sched.recovered",
                {
                    "operation_id": operation_id,
                    "execution_id": data["execution_id"],
                    "generation": generation,
                    "state": data["outcome"],
                    "reason": data["reason"],
                },
            )

    async def processes(
        self, task_id: str, *, execution_id: str
    ) -> list[tuple[int, int, str, str]]:
        """Read captured groups; an empty list is unknown, not proof of absence."""
        _text(execution_id, "execution_id")
        async with self.store.component_transaction("recovery_processes") as conn:
            rows = await (
                await conn.execute(
                    "SELECT pgid, leader_pid, boot_id, start_token FROM recovery_processes "
                    "WHERE task_id = ? AND execution_id = ? ORDER BY pgid, leader_pid, boot_id, start_token",
                    (task_id, execution_id),
                )
            ).fetchall()
            return [(int(row[0]), int(row[1]), row[2], row[3]) for row in rows]

    async def record_process(
        self,
        task_id: str,
        *,
        execution_id: str,
        pgid: int,
        leader_pid: int,
        boot_id: str,
        start_token: str,
    ) -> None:
        """Store captured identity, not a claim the group has exited."""
        _text(execution_id, "execution_id")
        _text(boot_id, "boot_id")
        _text(start_token, "start_token")
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0
            for n in (pgid, leader_pid)
        ):
            raise ValueError("process IDs must be positive integers")
        async with self.store.component_transaction("recovery_process") as conn:
            row = await _one(
                conn,
                "SELECT execution_id FROM recovery_provenance WHERE task_id = ?",
                (task_id,),
            )
            if row is None or row[0] != execution_id:
                raise TaskConflictError("stale process execution")
            await conn.execute(
                "INSERT OR IGNORE INTO recovery_processes VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, execution_id, pgid, leader_pid, boot_id, start_token),
            )
