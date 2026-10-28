"""Transactional worker holds and park cards, independent of priority policy."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Row

from localmelo.support.gateway.task_store import (
    TaskConflictError,
    TaskNotFoundError,
    TaskStoreError,
    compare_park_identity,
    read_operation_receipt,
)
from localmelo.support.scheduler.model import TERMINAL_PHASES
from localmelo.support.sessions.store import SessionStore


class ParkStoreError(TaskStoreError):
    """The requested transition is not valid for the current task state."""


@dataclass(frozen=True)
class ParkRecord:
    task_id: str
    state: str
    site: str
    step: int
    agent: str | None
    attempt: int
    reason: str
    card: str


class ParkStore:
    """Commit worker transitions before callers release a held provider.

    Hold coordinates must come from committed runner data supplied by the
    caller. Priority fields are never modified. Operation IDs must be globally
    fresh for the task; receipts may be pruned. A missing identity means no
    pause row exists, not a unique turn: the worker coordinator must fence
    pre-first-pause callbacks and absent-to-absent ABA.
    """

    def __init__(
        self, store: SessionStore, *, clock: Callable[[], float] = time.time
    ) -> None:
        if store.writer != "gateway":
            raise ValueError("park store needs the gateway's store")
        self._store = store
        self._clock = clock

    async def get(self, task_id: str) -> ParkRecord | None:
        async with self._store.component_transaction("park_get") as conn:
            row = await (
                await conn.execute(
                    "SELECT state, hold_site, wait_reason, card, level, reason FROM tasks WHERE task_id = ?",
                    (task_id,),
                )
            ).fetchone()
            if row is None:
                raise TaskNotFoundError(f"No task with id {task_id}")
            return self._record(task_id, row)

    @staticmethod
    def _record(task_id: str, row: Row) -> ParkRecord | None:
        state, raw, reason, card, level, priority_reason = row
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None
            preserved_hold = (
                state == "parked"
                and level == 5
                and priority_reason == "pause"
                and data.get("task_state") == "held"
                and data.get("status") == "held"
            )
            if data.get("task_state") != state and not preserved_hold:
                return None
            return ParkRecord(
                task_id,
                state,
                data["site"],
                data["step"],
                data["agent"],
                data["attempt"],
                data.get("reason", reason),
                card,
            )
        except (ValueError, TypeError, KeyError):
            return None

    async def reconcile(
        self, task_id: str, *, operation_id: str
    ) -> tuple[dict, bool] | None:
        """Read a durable receipt and whether it remains current.

        None is unknown, not proof of rollback: receipts may have been pruned.
        Reconcile only after the original writer has settled. A superseded or
        terminal receipt never authorizes opening a worker fence.
        """
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be non-empty")
        async with self._store.component_transaction("park_reconcile") as conn:
            task = await (
                await conn.execute(
                    "SELECT state FROM tasks WHERE task_id = ?", (task_id,)
                )
            ).fetchone()
            if task is None:
                raise TaskNotFoundError(f"No task with id {task_id}")
            return await read_operation_receipt(conn, task_id, operation_id)

    async def hold(
        self,
        task_id: str,
        *,
        site: str,
        step: int,
        attempt: int,
        reason: str,
        agent: str | None = None,
        expected_generation: int | None = None,
        expected_operation_id: str | None = None,
        operation_id: str | None = None,
    ) -> ParkRecord:
        if site not in {"plan", "reflect", "approval"}:
            raise ValueError("invalid hold site")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
            or not isinstance(reason, str)
            or not reason.strip()
            or (agent is not None and (not isinstance(agent, str) or not agent))
        ):
            raise ValueError("invalid hold coordinates or reason")
        return await self._transition(
            task_id,
            "held",
            site,
            step,
            agent,
            attempt,
            reason,
            "",
            expected_generation=expected_generation,
            expected_operation_id=expected_operation_id,
            operation_id=operation_id,
        )

    async def resume(
        self,
        task_id: str,
        *,
        expected_generation: int | None = None,
        expected_operation_id: str | None = None,
        operation_id: str | None = None,
    ) -> ParkRecord:
        return await self._transition(
            task_id,
            "running",
            expected_generation=expected_generation,
            expected_operation_id=expected_operation_id,
            operation_id=operation_id,
        )

    async def park(
        self,
        task_id: str,
        *,
        card: str,
        expected_generation: int | None = None,
        expected_operation_id: str | None = None,
        operation_id: str | None = None,
    ) -> ParkRecord:
        if not isinstance(card, str) or not card.strip():
            raise ValueError("card must be non-empty")
        return await self._transition(
            task_id,
            "parked",
            card=card,
            expected_generation=expected_generation,
            expected_operation_id=expected_operation_id,
            operation_id=operation_id,
        )

    async def _transition(
        self,
        task_id: str,
        target: str,
        site: str = "",
        step: int = 0,
        agent: str | None = None,
        attempt: int = 0,
        reason: str = "",
        card: str = "",
        *,
        expected_generation: int | None = None,
        expected_operation_id: str | None = None,
        operation_id: str | None = None,
    ) -> ParkRecord:
        if (
            (
                expected_generation is not None
                and (
                    not isinstance(expected_generation, int)
                    or isinstance(expected_generation, bool)
                    or expected_generation < 0
                )
            )
            or (expected_generation is None and expected_operation_id is not None)
            or not isinstance(operation_id, str)
            or not operation_id.strip()
            or (
                expected_operation_id is not None
                and (
                    not isinstance(expected_operation_id, str)
                    or not expected_operation_id.strip()
                )
            )
        ):
            raise ValueError(
                "guard requires expected generation, previous operation ID, and unique operation ID"
            )
        request = {
            "target": target,
            "site": site,
            "step": step,
            "agent": agent,
            "attempt": attempt,
            "reason": reason,
            "card": card,
            "expected_generation": expected_generation,
            "expected_operation_id": expected_operation_id,
        }
        async with self._store.component_event_transaction("park_" + target) as (
            conn,
            events,
        ):
            row = await (
                await conn.execute(
                    "SELECT state, hold_site, wait_reason, card, level, reason FROM tasks WHERE task_id = ?",
                    (task_id,),
                )
            ).fetchone()
            if row is None:
                raise TaskNotFoundError(f"No task with id {task_id}")
            state = row[0]
            if state in TERMINAL_PHASES:
                raise ParkStoreError(f"Task {task_id} is terminal")
            snapshot = await (
                await conn.execute(
                    "SELECT generation, operation_id, pause_pending FROM park_priority WHERE task_id = ?",
                    (task_id,),
                )
            ).fetchone()
            old = self._record(task_id, row)
            receipt = await read_operation_receipt(conn, task_id, operation_id)
            if receipt is not None:
                data, current = receipt
                if data.get("park_request") != request:
                    raise ParkStoreError("Operation ID reused with different arguments")
                if not current or old is None:
                    raise ParkStoreError("Operation superseded")
                return old
            if target == "running" and snapshot is not None and snapshot[2]:
                raise ParkStoreError("Cannot resume while pause is pending")
            phase = json.loads(row[1]).get("status") if old else None
            paused = state == "parked" and row[4] == 5 and row[5] == "pause"
            if (
                old
                and phase == target
                and (state == target or (paused and target == "held"))
            ):
                raise ParkStoreError(
                    "Transition already completed under another operation"
                )
            expected = "running" if target == "held" else "held"
            paused_hold = paused and target == "held" and old is None
            paused_park = (
                paused and target == "parked" and old is not None and phase == "held"
            )
            if not (paused_hold or paused_park) and (
                state != expected
                or (target != "held" and (old is None or phase != "held"))
            ):
                raise ParkStoreError(f"Cannot transition {state} to {target}")
            if target != "held":
                assert old is not None
                site, step, agent, attempt, reason = (
                    old.site,
                    old.step,
                    old.agent,
                    old.attempt,
                    old.reason,
                )
                if target == "running":
                    card = old.card
            task_state = "parked" if paused_hold else target
            record = ParkRecord(
                task_id, task_state, site, step, agent, attempt, reason, card
            )
            metadata = json.dumps(
                {
                    "status": target,
                    "task_state": task_state,
                    "site": site,
                    "step": step,
                    "agent": agent,
                    "attempt": attempt,
                    "reason": reason,
                },
                sort_keys=True,
            )
            await conn.execute(
                "UPDATE tasks SET state = ?, hold_site = ?, wait_reason = ?, card = ?, updated_at = ? WHERE task_id = ?",
                (
                    task_state,
                    metadata,
                    "" if target == "running" else reason,
                    card,
                    self._clock(),
                    task_id,
                ),
            )
            pending_id = (
                operation_id
                if target == "parked" and snapshot is None
                else None if target == "parked" else expected_operation_id
            )
            try:
                new_generation = await compare_park_identity(
                    conn,
                    task_id,
                    expected_generation=expected_generation,
                    expected_operation_id=expected_operation_id,
                    operation_id=pending_id,
                    create=target == "parked",
                )
            except TaskConflictError as exc:
                raise ParkStoreError(str(exc)) from exc
            if target == "parked":
                await conn.execute(
                    "UPDATE park_priority SET pause_pending = 0 WHERE task_id = ?",
                    (task_id,),
                )
            await self._store.append_component_event(
                conn,
                events,
                task_id,
                {
                    "held": "sched.held",
                    "running": "sched.resumed",
                    "parked": "sched.parked",
                }[target],
                {
                    "task_id": task_id,
                    "site": site,
                    "step": step,
                    "agent": agent,
                    "attempt": attempt,
                    "reason": reason,
                    "card": card,
                    "state": task_state,
                    "pause_pending": bool(
                        snapshot and snapshot[2] and target != "parked"
                    ),
                    "park_request": request,
                    "generation": new_generation,
                    "pending_operation_id": pending_id,
                    "operation_id": operation_id,
                },
            )
            return record
