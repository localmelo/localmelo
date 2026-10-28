"""Transactional worker holds and park cards, independent of priority policy."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Row

from localmelo.support.gateway.task_store import TaskNotFoundError, TaskStoreError
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
    caller. Priority fields are never modified.
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

    async def hold(
        self,
        task_id: str,
        *,
        site: str,
        step: int,
        attempt: int,
        reason: str,
        agent: str | None = None,
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
            task_id, "held", site, step, agent, attempt, reason, ""
        )

    async def resume(self, task_id: str) -> ParkRecord:
        return await self._transition(task_id, "running")

    async def park(self, task_id: str, *, card: str) -> ParkRecord:
        if not isinstance(card, str) or not card.strip():
            raise ValueError("card must be non-empty")
        return await self._transition(task_id, "parked", card=card)

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
    ) -> ParkRecord:
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
            old = self._record(task_id, row)
            phase = json.loads(row[1]).get("status") if old else None
            paused = state == "parked" and row[4] == 5 and row[5] == "pause"
            if (
                old
                and phase == target
                and (state == target or (paused and target == "held"))
            ):
                if target == "held" and (site, step, agent, attempt, reason) != (
                    old.site,
                    old.step,
                    old.agent,
                    old.attempt,
                    old.reason,
                ):
                    raise ParkStoreError("hold conflicts with the current hold")
                if target == "parked" and card != old.card:
                    raise ParkStoreError("park conflicts with the current card")
                return old
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
                },
            )
            return record
