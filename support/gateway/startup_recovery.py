"""Conservative startup classification over durable task and turn records."""

from __future__ import annotations

import uuid

from localmelo.support.gateway.park_card import build_park_card
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.task_store import TaskRecord, TaskStore
from localmelo.support.scheduler.model import TERMINAL_PHASES
from localmelo.support.sessions.store import RUN_STATUSES, SessionStore, StoredEvent


class StartupRecovery:
    """Reconcile persisted tasks before exposing scheduler operations.

    This stage never launches a worker. A later scheduler integration owns
    admission, workspace validation and explicit orphan-risk confirmation.
    """

    def __init__(self, store: SessionStore, tasks: TaskStore) -> None:
        self.store = store
        self.tasks = tasks
        self.recovery = RecoveryStore(store)

    async def run(self) -> None:
        for task in await self.tasks.list_tasks():
            if task.state in TERMINAL_PHASES:
                continue
            intent = await self.recovery.active_intent(task.task_id)
            if intent is not None:
                await self.recovery.commit(task.task_id, operation_id=intent[0])
                task = await self.tasks.get_task(task.task_id)
                if task.state in TERMINAL_PHASES:
                    continue
            await self._classify(task)

    async def _classify(self, task: TaskRecord) -> None:
        provenance = await self.recovery.provenance(task.task_id)
        if task.state == "interrupted":
            return
        if task.state == "queued" and provenance.first_start == "never_started":
            return
        if task.state == "parked" and (
            task.card.strip() or provenance.first_start == "never_started"
        ):
            return
        events: list[StoredEvent] = []
        after = 0
        while True:
            page = await self.store.list_events(task.task_id, after_id=after)
            events.extend(page)
            if len(page) < 1000:
                break
            assert page[-1].id is not None
            after = page[-1].id
        turns = await self.store.list_turns(task.task_id, limit=1)
        if task.state in {"running", "held"} and turns and provenance.execution_id:
            turn = turns[0]
            finished = [
                event
                for event in events
                if event.turn == turn.turn
                and event.event.get("type") == "run.finished"
                and event.event.get("run_id") == provenance.execution_id
            ]
            if (
                turn.run_id == provenance.execution_id
                and not turn.undone
                and turn.status in RUN_STATUSES
                and turn.cause in {"run", "crash"}
                and turn.result
                and finished
                and isinstance(finished[-1].event.get("data"), dict)
                and isinstance(finished[-1].event["data"].get("result"), dict)
                and finished[-1].event["data"]["result"].get("status") == turn.status
            ):
                phase = (
                    "done"
                    if turn.status == "completed"
                    else ("cancelled" if turn.status == "cancelled" else "failed")
                )
                generation, previous = await self.tasks.park_identity(task.task_id)
                operation_id = uuid.uuid4().hex
                await self.recovery.prepare(
                    task.task_id,
                    operation_id=operation_id,
                    execution_id=provenance.execution_id,
                    expected_generation=generation,
                    expected_operation_id=previous,
                    outcome=phase,
                    card=task.card,
                    reason="reconciled completed run",
                    crash=False,
                    unsafe_resume=True,
                )
                await self.recovery.commit(task.task_id, operation_id=operation_id)
                return
        evidence = [
            event
            for event in events
            if event.event.get("type")
            in {
                "turn.started",
                "run.started",
                "tool.started",
                "tool.finished",
                "assistant.message",
                "reflection",
            }
        ]
        card = task.card or (build_park_card(task.spec, evidence) if evidence else "")
        generation, previous = await self.tasks.park_identity(task.task_id)
        operation_id = uuid.uuid4().hex
        await self.recovery.prepare(
            task.task_id,
            operation_id=operation_id,
            execution_id=provenance.execution_id or f"unknown:{task.task_id}",
            expected_generation=generation,
            expected_operation_id=previous,
            outcome="interrupted",
            card=card,
            reason="gateway restarted; earlier effects or processes may still be active",
            crash=provenance.first_start == "started",
            unsafe_resume=True,
        )
        await self.recovery.commit(task.task_id, operation_id=operation_id)
