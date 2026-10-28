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
            intent = await self.recovery.active_intent(task.task_id)
            if intent is not None:
                receipt = await self.recovery.receipt(task.task_id, intent[0])
                if receipt is None:
                    raise RuntimeError("prepared recovery receipt is missing")
                if receipt[1] and task.state not in TERMINAL_PHASES:
                    await self.recovery.commit(task.task_id, operation_id=intent[0])
                else:
                    generation, previous = await self.tasks.park_identity(task.task_id)
                    await self.recovery.supersede(
                        task.task_id,
                        intent[0],
                        expected_generation=generation,
                        expected_operation_id=previous,
                    )
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
        if task.state == "parked" and provenance.first_start == "never_started":
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
        if task.state == "parked" and task.card.strip():
            parks = [
                event
                for event in events
                if event.event.get("type")
                in {
                    "sched.parked",
                    "sched.resumed",
                    "sched.held",
                    "sched.state_changed",
                    "turn.started",
                }
            ]
            if parks and parks[-1].event.get("type") == "sched.parked":
                data = parks[-1].event.get("data")
                if isinstance(data, dict) and data.get("card") == task.card:
                    return
        turns = await self.store.list_turns(task.task_id, limit=1)
        if task.state in {"running", "held"} and turns and provenance.execution_id:
            turn = turns[0]
            finished = [
                event
                for event in events
                if event.turn == turn.turn and event.event.get("type") == "run.finished"
            ]
            if (
                turn.client_request_id == provenance.execution_id
                and bool(turn.run_id)
                and not turn.undone
                and turn.status in RUN_STATUSES
                and turn.cause in {"run", "crash"}
                and turn.result
                and finished
                and finished[-1].event.get("run_id") == turn.run_id
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
        card = (
            build_park_card(task.spec, evidence)
            if evidence
            else ("" if task.state == "parked" else task.card)
        )
        generation, previous = await self.tasks.park_identity(task.task_id)
        operation_id = uuid.uuid4().hex
        # A legacy launch lacks provenance; the durable turn identifies it even
        # when no run.started event was recorded before the restart.
        abandoned_turn = (
            turns[0] if turns and task.state in {"running", "held"} else None
        )
        execution_id = provenance.execution_id or (
            f"turn:{abandoned_turn.id}" if abandoned_turn else f"unknown:{task.task_id}"
        )
        await self.recovery.prepare(
            task.task_id,
            operation_id=operation_id,
            execution_id=execution_id,
            expected_generation=generation,
            expected_operation_id=previous,
            outcome="interrupted",
            card=card,
            reason="gateway restarted; earlier effects or processes may still be active",
            crash=provenance.first_start == "started" or abandoned_turn is not None,
            unsafe_resume=True,
        )
        await self.recovery.commit(task.task_id, operation_id=operation_id)
