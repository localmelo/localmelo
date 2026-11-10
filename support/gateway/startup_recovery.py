"""The start decision for every unfinished task, from durable task and turn records."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from localmelo.support.gateway.park_card import build_park_card
from localmelo.support.gateway.process_identity import gone
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.resume_evidence import (
    Process,
    ResumeRefusedError,
    collect_evidence,
    in_flight,
)
from localmelo.support.gateway.task_store import (
    RESTART,
    TaskRecord,
    TaskStore,
    TaskStoreError,
)
from localmelo.support.scheduler.model import TERMINAL_PHASES
from localmelo.support.sessions.store import (
    RUN_STATUSES,
    SessionStore,
    SessionStoreError,
    StoredEvent,
    TurnInfo,
)

log = logging.getLogger(__name__)

FENCED_REASON = (
    "this task could not be restored after the restart; "
    "check its work before resuming it"
)
# A run a stop cut off, written by the stop itself: the next start decides it.
STOPPED_REASON = "localmelo stopped; this task continues when it opens again"
# Something may be half done: a call that can change something had not
# returned, or a process of the run may still be running.
IN_FLIGHT_REASON = "gateway restarted; earlier effects or processes may still be active"
# The person asked for a pause that had not settled when localmelo stopped.
PAUSED_REASON = "paused before localmelo stopped; resume it to go on"
# The run was cut off by a crash the time before too, with no turn finished since.
CRASHED_REASON = (
    "localmelo stopped unexpectedly twice while this task ran; "
    "check its work before continuing it"
)

Outcome = Literal["keep", "reconcile", "continue", "wait"]


@dataclass(frozen=True)
class RestartFacts:
    """What the records say of one unfinished task at start.

    ``waiting``: it already waits for the person (``interrupted`` by an earlier
    decision). ``continued``: a Resume or continue was committed and its turn
    has not begun. ``by_person``: a pause the person asked for (parked, or
    still pending), ``settled`` when it parked with the card the park wrote.
    ``finished``: its last run finished completely. ``crash``: it was found
    live, so no stop wrote it down.
    ``crashed_before``: a crash cut it before, with no finished turn since.
    ``in_flight``: something may be half done (see ``resume_evidence.in_flight``).
    """

    state: str
    never_started: bool = False
    waiting: bool = False
    continued: bool = False
    by_person: bool = False
    settled: bool = False
    finished: bool = False
    crash: bool = False
    crashed_before: bool = False
    in_flight: bool = True


def restart_outcome(facts: RestartFacts) -> Outcome:
    """The one start decision for a task, from its recorded facts."""
    if facts.state in TERMINAL_PHASES or facts.waiting or facts.continued:
        return "keep"
    if facts.never_started and facts.state in ("queued", "parked"):
        return "keep"  # starts by itself, or stays paused, as it was
    if facts.finished:
        return "reconcile"
    if facts.by_person:
        # The person paused it, or asked to: it never goes on by itself.
        return "keep" if facts.settled else "wait"
    if facts.in_flight or (facts.crash and facts.crashed_before):
        return "wait"
    return "continue"


def _store_failed(exc: Exception) -> bool:
    """Whether *exc* says the store itself failed, not one task's rows."""
    if isinstance(exc, SessionStoreError):
        return not isinstance(exc, TaskStoreError)
    return isinstance(exc, sqlite3.Error | OSError)


class StartupRecovery:
    """Decide every unfinished task once at start, from durable records.

    A task cut off with nothing in flight is continued here, by the same store
    operation as Resume; one that may have something half done waits for the
    person. This stage never launches a worker: ``TaskWorkers.open`` admits what
    it queued.
    """

    def __init__(
        self,
        store: SessionStore,
        tasks: TaskStore,
        *,
        verifier: Callable[[Process, str], bool] = gone,
    ) -> None:
        self.store = store
        self.tasks = tasks
        self.recovery = RecoveryStore(store)
        self.verifier = verifier

    async def run(self) -> None:
        """Decide every open task; one that cannot be decided is fenced.

        A failure of the store itself still stops the whole stage.
        """
        for task in await self.tasks.list_tasks():
            try:
                await self._recover(task)
            except Exception as exc:
                if _store_failed(exc):
                    raise
                log.exception("Task %s could not be recovered at startup", task.task_id)
                await self.recovery.fence(task.task_id, reason=FENCED_REASON)

    async def _recover(self, task: TaskRecord) -> None:
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
        if task.state not in TERMINAL_PHASES:
            await self._classify(task)

    async def _events(self, task_id: str) -> list[StoredEvent]:
        events: list[StoredEvent] = []
        after = 0
        while True:
            page = await self.store.list_events(task_id, after_id=after)
            events.extend(page)
            if len(page) < 1000:
                return events
            assert page[-1].id is not None
            after = page[-1].id

    async def _in_flight(self, task: TaskRecord) -> bool:
        try:
            evidence = await collect_evidence(
                self.tasks, self.recovery, task, verifier=self.verifier
            )
        except ResumeRefusedError:
            return True  # contradictory identity: unknown is in flight
        return in_flight(evidence)

    async def _classify(self, task: TaskRecord) -> None:
        provenance = await self.recovery.provenance(task.task_id)
        never_started = provenance.first_start == "never_started"
        cheap = RestartFacts(
            state=task.state,
            never_started=never_started,
            waiting=task.state == "interrupted" and task.wait_reason != STOPPED_REASON,
            continued=task.state == "queued"
            and await self.tasks.resume_waiting_turn(task.task_id),
        )
        if restart_outcome(cheap) == "keep":
            return  # decided by what the task row says: no event is read
        events = await self._events(task.task_id)
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
        last = parks[-1].event if parks else {}
        data = last.get("data") if last.get("type") == "sched.parked" else None
        settled = (
            task.state == "parked"
            and bool(task.card.strip())
            and isinstance(data, dict)
            and data.get("card") == task.card
        )
        turns = await self.store.list_turns(task.task_id, limit=1)
        finished = (
            task.state in {"running", "held"}
            and bool(turns)
            and provenance.execution_id is not None
            and _finished(turns[0], events, provenance.execution_id)
        )
        crash = task.state in {"running", "held"}
        facts = RestartFacts(
            state=task.state,
            never_started=never_started,
            waiting=cheap.waiting,
            continued=cheap.continued,
            by_person=(
                task.state == "parked" or await self.tasks.pause_pending(task.task_id)
            )
            and not _scheduler_park(events),
            settled=settled,
            finished=finished,
            crash=crash,
            crashed_before=task.crashes > 0,
            in_flight=await self._in_flight(task),
        )
        outcome = restart_outcome(facts)
        if outcome == "keep":
            return
        if outcome == "reconcile":
            turn = turns[0]
            phase = (
                "done"
                if turn.status == "completed"
                else ("cancelled" if turn.status == "cancelled" else "failed")
            )
            await self._commit(
                task,
                execution_id=provenance.execution_id or "",
                outcome=phase,
                card=task.card,
                reason="reconciled completed run",
                crash=False,
            )
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
            task.card
            if settled
            else (
                build_park_card(task.spec, evidence)
                if evidence
                else ("" if task.state == "parked" else task.card)
            )
        )
        # A legacy launch lacks provenance; the durable turn identifies it even
        # when no run.started event was recorded before the restart.
        abandoned_turn = turns[0] if turns and crash else None
        execution_id = provenance.execution_id or (
            f"turn:{abandoned_turn.id}" if abandoned_turn else f"unknown:{task.task_id}"
        )
        if outcome == "continue":
            reason = STOPPED_REASON
        elif facts.by_person:
            reason = PAUSED_REASON
        elif facts.crash and facts.crashed_before and not facts.in_flight:
            reason = CRASHED_REASON
        else:
            reason = IN_FLIGHT_REASON
        if not (outcome == "continue" and settled):
            # A settled park continues from its own card.
            await self._commit(
                task,
                execution_id=execution_id,
                outcome="interrupted",
                card=card,
                reason=reason,
                crash=crash,
            )
        if outcome == "continue":
            generation, previous = await self.tasks.park_identity(task.task_id)
            await self.tasks.resume_recovered(
                task.task_id,
                operation_id=uuid.uuid4().hex,
                expected_generation=generation,
                expected_operation_id=previous,
                expected_execution_id=execution_id,
                confirmed=False,
                first_start=False,
                actor=RESTART,
                because=RESTART,
            )

    async def _commit(
        self,
        task: TaskRecord,
        *,
        execution_id: str,
        outcome: str,
        card: str,
        reason: str,
        crash: bool,
    ) -> None:
        generation, previous = await self.tasks.park_identity(task.task_id)
        operation_id = uuid.uuid4().hex
        await self.recovery.prepare(
            task.task_id,
            operation_id=operation_id,
            execution_id=execution_id,
            expected_generation=generation,
            expected_operation_id=previous,
            outcome=outcome,
            card=card,
            reason=reason,
            crash=crash,
            unsafe_resume=True,
        )
        await self.recovery.commit(task.task_id, operation_id=operation_id)


def _finished(turn: TurnInfo, events: list[StoredEvent], execution_id: str) -> bool:
    """Whether *turn* is a run of *execution_id* that finished completely."""
    found = [
        event
        for event in events
        if event.turn == turn.turn and event.event.get("type") == "run.finished"
    ]
    if not found:
        return False
    data = found[-1].event.get("data")
    return (
        turn.client_request_id == execution_id
        and bool(turn.run_id)
        and not turn.undone
        and turn.status in RUN_STATUSES
        and turn.cause in {"run", "crash", "user_stop"}
        and bool(turn.result)
        and found[-1].event.get("run_id") == turn.run_id
        and isinstance(data, dict)
        and isinstance(data.get("result"), dict)
        and data["result"].get("status") == turn.status
    )


def _scheduler_park(events: list[StoredEvent]) -> bool:
    """Whether the newest pause was the scheduler's, making room for another task."""
    for event in reversed(events):
        if event.event.get("type") == "sched.pause_requested":
            data = event.event.get("data")
            request = data.get("task_request") if isinstance(data, dict) else None
            return isinstance(request, dict) and request.get("actor") == "scheduler"
    return False
