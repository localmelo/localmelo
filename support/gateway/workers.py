"""In-process task workers for the gateway's scheduler."""

from __future__ import annotations

import asyncio
import builtins
import contextvars
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict
from typing import Any, cast

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.providers import (
    LLM_REQUEST_ATTEMPT,
    LLM_REQUEST_KIND,
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
from localmelo.melo.executor.process_gate import PROCESS_RECORD
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import (
    ChatGate,
    GateCall,
    GatedEmbedding,
    GatedLLM,
    SessionClient,
)
from localmelo.support.gateway.park_card import build_park_card
from localmelo.support.gateway.park_store import ParkStore
from localmelo.support.gateway.process_identity import gone as process_gone
from localmelo.support.gateway.process_identity import identity as process_identity
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.resume_evidence import (
    ConfirmationRequiredError,
    RecoveryUnknownError,
    ResumeRefusedError,
    StaleConfirmationError,
    collect_evidence,
    decide,
)
from localmelo.support.gateway.resume_tokens import ResumeBinding, ResumeTokens
from localmelo.support.gateway.startup_recovery import StartupRecovery
from localmelo.support.gateway.task_policy import (
    TaskSpec,
    WorkspaceError,
    validate_task_workspace,
    worker_tool_allowlist,
)
from localmelo.support.gateway.task_store import (
    TaskConflictError,
    TaskNotFoundError,
    TaskNotPausedError,
    TaskRecord,
    TaskStore,
    TaskStoreError,
)
from localmelo.support.gateway.worker_evidence import (
    SafePointEvidence,
    committed_evidence,
)
from localmelo.support.scheduler.model import TERMINAL_PHASES, LevelSource, Phase, Task
from localmelo.support.sessions import display
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.approvals import ApprovalBroker, RememberingPolicy
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionStore, SessionStoreError

log = logging.getLogger(__name__)
MAX_WORKERS = 4
_CAPTURE_RETRY_SECONDS = 10.0


class _WorkerLLM(GatedLLM):
    def __init__(
        self,
        inner: BaseLLMProvider,
        client: SessionClient,
        ready: asyncio.Event,
        runner: Callable[[], SessionRunner | None],
        hold: Callable[[GateCall, SessionRunner, str, str], Awaitable[None]],
        release: Callable[[GateCall, SessionRunner, str], Awaitable[None]],
        started: Callable[[GateCall], None],
    ) -> None:
        super().__init__(inner, client)
        self.ready = ready
        self.runner = runner
        self.hold = hold
        self.release = release
        self.started = started

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        runner = self.runner()
        turn = runner.current_turn if runner is not None else None
        if turn is None or runner is None:
            raise RuntimeError("Worker call has no active turn")
        gate = self.client.gate
        kind = LLM_REQUEST_KIND.get()
        call = gate.submit(
            self.client.session_id,
            kind=kind,
            attempt=LLM_REQUEST_ATTEMPT.get(),
            turn_id=turn.id,
        )
        held = kind in ("plan", "reflect") and (
            call.meta.admitted_mono_us is None or not self.ready.is_set()
        )
        try:
            if held:
                self.ready.clear()
                await self.hold(call, runner, turn.id, kind)
            await gate.acquire(call)
            if held:
                await self.release(call, runner, turn.id)
            await self.ready.wait()
            if self.runner() is not runner or runner.current_turn != turn:
                raise RuntimeError("Worker turn changed before provider admission")
            self.started(call)
        except BaseException:
            gate.finish(call, "refused")
            raise
        try:
            message = await self.inner.chat(messages, tools=tools)
        except BaseException:
            gate.finish(call, "failed")
            raise
        self._copy()
        gate.finish(call, "ok", self.last_usage)
        return message


class _WorkerEmbedding(GatedEmbedding):
    def __init__(
        self, inner: BaseEmbeddingProvider, client: SessionClient, ready: asyncio.Event
    ) -> None:
        super().__init__(inner, client)
        self.ready = ready

    async def embed(self, texts: list[str]) -> list[list[float]]:
        await self.ready.wait()
        return await super().embed(texts)


class TaskWorkers:
    """Own only workers created or explicitly resumed during this process."""

    def __init__(
        self,
        store: SessionStore,
        gate: ChatGate,
        settings: Callable[[], Config],
        hub: SessionHub,
        broker: ApprovalBroker | None = None,
    ) -> None:
        self.store = store
        self.gate = gate
        self.settings = settings
        self.hub = hub
        self.broker = broker or ApprovalBroker()
        self.tasks = TaskStore(store)
        self.parks = ParkStore(store)
        self._lock = asyncio.Lock()
        self._eligible: set[str] = set()
        self._restart_fenced: set[str] = set()
        self._restart_first: set[str] = set()
        self._resume_tokens = ResumeTokens()
        self._process_verifier: Callable[[tuple[int, int, str, str], str], bool] = (
            process_gone
        )
        self._pending_execution: dict[str, str] = {}
        self._runs: dict[str, asyncio.Task[None]] = {}
        self._runners: dict[str, SessionRunner] = {}
        self._stopping: set[str] = set()
        self._invalid_workspace: set[str] = set()
        self._ready: dict[str, asyncio.Event] = {}
        self._points: dict[str, tuple[GateCall, SessionRunner, str]] = {}
        self._approval_points: dict[
            str, tuple[SessionRunner, str, SafePointEvidence]
        ] = {}
        self._pause_requested: set[str] = set()
        self._park_winners: set[str] = set()
        self._unavailable: set[str] = set()
        self._failure_events: dict[str, asyncio.Event] = {}
        self._quarantined: set[str] = set()
        self._shutdown = asyncio.Event()
        self._closed = False
        self._opened = False

    def is_active(self, task_id: str) -> bool:
        """Whether this process may still run or start this worker."""
        return task_id in self._eligible or task_id in self._runs

    async def open(self) -> None:
        if not self._opened:
            await self.tasks.open()
            try:
                await StartupRecovery(self.store, self.tasks).run()
                self._restart_fenced = set()
                self._eligible.difference_update(self._restart_first)
                self._restart_first.clear()
                for record in await self.tasks.list_tasks():
                    if record.state in TERMINAL_PHASES:
                        continue
                    provenance = await RecoveryStore(self.store).provenance(
                        record.task_id
                    )
                    if (
                        record.state == "queued"
                        and record.level < 5
                        and not record.card
                        and provenance.first_start == "never_started"
                    ):
                        self._restart_first.add(record.task_id)
                        self._register(record)
                        self._eligible.add(record.task_id)
                    elif (
                        record.state == "queued"
                        and record.level < 5
                        and await self.tasks.resume_waiting_turn(record.task_id)
                    ):
                        # A Resume committed before the restart whose turn never
                        # started is re-admitted, not fenced for another review.
                        self._register(record)
                        self._eligible.add(record.task_id)
                    elif (
                        record.state == "interrupted"
                        and record.level < 5
                        and await self.tasks.resume_waiting_turn(record.task_id)
                    ):
                        # The crash came after the task reached running but
                        # before the resumed turn began: startup interrupted
                        # it, and the committed Resume is queued again.
                        record = await self.tasks.change(
                            record.task_id, state="queued", actor="scheduler"
                        )
                        self._register(record)
                        self._eligible.add(record.task_id)
                    else:
                        self._restart_fenced.add(record.task_id)
            except Exception as exc:
                raise SessionStoreError("startup recovery unavailable") from exc
            async with self._lock:
                await self._schedule()
                self.store.add_retention_hook(self._reconcile)
                self._opened = True

    async def create(self, payload: dict) -> tuple[dict, bool]:
        if not isinstance(payload, dict):
            raise ValueError("task must be an object")
        unknown = set(payload) - {
            "objective",
            "prompt",
            "title",
            "workspace",
            "level",
            "deadline",
            "request_key",
        }
        if unknown:
            raise ValueError(f"unknown task fields: {', '.join(sorted(unknown))}")
        clean = dict(payload)
        key = clean.pop("request_key", None)
        if "deadline" in clean:
            raise ValueError("deadline is not supported yet")
        level = clean.get("level")
        if (
            isinstance(level, str)
            and len(level) == 2
            and level[0].lower() == "p"
            and level[1] in "012345"
        ):
            clean["level"] = int(level[1])
        spec = TaskSpec.from_user_payload(clean)
        task_id = uuid.uuid4().hex[:12]
        async with self._lock:
            if self._closed:
                raise RuntimeError("workers are closed")
            existing = await self.tasks.find_existing_request(
                spec, origin="user", request_key=key
            )
            if existing is not None:
                return await self._body(existing), True
            workspace = validate_task_workspace(
                spec.workspace, self.settings(), origin="user", task_id=task_id
            )
            record, repeated = await self.tasks.create_task(
                spec,
                origin="user",
                workspace=workspace,
                request_key=key,
                task_id=task_id,
                _recovery_proof=True,
            )
            if not repeated:
                self._register(record)
                self._eligible.add(record.task_id)
                if record.level == 5:
                    record = await self.tasks.change(
                        record.task_id, state="parked", actor="user"
                    )
                await self._schedule()
            return await self._body(record), repeated

    async def list(
        self, state: str | None = None, level: int | None = None
    ) -> list[dict]:
        records = await self.tasks.list_tasks()
        return [
            await self._body(r)
            for r in records
            if (state is None or r.state == state)
            and (level is None or r.level == level)
        ]

    async def get(self, task_id: str) -> dict:
        return await self._body(await self.tasks.get_task(task_id))

    async def update(
        self, task_id: str, level: object = None, state: str | None = None
    ) -> dict:
        if state is not None and state not in ("pause", "resume", "stop"):
            raise ValueError("state must be pause, resume or stop")
        if level is None and state is None:
            raise ValueError("level or state is required")
        if isinstance(level, str):
            level = level.strip().lower()
            if level == "first":
                raise ValueError("first is not supported yet")
            if level != "auto" and level not in {f"p{i}" for i in range(6)}:
                raise ValueError("level must be P0-P5 or auto")
        elif level is not None and (
            isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 5
        ):
            raise ValueError("level must be P0-P5 or auto")
        if task_id in self._unavailable:
            raise SessionStoreError("store unavailable")
        if task_id in self._quarantined and state != "stop":
            raise TaskStoreError("worker cleanup failed; slot held until restart")
        is_p5 = level == 5 or level == "p5"
        if state == "pause" and level is not None:
            raise ValueError("pause cannot be combined with a level")
        if state == "resume" and is_p5:
            raise ValueError("choose a runnable priority to resume a P5 task")
        if is_p5 and state != "stop":
            state, level = "pause", None
        if (state == "pause" or is_p5) and task_id in self._runs:
            self._pause_requested.add(task_id)
            self._ready[task_id].clear()
        if state == "stop" and task_id in self._runs:
            self._stopping.add(task_id)
            self._ready[task_id].clear()
            runner = self._runners.get(task_id)
            if runner is not None:
                runner.cancel(cause="user_stop")
        if state == "resume" and task_id in self._park_winners:
            cleanup = self._runs.get(task_id)
            if cleanup is not None:
                failure = asyncio.create_task(
                    self._failure_events.setdefault(task_id, asyncio.Event()).wait()
                )
                try:
                    done, _ = await asyncio.wait(
                        {cleanup, failure}, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    failure.cancel()
                    await asyncio.gather(failure, return_exceptions=True)
                if (
                    failure in done
                    or task_id in self._unavailable
                    or task_id in self._quarantined
                ):
                    raise SessionStoreError("store unavailable")
        async with self._lock:
            if state == "stop":
                # Stop wins: an old confirmation may not resume a task it
                # previewed before this transition.
                self._resume_tokens.invalidate(task_id)
            if task_id in self._unavailable:
                raise SessionStoreError("store unavailable")
            if task_id in self._quarantined and state != "stop":
                raise TaskStoreError("worker cleanup failed; slot held until restart")
            current = await self.tasks.get_task(task_id)
            if task_id in self._restart_fenced and state != "stop":
                raise TaskStoreError("restart recovery requires explicit safety review")
            if current.state in TERMINAL_PHASES:
                raise TaskStoreError("task is finished")
            if task_id in self._quarantined:
                generation, previous = await self.tasks.park_identity(task_id)
                args: dict[str, Any] = dict(
                    state="cancelled",
                    actor="user",
                    expected_generation=generation,
                    expected_operation_id=previous,
                    operation_id=uuid.uuid4().hex,
                )
                try:
                    record = await self.tasks.change(task_id, **args)
                except (Exception, asyncio.CancelledError):
                    record = await self.tasks.change(task_id, **args)
                self._eligible.discard(task_id)
                self.gate.finish_task(task_id, "cancelled")
                return await self._body(record)
            pending_resume = False
            if state == "pause" and current.level == 5:
                async with self.store.component_transaction("worker_pending") as conn:
                    pending = await (
                        await conn.execute(
                            "SELECT pause_pending FROM park_priority WHERE task_id = ?",
                            (task_id,),
                        )
                    ).fetchone()
                if pending and pending[0]:
                    if current.state == "held":
                        self._select_park(task_id)
                    return await self._body(current)
            if state == "resume" and current.state not in ("parked", "interrupted"):
                async with self.store.component_transaction("worker_pending") as conn:
                    row = await (
                        await conn.execute(
                            "SELECT pause_pending FROM park_priority WHERE task_id = ?",
                            (task_id,),
                        )
                    ).fetchone()
                pending_resume = bool(row and row[0])
                if not pending_resume:
                    raise TaskNotPausedError("task is not paused")
            if state == "stop" and task_id in self._runs:
                # Set the stop flag under the same lock Resume holds, so a
                # worker admitted by a Resume that finished committing just
                # before this Stop cannot start its turn.
                self._stopping.add(task_id)
                self._ready[task_id].clear()
                runner = self._runners.get(task_id)
                if runner is not None:
                    runner.cancel(cause="user_stop")
            if state == "resume" or (
                current.level == 5
                and level is not None
                and not is_p5
                and state != "stop"
            ):
                try:
                    await self._workspace(task_id)
                except WorkspaceError:
                    self._register(current)
                    record = await self.tasks.change(
                        task_id,
                        state="failed",
                        actor="scheduler",
                        failure_reason="invalid_workspace",
                    )
                    self._eligible.discard(task_id)
                    self._invalid_workspace.add(task_id)
                    self._ready[task_id].set()
                    runner = self._runners.get(task_id)
                    if runner is not None:
                        runner.cancel(cause="user_stop")
                    self.gate.finish_task(task_id, "failed")
                    if task_id not in self._runs:
                        self._invalid_workspace.discard(task_id)
                        self._ready.pop(task_id, None)
                    await self._schedule()
                    return await self._body(record)
            if state in ("pause", "resume") and level is None:
                self._register(current)
                if state == "resume" and current.state == "interrupted":
                    generation, previous = await self.tasks.park_identity(task_id)
                    record = await self.tasks.resume_interrupted(
                        task_id,
                        expected_generation=generation,
                        expected_operation_id=previous,
                        operation_id=uuid.uuid4().hex,
                    )
                elif state == "resume" and pending_resume:
                    generation, previous = await self.tasks.park_identity(task_id)
                    record = await self.tasks.cancel_pending_pause(
                        task_id,
                        expected_generation=generation,
                        expected_operation_id=previous,
                        operation_id=uuid.uuid4().hex,
                    )
                    if record.state == "held":
                        generation, previous = await self.tasks.park_identity(task_id)
                        await self.parks.resume(
                            task_id,
                            expected_generation=generation,
                            expected_operation_id=previous,
                            operation_id=uuid.uuid4().hex,
                        )
                        record = await self.tasks.get_task(task_id)
                elif (
                    state == "pause"
                    and task_id in self._runners
                    and self._runners[task_id].busy
                ):
                    generation, previous = await self.tasks.park_identity(task_id)
                    record = await self.tasks.request_pending_pause(
                        task_id,
                        expected_generation=generation,
                        expected_operation_id=previous,
                        operation_id=uuid.uuid4().hex,
                    )
                    if record.state != "held" and task_id in self._approval_points:
                        evidence = self._approval_points[task_id][2]
                        generation, previous = await self.tasks.park_identity(task_id)
                        await self.parks.hold(
                            task_id,
                            site="approval",
                            step=evidence.step,
                            attempt=evidence.attempt,
                            reason="approval",
                            expected_generation=generation,
                            expected_operation_id=previous,
                            operation_id=uuid.uuid4().hex,
                        )
                        record = await self.tasks.get_task(task_id)
                    if record.state != "held" and task_id in self._points:
                        prior = await self.parks.get(task_id)
                        if prior is not None and prior.state == "running":
                            generation, previous = await self.tasks.park_identity(
                                task_id
                            )
                            await self.parks.hold(
                                task_id,
                                site=prior.site,
                                step=prior.step,
                                attempt=prior.attempt,
                                reason=prior.reason,
                                agent=prior.agent,
                                expected_generation=generation,
                                expected_operation_id=previous,
                                operation_id=uuid.uuid4().hex,
                            )
                            record = await self.tasks.get_task(task_id)
                    if record.state == "held":
                        self._select_park(task_id)
                else:
                    record = await (
                        self.tasks.pause(task_id)
                        if state == "pause"
                        else self.tasks.resume(task_id)
                    )
                if state == "resume" and record.level != 5 and task_id in self._runs:
                    # TaskStore.change adopts the live capture of a
                    # running/held -> running transition. A resume after a
                    # full park goes back through _run, which launches a new
                    # execution with capture incomplete.
                    record = await self.tasks.change(
                        task_id,
                        state="running",
                        actor="scheduler",
                    )
                if state == "resume":
                    self._pause_requested.discard(task_id)
                restored: int | str = (
                    record.level if record.level_source == "user" else "auto"
                )
                self.gate.set_task_level(task_id, restored, rank=record.user_rank)
                if record.level == 5:
                    self._ready[task_id].clear()
                else:
                    self._eligible.add(task_id)
                    if task_id not in self._stopping:
                        self._ready[task_id].set()
                await self._schedule()
                return await self._body(record)
            parked = is_p5
            runnable = "running" if task_id in self._runs else "queued"
            desired = (
                "parked"
                if parked
                else (
                    runnable
                    if state == "resume" or (level is not None and current.level == 5)
                    else None
                )
            )
            if state == "stop" and task_id not in self._runs:
                desired = "cancelled"
            record = await self.tasks.change(
                task_id,
                priority=level,
                state=desired,
                actor="user",
            )
            self._register(current)
            if level is not None:
                self.gate.set_task_level(
                    task_id, cast(int | str, level), rank=record.user_rank
                )
            if state == "stop":
                self._eligible.discard(task_id)
                self._ready[task_id].set()
                if task_id not in self._runs:
                    self.gate.finish_task(task_id, "cancelled")
                    self._ready.pop(task_id, None)
            elif parked:
                self._ready[task_id].clear()
                self.gate.update_task(task_id, paused=True)
            elif state == "resume" or (level is not None and current.level == 5):
                self._pause_requested.discard(task_id)
                self._eligible.add(task_id)
                self.gate.update_task(task_id, paused=False)
                self._ready[task_id].set()
            await self._schedule()
            return await self._body(record)

    async def resume(
        self,
        task_id: str,
        *,
        operation_id: str,
        confirmation_token: str | None = None,
    ) -> dict:
        """Resume one recovered task through the evidence decision table.

        Without a token, an unverifiable execution raises
        :class:`ConfirmationRequiredError` and changes nothing. A token authorizes
        exactly the previewed evidence and CAS generation; an accepted
        transition consumes it once. Stop invalidates it and wins without
        replaying a delayed retry.
        """
        if (
            not isinstance(operation_id, str)
            or not operation_id.strip()
            or len(operation_id) > 200
        ):
            raise ValueError("operation_id must be a non-empty string")
        if confirmation_token is not None and (
            not isinstance(confirmation_token, str) or not confirmation_token
        ):
            raise ValueError("confirmation_token must be a non-empty string")
        async with self._lock:
            if self._closed or task_id in self._unavailable:
                raise SessionStoreError("store unavailable")
            current = await self.tasks.get_task(task_id)
            if await self.tasks.has_resume_receipt(task_id, operation_id):
                # A matching retry returns the current row without a second
                # turn, and re-admits a queue this process committed but did
                # not finish registering (for example after a cancelled commit).
                if current.state == "interrupted":
                    # The restart interrupted the committed queue. Re-admit it
                    # only when its turn never started; otherwise the work the
                    # operation promised has begun and a retry must not replay.
                    turn = await self.tasks.resume_turn(task_id, operation_id)
                    if turn is None or await self.tasks.turn_started(
                        task_id, turn=turn
                    ):
                        raise TaskConflictError(
                            "The resumed turn already started; this operation "
                            "cannot be replayed."
                        )
                    current = await self.tasks.change(
                        task_id, state="queued", actor="scheduler"
                    )
                if (
                    current.state == "queued"
                    and current.level < 5
                    and task_id not in self._quarantined
                    and task_id not in self._stopping
                ):
                    self._restart_fenced.discard(task_id)
                    self._restart_first.discard(task_id)
                    self._invalid_workspace.discard(task_id)
                    self._pause_requested.discard(task_id)
                    self._register(current)
                    self._eligible.add(task_id)
                    self._ready[task_id].set()
                    self.gate.update_task(task_id, paused=False)
                    await self._schedule()
                return await self._body(current)
            if task_id in self._quarantined:
                raise RecoveryUnknownError(
                    "Resume unavailable: the previous worker's cleanup has not "
                    "settled; Stop stays available.",
                    reason="cleanup_unresolved",
                )
            if task_id in self._stopping:
                raise ResumeRefusedError(
                    "Resume unavailable: Stop is in progress.", reason="stopping"
                )
            if confirmation_token is not None:
                grant = self._resume_tokens.binding_for(confirmation_token)
                if grant is None or grant.task_id != task_id:
                    raise StaleConfirmationError(
                        "The confirmation expired, was already used, or belongs "
                        "to another task."
                    )
            if await self.tasks.operation_receipt(task_id, operation_id) is not None:
                raise RecoveryUnknownError(
                    "Resume unavailable: this operation ID belongs to another transition."
                )
            recovery = RecoveryStore(self.store)
            if await recovery.receipt(task_id, operation_id) is not None:
                raise RecoveryUnknownError(
                    "Resume unavailable: this operation ID belongs to another recovery operation."
                )
            if await recovery.active_intent(task_id) is not None:
                raise RecoveryUnknownError()
            if current.state in TERMINAL_PHASES:
                raise ResumeRefusedError(
                    f"Task {task_id} is finished ({current.state}) and cannot be resumed.",
                    reason="terminal",
                )
            if current.state not in {"interrupted", "parked"}:
                raise ResumeRefusedError(
                    f"Task {task_id} is not waiting for recovery.",
                    reason="invalid_state",
                )
        # Reading the evidence is the slow part: workspace, five store reads
        # and one liveness check per recorded group. Take it outside the worker
        # lock so capture and Stop are not delayed by a large evidence set; the
        # commit below re-checks Stop and CASes the evidence generation.
        workspace_valid = await self._resume_workspace(task_id)
        evidence = await collect_evidence(
            self.tasks, recovery, current, verifier=self._process_verifier
        )
        decision = decide(evidence, workspace_valid=workspace_valid)
        binding = ResumeBinding(
            task_id=task_id,
            execution_id=evidence.execution_id,
            evidence_key=evidence.evidence_key,
            generation=evidence.generation,
            pending_operation_id=evidence.pending_operation_id,
        )
        async with self._lock:
            if self._closed or task_id in self._unavailable:
                raise SessionStoreError("store unavailable")
            if task_id in self._stopping:
                raise ResumeRefusedError(
                    "Resume unavailable: Stop is in progress.", reason="stopping"
                )
            if task_id in self._quarantined:
                raise RecoveryUnknownError(
                    "Resume unavailable: the previous worker's cleanup has not "
                    "settled; Stop stays available.",
                    reason="cleanup_unresolved",
                )
            if decision.kind == "confirm":
                if confirmation_token is None:
                    token = self._resume_tokens.issue(binding)
                    raise ConfirmationRequiredError(binding, token=token)
                if not self._resume_tokens.valid(confirmation_token, binding):
                    raise StaleConfirmationError(
                        "The recovery evidence or task identity changed; "
                        "review it again."
                    )
            record = await self.tasks.resume_recovered(
                task_id,
                operation_id=operation_id,
                expected_generation=evidence.generation,
                expected_operation_id=evidence.pending_operation_id,
                expected_execution_id=evidence.execution_id,
                confirmed=decision.kind == "confirm",
                first_start=decision.kind == "first_start",
            )
            if decision.kind == "confirm" and confirmation_token is not None:
                # The token was validated under this lock, just before the
                # commit. Consume it as a formality: an accepted transition
                # must never fail after its durable commit.
                self._resume_tokens.consume(confirmation_token, binding)
            # An accepted transition retires every preview issued for it.
            self._resume_tokens.invalidate(task_id)
            self._restart_fenced.discard(task_id)
            self._restart_first.discard(task_id)
            self._invalid_workspace.discard(task_id)
            self._pause_requested.discard(task_id)
            self._register(record)
            restored: int | str = (
                record.level if record.level_source == "user" else "auto"
            )
            self.gate.set_task_level(task_id, restored, rank=record.user_rank)
            if record.level >= 5:
                self.gate.update_task(task_id, paused=True)
                self._ready[task_id].clear()
            else:
                self.gate.update_task(task_id, paused=False)
                self._eligible.add(task_id)
                self._ready[task_id].set()
            await self._schedule()
            return await self._body(record)

    async def _resume_workspace(self, task_id: str) -> bool:
        try:
            await self._workspace(task_id)
        except WorkspaceError:
            return False
        return True

    def _select_park(self, task_id: str) -> None:
        point = self._points.get(task_id)
        if task_id in self._stopping:
            return
        if point is None:
            approval = self._approval_points.get(task_id)
            if approval is not None and approval[0].cancel_for_park(approval[1]):
                self._park_winners.add(task_id)
            return
        call, runner, turn_id = point
        if call.meta.admitted_mono_us is None:
            won = self.gate.try_park(task_id, turn_id, runner) is call
        else:
            won = runner.cancel_for_park(turn_id)
        if won:
            self._park_winners.add(task_id)

    async def _hold_call(
        self,
        task_id: str,
        call: GateCall,
        runner: SessionRunner,
        turn_id: str,
        kind: str,
    ) -> None:
        if self._runners.get(task_id) is not runner:
            raise asyncio.CancelledError()
        point = (call, runner, turn_id)
        self._points[task_id] = point
        evidence = await committed_evidence(
            runner,
            self.store,
            turn_id=turn_id,
            site="reflect" if kind == "reflect" else "plan",
        )
        async with self._lock:
            if self._points.get(task_id) != point or task_id in self._stopping:
                raise asyncio.CancelledError()
            generation, previous = await self.tasks.park_identity(task_id)
            await self.parks.hold(
                task_id,
                site=evidence.site,
                step=evidence.step,
                attempt=evidence.attempt,
                reason=self.gate.held(task_id) or "scheduler",
                expected_generation=generation,
                expected_operation_id=previous,
                operation_id=uuid.uuid4().hex,
            )
            current = await self.tasks.get_task(task_id)
            if task_id in self._pause_requested and current.level == 5:
                self._select_park(task_id)

    async def _release_call(
        self, task_id: str, call: GateCall, runner: SessionRunner, turn_id: str
    ) -> None:
        async with self._lock:
            if self._points.get(task_id) != (call, runner, turn_id):
                raise asyncio.CancelledError()
            if task_id in self._stopping or task_id in self._pause_requested:
                return
            generation, previous = await self.tasks.park_identity(task_id)
            await self.parks.resume(
                task_id,
                expected_generation=generation,
                expected_operation_id=previous,
                operation_id=uuid.uuid4().hex,
            )
            if task_id not in self._stopping and task_id not in self._pause_requested:
                self._ready[task_id].set()

    async def _hold_approval(
        self, task_id: str, runner: SessionRunner, request: ApprovalRequest
    ) -> tuple[SessionRunner, str, SafePointEvidence]:
        turn = runner.current_turn
        if self._runners.get(task_id) is not runner:
            raise asyncio.CancelledError()
        if turn is None:
            raise RuntimeError("Approval has no active worker turn")
        self._ready[task_id].clear()
        evidence = await committed_evidence(
            runner,
            self.store,
            turn_id=turn.id,
            site="approval",
            request_id=request.request_id,
        )
        point = (runner, turn.id, evidence)
        async with self._lock:
            if (
                task_id in self._stopping
                or self._runners.get(task_id) is not runner
                or runner.current_turn != turn
            ):
                raise asyncio.CancelledError()
            self._approval_points[task_id] = point
            generation, previous = await self.tasks.park_identity(task_id)
            await self.parks.hold(
                task_id,
                site="approval",
                step=evidence.step,
                attempt=evidence.attempt,
                reason="approval",
                expected_generation=generation,
                expected_operation_id=previous,
                operation_id=uuid.uuid4().hex,
            )
            current = await self.tasks.get_task(task_id)
            if task_id in self._pause_requested and current.level == 5:
                self._select_park(task_id)
        return point

    async def _release_approval(
        self, task_id: str, point: tuple[SessionRunner, str, SafePointEvidence]
    ) -> None:
        async with self._lock:
            if self._approval_points.get(task_id) != point or task_id in self._stopping:
                raise asyncio.CancelledError()
            if task_id in self._pause_requested:
                return
            generation, previous = await self.tasks.park_identity(task_id)
            await self.parks.resume(
                task_id,
                expected_generation=generation,
                expected_operation_id=previous,
                operation_id=uuid.uuid4().hex,
            )
            if task_id not in self._stopping and task_id not in self._pause_requested:
                self._ready[task_id].set()

    def _call_started(self, task_id: str, call: GateCall) -> None:
        point = self._points.get(task_id)
        if point is not None and point[0] is call:
            self._points.pop(task_id, None)

    async def _park_card(self, task_id: str) -> str:
        if self.store.failed is not None:
            raise RuntimeError("Event store is not writable")
        record = await self.tasks.get_task(task_id)
        after = 0
        events = []
        while True:
            page = await self.store.list_events(task_id, after_id=after)
            events.extend(page)
            if len(page) < 1000:
                break
            assert page[-1].id is not None
            after = page[-1].id
        return build_park_card(record.spec, events)

    async def _finalize_park(
        self,
        task_id: str,
        operation_id: str,
        interrupt_id: str,
        attempts: builtins.list[int],
    ) -> None:
        for candidate in (operation_id, interrupt_id):
            receipt = await self.parks.reconcile(task_id, operation_id=candidate)
            if receipt is not None:
                if not receipt[1]:
                    raise RuntimeError("Park operation was superseded")
                return
        card = await self._park_card(task_id)
        for attempt in range(3):
            if task_id in self._stopping or self._closed:
                raise RuntimeError("Park superseded by stop")
            generation, previous = await self.tasks.park_identity(task_id)
            if attempts[0] >= 3:
                await self.parks.interrupt(
                    task_id,
                    card=card,
                    expected_generation=generation,
                    expected_operation_id=previous,
                    operation_id=interrupt_id,
                )
                break
            try:
                await self.parks.park(
                    task_id,
                    card=card,
                    expected_generation=generation,
                    expected_operation_id=previous,
                    operation_id=operation_id,
                )
                break
            except (Exception, asyncio.CancelledError):
                attempts[0] += 1
                self._store_unavailable(task_id)
                receipt = await self.parks.reconcile(task_id, operation_id=operation_id)
                if receipt is not None:
                    if receipt[1]:
                        break
                    raise RuntimeError("Park operation was superseded") from None
                if self.store.failed is not None or await self.tasks.park_identity(
                    task_id
                ) != (generation, previous):
                    raise
                if attempts[0] >= 3:
                    await self.parks.interrupt(
                        task_id,
                        card=card,
                        expected_generation=generation,
                        expected_operation_id=previous,
                        operation_id=interrupt_id,
                    )
                    break
                marker_id = uuid.uuid4().hex
                try:
                    await self.parks.mark_park_retry(
                        task_id,
                        expected_generation=generation,
                        expected_operation_id=previous,
                        operation_id=marker_id,
                    )
                except (Exception, asyncio.CancelledError):
                    marker = await self.parks.reconcile(task_id, operation_id=marker_id)
                    if marker is not None and not marker[1]:
                        raise RuntimeError("Retry marker was superseded") from None
                    if marker is None and await self.tasks.park_identity(task_id) != (
                        generation,
                        previous,
                    ):
                        raise RuntimeError("Retry marker outcome is unknown") from None
                    log.warning("Could not record park retry for worker %s", task_id)
                await asyncio.sleep(0.05 * 2**attempt)

    async def _sync_park_phase(self, task_id: str) -> None:
        record = await self.tasks.get_task(task_id)
        if record.state in ("parked", "interrupted") and task_id not in self._stopping:
            self.gate.update_task(task_id, phase=cast(Phase, record.state), paused=True)

    def _store_unavailable(self, task_id: str) -> None:
        self._unavailable.add(task_id)
        self._failure_events.setdefault(task_id, asyncio.Event()).set()

    async def _retry_finalization(self, task_id: str, *, cleanup_failed: bool) -> None:
        delay = 0.1
        operation_id = uuid.uuid4().hex
        interrupt_id = uuid.uuid4().hex
        stop_id = uuid.uuid4().hex
        attempts = [0]
        identity: tuple[int | None, str | None] | None = None
        stop_identity: tuple[int | None, str | None] | None = None
        while not self._closed:
            try:
                if task_id in self._stopping:
                    async with self._lock:
                        current = await self.tasks.get_task(task_id)
                        if current.state == "cancelled" and stop_identity is None:
                            self.gate.finish_task(task_id, "cancelled")
                            self._eligible.discard(task_id)
                            return
                        if stop_identity is None:
                            stop_identity = await self.tasks.park_identity(task_id)
                        await self.tasks.change(
                            task_id,
                            state="cancelled",
                            actor="scheduler",
                            expected_generation=stop_identity[0],
                            expected_operation_id=stop_identity[1],
                            operation_id=stop_id,
                        )
                        self.gate.finish_task(task_id, "cancelled")
                        self._eligible.discard(task_id)
                    self._unavailable.discard(task_id)
                    return
                if cleanup_failed:
                    receipt = await self.parks.reconcile(
                        task_id, operation_id=operation_id
                    )
                    if receipt is not None:
                        if not receipt[1]:
                            raise RuntimeError("Cleanup interruption superseded")
                        await self._sync_park_phase(task_id)
                        self._unavailable.discard(task_id)
                        return
                    card = await self._park_card(task_id)
                    current_identity = await self.tasks.park_identity(task_id)
                    if identity is None:
                        identity = current_identity
                    elif identity != current_identity:
                        raise RuntimeError("Cleanup interruption outcome unknown")
                    generation, previous = identity
                    async with self._lock:
                        if task_id in self._stopping:
                            continue
                        await self.parks.interrupt(
                            task_id,
                            card=card,
                            reason="cleanup failed; slot held until restart",
                            allow_empty=True,
                            expected_generation=generation,
                            expected_operation_id=previous,
                            operation_id=operation_id,
                        )
                else:
                    await self._finalize_park(
                        task_id, operation_id, interrupt_id, attempts
                    )
                await self._sync_park_phase(task_id)
                self._unavailable.discard(task_id)
                if not cleanup_failed:
                    self._failure_events.setdefault(task_id, asyncio.Event()).clear()
                return
            except (Exception, asyncio.CancelledError):
                self._store_unavailable(task_id)
                log.exception("Could not persist park for worker %s; retrying", task_id)
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
                delay = min(delay * 2, 5.0)

    async def _close_worker(
        self, runner: SessionRunner | None, agent: Agent | None
    ) -> None:
        failed = False
        for close in (
            (
                (lambda: runner.close(cause="shutdown" if self._closed else "closed"))
                if runner is not None
                else None
            ),
            agent.close if agent is not None else None,
        ):
            if close is not None:
                try:
                    await close()
                except Exception:
                    failed = True
                    log.exception("Worker resource close failed")
        if failed:
            raise RuntimeError("Worker cleanup failed")

    async def _cleanup_park(
        self, task_id: str, runner: SessionRunner | None, agent: Agent | None
    ) -> bool:
        for attempt in range(4):
            try:
                closing = asyncio.create_task(self._close_worker(runner, agent))
                shutdown = asyncio.create_task(self._shutdown.wait())
                try:
                    done, _ = await asyncio.wait(
                        {closing, shutdown}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if closing not in done:
                        done, _ = await asyncio.wait({closing}, timeout=10)
                        if not done:
                            closing.cancel()
                            log.error(
                                "Worker %s cleanup timed out at shutdown", task_id
                            )
                            return False
                    await closing
                finally:
                    shutdown.cancel()
                return True
            except Exception:
                if self._closed:
                    return False
                if attempt == 0:
                    try:
                        generation, previous = await self.tasks.park_identity(task_id)
                        await self.parks.mark_park_retry(
                            task_id,
                            reason="park cleanup retrying",
                            expected_generation=generation,
                            expected_operation_id=previous,
                            operation_id=uuid.uuid4().hex,
                        )
                    except Exception:
                        self._store_unavailable(task_id)
                        log.exception("Could not persist cleanup retry for %s", task_id)
                if attempt < 3:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._shutdown.wait(), timeout=0.05 * 2**attempt
                        )
        return False

    def _register(self, record: TaskRecord) -> None:
        if record.task_id not in self._ready:
            self._ready[record.task_id] = asyncio.Event()
            if record.state != "parked" and record.level != 5:
                self._ready[record.task_id].set()
        if any(t.task_id == record.task_id for t in self.gate.registered_tasks):
            return
        self.gate.register_task(
            Task(
                task_id=record.task_id,
                root_id=record.root_id,
                origin="user",
                phase=cast(Phase, record.state),
                level=record.level,
                level_source=cast(LevelSource, record.level_source),
                user_rank=record.user_rank,
                paused=record.state == "parked" or record.level == 5,
                deadline_wall_us=(
                    None
                    if record.deadline is None
                    else int(record.deadline * 1_000_000)
                ),
                created_wall_us=int(record.created_at * 1_000_000),
            )
        )

    async def _schedule(self) -> None:
        if self._closed:
            return
        rows = [
            r
            for r in await self.tasks.list_tasks()
            if r.task_id in self._eligible
            and r.task_id not in self._runs
            and r.state == "queued"
            and r.level < 5
        ]
        rows.sort(
            key=lambda r: (
                r.level_eff,
                r.user_rank if r.user_rank is not None else float("inf"),
                r.created_at,
                r.task_id,
            )
        )
        for record in rows[: max(0, MAX_WORKERS - len(self._runs))]:
            self._runs[record.task_id] = asyncio.create_task(self._run(record.task_id))

    async def _workspace(self, task_id: str) -> str:
        session = await self.store.get_session(task_id)
        if session is None or session.workspace is None:
            raise WorkspaceError("The task workspace is missing")
        resolved = validate_task_workspace(
            session.workspace, self.settings(), origin="user", task_id=task_id
        )
        if resolved != session.workspace:
            raise WorkspaceError("The task workspace has changed")
        return resolved

    async def _capture_incomplete(self, task_id: str, execution_id: str) -> None:
        """Do not admit arbitrary code without durable incomplete evidence."""
        delay = 0.05
        try:
            async with asyncio.timeout(_CAPTURE_RETRY_SECONDS):
                while True:
                    try:
                        async with self._lock:
                            if self._closed or task_id in self._stopping:
                                raise asyncio.CancelledError()
                            generation, previous = await self.tasks.park_identity(
                                task_id
                            )
                            await RecoveryStore(self.store).mark_capture_incomplete(
                                task_id,
                                execution_id=execution_id,
                                expected_generation=generation,
                                expected_operation_id=previous,
                            )
                            if self._closed or task_id in self._stopping:
                                raise asyncio.CancelledError()
                        return
                    except TaskConflictError:
                        raise
                    except Exception:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 1.0)
        except TimeoutError as exc:
            raise RuntimeError(
                "execution evidence could not be recorded; the call was not run; task store unavailable"
            ) from exc

    async def _capture_process(self, task_id: str, execution_id: str, pid: int) -> None:
        try:
            pgid, leader, host, boot, birth = process_identity(pid)
            async with self._lock:
                if self._closed or task_id in self._stopping:
                    raise asyncio.CancelledError()
                generation, previous = await self.tasks.park_identity(task_id)
                await RecoveryStore(self.store).record_captured_process(
                    task_id,
                    execution_id=execution_id,
                    pgid=pgid,
                    leader_pid=leader,
                    boot_id=boot,
                    start_token=birth,
                    host_id=host,
                    expected_generation=generation,
                    expected_operation_id=previous,
                )
                if self._closed or task_id in self._stopping:
                    raise asyncio.CancelledError()
        except Exception:
            # A child may only pass its gate without a process record once the
            # execution is durably incomplete, including on unsupported systems.
            await self._capture_incomplete(task_id, execution_id)

    async def _run(self, task_id: str) -> None:
        agent = None
        runner = None
        phase = "failed"
        failure_reason = None
        deferred = False
        launch_uncertain = False
        execution_id: str | None = self._pending_execution.get(task_id)
        launch_committed = execution_id is not None
        turn_started = False
        post_fence_failure = False
        tool_token: contextvars.Token[Any] | None = None
        process_token: contextvars.Token[Any] | None = None
        try:
            if execution_id is not None:
                try:
                    retired = (
                        await RecoveryStore(self.store).active_capture(task_id)
                        != execution_id
                    )
                except BaseException:
                    launch_uncertain = True
                    self._eligible.discard(task_id)
                    self._store_unavailable(task_id)
                    raise
                if retired:
                    # An intervening transition (Stop, pause then resume) retired
                    # the fence's capture before its turn started, so nothing ran
                    # under it: this launch mints a fresh execution instead of
                    # failing post-fence.
                    self._pending_execution.pop(task_id, None)
                    execution_id = None
                    launch_committed = False
            async with self._lock:
                record = await self.tasks.get_task(task_id)
                if record.state == "parked":
                    deferred = True
                    return
            workspace = await self._workspace(task_id)
            try:
                provenance = await RecoveryStore(self.store).provenance(task_id)
            except BaseException:
                launch_uncertain = True
                self._eligible.discard(task_id)
                self._store_unavailable(task_id)
                raise
            if (
                task_id in self._restart_first
                or provenance.first_start == "never_started"
            ):
                async with self._lock:
                    current = await self.tasks.get_task(task_id)
                    if (
                        current.state != "queued"
                        or current.level >= 5
                        or task_id in self._stopping
                        or task_id in self._pause_requested
                        or self._closed
                    ):
                        deferred = True
                        return
                    generation, previous = await self.tasks.park_identity(task_id)
                    execution_id = uuid.uuid4().hex
                    try:
                        launch = await RecoveryStore(self.store).start(
                            task_id,
                            operation_id=uuid.uuid4().hex,
                            execution_id=execution_id,
                            expected_generation=generation,
                            expected_operation_id=previous,
                            _capture=True,
                        )
                        if not launch.acquired:
                            raise TaskStoreError(
                                "recovery launch did not acquire a permit"
                            )
                    except TaskConflictError:
                        launch_uncertain = True
                        self._eligible.discard(task_id)
                        self._restart_first.discard(task_id)
                        self._restart_fenced.add(task_id)
                        raise
                    except BaseException:
                        launch_uncertain = True
                        self._eligible.discard(task_id)
                        self._store_unavailable(task_id)
                        raise
                    launch_committed = True
                    self._pending_execution[task_id] = execution_id
                    self._restart_first.discard(task_id)
                    if (
                        task_id in self._stopping
                        or task_id in self._pause_requested
                        or self._closed
                    ):
                        deferred = True
                        return
            if execution_id is None:
                execution_id = uuid.uuid4().hex
            capture_execution = execution_id

            async def incomplete() -> None:
                await self._capture_incomplete(task_id, capture_execution)

            async def capture(pid: int) -> None:
                await self._capture_process(task_id, capture_execution, pid)

            tool_token = TOOL_ENTRY.set(incomplete)
            process_token = PROCESS_RECORD.set(capture)
            client = self.gate.client(task_id)
            agent = await build_session_agent(
                cfg=self.settings(),
                base_url=None,
                model=None,
                workspace=workspace,
                options=AgentOptions(
                    mode="gateway",
                    memory=False,
                    timeout_s=None,
                    builtins=worker_tool_allowlist("user").builtins,
                ),
                wrap_llm=lambda inner: _WorkerLLM(
                    inner,
                    client,
                    self._ready[task_id],
                    lambda: self._runners.get(task_id),
                    lambda call, runner, turn, kind: self._hold_call(
                        task_id, call, runner, turn, kind
                    ),
                    lambda call, runner, turn: self._release_call(
                        task_id, call, runner, turn
                    ),
                    lambda call: self._call_started(task_id, call),
                ),
                wrap_embedding=lambda inner: _WorkerEmbedding(
                    inner, client, self._ready[task_id]
                ),
            )
            runner = SessionRunner(agent, self.store, task_id, progress=True)

            async def paused_ask(request: ApprovalRequest) -> ApprovalDecision:
                answer = self.broker.begin(task_id, request)
                try:
                    point = await self._hold_approval(task_id, runner, request)
                    decision = await answer
                    await self._release_approval(task_id, point)
                    await self._ready[task_id].wait()
                    if (
                        self._approval_points.get(task_id) != point
                        or task_id in self._stopping
                    ):
                        raise asyncio.CancelledError()
                    self._approval_points.pop(task_id, None)
                    return decision
                finally:
                    answer.cancel()
                    await asyncio.gather(answer, return_exceptions=True)

            policy = RememberingPolicy(runner.observe(paused_ask))
            async with self._lock:
                self._runners[task_id] = runner
                if (
                    task_id in self._stopping
                    or self._closed
                    or task_id in self._invalid_workspace
                ):
                    phase = "cancelled"
                else:
                    current = await self.tasks.get_task(task_id)
                    if current.state == "parked":
                        deferred = True
                    else:
                        if not launch_committed or current.state != "running":
                            await self.tasks.change(
                                task_id,
                                state="running",
                                actor="scheduler",
                                _capture_execution=execution_id,
                                _capture_existing=launch_committed,
                            )
                        if (
                            task_id in self._stopping
                            or self._closed
                            or task_id in self._invalid_workspace
                        ):
                            # Stop always wins: it sets its flag without waiting
                            # for this lock, so the awaits above may have let it
                            # in; no turn starts once it was requested.
                            phase = "cancelled"
                        else:
                            self.gate.update_task(task_id, phase="running")
                            self.gate.turn_start(task_id)
                            if record.card:
                                await runner.resume()
                            await runner.start_turn(
                                record.card or str(record.spec["objective"]),
                                approval=policy,
                                client_request_id=(
                                    execution_id if launch_committed else None
                                ),
                            )
                            turn_started = True
                            self._pending_execution.pop(task_id, None)
            if phase != "cancelled" and not deferred:
                result = await runner.wait()
                while runner.busy:
                    await asyncio.sleep(0.05)
                phase = (
                    "done"
                    if result and result.status == "completed"
                    else (
                        "cancelled"
                        if result and result.status == "cancelled"
                        else "failed"
                    )
                )
        except WorkspaceError:
            if launch_committed and not turn_started:
                post_fence_failure = True
                log.exception("Worker %s failed after its launch fence", task_id)
            else:
                async with self._lock:
                    current = await self.tasks.get_task(task_id)
                    deferred = current.state == "parked"
                if not deferred:
                    failure_reason = "invalid_workspace"
                    log.warning("Worker %s has an invalid workspace", task_id)
        except asyncio.CancelledError:
            post_fence_failure = launch_committed and not turn_started
            raise
        except Exception:
            post_fence_failure = launch_committed and not turn_started
            log.exception("Worker %s failed", task_id)
        finally:
            if process_token is not None:
                PROCESS_RECORD.reset(process_token)
            if tool_token is not None:
                TOOL_ENTRY.reset(tool_token)
            park = task_id in self._park_winners and not (
                self._closed
                or task_id in self._stopping
                or task_id in self._invalid_workspace
            )
            if park:
                cleaned = await self._cleanup_park(task_id, runner, agent)
                if not cleaned and not self._closed:
                    self._quarantined.add(task_id)
                    self._failure_events.setdefault(task_id, asyncio.Event()).set()
                    await self._retry_finalization(task_id, cleanup_failed=True)
                    await self._shutdown.wait()
                    try:
                        closing = asyncio.create_task(self._close_worker(runner, agent))
                        done, _ = await asyncio.wait({closing}, timeout=10)
                        if not done:
                            closing.cancel()
                            log.error(
                                "Quarantined worker %s timed out at shutdown", task_id
                            )
                        else:
                            await closing
                    except (Exception, asyncio.CancelledError):
                        log.exception(
                            "Quarantined worker %s remained open at shutdown", task_id
                        )
                    raise asyncio.CancelledError()
                if cleaned and not self._closed:
                    await self._retry_finalization(task_id, cleanup_failed=False)
            else:
                try:
                    await self._close_worker(runner, agent)
                except Exception:
                    log.exception("Could not close worker %s", task_id)
            async with self._lock:
                park = task_id in self._park_winners and not (
                    self._closed
                    or task_id in self._stopping
                    or task_id in self._invalid_workspace
                )
                if park:
                    self._park_winners.discard(task_id)
                    self._points.pop(task_id, None)
                if post_fence_failure and not (
                    self._closed or task_id in self._stopping
                ):
                    # A fenced execution cannot be retried without a new safety
                    # decision, even when no turn was observed to start.
                    launch_uncertain = True
                    self._eligible.discard(task_id)
                    self._restart_first.discard(task_id)
                    self._restart_fenced.add(task_id)
                    try:
                        current = await self.tasks.get_task(task_id)
                        if current.state not in TERMINAL_PHASES:
                            card = current.card or await self._park_card(task_id)
                            generation, previous = await self.tasks.park_identity(
                                task_id
                            )
                            operation_id = uuid.uuid4().hex
                            assert execution_id is not None
                            recovery = RecoveryStore(self.store)
                            await recovery.prepare(
                                task_id,
                                operation_id=operation_id,
                                execution_id=execution_id,
                                expected_generation=generation,
                                expected_operation_id=previous,
                                outcome="interrupted",
                                card=card,
                                reason="launch failed before the turn; safety review required",
                                crash=True,
                                unsafe_resume=True,
                            )
                            await recovery.commit(task_id, operation_id=operation_id)
                            self.gate.update_task(
                                task_id, phase="interrupted", paused=True
                            )
                            self._pending_execution.pop(task_id, None)
                    except (Exception, asyncio.CancelledError):
                        self._store_unavailable(task_id)
                        log.exception(
                            "Could not reconcile launch failure for %s", task_id
                        )
                preserve = launch_uncertain or (
                    (deferred or park)
                    and not (
                        self._closed
                        or task_id in self._stopping
                        or task_id in self._invalid_workspace
                    )
                )
                if preserve:
                    current = await self.tasks.get_task(task_id)
                    if current.state == "running" and not launch_uncertain:
                        await self.tasks.change(
                            task_id, state="queued", actor="scheduler"
                        )
                        self.gate.update_task(task_id, phase="queued")
                else:
                    if task_id in self._invalid_workspace:
                        phase = "failed"
                    elif task_id in self._stopping:
                        phase = "cancelled"
                    try:
                        if (
                            task_id not in self._invalid_workspace
                            and (await self.tasks.get_task(task_id)).state
                            not in TERMINAL_PHASES
                        ):
                            await self.tasks.change(
                                task_id,
                                state=phase,
                                actor="scheduler",
                                failure_reason=(
                                    failure_reason if phase == "failed" else None
                                ),
                            )
                    except TaskNotFoundError:
                        pass
                    except TaskStoreError:
                        log.exception("Could not finish task %s", task_id)
                    self.gate.finish_task(task_id, phase)
                    self._pending_execution.pop(task_id, None)
                    self._restart_first.discard(task_id)
                    self._eligible.discard(task_id)
                    self._stopping.discard(task_id)
                    self._pause_requested.discard(task_id)
                    self._park_winners.discard(task_id)
                    self._invalid_workspace.discard(task_id)
                    self._ready.pop(task_id, None)
                self._points.pop(task_id, None)
                self._approval_points.pop(task_id, None)
                self._runners.pop(task_id, None)
                self._runs.pop(task_id, None)
                await self._schedule()

    async def _body(self, record: TaskRecord) -> dict:
        body = asdict(record)
        body["wait_reason"] = self.gate.held(record.task_id) or body["wait_reason"]
        session = await self.store.get_session(record.task_id)
        body["workspace"] = session.workspace if session else None
        body["workspace_label"] = (
            display.home_label(session.workspace)
            if session and session.workspace
            else ""
        )
        body["priority_events"] = await self.tasks.priority_events(record.task_id)
        return body

    async def _reconcile(self) -> int:
        async with self._lock:
            for task in self.gate.registered_tasks:
                try:
                    record = await self.tasks.get_task(task.task_id)
                except TaskNotFoundError:
                    record = None
                if task.task_id in self._runs:
                    continue
                if record is not None and (
                    record.state not in TERMINAL_PHASES
                    or record.updated_at >= self.store.retention_cutoff()
                ):
                    continue
                self._eligible.discard(task.task_id)
                self._ready.pop(task.task_id, None)
                if task.phase not in TERMINAL_PHASES:
                    self.gate.finish_task(task.task_id, "cancelled")
                self.gate.discard_task(task.task_id)
        return 0

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            self._shutdown.set()
            self._eligible.clear()
            for task_id in self._runs:
                self._stopping.add(task_id)
            for runner in self._runners.values():
                runner.cancel(cause="shutdown")
            running = list(self._runs.values())
        if running:
            await asyncio.gather(*running, return_exceptions=True)
