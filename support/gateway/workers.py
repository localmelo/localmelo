"""In-process task workers for the gateway's scheduler."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import cast

from localmelo.melo.agent import AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import (
    ChatGate,
    GatedEmbedding,
    GatedLLM,
    SessionClient,
)
from localmelo.support.gateway.task_policy import (
    TaskSpec,
    WorkspaceError,
    validate_task_workspace,
    worker_tool_allowlist,
)
from localmelo.support.gateway.task_store import (
    TaskNotFoundError,
    TaskNotPausedError,
    TaskRecord,
    TaskStore,
    TaskStoreError,
)
from localmelo.support.scheduler.model import TERMINAL_PHASES, LevelSource, Phase, Task
from localmelo.support.sessions import display
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.approvals import ApprovalBroker, RememberingPolicy
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionStore

log = logging.getLogger(__name__)
MAX_WORKERS = 4


class _WorkerLLM(GatedLLM):
    def __init__(
        self, inner: BaseLLMProvider, client: SessionClient, ready: asyncio.Event
    ) -> None:
        super().__init__(inner, client)
        self.ready = ready

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        await self.ready.wait()
        return await super().chat(messages, tools=tools)


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
        self._lock = asyncio.Lock()
        self._eligible: set[str] = set()
        self._runs: dict[str, asyncio.Task[None]] = {}
        self._runners: dict[str, SessionRunner] = {}
        self._stopping: set[str] = set()
        self._invalid_workspace: set[str] = set()
        self._ready: dict[str, asyncio.Event] = {}
        self._closed = False
        self._opened = False

    def is_active(self, task_id: str) -> bool:
        """Whether this process may still run or start this worker."""
        return task_id in self._eligible or task_id in self._runs

    async def open(self) -> None:
        if not self._opened:
            await self.tasks.open()
            await self.tasks.fail_nonterminal_on_restart()
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
        is_p5 = level == 5 or level == "p5"
        if state == "pause" and level is not None:
            raise ValueError("pause cannot be combined with a level")
        if state == "resume" and is_p5:
            raise ValueError("choose a runnable priority to resume a P5 task")
        async with self._lock:
            current = await self.tasks.get_task(task_id)
            if current.state in TERMINAL_PHASES:
                raise TaskStoreError("task is finished")
            if state == "resume" and current.state != "parked":
                raise TaskNotPausedError("task is not paused")
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
                record = await (
                    self.tasks.pause(task_id)
                    if state == "pause"
                    else self.tasks.resume(task_id)
                )
                if state == "resume" and record.level != 5 and task_id in self._runs:
                    record = await self.tasks.change(
                        task_id, state="running", actor="scheduler"
                    )
                restored: int | str = (
                    record.level if record.level_source == "user" else "auto"
                )
                self.gate.set_task_level(task_id, restored, rank=record.user_rank)
                if record.level == 5:
                    self._ready[task_id].clear()
                else:
                    self._eligible.add(task_id)
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
                task_id, priority=level, state=desired, actor="user"
            )
            self._register(current)
            if level is not None:
                self.gate.set_task_level(
                    task_id, cast(int | str, level), rank=record.user_rank
                )
            if state == "stop":
                self._eligible.discard(task_id)
                self._ready[task_id].set()
                runner = self._runners.get(task_id)
                if task_id in self._runs:
                    self._stopping.add(task_id)
                    if runner is not None:
                        runner.cancel(cause="user_stop")
                else:
                    self.gate.finish_task(task_id, "cancelled")
                    self._ready.pop(task_id, None)
            elif parked:
                self._ready[task_id].clear()
                self.gate.update_task(task_id, paused=True)
            elif state == "resume" or (level is not None and current.level == 5):
                self._eligible.add(task_id)
                self.gate.update_task(task_id, paused=False)
                self._ready[task_id].set()
            await self._schedule()
            return await self._body(record)

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

    async def _run(self, task_id: str) -> None:
        agent = None
        runner = None
        phase = "failed"
        failure_reason = None
        deferred = False
        try:
            async with self._lock:
                record = await self.tasks.get_task(task_id)
                if record.state == "parked":
                    deferred = True
                    return
            workspace = await self._workspace(task_id)
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
                wrap_llm=lambda inner: _WorkerLLM(inner, client, self._ready[task_id]),
                wrap_embedding=lambda inner: _WorkerEmbedding(
                    inner, client, self._ready[task_id]
                ),
            )
            runner = SessionRunner(agent, self.store, task_id, progress=True)
            ask = self.broker.asker(task_id)

            async def paused_ask(request: ApprovalRequest) -> ApprovalDecision:
                decision = await ask(request)
                await self._ready[task_id].wait()
                return decision

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
                        await self.tasks.change(
                            task_id, state="running", actor="scheduler"
                        )
                        self.gate.update_task(task_id, phase="running")
                        self.gate.turn_start(task_id)
                        await runner.start_turn(
                            str(record.spec["objective"]), approval=policy
                        )
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
            async with self._lock:
                current = await self.tasks.get_task(task_id)
                deferred = current.state == "parked"
            if not deferred:
                failure_reason = "invalid_workspace"
                log.warning("Worker %s has an invalid workspace", task_id)
        except Exception:
            log.exception("Worker %s failed", task_id)
        finally:
            try:
                if runner is not None:
                    await runner.close(cause="shutdown" if self._closed else "closed")
                if agent is not None:
                    await agent.close()
            except Exception:
                log.exception("Could not close worker %s", task_id)
            async with self._lock:
                preserve = deferred and not (
                    self._closed
                    or task_id in self._stopping
                    or task_id in self._invalid_workspace
                )
                if preserve:
                    current = await self.tasks.get_task(task_id)
                    if current.state == "running":
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
                        if task_id not in self._invalid_workspace:
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
                    self._eligible.discard(task_id)
                    self._stopping.discard(task_id)
                    self._invalid_workspace.discard(task_id)
                    self._ready.pop(task_id, None)
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
            self._eligible.clear()
            for task_id in self._runs:
                self._stopping.add(task_id)
            for runner in self._runners.values():
                runner.cancel(cause="shutdown")
            running = list(self._runs.values())
        if running:
            await asyncio.gather(*running, return_exceptions=True)
