"""Worker approvals remain blocked while a task is paused."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.approvals import AskFunction
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore


class FakeAgent:
    async def close(self) -> None:
        pass


class ApprovalRunner:
    instances: list[ApprovalRunner] = []

    def __init__(
        self, agent: FakeAgent, store: SessionStore, session_id: str, **kwargs: Any
    ) -> None:
        self.session_id = session_id
        self.busy = False
        self.executed = asyncio.Event()
        self.task: asyncio.Task[SimpleNamespace] | None = None
        self.__class__.instances.append(self)

    def observe(self, ask: AskFunction) -> AskFunction:
        return ask

    async def start_turn(self, text: str, *, approval: ApprovalPolicy) -> None:
        self.busy = True

        async def run() -> SimpleNamespace:
            try:
                decision = await approval.decide(
                    ApprovalRequest(
                        request_id="approval-1",
                        run_id="run-1",
                        call_id="call-1",
                        tool_name="write_file",
                        arguments={},
                        preview="write file",
                        workspace_root=None,
                    )
                )
                if decision.allow:
                    self.executed.set()
                return SimpleNamespace(status="completed")
            except asyncio.CancelledError:
                return SimpleNamespace(status="cancelled")
            finally:
                self.busy = False

        self.task = asyncio.create_task(run())

    async def wait(self) -> SimpleNamespace:
        assert self.task is not None
        return await self.task

    def cancel(self, *, cause: str) -> bool:
        return self.task.cancel() if self.task is not None else False

    async def close(self, **kwargs: Any) -> None:
        pass


async def eventually(check: Callable[[], bool]) -> None:
    async with asyncio.timeout(3):
        while not check():
            await asyncio.sleep(0.001)


@pytest.fixture
async def worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TaskWorkers, str, ApprovalRunner]]:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "data" / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)

    async def build(**kwargs: Any) -> FakeAgent:
        return FakeAgent()

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    monkeypatch.setattr(
        "localmelo.support.gateway.workers.SessionRunner", ApprovalRunner
    )
    ApprovalRunner.instances = []
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    try:
        row, _ = await workers.create(
            {"objective": "write", "workspace": str(workspace)}
        )
        task_id = row["task_id"]
        await eventually(lambda: bool(workers.broker.pending(task_id)))
        yield workers, task_id, ApprovalRunner.instances[0]
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_answered_approval_waits_for_resume(
    worker: tuple[TaskWorkers, str, ApprovalRunner],
) -> None:
    workers, task_id, runner = worker
    await workers.update(task_id, state="pause")
    assert workers.broker.answer("approval-1", ApprovalDecision(True)) == "resolved"
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner.executed.wait(), timeout=0.05)
    assert (await workers.get(task_id))["state"] == "parked"
    await workers.update(task_id, state="resume")
    await asyncio.wait_for(runner.executed.wait(), timeout=3)
    await eventually(lambda: task_id not in workers._runs)
    assert (await workers.get(task_id))["state"] == "done"


async def test_stop_while_paused_prevents_approved_execution(
    worker: tuple[TaskWorkers, str, ApprovalRunner],
) -> None:
    workers, task_id, runner = worker
    await workers.update(task_id, state="pause")
    assert workers.broker.answer("approval-1", ApprovalDecision(True)) == "resolved"
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner.executed.wait(), timeout=0.05)
    await workers.update(task_id, state="stop")
    await eventually(lambda: task_id not in workers._runs)
    assert not runner.executed.is_set()
    assert (await workers.get(task_id))["state"] == "cancelled"


async def test_stop_waits_for_delayed_start_and_cancels(
    worker: tuple[TaskWorkers, str, ApprovalRunner], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, _, _ = worker
    entered = asyncio.Event()
    release_start = asyncio.Event()
    allow_work = asyncio.Event()
    executed = asyncio.Event()

    class DelayedRunner(ApprovalRunner):
        async def start_turn(self, text: str, *, approval: ApprovalPolicy) -> None:
            entered.set()
            await release_start.wait()
            self.busy = True

            async def run() -> SimpleNamespace:
                try:
                    await allow_work.wait()
                    executed.set()
                    return SimpleNamespace(status="completed")
                except asyncio.CancelledError:
                    return SimpleNamespace(status="cancelled")
                finally:
                    self.busy = False

            self.task = asyncio.create_task(run())

    monkeypatch.setattr(
        "localmelo.support.gateway.workers.SessionRunner", DelayedRunner
    )
    row, _ = await workers.create({"objective": "delayed"})
    task_id = row["task_id"]
    await asyncio.wait_for(entered.wait(), timeout=3)
    stopping = asyncio.create_task(workers.update(task_id, state="stop"))
    await asyncio.sleep(0.05)
    assert not stopping.done()
    release_start.set()
    await asyncio.wait_for(stopping, timeout=3)
    allow_work.set()
    await eventually(lambda: task_id not in workers._runs)
    assert not executed.is_set()
    assert (await workers.get(task_id))["state"] == "cancelled"
