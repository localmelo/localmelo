"""Worker park withdraws approvals and a resumed turn asks again."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionStore


async def eventually(check: Callable[[], bool]) -> None:
    async with asyncio.timeout(3):
        while not check():
            await asyncio.sleep(0.001)


@pytest.fixture
async def worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TaskWorkers, str, Path]]:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "approval.txt"
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "data" / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)

    class LLM(BaseLLMProvider):
        def __init__(self, target: Path) -> None:
            self.target = target

        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            if self.target.exists():
                return Message(role="assistant", content="written")
            return Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="file_write",
                        arguments={"path": str(self.target), "content": "approved"},
                        id="write-call",
                    )
                ],
            )

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](LLM(Path(kwargs["workspace"]) / "approval.txt")),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    try:
        row, _ = await workers.create(
            {"objective": "write", "workspace": str(workspace)}
        )
        task_id = row["task_id"]
        await eventually(lambda: bool(workers.broker.pending(task_id)))
        runner = workers._runners[task_id]
        await runner.flush()
        events = await store.list_events(task_id)
        assert any(event.event["type"] == "step.started" for event in events)
        assert any(event.event["type"] == "approval.requested" for event in events)
        yield workers, task_id, target
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_approval_is_registered_before_request_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "early.txt"
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    entered, release = asyncio.Event(), asyncio.Event()

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            return Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="file_write",
                        arguments={"path": str(target), "content": "approved"},
                        id="early",
                    )
                ],
            )

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](LLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    original_hold = TaskWorkers._hold_approval

    async def blocked_hold(self: TaskWorkers, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await original_hold(self, *args, **kwargs)

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    monkeypatch.setattr(TaskWorkers, "_hold_approval", blocked_hold)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    try:
        row, _ = await workers.create(
            {"objective": "write", "workspace": str(workspace)}
        )
        task_id = row["task_id"]
        await asyncio.wait_for(entered.wait(), 3)
        await workers._runners[task_id].flush()
        events = await store.list_events(task_id)
        request = next(
            e.event["data"]["request_id"]
            for e in events
            if e.event["type"] == "approval.requested"
        )
        assert workers.broker.answer(request, ApprovalDecision(True)) == "resolved"
        assert not target.exists()
    finally:
        release.set()
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_answered_approval_is_discarded_and_resume_reasks(
    worker: tuple[TaskWorkers, str, Path],
) -> None:
    workers, task_id, target = worker
    first = workers.broker.pending(task_id)[0]
    # The broker resolves synchronously; pause closes delivery before the
    # waiting executor can resume on the next event-loop turn.
    assert workers.broker.answer(first.request_id, ApprovalDecision(True)) == "resolved"
    await workers.update(task_id, state="pause")
    await eventually(lambda: task_id not in workers._runs)
    assert not target.exists()
    parked = await workers.get(task_id)
    assert parked["state"] == "parked" and parked["card"]
    assert not workers.broker.pending(task_id)
    events = await workers.store.list_events(task_id)
    requested = [e for e in events if e.event["type"] == "approval.requested"]
    assert any(e.event["data"]["request_id"] == first.request_id for e in requested)
    assert any(
        e.event["type"] == "approval.resolved"
        and e.event["data"]["request_id"] == first.request_id
        and e.event["data"]["origin"] == "withdrawn"
        for e in events
    )

    await workers.update(task_id, state="resume")
    await eventually(lambda: bool(workers.broker.pending(task_id)))
    second = workers.broker.pending(task_id)[0]
    assert second.request_id != first.request_id
    assert not target.exists()
    assert (
        workers.broker.answer(second.request_id, ApprovalDecision(True)) == "resolved"
    )
    await eventually(lambda: task_id not in workers._runs)
    assert target.read_text() == "approved"
    assert (await workers.get(task_id))["state"] == "done"
    events = await workers.store.list_events(task_id)
    assert any(
        e.event["type"] == "approval.requested"
        and e.event["data"]["request_id"] == second.request_id
        and e.turn != requested[0].turn
        for e in events
    )


async def test_answer_after_park_cannot_authorize_old_request(
    worker: tuple[TaskWorkers, str, Path],
) -> None:
    workers, task_id, target = worker
    first = workers.broker.pending(task_id)[0]
    await workers.update(task_id, state="pause")
    await eventually(lambda: task_id not in workers._runs)
    assert (
        workers.broker.answer(first.request_id, ApprovalDecision(True))
        == "already_resolved"
    )
    assert not target.exists()
    assert not workers.broker.pending(task_id)
    await workers.update(task_id, state="stop")
    assert (await workers.get(task_id))["state"] == "cancelled"
    assert not target.exists()


async def test_stop_while_paused_prevents_approved_execution(
    worker: tuple[TaskWorkers, str, Path],
) -> None:
    workers, task_id, target = worker
    first = workers.broker.pending(task_id)[0]
    assert workers.broker.answer(first.request_id, ApprovalDecision(True)) == "resolved"
    await workers.update(task_id, state="pause")
    await workers.update(task_id, state="stop")
    await eventually(lambda: task_id not in workers._runs)
    assert not target.exists()
    assert not workers.broker.pending(task_id)
    assert (await workers.get(task_id))["state"] == "cancelled"


async def test_stop_waits_for_delayed_start_and_cancels(
    worker: tuple[TaskWorkers, str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, _, _ = worker
    entered = asyncio.Event()
    release_start = asyncio.Event()
    allow_work = asyncio.Event()
    executed = asyncio.Event()

    class ProbeLLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            await allow_work.wait()
            executed.set()
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](ProbeLLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    class DelayedRunner(SessionRunner):
        async def start_turn(self, *args: Any, **kwargs: Any) -> Any:
            entered.set()
            await release_start.wait()
            return await super().start_turn(*args, **kwargs)

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
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
    assert not (Path(row["workspace"]) / "approval.txt").exists()
    assert not workers.broker.pending(task_id)
    assert (await workers.get(task_id))["state"] == "cancelled"
