"""Retry and interruption after a worker's park write fails."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.scheduler.model import Machine
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore


@pytest.mark.parametrize("marker_fails", [False, True])
@pytest.mark.parametrize("uncertain_interrupt", [False, True])
async def test_failed_park_retries_then_interrupts_with_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_fails: bool,
    uncertain_interrupt: bool,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off", lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    hub = SessionHub(store)
    calls: list[list[Message]] = []
    attempts = 0

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            calls.append(messages)
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](LLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()

    async def fail_park(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("park write failed")

    monkeypatch.setattr(workers.parks, "park", fail_park)
    commit = store._commit
    interrupted_commit = False

    async def commit_then_cancel(conn: Any, events: Any, label: str) -> None:
        nonlocal interrupted_commit
        await commit(conn, events, label)
        if (
            uncertain_interrupt
            and label == "park_interrupted"
            and not interrupted_commit
        ):
            interrupted_commit = True
            raise asyncio.CancelledError()

    monkeypatch.setattr(store, "_commit", commit_then_cancel)
    if marker_fails:

        async def fail_marker(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("retry marker write failed")

        monkeypatch.setattr(workers.parks, "mark_park_retry", fail_marker)
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            while gate.held(task_id) != "Reservation":
                await asyncio.sleep(0.001)
        await workers.update(task_id, state="pause")
        async with asyncio.timeout(3):
            while task_id in workers._runs:
                await asyncio.sleep(0.001)
        interrupted = await workers.get(task_id)
        assert attempts == 3
        assert interrupted["state"] == "interrupted" and interrupted["card"]
        assert interrupted_commit == uncertain_interrupt
        assert [e.event["type"] for e in await store.list_events(task_id)].count(
            "sched.interrupted"
        ) == 1
        assert not calls
        gate.set_machine(Machine(lock="none"))
        await workers.update(task_id, state="resume")
        async with asyncio.timeout(3):
            while not calls:
                await asyncio.sleep(0.001)
        assert any(interrupted["card"] in message.content for message in calls[0])
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_uncertain_park_commit_reconciles_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off", lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    hub = SessionHub(store)
    calls: list[list[Message]] = []

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            calls.append(messages)
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](LLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    original = store._commit
    raised = False

    async def uncertain(conn: Any, events: Any, label: str) -> None:
        nonlocal raised
        await original(conn, events, label)
        if label == "park_parked" and not raised:
            raised = True
            raise asyncio.CancelledError()

    monkeypatch.setattr(store, "_commit", uncertain)
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            while gate.held(task_id) != "Reservation":
                await asyncio.sleep(0.001)
        await workers.update(task_id, state="pause")
        async with asyncio.timeout(3):
            while task_id in workers._runs:
                await asyncio.sleep(0.001)
        task = await workers.get(task_id)
        assert raised and task["state"] == "parked" and task["card"]
        assert not calls
        assert [e.event["type"] for e in await store.list_events(task_id)].count(
            "sched.parked"
        ) == 1
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
