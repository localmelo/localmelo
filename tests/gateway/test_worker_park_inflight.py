"""Worker pause must not describe an in-flight provider call as parked."""

from __future__ import annotations

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


@pytest.mark.parametrize("payload", [{"state": "pause"}, {"level": "P5"}])
async def test_pause_does_not_park_an_inflight_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingLLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            started.set()
            await release.wait()
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](BlockingLLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            await started.wait()
        paused = await workers.update(task_id, **payload)
        assert paused["state"] != "parked"
        async with store.component_transaction("check_pending_pause") as conn:
            pending = await (
                await conn.execute(
                    "SELECT pause_pending FROM park_priority WHERE task_id = ?",
                    (task_id,),
                )
            ).fetchone()
        assert pending is not None and pending[0] == 1
        assert task_id in workers._runs
        assert not release.is_set()
        resumed = await workers.update(task_id, state="resume")
        assert resumed["state"] == "running"
        async with store.component_transaction("check_resume") as conn:
            pending = await (
                await conn.execute(
                    "SELECT pause_pending FROM park_priority WHERE task_id = ?",
                    (task_id,),
                )
            ).fetchone()
        assert pending is not None and pending[0] == 0
    finally:
        release.set()
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_stop_fences_admission_before_stop_write(
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
    started = asyncio.Event()
    writing = asyncio.Event()
    finish_write = asyncio.Event()

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            started.set()
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
    original_change = workers.tasks.change

    async def delayed_change(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("actor") == "user":
            writing.set()
            await finish_write.wait()
        return await original_change(*args, **kwargs)

    monkeypatch.setattr(workers.tasks, "change", delayed_change)
    stop = None
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            while gate.held(task_id) != "Reservation":
                await asyncio.sleep(0.001)
        stop = asyncio.create_task(workers.update(task_id, state="stop"))
        async with asyncio.timeout(3):
            await writing.wait()
        gate.set_machine(Machine(lock="none"))
        await asyncio.sleep(0.05)
        assert not started.is_set()
    finally:
        finish_write.set()
        if stop is not None:
            await asyncio.gather(stop, return_exceptions=True)
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_pause_fences_admission_before_pending_write(
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
    started = asyncio.Event()
    writing = asyncio.Event()
    finish_write = asyncio.Event()

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            started.set()
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
    original_pause = workers.tasks.pause

    async def delayed_pause(*args: Any, **kwargs: Any) -> Any:
        writing.set()
        await finish_write.wait()
        return await original_pause(*args, **kwargs)

    monkeypatch.setattr(workers.tasks, "pause", delayed_pause)
    pause = None
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            while gate.held(task_id) != "Reservation":
                await asyncio.sleep(0.001)
        pause = asyncio.create_task(workers.update(task_id, state="pause"))
        async with asyncio.timeout(3):
            await writing.wait()
        gate.set_machine(Machine(lock="none"))
        # Let the gate's admission and provider coroutine run if unfenced.
        await asyncio.sleep(0.05)
        assert not started.is_set()
    finally:
        finish_write.set()
        if pause is not None:
            await asyncio.gather(pause, return_exceptions=True)
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
