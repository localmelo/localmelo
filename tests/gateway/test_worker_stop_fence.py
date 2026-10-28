"""Stop takes precedence while another task mutation is awaiting persistence."""

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


async def test_stop_cancels_before_blocked_pause_write_finishes(
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
    writing, finish_write = asyncio.Event(), asyncio.Event()

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
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
    pause = stop = None
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            while gate.held(task_id) != "Reservation":
                await asyncio.sleep(0.001)
        runner = workers._runners[task_id]
        cancellation = asyncio.Event()
        original_cancel = runner.cancel

        def cancel(*args: Any, **kwargs: Any) -> bool:
            cancellation.set()
            return original_cancel(*args, **kwargs)

        monkeypatch.setattr(runner, "cancel", cancel)
        pause = asyncio.create_task(workers.update(task_id, state="pause"))
        await asyncio.wait_for(writing.wait(), 3)
        stop = asyncio.create_task(workers.update(task_id, state="stop"))
        await asyncio.wait_for(cancellation.wait(), 0.2)
        assert not finish_write.is_set()
    finally:
        finish_write.set()
        await asyncio.gather(*(t for t in (pause, stop) if t), return_exceptions=True)
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_resume_waits_for_park_cleanup_and_uses_card(
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
    closing, release = asyncio.Event(), asyncio.Event()
    calls: list[list[Message]] = []
    builds = 0

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            calls.append(messages)
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        nonlocal builds
        builds += 1
        agent = Agent(
            llm=kwargs["wrap_llm"](LLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        if builds == 1:
            original_close = agent.close

            async def close() -> None:
                closing.set()
                await release.wait()
                await original_close()

            monkeypatch.setattr(agent, "close", close)
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    resume = None
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            while gate.held(task_id) != "Reservation":
                await asyncio.sleep(0.001)
        await workers.update(task_id, state="pause")
        await asyncio.wait_for(closing.wait(), 3)
        resume = asyncio.create_task(workers.update(task_id, state="resume"))
        await asyncio.sleep(0.05)
        assert not resume.done() and not calls
        release.set()
        await asyncio.wait_for(resume, 3)
        gate.set_machine(Machine(lock="none"))
        async with asyncio.timeout(3):
            while not calls:
                await asyncio.sleep(0.001)
        parked_card = (await workers.get(task_id))["card"]
        assert parked_card and any(
            parked_card in message.content for message in calls[0]
        )
    finally:
        release.set()
        if resume is not None:
            await asyncio.gather(resume, return_exceptions=True)
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
