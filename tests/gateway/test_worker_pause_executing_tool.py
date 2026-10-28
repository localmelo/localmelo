"""A pause during a real worker's executing tool parks at the next safe point."""

from __future__ import annotations

import asyncio
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
from localmelo.support.sessions.store import SessionStore


async def test_pause_waits_for_executing_tool_then_resumes_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "result.txt"
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    started, release, finished, cancelled = (asyncio.Event() for _ in range(4))
    calls: list[list[Message]] = []

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            calls.append(list(messages))
            if len(calls) == 1:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="file_write",
                            arguments={"path": str(target), "content": "finished"},
                            id="write-call",
                        )
                    ],
                )
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](LLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        original = agent.executor._callables["file_write"]

        async def blocked_write(path: str, content: str) -> Any:
            started.set()
            try:
                await release.wait()
                result = await original(path=path, content=content)
                finished.set()
                return result
            except asyncio.CancelledError:
                cancelled.set()
                raise

        agent.executor._callables["file_write"] = blocked_write
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    try:
        row, _ = await workers.create(
            {"objective": "write", "workspace": str(workspace)}
        )
        task_id = row["task_id"]
        async with asyncio.timeout(3):
            while not workers.broker.pending(task_id):
                await asyncio.sleep(0.001)
        request = workers.broker.pending(task_id)[0]
        assert (
            workers.broker.answer(request.request_id, ApprovalDecision(True))
            == "resolved"
        )
        await asyncio.wait_for(started.wait(), 3)
        assert not target.exists()
        paused = await workers.update(task_id, state="pause")
        assert paused["state"] != "parked"
        assert task_id in workers._runs
        assert not cancelled.is_set() and not finished.is_set()
        assert len(calls) == 1

        release.set()
        await asyncio.wait_for(finished.wait(), 3)
        async with asyncio.timeout(3):
            while task_id in workers._runs:
                await asyncio.sleep(0.001)
        assert not cancelled.is_set()
        assert target.read_text() == "finished"
        parked = await workers.get(task_id)
        assert parked["state"] == "parked" and parked["card"]
        assert len(calls) == 1
        card = parked["card"]
        events = await store.list_events(task_id)
        assert any(e.event["type"] == "tool.started" for e in events)
        assert any(e.event["type"] == "tool.finished" for e in events)
        assert "file_write:" in card
        assert "tool call without a recorded finish" not in card
        await workers.update(task_id, state="resume")
        async with asyncio.timeout(3):
            while (
                task_id in workers._runs
                or (await workers.get(task_id))["state"] != "done"
            ):
                await asyncio.sleep(0.001)
        assert len(calls) >= 2
        assert any(card in (message.content or "") for message in calls[1])
        assert target.read_text() == "finished"
    finally:
        release.set()
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
