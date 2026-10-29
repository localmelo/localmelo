"""A task that waits for approval lists it, and the approvals route answers it."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.gateway.console import Console
from localmelo.support.sessions.agents import session_options

from .conftest import PORT


class WriterLLM(BaseLLMProvider):
    """Asks to write a file once; any tool result (or the file) ends the task."""

    def __init__(self, target: Path) -> None:
        self.target = target

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if self.target.exists() or any(m.role == "tool" for m in messages):
            return Message(role="assistant", content="finished")
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


@pytest.fixture()
async def api(
    console: Console,
    app: FastAPI,
    workspace: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setenv("HOME", str(tmp_path))

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](WriterLLM(Path(kwargs["workspace"]) / "out.txt")),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    console.cfg.gateway.workspace = workspace
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url=f"http://127.0.0.1:{PORT}"
    ) as client:
        yield client


async def waiting_task(api: httpx.AsyncClient, workspace: str) -> tuple[str, str]:
    created = await api.post(
        "/api/tasks", json={"objective": "write a file", "workspace": workspace}
    )
    assert created.status_code == 202, created.text
    task_id = created.json()["task"]["task_id"]
    pending: list[dict[str, Any]] = []

    async def poll() -> None:
        while not pending:
            body = (await api.get(f"/api/tasks/{task_id}")).json()["task"]
            pending.extend(body["pending_approvals"])
            await asyncio.sleep(0.005)

    async with asyncio.timeout(5):
        await poll()
    return task_id, pending[0]["request_id"]


async def finished(api: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
    async with asyncio.timeout(5):
        while True:
            task: dict[str, Any] = (await api.get(f"/api/tasks/{task_id}")).json()[
                "task"
            ]
            if task["state"] in ("done", "failed", "cancelled"):
                return task
            await asyncio.sleep(0.005)


async def test_task_carries_its_pending_approval(
    api: httpx.AsyncClient, workspace: str
) -> None:
    assert (await api.get("/api/tasks")).json() == {"tasks": []}
    task_id, request_id = await waiting_task(api, workspace)
    one = (await api.get(f"/api/tasks/{task_id}")).json()["task"]
    listed = (await api.get("/api/tasks")).json()["tasks"][0]
    assert one["state"] == "held" and one["wait_reason"] == "approval"
    (approval,) = listed["pending_approvals"]
    assert approval == one["pending_approvals"][0]
    assert approval["request_id"] == request_id
    assert approval["tool_name"] == "file_write"
    assert approval["target"]["path_label"].endswith("out.txt")
    assert approval["arguments_complete"] is True


@pytest.mark.parametrize(
    ("choice", "remember", "written"),
    [("once", False, True), ("session", True, True), ("deny", False, False)],
)
async def test_answer_moves_the_task_on(
    api: httpx.AsyncClient,
    workspace: str,
    choice: str,
    remember: bool,
    written: bool,
) -> None:
    task_id, request_id = await waiting_task(api, workspace)
    body = {"choice": choice, **({"reason": "no"} if choice == "deny" else {})}
    answered = await api.post(f"/api/approvals/{request_id}", json=body)
    assert answered.status_code == 200
    assert answered.json()["remember"] is remember
    task = await finished(api, task_id)
    assert task["state"] == "done"
    assert task["pending_approvals"] == []
    assert (Path(workspace) / "out.txt").exists() is written
    again = await api.post(f"/api/approvals/{request_id}", json={"choice": "once"})
    assert again.status_code == 409


async def test_stop_while_waiting_cancels_and_withdraws_the_approval(
    api: httpx.AsyncClient, workspace: str
) -> None:
    task_id, request_id = await waiting_task(api, workspace)
    stopped = await api.patch(f"/api/tasks/{task_id}", json={"state": "stop"})
    assert stopped.status_code == 200
    task = await finished(api, task_id)
    assert task["state"] == "cancelled" and task["pending_approvals"] == []
    assert not (Path(workspace) / "out.txt").exists()
    late = await api.post(f"/api/approvals/{request_id}", json={"choice": "once"})
    assert late.status_code in (404, 409)


async def test_pause_while_waiting_withdraws_the_approval_and_resume_asks_again(
    api: httpx.AsyncClient, workspace: str
) -> None:
    task_id, first = await waiting_task(api, workspace)
    paused = await api.patch(f"/api/tasks/{task_id}", json={"state": "pause"})
    assert paused.status_code == 200
    async with asyncio.timeout(5):
        while True:
            task = (await api.get(f"/api/tasks/{task_id}")).json()["task"]
            if task["state"] == "parked":
                break
            await asyncio.sleep(0.005)
    assert task["pending_approvals"] == []
    resumed = await api.post(
        f"/api/tasks/{task_id}/resume", json={"operation_id": "op-1"}
    )
    assert resumed.status_code == 202
    async with asyncio.timeout(5):
        while True:
            task = (await api.get(f"/api/tasks/{task_id}")).json()["task"]
            if task["pending_approvals"]:
                break
            await asyncio.sleep(0.005)
    assert task["pending_approvals"][0]["request_id"] != first
