"""A console task runs with the same lengths as a chat, and reads ``[agent] reflection``."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore


@pytest.fixture
async def worker_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TaskWorkers, Config, Path]]:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    config = Config()
    config.gateway.workspace = str(workspace)
    workers = TaskWorkers(store, gate, lambda: config, hub)
    try:
        yield workers, config, workspace
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def _options_of_a_task(
    workers: TaskWorkers, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> AgentOptions:
    seen: list[AgentOptions] = []
    built = asyncio.Event()

    async def build(**kwargs: Any) -> Any:
        seen.append(kwargs["options"])
        built.set()
        raise RuntimeError("stop test worker")

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    await workers.tasks.open()
    await workers.tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}),
        origin="user",
        workspace=str(workspace),
        _recovery_proof=True,
    )
    await workers.open()
    await asyncio.wait_for(built.wait(), 3)
    return seen[0]


async def test_a_task_has_long_attempts_and_reflects_by_default(
    worker_setup: tuple[TaskWorkers, Config, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, _, workspace = worker_setup
    options = await _options_of_a_task(workers, workspace, monkeypatch)
    assert (options.max_steps, options.steps_per_attempt) == (100, 50)
    assert options.attempt_policy == "reflect"


async def test_a_task_never_reflects_when_the_switch_is_off(
    worker_setup: tuple[TaskWorkers, Config, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, config, workspace = worker_setup
    config.agent.reflection = False
    options = await _options_of_a_task(workers, workspace, monkeypatch)
    assert (options.max_steps, options.attempt_policy) == (100, "none")
