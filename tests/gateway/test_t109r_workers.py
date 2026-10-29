"""Regression tests for worker launch recovery and fencing."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskConflictError, TaskStoreError
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore, SessionStoreError


@pytest.fixture
async def worker_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TaskWorkers, Path]]:
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
        yield workers, workspace
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def _proof_task(workers: TaskWorkers, workspace: Path) -> str:
    await workers.tasks.open()
    task, _ = await workers.tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}),
        origin="user",
        workspace=str(workspace),
        _recovery_proof=True,
    )
    return task.task_id


async def test_open_retry_schedules_recovered_task(
    worker_setup: tuple[TaskWorkers, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, workspace = worker_setup
    task_id = await _proof_task(workers, workspace)
    original_schedule = workers._schedule
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def schedule() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SessionStoreError("temporary read failure")
        await original_schedule()

    async def build(**kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        raise RuntimeError("stop test worker")

    monkeypatch.setattr(workers, "_schedule", schedule)
    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    try:
        with pytest.raises(SessionStoreError):
            await workers.open()
        assert not workers._opened
        await workers.open()
        await asyncio.wait_for(entered.wait(), 3)
        assert workers._opened
        assert (await workers.tasks.get_task(task_id)).state == "running"
    finally:
        release.set()


async def test_start_conflict_is_fenced_but_stop_is_allowed(
    worker_setup: tuple[TaskWorkers, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, workspace = worker_setup
    task_id = await _proof_task(workers, workspace)

    async def conflict(self: RecoveryStore, *args: Any, **kwargs: Any) -> Any:
        raise TaskConflictError("stale task identity")

    async def build(**kwargs: Any) -> Any:
        pytest.fail("a conflicted launch must not build an agent")

    monkeypatch.setattr(RecoveryStore, "start", conflict)
    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    await workers.open()
    async with asyncio.timeout(3):
        while task_id in workers._runs:
            await asyncio.sleep(0.001)
    assert (await workers.tasks.get_task(task_id)).state == "queued"
    assert not workers.is_active(task_id)
    assert task_id not in workers._unavailable
    with pytest.raises(TaskStoreError, match="safety review"):
        await workers.update(task_id, level="p1")
    assert (await workers.update(task_id, state="stop"))["state"] == "cancelled"


@pytest.mark.parametrize("failure_site", ["build", "gate", "start_turn"])
async def test_post_fence_failure_interrupts_and_counts_once(
    worker_setup: tuple[TaskWorkers, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    workers, workspace = worker_setup
    task_id = await _proof_task(workers, workspace)

    class Agent:
        async def close(self) -> None:
            pass

    class Runner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def observe(self, ask: Any) -> Any:
            return ask

        async def start_turn(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("start turn failed")

        async def close(self, **kwargs: Any) -> None:
            pass

    async def build(**kwargs: Any) -> Any:
        if failure_site == "build":
            raise RuntimeError("build failed")
        return Agent()

    def fail_gate(task_id: str) -> None:
        raise RuntimeError("gate failed")

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    monkeypatch.setattr("localmelo.support.gateway.workers.SessionRunner", Runner)
    if failure_site == "gate":
        monkeypatch.setattr(workers.gate, "turn_start", fail_gate)
    await workers.open()
    async with asyncio.timeout(3):
        while task_id in workers._runs:
            await asyncio.sleep(0.001)
    task = await workers.tasks.get_task(task_id)
    provenance = await RecoveryStore(workers.store).provenance(task_id)
    assert task.state == "interrupted"
    assert task.crashes == 1
    assert task.card
    assert provenance.execution_id and provenance.unsafe_resume
    assert not workers.is_active(task_id)
    with pytest.raises(TaskStoreError, match="safety review"):
        await workers.update(task_id, state="resume")
    assert (await workers.update(task_id, state="stop"))["state"] == "cancelled"


async def test_ordinary_create_records_the_never_started_proof(
    worker_setup: tuple[TaskWorkers, Path],
) -> None:
    workers, workspace = worker_setup
    await workers.open()
    task, repeated = await workers.create(
        {"objective": "work", "workspace": str(workspace), "level": "p5"}
    )
    assert not repeated
    provenance = await RecoveryStore(workers.store).provenance(task["task_id"])
    # The production create writes the proof inside its durable transaction,
    # before any launch, so a task that dies queued is never-started, not
    # unknown.
    assert provenance.first_start == "never_started"
    assert provenance.execution_id is None
