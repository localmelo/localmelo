"""Task worker lifecycle without a model server or network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.task_policy import WorkspaceError
from localmelo.support.gateway.task_store import TaskConflictError
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.scheduler.model import Machine
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore


class FakeAgent:
    async def close(self) -> None:
        pass


class FakeRunner:
    instances: list[FakeRunner] = []

    def __init__(
        self, agent: Any, store: SessionStore, session_id: str, **kwargs: Any
    ) -> None:
        self.session_id = session_id
        self.progress = kwargs["progress"]
        self.done = asyncio.Event()
        self.cancelled = False
        self.busy = False
        self.__class__.instances.append(self)

    def observe(self, ask: Any) -> Any:
        return ask

    async def start_turn(self, text: str, *, approval: Any) -> None:
        self.busy = True
        assert approval is not None

    async def wait(self) -> SimpleNamespace:
        await self.done.wait()
        self.busy = False
        return SimpleNamespace(status="cancelled" if self.cancelled else "completed")

    def cancel(self, *, cause: str) -> bool:
        self.cancelled = True
        self.cause = cause
        self.done.set()
        return True

    async def close(self, **kwargs: Any) -> None:
        pass


async def eventually(check: Callable[[], bool]) -> None:
    async with asyncio.timeout(3):
        while not check():
            await asyncio.sleep(0.001)


WorkerSetup = tuple[TaskWorkers, Path, list[dict[str, Any]]]


@pytest.fixture
async def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[WorkerSetup]:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "data" / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    options: list[dict[str, Any]] = []

    async def build(**kwargs: Any) -> FakeAgent:
        options.append(kwargs)
        return FakeAgent()

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    monkeypatch.setattr("localmelo.support.gateway.workers.SessionRunner", FakeRunner)
    FakeRunner.instances = []
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    try:
        yield workers, workspace, options
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_capacity_stop_and_fifo(setup: WorkerSetup) -> None:
    workers, workspace, options = setup
    records = [
        await workers.create({"objective": str(i), "workspace": str(workspace)})
        for i in range(5)
    ]
    await eventually(lambda: len(FakeRunner.instances) == 4)
    assert len(options) == 4
    assert all(
        not x["options"].memory and x["options"].mode == "gateway" for x in options
    )
    assert all("shell_exec" not in x["options"].builtins for x in options)
    first = records[0][0]["task_id"]
    await workers.update(first, state="stop")
    await eventually(lambda: len(FakeRunner.instances) == 5)
    assert FakeRunner.instances[0].cause == "user_stop"
    assert (await workers.get(first))["state"] == "cancelled"
    assert FakeRunner.instances[4].session_id == records[4][0]["task_id"]


async def test_idempotency_and_workspace(setup: WorkerSetup) -> None:
    workers, workspace, _ = setup
    payload = {
        "prompt": "one",
        "workspace": str(workspace),
        "request_key": "x",
        "level": "P5",
    }
    row, repeated = await workers.create(payload)
    assert not repeated and row["workspace"] == str(workspace)
    assert row["deadline"] is None and row["state"] == "parked"
    with pytest.raises(ValueError, match="first"):
        await workers.update(row["task_id"], level="first")
    assert (await workers.get(row["task_id"]))["state"] == "parked"
    again, repeated = await workers.create(payload)
    assert repeated and again["task_id"] == row["task_id"]
    with pytest.raises(TaskConflictError):
        await workers.create({**payload, "prompt": "different"})
    with pytest.raises(ValueError, match="deadline"):
        await workers.create({**payload, "deadline": 123.0})
    workspace.rmdir()
    again, repeated = await workers.create(payload)
    assert repeated and again["task_id"] == row["task_id"]
    with pytest.raises(TaskConflictError):
        await workers.create({**payload, "prompt": "different"})
    await workers.update(row["task_id"], level="P2")
    await eventually(lambda: row["task_id"] not in workers._runs)
    failed = await workers.get(row["task_id"])
    assert failed["state"] == "failed" and failed["wait_reason"] == "invalid_workspace"
    with pytest.raises(WorkspaceError):
        await workers.create({"prompt": "bad", "workspace": "/"})
    with pytest.raises(ValueError):
        await workers.create({"prompt": "bad", "origin": "agent"})


async def test_pause_resume_restores_user_priority(setup: WorkerSetup) -> None:
    workers, workspace, _ = setup
    row, _ = await workers.create(
        {"prompt": "one", "workspace": str(workspace), "level": "P1"}
    )
    await eventually(lambda: len(FakeRunner.instances) == 1)
    task_id = row["task_id"]
    paused = await workers.update(task_id, state="pause")
    assert paused["state"] == "parked" and paused["level"] == 5
    again = await workers.update(task_id, level=" P5 ")
    assert again["reason"] == "pause" and again["state"] == "parked"
    resumed = await workers.update(task_id, state="resume")
    assert resumed["level"] == 1 and resumed["level_source"] == "user"
    assert resumed["state"] == "running"


async def test_pause_resume_queued_and_no_restart(setup: WorkerSetup) -> None:
    workers, workspace, _ = setup
    row, _ = await workers.create(
        {"prompt": "later", "workspace": str(workspace), "level": 5}
    )
    identifier = row["task_id"]
    assert not FakeRunner.instances
    assert (await workers.update(identifier, level="P2", state="resume"))[
        "state"
    ] == "queued"
    await eventually(lambda: len(FakeRunner.instances) == 1)
    await workers.update(identifier, state="stop")
    await eventually(lambda: identifier not in workers._runs)
    await workers.aclose()
    again = TaskWorkers(workers.store, workers.gate, workers.settings, workers.hub)
    await again.open()
    await asyncio.sleep(0)
    assert not again._runs
    await again.aclose()


async def test_patch_atomic_and_concurrent_key(setup: WorkerSetup) -> None:
    workers, workspace, _ = setup
    first, _ = await workers.create(
        {"prompt": "first", "workspace": str(workspace), "level": 0}
    )
    other, _ = await workers.create(
        {"prompt": "other", "workspace": str(workspace), "level": 5}
    )
    with pytest.raises(ValueError):
        await workers.update(other["task_id"], level=0, state="bad")
    assert (await workers.get(first["task_id"]))["level"] == 0
    assert (await workers.get(other["task_id"]))["level"] == 5
    payload = {
        "prompt": "same",
        "workspace": str(workspace),
        "request_key": "race",
        "level": 5,
    }
    results = await asyncio.gather(*(workers.create(payload) for _ in range(5)))
    assert len({row["task_id"] for row, _ in results}) == 1
    assert sum(repeated for _, repeated in results) == 4


async def test_bare_resume_of_initial_p5_and_normalized_auto(
    setup: WorkerSetup,
) -> None:
    workers, workspace, _ = setup
    first, _ = await workers.create(
        {"prompt": "one", "workspace": str(workspace), "level": 5}
    )
    resumed = await workers.update(first["task_id"], state="resume")
    assert resumed["level"] == 2 and resumed["level_source"] == "computed"
    second, _ = await workers.create(
        {"prompt": "two", "workspace": str(workspace), "level": 5}
    )
    changed = await workers.update(second["task_id"], level=" Auto ")
    assert changed["level"] == 2 and changed["level_source"] == "computed"


async def test_patch_normalizes_and_refuses_before_workspace_check(
    setup: WorkerSetup,
) -> None:
    workers, workspace, _ = setup
    row, _ = await workers.create(
        {"prompt": "one", "workspace": str(workspace), "level": 1}
    )
    await eventually(lambda: len(FakeRunner.instances) == 1)
    task_id = row["task_id"]
    paused = await workers.update(task_id, level=" P5 ")
    assert paused["level"] == 5 and paused["state"] == "parked"
    assert not workers._ready[task_id].is_set()
    workspace.rmdir()
    for level, state in (
        (" P5 ", "resume"),
        ("bogus", "resume"),
        ("P2", "pause"),
        (True, "resume"),
    ):
        with pytest.raises(ValueError):
            await workers.update(task_id, level=level, state=state)
        assert (await workers.get(task_id))["state"] == "parked"


async def test_pause_during_worker_build_does_not_start(
    setup: WorkerSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, workspace, _ = setup
    started, release = asyncio.Event(), asyncio.Event()

    async def build(**kwargs: Any) -> FakeAgent:
        started.set()
        await release.wait()
        return FakeAgent()

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    row, _ = await workers.create({"prompt": "one", "workspace": str(workspace)})
    task_id = row["task_id"]
    await started.wait()
    await workers.update(task_id, state="pause")
    release.set()
    await eventually(lambda: task_id not in workers._runs)
    assert (await workers.get(task_id))["state"] == "parked"
    assert not any(r.busy for r in FakeRunner.instances)
    await workers.update(task_id, state="resume")
    await eventually(
        lambda: task_id in workers._runs and any(r.busy for r in FakeRunner.instances)
    )


async def test_resume_while_deferred_worker_closes_is_rescheduled(
    setup: WorkerSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, workspace, _ = setup
    building, build_release, closing, close_release = (
        asyncio.Event() for _ in range(4)
    )
    builds = 0

    class SlowCloseAgent(FakeAgent):
        async def close(self) -> None:
            closing.set()
            await close_release.wait()

    async def build(**kwargs: Any) -> FakeAgent:
        nonlocal builds
        builds += 1
        if builds == 1:
            building.set()
            await build_release.wait()
            return SlowCloseAgent()
        return FakeAgent()

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    row, _ = await workers.create({"prompt": "one", "workspace": str(workspace)})
    task_id = row["task_id"]
    await building.wait()
    await workers.update(task_id, state="pause")
    build_release.set()
    await closing.wait()
    await workers.update(task_id, state="resume")
    close_release.set()
    await eventually(lambda: builds == 2 and any(r.busy for r in FakeRunner.instances))
    assert (await workers.get(task_id))["state"] == "running"


async def test_pause_before_failed_workspace_check_keeps_task_parked(
    setup: WorkerSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, workspace, _ = setup
    started, release = asyncio.Event(), asyncio.Event()

    async def invalid_workspace(task_id: str) -> str:
        started.set()
        await release.wait()
        raise WorkspaceError("workspace disappeared")

    monkeypatch.setattr(workers, "_workspace", invalid_workspace)
    row, _ = await workers.create({"prompt": "one", "workspace": str(workspace)})
    task_id = row["task_id"]
    await started.wait()
    await workers.update(task_id, state="pause")
    release.set()
    await eventually(lambda: task_id not in workers._runs)
    assert (await workers.get(task_id))["state"] == "parked"
    assert not FakeRunner.instances


async def test_retention_drops_old_identity_not_row_or_live_call(
    setup: WorkerSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers, workspace, _ = setup
    row, _ = await workers.create(
        {"prompt": "later", "workspace": str(workspace), "level": 5}
    )
    task_id = row["task_id"]
    call = workers.gate.submit(task_id)
    await workers.gate.acquire(call)
    await workers.update(task_id, state="stop")
    old = await workers.get(task_id)
    monkeypatch.setattr(
        workers.store, "retention_cutoff", lambda: old["updated_at"] + 1
    )
    await workers.store.apply_retention()
    assert any(t.task_id == task_id for t in workers.gate.registered_tasks)
    workers.gate.finish(call)
    await workers.store.apply_retention()
    assert all(t.task_id != task_id for t in workers.gate.registered_tasks)
    assert (await workers.get(task_id))["state"] == "cancelled"


async def test_sweep_discards_only_deleted_rows(setup: WorkerSetup) -> None:
    workers, workspace, _ = setup
    row, _ = await workers.create(
        {"prompt": "later", "workspace": str(workspace), "level": 5}
    )
    task_id = row["task_id"]
    await workers.update(task_id, state="stop")
    workers.gate.forget(task_id)
    await workers.store.apply_retention()
    assert any(t.task_id == task_id for t in workers.gate.registered_tasks)
    assert await workers.store.delete_session(task_id)
    await workers.store.apply_retention()
    assert all(t.task_id != task_id for t in workers.gate.registered_tasks)


async def test_blocked_gate_cancel_real_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "data" / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off", lock_gate="wait")
    gate.set_machine(Machine(lock="foreign"))
    hub = SessionHub(store)
    calls = []

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            calls.append(1)
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
    try:
        row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
        task_id = row["task_id"]
        await eventually(lambda: gate.held(task_id) == "Reservation")
        assert not calls
        assert (await workers.get(task_id))["wait_reason"] == "Reservation"
        await workers.update(task_id, state="stop")
        await eventually(lambda: task_id not in workers._runs)
        assert (await workers.get(task_id))["state"] == "cancelled"
        assert not calls
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
