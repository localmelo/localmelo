"""Worker quarantine when cleanup or finalization cannot be trusted."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.errors import ApiError
from localmelo.support.gateway.console.routes_tasks import _call
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.scheduler.model import Machine
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionStore


async def wait_for(predicate: Callable[[], bool], timeout: float = 3) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.fixture
async def worker_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TaskWorkers, ChatGate, list[list[Message]], Path]]:
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
    try:
        yield workers, gate, calls, workspace
    finally:
        # Remove injections before teardown so a broken implementation cannot strand it.
        monkeypatch.undo()
        async with asyncio.timeout(12):
            await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def start_held(workers: TaskWorkers, gate: ChatGate, workspace: Path) -> str:
    row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
    task_id = cast(str, row["task_id"])
    await wait_for(lambda: gate.held(task_id) == "Reservation")
    return task_id


@pytest.mark.parametrize("which", ["runner", "agent"])
async def test_permanent_close_failure_quarantines_slot(
    worker_env: tuple[TaskWorkers, ChatGate, list[list[Message]], Path],
    monkeypatch: pytest.MonkeyPatch,
    which: str,
) -> None:
    workers, gate, calls, workspace = worker_env
    target = SessionRunner if which == "runner" else Agent
    original = target.close
    attempts = 0

    async def fail_close(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("injected close failure")

    monkeypatch.setattr(target, "close", fail_close)
    task_id = await start_held(workers, gate, workspace)
    await workers.update(task_id, state="pause")
    await wait_for(lambda: attempts >= 3)

    async def interrupted() -> dict[str, Any]:
        async with asyncio.timeout(3):
            while True:
                row = await workers.get(task_id)
                if row["state"] == "interrupted":
                    return row
                await asyncio.sleep(0.001)

    row = await interrupted()
    assert row["card"]
    assert row["wait_reason"] == "cleanup failed; slot held until restart"
    assert task_id in workers._runs
    assert not calls
    assert "sched.parked" not in [
        event.event["type"] for event in await workers.store.list_events(task_id)
    ]
    # A quarantined worker remains counted by the scheduler's worker limit.
    from localmelo.support.gateway.workers import MAX_WORKERS

    others = []
    for _ in range(MAX_WORKERS):
        other, _ = await workers.create(
            {"prompt": "other", "workspace": str(workspace)}
        )
        others.append(other["task_id"])
    assert len(workers._runs) == MAX_WORKERS
    assert task_id in workers._runs
    assert sum(other in workers._runs for other in others) == MAX_WORKERS - 1
    assert not calls
    monkeypatch.setattr(target, "close", original)


@pytest.mark.parametrize("uncertain", [False, True])
async def test_stop_wins_when_park_cleanup_fails(
    worker_env: tuple[TaskWorkers, ChatGate, list[list[Message]], Path],
    monkeypatch: pytest.MonkeyPatch,
    uncertain: bool,
) -> None:
    workers, gate, calls, workspace = worker_env
    entered, release = asyncio.Event(), asyncio.Event()
    original = SessionRunner.close

    async def close(self: Any, *args: Any, **kwargs: Any) -> None:
        entered.set()
        await release.wait()
        raise RuntimeError("injected close failure")

    monkeypatch.setattr(SessionRunner, "close", close)
    task_id = await start_held(workers, gate, workspace)
    try:
        await workers.update(task_id, state="pause")
        await asyncio.wait_for(entered.wait(), 3)
        await workers.update(task_id, state="stop")
        commit = workers.store._commit
        fired = False

        async def commit_then_cancel(conn: Any, events: Any, label: str) -> None:
            nonlocal fired
            await commit(conn, events, label)
            if uncertain and not fired and label == "task_change":
                fired = True
                raise asyncio.CancelledError()

        monkeypatch.setattr(workers.store, "_commit", commit_then_cancel)
        release.set()
        async with asyncio.timeout(3):
            while (await workers.get(task_id))["state"] != "cancelled":
                await asyncio.sleep(0.001)
        if uncertain:
            await wait_for(lambda: fired and task_id in workers._unavailable)
            await wait_for(lambda: task_id not in workers._unavailable)
        assert task_id in workers._runs
        assert not calls
        assert "sched.parked" not in [
            event.event["type"] for event in await workers.store.list_events(task_id)
        ]
    finally:
        release.set()
        monkeypatch.setattr(SessionRunner, "close", original)


async def test_shutdown_bounds_a_blocked_initial_cleanup(
    worker_env: tuple[TaskWorkers, ChatGate, list[list[Message]], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, gate, calls, workspace = worker_env
    entered, release = asyncio.Event(), asyncio.Event()
    original = SessionRunner.close

    async def close(self: Any, *args: Any, **kwargs: Any) -> None:
        entered.set()
        await release.wait()
        await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionRunner, "close", close)
    task_id = await start_held(workers, gate, workspace)
    try:
        await workers.update(task_id, state="pause")
        await asyncio.wait_for(entered.wait(), 3)
        async with asyncio.timeout(11):
            await workers.aclose()
        assert not calls
    finally:
        release.set()
        monkeypatch.setattr(SessionRunner, "close", original)


async def test_quarantined_shutdown_is_bounded(
    worker_env: tuple[TaskWorkers, ChatGate, list[list[Message]], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, gate, calls, workspace = worker_env
    original = SessionRunner.close
    attempts = 0
    shutdown_close = asyncio.Event()
    unblock = asyncio.Event()

    async def close(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        if kwargs.get("cause") == "shutdown":
            shutdown_close.set()
            await unblock.wait()
        raise RuntimeError("injected close failure")

    monkeypatch.setattr(SessionRunner, "close", close)
    task_id = await start_held(workers, gate, workspace)
    await workers.update(task_id, state="pause")
    await wait_for(lambda: attempts >= 3)
    async with asyncio.timeout(3):
        while (await workers.get(task_id))["state"] != "interrupted":
            await asyncio.sleep(0.001)
    assert task_id in workers._runs
    try:
        async with asyncio.timeout(11):
            await workers.aclose()
        assert shutdown_close.is_set()
        assert not calls
    finally:
        unblock.set()
        monkeypatch.setattr(SessionRunner, "close", original)


async def test_stop_during_failed_park_waits_for_store_recovery(
    worker_env: tuple[TaskWorkers, ChatGate, list[list[Message]], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, gate, calls, workspace = worker_env
    entered, release = asyncio.Event(), asyncio.Event()
    recovered = False
    stop_attempts = 0
    change = workers.tasks.change

    async def fail_park(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        raise RuntimeError("park failed")

    async def fail_stop(*args: Any, **kwargs: Any) -> Any:
        nonlocal stop_attempts
        if kwargs.get("state") == "cancelled" and not recovered:
            stop_attempts += 1
            from localmelo.support.sessions.store import SessionStoreError

            raise SessionStoreError("store unavailable")
        return await change(*args, **kwargs)

    monkeypatch.setattr(workers.parks, "park", fail_park)
    monkeypatch.setattr(workers.tasks, "change", fail_stop)
    task_id = await start_held(workers, gate, workspace)
    try:
        await workers.update(task_id, state="pause")
        await asyncio.wait_for(entered.wait(), 3)
        await workers.update(task_id, state="stop")
        release.set()
        await wait_for(lambda: task_id in workers._unavailable)
        await wait_for(lambda: stop_attempts > 0)
        assert task_id in workers._runs and not workers._runs[task_id].done()
        assert not calls
        recovered = True
        async with asyncio.timeout(3):
            while (await workers.get(task_id))["state"] != "cancelled":
                await asyncio.sleep(0.001)
        await wait_for(lambda: task_id not in workers._runs)
    finally:
        release.set()
        recovered = True


async def test_finalization_store_failure_retains_runner_until_recovery(
    worker_env: tuple[TaskWorkers, ChatGate, list[list[Message]], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, gate, calls, workspace = worker_env
    park_attempts = 0
    interrupt_attempts = 0
    recovered = False
    park = workers.parks.park
    interrupt = workers.parks.interrupt

    async def fail_park(*args: Any, **kwargs: Any) -> Any:
        nonlocal park_attempts
        park_attempts += 1
        if not recovered:
            raise RuntimeError("injected park write failure")
        return await park(*args, **kwargs)

    async def fail_interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupt_attempts
        interrupt_attempts += 1
        if not recovered:
            raise RuntimeError("injected interrupt write failure")
        return await interrupt(*args, **kwargs)

    monkeypatch.setattr(workers.parks, "park", fail_park)
    monkeypatch.setattr(workers.parks, "interrupt", fail_interrupt)
    task_id = await start_held(workers, gate, workspace)
    runner = workers._runners[task_id]
    await workers.update(task_id, state="pause")
    await wait_for(lambda: interrupt_attempts > 0)
    row = await workers.get(task_id)
    assert park_attempts >= 3
    assert row["state"] == "held"
    assert workers._runners.get(task_id) is runner
    assert task_id in workers._runs
    assert not calls
    console = SimpleNamespace(store=workers.store, _store_error=None)
    console.store_api_error = lambda exc: Console.store_api_error(
        cast(Console, console), exc
    )
    for state in ("resume", "stop"):
        with pytest.raises(ApiError) as exc:
            async with asyncio.timeout(2):
                await _call(
                    cast(Console, console), workers.update(task_id, state=state)
                )
        assert exc.value.status_code == 503
        assert exc.value.code == "store_unavailable"
        assert workers._runners.get(task_id) is runner
        assert (await workers.get(task_id))["state"] == "held"
    recovered = True
    async with asyncio.timeout(3):
        while True:
            row = await workers.get(task_id)
            if row["state"] in ("parked", "interrupted"):
                break
            await asyncio.sleep(0.001)
    assert row["state"] == "interrupted"
    assert park_attempts == 3
    assert row["card"]
    assert not calls
