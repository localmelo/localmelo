"""Provider admission must wait for durable hold and resume transitions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
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
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionStore


async def eventually(check: Callable[[], bool]) -> None:
    async with asyncio.timeout(3):
        while not check():
            await asyncio.sleep(0.001)


@pytest.fixture
async def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TaskWorkers, SessionStore, ChatGate, Path, list[list[str]]]]:
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
    calls: list[list[str]] = []

    class LLM(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            assert messages
            calls.append([])
            task_id = next(iter(workers._runners))
            calls[-1].extend(e.event["type"] for e in await store.list_events(task_id))
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
        yield workers, store, gate, workspace, calls
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_admission_waits_for_hold_and_resume_commits(
    setup: tuple[TaskWorkers, SessionStore, ChatGate, Path, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, store, gate, workspace, calls = setup
    holding, held_continue = asyncio.Event(), asyncio.Event()
    resuming, resume_continue = asyncio.Event(), asyncio.Event()
    original_hold, original_resume = workers.parks.hold, workers.parks.resume

    async def block_hold(*args: Any, **kwargs: Any) -> Any:
        holding.set()
        await held_continue.wait()
        return await original_hold(*args, **kwargs)

    async def block_resume(*args: Any, **kwargs: Any) -> Any:
        resuming.set()
        await resume_continue.wait()
        return await original_resume(*args, **kwargs)

    monkeypatch.setattr(workers.parks, "hold", block_hold)
    monkeypatch.setattr(workers.parks, "resume", block_resume)
    row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
    task_id = row["task_id"]
    try:
        await asyncio.wait_for(holding.wait(), 3)
        assert isinstance(workers._runners[task_id], SessionRunner)
        call = next(c for c in gate._calls.values() if c.meta.session_id == task_id)
        gate.set_machine(Machine(lock="none"))
        await eventually(lambda: call.future is not None and call.future.done())
        assert not calls
        assert (await workers.tasks.get_task(task_id)).state == "running"
        assert "sched.held" not in [
            e.event["type"] for e in await store.list_events(task_id)
        ]
        held_continue.set()
        await asyncio.wait_for(resuming.wait(), 3)
        assert not calls
        assert (await workers.tasks.get_task(task_id)).state == "held"
        types = [e.event["type"] for e in await store.list_events(task_id)]
        assert types.count("sched.held") == 1 and "sched.resumed" not in types
        resume_continue.set()
        await eventually(lambda: bool(calls) and "sched.resumed" in calls[0])
        assert calls[0].count("sched.held") == calls[0].count("sched.resumed") == 1
        assert calls[0].index("sched.held") < calls[0].index("sched.resumed")
    finally:
        held_continue.set()
        resume_continue.set()


async def test_pause_during_resume_commit_parks_admitted_point(
    setup: tuple[TaskWorkers, SessionStore, ChatGate, Path, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, _, gate, workspace, calls = setup
    entered, release = asyncio.Event(), asyncio.Event()
    original = workers.parks.resume

    async def block_resume(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(workers.parks, "resume", block_resume)
    row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
    task_id = row["task_id"]
    pause = None
    try:
        await eventually(lambda: gate.held(task_id) == "Reservation")
        gate.set_machine(Machine(lock="none"))
        await asyncio.wait_for(entered.wait(), 3)
        pause = asyncio.create_task(workers.update(task_id, state="pause"))
        await eventually(lambda: task_id in workers._pause_requested)
        release.set()
        await asyncio.wait_for(pause, 3)
        await eventually(lambda: task_id not in workers._runs)
        assert (await workers.get(task_id))["state"] == "parked"
        assert not calls
    finally:
        release.set()
        if pause is not None:
            await asyncio.gather(pause, return_exceptions=True)


async def test_explicit_priority_resume_clears_park_fence(
    setup: tuple[TaskWorkers, SessionStore, ChatGate, Path, list[list[str]]],
) -> None:
    workers, _, gate, workspace, calls = setup
    row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
    task_id = row["task_id"]
    await eventually(lambda: gate.held(task_id) == "Reservation")
    await workers.update(task_id, state="pause")
    await eventually(lambda: task_id not in workers._runs)
    await workers.update(task_id, state="resume", level="P2")
    await eventually(lambda: gate.held(task_id) == "Reservation")
    gate.set_machine(Machine(lock="none"))
    await eventually(lambda: bool(calls))


async def test_failed_runner_cleanup_does_not_publish_park(
    setup: tuple[TaskWorkers, SessionStore, ChatGate, Path, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, _, gate, workspace, calls = setup
    failed, allow_close = asyncio.Event(), asyncio.Event()
    original = SessionRunner.close
    attempts = 0

    async def close(runner: SessionRunner, *args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            failed.set()
            raise RuntimeError("runner close failed")
        await allow_close.wait()
        await original(runner, *args, **kwargs)

    monkeypatch.setattr(SessionRunner, "close", close)
    row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
    task_id = row["task_id"]
    try:
        await eventually(lambda: gate.held(task_id) == "Reservation")
        await workers.update(task_id, state="pause")
        await asyncio.wait_for(failed.wait(), 3)
        await asyncio.sleep(0.05)
        assert task_id in workers._runs
        assert (await workers.get(task_id))["state"] != "parked"
        assert not calls
        allow_close.set()
        await eventually(lambda: task_id not in workers._runs)
        assert (await workers.get(task_id))["state"] == "parked"
    finally:
        allow_close.set()


@pytest.mark.parametrize("stage", ["hold", "resume"])
@pytest.mark.parametrize(
    "uncertain", [False, True], ids=["failed", "commit-then-cancelled"]
)
async def test_hold_or_resume_write_failure_never_starts_provider(
    setup: tuple[TaskWorkers, SessionStore, ChatGate, Path, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    uncertain: bool,
) -> None:
    workers, store, gate, workspace, calls = setup
    entered, release = asyncio.Event(), asyncio.Event()
    original = getattr(workers.parks, stage)
    original_commit = store._commit
    original_before_commit = store._before_commit
    label = "park_held" if stage == "hold" else "park_running"
    fired = False
    injected = False

    def fail_before_commit(actual: str) -> None:
        nonlocal injected
        original_before_commit(actual)
        if not uncertain and not injected and actual == label:
            injected = True
            raise RuntimeError("injected transition failure")

    async def fail_after_commit(conn: Any, events: Any, actual: str) -> None:
        nonlocal injected
        await original_commit(conn, events, actual)
        if uncertain and not injected and actual == label:
            injected = True
            raise asyncio.CancelledError()

    async def blocked(*args: Any, **kwargs: Any) -> Any:
        nonlocal fired
        if not fired:
            fired = True
            entered.set()
            await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "_before_commit", fail_before_commit)
    monkeypatch.setattr(store, "_commit", fail_after_commit)
    monkeypatch.setattr(workers.parks, stage, blocked)
    row, _ = await workers.create({"prompt": "hello", "workspace": str(workspace)})
    task_id = row["task_id"]
    try:
        if stage == "resume":
            await eventually(lambda: gate.held(task_id) == "Reservation")
            gate.set_machine(Machine(lock="none"))
        await asyncio.wait_for(entered.wait(), 3)
        assert isinstance(workers._runners[task_id], SessionRunner)
        if stage == "hold":
            call = next(c for c in gate._calls.values() if c.meta.session_id == task_id)
            gate.set_machine(Machine(lock="none"))
            await eventually(lambda: call.future is not None and call.future.done())
        assert not calls
        before = [e.event["type"] for e in await store.list_events(task_id)]
        assert before.count("sched.held") == (0 if stage == "hold" else 1)
        assert "sched.resumed" not in before
        release.set()
        await eventually(lambda: task_id not in workers._runs)
        assert injected
        assert not calls
        after = [e.event["type"] for e in await store.list_events(task_id)]
        assert after.count("sched.held") == (int(uncertain) if stage == "hold" else 1)
        assert after.count("sched.resumed") == (
            int(uncertain) if stage == "resume" else 0
        )
    finally:
        release.set()
