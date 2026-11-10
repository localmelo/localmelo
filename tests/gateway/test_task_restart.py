"""A task cut off by a restart continues by itself when nothing is in flight."""

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from localmelo.support.config import Config
from localmelo.support.gateway import workers as worker_module
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.park_store import ParkStore
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.startup_recovery import (
    FENCED_REASON,
    IN_FLIGHT_REASON,
    STOPPED_REASON,
    RestartFacts,
    StartupRecovery,
    restart_outcome,
)
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import Cause, SessionStore
from localmelo.tests.gateway.conftest import seed_process
from localmelo.tests.sessions.helpers import core_event, run_result


@pytest.mark.parametrize(
    ("facts", "outcome"),
    [
        # The person stopped it, or it ended.
        (RestartFacts(state="cancelled", in_flight=False), "keep"),
        # It already waits for the person, or a continue already waits to run.
        (RestartFacts(state="interrupted", waiting=True), "keep"),
        (RestartFacts(state="queued", continued=True), "keep"),
        # It never started: it starts, or stays paused, as before.
        (RestartFacts(state="queued", never_started=True), "keep"),
        (RestartFacts(state="parked", never_started=True), "keep"),
        # The person paused it: it never goes on by itself.
        (RestartFacts(state="parked", by_person=True, settled=True), "keep"),
        (RestartFacts(state="running", by_person=True, in_flight=False), "wait"),
        # Its last run finished completely.
        (RestartFacts(state="running", finished=True, crash=True), "reconcile"),
        # Cut off: something in flight waits, nothing in flight continues.
        (RestartFacts(state="running", crash=True, in_flight=True), "wait"),
        (RestartFacts(state="running", crash=True, in_flight=False), "continue"),
        (RestartFacts(state="interrupted", in_flight=False), "continue"),
        (RestartFacts(state="parked", in_flight=False), "continue"),
        (RestartFacts(state="queued", in_flight=False), "continue"),
        # A second crash in a row waits, whatever was in flight.
        (
            RestartFacts(
                state="running", crash=True, crashed_before=True, in_flight=False
            ),
            "wait",
        ),
        # A clean stop after a crash is no crash loop.
        (
            RestartFacts(state="interrupted", crashed_before=True, in_flight=False),
            "continue",
        ),
    ],
)
def test_restart_outcome_table(facts: RestartFacts, outcome: str) -> None:
    assert restart_outcome(facts) == outcome


# ── A cut-off task as the store leaves it ──


@pytest.fixture
async def tasks(tmp_path: Path) -> AsyncIterator[TaskStore]:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    try:
        yield tasks
    finally:
        await store.aclose()


async def _running(tasks: TaskStore, tmp_path: Path) -> str:
    """A task whose first execution was running, with nothing in flight."""
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}),
        origin="user",
        workspace=str(tmp_path),
        _recovery_proof=True,
    )
    generation, previous = await tasks.park_identity(task.task_id)
    await RecoveryStore(tasks.store).start(
        task.task_id,
        operation_id="launch",
        execution_id="run",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=True,
    )
    return task.task_id


async def _continued(tasks: TaskStore, task_id: str) -> bool:
    """Whether the start continued *task_id*: queued, its resumed turn waiting."""
    record = await tasks.get_task(task_id)
    return record.state == "queued" and await tasks.resume_waiting_turn(task_id)


async def _resumed(store: SessionStore, task_id: str) -> list[dict[str, Any]]:
    return [
        event.event["data"]
        for event in await store.list_events(task_id)
        if event.event.get("type") == "sched.resumed"
    ]


@pytest.mark.parametrize("gone", [True, False])
async def test_a_process_group_of_the_cut_run_decides(
    tasks: TaskStore, tmp_path: Path, gone: bool
) -> None:
    task_id = await _running(tasks, tmp_path)
    await seed_process(
        tasks.store,
        task_id,
        execution_id="run",
        pgid=4321,
        leader_pid=4321,
        boot_id="boot",
        start_token="1:1",
        host_id="host",
    )
    await StartupRecovery(tasks.store, tasks, verifier=lambda group, host: gone).run()
    record = await tasks.get_task(task_id)
    if gone:
        assert await _continued(tasks, task_id)
        assert record.crashes == 1 and record.wait_reason == ""
    else:
        # A group that may be alive: the task waits for the person.
        assert record.state == "interrupted"
        assert record.wait_reason == IN_FLIGHT_REASON


@pytest.mark.parametrize("how", ["scheduler park", "queued again"])
async def test_a_scheduler_park_and_an_earlier_started_queued_task_continue(
    tasks: TaskStore, tmp_path: Path, how: str
) -> None:
    task_id = await _running(tasks, tmp_path)
    await tasks.change(task_id, priority=3, actor="user")
    if how == "scheduler park":
        generation, previous = await tasks.park_identity(task_id)
        await tasks.request_pending_pause(
            task_id,
            expected_generation=generation,
            expected_operation_id=previous,
            operation_id="pause",
            actor="scheduler",
        )
        parks = ParkStore(tasks.store)
        generation, previous = await tasks.park_identity(task_id)
        await parks.hold(
            task_id,
            site="plan",
            step=1,
            attempt=1,
            reason="pause",
            operation_id="hold",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        generation, previous = await tasks.park_identity(task_id)
        await parks.park(
            task_id,
            card="Where stopped: the plan",
            operation_id="park",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        assert (await tasks.get_task(task_id)).state == "parked"
    else:
        await tasks.change(task_id, state="queued", actor="scheduler")
    await StartupRecovery(tasks.store, tasks).run()
    record = await tasks.get_task(task_id)
    assert await _continued(tasks, task_id)
    # Same task, same level and rank; a park or a queue is no crash.
    assert record.level == 3 and record.level_source == "user"
    assert record.crashes == 0
    assert [data.get("reason") for data in await _resumed(tasks.store, task_id)] == [
        "restart"
    ]


async def test_one_bad_record_costs_that_task_and_a_retry_continues_none_twice(
    tasks: TaskStore, tmp_path: Path
) -> None:
    before, bad, after = [await _running(tasks, tmp_path) for _ in range(3)]
    async with tasks.store.component_transaction("seed_bad_intent") as conn:
        await conn.execute(
            "INSERT INTO recovery_operations VALUES (?, 'bad', 'recovery', "
            "'{not json', 'prepared', NULL, NULL, 'running')",
            (bad,),
        )
    startup = StartupRecovery(tasks.store, tasks)
    await startup.run()
    # The open's retry loop, or a second start, runs the stage again.
    await startup.run()
    for task_id in (before, after):
        assert await _continued(tasks, task_id)
        assert len(await _resumed(tasks.store, task_id)) == 1
    fenced = await tasks.get_task(bad)
    assert fenced.state == "interrupted" and fenced.wait_reason == FENCED_REASON
    assert not await _resumed(tasks.store, bad)


# ── Through the workers, with a held model reply ──


class _Runs:
    """What the fake runner saw: each turn's text, and when one started."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.started = asyncio.Event()


def _fakes(monkeypatch: pytest.MonkeyPatch, runs: _Runs) -> None:
    class Agent:
        async def close(self) -> None:
            pass

    class Runner:
        def __init__(
            self, agent: Any, store: SessionStore, session_id: str, **kwargs: Any
        ) -> None:
            self.store, self.task_id = store, session_id
            self.ended = asyncio.Event()
            self.status, self.cause = "cancelled", "run"
            self.turn = 0
            self.busy = False

        def observe(self, ask: Any) -> Any:
            return ask

        async def resume(self, **kwargs: Any) -> int:
            return 0

        async def start_turn(self, text: str, **kwargs: Any) -> None:
            turn = await self.store.begin_turn(
                self.task_id,
                text,
                holder="worker",
                client_request_id=kwargs.get("client_request_id"),
            )
            self.turn, self.busy = turn.turn, True
            # A changing call that returned, so nothing is in flight.
            await self.store.append_events(
                self.task_id,
                turn.turn,
                [
                    core_event("run.started"),
                    core_event("tool.started", seq=2, name="file_write"),
                    core_event("tool.finished", seq=3, name="file_write"),
                ],
                holder="worker",
            )
            runs.texts.append(text)
            runs.started.set()

        async def wait(self) -> SimpleNamespace:
            await self.ended.wait()  # the model's reply is held until a stop
            await self.store.finish_turn(
                self.task_id,
                self.turn,
                holder="worker",
                status=self.status,
                cause=cast(Cause, self.cause),
                result=run_result(self.status),
                answer="",
            )
            self.busy = False
            return SimpleNamespace(status=self.status)

        def cancel(self, *, cause: str = "run", **kwargs: Any) -> bool:
            self.cause = cause
            self.ended.set()
            return True

        async def close(self, **kwargs: Any) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        return Agent()

    monkeypatch.setattr(worker_module, "SessionRunner", Runner)
    monkeypatch.setattr(worker_module, "build_session_agent", build)


@pytest.fixture
async def gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Callable[[], TaskWorkers], Path, _Runs]]:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    config = Config()
    config.gateway.workspace = str(workspace)
    runs = _Runs()
    _fakes(monkeypatch, runs)
    made: list[tuple[TaskWorkers, ChatGate, SessionHub]] = []

    def start() -> TaskWorkers:
        gate, hub = ChatGate(policy="off"), SessionHub(store)
        workers = TaskWorkers(store, gate, lambda: config, hub)
        made.append((workers, gate, hub))
        return workers

    try:
        yield start, workspace, runs
    finally:
        for workers, gate, hub in made:
            await workers.aclose()
            await hub.aclose()
            await gate.close()
        await store.aclose()


async def _a_running_task(workers: TaskWorkers, workspace: Path, runs: _Runs) -> str:
    await workers.open()
    body, _ = await workers.create({"objective": "work", "workspace": str(workspace)})
    await asyncio.wait_for(runs.started.wait(), 5)
    runs.started.clear()
    return str(body["task_id"])


async def test_a_clean_stop_then_a_start_continues_the_task(
    gateway: tuple[Callable[[], TaskWorkers], Path, _Runs],
) -> None:
    start, workspace, runs = gateway
    first = start()
    task_id = await _a_running_task(first, workspace, runs)
    await first.aclose()
    stopped = await first.tasks.get_task(task_id)
    # A stop is an involuntary pause, not the person's Stop.
    assert stopped.state == "interrupted" and stopped.wait_reason == STOPPED_REASON
    assert stopped.crashes == 0
    again = start()
    await again.open()
    await asyncio.wait_for(runs.started.wait(), 5)
    record = await again.tasks.get_task(task_id)
    assert record.state == "running"
    # The continued turn starts from the card of where the run stopped.
    assert runs.texts[-1].startswith("Where stopped")
    assert "file_write" in runs.texts[-1]
    resumed = await _resumed(again.store, task_id)
    assert [data.get("reason") for data in resumed] == ["restart"]
    assert (await again.get(task_id))["continued_at"] is not None


async def test_the_persons_stop_survives_a_restart(
    gateway: tuple[Callable[[], TaskWorkers], Path, _Runs],
) -> None:
    start, workspace, runs = gateway
    first = start()
    task_id = await _a_running_task(first, workspace, runs)
    await first.update(task_id, state="stop")
    async with asyncio.timeout(5):
        while task_id in first._runs:
            await asyncio.sleep(0.01)
    await first.aclose()
    assert (await first.tasks.get_task(task_id)).state == "cancelled"
    again = start()
    await again.open()
    assert (await again.tasks.get_task(task_id)).state == "cancelled"
    assert not again.is_active(task_id)
    assert len(runs.texts) == 1  # nothing ran again
