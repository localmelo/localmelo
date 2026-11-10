"""A task cut off by a restart continues by itself when nothing is in flight."""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
from localmelo.melo.schema import RESTART_CHECK_HEADER, Message, ToolCall, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway import workers as worker_module
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.park_store import ParkStore
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.startup_recovery import (
    CALL_REASON,
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
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import Cause, SessionStore
from localmelo.tests.gateway.conftest import seed_process
from localmelo.tests.sessions.helpers import core_event, run_result

# Nothing of the run in flight and no call open: what continues by itself.
CLEAR: dict[str, Any] = {"in_flight": False, "calls": ()}


@pytest.mark.parametrize(
    ("facts", "outcome"),
    [
        # The person stopped it, or it ended.
        (RestartFacts(state="cancelled", **CLEAR), "keep"),
        # It already waits for the person, or a continue already waits to run.
        (RestartFacts(state="interrupted", waiting=True), "keep"),
        (RestartFacts(state="queued", continued=True), "keep"),
        # It never started: it starts, or stays paused, as before.
        (RestartFacts(state="queued", never_started=True), "keep"),
        (RestartFacts(state="parked", never_started=True), "keep"),
        # The person paused it: it never goes on by itself.
        (RestartFacts(state="parked", by_person=True, settled=True), "keep"),
        (RestartFacts(state="running", by_person=True, **CLEAR), "wait"),
        # Its last run finished completely.
        (RestartFacts(state="running", finished=True, crash=True), "reconcile"),
        # Cut off: something in flight waits, nothing in flight continues.
        (RestartFacts(state="running", crash=True, in_flight=True, calls=()), "wait"),
        (RestartFacts(state="running", crash=True, **CLEAR), "continue"),
        (RestartFacts(state="interrupted", **CLEAR), "continue"),
        (RestartFacts(state="parked", **CLEAR), "continue"),
        (RestartFacts(state="queued", **CLEAR), "continue"),
        # One open call melo can check itself continues; any other open call waits:
        # one it cannot check, one recorded before the fact existed (or unreadable),
        # more than one.
        (RestartFacts(state="interrupted", in_flight=False, calls=(True,)), "continue"),
        (RestartFacts(state="interrupted", in_flight=False, calls=(False,)), "wait"),
        (RestartFacts(state="interrupted", in_flight=False, calls=(None,)), "wait"),
        (
            RestartFacts(state="interrupted", in_flight=False, calls=(True, False)),
            "wait",
        ),
        (
            RestartFacts(state="interrupted", in_flight=False, calls=(True, True)),
            "wait",
        ),
        (RestartFacts(state="interrupted", in_flight=False), "wait"),  # the default
        # A second crash in a row waits, whatever was in flight.
        (
            RestartFacts(state="running", crash=True, crashed_before=True, **CLEAR),
            "wait",
        ),
        # A clean stop after a crash is no crash loop.
        (
            RestartFacts(state="interrupted", crashed_before=True, **CLEAR),
            "continue",
        ),
        # A run that ended with no complete record of how is not run again.
        (RestartFacts(state="running", crash=True, finished=False), "wait"),
        # A folder missing at start waits; it does not end the task.
        (RestartFacts(state="interrupted", workspace_valid=False, **CLEAR), "wait"),
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


async def test_a_store_from_before_the_local_fact_opens_and_its_old_row_waits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    store = SessionStore(path, writer="gateway")
    await store.open()
    task_id = await _running(await _opened(store), tmp_path)
    await store.aclose()
    # The calls table as the version before made it, with one of its rows open.
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("ALTER TABLE recovery_calls DROP COLUMN local")
        conn.execute(
            "INSERT INTO recovery_calls VALUES (?, 'run', 'old', 'file_write')",
            (task_id,),
        )
    store = SessionStore(path, writer="gateway")
    await store.open()
    try:
        tasks = await _opened(store)
        recovery = RecoveryStore(store)
        # A changing call records its row in the upgraded table.
        await recovery.open_call(
            task_id, execution_id="run", call_id="new", tool="file_write", local=True
        )
        calls = await recovery.open_calls(task_id, execution_id="run")
        assert calls == [("file_write", None), ("file_write", True)]
        await recovery.close_call(task_id, execution_id="run", call_id="new")
        # The old row does not say whether its effect stays local: it waits.
        await StartupRecovery(store, tasks).run()
        record = await tasks.get_task(task_id)
        assert record.state == "interrupted" and record.wait_reason == CALL_REASON
    finally:
        await store.aclose()


async def _opened(store: SessionStore) -> TaskStore:
    tasks = TaskStore(store)
    await tasks.open()
    return tasks


# ── Through the workers, with a held model reply ──


class _Runs:
    """What the fake runner saw: each turn's text, and when one started.

    With *in_call* the held reply comes inside a call that can change
    something, of *tool*, whose effect stays local or not as *local* says.
    """

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.started = asyncio.Event()
        self.in_call = False
        self.tool, self.local = "shell_exec", False


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
            if not runs.in_call:
                runs.started.set()

        async def held(self) -> None:
            if runs.in_call:
                entry = TOOL_ENTRY.get()
                assert entry is not None
                async with entry(runs.tool, runs.local):
                    runs.started.set()
                    await asyncio.Event().wait()
            await asyncio.Event().wait()

        async def wait(self) -> SimpleNamespace:
            # The model's reply is held until a stop, which cancels what the
            # run was in the middle of.
            held = asyncio.create_task(self.held())
            await self.ended.wait()
            held.cancel()
            await asyncio.gather(held, return_exceptions=True)
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


async def test_the_stop_writes_a_cut_task_down_before_its_worker_closes(
    gateway: tuple[Callable[[], TaskWorkers], Path, _Runs],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, workspace, runs = gateway
    first = start()
    task_id = await _a_running_task(first, workspace, runs)
    closing, release = asyncio.Event(), asyncio.Event()
    close = TaskWorkers._close_worker

    async def slow_close(self: TaskWorkers, *args: Any) -> None:
        closing.set()
        await release.wait()  # a worker's tools that take long to close
        await close(self, *args)

    monkeypatch.setattr(TaskWorkers, "_close_worker", slow_close)
    stopping = asyncio.ensure_future(first.aclose())
    await asyncio.wait_for(closing.wait(), 5)
    # Written down already: a kill now still finds a clean stop.
    record = await first.tasks.get_task(task_id)
    assert record.state == "interrupted" and record.wait_reason == STOPPED_REASON
    release.set()
    await stopping
    assert (await first.tasks.get_task(task_id)).state == "interrupted"


@pytest.mark.parametrize("local", [False, True])
async def test_a_stop_in_the_middle_of_a_changing_call_waits_unless_melo_can_check_it(
    gateway: tuple[Callable[[], TaskWorkers], Path, _Runs], local: bool
) -> None:
    start, workspace, runs = gateway
    runs.in_call = True
    runs.tool, runs.local = ("file_write", True) if local else ("shell_exec", False)
    first = start()
    task_id = await _a_running_task(first, workspace, runs)
    await first.aclose()
    # The call may be half done: its record stays, with what the executor found.
    assert await RecoveryStore(first.store).open_calls(
        task_id, execution_id=str(await _execution(first, task_id))
    ) == [(runs.tool, local)]
    again = start()
    await again.open()
    if not local:
        # Not one melo can look at afterwards: it waits as after a kill.
        record = await again.tasks.get_task(task_id)
        assert record.state == "interrupted" and record.wait_reason == CALL_REASON
        assert not again.is_active(task_id) and len(runs.texts) == 1
        return
    # A write in its folder: the continued turn checks it first, as data.
    await asyncio.wait_for(runs.started.wait(), 5)
    assert runs.texts[-1].startswith(RESTART_CHECK_HEADER)
    assert "file_write" in runs.texts[-1] and "Where stopped" in runs.texts[-1]
    assert (await again.get(task_id))["continued_check"] is True


async def test_a_restart_never_carries_an_approval_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()

    class Writes(BaseLLMProvider):
        """A model that asks to write a file until it exists."""

        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            if (workspace / "out.txt").exists():
                return Message(role="assistant", content="written")
            call = ToolCall(
                tool_name="file_write",
                arguments={"path": str(workspace / "out.txt"), "content": "x"},
                id="write-call",
            )
            return Message(role="assistant", content="", tool_calls=[call])

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](Writes()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr(worker_module, "build_session_agent", build)
    made: list[tuple[TaskWorkers, ChatGate, SessionHub]] = []

    async def opened() -> TaskWorkers:
        gate, hub = ChatGate(policy="off"), SessionHub(store)
        workers = TaskWorkers(store, gate, lambda: config, hub)
        made.append((workers, gate, hub))
        await workers.open()
        return workers

    try:
        first = await opened()
        body, _ = await first.create(
            {"objective": "write", "workspace": str(workspace)}
        )
        task_id = str(body["task_id"])
        [before] = [r.request_id for r in await _card(first, task_id)]
        await first.aclose()
        again = await opened()
        # The continued run asks again, with a card of its own.
        [asked] = [r.request_id for r in await _card(again, task_id)]
        assert asked != before
        # An answer for the card from before the stop allows nothing.
        allow = ApprovalDecision(True, origin="user")
        assert again.broker.answer(before, allow) == "unknown"
        assert [r.request_id for r in again.broker.pending(task_id)] == [asked]
        assert not (workspace / "out.txt").exists()
    finally:
        for workers, gate, hub in made:
            await workers.aclose()
            await hub.aclose()
            await gate.close()
        await store.aclose()


async def test_the_persons_stop_survives_a_restart(
    gateway: tuple[Callable[[], TaskWorkers], Path, _Runs],
) -> None:
    start, workspace, runs = gateway
    first = start()
    task_id = await _a_running_task(first, workspace, runs)
    # The person's Stop lands while the gateway's stop has cut the run off.
    closing = asyncio.ensure_future(first.aclose())
    await asyncio.sleep(0)
    assert task_id in first._cut and task_id in first._runs
    with contextlib.suppress(Exception):  # the closing gateway may refuse the rest
        await first.update(task_id, state="stop")
    await closing
    assert (await first.tasks.get_task(task_id)).state == "cancelled"
    again = start()
    await again.open()
    assert (await again.tasks.get_task(task_id)).state == "cancelled"
    assert not again.is_active(task_id)
    assert len(runs.texts) == 1  # nothing ran again


async def _execution(workers: TaskWorkers, task_id: str) -> str | None:
    return (await RecoveryStore(workers.store).provenance(task_id)).execution_id


async def _card(workers: TaskWorkers, task_id: str) -> list[ApprovalRequest]:
    """The approval cards of *task_id* that wait, once there is one."""
    async with asyncio.timeout(5):
        while not workers.broker.pending(task_id):
            await asyncio.sleep(0.01)
    return workers.broker.pending(task_id)
