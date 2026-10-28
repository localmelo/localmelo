"""Worker capture admission and launch-fence boundaries."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
from localmelo.melo.executor.process_gate import PROCESS_RECORD, gated_exec
from localmelo.support.config import Config
from localmelo.support.gateway import workers as worker_module
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.park_store import ParkStore
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore, TaskStoreError
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore


@pytest.fixture
async def setup(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    worker = TaskWorkers.__new__(TaskWorkers)
    worker.store = store
    worker.tasks = tasks
    worker._lock = asyncio.Lock()
    worker._closed = False
    worker._stopping = set()
    try:
        yield store, tasks, worker
    finally:
        await store.aclose()


async def create(tasks, tmp_path, *, proof):
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}),
        origin="user",
        workspace=str(tmp_path),
        _recovery_proof=proof,
    )
    return task.task_id


async def test_ordinary_start_is_atomically_untrusted(setup, tmp_path, monkeypatch):
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=False)
    original = store._before_commit

    def fail(label):
        original(label)
        if label == "task_change":
            raise RuntimeError("rollback")

    monkeypatch.setattr(store, "_before_commit", fail)
    with pytest.raises(RuntimeError):
        await tasks.change(
            task, state="running", actor="scheduler", _capture_execution="run"
        )
    assert (await tasks.get_task(task)).state == "queued"
    monkeypatch.setattr(store, "_before_commit", original)
    await tasks.change(
        task, state="running", actor="scheduler", _capture_execution="run"
    )
    assert not await RecoveryStore(store).capture_complete(task, execution_id="run")
    await worker._capture_incomplete(task, "run")
    assert not await RecoveryStore(store).capture_complete(task, execution_id="run")


async def test_first_fence_receipt_survives_capture_and_hold_but_not_stop(
    setup, tmp_path
):
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=True)
    recovery = RecoveryStore(store)
    generation, previous = await tasks.park_identity(task)
    args = dict(
        operation_id="start",
        execution_id="run",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=True,
    )
    await recovery.start(task, **args)
    assert await recovery.capture_complete(task, execution_id="run")
    parks = ParkStore(store)
    generation, previous = await tasks.park_identity(task)
    await parks.hold(
        task,
        site="plan",
        step=0,
        attempt=1,
        reason="hold",
        expected_generation=generation,
        expected_operation_id=previous,
        operation_id="hold",
    )
    assert not (await recovery.start(task, **args)).acquired
    generation, previous = await tasks.park_identity(task)
    await parks.resume(
        task,
        expected_generation=generation,
        expected_operation_id=previous,
        operation_id="resume",
    )
    await worker._capture_incomplete(task, "run")
    assert not (await recovery.start(task, **args)).acquired
    generation, previous = await tasks.park_identity(task)
    await tasks.change(
        task,
        state="cancelled",
        operation_id="stop",
        expected_generation=generation,
        expected_operation_id=previous,
    )
    with pytest.raises(TaskStoreError):
        await recovery.start(task, **args)
    with pytest.raises(asyncio.CancelledError):
        worker._stopping.add(task)
        await worker._capture_incomplete(task, "run")


async def test_marker_retries_then_succeeds(setup, tmp_path, monkeypatch):
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=True)
    generation, previous = await tasks.park_identity(task)
    await RecoveryStore(store).start(
        task,
        operation_id="start",
        execution_id="run",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=True,
    )
    original = RecoveryStore.mark_capture_incomplete
    calls = 0

    async def flaky(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("store unavailable")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(RecoveryStore, "mark_capture_incomplete", flaky)
    await worker._capture_incomplete(task, "run")
    assert calls == 3
    assert not await RecoveryStore(store).capture_complete(task, execution_id="run")


async def test_marker_persistent_failure_never_admits_call(
    setup, tmp_path, monkeypatch
):
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=True)
    generation, previous = await tasks.park_identity(task)
    await RecoveryStore(store).start(
        task,
        operation_id="start",
        execution_id="run",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=True,
    )

    async def fail(*args, **kwargs):
        raise OSError("store unavailable")

    monkeypatch.setattr(worker_module, "_CAPTURE_RETRY_SECONDS", 0.12)
    monkeypatch.setattr(RecoveryStore, "mark_capture_incomplete", fail)
    with pytest.raises(
        RuntimeError,
        match="^execution evidence could not be recorded; the call was not run; task store unavailable$",
    ):
        await worker._capture_incomplete(task, "run")
    assert await RecoveryStore(store).capture_complete(task, execution_id="run")


async def test_identity_failure_requires_durable_incomplete(
    setup, tmp_path, monkeypatch
):
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=True)
    generation, previous = await tasks.park_identity(task)
    await RecoveryStore(store).start(
        task,
        operation_id="start",
        execution_id="run",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=True,
    )

    def fail(pid):
        raise OSError("unsupported")

    monkeypatch.setattr(worker_module, "process_identity", fail)
    await worker._capture_process(task, "run", 123)
    assert not await RecoveryStore(store).capture_complete(task, execution_id="run")


@pytest.mark.parametrize("record_fails", [False, True])
async def test_child_is_released_only_after_record_or_durable_fallback(
    setup, tmp_path, monkeypatch, record_fails
):
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=True)
    recovery = RecoveryStore(store)
    generation, previous = await tasks.park_identity(task)
    await recovery.start(
        task,
        operation_id="start",
        execution_id="run",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=True,
    )
    path = tmp_path / "child-ran"
    monkeypatch.setattr(
        worker_module, "process_identity", lambda pid: (pid, pid, "boot", "birth")
    )
    if record_fails:

        async def fail(*args, **kwargs):
            raise OSError("record failed")

        monkeypatch.setattr(RecoveryStore, "record_captured_process", fail)

    async def record(pid):
        assert not path.exists()
        await worker._capture_process(task, "run", pid)
        assert not path.exists()
        if record_fails:
            assert not await recovery.capture_complete(task, execution_id="run")
        else:
            assert await recovery.processes(task, execution_id="run") == [
                (pid, pid, "boot", "birth")
            ]

    proc = await gated_exec(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
            str(path),
        ],
        record=record,
    )
    assert await proc.wait() == 0
    assert path.exists()


async def test_failed_fallback_never_releases_child(setup, tmp_path, monkeypatch):
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=True)
    generation, previous = await tasks.park_identity(task)
    await RecoveryStore(store).start(
        task,
        operation_id="start",
        execution_id="run",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=True,
    )
    path = tmp_path / "child-ran"

    def fail_identity(pid):
        raise OSError("unsupported")

    async def fail_marker(*args, **kwargs):
        raise OSError("store unavailable")

    monkeypatch.setattr(worker_module, "process_identity", fail_identity)
    monkeypatch.setattr(worker_module, "_CAPTURE_RETRY_SECONDS", 0.1)
    monkeypatch.setattr(RecoveryStore, "mark_capture_incomplete", fail_marker)
    with pytest.raises(RuntimeError, match="the call was not run"):
        await gated_exec(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(path),
            ],
            record=lambda pid: worker._capture_process(task, "run", pid),
        )
    assert not path.exists()


@pytest.mark.parametrize("proof", [False, True])
async def test_worker_context_hooks_bind_execution(tmp_path, monkeypatch, proof):
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
    checked = asyncio.Event()
    observations = []

    class Agent:
        async def close(self):
            pass

    class Runner:
        busy = False

        def __init__(self, agent, store, session_id, **kwargs):
            self.session_id = session_id

        def observe(self, ask):
            return ask

        async def start_turn(self, text, **kwargs):
            observations.append(kwargs.get("client_request_id"))

        async def wait(self):
            async with store.component_transaction("capture_test") as conn:
                row = await (
                    await conn.execute(
                        "SELECT execution_id FROM recovery_capture WHERE task_id = ? AND active = 1",
                        (self.session_id,),
                    )
                ).fetchone()
            execution = row[0]
            assert observations[0] == (execution if proof else None)
            entry = TOOL_ENTRY.get()
            process = PROCESS_RECORD.get()
            assert entry is not None and process is not None
            recovery = RecoveryStore(store)
            assert (
                await recovery.capture_complete(self.session_id, execution_id=execution)
                is proof
            )
            await process(123)
            assert await recovery.processes(
                self.session_id, execution_id=execution
            ) == [(123, 123, "boot", "birth")]
            await entry()
            assert not await recovery.capture_complete(
                self.session_id, execution_id=execution
            )
            checked.set()
            return SimpleNamespace(status="completed")

        async def close(self, **kwargs):
            pass

    async def build(**kwargs):
        return Agent()

    monkeypatch.setattr(worker_module, "SessionRunner", Runner)
    monkeypatch.setattr(worker_module, "build_session_agent", build)
    monkeypatch.setattr(
        worker_module, "process_identity", lambda pid: (pid, pid, "boot", "birth")
    )
    try:
        if proof:
            await workers.tasks.open()
            await create(workers.tasks, workspace, proof=True)
        await workers.open()
        if not proof:
            await workers.create({"objective": "work", "workspace": str(workspace)})
        await asyncio.wait_for(checked.wait(), 3)
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
