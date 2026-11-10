"""Worker capture admission and launch-fence boundaries."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
from localmelo.melo.executor.process_gate import PROCESS_RECORD, gated_exec
from localmelo.melo.schema import RESTART_CHECK_HEADER
from localmelo.support.config import Config
from localmelo.support.gateway import workers as worker_module
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.park_store import ParkStore
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.restart_check import CHECK_ASK
from localmelo.support.gateway.startup_recovery import (
    CALL_REASON,
    StartupRecovery,
)
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore, TaskStoreError
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.gateway.conftest import seed_process

CaptureSetup = tuple[SessionStore, TaskStore, TaskWorkers]


@pytest.fixture
async def setup(tmp_path: Path) -> AsyncIterator[CaptureSetup]:
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
    worker._invalid_workspace = set()
    try:
        yield store, tasks, worker
    finally:
        await store.aclose()


async def create(tasks: TaskStore, tmp_path: Path, *, proof: bool) -> str:
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}),
        origin="user",
        workspace=str(tmp_path),
        _recovery_proof=proof,
    )
    return task.task_id


async def test_first_fence_receipt_survives_capture_and_hold_but_not_stop(
    setup: CaptureSetup, tmp_path: Path
) -> None:
    store, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=True)
    recovery = RecoveryStore(store)
    generation, previous = await tasks.park_identity(task)
    args: dict[str, Any] = dict(
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
    assert await recovery.capture_complete(task, execution_id="run")
    with pytest.raises(worker_module.TaskConflictError):
        await worker._capture_incomplete(task, "run")
    assert not (await recovery.start(task, **args)).acquired
    generation, previous = await tasks.park_identity(task)
    await parks.resume(
        task,
        expected_generation=generation,
        expected_operation_id=previous,
        operation_id="resume",
    )
    assert await recovery.capture_complete(task, execution_id="run")
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


async def test_marker_retries_then_succeeds(
    setup: CaptureSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    async def flaky(self: RecoveryStore, *args: Any, **kwargs: Any) -> None:
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
    setup: CaptureSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    async def fail(*args: Any, **kwargs: Any) -> None:
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
    setup: CaptureSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    def fail(pid: int) -> None:
        raise OSError("unsupported")

    monkeypatch.setattr(worker_module, "process_identity", fail)
    await worker._capture_process(task, "run", 123)
    assert not await RecoveryStore(store).capture_complete(task, execution_id="run")


@pytest.mark.parametrize("record_fails", [False, True])
async def test_child_is_released_only_after_record_or_durable_fallback(
    setup: CaptureSetup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_fails: bool,
) -> None:
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
        worker_module,
        "process_identity",
        lambda pid: (pid, pid, "host", "boot", "birth"),
    )
    if record_fails:

        async def fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("record failed")

        monkeypatch.setattr(RecoveryStore, "record_captured_process", fail)

    async def record(pid: int) -> None:
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


async def test_failed_fallback_never_releases_child(
    setup: CaptureSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    def fail_identity(pid: int) -> None:
        raise OSError("unsupported")

    async def fail_marker(*args: Any, **kwargs: Any) -> None:
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
async def test_worker_context_hooks_bind_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proof: bool
) -> None:
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
    observations: list[str | None] = []

    class Agent:
        async def close(self) -> None:
            pass

    class Runner:
        busy = False

        def __init__(
            self, agent: Any, store: SessionStore, session_id: str, **kwargs: Any
        ) -> None:
            self.session_id = session_id

        def observe(self, ask: Any) -> Any:
            return ask

        async def start_turn(self, text: str, **kwargs: Any) -> None:
            observations.append(kwargs.get("client_request_id"))

        async def wait(self) -> SimpleNamespace:
            async with store.component_transaction("capture_test") as conn:
                row = await (
                    await conn.execute(
                        "SELECT execution_id FROM recovery_capture WHERE task_id = ? AND active = 1",
                        (self.session_id,),
                    )
                ).fetchone()
            assert row is not None
            execution = row[0]
            assert observations[0] == execution
            entry = TOOL_ENTRY.get()
            process = PROCESS_RECORD.get()
            assert entry is not None and process is not None
            recovery = RecoveryStore(store)
            assert await recovery.capture_complete(
                self.session_id, execution_id=execution
            )
            await process(123)
            assert await recovery.processes(
                self.session_id, execution_id=execution
            ) == [(123, 123, "boot", "birth")]
            # A call that can change something is named while it runs.
            async with entry("shell_exec", False):
                assert await recovery.open_calls(
                    self.session_id, execution_id=execution
                ) == [("shell_exec", False)]
            assert not await recovery.open_calls(
                self.session_id, execution_id=execution
            )
            assert await recovery.capture_complete(
                self.session_id, execution_id=execution
            )
            checked.set()
            return SimpleNamespace(status="completed")

        def cancel(self, *args: Any, **kwargs: Any) -> bool:
            # The runner stays registered until its run is finalized (the ending's
            # folder reads included); like SessionRunner's, a cancel after the turn
            # finished stops nothing.
            return False

        async def close(self, **kwargs: Any) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        return Agent()

    monkeypatch.setattr(worker_module, "SessionRunner", Runner)
    monkeypatch.setattr(worker_module, "build_session_agent", build)
    monkeypatch.setattr(
        worker_module,
        "process_identity",
        lambda pid: (pid, pid, "host", "boot", "birth"),
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


async def test_current_execution_guard_only_preserves_its_own_capture(
    setup: CaptureSetup, tmp_path: Path
) -> None:
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
    assert await recovery.capture_complete(task, execution_id="run")

    # A running transition with no execution named keeps the live capture.
    await tasks.change(task, state="running", actor="scheduler")
    assert await recovery.capture_complete(task, execution_id="run")

    # A stale execution cannot revive or preserve the current capture.
    with pytest.raises(TaskStoreError, match="launch capture is no longer active"):
        await tasks.change(
            task,
            state="running",
            actor="scheduler",
            _capture_execution="stale",
            _capture_existing=True,
        )
    assert await recovery.capture_complete(task, execution_id="run")

    # A new ordinary execution retires it and starts complete: its row is
    # written before anything of it runs.
    other = await create(tasks, tmp_path, proof=False)
    await tasks.change(
        other, state="running", actor="scheduler", _capture_execution="new"
    )
    async with store.component_transaction("capture_row") as conn:
        row = await (
            await conn.execute(
                "SELECT complete, trusted FROM recovery_capture "
                "WHERE task_id = ? AND execution_id = ? AND active = 1",
                (other, "new"),
            )
        ).fetchone()
    assert row == (1, 1)
    assert await recovery.capture_complete(other, execution_id="new")

    # Leaving running retires the execution's capture.
    await tasks.change(task, state="done")
    assert not await recovery.capture_complete(task, execution_id="run")


async def test_park_resume_starts_a_new_execution(
    setup: CaptureSetup, tmp_path: Path
) -> None:
    store, tasks, _ = setup
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
    await seed_process(
        store,
        task,
        execution_id="run",
        pgid=11,
        leader_pid=11,
        boot_id="boot",
        start_token="old",
    )
    await tasks.pause(task)
    assert (await tasks.get_task(task)).state == "parked"
    await tasks.resume(task)
    assert (await tasks.get_task(task)).state == "queued"

    # The launch after a park is a new execution with its own capture; the
    # parked execution keeps its process rows as its own evidence.
    await tasks.change(
        task, state="running", actor="scheduler", _capture_execution="next"
    )
    assert await recovery.processes(task, execution_id="next") == []
    assert await recovery.processes(task, execution_id="run") == [
        (11, 11, "boot", "old")
    ]


# A separate process crashes with os._exit, so no Python cleanup runs between
# the durable record and the gate release.
_CAPTURE_CRASH_DRIVER = """
import asyncio, os, sys
from pathlib import Path

async def main():
    db, workspace, mode, marker, pid_file = sys.argv[1:6]
    from localmelo.melo.executor.process_gate import gated_exec
    from localmelo.support.gateway.recovery_store import RecoveryStore
    from localmelo.support.gateway.task_policy import TaskSpec
    from localmelo.support.gateway.task_store import TaskStore
    from localmelo.support.gateway.workers import TaskWorkers
    from localmelo.support.sessions.store import SessionStore

    store = SessionStore(Path(db), writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}), origin="user",
        workspace=workspace, _recovery_proof=True,
    )
    generation, previous = await tasks.park_identity(task.task_id)
    await RecoveryStore(store).start(
        task.task_id, operation_id="start", execution_id="run",
        expected_generation=generation, expected_operation_id=previous,
        _capture=True,
    )
    worker = TaskWorkers.__new__(TaskWorkers)
    worker.store, worker.tasks = store, tasks
    worker._lock, worker._closed, worker._stopping = asyncio.Lock(), False, set()
    worker._invalid_workspace = set()
    if mode in ("in-call", "after-call"):
        # A call melo cannot check itself (a command no review let run alone).
        async with TaskWorkers._call_record(
            worker, task.task_id, "run", "shell_exec", False
        ):
            if mode == "in-call":
                os._exit(9)
        os._exit(9)
    if mode == "in-write":
        # The executor classes a file write in the task's folder as local.
        from localmelo.melo.checker.checker import Checker
        from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
        from localmelo.melo.executor import Executor, register_builtins
        from localmelo.melo.executor.models import ExecutionRequest
        from localmelo.melo.memory.coordinator import Hippo

        hippo = Hippo()
        executor = Executor(hippo, Checker(mode="interactive"), workspace_root=workspace)
        register_builtins(
            executor, hippo, names={"file_write"}, workspace_root=workspace
        )

        async def dies(**kwargs):
            os._exit(9)

        executor.register("file_write", dies)
        TOOL_ENTRY.set(
            lambda tool, local: TaskWorkers._call_record(
                worker, task.task_id, "run", tool, local
            )
        )
        await executor.execute_structured(ExecutionRequest(
            tool_name="file_write", arguments={"path": "notes.md", "content": "x"},
        ))
        os._exit(3)  # the write returned: the test fails

    async def record(pid):
        Path(pid_file).write_text(str(pid))
        if mode == "before-record":
            os._exit(9)
        await TaskWorkers._capture_process(worker, task.task_id, "run", pid)
        if mode == "after-record":
            os._exit(9)

    proc = await gated_exec(
        [sys.executable, "-c",
         "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", marker],
        record=record,
    )
    if mode == "after-release":
        os._exit(9)
    await proc.wait()

asyncio.run(main())
"""


@pytest.mark.parametrize(
    ("mode", "recorded", "released"),
    [
        ("before-record", False, False),
        ("after-record", True, False),
        ("after-release", True, True),
    ],
)
async def test_worker_crash_keeps_capture_before_release(
    tmp_path: Path, mode: str, recorded: bool, released: bool
) -> None:
    db, workspace = tmp_path / "sessions.db", tmp_path / "workspace"
    workspace.mkdir()
    marker, pid_file = tmp_path / "marker", tmp_path / "pid"
    driver = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CAPTURE_CRASH_DRIVER,
        str(db),
        str(workspace),
        mode,
        str(marker),
        str(pid_file),
        stderr=asyncio.subprocess.PIPE,
    )
    _, errors = await asyncio.wait_for(driver.communicate(), 30)
    assert driver.returncode == 9, errors.decode()
    store = SessionStore(db, writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        (task,) = await tasks.list_tasks()
        recovery = RecoveryStore(store)
        assert await recovery.capture_complete(task.task_id, execution_id="run")
        processes = await recovery.processes(task.task_id, execution_id="run")
        assert bool(processes) is recorded
        if released:
            for _ in range(100):
                if marker.exists():
                    break
                await asyncio.sleep(0.05)
            assert marker.exists()
            return
        assert not marker.exists()
        pid = int(pid_file.read_text())
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("an unreleased child survived its crashed parent")
    finally:
        await store.aclose()


@pytest.mark.parametrize("mode", ["in-call", "after-call", "in-write"])
async def test_a_death_in_a_changing_call_waits_and_after_it_continues(
    tmp_path: Path, mode: str
) -> None:
    db, workspace = tmp_path / "sessions.db", tmp_path / "workspace"
    workspace.mkdir()
    driver = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CAPTURE_CRASH_DRIVER,
        str(db),
        str(workspace),
        mode,
        str(tmp_path / "marker"),
        str(tmp_path / "pid"),
        stderr=asyncio.subprocess.PIPE,
    )
    _, errors = await asyncio.wait_for(driver.communicate(), 30)
    assert driver.returncode == 9, errors.decode()
    store = SessionStore(db, writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        (task,) = await tasks.list_tasks()
        await StartupRecovery(store, tasks).run()
        record = await tasks.get_task(task.task_id)
        if mode == "in-call":
            # The call may be half done, and melo cannot check it: the task
            # waits for the person.
            assert record.state == "interrupted"
            assert record.wait_reason == CALL_REASON
            return
        assert record.state == "queued"
        assert await tasks.resume_waiting_turn(task.task_id)
        continued = await tasks.continued(task.task_id)
        assert continued is not None
        if mode == "in-write":
            # A write in its folder may be half done: its turn checks it first.
            assert continued.check.startswith(RESTART_CHECK_HEADER)
            assert "file_write" in continued.check and CHECK_ASK in continued.check
        else:
            assert continued.check == ""
    finally:
        await store.aclose()


async def test_the_hook_stands_down_when_the_folder_was_refused(
    setup: CaptureSetup, tmp_path: Path
) -> None:
    """A run whose folder was refused is cancelled; its tools never start."""
    _, tasks, worker = setup
    task = await create(tasks, tmp_path, proof=False)
    worker._invalid_workspace.add(task)
    with pytest.raises(asyncio.CancelledError):
        await worker._capture_incomplete(task, "no-capture-yet")
