"""Controlled Resume: the decision table, tokens and crash boundaries."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import localmelo.support.gateway.workers as gateway_workers
from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway import process_identity
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.resume_evidence import (
    ConfirmationRequiredError,
    RecoveryUnknownError,
    ResumeRefusedError,
    StaleConfirmationError,
)
from localmelo.support.gateway.resume_tokens import ResumeBinding, ResumeTokens
from localmelo.support.gateway.startup_recovery import StartupRecovery
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskConflictError, TaskStoreError
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.gateway.workers import (
    collect_evidence as collect_resume_evidence,
)
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore

OLD_BOOT = "11111111-1111-1111-1111-111111111111"
NEW_BOOT = "22222222-2222-2222-2222-222222222222"
HOST = "host-a"


@pytest.fixture
async def worker_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TaskWorkers, Path, SessionStore]]:
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
        yield workers, workspace, store
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def _no_schedule() -> None:
    return None


def _freeze(workers: TaskWorkers, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers, "_schedule", _no_schedule)


async def _set_card(store: SessionStore, task_id: str, card: str) -> None:
    async with store.component_transaction("seed_card") as conn:
        await conn.execute(
            "UPDATE tasks SET card = ? WHERE task_id = ?", (card, task_id)
        )


async def _started(
    workers: TaskWorkers,
    workspace: Path,
    *,
    capture: bool = False,
    processes: tuple[tuple[int, int, str, str], ...] = (),
    card: str = "Recovery card",
) -> str:
    """A launched execution, the way a restart leaves it just before opening."""
    tasks = workers.tasks
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "do work"}),
        origin="user",
        workspace=str(workspace),
        _recovery_proof=True,
    )
    generation, previous = await tasks.park_identity(task.task_id)
    recovery = RecoveryStore(workers.store)
    await recovery.start(
        task.task_id,
        operation_id="launch",
        execution_id="exec-1",
        expected_generation=generation,
        expected_operation_id=previous,
        _capture=capture,
    )
    for pgid, leader, boot, token in processes:
        await recovery.record_process(
            task.task_id,
            execution_id="exec-1",
            pgid=pgid,
            leader_pid=leader,
            boot_id=boot,
            start_token=token,
            host_id=HOST,
        )
    if card:
        await _set_card(workers.store, task.task_id, card)
    return task.task_id


async def _open_and_freeze(
    workers: TaskWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    await workers.open()
    _freeze(workers, monkeypatch)


async def _resume(
    workers: TaskWorkers, task_id: str, operation_id: str, token: str | None = None
) -> dict:
    return await workers.resume(
        task_id, operation_id=operation_id, confirmation_token=token
    )


async def _events(store: SessionStore, task_id: str, kind: str) -> list[Any]:
    found: list[Any] = []
    after = 0
    while True:
        page = await store.list_events(task_id, after_id=after)
        found.extend(e for e in page if e.event.get("type") == kind)
        if len(page) < 1000:
            return found
        assert page[-1].id is not None
        after = page[-1].id


async def test_trusted_complete_capture_with_group_gone_resumes_free(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(
        workers, workspace, capture=True, processes=((4321, 4321, "boot", "1:1"),)
    )
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: True
    task = await _resume(workers, task_id, "resume-1")
    assert task["state"] == "queued"
    assert not await _events(store, task_id, "sched.resume_confirmed")
    assert await workers.tasks.has_resume_receipt(task_id, "resume-1")


async def test_second_crash_advances_the_resumed_execution_identity(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace, capture=True)
    await _open_and_freeze(workers, monkeypatch)
    assert (await _resume(workers, task_id, "resume-1"))["state"] == "queued"
    # The resumed run starts a new execution through the ordinary path.
    await workers.tasks.change(
        task_id,
        state="running",
        actor="scheduler",
        _capture_execution="exec-2",
        _capture_existing=False,
    )
    await StartupRecovery(store, workers.tasks).run()
    provenance = await RecoveryStore(store).provenance(task_id)
    assert provenance.execution_id == "exec-2"
    with pytest.raises(ConfirmationRequiredError) as preview:
        await _resume(workers, task_id, "resume-2")
    accepted = await _resume(workers, task_id, "resume-2", preview.value.token)
    assert accepted["state"] == "queued"


async def test_unverifiable_process_requires_single_use_confirmation(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(
        workers, workspace, capture=True, processes=((4321, 4321, "boot", "1:1"),)
    )
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: False
    with pytest.raises(ConfirmationRequiredError) as preview:
        await _resume(workers, task_id, "resume-1")
    token = preview.value.token
    assert preview.value.warning.startswith("A process from before the restart")
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"
    accepted = await _resume(workers, task_id, "resume-1", token)
    assert accepted["state"] == "queued"
    assert len(await _events(store, task_id, "sched.resume_confirmed")) == 1
    assert workers._resume_tokens.pending(task_id) == 0
    with pytest.raises(StaleConfirmationError):
        await _resume(workers, task_id, "resume-2", token)


async def test_every_recorded_group_must_be_gone(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(
        workers,
        workspace,
        capture=True,
        processes=((4321, 4321, "boot", "1:1"), (8765, 8765, "boot", "2:2")),
    )
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: record[0] == 4321
    with pytest.raises(ConfirmationRequiredError):
        await _resume(workers, task_id, "resume-1")
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"


async def test_missing_card_blocks_a_free_candidate(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace, capture=True, card="")
    await _open_and_freeze(workers, monkeypatch)
    with pytest.raises(ResumeRefusedError) as refused:
        await _resume(workers, task_id, "resume-1")
    assert refused.value.reason == "missing_card"
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"


async def test_verifier_exception_is_never_evidence(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(
        workers, workspace, capture=True, processes=((4321, 4321, "boot", "1:1"),)
    )
    await _open_and_freeze(workers, monkeypatch)

    def denied(record: tuple[int, int, str, str], host: str) -> bool:
        raise PermissionError("cannot inspect process group")

    workers._process_verifier = denied
    with pytest.raises(ConfirmationRequiredError):
        await _resume(workers, task_id, "resume-1")
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"
    assert not await _events(store, task_id, "sched.resume_confirmed")


async def test_token_is_bound_to_the_cas_generation(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(
        workers, workspace, capture=True, processes=((4321, 4321, "boot", "1:1"),)
    )
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: False
    token = await _preview(workers, task_id, "resume-1")
    async with workers.store.component_transaction("seed_generation") as conn:
        await conn.execute(
            "UPDATE park_priority SET generation = generation + 1 WHERE task_id = ?",
            (task_id,),
        )
    with pytest.raises(StaleConfirmationError):
        await _resume(workers, task_id, "resume-1", token)
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"


async def test_incomplete_capture_is_unknown_not_death(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(
        workers,
        workspace,
        capture=False,
        processes=((4321, 4321, "boot", "1:1"),),
    )
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: True
    with pytest.raises(ConfirmationRequiredError):
        await _resume(workers, task_id, "resume-1")
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"


async def test_verified_boot_change_counts_as_gone(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(
        workers,
        workspace,
        capture=True,
        processes=((4321, 4321, OLD_BOOT, "1:1"),),
    )
    await _open_and_freeze(workers, monkeypatch)
    monkeypatch.setattr(process_identity, "_boot", lambda: NEW_BOOT)
    monkeypatch.setattr(process_identity, "_host", lambda: HOST)
    resumed = await _resume(workers, task_id, "resume-1")
    assert resumed["state"] == "queued"
    assert not await _events(store, task_id, "sched.resume_confirmed")


async def test_boot_change_on_another_host_requires_confirmation(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(
        workers,
        workspace,
        capture=True,
        processes=((4321, 4321, OLD_BOOT, "1:1"),),
    )
    await _open_and_freeze(workers, monkeypatch)
    monkeypatch.setattr(process_identity, "_boot", lambda: NEW_BOOT)
    # A boot id read on another host proves nothing about the host that
    # recorded the group: its processes may still be running there.
    monkeypatch.setattr(process_identity, "_host", lambda: "host-b")
    with pytest.raises(ConfirmationRequiredError):
        await _resume(workers, task_id, "resume-1")
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"


async def test_unreadable_boot_requires_confirmation(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(
        workers,
        workspace,
        capture=True,
        processes=((4321, 4321, OLD_BOOT, "1:1"),),
    )
    await _open_and_freeze(workers, monkeypatch)

    def unreadable() -> str:
        raise OSError("cannot read boot identity")

    monkeypatch.setattr(process_identity, "_boot", unreadable)
    with pytest.raises(ConfirmationRequiredError):
        await _resume(workers, task_id, "resume-1")
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"


async def test_proven_never_started_parked_first_starts_without_card_or_token(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    tasks = workers.tasks
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "do work"}),
        origin="user",
        workspace=str(workspace),
        _recovery_proof=True,
    )
    await tasks.change(task.task_id, state="parked")
    await _open_and_freeze(workers, monkeypatch)
    resumed = await _resume(workers, task.task_id, "resume-1")
    assert resumed["state"] == "queued"
    assert (await RecoveryStore(store).provenance(task.task_id)).first_start == (
        "never_started"
    )
    assert not await _events(store, task.task_id, "sched.resume_confirmed")


async def test_real_create_records_the_never_started_proof(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production create path, restarted, takes the first-start row."""
    workers, _, store = worker_setup
    _freeze(workers, monkeypatch)
    await workers.open()
    created, repeated = await workers.create({"objective": "do work", "level": "p5"})
    assert not repeated
    assert created["state"] == "parked"
    task_id = created["task_id"]
    assert (await RecoveryStore(store).provenance(task_id)).first_start == (
        "never_started"
    )
    gate = ChatGate(policy="off")
    restarted = TaskWorkers(store, gate, workers.settings, SessionHub(store))
    try:
        await restarted.open()
        _freeze(restarted, monkeypatch)
        resumed = await _resume(restarted, task_id, "resume-1")
        assert resumed["state"] == "queued"
        assert not await _events(store, task_id, "sched.resume_confirmed")
    finally:
        await restarted.aclose()
        await gate.close()


async def test_queued_never_started_create_is_not_resumed_by_the_route(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
) -> None:
    workers, workspace, _ = worker_setup
    tasks = workers.tasks
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "do work"}),
        origin="user",
        workspace=str(workspace),
        _recovery_proof=True,
    )
    with pytest.raises(ResumeRefusedError) as refused:
        await _resume(workers, task.task_id, "resume-1")
    assert refused.value.reason == "invalid_state"


async def test_missing_card_blocks_resume_but_not_stop(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    tasks = workers.tasks
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "do work"}),
        origin="user",
        workspace=str(workspace),
    )
    await _open_and_freeze(workers, monkeypatch)
    with pytest.raises(ResumeRefusedError) as refused:
        await _resume(workers, task.task_id, "resume-1")
    assert refused.value.reason == "missing_card"
    assert "Start again as a new task" in str(refused.value)
    assert (await workers.update(task.task_id, state="stop"))["state"] == "cancelled"


async def test_terminal_and_invalid_workspace_block_without_confirmation(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    workspace.rmdir()
    with pytest.raises(ResumeRefusedError) as refused:
        await _resume(workers, task_id, "resume-1")
    assert refused.value.reason == "invalid_workspace"
    with pytest.raises(ResumeRefusedError):
        await _resume(workers, task_id, "resume-1")
    await workers.update(task_id, state="stop")
    with pytest.raises(ResumeRefusedError) as terminal:
        await _resume(workers, task_id, "resume-2")
    assert terminal.value.reason == "terminal"


async def test_stale_execution_identity_blocks_resume(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace, capture=True)
    await _open_and_freeze(workers, monkeypatch)
    async with workers.store.component_transaction("seed_stale") as conn:
        await conn.execute(
            "UPDATE recovery_provenance SET execution_id = 'exec-other' "
            "WHERE task_id = ?",
            (task_id,),
        )
    with pytest.raises(ResumeRefusedError) as refused:
        await _resume(workers, task_id, "resume-1")
    assert refused.value.reason == "stale_identity"
    assert (await workers.tasks.get_task(task_id)).state == "interrupted"


async def test_paused_priority_survives_resume_and_snapshot_is_cleared(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace, capture=True)
    await workers.tasks.change(task_id, priority=3, actor="user")
    generation, previous = await workers.tasks.park_identity(task_id)
    await workers.tasks.request_pending_pause(
        task_id,
        expected_generation=generation,
        expected_operation_id=previous,
        operation_id="pause",
    )
    await _open_and_freeze(workers, monkeypatch)
    frozen = await workers.tasks.get_task(task_id)
    assert frozen.level == 5 and frozen.reason == "pause"
    resumed = await _resume(workers, task_id, "resume-1")
    assert resumed["state"] == "queued" and resumed["level"] == 3
    assert resumed["level_source"] == "user"
    async with workers.store.component_transaction("read_snapshot") as conn:
        row = await (
            await conn.execute(
                "SELECT snapshot_active FROM park_priority WHERE task_id = ?",
                (task_id,),
            )
        ).fetchone()
    assert row is not None and row[0] == 0


async def test_unresolved_operation_blocks_only_resume(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    recovery = RecoveryStore(workers.store)
    generation, previous = await workers.tasks.park_identity(task_id)
    await recovery.prepare(
        task_id,
        operation_id="unsettled",
        execution_id="exec-1",
        expected_generation=generation,
        expected_operation_id=previous,
        outcome="interrupted",
        card="Recovery card",
        reason="unknown",
        crash=True,
        unsafe_resume=True,
    )
    with pytest.raises(RecoveryUnknownError):
        await _resume(workers, task_id, "resume-1")
    assert (await workers.update(task_id, state="stop"))["state"] == "cancelled"


async def test_unresolved_cleanup_blocks_resume_but_not_stop(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    workers._register(await workers.tasks.get_task(task_id))
    workers._quarantined.add(task_id)
    with pytest.raises(RecoveryUnknownError) as blocked:
        await _resume(workers, task_id, "resume-1")
    assert blocked.value.reason == "cleanup_unresolved"
    assert (await workers.update(task_id, state="stop"))["state"] == "cancelled"


async def test_matching_retry_is_idempotent(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    token = await _preview(workers, task_id, "resume-1")
    accepted = await _resume(workers, task_id, "resume-1", token)
    again = await _resume(workers, task_id, "resume-1")
    assert accepted["state"] == again["state"] == "queued"
    assert len(await _events(store, task_id, "sched.resume_confirmed")) == 1
    assert len(await _events(store, task_id, "sched.resumed")) == 1


async def _preview(workers: TaskWorkers, task_id: str, operation_id: str) -> str:
    with pytest.raises(ConfirmationRequiredError) as preview:
        await _resume(workers, task_id, operation_id)
    return preview.value.token


async def test_delayed_retry_after_stop_does_not_resume(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    token = await _preview(workers, task_id, "resume-1")
    await _resume(workers, task_id, "resume-1", token)
    await workers.update(task_id, state="stop")
    retried = await _resume(workers, task_id, "resume-1")
    assert retried["state"] == "cancelled"
    assert (await workers.tasks.get_task(task_id)).state == "cancelled"
    assert not await _events(store, task_id, "turn.started")


async def test_stop_invalidates_a_pending_token(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    token = await _preview(workers, task_id, "resume-1")
    assert workers._resume_tokens.pending(task_id) == 1
    await workers.update(task_id, state="stop")
    assert workers._resume_tokens.pending(task_id) == 0
    with pytest.raises(StaleConfirmationError):
        await _resume(workers, task_id, "resume-2", token)


async def test_race_stop_and_resume_ends_cancelled_without_a_turn(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    token = await _preview(workers, task_id, "resume-1")
    results = await asyncio.gather(
        _resume(workers, task_id, "resume-1", token),
        workers.update(task_id, state="stop"),
        return_exceptions=True,
    )
    assert (await workers.tasks.get_task(task_id)).state == "cancelled"
    assert not await _events(store, task_id, "turn.started")
    assert any(
        isinstance(result, dict) and result.get("state") == "cancelled"
        for result in results
    )


class _DoneLLM(BaseLLMProvider):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


async def test_stop_before_worker_registration_cancels_the_resumed_turn(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await workers.open()
    built, release = asyncio.Event(), asyncio.Event()

    async def build(**kwargs: Any) -> Agent:
        built.set()
        await release.wait()
        agent = Agent(
            llm=kwargs["wrap_llm"](_DoneLLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    token = await _preview(workers, task_id, "resume-1")
    resumed = asyncio.create_task(_resume(workers, task_id, "resume-1", token))
    await asyncio.wait_for(built.wait(), 3)
    stopped = asyncio.create_task(workers.update(task_id, state="stop"))
    await asyncio.wait_for(stopped, 3)
    release.set()
    await asyncio.wait_for(resumed, 3)
    for _ in range(300):
        if (await workers.tasks.get_task(task_id)).state == "cancelled":
            break
        await asyncio.sleep(0.01)
    assert (await workers.tasks.get_task(task_id)).state == "cancelled"
    assert not await _events(store, task_id, "turn.started")


async def test_stop_started_before_resume_commits_still_wins(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await workers.open()
    built, release = asyncio.Event(), asyncio.Event()

    async def build(**kwargs: Any) -> Agent:
        built.set()
        await release.wait()
        agent = Agent(
            llm=kwargs["wrap_llm"](_DoneLLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    token = await _preview(workers, task_id, "resume-1")
    original = workers.tasks.resume_recovered
    ready, go = asyncio.Event(), asyncio.Event()

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        ready.set()
        await go.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(workers.tasks, "resume_recovered", delayed)
    resumed = asyncio.create_task(_resume(workers, task_id, "resume-1", token))
    await asyncio.wait_for(ready.wait(), 3)
    # Stop starts while Resume holds the lock, before the worker task exists.
    stopped = asyncio.create_task(workers.update(task_id, state="stop"))
    await asyncio.sleep(0)
    go.set()
    await asyncio.wait_for(asyncio.gather(resumed, stopped), 3)
    await asyncio.wait_for(built.wait(), 3)
    release.set()
    for _ in range(300):
        if (await workers.tasks.get_task(task_id)).state == "cancelled":
            break
        await asyncio.sleep(0.01)
    assert (await workers.tasks.get_task(task_id)).state == "cancelled"
    assert not await _events(store, task_id, "turn.started")


async def test_double_confirmation_commits_once(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    token = await _preview(workers, task_id, "resume-1")
    results = await asyncio.gather(
        _resume(workers, task_id, "resume-1", token),
        _resume(workers, task_id, "resume-2", token),
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in results) == 1
    assert len(await _events(store, task_id, "sched.resume_confirmed")) == 1
    assert (await workers.tasks.get_task(task_id)).state == "queued"


async def test_stale_generation_or_evidence_refuses_the_token(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace, processes=((4321, 4321, "b", "t"),))
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: False
    with pytest.raises(ConfirmationRequiredError) as preview:
        await _resume(workers, task_id, "resume-1")
    token = preview.value.token
    # A changed evidence identity (a newly recorded group) bumps the CAS and
    # must invalidate the token rather than silently re-authorize it.
    await RecoveryStore(workers.store).record_process(
        task_id,
        execution_id="exec-1",
        pgid=8765,
        leader_pid=8765,
        boot_id="b",
        start_token="t2",
    )
    with pytest.raises(StaleConfirmationError):
        await _resume(workers, task_id, "resume-1", token)


async def test_restart_drops_tokens(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    token = await _preview(workers, task_id, "resume-1")
    gate = ChatGate(policy="off")
    restarted = TaskWorkers(store, gate, Config, SessionHub(store))
    try:
        await restarted.open()
        _freeze(restarted, monkeypatch)
        with pytest.raises(StaleConfirmationError):
            await _resume(restarted, task_id, "resume-1", token)
        assert (await restarted.tasks.get_task(task_id)).state == "interrupted"
    finally:
        await restarted.aclose()
        await gate.close()


@pytest.mark.parametrize("after_commit", [False, True])
async def test_commit_boundary_is_reconciled(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
    after_commit: bool,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    token = await _preview(workers, task_id, "resume-1")
    original_before = store._before_commit
    original_commit = store._commit
    fired = False

    def before(label: str) -> None:
        nonlocal fired
        if label == "task_resume_recovered" and not after_commit and not fired:
            fired = True
            raise RuntimeError("injected before commit")
        original_before(label)

    async def commit(conn: Any, events: Any, label: str) -> None:
        nonlocal fired
        await original_commit(conn, events, label)
        if label == "task_resume_recovered" and after_commit and not fired:
            fired = True
            raise asyncio.CancelledError()

    monkeypatch.setattr(store, "_before_commit", before)
    monkeypatch.setattr(store, "_commit", commit)
    with pytest.raises(asyncio.CancelledError if after_commit else RuntimeError):
        await _resume(workers, task_id, "resume-1", token)
    assert fired
    if after_commit:
        # The receipt exists: the same operation returns success, no second
        # turn, and the committed queue is admitted in this process.
        retried = await _resume(workers, task_id, "resume-1")
        assert retried["state"] == "queued"
        assert task_id in workers._eligible
        assert workers._ready[task_id].is_set()
        assert len(await _events(store, task_id, "sched.resume_confirmed")) == 1
    else:
        assert (await workers.tasks.get_task(task_id)).state == "interrupted"
        accepted = await _resume(workers, task_id, "resume-1", token)
        assert accepted["state"] == "queued"


def test_token_expires_and_binding_must_match_exactly() -> None:
    now = [100.0]
    tokens = ResumeTokens(clock=lambda: now[0], ttl_s=300)
    binding = ResumeBinding("task", "exec", "key", 3, None)
    token = tokens.issue(binding)
    assert tokens.valid(token, binding)
    assert not tokens.valid(token, ResumeBinding("task", "exec", "other", 3, None))
    now[0] += 301
    assert not tokens.valid(token, binding)
    assert not tokens.consume(token, binding)


async def test_patch_stays_fenced_before_resume(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    with pytest.raises(TaskStoreError, match="safety review"):
        await workers.update(task_id, level="p1")
    with pytest.raises(TaskStoreError, match="safety review"):
        await workers.update(task_id, state="resume")


async def test_committed_resume_is_readmitted_after_a_restart(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace, capture=True)
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: True
    assert (await _resume(workers, task_id, "resume-1"))["state"] == "queued"
    gate = ChatGate(policy="off")
    restarted = TaskWorkers(store, gate, workers.settings, SessionHub(store))
    try:
        await restarted.open()
        _freeze(restarted, monkeypatch)
        # Re-admitted on its own: the committed queue survives the restart
        # without waiting for the client to retry.
        assert task_id in restarted._eligible
        assert (await restarted.tasks.get_task(task_id)).state == "queued"
        retried = await _resume(restarted, task_id, "resume-1")
        assert retried["state"] == "queued"
        assert task_id in restarted._eligible
        assert task_id not in restarted._restart_fenced
    finally:
        await restarted.aclose()
        await gate.close()


async def test_committed_resume_with_a_started_turn_conflicts_after_restart(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace, capture=True)
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: True
    assert (await _resume(workers, task_id, "resume-1"))["state"] == "queued"
    # The resumed work began before the restart: its turn exists.
    await store.begin_turn(task_id, "do work", holder="gateway")
    gate = ChatGate(policy="off")
    restarted = TaskWorkers(store, gate, workers.settings, SessionHub(store))
    try:
        await restarted.open()
        _freeze(restarted, monkeypatch)
        with pytest.raises(TaskConflictError):
            await _resume(restarted, task_id, "resume-1")
        assert task_id not in restarted._eligible
        assert (await restarted.tasks.get_task(task_id)).state == "interrupted"
    finally:
        await restarted.aclose()
        await gate.close()


async def test_stop_after_registration_admits_no_turn(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace)
    await workers.open()

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](_DoneLLM()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    token = await _preview(workers, task_id, "resume-1")
    original = workers.tasks.change
    ready, go = asyncio.Event(), asyncio.Event()

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("state") == "running":
            ready.set()
            await go.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(workers.tasks, "change", delayed)
    resumed = asyncio.create_task(_resume(workers, task_id, "resume-1", token))
    await asyncio.wait_for(ready.wait(), 3)
    # Stop is requested after the runner was registered but while the worker
    # waits inside its state change, holding the lock.
    stopped = asyncio.create_task(workers.update(task_id, state="stop"))
    await asyncio.sleep(0)
    go.set()
    await asyncio.wait_for(stopped, 3)
    await asyncio.wait_for(resumed, 3)
    for _ in range(300):
        if (await workers.tasks.get_task(task_id)).state == "cancelled":
            break
        await asyncio.sleep(0.01)
    assert (await workers.tasks.get_task(task_id)).state == "cancelled"
    assert not await _events(store, task_id, "turn.started")


async def test_evidence_is_read_outside_the_worker_lock(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, _ = worker_setup
    task_id = await _started(workers, workspace)
    await _open_and_freeze(workers, monkeypatch)
    held: list[bool] = []

    async def spy(*args: Any, **kwargs: Any) -> Any:
        held.append(workers._lock.locked())
        return await collect_resume_evidence(*args, **kwargs)

    monkeypatch.setattr(gateway_workers, "collect_evidence", spy)
    with pytest.raises(ConfirmationRequiredError):
        await _resume(workers, task_id, "resume-1")
    assert held == [False]


async def test_retry_with_another_payload_returns_the_committed_operation(
    worker_setup: tuple[TaskWorkers, Path, SessionStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers, workspace, store = worker_setup
    task_id = await _started(workers, workspace, capture=True)
    await _open_and_freeze(workers, monkeypatch)
    workers._process_verifier = lambda record, host: True
    first = await _resume(workers, task_id, "resume-1")
    # An operation ID names one operation: the retry never re-authorises a
    # transition, whatever payload it carries.
    again = await _resume(workers, task_id, "resume-1", "tok_whatever")
    assert again["task_id"] == first["task_id"]
    assert again["state"] == "queued"
    assert len(await _events(store, task_id, "sched.resumed")) == 1
