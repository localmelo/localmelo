"""The scheduler's task store: component v1, atomic tasks, levels and retention."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import (
    COMPONENT,
    SCHEDULER_VERSION,
    TaskConflictError,
    TaskNotPausedError,
    TaskRecord,
    TaskStore,
    TaskStoreError,
)
from localmelo.support.sessions.store import SessionStore, SessionStoreError


class Clock:
    """A wall clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def spec(objective: str = "Write the report", **values: Any) -> TaskSpec:
    source = values.pop("level_source", None)
    payload = {"objective": objective, **values}
    if source == "user":
        return TaskSpec.from_user_payload(payload)
    result = TaskSpec.from_payload(payload)
    return replace(result, level_source=source) if source is not None else result


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def store(tmp_path: Path, clock: Clock) -> AsyncIterator[SessionStore]:
    opened = SessionStore(tmp_path / "sessions.db", writer="gateway", clock=clock)
    await opened.open()
    try:
        yield opened
    finally:
        await opened.aclose()


@pytest.fixture
async def tasks(store: SessionStore, clock: Clock) -> AsyncIterator[TaskStore]:
    opened = TaskStore(store, clock=clock)
    await opened.open()
    yield opened


async def create_task(
    tasks: TaskStore,
    task_spec: TaskSpec,
    *,
    workspace: str | None = None,
    **kwargs: Any,
) -> tuple[TaskRecord, bool]:
    validated = workspace or str(Path(tasks.store.path).parent.resolve())
    return await tasks.create_task(task_spec, workspace=validated, **kwargs)


async def rows(store: SessionStore, sql: str, params: Sequence[Any] = ()) -> list[Any]:
    uri = Path(store.path).as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return list(conn.execute(sql, tuple(params)).fetchall())


async def one(store: SessionStore, sql: str, params: Sequence[Any] = ()) -> Any:
    found = await rows(store, sql, params)
    return found[0] if found else None


async def get_task(tasks: TaskStore, task_id: str) -> TaskRecord | None:
    row = await one(tasks.store, "SELECT * FROM tasks WHERE task_id = ?", (task_id,))
    return TaskRecord.from_row(row) if row is not None else None


async def component_version(store: SessionStore, name: str) -> int | None:
    row = await one(
        store, "SELECT version FROM schema_components WHERE name = ?", (name,)
    )
    return row[0] if row else None


async def priority_events(tasks: TaskStore, task_id: str) -> list[Any]:
    from types import SimpleNamespace

    found = await rows(
        tasks.store,
        "SELECT from_level, to_level, reason FROM priority_events "
        "WHERE task_id = ? ORDER BY id",
        (task_id,),
    )
    return [SimpleNamespace(from_level=x[0], to_level=x[1], reason=x[2]) for x in found]


async def recent_log(tasks: TaskStore, task_id: str) -> list[Any]:
    return await rows(
        tasks.store,
        "SELECT ts, kind, task_id, data FROM sched_log WHERE task_id = ? ORDER BY id",
        (task_id,),
    )


async def test_open_creates_seven_tables_and_leaves_user_version_alone(
    store: SessionStore, clock: Clock
) -> None:
    assert await component_version(store, COMPONENT) is None
    assert await one(store, "PRAGMA user_version") == (1,)

    tasks = TaskStore(store, clock=clock)
    assert await tasks.open() == SCHEDULER_VERSION
    assert tasks.version == SCHEDULER_VERSION
    assert await component_version(store, COMPONENT) == SCHEDULER_VERSION
    # A second open is a no-op, not a second migration.
    assert await tasks.open() == SCHEDULER_VERSION
    assert await one(store, "PRAGMA user_version") == (1,)

    objects = {
        row[0]
        for row in await rows(
            store, "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {
        "tasks",
        "task_deps",
        "priority_events",
        "questions",
        "sched_rules",
        "sched_log",
        "idle_windows",
    } <= objects
    indexes = {
        row[0]
        for row in await rows(
            store, "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    assert {
        "tasks_state_level",
        "priority_events_task",
        "sched_log_ts",
        "idle_windows_start",
    } <= indexes
    shapes = {
        "tasks": {
            "task_id",
            "front_id",
            "root_id",
            "origin",
            "request_key",
            "title",
            "spec",
            "state",
            "wait_reason",
            "hold_site",
            "level",
            "level_source",
            "level_pinned",
            "level_eff",
            "peak_level",
            "reason",
            "user_rank",
            "features",
            "deadline",
            "budget_s",
            "service_s",
            "wait_s",
            "tool_s",
            "counters",
            "crashes",
            "pgids",
            "card",
            "created_at",
            "started_at",
            "finished_at",
            "updated_at",
            "expires_at",
        },
        "task_deps": {"task_id", "waits_on", "created_by", "confirmed"},
        "priority_events": {
            "id",
            "task_id",
            "ts",
            "actor",
            "from_level",
            "to_level",
            "pinned",
            "reason",
            "quote",
        },
        "questions": {
            "qid",
            "task_id",
            "text",
            "options",
            "default_answer",
            "blocking",
            "created_at",
            "surfaced_at",
            "answered_at",
            "answer",
            "answered_by",
        },
        "sched_rules": {"id", "text", "match", "level", "enabled", "created_at"},
        "sched_log": {"id", "ts", "kind", "task_id", "data"},
        "idle_windows": {
            "start",
            "end",
            "power",
            "battery_pct",
            "thermal",
            "lock_state",
            "user_present",
        },
    }
    for table, expected in shapes.items():
        columns = {row[1] for row in await rows(store, f"PRAGMA table_info({table})")}
        assert columns == expected, table


async def test_a_newer_scheduler_component_is_refused(
    store: SessionStore, clock: Clock
) -> None:
    async with store.component_transaction("newer") as conn:
        await conn.execute(
            "INSERT INTO schema_components (name, version) VALUES (?, ?)",
            (COMPONENT, 2),
        )
    with pytest.raises(SessionStoreError, match="scheduler schema version 2"):
        await TaskStore(store, clock=clock).open()
    assert await one(store, "SELECT name FROM sqlite_master WHERE name='tasks'") is None


async def test_a_failed_migration_writes_no_table_and_no_version(
    store: SessionStore,
) -> None:
    with pytest.raises(sqlite3.OperationalError):
        await store.ensure_component(
            "broken",
            1,
            ("CREATE TABLE broken_ok (id INTEGER)", "THIS IS NOT SQL"),
        )
    assert (
        await one(store, "SELECT name FROM sqlite_master WHERE name='broken_ok'")
        is None
    )
    assert await component_version(store, "broken") is None


async def test_an_older_component_is_refused_without_migrating(
    store: SessionStore, clock: Clock
) -> None:
    async with store.component_transaction("older") as conn:
        await conn.execute(
            "INSERT INTO schema_components (name, version) VALUES (?, ?)",
            (COMPONENT, 0),
        )
    with pytest.raises(SessionStoreError, match="cannot migrate"):
        await TaskStore(store, clock=clock).open()
    assert await one(store, "SELECT name FROM sqlite_master WHERE name='tasks'") is None


async def test_a_closed_and_reopened_file_keeps_the_component_and_tasks(
    tmp_path: Path, clock: Clock
) -> None:
    path = tmp_path / "reopen.db"
    first = SessionStore(path, writer="gateway", clock=clock)
    await first.open()
    tasks = TaskStore(first, clock=clock)
    await tasks.open()
    record, _ = await create_task(
        tasks, spec("stays"), origin="user", request_key="stays"
    )
    await first.aclose()

    second = SessionStore(path, writer="gateway", clock=clock)
    await second.open()
    try:
        reopened = TaskStore(second, clock=clock)
        assert await reopened.open() == SCHEDULER_VERSION
        assert (await get_task(reopened, record.task_id)) is not None
        again, repeated = await create_task(
            reopened, spec("stays"), origin="user", request_key="stays"
        )
        assert repeated is True and again.task_id == record.task_id
    finally:
        await second.aclose()


async def test_the_cli_file_never_gets_the_component(tmp_path: Path) -> None:
    cli = SessionStore(tmp_path / "cli.db", writer="cli")
    await cli.open()
    try:
        with pytest.raises(ValueError, match="gateway"):
            TaskStore(cli)
        with pytest.raises(SessionStoreError, match="gateway"):
            await cli.ensure_component(COMPONENT, SCHEDULER_VERSION, ("SELECT 1",))
        assert await component_version(cli, COMPONENT) is None
    finally:
        await cli.aclose()


async def test_create_writes_the_task_session_and_its_first_rows(
    store: SessionStore, tasks: TaskStore
) -> None:
    record, repeated = await create_task(
        tasks,
        spec(
            "Make six slides",
            workspace="/tmp/talk",
            level=1,
            level_source="user",
            reason="P1 · due today",
        ),
        origin="user",
        request_key="req-1",
        front_id="front-1",
    )
    assert repeated is False
    assert record.task_id == record.root_id
    assert record.origin == "user"
    assert record.request_key == "req-1"
    assert record.state == "queued"
    assert record.level == 1
    assert record.level_source == "user"
    assert record.spec["objective"] == "Make six slides"
    assert record.spec["task_id"] == record.task_id
    session = await one(
        store,
        "SELECT kind, title, workspace FROM sessions WHERE id = ?",
        (record.task_id,),
    )
    assert session == (
        "task",
        "Make six slides",
        str(Path(store.path).parent.resolve()),
    )
    event = await one(
        store,
        "SELECT actor, from_level, to_level, reason, quote FROM priority_events "
        "WHERE task_id = ?",
        (record.task_id,),
    )
    assert event == ("user", None, 1, "P1 · due today", "")
    logged = await recent_log(tasks, record.task_id)
    assert [row[1] for row in logged] == ["task.created"]


async def test_create_persists_validated_workspace_separately_from_request(
    store: SessionStore,
    tasks: TaskStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from localmelo.support.gateway.task_policy import validate_task_workspace

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    root = home / "workspace"
    root.mkdir()
    cfg = SimpleNamespace(gateway=SimpleNamespace(workspace=str(root)))
    validated = validate_task_workspace(None, cfg, origin="agent", task_id="worker1")
    requested = spec("agent")
    agent, _ = await create_task(
        tasks, requested, origin="agent", task_id="worker1", workspace=validated
    )
    assert agent.spec["workspace"] == ""
    assert await one(
        store, "SELECT workspace FROM sessions WHERE id = ?", (agent.task_id,)
    ) == (validated,)

    explicit = home / "project"
    explicit.mkdir()
    validated_user = validate_task_workspace("~/project", cfg, origin="user")
    user, _ = await create_task(
        tasks,
        spec("user", workspace="~/project"),
        origin="user",
        workspace=validated_user,
    )
    assert user.spec["workspace"] == "~/project"
    assert await one(
        store, "SELECT workspace FROM sessions WHERE id = ?", (user.task_id,)
    ) == (validated_user,)


async def test_retry_keeps_the_original_validated_workspace(
    store: SessionStore, tasks: TaskStore, tmp_path: Path
) -> None:
    first_workspace = str(tmp_path.resolve())
    first, _ = await create_task(
        tasks,
        spec("same"),
        origin="user",
        request_key="repeat",
        workspace=first_workspace,
    )
    second_workspace = str((tmp_path / "new-root").resolve())
    repeated, is_repeated = await create_task(
        tasks,
        spec("same"),
        origin="user",
        request_key="repeat",
        workspace=second_workspace,
    )
    assert is_repeated and repeated.task_id == first.task_id
    assert await one(
        store, "SELECT workspace FROM sessions WHERE id = ?", (first.task_id,)
    ) == (first_workspace,)
    with pytest.raises(ValueError, match="validated absolute"):
        await tasks.create_task(spec(), origin="user", workspace="relative")


async def test_create_rolls_back_the_session_task_and_events_together(
    store: SessionStore, tasks: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(label: str) -> None:
        if label == "task_create":
            raise RuntimeError("stop before commit")

    monkeypatch.setattr(store, "_before_commit", fail)
    with pytest.raises(RuntimeError, match="stop before commit"):
        await create_task(tasks, spec(), origin="user", request_key="rollback")
    assert await one(store, "SELECT COUNT(*) FROM sessions") == (0,)
    assert await one(store, "SELECT COUNT(*) FROM tasks") == (0,)
    assert await one(store, "SELECT COUNT(*) FROM priority_events") == (0,)
    assert await one(store, "SELECT COUNT(*) FROM sched_log") == (0,)


async def test_a_request_key_is_idempotent_and_a_reuse_conflicts(
    tasks: TaskStore,
) -> None:
    first, repeated = await create_task(
        tasks, spec("one"), origin="user", request_key="same"
    )
    assert repeated is False
    again, repeated = await create_task(
        tasks, spec("one"), origin="user", request_key="same"
    )
    assert repeated is True
    assert again.task_id == first.task_id
    with pytest.raises(TaskConflictError, match="different task"):
        await create_task(tasks, spec("two"), origin="user", request_key="same")
    other, _ = await create_task(tasks, spec("one"), origin="user")
    assert other.task_id != first.task_id


async def test_find_existing_request_before_workspace_revalidation(
    tasks: TaskStore, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = spec("same", workspace=str(workspace))
    first, _ = await create_task(
        tasks,
        request,
        origin="user",
        request_key="key",
        front_id="front",
        workspace=str(workspace),
        deadline=123.0,
    )
    workspace.rmdir()
    assert (
        await tasks.find_existing_request(
            request, origin="user", request_key="key", front_id="front", deadline=123.0
        )
        == first
    )
    assert await tasks.find_existing_request(request, origin="user") is None
    assert (
        await tasks.find_existing_request(request, origin="agent", request_key="key")
        is None
    )
    for front_id, deadline in (("other", 123.0), ("front", 124.0)):
        with pytest.raises(TaskConflictError, match="different task"):
            await tasks.find_existing_request(
                request,
                origin="user",
                request_key="key",
                front_id=front_id,
                deadline=deadline,
            )
    with pytest.raises(TaskConflictError, match="different task"):
        await tasks.find_existing_request(
            spec("different"), origin="user", request_key="key"
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin": "invalid"},
        {"origin": "user", "request_key": ""},
        {"origin": "user", "request_key": 1},
        {"origin": "user", "front_id": 1},
        {"origin": "user", "task_id": 1},
        {"origin": "user", "meta": {"x": object()}},
    ],
)
async def test_find_existing_request_validates_create_arguments(
    tasks: TaskStore, kwargs: dict[str, Any]
) -> None:
    with pytest.raises((ValueError, TypeError)):
        await tasks.find_existing_request(spec(), **kwargs)


async def test_lookup_miss_does_not_replace_atomic_create_check(
    tasks: TaskStore,
) -> None:
    request = spec("concurrent")
    assert (
        await tasks.find_existing_request(request, origin="user", request_key="key")
        is None
    )
    results = await asyncio.gather(
        *(
            create_task(tasks, request, origin="user", request_key="key")
            for _ in range(2)
        )
    )
    assert results[0][0].task_id == results[1][0].task_id
    assert sorted(repeated for _, repeated in results) == [False, True]
    assert (
        await tasks.find_existing_request(request, origin="user", request_key="key")
        == results[0][0]
    )


async def test_change_noop_priority_does_not_add_event(tasks: TaskStore) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    await tasks.change(record.task_id, priority="P2")
    await tasks.change(record.task_id, priority="auto")
    assert len(await priority_events(tasks, record.task_id)) == 1
    await tasks.change(record.task_id, priority="P1")
    assert len(await priority_events(tasks, record.task_id)) == 2


async def test_change_persists_failure_reason(tasks: TaskStore) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    with pytest.raises(ValueError, match="failure_reason"):
        await tasks.change(
            record.task_id, state="running", failure_reason="invalid_workspace"
        )
    failed = await tasks.change(
        record.task_id,
        state="failed",
        actor="scheduler",
        failure_reason="invalid_workspace",
    )
    assert failed.state == "failed"
    assert failed.wait_reason == "invalid_workspace"
    assert (await tasks.get_task(record.task_id)).wait_reason == "invalid_workspace"


async def test_pause_resume_restores_user_priority_and_honors_later_changes(
    tasks: TaskStore,
) -> None:
    record, _ = await create_task(
        tasks, spec("task", level=1, level_source="user"), origin="user"
    )
    paused = await tasks.pause(record.task_id)
    assert (paused.level, paused.state) == (5, "parked")
    assert (await priority_events(tasks, record.task_id))[-1].reason == "pause"
    assert await tasks.pause(record.task_id) == paused
    resumed = await tasks.resume(record.task_id)
    assert (resumed.level, resumed.level_source, resumed.state) == (1, "user", "queued")
    await tasks.pause(record.task_id)
    await tasks.change(record.task_id, priority="P3")
    with pytest.raises(TaskStoreError, match="not paused"):
        await tasks.resume(record.task_id)
    assert (await tasks.get_task(record.task_id)).level == 3


async def test_reapplying_p5_preserves_pause_and_restores_priority(
    tasks: TaskStore,
) -> None:
    record, _ = await create_task(
        tasks,
        spec("task", level=1, level_source="user", reason="urgent"),
        origin="user",
    )
    await tasks.pause(record.task_id)
    changed = await tasks.change(record.task_id, priority="P5", state="parked")
    assert (changed.level, changed.reason) == (5, "pause")
    resumed = await tasks.resume(record.task_id)
    assert (resumed.state, resumed.level, resumed.level_source, resumed.reason) == (
        "queued",
        1,
        "user",
        "urgent",
    )
    assert [
        (event.from_level, event.to_level)
        for event in await priority_events(tasks, record.task_id)
    ] == [(None, 1), (1, 5), (5, 1)]


@pytest.mark.parametrize("created_at_p5", [True, False])
async def test_resume_parked_p5_without_pause_uses_computed_default(
    tasks: TaskStore, created_at_p5: bool
) -> None:
    record, _ = await create_task(
        tasks,
        spec("task", level=5 if created_at_p5 else 1, level_source="user"),
        origin="user",
    )
    if not created_at_p5:
        # An obsolete pause must not override a subsequent explicit P5.
        await tasks.pause(record.task_id)
        await tasks.resume(record.task_id)
        await tasks.change(record.task_id, priority="P5")
    parked = await tasks.change(record.task_id, state="parked")
    events_before = await priority_events(tasks, record.task_id)
    log_before = await recent_log(tasks, record.task_id)
    assert await tasks.pause(record.task_id) == parked
    assert await priority_events(tasks, record.task_id) == events_before
    assert await recent_log(tasks, record.task_id) == log_before
    resumed = await tasks.resume(record.task_id)
    assert (
        resumed.state,
        resumed.level,
        resumed.level_source,
        resumed.level_pinned,
    ) == ("queued", 2, "computed", False)
    assert (await priority_events(tasks, record.task_id))[-1].reason == "resume"


async def test_resume_after_pause_log_retention_uses_computed_default(
    tasks: TaskStore, clock: Clock
) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    await tasks.pause(record.task_id)
    clock.advance(31 * 86_400)
    await tasks.apply_retention()
    resumed = await tasks.resume(record.task_id)
    assert (resumed.state, resumed.level, resumed.level_source) == (
        "queued",
        2,
        "computed",
    )


async def test_resume_without_pause_and_restore_p0(tasks: TaskStore) -> None:
    first, _ = await create_task(
        tasks, spec("first", level=0, level_source="user"), origin="user"
    )
    await tasks.pause(first.task_id)
    second, _ = await create_task(
        tasks, spec("second", level=0, level_source="user"), origin="user"
    )
    assert (await tasks.resume(first.task_id)).level == 0
    assert (await tasks.get_task(second.task_id)).level == 1
    no_pause, _ = await create_task(
        tasks, spec("no pause", level=4, level_source="user"), origin="user"
    )
    with pytest.raises(TaskNotPausedError, match="not paused"):
        await tasks.resume(no_pause.task_id)


async def test_restart_fails_only_nonterminal_tasks(
    tasks: TaskStore, clock: Clock
) -> None:
    queued, _ = await create_task(tasks, spec("queued"), origin="user")
    running, _ = await create_task(tasks, spec("running"), origin="user")
    parked, _ = await create_task(tasks, spec("parked"), origin="user")
    done, _ = await create_task(tasks, spec("done"), origin="user")
    await tasks.change(running.task_id, state="running")
    await tasks.pause(parked.task_id)
    terminal = await tasks.change(done.task_id, state="done")
    clock.advance(1)
    assert await tasks.fail_nonterminal_on_restart() == 3
    for record in (queued, running, parked):
        failed = await tasks.get_task(record.task_id)
        assert failed.state == "failed" and failed.finished_at == clock.now
        assert failed.wait_reason == "gateway restarted; recovery arrives in S3b"
    assert await tasks.get_task(done.task_id) == terminal
    assert await tasks.fail_nonterminal_on_restart() == 0


async def test_deferred_input_is_rejected_before_request_key_lookup(
    tasks: TaskStore,
) -> None:
    from localmelo.support.gateway.task_policy import TaskSpecError

    first, _ = await create_task(
        tasks,
        TaskSpec.from_user_payload({"objective": "same"}),
        origin="user",
        request_key="request-1",
    )
    with pytest.raises(TaskSpecError, match="inputs"):
        request = TaskSpec.from_user_payload({"objective": "same", "inputs": ["file"]})
        await create_task(tasks, request, origin="user", request_key="request-1")
    assert await get_task(tasks, first.task_id) == first


async def test_trusted_create_pins_a_user_level(tasks: TaskStore) -> None:
    for level in (0, 1):
        request = TaskSpec.from_user_payload(
            {"objective": f"level {level}", "level": level}
        )
        record, _ = await create_task(tasks, request, origin="user")
        assert (record.level, record.level_source, record.level_pinned) == (
            level,
            "user",
            True,
        )
        with pytest.raises(TaskStoreError, match="set by the user"):
            await tasks.set_level(record.task_id, 3, source="computed")


async def test_a_second_user_p0_on_create_demotes_the_first(tasks: TaskStore) -> None:
    spec_p0 = spec("first", level=0, level_source="user")
    first, _ = await create_task(tasks, spec_p0, origin="user")
    second, _ = await create_task(
        tasks, spec("second", level=0, level_source="user"), origin="user"
    )
    demoted = await get_task(tasks, first.task_id)
    assert demoted is not None
    assert (demoted.level, demoted.level_eff, demoted.user_rank) == (1, 1, None)
    assert (await get_task(tasks, second.task_id)).level == 0  # type: ignore[union-attr]
    events = await priority_events(tasks, first.task_id)
    assert (events[-1].from_level, events[-1].to_level) == (0, 1)
    assert events[-1].reason == "p0_conflict"


async def test_user_can_set_priority_on_a_non_user_origin(tasks: TaskStore) -> None:
    record, _ = await create_task(tasks, spec("front task"), origin="front")
    changed = await tasks.set_priority(record.task_id, "P0")
    assert (changed.origin, changed.level, changed.level_source) == ("front", 0, "user")

    # Recovery must accept the persisted user priority for a front-origin task.
    from localmelo.support.gateway.chat_gate import ChatGate
    from localmelo.support.scheduler import Task

    gate = ChatGate(policy="class")
    gate.register_task(
        Task(
            task_id=changed.task_id,
            origin="front",
            level=changed.level,
            level_source="user",
            phase="queued",
        )
    )
    assert gate.level(changed.task_id) == 0
    await gate.close()


async def test_a_pinned_level_needs_force_to_become_computed(tasks: TaskStore) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    await tasks.set_level(record.task_id, 1, source="user")
    with pytest.raises(TaskStoreError, match="set by the user"):
        await tasks.set_level(record.task_id, 3, source="computed")
    changed = await tasks.set_level(record.task_id, 3, source="computed", force=True)
    assert (changed.level, changed.level_source, changed.level_pinned) == (
        3,
        "computed",
        0,
    )


async def test_a_user_p0_demotes_the_prior_p0_and_first_sets_a_rank(
    tasks: TaskStore, clock: Clock
) -> None:
    first, _ = await create_task(tasks, spec("first"), origin="user")
    second, _ = await create_task(tasks, spec("second"), origin="user")
    assert (await tasks.set_priority(first.task_id, "P0")).level == 0
    assert (await tasks.set_priority(second.task_id, "p0")).level == 0
    demoted = await get_task(tasks, first.task_id)
    assert demoted is not None
    assert (demoted.level, demoted.level_eff, demoted.user_rank) == (1, 1, None)
    events = await priority_events(tasks, first.task_id)
    assert [(event.from_level, event.to_level, event.reason) for event in events][
        -1
    ] == (
        0,
        1,
        "p0_conflict",
    )
    await tasks.set_priority(first.task_id, "first")
    ranked = await get_task(tasks, first.task_id)
    assert ranked is not None and ranked.user_rank is not None
    clock.advance(60)
    later = await get_task(tasks, second.task_id)
    assert later is not None
    await tasks.set_priority(second.task_id, "first")
    later = await get_task(tasks, second.task_id)
    assert later is not None and later.user_rank is not None
    assert ranked.user_rank < later.user_rank
    assert (await tasks.set_priority(first.task_id, "auto")).level == 2
    automatic = await get_task(tasks, first.task_id)
    assert automatic is not None
    assert (automatic.level, automatic.level_source, automatic.level_pinned) == (
        2,
        "computed",
        0,
    )
    assert (await tasks.set_priority(second.task_id, "P5")).level == 5
    later = await get_task(tasks, second.task_id)
    assert later is not None and later.level == 5


async def test_terminal_priority_change_does_not_demote_the_live_p0(
    tasks: TaskStore,
) -> None:
    live, _ = await create_task(tasks, spec("live"), origin="user")
    finished, _ = await create_task(tasks, spec("finished"), origin="agent")
    await tasks.set_level(live.task_id, 0)
    async with tasks.store.component_transaction("seed_terminal") as conn:
        await conn.execute(
            "UPDATE tasks SET state = 'done' WHERE task_id = ?", (finished.task_id,)
        )
    with pytest.raises(TaskStoreError, match="finished"):
        await tasks.set_level(finished.task_id, 0)
    current = await get_task(tasks, live.task_id)
    assert current is not None and current.level == 0


async def test_level_reason_and_session_timestamp_follow_changes(
    store: SessionStore, tasks: TaskStore, clock: Clock
) -> None:
    record, _ = await create_task(tasks, spec(reason="old"), origin="agent")
    clock.advance(60)
    changed = await tasks.set_level(record.task_id, 1, reason="new")
    assert changed.reason == "new"
    assert await one(
        store, "SELECT updated_at FROM sessions WHERE id = ?", (record.task_id,)
    ) == (clock.now,)


async def test_p0_demotion_touches_the_prior_task_session(
    store: SessionStore, tasks: TaskStore, clock: Clock
) -> None:
    first, _ = await create_task(tasks, spec("first"), origin="user")
    second, _ = await create_task(tasks, spec("second"), origin="agent")
    await tasks.set_level(first.task_id, 0, reason="urgent")
    clock.advance(60)
    await tasks.set_level(second.task_id, 0)
    prior = await get_task(tasks, first.task_id)
    assert prior is not None and (prior.level, prior.reason) == (1, "p0_conflict")
    assert await one(
        store, "SELECT updated_at FROM sessions WHERE id = ?", (first.task_id,)
    ) == (clock.now,)


async def test_unchanged_level_and_first_are_not_priority_labels(
    store: SessionStore, tasks: TaskStore
) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    await tasks.set_level(record.task_id, 2, source="user")
    await tasks.set_priority(record.task_id, "first")
    events = await rows(
        store,
        "SELECT from_level, to_level FROM priority_events WHERE task_id = ?",
        (record.task_id,),
    )
    assert events == [(None, 2)]


async def test_a_pinned_level_is_a_priority_event(tasks: TaskStore) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    changed = await tasks.set_level(
        record.task_id, 3, source="user", actor="user", reason="later today"
    )
    assert (changed.level, changed.level_source, changed.level_pinned) == (
        3,
        "user",
        1,
    )
    assert changed.peak_level == 2
    with pytest.raises(ValueError, match="P0 is the user"):
        await tasks.set_level(record.task_id, 0, source="computed")


async def test_stored_tasks_support_filters_and_ordering(
    tasks: TaskStore, clock: Clock
) -> None:
    first, _ = await create_task(tasks, spec("first"), origin="user")
    clock.advance(1)
    second, _ = await create_task(tasks, spec("second"), origin="agent")
    await tasks.set_priority(first.task_id, "P1")
    assert await rows(
        tasks.store, "SELECT task_id FROM tasks ORDER BY created_at DESC, task_id"
    ) == [(second.task_id,), (first.task_id,)]
    assert await rows(
        tasks.store, "SELECT task_id FROM tasks WHERE level = ?", (1,)
    ) == [(first.task_id,)]
    assert await rows(
        tasks.store,
        "SELECT task_id FROM tasks WHERE state = ? ORDER BY created_at DESC",
        ("queued",),
    ) == [(second.task_id,), (first.task_id,)]


async def test_retention_clears_old_logs_and_closed_idle_windows(
    store: SessionStore, tasks: TaskStore, clock: Clock
) -> None:
    now = clock.now
    old = now - 31 * 86_400.0
    async with store.component_transaction("seed") as conn:
        await conn.execute(
            "INSERT INTO sched_log (ts, kind, task_id, data) VALUES (?, ?, ?, ?)",
            (old, "old", None, "{}"),
        )
        await conn.execute(
            "INSERT INTO sched_log (ts, kind, task_id, data) VALUES (?, ?, ?, ?)",
            (now, "new", None, "{}"),
        )
        await conn.execute(
            "INSERT INTO idle_windows (start, end) VALUES (?, ?)", (old, old + 10)
        )
        await conn.execute(
            "INSERT INTO idle_windows (start, end) VALUES (?, ?)", (now, now + 10)
        )
    assert await tasks.apply_retention() == 2
    assert [row[1] for row in await rows(store, "SELECT ts, kind FROM sched_log")] == [
        "new"
    ]
    assert len(await rows(store, "SELECT start FROM idle_windows")) == 1


async def test_the_clock_keeps_rows_inside_the_retention_window(
    tasks: TaskStore, clock: Clock
) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    assert await tasks.apply_retention() == 0
    clock.advance(30 * 86_400.0 + 1)
    assert await tasks.apply_retention() == 1
    assert await recent_log(tasks, record.task_id) == []


async def test_the_gateway_sweep_runs_the_scheduler_retention(
    store: SessionStore, tasks: TaskStore, clock: Clock
) -> None:
    record, _ = await create_task(tasks, spec(), origin="user")
    clock.advance(31 * 86_400.0)
    assert await store.apply_retention() >= 1
    assert await recent_log(tasks, record.task_id) == []
    assert await get_task(tasks, record.task_id) is not None
