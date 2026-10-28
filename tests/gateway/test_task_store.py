"""The scheduler's task store: component v1, atomic tasks, levels and retention."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import (
    COMPONENT,
    SCHEDULER_VERSION,
    TaskConflictError,
    TaskNotFoundError,
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
    return TaskSpec.from_payload({"objective": objective, **values})


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


async def rows(store: SessionStore, sql: str, params: Sequence[Any] = ()) -> list[Any]:
    async with store.component_read() as conn:
        cursor = await conn.execute(sql, tuple(params))
        return list(await cursor.fetchall())


async def one(store: SessionStore, sql: str, params: Sequence[Any] = ()) -> Any:
    found = await rows(store, sql, params)
    return found[0] if found else None


async def test_open_creates_seven_tables_and_leaves_user_version_alone(
    store: SessionStore, clock: Clock
) -> None:
    assert await store.component_version(COMPONENT) is None
    assert await one(store, "PRAGMA user_version") == (1,)

    tasks = TaskStore(store, clock=clock)
    assert await tasks.open() == SCHEDULER_VERSION
    assert tasks.version == SCHEDULER_VERSION
    assert await store.component_version(COMPONENT) == SCHEDULER_VERSION
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
    assert await store.component_version("broken") is None


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
    record, _ = await tasks.create_task(
        spec("stays"), origin="user", request_key="stays"
    )
    await first.aclose()

    second = SessionStore(path, writer="gateway", clock=clock)
    await second.open()
    try:
        reopened = TaskStore(second, clock=clock)
        assert await reopened.open() == SCHEDULER_VERSION
        assert (await reopened.get_task(record.task_id)) is not None
        again, repeated = await reopened.create_task(
            spec("stays"), origin="user", request_key="stays"
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
        assert await cli.component_version(COMPONENT) is None
    finally:
        await cli.aclose()


async def test_create_writes_the_task_session_and_its_first_rows(
    store: SessionStore, tasks: TaskStore
) -> None:
    record, repeated = await tasks.create_task(
        spec(
            "Make six slides",
            done_when=["slides.pdf exists"],
            user_words="make slides",
            workspace="/tmp/talk",
            level=1,
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
    assert record.level_source == "computed"
    assert record.spec["objective"] == "Make six slides"
    assert record.spec["task_id"] == record.task_id
    session = await one(
        store,
        "SELECT kind, title, workspace FROM sessions WHERE id = ?",
        (record.task_id,),
    )
    assert session == ("task", "Make six slides", "/tmp/talk")
    event = await one(
        store,
        "SELECT actor, from_level, to_level, reason, quote FROM priority_events "
        "WHERE task_id = ?",
        (record.task_id,),
    )
    assert event == ("user", None, 1, "P1 · due today", "make slides")
    logged = await tasks.recent_log(task_id=record.task_id)
    assert [row[1] for row in logged] == ["task.created"]


async def test_create_rolls_back_the_session_task_and_events_together(
    store: SessionStore, tasks: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(label: str) -> None:
        if label == "task_create":
            raise RuntimeError("stop before commit")

    monkeypatch.setattr(store, "_before_commit", fail)
    with pytest.raises(RuntimeError, match="stop before commit"):
        await tasks.create_task(spec(), origin="user", request_key="rollback")
    assert await one(store, "SELECT COUNT(*) FROM sessions") == (0,)
    assert await one(store, "SELECT COUNT(*) FROM tasks") == (0,)
    assert await one(store, "SELECT COUNT(*) FROM priority_events") == (0,)
    assert await one(store, "SELECT COUNT(*) FROM sched_log") == (0,)


async def test_a_request_key_is_idempotent_and_a_reuse_conflicts(
    tasks: TaskStore,
) -> None:
    first, repeated = await tasks.create_task(
        spec("one"), origin="user", request_key="same"
    )
    assert repeated is False
    again, repeated = await tasks.create_task(
        spec("one"), origin="user", request_key="same"
    )
    assert repeated is True
    assert again.task_id == first.task_id
    with pytest.raises(TaskConflictError, match="different task"):
        await tasks.create_task(spec("two"), origin="user", request_key="same")
    other, _ = await tasks.create_task(spec("one"), origin="user")
    assert other.task_id != first.task_id


async def test_a_second_user_p0_on_create_demotes_the_first(tasks: TaskStore) -> None:
    spec_p0 = spec("first", level=0, level_source="user")
    first, _ = await tasks.create_task(spec_p0, origin="user")
    second, _ = await tasks.create_task(
        spec("second", level=0, level_source="user"), origin="user"
    )
    demoted = await tasks.get_task(first.task_id)
    assert demoted is not None
    assert (demoted.level, demoted.level_eff, demoted.user_rank) == (1, 1, None)
    assert (await tasks.get_task(second.task_id)).level == 0  # type: ignore[union-attr]
    events = await tasks.priority_events(first.task_id)
    assert (events[-1].from_level, events[-1].to_level) == (0, 1)
    assert events[-1].reason == "p0_conflict"


async def test_a_non_user_origin_cannot_carry_a_user_level(tasks: TaskStore) -> None:
    with pytest.raises(ValueError, match="user-origin"):
        await tasks.create_task(
            spec("front p0", level=0, level_source="user"), origin="front"
        )
    with pytest.raises(ValueError, match="user rule"):
        await tasks.create_task(
            spec("agent rule", level=1, level_source="user_rule"), origin="agent"
        )


async def test_a_pinned_level_needs_force_to_become_computed(tasks: TaskStore) -> None:
    record, _ = await tasks.create_task(spec(), origin="user")
    await tasks.set_level(record.task_id, 1, source="user")
    with pytest.raises(TaskStoreError, match="set by the user"):
        await tasks.set_level(record.task_id, 3, source="computed")
    changed = await tasks.set_level(record.task_id, 3, source="computed", force=True)
    assert (changed.level, changed.level_source, changed.level_pinned) == (
        3,
        "computed",
        0,
    )


async def test_a_terminal_state_clears_its_wait_and_cannot_resume(
    tasks: TaskStore,
) -> None:
    record, _ = await tasks.create_task(spec(), origin="user")
    await tasks.set_state(record.task_id, "waiting")
    await tasks.set_wait_reason(record.task_id, "input", hold_site="plan/step1")
    done = await tasks.set_state(record.task_id, "done")
    assert (done.wait_reason, done.hold_site) == ("", "")
    with pytest.raises(TaskStoreError, match="cannot become"):
        await tasks.set_state(record.task_id, "running")


async def test_a_user_p0_demotes_the_prior_p0_and_first_sets_a_rank(
    tasks: TaskStore, clock: Clock
) -> None:
    first, _ = await tasks.create_task(spec("first"), origin="user")
    second, _ = await tasks.create_task(spec("second"), origin="user")
    assert (await tasks.set_priority(first.task_id, "P0")).level == 0
    assert (await tasks.set_priority(second.task_id, "p0")).level == 0
    demoted = await tasks.get_task(first.task_id)
    assert demoted is not None
    assert (demoted.level, demoted.level_eff, demoted.user_rank) == (1, 1, None)
    events = await tasks.priority_events(first.task_id)
    assert [(event.from_level, event.to_level, event.reason) for event in events][
        -1
    ] == (
        0,
        1,
        "p0_conflict",
    )
    await tasks.set_priority(first.task_id, "first")
    ranked = await tasks.get_task(first.task_id)
    assert ranked is not None and ranked.user_rank is not None
    clock.advance(60)
    later = await tasks.get_task(second.task_id)
    assert later is not None
    await tasks.set_priority(second.task_id, "first")
    later = await tasks.get_task(second.task_id)
    assert later is not None and later.user_rank is not None
    assert ranked.user_rank < later.user_rank
    assert (await tasks.set_priority(first.task_id, "auto")).level == 2
    automatic = await tasks.get_task(first.task_id)
    assert automatic is not None
    assert (automatic.level, automatic.level_source, automatic.level_pinned) == (
        2,
        "computed",
        0,
    )
    assert (await tasks.set_priority(second.task_id, "P5")).level == 5
    later = await tasks.get_task(second.task_id)
    assert later is not None and later.level == 5


async def test_state_changes_log_and_time_the_task(tasks: TaskStore) -> None:
    record, _ = await tasks.create_task(spec(), origin="user")
    running = await tasks.set_state(record.task_id, "running")
    assert running.started_at is not None and running.finished_at is None
    cancelled = await tasks.set_state(record.task_id, "cancelled", reason="user_stop")
    assert cancelled.finished_at is not None
    assert [row[1] for row in await tasks.recent_log(task_id=record.task_id)] == [
        "task.created",
        "task.state",
        "task.state",
    ]
    with pytest.raises(ValueError, match="unknown task state"):
        await tasks.set_state(record.task_id, "paused")
    with pytest.raises(TaskNotFoundError):
        await tasks.set_state("nope", "running")


async def test_a_pinned_level_is_a_priority_event(tasks: TaskStore) -> None:
    record, _ = await tasks.create_task(spec(), origin="user")
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


async def test_list_tasks_filters_and_orders(tasks: TaskStore, clock: Clock) -> None:
    first, _ = await tasks.create_task(spec("first"), origin="user")
    clock.advance(1)
    second, _ = await tasks.create_task(spec("second"), origin="agent")
    await tasks.set_priority(first.task_id, "P1")
    newest = await tasks.list_tasks()
    assert [task.task_id for task in newest] == [second.task_id, first.task_id]
    assert [task.task_id for task in await tasks.list_tasks(level=1)] == [first.task_id]
    assert [task.task_id for task in await tasks.list_tasks(state="queued")] == [
        second.task_id,
        first.task_id,
    ]
    with pytest.raises(ValueError, match="unknown task state"):
        await tasks.list_tasks(state="paused")
    with pytest.raises(ValueError, match="limit"):
        await tasks.list_tasks(limit=0)
    with pytest.raises(ValueError, match="level"):
        await tasks.list_tasks(level=True)


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
    record, _ = await tasks.create_task(spec(), origin="user")
    assert await tasks.apply_retention() == 0
    clock.advance(30 * 86_400.0 + 1)
    assert await tasks.apply_retention() == 1
    assert await tasks.recent_log(task_id=record.task_id) == []


async def test_the_gateway_sweep_runs_the_scheduler_retention(
    store: SessionStore, tasks: TaskStore, clock: Clock
) -> None:
    record, _ = await tasks.create_task(spec(), origin="user")
    clock.advance(31 * 86_400.0)
    assert await store.apply_retention() >= 1
    assert await tasks.recent_log(task_id=record.task_id) == []
    assert await tasks.get_task(record.task_id) is not None
