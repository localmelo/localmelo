"""The procedural store: SqliteProcedures and MemoryProcedures.

Every SQLite file is under ``tmp_path``. Tests that both stores must pass
take the ``store`` fixture; the rest are about the file.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import stat
import statistics
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.procedures import store as store_module
from localmelo.melo.memory.procedures.normalize import (
    RoutineKey,
    key_for_query,
    key_for_routine,
    template_id,
)
from localmelo.melo.memory.procedures.store import (
    BUSY_TIMEOUT_MS,
    COLUMNS,
    SCHEMA_VERSION,
    MemoryProcedures,
    Procedures,
    ProceduresSchemaError,
    RunFacts,
    SqliteProcedures,
    episode_from_row,
    harness_version,
    routine_from_row,
    template_from_row,
)
from localmelo.melo.memory.procedures.template import (
    DEFAULT_PARAMS,
    Params,
    SkeletonItem,
    Template,
)
from localmelo.melo.memory.procedures.trace import Segment, Trace

OPTIONS = {"attempt_policy": "reflect", "max_steps": 30}
PLACE = "/work/project"
KEY = key_for_query("run pytest and fix the failures", PLACE)


def segments(steps: int) -> tuple[Segment, ...]:
    """R, W, then T on the last step, which passes."""
    return (
        Segment("R", "Read test_parser.py", "test_parser.py", (1, 1)),
        Segment("W", "Edit parser.py", "parser.py", (2, max(2, steps - 1))),
        Segment("T", "pytest", "pytest", (steps, steps), passed=True),
    )


class Runs:
    """Builds RunFacts with increasing task ids and times."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(
        self,
        steps: int,
        status: str = "completed",
        *,
        key: RoutineKey = KEY,
        model: str = "model-a",
        options: dict[str, Any] | None = None,
        attempts: int = 1,
    ) -> RunFacts:
        self.n += 1
        return RunFacts(
            task_id=f"task-{self.n:04d}",
            key=key,
            model=model,
            options=OPTIONS if options is None else options,
            status=status,
            steps=steps,
            attempts=attempts,
            duration_s=10.0 * steps,
            segments=segments(steps),
            first_pass_step=steps,
            cycles=1,
            n_tool=3,
            harness_version="1.2.3",
            created_at=float(self.n),
        )


@pytest.fixture
def runs() -> Runs:
    return Runs()


@pytest.fixture(params=["sqlite", "memory"])
async def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[Procedures]:
    if request.param == "sqlite":
        made: Procedures = SqliteProcedures(tmp_path / "procedures.db")
    else:
        made = MemoryProcedures()
    try:
        yield made
    finally:
        await made.aclose()


async def rows(store: Procedures, table: str) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in row.items() if k != "table"}
        for row in await store.export_rows()
        if row["table"] == table
    ]


async def lookup(
    store: Procedures, key: RoutineKey = KEY, model: str = "model-a", **kw: Any
) -> Any:
    return await store.lookup(key, model=model, options=kw.get("options", OPTIONS))


# ── the lifecycle (section 8) ───────────────────────────────────────────────


async def test_a_first_run_creates_the_routine_its_episode_and_version_1(
    store: Procedures, runs: Runs
) -> None:
    assert await lookup(store) is None
    result = await store.learn(runs(6))
    assert result.routine.key == KEY.key and result.routine.place == PLACE
    assert result.routine.query_text == "run pytest and fix the failures"
    assert result.episode.match == "new" and result.episode.level == 0
    assert result.episode.template_version is None
    assert result.version is not None and result.version.version == 1
    tid = template_id(result.routine.routine_id, "model-a", OPTIONS)
    assert result.episode.template_id == result.version.template_id == tid
    view = await lookup(store)
    assert view is not None
    assert (view.level, view.reason, view.match) == (1, "1 of 2 runs", "exact")
    assert [item.label for item in view.skeleton] == [
        "Read test_parser.py",
        "Edit parser.py",
        "pytest",
    ]


async def test_the_second_run_appends_version_2_and_supersedes_1(
    store: Procedures, runs: Runs
) -> None:
    await store.learn(runs(6))
    second = await store.learn(runs(8))
    assert second.episode.match == "exact"
    assert (second.episode.level, second.episode.template_version) == (1, 1)
    assert second.version is not None
    assert (second.version.version, second.version.parent_version) == (2, 1)
    statuses = {row["version"]: row["status"] for row in await rows(store, "templates")}
    assert statuses == {1: "superseded", 2: "active"}
    view = await lookup(store)
    assert view.level == 2 and view.version == 2
    assert view.stats.steps == (6, 8)


async def test_other_statuses_add_only_the_episode(
    store: Procedures, runs: Runs
) -> None:
    await store.learn(runs(6))
    result = await store.learn(runs(30, "budget_exhausted"))
    assert result.version is None
    assert len(await rows(store, "episodes")) == 2
    assert len(await rows(store, "templates")) == 1


async def test_a_template_belongs_to_one_model_and_one_set_of_options(
    store: Procedures, runs: Runs
) -> None:
    await store.learn(runs(6))
    await store.learn(runs(7))
    assert (await lookup(store)).level == 2
    assert await lookup(store, model="model-b") is None
    assert await lookup(store, options={**OPTIONS, "max_steps": 20}) is None
    other = await store.learn(runs(9, model="model-b"))
    assert other.episode.match == "exact" and other.episode.level == 0
    assert other.version is not None and other.version.version == 1


async def test_an_unknown_model_id_gets_level_1_at_most(
    store: Procedures, runs: Runs
) -> None:
    for steps in (6, 6, 6):
        await store.learn(runs(steps, model="default"))
    view = await lookup(store, model="default")
    assert (view.level, view.reason) == (1, "no model id")


async def test_the_window_holds_at_most_8_completed_runs(
    store: Procedures, runs: Runs
) -> None:
    for steps in range(4, 14):
        await store.learn(runs(steps))
        await store.learn(runs(steps, "failed"))
    view = await lookup(store)
    assert view.stats.steps == tuple(range(6, 14))
    active = [
        row for row in await rows(store, "templates") if row["status"] == "active"
    ]
    assert len(active) == 1
    assert len(template_from_row(active[0]).source_ids) == 8


async def test_the_daily_routine_gets_its_bar_on_day_4(
    store: Procedures, runs: Runs
) -> None:
    daily = key_for_routine("daily-mail", "Summarize today's mail")
    levels = []
    for steps in (9, 5, 5, 5):
        view = await lookup(store, key=daily)
        levels.append(None if view is None else view.level)
        await store.learn(runs(steps, key=daily))
    assert levels == [None, 1, 1, 2]
    view = await lookup(store, key=daily)
    assert view.match == "routine" and view.stats.steps == (5, 5, 5)


async def test_a_named_routine_is_one_routine_in_any_folder(
    store: Procedures, runs: Runs
) -> None:
    here = key_for_routine("daily-mail", "mail please")
    there = key_for_routine("daily-mail", "summarize mail")
    first = await store.learn(runs(5, key=here))
    second = await store.learn(runs(5, key=there))
    assert second.routine.routine_id == first.routine.routine_id
    assert second.episode.match == "routine"
    assert first.routine.name == "daily-mail"


async def test_shadow_scores_need_a_gate_that_passed(
    store: Procedures, runs: Runs
) -> None:
    first = await store.learn(runs(6))
    second = await store.learn(runs(9))  # one completed run: the gate fails
    third = await store.learn(runs(7))  # window 6, 9: the gate passes
    assert first.episode.shadow_bar_error is None
    assert second.episode.shadow_bar_error is None
    assert third.episode.shadow_bar_error is not None
    assert 0 < third.episode.shadow_bar_error < 0.2


async def test_a_model_with_bad_shadow_scores_is_demoted(
    store: Procedures, runs: Runs
) -> None:
    for i in range(5):
        key = key_for_query(f"routine {i}", PLACE)
        await store.learn(runs(4, key=key))
        await store.learn(runs(4, key=key))
        bad = await store.learn(runs(16, "failed", key=key))
        assert bad.episode.shadow_bar_error is not None
        assert bad.episode.shadow_bar_error > 0.18
    fresh = key_for_query("a sixth routine", PLACE)
    await store.learn(runs(5, key=fresh))
    await store.learn(runs(5, key=fresh))
    view = await lookup(store, key=fresh)
    assert (view.level, view.reason) == (1, "model demoted")
    # Another model keeps its own record.
    await store.learn(runs(5, key=fresh, model="model-b"))
    await store.learn(runs(5, key=fresh, model="model-b"))
    assert (await lookup(store, key=fresh, model="model-b")).level == 2


async def test_a_similar_task_borrows_named_steps_at_level_1(
    store: Procedures, runs: Runs
) -> None:
    texts = [
        "summarize today's unread mail into the daily notes file",
        "run pytest in the project and fix the failing tests",
        "convert the sales csv export into a json report",
        "copy the photos folder to the backup drive",
        "delete log files older than a week from the logs directory",
    ]
    learned = {}
    for text in texts:
        key = key_for_query(text, PLACE)
        learned[text] = await store.learn(runs(5, key=key))
        await store.learn(runs(6, key=key))
    reworded = key_for_query(
        "Summarize the unread mail of today into the daily notes", PLACE
    )
    view = await lookup(store, key=reworded)
    assert view is not None
    assert (view.level, view.match) == (1, "similar")
    assert view.routine_id == learned[texts[0]].routine.routine_id
    assert view.title == texts[0]
    elsewhere = key_for_query(
        "Summarize the unread mail of today into the daily notes", "/x"
    )
    assert await lookup(store, key=elsewhere) is None
    result = await store.learn(runs(5, key=reworded))
    assert result.episode.match == "similar" and result.episode.level == 1
    assert result.routine.similar_to == learned[texts[0]].routine.routine_id
    assert result.routine.routine_id != learned[texts[0]].routine.routine_id
    # The new routine earns its own template.
    assert (await lookup(store, key=reworded)).match == "exact"


async def test_the_view_given_to_learn_is_what_the_episode_records(
    store: Procedures, runs: Runs
) -> None:
    await store.learn(runs(6))
    await store.learn(runs(6))
    view = await lookup(store)
    result = await store.learn(runs(6), view=replace(view, level=1))
    assert (result.episode.level, result.episode.template_version) == (1, 2)


async def test_a_task_has_one_episode(store: Procedures, runs: Runs) -> None:
    facts = runs(6)
    await store.learn(facts)
    before = await store.export_rows()
    with pytest.raises(ValueError, match="already has an episode"):
        await store.learn(replace(facts, key=key_for_query("other", PLACE)))
    assert await store.export_rows() == before


async def test_a_failure_inside_learn_changes_nothing(
    store: Procedures, runs: Runs, monkeypatch: pytest.MonkeyPatch
) -> None:
    await store.learn(runs(6))
    before = await store.export_rows()

    async def broken(self: Any, template: Template) -> None:
        raise RuntimeError("disk full")

    for tx in (store_module._SqlTx, store_module._MemoryTx):
        monkeypatch.setattr(tx, "insert_template", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        await store.learn(runs(7, key=key_for_query("a new task", PLACE)))
    assert await store.export_rows() == before


async def test_learn_reads_a_trace(store: Procedures) -> None:
    trace = Trace(clock=lambda: 0.0)
    trace.step(1, 1)
    trace.tool("file_read", {"path": "/p/a.py"}, "success", "x")
    trace.step(2, 1)
    trace.tool("shell_exec", {"command": "pytest"}, "success", "2 passed in 0.1s")
    trace.step(3, 1)
    facts = RunFacts.from_trace(
        trace,
        task_id="t1",
        key=KEY,
        model="m",
        options=OPTIONS,
        status="completed",
        steps=3,
        attempts=1,
        duration_s=5.0,
    )
    assert (facts.first_pass_step, facts.n_tool, facts.cycles) == (2, 2, 0)
    result = await store.learn(facts)
    assert [s.label for s in result.episode.segments] == ["Read a.py", "pytest"]
    assert result.episode.created_at > 0


# ── sleep candidates (section 8) ────────────────────────────────────────────


def candidate(active: Template, *, parent: int, **evaluation: Any) -> Template:
    full = {
        "heldout_task_ids": ["task-0001"],
        "bar_error": 0.05,
        "parent_bar_error": 0.10,
        "cap_share": 0.0,
        "longer_rate": 0.1,
        "reason": "aligned",
        **evaluation,
    }
    return replace(
        active,
        skeleton=(SkeletonItem(("W", "T"), None, "Edit + pytest", 2),),
        parent_version=parent,
        eval=full,
    )


async def active_template(store: Procedures) -> Template:
    (row,) = [r for r in await rows(store, "templates") if r["status"] == "active"]
    return template_from_row(row)


async def test_a_sleep_candidate_is_activated_and_its_skeleton_kept(
    store: Procedures, runs: Runs
) -> None:
    await store.learn(runs(6))
    await store.learn(runs(7))
    active = await active_template(store)
    added = await store.add_candidate(candidate(active, parent=2))
    assert (added.version, added.status, added.source, added.admitted_by) == (
        3,
        "candidate",
        "sleep",
        "sleep",
    )
    assert (await lookup(store)).version == 2
    assert await store.decide_candidate(active.template_id, 3) == "active"
    statuses = {r["version"]: r["status"] for r in await rows(store, "templates")}
    assert statuses == {1: "superseded", 2: "superseded", 3: "active"}
    online = await store.learn(runs(8))
    assert online.version is not None and online.version.version == 4
    assert online.version.skeleton == added.skeleton
    assert online.version.source == "online"
    later = await store.learn(runs(7))
    assert later.version is not None and later.version.skeleton == added.skeleton


async def test_a_stale_candidate_is_rejected(store: Procedures, runs: Runs) -> None:
    await store.learn(runs(6))
    await store.learn(runs(7))
    active = await active_template(store)
    await store.add_candidate(candidate(active, parent=2))
    await store.learn(runs(6))  # an online version 4 lands first
    assert await store.decide_candidate(active.template_id, 3) == "rejected"
    row = next(r for r in await rows(store, "templates") if r["version"] == 3)
    assert row["status"] == "rejected"
    assert dict(template_from_row(row).eval or {})["reason"] == "stale"
    assert (await lookup(store)).version == 4


async def test_a_candidate_that_is_not_better_is_rejected(
    store: Procedures, runs: Runs
) -> None:
    await store.learn(runs(6))
    await store.learn(runs(7))
    active = await active_template(store)
    await store.add_candidate(candidate(active, parent=2, bar_error=0.2))
    await store.add_candidate(candidate(active, parent=2, cap_share=0.06))
    assert await store.decide_candidate(active.template_id, 3) == "rejected"
    assert await store.decide_candidate(active.template_id, 4) == "rejected"
    rejected = [
        template_from_row(r) for r in await rows(store, "templates") if r["version"] > 2
    ]
    assert [dict(t.eval or {})["reason"] for t in rejected] == ["aligned", "aligned"]
    assert (await lookup(store)).version == 2


async def test_candidate_refusals(store: Procedures, runs: Runs) -> None:
    await store.learn(runs(6))
    active = await active_template(store)
    with pytest.raises(ValueError, match="lacks"):
        await store.add_candidate(
            replace(active, parent_version=1, eval={"bar_error": 0.1})
        )
    with pytest.raises(ValueError, match="names the version"):
        await store.add_candidate(
            replace(candidate(active, parent=1), parent_version=None)
        )
    with pytest.raises(ValueError, match="no version 7"):
        await store.add_candidate(candidate(active, parent=7))
    with pytest.raises(ValueError, match="no candidate version 1"):
        await store.decide_candidate(active.template_id, 1)


# ── export and forget ───────────────────────────────────────────────────────


async def test_export_rows(store: Procedures, runs: Runs) -> None:
    await store.learn(runs(6))
    exported = await store.export_rows()
    assert [row["table"] for row in exported] == ["routines", "episodes", "templates"]
    for row in exported:
        assert list(row)[1:] == list(COLUMNS[row["table"]])
    routine, episode, template = exported
    assert routine_from_row(routine).key == KEY.key
    assert episode_from_row(episode).segments == segments(6)
    assert isinstance(episode["segments_json"], str)
    assert template_from_row(template).stats.steps == (6,)


async def test_forget_one_routine_or_everything(store: Procedures, runs: Runs) -> None:
    first = await store.learn(runs(6))
    await store.learn(runs(6))
    other = key_for_query("another task", PLACE)
    await store.learn(runs(5, key=other))
    deleted = await store.forget(first.routine.routine_id)
    assert deleted == 1 + 2 + 2
    assert await lookup(store) is None
    assert {r["key"] for r in await rows(store, "routines")} == {other.key}
    assert await store.forget("no-such-routine") == 0
    assert await store.forget() == 3
    assert await store.export_rows() == []


# ── the file ────────────────────────────────────────────────────────────────


async def test_the_file_mode_version_and_busy_timeout(
    tmp_path: Path, runs: Runs
) -> None:
    path = tmp_path / "procedures.db"
    store = SqliteProcedures(path)
    try:
        await store.learn(runs(6))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        live = await store._open()
        async with live.execute("PRAGMA busy_timeout") as cursor:
            assert tuple(await cursor.fetchall()) == ((BUSY_TIMEOUT_MS,),)
        async with live.execute("PRAGMA journal_mode") as cursor:
            assert tuple(await cursor.fetchall()) == (("wal",),)
    finally:
        await store.aclose()
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        indexes = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
    assert {
        "episodes_template",
        "templates_status",
        "routines_place",
        "templates_one_active",
    } <= indexes


async def test_a_missing_file_is_not_created_without_create(tmp_path: Path) -> None:
    store = SqliteProcedures(tmp_path / "procedures.db", create=False)
    with pytest.raises(FileNotFoundError):
        await store.export_rows()
    assert not (tmp_path / "procedures.db").exists()
    await store.aclose()


async def test_a_newer_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "procedures.db"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    store = SqliteProcedures(path)
    with pytest.raises(ProceduresSchemaError, match="schema version 2"):
        await store.export_rows()
    await store.aclose()


async def test_rows_are_append_only_in_the_file(tmp_path: Path, runs: Runs) -> None:
    path = tmp_path / "procedures.db"
    store = SqliteProcedures(path)
    await store.learn(runs(6))
    await store.learn(runs(7))
    await store.aclose()
    with contextlib.closing(sqlite3.connect(path)) as conn:
        for sql in (
            "UPDATE routines SET query_text = 'x'",
            "UPDATE episodes SET steps = 1",
            "UPDATE templates SET stats_json = '{}' WHERE version = 1",
            "UPDATE templates SET status = 'active' WHERE version = 1",
            "UPDATE templates SET status = 'candidate' WHERE version = 2",
            "UPDATE templates SET eval_json = '{}' WHERE version = 2",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
        # Two active versions of one template are refused too.
        conn.execute("DROP TRIGGER templates_append_only")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            conn.execute("UPDATE templates SET status = 'active' WHERE version = 1")


async def test_one_active_row_under_two_concurrent_writers(tmp_path: Path) -> None:
    path = tmp_path / "procedures.db"
    first, second = SqliteProcedures(path), SqliteProcedures(path)
    make = Runs()
    try:
        facts = [make(5 + i % 3) for i in range(40)]
        results = await asyncio.gather(
            *((first if i % 2 else second).learn(fact) for i, fact in enumerate(facts))
        )
    finally:
        await first.aclose()
        await second.aclose()
    versions = sorted(r.version.version for r in results if r.version is not None)
    assert versions == list(range(1, 41))
    with contextlib.closing(sqlite3.connect(path)) as conn:
        statuses = conn.execute(
            "SELECT version, status, parent_version FROM templates ORDER BY version"
        ).fetchall()
    assert [status for _, status, _ in statuses].count("active") == 1
    assert statuses[-1][1] == "active"
    # Each version's parent is the one before it: no update was lost.
    assert [parent for _, _, parent in statuses] == [None, *range(1, 40)]


async def test_a_cancelled_call_drops_the_connection_and_the_next_reopens(
    tmp_path: Path, runs: Runs
) -> None:
    path = tmp_path / "procedures.db"
    store = SqliteProcedures(path)
    await store.learn(runs(6))
    blocker = sqlite3.connect(path, timeout=0, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")  # holds the write lock
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(store.learn(runs(7)), timeout=0.1)
        assert store._conn is None
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    result = await store.learn(runs(8))
    assert result.version is not None and result.version.version == 2
    await store.aclose()
    assert not store._closing
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 2


async def test_a_locked_file_fails_after_the_busy_timeout(
    tmp_path: Path, runs: Runs
) -> None:
    path = tmp_path / "procedures.db"
    store = SqliteProcedures(path)
    await store.learn(runs(6))
    blocker = sqlite3.connect(path, timeout=0, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            await store.learn(runs(7))
        assert 0.4 < time.monotonic() - started < 3
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert (await store.learn(runs(8))).version is not None
    await store.aclose()


async def test_forget_overwrites_what_it_deletes(tmp_path: Path, runs: Runs) -> None:
    path = tmp_path / "procedures.db"
    store = SqliteProcedures(path)
    secret = key_for_query("a very distinctive private request zebra-quokka", PLACE)
    await store.learn(runs(6, key=secret))
    await store.forget()
    await store.aclose()
    data = b"".join(
        p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("procedures.db")
    )
    assert b"zebra-quokka" not in data


async def test_aclose_then_reopen(tmp_path: Path, runs: Runs) -> None:
    store = SqliteProcedures(tmp_path / "procedures.db")
    await store.learn(runs(6))
    await store.aclose()
    await store.learn(runs(7))
    assert (await lookup(store)).level == 2
    await store.aclose()
    await store.aclose()


def test_harness_version() -> None:
    assert isinstance(harness_version(), str) and harness_version()


async def test_lookup_and_learn_are_fast_with_1000_episodes(tmp_path: Path) -> None:
    store = SqliteProcedures(tmp_path / "procedures.db")
    make = Runs()
    keys = [key_for_query(f"task number {i}", PLACE) for i in range(100)]
    try:
        for i in range(1000):
            await store.learn(make(4 + i % 3, key=keys[i % 100]))
        looks: list[float] = []
        writes: list[float] = []
        for j in range(100):
            key = keys[j]
            started = time.perf_counter()
            await store.lookup(key, model="model-a", options=OPTIONS)
            looks.append(time.perf_counter() - started)
            started = time.perf_counter()
            await store.learn(make(5, key=key))
            writes.append(time.perf_counter() - started)
    finally:
        await store.aclose()

    def p95(values: list[float]) -> float:
        return statistics.quantiles(values, n=20)[-1]

    assert p95(looks) < 0.020
    assert p95(writes) < 0.050


async def test_a_memory_store_keeps_params(runs: Runs) -> None:
    store = MemoryProcedures(params=Params(drop="off", spread=0.6))
    await store.learn(runs(9))
    await store.learn(runs(6))
    view = await lookup(store)
    assert view.level == 2 and view.params == Params(drop="off", spread=0.6)
    assert DEFAULT_PARAMS.spread == 0.5


async def test_a_similar_view_records_no_version_of_its_own_template(
    store: Procedures, runs: Runs
) -> None:
    texts = [
        "summarize today's unread mail into the daily notes file",
        "run pytest in the project and fix the failing tests",
        "convert the sales csv export into a json report",
        "copy the photos folder to the backup drive",
        "delete log files older than a week from the logs directory",
    ]
    for text in texts:
        key = key_for_query(text, PLACE)
        await store.learn(runs(5, key=key))
        await store.learn(runs(6, key=key))
    reworded = key_for_query("Summarize the unread mail of today into notes", PLACE)
    view = await lookup(store, key=reworded)
    assert view is not None and view.match == "similar"
    result = await store.learn(runs(5, key=reworded), view=view)
    # The version shown belonged to the similar routine's template, not to
    # the template this episode belongs to.
    assert result.episode.template_id != view.template_id
    assert (result.episode.level, result.episode.template_version) == (1, None)


async def test_a_newer_file_is_not_written_to(tmp_path: Path) -> None:
    path = tmp_path / "procedures.db"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE notes (text TEXT)")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    before = path.read_bytes()
    store = SqliteProcedures(path)
    with pytest.raises(ProceduresSchemaError):
        await store.export_rows()
    await store.aclose()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["procedures.db"]


def test_segments_from_json_skips_infinite_numbers() -> None:
    from localmelo.melo.memory.procedures.store import segments_from_json

    text = '[{"cls": "R", "steps": [1, 1e999]}, {"cls": "W", "steps": [2, 2]}]'
    assert [s.cls for s in segments_from_json(text)] == ["W"]
    with pytest.raises(ValueError):
        segments_from_json(text, strict=True)
