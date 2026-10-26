"""History schema version 1 and its ``llm_calls`` table (R1).

Every database lives under ``tmp_path``. A file "written before R1" (version
0) is made with ``sqlite3`` and the unchanged ``_SCHEMA``, as an older
localmelo wrote it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import sqlite3
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory._sqlite import (
    THINKING_RETENTION_DAYS,
    THINKING_RETENTION_ENV,
    make_private_dir,
    thinking_retention_days,
    tighten_private_dir,
)
from localmelo.melo.memory.history.sqlite import (
    _SCHEMA,
    HISTORY_SCHEMA_VERSION,
    HistorySchemaError,
    SqliteHistory,
)
from localmelo.melo.schema import Message, StepRecord, TaskRecord, ToolDef

HISTORY_LOGGER = "localmelo.melo.memory.history.sqlite"

# PRAGMA table_info(llm_calls) of the DDL: cid, name, type, notnull,
# default, primary-key position.
LLM_CALLS_COLUMNS = [
    (0, "task_id", "TEXT", 1, None, 1),
    (1, "seq", "INTEGER", 1, None, 2),
    (2, "kind", "TEXT", 1, None, 0),
    (3, "step", "INTEGER", 0, None, 0),
    (4, "attempt", "INTEGER", 0, None, 0),
    (5, "outcome", "TEXT", 1, None, 0),
    (6, "content", "TEXT", 1, "''", 0),
    (7, "thinking", "TEXT", 1, "''", 0),
    (8, "reasoning_key", "TEXT", 1, "''", 0),
    (9, "step_ids_json", "TEXT", 0, None, 0),
    (10, "redactions", "INTEGER", 1, "0", 0),
    (11, "tool_calls_json", "TEXT", 0, None, 0),
    (12, "usage_json", "TEXT", 0, None, 0),
    (13, "duration_s", "REAL", 0, None, 0),
    (14, "timestamp", "REAL", 1, None, 0),
]


@contextlib.contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    try:
        yield connection
    finally:
        connection.close()


def _user_version(path: Path) -> int:
    with _connect(path) as connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _tables(path: Path) -> set[str]:
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def _rows(path: Path, table: str) -> list[tuple[Any, ...]]:
    with _connect(path) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()


def _write_v0(path: Path) -> None:
    """A history.db as localmelo wrote it before R1: tasks and steps, version 0."""
    with _connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT INTO tasks (task_id, query, status, result) VALUES (?, ?, ?, ?)",
            ("t1", "What is 6 * 7?", "completed", "42"),
        )
        connection.executemany(
            """INSERT INTO steps (step_id, task_id, thought, tool_call_json,
               tool_result_json, timestamp, seq) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    "s1",
                    "t1",
                    "Computing it.",
                    '{"tool_name": "python_exec"}',
                    "{}",
                    1.5,
                    0,
                ),
                ("s2", "t1", "", None, None, 2.5, 1),
            ],
        )
        connection.commit()


def _add_call_row(path: Path, task_id: str = "t1", seq: int = 0) -> None:
    with _connect(path) as connection:
        connection.execute(
            """INSERT INTO llm_calls (task_id, seq, kind, outcome, timestamp)
               VALUES (?, ?, 'plan', 'answer', 1.0)""",
            (task_id, seq),
        )
        connection.commit()


async def _open(path: Path) -> None:
    """Open *path* through SqliteHistory, as the first history call does."""
    history = SqliteHistory(path)
    try:
        await history.count_tasks()
    finally:
        await history.aclose()


def _migration_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == HISTORY_LOGGER and record.getMessage().startswith("Migrated")
    ]


# ── Schema version 1 ────────────────────────────────────────────────────────


async def test_a_new_file_is_at_version_1_with_the_table(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    await _open(path)
    assert HISTORY_SCHEMA_VERSION == 1
    assert _user_version(path) == 1
    with _connect(path) as connection:
        columns = connection.execute("PRAGMA table_info(llm_calls)").fetchall()
        stored = dict(
            connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name IN ('tasks', 'steps')"
            ).fetchall()
        )
        indexes = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = 'llm_calls'"
        ).fetchall()
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(llm_calls)"
        ).fetchall()
    assert columns == LLM_CALLS_COLUMNS
    # Only the primary key's own index, and no foreign key.
    assert [name for (name,) in indexes] == ["sqlite_autoindex_llm_calls_1"]
    assert foreign_keys == []
    # tasks and steps exactly as _SCHEMA creates them.
    reference = tmp_path / "reference.db"
    with _connect(reference) as connection:
        connection.executescript(_SCHEMA)
        expected = dict(
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        )
    assert stored == expected


async def test_a_v0_file_is_migrated_with_its_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.db"
    _write_v0(path)
    tasks, steps = _rows(path, "tasks"), _rows(path, "steps")
    assert _user_version(path) == 0
    caplog.set_level(logging.INFO, logger=HISTORY_LOGGER)

    await _open(path)

    assert _user_version(path) == 1
    assert (_rows(path, "tasks"), _rows(path, "steps")) == (tasks, steps)
    assert _rows(path, "llm_calls") == []
    assert _migration_lines(caplog) == [f"Migrated {path} to history schema version 1"]


async def test_reopening_a_v1_file_changes_nothing_and_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.db"
    _write_v0(path)
    await _open(path)
    _add_call_row(path)
    before = {table: _rows(path, table) for table in ("tasks", "steps", "llm_calls")}
    caplog.set_level(logging.DEBUG, logger=HISTORY_LOGGER)

    await _open(path)

    assert _user_version(path) == 1
    assert {table: _rows(path, table) for table in before} == before
    assert [r for r in caplog.records if r.name == HISTORY_LOGGER] == []


async def test_a_half_migrated_file_keeps_its_row(tmp_path: Path) -> None:
    # The table and a row, but version 0.
    path = tmp_path / "history.db"
    await _open(path)
    _add_call_row(path)
    with _connect(path) as connection:
        connection.execute("PRAGMA user_version = 0")
        connection.commit()
    row = _rows(path, "llm_calls")

    await _open(path)

    assert _user_version(path) == 1
    assert _rows(path, "llm_calls") == row


async def test_a_v1_file_without_the_table_gets_it_back(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    await _open(path)
    with _connect(path) as connection:
        connection.execute("DROP TABLE llm_calls")
        connection.commit()
    assert "llm_calls" not in _tables(path)

    await _open(path)

    assert _user_version(path) == 1
    assert "llm_calls" in _tables(path)


async def test_two_openers_of_a_v0_file_migrate_it_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.db"
    _write_v0(path)
    caplog.set_level(logging.INFO, logger=HISTORY_LOGGER)

    await asyncio.gather(_open(path), _open(path))

    assert _user_version(path) == 1
    assert len(_migration_lines(caplog)) == 1


def _write_v2(path: Path) -> None:
    with _connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute("PRAGMA user_version = 2")
        connection.commit()


async def test_a_newer_file_is_refused_at_every_call(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    _write_v2(path)
    history = SqliteHistory(path)
    message = f"{path} has schema version 2; this localmelo reads version 1"
    try:
        with pytest.raises(HistorySchemaError) as first:
            await history.save_task(TaskRecord(query="q", task_id="t1"))
        assert str(first.value) == message
        assert history._conn is None
        with pytest.raises(HistorySchemaError) as second:
            await history.add_step("t1", StepRecord(thought="x"))
        assert str(second.value) == message
    finally:
        await history.aclose()
    assert isinstance(first.value, RuntimeError)
    assert _user_version(path) == 2
    assert _rows(path, "tasks") == []
    assert _rows(path, "steps") == []
    assert "llm_calls" not in _tables(path)


class _FakeLLM(BaseLLMProvider):
    def __init__(self, responses: list[Message]) -> None:
        self.responses: deque[Message] = deque(responses)

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if not self.responses:
            return Message(role="assistant", content="(no response queued)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


async def test_a_run_on_a_newer_file_ends_crashed(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    path = memory_dir / "history.db"
    _write_v2(path)
    agent = Agent(
        llm=_FakeLLM([Message(role="assistant", content="42")]),
        options=AgentOptions(memory_dir=str(memory_dir)),
    )
    try:
        result = await agent.run_task("What is 6 * 7?")
    finally:
        await agent.close()
    assert result.status == "crashed"
    assert result.error == (
        f"HistorySchemaError: {path} has schema version 2; "
        "this localmelo reads version 1"
    )
    assert _rows(path, "tasks") == []


# ── File modes ──────────────────────────────────────────────────────────────

SQLITE_HELPERS_LOGGER = "localmelo.melo.memory._sqlite"


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


@contextlib.contextmanager
def _umask(mask: int) -> Iterator[None]:
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


async def test_a_new_history_file_and_its_wal_and_shm_are_0600(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    try:
        with _umask(0o022):
            await history.save_task(TaskRecord(query="q", task_id="t1"))
        wal, shm = Path(f"{path}-wal"), Path(f"{path}-shm")
        assert wal.exists() and shm.exists()
        assert (_mode(path), _mode(wal), _mode(shm)) == (0o600, 0o600, 0o600)
    finally:
        await history.aclose()


async def test_an_existing_history_file_keeps_its_mode(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    _write_v0(path)
    path.chmod(0o644)
    await _open(path)
    assert _mode(path) == 0o644


async def test_eight_openers_of_a_missing_file_make_one_private_v1_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.db"
    caplog.set_level(logging.INFO, logger=HISTORY_LOGGER)
    with _umask(0o022):
        await asyncio.gather(*(_open(path) for _ in range(8)))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["history.db"]
    assert _mode(path) == 0o600
    assert _user_version(path) == 1
    assert "llm_calls" in _tables(path)
    assert len(_migration_lines(caplog)) == 1


@pytest.mark.parametrize("mask", [0o022, 0o077])
def test_make_private_dir_makes_a_new_directory_0700(tmp_path: Path, mask: int) -> None:
    path = tmp_path / "parent" / "memory"
    with _umask(mask):
        make_private_dir(str(path))
    assert _mode(path) == 0o700
    # A missing parent gets the default mode.
    assert _mode(path.parent) == 0o777 & ~mask


def test_make_private_dir_leaves_an_existing_directory_alone(tmp_path: Path) -> None:
    path = tmp_path / "memory"
    path.mkdir()
    path.chmod(0o755)
    make_private_dir(str(path))
    assert _mode(path) == 0o755


def test_make_private_dir_raises_what_makedirs_raises(tmp_path: Path) -> None:
    a_file = tmp_path / "file"
    a_file.write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError):
        make_private_dir(str(a_file))
    with pytest.raises(NotADirectoryError):
        make_private_dir(str(a_file / "memory"))


def test_tighten_private_dir_closes_an_open_directory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "memory"
    path.mkdir()
    path.chmod(0o755)
    caplog.set_level(logging.DEBUG, logger=SQLITE_HELPERS_LOGGER)
    tighten_private_dir(str(path))
    assert _mode(path) == 0o700
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, f"Restricted {path} to mode 0700")
    ]


def test_tighten_private_dir_leaves_a_closed_directory_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "memory"
    path.mkdir()
    path.chmod(0o700)
    caplog.set_level(logging.DEBUG, logger=SQLITE_HELPERS_LOGGER)
    tighten_private_dir(str(path))
    assert _mode(path) == 0o700
    assert caplog.records == []


def test_tighten_private_dir_logs_a_failed_chmod(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "memory"
    path.mkdir()
    path.chmod(0o755)

    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", refuse)
    caplog.set_level(logging.DEBUG, logger=SQLITE_HELPERS_LOGGER)
    tighten_private_dir(str(path))
    monkeypatch.undo()
    assert _mode(path) == 0o755
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert str(path) in caplog.records[0].getMessage()


# ── Retention of thinking ───────────────────────────────────────────────────

DAY = 86400.0
NOW = 2_000_000_000.0

# (reasoning_key, thinking, content, content once the reasoning is cleared)
RETENTION_CASES = [
    ("thinking", "x", "<think>\nx\n</think>\n\nanswer", "answer"),
    ("thinking", "a</think>b", "<think>\na</think>b\n</think>\n\nanswer", "answer"),
    ("", "r", "<think>abc", ""),
    ("", "r", "x\n</think>\n\nanswer", "answer"),
    ("", "r", "use <think> tags", "use <think> tags"),
    ("", "r", "<think>a</think>\n\n<think>b</think>\n\nc", "c"),
    ("", "r", "x</think>y</think>z", "z"),
    ("", "r", "plain text", "plain text"),
    ("reasoning_content", "r", "x</think>y", "x</think>y"),
    ("reasoning_content", "r", "<think>plain text", "<think>plain text"),
]


def _write_retention_rows(path: Path) -> None:
    """For each case, a row 31 days old (seq 2n) and one 29 days old (2n + 1).

    An ``AFTER UPDATE`` trigger counts the rows written.
    """
    with _connect(path) as connection:
        connection.executescript("""
            CREATE TABLE update_count (n INTEGER NOT NULL);
            INSERT INTO update_count VALUES (0);
            CREATE TRIGGER count_updates AFTER UPDATE ON llm_calls
            BEGIN UPDATE update_count SET n = n + 1; END;
            """)
        for index, (key, thinking, content, _) in enumerate(RETENTION_CASES):
            for offset, age in enumerate((31, 29)):
                connection.execute(
                    """INSERT INTO llm_calls (task_id, seq, kind, step, attempt,
                       outcome, content, thinking, reasoning_key, step_ids_json,
                       redactions, tool_calls_json, usage_json, duration_s,
                       timestamp)
                       VALUES ('t1', ?, 'plan', 1, 1, 'answer', ?, ?, ?, '["s1"]',
                               1, '[]', '{"total_tokens": 3}', 0.5, ?)""",
                    (2 * index + offset, content, thinking, key, NOW - age * DAY),
                )
        connection.commit()


def _update_count(path: Path) -> int:
    with _connect(path) as connection:
        return int(connection.execute("SELECT n FROM update_count").fetchone()[0])


async def _open_at(path: Path, days: float, now: float = NOW) -> None:
    history = SqliteHistory(path, thinking_retention_days=days, clock=lambda: now)
    try:
        await history.count_tasks()
    finally:
        await history.aclose()


async def test_retention_clears_old_reasoning(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    await _open_at(path, 30.0, now=0.0)
    _write_retention_rows(path)
    before = _rows(path, "llm_calls")

    await _open_at(path, 30.0)

    after = _rows(path, "llm_calls")
    assert len(after) == len(before) == 2 * len(RETENTION_CASES)
    columns = [column[1] for column in LLM_CALLS_COLUMNS]
    content_at, thinking_at = columns.index("content"), columns.index("thinking")
    for index, (_, _, _, cleared) in enumerate(RETENTION_CASES):
        old_before, new_before = before[2 * index], before[2 * index + 1]
        old_after, new_after = after[2 * index], after[2 * index + 1]
        assert (old_after[thinking_at], old_after[content_at]) == ("", cleared)
        # Every other column of the old row, and the newer row, unchanged.
        assert [
            v for i, v in enumerate(old_after) if i not in (content_at, thinking_at)
        ] == [v for i, v in enumerate(old_before) if i not in (content_at, thinking_at)]
        assert new_after == new_before
    # One write per changed row: every old row had thinking.
    assert _update_count(path) == len(RETENTION_CASES)

    # A second instance finds nothing to write.
    await _open_at(path, 30.0)
    assert _rows(path, "llm_calls") == after
    assert _update_count(path) == len(RETENTION_CASES)


async def test_a_shorter_retention_clears_more(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    await _open_at(path, 30.0, now=0.0)
    _write_retention_rows(path)

    await _open_at(path, 7.0)

    rows = _rows(path, "llm_calls")
    thinking_at = [column[1] for column in LLM_CALLS_COLUMNS].index("thinking")
    content_at = [column[1] for column in LLM_CALLS_COLUMNS].index("content")
    assert [(row[thinking_at], row[content_at]) for row in rows] == [
        ("", cleared) for (_, _, _, cleared) in RETENTION_CASES for _ in range(2)
    ]
    assert _update_count(path) == 2 * len(RETENTION_CASES)
    await _open_at(path, 7.0)
    assert _update_count(path) == 2 * len(RETENTION_CASES)


@pytest.mark.parametrize("days", [0, -1, math.nan, math.inf])
def test_an_invalid_retention_is_refused(tmp_path: Path, days: float) -> None:
    with pytest.raises(ValueError, match="thinking_retention_days"):
        SqliteHistory(tmp_path / "history.db", thinking_retention_days=days)


async def test_retention_runs_once_per_instance(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    history = SqliteHistory(path, clock=lambda: NOW)
    try:
        await history.count_tasks()
        # Rows added after the pass, with old timestamps, keep their thinking
        # across a reconnection of the same instance.
        _write_retention_rows(path)
        await history.aclose()
        await history.count_tasks()
    finally:
        await history.aclose()
    assert _update_count(path) == 0

    await _open_at(path, 30.0)
    assert _update_count(path) == len(RETENTION_CASES)


def test_thinking_retention_days_reads_the_variable() -> None:
    name = "LOCALMELO_THINKING_RETENTION_DAYS"
    assert name == THINKING_RETENTION_ENV
    assert THINKING_RETENTION_DAYS == 30.0
    assert thinking_retention_days({}) == 30.0
    assert thinking_retention_days({name: ""}) == 30.0
    assert thinking_retention_days({name: "7"}) == 7.0
    assert thinking_retention_days({name: "0.5"}) == 0.5


@pytest.mark.parametrize("value", ["abc", "0", "-3", "nan", "inf"])
def test_thinking_retention_days_refuses_a_bad_value(value: str) -> None:
    with pytest.raises(ValueError) as refused:
        thinking_retention_days({"LOCALMELO_THINKING_RETENTION_DAYS": value})
    assert str(refused.value) == (
        "LOCALMELO_THINKING_RETENTION_DAYS must be a number of days greater "
        f"than 0, not '{value}'"
    )


def test_thinking_retention_days_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert thinking_retention_days() == 30.0
    monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "12")
    assert thinking_retention_days() == 12.0
