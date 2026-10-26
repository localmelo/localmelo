"""History schema version 1 and its ``llm_calls`` table (R1).

Every database lives under ``tmp_path``. A file "written before R1" (version
0) is made with ``sqlite3`` and the unchanged ``_SCHEMA``, as an older
localmelo wrote it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
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
