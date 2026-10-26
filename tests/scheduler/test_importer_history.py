"""The history importer reads tool times read-only and selects no text column.

Requirements 56-58 and the lead's decision 7: a ``history.db`` is opened
read-only, immutable when no ``-wal`` or ``-journal`` exists, so the
directory never changes; a database whose writer is still open is read
through its ``-wal`` file; NULL tool results give unknown tool time and
thought rows give zero; missing tables give an empty import with an
error; bad rows are skipped and counted; only ``duration_ms`` leaves the
tool-result JSON. Databases are built here with ``sqlite3``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.scheduler.sim import import_history, importer
from localmelo.support.scheduler.sim.importer import (
    HistoryFormatError,
    HistoryStep,
    tool_time_pool,
)

# localmelo's history schema, with timestamp nullable for the odd-row tests.
SCHEMA = """
CREATE TABLE tasks (task_id TEXT PRIMARY KEY, query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running', result TEXT NOT NULL DEFAULT '');
CREATE TABLE steps (step_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                    thought TEXT NOT NULL DEFAULT '', tool_call_json TEXT,
                    tool_result_json TEXT, timestamp REAL, seq INTEGER NOT NULL);
"""
CALL = json.dumps({"tool_name": "shell_exec", "arguments": {"cmd": "ls"}, "id": "c"})
Row = tuple[str, str, str | None, str | None, float | None, int]


def result(duration_ms: Any = 1500) -> str:
    return json.dumps(
        {"tool_name": "shell_exec", "output": "secret", "duration_ms": duration_ms}
    )


def make_db(
    path: Path,
    rows: Sequence[Row],
    tasks: Sequence[tuple[str, str]] = (),
    journal: str = "delete",
    schema: str = SCHEMA,
) -> sqlite3.Connection:
    """A history database at *path*; the caller closes the returned connection.

    Each row is ``(step_id, task_id, tool_call_json, tool_result_json,
    timestamp, seq)``; the thought is always secret text.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA journal_mode={journal}")
    conn.executescript(schema)
    if rows:
        conn.executemany(
            "INSERT INTO steps (step_id, task_id, thought, tool_call_json,"
            " tool_result_json, timestamp, seq) VALUES (?, ?, 'secret', ?, ?, ?, ?)",
            rows,
        )
    if tasks:
        conn.executemany(
            "INSERT INTO tasks (task_id, query, status) VALUES (?, 'secret', ?)", tasks
        )
    conn.commit()
    return conn


def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    return {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in directory.iterdir()
    }


ROWS: list[Row] = [
    ("s1", "t1", CALL, result(1500), 100.0, 0),
    ("s2", "t1", None, None, 101.0, 1),
    ("s3", "t2", CALL, result(250), 200.0, 0),
]


@pytest.mark.parametrize("journal", ["delete", "wal"])
def test_history_is_read_only_and_leaves_the_directory_unchanged(
    tmp_path: Path, journal: str
) -> None:
    db = tmp_path / "history.db"
    make_db(db, ROWS, [("t1", "done")], journal=journal).close()
    before = snapshot(tmp_path)
    assert sorted(before) == ["history.db"]
    tasks, report = import_history(db, label="run-1")
    assert snapshot(tmp_path) == before
    assert [t.task_key for t in tasks] == ["run-1#t1", "run-1#t2"]
    assert [t.status for t in tasks] == ["done", None]
    assert tasks[0].steps == (HistoryStep(0, 100.0, 1.5), HistoryStep(1, 101.0, 0.0))
    assert (report.files, report.lines, report.records_kept, report.tasks) == (
        1,
        3,
        3,
        2,
    )
    assert report.errors == () and not report.has_llm_calls


def test_a_database_with_a_wal_file_is_read_without_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "history.db"
    writer = make_db(db, ROWS, journal="wal")  # left open, as a running playground
    try:
        names = sorted(snapshot(tmp_path))
        assert names == ["history.db", "history.db-shm", "history.db-wal"]
        main_only = sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True)
        assert main_only.execute("SELECT name FROM sqlite_master").fetchall() == []
        main_only.close()  # every row is still in the -wal file only
        tasks, report = import_history(db)
        assert sorted(snapshot(tmp_path)) == names
        assert [t.task_key for t in tasks] == ["history.db#t1", "history.db#t2"]
        assert report.records_kept == 3
    finally:
        writer.close()

    # A writer that appears during an immutable read: read again through it.
    real: Callable[..., Any] = importer._read
    modes: list[bool] = []
    late: list[sqlite3.Connection] = []

    def read(path: Path, immutable: bool) -> Any:
        out = real(path, immutable)
        modes.append(immutable)
        if not late:
            late.append(sqlite3.connect(path))
            late[0].execute("INSERT INTO steps VALUES ('s4','t3','',NULL,NULL,300.0,0)")
            late[0].commit()
        return out

    monkeypatch.setattr(importer, "_read", read)
    try:
        tasks, _ = import_history(db)
    finally:
        late[0].close()
    assert modes == [True, False]
    assert [t.task_key for t in tasks][-1] == "history.db#t3"


def test_null_tool_results_give_unknown_tool_time_and_thought_rows_give_zero(
    tmp_path: Path,
) -> None:
    rows: list[Row] = [
        ("a", "t", None, None, 1.0, 0),  # a thought: 0.0, though its result is NULL
        ("b", "t", CALL, None, 2.0, 1),
        ("c", "t", CALL, result(2500), 3.0, 2),
        ("d", "t", CALL, result(-5), 4.0, 3),
        ("e", "t", CALL, result("12"), 5.0, 4),
        ("f", "t", CALL, json.dumps({"tool_name": "x"}), 6.0, 5),
        ("g", "t", CALL, '{"duration_ms": 1e400}', 7.0, 6),
        ("h", "t", CALL, result(0), 8.0, 7),
        ("i", "t", CALL, result(True), 9.0, 8),  # json_extract gives 1 (decision 5)
    ]
    db = tmp_path / "h.db"
    make_db(db, rows).close()
    (t,) = import_history(db)[0]
    assert [s.tool_s for s in t.steps] == [
        0.0,
        None,
        2.5,
        None,
        None,
        None,
        None,
        0.0,
        0.001,
    ]
    assert tool_time_pool([t]) == (0.001, 2.5)


def test_a_missing_steps_or_tasks_table_gives_an_empty_import_with_an_error_entry(
    tmp_path: Path,
) -> None:
    only_tasks = tmp_path / "a.db"
    conn = sqlite3.connect(only_tasks)
    conn.execute("CREATE TABLE tasks (task_id TEXT, status TEXT)")
    conn.execute("CREATE TABLE llm_calls (task_id TEXT, seq INTEGER)")
    conn.close()
    tasks, report = import_history(only_tasks)
    missing = "task_id, step_id, seq, timestamp, tool_call_json, tool_result_json"
    assert tasks == ()
    assert report.errors == (
        f"{only_tasks}: no usable steps table (missing: {missing})",
    )
    assert report.has_llm_calls and report.files_rejected == 1
    with pytest.raises(HistoryFormatError, match="no usable steps table"):
        import_history(only_tasks, strict=True)

    no_column = tmp_path / "b.db"
    narrow = SCHEMA.replace("tool_result_json TEXT,", "")
    make_db(no_column, [], schema=narrow).close()
    assert (
        import_history(no_column)[1].errors[0].endswith("(missing: tool_result_json)")
    )

    no_tasks = tmp_path / "c.db"
    make_db(no_tasks, ROWS[:1], schema=SCHEMA.split(";")[1] + ";").close()
    (t,) = import_history(no_tasks)[0]
    assert t.status is None and t.steps[0].tool_s == 1.5

    text = tmp_path / "d.db"
    text.write_text("not a database")
    with pytest.raises(HistoryFormatError, match="not a SQLite database"):
        import_history(text)


def test_malformed_json_and_null_timestamps_are_skipped_and_counted(
    tmp_path: Path,
) -> None:
    rows: list[Row] = [
        ("a", "t", CALL, "{broken", 1.0, 0),
        ("b", "t", CALL, result(), None, 1),
        ("c", "t", CALL, "{broken", None, 2),  # bad JSON is checked first
        ("d", "t", "{broken call", result(), 4.0, 3),  # the call JSON is not read
        ("e", "u", None, None, None, 0),
    ]
    db = tmp_path / "h.db"
    make_db(db, rows).close()
    tasks, report = import_history(db)
    assert [(t.task_key, len(t.steps)) for t in tasks] == [("h.db#t", 1), ("h.db#u", 0)]
    assert dict(report.dropped) == {"bad_json": 2, "null_timestamp": 2}
    assert report.lines == report.records_kept + sum(n for _, n in report.dropped)


def test_the_importer_selects_no_text_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "h.db"
    make_db(db, ROWS, [("t1", "done")]).close()
    statements: list[str] = []
    real = sqlite3.connect

    def connect(*args: Any, **kw: Any) -> sqlite3.Connection:
        conn: sqlite3.Connection = real(*args, **kw)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(importer.sqlite3, "connect", connect)
    import_history(db)
    assert any("FROM steps" in s for s in statements)
    assert any("FROM tasks" in s for s in statements)
    for sql in statements:
        words = re.findall(r"[A-Za-z_]+", sql)
        assert not {"query", "result", "thought"} & set(words), sql
        for i, word in enumerate(words):
            if word in ("tool_call_json", "tool_result_json"):
                inside = i > 0 and words[i - 1] in ("json_valid", "json_extract")
                tests_null = words[i + 1 : i + 3] == ["IS", "NULL"] or words[
                    i + 1 : i + 4
                ] == ["IS", "NOT", "NULL"]
                assert inside or tests_null, sql
