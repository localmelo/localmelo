"""The bench's contract with run-end learning: the real CLI in a subprocess.

The bench runs ``melo`` in direct mode, JSON output, no TTY, no sink, with
persistence on. A run must then send the same requests, print the same
stdout, exit with the same code, write the same history rows and nothing
new on stderr as the same CLI with learning switched off
(``procedures_base_child.py``); the only change is ``procedures.db`` in
the memory directory. The model is a local fake that replays its replies.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from localmelo.tests.cli.test_direct_mode_process import (
    FakeChatServer,
    _env,
    _tool_call,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")

BASE_CHILD = Path(__file__).with_name("procedures_base_child.py")
QUERY = "read notes.txt, run echo, and say done"
PROCEDURE_FILES = {"procedures.db", "procedures.db-wal", "procedures.db-shm"}


@pytest.fixture
def servers() -> Iterator[list[FakeChatServer]]:
    made = [FakeChatServer(), FakeChatServer()]
    try:
        yield made
    finally:
        for server in made:
            server.close()


def _script(server: FakeChatServer) -> None:
    server.replies.extend(
        [
            _tool_call("file_read", {"path": "notes.txt"}),
            _tool_call("shell_exec", {"command": "echo hi"}),
            {"role": "assistant", "content": "done"},
        ]
    )


def _melo(
    server: FakeChatServer,
    root: Path,
    *,
    base: bool,
    output_format: str = "json",
    persist: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run melo as the bench does, in ``root/work`` with ``root/memory``."""
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "notes.txt").write_text("a note\n", encoding="utf-8")
    env = _env(root)
    if persist:
        env["LOCALMELO_PERSIST_MEMORY"] = "1"
        env["LOCALMELO_MEMORY_DIR"] = str(root / "memory")
    program = [str(BASE_CHILD)] if base else ["-m", "localmelo"]
    return subprocess.run(
        [
            sys.executable,
            "-I",
            *program,
            "--base-url",
            server.base_url,
            "--chat-model",
            "fake-model",
            "--output-format",
            output_format,
            QUERY,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=work,
        check=False,
    )


def _record(stdout: str) -> dict[str, Any]:
    assert stdout.count("\n") == 1 and stdout.endswith("\n"), stdout
    record = json.loads(stdout)
    for masked in ("task_id", "duration_s"):
        record[masked] = "<masked>"
    record["config"]["memory_dir"] = "<masked>"
    return dict(record)


def _history(memory: Path) -> dict[str, list[Any]]:
    """tasks, steps and llm_calls, without ids, times and durations."""
    with closing(sqlite3.connect(memory / "history.db")) as db:
        tasks = db.execute("SELECT query, status, result FROM tasks").fetchall()
        steps = db.execute(
            "SELECT thought, tool_call_json, tool_result_json, seq FROM steps "
            "ORDER BY rowid"
        ).fetchall()
        calls = db.execute(
            "SELECT seq, kind, step, attempt, outcome, content, thinking, "
            "reasoning_key, step_ids_json, redactions, tool_calls_json, usage_json "
            "FROM llm_calls ORDER BY rowid"
        ).fetchall()
    kept_steps = []
    for thought, call, outcome, seq in steps:
        result = json.loads(outcome)
        result.pop("duration_ms", None)
        kept_steps.append([thought, json.loads(call), result, seq])
    kept_calls = []
    for row in calls:
        row = list(row)
        ids = row[8]
        row[8] = None if ids is None else len(json.loads(ids))
        kept_calls.append(row)
    return {
        "tasks": [list(t) for t in tasks],
        "steps": kept_steps,
        "llm_calls": kept_calls,
    }


@pytest.mark.parametrize("output_format", ["json", "text"])
def test_a_bench_run_changes_nothing_but_procedures_db(
    servers: list[FakeChatServer], tmp_path: Path, output_format: str
) -> None:
    learned_server, base_server = servers
    _script(learned_server)
    _script(base_server)
    learned = _melo(
        learned_server, tmp_path / "p2", base=False, output_format=output_format
    )
    base = _melo(base_server, tmp_path / "base", base=True, output_format=output_format)

    assert (learned.returncode, base.returncode) == (0, 0), learned.stderr
    assert learned.stderr == base.stderr == ""
    if output_format == "json":
        assert _record(learned.stdout) == _record(base.stdout)
        assert _record(learned.stdout)["status"] == "completed"
    else:
        assert learned.stdout == base.stdout == "done\n"
    assert learned_server.bodies == base_server.bodies
    assert len(learned_server.bodies) == 3

    memory, base_memory = tmp_path / "p2" / "memory", tmp_path / "base" / "memory"
    assert _history(memory) == _history(base_memory)
    added = {p.name for p in memory.iterdir()} - {p.name for p in base_memory.iterdir()}
    assert "procedures.db" in added and added <= PROCEDURE_FILES
    assert not PROCEDURE_FILES & {p.name for p in base_memory.iterdir()}
    for name in added:
        assert stat.S_IMODE((memory / name).stat().st_mode) == 0o600

    with closing(sqlite3.connect(memory / "procedures.db")) as db:
        episodes = db.execute(
            "SELECT status, steps, n_tool, model, match, level FROM episodes"
        ).fetchall()
        versions = db.execute("SELECT version, status FROM templates").fetchall()
    assert episodes == [("completed", 3, 2, "fake-model", "new", 0)]
    assert versions == [(1, "active")]


def test_runs_across_processes_build_on_each_other(
    servers: list[FakeChatServer], tmp_path: Path
) -> None:
    server = servers[0]
    for _ in range(3):
        _script(server)
        done = _melo(server, tmp_path, base=False)
        assert done.returncode == 0 and done.stderr == "", done.stderr
    with closing(sqlite3.connect(tmp_path / "memory" / "procedures.db")) as db:
        episodes = db.execute("SELECT match, level FROM episodes").fetchall()
        active = db.execute(
            "SELECT version FROM templates WHERE status = 'active'"
        ).fetchall()
    # The third run's gate passes (two completed runs, same length): the
    # episode records level 2, though level 2 is not shown.
    assert episodes == [("new", 0), ("exact", 1), ("exact", 2)]
    assert active == [(3,)]


def test_without_persistence_nothing_is_written(
    servers: list[FakeChatServer], tmp_path: Path
) -> None:
    server = servers[0]
    _script(server)
    done = _melo(server, tmp_path, base=False, persist=False)
    assert done.returncode == 0 and done.stderr == "", done.stderr
    written = [
        path.name
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name.startswith("procedures")
    ]
    assert written == []
    assert sorted(os.listdir(tmp_path / "work")) == ["notes.txt"]
