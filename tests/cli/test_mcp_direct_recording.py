"""MCP tool calls in direct mode are recorded like built-ins.

The benchmark's memory suite runs ``melo --mcp-only --mcp-config FILE --
"<task>"`` with ``LOCALMELO_PERSIST_MEMORY=1`` and ``LOCALMELO_MEMORY_DIR``,
then reads ``history.db`` and the JSON line. This test runs that command
against the stub model server and shows that two real MCP calls land in
``steps`` with the columns a ``shell_exec`` call gets, that the task row is
saved completed, and that the JSON line counts both calls.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.tests.fixtures.stub_llm import StubServer

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX subprocesses and shell commands"
)

FAKE_MCP_SERVER = Path(__file__).resolve().parents[1] / "mcp" / "fake_mcp_server.py"
PROXY_VARS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}

MCP_TASK = "call the MCP echo tool twice"
BUILTIN_TASK = "run echo twice"

SCENARIOS: dict[str, dict[str, Any]] = {
    "mcp_twice": {
        "models": ["stub-model"],
        "match": [MCP_TASK],
        "replies": [
            {
                "tool_calls": [
                    {"name": "mcp__fake__echo", "arguments": {"text": "alpha"}}
                ]
            },
            {
                "tool_calls": [
                    {"name": "mcp__fake__echo", "arguments": {"text": "beta"}}
                ]
            },
            {"content": "Called the MCP echo tool twice."},
        ],
    },
    "builtin_twice": {
        "models": ["stub-model"],
        "match": [BUILTIN_TASK],
        "replies": [
            {
                "tool_calls": [
                    {"name": "shell_exec", "arguments": {"command": "echo alpha"}}
                ]
            },
            {
                "tool_calls": [
                    {"name": "shell_exec", "arguments": {"command": "echo beta"}}
                ]
            },
            {"content": "Ran echo twice."},
        ],
    },
}


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer(SCENARIOS).start()
    try:
        yield server
    finally:
        server.close()


def _env(home: Path, memory_dir: Path) -> dict[str, str]:
    """This environment with a temp HOME and SQLite memory, as the bench sets it."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LOCALMELO_") and key.lower() not in PROXY_VARS
    }
    env.update(
        HOME=str(home),
        NO_PROXY="127.0.0.1,localhost",
        LOCALMELO_PERSIST_MEMORY="1",
        LOCALMELO_MEMORY_DIR=str(memory_dir),
    )
    return env


def _run(
    tmp_path: Path, stub: StubServer, task: str, *, mcp: bool
) -> tuple[dict[str, Any], Path]:
    """Run melo once as the bench does; return the JSON line and history.db."""
    home = tmp_path / ("mcp-home" if mcp else "builtin-home")
    home.mkdir()
    memory_dir = tmp_path / ("mcp-memory" if mcp else "builtin-memory")
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {
                        "command": sys.executable,
                        "args": ["-I", str(FAKE_MCP_SERVER)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    argv = [
        sys.executable,
        "-I",
        "-m",
        "localmelo",
        "--base-url",
        stub.base_url,
        "--output-format",
        "json",
    ]
    if mcp:
        argv += ["--mcp-only", "--mcp-config", str(mcp_config)]
    argv += ["--", task]
    done = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=_env(home, memory_dir),
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("\n") == 1, done.stdout
    return json.loads(done.stdout), memory_dir / "history.db"


def _steps(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The task's steps in order, as (tool call, tool result) JSON objects."""
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT tool_call_json, tool_result_json FROM steps ORDER BY seq"
        ).fetchall()
    return [(json.loads(call), json.loads(result)) for call, result in rows]


def test_an_mcp_tool_call_is_recorded_like_a_builtin(
    stub: StubServer, tmp_path: Path
) -> None:
    mcp_record, mcp_db = _run(tmp_path, stub, MCP_TASK, mcp=True)
    builtin_record, builtin_db = _run(tmp_path, stub, BUILTIN_TASK, mcp=False)

    # The JSON line counts the two calls in either mode: two tool steps and
    # the answer's planning call.
    assert mcp_record["status"] == "completed", mcp_record
    assert mcp_record["result"] == "Called the MCP echo tool twice."
    assert mcp_record["steps"] == 3
    assert mcp_record["n_tool"] == 2
    assert builtin_record["status"] == "completed", builtin_record
    assert builtin_record["steps"] == 3
    assert builtin_record["n_tool"] == 2

    # The task row is saved and completed, as in the built-in's run.
    for path, answer in (
        (mcp_db, "Called the MCP echo tool twice."),
        (builtin_db, "Ran echo twice."),
    ):
        with sqlite3.connect(path) as db:
            tasks = db.execute("SELECT status, result FROM tasks").fetchall()
        assert tasks == [("completed", answer)]

    # Each MCP call is one steps row, with both JSON columns filled in.
    mcp_steps = _steps(mcp_db)
    builtin_steps = _steps(builtin_db)
    assert len(mcp_steps) == len(builtin_steps) == 2
    assert [call["tool_name"] for call, _ in mcp_steps] == ["mcp__fake__echo"] * 2
    assert [call["arguments"] for call, _ in mcp_steps] == [
        {"text": "alpha"},
        {"text": "beta"},
    ]
    assert [result["tool_name"] for _, result in mcp_steps] == ["mcp__fake__echo"] * 2
    assert [result["output"] for _, result in mcp_steps] == ["alpha", "beta"]
    assert [result["error"] for _, result in mcp_steps] == ["", ""]

    # And the rows have exactly the shape a built-in's rows have.
    for (mcp_call, mcp_result), (builtin_call, builtin_result) in zip(
        mcp_steps, builtin_steps, strict=True
    ):
        assert set(mcp_call) == set(builtin_call) == {"tool_name", "arguments", "id"}
        assert (
            set(mcp_result)
            == set(builtin_result)
            == {
                "tool_name",
                "output",
                "error",
                "duration_ms",
            }
        )
        assert mcp_result["error"] == builtin_result["error"] == ""
        assert isinstance(mcp_result["duration_ms"], (int, float))
