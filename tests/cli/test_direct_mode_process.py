"""The real CLI in a subprocess: SIGTERM and SIGHUP, and what reaches stderr.

Each test starts Python processes, so these take seconds: the signal
tests wait for melo to stop a stubborn MCP server (about 4 s each).
Every process a test starts is killed should the test fail.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from localmelo.__main__ import SIGNAL_CLEANUP_S

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX signals and process groups"
)

FAKE_MCP_SERVER = Path(__file__).resolve().parents[1] / "mcp" / "fake_mcp_server.py"
CHILD = Path(__file__).with_name("direct_mode_child.py")
PROXY_VARS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}

# Starts a child in the shell's process group, records the shell's pid
# (which is also the group's id) and the child's, then waits for it.
SHELL_COMMAND = 'sleep 60 & echo $$ $! > "{pids}"; wait'


class FakeChatServer:
    """An OpenAI-compatible ``/chat/completions`` that replays queued replies.

    Once ``replies`` is empty, every reply is the final answer ``done``.
    ``bodies`` keeps the body of every request, in order.
    """

    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.requests = 0
        self.bodies: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.bodies.append(json.loads(body))
                server.requests += 1
                message = (
                    server.replies.pop(0)
                    if server.replies
                    else {"role": "assistant", "content": "done"}
                )
                body = json.dumps(
                    {"choices": [{"index": 0, "message": message}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # no request log in the test output

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def chat_server() -> Iterator[FakeChatServer]:
    server = FakeChatServer()
    try:
        yield server
    finally:
        server.close()


def _env(home: Path) -> dict[str, str]:
    """This environment without localmelo's settings and proxies, HOME at *home*.

    So nothing reaches the real ~/.cache/localmelo or leaves this machine.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LOCALMELO_") and key.lower() not in PROXY_VARS
    }
    env["HOME"] = str(home)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    return env


def _wait_until(check: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not check():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # also succeeds for a zombie not yet reaped
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False  # macOS: only zombies left, not yet reaped
    return False


def _kill_leftovers(melo: subprocess.Popen[bytes], pids: list[int]) -> None:
    """SIGKILL melo's group and every pid in *pids* that is still running."""
    if melo.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(melo.pid, signal.SIGKILL)
        melo.communicate(timeout=10)
    for pid in pids:
        if not _process_gone(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


def _wait_for_shell(
    melo: subprocess.Popen[bytes], pids_file: Path, timeout_s: float = 30.0
) -> tuple[int, int]:
    """The shell's and its child's pid, once the shell command runs."""
    deadline = time.monotonic() + timeout_s
    while True:
        text = pids_file.read_text() if pids_file.exists() else ""
        if text.endswith("\n"):
            shell, child = map(int, text.split())
            return shell, child
        if melo.poll() is not None:
            out, err = melo.communicate()
            pytest.fail(
                f"melo exited with {melo.returncode} before the shell command "
                f"ran\nstdout: {out!r}\nstderr: {err.decode(errors='replace')}"
            )
        if time.monotonic() >= deadline:
            pytest.fail(f"the shell command did not start within {timeout_s:g} s")
        time.sleep(0.05)


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


# ── SIGTERM and SIGHUP ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sig", "output_format"),
    [(signal.SIGTERM, "json"), (signal.SIGHUP, "text")],
    ids=["SIGTERM-json", "SIGHUP-text"],
)
def test_a_signal_stops_the_run_and_everything_it_started(
    chat_server: FakeChatServer,
    tmp_path: Path,
    sig: signal.Signals,
    output_format: str,
) -> None:
    """What the benchmark harness does on its timeout: signal melo's group.

    The shell command and the MCP server run in sessions of their own, so
    only melo can stop them. It must do so, save the task record and exit
    by the signal within SIGNAL_CLEANUP_S, printing no result.
    """
    pids_file = tmp_path / "shell.pids"
    server_pid_file = tmp_path / "mcp.pid"
    memory_dir = tmp_path / "memory"
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {
                        "command": sys.executable,
                        "args": ["-I", str(FAKE_MCP_SERVER)],
                        # Ignores SIGTERM and outlives its stdin.
                        "env": {
                            "FAKE_MCP_MODE": "stubborn",
                            "FAKE_MCP_PIDFILE": str(server_pid_file),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    command = SHELL_COMMAND.format(pids=pids_file)
    chat_server.replies.append(_tool_call("shell_exec", {"command": command}))

    melo = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-m",
            "localmelo",
            "--base-url",
            chat_server.base_url,
            "--mcp-config",
            str(mcp_config),
            "--memory-dir",
            str(memory_dir),
            "--tool-exposure",
            "all",
            "--output-format",
            output_format,
            "run the command",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # as the benchmark harness starts it
        env=_env(tmp_path),
        cwd=tmp_path,
    )
    started: list[int] = []
    try:
        shell, child = _wait_for_shell(melo, pids_file)
        server = int(server_pid_file.read_text())
        started += [shell, child, server]

        sent = time.monotonic()
        os.killpg(melo.pid, sig)
        out, err = melo.communicate(timeout=30)
        elapsed = time.monotonic() - sent
        stderr = err.decode(errors="replace")

        assert melo.returncode == -sig, stderr
        assert elapsed < SIGNAL_CLEANUP_S, f"exited {elapsed:.1f} s after {sig.name}"
        assert out == b""  # no result and no JSON line
        assert "Traceback" not in stderr, stderr
        assert chat_server.requests == 1  # cut short in its first step

        for pid in (shell, child, server):
            assert _wait_until(functools.partial(_process_gone, pid)), f"{pid} runs"
        assert _wait_until(lambda: _group_gone(shell)), "the shell's group remains"

        with contextlib.closing(sqlite3.connect(memory_dir / "history.db")) as db:
            rows = db.execute("SELECT status, result FROM tasks").fetchall()
        assert rows == [("crashed", f"Cancelled by {sig.name}")]
    finally:
        _kill_leftovers(melo, started)


def _read_methods(log: Path) -> list[str]:
    """The methods of the messages a fake MCP server has logged so far."""
    if not log.exists():
        return []
    # Only whole lines: the server may be writing the last one.
    lines = log.read_text().split("\n")[:-1]
    return [json.loads(line).get("method", "") for line in lines]


def test_a_signal_while_an_mcp_server_starts_keeps_to_the_limit(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    """SIGTERM while the second of two stubborn servers is starting.

    Its start-up then stops both servers itself, each with two grace
    periods of 2 s, about 8 s in all: melo must stop waiting in time, kill
    them, and still save the task record.
    """
    memory_dir = tmp_path / "memory"
    pid_files = {name: tmp_path / f"{name}.pid" for name in "ab"}
    b_log = tmp_path / "b.log"
    modes = {"a": "stubborn", "b": "stubborn_hang_initialize"}
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    name: {
                        "command": sys.executable,
                        "args": ["-I", str(FAKE_MCP_SERVER)],
                        "env": {
                            "FAKE_MCP_MODE": mode,
                            "FAKE_MCP_PIDFILE": str(pid_files[name]),
                            "FAKE_MCP_LOG": str(tmp_path / f"{name}.log"),
                        },
                    }
                    for name, mode in modes.items()
                }
            }
        ),
        encoding="utf-8",
    )
    melo = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-m",
            "localmelo",
            "--base-url",
            chat_server.base_url,
            "--mcp-config",
            str(mcp_config),
            "--memory-dir",
            str(memory_dir),
            "--output-format",
            "json",
            "use the servers",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=_env(tmp_path),
        cwd=tmp_path,
    )
    started: list[int] = []
    try:
        assert _wait_until(
            lambda: "initialize" in _read_methods(b_log), 30
        ), "server b got no initialize"
        started += [int(pid_files[name].read_text()) for name in "ab"]

        sent = time.monotonic()
        os.killpg(melo.pid, signal.SIGTERM)
        out, err = melo.communicate(timeout=30)
        elapsed = time.monotonic() - sent
        stderr = err.decode(errors="replace")

        assert melo.returncode == -signal.SIGTERM, stderr
        assert elapsed < SIGNAL_CLEANUP_S, f"exited {elapsed:.1f} s after SIGTERM"
        assert out == b""
        assert "Traceback" not in stderr, stderr
        assert chat_server.requests == 0  # the run never got to plan

        for pid in started:
            assert _wait_until(functools.partial(_process_gone, pid)), f"{pid} runs"
        with contextlib.closing(sqlite3.connect(memory_dir / "history.db")) as db:
            rows = db.execute("SELECT status, result FROM tasks").fetchall()
        assert rows == [("crashed", "Cancelled by SIGTERM")]
    finally:
        _kill_leftovers(melo, started)


# ── Tracebacks only for crashed runs ────────────────────────────────────────


def _run_child(
    tmp_path: Path, scenario: str, output_format: str
) -> subprocess.CompletedProcess[str]:
    """Run direct_mode_child.py, which calls main() with *scenario*'s failure."""
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        json.dumps({"mcpServers": {"stuck": {"command": "never-started"}}}),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, "-I", str(CHILD), scenario, output_format, str(mcp_config)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=_env(tmp_path),
        cwd=tmp_path,
        check=False,
    )


def _json_record(out: str) -> dict[str, Any]:
    assert out.count("\n") == 1 and out.endswith("\n"), out
    record = json.loads(out)
    assert isinstance(record, dict)
    assert list(record)[-1] == "localmelo_version"
    return record


@pytest.mark.parametrize("output_format", ["text", "json"])
@pytest.mark.parametrize(
    ("scenario", "warning"),
    [
        (
            "close_failure",
            r"Could not close tool provider FailingCloseProvider: "
            r"RuntimeError: close failed",
        ),
        (
            "save_failure",
            r"Could not save the final state of task [0-9a-f]{32}: "
            r"OperationalError: database is locked",
        ),
        (
            "unstoppable_mcp",
            r"Cannot stop MCP server 'stuck': process 4242 is still running "
            r"after SIGKILL",
        ),
    ],
)
def test_no_traceback_for_a_run_that_did_not_crash(
    tmp_path: Path, scenario: str, warning: str, output_format: str
) -> None:
    done = _run_child(tmp_path, scenario, output_format)
    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr, done.stderr
    # The failure is reported in one line of its own.
    assert [
        line for line in done.stderr.splitlines() if re.fullmatch(warning, line)
    ], done.stderr
    if output_format == "json":
        assert _json_record(done.stdout)["status"] == "completed"
    else:
        assert done.stdout == "done\n"


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_a_crashed_run_prints_its_traceback(tmp_path: Path, output_format: str) -> None:
    done = _run_child(tmp_path, "crashed", output_format)
    assert done.returncode == 1, done.stderr
    assert "Traceback (most recent call last)" in done.stderr
    assert "RuntimeError: the model server exploded" in done.stderr
    if output_format == "json":
        record = _json_record(done.stdout)
        assert (record["status"], record["error"]) == (
            "crashed",
            "RuntimeError: the model server exploded",
        )
    else:
        assert done.stdout == ""
        assert done.stderr.splitlines()[-1] == (
            "melo: run ended with status crashed: "
            "RuntimeError: the model server exploded"
        )


# ── python_exec ─────────────────────────────────────────────────────────────


def _run_melo(
    chat_server: FakeChatServer, tmp_path: Path, output_format: str
) -> subprocess.CompletedProcess[str]:
    """Run the real CLI on one query with every built-in tool offered."""
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "localmelo",
            "--base-url",
            chat_server.base_url,
            "--tool-exposure",
            "all",
            "--output-format",
            output_format,
            "run the code",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=_env(tmp_path),
        cwd=tmp_path,
        check=False,
    )


def _tool_messages(body: dict[str, Any]) -> list[str]:
    return [m["content"] for m in body["messages"] if m["role"] == "tool"]


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_sys_exit_in_python_exec_fails_only_the_call(
    chat_server: FakeChatServer, tmp_path: Path, output_format: str
) -> None:
    """Code that calls sys.exit(0) must not end melo, unreported, with 0."""
    code = "import sys\nprint('hi')\nsys.exit(0)"
    chat_server.replies.append(_tool_call("python_exec", {"code": code}))
    done = _run_melo(chat_server, tmp_path, output_format)

    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr, done.stderr
    if output_format == "json":
        record = _json_record(done.stdout)
        assert (record["status"], record["n_tool"]) == ("completed", 1)
    else:
        assert done.stdout == "done\n"
    # The model was told, and answered.
    assert chat_server.requests == 2
    assert _tool_messages(chat_server.bodies[1]) == [
        "Error: Code called sys.exit(0), which python_exec does not allow"
    ]


# Code that writes to fd 1 past python_exec's own capture of sys.stdout.
LEAKING_CODE = """\
import os, subprocess, sys
os.system("echo LEAK_FROM_OS_SYSTEM")
subprocess.run(["echo", "LEAK_FROM_SUBPROCESS"])
os.write(1, b"LEAK_FROM_OS_WRITE\\n")
sys.__stdout__.write("LEAK_FROM_DUNDER_STDOUT\\n")
print("printed")
"""
LEAKS = [
    "LEAK_FROM_OS_SYSTEM",
    "LEAK_FROM_SUBPROCESS",
    "LEAK_FROM_OS_WRITE",
    "LEAK_FROM_DUNDER_STDOUT",
]


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_what_tools_write_to_stdout_goes_to_stderr(
    chat_server: FakeChatServer, tmp_path: Path, output_format: str
) -> None:
    """Stdout carries the result or the one JSON line, and nothing else."""
    chat_server.replies.append(_tool_call("python_exec", {"code": LEAKING_CODE}))
    done = _run_melo(chat_server, tmp_path, output_format)

    assert done.returncode == 0, done.stderr
    if output_format == "json":
        assert _json_record(done.stdout)["status"] == "completed"
    else:
        assert done.stdout == "done\n"
    for leak in LEAKS:
        assert leak in done.stderr.splitlines(), done.stderr
    # What the code printed is still the call's output.
    assert _tool_messages(chat_server.bodies[1]) == ["printed"]
