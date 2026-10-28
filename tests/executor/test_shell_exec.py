"""shell_exec kills its whole process group on a timeout or a cancellation.

Each command writes the PIDs of its shell and of a background child to a
file and then waits, so the tests can check that both are gone, and that
no process of the shell's group is left. The kill is sent before the
clean-up's first await, so a second cancellation cannot prevent it. The
python_exec of a workspace root runs its child Python in a group of its
own too, killed the same way.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

import localmelo.melo.executor.builtins as builtin_tools
from localmelo.melo.checker import Checker
from localmelo.melo.executor import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo


def _command(pid_file: Path) -> str:
    return f'sleep 60 & echo $$ $! > "{pid_file}"; wait'


def _python_code(pid_file: Path) -> str:
    """Like :func:`_command`, for python_exec: its PID and a child's, then wait."""
    return (
        "import os, subprocess, time\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pid_file)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}\\n')\n"
        "time.sleep(60)\n"
    )


def _rooted_executor(root: Path) -> Executor:
    hippo = Hippo()
    executor = Executor(hippo, Checker(), workspace_root=str(root))
    register_builtins(executor, hippo, names={"python_exec"}, workspace_root=str(root))
    return executor


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _pids(pid_file: Path) -> tuple[int, int]:
    """The shell's and the background child's PIDs, once both are written."""
    for _ in range(500):
        text = pid_file.read_text() if pid_file.exists() else ""
        if text.endswith("\n"):
            shell, child = (int(pid) for pid in text.split())
            return shell, child
        await asyncio.sleep(0.01)
    raise AssertionError("the command did not write its PIDs")


async def _gone(pid: int, timeout_s: float = 5.0) -> bool:
    """Whether *pid* exits (and is reaped) within *timeout_s*."""
    deadline = time.monotonic() + timeout_s
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.02)
    return True


@pytest.fixture
def pid_file(tmp_path: Path) -> Iterator[Path]:
    """Where a command writes its PIDs; leftovers are killed afterwards."""
    path = tmp_path / "pids"
    yield path
    if path.exists():
        for pid in path.read_text().split():
            with contextlib.suppress(OSError):
                os.kill(int(pid), signal.SIGKILL)


@pytest.mark.asyncio
async def test_a_command_still_returns_its_output() -> None:
    assert await builtin_tools._shell_exec("echo out; echo err >&2") == "out\n\nerr"


@pytest.mark.asyncio
async def test_a_cancellation_kills_the_command_and_its_children(
    pid_file: Path,
) -> None:
    running = asyncio.create_task(builtin_tools._shell_exec(_command(pid_file)))
    shell, child = await _pids(pid_file)
    assert _alive(shell) and _alive(child)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert await _gone(shell)
    assert await _gone(child)


@pytest.mark.asyncio
async def test_its_own_timeout_kills_the_command_and_its_children(
    pid_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtin_tools, "SHELL_TIMEOUT_S", 1.0)
    with pytest.raises(TimeoutError):
        await builtin_tools._shell_exec(_command(pid_file))
    shell, child = await _pids(pid_file)
    assert await _gone(shell)
    assert await _gone(child)


@pytest.mark.asyncio
async def test_its_own_timeout_is_reported_with_its_limit(
    pid_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The command's own limit is shorter than the executor's 60 s, and the
    # error names it: before, the executor reported a TimeoutError raised
    # inside the tool as "Execution timed out after 60.0s".
    monkeypatch.setattr(builtin_tools, "SHELL_TIMEOUT_S", 0.5)
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="python", builtins={"shell_exec"}))
    register_builtins(executor, hippo, names={"shell_exec"})
    outcome = await executor.execute_structured(
        ExecutionRequest(
            tool_name="shell_exec", arguments={"command": _command(pid_file)}
        )
    )
    assert outcome.status == ExecutionStatus.TIMEOUT
    assert outcome.error_category == ErrorCategory.TIMEOUT
    assert outcome.error == "Command timed out after 0.5 s"
    shell, child = await _pids(pid_file)
    assert await _gone(shell)
    assert await _gone(child)


@pytest.mark.asyncio
async def test_the_executor_timeout_kills_the_command_and_its_children(
    pid_file: Path,
) -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="python", builtins={"shell_exec"}))
    register_builtins(executor, hippo, names={"shell_exec"})
    outcome = await executor.execute_structured(
        ExecutionRequest(
            tool_name="shell_exec",
            arguments={"command": _command(pid_file)},
            timeout_ms=1000,
        )
    )
    assert outcome.status == ExecutionStatus.TIMEOUT
    shell, child = await _pids(pid_file)
    assert await _gone(shell)
    assert await _gone(child)


@pytest.mark.asyncio
async def test_the_command_runs_in_its_own_process_group() -> None:
    out = await builtin_tools._shell_exec("ps -o pgid= -p $$; echo $$")
    group, shell = (int(value) for value in out.split())
    assert group == shell != os.getpgrp()


# ── The whole group, and the order of the kill ──────────────────────────────


async def _group_gone(pgid: int, timeout_s: float = 5.0) -> bool:
    """Whether no process of group *pgid* is left within *timeout_s*.

    ``os.killpg(pgid, 0)`` raises ``ProcessLookupError`` once the group is
    empty. ``PermissionError`` means "not gone yet", as in
    ``StdioMcpClient._wait_stopped``: macOS raises it for zombies that are
    not reaped yet.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_no_process_of_the_group_survives_a_cancellation(
    pid_file: Path,
) -> None:
    running = asyncio.create_task(builtin_tools._shell_exec(_command(pid_file)))
    shell, _ = await _pids(pid_file)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    # The shell leads the group (start_new_session).
    assert await _group_gone(shell)


@pytest.mark.asyncio
async def test_no_process_of_the_group_survives_its_own_timeout(
    pid_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtin_tools, "SHELL_TIMEOUT_S", 0.5)
    with pytest.raises(TimeoutError):
        await builtin_tools._shell_exec(_command(pid_file))
    shell, _ = await _pids(pid_file)
    assert await _group_gone(shell)


@pytest.mark.asyncio
async def test_no_process_of_the_group_survives_the_executor_timeout(
    pid_file: Path,
) -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="python", builtins={"shell_exec"}))
    register_builtins(executor, hippo, names={"shell_exec"})
    outcome = await executor.execute_structured(
        ExecutionRequest(
            tool_name="shell_exec",
            arguments={"command": _command(pid_file)},
            timeout_ms=500,
        )
    )
    assert outcome.status == ExecutionStatus.TIMEOUT
    shell, _ = await _pids(pid_file)
    assert await _group_gone(shell)


@pytest.mark.asyncio
async def test_kill_process_group_signals_before_its_first_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A second cancellation, at the clean-up's first await, cannot keep the
    # group from being killed: the signal has been sent by then.
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        builtin_tools.os, "killpg", lambda pgid, sig: signals.append((pgid, sig))
    )

    class Unreaped:
        """A process that is never reaped."""

        pid = 4242

        async def wait(self) -> int:
            await asyncio.Event().wait()
            raise AssertionError("not reached")

    process = cast(asyncio.subprocess.Process, Unreaped())
    killing = asyncio.create_task(builtin_tools._kill_process_group(process))
    await asyncio.sleep(0)  # the task runs up to its first await
    assert signals == [(4242, signal.SIGKILL)]
    killing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await killing


# ── python_exec with a workspace root ───────────────────────────────────────


@pytest.mark.asyncio
async def test_no_process_of_the_rooted_python_group_survives_a_cancellation(
    pid_file: Path, tmp_path: Path
) -> None:
    executor = _rooted_executor(tmp_path)
    running = asyncio.create_task(
        executor.execute_structured(
            ExecutionRequest(
                tool_name="python_exec", arguments={"code": _python_code(pid_file)}
            )
        )
    )
    python, child = await _pids(pid_file)
    assert python != os.getpid()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert await _group_gone(python)
    assert await _gone(child)


@pytest.mark.asyncio
async def test_no_process_of_the_rooted_python_group_survives_a_timeout(
    pid_file: Path, tmp_path: Path
) -> None:
    executor = _rooted_executor(tmp_path)
    outcome = await executor.execute_structured(
        ExecutionRequest(
            tool_name="python_exec",
            arguments={"code": _python_code(pid_file)},
            # Long enough for the child to start and write its PIDs.
            timeout_ms=2000,
        )
    )
    assert outcome.status == ExecutionStatus.TIMEOUT
    python, child = await _pids(pid_file)
    assert await _group_gone(python)
    assert await _gone(child)
