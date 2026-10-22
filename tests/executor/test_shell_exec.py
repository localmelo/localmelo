"""shell_exec kills its whole process group on a timeout or a cancellation.

Each command writes the PIDs of its shell and of a background child to a
file and then waits, so the tests can check that both are gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import localmelo.melo.executor.builtins as builtin_tools
from localmelo.melo.checker import Checker
from localmelo.melo.executor import (
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo


def _command(pid_file: Path) -> str:
    return f'sleep 60 & echo $$ $! > "{pid_file}"; wait'


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
async def test_the_executor_timeout_kills_the_command_and_its_children(
    pid_file: Path,
) -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker())
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
