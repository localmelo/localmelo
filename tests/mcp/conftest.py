"""Shared helpers for the MCP tests.

A config pointing at the fake server, and gates that hold the code under
test at a chosen await, so a test can race it without sleeps.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.mcp import McpServerConfig, StdioMcpClient

FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")


class Gate:
    """Holds every caller of hold() until release().

    ``reached`` is set by the first caller, so a test can act while the
    code under test is held there: races without sleeps.
    """

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self._released = asyncio.Event()

    async def hold(self) -> None:
        self.reached.set()
        await self._released.wait()

    def release(self) -> None:
        self._released.set()


class SpawnGate(Gate):
    """Stands in for ``asyncio.create_subprocess_exec`` in the MCP client.

    From the ``hold_from``-th spawn on (0-based; None holds none), the
    caller is held before the process is spawned, as if the spawn were
    slow. Every process spawned is kept in ``processes``.
    """

    def __init__(
        self, spawn: Callable[..., Awaitable[asyncio.subprocess.Process]]
    ) -> None:
        super().__init__()
        self._spawn = spawn
        self._calls = 0
        self.hold_from: int | None = None
        self.processes: list[asyncio.subprocess.Process] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        call = self._calls
        self._calls += 1
        if self.hold_from is not None and call >= self.hold_from:
            await self.hold()
        proc = await self._spawn(*args, **kwargs)
        self.processes.append(proc)
        return proc


async def wait_held(gate: Gate, task: asyncio.Task[Any]) -> None:
    """Return once *gate* holds a caller; fail if *task* ends first."""
    reached = asyncio.ensure_future(gate.reached.wait())
    await asyncio.wait([reached, task], return_when=asyncio.FIRST_COMPLETED)
    if not reached.done():
        reached.cancel()
        task.result()  # re-raises the task's error, if it has one
        raise AssertionError("the task ended without reaching the gate")


def fake_server_config(
    log_path: Path,
    *,
    name: str = "fake",
    env: dict[str, str] | None = None,
    **kwargs: Any,
) -> McpServerConfig:
    return McpServerConfig(
        name=name,
        command=sys.executable,
        args=["-I", str(FAKE_SERVER)],
        env={"FAKE_MCP_LOG": str(log_path), **(env or {})},
        **kwargs,
    )


def read_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines()]


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "fake-mcp.log"


@pytest.fixture
async def client(log_path: Path) -> AsyncIterator[StdioMcpClient]:
    mcp = StdioMcpClient(fake_server_config(log_path), shutdown_grace_s=0.5)
    await mcp.start()
    try:
        yield mcp
    finally:
        await mcp.close()


@pytest.fixture
async def spawn_gate(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SpawnGate]:
    """Routes the MCP client's spawns through a SpawnGate.

    Afterwards it kills any process a failing test left running.
    """
    gate = SpawnGate(asyncio.create_subprocess_exec)
    # client.py looks the function up on the asyncio module at each spawn.
    monkeypatch.setattr(
        "localmelo.support.mcp.client.asyncio.create_subprocess_exec", gate
    )
    try:
        yield gate
    finally:
        for proc in gate.processes:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
