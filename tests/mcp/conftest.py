"""Shared helpers for the MCP tests: a config pointing at the fake server."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.mcp import McpServerConfig, StdioMcpClient

FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")


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
