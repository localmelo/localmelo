"""The gateway tells ``on_ready`` once it accepts connections, and not before."""

from __future__ import annotations

import asyncio
import socket
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from localmelo.support.gateway import _ConsoleServer  # noqa: E402


async def test_on_ready_runs_once_the_server_accepts_connections() -> None:
    connected = threading.Event()

    def ready() -> None:
        port = server.servers[0].sockets[0].getsockname()[1]
        socket.create_connection(("127.0.0.1", port), timeout=5).close()
        connected.set()

    config = uvicorn.Config(FastAPI(), host="127.0.0.1", port=0, log_level="warning")
    server = _ConsoleServer(config, console=None, on_ready=ready)
    task = asyncio.create_task(server.serve())
    try:
        assert await asyncio.to_thread(connected.wait, 10)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)
