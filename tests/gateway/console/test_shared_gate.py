"""One gate per gateway process: the console and the workspace page share it."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import (
    Config,
    GatewayConfig,
    LocalBackendConfig,
    SchedulerConfig,
)
from localmelo.support.gateway import GatewayServer
from localmelo.support.gateway.chat_gate import ChatGate, GatedLLM
from localmelo.support.gateway.console import ConsoleOptions


class QuietProbe:
    def snapshot(self) -> None:
        return None

    async def aclose(self) -> None:
        pass


class _HeldLLM(BaseLLMProvider):
    """A provider whose calls wait until the test opens it, counting overlap."""

    model = "held"

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.in_flight -= 1
        return Message(role="assistant", content="ok")

    async def close(self) -> None:
        return None


def _cfg(*, host: str = "127.0.0.1", policy: str = "off", n_srv: int = 8) -> Config:
    return Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(
            chat_url="http://127.0.0.1:9/v1", chat_model="stub-model"
        ),
        gateway=GatewayConfig(port=8401, host=host),
        scheduler=SchedulerConfig(policy=policy, n_srv=n_srv, call_log=False),
    )


def test_the_console_uses_the_workspace_pages_gate(tmp_path: Path) -> None:
    server = GatewayServer(
        _cfg(),
        console_options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), probe=QuietProbe()
        ),
    )
    playground = server.app.state.smoke_playground

    assert server.console is not None
    assert server.console.gate is playground.gate
    # The console built no monitor of its own: the page's feeds the gate.
    assert server.console._monitor is None


def test_a_test_supplied_gate_is_kept(tmp_path: Path) -> None:
    gate = ChatGate(policy="off", call_log_dir=None)
    server = GatewayServer(
        _cfg(),
        console_options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), probe=QuietProbe(), gate=gate
        ),
    )

    assert server.console is not None
    assert server.console.gate is gate


def test_a_non_loopback_gateway_has_no_console_to_share_with() -> None:
    server = GatewayServer(_cfg(host="0.0.0.0"))
    assert server.console is None


async def test_two_surfaces_share_one_slot_budget() -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, call_log_dir=None)
    inner = _HeldLLM()
    console = GatedLLM(inner, gate.client("3f9c2a7b1d4e"))
    page = GatedLLM(inner, gate.client("workspace-chat"))
    try:
        first = asyncio.create_task(console.chat([Message(role="user", content="one")]))
        await inner.entered.wait()
        second = asyncio.create_task(page.chat([Message(role="user", content="two")]))
        await asyncio.sleep(0.05)

        # One slot, two sessions: the page's call waits for the console's.
        assert inner.max_in_flight == 1
        inner.release.set()
        await asyncio.gather(first, second)
        assert inner.calls == 2 and inner.max_in_flight == 1
    finally:
        await gate.close()


def test_the_console_still_serves_with_the_shared_gate(tmp_path: Path) -> None:
    server = GatewayServer(
        _cfg(),
        console_options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), probe=QuietProbe()
        ),
    )
    client = TestClient(server.app, base_url="http://127.0.0.1:8401")
    assert client.get("/api/health").status_code == 200
