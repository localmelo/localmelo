"""One gate per gateway process: the console and the workspace page share it."""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

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
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.sessions.helpers import started_agent

from .conftest import make_cfg


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
    # The registry that builds the Agent's providers got the same gate, so a
    # console turn's model call lands in the page's one budget.
    assert server.console.sessions._gate is playground.gate
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


async def test_a_shared_gate_stays_open_when_the_console_closes(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="off", call_log_dir=None)
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), probe=QuietProbe(), gate=gate
        ),
    )
    await console.start()

    await console.aclose()

    # The gateway closes the one shared gate, after both surfaces' sessions;
    # the console leaves a gate it did not build alone.
    assert gate.closed is False
    await gate.close()


async def test_a_gate_the_console_built_is_its_own_to_close(
    tmp_path: Path,
) -> None:
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), probe=QuietProbe()
        ),
    )
    await console.start()

    await console.aclose()

    assert console.gate.closed is True


async def test_a_console_turn_goes_through_the_given_gate(tmp_path: Path) -> None:
    gate = ChatGate(policy="fcfs", n_srv=1, call_log_dir=None)
    inner = _HeldLLM()

    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        return await started_agent(
            Path(workspace), GatedLLM(inner, gate.client(session_id))
        )

    workspace = tmp_path / "ws"
    workspace.mkdir()
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=QuietProbe(),
            gate=gate,
        ),
    )
    await console.start()
    assert console.sessions._gate is gate
    page = asyncio.create_task(
        GatedLLM(inner, gate.client("workspace-chat")).chat(
            [Message(role="user", content="page")]
        )
    )
    try:
        await asyncio.wait_for(inner.entered.wait(), 5)
        entry, turn = await console.sessions.create(str(workspace), ("Hi", "Hi"))
        assert turn is not None
        await asyncio.sleep(0.05)
        # The page holds the one slot, so the turn's call waits for it.
        assert inner.calls == 1 and inner.max_in_flight == 1
        inner.release.set()
        deadline = time.monotonic() + 5
        while True:
            turns = await console.store.list_turns(entry.session_id)
            if turns and turns[-1].status != "running":
                break
            assert time.monotonic() < deadline, "the turn did not finish"
            await asyncio.sleep(0.01)
        assert turns[-1].status == "completed"
        assert inner.calls == 2 and inner.max_in_flight == 1
    finally:
        page.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await page
        await console.aclose()
        await gate.close()


async def test_a_turn_waiting_for_a_slot_ends_cancelled_at_shutdown(
    tmp_path: Path,
) -> None:
    inner = _HeldLLM()
    holder: dict[str, ChatGate] = {}

    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        # The server hands the page's one gate to the console; the builder
        # asks for it once the server exists.
        return await started_agent(
            Path(workspace), GatedLLM(inner, holder["gate"].client(session_id))
        )

    server = GatewayServer(
        _cfg(policy="fcfs", n_srv=1),
        console_options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=QuietProbe(),
        ),
    )
    console = server.console
    assert console is not None
    gate = console.gate
    holder["gate"] = gate
    assert server.app.state.smoke_playground.gate is gate
    workspace = tmp_path / "ws"
    workspace.mkdir()
    page = asyncio.create_task(
        GatedLLM(inner, gate.client("workspace-chat")).chat(
            [Message(role="user", content="page")]
        )
    )
    session_id = ""
    try:
        async with server.app.router.lifespan_context(server.app):
            await asyncio.wait_for(inner.entered.wait(), 5)
            entry, turn = await console.sessions.create(str(workspace), ("Hi", "Hi"))
            assert turn is not None
            session_id = entry.session_id
            await asyncio.sleep(0.05)
            # The turn's model call queues behind the page's one slot.
            assert inner.calls == 1
    finally:
        page.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await page

    # The gateway closes the one shared gate after both surfaces' sessions,
    # so a turn waiting for its slot ends as a stop, not as a call that
    # failed on a gate closed under it.
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    try:
        turns = await store.list_turns(session_id)
    finally:
        await store.aclose()
    assert [(turn.status, turn.cause) for turn in turns] == [("cancelled", "shutdown")]
