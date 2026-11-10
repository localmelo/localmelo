"""Switching melo_serve's chat model from the console (serve_switch.py).

A fake melo_serve answers on a real Unix socket (tests/fixtures/melo_serve.py);
no model loads and nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent
from localmelo.melo.schema import RunResult
from localmelo.support import melo_serve
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.tests.fixtures.melo_serve import URL, FakeServe

from .conftest import PORT, FakeAgent, FakeProbe, make_cfg, quiet_gate, seed_turn

Serving = tuple[Console, httpx.AsyncClient, list["GatedAgent"], asyncio.Event]


class GatedAgent(FakeAgent):
    """An Agent on the model its settings name, whose run waits for *gate*."""

    def __init__(self, workspace: str, model: str, gate: asyncio.Event) -> None:
        super().__init__(workspace)
        self.model, self.gate, self.ran = model, gate, False

    async def run_task(self, text: str, **kwargs: Any) -> RunResult:
        self.ran = True
        await self.gate.wait()
        return RunResult(task_id="t", status="completed", result="done")


async def until(check: Callable[[], object]) -> None:
    for _ in range(500):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("never happened")


@pytest.fixture
async def serving(tmp_path: Path, serve_fake: FakeServe) -> AsyncIterator[Serving]:
    """A console whose model server is the fake melo_serve, and a client of its routes."""
    gate = asyncio.Event()
    agents: list[GatedAgent] = []

    async def build(settings: Config, folder: str, mcp: str | None, sid: str) -> Agent:
        agents.append(GatedAgent(folder, settings.melo_serve.chat_model, gate))
        return cast(Agent, agents[-1])

    cfg = make_cfg()
    cfg.chat_backend, cfg.melo_serve = "melo_serve", LocalBackendConfig(chat_url=URL)
    console = Console(
        cfg,
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    console.serve.poll_s = 0.01
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    try:
        yield console, client, agents, gate
    finally:
        gate.set()
        await client.aclose()
        await console.aclose()


async def test_a_switch_waits_for_a_running_turn_and_a_turn_meanwhile_follows_it(
    serving: Serving, serve_fake: FakeServe, workspace: str
) -> None:
    console, client, agents, gate = serving
    listed = (await client.get("/api/serve")).json()
    assert [row["words"] for row in listed["picks"]][1:] == [
        "oMLX · needs 5 GB · fits",
        "oMLX · needs 2.5 GB · fits",
        "Splash · needs 48 GB, 31 GB free",
    ]
    assert listed["picks"][0]["current"] and not listed["picks"][3]["loadable"]
    assert listed["foot"] == "Every chat on this Mac follows the switch."
    assert listed["switch"] is None
    first = (await seed_turn(console.store, workspace)).session_id
    second = (await seed_turn(console.store, workspace)).session_id
    await console.sessions.start_turn(first, "hi", "hi")
    await until(lambda: agents and agents[0].ran)  # mid-turn on the 27B

    started = await client.post("/api/serve/switch", json={"pick": "qwen3-8b-omlx"})
    assert started.status_code == 202
    switch = console.serve.last
    assert switch is not None
    await until(lambda: switch.waiting_for == 1)
    await console.sessions.start_turn(second, "hi", "hi")  # its turn waits
    await until(lambda: console.serve.held(second) == "Switching")
    # The count is sampled: a sample taken between the turn's start and its hold says 2.
    await until(lambda: switch.waiting_for == 1)
    health = (await client.get("/api/health")).json()["switch"]
    assert (health["state"], health["title"], health["detail"]) == (
        "waiting",
        "Switching to Qwen3 8B…",
        "Waiting for 1 chat to finish",  # the held turn is not one
    )
    again = await client.post("/api/serve/switch", json={"pick": "qwen3-4b-omlx"})
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "switch_running"
    assert serve_fake.switches == []  # never under a running turn

    gate.set()  # the first chat's turn ends
    await until(lambda: switch.state == "done")
    assert serve_fake.switches[0]["pick"] == "qwen3-8b-omlx"
    assert serve_fake.switches[0]["expected_generation"] == 1
    # The held turn runs on a new Agent, on what melo_serve serves now.
    await until(lambda: len(agents) == 3 and agents[2].ran)
    assert [agent.model for agent in agents] == [
        "Qwen3.8-27B",
        "Qwen3.8-27B",
        "Qwen3-8B-4bit",
    ]
    assert agents[1].closed and not agents[1].ran


async def test_a_wait_ends_by_cancel_or_time_and_a_switch_elsewhere_is_reported(
    serving: Serving, serve_fake: FakeServe, workspace: str
) -> None:
    console, client, agents, gate = serving
    blocked = await client.post("/api/serve/switch", json={"pick": "qwen3-70b-splash"})
    assert (blocked.status_code, blocked.json()["error"]["code"]) == (
        409,
        "pick_cannot_load",
    )
    chat = (await seed_turn(console.store, workspace)).session_id
    await console.sessions.start_turn(chat, "hi", "hi")
    await until(lambda: agents and agents[0].ran)

    await client.post("/api/serve/switch", json={"pick": "qwen3-8b-omlx"})
    cancelled = await client.post("/api/serve/switch/cancel")
    assert cancelled.json()["switch"]["state"] == "cancelled"
    console.serve.wait_s = 0.05  # the 10 minutes, shortened
    await client.post("/api/serve/switch", json={"pick": "qwen3-8b-omlx"})
    await until(lambda: console.serve.last and console.serve.last.state == "failed")
    health = (await client.get("/api/health")).json()["switch"]
    assert health["error"] == "1 chat is still working; try again when it finishes"

    console.serve.wait_s = 60
    await client.post("/api/serve/switch", json={"pick": "qwen3-8b-omlx"})
    serve_fake.current, serve_fake.generation = "qwen3-4b-omlx", 2
    serve_fake.rows[2]["name"] = "Qwen3\x1b[2J 4B\n"  # shown as the data it is
    serve_fake.write()  # `melo-serve use 4b` in a terminal meanwhile
    gate.set()
    await until(lambda: console.serve.last and console.serve.last.state == "failed")
    assert serve_fake.switches[-1]["expected_generation"] == 1
    health = (await client.get("/api/health")).json()["switch"]
    assert health["error"] == "The model changed to Qwen3 4B meanwhile; choose again."
    assert serve_fake.current == "qwen3-4b-omlx"  # not undone


async def test_a_load_that_never_answers_fails_and_held_turns_go_on(
    serving: Serving,
    serve_fake: FakeServe,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console, client, agents, gate = serving
    monkeypatch.setattr(melo_serve, "LOAD_BOUND_S", 0.3)
    serve_fake.gate = threading.Event()  # melo_serve takes it and never answers
    chat = (await seed_turn(console.store, workspace)).session_id
    await client.post("/api/serve/switch", json={"pick": "qwen3-8b-omlx"})
    switch = console.serve.last
    assert switch is not None
    await until(lambda: switch.state == "loading")
    await console.sessions.start_turn(chat, "hi", "hi")
    await until(lambda: console.serve.held(chat) == "Switching")

    await until(lambda: switch.state == "failed")
    await until(lambda: agents and agents[-1].ran)  # the turn goes on
    health = (await client.get("/api/health")).json()["switch"]
    assert health["title"] == "Did not switch to Qwen3 8B"
    assert health["detail"] == health["error"]
    assert health["error"] == "melo_serve did not answer for 0 minutes"


@pytest.mark.parametrize(
    "body",
    [
        {"pick": "nothing"},
        {"pick": "8b"},  # its short name
        {"pick": "Qwen3-8B-4bit"},  # the model it serves
        {"pick": "bge-m3-omlx"},  # not a chat pick
        {"pick": ""},
        {"pick": 7},
        {"pick": "x" * 201},
        {"pick": "qwen3-8b-omlx", "force": True},
    ],
    ids=str,
)
async def test_only_the_id_of_a_listed_chat_pick_reaches_melo_serve(
    serving: Serving, serve_fake: FakeServe, body: dict[str, object]
) -> None:
    client = serving[1]
    assert (await client.post("/api/serve/switch", json=body)).status_code == 400
    assert serve_fake.switches == []
