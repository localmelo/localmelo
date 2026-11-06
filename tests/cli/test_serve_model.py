"""``/model`` when melo_serve is the model server (support/cli/serve_model.py).

A fake melo_serve answers on a real Unix socket (tests/fixtures/melo_serve.py)
and a fake gateway on an httpx transport; no model loads.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support import config, melo_serve
from localmelo.support.backends import providers_for
from localmelo.support.cli import serve_model
from localmelo.support.cli.agents import AgentFactory, config_backend
from localmelo.support.cli.approvals import TerminalApprover
from localmelo.support.cli.main_chat import GatewayClient
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.picker import PickPicker
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.cli.session import CliSession
from localmelo.support.cli.theme import UNICODE
from localmelo.support.sessions import agents as session_agents
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.fixtures.melo_serve import URL, FakeServe
from localmelo.tests.sessions.helpers import ScriptedLLM, answer

from .test_session import FakeApp


class PickingApp(FakeApp):
    """The app, with the list of melo_serve's picks answered from a script."""

    def __init__(self) -> None:
        super().__init__()
        self.listed: list[list[melo_serve.Pick]] = []
        self.chosen: list[str | None] = []

    async def choose_pick(
        self, picks: list[melo_serve.Pick], *, foot: str
    ) -> melo_serve.Pick | None:
        self.listed.append(list(picks))
        name = self.chosen.pop(0)
        return next((pick for pick in picks if pick.name == name), None)


async def _until(check: Callable[[], object]) -> None:
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0.01)


@pytest.fixture
async def chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_fake: FakeServe
) -> AsyncIterator[tuple[CliSession, PickingApp, RecordingOut]]:
    """The terminal's own chat on melo_serve, its models scripted, no gateway running."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = config.Config(
        chat_backend="melo_serve",
        embedding_backend="none",
        melo_serve=config.LocalBackendConfig(chat_url=URL),
    )

    def providers(cfg: config.Config, reasoning: str) -> tuple[ScriptedLLM, None]:
        model = providers_for(cfg, reasoning)[0].model  # type: ignore[attr-defined]
        llm = ScriptedLLM([answer(f"{model} here.")])
        llm.model = model  # type: ignore[attr-defined]
        return llm, None

    monkeypatch.setattr(session_agents, "providers_for", providers)
    factory = AgentFactory(
        backend=config_backend(cfg, None),
        cfg=cfg,
        options=AgentOptions(memory=False, workspace_root=str(tmp_path)),
        workspace=str(tmp_path),
        saves=False,
    )
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    out = RecordingOut(100)
    approver = TerminalApprover()
    session = CliSession(
        store=store,
        factory=factory,
        agent=await factory.build(),
        renderer=TurnRenderer(
            out,
            symbols=UNICODE,
            backend=lambda: factory.backend,
            snapshots=approver.snapshots,
        ),
        approver=approver,
        mode="ask",
        out=out,
    )
    app = PickingApp()
    session.attach(app)  # type: ignore[arg-type]
    try:
        yield session, app, out
    finally:
        await session.shutdown()


async def test_the_list_and_the_typed_name_switch_melo_serve(
    chat: tuple[CliSession, PickingApp, RecordingOut], serve_fake: FakeServe
) -> None:
    session, app, out = chat
    app.chosen = ["Qwen3 8B"]
    session.enter("/model")
    await _until(lambda: "Switched to" in out.text() and session.state == "idle")
    assert [p.name for p in app.listed[0]][:2] == ["Qwen3.8 27B", "Qwen3 8B"]
    assert serve_fake.switches[0]["pick"] == "qwen3-8b-omlx"
    assert serve_fake.switches[0]["expected_generation"] == 1
    assert "✓ Switched to Qwen3 8B · oMLX" in out.text()
    # The terminal's own chat moves to the new model at once.
    assert session.factory.backend.model == "Qwen3-8B-4bit"
    assert session.agent._llm.model == "Qwen3-8B-4bit"  # type: ignore[attr-defined]

    session.enter("/model 4b")  # no list
    await _until(lambda: len(serve_fake.switches) == 2 and session.state == "idle")
    assert serve_fake.switches[1]["expected_generation"] == 2
    assert len(app.listed) == 1
    session.enter("/model 70b")
    await _until(lambda: "cannot load" in out.text())
    assert "✗ Qwen3 70B cannot load: Splash · needs 48 GB, 31 GB free." in out.text()
    assert len(serve_fake.switches) == 2


async def test_a_message_sent_while_the_picks_load_gets_no_switch_under_its_turn(
    chat: tuple[CliSession, PickingApp, RecordingOut],
    serve_fake: FakeServe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, out = chat
    gate = asyncio.Event()
    read = melo_serve.picks

    async def slow() -> list[melo_serve.Pick]:
        await gate.wait()
        return await read()

    monkeypatch.setattr(melo_serve, "picks", slow)
    session.enter("/model 8b")
    assert session.state == "switching"  # from the start
    assert session.enter("hello") is not None  # a hint: the draft stays, no turn
    assert session.state == "switching"
    gate.set()
    await _until(lambda: "Switched to" in out.text() and session.state == "idle")
    assert len(serve_fake.switches) == 1

    gate.clear()
    session.enter("/model 4b")
    await asyncio.sleep(0)
    assert session.stop()  # Esc while they load: nothing is sent
    await _until(lambda: session.state == "idle")
    assert len(serve_fake.switches) == 1


async def test_the_token_goes_only_to_a_gateway_on_this_mac(
    chat: tuple[CliSession, PickingApp, RecordingOut],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def token(url: str) -> dict[str, str]:
        raise AssertionError("the token was read for another host")

    other = SimpleNamespace(url="http://192.0.2.1:8401")
    monkeypatch.setattr(serve_model.gateway_endpoint, "live", lambda: other)
    monkeypatch.setattr(serve_model.task_cli, "client_headers", token)
    assert await serve_model._gateway(chat[0]) == (None, False)


@pytest.mark.parametrize(
    ("lost", "ends"),
    [
        ({"GET": 1}, "Switched to Qwen3 8B"),  # one read lost: it goes on
        ({"GET": 100}, "may still be running"),
        ({"POST": 1}, "may still be running"),
    ],
    ids=["one-read", "every-read", "the-request"],
)
async def test_a_gateway_that_stops_answering_never_claims_the_old_model(
    chat: tuple[CliSession, PickingApp, RecordingOut],
    serve_fake: FakeServe,
    monkeypatch: pytest.MonkeyPatch,
    lost: dict[str, int],
    ends: str,
) -> None:
    session, _, out = chat
    done = {"id": 1, "state": "done", "error": ""}

    def gateway(request: httpx.Request) -> httpx.Response:
        if lost.get(request.method, 0):
            lost[request.method] -= 1
            raise httpx.ConnectError("gone", request=request)
        if request.method == "POST":
            return httpx.Response(202, json={"switch": {**done, "state": "loading"}})
        serve_fake.current, serve_fake.generation = "qwen3-8b-omlx", 2
        serve_fake.write()
        return httpx.Response(200, json={"switch": done})

    async def found(session: CliSession) -> tuple[GatewayClient, bool]:
        client = GatewayClient(
            "http://127.0.0.1:1",
            transport=httpx.MockTransport(gateway),
            headers={"Authorization": "Bearer token"},
        )
        return client, True

    monkeypatch.setattr(serve_model, "_gateway", found)
    monkeypatch.setattr(serve_model, "POLL_S", 0.01)
    session.enter("/model 8b")
    await _until(lambda: ends in out.text() and session.state == "idle")
    assert "Still using" not in out.text()
    assert lost.get("POST", 0) == 0 and lost.get("GET", 0) != 1  # they did fail


async def test_with_a_gateway_the_switch_goes_through_its_route(
    chat: tuple[CliSession, PickingApp, RecordingOut],
    serve_fake: FakeServe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, app, out = chat
    asked: list[tuple[str, str, Any]] = []
    shown: list[tuple[str | None, str]] = []
    waiting = {"id": 1, "state": "waiting", "waiting_for": 1, "error": ""}

    def gateway(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        asked.append((request.method, request.url.path, body))
        assert request.headers["authorization"] == "Bearer token"
        if request.url.path == "/api/serve/switch":
            return httpx.Response(202, json={"switch": waiting})
        shown.append((session.activity, session.activity_note))
        serve_fake.current, serve_fake.generation = "qwen3-8b-omlx", 2
        serve_fake.write()  # the gateway's switch, once the chat finished
        return httpx.Response(200, json={"switch": {**waiting, "state": "done"}})

    async def found(session: CliSession) -> tuple[GatewayClient, bool]:
        client = GatewayClient(
            "http://127.0.0.1:1",
            transport=httpx.MockTransport(gateway),
            headers={"Authorization": "Bearer token"},
        )
        return client, True

    monkeypatch.setattr(serve_model, "_gateway", found)
    monkeypatch.setattr(serve_model, "POLL_S", 0.01)
    session.enter("/model qwen3 8b")
    await _until(lambda: "Switched to" in out.text() and session.state == "idle")
    assert asked[0] == ("POST", "/api/serve/switch", {"pick": "qwen3-8b-omlx"})
    assert asked[1][:2] == ("GET", "/api/health")
    assert shown[0] == ("Waiting for 1 chat to finish", "esc stop waiting")
    assert serve_fake.switches == []  # the gateway switched, not the terminal
    assert session.factory.backend.model == "Qwen3-8B-4bit"


async def test_the_list_marks_the_current_pick_and_its_arrows_skip_one_that_cannot_load(
    serve_fake: FakeServe,
) -> None:
    picks = await melo_serve.picks()
    picker = PickPicker(
        picks, foot="Every chat on this Mac follows the switch.",
        width=lambda: 92, invalidate=lambda: None, symbols=UNICODE,
    )  # fmt: skip
    picker.rows, picker.selected = picks, picker._start(picks)
    picker.loading = False
    lines = [text for _, text in picker._lines()]
    assert lines[0].startswith(" ❯ ✓ Qwen3.8 27B   Splash · needs 21 GB · fits")
    assert lines[3] == "     Qwen3 70B     Splash · needs 48 GB, 31 GB free"
    assert lines[-2:] == [
        " Every chat on this Mac follows the switch.",
        " ↑↓ select · enter switch · esc keep Qwen3.8 27B",
    ]
    picker.move(1)
    picker.move(1)
    picker.move(1)  # past the 4B: the 70B is skipped
    assert picker.rows[picker.selected].name == "Qwen3 4B"
