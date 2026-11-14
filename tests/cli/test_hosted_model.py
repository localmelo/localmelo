"""``/model`` once a hosted service has a key (support/cli/hosted_model.py).

A fake gateway answers on an httpx transport and no model server or hosted service is
called: the switch itself is the gateway's (tests/gateway/console/test_hosted_models.py).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support import config, hosted, keychain, model_choices
from localmelo.support.backends import providers_for
from localmelo.support.cli import hosted_model, serve_model
from localmelo.support.cli.agents import AgentFactory, config_backend
from localmelo.support.cli.approvals import TerminalApprover
from localmelo.support.cli.main_chat import GatewayClient
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.cli.session import CliSession
from localmelo.support.cli.theme import UNICODE
from localmelo.support.sessions import agents as session_agents
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.sessions.helpers import ScriptedLLM, answer

from .test_session import FakeApp

KEY = "sk-test-0123456789abcdef"
LOCAL = {
    "id": "local",
    "label": "vLLM",
    "kind": "local",
    "serves": False,
    "models": ["qwen"],
    "current": "qwen",
    "problem": "",
}
DEEPSEEK = {
    "id": "deepseek",
    "label": "DeepSeek",
    "kind": "hosted",
    "serves": False,
    "models": ["deepseek-v4-pro", "deepseek-flash"],
    "current": "",
    "problem": "",
}
FAILING = {
    "id": "openai",
    "label": "OpenAI",
    "kind": "hosted",
    "serves": False,
    "models": [],
    "current": "",
    "problem": "The key was refused (401).",
}


async def _until(check: Callable[[], object]) -> None:
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0.01)


class Gateway:
    """A gateway that lists the groups and, on a switch, saves it as the real one does."""

    def __init__(self, groups: list[dict[str, Any]]) -> None:
        self.groups = groups
        self.asked: list[tuple[str, str, Any]] = []
        self.refuse = ""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.asked.append((request.method, request.url.path, body))
        if request.url.path == "/api/chat-models/switch":
            if self.refuse:
                error = {"code": "no_model", "message": self.refuse, "details": {}}
                return httpx.Response(400, json={"error": error})
            cfg = config.load()
            found = hosted.get(body["group"])
            assert found is not None
            hosted.use_service(cfg, found, body["model"])
            config.save(cfg)
            return httpx.Response(200, json={"ok": True})
        cfg = config.load()
        now = {
            "group": model_choices.in_use_group(cfg),
            "model": getattr(getattr(cfg, cfg.chat_backend), "chat_model", ""),
        }
        return httpx.Response(
            200, json={"hosted": True, "current": now, "groups": self.groups}
        )


@pytest.fixture
async def chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[CliSession, FakeApp, RecordingOut, Gateway]]:
    """The terminal's own chat on a local server, a key saved for DeepSeek, a gateway stub."""
    monkeypatch.setenv("HOME", str(tmp_path))
    keychain.write("localmelo.deepseek", KEY)
    cfg = config.Config(
        chat_backend="vllm",
        embedding_backend="ollama",
        vllm=config.LocalBackendConfig(
            chat_url="http://127.0.0.1:9/v1", chat_model="qwen"
        ),
    )
    config.save(cfg)

    def providers(cfg: config.Config, reasoning: str) -> tuple[ScriptedLLM, None]:
        model = providers_for(cfg, reasoning)[0].model  # type: ignore[attr-defined]
        llm = ScriptedLLM([answer(f"{model} here.")])
        llm.model = model  # type: ignore[attr-defined]
        return llm, None

    monkeypatch.setattr(session_agents, "providers_for", providers)
    gateway = Gateway([LOCAL, DEEPSEEK, FAILING])

    async def found(session: CliSession) -> tuple[GatewayClient, bool]:
        client = GatewayClient(
            "http://127.0.0.1:1",
            transport=httpx.MockTransport(gateway),
            headers={"Authorization": "Bearer token"},
        )
        return client, True

    monkeypatch.setattr(serve_model, "_gateway", found)
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
    app = FakeApp()
    session.attach(app)  # type: ignore[arg-type]
    try:
        yield session, app, out, gateway
    finally:
        await session.shutdown()


async def test_the_one_list_switches_the_chat_and_keeps_the_embedding(
    chat: tuple[CliSession, FakeApp, RecordingOut, Gateway],
) -> None:
    session, app, out, gateway = chat
    app.picks = ["DeepSeek · deepseek-v4-pro"]
    session.enter("/model")
    await _until(
        lambda: "Saved as the default" in out.text() and session.state == "idle"
    )

    rows, _, current = app.offered[0]
    assert [row.id for row in rows] == [
        "vLLM · qwen",
        "DeepSeek · deepseek-v4-pro",
        "DeepSeek · deepseek-flash",
    ]  # a service that failed is no row; it is said above the list
    assert current == "vLLM · qwen"
    assert "⚠ OpenAI: The key was refused (401)." in out.text()
    assert gateway.asked[-1] == (
        "POST",
        "/api/chat-models/switch",
        {"group": "deepseek", "model": "deepseek-v4-pro"},
    )
    assert "Model: DeepSeek · deepseek-v4-pro. Saved as the default." in out.text()
    # The terminal's own chat is on the new model for its next message, with no key typed.
    assert session.factory.backend.model == "deepseek-v4-pro"
    assert session.agent._llm.model == "deepseek-v4-pro"  # type: ignore[attr-defined]
    # Embedding is not part of a switch, in the saved file or in the session.
    assert config.load().embedding_backend == "ollama"
    assert session.factory._cfg is not None
    assert session.factory._cfg.embedding_backend == "ollama"
    assert KEY not in out.text()


async def test_a_typed_name_switches_without_the_list_and_a_refusal_changes_nothing(
    chat: tuple[CliSession, FakeApp, RecordingOut, Gateway],
) -> None:
    session, app, out, gateway = chat
    session.enter("/model deepseek:deepseek-flash")
    await _until(
        lambda: "Saved as the default" in out.text() and session.state == "idle"
    )
    assert app.offered == []
    assert session.factory.backend.model == "deepseek-flash"

    session.enter("/model nothing-like-it")
    await _until(lambda: "No listed model is called 'nothing-like-it'" in out.text())
    assert [call[1] for call in gateway.asked].count("/api/chat-models/switch") == 1

    session.enter("/model deepseek-flash")  # the one in use
    await _until(lambda: "Still using DeepSeek · deepseek-flash." in out.text())

    gateway.refuse = "The service no longer lists gpt-x."
    session.enter("/model qwen")
    await _until(lambda: "✗ The service no longer lists gpt-x." in out.text())
    assert session.factory.backend.model == "deepseek-flash"
    assert config.load().chat_backend == "openai"  # the refused switch saved nothing


async def test_with_no_hosted_key_or_no_gateway_the_command_is_the_old_one(
    chat: tuple[CliSession, FakeApp, RecordingOut, Gateway],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _, gateway = chat
    assert hosted_model.applies(session)
    assert await _no_gateway(session, monkeypatch) is False
    assert gateway.asked == []

    keychain.remove("localmelo.deepseek")
    assert not hosted_model.applies(session)
    assert not session._model_lists_services()


async def _no_gateway(session: CliSession, monkeypatch: pytest.MonkeyPatch) -> bool:
    async def none(session: CliSession) -> tuple[None, bool]:
        return None, False

    monkeypatch.setattr(serve_model, "_gateway", none)
    return await hosted_model.command(session, "qwen")
