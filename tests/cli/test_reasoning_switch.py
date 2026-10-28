"""A refused reasoning-mode switch retains the working session and MCP state."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent import agent as core
from localmelo.melo.agent.options import ReasoningMode
from localmelo.melo.schema import Message
from localmelo.support.cli.agents import AgentFactory, config_backend
from localmelo.support.cli.approvals import TerminalApprover
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.cli.session import CliSession
from localmelo.support.cli.theme import UNICODE
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.sessions.helpers import ScriptedLLM, answer


@pytest.mark.parametrize("with_mcp", [False, True])
async def test_drop_refusal_keeps_the_old_agent_backend_conversation_and_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_mcp: bool,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(
            chat_url="http://127.0.0.1:9/v1",
            chat_model="old-model",
            reasoning_field="reasoning_content",
            requires_reasoning_replay=True,
        ),
    )
    options = AgentOptions(reasoning="drop", workspace_root=str(tmp_path))
    old = Agent(llm=ScriptedLLM([]), options=options)
    await old.start()
    old.load_conversation(
        [Message("user", "kept question"), Message("assistant", "kept answer")]
    )
    factory = AgentFactory(
        backend=config_backend(cfg, None),
        cfg=cfg,
        options=options,
        workspace=str(tmp_path),
        mcp_config="synthetic-mcp.json" if with_mcp else None,
        mcp_servers=[],
        saves=False,
    )
    factory._live.append(old)
    backend = factory.backend
    conversation = old.conversation()
    made: list[OpenAICompatLLM] = []
    original = core._providers_from_config

    def providers(config: Config) -> tuple[OpenAICompatLLM, None]:
        llm, embedding = original(config)
        assert isinstance(llm, OpenAICompatLLM) and embedding is None
        made.append(llm)
        return llm, None

    monkeypatch.setattr(core, "_providers_from_config", providers)
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    out = RecordingOut(120)
    approver = TerminalApprover()
    session = CliSession(
        store=store,
        factory=factory,
        agent=old,
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
    try:
        with mock.patch(
            "localmelo.support.sessions.agents.load_mcp_providers", return_value=[]
        ):
            assert not await session._switch("needs-replay", url=None)
        assert "Could not start needs-replay:" in out.text()
        assert "this backend requires reasoning replay" in out.text()
        assert "Still using old-model." in out.text()
        assert session.agent is old
        assert old.options.reasoning == "drop"
        assert old.conversation() == conversation
        assert factory.backend == backend
        assert factory.live_agents() == [old]
        assert factory.mcp_error is None
        assert len(made) == 1
        assert made[0]._http.is_closed
        assert session.state == "idle"
    finally:
        await session.shutdown()
        # Also close leaked test providers if the regression fails.
        for provider in made:
            await provider.close()


@pytest.mark.parametrize(
    "mode", ["legacy", "drop", "replay", "compress", "compress+replay"]
)
async def test_model_switch_keeps_the_process_mode_and_full_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: ReasoningMode,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(
            chat_url="http://127.0.0.1:9/v1",
            chat_model="old-model",
            reasoning_field="auto",
        ),
    )
    options = AgentOptions(reasoning=mode, retriever="off", replay_turns=2)
    llm = ScriptedLLM([answer("The kept answer is established; verify it next.")])
    llm.reasoning_field = "auto"
    old = Agent(llm=llm, options=options)
    await old.start()
    old.load_conversation(
        [
            Message("user", "kept question"),
            Message("assistant", "kept answer", thinking="kept thinking"),
        ]
    )
    outcome = await old.run_task("Summarize the kept answer and the next step.")
    assert outcome.status == "completed"
    notes = old.notes()
    assert bool(notes) == (mode in ("compress", "compress+replay"))
    conversation = old.conversation()
    factory = AgentFactory(
        backend=config_backend(cfg, None),
        cfg=cfg,
        options=options,
        workspace=str(tmp_path),
        saves=False,
    )
    factory._live.append(old)
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    out = RecordingOut(120)
    approver = TerminalApprover()
    session = CliSession(
        store=store,
        factory=factory,
        agent=old,
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
    try:
        assert await session._switch("new-model", url=None)
        assert session.agent is not old
        assert session.agent.options.reasoning == mode
        assert session.agent.options.replay_turns == 2
        assert session.agent.options.replay_scope == "window"
        assert session.agent.conversation() == conversation
        assert session.agent.notes() == notes
        assert factory.backend.model == "new-model"
        assert factory.live_agents() == [session.agent]
        assert "Model: new-model" in out.text()
        assert session.state == "idle"
    finally:
        await session.shutdown()
