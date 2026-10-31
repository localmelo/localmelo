"""Real console wrappers retain the provider's replay and request-kind contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.agent.options import ReasoningMode
from localmelo.melo.contracts.providers import LLM_REQUEST_KIND
from localmelo.melo.schema import Message
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway.chat_gate import ChatGate, GatedLLM
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.sessions.agents import build_session_agent
from localmelo.tests.fixtures.stub_llm import StubServer


@pytest.mark.parametrize(
    ("mode", "required", "replayed"),
    [
        ("legacy", False, False),
        ("drop", False, False),
        ("compress", False, False),
        ("replay", False, True),
        ("compress+replay", False, True),
        ("compress", True, True),
        ("replay", True, True),
    ],
)
async def test_gated_configured_session_normalizes_before_wrapping_and_replays(
    tmp_path: Path,
    mode: ReasoningMode,
    required: bool,
    replayed: bool,
) -> None:
    server = StubServer().start()
    gate = ChatGate()
    agent = None
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(
            chat_url=server.base_url,
            chat_model="stub-model",
            reasoning_field="reasoning_content",
            requires_reasoning_replay=required,
        ),
    )
    try:
        agent = await build_session_agent(
            cfg=cfg,
            base_url=None,
            model=None,
            workspace=str(tmp_path),
            options=AgentOptions(reasoning=mode, retriever="off"),
            wrap_llm=lambda provider: GatedLLM(provider, gate.client("test-session")),
        )
        assert isinstance(agent._llm, GatedLLM)
        assert agent._llm.reasoning_field == ("reasoning_content" if replayed else None)
        assert agent._llm.requires_reasoning_replay is required
        assert agent.options.reasoning == (
            "compress+replay" if required and mode == "compress" else mode
        )
        agent.load_conversation(
            [
                Message("user", "first"),
                Message("assistant", "answer", thinking="retained reasoning"),
            ]
        )
        outcome = await agent.run_task("follow")
        assert outcome.status == "completed"
        [body] = server.requests
        old = next(m for m in body["messages"] if m["role"] == "assistant")
        assert old.get("reasoning_content") == (
            "retained reasoning" if replayed else None
        )
        assert cfg.vllm.reasoning_field == "reasoning_content"
    finally:
        if agent is not None:
            await agent.close()
        await gate.close()
        server.close()


@pytest.mark.parametrize(
    "mode", ["legacy", "drop", "replay", "compress", "compress+replay"]
)
async def test_gated_native_provider_keeps_its_fixed_thinking_field(
    tmp_path: Path,
    mode: ReasoningMode,
) -> None:
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "Done."}}
        )

    provider = OllamaNativeChat(
        "http://synthetic.invalid", "stub", transport=httpx.MockTransport(respond)
    )
    gate = ChatGate()
    agent = None
    cfg = Config(chat_backend="ollama", embedding_backend="none")
    try:
        with mock.patch(
            "localmelo.melo.agent.agent._providers_from_config",
            return_value=(provider, None),
        ):
            agent = await build_session_agent(
                cfg=cfg,
                base_url=None,
                model=None,
                workspace=str(tmp_path),
                options=AgentOptions(reasoning=mode, retriever="off"),
                wrap_llm=lambda inner: GatedLLM(inner, gate.client("native-session")),
            )
        assert agent._llm.reasoning_field == "thinking"
        agent.load_conversation(
            [
                Message("user", "first"),
                Message("assistant", "answer", thinking="native reasoning"),
            ]
        )
        outcome = await agent.run_task("follow")
        assert outcome.status == "completed"
        [body] = bodies
        old = next(m for m in body["messages"] if m["role"] == "assistant")
        assert old.get("thinking") == (
            "native reasoning"
            if mode in ("legacy", "replay", "compress+replay")
            else None
        )
    finally:
        if agent is not None:
            await agent.close()
        await provider.close()
        await gate.close()


async def test_gated_retrieval_body_is_sent_only_on_retrieve_requests(
    tmp_path: Path,
) -> None:
    server = StubServer().start()
    gate = ChatGate()
    agent = None
    extra = {"temperature": 0.1, "chat_template_kwargs": {"enable_thinking": False}}
    try:
        agent = await build_session_agent(
            cfg=None,
            base_url=server.base_url,
            model="stub-model",
            workspace=str(tmp_path),
            options=AgentOptions(reasoning="compress", retriever="llm"),
            retrieval_extra_body=extra,
            wrap_llm=lambda provider: GatedLLM(
                provider, gate.client("retrieve-session")
            ),
        )
        assert agent._llm.extra_body_by_kind["retrieve"] == extra
        # A step review gets its own short, unthinking body (#339).
        assert set(agent._llm.extra_body_by_kind) == {"retrieve", "review"}
        extra["temperature"] = 0.9
        for kind in ("retrieve", "plan", "reflect"):
            token = LLM_REQUEST_KIND.set(kind)
            try:
                await agent._llm.chat([Message("user", "synthetic request")])
            finally:
                LLM_REQUEST_KIND.reset(token)
        retrieve, plan, reflect = server.requests
        assert retrieve["temperature"] == 0.1
        assert retrieve["chat_template_kwargs"] == {"enable_thinking": False}
        for body in (plan, reflect):
            assert "temperature" not in body
            assert "chat_template_kwargs" not in body
    finally:
        if agent is not None:
            await agent.close()
        await gate.close()
        server.close()
