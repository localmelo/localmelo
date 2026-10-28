"""Focused regressions from the independent R2b integration review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import LLM_REQUEST_KIND
from localmelo.melo.schema import Message
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate, GatedLLM
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.sessions.agents import build_session_agent
from localmelo.tests.sessions.helpers import ScriptedLLM, answer


@pytest.mark.parametrize("inline", [False, True])
async def test_loaded_legacy_replay_strips_thinking_before_evicting_facts(
    inline: bool,
) -> None:
    provider = ScriptedLLM([answer("Done.")])
    provider.reasoning_field = "reasoning_content"
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            reasoning="replay", replay_scope="window", builtins=frozenset()
        ),
    )
    thinking = "x" * 40_000
    messages = [
        message
        for index in range(3)
        for message in (
            Message("user", f"old question {index}"),
            Message(
                "assistant",
                (f"<think>{thinking}</think>" if inline else "") + f"kept fact {index}",
                thinking=thinking,
            ),
        )
    ]
    try:
        assert agent.load_conversation(messages) == 6
        outcome = await agent.run_task("follow")
        assert outcome.status == "completed"
        assert outcome.memory_stats is not None
        assert outcome.memory_stats["replay_stripped"] == 1
        [request] = provider.requests
        assert [m.content for m in request if m.role == "assistant"] == [
            f"kept fact {index}" for index in range(3)
        ]
        assert len([m for m in request if m.thinking]) == 2
        # Loading uses copies; neither request normalization nor cap fitting
        # rewrites a caller's checkpoint objects.
        assert all(m.thinking == thinking for m in messages if m.role == "assistant")
        assert ("<think>" in messages[1].content) is inline
    finally:
        await agent.close()


async def test_native_gated_retrieval_override_is_applied_only_to_retrieve(
    tmp_path: Path,
) -> None:
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "Done."}}
        )

    provider = OllamaNativeChat(
        "http://synthetic.invalid",
        "stub",
        extra_body={"think": True},
        transport=httpx.MockTransport(respond),
    )
    gate = ChatGate()
    agent = None
    try:
        with mock.patch(
            "localmelo.melo.agent.agent._providers_from_config",
            return_value=(provider, None),
        ):
            agent = await build_session_agent(
                cfg=Config(chat_backend="ollama", embedding_backend="none"),
                base_url=None,
                model=None,
                workspace=str(tmp_path),
                options=AgentOptions(reasoning="compress", retriever="llm"),
                retrieval_extra_body={"think": False},
                wrap_llm=lambda inner: GatedLLM(inner, gate.client("native-review")),
            )
        for kind in ("retrieve", "plan", "reflect"):
            token = LLM_REQUEST_KIND.set(kind)
            try:
                await agent._llm.chat([Message("user", "synthetic request")])
            finally:
                LLM_REQUEST_KIND.reset(token)
        assert [body["think"] for body in bodies] == [False, True, True]
        assert all(
            body["model"] == "stub" and body["stream"] is False for body in bodies
        )
    finally:
        if agent is not None:
            await agent.close()
        await provider.close()
        await gate.close()
