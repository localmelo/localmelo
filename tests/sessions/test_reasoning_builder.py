"""Session builders normalize wrapped providers and honor configured options."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.agent.options import ReasoningMode
from localmelo.melo.schema import Message
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.sessions.agents import build_session_agent
from localmelo.tests.fixtures.stub_llm import StubServer


def configured(url: str = "http://127.0.0.1:9/v1") -> Config:
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(
            chat_url=url, chat_model="stub-model", reasoning_field="reasoning_content"
        ),
    )


@pytest.mark.parametrize(
    "mode", ["legacy", "drop", "replay", "compress", "compress+replay"]
)
async def test_omitted_builder_options_honor_config_without_mutation(
    tmp_path: Path,
    mode: ReasoningMode,
) -> None:
    cfg = configured()
    cfg.agent.reasoning = mode
    cfg.agent.replay_turns = 2
    cfg.agent.note_writer = "heuristic"
    cfg.retrieval.retriever = "bm25"
    cfg.retrieval.k = 7
    before = copy.deepcopy(cfg)
    agent = await build_session_agent(
        cfg=cfg, base_url=None, model=None, workspace=str(tmp_path)
    )
    try:
        assert agent.options.reasoning == mode
        assert agent.options.replay_turns == 2
        assert agent.options.note_writer == "heuristic"
        assert agent.options.replay_scope == "window"
        assert agent.options.retriever == ("auto" if mode == "legacy" else "bm25")
        assert agent.options.recall_k == (3 if mode == "legacy" else 7)
        assert cfg == before
    finally:
        await agent.close()


async def test_explicit_builder_options_win_over_config(tmp_path: Path) -> None:
    cfg = configured()
    cfg.agent.reasoning = "compress"
    cfg.retrieval.retriever = "llm"
    agent = await build_session_agent(
        cfg=cfg,
        base_url=None,
        model=None,
        workspace=str(tmp_path),
        options=AgentOptions(reasoning="drop", retriever="off"),
    )
    try:
        assert agent.options.reasoning == "drop"
        assert agent.options.retriever == "off"
    finally:
        await agent.close()


@pytest.mark.parametrize("mode", ["legacy", "drop", "compress"])
async def test_wrapped_non_replay_config_sends_no_reasoning_key(
    tmp_path: Path,
    mode: ReasoningMode,
) -> None:
    server = StubServer().start()
    agent = None
    try:
        cfg = configured(server.base_url)
        agent = await build_session_agent(
            cfg=cfg,
            base_url=None,
            model=None,
            workspace=str(tmp_path),
            options=AgentOptions(reasoning=mode, retriever="off"),
            wrap_llm=lambda provider: provider,
        )
        await agent._llm.chat(
            [
                Message("user", "first"),
                Message("assistant", "answer", thinking="old reasoning"),
                Message("user", "follow"),
            ]
        )
        [body] = server.requests
        assert all(
            "reasoning_content" not in message and "reasoning" not in message
            for message in body["messages"]
        )
        assert cfg.vllm.reasoning_field == "reasoning_content"
    finally:
        if agent is not None:
            await agent.close()
        server.close()


async def test_url_builder_accepts_reasoning_field_and_retrieval_extra_body(
    tmp_path: Path,
) -> None:
    agent = await build_session_agent(
        cfg=None,
        base_url="http://127.0.0.1:9/v1",
        model="stub",
        workspace=str(tmp_path),
        options=AgentOptions(reasoning="replay", retriever="llm"),
        reasoning_field="reasoning",
        retrieval_extra_body={"temperature": 0.1},
    )
    try:
        assert agent._llm.reasoning_field == "reasoning"
        assert agent._llm.extra_body_by_kind == {"retrieve": {"temperature": 0.1}}
    finally:
        await agent.close()
