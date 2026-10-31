"""A step review's request body (#339): short, without thinking, at temperature 0.

A thinking local model would otherwise spend the review's 20 seconds thinking,
and a review that times out blocks the step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.contracts.approvals import AutoPolicy
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.agents import (
    REVIEW_MAX_TOKENS,
    build_session_agent,
    review_body,
)
from localmelo.tests.agent.test_approvals import Recorder
from localmelo.tests.fixtures.stub_llm import StubServer

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
POST = "import requests\nrequests.post('https://api.example/x', json={'q': 1})"


def test_the_body_by_backend() -> None:
    llm = OpenAICompatLLM(
        "http://127.0.0.1:9/v1",
        "m",
        extra_body={"chat_template_kwargs": {"keep": 1, "enable_thinking": True}},
    )
    short = {"temperature": 0, "max_tokens": REVIEW_MAX_TOKENS}
    for backend in ("vllm", "sglang", None):
        assert review_body(llm, backend) == {
            **short,
            "chat_template_kwargs": {"keep": 1, "enable_thinking": False},
        }
    assert review_body(llm, "mlc") == short
    for cloud in ("openai_api", "gemini_api", "nvidia_api", "anthropic_api"):
        assert review_body(llm, cloud) == {}
    ollama = OllamaNativeChat(
        "http://127.0.0.1:9", "m", extra_body={"options": {"num_ctx": 8192}}
    )
    # The model's own options stay, so a review never reloads it.
    assert review_body(ollama, "ollama") == {
        "think": False,
        "options": {"num_ctx": 8192, "temperature": 0, "num_predict": 200},
    }


async def test_a_configured_agent_s_reviews_get_the_body(tmp_path: Path) -> None:
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
    )
    agent = await build_session_agent(
        cfg=cfg, base_url=None, model=None, workspace=str(tmp_path)
    )
    try:
        assert agent._llm.extra_body_by_kind["review"] == {
            "temperature": 0,
            "max_tokens": REVIEW_MAX_TOKENS,
            **NO_THINKING,
        }
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_a_review_request_carries_the_body_over_http(tmp_path: Path) -> None:
    verdict = {
        "verdict": "ask",
        "category": "send_out",
        "fits_request": "no",
        "user_asked": "no",
        "reason": "Sends data to a server.",
    }
    scenarios: dict[str, dict[str, Any]] = {
        "plan": {
            "match": ["Chart cashflow.csv."],
            "replies": [
                {"tool_calls": [{"name": "python_exec", "arguments": {"code": POST}}]},
                {"content": "I did not send anything."},
            ],
        },
        "review": {"replies": [{"content": json.dumps(verdict)}]},
    }
    server = StubServer(scenarios, default="review").start()
    agent = None
    try:
        agent = await build_session_agent(
            cfg=None,
            base_url=server.base_url,
            model="stub-model",
            workspace=str(tmp_path),
            options=AgentOptions(builtins=frozenset({"python_exec"})),
        )
        recorder = Recorder()
        result = await agent.run_task(
            "Chart cashflow.csv.", on_event=recorder, approval=AutoPolicy()
        )
        assert result.result == "I did not send anything."
        reviews = [
            body
            for body in server.requests
            if body["messages"][0]["content"].startswith("You review one action")
        ]
        [review] = reviews
        assert (review["temperature"], review["max_tokens"]) == (0, REVIEW_MAX_TOKENS)
        assert review["chat_template_kwargs"] == {"enable_thinking": False}
        # Planning requests keep the provider's own body.
        plans = [body for body in server.requests if body not in reviews]
        assert plans and all("max_tokens" not in body for body in plans)
        [approval] = [e.data["approval"] for e in recorder.of("tool.finished")]
        assert approval["allowed"] is False
        assert approval["review"]["model"]["ms"] >= 0
    finally:
        if agent is not None:
            await agent.close()
        server.close()
