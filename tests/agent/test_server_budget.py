"""The context budget comes from the model server unless something sets one.

A chat and a task's worker are built as the console builds them, on a stub
model that reports its context length. With a length, every planning request
is fitted to ``min(length, DEFAULT_CONTEXT_CAP)`` tokens and an unset window
on a local server is the maximum, so the token fit is the limit that acts. A
set ``[agent]`` budget wins in both directions, and with no length nothing
changes. A length too small for the fixed part ends the run with a sentence
that names it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import (
    DEFAULT_CONTEXT_CAP,
    LOCAL_WINDOW_MESSAGES,
    SHORT_TERM_MAX,
    WINDOW_MESSAGES_MAX,
    Message,
    ToolDef,
)
from localmelo.support.config import AgentConfig, Config
from localmelo.support.sessions.agents import (
    build_session_agent,
    configured_run_options,
    configured_session_options,
)


class Reports(BaseLLMProvider):
    """Answers every request at once and reports *length* as its context."""

    model = "stub"

    def __init__(
        self, length: int | None, inner: BaseLLMProvider | None = None
    ) -> None:
        self.length = length
        self.inner = inner
        self.asked = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content="done")

    async def context_length(self) -> int | None:
        self.asked += 1
        return self.length

    async def close(self) -> None:
        if self.inner is not None:
            await self.inner.close()


async def _agent(
    tmp_path: Path, kind: str, length: int | None, cfg: Config | None = None
) -> tuple[Agent, Reports]:
    """A chat's or a task's agent on a stub that reports *length*."""
    cfg = cfg or Config(chat_backend="vllm")
    options = (
        configured_session_options(cfg)
        if kind == "chat"
        else configured_run_options(cfg, AgentOptions(mode="gateway", memory=False))
    )
    made: list[Reports] = []

    def stub(inner: BaseLLMProvider) -> BaseLLMProvider:
        made.append(Reports(length, inner))
        return made[0]

    agent = await build_session_agent(
        cfg=None,
        base_url="http://stub.test/v1",
        model="stub",
        workspace=str(tmp_path),
        options=options,
        wrap_llm=stub,
    )
    return agent, made[0]


def _limits(records: list[dict[str, object]] | None) -> set[object]:
    return {record["input_limit"] for record in records or ()}


@pytest.mark.parametrize("kind", ["chat", "task"])
async def test_a_reported_length_is_the_budget_of_a_chat_and_of_a_task(
    tmp_path: Path, kind: str
) -> None:
    agent, llm = await _agent(tmp_path, kind, 262144)
    try:
        first = await agent.run_task("Say done.")
        await agent.run_task("Say done again.")
    finally:
        await agent.close()
    # 131072 tokens, less 1/8 for the reply and 1/64 for the estimate's error.
    assert DEFAULT_CONTEXT_CAP == 131072
    assert _limits(first.context_budget) == {131072 - 16384 - 2048}
    assert agent.window_messages == WINDOW_MESSAGES_MAX
    assert llm.asked == 1


@pytest.mark.parametrize(("kind", "length"), [("chat", 262144), ("task", 4096)])
async def test_a_set_budget_wins_both_ways(
    tmp_path: Path, kind: str, length: int
) -> None:
    """Smaller than the server says (the chat) or larger (the task): it stands."""
    window = 8000 if length > 8000 else 300000
    cfg = Config(
        chat_backend="vllm",
        agent=AgentConfig(context_window=window, output_reserve=500, safety_margin=100),
    )
    agent, llm = await _agent(tmp_path, kind, length, cfg)
    try:
        result = await agent.run_task("Say done.")
    finally:
        await agent.close()
    assert _limits(result.context_budget) == {window - 600}
    assert llm.asked == 0


async def test_with_no_length_nothing_changes(tmp_path: Path) -> None:
    agent, llm = await _agent(tmp_path, "task", None)
    try:
        result = await agent.run_task("Say done.")
    finally:
        await agent.close()
    assert "context_budget" not in result.to_dict()
    assert agent.window_messages == LOCAL_WINDOW_MESSAGES
    assert llm.asked == 1


@pytest.mark.parametrize(
    ("cfg", "window"),
    [
        (Config(chat_backend="vllm", agent=AgentConfig(window_messages=32)), 32),
        (Config(chat_backend="openai"), SHORT_TERM_MAX),  # a cloud vendor keeps 20
    ],
)
async def test_a_window_that_is_set_or_a_cloud_default_stays(
    tmp_path: Path, cfg: Config, window: int
) -> None:
    agent, _ = await _agent(tmp_path, "task", 262144, cfg)
    try:
        result = await agent.run_task("Say done.")
    finally:
        await agent.close()
    assert result.context_budget and agent.window_messages == window


async def test_a_context_too_small_for_the_fixed_part_says_so(tmp_path: Path) -> None:
    agent, _ = await _agent(tmp_path, "task", 600)
    try:
        result = await agent.run_task("Say done.")
    finally:
        await agent.close()
    assert result.status == "context_budget_exceeded"
    assert result.error is not None
    assert result.error.startswith(
        "The model server runs this model with a context of 600 tokens, too small"
    )
    assert result.error.endswith("load the model with a longer context and try again.")
