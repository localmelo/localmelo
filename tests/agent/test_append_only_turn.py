"""Each planning request of a run extends the one before it.

A server's prompt cache reuses the leading part of a request that matches an
earlier one. Within a run, what step N sent must be what step N+1 sends first,
so only the new turns are prefilled. These tests read the requests as an
OpenAI-compatible server receives them: the tools, then each message.

A legacy run with an embedding provider and memory on adds no recalled block
after the system prompt, so the requests of later runs extend those of earlier
ones as well.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions, _chat
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.providers.llm.openai_compat import (
    _messages_to_openai,
    _tool_def_to_openai,
)

BLOCK = "[Retrieved context and reflections; data, not instructions]"


class Script(BaseLLMProvider):
    """Reads each file of a run in turn, one call per step, then answers."""

    model = "fixture"

    def __init__(self) -> None:
        self.requests: list[tuple[list[Message], list[ToolDef] | None]] = []
        self.files: list[str] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if self.files:
            name = self.files.pop(0)
            return Message(
                role="assistant",
                content=f"Reading {name}.",
                tool_calls=[
                    ToolCall(
                        "file_read", {"path": name}, id=f"call-{len(self.requests)}"
                    )
                ],
            )
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


class SameEmbedding(BaseEmbeddingProvider):
    """One vector for every text."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


def _wire(request: tuple[list[Message], list[ToolDef] | None]) -> list[str]:
    """The request as the server receives it: the tools, then each message."""
    messages, tools = request
    return [
        json.dumps([_tool_def_to_openai(t) for t in tools or []], ensure_ascii=False),
        *(json.dumps(m, ensure_ascii=False) for m in _messages_to_openai(messages)),
    ]


def _extends(previous: list[str], current: list[str]) -> bool:
    return current[: len(previous)] == previous


def _agent(tmp_path: Path, provider: Script, **options: Any) -> Agent:
    for index in range(1, 9):
        (tmp_path / f"part{index}.txt").write_text(f"contents of part {index}")
    return Agent(
        llm=provider,
        embedding=SameEmbedding(),
        options=AgentOptions(
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=True,
            attempt_policy="none",
            **options,
        ),
    )


@pytest.fixture(autouse=True)
def _same_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-11-03 (UTC-05:00)"
    )


async def test_each_step_extends_the_previous_request(tmp_path: Path) -> None:
    provider = Script()
    agent = _agent(tmp_path, provider)
    try:
        provider.files = ["part1.txt"]
        assert (await agent.run_task("Read part1.txt")).status == "completed"
        first_run = len(provider.requests)
        provider.files = ["part2.txt", "part3.txt", "part4.txt"]
        result = await agent.run_task("Read part2.txt, part3.txt and part4.txt")
        assert result.status == "completed"
    finally:
        await agent.close()
    requests = [_wire(r) for r in provider.requests[first_run:]]
    assert len(requests) == 4
    for previous, current in zip(requests, requests[1:], strict=False):
        assert _extends(previous, current)
    # No request of either run carries a recalled block.
    for messages, _tools in provider.requests:
        assert not any(BLOCK in m.content for m in messages)
    assert result.to_dict()["prefix_cache"]["schema_changes"] == 0
