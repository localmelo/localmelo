"""Opt-in fixed system prefix and per-run date contract."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent import _chat
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef


class Provider(BaseLLMProvider):
    model = "fixture"

    def __init__(self) -> None:
        self.requests: list[tuple[list[Message], list[ToolDef] | None]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if len(self.requests) % 2:
            return Message(
                role="assistant",
                tool_calls=[ToolCall("file_read", {"path": "sample.txt"}, id="c1")],
            )
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


async def test_prefix_frozen_across_steps_and_runs(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sample.txt").write_text("evidence")
    dates = iter(
        [
            "Date: 2026-10-28 (UTC-04:00)",
            "Date: 2026-10-28 (UTC-04:00)",
            "Date: 2026-10-29 (UTC-04:00)",
        ]
    )
    monkeypatch.setattr(_chat, "local_date_line", lambda: next(dates), raising=False)
    provider = Provider()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            fixed_prefix=True,
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            tool_exposure="all",
        ),
    )
    try:
        for _ in range(3):
            assert (await agent.run_task("Read sample.txt")).status == "completed"
    finally:
        await agent.close()
    systems = [messages[0].content for messages, _ in provider.requests]
    assert len(systems) == 6 and len(set(systems[:4])) == 1
    assert systems[4] == systems[5]
    assert systems[4].replace("2026-10-29", "2026-10-28") == systems[0]
    assert all("Date: 2026-10-28 (UTC-04:00)" in s for s in systems[:4])
    assert provider.requests[0][1] == provider.requests[1][1]
