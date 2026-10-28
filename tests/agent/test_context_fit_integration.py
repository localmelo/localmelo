"""Request fitting across the agent, provider error and event boundaries."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.providers._http import post_json


class Scripted(BaseLLMProvider):
    model = "fixture"

    def __init__(self, replies: list[Message | ProviderError]) -> None:
        self.replies = iter(replies)
        self.requests: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(copy.deepcopy(messages))
        reply = next(self.replies)
        if isinstance(reply, ProviderError):
            raise reply
        return reply

    async def close(self) -> None:
        pass


async def test_overflow_retry_shrinks_request_without_rerunning_tool(
    tmp_path: Path,
) -> None:
    (tmp_path / "input.txt").write_text("evidence " * 1000)
    llm = Scripted(
        [
            Message(
                role="assistant",
                tool_calls=[ToolCall("file_read", {"path": "input.txt"}, id="c1")],
            ),
            ProviderError(
                "context_length_exceeded", status_code=400, context_overflow=True
            ),
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            context_window=20000,
            output_reserve=100,
            safety_margin=100,
        ),
    )
    events: list[AgentEvent] = []
    try:
        result = await agent.run_task("Read input.txt", on_event=events.append)
    finally:
        await agent.close()
    assert result.status == "completed" and result.n_tool == 1
    assert len(llm.requests) == 3
    before = next(m.content for m in llm.requests[1] if m.role == "tool")
    after = next(m.content for m in llm.requests[2] if m.role == "tool")
    assert len(after) < len(before) and "characters omitted" in after
    assert [m.tool_call_id for m in llm.requests[1] if m.role == "tool"] == ["c1"]
    assert [m.tool_call_id for m in llm.requests[2] if m.role == "tool"] == ["c1"]
    records = result.to_dict()["context_budget"]
    assert len(records) == 3 and records[-1]["pruned"]["tool_results"] == 1
    assert sum(e.type == "tool.started" for e in events) == 1


async def test_initial_fit_sends_preview_without_changing_history(
    tmp_path: Path,
) -> None:
    output = "evidence " * 1000
    (tmp_path / "input.txt").write_text(output)
    llm = Scripted(
        [
            Message(
                role="assistant",
                tool_calls=[ToolCall("file_read", {"path": "input.txt"}, id="c1")],
            ),
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            context_window=3000,
            output_reserve=100,
            safety_margin=100,
        ),
    )
    try:
        result = await agent.run_task("Read input.txt")
        window = agent.hippo.working.get_window()
        history = await agent.hippo.history.get_steps(result.task_id)
        assert result.status == "completed" and result.n_tool == 1
        assert len(llm.requests) == 2
        assert "characters omitted" in next(
            m.content for m in llm.requests[1] if m.role == "tool"
        )
        assert any(m.role == "tool" and output in m.content for m in window)
        assert (
            history[0].tool_result is not None
            and output in history[0].tool_result.output
        )
        assert result.to_dict()["context_budget"][-1]["pruned"]["tool_results"] == 1
    finally:
        await agent.close()


async def test_second_overflow_does_not_retry_again(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("evidence " * 1000)

    def overflow() -> ProviderError:
        return ProviderError(
            "context_length_exceeded", status_code=400, context_overflow=True
        )

    llm = Scripted(
        [
            Message(
                role="assistant",
                tool_calls=[ToolCall("file_read", {"path": "input.txt"}, id="c1")],
            ),
            overflow(),
            overflow(),
        ]
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            memory=False,
            context_window=20000,
            output_reserve=100,
            safety_margin=100,
        ),
    )
    try:
        result = await agent.run_task("Read input.txt")
    finally:
        await agent.close()
    assert result.status == "context_budget_exceeded" and result.n_tool == 1
    assert len(llm.requests) == 3


async def test_impossible_request_is_a_noncrash_status() -> None:
    llm = Scripted([])
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            memory=False,
            context_window=30,
            output_reserve=10,
            safety_margin=10,
        ),
    )
    try:
        result = await agent.run_task("中文🙂" * 50)
    finally:
        await agent.close()
    assert result.status == "context_budget_exceeded"
    assert "increase" in result.error and llm.requests == []


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, {"error": {"code": "context_length_exceeded"}}, True),
        (
            400,
            {
                "error": {
                    "message": "This model's maximum context length is 8192 tokens"
                }
            },
            True,
        ),
        (400, {"error": "maximum context length is 8192 tokens"}, True),
        (429, {"error": "maximum context length is 8192 tokens"}, False),
        (400, {"error": {"message": "bad request"}}, False),
        (429, {"error": {"code": "context_length_exceeded"}}, False),
        (401, {"error": {"code": "context_length_exceeded"}}, False),
    ],
)
async def test_only_explicit_context_errors_are_classified(
    status: int, body: dict[str, Any], expected: bool
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=body)
        ),
        base_url="http://fixture",
    ) as client:
        with pytest.raises(ProviderError) as caught:
            await post_json(client, "/chat", {})
    assert caught.value.context_overflow is expected
