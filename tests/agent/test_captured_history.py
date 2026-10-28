"""Captured history is distinct from the bounded model preview."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._captured_history import (
    CAPTURE_LIMIT,
    PREVIEW_LIMIT,
    capture_result,
    preview_result,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor import ToolOutput
from localmelo.melo.schema import ImagePart, Message, ToolCall, ToolDef, ToolResult


class Provider(BaseLLMProvider):
    model = "fixture"

    def __init__(self) -> None:
        self.requests: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(copy.deepcopy(messages))
        if len(self.requests) == 1:
            return Message("assistant", tool_calls=[ToolCall("sample", {}, id="c1")])
        return Message("assistant", "done")

    async def close(self) -> None:
        pass


def test_capture_is_bounded_redacted_and_does_not_mutate() -> None:
    secret = "sk-" + "a" * 32
    raw = ToolResult("sample", output=secret + "🙂" * (CAPTURE_LIMIT + 20))
    result, metadata = capture_result(raw)
    assert secret not in result.output and raw.output.startswith(secret)
    assert len(result.output) <= CAPTURE_LIMIT
    assert metadata["output_omitted_chars"] > 0
    assert metadata["acquisition_omitted_chars"] is None


def test_second_preview_keeps_exact_omitted_count() -> None:
    from localmelo.melo.agent.agent import _cut_to

    original = "x" * 100_000
    preview, cut = preview_result(ToolResult("sample", output=original))
    again = _cut_to(preview.output, 10_000, cut_before=cut)
    body, marker = again.split("\n... [truncated: ", 1)
    assert int(marker.split()[0]) == len(original) - len(body)


def test_preview_bounds_errors_and_preserves_image_drop_notes() -> None:
    raw = ToolResult("sample", output="unused" * 20_000, error="failed🙂" * 20_000)
    preview, _ = preview_result(raw, ["[1 image dropped: invalid]"])
    assert len(preview.error) + len(preview.output) <= PREVIEW_LIMIT
    assert "[1 image dropped: invalid]" in preview.error
    assert "truncated:" in preview.error
    assert len(raw.error) > PREVIEW_LIMIT


def test_capture_error_before_output_and_exact_redacted_counts() -> None:
    raw = ToolResult("sample", output="o" * 100, error="错误" * CAPTURE_LIMIT)
    result, info = capture_result(raw)
    assert len(result.error) + len(result.output) == CAPTURE_LIMIT
    assert info["error_omitted_chars"] == len(raw.error) - len(result.error)
    assert info["output_omitted_chars"] == len(raw.output) - len(result.output)
    assert info["acquisition_omitted_chars"] is None


@pytest.mark.parametrize("memory", [True, False])
async def test_history_is_separate_from_model_and_event_preview(
    tmp_path: Path, memory: bool
) -> None:
    secret = "sk-" + "a" * 32
    text = "x" * (PREVIEW_LIMIT + 500) + " " + secret + " tail"
    provider = Provider()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            captured_history=True,
            memory=memory,
            memory_dir=str(tmp_path),
            tool_exposure="all",
            builtins=frozenset(),
        ),
    )
    agent.hippo.register_tool(ToolDef("sample", "sample", {}))

    async def sample() -> str:
        return text

    agent.executor.register("sample", sample)
    events: list[AgentEvent] = []
    try:
        result = await agent.run_task("run sample", on_event=events.append)
        history = await agent.hippo.history.get_steps(result.task_id)
        assert result.status == "completed" and len(history) == 1
        stored = history[0].tool_result
        assert (
            stored is not None
            and "tail" in stored.output
            and secret not in stored.output
        )
        assert history[0].capture_metadata is not None
        tool = next(m for m in provider.requests[1] if m.role == "tool")
        assert len(tool.content) <= PREVIEW_LIMIT and "tail" not in tool.content
        assert tool.tool_result is not None and "tail" not in tool.tool_result.output
        finished = next(event for event in events if event.type == "tool.finished")
        assert "tail" not in str(finished.data)
    finally:
        await agent.close()


async def test_crash_after_capture_keeps_only_redacted_history(tmp_path: Path) -> None:
    secret = "sk-" + "a" * 32

    class Crash(Provider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            self.requests.append(copy.deepcopy(messages))
            if len(self.requests) == 1:
                return Message(
                    "assistant", tool_calls=[ToolCall("sample", {}, id="c1")]
                )
            raise RuntimeError("test provider crash")

    agent = Agent(
        llm=Crash(),
        options=AgentOptions(
            captured_history=True,
            memory_dir=str(tmp_path),
            tool_exposure="all",
            builtins=frozenset(),
        ),
    )
    agent.hippo.register_tool(ToolDef("sample", "sample", {}))

    async def sample() -> str:
        return "prefix " + secret + " suffix"

    agent.executor.register("sample", sample)
    try:
        result = await agent.run_task("run sample")
        rows = await agent.hippo.history.get_steps(result.task_id)
        assert result.status == "crashed" and len(rows) == 1
        assert (
            rows[0].tool_result is not None and secret not in rows[0].tool_result.output
        )
        assert rows[0].capture_metadata is not None
    finally:
        await agent.close()


async def test_overflow_retry_does_not_duplicate_capture() -> None:
    from localmelo.melo.contracts.providers import ProviderError
    from localmelo.tests.agent.test_context_fit_integration import Scripted

    provider = Scripted(
        [
            Message("assistant", tool_calls=[ToolCall("sample", {}, id="c1")]),
            ProviderError(
                "context_length_exceeded", status_code=400, context_overflow=True
            ),
            Message("assistant", "done"),
        ]
    )
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            captured_history=True,
            tool_exposure="all",
            builtins=frozenset(),
            context_window=20_000,
            output_reserve=100,
            safety_margin=100,
        ),
    )
    agent.hippo.register_tool(ToolDef("sample", "sample", {}))
    calls = 0

    async def sample() -> str:
        nonlocal calls
        calls += 1
        return "evidence " * 1000

    agent.executor.register("sample", sample)
    try:
        result = await agent.run_task("run sample")
        rows = await agent.hippo.history.get_steps(result.task_id)
        assert result.status == "completed" and calls == 1 and len(rows) == 1
        assert rows[0].capture_metadata is not None
    finally:
        await agent.close()


async def test_tool_error_is_bounded_and_stored_with_status(tmp_path: Path) -> None:
    provider = Provider()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            captured_history=True,
            memory_dir=str(tmp_path),
            tool_exposure="all",
            builtins=frozenset(),
        ),
    )
    agent.hippo.register_tool(ToolDef("sample", "sample", {}))

    async def sample() -> ToolOutput:
        return ToolOutput("failure " * 10_000 + " error-end", is_error=True)

    agent.executor.register("sample", sample)
    try:
        result = await agent.run_task("run sample")
        rows = await agent.hippo.history.get_steps(result.task_id)
        assert (
            rows[0].tool_result is not None and "error-end" in rows[0].tool_result.error
        )
        assert (
            rows[0].capture_metadata is not None
            and rows[0].capture_metadata["status"] == "error"
        )
        tool = next(m for m in provider.requests[1] if m.role == "tool")
        assert len(tool.content) <= PREVIEW_LIMIT and "error-end" not in tool.content
        assert (
            tool.tool_result is not None and "error-end" not in tool.tool_result.error
        )
        assert (tmp_path / "history.db").stat().st_mode & 0o077 == 0
    finally:
        await agent.close()


async def test_image_drop_notice_survives_model_fitting() -> None:
    class TwoCalls(Provider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            self.requests.append(copy.deepcopy(messages))
            if len(self.requests) == 1:
                return Message(
                    "assistant",
                    tool_calls=[
                        ToolCall("sample", {}, id="c1"),
                        ToolCall("sample", {}, id="c2"),
                    ],
                )
            return Message("assistant", "done")

    provider = TwoCalls()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            captured_history=True, tool_exposure="all", builtins=frozenset()
        ),
    )
    agent.hippo.register_tool(ToolDef("sample", "sample", {}))

    async def sample() -> ToolOutput:
        return ToolOutput(
            "x" * 100_000, [ImagePart.from_bytes(b"image", "application/invalid")]
        )

    agent.executor.register("sample", sample)
    try:
        result = await agent.run_task("run sample")
        assert result.status == "completed"
        tools = [m for m in provider.requests[1] if m.role == "tool"]
        assert len(tools) == 2
        rows = await agent.hippo.history.get_steps(result.task_id)
        assert len(rows) == 2
        assert all(
            row.capture_metadata is not None
            and row.capture_metadata["images_omitted"] == 1
            for row in rows
        )
        assert all(
            row.tool_result is not None and not row.tool_result.images for row in rows
        )
        assert all(
            len(tool.content) <= PREVIEW_LIMIT and "image dropped:" in tool.content
            for tool in tools
        )
        assert all(
            tool.tool_result is not None and tool.tool_result.output in tool.content
            for tool in tools
        )
    finally:
        await agent.close()


async def test_cancel_preserves_prior_captured_call_in_order(tmp_path: Path) -> None:
    class TwoCalls(Provider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            return Message(
                "assistant",
                tool_calls=[
                    ToolCall("sample", {"n": 1}, id="c1"),
                    ToolCall("sample", {"n": 2}, id="c2"),
                ],
            )

    entered = asyncio.Event()
    agent = Agent(
        llm=TwoCalls(),
        options=AgentOptions(
            captured_history=True,
            memory_dir=str(tmp_path),
            tool_exposure="all",
            builtins=frozenset(),
        ),
    )
    agent.hippo.register_tool(
        ToolDef(
            "sample",
            "sample",
            {"type": "object", "properties": {"n": {"type": "integer"}}},
        )
    )

    async def sample(n: int) -> str:
        if n == 2:
            entered.set()
            await asyncio.Event().wait()
        return "x" * (PREVIEW_LIMIT + 100) + " captured-end"

    agent.executor.register("sample", sample)
    events: list[AgentEvent] = []
    running = asyncio.create_task(agent.run_task("run sample", on_event=events.append))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        task_id = events[0].run_id
        history = await agent.hippo.history.get_steps(task_id)
        assert [step.tool_call.id for step in history if step.tool_call] == ["c1", "c2"]
        assert (
            history[0].tool_result is not None
            and "captured-end" in history[0].tool_result.output
        )
        assert history[0].capture_metadata is not None
        assert history[1].capture_metadata is None
    finally:
        await agent.close()
