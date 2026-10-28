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


def test_multibyte_preview_cut_counts_characters() -> None:
    original = "🙂汉" * 40_000
    result, _ = preview_result(ToolResult("sample", output=original))
    body, marker = result.output.split("\n... [truncated: ", 1)
    assert len(result.output) <= PREVIEW_LIMIT
    assert int(marker.split()[0]) == len(original) - len(body)
    assert result.output.encode("utf-8").decode("utf-8") == result.output


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
    text = secret + " " + "x" * (PREVIEW_LIMIT + 500) + " tail"
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
        assert secret in tool.content
        assert tool.tool_result is not None and "tail" not in tool.tool_result.output
        assert secret not in tool.tool_result.output
        finished = next(event for event in events if event.type == "tool.finished")
        assert "tail" not in str(finished.data) and secret not in str(finished.data)
    finally:
        await agent.close()


async def test_redaction_precedes_preview_cut() -> None:
    secret = "sk-" + "a" * 80
    text = "x" * (PREVIEW_LIMIT - 80) + " " + secret + " tail"
    provider = Provider()
    agent = Agent(
        llm=provider,
        options=AgentOptions(
            captured_history=True, tool_exposure="all", builtins=frozenset()
        ),
    )
    agent.hippo.register_tool(ToolDef("sample", "sample", {}))

    async def sample() -> str:
        return text

    agent.executor.register("sample", sample)
    events: list[AgentEvent] = []
    try:
        assert (
            await agent.run_task("run sample", on_event=events.append)
        ).status == "completed"
        tool = next(m for m in provider.requests[1] if m.role == "tool")
        assert "sk-" in tool.content and secret not in tool.content
        assert tool.tool_result is not None and "sk-" not in tool.tool_result.output
        assert "sk-" not in str(
            next(event.data for event in events if event.type == "tool.finished")
        )
    finally:
        await agent.close()


async def test_reopen_preserves_metadata_and_reads_legacy_row(tmp_path: Path) -> None:
    from localmelo.melo.memory.history.sqlite import SqliteHistory
    from localmelo.melo.schema import StepRecord, TaskRecord

    path = tmp_path / "history.db"
    history = SqliteHistory(path)
    task = TaskRecord("task")
    await history.save_task(task)
    await history.add_step(
        task.task_id, StepRecord(tool_result=ToolResult("sample", output="legacy"))
    )
    metadata = {"output_omitted_chars": 3, "acquisition_omitted_chars": None}
    await history.add_step(
        task.task_id,
        StepRecord(
            tool_result=ToolResult("sample", output="checked"),
            capture_metadata=metadata,
        ),
    )
    await history.aclose()
    reopened = SqliteHistory(path)
    try:
        rows = await reopened.get_steps(task.task_id)
        assert [row.capture_metadata for row in rows] == [None, metadata]
        assert [row.tool_result.output for row in rows if row.tool_result] == [
            "legacy",
            "checked",
        ]
    finally:
        await reopened.aclose()


async def test_failed_history_write_does_not_persist_raw_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-" + "a" * 32
    agent = Agent(
        llm=Provider(),
        options=AgentOptions(
            captured_history=True,
            memory_dir=str(tmp_path),
            tool_exposure="all",
            builtins=frozenset(),
        ),
    )
    agent.hippo.register_tool(ToolDef("sample", "sample", {}))

    async def sample() -> str:
        return secret + " raw"

    async def fail_write(task_id: str, step: object) -> str:
        raise RuntimeError("injected history failure")

    agent.executor.register("sample", sample)
    monkeypatch.setattr(agent.hippo, "store_step", fail_write)
    events: list[AgentEvent] = []
    try:
        result = await agent.run_task("run sample", on_event=events.append)
        assert result.status == "crashed"
        assert await agent.hippo.history.get_steps(result.task_id) == []
        assert secret not in str(
            next(event.data for event in events if event.type == "tool.finished")
        )
    finally:
        await agent.close()
    assert all(
        secret.encode() not in path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    )


async def test_run_tool_keeps_legacy_result_with_capture_enabled() -> None:
    outputs = []
    for enabled in (False, True):
        agent = Agent(
            llm=Provider(),
            options=AgentOptions(captured_history=enabled, builtins=frozenset()),
        )
        agent.hippo.register_tool(ToolDef("sample", "sample", {}))

        async def sample() -> str:
            return "x" * 60_000

        agent.executor.register("sample", sample)
        try:
            outputs.append((await agent.run_tool("sample", {})).output)
        finally:
            await agent.close()
    assert outputs[0] == outputs[1]


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
        assert rows[0].capture_metadata["output_omitted_chars"] == 0
        assert rows[0].capture_metadata["acquisition_omitted_chars"] is None
    finally:
        await agent.close()


async def test_multiple_calls_record_exact_success_and_error_omissions() -> None:
    class TwoCalls(Provider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            self.requests.append(copy.deepcopy(messages))
            if len(self.requests) == 1:
                return Message(
                    "assistant",
                    tool_calls=[
                        ToolCall("sample", {"n": 1}, id="c1"),
                        ToolCall("sample", {"n": 2}, id="c2"),
                    ],
                )
            return Message("assistant", "done")

    agent = Agent(
        llm=TwoCalls(),
        options=AgentOptions(
            captured_history=True, tool_exposure="all", builtins=frozenset()
        ),
    )
    agent.hippo.register_tool(
        ToolDef(
            "sample",
            "sample",
            {"type": "object", "properties": {"n": {"type": "integer"}}},
        )
    )

    async def sample(n: int) -> ToolOutput:
        return ToolOutput(
            "文" * (CAPTURE_LIMIT + (17 if n == 1 else 29)), is_error=n == 2
        )

    agent.executor.register("sample", sample)
    try:
        result = await agent.run_task("run sample")
        rows = await agent.hippo.history.get_steps(result.task_id)
        assert len(rows) == 2
        first, second = [row.capture_metadata for row in rows]
        assert first is not None and second is not None
        assert (first["output_omitted_chars"], first["error_omitted_chars"]) == (17, 0)
        assert (second["output_omitted_chars"], second["error_omitted_chars"]) == (
            0,
            29,
        )
        assert (
            first["acquisition_omitted_chars"] is None
            and second["acquisition_omitted_chars"] is None
        )
    finally:
        await agent.close()


async def test_tool_error_is_bounded_and_stored_with_status(tmp_path: Path) -> None:
    secret = "sk-" + "a" * 32
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
        return ToolOutput(
            secret + " " + "failure " * 10_000 + " error-end", is_error=True
        )

    agent.executor.register("sample", sample)
    events: list[AgentEvent] = []
    try:
        result = await agent.run_task("run sample", on_event=events.append)
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
            tool.tool_result is not None
            and "error-end" not in tool.tool_result.error
            and secret not in tool.tool_result.error
        )
        assert secret not in str(
            next(event.data for event in events if event.type == "tool.finished")
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
            tool.tool_result is not None
            and len(tool.tool_result.output) <= PREVIEW_LIMIT
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
