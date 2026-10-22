"""Executor: rich tool output (text + images), tool errors, per-tool timeouts."""

from __future__ import annotations

import asyncio
from typing import Any

from localmelo.melo.checker import Checker
from localmelo.melo.executor import Executor, ToolOutput
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import ImagePart, ToolDef

# A 1x1 PNG.
PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
    "AAAABJRU5ErkJggg=="
)


def _executor(name: str, fn: Any, **register_kwargs: Any) -> Executor:
    hippo = Hippo()
    hippo.register_tool(
        ToolDef(name=name, description=name, parameters={"type": "object"})
    )
    executor = Executor(hippo, Checker())
    executor.register(name, fn, **register_kwargs)
    return executor


async def test_tool_output_carries_images() -> None:
    image = ImagePart(mime_type="image/png", data=PNG_1X1)

    async def shot() -> ToolOutput:
        return ToolOutput(text="captured", images=[image])

    outcome = await _executor("shot", shot).execute_structured(
        ExecutionRequest(tool_name="shot")
    )
    assert outcome.status == ExecutionStatus.SUCCESS
    assert outcome.output == "captured"
    assert outcome.images == [image]
    assert outcome.to_tool_result().images == [image]


async def test_tool_reported_error_keeps_images() -> None:
    image = ImagePart(mime_type="image/png", data=PNG_1X1)

    async def broken() -> ToolOutput:
        return ToolOutput(text="window not found", images=[image], is_error=True)

    outcome = await _executor("broken", broken).execute_structured(
        ExecutionRequest(tool_name="broken")
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.TOOL_ERROR
    assert outcome.error == "window not found"
    assert outcome.output == ""
    assert outcome.images == [image]


async def test_tool_reported_error_without_text() -> None:
    async def broken() -> ToolOutput:
        return ToolOutput(is_error=True)

    outcome = await _executor("broken", broken).execute_structured(
        ExecutionRequest(tool_name="broken")
    )
    assert outcome.error == "Tool reported an error"


async def test_plain_string_output_unchanged() -> None:
    async def plain() -> str:
        return "text only"

    outcome = await _executor("plain", plain).execute_structured(
        ExecutionRequest(tool_name="plain")
    )
    assert outcome.output == "text only"
    assert outcome.images == []


async def _sleep_forever(**_: Any) -> str:
    await asyncio.sleep(30)
    return "never"


async def test_per_tool_timeout() -> None:
    executor = _executor("slow", _sleep_forever, timeout_ms=50)
    outcome = await executor.execute_structured(ExecutionRequest(tool_name="slow"))
    assert outcome.status == ExecutionStatus.TIMEOUT


async def test_request_timeout_beats_tool_timeout() -> None:
    executor = _executor("slow", _sleep_forever, timeout_ms=30_000)
    outcome = await executor.execute_structured(
        ExecutionRequest(tool_name="slow", timeout_ms=50)
    )
    assert outcome.status == ExecutionStatus.TIMEOUT


async def test_reregister_clears_tool_timeout() -> None:
    executor = _executor("slow", _sleep_forever, timeout_ms=50)
    executor.register("slow", _sleep_forever)
    assert "slow" not in executor._timeouts_ms


async def test_argument_names_need_not_be_identifiers() -> None:
    """MCP tools may take arguments such as "file-path"."""
    seen: dict[str, Any] = {}

    async def any_args(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "ok"

    outcome = await _executor("mcp_tool", any_args).execute_structured(
        ExecutionRequest(tool_name="mcp_tool", arguments={"file-path": "a.txt"})
    )
    assert outcome.status == ExecutionStatus.SUCCESS
    assert seen == {"file-path": "a.txt"}
