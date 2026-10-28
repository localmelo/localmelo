"""Executor: a built-in tool's argument of the wrong JSON type never runs.

The executor answers with an INVALID_ARGUMENTS error before the checker, the
workspace policy and the tool see the call. Before, a list or a number where
the schema declares a string raised outside the guarded call and crashed the
run. Provided tools are left alone: their servers check their own arguments.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker import Checker
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tools import ProvidedTool
from localmelo.melo.executor import Executor, ToolOutput, register_builtins
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import CheckResult, Message, ToolCall, ToolDef


def _with_builtins(
    checker: Checker | None = None, workspace_root: str | None = None
) -> Executor:
    hippo = Hippo()
    executor = Executor(
        hippo, checker or Checker(mode="interactive"), workspace_root=workspace_root
    )
    register_builtins(executor, hippo, mode="interactive")
    return executor


class _BlockEverything(Checker):
    async def pre_execute(
        self, tool_call: ToolCall, tool_def: ToolDef | None
    ) -> CheckResult:
        return CheckResult(allowed=False, reason="blocked by the test")


@pytest.mark.parametrize(
    ("command", "json_type"),
    [
        (["ls", "-la"], "array"),
        (None, "null"),
        (42, "integer"),
        (1.5, "number"),
        # bool is an int subclass in Python, but not a JSON integer.
        (True, "boolean"),
        ({"cmd": "ls"}, "object"),
    ],
    ids=["array", "null", "integer", "number", "boolean", "object"],
)
async def test_shell_exec_command_of_the_wrong_type(
    command: Any, json_type: str
) -> None:
    executor = _with_builtins()
    calls: list[dict[str, Any]] = []

    async def spy(**arguments: Any) -> str:
        calls.append(arguments)
        return ""

    executor.register("shell_exec", spy)

    outcome = await executor.execute_structured(
        ExecutionRequest(tool_name="shell_exec", arguments={"command": command})
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == (
        f"Arguments for shell_exec: 'command' must be of type string, "
        f"not {json_type}"
    )
    assert calls == []


async def test_rejected_before_the_checker() -> None:
    outcome = await _with_builtins(_BlockEverything()).execute_structured(
        ExecutionRequest(tool_name="shell_exec", arguments={"command": ["ls"]})
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS


async def test_file_path_of_the_wrong_type_under_a_workspace_root(
    tmp_path: Path,
) -> None:
    # The workspace policy resolves the path outside the guarded call, so a
    # list here used to raise TypeError out of the executor.
    outcome = await _with_builtins(workspace_root=str(tmp_path)).execute_structured(
        ExecutionRequest(tool_name="file_read", arguments={"path": ["notes.txt"]})
    )
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == (
        "Arguments for file_read: 'path' must be of type string, not array"
    )


async def test_file_write_content_of_the_wrong_type_leaves_the_file(
    tmp_path: Path,
) -> None:
    # file_write opens the file for writing before it writes, so a content
    # that is not text used to empty the file and then fail.
    target = tmp_path / "notes.txt"
    target.write_text("keep me")

    outcome = await _with_builtins().execute_structured(
        ExecutionRequest(
            tool_name="file_write", arguments={"path": str(target), "content": 42}
        )
    )
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == (
        "Arguments for file_write: 'content' must be of type string, not integer"
    )
    assert target.read_text() == "keep me"


async def test_arguments_of_the_declared_types_still_run() -> None:
    outcome = await _with_builtins().execute_structured(
        ExecutionRequest(tool_name="python_exec", arguments={"code": "print(6 * 7)"})
    )
    assert outcome.status == ExecutionStatus.SUCCESS
    assert outcome.output == "42"


async def test_provided_tool_arguments_are_left_to_the_tool() -> None:
    calls: list[dict[str, Any]] = []

    async def click(**arguments: Any) -> ToolOutput:
        calls.append(arguments)
        return ToolOutput(text="clicked")

    tool = ProvidedTool(
        definition=ToolDef(
            name="mcp__gui__click",
            description="Click at a point.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
        ),
        function=click,
        timeout_ms=5_000,
    )
    hippo = Hippo()
    executor = Executor(hippo, Checker())
    hippo.register_tool(tool.definition)
    executor.register(tool.definition.name, tool.function, timeout_ms=tool.timeout_ms)

    outcome = await executor.execute_structured(
        ExecutionRequest(tool_name="mcp__gui__click", arguments={"x": "10", "y": 20})
    )
    assert outcome.status == ExecutionStatus.SUCCESS
    assert calls == [{"x": "10", "y": 20}]


class _FakeLLM(BaseLLMProvider):
    """Pops queued responses and records every request."""

    def __init__(self, responses: list[Message]) -> None:
        self.responses: deque[Message] = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return Message(role="assistant", content="(no response queued)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


async def test_the_run_goes_on_after_a_wrong_type() -> None:
    llm = _FakeLLM(
        [
            # A Codex-style shell call: the command as an argv array.
            Message(
                role="assistant",
                content="",
                tool_call=ToolCall(
                    tool_name="shell_exec", arguments={"command": ["ls", "-la"]}
                ),
            ),
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(tool_exposure="all", builtins=frozenset({"shell_exec"})),
    )
    try:
        result = await agent.run_task("list the files")
    finally:
        await agent.close()

    assert (result.status, result.result) == ("completed", "done")
    assert [m.content for m in llm.calls[1]["messages"] if m.role == "tool"] == [
        "Error: Arguments for shell_exec: 'command' must be of type string, "
        "not array"
    ]
