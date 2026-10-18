"""Executor: a tool call whose arguments are not a JSON object never runs.

Providers keep such arguments as ``{RAW_ARGUMENTS_KEY: raw}``. The
executor answers with an INVALID_ARGUMENTS error, before the checker and
without calling the tool function.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.tools import ProvidedTool
from localmelo.melo.executor import BUILTINS, Executor, ToolOutput, register_builtins
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import RAW_ARGUMENTS_KEY, CheckResult, ToolCall, ToolDef


def _with_builtins(checker: Checker | None = None) -> Executor:
    hippo = Hippo()
    executor = Executor(hippo, checker or Checker())
    register_builtins(executor, hippo)
    return executor


def _raw(value: Any) -> dict[str, Any]:
    return {RAW_ARGUMENTS_KEY: value}


class _BlockEverything(Checker):
    async def pre_execute(
        self, tool_call: ToolCall, tool_def: ToolDef | None
    ) -> CheckResult:
        return CheckResult(allowed=False, reason="blocked by the test")


async def test_builtin_tool_is_not_called() -> None:
    executor = _with_builtins()
    python_exec = next(fn for td, fn in BUILTINS if td.name == "python_exec")
    calls: list[dict[str, Any]] = []

    async def spy(**arguments: Any) -> str:
        calls.append(arguments)
        return await python_exec(**arguments)

    executor.register("python_exec", spy)

    # The model sent the code itself instead of {"code": ...}.
    outcome = await executor.execute_structured(
        ExecutionRequest(tool_name="python_exec", arguments=_raw("print(6 * 7)"))
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == (
        "Arguments for python_exec are not a JSON object: print(6 * 7)"
    )
    assert outcome.output == ""
    assert calls == []

    # The same callable runs once the arguments parse.
    ok = await executor.execute_structured(
        ExecutionRequest(tool_name="python_exec", arguments={"code": "print(6 * 7)"})
    )
    assert ok.output == "42"
    assert calls == [{"code": "print(6 * 7)"}]


async def test_provided_tool_is_not_called() -> None:
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

    raw = '{"x": 10, "y": ' + "9" * 300  # cut off in the middle
    outcome = await executor.execute_structured(
        ExecutionRequest(tool_name="mcp__gui__click", arguments=_raw(raw))
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    # Only the first 200 characters of the raw text are quoted.
    assert outcome.error == (
        f"Arguments for mcp__gui__click are not a JSON object: {raw[:200]}"
    )
    assert calls == []


async def test_rejected_before_the_checker() -> None:
    outcome = await _with_builtins(_BlockEverything()).execute_structured(
        ExecutionRequest(tool_name="shell_exec", arguments=_raw("ls"))
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    ("raw", "quoted"),
    [
        ([1, 2], "[1, 2]"),
        # A number cannot be sliced, so it must be turned into text first.
        (7, "7"),
        # A long list is cut after 200 characters of its text, not 200 items.
        (list(range(100)), str(list(range(100)))[:200]),
        # Falsy values: the executor must test for the key, not for a
        # truthy value under it.
        (0, "0"),
        (False, "False"),
    ],
    ids=["list", "number", "long_list", "zero", "false"],
)
async def test_non_string_raw_value_is_quoted_as_text(raw: Any, quoted: str) -> None:
    outcome = await _with_builtins().execute_structured(
        ExecutionRequest(tool_name="python_exec", arguments=_raw(raw))
    )
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == f"Arguments for python_exec are not a JSON object: {quoted}"


async def test_unknown_tool_is_still_not_found() -> None:
    outcome = await _with_builtins().execute_structured(
        ExecutionRequest(tool_name="no_such_tool", arguments=_raw("{"))
    )
    assert outcome.error_category == ErrorCategory.TOOL_NOT_FOUND
