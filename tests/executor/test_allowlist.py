"""Executor: ``allowed`` refuses registered tools the caller did not offer.

The agent passes the names of the tools it offered in a planning call.
A call to any other registered tool gets a TOOL_NOT_OFFERED error and the
tool function is not called. ``allowed=None`` keeps the old behavior.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.executor import Executor, register_builtins
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import RAW_ARGUMENTS_KEY, CheckResult, ToolCall, ToolDef

NOT_OFFERED = "Tool 'echo' was not offered in this step."


class _BlockEverything(Checker):
    async def pre_execute(
        self, tool_call: ToolCall, tool_def: ToolDef | None
    ) -> CheckResult:
        return CheckResult(allowed=False, reason="blocked by the test")


def _echo_executor(
    calls: list[dict[str, Any]], checker: Checker | None = None
) -> Executor:
    """An executor with the built-ins plus an ``echo`` tool that records calls."""
    hippo = Hippo()
    executor = Executor(hippo, checker or Checker())
    register_builtins(executor, hippo)

    async def echo(**arguments: Any) -> str:
        calls.append(arguments)
        return str(arguments.get("text", ""))

    hippo.register_tool(
        ToolDef(
            name="echo",
            description="echo text back",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        )
    )
    executor.register("echo", echo)
    return executor


def _echo(text: str = "hi") -> ExecutionRequest:
    return ExecutionRequest(tool_name="echo", arguments={"text": text})


async def test_tool_not_offered_is_not_called() -> None:
    calls: list[dict[str, Any]] = []
    outcome = await _echo_executor(calls).execute_structured(
        _echo(), allowed={"python_exec"}
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.TOOL_NOT_OFFERED
    assert outcome.error_category == "tool_not_offered"
    assert outcome.error == NOT_OFFERED
    assert outcome.output == ""
    assert calls == []


async def test_empty_allowlist_refuses_every_tool() -> None:
    calls: list[dict[str, Any]] = []
    outcome = await _echo_executor(calls).execute_structured(_echo(), allowed=set())
    assert outcome.error_category == ErrorCategory.TOOL_NOT_OFFERED
    assert calls == []


@pytest.mark.parametrize(
    "allowed",
    [["echo"], ("python_exec", "echo"), {"echo"}, frozenset({"echo"}), {"echo": 1}],
    ids=["list", "tuple", "set", "frozenset", "dict"],
)
async def test_offered_tool_runs(allowed: Collection[str]) -> None:
    calls: list[dict[str, Any]] = []
    outcome = await _echo_executor(calls).execute_structured(_echo(), allowed=allowed)
    assert outcome.status == ExecutionStatus.SUCCESS
    assert outcome.output == "hi"
    assert calls == [{"text": "hi"}]


async def test_no_allowlist_is_unchanged() -> None:
    calls: list[dict[str, Any]] = []
    executor = _echo_executor(calls)
    default = await executor.execute_structured(_echo("a"))
    explicit = await executor.execute_structured(_echo("b"), allowed=None)
    assert (default.status, default.output) == (ExecutionStatus.SUCCESS, "a")
    assert (explicit.status, explicit.output) == (ExecutionStatus.SUCCESS, "b")
    assert calls == [{"text": "a"}, {"text": "b"}]


@pytest.mark.parametrize(
    "allowed",
    [frozenset(), frozenset({"no_such_tool"})],
    ids=["not_listed", "listed"],
)
async def test_unknown_tool_is_still_not_found(allowed: frozenset[str]) -> None:
    outcome = await _echo_executor([]).execute_structured(
        ExecutionRequest(tool_name="no_such_tool"), allowed=allowed
    )
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.TOOL_NOT_FOUND


async def test_not_offered_wins_over_raw_arguments() -> None:
    calls: list[dict[str, Any]] = []
    outcome = await _echo_executor(calls).execute_structured(
        ExecutionRequest(tool_name="echo", arguments={RAW_ARGUMENTS_KEY: "hi"}),
        allowed=frozenset(),
    )
    assert outcome.error_category == ErrorCategory.TOOL_NOT_OFFERED
    assert outcome.error == NOT_OFFERED
    assert calls == []


async def test_offered_with_raw_arguments_is_invalid() -> None:
    calls: list[dict[str, Any]] = []
    outcome = await _echo_executor(calls).execute_structured(
        ExecutionRequest(tool_name="echo", arguments={RAW_ARGUMENTS_KEY: "hi"}),
        allowed=frozenset({"echo"}),
    )
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert outcome.error == "Arguments for echo are not a JSON object: hi"
    assert calls == []


@pytest.mark.parametrize(
    ("allowed", "status", "category"),
    [
        (frozenset(), ExecutionStatus.ERROR, ErrorCategory.TOOL_NOT_OFFERED),
        (
            frozenset({"shell_exec"}),
            ExecutionStatus.BLOCKED,
            ErrorCategory.BLOCKED_BY_CHECKER,
        ),
    ],
    ids=["not_offered", "offered"],
)
async def test_allowlist_comes_before_the_checker(
    allowed: frozenset[str], status: ExecutionStatus, category: ErrorCategory
) -> None:
    outcome = await _echo_executor([], _BlockEverything()).execute_structured(
        ExecutionRequest(tool_name="shell_exec", arguments={"command": "ls"}),
        allowed=allowed,
    )
    assert outcome.status == status
    assert outcome.error_category == category
