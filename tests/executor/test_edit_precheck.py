"""A file_edit that cannot apply fails at once and never asks.

The text to replace must occur exactly once. It is checked before the
approval gate is asked, so an edit with no match, or with several, costs
the person no click; only an edit that can apply asks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.executor import (
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo

TEXT = "alpha\nbeta\nbeta again\ngamma\n"


class Gate:
    def __init__(self) -> None:
        self.asked: list[ExecutionRequest] = []

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        self.asked.append(request)
        return ApprovalDecision(True)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    (path / "a.txt").write_text(TEXT)
    return path


def _executor(root: Path) -> Executor:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="gateway"), workspace_root=str(root))
    register_builtins(
        executor,
        hippo,
        names={"file_edit"},
        mode="gateway",
        workspace_root=str(root),
    )
    return executor


async def _edit(
    executor: Executor, gate: Gate | None, path: str, old: str
) -> ExecutionOutcome:
    request = ExecutionRequest("file_edit", {"path": path, "old": old, "new": "NEW"})
    return await executor.execute_structured(request, approve=gate)


async def test_an_edit_with_one_match_asks_and_applies(root: Path) -> None:
    gate = Gate()

    outcome = await _edit(_executor(root), gate, "a.txt", "gamma")

    assert outcome.status == ExecutionStatus.SUCCESS
    assert len(gate.asked) == 1
    assert (root / "a.txt").read_text() == TEXT.replace("gamma", "NEW")


async def test_an_edit_with_no_match_fails_without_asking(root: Path) -> None:
    gate = Gate()

    outcome = await _edit(_executor(root), gate, "a.txt", "delta")

    assert gate.asked == []
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert "is not in" in outcome.error and "read the file again" in outcome.error
    assert (root / "a.txt").read_text() == TEXT


async def test_an_edit_with_two_matches_fails_without_asking(root: Path) -> None:
    gate = Gate()

    outcome = await _edit(_executor(root), gate, "a.txt", "beta")

    assert gate.asked == []
    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.INVALID_ARGUMENTS
    assert "occurs 2 times" in outcome.error and "more lines around it" in outcome.error
    assert (root / "a.txt").read_text() == TEXT


async def test_an_empty_text_and_a_missing_file_fail_without_asking(
    root: Path,
) -> None:
    gate = Gate()
    executor = _executor(root)

    empty = await _edit(executor, gate, "a.txt", "")
    missing = await _edit(executor, gate, "nope.txt", "x")

    assert gate.asked == []
    assert "old must not be empty" in empty.error
    assert missing.error == (
        f"{os.path.realpath(root) + '/nope.txt'} does not exist; "
        "file_edit changes an existing file"
    )


async def test_a_text_that_changes_while_waiting_still_fails_when_it_runs(
    root: Path,
) -> None:
    class Racing(Gate):
        async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
            (root / "a.txt").write_text("nothing left\n")
            return await super().__call__(request)

    gate = Racing()

    outcome = await _edit(_executor(root), gate, "a.txt", "gamma")

    assert len(gate.asked) == 1
    assert outcome.status == ExecutionStatus.ERROR
    assert "is not in" in outcome.error


async def test_a_replaced_file_tool_keeps_asking(root: Path) -> None:
    # Registering a tool again drops its precheck: a stand-in is asked as before.
    executor = _executor(root)

    async def stand_in(path: str, old: str, new: str) -> str:
        return "edited"

    executor.register("file_edit", stand_in, requires_approval=True)
    gate = Gate()

    outcome = await _edit(executor, gate, "a.txt", "delta")

    assert outcome.status == ExecutionStatus.SUCCESS and len(gate.asked) == 1
