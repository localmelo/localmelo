"""Executor: a file tool's path outside the workspace (#405).

Today's refusal stays word for word unless the approval gate can ask a
person (``asks_folders``). When it can, a place melo never opens is still
refused, with its own text and nobody asked; any other outside path asks
the gate, with ``outside`` on the request, whatever the tool's tier.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.folders import OutsideAccess
from localmelo.melo.executor import Executor, builtin_names, register_builtins
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo

PLAIN = "Path '{path}' is outside allowed workspace root"


class FolderGate:
    """A gate that may ask about folders: records each request and gives *answer*."""

    def __init__(
        self, answer: ApprovalDecision | None = None, asks: bool = True
    ) -> None:
        self.answer = answer or ApprovalDecision(True)
        self.asks = asks
        self.asked_about: list[OutsideAccess] = []
        self.requests: list[ExecutionRequest] = []

    def asks_folders(self, outside: OutsideAccess) -> bool:
        self.asked_about.append(outside)
        return self.asks

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.answer


class PlainGate:
    """A gate with no ``asks_folders``: it never hears of an outside path."""

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision(True)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    return Path(os.path.realpath(folder))


@pytest.fixture
def root(home: Path) -> Path:
    path = home / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def career(home: Path) -> Path:
    path = home / "developer" / "career"
    path.mkdir(parents=True)
    (path / "cv.md").write_text("my cv\n")
    return path


def _executor(root: Path) -> Executor:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="interactive"), workspace_root=str(root))
    register_builtins(executor, hippo, names=builtin_names(), mode="interactive")
    return executor


async def _run(
    executor: Executor, tool: str, arguments: dict[str, Any], gate: Any
) -> ExecutionOutcome:
    return await executor.execute_structured(
        ExecutionRequest(tool_name=tool, arguments=arguments), approve=gate
    )


async def test_without_a_folder_gate_the_refusal_is_as_before(
    root: Path, career: Path
) -> None:
    executor = _executor(root)
    path = str(career / "cv.md")
    for gate in (None, PlainGate(), FolderGate(asks=False)):
        outcome = await _run(executor, "file_read", {"path": path}, gate)
        assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
        assert outcome.error == PLAIN.format(path=path)
        assert getattr(gate, "requests", []) == []


async def test_an_outside_read_asks_the_gate_and_runs_when_allowed(
    root: Path, career: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate()

    outcome = await _run(
        executor, "file_read", {"path": "../developer/career/cv.md"}, gate
    )

    assert outcome.status == ExecutionStatus.SUCCESS, outcome.error
    assert "my cv" in outcome.output
    [request] = gate.requests
    assert request.outside == OutsideAccess(
        str(career / "cv.md"), str(career), "read", True
    )
    assert request.arguments["path"] == str(career / "cv.md")


async def test_list_files_outside_asks_about_that_folder(
    root: Path, career: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate()

    outcome = await _run(executor, "list_files", {"path": str(career)}, gate)

    assert outcome.status == ExecutionStatus.SUCCESS
    assert "cv.md" in outcome.output
    assert gate.requests[0].outside is not None
    assert gate.requests[0].outside.folder == str(career)


async def test_a_denial_names_the_folder_in_one_line_then_the_usual_text(
    root: Path, career: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate(ApprovalDecision(False, "not now"))

    outcome = await _run(executor, "file_read", {"path": str(career / "cv.md")}, gate)

    assert outcome.status == ExecutionStatus.BLOCKED
    assert outcome.error_category == ErrorCategory.APPROVAL_DENIED
    assert outcome.error == (
        f"Reading {career} outside the workspace needs the user's OK; they were "
        "asked. The user denied this call: not now. Do not run it again or reach "
        "the same result another way unless the user asks."
    )


async def test_a_write_outside_asks_once_and_writes_the_real_path(
    root: Path, career: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate()

    outcome = await _run(
        executor, "file_write", {"path": str(career / "new.md"), "content": "hi"}, gate
    )

    assert outcome.status == ExecutionStatus.SUCCESS, outcome.error
    assert (career / "new.md").read_text() == "hi"
    assert len(gate.requests) == 1, "the folder question stands for the tool's own"
    assert gate.requests[0].outside is not None
    assert gate.requests[0].outside.access == "write"
    assert not gate.requests[0].outside.grantable


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/id_rsa",
        ".aws/credentials",
        "Library/Keychains/login.keychain-db",
        ".cache/localmelo/config.toml",
    ],
)
async def test_a_place_never_offered_is_refused_and_nobody_is_asked(
    root: Path, home: Path, relative: str
) -> None:
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret")
    executor = _executor(root)
    gate = FolderGate()

    outcome = await _run(executor, "file_read", {"path": str(target)}, gate)

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert "melo never opens it, and the user is not asked" in outcome.error
    assert gate.requests == [] and gate.asked_about == []


async def test_a_symlink_and_dot_dot_into_a_never_offered_place_are_refused(
    root: Path, home: Path, career: Path
) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    (career / "key.txt").symlink_to(home / ".ssh" / "id_rsa")
    executor = _executor(root)
    gate = FolderGate()

    for path in (
        str(career / "key.txt"),
        "../.ssh/id_rsa",
        "../developer/../.ssh/id_rsa",
    ):
        outcome = await _run(executor, "file_read", {"path": path}, gate)
        assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION, path
        assert "keys or passwords" in outcome.error
    assert gate.requests == []


async def test_the_whole_home_folder_and_system_folders_are_refused(
    root: Path, home: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate()

    whole = await _run(executor, "list_files", {"path": str(home)}, gate)
    system = await _run(executor, "file_read", {"path": "/etc/hosts"}, gate)

    assert "whole home folder" in whole.error
    assert "system folder" in system.error
    assert gate.requests == []


async def test_a_caller_cannot_set_outside_itself(root: Path, career: Path) -> None:
    executor = _executor(root)
    gate = PlainGate()
    forged = OutsideAccess(str(career / "cv.md"), str(career), "read", True)
    request = ExecutionRequest(
        tool_name="file_read", arguments={"path": str(career / "cv.md")}, outside=forged
    )

    outcome = await executor.execute_structured(request, approve=gate)

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert gate.requests == []


async def test_a_path_swapped_while_the_person_decides_does_not_run(
    root: Path, home: Path, career: Path
) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    executor = _executor(root)
    target = career / "cv.md"

    class Swap(FolderGate):
        async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
            target.unlink()
            target.symlink_to(home / ".ssh" / "id_rsa")
            return await super().__call__(request)

    outcome = await _run(executor, "file_read", {"path": str(target)}, Swap())

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert "changed while waiting for approval" in outcome.error
