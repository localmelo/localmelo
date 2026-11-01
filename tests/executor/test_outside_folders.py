"""Executor: a file tool's path outside the workspace (#405).

Today's refusal stays word for word unless the approval gate can ask a
person (``asks_folders``). When it can, a place melo never opens is still
refused, with its own text and nobody asked; any other outside path asks
the gate, with ``outside`` on the request, whatever the tool's tier. Paths
are named by the file system: a macOS alias (``/.vol`` and others) or
another letter case counts as what it names, and the tool opens only the
file that was checked.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.folders import OutsideAccess, fd_path
from localmelo.melo.executor import Executor, builtin_names, register_builtins
from localmelo.melo.executor import executor as executor_module
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
def letters(home: Path) -> Path:
    path = home / "Documents" / "letters"
    path.mkdir(parents=True)
    (path / "cover-letter.md").write_text("my letter\n")
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
    root: Path, letters: Path
) -> None:
    executor = _executor(root)
    path = str(letters / "cover-letter.md")
    for gate in (None, PlainGate(), FolderGate(asks=False)):
        outcome = await _run(executor, "file_read", {"path": path}, gate)
        assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
        assert outcome.error == PLAIN.format(path=path)
        assert getattr(gate, "requests", []) == []


async def test_an_outside_read_asks_the_gate_and_runs_when_allowed(
    root: Path, letters: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate()

    outcome = await _run(
        executor, "file_read", {"path": "../Documents/letters/cover-letter.md"}, gate
    )

    assert outcome.status == ExecutionStatus.SUCCESS, outcome.error
    assert "my letter" in outcome.output
    [request] = gate.requests
    assert request.outside == OutsideAccess(
        str(letters / "cover-letter.md"), str(letters), "read", True
    )
    assert request.arguments["path"] == str(letters / "cover-letter.md")


async def test_list_files_outside_asks_about_that_folder(
    root: Path, letters: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate()

    outcome = await _run(executor, "list_files", {"path": str(letters)}, gate)

    assert outcome.status == ExecutionStatus.SUCCESS
    assert "cover-letter.md" in outcome.output
    assert gate.requests[0].outside is not None
    assert gate.requests[0].outside.folder == str(letters)


async def test_a_denial_names_the_folder_in_one_line_then_the_usual_text(
    root: Path, letters: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate(ApprovalDecision(False, "not now"))

    outcome = await _run(
        executor, "file_read", {"path": str(letters / "cover-letter.md")}, gate
    )

    assert outcome.status == ExecutionStatus.BLOCKED
    assert outcome.error_category == ErrorCategory.APPROVAL_DENIED
    assert outcome.error == (
        f"Reading {letters} outside the workspace needs the user's OK; they were "
        "asked. The user denied this call: not now. Do not run it again or reach "
        "the same result another way unless the user asks."
    )


async def test_a_write_outside_asks_once_and_writes_the_real_path(
    root: Path, letters: Path
) -> None:
    executor = _executor(root)
    gate = FolderGate()

    outcome = await _run(
        executor, "file_write", {"path": str(letters / "new.md"), "content": "hi"}, gate
    )

    assert outcome.status == ExecutionStatus.SUCCESS, outcome.error
    assert (letters / "new.md").read_text() == "hi"
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
    assert gate.requests == []


async def test_a_gate_that_cannot_ask_keeps_the_old_refusal_for_every_place(
    root: Path, home: Path
) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    executor = _executor(root)
    gate = FolderGate(asks=False)

    for path in (str(home / ".ssh" / "id_rsa"), str(home / "x.txt"), "/etc/hosts"):
        outcome = await _run(executor, "file_read", {"path": path}, gate)
        assert outcome.error == PLAIN.format(path=path)
    assert gate.requests == []


async def test_a_symlink_and_dot_dot_into_a_never_offered_place_are_refused(
    root: Path, home: Path, letters: Path
) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    (letters / "key.txt").symlink_to(home / ".ssh" / "id_rsa")
    executor = _executor(root)
    gate = FolderGate()

    for path in (
        str(letters / "key.txt"),
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


async def test_a_caller_cannot_set_outside_itself(root: Path, letters: Path) -> None:
    executor = _executor(root)
    gate = PlainGate()
    forged = OutsideAccess(str(letters / "cover-letter.md"), str(letters), "read", True)
    request = ExecutionRequest(
        tool_name="file_read",
        arguments={"path": str(letters / "cover-letter.md")},
        outside=forged,
    )

    outcome = await executor.execute_structured(request, approve=gate)

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert gate.requests == []


async def test_a_path_swapped_while_the_person_decides_does_not_run(
    root: Path, home: Path, letters: Path
) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("key")
    executor = _executor(root)
    target = letters / "cover-letter.md"

    class Swap(FolderGate):
        async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
            target.unlink()
            target.symlink_to(home / ".ssh" / "id_rsa")
            return await super().__call__(request)

    outcome = await _run(executor, "file_read", {"path": str(target)}, Swap())

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert "changed while waiting for approval" in outcome.error


async def test_a_part_swapped_after_the_last_check_is_not_opened(
    root: Path, home: Path, letters: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool's own descriptor must name the checked file, after every check."""
    (home / "keys").mkdir()
    (home / "keys" / "cover-letter.md").write_text("PRIVATE")
    probe = os.open(letters / "cover-letter.md", os.O_RDONLY)
    nameable = fd_path(probe) is not None
    os.close(probe)
    if not nameable:
        pytest.skip("this system cannot name a descriptor")
    # Step 9's check passes: the swap lands between it and the open.
    monkeypatch.setattr(
        executor_module, "_resolved_again", lambda policy, resolved, outside: resolved
    )

    class Swap(FolderGate):
        async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
            letters.rename(letters.parent / "letters-old")
            letters.symlink_to(home / "keys")
            return await super().__call__(request)

    outcome = await _run(
        _executor(root), "file_read", {"path": str(letters / "cover-letter.md")}, Swap()
    )

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert "changed before it was opened" in outcome.error
    assert "PRIVATE" not in (outcome.output or "")


async def test_a_write_opens_the_checked_folder_only(
    root: Path, home: Path, letters: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (home / "elsewhere").mkdir()
    probe = os.open(letters, os.O_RDONLY)
    nameable = fd_path(probe) is not None
    os.close(probe)
    if not nameable:
        pytest.skip("this system cannot name a descriptor")
    monkeypatch.setattr(
        executor_module, "_resolved_again", lambda policy, resolved, outside: resolved
    )

    class Swap(FolderGate):
        async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
            letters.rename(letters.parent / "letters-old")
            letters.symlink_to(home / "elsewhere")
            return await super().__call__(request)

    outcome = await _run(
        _executor(root),
        "file_write",
        {"path": str(letters / "new.md"), "content": "x"},
        Swap(),
    )

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert "changed before it was opened" in outcome.error
    assert not (home / "elsewhere" / "new.md").exists()


async def test_listing_an_outside_folder_never_enters_a_place_never_offered(
    root: Path, home: Path
) -> None:
    library = home / "Library"
    (library / "Keychains").mkdir(parents=True)
    (library / "Keychains" / "login.keychain-db").write_text("k")
    (library / "Recipes").mkdir()
    (library / "Recipes" / "soup.md").write_text("s")
    executor = _executor(root)

    outcome = await _run(
        executor, "list_files", {"path": str(library), "depth": 2}, FolderGate()
    )

    assert outcome.status == ExecutionStatus.SUCCESS, outcome.error
    lines = outcome.output.splitlines()
    assert "Keychains/ (not offered)" in lines
    assert "login.keychain-db" not in outcome.output
    assert "Recipes/" in lines and any("soup.md" in line for line in lines)


async def test_listing_inside_the_workspace_is_as_before(root: Path) -> None:
    (root / ".ssh").mkdir()
    (root / ".ssh" / "known.txt").write_text("k")
    executor = _executor(root)

    outcome = await _run(executor, "list_files", {"path": ".", "depth": 2}, None)

    assert ".ssh/" in outcome.output.splitlines()
    assert "known.txt" in outcome.output


# ── Aliases (macOS) ──────────────────────────────────────────────────────────

needs_aliases = pytest.mark.skipif(
    not (sys.platform == "darwin" and os.path.isdir("/.vol")),
    reason="no file-system aliases here",
)


def _vol(path: Path) -> str:
    info = os.stat(path)
    return f"/.vol/{info.st_dev}/{info.st_ino}"


def _forms(folder: Path, name: str) -> list[str]:
    path = folder / name
    return [
        _vol(path),
        f"{_vol(folder)}/{name}",
        f"/.resolve/1{path}",
        f"/.nofollow{path}",
        "/.VOL" + _vol(path)[len("/.vol") :],
    ]


@needs_aliases
@pytest.mark.parametrize("form", range(5))
@pytest.mark.parametrize("tool", ["file_read", "file_write", "list_files"])
async def test_an_alias_is_refused_and_nobody_is_asked(
    root: Path, home: Path, letters: Path, form: int, tool: str
) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("PRIVATE")
    executor = _executor(root)
    gate = FolderGate()
    if tool == "list_files":
        paths = [_forms(home, ".ssh")[form], _forms(letters.parent, "letters")[form]]
    else:
        paths = [
            _forms(home / ".ssh", "id_rsa")[form],
            _forms(letters, "cover-letter.md")[form],
        ]
    arguments: dict[str, Any] = {"content": "x"} if tool == "file_write" else {}

    for path in paths:
        outcome = await _run(executor, tool, {"path": path, **arguments}, gate)
        assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION, path
        assert "melo never opens it" in outcome.error
        assert "PRIVATE" not in (outcome.output or "")
    assert gate.requests == []
    assert (home / ".ssh" / "id_rsa").read_text() == "PRIVATE"
    assert (letters / "cover-letter.md").read_text() == "my letter\n"


@needs_aliases
async def test_an_alias_into_the_workspace_is_the_workspace(root: Path) -> None:
    (root / "notes.txt").write_text("inside")
    executor = _executor(root)

    outcome = await _run(
        executor, "file_read", {"path": _vol(root / "notes.txt")}, None
    )

    assert (outcome.status, outcome.output) == (ExecutionStatus.SUCCESS, "inside")
    assert [a.path for a in outcome.artifacts] == [str(root / "notes.txt")]
