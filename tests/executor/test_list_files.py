"""Executor: ``list_files``, a read-only listing of one folder of the workspace.

It names folders (with a ``/``) and files (with their size), folders first,
never reads a file or follows a link, and is fenced by the executor's
workspace policy like ``file_read``.
"""

from __future__ import annotations

import builtins
import os
import stat
from pathlib import Path
from typing import Any

import pytest

import localmelo.melo.executor.builtins as tool
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ExecutorResultPayload
from localmelo.melo.checker.validators import MAX_OUTPUT_LEN, TRUNCATION_SUFFIX
from localmelo.melo.executor import (
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return Path(os.path.realpath(path))


async def _run(root: Path | None, **arguments: Any) -> ExecutionOutcome:
    hippo, top = Hippo(), None if root is None else str(root)
    executor = Executor(hippo, Checker(mode="interactive"), workspace_root=top)
    register_builtins(executor, hippo, mode="interactive", workspace_root=top)
    request = ExecutionRequest(tool_name="list_files", arguments=arguments)
    return await executor.execute_structured(request)


async def _text(root: Path, **arguments: Any) -> str:
    outcome = await _run(root, **arguments)
    assert outcome.status == ExecutionStatus.SUCCESS, outcome.error
    return outcome.output


def _tree(root: Path) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "Docs").mkdir()
    (root / "src" / "pkg" / "mod.py").write_bytes(b"x" * 3000)
    (root / "src" / "app.py").write_bytes(b"x")
    (root / "b.txt").write_bytes(b"x" * 5)
    (root / "a.txt").write_bytes(b"x" * 2_500_000)


async def test_folders_first_then_files_sorted_with_sizes(root: Path) -> None:
    _tree(root)
    expected = "Docs/\nsrc/\na.txt  2.4 MB\nb.txt  5 B"

    # No path, an empty one and a dot all mean the root.
    for arguments in ({}, {"path": ""}, {"path": "."}):
        assert await _text(root, **arguments) == expected
    assert await _text(root, path="src") == "pkg/\napp.py  1 B"


async def test_depth_goes_down_indented_and_is_kept_between_1_and_3(
    root: Path,
) -> None:
    _tree(root)

    assert (
        await _text(root, path="src", depth=2) == "pkg/\n  mod.py  2.9 KB\napp.py  1 B"
    )
    assert await _text(root, depth=0) == await _text(root, depth=1)
    assert "mod.py" not in await _text(root, depth=1)
    assert await _text(root, depth=9) == await _text(root, depth=3)
    assert "    mod.py  2.9 KB" in await _text(root, depth=9)


async def test_git_is_skipped_at_every_level_and_nothing_to_show_is_empty(
    root: Path,
) -> None:
    (root / ".git").mkdir()
    (root / "sub").mkdir()
    (root / "sub" / ".git").write_text("gitdir: elsewhere")

    assert await _text(root, depth=2) == "sub/"
    assert await _text(root, path="sub") == "(empty)"


async def test_after_the_cap_the_rest_is_counted_up_to_a_limit(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for number in range(tool.LIST_FILES_MAX_ENTRIES + 7):
        (root / f"f{number:03}").write_bytes(b"")

    lines = (await _text(root)).splitlines()

    assert len(lines) == tool.LIST_FILES_MAX_ENTRIES + 1
    assert (lines[0], lines[-1]) == ("f000  0 B", "... and 7 more")
    monkeypatch.setattr(tool, "_LIST_FILES_COUNT_LIMIT", 4)
    assert (await _text(root)).splitlines()[-1] == "... and 4+ more"


async def test_links_are_shown_and_never_followed_nor_any_file_read(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOKEN")
    (root / "real").mkdir()
    (root / "real" / "inner.txt").write_text("hi")
    (root / "out").symlink_to(outside)  # a folder link that points out
    (root / "same").symlink_to(root / "real")  # one that points inside
    (root / "file").symlink_to(outside / "secret.txt")

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("list_files read a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(os, "read", refuse)

    text = await _text(root, depth=3)

    assert text == "real/\n  inner.txt  2 B\nfile@\nout@\nsame@"
    assert "TOKEN" not in text and "secret" not in text


async def test_pipes_and_odd_names_cannot_break_the_listing(root: Path) -> None:
    os.mkfifo(root / "pipe")
    (root / "two\nlines").write_text("x")

    assert await _text(root) == "pipe\ntwo\\nlines  1 B"


async def test_a_folder_that_cannot_be_read_is_marked(root: Path) -> None:
    (root / "closed").mkdir()
    (root / "closed").chmod(0)
    try:
        if os.access(root / "closed", os.R_OK):
            pytest.skip("permissions are not enforced for this user")
        assert await _text(root, depth=2) == "closed/\n  (unreadable)"
    finally:
        (root / "closed").chmod(stat.S_IRWXU)


@pytest.mark.parametrize("path", ["/", "..", "src/../..", "out", "ELSEWHERE"])
async def test_a_path_outside_the_root_is_refused(
    root: Path, tmp_path: Path, path: str
) -> None:
    (tmp_path / "elsewhere").mkdir()
    (root / "src").mkdir()
    (root / "out").symlink_to(tmp_path / "elsewhere")
    path = str(tmp_path / "elsewhere") if path == "ELSEWHERE" else path

    outcome = await _run(root, path=path)

    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert outcome.error == f"Path '{path}' is outside allowed workspace root"
    assert outcome.output == ""


async def test_a_file_or_a_missing_folder_is_an_error_the_model_can_read(
    root: Path,
) -> None:
    (root / "notes.txt").write_text("x")

    file, missing = await _run(root, path="notes.txt"), await _run(root, path="nope")

    assert file.error_category == missing.error_category == ErrorCategory.RUNTIME_ERROR
    assert "Not a directory" in file.error and "No such file" in missing.error


async def test_the_result_check_still_caps_the_output(root: Path) -> None:
    for number in range(tool.LIST_FILES_MAX_ENTRIES):
        (root / f"{number:03}-{'n' * 250}").write_bytes(b"")
    outcome = await _run(root)
    payload = ExecutorResultPayload("list_files", outcome.output, "", 0.0)

    checked = Checker().check_executor_result(payload).sanitized_payload

    assert len(outcome.output) > MAX_OUTPUT_LEN
    assert checked is not None
    assert len(checked.output) == MAX_OUTPUT_LEN + len(TRUNCATION_SUFFIX)


async def test_without_a_root_it_lists_the_process_folder(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree(root)
    monkeypatch.chdir(root)

    assert (await _run(None)).output == "Docs/\nsrc/\na.txt  2.4 MB\nb.txt  5 B"
    assert (await _run(None, path="src")).output == "pkg/\napp.py  1 B"
