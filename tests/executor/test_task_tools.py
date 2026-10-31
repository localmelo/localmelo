"""A task's own built-ins: git_status reads and never asks, file_edit edits and asks.

Neither is in any mode's default set or in ``--builtins all``; a worker names
them. git_status builds git's argv itself (no shell), refuses a path outside
the workspace and a revision that looks like an option, and leaves the
repository as it found it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from localmelo.melo.checker import DEFAULT_BUILTINS, TASK_BUILTINS, Checker
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.executor import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    call_preview,
    register_builtins,
)
from localmelo.melo.executor.git_status import (
    MAX_LOG_COUNT,
    MAX_OUTPUT_CHARS,
    git_argv,
    run_git_status,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.support.sessions.display import approval_target


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.py").write_text("one\ntwo\nthree\n")
    _git(root, "add", "a.py")
    _git(root, "commit", "-q", "-m", "First")
    (root / "a.py").write_text("one\nTWO\nthree\n")
    (root / "new.txt").write_text("new\n")
    return root


class Gate:
    def __init__(self) -> None:
        self.asked: list[ExecutionRequest] = []

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        self.asked.append(request)
        return ApprovalDecision(True)


def _executor(root: Path) -> Executor:
    hippo = Hippo()
    executor = Executor(hippo, Checker(mode="gateway"), workspace_root=str(root))
    register_builtins(
        executor,
        hippo,
        names={"git_status", "file_edit", "file_read"},
        mode="gateway",
        workspace_root=str(root),
    )
    return executor


def _tree(folder: Path) -> str:
    """A digest of every file under *folder*: names, sizes, times and bytes."""
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            info = path.stat()
            digest.update(f"{path}:{info.st_size}:{info.st_mtime_ns}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_no_mode_offers_them_by_default() -> None:
    assert {"git_status", "file_edit"} == TASK_BUILTINS
    for names in DEFAULT_BUILTINS.values():
        assert not names & TASK_BUILTINS


async def test_each_read_shows_the_repository(repo: Path) -> None:
    status = await run_git_status(str(repo), "status")
    assert status.splitlines()[0] == "## main"
    assert " M a.py" in status and "?? new.txt" in status
    assert "main" in await run_git_status(str(repo), "branch")
    assert "First" in await run_git_status(str(repo), "log", count=3)
    stat = await run_git_status(str(repo), "diff", stat=True)
    assert "a.py" in stat and "1 file changed" in stat
    diff = await run_git_status(str(repo), "diff", paths=["a.py"])
    assert "-two" in diff and "+TWO" in diff
    assert "a.py" in await run_git_status(str(repo), "show", rev="HEAD")
    assert await run_git_status(str(repo), "show", rev="HEAD:a.py") == "one\ntwo\nthree"


async def test_reads_leave_the_repository_as_it_was(repo: Path) -> None:
    before = _tree(repo / ".git")
    for command in ("status", "branch", "log", "diff", "show"):
        await run_git_status(str(repo), command)
    assert _tree(repo / ".git") == before


@pytest.mark.parametrize(
    ("command", "options", "message"),
    [
        ("commit", {}, "command must be one of"),
        ("checkout", {}, "command must be one of"),
        ("log", {"rev": "--output=/tmp/x"}, "Not a revision"),
        ("diff", {"rev": "-p"}, "Not a revision"),
        ("show", {"rev": "HEAD; rm -rf ."}, "Not a revision"),
        ("diff", {"paths": ["../outside"]}, "outside allowed workspace root"),
        ("diff", {"paths": ["/etc/hosts"]}, "outside allowed workspace root"),
        ("diff", {"paths": ["a.py"] * 21}, "At most 20 paths"),
        ("status", {"paths": ["a.py"]}, "paths apply to log and diff"),
        ("log", {"stat": True}, "stat and staged apply to diff only"),
        ("branch", {"rev": "main"}, "rev applies to log, diff and show"),
        ("log", {"count": 0}, "count must be a positive integer"),
        ("show", {"rev": "HEAD:../outside"}, "outside allowed workspace root"),
        ("show", {"rev": "HEAD:.env"}, "may hold passwords or keys"),
        ("diff", {"paths": ["config/.env.local"]}, "may hold passwords or keys"),
        ("log", {"paths": ["keys/id_ed25519"]}, "may hold passwords or keys"),
    ],
)
def test_a_read_outside_its_bounds_is_refused(
    repo: Path, command: str, options: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        git_argv(str(repo), command, **options)


def test_a_link_out_of_the_workspace_is_refused(repo: Path, tmp_path: Path) -> None:
    (repo / "out").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="outside allowed workspace root"):
        git_argv(str(repo), "diff", paths=["out/x"])


def test_the_argv_is_bounded_and_runs_no_program_of_the_repository(
    repo: Path,
) -> None:
    argv = git_argv(str(repo), "log", count=500, paths=["a.py"], rev="HEAD")
    assert f"-n{MAX_LOG_COUNT}" in argv
    assert argv[-2:] == ["--", "a.py"]
    for setting in ("core.fsmonitor=false", "core.hooksPath=/dev/null"):
        assert setting in argv
    assert "--no-optional-locks" in argv
    assert "--no-ext-diff" in git_argv(str(repo), "diff")


async def test_a_long_output_is_cut(repo: Path) -> None:
    (repo / "big.txt").write_text("x\n" * 30_000)
    _git(repo, "add", "big.txt")
    text = await run_git_status(str(repo), "diff", staged=True)
    assert len(text) <= MAX_OUTPUT_CHARS
    assert "[cut:" in text


async def test_outside_a_repository_the_error_is_git_s(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="git status failed"):
        await run_git_status(str(tmp_path), "status")


async def test_git_status_never_asks(repo: Path) -> None:
    executor = _executor(repo)
    gate = Gate()
    assert not executor.requires_approval("git_status")
    outcome = await executor.execute_structured(
        ExecutionRequest("git_status", {"command": "status"}), approve=gate
    )
    assert outcome.status == ExecutionStatus.SUCCESS
    assert "a.py" in outcome.output
    assert gate.asked == []


async def test_file_edit_replaces_the_one_match_and_asks(repo: Path) -> None:
    executor = _executor(repo)
    gate = Gate()
    assert executor.requires_approval("file_edit")
    outcome = await executor.execute_structured(
        ExecutionRequest("file_edit", {"path": "a.py", "old": "TWO", "new": "2"}),
        approve=gate,
    )
    assert outcome.status == ExecutionStatus.SUCCESS, outcome.error
    assert (repo / "a.py").read_text() == "one\n2\nthree\n"
    assert [request.arguments["path"] for request in gate.asked] == [
        os.path.realpath(repo / "a.py")
    ]


@pytest.mark.parametrize(
    ("old", "message"),
    [
        ("missing", "is not in"),
        ("e", "occurs 3 times"),
        ("", "old must not be empty"),
    ],
)
async def test_file_edit_refuses_a_missing_or_ambiguous_text(
    repo: Path, old: str, message: str
) -> None:
    outcome = await _executor(repo).execute_structured(
        ExecutionRequest("file_edit", {"path": "a.py", "old": old, "new": "x"}),
        approve=Gate(),
    )
    assert outcome.status != ExecutionStatus.SUCCESS
    assert message in outcome.error
    assert (repo / "a.py").read_text() == "one\nTWO\nthree\n"


async def test_file_edit_outside_the_workspace_is_refused_before_asking(
    repo: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n")
    gate = Gate()
    outcome = await _executor(repo).execute_structured(
        ExecutionRequest(
            "file_edit", {"path": str(outside), "old": "keep", "new": "gone"}
        ),
        approve=gate,
    )
    assert outcome.error_category == ErrorCategory.PATH_POLICY_VIOLATION
    assert gate.asked == []
    assert outside.read_text() == "keep\n"


def test_the_approval_card_shows_an_edit_as_a_diff(repo: Path) -> None:
    target = approval_target(
        {
            "tool_name": "file_edit",
            "arguments": {"path": str(repo / "a.py"), "old": "TWO", "new": "2"},
            "workspace_root": str(repo),
        }
    )
    assert target is not None
    assert target["path_label"] == "a.py"
    diff = target["diff"]
    assert (diff["added"], diff["removed"], diff["note"]) == (1, 1, "")
    assert {"kind": "removed", "line": 2, "text": "TWO"} in diff["rows"]
    assert {"kind": "added", "line": 2, "text": "2"} in diff["rows"]
    missing = approval_target(
        {
            "tool_name": "file_edit",
            "arguments": {"path": str(repo / "a.py"), "old": "nope", "new": "2"},
        }
    )
    assert missing is not None
    assert "is not in the file" in missing["diff"]["note"]


def test_the_previews_name_the_edit_and_the_read() -> None:
    assert call_preview("file_edit", {"path": "a.py", "old": "x", "new": "y"}) == (
        "a.py (edit)"
    )
    assert call_preview("git_status", {"command": "log", "rev": "HEAD~2"}) == (
        "git log HEAD~2"
    )
