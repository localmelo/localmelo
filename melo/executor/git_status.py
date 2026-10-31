"""``git_status``: a task's read-only look at its git repository.

The tool never takes a command line. It takes one of five words and builds
git's argv itself, so a call cannot run anything but these reads:

- ``status``: ``git status --short --branch``;
- ``branch``: ``git branch --list -vv``;
- ``log``: ``git log --oneline -n <count> [<rev>] [-- <paths>]``, at most
  :data:`MAX_LOG_COUNT` commits;
- ``diff``: ``git diff [--stat] [--cached] [<rev>] [-- <paths>]``;
- ``show``: ``git show --stat <rev>`` (``HEAD`` without one).

No shell runs, and git runs without hooks, external diff or text conversion
programs, a pager, an fsmonitor, a terminal prompt or the optional index
refresh (``GIT_OPTIONAL_LOCKS=0``), so nothing it runs can write. A path
must resolve inside the root and a revision must look like one (no leading
``-``, no spaces): anything else is refused before git starts. The output
keeps its first :data:`MAX_OUTPUT_CHARS` characters.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Sequence
from typing import Any, Final

from localmelo.melo.executor.env import child_env
from localmelo.melo.schema import ToolDef

COMMANDS: Final = ("status", "branch", "log", "diff", "show")
MAX_LOG_COUNT: Final = 50
DEFAULT_LOG_COUNT: Final = 10
MAX_PATHS: Final = 20
MAX_OUTPUT_CHARS: Final = 20_000
TIMEOUT_S: Final = 20.0
# A revision or range: a name, a hash, HEAD~2, main..HEAD, stash@{0}.
_REV: Final = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./~^@{}:-]{0,199}")
# Settings that would let a read run a program or write a file.
_SAFE: Final = (
    "core.fsmonitor=false",
    "core.hooksPath=/dev/null",
    "core.pager=cat",
    "diff.external=",
    "color.ui=false",
)

GIT_STATUS_DEF: Final = ToolDef(
    name="git_status",
    description=(
        "Read the git state of the workspace without changing anything: "
        "status, branch, log, diff or show. It never asks for approval; "
        "use it instead of running git through python_exec."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": list(COMMANDS),
                "description": "What to read: status, branch, log, diff or show",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files or folders in the workspace (log and diff)",
            },
            "rev": {
                "type": "string",
                "description": "A commit, branch or range, such as HEAD~1 or "
                "main..HEAD (log, diff and show)",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_LOG_COUNT,
                "description": f"Commits to list (log; default {DEFAULT_LOG_COUNT})",
            },
            "stat": {
                "type": "boolean",
                "description": "Only the changed files and line counts (diff)",
            },
            "staged": {
                "type": "boolean",
                "description": "The staged changes instead of the unstaged (diff)",
            },
        },
        "required": ["command"],
    },
    semantic_tags=["git", "status", "diff", "log", "branch", "commit", "changes"],
)


def git_argv(
    root: str,
    command: str,
    *,
    paths: Sequence[str] | None = None,
    rev: str = "",
    count: int = DEFAULT_LOG_COUNT,
    stat: bool = False,
    staged: bool = False,
) -> list[str]:
    """The argv of one read, or ``ValueError`` saying why it is refused."""
    if command not in COMMANDS:
        raise ValueError(f"command must be one of {', '.join(COMMANDS)}")
    if rev and (not isinstance(rev, str) or not _REV.fullmatch(rev)):
        raise ValueError(f"Not a revision: {rev!r}")
    if paths is not None and (
        isinstance(paths, str) or not all(isinstance(p, str) for p in paths)
    ):
        raise ValueError("paths must be a list of strings")
    fenced = _fenced(root, list(paths or []))
    if fenced and command not in ("log", "diff"):
        raise ValueError(f"paths apply to log and diff, not {command}")
    if (stat or staged) and command != "diff":
        raise ValueError("stat and staged apply to diff only")
    if rev and command not in ("log", "diff", "show"):
        raise ValueError(f"rev applies to log, diff and show, not {command}")
    git = shutil.which("git") or "git"
    head = [git, "--no-pager", "--no-optional-locks"]
    for setting in _SAFE:
        head += ["-c", setting]
    if command == "status":
        return [*head, "status", "--short", "--branch"]
    if command == "branch":
        return [*head, "branch", "--list", "-vv"]
    if command == "show":
        return [
            *head,
            "show",
            "--stat",
            "--no-ext-diff",
            "--no-textconv",
            rev or "HEAD",
        ]
    if command == "log":
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        argv = [*head, "log", "--oneline", f"-n{min(count, MAX_LOG_COUNT)}"]
    else:
        argv = [*head, "diff", "--no-ext-diff", "--no-textconv"]
        argv += ["--stat"] if stat else []
        argv += ["--cached"] if staged else []
    argv += [rev] if rev else []
    return [*argv, "--", *fenced] if fenced else argv


def _fenced(root: str, paths: list[str]) -> list[str]:
    """*paths* relative to *root*; one outside it, or too many, is refused."""
    if len(paths) > MAX_PATHS:
        raise ValueError(f"At most {MAX_PATHS} paths")
    real_root = os.path.realpath(root)
    out = []
    for path in paths:
        if not path or "\0" in path:
            raise ValueError("A path must be a non-empty string")
        real = os.path.realpath(os.path.join(real_root, path))
        if os.path.commonpath([real_root, real]) != real_root:
            raise ValueError(f"Path '{path}' is outside allowed workspace root")
        out.append(os.path.relpath(real, real_root))
    return out


async def run_git_status(root: str, command: str, **options: Any) -> str:
    """Run one read in *root*; its output, or the error git printed."""
    argv = git_argv(root, command, **options)
    env = child_env(
        {"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"}
    )
    for name in ("GIT_EXTERNAL_DIFF", "GIT_DIR", "GIT_WORK_TREE"):
        env.pop(name, None)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=root,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_S)
    except BaseException:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        raise
    text = out.decode(errors="replace")
    if proc.returncode:
        problem = err.decode(errors="replace").strip() or text.strip()
        raise RuntimeError(f"git {command} failed: {problem[:2000]}")
    if not text.strip():
        return "(nothing)" if command != "status" else "(clean)"
    if len(text) > MAX_OUTPUT_CHARS:
        note = f"\n... [cut: {len(text)} characters in all]"
        return text[: MAX_OUTPUT_CHARS - len(note)] + note
    return text.rstrip("\n")


__all__ = [
    "COMMANDS",
    "GIT_STATUS_DEF",
    "MAX_LOG_COUNT",
    "git_argv",
    "run_git_status",
]
