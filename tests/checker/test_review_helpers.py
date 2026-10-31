"""Read-only commands through a local helper are routine; anything else still asks.

Workers wrap their commands in a helper (``def run(*a):
subprocess.run(list(a))``) and call it with literal commands. The review
follows the helper to each call site: when every one is a literal read
(git status/log/diff, ls, grep, cat…), the call is routine, so a session's
grant covers it; a write, a commit, a checkout, a command it cannot read or a
helper it cannot follow keeps the call undecided (or high impact).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.checker.review import review_action
from localmelo.melo.checker.review.python import read_only_argv

VARARGS = (
    "import subprocess\n"
    "def run(*a):\n"
    "    r = subprocess.run(list(a), capture_output=True, text=True)\n"
    "    print(r.stdout, r.stderr)\n"
)
ONE_LIST = (
    "import subprocess\n"
    "def sh(cmd, cwd=None):\n"
    "    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout\n"
)


def verdict(tmp_path: Path, code: str) -> str:
    home = tmp_path / "home"
    root = home / "project"
    root.mkdir(parents=True, exist_ok=True)
    payload = ActionReviewPayload(
        "python_exec", {"code": code}, str(root), home=str(home)
    )
    return review_action(payload).verdict


@pytest.mark.parametrize(
    "calls",
    [
        'run("git", "status")\nrun("ls", "-la")\nrun("grep", "-rn", "TODO", ".")\n'
        'run("cat", "README.md")',
        'run("git", "log", "-n", "5")\nrun("git", "diff", "--stat")',
        'run("git", "-C", "sub", "show", "--stat", "HEAD")\nrun("git", "branch")',
        'run("find", ".", "-name", "*.py")\nrun("wc", "-l", "a.py")',
    ],
)
def test_literal_reads_through_a_helper_are_routine(tmp_path: Path, calls: str) -> None:
    assert verdict(tmp_path, VARARGS + calls) == "routine"


def test_a_helper_that_takes_one_list_is_followed_too(tmp_path: Path) -> None:
    code = ONE_LIST + 'print(sh(["git", "status"]))\nprint(sh(["ls"]))'
    assert verdict(tmp_path, code) == "routine"


@pytest.mark.parametrize(
    "calls",
    [
        'run("git", "status")\nrun("git", "commit", "-m", "x")',
        'run("git", "checkout", "main")',
        'run("git", "branch", "topic")',
        'run("git", "add", ".")',
        'run("touch", "x")',
        'run("rm", "-rf", "build")',
        'run("find", ".", "-delete")',
        'run("git", "diff", "--output=patch.txt")',
        'run("python", "-c", "print(1)")',
    ],
)
def test_a_write_commit_or_checkout_is_not_routine(tmp_path: Path, calls: str) -> None:
    assert verdict(tmp_path, VARARGS + calls) != "routine"


@pytest.mark.parametrize(
    "code",
    [
        # A command built at run time, or spread from a variable.
        VARARGS + 'name = input()\nrun("git", name)',
        VARARGS + 'for c in [("git", "status")]:\n    run(*c)',
        # The helper passed around: its calls cannot be read.
        VARARGS + 'list(map(run, ["ls"]))',
        # A helper that runs a shell, or starts a session of its own.
        "import subprocess\ndef sh(c):\n    subprocess.run(c, shell=True)\nsh('ls')",
        "import subprocess\ndef sh(*a):\n"
        "    subprocess.run(list(a), start_new_session=True)\nsh('ls')",
        # A helper that changes its arguments before running them.
        "import subprocess\ndef sh(a):\n    subprocess.run(fix(a))\nsh(['ls'])",
    ],
)
def test_a_helper_melo_cannot_follow_is_not_routine(tmp_path: Path, code: str) -> None:
    # Undecided, or high impact (a session of its own keeps running).
    assert verdict(tmp_path, code) != "routine"


def test_a_read_of_credentials_through_a_helper_is_high_impact(
    tmp_path: Path,
) -> None:
    assert verdict(tmp_path, VARARGS + 'run("cat", "~/.ssh/id_ed25519")') == "high"


def test_other_code_is_still_reviewed(tmp_path: Path) -> None:
    code = VARARGS + 'run("git", "status")\nopen("/etc/hosts", "w").write("x")'
    assert verdict(tmp_path, code) == "high"


@pytest.mark.parametrize(
    ("argv", "reads"),
    [
        (("git", "status"), True),
        (("git", "remote", "-v"), True),
        (("git", "remote", "add", "o", "u"), False),
        (("git", "push"), False),
        (("ls",), True),
        (("sort", "-o", "x"), False),
        ((), False),
    ],
)
def test_read_only_argv(argv: tuple[str, ...], reads: bool) -> None:
    assert read_only_argv(argv) is reads
