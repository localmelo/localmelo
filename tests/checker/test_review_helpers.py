"""Read-only commands through a local helper are routine; anything else still asks.

Workers wrap their commands in a helper (``def run(*a):
subprocess.run(list(a))``) and call it with literal commands. The review
follows the helper to each call site: when every one is a literal read
(git status/log/diff, ls, grep of a file, cat…), the call is routine, so a session's
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
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / ".env").write_text("API_KEY=secret\n")
    payload = ActionReviewPayload(
        "python_exec", {"code": code}, str(root), home=str(home)
    )
    return review_action(payload).verdict


@pytest.mark.parametrize(
    "calls",
    [
        'run("git", "status")\nrun("ls", "-la")\nrun("grep", "-n", "TODO", "a.py")\n'
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
        "import subprocess\ndef sh(*a):\n    a = ('rm', '-rf', 'x')\n"
        "    subprocess.run(list(a))\nsh('ls')",
        "import subprocess\ndef sh(c):\n    c.append('--force')\n"
        "    subprocess.run(c)\nsh(['ls'])",
        "import subprocess\ndef sh(c):\n    c[0] = 'rm'\n    subprocess.run(c)\nsh(['ls'])",
        # A method, or a helper reached another way than by its name.
        "import subprocess\nclass T:\n    def run(self, *a):\n"
        "        subprocess.run(list(a))\nT().run('rm', '-rf', 'x')",
        VARARGS + 'globals()["run"]("rm", "-rf", "x")',
        "import subprocess, functools\n@functools.cache\ndef run(*a):\n"
        "    subprocess.run(list(a))\nrun('ls')",
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
        (("git", "diff", "--ext-diff"), False),
        (("git", "log", "-p", "--textconv"), False),
        (("git", "-c", "core.pager=less", "log"), False),
        (("git", "--exec-path=/tmp", "status"), False),
        (("git", "--git-dir=/tmp/x", "status"), False),
        (("git", "--no-pager", "-C", "src", "log"), True),
        (("ls",), True),
        (("grep", "-n", "x", "a.py"), True),
        (("grep", "-rn", "x", "."), False),
        (("grep", "-Rl", "x"), False),
        (("egrep", "--recursive", "x"), False),
        (("grep", "--directories=recurse", "x"), False),
        (("git", "grep", "x"), False),
        (("grep", "--rec", "x"), False),
        (("grep", "-nd", "recurse", "x"), False),
        (("grep", "-d", "skip", "x", "a.py"), True),
        (("tree", "-a"), True),
        (("tree", "-ao", "out.txt"), False),
        (("file", "-C"), False),
        (("rg", "--pre", "sh", "x", "a.py"), False),
        (("git", "branch", "-a"), True),
        (("git", "branch", "--unset-upstream"), False),
        (("git", "branch", "-D", "x"), False),
        (("sort", "-o", "x"), False),
        ((), False),
    ],
)
def test_read_only_argv(argv: tuple[str, ...], reads: bool) -> None:
    assert read_only_argv(argv) is reads


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        # Reads outside the workspace ask, as before helpers were followed.
        ('run("cat", "/etc/passwd")', "undecided"),
        ('run("grep", "-r", "token", "/Users")', "undecided"),
        ('run("cat", "../../outside.txt")', "undecided"),
        ('run("git", "-C", "..", "status")', "undecided"),
        ('run("grep", "--file=/etc/hosts", "src")', "undecided"),
        # Files that hold passwords or keys, inside the workspace too.
        ('run("cat", ".env")', "high"),
        ('run("cat", "config/.env.local")', "high"),
        ('run("git", "show", "HEAD:.env")', "high"),
        ('run("cat", "certs/server.pem")', "high"),
        ('run("head", "deploy/id_ed25519")', "high"),
        ('run("cat", "credentials.json")', "high"),
        # git options that run a program the repository or the command names.
        ('run("git", "diff", "--ext-diff")', "undecided"),
        ('run("git", "log", "-p", "--textconv")', "undecided"),
        ('run("git", "-c", "core.pager=less", "log")', "undecided"),
        ('run("git", "--exec-path=/tmp/x", "status")', "undecided"),
        ('run("git", "--git-dir=/tmp/x", "status")', "undecided"),
        # The same reads, inside the workspace and of plain files: routine.
        ('run("cat", "README.md")', "routine"),
        ('run("grep", "-n", "x", "src/a.py")', "routine"),
        ('run("grep", "-n", "token", "src/a.py")', "routine"),
        ('run("rg", "-n", "x", "src/a.py")', "routine"),
        # A search through every file below, hidden ones (.env) too, asks:
        # search_files is the worker's way, and never reads them.
        ('run("grep", "-rn", "token", ".")', "undecided"),
        ('run("grep", "-r", "API_KEY")', "undecided"),
        ('run("grep", "-R", "x", "src")', "undecided"),
        ('run("grep", "-nri", "x", ".")', "undecided"),
        ('run("grep", "--recursive", "x", ".")', "undecided"),
        ('run("grep", "-d", "recurse", "x", ".")', "undecided"),
        ('run("git", "grep", "API_KEY")', "undecided"),
        ('run("rg", "API_KEY")', "undecided"),
        ('run("rg", "x", "src")', "undecided"),
        ('run("git", "show", "HEAD:docs/a.md")', "routine"),
        ('run("git", "-C", "src", "log", "-n", "3")', "routine"),
    ],
)
def test_a_helper_read_is_routine_only_inside_the_workspace_and_of_no_secret(
    tmp_path: Path, call: str, expected: str
) -> None:
    assert verdict(tmp_path, VARARGS + call) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # The helper's folder is its default: a literal inside the workspace.
        (ONE_LIST + 'sh(["cat", "a.md"])', "routine"),
        (
            'import subprocess\ndef sh(cmd, cwd="src"):\n    subprocess.run(cmd, cwd=cwd)\n'
            'sh(["cat", "../README.md"])',
            "routine",
        ),
        (
            'import subprocess\ndef sh(cmd, cwd="/etc"):\n    subprocess.run(cmd, cwd=cwd)\n'
            'sh(["ls"])',
            "undecided",
        ),
        # A call that passes the folder, or a helper whose first parameter takes it.
        (ONE_LIST + 'sh(["ls"], "/etc")', "undecided"),
        (
            "import subprocess\ndef run(cwd, *a):\n    subprocess.run(list(a), cwd=cwd)\n"
            'run("/etc", "ls")',
            "undecided",
        ),
    ],
)
def test_a_helper_runs_in_its_folder_only_when_melo_can_place_it(
    tmp_path: Path, code: str, expected: str
) -> None:
    assert verdict(tmp_path, code) == expected
