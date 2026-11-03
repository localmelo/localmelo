"""Steps that reach macOS or destroy work are never routine.

Three families, each of which used to read as ``routine`` (no question in Auto
mode, plain code in Ask mode):

- a script that drives the system through a scripting bridge without naming
  an app (AppleScriptObjC, JXA written ``Application ("X")`` or through an
  alias), and ``open`` of a link with a custom scheme (``shortcuts://``);
- commands that erase a disk, discard uncommitted work or kill other
  processes;
- the same commands seen through wrappers (``sudo``, ``env``, ``nice``,
  ``command``, ``xargs``, ``sh -c``, ``bash -lc``, a Python process call).

The read-only forms (``git status``, ``kill -0``, ``diskutil list``) stay
routine, and ``http(s)`` and ``file`` links keep their handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.contracts.review import ActionReview, review_title


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    folder = tmp_path / "home" / "ws"
    folder.mkdir(parents=True)
    return folder


def review(ws: Path, tool: str, arguments: dict[str, Any]) -> ActionReview:
    return Checker().review_action(
        ActionReviewPayload(
            tool,
            arguments,
            str(ws),
            home=str(ws.parent),
            executable="/opt/melo/bin/python3",
            prefix="/opt/melo",
        )
    )


def sh(ws: Path, command: str) -> ActionReview:
    return review(ws, "shell_exec", {"command": command})


def py(ws: Path, code: str) -> ActionReview:
    return review(ws, "python_exec", {"code": code})


def is_system(found: ActionReview) -> bool:
    return found.verdict == "high" and "system" in found.categories


# ── 1. Steps that reach macOS without naming an app ──

BRIDGES = [
    # AppleScriptObjC: a framework loaded and used through `current application`.
    """osascript -e 'use framework "Foundation"' -e "set t to current application's NSTask\"""",
    """osascript -e 'use framework "AppKit"' -e "current application's NSWorkspace's sharedWorkspace()\"""",
    """osascript -e "current application's NSWorkspace's sharedWorkspace()\"""",
    # An app held in a variable, no `tell application` in sight.
    """osascript -e 'set a to application "Terminal"' -e 'tell a to activate'""",
    """osascript -e 'set a to app "Terminal"' -e 'tell a to activate'""",
    # A loaded script, a file opened for access, a link opened by the system.
    """osascript -e 'set s to load script POSIX file "/tmp/x.scpt"' -e 'tell s to go()'""",
    """osascript -e 'set f to open for access POSIX file "/tmp/a"' -e 'read f'""",
    """osascript -e 'open location "shortcuts://run-shortcut?name=Foo"'""",
    # JXA: a space before the parenthesis, an alias, a destructured name, a
    # computed member, eval, the bridge by index.
    """osascript -l JavaScript -e 'Application  ("Terminal").doScript("id")'""",
    """osascript -l JavaScript -e 'Application ("Terminal").doScript("id")'""",
    """osascript -l JavaScript -e 'var A = Application; A("Terminal").doScript("id")'""",
    """osascript -l JavaScript -e 'const f = Application; f("Safari").activate()'""",
    """osascript -l JavaScript -e 'const {Application: A} = this; A("Terminal").doScript("id")'""",
    """osascript -l JavaScript -e 'globalThis["Appl"+"ication"]("Terminal").doScript("id")'""",
    """osascript -l JavaScript -e 'eval("Appl"+"ication(\\"Terminal\\").doScript(\\"id\\")")'""",
    """osascript -l JavaScript -e '$["NSTask"].alloc.init'""",
]


@pytest.mark.parametrize("command", BRIDGES)
def test_a_script_that_reaches_the_system_without_naming_an_app_is_system(
    ws: Path, command: str
) -> None:
    found = sh(ws, command)
    assert found.verdict == "high"
    assert found.categories == ("system",)
    assert review_title(found, "ws") == "Change system settings?"


def test_a_bridge_script_in_a_python_process_call_is_system(ws: Path) -> None:
    code = (
        "import subprocess\nsubprocess.run(['osascript', '-e', 'use framework "
        "\"Foundation\"', '-e', \"set t to current application's NSTask\"])"
    )
    assert is_system(py(ws, code))


def test_a_plain_jxa_call_with_a_space_is_app_control_not_routine(ws: Path) -> None:
    found = sh(ws, """osascript -l JavaScript -e 'Application ("Safari").activate()'""")
    assert (found.verdict, found.categories) == ("high", ("app_control",))
    assert found.reasons == ("Operates Safari through a script.",)


@pytest.mark.parametrize(
    "command",
    [
        """osascript -e 'display dialog "The application finished"'""",
        """osascript -e 'display notification "use framework later"'""",
        """osascript -e 'display notification "done"'""",
    ],
)
def test_a_keyword_inside_a_string_does_not_make_a_script_a_bridge(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


LINKS = [
    "open 'shortcuts://run-shortcut?name=Foo'",
    "open 'shortcuts://run-shortcut?name=Foo&input=text&text=hi'",
    "open -u 'shortcuts://run-shortcut?name=Foo'",
    "open --url 'shortcuts://run-shortcut?name=Foo'",
    "open -g 'shortcuts://run-shortcut?name=Foo'",
    "open 'Shortcuts://run-shortcut?name=Foo'",
    "open 'obsidian://open?vault=notes'",
    "open 'vscode://file/tmp/x.txt'",
    "open 'ssh://example.com'",
    "open 'ftp://example.com/x'",
    "open tel:5551234",
    "open 'mailto:a@example.com?subject=hi'",
    "open 'x-apple.systempreferences:com.apple.preference.security'",
    "open https://example.com 'shortcuts://run-shortcut?name=Foo'",
    # wrappers
    "env LANG=C open 'shortcuts://run-shortcut?name=Foo'",
    "nice -n 5 open 'shortcuts://run-shortcut?name=Foo'",
    "command open 'shortcuts://run-shortcut?name=Foo'",
    "/usr/bin/open 'shortcuts://run-shortcut?name=Foo'",
    "python3 -m webbrowser 'shortcuts://run-shortcut?name=Foo'",
    """sh -c "open 'shortcuts://run-shortcut?name=Foo'\"""",
]


@pytest.mark.parametrize("command", LINKS)
def test_open_of_a_custom_scheme_is_system(ws: Path, command: str) -> None:
    """Decision: every link that is not http(s) or file is ``system``.

    The app that answers a custom scheme is whatever the person installed, and
    a link such as ``shortcuts://`` or ``obsidian://`` can make that app run
    things. So none is ``app_control``, which is only for an app named by
    the script itself and allow-listed. Both are click-only, so this changes
    only the title and the reason.
    """
    found = sh(ws, command)
    assert found.verdict == "high"
    assert "system" in found.categories
    assert "app_control" not in found.categories


def test_a_shortcuts_link_is_classed_like_shortcuts_run(ws: Path) -> None:
    ran = sh(ws, "shortcuts run Foo")
    link = sh(ws, "open 'shortcuts://run-shortcut?name=Foo'")
    assert (link.verdict, link.categories) == (ran.verdict, ran.categories)
    assert link.reasons == ran.reasons


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['open', 'shortcuts://run-shortcut?name=Foo'])",
        "import os\nos.system(\"open 'shortcuts://run-shortcut?name=Foo'\")",
        "import webbrowser\nwebbrowser.open('shortcuts://run-shortcut?name=Foo')",
    ],
)
def test_a_custom_link_opened_from_python_is_system(ws: Path, code: str) -> None:
    assert is_system(py(ws, code))


@pytest.mark.parametrize(
    "command",
    [
        "open https://example.com",
        "open http://example.com/a/b?c=d",
        "open file:///tmp/notes.txt",
        "open notes.txt",
    ],
)
def test_web_and_file_links_keep_their_handling(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


def test_a_web_page_opened_from_python_stays_routine(ws: Path) -> None:
    code = "import webbrowser\nwebbrowser.open('https://example.com')"
    assert py(ws, code).verdict == "routine"


# ── 2. Destructive commands ──

DISKS = [
    "diskutil eraseDisk JHFS+ Scratch disk2",
    "diskutil eraseVolume APFS Scratch disk2s1",
    "diskutil partitionDisk disk2 GPT JHFS+ Scratch 0b",
    "diskutil zeroDisk disk2",
    "diskutil secureErase 0 disk2",
    "diskutil randomDisk 1 disk2",
    "diskutil reformat disk2s1",
    "diskutil erasedisk JHFS+ Scratch disk2",
    "diskutil apfs deleteContainer disk3",
    "diskutil coreStorage delete disk3",
    "sudo diskutil eraseDisk JHFS+ Scratch disk2",
    "env LANG=C /usr/sbin/diskutil eraseDisk JHFS+ Scratch disk2",
    "nice -n 5 diskutil eraseDisk JHFS+ Scratch disk2",
    "command diskutil eraseDisk JHFS+ Scratch disk2",
    """sh -c "diskutil eraseDisk JHFS+ Scratch disk2\"""",
    """bash -lc "diskutil eraseDisk JHFS+ Scratch disk2\"""",
    "echo disk2 | xargs diskutil zeroDisk",
    "diskutil list && diskutil eraseDisk JHFS+ Scratch disk2",
]


@pytest.mark.parametrize("command", DISKS)
def test_erasing_or_repartitioning_a_disk_is_system_and_delete(
    ws: Path, command: str
) -> None:
    found = sh(ws, command)
    assert found.verdict == "high"
    assert {"system", "delete"} <= set(found.categories)
    assert review_title(found, "ws") == "Change system settings?"


def test_erasing_a_disk_from_python_is_system(ws: Path) -> None:
    code = "import subprocess\nsubprocess.run(['diskutil', 'eraseDisk', 'JHFS+', 'S', 'disk2'])"
    assert {"system", "delete"} <= set(py(ws, code).categories)


@pytest.mark.parametrize(
    "command",
    [
        "diskutil list",
        "diskutil info disk0",
        "diskutil info -plist disk0",
        "diskutil mount disk2s1",
        "diskutil apfs list",
        "echo diskutil eraseDisk JHFS+ Scratch disk2",
    ],
)
def test_reading_disks_stays_routine(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


DISCARDS = [
    "git checkout -- src/a.py",
    "git checkout .",
    "git checkout -- .",
    "git checkout HEAD~1 -- src/a.py",
    "git checkout main src/a.py",
    "git checkout -f main",
    "git checkout --force main",
    "git checkout --theirs src/a.py",
    "git checkout -p",
    "git -C sub checkout -- .",
    "git --no-pager checkout -- .",
    "git switch -f main",
    "git switch --discard-changes main",
    "git restore src/a.py",
    "git restore .",
    "git restore --worktree src/a.py",
    "git restore --staged --worktree src/a.py",
    "git restore --source=HEAD~1 src/a.py",
    "git reset --hard",
    "git reset --hard HEAD~1",
    "git --no-pager reset --hard",
    "git -C sub reset --hard",
    "git clean -f",
    "git clean -fd",
    "git clean -fdx",
    "git clean --force -d",
    "git clean -f build",
    "git stash drop",
    "git stash drop 'stash@{1}'",
    "git stash clear",
    # wrappers
    "nice git reset --hard",
    "command git clean -fd",
    """sh -c "git checkout -- .\"""",
    """bash -lc "git checkout -- .\"""",
    """bash -c "git restore .\"""",
    "echo a.py | xargs git checkout --",
    "git status && git checkout -- .",
    "/usr/bin/git checkout -- .",
]


@pytest.mark.parametrize("command", DISCARDS)
def test_discarding_uncommitted_work_is_delete(ws: Path, command: str) -> None:
    found = sh(ws, command)
    assert found.verdict == "high"
    assert "delete" in found.categories
    assert review_title(found, "ws") == "Delete files that cannot be restored?"


@pytest.mark.parametrize(
    "command",
    [
        "sudo git checkout -- .",
        "env GIT_PAGER=cat git checkout -- .",
    ],
)
def test_a_wrapper_that_makes_it_stricter_still_reads_the_discard(
    ws: Path, command: str
) -> None:
    """``sudo`` is system and ``env VAR=…`` an undecided call; both also see the discard."""
    found = sh(ws, command)
    assert found.verdict == "high"
    assert "delete" in found.categories


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['git', 'checkout', '--', '.'])",
        "import subprocess\nsubprocess.run(['git', 'restore', 'src/a.py'])",
        "import subprocess\nsubprocess.run(['git', 'clean', '-fd'])",
        "import os\nos.system('git stash drop')",
    ],
)
def test_discarding_from_python_is_delete(ws: Path, code: str) -> None:
    found = py(ws, code)
    assert found.verdict == "high"
    assert "delete" in found.categories


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git status --short",
        "git diff",
        "git diff --staged",
        "git log --oneline -5",
        "git show HEAD",
        "git stash",
        "git stash push -m wip",
        "git stash list",
        "git stash pop",
        "git stash apply",
        "git checkout main",
        "git checkout feature/login",
        "git checkout -b feature",
        "git checkout -b feature main",
        "git checkout -",
        "git checkout --detach HEAD",
        "git switch main",
        "git switch -c feature",
        "git restore --staged src/a.py",
        "git restore -S src/a.py",
        "git clean -n",
        "git clean --dry-run -d",
        "git clean -nfd",
        "git reset --soft HEAD~1",
        "git reset HEAD src/a.py",
        "git add src/a.py",
        """git commit -m "do not git checkout -- . or pkill foo\"""",
        "echo git checkout -- .",
        "grep -rn 'git reset --hard' docs",
    ],
)
def test_git_commands_that_lose_nothing_stay_routine(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


def test_a_checkout_of_a_file_that_is_there_discards_it(ws: Path) -> None:
    """``git checkout notes.txt`` is a path when no branch has that name."""
    (ws / "notes.txt").write_text("edited\n")
    assert "delete" in sh(ws, "git checkout notes.txt").categories
    assert sh(ws, "git checkout feature").verdict == "routine"


KILLS = [
    "pkill -9 -f python",
    "pkill node",
    "killall node",
    "killall -9 Safari",
    "kill 1234",
    "kill -9 1234",
    "kill -TERM 1234",
    "kill -s KILL 1234",
    "kill $(pgrep foo)",
    "kill `pgrep foo`",
    "kill $pid",
    "pid=$(pgrep foo); kill $pid",
    "kill -9 -1",
    "kill 1234 5678",
    "sleep 5 & kill $! 1234",
    "pgrep foo | xargs kill",
    "pgrep foo | xargs kill -9",
    "sudo pkill foo",
    "env LANG=C pkill foo",
    "nice -n 5 pkill foo",
    "command killall foo",
    "timeout 5 pkill foo",
    "/usr/bin/pkill foo",
    "sh -c 'pkill -f foo'",
    "bash -lc 'pkill -f foo'",
    "bash -c 'kill 1234'",
    "pgrep foo && pkill foo",
]


@pytest.mark.parametrize("command", KILLS)
def test_killing_other_processes_is_system(ws: Path, command: str) -> None:
    found = sh(ws, command)
    assert found.verdict == "high"
    assert "system" in found.categories


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['pkill', '-f', 'foo'])",
        "import subprocess\nsubprocess.run(['killall', 'node'])",
        "import os\nos.system('pkill foo')",
        "import os\nos.kill(1234, 15)",
        "import os\nos.killpg(1234, 9)",
        "import os\nos.kill(pid, 9)",
    ],
)
def test_killing_from_python_is_system(ws: Path, code: str) -> None:
    assert is_system(py(ws, code))


@pytest.mark.parametrize(
    "command",
    [
        "pgrep -f python",
        "pgrep node",
        "kill -0 1234",
        "pkill -0 foo",
        "kill -l",
        "kill -l 9",
        "sleep 5 & kill $!",
        "sleep 5 & kill -9 $!",
        """sleep 5 & kill "$!\"""",
        "sleep 5 & pid=$!; kill $pid",
        "sleep 5 & pid=$!; kill -9 $pid",
        "sleep 5 & pid=$!; kill ${pid}",
        "sleep 5 & kill %1",
        "kill -TERM $$",
        "echo pkill foo",
        "grep -rn pkill scripts",
        "command -v kill",
    ],
)
def test_reading_processes_and_stopping_this_steps_own_stay_routine(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.kill(1234, 0)",
        "import os, subprocess\np = subprocess.Popen(['sleep', '5'])\nos.kill(p.pid, 15)",
        "import subprocess\np = subprocess.Popen(['sleep', '5'])\np.kill()",
    ],
)
def test_python_may_stop_a_process_it_started(ws: Path, code: str) -> None:
    assert py(ws, code).verdict == "routine"


def test_a_shell_started_with_a_combined_flag_is_read_like_dash_c(ws: Path) -> None:
    """``bash -lc 'cmd'`` is seen through, and stays at least as strict as before."""
    quiet = sh(ws, "bash -lc 'echo hi'")
    assert quiet.verdict != "high"  # was undecided; not closed, not loosened
    assert sh(ws, "zsh -ic 'kill 1234'").verdict == "high"


def test_a_step_that_destroys_is_not_covered_by_a_grant(ws: Path) -> None:
    from localmelo.melo.contracts.review import covered_by_grant, offers_grant

    found = sh(ws, "git checkout -- .")
    assert covered_by_grant(found) is False
    assert offers_grant(found) is False
