"""Spellings of destructive and system-reaching steps that must not read as routine.

Each case is a shape that an independent review found passing as ``routine``
(no question in Auto mode, plain code in Ask mode) after the rules for disks,
discarded work, other processes and scripting bridges were added:

- options read where the program does not read them (a signal after the
  target, a value glued to a short option, ``pkill -l``);
- a process name that is not this step's own after all;
- a script that hides where its comments start, or names its language in
  another spelling;
- shell forms around the command (loops, a backquoted command with spaces,
  ``bash -c -l '…'``, an upper-case program name);
- Python calls that stop a process or open a link through another object.

The read-only forms next to them stay routine.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.contracts.review import ActionReview


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


def is_delete(found: ActionReview) -> bool:
    return found.verdict == "high" and "delete" in found.categories


# ── pkill -l is long output, not a list ──


@pytest.mark.parametrize(
    "command",
    [
        "pkill -l Safari",
        "pkill -L -F /tmp/p.pid",
        "pkill -fl node",
        "pkill -9 -l Safari",
    ],
)
def test_pkill_with_long_output_still_stops_programs(ws: Path, command: str) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize("command", ["kill -l", "kill -l 9", "killall -l"])
def test_listing_the_signals_stays_routine(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


# ── Options are read only before the first target ──


@pytest.mark.parametrize(
    "command",
    [
        "kill -9 1234 -0",
        "pkill Safari -0",
        "killall Safari -s",
        "pkill -STOP node -0",
        "kill -9 -1234",
        "kill -- -1234",
        "pkill -s 0 Safari",  # -s is a session for pkill, not a signal
        "killall -s0 Safari -u bob",
    ],
)
def test_a_signal_zero_or_print_flag_after_the_target_is_still_a_kill(
    ws: Path, command: str
) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "kill -0 1234",
        "kill -s 0 1234",
        "kill -n 0 1234",
        "pkill -0 Safari",
        "pkill -0 -f node",
        "killall -s Safari",
        "killall -0 Safari",
    ],
)
def test_a_signal_zero_or_print_flag_before_the_target_stays_routine(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


# ── A value glued to a short option is not a set of options ──


@pytest.mark.parametrize(
    "command",
    [
        "git clean -fd -enode_modules",
        "git clean -fd -e.env",
        "git clean -fd -e n",
        "git clean -fdx -enode_modules",
        "git restore -sS .",
        "git restore -sS -- a.py",
    ],
)
def test_the_letters_of_a_glued_value_are_not_options(ws: Path, command: str) -> None:
    assert is_delete(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "git clean -nd",
        "git clean -n -e x",
        "git clean -ne x",
        "git restore -S .",
        "git restore -S -s HEAD .",
        "git restore --staged .",
    ],
)
def test_a_dry_run_or_an_unstaging_stays_routine(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


# ── A name that held $! and then something else is not this step's process ──


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & pid=$!; pid=1234; kill $pid",
        "sleep 9 & pid=$!; pid=$(pgrep Safari); kill $pid",
        "sleep 9 & pid=$!; for pid in $(pgrep Safari); do kill $pid; done",
        "sleep 9 & pid=$!; read pid < /tmp/p; kill $pid",
        "export pid=1234; (sleep 9 & pid=$!); kill $pid",
        "sleep 9 & pid=$!; declare pid=1234; kill $pid",
        "sleep 9 & pid=$!; eval 'pid=1234'; kill $pid",
        "sleep 9 & pid=$!; export pid=1234; kill $pid",
    ],
)
def test_a_pid_set_again_after_dollar_bang_is_not_own(ws: Path, command: str) -> None:
    assert is_system(sh(ws, command))


def test_a_loop_variable_named_like_a_started_pid_is_not_own(ws: Path) -> None:
    found = sh(ws, "sleep 9 & pid=$!; for pid in 1 2 3; do kill $pid; done")
    assert is_system(found)


def test_a_pid_set_in_a_subshell_is_not_own(ws: Path) -> None:
    assert is_system(sh(ws, "(sleep 9 & pid=$!); kill $pid"))


def test_xargs_kill_is_never_own(ws: Path) -> None:
    assert is_system(sh(ws, "sleep 9 & value=$!; pgrep Safari | xargs kill"))
    assert is_system(sh(ws, "pgrep Safari | xargs kill"))


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & pid=$!; kill $pid",
        "sleep 9 & pid=$!; wait $pid; kill $pid",
        "for i in 1 2; do sleep 9 & pid=$!; done; kill $pid",
        "sleep 9 & export pid=$!; kill $pid",
        "sleep 9 & a=$!; sleep 9 & b=$!; kill $a $b",
        "{ sleep 9 & pid=$!; }; kill $pid",
    ],
)
def test_a_pid_that_only_ever_holds_dollar_bang_is_own(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


# ── Scripts that steer where their comments start, or how they are named ──


@pytest.mark.parametrize(
    "command",
    [
        """osascript -e 'set |--| to application "Finder"' -e 'tell |--| to activate'""",
        """osascript -e 'set |#| to application "Finder"' -e 'tell |#| to activate'""",
        """osascript -lJavaScript -e 'Application ("Terminal").doScript("id")'""",
        """osascript -lJavaScript -e 'this["App"+"lication"]("Finder").activate()'""",
        """osascript -l JavaScript -e 'var x = `${Application("Terminal")}`'""",
        """osascript -l JavaScript -e 'var r = /"/; Application ("Terminal")'""",
        """osascript -l JavaScript -e 'var A = this.constructor.constructor("return 1")'""",
        """osascript -l JavaScript -e '1 + 1'""",
        """osascript -l JXA -e '$.NSLog("x")'""",
        """osascript -l Python -e 'x'""",
    ],
)
def test_a_script_that_hides_its_reach_is_system(ws: Path, command: str) -> None:
    assert is_system(sh(ws, command))


def test_a_pipe_inside_an_applescript_comment_or_string_is_not_an_identifier(
    ws: Path,
) -> None:
    command = "osascript -e 'display dialog \"a | b\"'"
    assert sh(ws, command).verdict == "routine"


@pytest.mark.parametrize(
    "command",
    [
        """osascript -e 'display notification "done" with title "melo"'""",
        """osascript -e 'display dialog "The application finished"'""",
    ],
)
def test_plain_applescript_without_a_reach_stays_routine(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


def test_a_plain_javascript_app_call_is_still_app_control(ws: Path) -> None:
    found = sh(ws, """osascript -l JavaScript -e 'Application("Safari").activate()'""")
    assert found.categories == ("app_control",)
    glued = sh(ws, """osascript -lJavaScript -e 'Application("Safari").activate()'""")
    assert glued.verdict == "high"


# ── Shell forms around the command ──


@pytest.mark.parametrize(
    "command",
    [
        "RM -rf ~/Documents/x",
        "Rm -rf ~/Documents/x",
        "GIT reset --hard",
        "PKILL Safari",
        "DiskUtil eraseDisk APFS X disk9",
        "Kill 123",
        "SUDO diskutil eraseDisk APFS X disk9",
    ],
)
def test_a_program_named_in_upper_case_is_the_same_program(
    ws: Path, command: str
) -> None:
    found = sh(ws, command)
    assert found.verdict == "high"
    assert found.categories


def test_a_read_only_program_in_upper_case_stays_routine(ws: Path) -> None:
    assert sh(ws, "GIT status").verdict == "routine"
    assert sh(ws, "LS").verdict == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "bash -c -l 'rm -rf ~/Documents/x'",
        "bash -c -- 'rm -rf ~/Documents/x'",
        "bash -o pipefail -c -l 'rm -rf ~/Documents/x'",
        "bash -c -o posix 'rm -rf ~/Documents/x'",
        "bash -lc 'rm -rf ~/Documents/x'",
        "sh -c -e 'git reset --hard'",
    ],
)
def test_a_command_after_more_options_than_dash_c_is_read(
    ws: Path, command: str
) -> None:
    assert is_delete(sh(ws, command))


def test_a_harmless_command_after_dash_c_and_an_option_stays_routine(ws: Path) -> None:
    assert sh(ws, "bash -c -l 'echo hi'").verdict != "high"
    assert sh(ws, "bash -c 'echo hi'").verdict == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "echo `rm -rf ~/Documents/x`",
        "echo `git reset --hard`",
        "echo `pkill Safari`",
        "X=`rm -rf ~/Documents/x`",
        "ls `git clean -fd`",
        'echo "`rm -rf ~/Documents/x`"',
    ],
)
def test_a_backquoted_command_with_spaces_is_reviewed(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "high"


def test_a_backquoted_harmless_command_and_single_quoted_text_stay_routine(
    ws: Path,
) -> None:
    assert sh(ws, "echo `date +%Y`").verdict == "routine"
    assert sh(ws, "echo `date +%Y` done").verdict == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "for f in a b; do rm -rf ~/Documents/x; done",
        "if true; then rm -rf ~/Documents/x; fi",
        "while true; do git reset --hard; done",
        "for f in a; do pkill Safari; done",
        "f() { rm -rf ~/Documents/x; }; f",
        "true && { git clean -fd; }",
    ],
)
def test_a_command_inside_a_loop_or_condition_is_reviewed(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "high"


def test_a_loop_of_harmless_commands_stays_routine(ws: Path) -> None:
    assert sh(ws, "for f in a b; do echo $f; done").verdict == "routine"
    assert sh(ws, "if [ -f x ]; then echo ok; fi").verdict == "routine"


# ── More forms of the same commands ──


@pytest.mark.parametrize(
    "command",
    [
        "git stash -q drop",
        "git checkout --pathspec-from-file=list.txt",
        "git reset --har",
        "git --config-env=x.y=Z checkout -- .",
        "git -C sub stash --quiet clear",
    ],
)
def test_more_spellings_of_a_git_discard_are_delete(ws: Path, command: str) -> None:
    assert is_delete(sh(ws, command))


@pytest.mark.parametrize(
    "command", ["git stash list", "git stash -q list", "git stash"]
)
def test_stashing_work_is_not_discarding_it(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "diskutil appleRAID delete x",
        "diskutil ar delete x",
        "newfs_apfs disk9",
        "sudo newfs_hfs -v X disk9s2",
        "asr restore --erase --source a.dmg --target /Volumes/y",
    ],
)
def test_more_ways_to_erase_a_disk_are_system_and_delete(
    ws: Path, command: str
) -> None:
    found = sh(ws, command)
    assert is_system(found) and is_delete(found)


def test_launchctl_kill_is_system(ws: Path) -> None:
    assert is_system(sh(ws, "launchctl kill 9 system/com.example.job"))
    assert sh(ws, "launchctl list").verdict == "routine"


# ── Python: psutil and a browser object ──


@pytest.mark.parametrize(
    "code",
    [
        "import psutil\npsutil.Process(1234).kill()",
        "import psutil\npsutil.Process(1234).terminate()",
        "import psutil\np = psutil.Process(1234)\np.send_signal(9)",
        "import psutil\nfor p in psutil.process_iter():\n    p.kill()",
        "from psutil import Process\nProcess(1).kill()",
    ],
)
def test_psutil_kill_is_system(ws: Path, code: str) -> None:
    assert is_system(py(ws, code))


def test_a_process_this_code_started_may_be_stopped(ws: Path) -> None:
    code = "import psutil, subprocess\np = subprocess.Popen(['sleep', '9'])\np.kill()"
    assert py(ws, code).verdict == "routine"
    popen = "import psutil\np = psutil.Popen(['sleep', '9'])\np.terminate()"
    assert py(ws, popen).verdict == "routine"


@pytest.mark.parametrize(
    "code",
    [
        "import webbrowser\nwebbrowser.get().open('shortcuts://run-shortcut?name=x')",
        "import webbrowser\nwebbrowser.get('safari').open_new('obsidian://open')",
        "import webbrowser\nb = webbrowser.get()\nb.open('shortcuts://x')",
        "import webbrowser\nwebbrowser.MacOSXOSAScript('safari').open('ssh://h')",
    ],
)
def test_a_custom_link_opened_through_a_browser_object_is_system(
    ws: Path, code: str
) -> None:
    assert is_system(py(ws, code))


def test_a_web_link_opened_through_a_browser_object_stays_routine(ws: Path) -> None:
    code = "import webbrowser\nwebbrowser.get().open('https://example.com')"
    assert py(ws, code).verdict == "routine"


# ── Cost ──


BIG = 100_000


def timed(ws: Path, command: str) -> float:
    from localmelo.melo.checker.review.gist import step_gist

    arguments = {"command": command, "why": "Delete a.txt"}
    # CPU time of the fastest of three runs, so that a busy machine (a parallel
    # test run) cannot fail a bound meant to catch a rule that backtracks.
    times = []
    for _ in range(3):
        started = time.process_time()
        step_gist("shell_exec", arguments, review(ws, "shell_exec", arguments))
        times.append(time.process_time() - started)
    return min(times)


@pytest.mark.parametrize(
    "command",
    [
        "bash -" + "c" * BIG + "1 x",
        "git clean -" + "x" * BIG + "! .",
        "open -" + "a" * BIG + "! x",
        "kill " + "-a " * (BIG // 3) + "1",
        "pkill " + "-f " * (BIG // 3) + "x",
        "sleep 1 & " + "".join(f"p{i}=$!; " for i in range(BIG // 12)) + "kill $p1",
    ],
)
def test_review_of_a_100kb_command_is_fast(ws: Path, command: str) -> None:
    """Under 2 seconds, or three times what a plain command of the same size costs."""
    plain = timed(ws, "echo " + "a" * BIG)
    assert timed(ws, command) < max(2.0, 3 * plain)
