"""The ``app_control`` kind: a script that operates another app (not a system change).

A script that drives an app (AppleScript or ``osascript``, JXA, ``open -a``,
``shortcuts run``) is ``app_control``; what really changes the system, or runs
a shell, stays ``system``. ``app_control`` is never looser than ``system`` was:
the last tests prove it in every mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.contracts.review import (
    CATEGORY_WORDS,
    CLICK_ONLY,
    MODEL_CAN_ALLOW,
    TITLES,
    WORDS_CAN_ALLOW,
    ActionReview,
    covered_by_grant,
    offers_grant,
    review_title,
)

SAFARI_SCRIPT = """import subprocess
script = '''
tell application "Safari"
  repeat with w in windows
    set i to (count of tabs of w)
    repeat while i > 0
      set t to tab i of w
      if (URL of t) contains "example.com" then
        close t
      end if
      set i to i - 1
    end repeat
  end repeat
end tell
'''
r = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
print(r.returncode, r.stdout, r.stderr)
"""


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


# ── The new kind ──


def test_the_kind_has_every_word_the_system_kind_has() -> None:
    assert "app_control" in TITLES and "system" in TITLES
    assert TITLES["app_control"] == "Control an app?"
    assert "app_control" in CATEGORY_WORDS
    # Listed right after system: a step with both is titled by the stronger.
    order = list(TITLES)
    assert order.index("app_control") == order.index("system") + 1


def test_the_case_that_asked_for_the_kind(ws: Path) -> None:
    """A python step closing browser tabs through osascript was titled as a system change."""
    found = py(ws, SAFARI_SCRIPT)
    assert found.verdict == "high"
    assert found.categories == ("app_control",)
    assert found.reasons == ("Operates Safari through a script.",)


@pytest.mark.parametrize(
    ("command", "apps"),
    [
        ("osascript -e 'tell application \"Safari\" to close front window'", "Safari"),
        ("osascript -e 'tell app \"Notes\" to make new note'", "Notes"),
        (
            "osascript -e 'tell application \"Safari\" to quit' "
            "-e 'tell application \"Notes\" to quit'",
            "Safari and Notes",
        ),
        ("osascript -l JavaScript -e 'Application(\"Safari\").quit()'", "Safari"),
        ("osascript -l JXA -e \"Application('Mail').quit()\"", "Mail"),
        ("open -a Safari", "Safari"),
        ("open -a 'Google Chrome' https://example.com", "Google Chrome"),
        ("open -b com.apple.Safari", "com.apple.Safari"),
    ],
)
def test_a_script_that_operates_an_app_is_app_control(
    ws: Path, command: str, apps: str
) -> None:
    found = sh(ws, command)
    assert found.verdict == "high", command
    assert found.categories == ("app_control",), command
    assert apps in found.reasons[0], command
    # The same command from Python, as a process's argv.
    import shlex

    code = f"import subprocess\nsubprocess.run({shlex.split(command)!r})"
    assert py(ws, code).categories == ("app_control",), command


def test_running_a_shortcut_is_app_control(ws: Path) -> None:
    found = sh(ws, "shortcuts run 'Resize image'")
    assert found.categories == ("app_control",)
    assert "Shortcut" in found.reasons[0]


@pytest.mark.parametrize(
    "command",
    [
        # Real changes to the system stay system.
        "defaults write com.apple.dock autohide -bool true",
        "systemsetup -setremotelogin on",
        "tccutil reset All",
        "sudo ls",
        "echo hi > /etc/hosts",
        "cp x /Library/foo",
        # Login items and keystrokes: System Events changes the system.
        'osascript -e \'tell application "System Events" to make login item at end '
        'with properties {path:"/Applications/X.app"}\'',
        'osascript -e \'tell application "System Events" to keystroke "a"\'',
        "osascript -e 'tell application \"System Settings\" to activate'",
        # A shell reached through a script.
        "osascript -e 'do shell script \"ls\"'",
        'osascript -e \'tell application "Safari" to do shell script "ls"\'',
        "osascript -e 'do shell script \"ls\" with administrator privileges'",
        'osascript -e \'tell application "Terminal" to do script "ls"\'',
        'osascript -l JavaScript -e \'Application("Terminal").doScript("ls")\'',
        "osascript -l JavaScript -e 'ObjC.import(\"Foundation\"); $.NSTask'",
        # A script melo cannot read, or whose target it cannot name.
        "osascript ./drive.scpt",
        "osascript",
        "osascript -e 'tell application appName to quit'",
        # A program started from a file, or an app that runs commands.
        "open -a Terminal ./x.command",
        "open -b com.apple.Terminal",
        "open -a ./evil.app",
        "open -a Safari --args --some-flag",
        "open ./x.command",
        "automator x.workflow",
    ],
)
def test_a_real_system_change_or_a_shell_stays_system(ws: Path, command: str) -> None:
    found = sh(ws, command)
    assert found.verdict == "high", command
    assert "system" in found.categories, command
    assert "app_control" not in found.categories, command


def test_a_step_with_both_is_titled_system(ws: Path) -> None:
    found = sh(
        ws,
        "osascript -e 'tell application \"Safari\" to quit'; defaults write a b -int 1",
    )
    assert found.categories == ("system", "app_control")
    assert review_title(found, "ws") == "Change system settings?"


def test_a_notification_stays_routine(ws: Path) -> None:
    assert sh(ws, "osascript -e 'display notification \"done\"'").verdict == "routine"


# ── Never looser than system, in every mode ──


def test_app_control_can_be_allowed_only_by_a_click() -> None:
    assert "app_control" in CLICK_ONLY
    assert "app_control" not in WORDS_CAN_ALLOW
    assert "app_control" not in MODEL_CAN_ALLOW
    # Whatever words or a model can never allow, system could not either.
    assert "system" in CLICK_ONLY and "system" not in WORDS_CAN_ALLOW
    assert CLICK_ONLY.isdisjoint(WORDS_CAN_ALLOW)


@pytest.mark.parametrize("category", ["system", "app_control"])
def test_a_high_impact_call_of_either_kind_is_never_covered_by_a_grant(
    category: str,
) -> None:
    found = ActionReview("high", (category,), ("Does it.",))
    assert covered_by_grant(found) is False
    assert offers_grant(found) is False
