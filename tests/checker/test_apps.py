"""The allow-list parser behind the ``app_control`` kind (``review/apps.py``).

A script counts only when every word of it is on the list; the parse says
``None`` whenever it is unsure, and the caller keeps "Change system settings?".
"""

from __future__ import annotations

import pytest

from localmelo.melo.checker.review import apps

PLAIN = [
    ('tell application "Safari" to activate', "Safari"),
    ('tell app "Notes" to quit', "Notes"),
    ('TELL APPLICATION "Safari" TO ACTIVATE', "Safari"),
    (
        'tell application "Safari" to open location "https://example.com/a?b=1"',
        "Safari",
    ),
    ('tell application "Safari" to close front window', "Safari"),
    ('tell application "Safari" to get URL of current tab of front window', "Safari"),
    (
        'tell application "Safari"\n  activate\n  set URL of front document to '
        '"https://example.com"\nend tell',
        "Safari",
    ),
    (
        'tell application "Safari"\n'
        "  repeat with w in windows\n"
        '    close (every tab of w whose URL contains "example.com")\n'
        "  end repeat\n"
        "end tell",
        "Safari",
    ),
    (
        'tell application "Safari"\n'
        "  repeat with w in windows\n"
        "    repeat with t in tabs of w\n"
        '      close (t whose URL contains "example.com")\n'
        "    end repeat\n"
        "  end repeat\n"
        "end tell",
        "Safari",
    ),
]
NOT_PLAIN = [
    "",
    "activate",
    'tell application "Safari" to activate\ntell application "Notes" to quit',
    'tell application "Safari"\n  activate\n',  # no end tell
    'tell application "Safari"\n  activate\nend tell\nquit',
    'tell application "Safari"\n  repeat with w in windows\n    close w\nend tell',
    'tell application "Safari"\n  end repeat\nend tell',
    'tell application "Safari" to make new document',
    'tell application "Safari" to do JavaScript "1"',
    'tell application "Safari" to open location "file:///etc/hosts"',
    'tell application "Safari" to open location "javascript:alert(1)"',
    'tell application "Safari" to set URL of front document to "javascript:alert(1)"',
    'tell application "Safari" to set URL of front document to someVariable',
    'tell application "Safari" to set the clipboard to "x"',
    'tell application "Safari" to quit -- done',
    'tell application "Safari" to quit # done',
    'tell application "Safari" to (* x *) quit',
    'tell application "Safari" to quit ¬\n',
    'tell application "Safari" to activate; quit',
    'tell application "Safari" to get (do shell script "id")',
    'tell application "Safari" to close (run script "x")',
    'tell application "Safari" to quit with administrator privileges',
    'tell application ("Saf" & "ari") to quit',
    'tell application "Safari" of machine "eppc://example.com" to quit',
    'tell application id "com.apple.Safari" to quit',
    'tell application "Safari"\ntell application "Terminal" to activate\nend tell',
    'tell application "Terminal" to activate',
    'tell application "Safari.app" to activate',
    'tell application "/Applications/Safari.app" to activate',
    "tell application 'Safari' to activate",
]


@pytest.mark.parametrize(("script", "name"), PLAIN)
def test_a_plain_script_names_its_app(script: str, name: str) -> None:
    assert apps.applescript_app(script) == name


@pytest.mark.parametrize("script", NOT_PLAIN)
def test_anything_else_is_not_plain(script: str) -> None:
    assert apps.applescript_app(script) is None


@pytest.mark.parametrize(
    ("script", "name"),
    [
        ('Application("Safari").quit()', "Safari"),
        ("Application('Mail').activate();", "Mail"),
        ('Application("Safari").windows[0].close()', "Safari"),
        ('Application("Safari").openLocation("https://example.com")', "Safari"),
    ],
)
def test_a_plain_javascript_call_names_its_app(script: str, name: str) -> None:
    assert apps.jxa_app(script) == name


@pytest.mark.parametrize(
    "script",
    [
        'Application("Terminal").doScript("id")',
        'Application("Safari").doShellScript("id")',
        'Application("Safari").quit(); Application("Safari").quit()',
        'Application("Safari")["quit"]()',
        'Application("Safari").openLocation("javascript:1")',
        'var A = Application; A("Safari").quit()',
        'eval("1")',
        "",
    ],
)
def test_other_javascript_is_not_plain(script: str) -> None:
    assert apps.jxa_app(script) is None


def test_osascript_reads_only_the_options_it_knows() -> None:
    quit_safari = ['tell application "Safari" to quit']
    assert apps.osascript_app(["-e", *quit_safari]) == "Safari"
    assert apps.osascript_app(["-l", "AppleScript", "-e", *quit_safari]) == "Safari"
    assert apps.osascript_app(["-s", "h", "-e", *quit_safari]) is None
    assert apps.osascript_app(["file.scpt"]) is None
    assert apps.osascript_app(["-e", *quit_safari, "arg"]) is None
    assert apps.osascript_app([]) is None
    pieces = ["-e", 'tell application "Safari"', "-e", "activate", "-e", "end tell"]
    assert apps.osascript_app(pieces) == "Safari"


def test_open_reads_exactly_one_app() -> None:
    assert apps.open_app(["-a", "Safari"]) == "Safari"
    assert apps.open_app(["-a", "Google Chrome"]) == "Google Chrome"
    assert apps.open_app(["-b", "com.apple.Safari"]) == "Safari"
    for args in (
        ["-a", "Safari", "x.sh"],
        ["-a", "Safari", "-a", "Terminal"],
        ["-b", "com.apple.Safari", "-a", "Terminal"],
        ["-b", "com.apple.Terminal"],
        ["-b", "dev.warp.Warp-Stable"],
        ["-a", "Terminal"],
        ["-a", "Safari.app"],
        ["-a", "/Applications/Safari.app"],
        ["-a"],
        ["-n", "-a", "Safari"],
        [],
    ):
        assert apps.open_app(args) is None, args
