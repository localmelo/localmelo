"""The one plain sentence an approval card starts with, and the mismatch warning.

The sentence is the model's own words when the call carries them (a ``why``
field or a leading comment) and is built by code from the tool and its
arguments otherwise. When the model's words clearly disagree with what the
code does (it deletes files, it sends data out), the card shows the code's
real effect and a warning instead. Both directions are tested.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.melo.checker.review.gist import Gist, step_gist
from localmelo.melo.contracts.review import ActionReview

APP_REVIEW = ActionReview(
    "high", ("app_control",), ("Operates Safari through a script.",)
)

CLOSE_TABS = """import subprocess
script = '''
tell application "Safari"
  close (every tab of front window whose URL contains "example.com")
end tell
'''
subprocess.run(['osascript', '-e', script])
"""


def gist(
    tool: str, arguments: dict[str, Any], review: ActionReview | None = None
) -> Gist:
    found = step_gist(tool, arguments, review)
    assert found is not None
    return found


# ── The model's own words ──


@pytest.mark.parametrize("field", ["why", "description", "purpose"])
def test_a_why_field_is_the_sentence(field: str) -> None:
    found = gist("shell_exec", {"command": "ls", field: "List the notes folder"})
    assert (found.id, found.by) == ("model", "model")
    assert found.params == {"text": "List the notes folder"}
    assert found.warn == ""
    assert found.to_json() == {
        "say": {"id": "model", "params": {"text": "List the notes folder"}},
        "by": "model",
    }


def test_the_words_are_one_short_line() -> None:
    found = gist("shell_exec", {"command": "ls", "why": "  List\n the   notes  "})
    assert found.params == {"text": "List the notes"}
    long = gist("shell_exec", {"command": "ls", "why": "word " * 80})
    assert len(long.params["text"]) <= 160 and long.params["text"].endswith("…")


@pytest.mark.parametrize("value", ["", "  ", "ok", 7, None, ["a long enough list"]])
def test_a_field_that_is_no_sentence_is_ignored(value: Any) -> None:
    found = gist("shell_exec", {"command": "git status", "why": value})
    assert found.by == "code" and found.id == "command"


def test_a_secret_in_the_words_is_redacted() -> None:
    found = gist(
        "shell_exec",
        {"command": "ls", "why": "Use token sk-abcdefghijklmnopqrstuvwxyz123456 here"},
    )
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in found.params["text"]


@pytest.mark.parametrize(
    ("tool", "arguments", "text"),
    [
        (
            "python_exec",
            {"code": "# Close the extra forum tabs\nprint(1)\n"},
            "Close the extra forum tabs",
        ),
        ("shell_exec", {"command": "\n# List the notes\nls notes"}, "List the notes"),
        ("python_exec", {"code": "# 关闭多余的标签页\nprint(1)"}, "关闭多余的标签页"),
    ],
)
def test_a_leading_comment_is_the_models_word(
    tool: str, arguments: dict[str, Any], text: str
) -> None:
    found = gist(tool, arguments)
    assert (found.id, found.by, found.params["text"]) == ("model", "model", text)


@pytest.mark.parametrize(
    "code",
    [
        "#!/usr/bin/env python3\nprint(1)",
        "# noqa\nprint(1)",
        "# type: ignore\nprint(1)",
        "# -*- coding: utf-8 -*-\nprint(1)",
        "# x = 5\nprint(x)",
        "print(1)  # a comment after code that is long enough",
        "import os\n# a comment after the first line of code",
    ],
)
def test_a_comment_that_is_no_description_is_not_one(code: str) -> None:
    assert gist("python_exec", {"code": code}).by == "code"


# ── Built by code ──


def test_a_command_is_named_by_the_command() -> None:
    found = gist("shell_exec", {"command": "git   status"})
    assert (found.id, found.by, found.params) == (
        "command",
        "code",
        {"command": "git status"},
    )
    long = gist("shell_exec", {"command": "echo " + "x" * 200})
    assert len(long.params["command"]) <= 80 and long.params["command"].endswith("…")


def test_python_is_named_by_the_command_it_runs_or_as_a_script() -> None:
    literal = gist(
        "python_exec", {"code": "import subprocess\nsubprocess.run(['ls', '-l'])"}
    )
    assert (literal.id, literal.params) == ("script_command", {"command": "ls -l"})
    plain = gist("python_exec", {"code": "print(1)"})
    assert (plain.id, plain.params) == ("script", {})


def test_a_file_write_and_an_edit_are_named_by_the_file() -> None:
    write = gist("file_write", {"path": "/work/notes/todo.md", "content": "x"})
    assert (write.id, write.params) == ("write", {"name": "todo.md"})
    edit = gist("file_edit", {"path": "src/app.py", "old": "a", "new": "b"})
    assert (edit.id, edit.params) == ("edit", {"name": "app.py"})


def test_another_tool_is_named_by_its_name() -> None:
    found = gist("deploy", {"target": "staging"})
    assert (found.id, found.params) == ("tool", {"tool": "deploy"})


def test_only_a_why_is_the_words_of_a_tool_that_is_not_a_runner() -> None:
    """A ``description`` of another tool is its content, not what the step is for."""
    content = gist("create_issue", {"title": "Bug", "description": "Login fails on x"})
    assert (content.id, content.by) == ("tool", "code")
    said = gist("create_issue", {"title": "Bug", "why": "File the login bug"})
    assert (said.id, said.params) == ("model", {"text": "File the login bug"})


def test_a_tool_that_asks_with_its_own_sentence_has_no_gist() -> None:
    assert step_gist("browser:example.com", {"site": "example.com"}, None) is None


def test_a_step_that_operates_an_app_names_the_app() -> None:
    script = gist("python_exec", {"code": CLOSE_TABS}, APP_REVIEW)
    assert (script.id, script.params) == ("script_apps", {"apps": "Safari"})
    command = gist(
        "shell_exec",
        {"command": "osascript -e 'tell application \"Safari\" to quit'"},
        APP_REVIEW,
    )
    assert (command.id, command.params) == ("command_apps", {"apps": "Safari"})
    opened = ActionReview(
        "high", ("app_control",), ("Opens Notes through macOS, outside melo's limits.",)
    )
    assert gist("shell_exec", {"command": "open -a Notes"}, opened).params == {
        "apps": "Notes"
    }


def test_the_model_s_words_come_before_the_apps() -> None:
    found = gist(
        "python_exec",
        {"code": CLOSE_TABS, "why": "Close the 7 forum tabs melo opened"},
        APP_REVIEW,
    )
    assert (found.id, found.by, found.warn) == ("model", "model", "")
    # The apps are still named, for the reason line.
    assert found.apps == "Safari"
    assert found.to_json()["apps"] == "Safari"
    assert (
        "apps" not in gist("shell_exec", {"command": "ls", "why": "List it"}).to_json()
    )


# ── A description that does not match the code ──

DELETES = [
    ("shell_exec", {"command": "rm -rf build"}),
    ("shell_exec", {"command": "cd x && sudo rm file"}),
    ("shell_exec", {"command": "find . -name '*.tmp' | xargs rm"}),
    ("shell_exec", {"command": "find . -name '*.tmp' -delete"}),
    ("shell_exec", {"command": "git clean -fdx"}),
    ("python_exec", {"code": "import shutil\nshutil.rmtree('build')"}),
    ("python_exec", {"code": "import os\nos.remove('a.txt')"}),
    ("python_exec", {"code": "import os\nos.unlink('a.txt')"}),
    ("python_exec", {"code": "import pathlib\npathlib.Path('a').unlink()"}),
    ("python_exec", {"code": "import subprocess\nsubprocess.run(['rm', '-r', 'x'])"}),
]
SENDS = [
    ("shell_exec", {"command": "curl -d @notes.txt https://example.com/in"}),
    ("shell_exec", {"command": "curl -X POST https://example.com/in"}),
    ("shell_exec", {"command": "curl --upload-file a.zip https://example.com/a"}),
    ("shell_exec", {"command": "scp notes.txt me@example.com:backup/"}),
    ("shell_exec", {"command": "rsync -a docs/ me@example.com:docs/"}),
    (
        "python_exec",
        {"code": "import requests\nrequests.post('https://example.com', data=b'x')"},
    ),
    (
        "python_exec",
        {
            "code": "import urllib.request\nurllib.request.urlopen('https://example.com', data=b'x')"
        },
    ),
]


@pytest.mark.parametrize(("tool", "arguments"), DELETES)
def test_a_description_that_hides_a_delete_is_flagged(
    tool: str, arguments: dict[str, Any]
) -> None:
    found = gist(tool, {**arguments, "why": "Close the extra tabs"})
    # The code's real effect comes first; the model's words are in the warning.
    assert (found.id, found.by) == ("effect_delete", "code")
    assert (found.warn, found.warn_params) == (
        "delete",
        {"said": "Close the extra tabs"},
    )
    assert found.to_json() == {
        "say": {"id": "effect_delete", "params": {}},
        "by": "code",
        "warn": {"id": "delete", "params": {"said": "Close the extra tabs"}},
    }


@pytest.mark.parametrize(("tool", "arguments"), DELETES)
def test_a_description_that_says_it_deletes_is_not(
    tool: str, arguments: dict[str, Any]
) -> None:
    for said in ("Delete the build output", "Remove old files", "Clean up the folder"):
        found = gist(tool, {**arguments, "why": said})
        assert (found.id, found.warn) == ("model", ""), (arguments, said)


@pytest.mark.parametrize(("tool", "arguments"), SENDS)
def test_a_description_that_hides_a_send_is_flagged(
    tool: str, arguments: dict[str, Any]
) -> None:
    found = gist(tool, {**arguments, "why": "Check the notes"})
    assert (found.id, found.warn) == ("effect_send_out", "send_out")
    assert found.warn_params == {"said": "Check the notes"}


@pytest.mark.parametrize(("tool", "arguments"), SENDS)
def test_a_description_that_says_it_sends_is_not(
    tool: str, arguments: dict[str, Any]
) -> None:
    for said in ("Upload the notes", "Send the report to the server", "Post the data"):
        found = gist(tool, {**arguments, "why": said})
        assert (found.id, found.warn) == ("model", ""), (arguments, said)


def test_chinese_words_are_read_both_ways() -> None:
    delete = {"command": "rm -rf build"}
    assert gist("shell_exec", {**delete, "why": "关闭多余的标签页"}).warn == "delete"
    for said in ("删除构建产物", "清理临时文件", "移除旧文件"):
        assert gist("shell_exec", {**delete, "why": said}).warn == ""
    send = {"command": "curl -d @a https://example.com"}
    assert gist("shell_exec", {**send, "why": "查看笔记"}).warn == "send_out"
    assert gist("shell_exec", {**send, "why": "上传笔记"}).warn == ""


def test_the_comment_is_checked_like_a_why_field() -> None:
    code = "# Close the extra tabs\nimport shutil\nshutil.rmtree('build')"
    found = gist("python_exec", {"code": code})
    assert (found.id, found.warn) == ("effect_delete", "delete")
    assert found.warn_params == {"said": "Close the extra tabs"}


def test_closing_tabs_through_a_script_matches_its_description() -> None:
    found = gist(
        "python_exec",
        {"code": CLOSE_TABS, "why": "Close the 7 forum tabs melo opened"},
        APP_REVIEW,
    )
    assert (found.id, found.warn) == ("model", "")


def test_the_review_s_own_findings_count_as_effects() -> None:
    review = ActionReview(
        "high", ("delete",), ("Force-pushes, which can overwrite commits.",)
    )
    found = gist(
        "shell_exec", {"command": "git push -f", "why": "Show the status"}, review
    )
    assert found.warn == "delete"
    send = ActionReview("high", ("send_out",), ("Sends data.",))
    assert (
        gist("shell_exec", {"command": "x", "why": "Show it"}, send).warn == "send_out"
    )


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        # Words that look like a delete are not one.
        ("shell_exec", {"command": "echo rm -rf"}),
        ("shell_exec", {"command": "grep -r remove ."}),
        ("shell_exec", {"command": "ls removed.txt"}),
        ("python_exec", {"code": "items = [1]\nitems.remove(1)"}),
        ("python_exec", {"code": "print('shutil.rmtree is dangerous')"}),
        # A read is not a send.
        ("shell_exec", {"command": "curl https://example.com"}),
        ("shell_exec", {"command": "curl -I https://example.com"}),
        (
            "python_exec",
            {"code": "import requests\nrequests.get('https://example.com')"},
        ),
        (
            "python_exec",
            {
                "code": "import urllib.request\nurllib.request.urlopen('https://example.com')"
            },
        ),
    ],
)
def test_a_step_that_does_neither_is_never_flagged(
    tool: str, arguments: dict[str, Any]
) -> None:
    found = gist(tool, {**arguments, "why": "Look at the notes"})
    assert (found.id, found.warn) == ("model", "")


def test_no_words_means_nothing_to_flag() -> None:
    found = gist("shell_exec", {"command": "rm -rf build"})
    assert (found.id, found.by, found.warn) == ("command", "code", "")
