"""call_preview: one line of at most 200 characters per tool call."""

from __future__ import annotations

import json
from typing import Any

import pytest

from localmelo.melo.executor import call_preview
from localmelo.melo.executor.preview import PREVIEW_CHARS


class _Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no text")


@pytest.mark.parametrize(
    ("tool", "arguments", "preview"),
    [
        ("shell_exec", {"command": "ls   -la\n\t /tmp "}, "ls -la /tmp"),
        ("python_exec", {"code": "print(1)"}, "print(1)"),
        (
            "python_exec",
            {"code": "\n\n  import os  \nprint(os.getcwd())\n\nprint(2)\n\n"},
            "print(os.getcwd()) (+2 more lines)",
        ),
        ("file_read", {"path": "notes/today.txt"}, "notes/today.txt"),
        ("list_files", {}, "."),
        ("list_files", {"path": "src", "depth": 1}, "src"),
        ("list_files", {"path": "src", "depth": 3}, "src (3 levels)"),
        (
            "file_write",
            {"path": "out.txt", "content": "héllo\n"},
            "out.txt (6 chars)",
        ),
        (
            "weather",
            {"units": "metric", "city": "Zürich"},
            '{"city":"Zürich","units":"metric"}',
        ),
    ],
)
def test_each_row(tool: str, arguments: dict[str, Any], preview: str) -> None:
    assert call_preview(tool, arguments) == preview


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("shell_exec", {}),
        ("shell_exec", {"command": ["ls"]}),
        ("python_exec", {"code": 3}),
        ("python_exec", {"code": "  \n\n"}),
        ("file_read", {"path": None}),
        ("list_files", {"path": 7}),
        ("list_files", {"depth": "2"}),
        ("file_write", {"path": "a.txt"}),
        ("file_write", {"path": 1, "content": "x"}),
    ],
)
def test_a_missing_or_non_str_value_gives_json(
    tool: str, arguments: dict[str, Any]
) -> None:
    assert call_preview(tool, arguments) == json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def test_json_names_other_values_by_their_text() -> None:
    thing = object()
    assert call_preview("probe", {"x": thing}) == f'{{"x":"{thing}"}}'


def test_a_long_preview_ends_in_three_dots() -> None:
    preview = call_preview("shell_exec", {"command": "echo " + "x" * 300})
    assert len(preview) == PREVIEW_CHARS == 200
    assert preview == ("echo " + "x" * 300)[:197] + "..."
    exact = "y" * 200
    assert call_preview("file_read", {"path": exact}) == exact


def test_it_is_one_line() -> None:
    assert call_preview("file_read", {"path": "a\nb\r\nc"}) == "a b c"
    assert "\n" not in call_preview("probe", {"text": "line 1\nline 2"})


def test_it_never_raises() -> None:
    # Keys json.dumps cannot sort together, and a value whose str() raises.
    assert call_preview("probe", {1: "a", "b": 2})  # type: ignore[dict-item]
    assert call_preview("probe", {"x": _Unprintable()})


# ── python_exec: what the code does, not its imports ──

_GREP = (
    "import subprocess\n"
    "found = subprocess.run(\n"
    '    ["grep", "-rn", "DEFAULT_WORKSPACE", "."], capture_output=True, text=True\n'
    ")\n"
    "print(found.stdout)\n"
)
_HELPER = (
    "import subprocess\n"
    "\n"
    "def run(*args):\n"
    "    return subprocess.run(args, capture_output=True, text=True).stdout\n"
    "\n"
    'print(run("git", "status"))\n'
    'print(run("git", "log", "-1"))\n'
)


@pytest.mark.parametrize(
    ("code", "preview"),
    [
        # A literal command list reads as the command, quoted as a shell would.
        (_GREP, "grep -rn DEFAULT_WORKSPACE . (+1 more lines)"),
        (
            "import subprocess\nsubprocess.check_output(['git', 'commit', '-m', 'a b'])",
            "git commit -m 'a b'",
        ),
        (
            "import subprocess\nsubprocess.run('pytest -q  tests', shell=True)",
            "pytest -q tests",
        ),
        # A helper the code defines, called with literal arguments.
        (_HELPER, "git status (+1 more lines)"),
        (
            "def sh(cmd):\n    import subprocess\n    return subprocess.call(cmd)\nsh(['ls', '-la'])",
            "ls -la",
        ),
        # Plain code: the first line that is no import, comment or blank.
        ("import os\nimport sys\n# note\n\nprint(os.getcwd())", "print(os.getcwd())"),
        (
            "from pathlib import (\n    Path,\n    PurePath,\n)\nroot = Path('.')\nprint(root)",
            "root = Path('.') (+1 more lines)",
        ),
        # A command with a name in it is not literal: fall back to the code.
        (
            "import subprocess\nname = 'a'\nsubprocess.run(['echo', name])",
            "name = 'a' (+1 more lines)",
        ),
        # Only imports, or code that does not parse: the first line there is.
        ("import os\nimport sys\n", "import os (+1 more lines)"),
        ("import os", "import os"),
        ("x = (\nprint(1)", "x = ( (+1 more lines)"),
        ("# only a comment\nx = (", "x = ("),
    ],
)
def test_python_exec_names_what_it_does(code: str, preview: str) -> None:
    assert call_preview("python_exec", {"code": code}) == preview


def test_python_exec_code_with_odd_escapes_still_previews() -> None:
    assert call_preview("python_exec", {"code": "print('\\d')"}) == "print('\\d')"
