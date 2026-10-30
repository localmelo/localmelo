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
            "import os (+3 more lines)",
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
