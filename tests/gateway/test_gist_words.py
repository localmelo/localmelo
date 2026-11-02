"""The words of the card's first sentence, in both catalogs.

``melo.checker.review.gist`` names a sentence and a warning by id; the catalogs
hold ``gist.say.<id>`` and ``gist.warn.<id>`` for each, with the parameters the
code fills in. The terminal reads them in English through ``gist_text``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from localmelo.melo.checker.review.gist import SAY_IDS, WARN_IDS, step_gist
from localmelo.melo.contracts.review import ActionReview
from localmelo.support.words import catalog, gist_text

APP = ActionReview(
    "high", ("app_control",), ("Operates Safari and Notes through a script.",)
)
# One call per sentence id, so the parameters the code gives are the catalog's.
SAMPLES: dict[str, tuple[str, dict[str, Any], ActionReview | None]] = {
    "model": ("shell_exec", {"command": "ls", "why": "List the notes"}, None),
    "command": ("shell_exec", {"command": "git status"}, None),
    "command_apps": ("shell_exec", {"command": "open -a Safari"}, APP),
    "script": ("python_exec", {"code": "print(1)"}, None),
    "script_apps": ("python_exec", {"code": "print(1)"}, APP),
    "script_command": (
        "python_exec",
        {"code": "import subprocess\nsubprocess.run(['ls'])"},
        None,
    ),
    "write": ("file_write", {"path": "a/todo.md", "content": "x"}, None),
    "edit": ("file_edit", {"path": "a/todo.md", "old": "x", "new": "y"}, None),
    "tool": ("deploy", {"target": "staging"}, None),
    "effect_delete": ("shell_exec", {"command": "rm x", "why": "Close the tabs"}, None),
    "effect_send_out": (
        "shell_exec",
        {"command": "curl -d a https://example.com", "why": "Close the tabs"},
        None,
    ),
}
PARAM = re.compile(r"\{(\w+)\}")


def test_every_sentence_and_warning_has_words_in_both_languages() -> None:
    for language in ("en", "zh-Hans"):
        words = catalog(language)
        for name in SAY_IDS:
            assert f"gist.say.{name}" in words, (language, name)
        for name in WARN_IDS:
            assert f"gist.warn.{name}" in words, (language, name)


def test_every_sentence_is_sampled() -> None:
    assert sorted(SAMPLES) == sorted(SAY_IDS)


@pytest.mark.parametrize("name", SAY_IDS)
def test_the_code_gives_the_parameters_the_words_take(name: str) -> None:
    tool, arguments, review = SAMPLES[name]
    found = step_gist(tool, arguments, review)
    assert found is not None and found.id == name
    for language in ("en", "zh-Hans"):
        entry = catalog(language)[f"gist.say.{name}"]
        assert isinstance(entry, str)
        assert set(PARAM.findall(entry)) == set(found.params), (language, name)


@pytest.mark.parametrize("name", WARN_IDS)
def test_a_warning_takes_what_the_model_said(name: str) -> None:
    for language in ("en", "zh-Hans"):
        entry = catalog(language)[f"gist.warn.{name}"]
        assert isinstance(entry, str)
        assert set(PARAM.findall(entry)) == {"said"}


def test_the_terminal_reads_the_sentence_in_english() -> None:
    def read(name: str, language: str = "en") -> tuple[str, str]:
        tool, arguments, review = SAMPLES[name]
        found = step_gist(tool, arguments, review)
        assert found is not None
        return gist_text(found.to_json(), language)

    assert read("model") == ("List the notes", "")
    assert read("command") == ("Run the command git status", "")
    assert read("script_apps") == (
        "Run a Python script that operates Safari and Notes",
        "",
    )
    assert read("write") == ("Write todo.md", "")
    assert read("effect_delete") == (
        "Delete files",
        "melo described this as “Close the tabs”, but the code deletes files.",
    )


def test_the_same_sentence_reads_in_chinese() -> None:
    found = step_gist("shell_exec", {"command": "rm x", "why": "关闭标签页"})
    assert found is not None
    assert gist_text(found.to_json(), "zh-Hans") == (
        "删除文件",
        "melo 把这一步描述为“关闭标签页”，但代码会删除文件。",
    )
    plain = step_gist("shell_exec", {"command": "ls", "why": "列出笔记"})
    assert plain is not None
    assert gist_text(plain.to_json(), "zh-Hans") == ("列出笔记", "")


def test_a_gist_the_reader_cannot_read_gives_nothing() -> None:
    assert gist_text(None, "en") == ("", "")
    assert gist_text({}, "en") == ("", "")
    assert gist_text({"say": "x"}, "en") == ("", "")
