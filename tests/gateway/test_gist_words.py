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
from localmelo.support.words import (
    catalog,
    gist_names,
    gist_note,
    gist_sentence,
    gist_text,
    said,
    said_in,
)

APP = ActionReview(
    "high",
    ("app_control",),
    ("Operates Safari through a script.", "Opens Notes through macOS, x."),
)
SAFARI = "osascript -e 'tell application \"Safari\" to quit'"
# One call per sentence id, so the parameters the code gives are the catalog's.
SAMPLES: dict[str, tuple[str, dict[str, Any], ActionReview | None]] = {
    "model": ("shell_exec", {"command": "ls", "why": "List the notes"}, None),
    "command": ("shell_exec", {"command": "git status"}, None),
    "operate": ("shell_exec", {"command": SAFARI + "; open -a Notes"}, APP),
    "then": ("shell_exec", {"command": SAFARI + "; rm x"}, APP),
    "more": (
        "shell_exec",
        {"command": "; ".join(f"echo {n}" for n in range(30))},
        None,
    ),
    "script": ("python_exec", {"code": "print(1)"}, None),
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
    # An effect only the rules saw, in a script whose code shows nothing.
    "script_effect_delete": (
        "python_exec",
        {"code": "exec(open('job.py').read())"},
        ActionReview("high", ("delete",), ("Deletes files.",)),
    ),
    "script_effect_send_out": (
        "python_exec",
        {"code": "exec(open('job.py').read())"},
        ActionReview("high", ("send_out",), ("Sends data.",)),
    ),
}
PARAM = re.compile(r"\{(\w+)\}")


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
    """The words, and for a target warning also where the code acts."""
    wanted = {"said", "target"} if name.endswith("_target") else {"said"}
    for language in ("en", "zh-Hans"):
        entry = catalog(language)[f"gist.warn.{name}"]
        assert isinstance(entry, str)
        assert set(PARAM.findall(entry)) == wanted


def test_the_terminal_reads_the_sentence_in_english() -> None:
    def read(name: str, language: str = "en") -> tuple[str, str]:
        tool, arguments, review = SAMPLES[name]
        found = step_gist(tool, arguments, review)
        assert found is not None
        return gist_text(found.to_json(), language)

    # The model's words always carry their label.
    assert read("model") == ("melo says: List the notes", "")
    assert read("command") == ("Run the command git status", "")
    assert read("operate") == ("Operate Safari and Notes", "")
    assert read("then") == ("Operate Safari, then delete files", "")
    assert read("write") == ("Write todo.md", "")
    assert read("script_effect_delete") == (
        "Run a Python script that deletes files",
        "",
    )
    assert read("script_effect_send_out") == (
        "Run a Python script that sends data to another computer",
        "",
    )
    assert read("effect_delete") == (
        "Delete files",
        "melo described this as “Close the tabs”, but the code deletes files.",
    )
    sentence, _ = read("more")
    assert sentence.startswith("Run the command echo 0; echo 1")
    assert sentence.endswith(" more") and ", and " in sentence


def test_a_compound_step_reads_in_order_with_the_next_part_in_lower_case() -> None:
    found = step_gist(
        "shell_exec",
        {"command": "git status && curl -d @x https://example.com && rm -rf y"},
    )
    assert found is not None
    assert gist_text(found.to_json(), "en") == (
        "Run the command git status, then send data to another computer, then delete files",
        "",
    )
    assert gist_text(found.to_json(), "zh-Hans") == (
        "运行命令 git status，然后向另一台电脑发送数据，然后删除文件",
        "",
    )


def test_the_apps_are_joined_in_the_readers_language() -> None:
    assert gist_names(["Safari"], "en") == "Safari"
    assert gist_names(["Safari", "Notes"], "en") == "Safari and Notes"
    assert gist_names(["Safari", "Notes", "Mail"], "en") == "Safari, Notes and Mail"
    assert gist_names(["Safari", "Notes"], "zh-Hans") == "Safari和Notes"
    assert gist_names(["Safari", "Notes", "Mail"], "zh-Hans") == "Safari、Notes和Mail"
    # The sentence takes a list, not a joined string.
    node = {"id": "operate", "params": {"apps": ["Safari", "Notes"]}}
    assert gist_sentence(node, "zh-Hans") == "操作 Safari和Notes"
    assert "and" not in gist_sentence(node, "zh-Hans")


def test_the_models_words_below_a_risky_sentence_are_labelled() -> None:
    risky = ActionReview("high", ("install",), ("Installs numpy with pip.",))
    found = step_gist(
        "shell_exec", {"command": "pip install numpy", "why": "Check the notes"}, risky
    )
    assert found is not None
    assert gist_text(found.to_json(), "en") == ("Run the command pip install numpy", "")
    assert gist_note(found.to_json(), "en") == "melo says: Check the notes"
    assert gist_note(found.to_json(), "zh-Hans") == "melo 说：Check the notes"
    assert gist_note({"say": {}}, "en") == ""
    assert gist_note(None, "en") == ""


def test_a_stored_sentence_reads_in_any_language() -> None:
    node = {
        "id": "then",
        "params": {
            "first": {"id": "operate", "params": {"apps": ["Safari"]}},
            "next": {"id": "effect_delete", "params": {}},
        },
    }
    words = said("cards.wants.step", what={"gist": node})
    assert said_in("en", words) == "Needs your OK: Operate Safari, then delete files"
    assert said_in("zh-Hans", words) == "需要你同意：操作 Safari，然后删除文件"


def test_the_same_sentence_reads_in_chinese() -> None:
    found = step_gist("shell_exec", {"command": "rm x", "why": "关闭标签页"})
    assert found is not None
    assert gist_text(found.to_json(), "zh-Hans") == (
        "删除文件",
        "melo 把这一步描述为“关闭标签页”，但代码会删除文件。",
    )
    plain = step_gist("shell_exec", {"command": "ls", "why": "列出笔记"})
    assert plain is not None
    assert gist_text(plain.to_json(), "zh-Hans") == ("melo 说：列出笔记", "")


def test_a_gist_the_reader_cannot_read_gives_nothing() -> None:
    assert gist_text(None, "en") == ("", "")
    assert gist_text({}, "en") == ("", "")
    assert gist_text({"say": "x"}, "en") == ("", "")
