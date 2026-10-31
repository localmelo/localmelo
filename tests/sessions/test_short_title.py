"""short_title: a request's first clause, cut at a word, never wider than 48."""

from __future__ import annotations

import pytest

from localmelo.support.sessions.display import TITLE_WIDTH, short_title, title_width


@pytest.mark.parametrize(
    ("text", "title"),
    [
        # Short requests stay whole, tails of punctuation go.
        ("Fix the login bug", "Fix the login bug"),
        ("  Fix   the\tlogin bug.  ", "Fix the login bug"),
        ("Plan the trip!", "Plan the trip"),
        # The first non-blank line, and its first sentence.
        ("\n\n  Second line here\nthird", "Second line here"),
        ("Run the tests. Then commit.", "Run the tests"),
        ("See test_e2e_pty.py now", "See test_e2e_pty.py now"),
        # A long sentence breaks at a clause of its own, a comma away.
        (
            "DEFAULT_WORKSPACE is defined twice, in support/gateway/task_policy.py "
            "and support/gateway/console/registry.py. Make registry.py import it.",
            "DEFAULT_WORKSPACE is defined twice",
        ),
        # A short lead-in is no title: the cut falls at a word instead.
        (
            "Fix this today: in tests/cli/test_e2e_pty.py the list_files test "
            "only checks that the word list_files appears on screen.",
            "Fix this today: in tests/cli/test_e2e_pty.py…",
        ),
        (
            "No rush, whenever you are free: docs/guides/embedding.md and "
            "docs/zh-CN/guides/embedding.md have no picture.",
            "No rush, whenever you are free…",
        ),
        # Chinese counts two columns a character and breaks between characters.
        (
            "刚才那几个任务里，哪个提交了几个文件？简单说。",
            "刚才那几个任务里，哪个提交了几个文件",
        ),
        (
            "请把这个仓库里所有测试文件的名字都列出来然后统计每个文件有多少个测试函数",
            "请把这个仓库里所有测试文件的名字都列出来然后统…",
        ),
        # Mixed text breaks at a word or at a Chinese character.
        (
            "Update README and docs 中文混合 text that goes on and on and on forever",
            "Update README and docs 中文混合 text that goes…",
        ),
        # One long word has no break: cut where it must be.
        ("x" * 100, "x" * 47 + "…"),
        ("", "Task"),
        ("   \n  ", "Task"),
    ],
)
def test_short_title(text: str, title: str) -> None:
    assert short_title(text) == title
    assert title_width(short_title(text)) <= TITLE_WIDTH


def test_a_cut_never_ends_in_the_middle_of_a_word() -> None:
    text = "Rewrite the approval card so that every answered request folds away " * 3
    title = short_title(text)
    assert title.endswith("…")
    assert text.startswith(title.removesuffix("…"))
    assert text[len(title) - 1] == " "
