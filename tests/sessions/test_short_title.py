"""short_title: the one title for a task or a chat, from the words that asked for it."""

from __future__ import annotations

import pytest

from localmelo.support.sessions.display import TITLE_WIDTH, short_title, title_width
from localmelo.tests.linear import assert_linear


@pytest.mark.parametrize(
    ("text", "title"),
    [
        # Short requests stay whole, tails of punctuation go.
        ("Fix the login bug", "Fix the login bug"),
        ("  Fix   the\tlogin bug.  ", "Fix the login bug"),
        ("Plan the trip!", "Plan the trip"),
        # A question keeps its mark.
        ("What is the capital of France?", "What is the capital of France?"),
        ("哪个提交了几个文件？简单说。", "哪个提交了几个文件？"),
        # The first non-blank line, and its first sentence.
        ("\n\n  Second line here\nthird", "Second line here"),
        ("Run the tests. Then commit.", "Run the tests"),
        ("Fix CI. Then tell me.", "Fix CI"),
        ("1. Fix the login bug. 2. Add a test", "1. Fix the login bug"),
        ("See test_e2e_pty.py now", "See test_e2e_pty.py now"),
        # A polite lead goes, but only as a whole word.
        ("Please fix the login bug", "fix the login bug"),
        ("Could you please, kindly check the log", "check the log"),
        ("/task fix the build", "fix the build"),
        ("请帮我整理一下会议记录", "整理一下会议记录"),
        ("请问怎么重置密码", "怎么重置密码"),
        ("Pleased to meet you", "Pleased to meet you"),
        ("Can your team review this", "Can your team review this"),
        ("请假三天怎么写", "请假三天怎么写"),
        ("Please", "Please"),
        # A first line that is only a greeting says nothing about the request: the next
        # line names it.
        (
            "Hi\nCan you help me plan a three day trip to Kyoto?",
            "help me plan a three day trip to Kyoto?",
        ),
        ("Good morning,\n\nFix the login bug", "Fix the login bug"),
        ("hello!\nhey there\nsummarize the notes", "summarize the notes"),
        ("你好\n帮我整理一下会议记录", "整理一下会议记录"),
        ("您好，\n请问怎么重置密码", "怎么重置密码"),
        # A leading code block is skipped whole: the line after its closing fence names it.
        (
            "```python\ndef add(a, b):\n    return a + b\n```\n\nwhy does this fail?",
            "why does this fail?",
        ),
        ("Hi\n```\nprint(1)\n```\nsummarize the log", "summarize the log"),
        (
            "Hi there, can you help me plan a trip",
            "Hi there, can you help me plan a trip",
        ),
        # With nothing else to name it, the first line stays.
        ("Hi", "Hi"),
        ("Hello!\n\n", "Hello"),
        ("```\n```", "```"),
        ("```python\nprint(1)\n```", "```python"),  # nothing after the block
        ("```\nwhy does this fail?", "```"),  # a block that never closes
        # A level word that opens the request and is followed by a colon or comma
        # goes; anywhere else, or with nothing after it, it stays.
        ("不急：主对话里英文的提示语要改成中文", "主对话里英文的提示语要改成中文"),
        ("今天：修小药丸", "修小药丸"),
        ("今天修一下：路由规则改掉", "今天修一下：路由规则改掉"),
        ("no rush: tidy the photos", "tidy the photos"),
        ("Please, no rush: tidy the photos", "tidy the photos"),
        ("today, fix the pill", "fix the pill"),
        ("Whenever possible, update the docs", "Whenever possible, update the docs"),
        ("today", "today"),
        ("today:", "today"),
        ("Today's news summary", "Today's news summary"),
        # A path reads as its file's name; a link stays whole.
        (
            "Read /Users/alex/Documents/notes/todo.md and tell me what is left",
            "Read todo.md and tell me what is left",
        ),
        ("see https://example.com/a/b.html", "see https://example.com/a/b.html"),
        ("Tidy ~/Pictures/trip/2024/ now", "Tidy ~/Pictures/trip/2024/ now"),
        # A long sentence breaks at a clause of its own, a comma away.
        (
            "DEFAULT_WORKSPACE is defined twice, in support/gateway/task_policy.py "
            "and support/gateway/console/registry.py. Make registry.py import it.",
            "DEFAULT_WORKSPACE is defined twice",
        ),
        # A short lead-in is no title: the cut falls at a word instead, with no
        # "in" or "and" left hanging.
        (
            "Fix this today: in tests/cli/test_e2e_pty.py the list_files test "
            "only checks that the word list_files appears on screen.",
            "Fix this today: in test_e2e_pty.py…",
        ),
        (
            "No rush, whenever you are free: docs/guides/embedding.md and "
            "docs/zh-CN/guides/embedding.md have no picture.",
            "whenever you are free: embedding.md…",
        ),
        (
            "summarize every chapter of the handbook and write the findings down",
            "summarize every chapter of the handbook…",
        ),
        (
            "Please fix the failing test in support/gateway/console/routes_turns.py "
            "and run the suite",
            "fix the failing test in routes_turns.py and run…",
        ),
        # Chinese counts two columns a character and breaks between characters.
        (
            "刚才那几个任务里，哪个提交了几个文件？简单说。",
            "刚才那几个任务里，哪个提交了几个文件？",
        ),
        (
            "请把这个仓库里所有测试文件的名字都列出来然后统计每个文件有多少个测试函数",
            "把这个仓库里所有测试文件的名字都列出来然后统计…",
        ),
        (
            "帮我整理一下这周的会议记录并发给团队成员们看看吧好吗谢谢你了",
            "整理一下这周的会议记录并发给团队成员们看看吧好…",
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


def test_a_huge_first_line_costs_no_more_than_a_short_one() -> None:
    assert_linear(lambda size: "word/" * (size // 5), short_title)
