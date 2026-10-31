"""The main chat's rules in code: router cases in English and Chinese, levels, the principal rule."""

from __future__ import annotations

import pytest

from localmelo.support.gateway.front_rules import (
    LEVEL_MEANINGS,
    LEVEL_WORDS,
    TaskRef,
    allows,
    allows_level,
    allows_message,
    intention_only,
    is_chinese,
    is_question,
    level_code,
    level_label,
    level_menu,
    level_number,
    level_word,
    reply_language,
    route,
    short_title,
    triage,
)

SLIDES = TaskRef("a1b2c3d4e5f6", "Slides for the 15:00 meeting", "running")
CI = TaskRef("0f0f0f0f0f0f", "Fix CI", "queued")
PHOTOS = TaskRef("111111111111", "整理照片", "parked")
DONE = TaskRef("222222222222", "Old report", "done")
TASKS = [SLIDES, CI, PHOTOS, DONE]


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("hello", "answer"),
        ("Hi there!", "answer"),
        ("你好", "answer"),
        ("谢谢", "answer"),
        ("what time is my meeting?", "answer"),
        ("我的组会是哪天？", "answer"),
        ("summarize notes.md", "plan"),
        ("write a report on battery chemistry", "plan"),
        (
            "what is in ~/talk/notes.md?",
            "plan",
        ),  # names a file: the front's step decides
        ("帮我写一份周报", "plan"),
        # Repeating phrases are chores, left for the scheduled tab.
        ("every weekday at 8:30 remind me to stretch", "chore"),
        ("每天早上提醒我喝水", "chore"),
        ("pause the report", "plan"),  # no open task is named so
        ("stop the old report", "plan"),  # a finished task is not changed
        ("stop worrying about the slides font and make it bigger please", "plan"),
        ("don't stop the slides", "plan"),
        ("别停 Fix CI", "plan"),
    ],
)
def test_routes(typed: str, expected: str) -> None:
    assert route(typed, TASKS).route == expected


@pytest.mark.parametrize(
    ("typed", "action", "task"),
    [
        ("stop the slides", "stop", SLIDES),
        ("Pause Fix CI", "pause", CI),
        ("make the slides first", "first", SLIDES),
        ("slides top priority", "first", SLIDES),
        ("pause a1b2c3d4e5f6", "pause", SLIDES),
        ("暂停 Fix CI", "pause", CI),
        ("继续整理照片", "resume", PHOTOS),
        ("停止 整理照片", "stop", PHOTOS),
        ("先做 slides", "first", SLIDES),
    ],
)
def test_control_phrases_name_a_task(typed: str, action: str, task: TaskRef) -> None:
    routed = route(typed, TASKS)
    assert (routed.route, routed.action, routed.task) == ("control", action, task)


def test_delegation_by_command_or_composer_action() -> None:
    assert route("/task make slides for 3pm", TASKS).text == "make slides for 3pm"
    assert route("/task", TASKS).text == ""
    assert route("hello", TASKS, as_task=True).route == "delegate"
    # A control phrase sent as a task is a task, not a control.
    assert route("stop the slides", TASKS, as_task=True).route == "delegate"


@pytest.mark.parametrize(
    ("typed", "level"),
    [
        ("make slides from notes.md", 2),
        ("make slides, it's urgent", 1),
        ("make slides by 3pm", 1),
        ("finish it before 15:00", 1),
        ("尽快整理照片", 1),
        ("今天下午三点前做完", 1),
        ("make slides, no rush", 3),
        ("this is not urgent", 3),
        ("不急，慢慢来", 3),
        # A word that only looks like a deadline is none.
        ("summarize today's news", 2),
        ("split the bill by 5 people", 2),
        # The bare word for today, anywhere in the request, is Today.
        ("make slides today", 1),
        ("today, make slides from notes.md", 1),
        ("clean up the inbox, today if you can", 1),
        ("今天整理照片", 1),
        ("帮我整理照片，今天", 1),
        ("整理今天的照片", 1),
        # Negated, it asks for nothing; a possessive only names a thing about today.
        ("not today", 2),
        ("make slides tomorrow, not today", 2),
        ("no need to finish it today", 2),
        ("今天不用整理照片", 2),
        ("今天不用", 2),
        ("不是今天要做，先记着", 2),
        ("明天再做", 2),
        ("what does today's news say", 2),
        ("summarize today's and yesterday's news", 2),
        # Ease still wins: "no rush" says whenever, whatever else the message holds.
        ("make slides today, no rush", 3),
    ],
)
def test_triage_from_the_persons_words(typed: str, level: int) -> None:
    assert triage(typed)[0] == level


def test_the_bare_today_is_the_reason_too() -> None:
    assert triage("do it today")[1] == 'you said "today"'
    assert triage("今天整理照片")[1] == 'you said "今天"'
    assert triage("not today")[1] == "you asked for it"
    # The person's own "today" raises a task they name; a negated one does not.
    assert allows_level("the slides today please", True, SLIDES)
    assert not allows_level("not the slides today", True, SLIDES)


def test_levels_in_words() -> None:
    # The words stay what a person may type and what the API's level_word answers.
    assert LEVEL_WORDS == ("Now", "Today", "Normal", "Whenever", "If idle", "Later")
    assert [level_word(n) for n in (0, 2, 5)] == ["Now", "Normal", "Later"]
    assert level_word(None) == "Normal"
    assert [level_number(v) for v in ("today", "If idle", "P3", 4, "soon", 9)] == [
        1,
        4,
        3,
        4,
        None,
        None,
    ]
    assert triage("urgent please")[1] == 'you said "urgent"'


def test_every_level_reads_as_its_code_and_meaning() -> None:
    assert LEVEL_MEANINGS == (
        "runs first; others wait",
        "today",
        "normal",
        "when there is time",
        "only when the computer is idle",
        "on hold until you start it",
    )
    assert [level_code(n) for n in range(6)] == [f"P{n}" for n in range(6)]
    assert [level_label(n) for n in range(6)] == [
        f"P{n} · {meaning}" for n, meaning in enumerate(LEVEL_MEANINGS)
    ]
    assert level_label(1) == "P1 · today"
    # Anything that is not a level reads as P2, as level_word does.
    assert [level_code(v) for v in (None, 6, -1, "1", True)] == ["P2"] * 5
    assert level_menu().startswith("P0 (runs first; others wait), P1 (today), P2")


def test_old_words_and_codes_set_the_same_level() -> None:
    same = (
        (0, ("Now", "now", "P0", "p0", "0", 0)),
        (1, ("Today", "today", " TODAY ", "P1", "p1", "1", 1)),
        (2, ("Normal", "normal", "P2", "p2", 2)),
        (3, ("Whenever", "whenever", "P3", "p3", 3)),
        (4, ("If idle", "if  idle", "P4", "p4", 4)),
        (5, ("Later", "later", "P5", "p5", 5)),
    )
    for level, spellings in same:
        assert {level_number(spelling) for spelling in spellings} == {level}
    assert [level_number(v) for v in ("P6", "p", "6", "-1", "", True)] == [None] * 6


def test_questions() -> None:
    assert is_question("how did the slides go?")
    assert is_question("when is my group meeting")
    assert is_question("组会是星期几")
    assert not is_question("make slides for 3pm")


def test_the_principal_rule_reads_only_the_persons_own_words() -> None:
    assert allows("please stop the slides one, I don't need it", "stop", SLIDES)
    assert allows("暂停 Fix CI", "pause", CI)
    # Typed words that ask for something else, or name another task, allow nothing.
    assert not allows("summarize the meeting page", "stop", SLIDES)
    assert not allows("pause Fix CI", "stop", CI)
    assert not allows("stop Fix CI", "stop", SLIDES)
    assert not allows("don't stop the slides", "stop", SLIDES)
    assert not allows("stop", "stop", SLIDES)  # no task named
    assert allows_level("make the slides urgent please", True, SLIDES)
    assert allows_level("the slides can wait, no rush", False, SLIDES)
    assert not allows_level("make it urgent", True, SLIDES)
    assert not allows_level("the slides are done?", True, SLIDES)


# ── Questions, preambles, work and follow-ups ──


def test_a_question_mark_anywhere_or_quick_question_is_a_question() -> None:
    assert is_question(
        "Quick question: which file defines DEFAULT_WORKSPACE? Answer in one line."
    )
    assert is_question("quick question: the folder for notes")
    assert is_question("请问 notes.md 在哪个文件夹")
    # A link's query string is not a question.
    assert not is_question("summarize https://example.com/page?id=3")


@pytest.mark.parametrize(
    "reply",
    [
        "I'll start by looking at the test file and finding the relevant code.",
        "Sure! Let me check.",
        "Let me look at the tests first. Then I'll fix the check and commit.",
        "I'll start by:\n1. Reading the test\n2. Fixing it",
        "OK, I'm going to search the repository for it…",
        "我先看一下测试文件。",
        "好的，我来检查一下这个文件，然后修改它。",
    ],
)
def test_a_reply_that_only_announces_what_it_will_do(reply: str) -> None:
    assert intention_only(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "Let me know if you need more.",
        "I'll need the folder name to continue.",
        "Let me check. The file is support/gateway/task_policy.py.",
        "Which folder should I look in?",
        "I'll look at it — which branch?",
        "Started the slides; I'll tell you when they're done.",
        "DEFAULT_WORKSPACE is defined in support/gateway/task_policy.py:24.",
        "已交给后台任务处理：放宽优先级判定。",
        "我先说结论：在 task_policy.py 里。是否还要别的？",
        "",
    ],
)
def test_an_answer_is_not_a_preamble(reply: str) -> None:
    assert not intention_only(reply)


@pytest.mark.parametrize(
    ("typed", "title"),
    [
        (
            "Please fix this today: in tests/cli/test_e2e_pty.py the list_files "
            "test only checks one word. Make it check a file name and commit.",
            "fix this today: in test_e2e_pty.py",
        ),
        (
            "add a test for the parser in tests/test_app.py",
            "add a test for the parser in test_app.py",
        ),
        ("In support/x.py, rename foo to bar", "In x.py, rename foo to bar"),
        (
            "把 docs/guides/embedding.md 里的图片换成 webp",
            "把 embedding.md 里的图片换成",
        ),
        ("commit support/gateway/console/registry.py", "commit registry.py"),
    ],
)
def test_an_imperative_change_to_named_files_is_work(typed: str, title: str) -> None:
    routed = route(typed, TASKS)
    assert (routed.route, routed.title, routed.text) == ("work", title, typed)


@pytest.mark.parametrize(
    "typed",
    [
        "Quick question: which file defines DEFAULT_WORKSPACE? Answer in one line.",
        "can you fix tests/cli/test_e2e_pty.py",  # a question: the front's step
        "summarize notes.md",  # reads, changes nothing
        "make slides from notes.md",  # makes something new: the front's step
        "fix and/or improve the docs",  # no file
        "write a report on battery chemistry",
        "The report task stopped before its commit. Please finish it: commit docs/a.md",
        "finish the old report: commit docs/report.md",  # names a task: may follow it
    ],
)
def test_other_messages_are_not_work(typed: str) -> None:
    assert route(typed, TASKS).route != "work"


def test_short_titles_cut_at_a_word() -> None:
    long = "summarize every chapter of the handbook and write the findings down"
    # At most 48 characters, at a word, without a trailing "and".
    assert short_title(long) == "summarize every chapter of the handbook"
    assert short_title("Fix CI. Then tell me.") == "Fix CI"
    assert short_title("x" * 80) == "x" * 48
    assert short_title(
        "帮我整理一下这周的会议记录并发给团队成员们看看吧好吗谢谢你了"
    ) == ("整理一下这周的会议记录并发给团队成员们看看吧好吗")
    assert short_title("see https://example.com/a/b.html") == (
        "see https://example.com/a/b.html"
    )
    assert short_title("   ") == "Task"


@pytest.mark.parametrize(
    ("typed", "task"),
    [
        ("continue the old report", DONE),
        ("继续 Old report", DONE),
        ("commit Fix CI", CI),
        (f"finish {DONE.task_id}", DONE),
    ],
)
def test_a_follow_up_word_and_a_named_task_follow_it(typed: str, task: TaskRef) -> None:
    routed = route(typed, TASKS)
    assert (routed.route, routed.task) == ("follow", task)


def test_follow_ups_that_code_leaves_to_the_front() -> None:
    assert route("finish it", TASKS).route == "plan"  # "it": the front knows which
    assert route("did you finish the old report?", TASKS).route != "follow"
    assert route("don't commit Fix CI", TASKS).route != "follow"


def test_a_message_needs_the_persons_own_words_pointing_at_the_task() -> None:
    assert allows_message("tell the slides to use a dark theme", SLIDES, newest=False)
    assert allows_message("finish the old report: commit it", DONE, newest=False)
    # "it" points at the newest task the main chat showed, and only with a
    # follow-up word.
    assert allows_message("finish it", DONE, newest=True)
    assert allows_message("继续那个任务", DONE, newest=True)
    assert not allows_message("finish it", DONE, newest=False)
    assert not allows_message("summarize notes.md", DONE, newest=True)
    assert not allows_message("summarize notes.md", SLIDES, newest=False)
    assert not allows_message("", DONE, newest=True)


def test_the_language_code_writes_in_follows_the_script() -> None:
    assert (reply_language("fix it"), reply_language("修一下")) == (
        "English",
        "Chinese",
    )
    assert is_chinese("把 a.md 改一下") and not is_chinese("fix a.md")


def test_a_title_is_named_without_its_filler_words() -> None:
    e2e = TaskRef("444444444444", "Fix the e2e check", "done")
    routed = route("commit Fix the e2e check", [*TASKS, e2e])
    assert (routed.route, routed.task) == ("follow", e2e)
    assert route("pause the slides for the 15:00 meeting", TASKS).task == SLIDES
