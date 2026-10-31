"""The main chat's rules in code: router cases in English and Chinese, levels, the principal rule."""

from __future__ import annotations

import pytest

from localmelo.support.gateway.front_rules import (
    LEVEL_WORDS,
    TaskRef,
    allows,
    allows_level,
    is_question,
    level_number,
    level_word,
    route,
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
