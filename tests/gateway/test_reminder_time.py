"""Reminders in code: the cue, the time it names and the work that is left.

Every case reads against Wednesday 2026-10-28 10:00 on the person's wall clock.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from localmelo.support.gateway.chore_schedule import Schedule
from localmelo.support.gateway.reminder_time import (
    Reminder,
    parse_reminder,
    parse_when,
)

NOW = datetime(2026, 10, 28, 10, 0)  # a Wednesday


def hm(hour: int, minute: int = 0) -> int:
    return hour * 60 + minute


def read(text: str, now: datetime = NOW) -> tuple[str, str, str, str]:
    """``(work, date, HH:MM, problem)`` of a reminder; the date and time are empty without one."""
    found = parse_reminder(text, now)
    assert found is not None, text
    when = found.on.isoformat() if found.on else ""
    clock = f"{found.minute // 60:02d}:{found.minute % 60:02d}" if found.on else ""
    return found.work, when, clock, found.problem


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Days and clock times.
        (
            "remind me to send the report on Friday at 9",
            ("send the report", "2026-10-30", "09:00"),
        ),
        ("Remind me tomorrow at 9am to call mom", ("call mom", "2026-10-29", "09:00")),
        (
            "remind me tomorrow to send the report",
            ("send the report", "2026-10-29", "09:00"),
        ),
        (
            "remind me at 14:30 to join the call",
            ("join the call", "2026-10-28", "14:30"),
        ),
        ("remind me to log in tomorrow at 7:45 am", ("log in", "2026-10-29", "07:45")),
        (
            "remind me tomorrow evening at 8 to take pills",
            ("take pills", "2026-10-29", "20:00"),
        ),
        ("remind me tomorrow morning to stretch", ("stretch", "2026-10-29", "08:00")),
        ("remind me tomorrow noon to eat lunch", ("eat lunch", "2026-10-29", "12:00")),
        (
            "set a reminder for tomorrow at 5pm to call dad",
            ("call dad", "2026-10-29", "17:00"),
        ),
        ("reminder: stretch tomorrow 6pm", ("stretch", "2026-10-29", "18:00")),
        (
            "don't let me forget to pay rent on Nov 2",
            ("pay rent", "2026-11-02", "09:00"),
        ),
        (
            "remind me about the meeting on 2026-11-02 at 15:00",
            ("the meeting", "2026-11-02", "15:00"),
        ),
        ("remind me to call at 9 on the 30th of nov", ("call", "2026-11-30", "09:00")),
        # 12 am and 12 pm, 24-hour times, a bare small hour.
        ("remind me at 12 am to sleep", ("sleep", "2026-10-29", "00:00")),
        ("remind me at midnight to sleep", ("sleep", "2026-10-29", "00:00")),
        ("remind me at 12 pm to eat", ("eat", "2026-10-28", "12:00")),
        ("remind me at 12:30 pm to eat", ("eat", "2026-10-28", "12:30")),
        ("remind me at 5 to leave", ("leave", "2026-10-28", "17:00")),
        ("remind me tomorrow at 7 to run", ("run", "2026-10-29", "07:00")),
        ("remind me tomorrow at 21:15 to lock up", ("lock up", "2026-10-29", "21:15")),
        # Relative times.
        ("remind me in 2 hours to drink water", ("drink water", "2026-10-28", "12:00")),
        ("remind me in half an hour to stretch", ("stretch", "2026-10-28", "10:30")),
        ("remind me in 45 min to call", ("call", "2026-10-28", "10:45")),
        ("remind me in an hour to leave", ("leave", "2026-10-28", "11:00")),
        (
            "remind me in 3 days to renew the card",
            ("renew the card", "2026-10-31", "10:00"),
        ),
        (
            "remind me in 2 weeks to renew the card",
            ("renew the card", "2026-11-11", "10:00"),
        ),
        # A bare time: today when it is ahead, else tomorrow.
        ("remind me at 9:15 to stretch", ("stretch", "2026-10-29", "09:15")),
        ("remind me at 11 to stretch", ("stretch", "2026-10-28", "11:00")),
        (
            "remind me in the morning to take pills",
            ("take pills", "2026-10-29", "08:00"),
        ),
        # Today's words: ahead is today.
        (
            "remind me tonight to lock the door",
            ("lock the door", "2026-10-28", "21:00"),
        ),
        ("remind me this evening at 7 to call", ("call", "2026-10-28", "19:00")),
        ("remind me today at 3pm to leave", ("leave", "2026-10-28", "15:00")),
    ],
)
def test_english_reminders(text: str, expected: tuple[str, str, str]) -> None:
    assert read(text) == (*expected, "")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("提醒我明天九点给导师发邮件", ("给导师发邮件", "2026-10-29", "09:00")),
        ("记得明天九点给导师发邮件", ("给导师发邮件", "2026-10-29", "09:00")),
        ("别忘了周五交报告", ("交报告", "2026-10-30", "09:00")),
        ("明天早上八点提醒我吃药", ("吃药", "2026-10-29", "08:00")),
        ("提醒我周三下午三点开会", ("开会", "2026-11-04", "15:00")),
        ("提醒我今晚八点吃药", ("吃药", "2026-10-28", "20:00")),
        ("提醒我明晚九点半关灯", ("关灯", "2026-10-29", "21:30")),
        ("提醒我后天中午十二点吃饭", ("吃饭", "2026-10-30", "12:00")),
        ("提醒我10月30日交报告", ("交报告", "2026-10-30", "09:00")),
        ("两小时后提醒我喝水", ("喝水", "2026-10-28", "12:00")),
        ("半小时后提醒我关火", ("关火", "2026-10-28", "10:30")),
        ("一个半小时后提醒我取快递", ("取快递", "2026-10-28", "11:30")),
        ("三天后提醒我续费", ("续费", "2026-10-31", "10:00")),
        ("提醒我晚上七点半吃饭", ("吃饭", "2026-10-28", "19:30")),
        # 下周X is that day of the Monday-to-Sunday week after this one.
        ("提醒我下周一交作业", ("交作业", "2026-11-02", "09:00")),
        ("提醒我下周日交作业", ("交作业", "2026-11-08", "09:00")),
        ("提醒我下周五交作业", ("交作业", "2026-11-06", "09:00")),
    ],
)
def test_chinese_reminders(text: str, expected: tuple[str, str, str]) -> None:
    assert read(text) == (*expected, "")


def test_a_weekday_is_the_next_one_and_today_is_next_week() -> None:
    assert read("remind me on Wednesday to pay")[1] == "2026-11-04"
    assert read("remind me on Thursday to pay")[1] == "2026-10-29"
    assert read("remind me on Tuesday to pay")[1] == "2026-11-03"
    assert read("remind me next Monday to pay")[1] == "2026-11-02"
    assert read("remind me on sat to pay")[1] == "2026-10-31"
    # On a Sunday, 下周一 is the next day; on a Monday it is the Monday after.
    assert read("提醒我下周一交作业", datetime(2026, 11, 1, 10, 0))[1] == "2026-11-02"
    assert read("提醒我下周一交作业", datetime(2026, 11, 2, 10, 0))[1] == "2026-11-09"
    assert read("提醒我周三交作业", datetime(2026, 10, 28, 10, 0))[1] == "2026-11-04"


def test_a_date_that_has_gone_this_year_is_next_year() -> None:
    assert read("remind me to call dad on Jan 5")[1] == "2027-01-05"
    assert read("remind me to call dad on Oct 27 at 9") == (
        "call dad",
        "2027-10-27",
        "09:00",
        "",
    )
    # Today's date with a time that has passed is past, not next year.
    assert read("remind me to call dad on Oct 28 at 9")[3] == "past"


def test_a_bare_time_that_has_passed_is_tomorrow_but_today_words_are_not() -> None:
    assert read("remind me at 9 to stand up")[1:3] == ("2026-10-29", "09:00")
    # The words say today: a time that has passed is a question, never tomorrow.
    for text in (
        "remind me today at 9am to stand up",
        "remind me this morning to call mom",
        "remind me tonight at 8 to call",  # 20:00 is ahead...
    ):
        found = parse_reminder(text, datetime(2026, 10, 28, 22, 0))
        assert found is not None and found.problem == "past", text
    assert parse_reminder("remind me tonight at 8 to call", NOW).problem == ""  # type: ignore[union-attr]
    # An explicit past day with a time is past too.
    assert read("remind me on 2026-10-01 to call dad")[3] == "past"
    assert read("remind me yesterday at 9 to call dad")[3] == "past"
    assert read("提醒我昨天九点开会")[3] == "past"
    # The minute itself is already past.
    assert (
        read("remind me at 10:00 to call", datetime(2026, 10, 28, 10, 0, 30))[1]
        == "2026-10-29"
    )


def test_a_strong_cue_without_a_time_asks_and_never_guesses() -> None:
    for text in (
        "remind me to call mom",
        "提醒我吃药",
        "set a reminder to send the report",
    ):
        found = parse_reminder(text, NOW)
        assert (
            found is not None and found.problem == "time" and found.schedule is None
        ), text
        assert found.work


def test_a_time_that_cannot_be_read_is_a_problem() -> None:
    for text in (
        "remind me at 25:00 to call",
        "remind me at 13pm to call",
        "remind me at 9:75 to call",
        "remind me on Feb 30 to renew it",
        "remind me on 2026-02-30 to renew it",
    ):
        assert read(text)[3] == "time", text


def test_a_time_more_than_a_year_ahead_is_far() -> None:
    assert read("remind me on 2028-01-05 to renew it")[3] == "far"
    assert read("remind me in 400 days to renew it")[3] == "far"
    assert (
        read("remind me on 2027-10-29 to renew it")[3] == ""
    )  # 366 days: still allowed


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "what time is it?",
        "remind me what the report said",
        "remind me tomorrow at 9",  # no work left
        "提醒我",
        "我记得明天有会",  # a weak cue is only at the start of the clause
        "你还记得明天九点的会吗？",
        "do you remember tomorrow at 9?",
        "记得保存文件",  # a weak cue with no time
        "don't forget to save the file",
        "summarize notes.md tomorrow",
    ],
)
def test_other_messages_are_not_reminders(text: str) -> None:
    assert parse_reminder(text, NOW) is None


def test_a_reminder_gives_a_one_time_schedule_only_with_a_usable_time() -> None:
    found = parse_reminder("remind me to send the report on Friday at 9", NOW)
    assert found == Reminder("send the report", date(2026, 10, 30), hm(9), "")
    assert found.schedule == Schedule("once", minute=hm(9), on=date(2026, 10, 30))  # type: ignore[union-attr]
    assert parse_reminder("remind me to send the report", NOW).schedule is None  # type: ignore[union-attr]


def test_numbers_that_are_not_times_are_left_in_the_work() -> None:
    assert read("remind me tomorrow to buy 3 apples") == (
        "buy 3 apples",
        "2026-10-29",
        "09:00",
        "",
    )
    assert read("remind me to meet at 5th avenue tomorrow")[0] == "meet at 5th avenue"
    assert read("remind me to take 2 pills at 8am")[0:3] == (
        "take 2 pills",
        "2026-10-29",
        "08:00",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Friday at 9", ("2026-10-30", "09:00")),
        ("tomorrow 9am", ("2026-10-29", "09:00")),
        ("in 2 hours", ("2026-10-28", "12:00")),
        ("at 5", ("2026-10-28", "17:00")),
        ("9pm please", ("2026-10-28", "21:00")),
        ("明天九点", ("2026-10-29", "09:00")),
        ("下周一", ("2026-11-02", "09:00")),
    ],
)
def test_an_answer_that_is_only_a_time(text: str, expected: tuple[str, str]) -> None:
    found = parse_when(text, NOW)
    assert found is not None and found.work == ""
    assert (found.on.isoformat(), f"{found.minute // 60:02d}:{found.minute % 60:02d}") == expected  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "text", ["hello", "yes please", "send the report tomorrow", "确定"]
)
def test_an_answer_with_more_than_a_time_is_not_one(text: str) -> None:
    assert parse_when(text, NOW) is None


def test_full_width_digits_and_colons_are_read() -> None:
    assert read("提醒我明天９：３０开会") == ("开会", "2026-10-29", "09:30", "")


def test_a_long_message_is_not_scanned() -> None:
    long = "remind me to call mom tomorrow at 9 " + "-" * 5000
    assert parse_reminder(long, NOW) is None
    assert parse_when("tomorrow " + "-" * 5000, NOW) is None
    assert (
        parse_reminder("remind me to call mom tomorrow at 9 " + "-" * 900, NOW)
        is not None
    )
