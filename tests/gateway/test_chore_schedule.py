"""Chore schedules: when a chore is next due, and the phrases that name one."""

from __future__ import annotations

from datetime import UTC, date, datetime, tzinfo
from zoneinfo import ZoneInfo

import pytest

from localmelo.support.gateway.chore_schedule import (
    MAX_AHEAD,
    MAX_HOURS,
    Schedule,
    ScheduleError,
    clock_text,
    parse_phrase,
)
from localmelo.support.gateway.i18n import said_text


def at(text: str, tz: tzinfo = UTC) -> float:
    """Epoch seconds of a local ``YYYY-MM-DD HH:MM`` in *tz*."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=tz).timestamp()


def test_daily_runs_today_when_the_time_has_not_come_else_tomorrow() -> None:
    schedule = Schedule("daily", minute=8 * 60 + 30)
    assert schedule.next_after(at("2026-10-28 07:00"), UTC) == at("2026-10-28 08:30")
    # Strictly after: at the minute itself the next run is tomorrow's.
    assert schedule.next_after(at("2026-10-28 08:30"), UTC) == at("2026-10-29 08:30")
    assert schedule.next_after(at("2026-10-28 23:59"), UTC) == at("2026-10-29 08:30")


def test_weekdays_skip_the_weekend() -> None:
    schedule = Schedule("weekdays", minute=8 * 60 + 30)
    # 2026-10-30 is a Friday; 31 and 1 are the weekend.
    assert datetime.fromtimestamp(at("2026-10-30 09:00"), UTC).weekday() == 4
    assert schedule.next_after(at("2026-10-30 09:00"), UTC) == at("2026-11-02 08:30")
    assert schedule.next_after(at("2026-10-31 12:00"), UTC) == at("2026-11-02 08:30")
    assert schedule.next_after(at("2026-11-02 08:00"), UTC) == at("2026-11-02 08:30")


def test_every_n_hours_counts_from_the_given_time() -> None:
    schedule = Schedule("hours", hours=4)
    assert schedule.next_after(1000.0) == 1000.0 + 4 * 3600


def test_a_daily_time_stays_on_the_wall_clock_across_a_clock_change() -> None:
    new_york = ZoneInfo("America/New_York")
    schedule = Schedule("daily", minute=8 * 60 + 30)
    # Clocks go back on 2026-11-01: the day is 25 hours long, 8:30 stays 8:30.
    assert schedule.next_after(at("2026-10-31 09:00", new_york), new_york) == at(
        "2026-11-01 08:30", new_york
    )
    assert schedule.next_after(at("2026-11-01 09:00", new_york), new_york) == at(
        "2026-11-02 08:30", new_york
    )


def test_a_schedule_round_trips_through_json_and_says_itself() -> None:
    for data, words in (
        ({"kind": "weekdays", "at": "08:30"}, "every weekday at 8:30"),
        ({"kind": "daily", "at": "21:05"}, "every day at 21:05"),
        ({"kind": "hours", "every": 4}, "every 4 hours"),
        ({"kind": "hours", "every": 1}, "every hour"),
    ):
        schedule = Schedule.from_json(data)
        assert schedule.to_json() == data
        assert schedule.describe() == words
    assert clock_text(0) == "0:00" and clock_text(23 * 60 + 59) == "23:59"


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {"kind": "weekly", "at": "08:30"},
        {"kind": "daily"},
        {"kind": "daily", "at": "8:3"},
        {"kind": "daily", "at": "24:00"},
        {"kind": "daily", "at": "08:60"},
        {"kind": "daily", "at": "08:30", "every": 2},
        {"kind": "hours", "every": 0},
        {"kind": "hours", "every": MAX_HOURS + 1},
        {"kind": "hours", "every": True},
        {"kind": "hours", "every": "4"},
        {"kind": "hours", "at": "08:30"},
    ],
)
def test_a_schedule_that_is_not_one_of_the_three_is_refused(data: object) -> None:
    with pytest.raises(ScheduleError):
        Schedule.from_json(data)


@pytest.mark.parametrize(
    ("text", "when", "work"),
    [
        (
            "every weekday at 8:30, brief me on my calendar",
            "every weekday at 8:30",
            "brief me on my calendar",
        ),
        (
            "brief me on my calendar every weekday at 8:30",
            "every weekday at 8:30",
            "brief me on my calendar",
        ),
        (
            "Every weekday morning at 8 summarize my email",
            "every weekday at 8:00",
            "summarize my email",
        ),
        ("every day at 9am check the news", "every day at 9:00", "check the news"),
        ("every day at 9 amazing things", "every day at 9:00", "amazing things"),
        ("daily at 6 pm back up notes.md", "every day at 18:00", "back up notes.md"),
        (
            "every evening at 8 remind me to stretch",
            "every day at 20:00",
            "remind me to stretch",
        ),
        ("every morning summarize the news", "every day at 8:00", "summarize the news"),
        ("every night at 11 pm lock up", "every day at 23:00", "lock up"),
        ("every day at noon eat lunch", "every day at 12:00", "eat lunch"),
        ("daily check the build", "every day at 9:00", "check the build"),
        (
            "Monday to Friday at 7:45 post the standup",
            "every weekday at 7:45",
            "post the standup",
        ),
        (
            "remind me every day at 9 to stretch",
            "every day at 9:00",
            "remind me to stretch",
        ),
        ("every 2 hours check disk space", "every 2 hours", "check disk space"),
        ("hourly check the build", "every hour", "check the build"),
        ("check the build every hour", "every hour", "check the build"),
        ("每天早上九点 总结我的邮件", "every day at 9:00", "总结我的邮件"),
        ("每天早上9点30分提醒我喝水", "every day at 9:30", "提醒我喝水"),
        ("每个工作日 8:30 帮我看日历", "every weekday at 8:30", "帮我看日历"),
        ("工作日早上八点半 整理待办", "every weekday at 8:30", "整理待办"),
        ("每晚十点 备份笔记", "every day at 22:00", "备份笔记"),
        ("每天下午三点 写日报", "every day at 15:00", "写日报"),
        ("每天提醒我喝水", "every day at 9:00", "提醒我喝水"),
        ("每小时 检查一下构建", "every hour", "检查一下构建"),
        ("每隔两小时提醒我站起来", "every 2 hours", "提醒我站起来"),
        ("每隔2个小时看一下服务", "every 2 hours", "看一下服务"),
    ],
)
def test_a_repeating_phrase_gives_the_schedule_and_the_work_that_is_left(
    text: str, when: str, work: str
) -> None:
    parsed = parse_phrase(text)
    assert parsed is not None
    assert (parsed.schedule.describe(), parsed.work) == (when, work)
    assert parsed.phrase and parsed.phrase in text


@pytest.mark.parametrize(
    "text",
    [
        "make slides from notes.md",
        "what time is it?",
        "write a poem",
        "你好",
        # Repeating, but not one of the three schedules: left to the model, as before.
        "every Monday at 9 write a report",
        "every 30 minutes ping the server",
        "每周一 开会",
        # A time that is not a time, or nothing left to do.
        "every day at 25 do the thing",
        "every day at 9",
        "每天早上九点",
        "every 99 hours check",
    ],
)
def test_a_message_that_names_no_supported_schedule_is_not_read(text: str) -> None:
    assert parse_phrase(text) is None


# ── One-time schedules (reminders) ──

FRIDAY_NINE = {"kind": "once", "at": "2026-10-30T09:00"}


def test_a_one_time_schedule_round_trips_and_says_its_date_in_both_languages() -> None:
    schedule = Schedule.from_json(FRIDAY_NINE)
    assert schedule == Schedule("once", minute=9 * 60, on=date(2026, 10, 30))
    assert schedule.to_json() == FRIDAY_NINE
    assert schedule.describe() == "once, Fri Oct 30 at 9:00"
    assert said_text(schedule.said_at(), "en") == "Fri Oct 30 at 9:00"
    assert said_text(schedule.said(), "zh-Hans") == "仅一次，10月30日 周五 9:00"
    late = Schedule("once", minute=21 * 60 + 5, on=date(2027, 1, 3))
    assert late.describe() == "once, Sun Jan 3 at 21:05"


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "once"},
        {"kind": "once", "at": "09:00"},
        {"kind": "once", "at": "2026-10-30"},
        {"kind": "once", "at": "2026-10-30 09:00"},
        {"kind": "once", "at": "2026-10-30T09:00:00"},
        {"kind": "once", "at": "2026-02-30T09:00"},
        {"kind": "once", "at": "2026-10-30T24:00"},
        {"kind": "once", "at": "2026-10-30T09:60"},
        {"kind": "once", "at": 5},
        {"kind": "once", "at": "2026-10-30T09:00", "every": 2},
        {"kind": "daily", "at": "2026-10-30T09:00"},
    ],
)
def test_a_one_time_schedule_that_is_not_a_date_and_a_time_is_refused(
    data: object,
) -> None:
    with pytest.raises(ScheduleError):
        Schedule.from_json(data)


def test_a_one_time_schedule_is_due_at_its_own_time_only_if_ahead_and_within_a_year() -> (
    None
):
    schedule = Schedule.from_json(FRIDAY_NINE)
    due = at("2026-10-30 09:00")
    assert schedule.when(UTC) == due
    # The next time is its own time, whatever day it is asked on.
    assert schedule.next_after(at("2026-10-28 10:00"), UTC) == due
    assert schedule.next_after(due - 60, UTC) == due
    # Past, or the very minute: no next time.
    for now in (due, due + 1, at("2026-11-02 09:00")):
        with pytest.raises(ScheduleError, match="passed"):
            schedule.next_after(now, UTC)
    # At most a year ahead.
    far = Schedule("once", minute=0, on=date(2027, 11, 5))
    with pytest.raises(ScheduleError, match="year"):
        far.next_after(at("2026-10-28 10:00"), UTC)
    edge = datetime(2026, 10, 28, 10, 0, tzinfo=UTC) + MAX_AHEAD
    inside = Schedule("once", minute=10 * 60, on=edge.date())
    assert inside.next_after(at("2026-10-28 10:00"), UTC) == inside.when(UTC)


def test_a_one_time_time_is_on_the_wall_clock_of_the_given_zone() -> None:
    new_york = ZoneInfo("America/New_York")
    schedule = Schedule.from_json({"kind": "once", "at": "2026-11-01T08:30"})
    assert schedule.when(new_york) == at("2026-11-01 08:30", new_york)
    assert schedule.when(new_york) != schedule.when(UTC)


def test_only_a_one_time_schedule_has_a_date() -> None:
    with pytest.raises(ScheduleError):
        Schedule("once", minute=60)
    with pytest.raises(ScheduleError):
        Schedule("daily", minute=60, on=date(2026, 10, 30))
    with pytest.raises(ScheduleError):
        Schedule("daily", minute=60).when()
