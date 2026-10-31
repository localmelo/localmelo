"""Chore schedules in code: when a chore is next due, and the phrases that name one.

A chore repeats on one of three schedules: every day at a time, every weekday
at a time, or every N hours; or it runs once, on a date at a time (a reminder).
:meth:`Schedule.next_after` answers "when next?"
from a clock reading and an optional time zone (the computer's own when
``None``), and :func:`parse_phrase` reads a repeating phrase in a message
("every weekday at 8:30, brief me on my calendar", "每天早上九点 …") into a
schedule and the work that is left. Both are plain functions with no clock,
store or model: the same phrase always gives the same answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Final, Literal

from localmelo.support.gateway.i18n import said, said_text

Kind = Literal["daily", "weekdays", "hours", "once"]
KINDS: Final = ("daily", "weekdays", "hours", "once")
MAX_HOURS: Final = 24
# A one-time schedule is at most this far ahead.
MAX_AHEAD: Final = timedelta(days=366)
# A run starts up to this long after its time, so chores do not all start on the minute.
JITTER_S: Final = 120.0
# What a phrase without a time means: a day part's own hour, else this.
DEFAULT_MINUTE: Final = 9 * 60


class ScheduleError(ValueError):
    """A schedule that is not one of the kinds, or out of range."""


def clock_text(minute: int) -> str:
    """``8:30`` for 510 minutes after midnight."""
    return f"{minute // 60}:{minute % 60:02d}"


@dataclass(frozen=True)
class Schedule:
    """``daily`` and ``weekdays`` run at *minute* (after local midnight); ``hours`` every *hours*.

    ``once`` runs on the local date *on* at *minute* and then never again.
    """

    kind: Kind
    minute: int = 0
    hours: int = 0
    on: date | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ScheduleError("kind: every day, every weekday, every N hours or once")
        if (self.on is not None) != (self.kind == "once"):
            raise ScheduleError("on: a date, for a one-time schedule only")
        if self.kind == "hours":
            if not 1 <= self.hours <= MAX_HOURS:
                raise ScheduleError(f"every: 1 to {MAX_HOURS} hours")
        elif not 0 <= self.minute < 24 * 60:
            raise ScheduleError("at: a time of day, such as 08:30")

    @classmethod
    def from_json(cls, data: object) -> Schedule:
        """``{"kind": "weekdays", "at": "08:30"}``, ``{"kind": "hours", "every": 4}`` or ``{"kind": "once", "at": "2026-10-30T09:00"}``."""
        if not isinstance(data, Mapping):
            raise ScheduleError("schedule must be an object")
        kind = data.get("kind")
        extra = set(data) - {"kind", "every" if kind == "hours" else "at"}
        if extra:
            raise ScheduleError(f"unknown schedule fields: {', '.join(sorted(extra))}")
        if kind == "hours":
            every = data.get("every")
            if isinstance(every, bool) or not isinstance(every, int):
                raise ScheduleError(f"every: 1 to {MAX_HOURS} hours")
            return cls("hours", hours=every)
        if kind not in KINDS:
            raise ScheduleError("kind: every day, every weekday, every N hours or once")
        at = data.get("at")
        if kind == "once":
            return cls._once(at)
        found = re.fullmatch(r"(\d{1,2}):(\d{2})", at) if isinstance(at, str) else None
        if found is None or int(found[1]) > 23 or int(found[2]) > 59:
            raise ScheduleError("at: a time of day, such as 08:30")
        return cls(kind, minute=int(found[1]) * 60 + int(found[2]))

    @classmethod
    def _once(cls, at: object) -> Schedule:
        found = (
            re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})", at)
            if isinstance(at, str)
            else None
        )
        try:
            if found is None:
                raise ValueError
            year, month, day, hour, minute = (int(part) for part in found.groups())
            if hour > 23 or minute > 59:
                raise ValueError
            return cls("once", minute=hour * 60 + minute, on=date(year, month, day))
        except ValueError:
            raise ScheduleError(
                "at: a date and time, such as 2026-10-30T09:00"
            ) from None

    def to_json(self) -> dict[str, Any]:
        if self.kind == "hours":
            return {"kind": "hours", "every": self.hours}
        if self.on is not None:
            return {"kind": "once", "at": f"{self.on.isoformat()}T{self._clock()}"}
        return {
            "kind": self.kind,
            "at": self._clock(),
        }

    def _clock(self) -> str:
        return f"{self.minute // 60:02d}:{self.minute % 60:02d}"

    def describe(self) -> str:
        """``every weekday at 8:30``, ``every day at 21:00``, ``every 4 hours``, ``once, Fri Oct 30 at 9:00``."""
        if self.kind == "hours":
            return "every hour" if self.hours == 1 else f"every {self.hours} hours"
        if self.kind == "once":
            return said_text(self.said(), "en")
        day = "every weekday" if self.kind == "weekdays" else "every day"
        return f"{day} at {clock_text(self.minute)}"

    def said(self) -> dict[str, Any]:
        """:meth:`describe` as words a record keeps for the reader's language."""
        if self.kind == "hours":
            return said("schedule.hours", n=self.hours)
        if self.on is not None:
            return said("schedule.once", at=self.said_at())
        return said(f"schedule.{self.kind}", time=clock_text(self.minute))

    def said_at(self) -> dict[str, Any]:
        """A one-time schedule's date and time without "once": ``Fri Oct 30 at 9:00``."""
        if self.on is None:
            raise ScheduleError("only a one-time schedule has a date")
        weekday, month = self.on.weekday(), self.on.month
        return said(
            "schedule.at",
            weekday=said(f"time.weekday.{weekday}"),
            month=said(f"time.month.{month}"),
            day=self.on.day,
            time=clock_text(self.minute),
        )

    def next_after(self, after: float, tz: tzinfo | None = None) -> float:
        """The first time strictly after *after* (epoch seconds) this schedule is due.

        A one-time schedule has one: its own time, which must be after *after*
        and at most :data:`MAX_AHEAD` later.
        """
        if self.on is not None:
            when = self.when(tz)
            if when <= after:
                raise ScheduleError("That time has passed. Choose a new time.")
            if when > after + MAX_AHEAD.total_seconds():
                raise ScheduleError("Reminders can be at most a year ahead.")
            return when
        if self.kind == "hours":
            return after + self.hours * 3600.0
        today = datetime.fromtimestamp(after, tz).date()
        # Seven days cover a weekend; the loop ends on the first day that fits.
        for ahead in range(8):
            day = today + timedelta(days=ahead)
            if self.kind == "weekdays" and day.weekday() >= 5:
                continue
            at = _at(day, self.minute, tz)
            if at > after:
                return at
        raise ScheduleError("no next time")  # unreachable: a week always has a weekday

    def when(self, tz: tzinfo | None = None) -> float:
        """A one-time schedule's own time (epoch seconds), past or not."""
        if self.on is None:
            raise ScheduleError("only a one-time schedule has a time")
        return _at(self.on, self.minute, tz)


def _at(day: date, minute: int, tz: tzinfo | None) -> float:
    naive = datetime.combine(day, time(minute // 60, minute % 60))
    return (naive if tz is None else naive.replace(tzinfo=tz)).timestamp()


# ── Phrases ──


@dataclass(frozen=True)
class ParsedChore:
    """A repeating phrase read from a message: the schedule, the work left, the phrase."""

    schedule: Schedule
    work: str
    phrase: str


_ZH_DIGITS: Final = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                     "六": 6, "七": 7, "八": 8, "九": 9}  # fmt: skip
_ZH_NUM: Final = r"(?:\d{1,2}|[零〇一二两三四五六七八九十]{1,3})"
# A day part's default hour, and whether it moves a plain 1-12 hour to the afternoon.
_PART_HOUR: Final = {
    "morning": 8,
    "noon": 12,
    "afternoon": 14,
    "evening": 18,
    "night": 21,
}
_PM_PARTS: Final = frozenset({"afternoon", "evening", "night"})
_ZH_PARTS: Final = {"凌晨": "morning", "清晨": "morning", "早晨": "morning", "早上": "morning",
                    "早": "morning", "上午": "morning", "中午": "noon", "下午": "afternoon",
                    "傍晚": "evening", "晚上": "night", "晚间": "night", "晚": "night",
                    "夜里": "night", "夜间": "night"}  # fmt: skip

_I = re.IGNORECASE
_WEEKDAYS_EN = re.compile(
    r"\b(?:every\s+(?:week\s?day|workday|working\s+day|business\s+day)s?"
    r"|(?:on\s+)?weekdays|mon(?:day)?\s*(?:-|to|through|thru)\s*fri(?:day)?)\b",
    _I,
)
_HOURS_EN = re.compile(r"\b(?:hourly|every\s+(?:(?P<n>\d{1,2})\s+)?hours?)\b", _I)
_DAILY_EN = re.compile(
    r"\b(?:every\s?day|daily|each\s+day|every\s+(?P<part>morning|afternoon|evening|night))\b",
    _I,
)
_WEEKDAYS_ZH = re.compile(r"每个?工作日|工作日|每?周一[到至]周?五")
_HOURS_ZH = re.compile(rf"每隔?(?P<n>{_ZH_NUM})?个?小时")
_DAILY_ZH = re.compile(r"每天|每日|天天|每(?=[早晚])")
_PART_EN = re.compile(
    r"\s*,?\s*(?:in\s+the\s+)?(?P<part>morning|afternoon|evening|night)\b", _I
)
_PART_ZH = re.compile(
    r"的?(?P<part>" + "|".join(sorted(_ZH_PARTS, key=len, reverse=True)) + ")"
)
_AP = r"(?P<{0}>[ap]\.?m\.?)(?![a-z])"
_AP_MAYBE = r"(?:" + _AP + r")?"
_TIME_EN = re.compile(
    "".join(
        (
            r"\s*,?\s*(?:",
            r"(?:at|@|around|by)\s*(?P<h1>\d{1,2})(?:[:.](?P<m1>\d{2}))?\s*",
            _AP_MAYBE.format("a1"),
            r"|(?P<h2>\d{1,2}):(?P<m2>\d{2})\s*" + _AP_MAYBE.format("a2"),
            r"|(?P<h3>\d{1,2})\s*" + _AP.format("a3"),
            r"|at\s+(?P<word>noon|midday|midnight))",
        )
    ),
    _I,
)
_TIME_ZH = re.compile(
    rf"\s*(?P<h>{_ZH_NUM})\s*[点時时:：]\s*(?:(?P<m>{_ZH_NUM})\s*分?|(?P<half>半))?"
)
_LEAD: Final = re.compile(
    r"^(?:[\s,;:.，；：。、!！-]|please\b|and\b|then\b|to\b)+", _I
)
_TRAIL: Final = re.compile(r"[\s,;:，；：、-]+$")


def _zh_number(text: str) -> int | None:
    """0-99 from digits or Chinese numerals (``九``, ``十二``, ``二十五``)."""
    if text.isdigit():
        return int(text)
    if "十" in text:
        tens, _, ones = text.partition("十")
        high = 1 if tens == "" else _ZH_DIGITS.get(tens)
        low = 0 if ones == "" else _ZH_DIGITS.get(ones)
        return None if high is None or low is None else high * 10 + low
    return _ZH_DIGITS.get(text) if len(text) == 1 else None


def _hour_of(hour: int, ap: str, part: str) -> int | None:
    """A clock hour 0-23 from the hour as typed, ``am``/``pm`` or ``""``, and a day part."""
    if ap:
        if not 1 <= hour <= 12:
            return None
        return hour % 12 + (12 if ap[0].lower() == "p" else 0)
    if hour < 12 and (part in _PM_PARTS or (part == "noon" and hour <= 6)):
        hour += 12
    return hour if hour < 24 else None


Reader = Callable[[str, int], tuple[int | None, int, bool]]


def _time_en(text: str, at: int) -> tuple[int | None, int, bool]:
    """``(minute, end, ok)`` for the optional day part and time that follow a phrase at *at*."""
    part = ""
    found = _PART_EN.match(text, at)
    if found:
        part, at = found["part"].lower(), found.end()
    clock = _TIME_EN.match(text, at)
    if clock is None:
        return (_PART_HOUR[part] * 60 if part else None), at, True
    if clock["word"]:
        word = clock["word"].lower()
        return (0 if word == "midnight" else 12 * 60), clock.end(), True
    which = next(n for n in "123" if clock[f"h{n}"])
    ap = (clock[f"a{which}"] or "").replace(".", "")
    hour = _hour_of(int(clock[f"h{which}"]), ap, part)
    minute = int(clock[f"m{which}"]) if which != "3" and clock[f"m{which}"] else 0
    if hour is None or minute > 59:
        return None, at, False
    return hour * 60 + minute, clock.end(), True


def _time_zh(text: str, at: int) -> tuple[int | None, int, bool]:
    part = ""
    found = _PART_ZH.match(text, at)
    if found:
        part, at = _ZH_PARTS[found["part"]], found.end()
    clock = _TIME_ZH.match(text, at)
    if clock is None:
        return (_PART_HOUR[part] * 60 if part else None), at, True
    hour = _zh_number(clock["h"])
    minute = 30 if clock["half"] else _zh_number(clock["m"]) if clock["m"] else 0
    hour = None if hour is None else _hour_of(hour, "", part)
    if hour is None or minute is None or minute > 59:
        return None, at, False
    return hour * 60 + minute, clock.end(), True


def _cjk(char: str) -> bool:
    return bool(char) and "\u3400" <= char <= "\u9fff"


def _tidy(text: str) -> str:
    return _TRAIL.sub("", _LEAD.sub("", text.strip())).strip()


def parse_phrase(text: str) -> ParsedChore | None:
    """The schedule a message names and the work that is left, or ``None``.

    Reads "every day", "every weekday" (and "weekdays", "Monday to Friday"),
    "every morning/evening/night", "daily", "hourly" and "every N hours" in
    English, and 每天, 每个工作日, 每晚, 每小时, 每隔N小时 and the like in Chinese,
    each with an optional day part and time ("at 8:30", "at 9 pm", "早上九点半").
    A phrase with no time means its day part's hour, else 9:00. Anything else
    ("every Monday", "every 30 minutes") is not read: the message goes on as
    it always did. Nothing is left to do ("every day at 9") also gives ``None``.
    """
    rules: tuple[tuple[re.Pattern[str], Kind, Reader | None], ...] = (
        (_WEEKDAYS_EN, "weekdays", _time_en),
        (_WEEKDAYS_ZH, "weekdays", _time_zh),
        (_HOURS_EN, "hours", None),
        (_HOURS_ZH, "hours", None),
        (_DAILY_EN, "daily", _time_en),
        (_DAILY_ZH, "daily", _time_zh),
    )
    for pattern, kind, reader in rules:
        found = pattern.search(text)
        if found is None:
            continue
        end = found.end()
        if reader is None:
            raw = found.groupdict().get("n")
            every = (
                (_zh_number(raw) if raw else 1)
                if pattern is _HOURS_ZH
                else int(raw or 1)
            )
            if every is None or not 1 <= every <= MAX_HOURS:
                return None
            schedule = Schedule("hours", hours=every)
        else:
            if found.groupdict().get("part"):
                end = found.start(
                    "part"
                )  # "every evening at 8": the day part is read again
            minute, end, ok = reader(text, end)
            if not ok:
                return None
            schedule = Schedule(
                kind, minute=DEFAULT_MINUTE if minute is None else minute
            )
        left, right = text[: found.start()].strip(), text[end:].strip()
        joined = left + ("" if _cjk(left[-1:]) or _cjk(right[:1]) else " ") + right
        work = _tidy(joined)
        if len(work) < 2:
            return None
        return ParsedChore(schedule, work, text[found.start() : end].strip())
    return None


__all__ = [
    "DEFAULT_MINUTE",
    "JITTER_S",
    "KINDS",
    "MAX_HOURS",
    "ParsedChore",
    "Schedule",
    "ScheduleError",
    "clock_text",
    "parse_phrase",
]
