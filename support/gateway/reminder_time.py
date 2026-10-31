"""Reminders in code: the cue words in a message and the time they name.

"remind me to send the report on Friday at 9", "两小时后提醒我喝水":
:func:`parse_reminder` reads the cue (``remind me``, ``set a reminder``, ``提醒我``;
``don't let me forget``, ``记得``, ``别忘了`` count only with a time), the time
and the work that is left into a :class:`Reminder`. The time is a date and a
minute on the person's wall clock, resolved against *now*, so the same words and
the same *now* always give the same answer. :func:`parse_when` reads a message
that is only a time ("Friday at 9"), the answer to "When should I remind you?".

What it reads: ``in 2 hours``, ``in 3 days``, ``today``, ``tonight``,
``tomorrow``, a weekday (``Friday``: the next one, never today), a date
(``Nov 2``, ``2026-11-02``), a day part (``morning`` 8, ``noon`` 12,
``afternoon`` 14, ``evening`` 18, ``night`` 21) and a clock time (``9``,
``9:30``, ``9 pm``, ``14:30``, ``noon``, ``midnight``); in Chinese ``两小时后``,
``明天``, ``下周一``, ``10月30日``, ``早上``, ``九点半``. A day with no clock time
means 9:00. A bare hour from 1 to 6 with no ``am``/``pm`` and no day part is the
afternoon (``at 5`` is 17:00). A time with no day is today when it is still
ahead, else tomorrow; a time that says today (``today``, ``tonight``, ``this
morning``) and has passed is a ``past`` problem, never tomorrow.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Literal

from .chore_schedule import (
    _PART_HOUR,
    _ZH_PARTS,
    DEFAULT_MINUTE,
    MAX_AHEAD,
    Schedule,
    _hour_of,
    _zh_number,
)

Problem = Literal["", "time", "past", "far"]

_I = re.IGNORECASE
_N = r"(?:\d{1,3}|[零〇一二两三四五六七八九十]{1,3})"
_STRONG = re.compile(
    r"\bremind\s+me\b(?!\s+(?:what|how|why|who|when|where|which|whether|if)\b)"
    r"|\bset\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?reminder\b|\breminder\s*(?:to|that|for|:)"
    r"|提醒(?:一下)?我|设(?:置|定)?(?:个|一个)?提醒",
    _I,
)
_WEAK = re.compile(
    r"\bdon['’]?t\s+(?:let\s+me\s+)?forget\b|记得|别忘了|不要忘了|别忘记|不要忘记", _I
)
_QUESTION = re.compile(r"[?？]\s*$|吗\s*[?？]?\s*$")
# What may come before a weak cue (once the time words are taken out): "请记得", "你别忘了".
_BEFORE_WEAK = re.compile(r"(?:请|麻烦你?|一定|务必|千万|你|please|pls)*", _I)

_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}  # fmt: skip
_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
_UNITS_ZH = {"分钟": "m", "小时": "h", "钟头": "h", "天": "d", "周": "w", "星期": "w", "礼拜": "w"}  # fmt: skip
_DAYS_EN = {"yesterday": -1, "today": 0, "tonight": 0, "tomorrow": 1, "tmrw": 1, "tmr": 1}  # fmt: skip
_DAYS_ZH = {"昨天": (-1, ""), "今天": (0, ""), "今晚": (0, "night"), "今早": (0, "morning"),
            "明天": (1, ""), "明早": (1, "morning"), "明晚": (1, "night"),
            "后天": (2, ""), "大后天": (3, "")}  # fmt: skip
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEEKDAYS_ZH = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")  # fmt: skip


def _month(group: str) -> str:
    return (
        f"(?P<{group}>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
        "|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    )


_ON = r"(?:(?:on|for|by)\s+)?"
_REL_EN = re.compile(
    r"\bin\s+(?P<n>\d+(?:\.\d+)?|half\s+an?|an?|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s*"
    r"(?P<u>minutes?|mins?|hours?|hrs?|h|days?|weeks?)\b",
    _I,
)
_REL_ZH = re.compile(
    rf"(?:(?P<n>{_N})\s*个?(?P<half>半)?|(?P<h0>半)个?)\s*(?P<u>小时|钟头|分钟|天|周|星期|礼拜)(?:之|以)?后"
)
_ISO = re.compile(rf"\b{_ON}(\d{{4}})-(\d{{2}})-(\d{{2}})\b", _I)
_DATE_EN = re.compile(
    rf"\b{_ON}{_month('mon')}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*[ap]\.?m\b|[:\d])"
    rf"|\b{_ON}(?:the\s+)?(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_month('mon2')}\b",
    _I,
)
_DATE_ZH = re.compile(r"(?P<M>\d{1,2})月(?P<d>\d{1,2})[日号]")
_DAY_EN = re.compile(
    rf"\b{_ON}(?P<w>yesterday|today|tonight|tomorrow|tmrw|tmr)\b|\bthis\s+(?P<p>morning|afternoon|evening)\b",
    _I,
)
_AFTER_EN = re.compile(rf"\b{_ON}(?:the\s+)?day\s+after\s+tomorrow\b", _I)
_WEEKDAY_EN = re.compile(
    r"\b(?:(?:on|for|by|this|next|coming)\s+)?(?P<w>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(?:on|for|by|this|next)\s+(?P<a>mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)\b\.?",
    _I,
)
_DAY_ZH = re.compile("|".join(sorted(_DAYS_ZH, key=len, reverse=True)))
_WEEKDAY_ZH = re.compile(
    r"(?P<pre>下个?|这个?|本)?(?:周|星期|礼拜)(?P<w>[一二三四五六日天])"
)
_PART_EN = re.compile(
    r"\b(?:(?:in|at)\s+the\s+|at\s+)?(?P<p>morning|afternoon|evening|night)\b", _I
)
_PART_ZH = re.compile(
    "的?(?P<p>"
    + "|".join(sorted((k for k in _ZH_PARTS if len(k) > 1), key=len, reverse=True))
    + ")"
)
_PRE = r"(?:(?<![A-Za-z])(?:at|around|by|for)\s+|@\s*)"
_CLOCK_EN = re.compile(
    rf"{_PRE}?(?<![\d:.])(?:(?P<h1>\d{{1,2}})(?::(?P<m1>\d{{2}}))?\s*(?P<ap>[ap]\.?m\.?)(?![A-Za-z])"
    r"|(?P<h2>\d{1,2}):(?P<m2>\d{2})(?![\d:])|(?P<w>noon|midday|midnight)\b)"
    rf"|{_PRE}(?P<h3>\d{{1,2}})(?:\s*o['’]?clock)?(?!\d|[:%]|\.\d|(?:st|nd|rd|th)\b)",
    _I,
)
_CLOCK_ZH = re.compile(
    rf"(?P<h>{_N})\s*[点時时]钟?\s*(?:(?P<half>半)|(?P<m>{_N})\s*分?|整)?"
)
_LEAD = re.compile(
    r"^(?:[\s,;:.，；：。、!！?？-]|please\b|to\b|that\b|about\b|and\b|then\b|的)+", _I
)
_TRAIL = re.compile(r"[\s,;:.，；：。、!！?？-]+$")


@dataclass(frozen=True)
class Reminder:
    """What a message asks to be reminded of and when: ``on`` and ``minute`` are local wall-clock."""

    work: str
    on: date | None = None
    minute: int = 0
    # "" when the time is usable; "time": none was named or it cannot be read;
    # "past": it has passed; "far": it is more than a year ahead.
    problem: Problem = ""

    @property
    def schedule(self) -> Schedule | None:
        """The one-time schedule, when the time is usable."""
        if self.on is None or self.problem:
            return None
        return Schedule("once", minute=self.minute, on=self.on)


@dataclass
class _When:
    """The time words read so far."""

    seen: list[str] = field(default_factory=list)
    delta: timedelta | None = None
    day: date | None = None
    # The day came from a count (`in 3 days`): its clock time is now's, unless one is named.
    counted: bool = False
    # The words say today (`today`, `tonight`, `this morning`): past is past, not tomorrow.
    today: bool = False
    part: str = ""
    minute: int | None = None
    bad: bool = False

    @property
    def named(self) -> bool:
        return bool(self.seen)


def _amount(raw: str) -> float | None:
    raw = raw.lower()
    if raw.startswith("half"):
        return 0.5
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return float(raw)
    return _WORDS.get(raw)


def _read(text: str, now: datetime) -> tuple[_When, str]:
    """The time words in *text* and the text without them."""
    when, today = _When(), now.date()

    def take(pattern: re.Pattern[str], fold: Callable[[re.Match[str]], None]) -> None:
        nonlocal text
        found = pattern.search(text)
        if found is not None:
            text = f"{text[: found.start()]} {text[found.end() :]}"
            when.seen.append(found.group())
            fold(found)

    def set_day(day: date, *, now_only: bool = False) -> None:
        when.day, when.today = day, when.today or now_only

    def relative(found: re.Match[str]) -> None:
        names = found.groupdict()
        if names.get("h0"):
            number: float | None = 0.5
            unit = "h"
        elif names["u"] in _UNITS_ZH:
            number = _zh_number(names["n"])
            number = None if number is None else number + (0.5 if names["half"] else 0)
            unit = _UNITS_ZH[names["u"]]
        else:
            number, unit = _amount(names["n"]), names["u"].lower()[0]
        if not number:
            when.bad = True
            return
        span = timedelta(seconds=number * _SECONDS[unit])
        if unit in "mh":
            when.delta = span
        else:
            when.day, when.counted = (now + span).date(), True

    def english_day(found: re.Match[str]) -> None:
        word = (found["w"] or "").lower()
        if not word:  # this morning, this afternoon, this evening
            set_day(today, now_only=True)
            when.part = found["p"].lower()
            return
        set_day(
            today + timedelta(days=_DAYS_EN[word]),
            now_only=word in ("today", "tonight"),
        )
        when.part = "night" if word == "tonight" else when.part

    def weekday(found: re.Match[str]) -> None:
        name = (found["w"] or found["a"]).lower()[:3]
        set_day(
            today + timedelta(days=(_WEEKDAYS.index(name) - today.weekday()) % 7 or 7)
        )

    def chinese_day(found: re.Match[str]) -> None:
        offset, part = _DAYS_ZH[found.group()]
        set_day(today + timedelta(days=offset), now_only=offset == 0)
        when.part = part or when.part

    def chinese_weekday(found: re.Match[str]) -> None:
        index = _WEEKDAYS_ZH[found["w"]]
        if (found["pre"] or "").startswith(
            "下"
        ):  # the Monday-to-Sunday week after this one
            set_day(today + timedelta(days=7 - today.weekday() + index))
        else:
            set_day(today + timedelta(days=(index - today.weekday()) % 7 or 7))

    def month_day(month: int, day: int) -> None:
        try:
            found = date(today.year, month, day)
            set_day(found if found >= today else date(today.year + 1, month, day))
        except ValueError:
            when.bad = True

    def english_date(found: re.Match[str]) -> None:
        month = (found["mon"] or found["mon2"]).lower()[:3]
        month_day(_MONTHS.index(month) + 1, int(found["d"] or found["d2"]))

    def iso(found: re.Match[str]) -> None:
        try:
            set_day(date(*(int(part) for part in found.groups())))
        except ValueError:
            when.bad = True

    def part(found: re.Match[str]) -> None:
        raw = found["p"]
        when.part = _ZH_PARTS.get(raw) or raw.lower()

    def clock(found: re.Match[str]) -> None:
        names = found.groupdict()
        if names.get("w"):
            when.minute = 0 if names["w"].lower() == "midnight" else 12 * 60
            return
        if "h" in names:
            hour: int | None = _zh_number(found["h"])
            minute = (
                30 if found["half"] else _zh_number(found["m"]) if found["m"] else 0
            )
            ap = ""
        else:
            which = next(n for n in "123" if names[f"h{n}"])
            hour = int(names[f"h{which}"])
            minute = int(names.get(f"m{which}") or 0)
            ap = (names.get("ap") or "").replace(".", "")
        if hour is not None and not ap and not when.part and 1 <= hour <= 6:
            hour += 12  # a bare 1 to 6 is the afternoon
        hour = None if hour is None else _hour_of(hour, ap, when.part)
        if hour is None or minute is None or minute > 59:
            when.bad = True
            return
        when.minute = hour * 60 + minute

    take(_REL_EN, relative)
    take(_REL_ZH, relative)
    take(_ISO, iso)
    take(_DATE_EN, english_date)
    take(_DATE_ZH, lambda found: month_day(int(found["M"]), int(found["d"])))
    take(_AFTER_EN, lambda found: set_day(today + timedelta(days=2)))
    take(_DAY_EN, english_day)
    take(_DAY_ZH, chinese_day)
    take(_WEEKDAY_EN, weekday)
    take(_WEEKDAY_ZH, chinese_weekday)
    take(_PART_EN, part)
    take(_PART_ZH, part)
    take(_CLOCK_EN, clock)
    take(_CLOCK_ZH, clock)
    return when, text


def _settle(when: _When, now: datetime) -> tuple[date | None, int, Problem]:
    """The date and minute *when* names against *now*, or the problem."""
    if not when.named or when.bad:
        return None, 0, "time"
    stamp = now.replace(tzinfo=None)
    if when.delta is not None:
        soon = (stamp + when.delta).replace(second=0, microsecond=0)
        return soon.date(), soon.hour * 60 + soon.minute, ""
    if when.minute is not None:
        minute = when.minute
    elif when.part:
        minute = _PART_HOUR[when.part] * 60
    elif when.counted:
        minute = stamp.hour * 60 + stamp.minute
    else:
        minute = DEFAULT_MINUTE
    at = datetime.combine(when.day or stamp.date(), time()) + timedelta(minutes=minute)
    if at <= stamp:
        if when.day is not None or when.today:
            return None, 0, "past"
        at += timedelta(days=1)  # a bare time that has passed means tomorrow
    if at - stamp > MAX_AHEAD:
        return None, 0, "far"
    return at.date(), at.hour * 60 + at.minute, ""


def _tidy(text: str) -> str:
    return _TRAIL.sub("", _LEAD.sub("", " ".join(text.split()))).strip()


def parse_reminder(text: str, now: datetime) -> Reminder | None:
    """The reminder a message asks for, or ``None`` when it asks for none.

    ``None`` too for a cue with no work left ("remind me tomorrow at 9") and for
    a weak cue (``记得``, ``don't forget``) with no time or in a question. A strong
    cue with no usable time gives a :class:`Reminder` whose ``problem`` says why.
    """
    text = unicodedata.normalize("NFKC", text).strip()
    cue = _STRONG.search(text)
    weak = cue is None
    cue = cue or _WEAK.search(text)
    if cue is None:
        return None
    when, rest = _read(f"{text[: cue.start()]} {text[cue.end() :]}", now)
    work = _tidy(rest)
    if len(work) < 2:
        return None
    if weak:
        # "记得明天九点…" is a reminder; "我记得明天有会", "do you remember…?" are not.
        before = _tidy(_read(text[: cue.start()], now)[1])
        if (
            not when.named
            or _QUESTION.search(text)
            or not _BEFORE_WEAK.fullmatch(before)
        ):
            return None
    on, minute, problem = _settle(when, now)
    return Reminder(work, on, minute, problem)


def parse_when(text: str, now: datetime) -> Reminder | None:
    """A message that is only a time ("Friday at 9", "in 2 hours"), else ``None``."""
    when, rest = _read(unicodedata.normalize("NFKC", text).strip(), now)
    if not when.named or _tidy(rest):
        return None
    on, minute, problem = _settle(when, now)
    return Reminder("", on, minute, problem)


__all__ = ["Reminder", "parse_reminder", "parse_when"]
