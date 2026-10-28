"""Narrow, versioned assertions; unrecognized or ambiguous text stays unparsed."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from typing import Any

_VERSION = 1
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME = re.compile(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b")
_EVENT = re.compile(
    r"^(?P<subject>[A-Za-z][A-Za-z0-9 &'’.-]{1,80}?)\s+(?:concert|show)\b",
    re.IGNORECASE,
)
_VENUE = re.compile(
    r"\bat\s+(?P<venue>[A-Za-z0-9 &'’.-]{2,80}?)\s+on\s+(?:(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+)?\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
_DOORS = re.compile(
    r"\bdoors\s+(?P<verb>at|opens?\s+at|moved\s+to|changed\s+to|pushed\s+(?:back\s+)?to|postponed\s+to|delayed\s+to|rescheduled\s+to|now(?:\s+at)?|instead\s+at)\s+(?P<value>(?:[01]\d|2[0-3]):[0-5]\d)\b",
    re.IGNORECASE,
)
_UNLINKED_DOORS = re.compile(
    r"^doors\s+(?:pushed\s+(?:back\s+)?to|postponed\s+to|delayed\s+to|rescheduled\s+to|moved\s+to)\s+(?P<hour>\d{1,2})\s+tonight(?:[,;][^?？]*)?[.!！]?$",
    re.IGNORECASE,
)
_RESIDENCE = re.compile(
    r"^(?P<name>[A-Z][a-z]{1,30})\s+(?P<verb>lives in|does not live in|no longer lives in|now lives in|moved to|changed residence to)\s+(?P<value>[A-Z][A-Za-z -]{1,60}?)(?:\s+(?P<change>now|instead))?[.!]?$",
    re.IGNORECASE,
)
_CJK_RESIDENCE = re.compile(
    r"^(?P<name>[一-鿿]{2,4}?)(?P<negative>不再)?住在(?P<value>[一-鿿]{2,12})[。.!！]?$"
)
_CJK_MEETING = re.compile(
    r"^(?P<name>[一-鿿]{2,12}?)会议(?P<date>\d{4}-\d{2}-\d{2})(?P<verb>定于|改到|改成|改为|推迟到|延后到|提前到)(?P<value>(?:[01]\d|2[0-3]):[0-5]\d)[。.!！]?$"
)
_CJK_MEETING_CANCELLED = re.compile(
    r"^(?P<name>[一-鿿]{2,12}?)会议(?P<date>\d{4}-\d{2}-\d{2})(?:已|被)?取消[。.!！]?$"
)
_CANCELLED_DOORS = re.compile(
    r"\bdoors\s+(?:is\s+|are\s+|was\s+|were\s+)?cancel(?:l)?ed\b", re.IGNORECASE
)
_UNCERTAIN = re.compile(
    r"\b(?:if|unless|whether|maybe|might|could|not\s+not|not\s+only)\b|如果|假如|是否|不但|不仅|可能|也许|[?？\"“”]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Assertion:
    """A parsed assertion, never evidence by itself."""

    subject_key: str | None
    predicate_key: str
    value_text: str
    polarity: str
    state: str
    changed: bool
    original: str
    version: int = _VERSION
    effective_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verified_record(
    text: str, source_ids: tuple[str, ...], evidence: str
) -> tuple[bool, date | None]:
    """Require a complete source object, matching id and an exact textual field."""
    from localmelo.melo.checker.redaction import redact_secrets

    if len(source_ids) != 1 or len(evidence) > 500:
        return False, None
    try:
        record = json.loads(evidence)
    except (ValueError, TypeError):
        return False, None
    if (
        not isinstance(record, dict)
        or str(record.get("record_id", record.get("id", ""))) != source_ids[0]
    ):
        return False, None
    if not any(
        isinstance(record.get(key), str)
        and redact_secrets(record[key])[0].strip() == text.strip()
        for key in ("text", "body", "content")
    ):
        return False, None
    parsed = _source_time(record)
    return True, parsed.date() if parsed is not None else None


def _source_time(record: dict[str, Any]) -> datetime | None:
    """The record's timezone-aware ``timestamp`` or ``date``, if it has one."""
    stamp = record.get("timestamp")
    if not isinstance(stamp, str):
        stamp = record.get("date")
    if not isinstance(stamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.utcoffset() is not None else None


def record_observed_at(evidence: str) -> float | None:
    """A timezone-aware source timestamp, not the later ingestion time."""
    try:
        record = json.loads(evidence)
        parsed = _source_time(record) if isinstance(record, dict) else None
        return parsed.timestamp() if parsed is not None else None
    except (ValueError, TypeError, AttributeError, OverflowError, OSError):
        return None


def decide_assertion(
    incoming: dict[str, Any],
    rows: list[tuple[int, dict[str, Any], float]],
    observed_at: float,
) -> tuple[str, int | None, str]:
    """Compare verified typed assertions; unlinked subjects never supersede."""
    key = (incoming.get("subject_key"), incoming.get("predicate_key"))
    if incoming.get("version") != _VERSION or not all(key):
        return "NEW", None, "unlinked assertion"
    matching = [
        (entry_id, other, created_at)
        for entry_id, other, created_at in rows
        if other.get("version") == _VERSION
        and (other.get("subject_key"), other.get("predicate_key")) == key
    ]
    if not matching:
        return "NEW", None, "new typed assertion"
    matching.sort(key=lambda item: (item[2], item[0]), reverse=True)
    entry_id, other, _created_at = matching[0]
    same = all(
        str(incoming.get(name, "")).casefold() == str(other.get(name, "")).casefold()
        for name in ("value_text", "polarity", "state")
    )
    if same:
        return "IGNORE", entry_id, "duplicate typed assertion"
    incoming_order = incoming.get("observed_at")
    previous_order = other.get("observed_at")
    if (
        incoming.get("changed") is True
        and len(matching) == 1
        and isinstance(incoming_order, (int, float))
        and not isinstance(incoming_order, bool)
        and isinstance(previous_order, (int, float))
        and not isinstance(previous_order, bool)
        and incoming_order > previous_order > 0
    ):
        return "UPDATE", entry_id, "ordered explicit change"
    return "CONFLICT", entry_id, "different or unordered typed assertion"


def parse_assertion(text: str, *, source_date: date | None = None) -> Assertion | None:
    """Parse explicit, bounded shapes; callers must separately verify evidence."""
    if not text or len(text) > 1_000 or _UNCERTAIN.search(text):
        return None
    match = _RESIDENCE.fullmatch(text.strip())
    if match:
        verb = match.group("verb").lower()
        return Assertion(
            "person:" + match.group("name").casefold(),
            "residence",
            match.group("value").strip(),
            (
                "negative"
                if verb in ("does not live in", "no longer lives in")
                else "positive"
            ),
            "asserted",
            verb
            in (
                "no longer lives in",
                "now lives in",
                "moved to",
                "changed residence to",
            )
            or (match.group("change") or "").lower() in ("now", "instead"),
            text,
        )
    match = _CJK_RESIDENCE.fullmatch(text.strip())
    if match:
        return Assertion(
            "person:" + match.group("name"),
            "residence",
            match.group("value"),
            "negative" if match.group("negative") else "positive",
            "asserted",
            bool(match.group("negative")),
            text,
        )
    meeting = _CJK_MEETING.fullmatch(text.strip())
    if meeting:
        try:
            date.fromisoformat(meeting.group("date"))
        except ValueError:
            return None
        day = meeting.group("date")
        return Assertion(
            f"meeting:{meeting.group('name')}:{day}",
            "meeting_time",
            meeting.group("value"),
            "positive",
            "asserted",
            meeting.group("verb") != "定于",
            text,
            effective_date=day,
        )
    cancelled_meeting = _CJK_MEETING_CANCELLED.fullmatch(text.strip())
    if cancelled_meeting:
        day = cancelled_meeting.group("date")
        try:
            date.fromisoformat(day)
        except ValueError:
            return None
        return Assertion(
            f"meeting:{cancelled_meeting.group('name')}:{day}",
            "meeting_time",
            "",
            "positive",
            "cancelled",
            True,
            text,
            effective_date=day,
        )
    relative = _UNLINKED_DOORS.fullmatch(text.strip())
    if relative:
        hour = int(relative.group("hour"))
        if not 1 <= hour <= 11:
            return None
        return Assertion(
            None,
            "doors_time",
            f"{hour + 12:02}:00",
            "positive",
            "asserted",
            True,
            text,
            effective_date=source_date.isoformat() if source_date else None,
        )
    event = _EVENT.match(text)
    door = _DOORS.search(text)
    cancelled = _CANCELLED_DOORS.search(text)
    dates = _DATE.findall(text)
    times = _TIME.findall(text)
    venue = _VENUE.search(text)
    subject = None
    if event and venue and venue.start() > event.end() and len(dates) == 1:
        name = " ".join(event.group("subject").casefold().split())
        place = " ".join(venue.group("venue").casefold().split())
        subject = f"event:{name}:{place}:{dates[0]}"
    if (
        event
        and cancelled
        and len(dates) == 1
        and not times
        and cancelled.start() > event.end()
    ):
        try:
            date.fromisoformat(dates[0])
        except ValueError:
            return None
        return Assertion(
            subject,
            "doors_time",
            "",
            "positive",
            "cancelled",
            True,
            text,
            effective_date=dates[0],
        )
    if not event or not door or len(dates) != 1 or len(times) != 1:
        return None
    try:
        date.fromisoformat(dates[0])
        time.fromisoformat(door.group("value"))
    except ValueError:
        return None
    if "'" in text or "’" in text or door.start() < event.end():
        return None
    verb = door.group("verb").lower()
    return Assertion(
        subject,
        "doors_time",
        door.group("value"),
        "positive",
        "asserted",
        verb not in ("at", "open at", "opens at"),
        text,
        effective_date=dates[0],
    )
