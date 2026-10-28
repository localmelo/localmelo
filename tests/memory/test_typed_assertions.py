"""Conservative typed facts keep identity, polarity and the original sentence."""

from __future__ import annotations

import pytest

from localmelo.melo.memory.assertions import parse_assertion


@pytest.mark.parametrize(
    "text",
    [
        "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11, doors at 20:00.",
        "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11: doors moved to 21:00 (the opener cancelled; Lena told Jamie).",
    ],
)
def test_concert_doors_have_one_typed_identity(text: str) -> None:
    assertion = parse_assertion(text)
    assert assertion is not None
    assert assertion.subject_key == "event:static bloom:the paradise lot:2026-04-11"
    assert assertion.predicate_key == "doors_time"
    assert assertion.original == text
    assert assertion.polarity == "positive" and assertion.state == "asserted"
    assert assertion.value_text == ("21:00" if "21:00" in text else "20:00")
    assert assertion.changed is ("moved" in text)


@pytest.mark.parametrize(
    ("text", "polarity", "changed"),
    [
        ("Mina lives in Paris.", "positive", False),
        ("Mina does not live in Paris.", "negative", False),
        ("Mina no longer lives in Paris.", "negative", True),
        ("Mina now lives in Rome.", "positive", True),
    ],
)
def test_residence_preserves_polarity(text: str, polarity: str, changed: bool) -> None:
    assertion = parse_assertion(text)
    assert assertion is not None
    assert (
        assertion.subject_key == "person:mina"
        and assertion.predicate_key == "residence"
    )
    assert assertion.polarity == polarity and assertion.changed is changed


@pytest.mark.parametrize(
    ("text", "value", "polarity", "changed"),
    [
        ("小王住在北京。", "北京", "positive", False),
        ("小王不再住在北京。", "北京", "negative", True),
    ],
)
def test_chinese_residence(text: str, value: str, polarity: str, changed: bool) -> None:
    assertion = parse_assertion(text)
    assert assertion is not None
    assert (assertion.subject_key, assertion.predicate_key, assertion.value_text) == (
        "person:小王",
        "residence",
        value,
    )
    assert (assertion.polarity, assertion.changed) == (polarity, changed)


def test_event_without_venue_is_not_linked() -> None:
    assertion = parse_assertion("Static Bloom concert 2026-04-11 doors at 20:00.")
    assert assertion is not None and assertion.subject_key is None


def test_event_cancellation_is_state_not_opener_denial() -> None:
    old = parse_assertion(
        "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11 doors at 20:00."
    )
    cancelled = parse_assertion(
        "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11 doors cancelled."
    )
    assert old and cancelled
    assert cancelled.subject_key == old.subject_key
    assert cancelled.predicate_key == old.predicate_key
    assert cancelled.state == "cancelled" and cancelled.changed
    assert cancelled.polarity == "positive"


@pytest.mark.parametrize(
    "marker", ["改到", "改成", "改为", "推迟到", "延后到", "提前到"]
)
def test_chinese_named_meeting_changes(marker: str) -> None:
    original = parse_assertion("团队会议2026-04-11定于20:00。")
    updated = parse_assertion(f"团队会议2026-04-11{marker}21:00。")
    assert original and updated
    assert original.subject_key == updated.subject_key
    assert original.predicate_key == updated.predicate_key == "meeting_time"
    assert updated.changed and updated.value_text == "21:00"


def test_unlinked_relative_doors_keep_original_without_guessing_date() -> None:
    from datetime import date

    text = "Doors pushed to 9 tonight, the opener cancelled!"
    unknown = parse_assertion(text)
    assert unknown is not None and unknown.subject_key is None
    assert unknown.value_text == "21:00" and unknown.effective_date is None
    dated = parse_assertion(text, source_date=date(2026, 4, 11))
    assert dated is not None and dated.original == text
    assert dated.effective_date == "2026-04-11" and dated.changed
    assert dated.polarity == "positive" and dated.state == "asserted"


def test_source_record_verification_requires_complete_matching_evidence() -> None:
    from localmelo.melo.memory.assertions import verified_record

    text = "Doors pushed to 9 tonight, the opener cancelled!"
    evidence = '{"id":"F3","text":"Doors pushed to 9 tonight, the opener cancelled!","timestamp":"2026-04-11T16:50:00-04:00"}'
    verified, day = verified_record(text, ("F3",), evidence)
    assert verified and str(day) == "2026-04-11"
    assert verified_record(text, ("F4",), evidence) == (False, None)
    assert verified_record(text, ("F3",), evidence[:70]) == (False, None)
    quoted = '{"id":"F3","text":"Someone said: Doors pushed to 9 tonight, the opener cancelled!"}'
    assert verified_record(text, ("F3",), quoted) == (False, None)


def test_typed_relations_require_ordered_explicit_change() -> None:
    from localmelo.melo.memory.assertions import decide_assertion

    old = parse_assertion("Mina lives in Paris.")
    changed = parse_assertion("Mina now lives in Rome.")
    conflict = parse_assertion("Mina lives in Rome.")
    assert old and changed and conflict
    rows = [(1, old.to_dict(), 10.0)]
    assert decide_assertion(changed.to_dict(), rows, 20.0)[:2] == ("UPDATE", 1)
    assert decide_assertion(conflict.to_dict(), rows, 20.0)[:2] == ("CONFLICT", 1)
    assert decide_assertion(changed.to_dict(), rows, 9.0)[:2] == ("CONFLICT", 1)
    assert decide_assertion(changed.to_dict(), rows, 10.0)[:2] == ("CONFLICT", 1)
    assert decide_assertion(old.to_dict(), rows, 20.0)[:2] == ("IGNORE", 1)


@pytest.mark.parametrize(
    "text",
    [
        "If Mina now lives in Rome, call me.",
        "Does Mina live in Rome?",
        "He said 'Mina now lives in Rome'.",
        "Mina does not not live in Rome.",
        "Static Bloom show doors moved to 21:00.",
        "Static Bloom concert 2026-04-11 and 2026-04-12 doors moved to 21:00.",
        "Static Bloom concert 2026-04-11 doors moved from 20:00 to 21:00.",
    ],
)
def test_uncertain_identity_scope_or_value_abstains(text: str) -> None:
    assert parse_assertion(text) is None
