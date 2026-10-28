"""Regression cases for heuristic fact extraction and attribution."""

from __future__ import annotations

import time

import pytest

from localmelo.melo.memory.notes import (
    NOTE_MAX_FACTS,
    _date_keys,
    _item_source_ids,
    _strong_keys,
    is_meta_claim,
    note_from_reply,
    parse_note_block,
)


@pytest.mark.parametrize(
    "text",
    [
        "I could check the offsite on 4/16.",
        "Is yoga at 9:00?",
        "Not on 4/9.",
        "The plumber is not coming on 4/9.",
        "Option A: meet on 4/9 at 9:00.",
        "Should we book the room for 4/16 at 9:00?",
        "Maybe the meeting is at 9:00.",
        "The meeting was cancelled on 4/9.",
        "We can meet at 9:00.",
    ],
)
def test_nonfacts_are_not_found(text: str) -> None:
    note, _ = note_from_reply(text, writer="heuristic")
    assert not note.found
    assert not note.decided
    assert not note.remember
    if text.endswith("?"):
        assert note.open_questions


def test_mixed_bullet_preserves_fact_and_question() -> None:
    note, _ = note_from_reply("- Yoga starts at 9:00. Is it at Studio Verde?")
    assert note.found == ["Yoga starts at 9:00."]
    assert note.open_questions == ["Is it at Studio Verde?"]


@pytest.mark.parametrize(
    "text",
    [
        "3/4 cup",
        "1/2 price",
        "24/7",
        "We will march 5 miles",
        "may 5 people",
        "April 31",
    ],
)
def test_not_dates(text: str) -> None:
    assert not _date_keys(text)


def test_date_context_and_order() -> None:
    assert "date:04-09" in _date_keys("on 4/9")
    assert "date:04-13" in _date_keys("13/4")
    assert "date:09-04" in _date_keys("on 9/4")
    assert "date:04-09" in _date_keys("9 April")
    assert not _date_keys("3/4 cup")


def test_numbers_are_not_accidental_keys() -> None:
    for text in ("3rd", "Parent: 3", "current: 5", "3-4", "4B", "v2.1"):
        assert not _strong_keys(text), text
    assert not any(key.startswith("number:") for key in _strong_keys("555-123-4567"))


def test_conflicting_source_keys_and_ties() -> None:
    assert not _item_source_ids("Call 555-123-4567", [("wrong", "Call 444-123-9876")])
    assert not _item_source_ids("payment $100", [("wrong", "payment EUR 100")])
    assert _item_source_ids(
        "meeting on 2027-04-09",
        [("old", "meeting on 2026-04-09"), ("right", "meeting on 2027-04-09")],
    ) == ["right"]
    assert _item_source_ids(
        "offsite on 4/9 9:30",
        [("wrong", "offsite 2026-04-10 9:30"), ("right", "offsite 2026-04-09")],
    ) == ["right"]
    assert not _item_source_ids(
        "yoga at 9:00", [("one", "yoga at 9:00"), ("two", "yoga at 9:00")]
    )


def test_phone_and_unit_keys_are_not_partial_matches() -> None:
    assert _item_source_ids("Call 555-123-4567", [("right", "Call 555-123-4567")]) == [
        "right"
    ]
    assert not _item_source_ids("Call 555-123-4567", [("addr", "123 Main Street")])
    assert not _item_source_ids("payment $100", [("bare", "payment 100")])
    assert _item_source_ids("payment $100", [("usd", "payment USD 100")]) == ["usd"]
    assert not _item_source_ids("Meeting on the 3rd", [("floor", "3rd floor office")])
    assert not _strong_keys("1" * 50_000)


def test_meta_frame_and_content() -> None:
    for text in (
        "I've read all records at 9:00.",
        "I've read and remembered all of Jamie's records for the reading group on 4/9.",
    ):
        assert is_meta_claim(text), text
    for text in (
        "I've noted that yoga starts at 9:00.",
        "I've noted the record says yoga is on April 9.",
        "I've read the record: yoga on 4/11.",
    ):
        assert not is_meta_claim(text), text


def test_fact_caps_are_twelve() -> None:
    assert NOTE_MAX_FACTS == 12
    note = parse_note_block("found: " + "; ".join(f"item {i}" for i in range(13)))
    assert len(note.found) == 12


@pytest.mark.parametrize("text", ["1" * 50_000, "a." * 25_000, "Alice Bob " * 5_000])
def test_long_reply_is_bounded(text: str) -> None:
    start = time.monotonic()
    note_from_reply(text, writer="heuristic")
    is_meta_claim(text)
    assert time.monotonic() - start < 1.0
