"""T68: a factual bullet reply and the meta claims that are not facts."""

from __future__ import annotations

import json
from pathlib import Path

from localmelo.melo.memory.hippo import decide
from localmelo.melo.memory.notes import (
    NOTE_MAX_FACTS,
    ReasoningNote,
    candidate_items,
    is_meta_claim,
    note_from_reply,
)
from localmelo.melo.schema import CandidateItem, ToolCall, ToolResult

FIXTURE = Path(__file__).parent / "fixtures" / "t68_bullet_reply.json"


def _bullet_reply() -> ReasoningNote:
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    call = ToolCall("mcp__pab__feed_day", {})
    result = ToolResult("mcp__pab__feed_day", output=case["tool_result"])
    note, _rest = note_from_reply(
        case["content"], "", [call], [result], kind="answer", writer="heuristic"
    )
    return note


def _source(items: list[CandidateItem], fragment: str) -> tuple[str, ...]:
    for item in items:
        if fragment in item.text:
            assert item.source_ids is not None
            return item.source_ids
    raise AssertionError(f"no item holds {fragment!r}")


def test_a_bullet_reply_stores_its_facts_and_their_records() -> None:
    note = _bullet_reply()
    assert len(note.found) == NOTE_MAX_FACTS
    items = list(candidate_items(note))
    assert _source(items, "Yoga") == ("ra4ec148cb4",)
    assert _source(items, "Q2 planning offsite") == ("r38ad45f23d",)
    assert _source(items, "Game night") == ("rd66a8ac4ba",)
    assert _source(items, "Library hold") == ("rb8e5446dd4",)
    assert _source(items, "Tom Whitfield") == ("r4af3c64af7",)
    assert _source(items, "Grace Liu") == ("r1e4be03db0",)
    assert _source(items, "Priya Raman") == ("rfea11d459a",)


def test_a_meta_claim_is_ignored_with_a_reason() -> None:
    claims = [
        "I've read and remembered all of Jamie's records.",
        "I've read and remembered all of Jamie's records from today "
        "(2026-04-06), including event details, people, places, dates, "
        "and financial amounts.",
        "The session is complete.",
        "I've reviewed all 8 of today's records (2026-04-07) and memorized "
        "the key facts about Jamie's plans, people, places, dates, and amounts.",
        "All records from 2026-04-12 have been reviewed and the key facts "
        "are memorized.",
    ]
    for text in claims:
        for source in ("reasoning_note", "digest"):
            decision = decide(
                text,
                kind="episodic",
                source=source,
                provenance={},
                active=[],
            )
            assert decision.action == "IGNORE", (source, text)
            assert "meta claim" in decision.reason


def test_a_meta_claim_that_carries_a_fact_is_not_dropped() -> None:
    text = (
        "all 10 records read and summarized — plumber 4/10 1-3PM Adler Plumbing "
        "$180 Ben Okoro 14 Orchard Ln Apt 3"
    )
    decision = decide(
        text, kind="episodic", source="reasoning_note", provenance={}, active=[]
    )
    assert decision.action == "NEW"


def test_a_slash_date_matches_an_iso_record_and_not_a_bare_number() -> None:
    item = "reading group 4/9 3-4 Room 4B"
    plumber = ("plumber", '{"body": "plumber Friday 2026-04-10 1-3PM Apt 3"}')
    reading = ("reading", '{"body": "reading group 2026-04-09 3-4 Room 4B"}')
    note = ReasoningNote(
        found=[item],
        source_ids=[record_id for record_id, _text in (plumber, reading)],
        source_records=[plumber, reading],
    )
    assert [candidate.source_ids for candidate in candidate_items(note)] == [
        ("reading",)
    ]
    only_plumber = ReasoningNote(
        found=[item],
        source_ids=["plumber"],
        source_records=[plumber],
    )
    assert [candidate.source_ids for candidate in candidate_items(only_plumber)] == [()]


def test_a_digest_with_a_task_line_is_content() -> None:
    # A digest's task line and its outcome are content even when the
    # sentence reads like a completion claim (no dates or amounts at all).
    text = "task: transpose() … outcome: completed"
    assert not is_meta_claim(text)
    decision = decide(text, kind="episodic", source="digest", provenance={}, active=[])
    assert decision.action == "NEW", decision.reason


def test_a_claim_that_names_a_file_or_function_is_content() -> None:
    for text in (
        "I've read transpose.py and remembered everything about Jamie's records.",
        "I've read transpose() and remembered everything about Jamie's records.",
        "I've read `transpose_matrix` and remembered all of Jamie's records.",
    ):
        assert not is_meta_claim(text)
        decision = decide(
            text, kind="episodic", source="reasoning_note", provenance={}, active=[]
        )
        assert decision.action == "NEW", (text, decision.reason)


def test_an_outcome_admitted_completed_digest_is_exempt_from_the_meta_filter() -> None:
    text = "I've read and remembered all of Jamie's records."
    decision = decide(
        text,
        kind="episodic",
        source="digest",
        provenance={"final_status": "completed"},
        active=[],
        admitted_by="outcome",
    )
    assert decision.action == "NEW", decision.reason
    not_completed = decide(
        text,
        kind="episodic",
        source="digest",
        provenance={"final_status": "budget_exhausted"},
        active=[],
        admitted_by="outcome",
    )
    assert not_completed.action == "IGNORE"
    assert "meta claim" in not_completed.reason


def test_an_aider_final_digest_is_admitted_as_a_user_request_digest() -> None:
    text = (
        "task: Implement transpose in transpose.py so the tests in "
        "transpose_test.py pass\noutcome: completed"
    )
    for admitted_by in ("", "user_request", "outcome"):
        decision = decide(
            text,
            kind="episodic",
            source="digest",
            provenance={"final_status": "completed"},
            active=[],
            admitted_by=admitted_by,
        )
        assert decision.action == "NEW", (admitted_by, decision.reason)


def test_a_fenced_code_block_is_not_a_fact() -> None:
    content = (
        "```python\nmeeting = '2026-04-09 9:00'\n```\n"
        "The meeting is at 9:00 on 2026-04-09."
    )
    note, _rest = note_from_reply(content)
    assert note.found == ["The meeting is at 9:00 on 2026-04-09."]
