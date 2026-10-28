"""Hippo's write path: staging, signals, the checked write and the log."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from localmelo.melo.memory import coordinator
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.notes import ReasoningNote, candidate_items
from localmelo.melo.schema import CandidateItem, MemoryCandidate


def _candidate(
    text: str,
    *,
    candidate_id: str = "n1",
    source: str = "reasoning_note",
    task_id: str = "t1",
    remember: bool = True,
    items: tuple[CandidateItem, ...] = (),
    provenance: dict[str, Any] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        source=source,
        text=text,
        remember_by="model",
        task_id=task_id,
        provenance={"task_text": "plan the migration", **(provenance or {})},
        remember=remember,
        items=items,
    )


async def test_without_a_signal_a_candidate_is_never_written() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(_candidate("I live in Berlin and work at Acme"))
    assert await store.count_entries() == 0
    assert await store.decisions() == []


async def test_a_rehearsed_candidate_is_written_once() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    candidate = _candidate("I live in Berlin and work at Acme")
    hippo.propose(candidate)
    first = await hippo.admit("n1", "rehearsal")
    assert [decision.action for decision in first] == ["NEW"]
    assert first[0].entry_id is not None
    assert await store.count_entries() == 1
    second = await hippo.admit("n1", "rehearsal")
    assert second == []
    assert await store.count_entries() == 1
    assert len(await store.decisions()) == 1


async def test_a_restatement_re_proposes_and_updates() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(_candidate("I live in Berlin near the river and work at Acme"))
    await hippo.admit("n1", "rehearsal")
    hippo.propose(_candidate("I live in Berlin near the park and work at Acme"))
    decisions = await hippo.admit("n1", "rehearsal")
    assert [decision.action for decision in decisions] == ["UPDATE"]
    active = await store.active_entries()
    assert [entry.text for entry in active] == [
        "I live in Berlin near the park and work at Acme"
    ]
    assert active[0].supersedes == 1


async def test_an_unknown_signal_is_a_value_error() -> None:
    hippo = Hippo(long=LongTerm())
    hippo.propose(_candidate("I live in Berlin and work at Acme"))
    with pytest.raises(ValueError):
        await hippo.admit("n1", "because")


async def test_a_run_request_admits_remembered_notes_and_the_digest_only() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    note = _candidate("I live in Berlin and work at Acme")
    refused_note = _candidate(
        "the migration is done and the test passed", candidate_id="n2", remember=False
    )
    digest = _candidate(
        "task: migrate\nfacts: the migration is done",
        candidate_id="d1",
        source="digest",
    )
    reflection = _candidate(
        "Attempt 1: tried the migration", candidate_id="r1", source="reflection"
    )
    for candidate in (note, refused_note, digest, reflection):
        hippo.propose(candidate)
    decisions = await hippo.flush(
        run_request=True, digest=digest, final_reflection=reflection
    )
    assert len(decisions) == 2
    assert {decision.action for decision in decisions} <= {
        "NEW",
        "UPDATE",
        "RELATED",
        "CONFLICT",
    }
    assert {entry.text for entry in await store.active_entries()} == {
        "I live in Berlin and work at Acme",
        "task: migrate\nfacts: the migration is done",
    }
    assert {decision.admitted_by for decision in decisions} == {"user_request"}


async def test_the_outcome_switch_off_writes_nothing_and_on_writes_only_the_digest() -> (
    None
):
    store = LongTerm()
    digest = _candidate(
        "task: migrate\nfacts: the migration is done",
        candidate_id="d1",
        source="digest",
    )
    digest.provenance["final_status"] = "completed"
    reflection = _candidate(
        "Attempt 1: tried the migration", candidate_id="r1", source="reflection"
    )
    off = Hippo(long=store)
    off.propose(digest)
    off.propose(reflection)
    assert (
        await off.flush(run_request=False, digest=digest, final_reflection=reflection)
        == []
    )
    assert await store.count_entries() == 0

    on = Hippo(long=LongTerm(), admit_on_outcome=True)
    on.propose(digest)
    on.propose(reflection)
    decisions = await on.flush(
        run_request=False, digest=digest, final_reflection=reflection
    )
    assert [decision.admitted_by for decision in decisions] == ["outcome"]
    entries = await on.long.active_entries()
    assert [entry.text for entry in entries] == [
        "task: migrate\nfacts: the migration is done"
    ]
    assert entries[0].provenance["final_status"] == "completed"


async def test_the_checked_write_refuses_a_profile_entry_from_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(_candidate("my birthday is 12 May 1990 and I like tea"))
    monkeypatch.setattr(coordinator, "classify_kind", lambda *a, **k: "profile")
    decisions = await hippo.admit("n1", "rehearsal")
    assert [decision.refused for decision in decisions] == [True]
    assert "must come from the user" in decisions[0].reason
    assert await store.count_entries() == 0
    assert await store.decisions() == []


async def test_a_secret_is_stored_redacted_and_counted() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    key = "sk-" + "a" * 24
    hippo.propose(_candidate(f"I use the api key {key} for the deployment"))
    decisions = await hippo.admit("n1", "rehearsal")
    assert decisions[0].redactions == 1
    entry = (await store.active_entries())[0]
    assert key not in entry.text
    assert "[redacted]" in entry.text


async def test_a_failed_write_is_logged_and_flush_does_not_raise(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    digest = _candidate(
        "task: migrate\nfacts: the migration is done",
        source="digest",
        candidate_id="d1",
    )
    hippo.propose(digest)

    async def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "apply", fail)
    with caplog.at_level(logging.WARNING, logger="localmelo.melo.memory.coordinator"):
        decisions = await hippo.flush(
            run_request=True, digest=digest, final_reflection=None
        )
    assert decisions == []
    assert "disk full" in caplog.text


def test_an_item_keeps_the_records_that_share_its_number_or_name() -> None:
    note = ReasoningNote(
        found=["the amount is 42 thousand", "the office is on Main Street"],
        source_ids=["r1", "r2"],
        source_records=[
            ("r1", '{"amount": 42}'),
            ("r2", '{"address": "Main Street"}'),
        ],
    )
    items = candidate_items(note)
    assert [item.source_ids for item in items] == [("r1",), ("r2",)]


def test_a_note_with_no_matching_record_keeps_all_its_ids() -> None:
    note = ReasoningNote(
        found=["the deployment happens on Friday"],
        source_ids=["r1", "r2"],
        source_records=[("r1", '{"name": "Alice"}'), ("r2", '{"name": "Bob"}')],
    )
    items = candidate_items(note)
    assert items[0].source_ids == ("r1", "r2")
