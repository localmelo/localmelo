"""Hippo's write path: staging, signals, the checked write and the log."""

from __future__ import annotations

import json
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
    hippo.propose(
        _candidate(
            f"I use the api key {key} for the deployment",
            provenance={
                "task_text": f"deploy with {key}",
                "paths": [f"/tmp/{key}.pem"],
                "commands": [f"curl -H 'Authorization: Bearer {key}'"],
            },
        )
    )
    decisions = await hippo.admit("n1", "rehearsal")
    assert decisions[0].redactions == 1
    entry = (await store.active_entries())[0]
    assert key not in entry.text
    assert "[redacted]" in entry.text
    # T53R finding 2: the provenance is stored in the same row, so its
    # texts are redacted too (and bounded).
    assert key not in json.dumps(entry.provenance)
    assert entry.provenance["task_text"] == "deploy with [redacted]"
    assert entry.provenance["paths"] == ["/tmp/[redacted].pem"]
    assert entry.provenance["commands"][0].endswith("Bearer [redacted]'")


async def test_a_provenance_text_is_bounded_before_it_is_stored() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(
        _candidate(
            "the deployment checklist is owned by Alice",
            provenance={
                "task_text": "t" * 500,
                "paths": ["p" * 500],
                "commands": ["c" * 500],
            },
        )
    )
    await hippo.admit("n1", "rehearsal")
    entry = (await store.active_entries())[0]
    assert len(entry.provenance["task_text"]) == 300
    assert entry.provenance["paths"] == ["p" * 300]
    assert entry.provenance["commands"] == ["c" * 300]


async def test_two_ordinary_items_of_one_note_both_stay() -> None:
    # T53R finding 4: both are episodic and share the note, but they are
    # two traces; the second must not supersede the first.
    store = LongTerm()
    hippo = Hippo(long=store)
    candidate = _candidate(
        "two events",
        items=(
            CandidateItem("the build is green after the fix", ()),
            CandidateItem("the tests are slow on the new runner", ()),
        ),
    )
    hippo.propose(candidate)
    decisions = await hippo.admit("n1", "rehearsal")
    assert [decision.action for decision in decisions] == ["NEW", "NEW"]
    assert {entry.text for entry in await store.active_entries()} == {
        "the build is green after the fix",
        "the tests are slow on the new runner",
    }


async def test_a_restatement_of_the_same_item_updates_it() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    first = "the deployment is green after the fix"
    hippo.propose(
        _candidate(first, items=(CandidateItem(first, ()),)),
    )
    await hippo.admit("n1", "rehearsal")
    updated = "the deployment failed on the second runner"
    hippo.propose(
        _candidate(updated, items=(CandidateItem(updated, ()),)),
        # The same candidate id and item index: a restatement, not a new fact.
    )
    decisions = await hippo.admit("n1", "rehearsal")
    assert [decision.action for decision in decisions] == ["UPDATE"]
    assert [entry.text for entry in await store.active_entries()] == [updated]


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
            ("r1", '{"amount": "42 thousand"}'),
            ("r2", '{"address": "Main Street"}'),
        ],
    )
    items = candidate_items(note)
    assert [item.source_ids for item in items] == [("r1",), ("r2",)]


def test_a_note_with_no_matching_record_gets_no_item_source_ids() -> None:
    note = ReasoningNote(
        found=["the deployment happens on Friday"],
        source_ids=["r1", "r2"],
        source_records=[("r1", '{"name": "Alice"}'), ("r2", '{"name": "Bob"}')],
    )
    items = candidate_items(note)
    # A false source is worse than none: the records exist and neither
    # shares a strong key or two distinctive words with the item.
    assert items[0].source_ids == ()


def test_a_note_with_no_record_text_keeps_all_its_ids() -> None:
    note = ReasoningNote(
        found=["the deployment happens on Friday"],
        source_ids=["r1", "r2"],
    )
    items = candidate_items(note)
    assert items[0].source_ids == ("r1", "r2")


async def test_flush_does_not_admit_an_earlier_run_s_candidate() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    old = _candidate("I live in Berlin near the river and work at Acme", task_id="t1")
    hippo.propose(old)
    assert (
        await hippo.flush(
            run_request=True, digest=None, final_reflection=None, task_id="t2"
        )
        == []
    )
    assert await store.count_entries() == 0
    decisions = await hippo.flush(
        run_request=True, digest=None, final_reflection=None, task_id="t1"
    )
    assert [decision.action for decision in decisions] == ["NEW"]


@pytest.mark.parametrize(
    "text",
    [
        "I could check the offsite on 4/16.",
        "Is yoga at 9:00?",
        "Option A: meet on 4/9 at 9:00.",
        "Maybe the meeting is at 9:00.",
    ],
)
async def test_heuristic_nonfacts_are_not_admitted_with_user_signal(text: str) -> None:
    from localmelo.melo.memory.notes import note_from_reply

    note, _ = note_from_reply(text, writer="heuristic")
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(_candidate(text, remember=note.remember, items=candidate_items(note)))
    assert await hippo.flush(run_request=True, digest=None, final_reflection=None) == []
    assert await store.count_entries() == 0


@pytest.mark.parametrize(
    "positive,negative",
    [
        ("The meeting is at 20:00.", "The meeting is not at 20:00."),
        (
            "会议在20:00举行，请大家注意时间安排。",
            "会议不在20:00举行，请大家注意时间安排。",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
async def test_opposite_polarities_are_never_duplicates(
    positive: str, negative: str, reverse: bool
) -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    first, second = (negative, positive) if reverse else (positive, negative)
    hippo.propose(_candidate(first, candidate_id="a"))
    await hippo.admit("a", "rehearsal")
    hippo.propose(_candidate(second, candidate_id="b"))
    decisions = await hippo.admit("b", "rehearsal")
    assert decisions[0].action != "IGNORE"
    assert {entry.text for entry in await store.active_entries()} == {first, second}


async def test_explicit_negative_fact_is_admitted_without_losing_negation() -> None:
    from localmelo.melo.memory.notes import note_from_reply

    text = "The plumber is not coming on 4/9."
    note, _ = note_from_reply(text, writer="heuristic")
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(_candidate(text, remember=note.remember, items=candidate_items(note)))
    await hippo.flush(run_request=True, digest=None, final_reflection=None)
    assert [entry.text for entry in await store.active_entries()] == [text]


async def test_mixed_bullet_admits_the_fact_not_the_question() -> None:
    from localmelo.melo.memory.notes import note_from_reply

    note, _ = note_from_reply(
        "- Yoga starts at 9:00 every Saturday. Is it at Studio Verde?"
    )
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(
        _candidate("Yoga starts at 9:00 every Saturday.", items=candidate_items(note))
    )
    await hippo.flush(run_request=True, digest=None, final_reflection=None)
    assert [entry.text for entry in await store.active_entries()] == [
        "Yoga starts at 9:00 every Saturday."
    ]


async def test_no_item_match_does_not_restore_candidate_sources() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    note = ReasoningNote(
        found=["the deployment happens on Friday"],
        source_ids=["unrelated"],
        source_records=[("unrelated", '{"name": "Alice"}')],
    )
    items = candidate_items(note)
    assert items[0].source_ids == ()
    hippo.propose(
        _candidate(note.found[0], items=items, provenance={"source_ids": ["unrelated"]})
    )
    decisions = await hippo.admit("n1", "rehearsal")
    assert decisions[0].action == "NEW"
    assert decisions[0].source_ids == ()
    assert (await store.active_entries())[0].provenance["source_ids"] == []
    assert (await store.decisions())[0]["source_ids"] == []


async def test_missing_item_sources_fall_back_to_candidate_sources() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    hippo.propose(
        _candidate(
            "the deployment happens on Friday", provenance={"source_ids": ["r1"]}
        )
    )
    decisions = await hippo.admit("n1", "rehearsal")
    assert decisions[0].source_ids == ("r1",)


async def test_completed_outcome_exempts_a_bare_meta_digest() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, admit_on_outcome=True)
    text = "I've read and remembered all of Jamie's records."
    digest = _candidate(
        text,
        candidate_id="d1",
        source="digest",
        provenance={"final_status": "completed"},
    )
    decisions = await hippo.flush(
        run_request=False, digest=digest, final_reflection=None
    )
    assert [(d.action, d.admitted_by) for d in decisions] == [("NEW", "outcome")]
    assert [entry.text for entry in await store.active_entries()] == [text]


async def test_each_item_keeps_its_own_source_ids_in_the_entry() -> None:
    store = LongTerm()
    hippo = Hippo(long=store)
    candidate = _candidate(
        "two facts",
        items=(
            CandidateItem("the amount is 42 thousand for the migration", ("r1",)),
            CandidateItem("the office is on Main Street in Springfield", ("r2",)),
        ),
    )
    hippo.propose(candidate)
    await hippo.admit("n1", "rehearsal")
    entries = await store.active_entries()
    assert [entry.provenance["source_ids"] for entry in entries] == [
        ["r1"],
        ["r2"],
    ]
