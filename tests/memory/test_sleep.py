"""Sleep by rules: tidy the notes with no model call, report it, undo it.

Synthetic notes (written straight to the store, so duplicates can exist) and
chat turns (real turns of a session file with their ``turn.memory`` event) on a
temporary memory folder; no model, no embedding server. Each operation does
its job: O1 merges duplicates, O2 writes the newer value of a note's subject,
O3 recovers a lasting statement no note kept. A note the user reworded is not
merged away, a wrong value or owner is refused, a run makes at most 20
changes, a watermark keeps a second run from redoing a turn, undo puts a run
back exactly, and ``off`` does nothing.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.long import StoredEntry
from localmelo.melo.schema import new_record_id
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, records, sleep
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo

HOLDER = "test-host:1:sleep"
LONG_AGO, SOON = time.time() - 10_000, time.time() + 10_000


@pytest.fixture()
async def chats(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory.sleep = "rules"
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def put(
    memory: ChatMemory,
    text: str,
    at: float = LONG_AGO,
    *,
    session: str = "old-chat",
    turn: int = 1,
    vector: list[float] | None = None,
    edited: bool = False,
) -> int:
    """A note, written as it is (no supersede), kept *at* that time."""
    store, _ = memory.stores()
    provenance = {"session_id": session, "turn": turn, "topic": "about", "quote": ""}
    entry_id, _ = await store.apply(
        text=text,
        kind="profile",
        source="user",
        admitted_by="user_statement",
        provenance=provenance,
        task_id="",
        session_id=session,
        created_at=at,
        embedding=vector,
        action="NEW",
        target_id=None,
        reason="seeded",
        source_ids=(),
        keep_text=False,
        candidate_id=new_record_id(),
        record={"topic": "about", "quote": "", "embed_model": "m" if vector else ""},
    )
    assert entry_id is not None
    if edited:
        await memory.edit_record(entry_id, text)
    return entry_id


async def chat(
    chats: SessionStore,
    typed: str,
    *,
    noted: tuple[str, ...] = (),
    unsaved: bool = False,
) -> TurnInfo:
    """A finished chat turn of *typed* whose ``turn.memory`` event lists *noted*."""
    turn = await chats.begin_turn(None, typed, holder=HOLDER, new_session=NewSession())
    data = {
        "turn": turn.turn,
        "turn_id": turn.id,
        "used": [],
        "remembered": [
            {"id": 900 + n, "text": t, "topic": "about"} for n, t in enumerate(noted)
        ],
        "refused": "",
        "recall": {},
        "unsaved_claim": unsaved,
    }
    await chats.append_events(
        turn.session_id,
        turn.turn,
        [{"type": "turn.memory", "data": data}],
        holder=HOLDER,
    )
    return await chats.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="Noted.",
    )


async def texts(memory: ChatMemory) -> set[str]:
    return {entry.text for entry in await memory.records()}


def rows(memory: ChatMemory) -> list[tuple[int, str, str]]:
    """Every row of the records file, superseded ones too: id, text, status."""
    with sqlite3.connect(Path(memory.directory) / "records.db") as conn:
        return conn.execute(
            "SELECT id, text, status FROM long_term ORDER BY id"
        ).fetchall()


# ── O1: merge duplicates ──


async def test_o1_the_older_note_of_one_subject_is_superseded_by_the_newest(
    memory: ChatMemory,
) -> None:
    older = await put(memory, "You live in Lisbon.", LONG_AGO)
    newer = await put(memory, "You live in Utrecht.", LONG_AGO + 5)
    other = await put(memory, "Your dentist is Lee Park.", LONG_AGO + 1)
    report = await memory.sleep()
    (change,) = report.changes
    assert (change.op, change.before_ids, change.after_id) == ("O1", [older], newer)
    assert await texts(memory) == {"Your dentist is Lee Park.", "You live in Utrecht."}
    # Nothing is deleted: the older note is kept, superseded.
    assert [(i, s) for i, _, s in rows(memory)] == [
        (older, "superseded"),
        (newer, "active"),
        (other, "active"),
    ]


async def test_o1_notes_close_in_meaning_and_wording_merge_but_other_subjects_do_not(
    memory: ChatMemory,
) -> None:
    same = [1.0, 0.0, 0.0]
    await put(memory, "You prefer short answers.", LONG_AGO, vector=same)
    await put(memory, "You prefer short answers always.", LONG_AGO + 5, vector=same)
    # The same vector and most of the words, but two subjects: both stay.
    await put(memory, "Your flight to Porto is at gate 22.", LONG_AGO + 6, vector=same)
    await put(memory, "Your flight to Faro is at gate 9.", LONG_AGO + 7, vector=same)
    report = await memory.sleep()
    assert [c.op for c in report.changes] == ["O1"]
    assert await texts(memory) == {
        "Your flight to Faro is at gate 9.",
        "Your flight to Porto is at gate 22.",
        "You prefer short answers always.",
    }


async def test_a_note_the_user_reworded_is_not_merged_away(memory: ChatMemory) -> None:
    await put(memory, "You live in Lisbon.", LONG_AGO, edited=True)
    await put(memory, "You live in Utrecht.", LONG_AGO + 5)
    assert (await memory.sleep()).changes == []
    assert await texts(memory) == {"You live in Lisbon.", "You live in Utrecht."}


# ── O2: a newer typed value supersedes ──


async def test_o2_a_newer_turn_that_states_another_value_replaces_the_note(
    memory: ChatMemory, chats: SessionStore
) -> None:
    old = await put(memory, "You go by Pip.")
    turn = await chat(chats, "My name is Juno.")
    report = await memory.sleep(chats=chats)
    (change,) = report.changes
    assert (change.op, change.before_ids) == ("O2", [old])
    assert change.source_turn == [turn.session_id, turn.turn]
    (note,) = await memory.records()
    assert (note.id, note.text) == (change.after_id, "Your name is Juno.")
    assert (note.admitted_by, note.source, note.supersedes) == ("sleep", "sleep", old)
    assert note.provenance["source"] == "sleep"
    assert note.provenance["run_id"] == report.run_id
    assert (note.provenance["session_id"], note.provenance["turn"]) == (
        turn.session_id,
        turn.turn,
    )
    assert note.provenance["quote"] == "My name is Juno."
    assert note.session_id == turn.session_id
    assert (old, "You go by Pip.", "superseded") in rows(memory)
    # Recall reads a note sleep wrote like any other: the newest value only.
    found = await memory.search("what is my name?")
    assert [item.text for item in found if item.is_record] == ["Your name is Juno."]


async def test_o2_a_note_newer_than_the_turn_stays_and_the_same_value_is_no_change(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await put(memory, "You go by Juno.", SOON)  # said after the turn below
    await put(memory, "Your dentist is Lee Park.")
    await chat(chats, "My name is Pip.")
    await chat(chats, "My dentist is Lee Park.")  # the same value again
    assert (await memory.sleep(chats=chats)).changes == []
    assert await texts(memory) == {"Your dentist is Lee Park.", "You go by Juno."}


async def test_o2_a_note_the_user_reworded_gives_way_only_to_a_contradicting_turn(
    memory: ChatMemory, chats: SessionStore
) -> None:
    pip = await put(memory, "You go by Pip.", edited=True)
    await put(memory, "Your dentist is Lee Park.", edited=True)
    await chat(chats, "I prefer short answers.")  # no subject of an edited note
    await chat(chats, "My dentist is Lee Park.")  # not another value
    assert (await memory.sleep(chats=chats)).changes == []
    await chat(chats, "My name is Juno.")
    second = await memory.sleep(chats=chats)
    assert [(c.op, c.before_ids) for c in second.changes] == [("O2", [pip])]


# ── O3: recover lost facts ──


async def test_o3_a_lasting_statement_no_note_kept_is_written(
    memory: ChatMemory, chats: SessionStore
) -> None:
    turn = await chat(chats, "I live in Utrecht.")
    report = await memory.sleep(chats=chats)
    (change,) = report.changes
    assert (change.op, change.before_ids, change.text) == (
        "O3",
        [],
        "You live in Utrecht.",
    )
    (note,) = await memory.records()
    assert note.admitted_by == "sleep" and note.created_at == turn.started_at
    assert report.model_calls == 0 and report.variant == "rules" and not report.aborted


async def test_o3_what_a_note_of_the_turn_keeps_or_the_user_undid_is_not_written_again(
    memory: ChatMemory, chats: SessionStore
) -> None:
    # Kept by the model, then undone by the user: the event still lists it.
    await chat(chats, "I live in Utrecht.", noted=("You live in Utrecht.",))
    await chat(chats, "My dentist is Lee Park.")  # not kept: recovered
    report = await memory.sleep(chats=chats)
    assert [c.text for c in report.changes] == ["Your dentist is Lee Park."]


async def test_o3_questions_secrets_other_people_and_a_replaced_turn_write_nothing(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await chat(chats, "Where do I live?")
    await chat(chats, "My password is hunter2.")
    await chat(chats, "Marco's studio is on Main Street.")
    await chat(chats, "I live in Lisbon.")  # a newer turn replaces it
    await chat(chats, "I live in Utrecht.")
    report = await memory.sleep(chats=chats)
    assert [c.text for c in report.changes] == ["You live in Utrecht."]


async def test_an_undone_turn_and_a_turn_without_a_memory_event_are_not_read(
    memory: ChatMemory, chats: SessionStore
) -> None:
    undone = await chat(chats, "I live in Lisbon.")
    await chats.undo_last_turn(undone.session_id)
    # Memory was off for this turn: it has no turn.memory event.
    bare = await chats.begin_turn(
        None, "My dentist is Lee Park.", holder=HOLDER, new_session=NewSession()
    )
    await chats.finish_turn(
        bare.session_id,
        bare.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="Ok.",
    )
    assert (await memory.sleep(chats=chats)).changes == []


# ── The checks, the cap, the watermark, undo, off ──


async def test_a_wrong_value_a_wrong_owner_and_a_secret_are_refused(
    memory: ChatMemory,
) -> None:
    turn = sleep.SleepTurn("s", 1, 1.0, 2.0, "I live in Utrecht.")
    for text, quote in (
        ("You live in Lisbon.", "I live in Utrecht."),  # not the user's value
        ("Your studio is on Main Street.", "Marco's studio is on Main Street."),
        ("Your password is hunter2.", "My password is hunter2."),
    ):
        cand = records.Candidate(text, quote, "about", "sleep")
        typed = sleep.SleepTurn("s", 1, 1.0, 2.0, quote)
        assert not sleep._allowed(typed, cand, [])
    assert sleep._allowed(
        turn, records.Candidate("You live in Utrecht.", "", "about", "sleep"), []
    )


async def test_a_candidate_that_fails_the_checks_is_not_written(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await chat(chats, "I live in Utrecht.")
    wrong = records.Candidate("You live in Lisbon.", "I live in Utrecht.", "about", "x")
    monkeypatch.setattr(records, "statements", lambda typed: [wrong])
    assert (await memory.sleep(chats=chats)).changes == []
    assert await memory.records() == []


async def test_a_run_makes_at_most_20_changes_and_the_next_run_goes_on(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sleep.MAX_CHANGES == 20
    monkeypatch.setattr(sleep, "MAX_CHANGES", 3)
    for who in ("advisor", "dentist", "coach", "tutor", "mentor"):
        await chat(chats, f"My {who} is Sam.")
    first = await memory.sleep(chats=chats)
    assert (len(first.changes), first.capped) == (3, True)
    assert await memory.stores()[1].meta("sleep_until") == 0  # not past what is left
    second = await memory.sleep(chats=chats)
    assert (len(second.changes), second.capped) == (2, False)
    assert len(await memory.records()) == 5


async def test_the_watermark_keeps_a_second_run_from_redoing_a_turn(
    memory: ChatMemory, chats: SessionStore
) -> None:
    first_turn = await chat(chats, "I live in Utrecht.")
    (change,) = (await memory.sleep(chats=chats)).changes
    assert change.after_id is not None
    until = await memory.stores()[1].meta("sleep_until")
    assert until == first_turn.finished_at
    await memory.delete_record(change.after_id)  # the user forgot the note
    assert (await memory.sleep(chats=chats)).changes == []  # the turn is not read again
    await chat(chats, "My dentist is Lee Park.")
    assert [c.text for c in (await memory.sleep(chats=chats)).changes] == [
        "Your dentist is Lee Park."
    ]
    assert await texts(memory) == {"Your dentist is Lee Park."}


async def test_undo_puts_everything_back_exactly(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await put(memory, "You live in Lisbon.", LONG_AGO)
    await put(memory, "You live in Utrecht.", LONG_AGO + 5)  # O1
    await put(memory, "You go by Pip.")  # O2
    await chat(chats, "My name is Juno.")
    await chat(chats, "My advisor is Dr. Okafor.")  # O3
    before = rows(memory)
    report = await memory.sleep(chats=chats)
    assert sorted(c.op for c in report.changes) == ["O1", "O2", "O3"]
    assert rows(memory) != before
    undone = await memory.undo_sleep(report.run_id)
    assert undone == {"run_id": report.run_id, "restored": 2, "removed": 2, "kept": 0}
    assert rows(memory) == before
    assert await memory.undo_sleep(report.run_id) is None  # once
    assert await memory.undo_sleep("nope") is None


async def test_undo_leaves_a_note_a_later_note_replaced(
    memory: ChatMemory, chats: SessionStore
) -> None:
    await chat(chats, "I live in Lisbon.")
    report = await memory.sleep(chats=chats)
    store, _ = memory.stores()
    await records.write(
        store,
        [
            records.Candidate(
                "You live in Utrecht.", "I live in Utrecht.", "about", "user_statement"
            )
        ],
        [None],
        source={"session_id": "later", "turn": 1},
        embed_model="",
        created_at=SOON,
    )
    undone = await memory.undo_sleep(report.run_id)
    assert undone is not None and (undone["removed"], undone["kept"]) == (0, 1)
    assert await texts(memory) == {"You live in Utrecht."}


async def test_a_failure_keeps_the_changes_made_and_says_why(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = await put(memory, "You live in Lisbon.", LONG_AGO)
    await put(memory, "You live in Utrecht.", LONG_AGO + 5)
    await chat(chats, "My dentist is Lee Park.")

    async def boom(*args: Any, **kwargs: Any) -> list[records.Written]:
        raise RuntimeError("the disk is full")

    monkeypatch.setattr(records, "write", boom)
    report = await memory.sleep(chats=chats)
    assert [c.op for c in report.changes] == ["O1"]
    assert report.aborted == "RuntimeError: the disk is full"
    assert await memory.stores()[1].meta("sleep_until") == 0  # the turn is read again
    undone = await memory.undo_sleep(report.run_id)  # what was done can be undone
    assert undone is not None and undone["restored"] == 1
    assert older in {e.id for e in await memory.records()}


async def test_off_does_nothing_and_model_variants_keep_rules_when_model_unavailable(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await put(memory, "You live in Lisbon.", LONG_AGO)
    await put(memory, "You live in Utrecht.", LONG_AGO + 5)
    await chat(chats, "My dentist is Lee Park.")
    before: Any = rows(memory)
    off = await memory.sleep("off", chats=chats)
    assert (off.variant, off.changes, off.aborted) == ("off", [], "sleep is off")
    assert rows(memory) == before and await memory.stores()[1].meta("sleep_until") == 0
    monkeypatch.setattr(memory, "chat_provider", lambda: None)
    for variant in ("model", "model+tags"):
        report = await memory.sleep(variant, chats=chats)
        assert report.variant == variant and report.model_calls == 0
        assert [c.op for c in report.changes] == ["O1", "O3"]
        assert report.aborted == "RuntimeError: chat model is not configured"
        assert await memory.stores()[1].meta("sleep_until") == 0
        await memory.undo_sleep(report.run_id)
        assert rows(memory) == before


async def test_the_setting_picks_the_variant_and_memory_off_does_nothing(
    tmp_path: Path,
) -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        assert (await memory.sleep()).variant == "off"  # the default
        cfg.memory.sleep = "rules"
        assert (await memory.sleep()).variant == "rules"
        cfg.memory.across_chats = False
        assert (await memory.sleep()).variant == "off"
    finally:
        await memory.aclose()


def test_duplicates_and_the_candidates_are_pure_functions_of_the_notes_and_turns() -> (
    None
):
    def note(entry_id: int, text: str, at: float) -> StoredEntry:
        return StoredEntry(
            id=entry_id,
            text=text,
            kind="profile",
            source="user",
            status="active",
            admitted_by="user_statement",
            provenance={},
            task_id="",
            session_id="s",
            uses=0,
            created_at=at,
            last_used_at=0,
            supersedes=None,
            links=(),
            embed_model="",
            embed_dim=0,
            vector=None,
        )

    kept, gone = sleep.duplicates(
        [note(1, "You live in Lisbon.", 1), note(2, "You live in Utrecht.", 2)]
    )[0]
    assert (kept.id, gone) == (2, [1])
    flagged = sleep.SleepTurn("a", 1, 5.0, 6.0, "My dentist is Lee Park.", (), True)
    plain = sleep.SleepTurn("a", 2, 7.0, 8.0, "I live in Utrecht.")
    # Flagged first, then the newest.
    assert [t.turn for t, _ in sleep.recoverable([plain, flagged], [])] == [1, 2]
