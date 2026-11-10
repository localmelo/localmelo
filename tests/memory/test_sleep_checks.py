"""Sleep keeps paraphrases out, retires a note a newer one replaced and never pins.

Stub embeddings return cosines measured with a real embedding model, so the
guard is checked on the numbers that decide it. No model, no server.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from localmelo.melo.memory.long import StoredEntry
from localmelo.support.memory import ChatMemory, sleep
from localmelo.support.memory.recall import current_only, newer_first

from . import test_sleep as rules

memory = rules.memory
chats = rules.chats

TRAIN = "You take the 7:40 train to work."
PARAPHRASE = "Your usual train to the office is the 7:40."
AT = rules.LONG_AGO


def embedding(cosine: float) -> Any:
    """An embed callback whose vector has *cosine* with the note vector ``[1, 0]``."""

    async def embed(text: str) -> tuple[list[float], str]:
        return [cosine, math.sqrt(1 - cosine * cosine)], "m"

    return embed


def turn(session: str, typed: str, at: float, number: int = 1) -> sleep.SleepTurn:
    return sleep.SleepTurn(session, number, at, at + 1, typed)


# ── Fix 1: what an active note already holds ──


@pytest.mark.parametrize(
    ("note", "candidate", "cosine", "held"),
    [
        # A paraphrase is held.
        (TRAIN, PARAPHRASE, 0.878, True),
        (TRAIN, PARAPHRASE, 0.85, True),
        (TRAIN, PARAPHRASE, 0.84, False),
        # A changed value scores like a paraphrase and is never held.
        (
            "Your locker number at the pool is 31.",
            "Your locker number at the pool is 58.",
            0.886,
            False,
        ),
        ("Your gym day is Monday.", "Your gym day moved to Friday.", 0.785, False),
        ("你的牙医预约在周二。", "你的牙医预约改到了周四。", 0.747, False),
        # The same values with the opposite polarity are never held, either way.
        (
            "You live in Lisbon for the summer.",
            "You no longer live in Lisbon.",
            0.95,
            False,
        ),
        (
            "You no longer live in Lisbon.",
            "You live in Lisbon for the summer.",
            0.95,
            False,
        ),
        ("你住在里斯本。", "你不再住在里斯本。", 0.95, False),
        # Other subjects stay apart, whatever the cosine.
        ("Your gym day is Monday.", "Your swim day is Monday.", 0.656, False),
        ("Your gym day is Monday.", "Your swim day is Monday.", 0.99, False),
        ("You live in Utrecht.", "You live on the 3rd floor.", 0.547, False),
        ("You live in Utrecht.", "You live on the 3rd floor.", 0.99, False),
    ],
)
async def test_held_needs_values_polarity_subject_and_cosine(
    memory: ChatMemory, note: str, candidate: str, cosine: float, held: bool
) -> None:
    await rules.put(memory, note, vector=[1.0, 0.0])
    guard = sleep._FactGuard(embedding(cosine))
    assert await guard.held(candidate, await memory.records()) is held


async def test_reading_says_why_a_fact_is_or_is_not_held(memory: ChatMemory) -> None:
    train = await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    swim = await rules.put(memory, "Your gym day is Monday.", vector=[0.0, 1.0])
    notes = await memory.records()
    guard = sleep._FactGuard(embedding(0.878))
    assert await guard.check(PARAPHRASE, notes) == {
        "note_id": train,
        "coverage": True,
        "subject_same": True,
        "polarity_same": True,
        "cosine": 0.878,
        "overlap": 0.3333,  # the words alone would not hold it
        "held": True,
    }
    other = await guard.check("Your swim day is Monday.", notes)
    assert other["note_id"] == swim and other["coverage"] is True
    assert other["subject_same"] is False and other["cosine"] is None
    assert other["held"] is False
    lost = await guard.check("Your locker number is 58.", notes)
    assert lost["coverage"] is False and lost["held"] is False
    assert lost["note_id"] is None and lost["cosine"] is None
    none = await guard.check("You enjoy kitesurfing.", notes)  # no value to match
    assert none["coverage"] is False and none["held"] is False


async def test_a_changed_value_is_not_embedded(memory: ChatMemory) -> None:
    await rules.put(memory, "Your locker number at the pool is 31.", vector=[1.0, 0.0])

    async def forbidden(text: str) -> tuple[list[float], str]:
        pytest.fail("no note has the value, so the candidate need not be embedded")

    reading = await sleep._FactGuard(forbidden).check(
        "Your locker number at the pool is 58.", await memory.records()
    )
    assert reading["coverage"] is False and reading["cosine"] is None


@pytest.mark.parametrize(
    ("text", "negated"),
    [
        ("You no longer live in Lisbon.", True),
        ("You never drink coffee.", True),
        ("You don't live in Lisbon.", True),
        ("You can't eat peanuts.", True),
        ("Your dentist is not Dr. Lin.", True),
        ("你不再住在里斯本。", True),
        ("你没有猫。", True),
        ("You live in Lisbon for the summer.", False),
        ("Your usual train to the office is the 7:40.", False),
        ("你住在里斯本。", False),
    ],
)
def test_polarity_is_a_general_negation_check(text: str, negated: bool) -> None:
    assert sleep._negated(text) is negated


async def test_paraphrase_at_the_measured_cosine_is_held_and_logged(
    memory: ChatMemory,
) -> None:
    old = await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    source = turn("s", "My usual train to the office is the 7:40.", AT + 10)
    report = await sleep.run(
        *memory.stores(), [source], variant="rules", embed_note=embedding(0.878)
    )
    assert not report.changes
    assert [(n.id, n.text) for n in await memory.records()] == [(old, TRAIN)]
    (skipped,) = report.to_json()["skipped"]  # the rule skipped it, with its numbers
    assert (skipped["session_id"], skipped["turn"]) == ("s", 1)
    assert skipped["text"] == PARAPHRASE
    assert skipped["guard"]["held"] is True and skipped["guard"]["cosine"] == 0.878
    assert skipped["guard"]["coverage"] and skipped["guard"]["polarity_same"]


async def test_paraphrase_below_the_cosine_is_written_with_its_numbers(
    memory: ChatMemory,
) -> None:
    await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    source = turn("s", "My usual train to the office is the 7:40.", AT + 10)
    report = await sleep.run(
        *memory.stores(), [source], variant="rules", embed_note=embedding(0.8)
    )
    (change,) = report.changes
    assert change.op == "O3" and not report.skipped
    assert change.guard is not None
    assert change.guard["held"] is False and change.guard["cosine"] == 0.8


async def test_locker_changed_value_is_written_even_next_to_a_close_vector(
    memory: ChatMemory,
) -> None:
    old = await rules.put(
        memory, "Your locker number at the pool is 31.", vector=[1.0, 0.0]
    )
    source = turn("s", "My locker number at the pool is 58.", AT + 10)

    async def forbidden(text: str) -> tuple[list[float], str]:
        pytest.fail("a changed value is decided by its values, not by cosine")

    report = await sleep.run(
        *memory.stores(), [source], variant="rules", embed_note=forbidden
    )
    (change,) = report.changes
    assert (change.op, change.before_ids) == ("O2", [old])
    assert change.guard is not None and change.guard["coverage"] is False
    assert await rules.texts(memory) == {"Your locker number at the pool is 58."}


@pytest.mark.parametrize(
    ("note", "quote", "text"),
    [
        (
            "You live in Lisbon for the summer.",
            "I no longer live in Lisbon.",
            "You no longer live in Lisbon.",
        ),
        (
            "You no longer live in Lisbon.",
            "I live in Lisbon for the summer.",
            "You live in Lisbon for the summer.",
        ),
    ],
)
async def test_the_other_polarity_is_a_different_fact_not_an_existing_one(
    memory: ChatMemory, note: str, quote: str, text: str
) -> None:
    await rules.put(memory, note, vector=[1.0, 0.0])
    guard = sleep._FactGuard(embedding(0.95))
    reading = await guard.check(text, await memory.records())
    assert reading["coverage"] is True and reading["cosine"] == 0.95
    assert reading["polarity_same"] is False and reading["held"] is False


# ── Fix 2: sleep notes are recall only ──


async def test_a_sleep_note_is_never_pinned_but_a_user_note_of_the_topic_is(
    memory: ChatMemory,
) -> None:
    await rules.put(
        memory, "You live in Utrecht.", AT, quote="I live in Utrecht.", session="a"
    )
    source = turn("b", "My advisor is Dr. Okafor.", AT + 10)
    report = await sleep.run(*memory.stores(), [source], variant="rules")
    (change,) = report.changes
    assert change.op == "O3"
    written = {n.text: n for n in await memory.records()}
    note = written["Your advisor is Dr. Okafor."]
    assert note.admitted_by == "sleep" and note.provenance["topic"] == "about"
    assert note.provenance["quote"] == "My advisor is Dr. Okafor."
    pinned = await memory.pinned_quotes()
    assert "I live in Utrecht." in pinned and "Okafor" not in pinned
    # Recall still finds it.
    found = await memory.search("who is my advisor")
    assert any("Okafor" in item.text for item in found if item.is_record)


# ── Fix 3: a newer active note replaces the older one of its subject ──


async def pair(
    memory: ChatMemory,
    old: tuple[str, str],
    new: tuple[str, str],
    *,
    topics: tuple[str, str] = ("about", "about"),
    edited: str = "",
    tag: str = "",
) -> tuple[int, int]:
    """The older note (session ``o``) and the newer one (session ``n``), both
    active, each with its text and the quote the user typed. *tag* tells the
    sessions of two pairs apart."""
    older = await rules.put(
        memory,
        old[0],
        AT,
        session="o" + tag,
        topic=topics[0],
        quote=old[1],
        vector=[0.0, 1.0],
        edited=edited == "old",
    )
    newer = await rules.put(
        memory,
        new[0],
        AT + 20,
        session="n" + tag,
        topic=topics[1],
        quote=new[1],
        vector=[1.0, 0.0],
        edited=edited == "new",
    )
    return older, newer


async def states(memory: ChatMemory) -> dict[int, str]:
    return {row[0]: row[2] for row in rules.rows(memory)}


C3 = (
    ("你的牙医预约在周二。", "我的牙医预约在周二。"),
    ("你的牙医预约改到了周四。", "我的牙医预约改到周四了。"),
)
C4 = (
    ("My manager is Priya Shah.", "My manager is Priya Shah."),
    (
        "My new manager is Daniel Kim from next week.",
        "My new manager is Daniel Kim from next week.",
    ),
)


@pytest.mark.parametrize(
    ("newer", "older", "expected"),
    [
        ("My new manager is Daniel Kim.", "My manager is Priya Shah.", True),
        ("Your current gym day is Friday.", "Your gym day is Monday.", True),
        ("你的牙医预约改到了周四。", "你的牙医预约在周二。", True),
        # Another subject, a restatement, or only an extra detail.
        ("Your new bike is a red Brompton.", "My manager is Priya Shah.", False),
        ("Your swim day is Thursday.", "Your gym day is Monday.", False),
        ("Your gym day is Monday.", "Your gym day is Monday.", False),
        ("Your dentist is Dr. Lin.", "Your dentist is Dr. Lin on Main Street.", False),
        # The older note says the new one's values and more: not a changed value.
        ("You no longer live in Lisbon.", "You live in Lisbon for the summer.", False),
        # A relative's name is not the user's.
        ("Your new sister's name is Mei.", "Your name is Juno.", False),
    ],
)
def test_replaced_by_wants_one_subject_and_a_swapped_value(
    newer: str, older: str, expected: bool
) -> None:
    def note(entry_id: int, text: str, at: float) -> StoredEntry:
        return StoredEntry(
            id=entry_id,
            text=text,
            kind="profile",
            source="user",
            status="active",
            admitted_by="user_statement",
            provenance={"session_id": "s", "turn": entry_id},
            task_id="",
            session_id="s",
            uses=0,
            created_at=at,
            last_used_at=0.0,
            supersedes=None,
            links=(),
            embed_model="",
            embed_dim=0,
            vector=None,
        )

    assert sleep.replaced_by(note(2, newer, 20), note(1, older, 10), {}) is expected
    assert sleep.replaced_by(note(1, older, 10), note(2, newer, 20), {}) is False


# ── The rule O2b ──


async def tidy(memory: ChatMemory) -> sleep.SleepReport:
    """A rules run over the notes as they are (no chat turn)."""
    return await sleep.run(*memory.stores(), [], variant="rules")


@pytest.mark.parametrize(
    ("old", "new", "topics"),
    [
        (*C3, ("projects", "decisions")),
        (*C4, ("people", "people")),
        (
            ("Your gym day is Monday.", "My gym day is Monday."),
            ("Your gym day moved to Friday.", "my gym day moved to Friday"),
            ("preferences", "preferences"),
        ),
    ],
)
async def test_rules_retire_the_older_note_a_newer_active_one_replaced(
    memory: ChatMemory,
    old: tuple[str, str],
    new: tuple[str, str],
    topics: tuple[str, str],
) -> None:
    older, newer = await pair(memory, old, new, topics=topics)
    report = await tidy(memory)
    (change,) = report.changes
    assert (change.op, change.before_ids, change.after_id) == ("O2b", [older], newer)
    assert change.before_texts == [old[0]] and change.quote == new[1]
    assert await states(memory) == {older: "superseded", newer: "active"}
    assert await rules.texts(memory) == {new[0]}  # nothing is written
    assert (await tidy(memory)).changes == []  # a second run has nothing to do


async def test_rules_read_the_time_of_the_source_turns(memory: ChatMemory) -> None:
    """The note said later wins whatever order its row was written in, and the
    older value is not recovered from its turn once the newer one replaced it."""
    newer = await rules.put(memory, C4[1][0], AT, session="n", quote=C4[1][1])
    older = await rules.put(memory, C4[0][0], AT + 90, session="o", quote=C4[0][1])
    turns = [turn("o", C4[0][1], AT - 50), turn("n", C4[1][1], AT + 20)]
    report = await sleep.run(*memory.stores(), turns, variant="rules")
    assert [(c.op, c.before_ids, c.after_id) for c in report.changes] == [
        ("O2b", [older], newer)
    ]
    assert await rules.texts(memory) == {C4[1][0]}


async def test_rule_undo_puts_the_older_note_back(memory: ChatMemory) -> None:
    older, newer = await pair(memory, *C3)
    report = await tidy(memory)
    undone = await memory.undo_sleep(report.run_id)
    assert undone == {"run_id": report.run_id, "restored": 1, "removed": 0, "kept": 0}
    assert await states(memory) == {older: "active", newer: "active"}


async def test_undo_leaves_the_older_note_when_a_later_note_replaced_the_kept_one(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    report = await tidy(memory)
    await memory.stores()[0].set_status(
        [newer], "superseded", keeper=None, admitted_by="user", reason="later"
    )
    undone = await memory.undo_sleep(report.run_id)
    assert undone is not None and undone["kept"] == 1 and undone["restored"] == 0
    assert await states(memory) == {older: "superseded", newer: "superseded"}


async def test_rule_o2b_counts_against_the_change_cap_and_the_next_run_goes_on(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sleep, "MAX_CHANGES", 1)
    first = await pair(memory, *C3, tag="1")
    second = await pair(memory, *C4, tag="2")
    report = await tidy(memory)
    assert report.capped and len(report.changes) == 1
    done = {n for n, s in (await states(memory)).items() if s == "superseded"}
    assert len(done) == 1 and done <= {first[0], second[0]}
    again = await tidy(memory)
    assert len(again.changes) == 1 and again.changes[0].op == "O2b"
    assert {n for n, s in (await states(memory)).items() if s == "active"} == {
        first[1],
        second[1],
    }


@pytest.mark.parametrize(
    ("old", "new"),
    [
        # Other subjects: a gym day is not a swim day, nor a piano class a dance one.
        (
            ("Your gym day is Monday.", "My gym day is Monday."),
            ("Your swim day is Thursday.", "My swim day is Thursday."),
        ),
        (
            ("你的钢琴课在周三。", "我的钢琴课在周三。"),
            ("你的舞蹈课在周五。", "我的舞蹈课在周五。"),
        ),
        (
            ("Your piano class is on Wednesday.", "My piano class is on Wednesday."),
            ("Your dance class is on Friday.", "My dance class is on Friday."),
        ),
        # The summer and the no-longer note, in the order a bench seeds them.
        (
            ("You live in Lisbon for the summer.", "I live in Lisbon for the summer."),
            ("You no longer live in Lisbon.", "I no longer live in Lisbon."),
        ),
        # A third person's note is not the user's.
        (
            ("Priya's manager is Tom Lee.", "Priya's manager is Tom Lee."),
            ("Your new manager is Daniel Kim.", "My new manager is Daniel Kim."),
        ),
        (
            ("Your manager is Daniel Kim.", "My manager is Daniel Kim."),
            ("Priya's new manager is Tom Lee.", "Priya's new manager is Tom Lee."),
        ),
        (
            (
                "Tom Briggs's office is in room 9C.",
                "Tom Briggs's office is in room 9C.",
            ),
            (
                "Tom Briggs's new office is in room 12.",
                "Tom Briggs's new office is in room 12.",
            ),
        ),
    ],
)
async def test_rules_leave_other_subjects_scopes_and_other_peoples_notes(
    memory: ChatMemory, old: tuple[str, str], new: tuple[str, str]
) -> None:
    older, newer = await pair(memory, old, new)
    report = await tidy(memory)
    assert not [c for c in report.changes if c.op == "O2b"]
    assert not report.changes
    assert await states(memory) == {older: "active", newer: "active"}


@pytest.mark.parametrize("edited", ["old", "new"])
async def test_rules_never_retire_or_keep_a_note_the_user_reworded(
    memory: ChatMemory, edited: str
) -> None:
    older, newer = await pair(memory, *C4, edited=edited)
    report = await tidy(memory)
    assert not report.changes
    assert await states(memory) == {older: "active", newer: "active"}


# ── A retired value does not come back through its exchange ──


async def kept_replaces(memory: ChatMemory, note: int) -> Any:
    entry = await memory.stores()[0].entry(note)
    assert entry is not None
    return entry.provenance.get("replaces")


@pytest.mark.parametrize("shape", [C3, C4])
async def test_o2b_adds_the_older_turn_to_the_kept_notes_replaces(
    memory: ChatMemory, shape: tuple[tuple[str, str], tuple[str, str]]
) -> None:
    older, newer = await pair(memory, *shape)
    assert await kept_replaces(memory, newer) is None
    report = await tidy(memory)
    (change,) = report.changes
    assert change.op == "O2b" and change.after_id == newer
    assert change.replaces_before is None and change.replaces_after == [["o", 1]]
    assert await kept_replaces(memory, newer) == [["o", 1]]
    # Only the replaces changed: the note keeps its text, quote and place.
    kept = await memory.stores()[0].entry(newer)
    assert kept is not None and kept.text == shape[1][0]
    assert kept.provenance["quote"] == shape[1][1] and kept.provenance["turn"] == 1


async def test_recall_drops_the_exchange_of_a_retired_value_and_undo_brings_it_back(
    memory: ChatMemory,
) -> None:
    from localmelo.support.memory.recall import Corpus, Item, replaced_turns
    from localmelo.support.memory.service import _record_item

    older, newer = await pair(memory, *C3)
    store = memory.stores()[0]
    old_exchange = Item("d:1", "User: 我的牙医预约在周二。", "chat o", "o", 1, AT)
    message = "我的牙医预约是什么时候？"

    async def recall() -> tuple[dict[tuple[str, int], Item], list[Item]]:
        entry = await store.entry(newer)
        assert entry is not None
        corpus = Corpus()
        corpus.add(_record_item(entry))
        replaced = replaced_turns(corpus)
        return replaced, current_only(newer_first([old_exchange], replaced), message)

    replaced, shown = await recall()
    assert replaced == {} and [i.key for i in shown] == ["d:1"]  # nothing yet
    report = await tidy(memory)
    replaced, shown = await recall()
    assert list(replaced) == [("o", 1)] and replaced[("o", 1)].key == f"r:{newer}"
    assert shown == [] or [i.key for i in shown] == [f"r:{newer}"]
    assert "d:1" not in [i.key for i in shown]  # the old Tuesday is not given
    assert await memory.undo_sleep(report.run_id) is not None
    replaced, shown = await recall()
    assert replaced == {} and [i.key for i in shown] == ["d:1"]
    assert await kept_replaces(memory, newer) is None
    assert await states(memory) == {older: "active", newer: "active"}


async def test_the_running_recall_sees_the_turns_a_sleep_replaced_and_its_undo(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C3)
    await memory._sync()  # recall has read the notes as they are
    assert memory._replaced(memory._corpus) == {}
    report = await memory.sleep("rules")
    assert [c.op for c in report.changes] == ["O2b"]
    await memory._sync()
    assert list(memory._replaced(memory._corpus)) == [("o", 1)]
    assert await memory.undo_sleep(report.run_id) is not None
    await memory._sync()
    assert memory._replaced(memory._corpus) == {}
    assert await states(memory) == {older: "active", newer: "active"}


async def test_undo_restores_the_kept_notes_provenance_exactly(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    store = memory.stores()[0]
    entry = await store.entry(newer)
    assert entry is not None
    before = dict(entry.provenance)
    report = await tidy(memory)
    assert await kept_replaces(memory, newer) == [["o", 1]]
    assert await memory.undo_sleep(report.run_id) is not None
    after = await store.entry(newer)
    # The undo sticks: the pair it separated is the only thing it adds.
    assert after is not None
    assert after.provenance == {**before, "kept_apart": [older]}


async def test_a_kept_notes_earlier_replaces_stay_and_the_older_ones_join_them(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    store = memory.stores()[0]
    await store.set_provenance(newer, {"replaces": [["x", 5]]})
    # The older note replaced two turns itself, one of them the kept note's own.
    await store.set_provenance(older, {"replaces": [["n", 1], ["w", 2], ["x", 5]]})
    report = await tidy(memory)
    (change,) = report.changes
    expected = [["x", 5], ["w", 2], ["o", 1]]  # never ["n", 1]; no repeat
    assert await kept_replaces(memory, newer) == expected
    assert change.replaces_before == [["x", 5]] and change.replaces_after == expected
    assert await memory.undo_sleep(report.run_id) is not None
    assert await kept_replaces(memory, newer) == [["x", 5]]


async def test_a_kept_note_that_already_replaces_the_turn_is_not_touched(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    await memory.stores()[0].set_provenance(newer, {"replaces": [["o", 1]]})
    report = await tidy(memory)
    (change,) = report.changes
    assert change.op == "O2b" and change.replaces_after is None
    assert await kept_replaces(memory, newer) == [["o", 1]]
    assert await memory.undo_sleep(report.run_id) is not None
    assert await kept_replaces(memory, newer) == [["o", 1]]


async def test_a_merged_duplicate_adds_its_turn_to_the_kept_note(
    memory: ChatMemory,
) -> None:
    older = await rules.put(memory, "Your gym day is Monday.", AT, session="o")
    newer = await rules.put(memory, "Your gym day is Monday.", AT + 20, session="n")
    report = await tidy(memory)
    (change,) = report.changes
    assert (change.op, change.before_ids, change.after_id) == ("O1", [older], newer)
    assert await kept_replaces(memory, newer) == [["o", 1]]
    assert await memory.undo_sleep(report.run_id) is not None
    assert await kept_replaces(memory, newer) is None


async def test_notes_sleep_writes_replace_the_turns_of_the_notes_they_supersede(
    memory: ChatMemory,
) -> None:
    old = await rules.put(
        memory, "Your locker number at the pool is 31.", AT, session="o"
    )
    source = turn("n", "My locker number at the pool is 58.", AT + 20)
    report = await sleep.run(*memory.stores(), [source], variant="rules")
    (change,) = report.changes
    assert (change.op, change.before_ids) == ("O2", [old])
    assert await kept_replaces(memory, int(change.after_id or 0)) == [["o", 1]]


async def test_set_provenance_merges_removes_and_changes_nothing_else(
    memory: ChatMemory,
) -> None:
    note = await rules.put(
        memory, "Your gym day is Monday.", AT, session="o", vector=[1.0, 0.0]
    )
    gone = await rules.put(memory, "Your swim day is Friday.", AT, session="o")
    store = memory.stores()[0]
    await store.set_status(
        [gone], "superseded", keeper=None, admitted_by="user", reason="gone"
    )
    before = await store.entry(note)
    assert before is not None
    assert await store.set_provenance(note, {"replaces": [["x", 5]], "quote": "q"})
    mid = await store.entry(note)
    assert mid is not None and mid.provenance["replaces"] == [["x", 5]]
    assert mid.provenance["quote"] == "q" and mid.provenance["topic"] == "about"
    assert (mid.text, mid.status, mid.vector) == (
        before.text,
        before.status,
        before.vector,
    )
    assert await store.set_provenance(note, {"replaces": None})
    end = await store.entry(note)
    assert end is not None and "replaces" not in end.provenance
    # A superseded or unknown note has no provenance to change.
    assert not await store.set_provenance(gone, {"replaces": [["x", 5]]})
    assert not await store.set_provenance(999, {"replaces": [["x", 5]]})
    assert "replaces" not in (await store.entry(gone) or before).provenance
