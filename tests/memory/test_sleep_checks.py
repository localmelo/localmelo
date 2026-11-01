"""Sleep keeps paraphrases out, retires a note a newer one replaced and never pins.

Stub embeddings return cosines measured with a real embedding model, so the
guard is checked on the numbers that decide it. No model, no server.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from localmelo.melo.memory.long import StoredEntry
from localmelo.support.memory import ChatMemory, records, sleep
from localmelo.support.memory.recall import current_only, newer_first

from . import test_sleep as rules
from .test_sleep_model import Scripted, op

memory = rules.memory
chats = rules.chats


@pytest.fixture()
def model_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch the ``O2b`` rule off, so a test sees what the model's proposal does."""
    monkeypatch.setattr(sleep, "replacements", lambda notes, sources: [])


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


async def run_model(
    memory: ChatMemory, turns: list[sleep.SleepTurn], replies: list[Any]
) -> sleep.SleepReport:
    """A model run whose candidate embeds close to the newer note ``[1, 0]``."""
    return await sleep.run(
        *memory.stores(),
        turns,
        variant="model",
        provider=Scripted(replies),
        embed_note=embedding(0.99),
    )


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


@pytest.mark.parametrize("variant", ["rules", "model", "model+tags"])
async def test_paraphrase_at_the_measured_cosine_is_held_and_logged(
    memory: ChatMemory, variant: str
) -> None:
    old = await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    source = turn("s", "My usual train to the office is the 7:40.", AT + 10)
    report = await sleep.run(
        *memory.stores(),
        [source],
        variant=variant,
        provider=Scripted([[op("ADD", source.typed, PARAPHRASE)]]),
        embed_note=embedding(0.878),
    )
    assert not report.changes
    assert [(n.id, n.text) for n in await memory.records()] == [(old, TRAIN)]
    (skipped,) = report.to_json()["skipped"]  # the rule skipped it, with its numbers
    assert (skipped["session_id"], skipped["turn"]) == ("s", 1)
    assert skipped["text"] == PARAPHRASE
    assert skipped["guard"]["held"] is True and skipped["guard"]["cosine"] == 0.878
    assert skipped["guard"]["coverage"] and skipped["guard"]["polarity_same"]
    if variant != "rules":
        (proposal,) = report.to_json()["proposals"]
        assert proposal["verdict"] == "refused:decide_mismatch"
        assert proposal["guard"] == skipped["guard"]  # the model ADD, same numbers


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


async def test_a_model_written_sleep_note_is_not_pinned_either(
    memory: ChatMemory,
) -> None:
    source = turn("b", "I'm allergic to peanuts.", AT + 10)
    report = await run_model(
        memory, [source], [[op("ADD", source.typed, "You are allergic to peanuts.")]]
    )
    assert [c.op for c in report.changes] == ["ADD"]
    assert await memory.pinned_quotes() == ""


async def test_the_model_picks_the_topic_as_the_chat_does(memory: ChatMemory) -> None:
    source = turn("s", "My flight to Faro leaves from gate 9.", AT + 10)
    proposal = op("ADD", source.typed, "Your flight to Faro leaves from gate 9.")
    proposal["topic"] = "projects"
    await run_model(memory, [source], [[proposal]])
    (note,) = await memory.records()
    assert note.provenance["topic"] == "projects" and note.kind == "episodic"


@pytest.mark.parametrize("topic", [None, "trips", 7, ["about"]])
async def test_no_or_an_unknown_topic_falls_back_to_the_chat_rule(
    memory: ChatMemory, topic: Any
) -> None:
    source = turn("s", "I enjoy kitesurfing.", AT + 10)
    proposal = op("ADD", source.typed, "You enjoy kitesurfing.")
    proposal["topic"] = topic
    await run_model(memory, [source], [[proposal]])
    (note,) = await memory.records()
    assert note.provenance["topic"] == records.topic_of(note.text, request=False)


async def test_a_pronoun_rewrite_keeps_the_topic_of_the_note(
    memory: ChatMemory,
) -> None:
    source = turn("s", "Dr. Ada's office is on Pine Street.", AT + 10)
    old = await rules.put(
        memory, "Her office is on Pine Street.", AT, session="s", topic="projects"
    )
    await run_model(
        memory,
        [source],
        [[], [op("UPDATE", source.typed, source.typed, old)]],
    )
    (note,) = await memory.records()
    assert note.text == source.typed and note.provenance["topic"] == "projects"


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


# What a refused update of the older note or of the newer one comes back as.
REFUSED = {"older": "refused:decide_mismatch", "newer": "refused:self_target"}
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


@pytest.mark.parametrize("target", ["older", "newer"])
@pytest.mark.parametrize(
    ("old", "new", "text", "topics"),
    [
        (*C3, "你的牙医预约改到了周四。", ("projects", "decisions")),
        (*C4, "Your new manager is Daniel Kim from next week.", ("people", "people")),
        (
            ("Your gym day is Monday.", "My gym day is Monday."),
            ("Your gym day moved to Friday.", "my gym day moved to Friday"),
            "Your gym day moved to Friday.",
            ("preferences", "preferences"),
        ),
    ],
)
async def test_model_update_retires_the_older_note_a_newer_active_one_holds(
    memory: ChatMemory,
    model_only: None,
    old: tuple[str, str],
    new: tuple[str, str],
    text: str,
    topics: tuple[str, str],
    target: str,
) -> None:
    older, newer = await pair(memory, old, new, topics=topics)
    source = turn("n", new[1], AT + 20)
    named = older if target == "older" else newer
    report = await run_model(memory, [source], [[op("UPDATE", new[1], text, named)]])
    (change,) = report.changes
    assert (change.op, change.before_ids, change.after_id) == ("O2b", [older], newer)
    assert change.before_texts == [old[0]] and change.quote == new[1]
    assert change.verdict == "applied" and change.proposal is not None
    assert change.proposal["target_text"] in (old[0], new[0])
    assert await states(memory) == {older: "superseded", newer: "active"}
    assert await rules.texts(memory) == {new[0]}  # nothing is written
    assert not report.refused


async def test_o2b_is_undone_as_one_run_and_never_removes_the_kept_note(
    memory: ChatMemory, model_only: None
) -> None:
    older, newer = await pair(memory, *C4)
    source = turn("n", C4[1][1], AT + 20)
    report = await run_model(
        memory,
        [source],
        [
            [
                op(
                    "UPDATE",
                    C4[1][1],
                    "Your new manager is Daniel Kim from next week.",
                    older,
                )
            ]
        ],
    )
    before = rules.rows(memory)
    assert await states(memory) == {older: "superseded", newer: "active"}
    undone = await memory.undo_sleep(report.run_id)
    assert undone == {"run_id": report.run_id, "restored": 1, "removed": 0, "kept": 0}
    assert await states(memory) == {older: "active", newer: "active"}
    assert [r[:2] for r in rules.rows(memory)] == [r[:2] for r in before]


async def test_undo_leaves_the_older_note_when_a_later_note_replaced_the_kept_one(
    memory: ChatMemory, model_only: None
) -> None:
    older, newer = await pair(memory, *C4)
    source = turn("n", C4[1][1], AT + 20)
    report = await run_model(
        memory,
        [source],
        [
            [
                op(
                    "UPDATE",
                    C4[1][1],
                    "Your new manager is Daniel Kim from next week.",
                    older,
                )
            ]
        ],
    )
    await memory.stores()[0].set_status(
        [newer], "superseded", keeper=None, admitted_by="user", reason="later"
    )
    undone = await memory.undo_sleep(report.run_id)
    assert undone is not None and undone["kept"] == 1 and undone["restored"] == 0
    assert await states(memory) == {older: "superseded", newer: "superseded"}


async def test_the_twenty_change_cap_still_holds_for_the_models_o2b(
    memory: ChatMemory, model_only: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sleep, "MAX_CHANGES", 1)
    first = await pair(memory, *C4)
    second = await pair(memory, *C3)
    source = turn("n", C4[1][1] + " " + C3[1][1], AT + 20)
    report = await run_model(
        memory,
        [source],
        [
            [
                op(
                    "UPDATE",
                    C4[1][1],
                    "Your new manager is Daniel Kim from next week.",
                    first[0],
                ),
                op("UPDATE", C3[1][1], C3[1][0], second[0]),
            ]
        ],
    )
    assert report.capped and len(report.changes) == 1
    assert [r["verdict"] for r in report.proposals] == ["applied", "noop"]
    assert (await states(memory))[second[0]] == "active"


# Controls: nothing is retired.


async def test_a_refused_update_of_a_note_from_its_own_turn_changes_nothing(
    memory: ChatMemory,
) -> None:
    """K3 in the order a bench seeds it: the model targets each note from its own turn."""
    older, newer = await pair(
        memory,
        ("You live in Lisbon for the summer.", "I live in Lisbon for the summer."),
        ("You no longer live in Lisbon.", "I no longer live in Lisbon."),
    )
    first = turn("o", "I live in Lisbon for the summer.", AT)
    second = turn("n", "I no longer live in Lisbon.", AT + 20)
    report = await run_model(
        memory,
        [first, second],
        [
            [op("UPDATE", first.typed, "You live in Lisbon for the summer.", older)],
            [op("RETIRE", second.typed, note=newer)],
        ],
    )
    assert not report.changes
    assert [r["verdict"] for r in report.proposals] == [
        "refused:self_target",
        "refused:self_target",
    ]
    assert await states(memory) == {older: "active", newer: "active"}


@pytest.mark.parametrize("target", ["older", "newer"])
async def test_lisbon_scope_change_is_left_whatever_the_model_targets(
    memory: ChatMemory, target: str
) -> None:
    older, newer = await pair(
        memory,
        ("You live in Lisbon for the summer.", "I live in Lisbon for the summer."),
        ("You no longer live in Lisbon.", "I no longer live in Lisbon."),
    )
    source = turn("n", "I no longer live in Lisbon.", AT + 20)
    named = older if target == "older" else newer
    report = await run_model(
        memory,
        [source],
        [[op("UPDATE", source.typed, "You no longer live in Lisbon.", named)]],
    )
    assert not report.changes
    assert [r["verdict"] for r in report.proposals] == [REFUSED[target]]
    assert await states(memory) == {older: "active", newer: "active"}


@pytest.mark.parametrize("target", ["older", "newer"])
@pytest.mark.parametrize(
    ("old", "new", "text"),
    [
        (
            ("Your gym day is Monday.", "My gym day is Monday."),
            ("Your swim day is Thursday.", "My swim day is Thursday."),
            "Your swim day is Thursday.",
        ),
        (
            ("你的钢琴课在周三。", "我的钢琴课在周三。"),
            ("你的舞蹈课在周五。", "我的舞蹈课在周五。"),
            "你的舞蹈课在周五。",
        ),
        (
            ("Your piano class is on Wednesday.", "My piano class is on Wednesday."),
            ("Your dance class is on Friday.", "My dance class is on Friday."),
            "Your dance class is on Friday.",
        ),
    ],
)
async def test_other_subjects_are_never_retired(
    memory: ChatMemory,
    old: tuple[str, str],
    new: tuple[str, str],
    text: str,
    target: str,
) -> None:
    older, newer = await pair(memory, old, new)
    source = turn("n", new[1], AT + 20)
    named = older if target == "older" else newer
    report = await run_model(memory, [source], [[op("UPDATE", new[1], text, named)]])
    assert not report.changes
    assert [r["verdict"] for r in report.proposals] == [REFUSED[target]]
    assert await states(memory) == {older: "active", newer: "active"}


@pytest.mark.parametrize("edited", ["old", "new"])
@pytest.mark.parametrize("target", ["older", "newer"])
async def test_a_note_the_user_edited_is_never_retired_or_kept(
    memory: ChatMemory, edited: str, target: str
) -> None:
    older, newer = await pair(memory, *C4, edited=edited)
    source = turn("n", C4[1][1], AT + 20)
    named = older if target == "older" else newer
    report = await run_model(
        memory,
        [source],
        [
            [
                op(
                    "UPDATE",
                    C4[1][1],
                    "Your new manager is Daniel Kim from next week.",
                    named,
                )
            ]
        ],
    )
    assert not report.changes
    expected = {
        ("old", "older"): "refused:edited",  # the user's note, named as the target
        ("old", "newer"): "refused:self_target",
        ("new", "older"): "refused:decide_mismatch",  # the kept note was reworded
        ("new", "newer"): "refused:self_target",
    }
    assert [r["verdict"] for r in report.proposals] == [expected[edited, target]]
    assert await states(memory) == {older: "active", newer: "active"}


async def test_an_older_value_never_retires_a_newer_note(
    memory: ChatMemory, model_only: None
) -> None:
    older, newer = await pair(memory, *C4)
    source = turn("o", C4[0][1], AT)
    report = await run_model(
        memory,
        [source],
        [[op("UPDATE", C4[0][1], "Your manager is Priya Shah.", newer)]],
    )
    assert not report.changes
    assert [r["verdict"] for r in report.proposals] == ["refused:not_newer"]
    assert await states(memory) == {older: "active", newer: "active"}


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


# ── Fix 4: a refusal says what it targeted ──


async def test_every_proposal_names_the_text_of_its_target(memory: ChatMemory) -> None:
    secret = await rules.put(memory, "My password is hunter2.")
    long = await rules.put(memory, "Your notes are " + "very " * 80 + "long.")
    plain = await rules.put(memory, "You are allergic to peanuts.")
    source = turn("s", "I'm allergic to peanuts.", AT + 10)
    report = await run_model(
        memory,
        [source],
        [
            [
                op("NOOP", source.typed, note=plain),
                op("UPDATE", "invented", "You are allergic to peanuts.", plain),
                op("UPDATE", source.typed, "x", secret),
                op("RETIRE", source.typed, note=long),
                op("RETIRE", source.typed, note=9999),
                op("ADD", source.typed, "You enjoy kitesurfing."),
            ]
        ],
    )
    rows = report.to_json()["proposals"]
    assert [r["target_text"] for r in rows][:2] == ["You are allergic to peanuts."] * 2
    assert rows[2]["target_text"] == "[redacted]" and "hunter2" not in str(rows)
    assert rows[3]["target_text"].startswith("Your notes are very")
    assert len(rows[3]["target_text"]) == 200
    assert rows[4]["target_text"] == "" and rows[5]["target_text"] == ""


async def test_a_note_of_the_proposals_own_turn_is_refused_as_self_target(
    memory: ChatMemory,
) -> None:
    """The model retired the negation note written from the same turn."""
    older, newer = await pair(
        memory,
        ("You live in Lisbon for the summer.", "I live in Lisbon for the summer."),
        ("You no longer live in Lisbon.", "I no longer live in Lisbon."),
    )
    source = turn("n", "I no longer live in Lisbon.", AT + 20)
    report = await run_model(
        memory, [source], [[op("RETIRE", source.typed, note=newer)]]
    )
    (row,) = report.to_json()["proposals"]
    assert row["verdict"] == "refused:self_target"
    assert (
        row["note_id"] == newer
        and row["target_text"] == "You no longer live in Lisbon."
    )
    assert report.refused == 1 and not report.changes
    assert await states(memory) == {older: "active", newer: "active"}


async def test_a_note_listed_by_the_turn_is_its_own_too(memory: ChatMemory) -> None:
    older, newer = await pair(memory, *C4)
    source = sleep.SleepTurn(
        "elsewhere",
        1,
        AT + 20,
        AT + 21,
        C4[1][1],
        remembered=(records.Written(newer, C4[1][0], "people"),),
    )
    notes = {n.id: n for n in await memory.records()}
    assert sleep._written_from(notes[newer], source)
    assert not sleep._written_from(notes[older], source)


async def test_the_pronoun_rewrite_targets_its_own_turn_by_design(
    memory: ChatMemory,
) -> None:
    source = turn("s", "Dr. Ada's office is on Pine Street.", AT + 10)
    old = await rules.put(memory, "Her office is on Pine Street.", AT, session="s")
    report = await run_model(
        memory, [source], [[], [op("UPDATE", source.typed, source.typed, old)]]
    )
    assert [c.op for c in report.changes] == ["O4"]
    assert report.to_json()["proposals"][-1]["verdict"] == "applied"


async def test_the_prompt_lists_notes_with_ids_and_dates_and_names_the_older_note(
    memory: ChatMemory,
) -> None:
    note = await rules.put(memory, "You are allergic to peanuts.", AT)
    source = turn("s", "I'm allergic to cashews.", AT + 10)
    provider = Scripted([[]])
    await sleep.run(*memory.stores(), [source], variant="model", provider=provider)
    system, user = provider.calls[0]
    payload = json.loads(user.content or "")
    assert payload["notes"] == [
        f"#{note} ({sleep._day(AT)}) You are allergic to peanuts."
    ]
    assert payload["date"] == sleep._day(source.started_at)
    prompt = system.content or ""
    assert '"#id (date) text"' in prompt
    assert "older note" in prompt and "never a note written from that same" in prompt


# ── The rule O2b: no model call ──


async def tidy(memory: ChatMemory) -> sleep.SleepReport:
    """A rules run over the notes as they are (no chat turn, no model)."""
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
    assert change.proposal is None and report.model_calls == 0
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


async def test_a_run_that_retired_a_note_makes_the_models_proposals_noops(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, *C4)
    source = turn("n", C4[1][1], AT + 20)
    text = "Your new manager is Daniel Kim from next week."
    report = await run_model(
        memory,
        [source],
        [[op("UPDATE", C4[1][1], text, older), op("UPDATE", C4[1][1], text, newer)]],
    )
    (change,) = report.changes  # the rule's, counted once
    assert (change.op, change.before_ids, change.after_id) == ("O2b", [older], newer)
    assert change.proposal is None
    assert [r["verdict"] for r in report.proposals] == ["noop", "noop"]
    assert report.refused == 0 and not report.capped
    assert await states(memory) == {older: "superseded", newer: "active"}
    # One undo of the run puts the older note back.
    undone = await memory.undo_sleep(report.run_id)
    assert undone is not None and undone["restored"] == 1
    assert await states(memory) == {older: "active", newer: "active"}


# ── A retired value does not come back through its exchange ──


async def kept_replaces(memory: ChatMemory, note: int) -> Any:
    entry = await memory.stores()[0].entry(note)
    assert entry is not None
    return entry.provenance.get("replaces")


@pytest.mark.parametrize("path", ["rule", "model"])
@pytest.mark.parametrize("shape", [C3, C4])
async def test_o2b_adds_the_older_turn_to_the_kept_notes_replaces(
    memory: ChatMemory,
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[tuple[str, str], tuple[str, str]],
    path: str,
) -> None:
    older, newer = await pair(memory, *shape)
    assert await kept_replaces(memory, newer) is None
    if path == "rule":
        report = await tidy(memory)
    else:
        monkeypatch.setattr(sleep, "replacements", lambda notes, sources: [])
        text = "Your new manager is Daniel Kim from next week."
        proposal = op("UPDATE", shape[1][1], text, older)
        if shape is C3:
            proposal["text"] = "你的牙医预约改到了周四。"
        report = await run_model(
            memory, [turn("n", shape[1][1], AT + 20)], [[proposal]]
        )
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
    old = await rules.put(memory, "Your gym day is Monday.", AT, session="o")
    source = turn("n", "My gym day moved to Friday.", AT + 20)
    text = "Your gym day moved to Friday."
    report = await run_model(
        memory, [source], [[op("UPDATE", source.typed, text, old)]]
    )
    (change,) = report.changes
    assert (change.op, change.before_ids) == ("UPDATE", [old])
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


# ── An update a newer note already holds is not written again ──

LOCKER_OLD = (
    "Your locker number at the pool is 31.",
    "My locker number at the pool is 31.",
)
LOCKER_NEW = (
    "The pool gave me a new locker, number 58.",
    "The pool gave me a new locker, number 58.",
)
LOCKER_TEXT = "Your locker number at the pool is 58."


async def test_an_update_whose_value_a_newer_note_holds_writes_no_copy(
    memory: ChatMemory,
) -> None:
    older, newer = await pair(memory, LOCKER_OLD, LOCKER_NEW)
    source = turn("n", LOCKER_NEW[1], AT + 20)
    report = await run_model(
        memory, [source], [[op("UPDATE", source.typed, LOCKER_TEXT, older)]]
    )
    (change,) = report.changes
    assert (change.op, change.before_ids, change.after_id) == ("O2b", [older], newer)
    assert change.verdict == "applied" and change.replaces_after == [["o", 1]]
    assert await states(memory) == {older: "superseded", newer: "active"}
    assert await rules.texts(memory) == {LOCKER_NEW[0]}  # no #3: nothing is written
    assert len(rules.rows(memory)) == 2
    (proposal,) = report.proposals
    assert proposal["verdict"] == "applied" and proposal["guard"]["note_id"] == newer
    assert await kept_replaces(memory, newer) == [["o", 1]]
    undone = await memory.undo_sleep(report.run_id)
    assert undone == {"run_id": report.run_id, "restored": 1, "removed": 0, "kept": 0}
    assert await states(memory) == {older: "active", newer: "active"}
    assert await kept_replaces(memory, newer) is None


@pytest.mark.parametrize(
    ("holder", "typed", "edited"),
    [
        # Another value: nothing holds 58, so the update is written.
        (("The pool gave me a new locker, number 62.",) * 2, LOCKER_NEW[1], ""),
        # The opposite statement is no holder.
        (
            ("The pool never gave me locker number 58.",) * 2,
            "The pool gave me a new locker, number 58.",
            "",
        ),
        # The user's own wording is left alone.
        (LOCKER_NEW, LOCKER_NEW[1], "new"),
        # A note that holds the value of another subject (a "day" in common is
        # no subject) is no holder: the swim day is not the gym day.
        (
            ("Your swim day is Thursday.", "My swim day is Thursday."),
            "My gym day is Thursday.",
            "",
        ),
    ],
)
async def test_an_update_is_still_written_when_no_note_holds_what_it_says(
    memory: ChatMemory,
    monkeypatch: pytest.MonkeyPatch,
    holder: tuple[str, str],
    typed: str,
    edited: str,
) -> None:
    # The rules would write this statement themselves; the model's proposal is on trial.
    monkeypatch.setattr(sleep, "recoverable", lambda fresh, notes: [])
    gym = holder[0].startswith("Your swim")
    old = ("Your gym day is Monday.", "My gym day is Monday.") if gym else LOCKER_OLD
    text = "Your gym day is Thursday." if gym else LOCKER_TEXT
    older, newer = await pair(memory, old, holder, edited=edited)
    source = turn("m", typed, AT + 30, number=2)
    report = await run_model(memory, [source], [[op("UPDATE", typed, text, older)]])
    (change,) = report.changes
    assert (change.op, change.before_ids) == ("UPDATE", [older])
    assert change.after_id not in (older, newer)
    assert await rules.texts(memory) == {holder[0], text}
    assert (await states(memory))[newer] == "active"
    assert (await states(memory))[older] == "superseded"


# ── A model note keeps the person and the assistant apart ──


@pytest.mark.parametrize(
    "text",
    [
        "You prefer that when you explain a technical topic to me you choose plain words.",
        "You like my plain words.",
        "I prefer plain words.",
        "You told me to use plain words.",
        "你喜欢我用简单的话解释。",
        "你说我的牙医预约在周四。",
    ],
)
@pytest.mark.parametrize("op_name", ["ADD", "UPDATE"])
async def test_a_first_person_model_note_is_refused(
    memory: ChatMemory, text: str, op_name: str
) -> None:
    old = await rules.put(memory, "Your gym day is Monday.", AT, session="o")
    source = turn("n", text, AT + 20)
    proposal = op(op_name, source.typed, text, old if op_name == "UPDATE" else None)
    report = await run_model(memory, [source], [[proposal]])
    assert not report.changes and report.refused == 1
    assert [r["verdict"] for r in report.proposals] == ["refused:first_person"]
    assert await states(memory) == {old: "active"}


@pytest.mark.parametrize(
    "text",
    [
        "You prefer that melo explains technical topics in plain words.",
        'Your favorite line is "I think therefore I am".',
        "你喜欢「我思故我在」这句话。",
    ],
)
async def test_second_person_and_quoted_first_person_model_notes_are_kept(
    memory: ChatMemory, text: str
) -> None:
    typed = (
        text.replace("You prefer", "I prefer").replace("Your", "My").replace("你", "我")
    )
    source = turn("n", typed, AT + 20)
    report = await run_model(memory, [source], [[op("ADD", typed, text)]])
    assert [r["verdict"] for r in report.proposals] == ["applied"]
    assert await rules.texts(memory) == {text}


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("I live in Lisbon.", True),
        ("You said I'm late.", True),
        ("Tell me.", True),
        ("MY gym day", True),
        ("The shop is mine.", True),
        ("You did it for myself.", True),
        ("我住在里斯本。", True),
        ('Your line is "I think".', True),  # the checker strips quotes first
        ("Mimi lives in Maine.", False),
        ("You like mythology and Amy's hi-fi.", False),
        ("Your ill neighbour Imran.", False),
        ("You live in Lisbon.", False),
        ("你住在里斯本。", False),
    ],
)
def test_first_person_pattern_finds_whole_words_only(text: str, found: bool) -> None:
    assert (sleep._FIRST_PERSON.search(text) is not None) is found


def test_the_prompt_asks_for_you_and_melo() -> None:
    prompt = sleep._MODEL_PROMPT
    assert 'Write "you" for the user and "melo" for the assistant' in prompt
    assert "You prefer that melo explains technical topics in plain words" in prompt
