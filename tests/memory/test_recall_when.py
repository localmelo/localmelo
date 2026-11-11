"""The recall block says when and who: dates a reader can order, newer values first.

A date has the year when it is not the year of "now" (``LOCALMELO_NOW`` pins it)
and the time of day when another item of the block is from the same day. Notes of
one subject with different values are listed newest first, the older marked and
kept. A turn reads as what the user said and what melo replied, so a past reply
is never taken for the user's own statement. Synthetic items only.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime
from pathlib import Path

import pytest

from localmelo.melo import clock
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.index import document_text
from localmelo.support.memory.recall import (
    BLOCK_CHARS,
    BLOCK_HEADER,
    OLDER,
    Corpus,
    Item,
    block,
    day,
    replaced_turns,
    snippet,
    times,
    turn_said,
)
from localmelo.support.sessions.store import NewSession, SessionStore


def at(year: int, month: int, mday: int, hour: int = 12, minute: int = 0) -> float:
    return datetime(year, month, mday, hour, minute).timestamp()


def note(key: int, text: str, when: float, chat: str = "chat A") -> Item:
    """A note the person's own words made: nothing from a page replaces it."""
    return Item(
        f"r:{key}",
        text,
        f"{chat}, {day(when)}",
        "s",
        key,
        when,
        quote=text,
        standing=records.OWN,
    )


def turn(key: int, typed: str, answer: str, when: float) -> Item:
    return Item(
        f"d:{key}",
        document_text(typed, answer),
        f"chat B, {day(when)}",
        "t",
        key,
        when,
    )


@pytest.fixture(autouse=True)
def real_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(clock.ENV, raising=False)


def test_a_day_has_the_year_only_when_it_is_not_the_year_of_now() -> None:
    may = at(2023, 5, 20, 13, 40)
    assert day(may) == "May 20"  # no "now": as it was
    assert day(may, at(2023, 11, 1)) == "May 20"
    assert day(may, at(2024, 1, 2)) == "May 20, 2023"
    assert day(may, at(2023, 11, 1), timed=True) == "May 20, 13:40"
    assert day(may, at(2026, 10, 31), timed=True) == "May 20, 2023, 13:40"
    assert day(at(2023, 5, 3, 9, 5), at(2023, 6, 1), timed=True) == "May 3, 09:05"


def test_a_2023_item_shows_no_year_under_a_2023_clock_and_does_under_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = note(1, "dentist is Okafor", at(2023, 5, 20))
    monkeypatch.setenv(clock.ENV, "2023-06-01T09:00:00")
    pinned, _ = block([item])
    assert "(chat A, May 20)" in pinned and "2023" not in pinned
    monkeypatch.delenv(clock.ENV)
    today, _ = block([item])
    assert "(chat A, May 20, 2023)" in today
    assert item.prompt.endswith("(chat A, May 20, 2023)")  # search results too
    # An item of the year before a pinned "now" has its year.
    monkeypatch.setenv(clock.ENV, "2024-01-02T09:00:00")
    assert "(chat A, May 20, 2023)" in block([item])[0]


def test_two_items_of_one_day_both_show_the_time() -> None:
    first = note(1, "weekly sync is on Wednesday.", at(2026, 10, 30, 9, 15))
    second = note(2, "weekly sync is on Friday.", at(2026, 10, 30, 14, 5))
    other = note(3, "dentist is Okafor.", at(2026, 10, 29, 14, 5))
    text, used = block([first, second, other])
    lines = text.split("\n")[1:]
    assert used == [first, second, other]
    assert lines[0].endswith("(chat A, Oct 30, 09:15)")
    assert lines[1].endswith("(chat A, Oct 30, 14:05)")
    assert lines[2].endswith("(chat A, Oct 29)")  # alone in its day: no time
    assert times([first, second, other]) == {"r:1", "r:2"}
    # A record and a turn of one day count together.
    mixed, _ = block([first, turn(4, "hi", "hello", at(2026, 10, 30, 20, 0))])
    assert "(chat A, Oct 30, 09:15)" in mixed and "(chat B, Oct 30, 20:00)" in mixed
    # Old and from one day: year and time both.
    old = [note(5, "a", at(2023, 5, 20, 8, 0)), note(6, "b", at(2023, 5, 20, 9, 0))]
    assert "(chat A, May 20, 2023, 08:00)" in block(old)[0]


def test_an_item_whose_same_day_partner_did_not_fit_shows_no_time() -> None:
    day_at = at(2026, 10, 30, 9, 0)
    small = note(1, "tea.", day_at)
    big = note(2, "x" * (BLOCK_CHARS - 150), day_at + 60)
    text, used = block([small, big])
    assert used == [small]
    assert "(chat A, Oct 30)" in text and "09:00" not in text


def test_the_dates_fit_in_the_block() -> None:
    items = [
        note(n, "plan " + "z" * 130, at(2023, 5, 20, 8, n)) for n in range(1, 6)
    ] + [turn(9, "q" * 300, "a" * 300, at(2023, 5, 20, 9, 0))]
    text, used = block(items)
    assert 0 < len(used) <= len(items)
    assert len(text) <= BLOCK_CHARS
    assert all("2023" in line for line in text.split("\n")[1:])


def test_an_item_whose_where_has_no_date_is_left_alone() -> None:
    item = Item("r:1", "Your name is Juno.", "chat A", "s", 1, 1.0, quote="Juno")
    other = Item("r:2", "Your cat is Tom.", "chat A", "s", 2, 2.0, quote="Tom")
    assert item.prompt == "the user said “Juno” (chat A)"
    lines = block([item, other])[0].split("\n")[1:]
    assert all(line.endswith(" (chat A)") for line in lines) and len(lines) == 2


SYNC_OLD = "Your weekly sync is on Wednesday."
SYNC_NEW = "Your weekly sync is on Friday."


def test_notes_of_one_subject_with_different_values_are_listed_newest_first() -> None:
    older = note(1, SYNC_OLD, at(2026, 10, 30, 9, 0))
    middle = note(2, "dentist is Okafor.", at(2026, 10, 1))
    newer = note(3, SYNC_NEW, at(2026, 10, 30, 14, 5))
    assert records.subject_key(SYNC_OLD) == records.subject_key(SYNC_NEW) != ""
    ordered = records.newer_values_first([older, middle, newer])
    assert [i.key for i in ordered] == ["r:3", "r:2", "r:1"]
    assert [i.older for i in ordered] == ["", "", OLDER]
    assert [i.older for i in (older, newer)] == ["", ""]  # the items stay as they were
    text, used = block(ordered)
    lines = text.split("\n")[1:]
    assert "Friday" in lines[0] and "Wednesday" in lines[2]
    assert lines[0].endswith("(chat A, Oct 30, 14:05)")
    assert lines[2].endswith(f"(chat A, Oct 30, 09:00) ({OLDER})")
    assert used == ordered  # nothing is dropped


def test_the_newer_note_comes_first_in_the_places_the_pair_had() -> None:
    older = note(1, SYNC_OLD, at(2026, 10, 30))
    other = note(2, "dentist is Okafor.", at(2026, 10, 29))
    newer = note(3, SYNC_NEW, at(2026, 10, 31))
    ordered = records.newer_values_first([other, older, newer])
    assert [i.key for i in ordered] == ["r:2", "r:3", "r:1"]
    assert ordered[2].older == OLDER and not ordered[1].older and not ordered[0].older


def test_the_same_value_twice_or_one_note_is_not_marked() -> None:
    first = note(1, SYNC_NEW, at(2026, 10, 29))
    again = note(2, SYNC_NEW, at(2026, 10, 30))
    assert [i.older for i in records.newer_values_first([first, again])] == ["", ""]
    # No different values: the order is the ranking's.
    assert [i.key for i in records.newer_values_first([first, again])] == [
        "r:1",
        "r:2",
    ]
    alone = note(3, "dentist is Okafor.", at(2026, 10, 30))
    assert records.newer_values_first([alone]) == [alone]
    assert records.newer_values_first([]) == []
    # Three values: only the newest is current; an equal one is not marked older.
    wed = note(4, SYNC_OLD, at(2026, 10, 27))
    ordered = records.newer_values_first([wed, first, again])
    assert [(i.key, i.older) for i in ordered] == [
        ("r:2", ""),
        ("r:1", ""),
        ("r:4", OLDER),
    ]


def test_a_tie_in_time_goes_to_the_later_note() -> None:
    when = at(2026, 10, 30, 9, 0)
    a, b = note(1, SYNC_OLD, when), note(2, SYNC_NEW, when)
    assert [i.key for i in records.newer_values_first([a, b])] == ["r:2", "r:1"]


BOB = "Your manager is Bob."
ALICE = "Your manager is Alice Wu."
CAROL = "Your manager is Carol Ng."


def source_note(key: int, text: str, when: float, how: str = "pointed") -> Item:
    """A note from a page (or a result): its ``how`` says so and it quotes nothing."""
    return dataclasses.replace(
        note(key, text, when),
        how=how,
        standing=records.TAKEN,
        quote="",
        where=f"from jane.example.com, {day(when)}",
    )


def test_a_newer_note_from_a_page_does_not_mark_the_persons_own_older() -> None:
    mine = note(1, BOB, at(2026, 10, 29))
    page = source_note(2, ALICE, at(2026, 10, 30))
    for items in ([mine, page], [page, mine]):
        ordered = records.newer_values_first(items)
        assert [i.key for i in ordered] == ["r:1", "r:2"]
        assert [i.older for i in ordered] == ["", ""]
    text, used = block(records.newer_values_first([page, mine]))
    assert OLDER not in text and len(used) == 2
    mine_line, page_line = text.split("\n")[1:]
    assert "the user said" in mine_line and BOB[:-1].split(" is ")[1] in mine_line
    assert "from jane.example.com" in page_line and "Alice Wu" in page_line


def test_a_newer_note_of_the_person_marks_an_older_one_from_a_page() -> None:
    page = source_note(1, ALICE, at(2026, 10, 29), "result")
    mine = note(2, BOB, at(2026, 10, 30))
    ordered = records.newer_values_first([page, mine])
    assert [(i.key, i.older) for i in ordered] == [("r:2", ""), ("r:1", OLDER)]


def test_the_persons_newest_note_is_current_beside_a_note_from_a_page() -> None:
    first = note(1, BOB, at(2026, 10, 27))
    page = source_note(2, ALICE, at(2026, 10, 28))
    second = note(3, CAROL, at(2026, 10, 29))
    ordered = records.newer_values_first([first, page, second])
    # The page's note is older than the person's newest one, which replaced it.
    assert [(i.key, i.older) for i in ordered] == [
        ("r:3", ""),
        ("r:1", OLDER),
        ("r:2", OLDER),
    ]
    newest = source_note(4, "Your manager is Dev Rao.", at(2026, 10, 30))
    ordered = records.newer_values_first([first, page, second, newest])
    assert [(i.key, i.older) for i in ordered] == [
        ("r:3", ""),
        ("r:1", OLDER),
        ("r:4", ""),
        ("r:2", OLDER),
    ]


def test_notes_of_one_kind_are_ordered_and_marked_as_before() -> None:
    first, second = note(1, BOB, at(2026, 10, 27)), note(2, CAROL, at(2026, 10, 28))
    ordered = records.newer_values_first([first, second])
    assert [(i.key, i.older) for i in ordered] == [("r:2", ""), ("r:1", OLDER)]
    pages = [
        source_note(3, BOB, at(2026, 10, 27)),
        source_note(4, ALICE, at(2026, 10, 28)),
    ]
    ordered = records.newer_values_first(pages)
    assert [(i.key, i.older) for i in ordered] == [("r:4", ""), ("r:3", OLDER)]


def reworded_note(key: int, text: str, when: float) -> Item:
    """A note from a page that the person edited by hand."""
    return dataclasses.replace(source_note(key, text, when), standing=records.REWORDED)


def test_a_reworded_note_is_listed_first_whatever_its_source_label() -> None:
    """A note from a page that the person reworded is kept beside a newer page note:
    ``standing``, not ``how``, tells recall."""
    edited = reworded_note(1, BOB, at(2026, 10, 27))
    newer = source_note(2, ALICE, at(2026, 10, 28))
    for items in ([edited, newer], [newer, edited]):
        ordered = records.newer_values_first(items)
        assert [(i.key, i.older) for i in ordered] == [("r:1", ""), ("r:2", "")]
    # A note with no source label that stands no higher is replaced like a page's.
    plain = dataclasses.replace(note(3, BOB, at(2026, 10, 27)), standing=records.TAKEN)
    ordered = records.newer_values_first([plain, newer])
    assert [(i.key, i.older) for i in ordered] == [("r:2", ""), ("r:3", OLDER)]


def test_only_a_note_of_the_persons_own_marks_a_turn_older() -> None:
    corpus = Corpus()
    said = (("chat-a", 1),)
    for key, make in enumerate((note, reworded_note, source_note), 1):
        item = make(key, BOB, at(2026, 10, 27))
        corpus.add(dataclasses.replace(item, replaces=said, session_id=f"s{key}"))
    assert {turn: item.key for turn, item in replaced_turns(corpus).items()} == {
        ("chat-a", 1): "r:1"
    }
    corpus.remove("r:1")
    assert replaced_turns(corpus) == {}


def test_a_reworded_note_never_marks_the_persons_own_older() -> None:
    mine = note(1, BOB, at(2026, 10, 27))
    edited = reworded_note(2, ALICE, at(2026, 10, 28))
    for items in ([mine, edited], [edited, mine]):
        ordered = records.newer_values_first(items)
        assert [(i.key, i.older) for i in ordered] == [("r:1", ""), ("r:2", "")]
    # The person's own newer note marks it, and an older page note under it.
    page = source_note(3, CAROL, at(2026, 10, 26))
    later = note(4, "Your manager is Dev Rao.", at(2026, 10, 29))
    ordered = records.newer_values_first([page, edited, later])
    assert [(i.key, i.older) for i in ordered] == [
        ("r:4", ""),
        ("r:2", OLDER),
        ("r:3", OLDER),
    ]
    # With no note of the person's own, the reworded one marks the older page note.
    ordered = records.newer_values_first([page, edited])
    assert [(i.key, i.older) for i in ordered] == [("r:2", ""), ("r:3", OLDER)]


def test_a_turn_says_who_said_what() -> None:
    item = turn(1, "My dentist is Okafor.", "Noted, Okafor it is.", at(2026, 10, 30))
    assert item.prompt == (
        "the user said “My dentist is Okafor.”; melo replied “Noted, Okafor it is.” "
        "(chat B, Oct 30)"
    )
    assert turn_said("Task: Plan\nUser: hi\nmelo: hello") == (
        "the user said “hi”; melo replied “hello”"
    )
    # A reply of melo's that holds an old value is melo's, not the user's.
    old = turn(2, "Where is my dentist?", "Your dentist was Park.", at(2026, 10, 1))
    assert "the user said “Where is my dentist?”; melo replied “Your dentist" in (
        old.prompt
    )
    assert "User:" not in old.prompt and "melo:" not in old.prompt


def test_a_turns_parts_are_bounded_and_the_users_words_stay_whole_when_they_fit() -> (
    None
):
    typed = "t" * 120
    said = turn_said(document_text(typed, "r" * 900))
    user, reply = said.split("; melo replied ")
    assert user == f"the user said “{typed}”"
    assert reply.endswith("…”") and len(reply) >= 60
    assert len(said) <= 234 + len("the user said “”; melo replied “”")
    # Long words from the user show as far as the plain 240-character head did,
    # which leaves no room for the reply.
    said = turn_said(document_text("u" * 900, "r" * 900))
    assert said == "the user said “" + "u" * 233 + "…”"
    # Everything fits: both whole. No reply: the user's words alone.
    assert turn_said(document_text("a" * 90, "b" * 90)).count("…") == 0
    assert turn_said("User: question\nmelo: ") == "the user said “question”"
    assert turn_said("User: " + "q" * 600 + "\nmelo: ").endswith("…”")


def test_text_that_is_no_turn_reads_as_it_did() -> None:
    summary = "Summary of chat “Fix CI”: the fixture port was taken."
    asked = "Task asked: which folder? Answer: the repo root."
    flat = "User: a melo: b"
    assert turn_said(summary) == snippet(summary)
    assert turn_said(asked) == snippet(asked)
    assert turn_said(flat) == snippet(flat)


def test_the_page_keeps_the_turns_text_as_it_was() -> None:
    item = turn(1, "My dentist is Okafor.", "Noted.", at(2026, 10, 30))
    data = item.to_json()
    assert data["text"] == "User: My dentist is Okafor. melo: Noted."
    assert data["source"] == "chat B, Oct 30"
    older = note(2, SYNC_OLD, at(2026, 10, 30))
    assert older.to_json()["source"] == "you said it · chat A, Oct 30"


def test_a_block_of_turns_fits_with_both_parts() -> None:
    items = [turn(n, "w" * 150, "x" * 600, at(2026, 10, n)) for n in range(1, 4)] + [
        note(9, "dentist is Okafor.", at(2026, 10, 9))
    ]
    text, used = block(items)
    assert len(text) <= BLOCK_CHARS
    assert used and all(
        ("the user said" in line and "melo replied" in line) or "a note" in line
        for line in text.split("\n")[1:]
        if "dentist" not in line
    )


@pytest.fixture()
async def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


async def put(memory: ChatMemory, text: str, when: float) -> None:
    """A note kept as it is, with no supersede, at *when* (two of one subject)."""
    ids, _ = memory.stores()
    await ids.apply(
        text=text,
        kind=records.TOPIC_KINDS["about"],
        source="user",
        admitted_by="user_statement",
        provenance={"session_id": "old", "turn": 1, "topic": "about", "quote": ""},
        task_id="",
        session_id="old",
        created_at=when,
        embedding=None,
        action="NEW",
        target_id=None,
        reason="seeded",
        source_ids=(),
        keep_text=False,
        candidate_id=text,
    )


async def test_the_block_a_message_gets_orders_two_values_of_one_day(
    tmp_path: Path, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Noon, so an hour earlier is the same local day whenever the test runs.
    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    monkeypatch.setenv(clock.ENV, noon.isoformat())
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        now = noon.timestamp()
        await put(memory, SYNC_OLD, now - 3_600)
        await put(memory, SYNC_NEW, now - 60)
        question = "Which day is my weekly sync?"
        info = await store.begin_turn(
            None,
            question,
            holder="test-host:1:when",
            title=question,
            new_session=NewSession(),
        )
        found = await memory.before_turn(store, info, question)
        lines = found.block.split("\n")[3:]
        assert len(lines) == 2 and len(found.used) == 2
        assert "Friday" in lines[0] and "Wednesday" in lines[1]
        assert lines[1].endswith(f"({OLDER})") and OLDER not in lines[0]
        assert all(
            re.search(r", \d\d:\d\d\)", line.split(" (older")[0]) for line in lines
        )
        assert [i.older for i in found.used] == ["", OLDER]
        assert BLOCK_HEADER in found.block and len(found.block) <= BLOCK_CHARS + 2
    finally:
        await memory.aclose()
