"""The recall block gives the current value only, and notes say who they are about.

A turn whose fact a newer note or message replaced stays out of the block unless
the message asks about the past (English and Chinese); the Memory tab and
``search_memory`` still show it. Every text that shows notes to the model agrees
that a note is written to the user, so "you" in it is the user.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.support.memory import records
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    OLDER,
    OLDER_ITEM,
    OLDER_TURN,
    Item,
    asks_about_past,
    block,
    current_only,
    gate_terms,
    newer_first,
)
from localmelo.support.memory.service import REMEMBER, SEARCH_MEMORY
from localmelo.support.sessions.identity import NOTES_TEXT, PINNED_TEXT


def _stale() -> tuple[list[Item], list[Item]]:
    """A note and two turns an earlier one replaced: (all items, the current ones)."""
    note = Item("r:3", "You live in Utrecht.", "chat C", "c", 1, 3.0)
    note.replaces = (("a", 1),)
    old = Item("d:1", "User: I live in Lisbon.", "chat A", "a", 1, 1.0)
    older = Item("d:2", "User: I live in Porto.", "chat B", "b", 1, 2.0)
    other = Item("d:4", "User: I like tea.", "chat D", "d", 1, 4.0)
    replaced = {("a", 1): note, ("b", 1): note}
    ordered = newer_first([old, older, other], replaced)
    assert [i.key for i in ordered] == ["r:3", "d:4", "d:1", "d:2"]
    assert [i.older for i in ordered] == ["", "", OLDER, OLDER]
    return ordered, ordered[:2]


def test_an_item_a_newer_one_replaced_stays_out_of_the_block() -> None:
    ordered, current = _stale()
    kept = current_only(ordered, "Where do I live?")
    assert kept == current
    text, used = block(kept)
    assert used == current and "Lisbon" not in text and "Porto" not in text
    assert "You live in Utrecht." in text
    # search_memory reads the items, not the block: it still names the older ones.
    assert [i.older for i in ordered[2:]] == [OLDER, OLDER]
    assert ordered[2].prompt.endswith(f"(chat A) ({OLDER})")


def test_a_turn_a_newer_turn_replaced_stays_out_too() -> None:
    new = Item("d:2", "User: I live in Utrecht.", "chat B", "b", 2, 2.0)
    old = Item("d:1", "User: I live in Lisbon.", "chat A", "a", 1, 1.0)
    ordered = newer_first([old], {("a", 1): new})
    assert ordered[-1].older == OLDER_TURN
    assert current_only(ordered, "Where do I live?") == [new]


def _item(key: str, text: str, session: str, turn: int, **extra: Any) -> Item:
    """A note (``r:``) or a memory item (``m:``) of chat *session*'s turn *turn*."""
    return Item(key, text, f"chat {session}", session, turn, 1.0, **extra)


def test_a_note_follows_a_replaced_item_that_says_the_same_in_its_own_turn() -> None:
    said = "Your book club meets on Tuesdays."
    club = "The book club meets on Tuesdays."
    old = _item("m:1", club, "a", 2, sources=(("a", 2),), replaced_by="m:2")
    new = _item("m:2", "The book club meets on Thursdays.", "c", 2)
    kept = _item("r:1", said, "a", 2)
    others = [
        _item("r:2", "Your cat is called Miso.", "a", 2),  # another fact of the turn
        _item("r:3", said, "b", 4),  # the same words, another turn
        _item("r:4", said, "", 0),  # an import: no turn
        _item("r:5", "Your chess club meets on Fridays.", "a", 3),
        # The note above is of an item that is not replaced.
        _item("m:3", "Your chess club meets on Fridays.", "a", 3, sources=(("a", 3),)),
    ]
    followed = records.notes_of_replaced([old, new, kept, *others])
    assert followed == {"r:1": old}
    # The note reads as older after the newer item, which comes in, and stays in the block.
    known = {"m:1": old, "m:2": new}
    ordered = newer_first([kept], {}, known=known, followed=followed)
    assert [(i.key, i.older) for i in ordered] == [("m:2", ""), ("r:1", OLDER_ITEM)]
    assert current_only(ordered, "When does the book club meet?") == ordered
    # Without the newest replacement known, or without the map, it reads as it did.
    assert newer_first([kept], {}, known={"m:1": old}, followed=followed) == [kept]
    assert newer_first([kept], {}, known=known) == [kept]


def test_a_note_that_reads_as_older_already_keeps_its_mark_beside_a_newer_one() -> None:
    marked = _item("r:1", "Your weekly sync is on Friday.", "a", 2, older=OLDER_ITEM)
    other = _item("r:2", "Your weekly sync is on Monday.", "b", 3)
    assert records.newer_values_first([other, marked]) == [other, marked]


def test_an_item_follows_its_turns_only_when_a_note_replaced_all_of_them() -> None:
    note = _item("r:3", "Your weekly sync is on Friday.", "c", 1)
    only = _item("m:1", "The weekly sync is on Wednesday.", "a", 3, sources=(("a", 3),))
    both = _item("m:2", "The weekly sync is in room 4.", "a", 3)
    both.sources = (("a", 3), ("a", 4))
    ordered = newer_first([only, both], {("a", 3): note})
    # The note comes in ahead of the old item; the item with a turn that stands stays.
    assert [(i.key, i.older) for i in ordered] == [
        ("r:3", ""),
        ("m:2", ""),
        ("m:1", OLDER),
    ]
    assert current_only(ordered, "When is the weekly sync?") == ordered[:2]
    assert current_only(ordered, "When was the weekly sync before?") == ordered
    # A turn a newer turn replaced is not a note: the item reads as it did.
    turn = _item("d:9", "User: The weekly sync is on Friday.", "c", 1)
    assert newer_first([only], {("a", 3): turn}) == [only]


@pytest.mark.parametrize(
    "message",
    [
        "Where did I live before?",
        "Where did I use to live?",
        "What was my earlier address?",
        "What is my previous job?",
        "Which was my old name?",
        "What was my name previously?",
        "Who was my coach formerly?",
        "我以前住在哪里？",
        "我之前的名字是什么？",
        "我原来的导师是谁？",
        "我过去在哪里工作？",
    ],
)
def test_a_message_about_the_past_keeps_the_older_item(message: str) -> None:
    ordered, _ = _stale()
    assert asks_about_past(message)
    assert current_only(ordered, message) == ordered


@pytest.mark.parametrize(
    "message",
    [
        "Where do I live?",
        "What is my name?",
        "How old am I?",
        "Tell me what I told you.",
        "我现在住在哪里？",
        "我的健身教练是谁？",
        "",
    ],
)
def test_other_messages_get_the_current_value_only(message: str) -> None:
    ordered, current = _stale()
    assert not asks_about_past(message)
    assert current_only(ordered, message) == current


def test_words_that_ask_for_an_older_value_are_not_content_words() -> None:
    # So "Where did I live before?" finds what "Where do I live?" finds.
    assert gate_terms("Where did I live before?") == gate_terms("Where do I live?")
    assert gate_terms("What was it earlier?") == gate_terms("What was it?")


def test_a_note_reads_as_the_users_whichever_form_it_has() -> None:
    def prompt(text: str, quote: str = "") -> str:
        return Item("r:1", text, "chat A", "a", 1, 1.0, quote=quote).prompt

    # The user's own words when the note has them.
    assert prompt("Your name is Juno.", "My name is Juno.") == (
        "the user said “My name is Juno.” (chat A)"
    )
    # Without, a note in the second person is written to the user ...
    assert (
        prompt("Your name is Juno.")
        == "a note to the user “Your name is Juno.” (chat A)"
    )
    assert prompt("你的导师是周教授。").startswith("a note to the user “你的导师")
    assert prompt("The user's name is Juno.").startswith("a note to the user “The user")
    # ... and one in the first person is the user's own, shown unchanged.
    for first in (
        "I prefer tea.",
        "My name is Juno.",
        "我习惯喝茶。",
        "We meet on Fridays.",
    ):
        assert prompt(first) == f"the user's note “{first}” (chat A)"
    # A word that only starts like one is not.
    assert prompt("Weekly sync is on Fridays.").startswith("a note to the user")
    assert prompt("In March you move.").startswith("a note to the user")


def test_the_block_header_says_what_you_means() -> None:
    text, _ = block([Item("r:1", "Your name is Juno.", "chat A", "a", 1, 1.0)])
    assert text.split("\n")[0] == BLOCK_HEADER
    assert "“you” is the user" in BLOCK_HEADER


def test_every_text_that_shows_notes_names_the_same_form() -> None:
    """The model reads notes in the second person, written to the user."""
    for text in (NOTES_TEXT, REMEMBER.description, SEARCH_MEMORY.description):
        assert "“you”" in text.lower()
    for text in (NOTES_TEXT, REMEMBER.description):
        assert "second person" in text
        assert "in the user's words" not in text
    assert "“you” in it means the user, never you" in NOTES_TEXT
    # The pinned lines are the user's own words (quotes), and say so.
    assert "in their words" in PINNED_TEXT
    # What the tool saves reads the way the rule says (and a change too).
    assert records.second_person("I work at Acme now.") == "You work at Acme now."
    assert records.second_person("我的导师是周教授。") == "你的导师是周教授。"


def test_the_tool_text_and_the_notes_text_agree_on_language_and_say_nothing_of_claims() -> (
    None
):
    """A note is in the language of this turn's message. Neither text tells the
    model when it may say a fact is noted: quoting the tool's output made it
    imitate the output ("Remembered: ..." with no call), and a rule about
    claiming saves made it refuse recalled facts that were never kept as notes.
    Code makes a claimed save real instead (``ChatMemory.keep_claims``)."""
    for text in (NOTES_TEXT, REMEMBER.description):
        assert "in the language of the user's message in this turn" in text
        assert "noted only after" not in text
        assert "say plainly that it was not kept" not in text
        assert "“Remembered”" not in text and "“Not remembered”" not in text
    assert "in the user's language" not in REMEMBER.description


def test_the_notes_text_says_recalled_items_answer_whether_or_not_kept_as_notes() -> (
    None
):
    """A recalled earlier message is as good as a note for answering."""
    assert "Use them to answer whether or not they were kept as notes" in NOTES_TEXT
    assert "an earlier message of this chat or another counts as much as a note" in (
        NOTES_TEXT
    )


def test_the_memory_rule_asks_for_the_current_value_and_for_what_was_asked() -> None:
    for rule in (
        "Answer with the newest value only, and mention an older value only when "
        "the user asks about the past.",
        "Use only the notes that answer the question; leave out other remembered "
        "facts.",
        "the newest one is true",
    ):
        assert rule in NOTES_TEXT
    assert "\n" not in NOTES_TEXT


def test_a_change_is_a_statement_to_keep_in_that_turn() -> None:
    change = "“I changed jobs; I work at Acme now.”"
    assert change in NOTES_TEXT
    assert "keep the new note in that turn, it replaces the old one" in NOTES_TEXT
    assert "Never ask whether to remember something" in NOTES_TEXT
    assert f"{change} → You work at Acme. (it replaces the old note)" in (
        REMEMBER.description
    )
    # The note the tool is told to write replaces the one it names.
    old, new = "You work at Spruce Labs.", "You work at Acme."
    assert records.subject(old) == records.subject(new) != ""


def test_the_notes_text_says_reminders_are_scheduled_not_noted() -> None:
    """A "remind me … at <time>" goes to the scheduler; the chat never keeps it as a note."""
    assert "Reminders are scheduled, not noted" in NOTES_TEXT
    assert "do not save it as a note" in NOTES_TEXT
