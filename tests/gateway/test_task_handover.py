"""What a chat hands a task it starts: the person's words, its turns, what it recalled.

- The person's own words are shown when the request does not hold them, cut
  at a sentence end past 1200 characters.
- The chat's newest earlier turns are the person's (never a task's question
  to the chat, never the running turn), oldest first, in the earlier-turns
  block's own bounds.
- What the chat's running turn recalled, searched and opened goes to the
  task only when that turn is the person's, with memory on.
- Those lines come opened first, then searched, then recalled, then the
  other chats' digest; each turn, note or item once (the earlier-turns block
  included); no People or Decisions note; no id a task cannot open; never
  past their room, opened text cut at a sentence end before a line goes.

A real memory (words only) and a real session store; no model.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent._earlier import EARLIER_BLOCK_CHARS, EARLIER_TURNS
from localmelo.support.gateway.handover import (
    OPENED_FLOOR,
    PICKED_CHARS,
    WORDS_CHARS,
    Handover,
    handover_of,
    own_words,
    picked_lines,
    turn_lines,
)
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.digest import DigestItem
from localmelo.support.memory.recall import CUT_MARK, Item
from localmelo.support.memory.service import ASK_OWNER, ChatTurn, Recall, TurnMemory
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.memory.test_recall_list import add_turn

HOLDER = "test-host:1:handover"
SENTENCE = "Alex wants a staff role at a large company, close to the city. "


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def said(
    store: SessionStore,
    chat: str | None,
    typed: str,
    answer: str = "Noted.",
    *,
    origin: str = "",
    finish: bool = True,
) -> TurnInfo:
    """A turn of *typed* in *chat* (a new chat when None), finished unless not *finish*."""
    info = await store.begin_turn(
        chat,
        typed,
        holder=HOLDER,
        typed_text=typed,
        new_session=None if chat else NewSession(kind="front"),
        origin=origin,
    )
    if finish:
        await store.finish_turn(
            info.session_id,
            info.turn,
            holder=HOLDER,
            status="completed",
            cause="run",
            result=None,
            answer=answer,
        )
    return info


async def note(
    memory: ChatMemory, text: str, quote: str, topic: str, **extra: Any
) -> str:
    """One note of *topic*; its key (``r:<id>``)."""
    source = {"session_id": "s1", "turn": 1, "chat": "Main chat", **extra}
    cand = records.Candidate(text, quote, topic, "user_statement")
    written = await records.write(
        memory.stores()[0], [cand], [None], source=source, embed_model=""
    )
    return f"r:{written[0].id}"


async def item_of(memory: ChatMemory, key: str) -> Item:
    """The recall item of note *key*, as the chat's turn would hold it."""
    (text,) = [e.text for e in await memory.records() if f"r:{e.id}" == key]
    for item in await memory.search(text, records_k=8, docs_k=0):
        if item.key == key:
            return item
    raise AssertionError(f"{key} is not found")


def chat_turn(
    store: SessionStore,
    info: TurnInfo,
    *,
    used: tuple[Item, ...] = (),
    searched: list[Item] | None = None,
    opened: list[dict[str, Any]] | None = None,
    memory_on: bool = True,
) -> ChatTurn:
    found = TurnMemory(used=used, recall=Recall() if memory_on else None)
    return ChatTurn(
        store,
        info,
        info.user_text,
        found,
        searched=list(searched or []),
        opened=list(opened or []),
    )


# ── The person's own words ──


def test_words_the_request_holds_are_not_shown_again() -> None:
    assert own_words("", "Fix registry.py") == ""
    assert own_words("Fix registry.py", "Fix registry.py") == ""
    assert own_words("  fix   Registry.py ", "Fix registry.py") == ""
    # The request is the message with an attachment, or with melo's reading after it.
    assert own_words("Read this", 'Read this\n\n<attachment name="a.txt">…') == ""


def test_words_the_request_does_not_hold_are_shown_whole() -> None:
    words = "read his résumé first, then tell me which roles fit"
    assert own_words(words, "Read Alex's résumé and list roles that fit.") == words
    assert own_words("去查一下", "Look up openings for Alex at large companies.") == (
        "去查一下"
    )


def test_long_words_are_cut_at_a_sentence_end_with_a_mark() -> None:
    words = SENTENCE * 40
    shown = own_words(words, "Plan Alex's search.")
    assert len(shown) <= WORDS_CHARS
    assert shown.endswith(f"city. {CUT_MARK}")
    assert words.startswith(shown[: -len(CUT_MARK) - 1])


# ── The chat's earlier turns ──


async def test_the_newest_turns_of_the_person_come_oldest_first(
    store: SessionStore,
) -> None:
    first = await said(store, None, "My friend Alex is looking for work.")
    chat = first.session_id
    await said(store, chat, "He has eight years of backend work.")
    await said(store, chat, "Where is Alex now?", origin=ASK_OWNER)
    await said(store, chat, "He lives near the city.")
    await said(store, chat, "He wants a large company.")
    now = await said(store, chat, "go look it up", finish=False)
    handover = await handover_of(store, chat, now, None)
    assert [words for _, _, words, _ in handover.turns] == [
        "He has eight years of backend work.",
        "He lives near the city.",
        "He wants a large company.",
    ]
    assert len(handover.turns) == EARLIER_TURNS
    assert not handover.used and not handover.opened
    lines, shown = turn_lines(handover)
    assert lines.splitlines()[0] == "- the user: He has eight years of backend work."
    assert "Where is Alex now?" not in lines
    assert shown == {(chat, turn) for _, turn, _, _ in handover.turns}


async def test_only_the_turns_that_fit_the_block_are_counted_as_shown(
    store: SessionStore,
) -> None:
    first = await said(store, None, SENTENCE * 6, SENTENCE * 8)
    chat = first.session_id
    second = await said(store, chat, SENTENCE * 6, SENTENCE * 8)
    third = await said(store, chat, SENTENCE * 6, SENTENCE * 8)
    handover = await handover_of(store, chat, None, None)
    lines, shown = turn_lines(handover)
    assert lines and len(lines) < EARLIER_BLOCK_CHARS
    kept = sum(1 for line in lines.splitlines() if line.startswith("- "))
    assert kept < 3
    assert shown == {(chat, info.turn) for info in (first, second, third)[3 - kept :]}


async def test_a_chat_with_no_earlier_turn_hands_none(store: SessionStore) -> None:
    now = await said(store, None, "go look it up", finish=False)
    handover = await handover_of(store, now.session_id, now, None)
    assert handover == Handover()
    assert turn_lines(handover) == ("", frozenset())


# ── What the chat's turn recalled ──


async def test_the_persons_turn_hands_what_it_recalled_searched_and_opened(
    store: SessionStore, memory: ChatMemory
) -> None:
    key = await note(memory, "Your friend Alex wants a staff role.", "", "about")
    item = await item_of(memory, key)
    now = await said(store, None, "read his résumé first", finish=False)
    opened = [{"session_id": "s9", "turn": 2, "id": 7}]
    turn = chat_turn(store, now, used=(item,), searched=[item], opened=opened)
    handover = await handover_of(store, now.session_id, now, turn)
    assert handover.used == (item,) and handover.searched == (item,)
    assert handover.opened == tuple(opened)


async def test_a_turn_melo_began_or_another_turn_hands_no_items(
    store: SessionStore, memory: ChatMemory
) -> None:
    key = await note(memory, "Your friend Alex wants a staff role.", "", "about")
    item = await item_of(memory, key)
    first = await said(store, None, "Tell me about Alex.")
    chat = first.session_id
    # A task's question to the chat: agent data, never the person's notes.
    asked = await said(store, chat, "Which city?", origin=ASK_OWNER)
    handover = await handover_of(
        store, chat, asked, chat_turn(store, asked, used=(item,))
    )
    assert not handover.used and [w for _, _, w, _ in handover.turns] == [
        "Tell me about Alex."
    ]
    # The memory's turn is another turn than the one that starts the task.
    now = await said(store, chat, "go look it up", finish=False)
    stale = chat_turn(store, first, used=(item,))
    assert not (await handover_of(store, chat, now, stale)).used
    # No turn named: the items stay.
    assert not (
        await handover_of(store, chat, None, chat_turn(store, now, used=(item,)))
    ).used


async def test_memory_off_for_the_turn_hands_no_items(
    store: SessionStore, memory: ChatMemory
) -> None:
    key = await note(memory, "Your friend Alex wants a staff role.", "", "about")
    item = await item_of(memory, key)
    now = await said(store, None, "read his résumé first", finish=False)
    off = chat_turn(store, now, used=(item,), memory_on=False)
    assert (await handover_of(store, now.session_id, now, off)).used == ()


async def test_the_lines_come_opened_searched_recalled_then_the_digest(
    store: SessionStore, memory: ChatMemory
) -> None:
    about = await note(memory, "Your friend Alex wants a staff role.", "", "about")
    tea = await note(memory, "You drink tea.", "I drink tea.", "preferences")
    landlord = await note(memory, "Your landlord is Dan.", "", "people")
    postgres = await note(memory, "You chose Postgres.", "", "decisions")
    gone = await note(memory, "Your old résumé is in Drive.", "", "about")
    opened_chat, (opened_id,) = await add_turn(
        memory, store, "What does Alex's résumé say?", "Alex led the search team."
    )
    used_chat, _ = await add_turn(
        memory, store, "Alex asked about remote work.", "Remote is fine for him."
    )
    used_turn = next(
        item
        for item in await memory.search("remote work", docs_k=3)
        if item.session_id == used_chat
    )
    notes = {
        k: await item_of(memory, k) for k in (about, tea, landlord, postgres, gone)
    }
    await memory.delete_record(int(gone[2:]))
    handover = Handover(
        opened=({"session_id": opened_chat, "turn": 1, "id": opened_id},),
        searched=(notes[tea], notes[landlord]),
        used=(notes[about], notes[tea], notes[postgres], notes[gone], used_turn),
        digest=(
            DigestItem("s7", 3, "Main chat, 09:10 · the user: Alex has two offers"),
        ),
    )
    lines, kept = await picked_lines(memory, handover)
    body = lines.splitlines()
    assert [line.split(" (")[0] for line in body] == [
        "- the user said “What does Alex's résumé say?”; melo replied “Alex led the "
        "search team.”",
        "- the user said “I drink tea.”",
        "- a note to the user “Your friend Alex wants a staff role.”",
        "- the user said “Alex asked about remote work.”; melo replied “Remote is "
        "fine for him.”",
        "- Main chat, 09:10 · the user: Alex has two offers",
    ]
    for left_out in ("Dan", "Postgres", "Drive"):
        assert left_out not in lines
    assert kept == {about, tea}


async def test_a_turn_reached_twice_or_shown_in_the_earlier_turns_is_listed_once(
    store: SessionStore, memory: ChatMemory
) -> None:
    chat, ids = await add_turn(
        memory, store, "Alex's résumé is two pages.", "Good.", title="Résumé"
    )
    other, _ = await add_turn(memory, store, "Alex likes Rust.", "Noted.", title="Code")
    found = {item.session_id: item for item in await memory.search("Alex", docs_k=4)}
    same, rust = found[chat], found[other]
    handover = Handover(
        opened=({"session_id": chat, "turn": 1, "id": ids[0]},),
        searched=(same, rust),
        used=(same, rust),
        digest=(DigestItem(other, 1, "chat, 09:00 · the user: Alex likes Rust."),),
    )
    lines, _ = await picked_lines(memory, handover)
    assert lines.count("two pages") == 1 and lines.count("Rust") == 1
    # The earlier-turns block shows the Rust turn already.
    lines, _ = await picked_lines(memory, handover, shown={(other, 1)})
    assert "Rust" not in lines and "two pages" in lines


async def test_no_line_names_an_id_a_task_cannot_open(
    store: SessionStore, memory: ChatMemory
) -> None:
    chat, _ = await add_turn(memory, store, "Alex's résumé is two pages.", "Good.")
    turn = next(item for item in await memory.search("Alex", docs_k=2) if item.is_turn)
    # As the excerpt and list forms showed them to the chat: with ids.
    shown = dataclasses.replace(turn, shown=f"[{turn.ref}] {turn.line()}")
    held = Item(
        key="m:12",
        text="Alex wants a staff role.",
        where="Main chat, Oct 30",
        session_id="s1",
        turn=1,
        created_at=time.time(),
        title="Alex's search",
        conflicts=(3, 5),
    )
    lines, _ = await picked_lines(memory, Handover(used=(shown, held)))
    assert "[t" not in lines and "m12" not in lines and "m3" not in lines
    assert "conflicts" not in lines
    assert "“Alex's search”: Alex wants a staff role." in lines


async def test_a_turn_of_the_starting_chat_is_named_so(
    store: SessionStore, memory: ChatMemory
) -> None:
    item = Item(
        key="d:4",
        text="User: Alex is my friend.\nmelo: Got it.",
        where="this chat, Oct 30",
        session_id="c1",
        turn=1,
        created_at=time.time(),
    )
    lines, _ = await picked_lines(memory, Handover(used=(item,)))
    assert "(the chat that started this task, Oct 30)" in lines
    assert "this chat," not in lines


async def test_a_note_from_a_page_or_inferred_never_reads_as_the_persons_words(
    memory: ChatMemory,
) -> None:
    def record(key: str, how: str, where: str) -> Item:
        return Item(
            key=key,
            text="Alex's team ships weekly.",
            where=where,
            session_id="s1",
            turn=1,
            created_at=time.time(),
            quote="Alex's team ships weekly.",
            how=how,
        )

    pointed = await note(memory, "Alex's team ships weekly.", "", "about")
    inferred = await note(memory, "Alex prefers small teams.", "", "about")
    handover = Handover(
        used=(
            record(pointed, "pointed", "from example.com, Oct 30"),
            record(inferred, "inferred", "Main chat, Oct 30"),
        )
    )
    lines, _ = await picked_lines(memory, handover)
    first, second = lines.splitlines()
    assert first.startswith(
        "- a note to the user “Alex's team ships weekly.” (from example.com"
    )
    assert second.startswith("- melo inferred “Alex's team ships weekly.”")
    assert "the user said" not in lines


async def test_opened_text_is_cut_at_a_sentence_end_before_a_line_goes(
    store: SessionStore, memory: ChatMemory
) -> None:
    long_chat, long_ids = await add_turn(
        memory, store, "What did Alex's review say?", SENTENCE * 30
    )
    small = [
        await note(memory, text, "", "about")
        for text in (
            "Your friend Alex hikes on weekends.",
            "Your sister reads science fiction.",
            "Your cat is called Miso.",
        )
    ]
    items = tuple([await item_of(memory, key) for key in small])
    handover = Handover(
        opened=({"session_id": long_chat, "turn": 1, "id": long_ids[0]},),
        used=items,
    )
    lines, kept = await picked_lines(memory, handover)
    assert len(lines) <= PICKED_CHARS
    first, *rest = lines.splitlines()
    assert f"city. {CUT_MARK} (" in first, first
    assert len(rest) == 3 and kept == set(small)
    # Less room: the opened text goes down to its floor, then the lowest lines go.
    lines, kept = await picked_lines(memory, handover, chars=OPENED_FLOOR + 140)
    assert len(lines) <= OPENED_FLOOR + 140
    first, *rest = lines.splitlines()
    assert CUT_MARK in first and len(rest) < 3
    assert kept == set(small[: len(rest)])


async def test_a_turn_or_item_opened_and_gone_since_is_left_out(
    store: SessionStore, memory: ChatMemory
) -> None:
    handover = Handover(
        opened=(
            {"session_id": "gone", "turn": 1, "id": 999},
            {"item": 999, "sources": []},
        )
    )
    assert await picked_lines(memory, handover) == ("", frozenset())


async def test_with_memory_items_the_lines_are_items_and_carry_no_id(
    tmp_path: Path,
) -> None:
    from localmelo.support.memory.index import MemoryItem
    from localmelo.tests.memory.test_recall_list import config

    memory = ChatMemory(
        lambda: config(recall_source="items"), directory=str(tmp_path / "items")
    )
    try:
        item_id = await memory.stores()[1].add_item(
            MemoryItem(
                session_id="s1",
                title="Alex's search",
                text="Alex wants a staff role at a large company.",
                kind="semantic",
                created_at=time.time(),
                sources=(("s1", 1),),
                conflicts=(4,),
            )
        )
        found = await memory.open_item(item_id)
        assert not isinstance(found, str)
        handover = Handover(
            opened=({"item": item_id, "sources": [["s1", 1]]},), used=(found.item,)
        )
        lines, kept = await picked_lines(memory, handover)
        assert lines.splitlines() == [
            f"- “Alex's search”: Alex wants a staff role at a large company. "
            f"({found.item.where})"
        ]
        assert f"m{item_id}" not in lines and "m4" not in lines and kept == set()
    finally:
        await memory.aclose()
