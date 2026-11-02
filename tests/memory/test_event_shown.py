"""``turn.memory`` records the exact line the model read for each item.

Each entry of ``used[]`` (the recall block's items) and ``searched[]`` (what
``search_memory`` returned) has a ``shown`` key: the line as the block or the
search result worded it, without a list number, whatever ``recall_item_chars``,
``recall_excerpt`` and ``recall_form`` made of it. ``text`` stays the stored text.
Synthetic notes and turns, words only, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    LIST_GUIDE,
    Item,
    Shown,
    block,
    day,
    snippet,
)
from localmelo.support.memory.service import CHAT_TURN, MemoryTools, TurnMemory
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import teach
from localmelo.tests.memory.test_recall_bars import _turn, config
from localmelo.tests.memory.test_recall_excerpt import (
    QUESTION,
    TARGET,
    TYPED,
    english,
)

NOTE = "My tomato harvest is the pride of my garden."
DAY = 86_400.0


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


async def build(tmp_path: Path, store: SessionStore, cfg: Any) -> ChatMemory:
    """A note and two long chat turns, all of which the question finds (words)."""
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    await teach(memory, store, await _turn(store, "Pride"), NOTE)
    _, index = memory.stores()
    now = time.time()
    for n in (1, 2):
        turn = await _turn(store, f"{TYPED} {n}")
        await index.add(
            Document(
                session_id=turn.session_id,
                turn=1,
                surface="gateway",
                kind="chat",
                title=f"{TYPED} {n}",
                text=document_text(TYPED, english(1500 + n * 40, 600)),
                created_at=now - n * 3 * DAY,
            )
        )
    return memory


def body(text: str) -> list[str]:
    """The lines of a recall block under its header."""
    lines = text.strip("\n").split("\n")
    assert lines[0] == BLOCK_HEADER
    return lines[1:]


def read_back(entries: list[dict[str, Any]], listing: bool) -> list[str]:
    """The lines a block of these entries is made of, from their ``shown`` keys."""
    if not listing:
        return [f"- {entry['shown']}" for entry in entries]
    notes = [f"- {e['shown']}" for e in entries if e["kind"] == "record"]
    turns = [e["shown"] for e in entries if e["kind"] == "chat"]
    return [*notes, LIST_GUIDE, *turns] if turns else notes


def numbered(entries: list[dict[str, Any]], listing: bool) -> list[str]:
    """The lines a search result of these entries is made of."""
    notes = [e["shown"] for e in entries if e["kind"] == "record"]
    turns = [e["shown"] for e in entries if e["kind"] == "chat"]
    if not listing:
        return [f"{n}. {e['shown']}" for n, e in enumerate(entries, 1)]
    lines = [f"{n}. {shown}" for n, shown in enumerate(notes, 1)]
    return [*lines, LIST_GUIDE, *turns] if turns else lines


SETTINGS = [
    pytest.param({}, id="defaults"),
    pytest.param({"recall_item_chars": 800, "recall_chars": 4000}, id="larger-room"),
    pytest.param(
        {"recall_excerpt": "match", "recall_item_chars": 400, "recall_chars": 4000},
        id="match",
    ),
    pytest.param({"recall_form": "list"}, id="list"),
]


@pytest.mark.parametrize("settings", SETTINGS)
async def test_used_entries_carry_the_line_the_block_holds(
    tmp_path: Path, store: SessionStore, settings: dict[str, Any]
) -> None:
    cfg = config(**settings)
    memory = await build(tmp_path, store, cfg)
    try:
        found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        data = found.event()
        assert data is not None
        entries = data["used"]
        assert len(entries) >= 2 and {e["kind"] for e in entries} == {"record", "chat"}
        listing = cfg.memory.recall_form == "list"
        assert body(found.block) == read_back(entries, listing)
        # The stored text is the same as ever: a note whole, a turn's head.
        for entry, item in zip(entries, found.used, strict=True):
            assert entry["text"] == (
                item.text if item.is_record else snippet(item.text)
            )
            assert entry["shown"] == item.shown
        chats = [e for e in entries if e["kind"] == "chat"]
        assert all(TARGET not in e["text"] for e in chats)
        if settings.get("recall_excerpt") == "match":
            assert all(TARGET in e["shown"] for e in chats)
        if settings.get("recall_item_chars") == 800:
            assert all(len(e["shown"]) > 600 for e in chats)
    finally:
        await memory.aclose()


@pytest.mark.parametrize("settings", SETTINGS)
async def test_searched_entries_carry_the_line_the_search_returned(
    tmp_path: Path, store: SessionStore, settings: dict[str, Any]
) -> None:
    cfg = config(**settings)
    memory = await build(tmp_path, store, cfg)
    try:
        emitted: list[dict[str, Any]] = []
        typed = "hi there"  # nothing here recalls automatically
        await memory.before_turn(
            store, await _turn(store, typed), typed, emit=emitted.append
        )
        reply = await MemoryTools(memory).search(query=QUESTION)
        entries = emitted[-1]["searched"]
        assert len(entries) >= 2 and {e["kind"] for e in entries} == {"record", "chat"}
        listing = cfg.memory.recall_form == "list"
        assert reply.split("\n") == numbered(entries, listing)
        chat = CHAT_TURN.get()
        assert chat is not None
        for entry, item in zip(entries, chat.searched, strict=True):
            assert entry["text"] == (
                item.text if item.is_record else snippet(item.text)
            )
            assert entry["shown"] == item.shown and entry["shown"]
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_the_default_event_shows_a_turns_head_and_a_larger_room_more(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_chars=4000)
    memory = await build(tmp_path, store, cfg)
    try:

        async def chats() -> list[dict[str, Any]]:
            found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
            data = found.event()
            assert data is not None
            return [e for e in data["used"] if e["kind"] == "chat"]

        head = await chats()
        assert head and all(TARGET not in e["shown"] for e in head)
        cfg.memory.recall_excerpt = "match"
        match = await chats()
        assert all(TARGET in e["shown"] and "…" in e["shown"] for e in match)
        cfg.memory.recall_item_chars = 800
        wide = await chats()
        assert all(
            len(w["shown"]) > len(m["shown"]) + 100
            for w, m in zip(wide, match, strict=True)
        )
        # What the page and the JSON keep does not follow the settings.
        assert [e["text"] for e in head] == [e["text"] for e in match]
        assert [e["text"] for e in match] == [e["text"] for e in wide]
    finally:
        await memory.aclose()


async def test_an_item_found_again_keeps_the_line_of_its_first_search(
    tmp_path: Path, store: SessionStore
) -> None:
    """The same item found again adds no entry and changes none."""
    cfg = config()
    memory = await build(tmp_path, store, cfg)
    try:
        emitted: list[dict[str, Any]] = []
        typed = "hi there"
        await memory.before_turn(
            store, await _turn(store, typed), typed, emit=emitted.append
        )
        tools = MemoryTools(memory)
        await tools.search(query=QUESTION)
        first = list(emitted[-1]["searched"])
        cfg.memory.recall_excerpt = "match"
        await tools.search(query=QUESTION)
        assert emitted[-1]["searched"] == first
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_an_item_left_out_of_the_block_has_no_entry(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_chars=900, recall_item_chars=2000, recall_excerpt="match")
    memory = await build(tmp_path, store, cfg)
    try:
        found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        data = found.event()
        assert data is not None
        # Only the note fits; the turns are too long for the room.
        assert [e["kind"] for e in data["used"]] == ["record"]
        assert body(found.block) == read_back(data["used"], False)
        assert found.recall is not None and found.recall.used == 1
    finally:
        await memory.aclose()


async def test_a_turn_with_no_recall_has_no_used_entries(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, cfg)
    try:
        typed = "hi there"
        found = await memory.before_turn(store, await _turn(store, "Hi"), typed)
        data = found.event()
        assert data is not None and data["used"] == [] and found.block == ""
        assert "searched" not in data and "opened" not in data
        # With memory off there is no event at all.
        assert TurnMemory().event() is None
    finally:
        await memory.aclose()


def day_of(item: Item) -> str:
    return day(item.created_at)


def _turn_item(n: int, at: float, text: str) -> Item:
    return Item(f"d:{n}", text, f"chat “Garden {n}”, {day(at)}", f"s{n}", 1, at)


def test_block_gives_the_final_line_not_the_one_it_tried() -> None:
    """Two turns of one day read with the time of day; when only one fits, the
    block drops the time, and ``shown`` is the line the block holds."""
    day = time.mktime((2026, 10, 30, 9, 0, 0, 0, 0, -1))
    first = _turn_item(1, day, document_text("Where do the tomatoes grow?", "East."))
    second = _turn_item(
        2, day + 3600, document_text("Tomatoes?", "Late " + "August " * 80)
    )
    both, used = block([first, second], chars=4000)
    assert [item.key for item in used] == ["d:1", "d:2"]
    assert body(both) == [f"- {item.shown}" for item in used]
    assert "09:00" in used[0].shown and "10:00" in used[1].shown
    alone, kept = block([first, second], chars=300)
    assert [item.key for item in kept] == ["d:1"]
    assert body(alone) == [f"- {kept[0].shown}"]
    assert "09:00" not in kept[0].shown
    # The items the caller gave are not changed; the copies carry the lines.
    assert first.shown == "" and second.shown == ""
    assert kept == [first]


def test_block_in_the_list_form_gives_the_list_lines() -> None:
    day = time.mktime((2026, 10, 30, 9, 0, 0, 0, 0, -1))
    note = Item(
        "r:7", "You like tomatoes.", "chat “Pride”, Oct 30", "s0", 1, day, quote="x"
    )
    turn = _turn_item(
        3, day - 3 * DAY, document_text("Where do tomatoes grow?", "East.")
    )
    text, used = block([note, turn], show=Shown.of(200, "tomatoes"), listing=True)
    assert body(text) == [f"- {used[0].shown}", LIST_GUIDE, used[1].shown]
    assert used[0].shown.startswith("the user said “x”")
    assert used[1].shown.startswith(f"[t3] {day_of(turn)} · ")


def test_an_item_nobody_was_shown_has_no_shown_key() -> None:
    item = _turn_item(1, 1.0, document_text("Q?", "A."))
    assert "shown" not in item.to_json() and item.shown == ""
    assert item.to_json()["text"] == snippet(item.text)
