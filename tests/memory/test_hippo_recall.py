"""Recall from memory items: ``[memory] recall_source = "items"``.

Items a chat model encoded from the chats take the place of the turns: recall ranks,
bars, picks and shows them as it does turns, a list line is ``[m12] …`` under its own
guide, ``open_memory`` opens an item (and, with ``open_sources``, the turns it came
from), a replaced item reads as older, a conflicting one names its partner, and the
turn's memory event says which items were used, searched and opened. With the default
source, ``turns``, nothing changes. Synthetic items, words and fixed vectors, no model.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import MemoryItem
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    ITEM_LIST_GUIDE,
    LIST_GUIDE,
    OLDER_ITEM,
    Corpus,
    Item,
    day,
    listed_title,
    ranked,
)
from localmelo.support.memory.service import (
    CHAT_TURN,
    IN_THIS_CHAT,
    NO_SUCH_TURN,
    OPEN_USAGE_ITEMS,
    MemoryTools,
    open_memory_items,
    search_memory_listing,
)
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.test_recall_bars import (
    FAR_QUESTION,
    MODEL,
    FixedEmbedding,
    vector_at,
)
from localmelo.tests.memory.test_recall_list import (
    DAY,
    QUESTION,
    _turn,
    add_turn,
    ask,
    config,
)

NOTHING_FOUND = "Nothing in melo's memory matches that."
TITLE = "Tomato garden stakes"
TEXT = "The user stakes the tomato garden with bamboo."
TURN = ("Which tomato garden bed gets the sun?", "Bed one faces south all day.")


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


async def put(
    memory: ChatMemory,
    session_id: str,
    title: str,
    text: str,
    *,
    turns: tuple[int, ...] = (1,),
    kind: str = "semantic",
    at: float | None = None,
    cosine: float | None = None,
) -> int:
    """Store a memory item of chat *session_id* and give its number."""
    _, index = memory.stores()
    return await index.add_item(
        MemoryItem(
            session_id=session_id,
            title=title,
            text=text,
            kind=kind,
            created_at=at if at is not None else time.time() - 3 * DAY,
            sources=tuple((session_id, turn) for turn in turns),
            vector=tuple(vector_at(cosine)) if cosine is not None else None,
            embed_model=MODEL if cosine is not None else "",
            encoder="1:stub",
        )
    )


class Held:
    """A memory with one chat whose turn and item both speak of the tomato garden."""

    def __init__(self, memory: ChatMemory, session_id: str, doc: int, item: int):
        self.memory, self.session_id, self.doc, self.item = (
            memory,
            session_id,
            doc,
            item,
        )


async def held(tmp_path: Path, store: SessionStore, cfg: Any, **item: Any) -> Held:
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    when = time.time() - 3 * DAY
    session_id, (doc,) = await add_turn(
        memory, store, TURN[0], TURN[1], at=when, title="Garden"
    )
    number = await put(memory, session_id, TITLE, TEXT, at=when, **item)
    return Held(memory, session_id, doc, number)


def lines_of(text: str) -> list[str]:
    """The item lines of a block or a search result: ``[m12] …``."""
    return [line for line in text.split("\n") if re.match(r"\[m\d+\] ", line)]


def has_item_line(text: str) -> bool:
    return bool(lines_of(text))


# ── The default is unchanged ──


async def test_the_default_source_is_turns_and_items_in_the_index_change_nothing(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    assert cfg.memory.recall_source == "turns" and cfg.memory.open_sources == "off"
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        session_id, _ = await add_turn(
            memory, store, TURN[0], TURN[1], at=time.time() - 3 * DAY, title="Garden"
        )
        before = await ask(memory, store)
        search = await MemoryTools(memory).search(query=QUESTION)
        names = [t.definition.name for t in await MemoryTools(memory).start()]
        await put(memory, session_id, TITLE, TEXT)
        after = await ask(memory, store)
        assert after.block == before.block != ""
        assert [i.key for i in after.used] == [i.key for i in before.used]
        assert await MemoryTools(memory).search(query=QUESTION) == search
        assert [t.definition.name for t in await MemoryTools(memory).start()] == names
        assert not memory.recall_settings().items
        assert not any(i.is_memory for i in memory._corpus.items.values())
        assert not has_item_line(after.block) and TITLE not in after.block
    finally:
        await memory.aclose()


async def test_the_item_settings_are_read_each_time(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        assert not memory.recall_settings().items
        cfg.memory.recall_source = "items"
        cfg.memory.open_sources = "on"
        settings = memory.recall_settings()
        assert settings.items and settings.open_sources
        cfg.memory.recall_source = "turns"
        assert not memory.recall_settings().items
    finally:
        await memory.aclose()


# ── The excerpt form ──


async def test_an_item_is_recalled_in_place_of_the_turn_as_its_title_and_text(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        found = await ask(memory, store)
        when = day(time.time() - 3 * DAY)
        assert found.block == (
            f"\n\n{BLOCK_HEADER}\n- “{TITLE}”: {TEXT} (chat “Garden”, {when})"
        )
        assert [i.key for i in found.used] == [f"m:{fill.item}"]
        # The turn itself is not in the corpus, and no list line is in the block.
        assert not any(i.is_turn for i in memory._corpus.items.values())
        assert LIST_GUIDE not in found.block and not has_item_line(found.block)
        reply = await MemoryTools(memory).search(query=QUESTION)
        assert reply == f"1. “{TITLE}”: {TEXT} (chat “Garden”, {when})"
    finally:
        await memory.aclose()


async def test_the_item_shows_its_own_words_the_way_a_turn_shows_its_head(
    tmp_path: Path, store: SessionStore
) -> None:
    long = "The user stakes the tomato garden with bamboo. " + "It rains. " * 120
    cfg = config(recall_source="items", recall_item_chars=300)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        session_id, _ = await add_turn(
            memory, store, TURN[0], TURN[1], at=time.time() - 3 * DAY, title="Garden"
        )
        await put(memory, session_id, TITLE, long)
        found = await ask(memory, store)
        (line,) = found.block.split("\n")[3:]
        assert line.startswith(f"- “{TITLE}”: The user stakes the tomato garden")
        assert len(line) < 300 + len(TITLE) + 60 and "…" in line
    finally:
        await memory.aclose()


async def test_an_item_is_found_by_its_title_too(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        session_id, _ = await add_turn(
            memory, store, "Hello there", "Hi.", at=time.time() - 3 * DAY
        )
        number = await put(
            memory, session_id, "Zucchini trellis plans", "Needs six posts."
        )
        found = await ask(memory, store, "what about the zucchini trellis plans?")
        assert [i.key for i in found.used] == [f"m:{number}"]
    finally:
        await memory.aclose()


async def test_a_question_that_matches_no_item_recalls_nothing(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    fill = await held(tmp_path, store, cfg)
    try:
        found = await ask(fill.memory, store, "what is the capital of Mongolia?")
        assert found.block == "" and not found.used
        assert (
            await MemoryTools(fill.memory).search(query="capital of Mongolia")
            == NOTHING_FOUND
        )
    finally:
        await fill.memory.aclose()


async def test_the_bars_apply_to_items_as_to_turns() -> None:
    corpus = Corpus()
    near = Item(
        "m:1",
        "a dog barks in the yard",
        "chat",
        "s",
        1,
        time.time(),
        vector=np.asarray(vector_at(0.60), np.float32),
        embed_model=MODEL,
        title="Dog",
    )
    corpus.add(near)
    query = np.asarray([1.0, 0.0, 0.0], np.float32)
    now = time.time() + 1
    assert ranked(corpus, FAR_QUESTION, query, MODEL, now=now) == [near]
    assert ranked(corpus, FAR_QUESTION, query, MODEL, now=now, min_cosine=0.65) == []
    # The title is searched as well as the text.
    assert ranked(corpus, "dog", None, "", now=now) == [near]
    titled = Item("m:2", "plans for six posts", "chat", "s", 1, now, title="Zucchini")
    corpus.add(titled)
    assert ranked(corpus, "zucchini", None, "", now=now) == [titled]


async def test_a_vector_found_item_is_listed_with_its_similarity(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    provider = FixedEmbedding()
    memory.embedding = lambda: (provider, MODEL)  # type: ignore[method-assign]
    try:
        session_id, _ = await add_turn(
            memory, store, "Hello there", "Hi.", at=time.time() - 3 * DAY
        )
        number = await put(
            memory,
            session_id,
            "Cat on the sofa",
            "The cat sleeps on the sofa.",
            cosine=0.70,
        )
        reply = await MemoryTools(memory).search(query=FAR_QUESTION)
        (line,) = lines_of(reply)
        assert line.startswith(f"[m{number}] ") and line.endswith(" · 0.70")
    finally:
        await memory.aclose()


# ── The list form ──


async def test_an_item_is_one_list_line_under_its_own_guide(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        found = await ask(memory, store)
        when = day(time.time() - 3 * DAY)
        assert found.block.split("\n")[2:] == [
            BLOCK_HEADER,
            ITEM_LIST_GUIDE,
            f"[m{fill.item}] {when} · chat “Garden” · “{TITLE}” · words",
        ]
        assert LIST_GUIDE not in found.block
        assert ITEM_LIST_GUIDE == (
            "Memory items that may hold the answer, closest first. Open the ones you "
            "need with open_memory before you answer from them; a title is not the "
            "content."
        )
        reply = await MemoryTools(memory).search(query=QUESTION)
        assert reply.split("\n") == [
            ITEM_LIST_GUIDE,
            f"[m{fill.item}] {when} · chat “Garden” · “{TITLE}” · words",
        ]
    finally:
        await memory.aclose()


async def test_the_tools_of_the_item_list_form_are_ascii_and_name_the_ids(
    tmp_path: Path,
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        search, remember, opened = await MemoryTools(memory).start()
        assert [t.definition.name for t in (search, remember, opened)] == [
            "search_memory",
            "remember",
            "open_memory",
        ]
        assert search.definition.description == (
            search_memory_listing(cfg.memory.recall_list_max, True).description
        )
        for tool in (search.definition, opened.definition):
            # English only; the curly quotes are the note wording's, as in turns mode.
            assert all(ord(c) < 128 or c in "“”" for c in tool.description)
            assert str(tool.parameters).isascii()
            assert "memory item" in tool.description.lower()
        ids = opened.definition.parameters["properties"]["ids"]
        assert ids["description"] == "Ids from the list, such as m12."
        assert "came from" not in opened.definition.description
        assert "came from" in open_memory_items(True).description
        assert open_memory_items(True).description.isascii()
        # Turns mode keeps the turn tools.
        cfg.memory.recall_source = "turns"
        again = await MemoryTools(memory).start()
        assert again[2].definition.parameters["properties"]["ids"]["description"] == (
            "Ids from the list, such as t123."
        )
    finally:
        await memory.aclose()


async def test_open_memory_opens_an_item_with_its_title_and_text(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        tools = MemoryTools(memory)
        when = day(time.time() - 3 * DAY)
        reply = await tools.open_memory(ids=[f"m{fill.item}"])
        assert reply == f"[m{fill.item}] {when} · chat “Garden”\n{TITLE}\n{TEXT}"
        # A bracketed id and capitals read the same.
        assert await tools.open_memory(ids=[f"[M{fill.item}]"]) == reply
        assert await tools.open_memory(ids=f"m{fill.item}") == reply
    finally:
        await memory.aclose()


async def test_open_memory_says_why_an_id_opens_nothing(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        tools = MemoryTools(memory)
        assert await tools.open_memory(ids=[]) == OPEN_USAGE_ITEMS
        assert OPEN_USAGE_ITEMS == "Pass 1 to 5 ids from the list, such as m12."
        reply = await tools.open_memory(ids=["m99999", f"t{fill.doc}", "r1", "nope"])
        assert reply.split("\n\n") == [
            f"m99999: {NO_SUCH_TURN}",
            f"t{fill.doc}: not an id from the list; ids look like m12.",
            "r1: that is a note, which is shown in full already.",
            "nope: not an id from the list; ids look like m12.",
        ]
    finally:
        await memory.aclose()


async def test_an_item_of_a_deleted_chat_is_gone(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        tools = MemoryTools(memory)
        assert (await tools.open_memory(ids=[f"m{fill.item}"])).startswith("[m")
        _, index = memory.stores()
        await index.delete(fill.session_id)
        assert await tools.open_memory(ids=[f"m{fill.item}"]) == (
            f"m{fill.item}: {NO_SUCH_TURN}"
        )
        found = await ask(memory, store)
        assert found.block == "" and not found.used
    finally:
        await memory.aclose()


async def test_the_sources_of_an_item_come_with_it_when_asked_for(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list", open_sources="on")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        session_id, _ = await add_turn(
            memory, store, "Which bed gets the sun?", "Bed one.", at=when, title="G"
        )
        await add_turn(
            memory,
            store,
            "What do I stake it with?",
            "Bamboo works well.",
            at=when + 60,
            session_id=session_id,
            turn=2,
            title="G",
        )
        number = await put(memory, session_id, TITLE, TEXT, turns=(1, 2), at=when)
        reply = await MemoryTools(memory).open_memory(ids=[f"m{number}"])
        head, body = reply.split("\n", 1)
        assert head.startswith(f"[m{number}] ") and body.startswith(f"{TITLE}\n{TEXT}")
        assert "Source turns:" in body
        turns = body.split("Source turns:\n")[1].split("\n")
        assert len(turns) == 2
        assert "Which bed gets the sun?" in turns[0] and "Bed one." in turns[0]
        assert "What do I stake it with?" in turns[1] and "Bamboo" in turns[1]
        # Without the setting the item opens alone.
        cfg.memory.open_sources = "off"
        alone = await MemoryTools(memory).open_memory(ids=[f"m{number}"])
        assert "Source turns:" not in alone and alone.endswith(TEXT)
    finally:
        await memory.aclose()


async def test_an_opened_item_and_its_sources_fit_in_recall_open_chars(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(
        recall_source="items",
        recall_form="list",
        open_sources="on",
        recall_open_chars=1000,
    )
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        sentence = "Sentence {n} says to water the tomato plants early. "
        session_id = ""
        for turn in (1, 2, 3):
            session_id, _ = await add_turn(
                memory,
                store,
                f"Question {turn}?",
                "".join(sentence.format(n=n) for n in range(80)),
                at=when + turn,
                session_id=session_id or None,
                turn=turn,
                title="G",
            )
        number = await put(memory, session_id, TITLE, TEXT, turns=(1, 2, 3), at=when)
        reply = await MemoryTools(memory).open_memory(ids=[f"m{number}"])
        assert len(reply) <= 1000
        assert TEXT in reply and "Source turns:" in reply
        assert "… (cut)" in reply
    finally:
        await memory.aclose()


async def test_a_chat_that_shows_the_sources_does_not_open_the_item(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        info = await _turn(store, "Garden")
        _, _ = await add_turn(
            memory,
            store,
            "Where?",
            "Over there.",
            session_id=info.session_id,
            turn=3,
            title="Garden",
        )
        number = await put(
            memory, info.session_id, TITLE, TEXT, turns=(3,), at=time.time() - DAY
        )
        shown = await memory.open_item(
            number, exclude_session=info.session_id, keep_from_turn=2
        )
        assert shown == IN_THIS_CHAT
        past = await memory.open_item(
            number, exclude_session=info.session_id, keep_from_turn=4
        )
        assert not isinstance(past, str)
        assert past.item.where.startswith("this chat, ")
        other = await memory.open_item(
            number, exclude_session="elsewhere", keep_from_turn=0
        )
        assert not isinstance(other, str) and other.item.where.startswith("chat “")
    finally:
        await memory.aclose()


# ── Older and conflicting items ──


async def test_a_replaced_item_reads_as_older_and_its_replacement_leads(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        now = time.time()
        session_id, _ = await add_turn(
            memory, store, "Where?", "Here.", at=now - 5 * DAY, title="Sync"
        )
        old = await put(
            memory,
            session_id,
            "Weekly sync day",
            "The weekly sync is on Wednesday.",
            at=now - 4 * DAY,
        )
        new = await put(
            memory,
            session_id,
            "Weekly sync day",
            "The weekly sync is now on Friday.",
            at=now - 2 * DAY,
        )
        _, index = memory.stores()
        await index.mark_replaced(old, new)
        found = await ask(memory, store, "when is the weekly sync?")
        assert [i.key for i in found.used] == [f"m:{new}"]
        assert OLDER_ITEM not in found.block
        # Asked about the past, the old one is listed after the new one, marked.
        past = await ask(memory, store, "when was the weekly sync before?")
        assert [i.key for i in past.used] == [f"m:{new}", f"m:{old}"]
        assert f"({OLDER_ITEM})" in past.block
        opened = await MemoryTools(memory).open_memory(ids=[f"m{old}"])
        assert opened.split("\n")[0].endswith(f"({OLDER_ITEM})")
        fresh = await MemoryTools(memory).open_memory(ids=[f"m{new}"])
        assert "older" not in fresh.split("\n")[0]
        # A search lists the old one after the new one.
        reply = await MemoryTools(memory).search(query="weekly sync")
        listed = lines_of(reply)
        assert [line[: line.index("]") + 1] for line in listed] == [
            f"[m{new}]",
            f"[m{old}]",
        ]
        assert listed[-1].endswith(f"({OLDER_ITEM})")
        assert OLDER_ITEM == "older; replaced by a newer item"
    finally:
        await memory.aclose()


async def test_a_conflicting_item_names_its_partners_in_the_line_and_the_heading(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        now = time.time()
        session_id, _ = await add_turn(
            memory, store, "Food?", "Noted.", at=now - 5 * DAY, title="Food"
        )
        like = await put(
            memory,
            session_id,
            "Spicy food",
            "The user eats spicy food every day.",
            at=now - 4 * DAY,
        )
        never = await put(
            memory,
            session_id,
            "Spicy food",
            "The user never eats spicy food.",
            at=now - 3 * DAY,
        )
        _, index = memory.stores()
        await index.link_items(never, like, conflict=True)
        reply = await MemoryTools(memory).search(query="spicy food")
        listed = lines_of(reply)
        assert len(listed) == 2
        by_id = {line[1 : line.index("]")]: line for line in listed}
        assert by_id[f"m{like}"].endswith(f" (conflicts with m{never})")
        assert by_id[f"m{never}"].endswith(f" (conflicts with m{like})")
        opened = await MemoryTools(memory).open_memory(ids=[f"m{like}"])
        assert opened.split("\n")[0].endswith(f" (conflicts with m{never})")
        # Both stay recallable: a conflict is not a replacement.
        found = await ask(memory, store, "does the user eat spicy food?")
        assert {i.key for i in found.used} == {f"m:{like}", f"m:{never}"}
        assert "older" not in found.block
    finally:
        await memory.aclose()


async def test_the_corpus_reads_an_item_again_when_its_status_changes(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        now = time.time()
        session_id, _ = await add_turn(
            memory, store, "Where?", "Here.", at=now - 5 * DAY, title="Sync"
        )
        first = await put(
            memory, session_id, "Sync day", "The weekly sync is on Wednesday."
        )
        second = await put(
            memory, session_id, "Sync day", "The weekly sync is on Friday."
        )
        await ask(memory, store, "weekly sync")
        assert not memory._corpus.items[f"m:{first}"].replaced_by
        _, index = memory.stores()
        await index.mark_replaced(first, second)
        await ask(memory, store, "weekly sync")
        assert memory._corpus.items[f"m:{first}"].replaced_by == f"m:{second}"
        await index.delete(session_id)
        await ask(memory, store, "weekly sync")
        assert not memory._corpus.keys("m:")
    finally:
        await memory.aclose()


# ── The turn's memory event ──


async def test_the_event_names_the_items_used_searched_and_opened(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list", open_sources="on")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        emitted: list[dict[str, Any]] = []
        typed = QUESTION
        await memory.before_turn(
            store, await _turn(store, typed), typed, emit=emitted.append
        )
        chat = CHAT_TURN.get()
        assert chat is not None
        data = chat.event()
        assert data is not None
        (used,) = data["used"]
        assert used["kind"] == "item" and used["item"] == fill.item
        assert used["id"] == fill.item and used["sources"] == [[fill.session_id, 1]]
        assert used["source"].startswith("chat “Garden”")
        tools = MemoryTools(memory)
        await tools.search(query="tomato garden stakes")
        searched = emitted[-1]["searched"]
        assert [(s["kind"], s["item"]) for s in searched] == [("item", fill.item)]
        await tools.open_memory(ids=[f"m{fill.item}"])
        assert emitted[-1]["opened"] == [
            {
                "item": fill.item,
                "sources": [[fill.session_id, 1]],
            }
        ]
        count = len(emitted)
        await tools.open_memory(ids=[f"m{fill.item}"])  # nothing new, nothing sent
        assert len(emitted) == count
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


def test_an_item_reads_as_json_with_its_sources() -> None:
    at = time.time()
    item = Item(
        "m:12",
        "The user stakes the tomato garden.",
        f"chat “Fix CI”, {day(at)}",
        "s1",
        3,
        at,
        title="Stakes",
        sources=(("s1", 3), ("s2", 1)),
    )
    data = item.to_json()
    assert data["kind"] == "item" and data["id"] == 12 and data["item"] == 12
    assert data["sources"] == [["s1", 3], ["s2", 1]]
    assert data["text"].startswith("The user stakes")
    assert item.ref == "m12" and item.is_memory and not item.is_turn
    assert item.opened_heading() == f"[m12] {day(at)} · chat “Fix CI”"
    assert listed_title(item) == "Stakes"
    turn = Item("d:12", "User: hi\nmelo: hello", "chat", "s1", 3, at)
    assert turn.to_json()["kind"] == "chat" and "item" not in turn.to_json()
    assert turn.ref == "t12" and turn.is_turn and not turn.is_memory


# ── Forgetting ──


async def test_forgetting_a_chat_or_everything_takes_the_items_and_a_reindex_keeps_them(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        first, _ = await add_turn(
            memory, store, "Which bed?", "Bed one.", at=when, title="One"
        )
        second, _ = await add_turn(
            memory, store, "Which pot?", "Pot two.", at=when, title="Two"
        )
        kept = await put(memory, first, "Bed", "The user plants tomatoes in bed one.")
        gone = await put(memory, second, "Pot", "The user keeps basil in pot two.")
        _, index = memory.stores()
        await memory.reindex(None)
        assert await index.count_items() == 2
        await memory.forget_session(second)
        assert [i.id for i in await index.fetch_items([kept, gone])] == [kept]
        await memory.forget_all()
        assert await index.count_items() == 0
        assert not memory._corpus.keys("m:")
        assert (await ask(memory, store, "tomatoes in bed one")).block == ""
    finally:
        await memory.aclose()
