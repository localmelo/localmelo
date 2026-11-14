"""Recall from memory items: ``[memory] recall_source = "items"``.

Items a chat model encoded from the chats take the place of the turns: recall ranks,
bars, picks and shows them as it does turns, a list line is ``[m12] …`` under its own
guide, ``open_memory`` opens an item (and names the turns it came from), a
replaced item reads as older, a conflicting one names its partner, and the
turn's memory event says which items were used, searched and opened. A turn leaves
recall only while a stored item cites it (whatever the item's status), whatever the
chat's progress mark says: a chat nobody encoded, or a turn the encoder left out, is
not lost, and a store with no items reads as the turns source. With every turn cited,
recall reads items alone. With the default source, ``turns``, nothing changes. With
``auto``, a turn reads turns until the encoder has read nine in ten of them, then
items, when the chat model and every stored item's encoder are on the checked list.
Synthetic items, words and fixed vectors, no model.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.backends.openai_family import CLOUD_BACKENDS
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, hippo_encode, service
from localmelo.support.memory.hippo_route import Judged, Router
from localmelo.support.memory.index import SUMMARY_TURN, MemoryItem
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    ITEM_CHARS,
    ITEM_LIST_GUIDE,
    LIST_GUIDE,
    OLDER,
    OLDER_ITEM,
    PREMISE_GUIDE,
    Corpus,
    Item,
    day,
    listed_title,
    ranked,
)
from localmelo.support.memory.service import (
    CAUGHT_UP,
    CHAT_TURN,
    IN_THIS_CHAT,
    NO_SUCH_TURN,
    OPEN_MEMORY,
    OPEN_MEMORY_ITEMS,
    OPEN_USAGE_ITEMS,
    MemoryTools,
    Reading,
    search_memory_listing,
    search_memory_opening,
)
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.agent.test_tool_exposure import FakeLLM
from localmelo.tests.memory.hippo_helpers import StubEncoder, relate
from localmelo.tests.memory.test_context_pack import note
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
    filled,
)

NOTHING_FOUND = "Nothing in melo's memory matches that."
TITLE = "Tomato garden stakes"
TEXT = "The user stakes the tomato garden with bamboo."
TURN = ("Which tomato garden bed gets the sun?", "Bed one faces south all day.")
# A later turn of the same chat, which no item holds.
LATER = ("Which tomato garden pot needs water?", "Pot two needs water daily.")


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
    encoded: bool = True,
    encoder: str = "1:stub",
) -> int:
    """Store a memory item of chat *session_id* and give its number. As the encoder
    does (:meth:`ChatMemory.compress_chat`), it also marks the chat encoded up to the
    item's last source turn, unless *encoded* is off."""
    _, index = memory.stores()
    number = await index.add_item(
        MemoryItem(
            session_id=session_id,
            title=title,
            text=text,
            kind=kind,
            created_at=at if at is not None else time.time() - 3 * DAY,
            sources=tuple((session_id, turn) for turn in turns),
            vector=tuple(vector_at(cosine)) if cosine is not None else None,
            embed_model=MODEL if cosine is not None else "",
            encoder=encoder,
        )
    )
    if encoded:
        await index.mark_encoded(session_id, max(turns))
    return number


class Held:
    """A memory with one chat whose turn and item both speak of the tomato garden;
    the turn is encoded (the chat's mark is on it). With *later*, the chat has a
    second turn (``later`` is its document) that no item holds."""

    def __init__(
        self, memory: ChatMemory, session_id: str, doc: int, item: int, later: int = 0
    ):
        self.memory, self.session_id, self.doc, self.item, self.later = (
            memory,
            session_id,
            doc,
            item,
            later,
        )


async def held(
    tmp_path: Path,
    store: SessionStore,
    cfg: Any,
    later: tuple[str, str] | None = None,
    **item: Any,
) -> Held:
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    when = time.time() - 3 * DAY
    session_id, (doc,) = await add_turn(
        memory, store, TURN[0], TURN[1], at=when, title="Garden"
    )
    number = await put(memory, session_id, TITLE, TEXT, at=when, **item)
    after = 0
    if later is not None:
        _, (after,) = await add_turn(
            memory,
            store,
            *later,
            at=when + 60,
            session_id=session_id,
            turn=2,
            title="Garden",
        )
    return Held(memory, session_id, doc, number, after)


def lines_of(text: str) -> list[str]:
    """The item lines of a block or a search result: ``[m12] …``."""
    return [line for line in text.split("\n") if re.match(r"\[m\d+\] ", line)]


def has_item_line(text: str) -> bool:
    return bool(lines_of(text))


def refs_of(text: str) -> list[str]:
    """The ids of the list lines of a block or a search result, items and turns:
    ``["m12", "t34"]``."""
    return re.findall(r"^\[([mt]\d+)\] ", text, re.MULTILINE)


# ── The default is unchanged ──


async def test_the_default_source_is_turns_and_items_in_the_index_change_nothing(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    assert cfg.memory.recall_source == "turns"
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
        assert memory.recall_settings().items
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
    long = "The user stakes the tomato garden with bamboo. " + "It rains. " * 300
    cfg = config(recall_source="items")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        session_id, _ = await add_turn(
            memory, store, TURN[0], TURN[1], at=time.time() - 3 * DAY, title="Garden"
        )
        await put(memory, session_id, TITLE, long)
        found = await ask(memory, store)
        (line,) = found.block.split("\n")[3:]
        assert line.startswith(f"- “{TITLE}”: The user stakes the tomato garden")
        assert len(line) < ITEM_CHARS + len(TITLE) + 60 and "…" in line
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
            f"{ITEM_LIST_GUIDE} {PREMISE_GUIDE}",
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
            f"{ITEM_LIST_GUIDE} {PREMISE_GUIDE}",
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
        assert ids["description"] == "Ids from the list, such as m12 or t123."
        # The item comes with the ids of its turns, not the turns.
        assert "and the ids of the turns it came from" in opened.definition.description
        assert "the earlier chat turns" not in opened.definition.description
        assert opened.definition.description.isascii()
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
        assert reply == (
            f"[m{fill.item}] {when} · chat “Garden”\n{TITLE}\n{TEXT}\n"
            f"From turn t{fill.doc}: open it for what was said word for word."
        )
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
        assert OPEN_USAGE_ITEMS == "Pass 1 to 5 ids from the list, such as m12 or t123."
        reply = await tools.open_memory(ids=["m99999", "t99999", "r1", "nope"])
        assert reply.split("\n\n") == [
            f"m99999: {NO_SUCH_TURN}",
            f"t99999: {NO_SUCH_TURN}",
            "r1: that is a note, which is shown in full already.",
            "nope: not an id from the list; ids look like m12 or t123.",
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


# ── An opened item names the turns it came from ──


async def chat_of_turns(
    memory: ChatMemory,
    store: SessionStore,
    answers: list[str],
    *,
    session_id: str | None = None,
    at: float,
) -> tuple[str, list[list[int]]]:
    """Index one turn for each of *answers* (turn numbers from 1, a minute apart) in
    a chat; its id and, for each turn, its documents (chunk 0 first)."""
    docs: list[list[int]] = []
    for number, answer in enumerate(answers, 1):
        session_id, ids = await add_turn(
            memory,
            store,
            f"Which tomato garden question {number}?",
            answer,
            at=at + 60 * number,
            session_id=session_id,
            turn=number,
            title="G",
        )
        docs.append(ids)
    assert session_id is not None
    return session_id, docs


def pointer_of(reply: str) -> str:
    """The last line of an opened item."""
    return reply.rsplit("\n", 1)[-1]


@pytest.mark.parametrize("source", ["items", "auto"])
async def test_an_opened_item_names_its_source_turns_and_each_id_opens_the_turn(
    tmp_path: Path, store: SessionStore, source: str
) -> None:
    cfg = (
        auto_config()
        if source == "auto"
        else config(recall_source="items", recall_form="list")
    )
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        tail = "The sealant is the grey one."
        long_answer = "Bed one faces south all day. " * 80 + tail
        session_id, (first, second) = await chat_of_turns(
            memory, store, ["Bed two is shady.", long_answer], at=when
        )
        assert len(second) > 1  # the second turn is several documents
        number = await put(
            memory, session_id, TITLE, TEXT, turns=(1, 2), at=when, encoder=ENCODER
        )
        tools = MemoryTools(memory)
        if source == "auto":  # every turn is read: the chat reads items
            assert await memory.reading() == Reading(True, 1.0)
        assert (await tools.start())[2].definition == OPEN_MEMORY_ITEMS
        reply = await tools.open_memory(ids=[f"m{number}"])
        # Ids only, in time order; a turn's head names it.
        assert reply == (
            f"[m{number}] {day(when)} · chat “G”\n{TITLE}\n{TEXT}\n"
            f"From turns t{first[0]}, t{second[0]}: "
            "open one for what was said word for word."
        )
        assert "Bed two" not in reply and tail not in reply
        # Each named id opens the whole turn, though the item stands for it.
        opened = await tools.open_memory(ids=[f"t{first[0]}", f"t{second[0]}"])
        short, whole = opened.split("\n\n")
        assert short.startswith(f"[t{first[0]}] {day(when)} · chat “G”\n")
        assert "Which tomato garden question 1?" in short and "Bed two" in short
        assert "Which tomato garden question 2?" in whole and tail in whole
        # The lookup names heads and reads no text; a turn not indexed has none.
        heads = await memory.stores()[1].turn_heads(
            [(session_id, 2), (session_id, 9), (session_id, 1)]
        )
        assert [head.id for head in heads] == [second[0], first[0]]
    finally:
        await memory.aclose()


async def test_the_pointer_leaves_out_turns_the_chat_would_not_show(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        info = await _turn(store, "G")
        when = time.time() - 3 * DAY
        session_id, docs = await chat_of_turns(
            memory,
            store,
            ["One.", "Two.", "Three."],
            session_id=info.session_id,
            at=when,
        )
        # Turn 9 was never indexed, so it is left out.
        number = await put(memory, session_id, TITLE, TEXT, turns=(1, 2, 3, 9), at=when)
        tools = MemoryTools(memory)
        await memory.before_turn(store, info, QUESTION)
        chat = CHAT_TURN.get()
        assert chat is not None
        reply = await tools.open_memory(ids=[f"m{number}"])
        assert pointer_of(reply).startswith(
            f"From turns t{docs[0][0]}, t{docs[1][0]}, t{docs[2][0]}: "
        )
        # The chat still shows turns 3 and up, so they are not named.
        chat.keep_from_turn = 3
        reply = await tools.open_memory(ids=[f"m{number}"])
        assert pointer_of(reply) == (
            f"From turns t{docs[0][0]}, t{docs[1][0]}: "
            "open one for what was said word for word."
        )
        chat.keep_from_turn = 2
        reply = await tools.open_memory(ids=[f"m{number}"])
        assert pointer_of(reply) == (
            f"From turn t{docs[0][0]}: open it for what was said word for word."
        )
        # Another chat's view leaves nothing out.
        found = await memory.open_item(number)
        assert not isinstance(found, str)
        assert await memory.source_turns(
            found, exclude_session="elsewhere", keep_from_turn=0
        ) == [ids[0] for ids in docs]
        # No source left to name: no line (outside the chat's own view).
        CHAT_TURN.set(None)
        none = await put(memory, session_id, TITLE, TEXT, turns=(8, 9), at=when)
        reply = await tools.open_memory(ids=[f"m{none}"])
        assert reply.endswith(f"\n{TITLE}\n{TEXT}") and "From turn" not in reply
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_the_pointer_names_the_newest_turns_when_there_are_many(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        count = service.OPEN_AT_ONCE + 2
        session_id, docs = await chat_of_turns(
            memory, store, [f"Answer {n}." for n in range(count)], at=when
        )
        # Cited out of order: the line is still in time order.
        number = await put(
            memory,
            session_id,
            TITLE,
            TEXT,
            turns=tuple(reversed(range(1, count + 1))),
            at=when,
        )
        named = ", ".join(f"t{ids[0]}" for ids in docs[-service.OPEN_AT_ONCE :])
        reply = await MemoryTools(memory).open_memory(ids=[f"m{number}"])
        assert pointer_of(reply) == (
            f"From turns {named}: open one for what was said word for word."
        )
    finally:
        await memory.aclose()


async def test_an_item_fits_recall_open_chars_with_its_pointer(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list", recall_open_chars=1000)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        session_id, docs = await chat_of_turns(memory, store, ["One."], at=when)
        sentence = "The user stakes the tomato plants with bamboo. "
        number = await put(
            memory, session_id, TITLE, sentence * 20, turns=(1,), at=when
        )
        reply = await MemoryTools(memory).open_memory(ids=[f"m{number}"])
        assert len(reply) <= 1000 and "… (cut)" in reply
        assert pointer_of(reply) == (
            f"From turn t{docs[0][0]}: open it for what was said word for word."
        )
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
        assert past.where.startswith("this chat, ")
        other = await memory.open_item(
            number, exclude_session="elsewhere", keep_from_turn=0
        )
        assert not isinstance(other, str) and other.where.startswith("chat “")
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
        await relate(index, replace=(old, new))
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


async def book_club(
    memory: ChatMemory, store: SessionStore, said: str
) -> tuple[int, int, int]:
    """Two memory items about one book club, the first (turn 2 of chat A) replaced by
    the second (turn 2 of chat C), and a note of chat A's turn 2 that says *said*: the
    numbers of the new item and of the note, then of the old item."""
    now = time.time()
    a, _ = await add_turn(memory, store, "Club?", "Noted.", at=now - 5 * DAY, title="A")
    c, _ = await add_turn(
        memory, store, "Moved?", "Noted.", at=now - 2 * DAY, title="C"
    )
    old = await put(
        memory, a, "Club day", "The book club meets on Tuesdays.", turns=(2,)
    )
    new = await put(
        memory, c, "Club day", "The book club now meets on Thursdays.", turns=(2,)
    )
    await relate(memory.stores()[1], replace=(old, new))
    kept = await note(
        memory, said, "My book club meets on Tuesdays.", "about", session_id=a, turn=2
    )
    return new, kept, old


async def test_a_note_of_a_replaced_items_turn_reads_as_older_and_stays_in_the_block(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        new, kept, old = await book_club(
            memory, store, "Your book club meets on Tuesdays."
        )
        found = await ask(memory, store, "when does the book club meet?")
        # The newer item first, then the note, marked; the old item is left out.
        assert [i.key for i in found.used] == [f"m:{new}", f"r:{kept}"]
        newer, said = (
            line for line in found.block.split("\n") if line.startswith("- ")
        )
        assert "Thursdays" in newer and "older" not in newer
        assert "Tuesdays" in said and said.endswith(f"({OLDER_ITEM})")
        # The search tool lists the same marks, on the old item and on the note.
        reply = await MemoryTools(memory).search(query="when does the book club meet?")
        lines = reply.split("\n")
        assert len(lines) == 3 and "older" not in lines[0]
        assert all(line.endswith(f"({OLDER_ITEM})") for line in lines[1:])
        assert sum("the user said" in line for line in lines[1:]) == 1
        # Asked about the past too, the old item joins the note, both marked.
        past = await ask(memory, store, "when did the book club meet before?")
        assert past.used[0].key == f"m:{new}"
        assert {i.key for i in past.used[1:]} == {f"m:{old}", f"r:{kept}"}
        assert past.block.count(f"({OLDER_ITEM})") == 2
    finally:
        await memory.aclose()


async def test_the_newer_item_comes_in_when_the_search_finds_only_the_note(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        new, kept, _ = await book_club(
            memory, store, "Your book club meets on Tuesdays in the library hall."
        )
        found = await ask(memory, store, "which library hall?")
        assert [i.key for i in found.used] == [f"m:{new}", f"r:{kept}"]
        assert "Thursdays" in found.block and found.block.endswith(f"({OLDER_ITEM})")
    finally:
        await memory.aclose()


async def test_an_item_whose_turns_a_note_replaced_is_left_out_unless_asked_about_the_past(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        now = time.time()
        # The turn shares no word with the questions below: a turn that did would
        # take one of the two turn places of the block from the item the past needs.
        a, _ = await add_turn(
            memory, store, "Planning?", "Noted.", at=now - 5 * DAY, title="A"
        )
        c, _ = await add_turn(
            memory, store, "Moved?", "Noted.", at=now - 2 * DAY, title="C"
        )
        # Chat C's note replaces chat A's: it replaced what chat A's turn 3 said.
        await note(
            memory,
            "Your weekly sync is on Wednesday.",
            "",
            "about",
            session_id=a,
            turn=3,
        )
        kept = await note(
            memory, "Your weekly sync is on Friday.", "", "about", session_id=c, turn=1
        )
        only = await put(
            memory, a, "Sync day", "The weekly sync is on Wednesday.", turns=(3,)
        )
        # The second item cites a turn that still stands, so it stays.
        both = await put(
            memory, a, "Sync room", "The weekly sync is in room 4.", turns=(3, 4)
        )
        found = await ask(memory, store, "when is the weekly sync?")
        assert {i.key for i in found.used} == {f"r:{kept}", f"m:{both}"}
        assert "older" not in found.block
        past = await ask(memory, store, "when was the weekly sync before?")
        assert [i.key for i in past.used][-1] == f"m:{only}"
        assert past.block.endswith(f"({OLDER})")
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
        await relate(index, link=(never, like), conflict=True)
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
        await relate(index, replace=(first, second))
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
    cfg = config(recall_source="items", recall_form="list")
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


# ── Turns not encoded yet ──


@pytest.mark.parametrize("form", ["list", "excerpt"])
async def test_with_nothing_encoded_recall_reads_what_the_turns_source_reads(
    tmp_path: Path, store: SessionStore, form: str
) -> None:
    cfg = config(recall_form=form)
    fill = await filled(tmp_path, store, cfg)
    memory = fill.memory
    try:
        before = await ask(memory, store)
        search = await MemoryTools(memory).search(query=QUESTION)
        assert any(i.is_record for i in before.used)
        assert any(i.is_turn for i in before.used)
        cfg.memory.recall_source = "items"
        after = await ask(memory, store)
        assert after.block == before.block
        assert [i.key for i in after.used] == [i.key for i in before.used]
        assert await MemoryTools(memory).search(query=QUESTION) == search
        assert not memory._corpus.keys("m:") and len(memory._corpus.keys("d:")) == 3
        assert (LIST_GUIDE in after.block) == (form == "list")
        assert ITEM_LIST_GUIDE not in after.block
    finally:
        await memory.aclose()


@pytest.mark.parametrize("form", ["list", "excerpt"])
async def test_a_half_encoded_chat_recalls_its_later_turns_beside_its_items(
    tmp_path: Path, store: SessionStore, form: str
) -> None:
    cfg = config(recall_source="items", recall_form=form)
    fill = await held(tmp_path, store, cfg, later=LATER)
    memory = fill.memory
    try:
        found = await ask(memory, store)
        assert {i.key for i in found.used} == {f"m:{fill.item}", f"d:{fill.later}"}
        reply = await MemoryTools(memory).search(query=QUESTION)
        # The earlier turn is in the item; only the later one is listed or shown.
        if form == "list":
            assert sorted(refs_of(found.block)) == sorted(refs_of(reply))
            assert sorted(refs_of(reply)) == [f"m{fill.item}", f"t{fill.later}"]
            assert ITEM_LIST_GUIDE in found.block and LIST_GUIDE not in found.block
            assert reply.split("\n")[0].startswith(ITEM_LIST_GUIDE[:20])
        else:
            for text in (found.block, reply):
                assert TEXT in text and LATER[0] in text and TURN[0] not in text
        assert f"d:{fill.doc}" not in memory._corpus.items
    finally:
        await memory.aclose()


async def test_a_list_of_turns_and_items_has_one_guide_line_with_the_premise_sentence(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg, later=LATER)
    memory = fill.memory
    try:
        found = await ask(memory, store)
        guide = f"{ITEM_LIST_GUIDE} {PREMISE_GUIDE}"
        assert found.block.count(guide) == 1 and LIST_GUIDE not in found.block
        reply = await MemoryTools(memory).search(query=QUESTION)
        assert reply.count(guide) == 1 and LIST_GUIDE not in reply
    finally:
        await memory.aclose()


async def test_a_turn_leaves_recall_when_an_item_cites_it_not_when_a_mark_passes_it(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg, later=LATER)
    memory = fill.memory
    try:
        assert f"t{fill.later}" in refs_of((await ask(memory, store)).block)
        _, index = memory.stores()
        # The encoder read turn 2 and made nothing of it: a mark says a turn was
        # read, not that anything stands for it, so it stays recalled.
        await index.mark_encoded(fill.session_id, 2)
        found = await ask(memory, store)
        assert {i.key for i in found.used} == {f"m:{fill.item}", f"d:{fill.later}"}
        # An item that cites it takes it out at the next sync.
        await put(memory, fill.session_id, "Pot", "Pot two needs water.", turns=(2,))
        await ask(memory, store)
        assert not memory._corpus.keys("d:")
    finally:
        await memory.aclose()


async def test_a_cited_turn_is_not_recalled_whatever_the_chats_mark_says(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg, encoded=False)
    memory = fill.memory
    try:
        assert await memory.stores()[1].encoded_turn(fill.session_id) is None
        found = await ask(memory, store)
        assert {i.key for i in found.used} == {f"m:{fill.item}"}
    finally:
        await memory.aclose()


async def test_a_summary_stays_recalled_when_its_chat_has_a_mark(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        await memory.side_summary(
            fill.session_id, "Garden", "The tomato garden is staked with bamboo."
        )
        _, index = memory.stores()
        assert await index.encoded_turn(fill.session_id) == 1
        found = await ask(memory, store)
        (summary,) = (i for i in found.used if i.is_turn)
        # The summary (turn -1) is recalled beside the item; the encoded turn is not.
        assert summary.turn == SUMMARY_TURN and summary.key != f"d:{fill.doc}"
        assert {i.key for i in found.used} == {f"m:{fill.item}", summary.key}
        assert f"d:{fill.doc}" not in memory._corpus.items
    finally:
        await memory.aclose()


async def test_open_memory_opens_an_item_and_a_turn_in_one_call(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg, later=LATER)
    memory = fill.memory
    try:
        when = day(time.time() - 3 * DAY)
        reply = await MemoryTools(memory).open_memory(
            ids=[f"m{fill.item}", f"T{fill.later}", f"t{fill.doc}"]
        )
        item, turn, encoded = reply.split("\n\n")
        assert item == (
            f"[m{fill.item}] {when} · chat “Garden”\n{TITLE}\n{TEXT}\n"
            f"From turn t{fill.doc}: open it for what was said word for word."
        )
        assert turn.startswith(f"[t{fill.later}] {when} · chat “Garden”\n")
        assert LATER[0] in turn and LATER[1] in turn
        # An encoded turn is no longer listed, but the index has it, so it opens.
        assert encoded.startswith(f"[t{fill.doc}] {when} · chat “Garden”\n")
        assert TURN[0] in encoded and TURN[1] in encoded
        # Without the items source a turn id still opens a turn and an item id does not.
        cfg.memory.recall_source = "turns"
        again = await MemoryTools(memory).open_memory(
            ids=[f"m{fill.item}", f"t{fill.doc}"]
        )
        assert again.split("\n\n")[0] == (
            f"m{fill.item}: not an id from the list; ids look like t123."
        )
        assert again.split("\n\n")[1] == encoded  # the turn path is the same
    finally:
        await memory.aclose()


async def test_the_event_names_items_and_turns_used_searched_and_opened(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg, later=LATER)
    memory = fill.memory
    try:
        emitted: list[dict[str, Any]] = []
        await memory.before_turn(
            store, await _turn(store, QUESTION), QUESTION, emit=emitted.append
        )
        chat = CHAT_TURN.get()
        assert chat is not None
        data = chat.event()
        assert data is not None
        assert {(u["kind"], u["id"]) for u in data["used"]} == {
            ("item", fill.item),
            ("chat", fill.later),
        }
        tools = MemoryTools(memory)
        await tools.search(query=QUESTION)
        assert {(s["kind"], s["id"]) for s in emitted[-1]["searched"]} == {
            ("item", fill.item),
            ("chat", fill.later),
        }
        await tools.open_memory(ids=[f"t{fill.later}", f"m{fill.item}"])
        assert emitted[-1]["opened"] == [
            {"session_id": fill.session_id, "turn": 2, "id": fill.later},
            {"item": fill.item, "sources": [[fill.session_id, 1]]},
        ]
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_forgetting_one_turn_keeps_the_chats_mark(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg, later=LATER)
    memory = fill.memory
    try:
        _, index = memory.stores()
        await memory.forget_session(fill.session_id, 2)
        assert await index.encoded_turn(fill.session_id) == 1
        assert [i.key for i in (await ask(memory, store)).used] == [f"m:{fill.item}"]
    finally:
        await memory.aclose()


async def test_forgetting_one_source_of_an_item_drops_it_and_frees_the_other_turn(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg, later=LATER, turns=(1, 2))
    memory = fill.memory
    try:
        found = await ask(memory, store)
        assert [i.key for i in found.used] == [f"m:{fill.item}"]
        _, index = memory.stores()
        await memory.forget_session(fill.session_id, 2)
        # The item rested on both turns: it goes whole, with the text it drew from
        # the forgotten one. The turn that is left is recalled as a turn again, not
        # encoded again.
        assert await index.count_items() == 0
        assert await index.encoded_turn(fill.session_id) == 2
        for text in (
            (await ask(memory, store)).block,
            await MemoryTools(memory).search(query=QUESTION),
        ):
            assert f"t{fill.doc}" in text and TURN[0] in text
            assert TEXT not in text and LATER[0] not in text and LATER[1] not in text
    finally:
        await memory.aclose()


HOME = "The user said they live in Pittsburgh."
MOVED = "The user said they now live in Utrecht."


async def repeated(
    tmp_path: Path, store: SessionStore, cfg: Any
) -> tuple[ChatMemory, Router, str, str]:
    """A memory in which a first chat states a fact and a second chat repeats it: the
    router keeps one item, which cites the turns of both chats."""
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        router = Router(memory.stores()[1])
        sessions = []
        for n, typed in enumerate(
            ("I live in Pittsburgh.", "Yes, I live in Pittsburgh.")
        ):
            when = time.time() - (4 - n) * DAY
            session_id, _ = await add_turn(
                memory, store, typed, "Noted, Pittsburgh.", at=when, title=f"Chat {n}"
            )
            sessions.append(session_id)
            item = MemoryItem(
                session_id,
                "Home",
                HOME,
                "profile",
                when,
                ((session_id, 1),),
                encoder="1:x",
            )
            assert (await router.route(item)).action == ("NEW" if n == 0 else "IGNORE")
        first, second = sessions
        (kept,) = await memory.stores()[1].fetch_items([1])
        assert kept.sources == ((first, 1), (second, 1))
    except BaseException:
        await memory.aclose()
        raise
    return memory, router, first, second


async def test_a_repeated_fact_that_is_replaced_leaves_no_stale_turn(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory, router, _, _ = await repeated(tmp_path, store, cfg)
    try:
        third, _ = await add_turn(
            memory, store, "I now live in Utrecht.", "Utrecht.", at=time.time() - DAY
        )
        moved = MemoryItem(
            third, "Home", MOVED, "profile", time.time() - DAY, ((third, 1),)
        )
        near = await router.shortlist(moved)
        assert (
            await router.route(moved, Judged("changes", near[0].id, near))
        ).action == ("UPDATE")
        found = await ask(memory, store, "Where do I live now?")
        # The repeat's turn is behind the replaced item: it is not a second, current
        # statement of the old fact beside the new one.
        assert [i.key for i in found.used] == ["m:2"]
        assert not memory._corpus.keys("d:")
        assert "Utrecht" in found.block and "Pittsburgh" not in found.block
    finally:
        await memory.aclose()


SYNC_TURN = (
    "Heads up, Wednesday is when our weekly team sync happens.",
    "Noted, Wednesday it is.",
)
SYNC_NOW = "The weekly team sync is now on Thursday."
SYNC_BEFORE = "The weekly team sync is on Wednesday."


@pytest.mark.parametrize(
    "variant",
    [
        {"recall_source": "items"},
        {"recall_source": "items", "recall_form": "list"},
        {"recall_source": "auto"},
    ],
    ids=["items", "items-list", "auto"],
)
async def test_a_turn_cited_by_a_replaced_item_is_not_recalled_as_a_turn_on_any_path(
    tmp_path: Path, store: SessionStore, variant: dict[str, str]
) -> None:
    """Item 1 cites chat A's turn and item 2, from chat B, replaces it through the
    router. A new chat that asks about both finds the items, never A's turn: not in the
    block, nor in a search, nor in the corpus, whatever the question (a past one
    included)."""
    encoder = ENCODER if variant["recall_source"] == "auto" else "1:stub"
    cfg = auto_config() if variant["recall_source"] == "auto" else config(**variant)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        _, index = memory.stores()
        router = Router(index)
        items: list[int] = []
        for n, (typed, answer, text) in enumerate(
            (
                (*SYNC_TURN, SYNC_BEFORE),
                ("It moved to Thursday.", "Thursday, noted.", SYNC_NOW),
            )
        ):
            when = time.time() - (4 - 2 * n) * DAY
            session_id, _ = await add_turn(
                memory, store, typed, answer, at=when, title=f"Sync {n}"
            )
            item = MemoryItem(
                session_id,
                "Sync day",
                text,
                "episodic",
                when,
                ((session_id, 1),),
                encoder=encoder,
            )
            near = await router.shortlist(item)
            route = await router.route(item, Judged("changes", 1, near) if n else None)
            assert route.action == ("UPDATE" if n else "NEW") and route.item
            await index.mark_encoded(session_id, 1)
            items.append(route.item)
        old, new = items
        older, newer = await index.fetch_items([old, new])
        assert (older.status, older.replaced_by, newer.status) == (
            "replaced",
            new,
            "active",
        )
        assert await index.ids(uncited=True) == set()
        for question, keys in (
            ("when is the weekly team sync?", [f"m:{new}"]),
            ("all my weekly team sync before", None),
        ):
            found = await ask(memory, store, question)
            assert found.recall and found.recall.gather == (keys is None)
            assert found.used and all(i.key.startswith("m:") for i in found.used)
            if keys is not None:
                assert [i.key for i in found.used] == keys
            else:
                assert [i.key for i in found.used] == [f"m:{new}", f"m:{old}"]
            assert "Noted, Wednesday it is" not in found.block
            reply = await MemoryTools(memory).search(query="weekly team sync")
            assert "Noted, Wednesday it is" not in reply
            assert not [ref for ref in refs_of(reply) if ref.startswith("t")]
            assert SYNC_NOW in reply or f"m{new}" in reply
        for corpus in (memory._corpus, memory._auto_items):
            assert not corpus.keys("d:")
    finally:
        await memory.aclose()


async def test_forgetting_the_first_chat_of_a_repeated_fact_brings_the_second_back(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    memory, _, first, _ = await repeated(tmp_path, store, cfg)
    try:
        _, index = memory.stores()
        before = await ask(memory, store, "Where do I live?")
        assert [i.key for i in before.used] == ["m:1"]
        assert not memory._corpus.keys("d:")
        await memory.forget_session(first)
        # The item goes whole with its first source; the second chat's turn, which it
        # also cited, is recalled as a turn, so the fact is still found.
        assert await index.count_items() == 0
        (doc,) = await index.ids()
        found = await ask(memory, store, "Where do I live?")
        assert [i.key for i in found.used] == [f"d:{doc}"]
        assert "Pittsburgh" in found.block
    finally:
        await memory.aclose()


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
        assert [await index.encoded_turn(s) for s in (first, second)] == [1, 1]
        await memory.forget_session(second)
        assert [i.id for i in await index.fetch_items([kept, gone])] == [kept]
        assert [await index.encoded_turn(s) for s in (first, second)] == [1, None]
        await memory.forget_all()
        assert await index.count_items() == 0
        assert await index.encoded_turn(first) is None
        assert not memory._corpus.keys("m:")
        assert (await ask(memory, store, "tomatoes in bed one")).block == ""
    finally:
        await memory.aclose()


async def test_a_deleted_item_leaves_recall_at_once_its_turn_returns_and_an_undo_reverses_both(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_source="items")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        before = await ask(memory, store)
        assert [i.key for i in before.used] == [f"m:{fill.item}"]
        assert await memory.delete_item(fill.item)
        # The next search reads the turn the item stood for, as a turn, and not the item.
        after = await ask(memory, store)
        assert [i.key for i in after.used] == [f"d:{fill.doc}"]
        assert TITLE not in after.block and "Bed one faces south" in after.block
        assert not await memory.delete_item(fill.item), "a second delete finds nothing"
        assert await memory.restore_item(fill.item)
        again = await ask(memory, store)
        assert again.block == before.block
        assert [i.key for i in again.used] == [f"m:{fill.item}"]
        assert not await memory.restore_item(fill.item), "an undo is used once"
    finally:
        await memory.aclose()


async def test_deleting_the_item_that_replaced_another_does_not_bring_the_older_fact_back(
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
            "Sync day",
            "The weekly sync is on Wednesday.",
            at=now - 4 * DAY,
        )
        new = await put(
            memory,
            session_id,
            "Sync day",
            "The weekly sync is now on Friday.",
            at=now - 2 * DAY,
        )
        await relate(memory.stores()[1], replace=(old, new))
        assert await memory.delete_item(new)
        # The Wednesday fact is still the older one: it is not the current answer ...
        found = await ask(memory, store, "when is the weekly sync?")
        assert not found.used and "Friday" not in found.block
        # ... and only a question about the past finds it, marked older.
        past = await ask(memory, store, "when was the weekly sync before?")
        assert [i.key for i in past.used] == [f"m:{old}"]
        assert f"({OLDER_ITEM})" in past.block
        # An undo makes the newer fact the answer again.
        assert await memory.restore_item(new)
        back = await ask(memory, store, "when is the weekly sync?")
        assert [i.key for i in back.used] == [f"m:{new}"]
    finally:
        await memory.aclose()


async def test_an_undo_runs_out_and_forgetting_the_chat_or_everything_ends_it(
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
        bed = await put(memory, first, "Bed", "The user plants tomatoes in bed one.")
        pot = await put(memory, second, "Pot", "The user keeps basil in pot two.")
        # Time runs out.
        assert await memory.delete_item(bed)
        with mock.patch.object(service, "ITEM_UNDO_S", 0):
            assert not await memory.restore_item(bed)
        assert not await memory.delete_item(bed), "an expired item is gone for good"
        # Forgetting another chat leaves an undo; forgetting a turn the item cites ends it.
        assert await memory.delete_item(pot)
        await memory.forget_session(first)
        assert await memory.restore_item(pot)
        assert await memory.delete_item(pot)
        await memory.forget_session(second, 1)
        assert not await memory.restore_item(pot)
        # Forgetting everything ends every undo.
        assert await memory.delete_item(
            await put(memory, first, "Bed", "Tomatoes again.")
        )
        await memory.forget_all()
        assert not memory._trash
        assert await memory.stores()[1].count_items() == 0
    finally:
        await memory.aclose()


class PlantEmbedding(BaseEmbeddingProvider):
    """A second embedding model: text about plants (tomato, vegetable) is
    ``[1, 0, 0]``, any other text ``[0, 0, 1]``. With *fail*, it raises."""

    model = "new-embed"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("the embedding server is down")
        return [
            [1.0, 0.0, 0.0] if re.search("tomato|vegetable", t) else [0.0, 0.0, 1.0]
            for t in texts
        ]

    async def close(self) -> None:
        return None


async def test_a_reindex_gives_the_items_vectors_of_the_new_embedding_model(
    tmp_path: Path, store: SessionStore, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = config(recall_source="items", retriever="embedding")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        session_id, _ = await add_turn(
            memory, store, TURN[0], TURN[1], at=time.time() - 3 * DAY, title="Garden"
        )
        number = await put(memory, session_id, TITLE, TEXT, cosine=0.0)
        provider = PlantEmbedding()
        memory.embedding = lambda: (provider, provider.model)  # type: ignore[method-assign]
        _, index = memory.stores()
        # A question that shares no word with the item, under the new model: the old
        # vector is of another model, so it is not compared.
        paraphrase = "How do I prop up my vegetable beds?"
        assert not [i for i in await memory.search(paraphrase) if i.is_memory]
        # A model that fails leaves the old vector, and the log says how many.
        provider.fail = True
        await memory.reindex(None)
        (kept,) = await index.fetch_items([number])
        assert kept.embed_model == MODEL
        assert any("1 memory items kept" in line for line in caplog.messages)
        provider.fail = False
        await memory.reindex(None)
        (stored,) = await index.fetch_items([number])
        assert (stored.embed_model, stored.vector) == ("new-embed", (1.0, 0.0, 0.0))
        found = await memory.search(paraphrase)
        assert [i.key for i in found if i.is_memory] == [f"m:{number}"]
    finally:
        await memory.aclose()


# ── recall_source = "auto": turns until the encoder has caught up ──

CHECKED = next(iter(hippo_encode.CHECKED_ENCODERS))
ENCODER = f"3:{CHECKED}"
# A chat model's address that is never called: only its name is read.
STUB_URL = "http://127.0.0.1:59999/v1"
NEWER = "The user now props the plants with cedar poles."


def auto_config(**memory: Any) -> Config:
    """``recall_source = "auto"`` with a chat model on the checked list."""
    cfg = config(recall_source="auto", **memory)
    cfg.vllm.chat_model = CHECKED
    return cfg


async def catch_up(fill: Held, store: SessionStore, turns: int, read: int) -> None:
    """Give the chat *turns* turns in all, *read* of them read by the encoder (the
    first is already)."""
    for turn in range(2, turns + 1):
        await add_turn(
            fill.memory,
            store,
            f"Which tomato garden bed {turn}?",
            f"Bed {turn}.",
            session_id=fill.session_id,
            turn=turn,
            title="Garden",
        )
    await fill.memory.stores()[1].mark_encoded(fill.session_id, read)


async def test_auto_reads_turns_until_the_encoder_has_caught_up_and_then_items(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = auto_config()  # recall_form stays at excerpt
    fill = await held(tmp_path, store, cfg, later=LATER, encoder=ENCODER)
    memory, index = fill.memory, fill.memory.stores()[1]
    try:
        assert await index.encoded_share() == 0.5
        auto = await ask(memory, store)
        search = await MemoryTools(memory).search(query=QUESTION)
        cfg.memory.recall_source = "turns"
        plain = await ask(memory, store)
        assert auto.block == plain.block != ""
        assert [i.key for i in auto.used] == [i.key for i in plain.used]
        assert await MemoryTools(memory).search(query=QUESTION) == search
        assert not has_item_line(auto.block) and TITLE not in auto.block
        # The encoder reads the last turn: items, in the list form.
        await index.mark_encoded(fill.session_id, 2)
        cfg.memory.recall_source = "auto"
        auto = await ask(memory, store)
        search = await MemoryTools(memory).search(query=QUESTION)
        cfg.memory.recall_source, cfg.memory.recall_form = "items", "list"
        plain = await ask(memory, store)
        assert auto.block == plain.block != ""
        assert refs_of(auto.block) == [f"m{fill.item}", f"t{fill.later}"]
        assert ITEM_LIST_GUIDE in auto.block
        assert await MemoryTools(memory).search(query=QUESTION) == search
    finally:
        await memory.aclose()


@pytest.mark.parametrize(
    ("turns", "read", "items"),
    [(10, 9, True), (20, 18, True), (19, 17, False), (10, 8, False), (1, 1, True)],
)
async def test_the_switch_is_at_nine_tenths_of_the_turns_read_with_no_rounding(
    tmp_path: Path, store: SessionStore, turns: int, read: int, items: bool
) -> None:
    fill = await held(tmp_path, store, auto_config(), encoder=ENCODER)
    try:
        await catch_up(fill, store, turns, read)
        assert CAUGHT_UP == 0.9
        assert await fill.memory.reading() == Reading(items, read / turns)
        found = await ask(fill.memory, store)
        assert found.recall is not None
        assert found.recall.source == ("items" if items else "turns")
    finally:
        await fill.memory.aclose()


@pytest.mark.parametrize(
    ("encoder", "items"),
    [
        ("3:small-model", False),
        ("", False),
        ("3:", False),
        (f":{CHECKED}", False),  # no version
        (ENCODER, True),
        (hippo_encode.RULES_ENCODER, True),
    ],
)
async def test_a_stored_item_an_unchecked_encoder_wrote_keeps_recall_on_turns(
    tmp_path: Path, store: SessionStore, encoder: str, items: bool
) -> None:
    fill = await held(tmp_path, store, auto_config(), encoder=encoder)
    memory = fill.memory
    try:
        assert await memory.reading() == Reading(items, 1.0)  # every turn is read
        if not items:
            # Until a person deletes it: the check is on what would be read.
            assert await memory.stores()[1].delete_item(fill.item) is not None
            assert await memory.reading() == Reading(True, 1.0)
    finally:
        await memory.aclose()


@pytest.mark.parametrize(
    ("encoder", "opens"),
    [
        ("3:small-model", False),
        ("", False),
        (f":{CHECKED}", False),
        (ENCODER, True),
        (hippo_encode.RULES_ENCODER, True),
    ],
)
async def test_auto_opens_an_item_only_when_its_encoder_is_checked(
    tmp_path: Path, store: SessionStore, encoder: str, opens: bool
) -> None:
    cfg = auto_config()
    fill = await held(tmp_path, store, cfg, encoder=encoder)
    memory = fill.memory
    try:
        await catch_up(fill, store, 5, 1)  # the chat reads turns now
        assert (await memory.reading()) == Reading(False, 0.2)
        reply = await memory.open_item(fill.item)
        said = await MemoryTools(memory).open_memory(ids=[f"m{fill.item}"])
        if opens:
            assert not isinstance(reply, str) and TEXT in said
        else:
            # The same answer as an id nothing has.
            assert reply == NO_SUCH_TURN == await memory.open_item(fill.item + 100)
            assert said == f"m{fill.item}: {NO_SUCH_TURN}"
        # The explicit setting opens every item, as it always did.
        cfg.memory.recall_source = "items"
        assert not isinstance(await memory.open_item(fill.item), str)
    finally:
        await memory.aclose()


async def test_the_tools_register_the_turns_state_when_the_state_cannot_be_read(
    tmp_path: Path,
    store: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fill = await held(tmp_path, store, auto_config(), encoder=ENCODER)
    memory = fill.memory
    try:
        # Every turn is read, so a working read would give the items state.
        assert (await memory.reading()) == Reading(True, 1.0)

        async def broken() -> Reading:
            raise sqlite3.OperationalError("disk I/O error on a private path")

        monkeypatch.setattr(memory, "reading", broken)
        tools = await MemoryTools(memory).start()
        names = [t.definition.name for t in tools]
        assert names == ["search_memory", "remember", "open_memory"]
        assert tools[0].definition == search_memory_opening()
        assert tools[2].definition == OPEN_MEMORY
        # The log holds the error's class and nothing it said.
        assert any("OperationalError" in line for line in caplog.messages)
        assert not any("private path" in line for line in caplog.messages)
    finally:
        await memory.aclose()


async def test_a_replaced_item_of_an_unchecked_encoder_counts_too(
    tmp_path: Path, store: SessionStore
) -> None:
    fill = await held(tmp_path, store, auto_config(), encoder="3:small-model")
    memory = fill.memory
    try:
        newer = await put(memory, fill.session_id, "Poles", NEWER, encoder=ENCODER)
        await relate(memory.stores()[1], replace=(fill.item, newer))
        assert await memory.reading() == Reading(False, 1.0)
    finally:
        await memory.aclose()


@pytest.mark.parametrize("model", ["Qwen3.8-Flash", "", "stub-encoder"])
async def test_a_local_model_off_the_list_or_without_an_id_is_not_checked(
    tmp_path: Path, store: SessionStore, model: str
) -> None:
    cfg = auto_config()
    cfg.vllm.chat_model = model
    fill = await held(tmp_path, store, cfg, encoder=ENCODER)
    try:
        assert not fill.memory.chat_model_checked()
        assert await fill.memory.reading() == Reading(False, 1.0)
        found = await ask(fill.memory, store)
        assert found.recall is not None and found.recall.source == "turns"
    finally:
        await fill.memory.aclose()


async def test_a_hosted_backend_is_not_checked_and_a_named_server_is_local(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = auto_config()
    fill = await held(tmp_path, store, cfg, encoder=ENCODER)
    memory = fill.memory
    try:
        assert memory.chat_model_checked()
        cfg.chat_backend = CLOUD_BACKENDS[0]
        # The model has the checked id, but a hosted backend is never trusted.
        memory.chat_provider = lambda: StubEncoder(model=CHECKED)  # type: ignore[method-assign]
        assert not memory.chat_model_checked()
        assert await memory.reading() == Reading(False, 1.0)
        named = ChatMemory(
            lambda: cfg,
            directory=str(tmp_path / "named"),
            chat_url=STUB_URL,
            chat_model=CHECKED,
        )
        try:
            assert named.chat_model_checked()
        finally:
            await named.aclose()
    finally:
        await memory.aclose()


async def test_the_event_says_what_auto_read_and_how_much_the_encoder_had_read(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = auto_config()
    fill = await held(tmp_path, store, cfg, later=LATER, encoder=ENCODER)
    memory = fill.memory
    try:
        found = await ask(memory, store)
        data = found.event()
        assert data is not None
        assert data["recall"]["source"] == "turns"
        assert data["recall"]["encoded_share"] == 0.5
        await memory.stores()[1].mark_encoded(fill.session_id, 2)
        data = (await ask(memory, store)).event()
        assert data is not None
        assert data["recall"]["source"] == "items"
        assert data["recall"]["encoded_share"] == 1.0
        # With turns or items the two fields are not there at all.
        for source in ("turns", "items"):
            cfg.memory.recall_source = source
            data = (await ask(memory, store)).event()
            assert data is not None
            assert "source" not in data["recall"]
            assert "encoded_share" not in data["recall"]
    finally:
        await memory.aclose()


async def test_two_turns_can_read_in_different_states_at_once(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = auto_config(retriever="llm")
    fill = await held(tmp_path, store, cfg, encoder=ENCODER)
    memory = fill.memory
    try:
        newer = await put(memory, fill.session_id, "Poles", NEWER, encoder=ENCODER)
        await relate(memory.stores()[1], replace=(fill.item, newer))
        # The chat model answers nothing readable, and the first call waits.
        gate, entered = asyncio.Event(), asyncio.Event()
        stub = StubEncoder("no ids here", model=CHECKED)
        answer, calls = stub.chat, []

        async def chat(messages: Any, tools: Any = None) -> Any:
            calls.append(messages)
            if len(calls) == 1:
                entered.set()
                await gate.wait()
            return await answer(messages, tools)

        stub.chat = chat  # type: ignore[method-assign]
        memory.chat_provider = lambda: stub  # type: ignore[method-assign]
        first = asyncio.create_task(ask(memory, store))
        await entered.wait()
        # Every turn is read, so the first turn is in the items state and holds the
        # items it found. Now the chat gets a turn nobody has read: the next turn
        # reads turns, and its sync must not take the first one's items away.
        await add_turn(
            memory, store, *LATER, session_id=fill.session_id, turn=2, title="Garden"
        )
        second = await ask(memory, store)
        assert second.recall is not None and second.recall.source == "turns"
        assert not has_item_line(second.block)
        gate.set()
        found = await first
        assert found.recall is not None and found.recall.source == "items"
        # The item that replaced the one its words matched is what the block shows.
        assert refs_of(found.block) == [f"m{newer}"]
        assert memory._auto_items.keys("m:") == {f"m:{fill.item}", f"m:{newer}"}
        assert not memory._corpus.keys("m:")
        assert len(memory._corpus.keys("d:")) == 2
    finally:
        await memory.aclose()


async def test_one_agent_keeps_working_as_its_chat_crosses_between_the_states(
    tmp_path: Path,
    store: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)
    cfg = auto_config()
    fill = await held(tmp_path, store, cfg, later=LATER, encoder=ENCODER)
    memory = fill.memory
    index = memory.stores()[1]
    llm = FakeLLM()
    # One Agent for the whole chat: it registers the memory tools once, at its first
    # run, in the state the chat is in then.
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            attempt_policy="none",
            max_steps=4,
        ),
        tool_providers=[MemoryTools(memory)],
    )

    def called(name: str, arguments: dict[str, Any]) -> Message:
        return Message(
            "assistant", "", tool_call=ToolCall(tool_name=name, arguments=arguments)
        )

    async def turn(ids: list[str]) -> tuple[str, list[ToolDef], str, str]:
        """One turn: the recall block, the tools the Agent offered, and what the
        model's ``search_memory`` and ``open_memory`` calls read."""
        block = (await ask(memory, store)).block
        llm.responses.extend(
            [
                called("search_memory", {"query": QUESTION}),
                called("open_memory", {"ids": ids}),
                Message("assistant", "done"),
            ]
        )
        assert await agent.run(QUESTION) == "done"
        tools = llm.calls[-3]["tools"]
        searched, opened = [
            m.content for m in llm.calls[-1]["messages"] if m.role == "tool"
        ][-2:]
        return block, tools, searched, opened

    try:
        # Half the turns are read: the chat reads turns, and `open_memory` is offered.
        block, tools, searched, opened = await turn([f"t{fill.doc}"])
        registered = {t.name: t for t in tools}
        assert set(registered) == {"search_memory", "remember", "open_memory"}
        assert registered["search_memory"] == search_memory_opening()
        assert registered["open_memory"] == OPEN_MEMORY
        assert not has_item_line(block) and not has_item_line(searched)
        assert TURN[0] in opened and TURN[1] in opened
        # The encoder catches up: items, as with the list form, though the tools
        # keep their old wording. A turn the chat saw before still opens, and so
        # does an item.
        await index.mark_encoded(fill.session_id, 2)
        block, tools, searched, opened = await turn(
            [f"t{fill.doc}", f"m{fill.item}", f"t{fill.later}"]
        )
        assert tools == list(registered.values())
        assert (
            refs_of(block) == refs_of(searched) == [f"m{fill.item}", f"t{fill.later}"]
        )
        assert TURN[0] in opened and TEXT in opened and LATER[0] in opened
        # The chat gets turns nobody has read: back to turns, and the item it saw
        # still opens, beside a turn it saw.
        await catch_up(fill, store, 5, 2)
        block, tools, searched, opened = await turn([f"m{fill.item}", f"t{fill.doc}"])
        assert tools == list(registered.values())
        assert not has_item_line(block) and not has_item_line(searched)
        assert TEXT in opened and TURN[0] in opened
    finally:
        await agent.close()
        await memory.aclose()
