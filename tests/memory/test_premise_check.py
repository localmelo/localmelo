"""The premise check: one more sentence in the list form's guide line.

With ``recall_form = "list"``, the line above the list (the turns' or the memory
items') and the same line in the memory search result end with one sentence that
tells the model to say it has no record of what the question names when none of the
lines mention it; on a turn that asks to count, total or compare it follows the
gather sentence. In the excerpt form (the default form) no guide line is shown, so
every byte the model reads is what it was. The texts of both guides are pinned here
as constants so that a change to either one fails this file.
Synthetic items, words and fixed vectors, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    GATHER_GUIDE,
    ITEM_LIST_GUIDE,
    LIST_GUIDE,
    PREMISE_GUIDE,
    day,
)
from localmelo.support.memory.service import MemoryTools
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.test_gather_recall import GATHER, PLAIN, WORDS, memory_of
from localmelo.tests.memory.test_hippo_recall import TITLE, TURN, held, lines_of
from localmelo.tests.memory.test_recall_list import (
    DAY,
    QUESTION,
    add_turn,
    ask,
    config,
)

# The guides, word for word, as they always read.
TURNS_GUIDE = (
    "Older chat turns that may hold the answer, closest first. Open the ones you "
    "need with open_memory before you answer from them; a title is not the content."
)
ITEMS_GUIDE = (
    "Memory items that may hold the answer, closest first. Open the ones you "
    "need with open_memory before you answer from them; a title is not the content."
)
PREMISE = (
    "If the question names a person, thing, place or role that none of these "
    "mention, say first that you have no record of it, and do not answer it from "
    "something similar."
)


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def test_the_guides_and_the_sentence_are_pinned() -> None:
    assert LIST_GUIDE == TURNS_GUIDE
    assert ITEM_LIST_GUIDE == ITEMS_GUIDE
    assert PREMISE_GUIDE == PREMISE
    assert "\n" not in PREMISE_GUIDE and "\n" not in f"{LIST_GUIDE} {PREMISE_GUIDE}"
    assert "\n" not in f"{LIST_GUIDE} {GATHER_GUIDE} {PREMISE_GUIDE}"


# ── Memory items, list form ──


async def test_the_items_guide_line_ends_with_the_sentence(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        when = day(time.time() - 3 * DAY)
        line = f"[m{fill.item}] {when} · chat “Garden” · “{TITLE}” · words"
        found = await ask(memory, store)
        assert found.block == f"\n\n{BLOCK_HEADER}\n{ITEMS_GUIDE} {PREMISE}\n{line}"
        assert found.block.count(PREMISE) == 1
        search = await MemoryTools(memory).search(query=QUESTION)
        assert search == f"{ITEMS_GUIDE} {PREMISE}\n{line}"
        assert search.count(PREMISE) == 1
        assert lines_of(found.block) == [line]
    finally:
        await memory.aclose()


# ── Chat turns, list form ──


async def test_the_turns_guide_line_ends_with_the_sentence(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        _, (doc,) = await add_turn(
            memory, store, TURN[0], TURN[1], at=when, title="Garden"
        )
        line = f"[t{doc}] {day(when)} · chat “Garden” · “{TURN[0]}” · words"
        found = await ask(memory, store)
        assert found.block == f"\n\n{BLOCK_HEADER}\n{TURNS_GUIDE} {PREMISE}\n{line}"
        assert found.block.count(PREMISE) == 1 and ITEMS_GUIDE not in found.block
        search = await MemoryTools(memory).search(query=QUESTION)
        assert search == f"{TURNS_GUIDE} {PREMISE}\n{line}"
    finally:
        await memory.aclose()


async def test_a_block_with_no_list_has_no_guide_and_so_no_sentence(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        found = await ask(memory, store)
        assert found.block == "" and PREMISE not in found.block
        assert await MemoryTools(memory).search(query=QUESTION) == (
            "Nothing in melo's memory matches that."
        )
    finally:
        await memory.aclose()


# ── With the gather sentence ──


async def test_on_a_gathering_turn_the_order_is_guide_gather_premise(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="list")
    both = f"{LIST_GUIDE} {GATHER_GUIDE} {PREMISE_GUIDE}"
    try:
        # A gathering turn: the guide, the gather sentence, the premise sentence,
        # each once and after one space, in the block and in the search result.
        found = await ask(memory, store, GATHER)
        assert found.block.split("\n")[3] == both
        assert found.block.count(PREMISE_GUIDE) == 1
        assert found.block.count(GATHER_GUIDE) == 1
        reply = await MemoryTools(memory).search(query=WORDS)
        assert reply.split("\n")[0] == both
        assert reply.count(PREMISE_GUIDE) == 1 and reply.count(GATHER_GUIDE) == 1
        # Any other turn: the guide and the premise sentence only.
        plain = await ask(memory, store, PLAIN)
        assert plain.block.split("\n")[3] == f"{LIST_GUIDE} {PREMISE_GUIDE}"
        assert GATHER_GUIDE not in plain.block
    finally:
        await memory.aclose()


# ── The excerpt form ──


@pytest.mark.parametrize("source", ["turns", "items"])
async def test_the_excerpt_form_has_no_guide_line_and_so_no_sentence(
    tmp_path: Path,
    store: SessionStore,
    source: str,
) -> None:
    cfg = config(recall_source=source, recall_form="excerpt")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        block = (await ask(memory, store)).block
        search = await MemoryTools(memory).search(query=QUESTION)
        tools = [t.definition for t in await MemoryTools(memory).start()]
        assert block != "" and search != ""
        for text in (block, search, *(t.description for t in tools)):
            assert PREMISE not in text
    finally:
        await memory.aclose()
