"""``[memory] premise_check``: one more sentence in the list form's guide line.

By default (``"on"``) and with ``recall_form = "list"``, the line above the list (the
turns' or the memory items') and the same line in the memory search result end with
one sentence that tells the model to say it has no record of what the question names
when none of the lines mention it; on a turn that asks to count, total or compare it
follows the gather sentence. With ``"off"``, or in the excerpt form (the default
form), every byte the model reads is what it was. The texts of both guides are pinned
here as constants so that a change to either one fails this file.
Synthetic items, words and fixed vectors, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import Config, MemoryConfig
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    GATHER_GUIDE,
    ITEM_LIST_GUIDE,
    LIST_GUIDE,
    PREMISE_GUIDE,
    day,
)
from localmelo.support.memory.service import MemoryTools, RecallSettings
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

# The guides, word for word, as they read before the setting existed.
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


def default_config(**memory: Any) -> Config:
    """A config that leaves ``premise_check`` as a new install has it (the shared
    helper sets it off, to pin the guide line as it was)."""
    cfg = config(**memory)
    cfg.memory.premise_check = MemoryConfig().premise_check
    return cfg


def test_the_guides_and_the_sentence_are_pinned() -> None:
    assert LIST_GUIDE == TURNS_GUIDE
    assert ITEM_LIST_GUIDE == ITEMS_GUIDE
    assert PREMISE_GUIDE == PREMISE
    assert "\n" not in PREMISE_GUIDE and "\n" not in f"{LIST_GUIDE} {PREMISE_GUIDE}"
    assert "\n" not in f"{LIST_GUIDE} {GATHER_GUIDE} {PREMISE_GUIDE}"


async def test_the_default_is_on_and_the_key_is_read_each_time(tmp_path: Path) -> None:
    assert MemoryConfig().premise_check == "on"
    assert RecallSettings().premise is True
    assert ChatMemory(lambda: None).recall_settings().premise is True
    cfg = default_config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        assert memory.recall_settings().premise
        cfg.memory.premise_check = "off"
        assert not memory.recall_settings().premise
        cfg.memory.premise_check = "on"
        assert memory.recall_settings().premise
    finally:
        await memory.aclose()


# ── Memory items, list form ──


async def test_the_items_guide_line_ends_with_the_sentence_and_off_leaves_it_out(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = default_config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        when = day(time.time() - 3 * DAY)
        line = f"[m{fill.item}] {when} · chat “Garden” · “{TITLE}” · words"
        on = await ask(memory, store)
        assert on.block == f"\n\n{BLOCK_HEADER}\n{ITEMS_GUIDE} {PREMISE}\n{line}"
        assert on.block.count(PREMISE) == 1
        search = await MemoryTools(memory).search(query=QUESTION)
        assert search == f"{ITEMS_GUIDE} {PREMISE}\n{line}"
        assert search.count(PREMISE) == 1

        cfg.memory.premise_check = "off"
        off = await ask(memory, store)
        assert off.block == f"\n\n{BLOCK_HEADER}\n{ITEMS_GUIDE}\n{line}"
        off_search = await MemoryTools(memory).search(query=QUESTION)
        assert off_search == f"{ITEMS_GUIDE}\n{line}"
        # Nothing else moved: the same items, the same lines.
        assert [i.key for i in on.used] == [i.key for i in off.used]
        assert lines_of(on.block) == lines_of(off.block) == [line]
    finally:
        await memory.aclose()


# ── Chat turns, list form ──


async def test_the_turns_guide_line_ends_with_the_sentence_and_off_leaves_it_out(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = default_config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        when = time.time() - 3 * DAY
        _, (doc,) = await add_turn(
            memory, store, TURN[0], TURN[1], at=when, title="Garden"
        )
        line = f"[t{doc}] {day(when)} · chat “Garden” · “{TURN[0]}” · words"
        on = await ask(memory, store)
        assert on.block == f"\n\n{BLOCK_HEADER}\n{TURNS_GUIDE} {PREMISE}\n{line}"
        assert on.block.count(PREMISE) == 1 and ITEMS_GUIDE not in on.block
        search = await MemoryTools(memory).search(query=QUESTION)
        assert search == f"{TURNS_GUIDE} {PREMISE}\n{line}"

        cfg.memory.premise_check = "off"
        off = await ask(memory, store)
        assert off.block == f"\n\n{BLOCK_HEADER}\n{TURNS_GUIDE}\n{line}"
        assert await MemoryTools(memory).search(query=QUESTION) == (
            f"{TURNS_GUIDE}\n{line}"
        )
        assert [i.key for i in on.used] == [i.key for i in off.used]
    finally:
        await memory.aclose()


async def test_a_block_with_no_list_has_no_guide_and_so_no_sentence(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = default_config(recall_form="list")
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
    memory, _, cfg = await memory_of(tmp_path, store, recall_form="list")
    cfg.memory.premise_check = MemoryConfig().premise_check  # the default, on
    assert cfg.memory.gather == "guide"
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
        # Off drops the premise sentence and nothing else.
        cfg.memory.premise_check = "off"
        gather_only = f"{LIST_GUIDE} {GATHER_GUIDE}"
        assert (await ask(memory, store, GATHER)).block.split("\n")[3] == gather_only
        assert (await MemoryTools(memory).search(query=WORDS)).split("\n")[
            0
        ] == gather_only
    finally:
        await memory.aclose()


# ── The excerpt form ──


@pytest.mark.parametrize("source", ["turns", "items"])
async def test_the_excerpt_form_reads_the_same_whether_the_setting_is_on_or_off(
    tmp_path: Path,
    store: SessionStore,
    source: str,
) -> None:
    cfg = default_config(recall_source=source, recall_form="excerpt")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        assert cfg.memory.premise_check == "on"
        block = (await ask(memory, store)).block
        search = await MemoryTools(memory).search(query=QUESTION)
        tools = [t.definition for t in await MemoryTools(memory).start()]
        assert block != "" and search != ""
        cfg.memory.premise_check = "off"
        assert (await ask(memory, store)).block == block
        assert await MemoryTools(memory).search(query=QUESTION) == search
        assert [t.definition for t in await MemoryTools(memory).start()] == tools
        for text in (block, search, *(t.description for t in tools)):
            assert PREMISE not in text
    finally:
        await memory.aclose()


@pytest.mark.parametrize("source", ["turns", "items"])
async def test_the_tool_definitions_of_the_list_form_do_not_change(
    tmp_path: Path, source: str
) -> None:
    cfg = default_config(recall_source=source, recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        on = [t.definition for t in await MemoryTools(memory).start()]
        cfg.memory.premise_check = "off"
        assert [t.definition for t in await MemoryTools(memory).start()] == on
    finally:
        await memory.aclose()
