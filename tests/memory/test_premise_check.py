"""``[memory] premise_check``: one more sentence in the list form's guide line.

With ``premise_check = "on"`` and ``recall_form = "list"``, the line above the list
(the turns' or the memory items') and the same line in the memory search result end
with one sentence that tells the model to say it has no record of what the question
names when none of the lines mention it. With ``"off"`` (the default), or in the
excerpt form, every byte the model reads is what it was. The texts of both guides are
pinned here as constants so that a change to either one fails this file.
Synthetic items, words and fixed vectors, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.config import MemoryConfig
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    ITEM_LIST_GUIDE,
    LIST_GUIDE,
    PREMISE_GUIDE,
    day,
    list_guide,
)
from localmelo.support.memory.service import MemoryTools, RecallSettings
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.test_hippo_recall import TITLE, TURN, held, lines_of
from localmelo.tests.memory.test_recall_list import (
    DAY,
    QUESTION,
    add_turn,
    ask,
    config,
)

# Today's guides, word for word, as they read before the setting existed.
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


def test_list_guide_appends_the_sentence_after_one_space_and_only_when_asked() -> None:
    assert list_guide(False) == list_guide(False, False) == TURNS_GUIDE
    assert list_guide(True) == list_guide(True, False) == ITEMS_GUIDE
    assert list_guide(False, True) == f"{TURNS_GUIDE} {PREMISE}"
    assert list_guide(True, True) == f"{ITEMS_GUIDE} {PREMISE}"
    assert list_guide(True, True).count(PREMISE) == 1


def test_the_settings_read_the_key_and_only_the_list_form_carries_the_sentence() -> (
    None
):
    assert MemoryConfig().premise_check == "off"
    assert RecallSettings().premise is False
    assert RecallSettings().guide == TURNS_GUIDE
    assert RecallSettings(items=True).guide == ITEMS_GUIDE
    # The sentence needs the list form: the excerpt form has no guide line to carry it.
    assert RecallSettings(premise=True).guide == TURNS_GUIDE
    assert RecallSettings(premise=True, items=True).guide == ITEMS_GUIDE
    assert RecallSettings(premise=True, listing=True).guide == (
        f"{TURNS_GUIDE} {PREMISE}"
    )
    assert RecallSettings(premise=True, listing=True, items=True).guide == (
        f"{ITEMS_GUIDE} {PREMISE}"
    )
    assert RecallSettings(listing=True, items=True).guide == ITEMS_GUIDE


async def test_the_key_is_read_each_time(tmp_path: Path) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        assert not memory.recall_settings().premise
        cfg.memory.premise_check = "on"
        assert memory.recall_settings().premise
        assert memory.recall_settings().guide == f"{TURNS_GUIDE} {PREMISE}"
        cfg.memory.premise_check = "off"
        assert not memory.recall_settings().premise
        assert memory.recall_settings().guide == TURNS_GUIDE
        assert ChatMemory(lambda: None).recall_settings().premise is False
    finally:
        await memory.aclose()


# ── Memory items, list form ──


async def test_the_items_guide_line_ends_with_the_sentence_when_on_and_is_unchanged_off(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = config(recall_source="items", recall_form="list")
    fill = await held(tmp_path, store, cfg)
    memory = fill.memory
    try:
        when = day(time.time() - 3 * DAY)
        line = f"[m{fill.item}] {when} · chat “Garden” · “{TITLE}” · words"
        off_block = f"\n\n{BLOCK_HEADER}\n{ITEMS_GUIDE}\n{line}"
        off = await ask(memory, store)
        assert off.block == off_block
        off_search = await MemoryTools(memory).search(query=QUESTION)
        assert off_search == f"{ITEMS_GUIDE}\n{line}"
        # "off" written out is the default.
        cfg.memory.premise_check = "off"
        assert (await ask(memory, store)).block == off_block

        cfg.memory.premise_check = "on"
        on = await ask(memory, store)
        assert on.block == f"\n\n{BLOCK_HEADER}\n{ITEMS_GUIDE} {PREMISE}\n{line}"
        assert on.block.count(PREMISE) == 1
        guide = next(row for row in on.block.split("\n") if "Memory items" in row)
        assert guide.endswith(PREMISE) and guide.startswith(ITEMS_GUIDE)
        search = await MemoryTools(memory).search(query=QUESTION)
        assert search == f"{ITEMS_GUIDE} {PREMISE}\n{line}"
        assert search.count(PREMISE) == 1
        # Nothing else moved: the same items, the same lines, the same event.
        assert [i.key for i in on.used] == [i.key for i in off.used]
        assert lines_of(on.block) == lines_of(off.block) == [line]
        assert on.block.replace(f" {PREMISE}", "") == off.block
        assert search.replace(f" {PREMISE}", "") == off_search
    finally:
        await memory.aclose()


# ── Chat turns, list form ──


async def test_the_turns_guide_line_ends_with_the_sentence_when_on_and_is_unchanged_off(
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
        off_block = f"\n\n{BLOCK_HEADER}\n{TURNS_GUIDE}\n{line}"
        off = await ask(memory, store)
        assert off.block == off_block
        off_search = await MemoryTools(memory).search(query=QUESTION)
        assert off_search == f"{TURNS_GUIDE}\n{line}"

        cfg.memory.premise_check = "on"
        on = await ask(memory, store)
        assert on.block == f"\n\n{BLOCK_HEADER}\n{TURNS_GUIDE} {PREMISE}\n{line}"
        assert on.block.count(PREMISE) == 1 and ITEMS_GUIDE not in on.block
        search = await MemoryTools(memory).search(query=QUESTION)
        assert search == f"{TURNS_GUIDE} {PREMISE}\n{line}"
        assert [i.key for i in on.used] == [i.key for i in off.used]
        assert on.block.replace(f" {PREMISE}", "") == off.block
        assert search.replace(f" {PREMISE}", "") == off_search
    finally:
        await memory.aclose()


async def test_a_block_with_no_list_has_no_guide_and_so_no_sentence(
    tmp_path: Path,
    store: SessionStore,
) -> None:
    cfg = config(recall_form="list", premise_check="on")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        found = await ask(memory, store)
        assert found.block == "" and PREMISE not in found.block
        assert await MemoryTools(memory).search(query=QUESTION) == (
            "Nothing in melo's memory matches that."
        )
    finally:
        await memory.aclose()


# ── The excerpt form ──


@pytest.mark.parametrize("source", ["turns", "items"])
async def test_the_excerpt_form_reads_the_same_with_the_setting_on(
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
        cfg.memory.premise_check = "on"
        assert (await ask(memory, store)).block == block
        assert await MemoryTools(memory).search(query=QUESTION) == search
        assert [t.definition for t in await MemoryTools(memory).start()] == tools
        for text in (block, search, *(t.description for t in tools)):
            assert PREMISE not in text
        assert memory.recall_settings().guide in (TURNS_GUIDE, ITEMS_GUIDE)
        assert PREMISE not in memory.recall_settings().guide
    finally:
        await memory.aclose()


@pytest.mark.parametrize("source", ["turns", "items"])
async def test_the_tool_definitions_of_the_list_form_do_not_change(
    tmp_path: Path, source: str
) -> None:
    cfg = config(recall_source=source, recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        off = [t.definition for t in await MemoryTools(memory).start()]
        cfg.memory.premise_check = "on"
        assert [t.definition for t in await MemoryTools(memory).start()] == off
    finally:
        await memory.aclose()
