"""The recall block's shipped size and the settings that still shape it.

The notes and turns a block holds (3 and 2), its size (4000 characters) and the room
of a turn in it (900) are constants (#349); ``recall_min_cosine``, ``recall_min_coverage``
and the form settings are read each turn. Synthetic notes and turns, words only, no
model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.config import Config, MemoryConfig
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import (
    AUTO_DOCS,
    AUTO_RECORDS,
    BLOCK_CHARS,
    BLOCK_HEADER,
    ITEM_CHARS,
    LIST_MAX,
    MIN_COSINE,
    MIN_COVERAGE,
    OPEN_CHARS,
    TURN_CHARS,
    Item,
    block,
)
from localmelo.support.memory.service import RecallSettings
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import teach

HOLDER = "test-host:1:recall-settings"
SMALL = 900  # a block bound smaller than the shipped one, which the items overflow
QUESTION = "what do I plant in the tomato garden?"
# Nine notes and nine chat turns, each of which the question finds.
NOTES = [
    f"I {verb} the tomato garden {where}."
    for verb, where in (
        ("water", "every morning"),
        ("prune", "in June"),
        ("mulch", "with straw"),
        ("stake", "with bamboo"),
        ("feed", "with compost tea"),
        ("weed", "on Sundays"),
        ("harvest", "in late August"),
        ("cover", "against frost"),
        ("photograph", "for my blog"),
    )
]
TURNS = [
    (f"Which tomato garden bed {n} gets the sun?", f"Bed {n} faces south all day.")
    for n in range(9)
]


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


async def _turn(store: SessionStore, title: str):  # type: ignore[no-untyped-def]
    return await store.begin_turn(
        None, title, holder=HOLDER, title=title, new_session=NewSession()
    )


async def build(tmp_path: Path, store: SessionStore, cfg: Config) -> ChatMemory:
    """A memory over *cfg* holding NOTES as notes and TURNS as chat turns."""
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    for n, said in enumerate(NOTES):
        await teach(memory, store, await _turn(store, f"Note {n}"), said)
    _, index = memory.stores()
    for n, (typed, answer) in enumerate(TURNS):
        turn = await _turn(store, f"Turn {n}")
        await index.add(
            Document(
                session_id=turn.session_id,
                turn=1,
                surface="gateway",
                kind="chat",
                title=typed,
                text=document_text(typed, answer),
                created_at=time.time(),
            )
        )
    return memory


def config(**memory: object) -> Config:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return cfg


def kinds(items: tuple[Item, ...]) -> tuple[int, int]:
    """How many of *items* are notes and how many chat turns."""
    notes = sum(item.is_record for item in items)
    return notes, len(items) - notes


async def recall(memory: ChatMemory, store: SessionStore, typed: str = QUESTION):  # type: ignore[no-untyped-def]
    return await memory.before_turn(store, await _turn(store, "Ask"), typed)


def test_the_defaults_are_the_shipped_numbers() -> None:
    cfg = MemoryConfig()
    assert (AUTO_RECORDS, AUTO_DOCS, BLOCK_CHARS, ITEM_CHARS) == (3, 2, 4000, 900)
    assert ChatMemory(lambda: None).recall_settings() == RecallSettings()
    assert ChatMemory(lambda: Config()).recall_settings() == RecallSettings()
    assert RecallSettings() == (
        MIN_COSINE,
        MIN_COVERAGE,
        False,  # match: the head of a turn
        False,  # listing: recall_form is excerpt
        LIST_MAX,
        OPEN_CHARS,
        False,  # items: recall_source is turns
        False,  # auto: recall_source is turns
    )
    assert (cfg.recall_min_cosine, cfg.recall_min_coverage) == (0.55, 0.6)
    assert (MIN_COSINE, MIN_COVERAGE) == (0.55, 0.6)
    assert RecallSettings().shown("tomato").chars == ITEM_CHARS
    assert cfg.recall_excerpt == "head"
    assert (cfg.recall_form, cfg.recall_list_max, cfg.recall_open_chars) == (
        "excerpt",
        20,
        4000,
    )
    assert (cfg.recall_source, cfg.compress_cache) == ("turns", "")
    # The head of a turn is a floor, not a default: it stays what it was.
    assert TURN_CHARS == 234


async def test_the_defaults_give_three_notes_and_two_turns_in_4000_characters(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = await build(tmp_path, store, config())
    try:
        found = await recall(memory, store)
        assert kinds(found.used) == (AUTO_RECORDS, AUTO_DOCS)
        assert BLOCK_HEADER in found.block
        assert len(found.block) <= BLOCK_CHARS + 2
        assert found.event()["recall"]["used"] == 5  # type: ignore[index]
        # Nothing matched: nothing is added, not even a hint.
        nothing = await recall(memory, store, "write a poem about autumn")
        assert nothing.used == () and nothing.block == ""
        assert "search_hint" not in found.event()  # type: ignore[operator]
    finally:
        await memory.aclose()


async def test_the_size_bound_leaves_items_out_the_best_first(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = await build(tmp_path, store, config())
    try:
        hits = await memory.search(QUESTION, records_k=8, docs_k=8)
        assert kinds(tuple(hits)) == (8, 8)
        text, used = block(hits, chars=BLOCK_CHARS)
        assert kinds(tuple(used)) == (8, 8)
        assert SMALL < len(text) <= BLOCK_CHARS + 2 and text.count("\n- ") == 16
        # The size bound alone leaves items out, the best first.
        tight_text, tight = block(hits, chars=SMALL)
        assert 0 < len(tight) < 16 and len(tight_text) <= SMALL + 2
        assert [i.key for i in tight] == [i.key for i in used[: len(tight)]]
        assert len(block(hits, chars=200)[1]) < len(tight)
    finally:
        await memory.aclose()
