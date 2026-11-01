"""The recall settings: block size, recall without notes, a hint when nothing matched.

``[memory] recall_records``, ``recall_docs`` and ``recall_chars`` size the block
a chat message carries (defaults 3, 2 and 900: what recall always did);
``recall_notes = "off"`` leaves notes out of the block and the pinned lines and
recalls turns only; ``search_when_empty = "hint"`` gives a block that recalled
nothing one fixed line. Synthetic notes and turns, words only, no model.
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
    Corpus,
    Item,
)
from localmelo.support.memory.service import SEARCH_HINT, RecallSettings
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import teach

HOLDER = "test-host:1:recall-settings"
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
NOTHING = "write a poem about autumn"


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


def test_the_defaults_are_the_numbers_recall_always_used() -> None:
    cfg = MemoryConfig()
    assert (cfg.recall_records, cfg.recall_docs, cfg.recall_chars) == (
        AUTO_RECORDS,
        AUTO_DOCS,
        BLOCK_CHARS,
    )
    assert (cfg.recall_notes, cfg.search_when_empty) == ("on", "off")
    assert ChatMemory(lambda: None).recall_settings() == RecallSettings()
    assert ChatMemory(lambda: Config()).recall_settings() == RecallSettings()
    assert RecallSettings() == (AUTO_RECORDS, AUTO_DOCS, BLOCK_CHARS, True, False)


async def test_the_defaults_give_three_notes_and_two_turns_in_900_characters(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = await build(tmp_path, store, config())
    try:
        found = await recall(memory, store)
        assert kinds(found.used) == (AUTO_RECORDS, AUTO_DOCS)
        assert BLOCK_HEADER in found.block and SEARCH_HINT not in found.block
        assert len(found.block) <= BLOCK_CHARS + 2
        assert found.event()["recall"]["used"] == 5  # type: ignore[index]
        assert "search_hint" not in found.event()  # type: ignore[operator]
    finally:
        await memory.aclose()


async def test_a_bigger_budget_puts_eight_notes_and_eight_turns_in_4000_characters(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_records=8, recall_docs=8, recall_chars=4000)
    memory = await build(tmp_path, store, cfg)
    try:
        found = await recall(memory, store)
        assert kinds(found.used) == (8, 8)
        assert BLOCK_CHARS < len(found.block) <= 4000 + 2
        assert found.block.count("\n- ") == 16
        assert found.recall is not None and found.recall.used == 16
        # The size bound alone leaves items out, the best first.
        cfg.memory.recall_chars = BLOCK_CHARS
        tight = await recall(memory, store)
        assert 0 < len(tight.used) < 16 and len(tight.block) <= BLOCK_CHARS + 2
        assert list(tight.used) == list(found.used[: len(tight.used)])
        cfg.memory.recall_chars = 200
        assert len((await recall(memory, store)).used) < len(tight.used)
        # A smaller budget than the default, and none at all.
        cfg.memory.recall_chars = 4000
        cfg.memory.recall_records, cfg.memory.recall_docs = 1, 0
        assert kinds((await recall(memory, store)).used) == (1, 0)
        cfg.memory.recall_records = 0
        assert (await recall(memory, store)).block == ""
    finally:
        await memory.aclose()


async def test_the_settings_are_read_again_each_turn_and_a_bad_one_is_the_default(
    tmp_path: Path,
) -> None:
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    cfg.memory.recall_records = 8
    cfg.memory.search_when_empty = "hint"
    assert memory.recall_settings() == RecallSettings(8, AUTO_DOCS, 900, True, True)
    cfg.memory.recall_notes = "off"
    assert memory.recall_settings().notes is False
    for bad in (-1, 21, True, "3", 2.5):
        cfg.memory.recall_docs = bad  # type: ignore[assignment]
        assert memory.recall_settings().docs == AUTO_DOCS, bad
    cfg.memory.recall_chars = 100
    assert memory.recall_settings().chars == BLOCK_CHARS
    cfg.memory.recall_notes = "maybe"
    cfg.memory.search_when_empty = "always"
    assert memory.recall_settings().notes and not memory.recall_settings().hint
    await memory.aclose()


async def test_without_notes_the_block_holds_turns_and_nothing_is_pinned(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_records=8, recall_docs=8, recall_chars=4000)
    memory = await build(tmp_path, store, cfg)
    try:
        # A note the user stated about themselves is pinned in every prompt.
        turn = await _turn(store, "About tea")
        await teach(memory, store, turn, "I prefer green tea.")
        assert (await memory.pinned_quotes()).startswith("- “I prefer green tea.”\n")
        with_notes = await recall(memory, store)
        assert kinds(with_notes.used) == (8, 8)

        cfg.memory.recall_notes = "off"
        found = await recall(memory, store)
        assert kinds(found.used) == (0, 8)
        assert "the user said" not in found.block and BLOCK_HEADER in found.block
        assert found.block.count("\n- ") == 8
        assert await memory.pinned_quotes() == ""

        # The default block, too, is turns only, up to its own bounds.
        cfg.memory.recall_records, cfg.memory.recall_docs = 3, 2
        cfg.memory.recall_chars = BLOCK_CHARS
        assert kinds((await recall(memory, store)).used) == (0, AUTO_DOCS)

        # The note tool and the search tool are unchanged.
        cfg.memory.recall_notes = "on"
        again = await memory.before_turn(
            store, await _turn(store, "Tea again"), "I prefer oolong tea."
        )
        assert again.recall is not None
        cfg.memory.recall_notes = "off"
        said = await memory.remember("You prefer oolong tea.", "preferences")
        assert said.startswith("Remembered: You prefer oolong tea.")
        assert await memory.pinned_quotes() == ""
        hits = await memory.search(QUESTION, records_k=8, docs_k=0)
        assert kinds(tuple(hits)) == (8, 0)
        cfg.memory.recall_notes = "on"
        assert "oolong" in await memory.pinned_quotes()
    finally:
        await memory.aclose()


def test_a_note_never_makes_a_turn_older_when_notes_are_off(tmp_path: Path) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    corpus = Corpus()
    note = Item(
        "r:1", "You live in Utrecht.", "chat", "s2", 1, 2.0, replaces=(("s1", 1),)
    )
    corpus.add(note)
    corpus.add(Item("d:1", "I live in Lisbon.", "chat", "s1", 1, 1.0))
    assert memory._replaced(corpus) == {("s1", 1): note}
    assert memory._replaced(corpus, notes=False) == {}


async def test_the_hint_is_in_the_block_exactly_when_no_item_was_recalled(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(search_when_empty="hint")
    memory = await build(tmp_path, store, cfg)
    try:
        empty = await recall(memory, store, NOTHING)
        assert empty.used == ()
        assert empty.block == f"\n\n{BLOCK_HEADER}\n{SEARCH_HINT}"
        assert empty.search_hint is True
        assert empty.event()["search_hint"] is True  # type: ignore[index]
        assert empty.recall is not None
        assert empty.recall.used == 0 and empty.recall.block_chars == len(empty.block)
        assert SEARCH_HINT == (
            "Nothing from earlier chats matched this message. If it asks about the "
            "person or something said before, call the memory search tool first; "
            "if that finds nothing, say you don't know."
        )

        found = await recall(memory, store)
        assert len(found.used) == AUTO_RECORDS + AUTO_DOCS
        assert SEARCH_HINT not in found.block and found.search_hint is False
        assert "search_hint" not in found.event()  # type: ignore[operator]

        # A block that kept notes names them first, then carries the hint.
        said = "Remember that my group meeting is on Thursdays at 3 pm."
        kept = await recall(memory, store, said)
        assert kept.used == () and kept.remembered
        lines = kept.block.split("\n")
        assert lines[2] == BLOCK_HEADER and lines[3].startswith("melo kept from")
        assert lines[4:] == [SEARCH_HINT]

        # No item: the budget allows none, so even a matching message gets it.
        cfg.memory.recall_records = cfg.memory.recall_docs = 0
        none = await recall(memory, store)
        assert none.used == () and none.block.endswith(SEARCH_HINT)
        assert none.event()["search_hint"] is True  # type: ignore[index]

        # An empty message recalls nothing and is told nothing.
        blank = await recall(memory, store, "   ")
        assert blank.block == "" and blank.search_hint is False
        assert blank.event() is not None and "search_hint" not in blank.event()  # type: ignore[operator]
    finally:
        await memory.aclose()


async def test_the_hint_is_off_by_default_and_with_off(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, cfg)
    try:
        for setting in ("off", "off"):
            cfg.memory.search_when_empty = setting
            empty = await recall(memory, store, NOTHING)
            assert empty.block == "" and not empty.search_hint
            assert "search_hint" not in empty.event()  # type: ignore[operator]
        cfg.memory.across_chats = False
        cfg.memory.search_when_empty = "hint"
        off = await recall(memory, store, NOTHING)
        assert off.block == "" and off.event() is None
    finally:
        await memory.aclose()


async def test_the_hint_and_no_notes_work_together(
    tmp_path: Path, store: SessionStore
) -> None:
    """A question only a note answers recalls nothing without notes: the hint says so."""
    cfg = config(recall_notes="off", search_when_empty="hint")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        await teach(memory, store, await _turn(store, "Home"), "I live in Pittsburgh.")
        found = await recall(memory, store, "where do I live?")
        assert found.used == () and found.block.endswith(SEARCH_HINT)
        cfg.memory.recall_notes = "on"
        found = await recall(memory, store, "where do I live?")
        assert kinds(found.used) == (1, 0) and SEARCH_HINT not in found.block
    finally:
        await memory.aclose()
