"""The room a recalled chat turn and the recall block take by default (#349).

The shipped defaults of ``[memory] recall_item_chars`` and ``recall_chars`` are 900
and 4000: a larger room answered more questions from earlier chats. The head of a
turn, 234 characters (:data:`TURN_CHARS`), is a floor and not a default, and a
config that sets the earlier numbers reads exactly as recall did before.
Synthetic turns, words only, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import (
    BLOCK_CHARS,
    BLOCK_HEADER,
    ITEM_CHARS,
    TURN_CHARS,
)
from localmelo.support.memory.service import MemoryTools, RecallSettings
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.test_event_shown import build as build_note_and_turns
from localmelo.tests.memory.test_recall_bars import _turn, config
from localmelo.tests.memory.test_recall_excerpt import (
    QUESTION,
    TYPED,
    english,
    legacy_turn_said,
    parts,
)

DAY = 86_400.0


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


async def add_turns(memory: ChatMemory, store: SessionStore, count: int) -> None:
    """*count* long chat turns, each in a chat of its own, that the question finds."""
    _, index = memory.stores()
    now = time.time()
    for n in range(1, count + 1):
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


def read(line: str) -> str:
    """What a recall line says of a turn, without its trailing ``(chat, date)``."""
    return line[: line.rindex(" (")]


def user_words(shown: str) -> str:
    """The user's part of a turn's recall line, without its quotes."""
    said = read(shown).removeprefix("the user said “")
    return said.partition("”; melo replied “")[0].removesuffix("”")


def test_the_shipped_defaults_are_900_and_4000_and_the_head_stays_234() -> None:
    assert (ITEM_CHARS, BLOCK_CHARS) == (900, 4000)
    assert TURN_CHARS == 234
    settings = RecallSettings()
    assert (settings.item_chars, settings.chars) == (900, 4000)
    assert settings.shown(QUESTION).chars == 900


async def test_with_no_config_a_turn_takes_900_characters(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    await add_turns(memory, store, 6)
    try:
        assert memory.recall_settings() == RecallSettings()
        found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        assert len(found.used) == 2  # recall_docs, as ever
        # A turn's two parts take 900 characters together: the user's words whole
        # (they fit) and melo's reply in the rest.
        for item in found.used:
            user, reply = parts(read(item.shown))
            assert user == TYPED
            assert len(user) + len(reply) == ITEM_CHARS and reply.endswith("…")
        assert 1800 < len(found.block) <= BLOCK_CHARS + 2
        # The search tool's list reads a turn in the same room.
        said = (await MemoryTools(memory).search(query=QUESTION)).split("\n")
        user, reply = parts(read(said[0]).removeprefix("1. "))
        assert len(user) + len(reply) == ITEM_CHARS
    finally:
        await memory.aclose()


async def test_with_no_size_set_the_block_holds_turns_up_to_4000_characters(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_docs=8)  # no recall_chars: the shipped bound
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    await add_turns(memory, store, 6)
    try:
        found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        # As many 900-character turns as fit in 4000, the best first, and no more.
        assert 3 <= len(found.used) < 6
        assert 3000 < len(found.block) <= BLOCK_CHARS + 2
        assert found.recall is not None and found.recall.block_chars == len(found.block)
        # The earlier bound, 900, leaves such a turn out: it does not fit in it.
        cfg.memory.recall_chars = 900
        assert (
            await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        ).used == ()
    finally:
        await memory.aclose()


async def test_the_default_config_and_no_config_read_alike(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    with_cfg = ChatMemory(lambda: cfg, directory=str(tmp_path / "with"))
    without = ChatMemory(lambda: None, directory=str(tmp_path / "without"))
    try:
        assert with_cfg.recall_settings() == without.recall_settings()
        for memory in (with_cfg, without):
            await add_turns(memory, store, 2)
        one = await with_cfg.before_turn(store, await _turn(store, "Ask"), QUESTION)
        two = await without.before_turn(store, await _turn(store, "Ask"), QUESTION)
        assert [read(item.shown) for item in one.used] == [
            read(item.shown) for item in two.used
        ]
    finally:
        await with_cfg.aclose()
        await without.aclose()


async def test_a_config_with_the_earlier_numbers_reads_exactly_as_it_did(
    tmp_path: Path, store: SessionStore
) -> None:
    older = config(recall_item_chars=234, recall_chars=900)
    memory = await build_note_and_turns(tmp_path, store, older)
    try:
        assert memory.recall_settings() == RecallSettings(chars=900, item_chars=234)
        found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        data = found.event()
        assert data is not None
        kinds = [e["kind"] for e in data["used"]]
        assert kinds.count("record") == 1 and kinds.count("chat") == 2
        # Byte for byte: the block header, then each line as the head of 234
        # characters always read, with the note whole.
        expected = [BLOCK_HEADER]
        for item in found.used:
            line = item.shown
            if item.is_record:
                expected.append(f"- {line}")
            else:
                where = line[line.rindex(" (") :]
                expected.append(f"- {legacy_turn_said(item.text)}{where}")
        assert found.block == "\n\n" + "\n".join(expected)
        assert len(found.block) <= 900 + 2
        # The search tool's list reads a turn in the same head.
        said = (await MemoryTools(memory).search(query=QUESTION)).split("\n")
        turns = [item for item in found.used if not item.is_record]
        assert len(said) == 3 and len(turns) == 2
        for line in said:
            if "melo replied" in line:
                number, _, rest = line.partition(". ")
                assert any(
                    read(rest) == legacy_turn_said(turn.text) for turn in turns
                ), number
        # The shipped defaults show these turns in a larger room.
        shipped = config()
        memory_default = await build_note_and_turns(
            tmp_path / "shipped", store, shipped
        )
        try:
            again = await memory_default.before_turn(
                store, await _turn(store, "Ask again"), QUESTION
            )
            assert len(again.block) > len(found.block) + 600
            assert "\n\n" + BLOCK_HEADER in again.block
        finally:
            await memory_default.aclose()
    finally:
        await memory.aclose()


@pytest.mark.parametrize(
    ("room", "user_chars"),
    [(ITEM_CHARS, ITEM_CHARS - 60), (250, TURN_CHARS)],
)
async def test_long_words_take_the_room_less_the_reply_but_never_fewer_than_the_head(
    tmp_path: Path, store: SessionStore, room: int, user_chars: int
) -> None:
    """The head of a turn, 234 characters, is a floor of the user's words in any room."""
    long_words = " ".join(f"word{n}" for n in range(180))  # about 1200 characters
    memory = ChatMemory(
        lambda: config(recall_item_chars=room), directory=str(tmp_path / "memory")
    )
    _, index = memory.stores()
    turn = await _turn(store, "Long")
    await index.add(
        Document(
            session_id=turn.session_id,
            turn=1,
            surface="gateway",
            kind="chat",
            title="Long",
            text=document_text(long_words, "Fine."),
            created_at=time.time(),
        )
    )
    try:
        found = await memory.before_turn(
            store, await _turn(store, "Ask"), "word7 word9?"
        )
        (item,) = found.used
        user = user_words(item.shown)
        assert len(user) == user_chars and user.endswith("…")
    finally:
        await memory.aclose()
