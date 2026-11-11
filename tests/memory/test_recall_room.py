"""The room a recalled chat turn and the recall block take by default (#349).

The room of a recalled turn and the size of the block are 900 and 4000 characters: a
larger room answered more questions from earlier chats. The head of a turn, 234
characters (:data:`TURN_CHARS`), is a floor and not the room. Synthetic turns, words
only, no model.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import (
    BLOCK_CHARS,
    ITEM_CHARS,
    TURN_CHARS,
    Item,
    Shown,
    block,
)
from localmelo.support.memory.service import MemoryTools, RecallSettings
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.test_recall_bars import _turn, config
from localmelo.tests.memory.test_recall_excerpt import (
    QUESTION,
    TYPED,
    english,
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


def id_prefix(line: str) -> str:
    """The id a cut line starts with (``[t1] ``): it must be there, by default."""
    found = re.match(r"\[t\d+\] ", line)
    assert found, line
    return found.group()


def user_words(shown: str) -> str:
    """The user's part of a turn's recall line, without its id and quotes."""
    said = re.sub(r"^\[t\d+\] ", "", read(shown)).removeprefix("the user said “")
    return said.partition("”; melo replied “")[0].removesuffix("”")


def test_the_shipped_defaults_are_900_and_4000_and_the_head_stays_234() -> None:
    assert (ITEM_CHARS, BLOCK_CHARS) == (900, 4000)
    assert TURN_CHARS == 234
    assert RecallSettings().shown().chars == 900


async def test_with_no_config_a_turn_takes_900_characters(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    await add_turns(memory, store, 6)
    try:
        assert memory.recall_settings() == RecallSettings()
        found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        assert len(found.used) == 2  # AUTO_DOCS, as ever
        # A turn's two parts take 900 characters together: the user's words whole
        # (they fit) and melo's reply in the rest. By default a cut line starts
        # with its id (``[t1] ``), which takes its characters from the same room.
        for item in found.used:
            prefix = id_prefix(item.shown)
            user, reply = parts(read(item.shown.removeprefix(prefix)))
            assert user == TYPED
            assert len(user) + len(reply) == ITEM_CHARS - len(prefix)
            assert reply.endswith("…")
        assert 1800 < len(found.block) <= BLOCK_CHARS + 2
        # The search tool's list reads a turn in the same room.
        said = (await MemoryTools(memory).search(query=QUESTION)).split("\n")
        line = said[0].removeprefix("1. ")
        prefix = id_prefix(line)
        user, reply = parts(read(line).removeprefix(prefix))
        assert len(user) + len(reply) == ITEM_CHARS - len(prefix)
    finally:
        await memory.aclose()


async def test_the_block_holds_turns_up_to_4000_characters(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    await add_turns(memory, store, 6)
    try:
        hits = await memory.search(QUESTION, records_k=0, docs_k=8)
        assert len(hits) == 6
        show = RecallSettings().shown()
        text, used = block(hits, show=show)
        # As many 900-character turns as fit in 4000, the best first, and no more.
        assert 3 <= len(used) < 6
        assert 3000 < len(text) <= BLOCK_CHARS + 2
        assert [i.key for i in used] == [i.key for i in hits[: len(used)]]
        # A bound of 900 leaves such a turn out: it does not fit in it.
        assert block(hits, chars=900, show=show)[1] == []
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


async def test_long_words_take_the_room_less_the_reply(
    tmp_path: Path, store: SessionStore
) -> None:
    long_words = " ".join(f"word{n}" for n in range(180))  # about 1200 characters
    memory = ChatMemory(lambda: config(), directory=str(tmp_path / "memory"))
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
        # The room kept for the reply comes off 900; the id the cut line starts with
        # is counted in the reply's part.
        id_prefix(item.shown)
        assert len(user) == ITEM_CHARS - 60
        assert user.endswith("…")
    finally:
        await memory.aclose()


def test_the_head_of_a_turn_is_a_floor_of_the_users_words_in_any_room() -> None:
    long_words = " ".join(f"word{n}" for n in range(180))  # about 1200 characters
    turn = Item("d:1", document_text(long_words, "Fine."), "chat “Long”", "s", 1, 1.0)
    user = user_words(turn.line(1.0, show=Shown.of(250)))
    assert len(user) == TURN_CHARS and user.endswith("…")
