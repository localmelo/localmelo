"""The recall bars: how close in meaning, and how many of the question's words.

``MIN_COSINE`` (0.45) is the least similarity a vector needs to join the vector
list and ``MIN_COVERAGE`` (0.4) the share of the question's content words an item
must hold to join the word list. Both apply to the recall block and to the memory
search tool. Synthetic notes, turns and fixed vectors, no model.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.support.config import Config, MemoryConfig
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import (
    MIN_COSINE,
    MIN_COVERAGE,
    Corpus,
    Item,
    ranked,
)
from localmelo.support.memory.service import MemoryTools
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeChat, teach

HOLDER = "test-host:1:recall-bars"
NOW = 1_000_000.0
MODEL = "fixed-embed"
NOTHING_FOUND = "Nothing in melo's memory matches that."
# No word of the turn below is in this question, so only vectors can find it.
FAR_QUESTION = "zebra yodel"
# Half of this question's four content words are in the turn and the note below.
HALF_QUESTION = "tomato garden zebra yodel"
# One of this question's four content words is in the turn and the note below.
FEW_QUESTION = "tomato zebra yodel quartz"
# A vector just above the cosine bar and one just below it.
ABOVE, BELOW = 0.47, 0.43
TURN = ("Which tomato garden bed gets the sun?", "Bed one faces south all day.")
NOTE = "I stake the tomato garden with bamboo."


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def vector_at(cosine: float) -> list[float]:
    """A unit vector whose cosine to ``[1, 0, 0]`` is *cosine*."""
    return [cosine, math.sqrt(1.0 - cosine * cosine), 0.0]


class FixedEmbedding(BaseEmbeddingProvider):
    """The question (it holds "zebra") is ``[1, 0, 0]``; any other text is orthogonal."""

    model = MODEL

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0] if "zebra" in text else [0.0, 0.0, 1.0] for text in texts
        ]

    async def close(self) -> None:
        return None


def config(**memory: object) -> Config:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return cfg


async def _turn(store: SessionStore, title: str):  # type: ignore[no-untyped-def]
    return await store.begin_turn(
        None, title, holder=HOLDER, title=title, new_session=NewSession()
    )


async def memory_with(
    tmp_path: Path,
    store: SessionStore,
    cfg: Config,
    *,
    cosine: float | None = None,
    note: bool = False,
) -> ChatMemory:
    """A memory holding TURN as a chat turn (with a stored vector at *cosine* to the
    far question, when given) and, with *note*, NOTE as a note."""
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    if cosine is not None:
        provider = FixedEmbedding()
        memory.embedding = lambda: (provider, MODEL)  # type: ignore[method-assign]
    if note:
        await teach(memory, store, await _turn(store, "Stakes"), NOTE)
    typed, answer = TURN
    turn = await _turn(store, typed)
    _, index = memory.stores()
    await index.add(
        Document(
            session_id=turn.session_id,
            turn=1,
            surface="gateway",
            kind="chat",
            title=typed,
            text=document_text(typed, answer),
            created_at=time.time(),
            vector=tuple(vector_at(cosine)) if cosine is not None else None,
            embed_model=MODEL if cosine is not None else "",
        )
    )
    return memory


def doc_item(key: str, text: str, cosine: float | None = None) -> Item:
    vector = None if cosine is None else np.asarray(vector_at(cosine), np.float32)
    return Item(
        key,
        text,
        "chat",
        f"s{key}",
        1,
        NOW,
        vector=vector,
        embed_model=MODEL if cosine is not None else "",
    )


QUERY_VECTOR = np.asarray([1.0, 0.0, 0.0], np.float32)


def test_ranked_takes_a_vector_above_the_cosine_bar_and_leaves_one_below() -> None:
    above = doc_item("d:1", "the cat sleeps on the sofa", ABOVE)
    below = doc_item("d:2", "a dog barks in the yard", BELOW)
    corpus = Corpus()
    corpus.add(above)
    corpus.add(below)
    now = NOW + 1
    assert ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now) == [above]
    # A lower bar takes what the default leaves, the closer one first.
    low = ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.40)
    assert low == [above, below]
    # A higher bar takes away what the default keeps.
    assert (
        ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.65)
        == []
    )


def test_ranked_takes_half_the_question_words_and_leaves_a_quarter() -> None:
    held = doc_item("d:1", "I stake the tomato garden with bamboo.")
    corpus = Corpus()
    corpus.add(held)
    now = NOW + 1
    assert ranked(corpus, HALF_QUESTION, None, "", now=now) == [held]
    assert ranked(corpus, FEW_QUESTION, None, "", now=now) == []
    # A stricter share takes away what the default keeps, a zero share lets all in.
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=0.6) == []
    assert ranked(corpus, FEW_QUESTION, None, "", now=now, coverage=0.0) == [held]
    # Every word of the question is a bar of 1.0.
    assert ranked(corpus, "tomato garden", None, "", now=now, coverage=1.0) == [held]
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=1.0) == []


def test_ranked_leaves_japanese_particles_out_of_the_question_words() -> None:
    dog = doc_item("d:1", "User: 私の犬の名前はポチです。\nmelo: 覚えました。")
    meeting = doc_item(
        "d:2", "User: 明日の会議は午後三時からです。\nmelo: 了解しました。"
    )
    hobby = doc_item("d:3", "User: 母の趣味は庭の手入れです。\nmelo: いいですね。")
    corpus = Corpus()
    for item in (dog, meeting, hobby):
        corpus.add(item)
    now = NOW + 1
    # An ordinary question still finds its turn.
    assert ranked(corpus, "私の犬の名前は何ですか", None, "", now=now) == [dog]
    assert ranked(corpus, "明日の会議は何時からですか", None, "", now=now) == [meeting]
    # A message that is one particle names nothing, so no turn that holds it counts.
    for particle in "のはがをにへでともかねよ":
        assert ranked(corpus, particle, None, "", now=now) == []


def test_ranked_with_the_default_bars_is_ranked_with_the_constants() -> None:
    corpus = Corpus()
    for item in (
        doc_item("d:1", "I stake the tomato garden with bamboo.", 0.70),
        doc_item("d:2", "the tomato garden gets the sun", 0.40),
        doc_item("d:3", "a dog barks in the yard", 0.58),
        doc_item("d:4", "zebra yodel contest tickets", 0.20),
    ):
        corpus.add(item)
    for question in (HALF_QUESTION, FAR_QUESTION, "tomato garden"):
        assert ranked(corpus, question, QUERY_VECTOR, MODEL, now=NOW + 1) == ranked(
            corpus,
            question,
            QUERY_VECTOR,
            MODEL,
            now=NOW + 1,
            coverage=MIN_COVERAGE,
            min_cosine=MIN_COSINE,
        )


async def test_a_turn_joins_the_block_above_the_cosine_bar_and_not_below_it(
    tmp_path: Path, store: SessionStore
) -> None:
    for cosine, joins in ((ABOVE, True), (BELOW, False)):
        memory = await memory_with(
            tmp_path / str(cosine), store, config(), cosine=cosine
        )
        try:
            found = await memory.before_turn(
                store, await _turn(store, "Ask"), FAR_QUESTION
            )
            assert [item.key[:2] for item in found.used] == (["d:"] if joins else [])
            assert (TURN[0] in found.block) is joins
        finally:
            await memory.aclose()


async def test_the_search_tool_honours_the_cosine_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    for cosine, joins in ((ABOVE, True), (BELOW, False)):
        memory = await memory_with(
            tmp_path / str(cosine), store, config(), cosine=cosine
        )
        tools = MemoryTools(memory)
        try:
            found = await memory.search(FAR_QUESTION)
            assert [item.key[:2] for item in found] == (["d:"] if joins else [])
            text = await tools.search(query=FAR_QUESTION, scope="chats")
            if joins:
                assert text.startswith("1. ") and TURN[0] in text
            else:
                assert text == NOTHING_FOUND
        finally:
            await memory.aclose()


async def test_half_the_question_words_join_the_block_and_a_quarter_does_not(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = await memory_with(tmp_path, store, config(), note=True)
    try:
        found = await memory.before_turn(
            store, await _turn(store, "Ask"), f"which {HALF_QUESTION}?"
        )
        assert sorted(item.key[:2] for item in found.used) == ["d:", "r:"]
        few = await memory.before_turn(
            store, await _turn(store, "Ask 2"), f"which {FEW_QUESTION}?"
        )
        assert few.used == ()
    finally:
        await memory.aclose()


async def test_the_search_tool_honours_the_coverage_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = await memory_with(tmp_path, store, config(), note=True)
    tools = MemoryTools(memory)
    try:
        found = await memory.search(HALF_QUESTION)
        assert sorted(item.key[:2] for item in found) == ["d:", "r:"]
        assert (await tools.search(query=HALF_QUESTION)).startswith("1. ")
        assert len((await tools.search(query=HALF_QUESTION)).splitlines()) == 2
        assert await memory.search(FEW_QUESTION) == []
        assert await tools.search(query=FEW_QUESTION) == NOTHING_FOUND
    finally:
        await memory.aclose()


async def test_the_llm_retriever_keeps_its_loose_candidates_whatever_the_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    """The model chooses among candidates that took no coverage bar, so a question
    that holds too few of the turn's words for the bar still has them."""
    cfg = config(retriever="llm")
    memory = await memory_with(tmp_path, store, cfg)
    chat = FakeChat(needle="tomato")
    memory.chat_provider = lambda: chat  # type: ignore[method-assign]
    try:
        found = await memory.search(FEW_QUESTION)
        assert [item.key[:2] for item in found] == ["d:"]
        assert chat.calls  # the model was asked about the loose candidates
    finally:
        await memory.aclose()
