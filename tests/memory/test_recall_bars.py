"""The recall bars: how close in meaning, and how many of the question's words.

``[memory] recall_min_cosine`` (default 0.55) is the least similarity a vector
needs to join the vector list and ``recall_min_coverage`` (default 0.6) the
share of the question's content words an item must hold to join the word list.
Both apply to the recall block and to the memory search tool, and a default
changes no ranking. Synthetic notes, turns and fixed vectors, no model.
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
from localmelo.support.memory.service import MemoryTools, RecallSettings
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


def test_ranked_takes_a_vector_at_half_the_cosine_only_under_a_lower_bar() -> None:
    far = doc_item("d:1", "the cat sleeps on the sofa", 0.50)
    corpus = Corpus()
    corpus.add(far)
    now = NOW + 1
    assert ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now) == []
    assert (
        ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.55)
        == []
    )
    low = ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.45)
    assert low == [far]
    # A higher bar than the default takes away what the default keeps.
    near = doc_item("d:2", "a dog barks in the yard", 0.60)
    corpus.add(near)
    both = ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.45)
    assert both == [near, far]  # the closer one first
    assert ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now) == [near]
    assert (
        ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.65)
        == []
    )


def test_ranked_takes_half_the_question_words_only_under_a_lower_coverage() -> None:
    held = doc_item("d:1", "I stake the tomato garden with bamboo.")
    corpus = Corpus()
    corpus.add(held)
    now = NOW + 1
    assert ranked(corpus, HALF_QUESTION, None, "", now=now) == []
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=MIN_COVERAGE) == []
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=0.4) == [held]
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=0.0) == [held]
    # Every word of the question is a bar of 1.0.
    assert ranked(corpus, "tomato garden", None, "", now=now, coverage=1.0) == [held]
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=1.0) == []


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


def test_the_settings_read_the_bars_each_time_and_a_bad_one_is_the_default(
    tmp_path: Path,
) -> None:
    assert ChatMemory(lambda: None).recall_settings() == RecallSettings()
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    assert memory.recall_settings().min_cosine == MIN_COSINE
    assert memory.recall_settings().min_coverage == MIN_COVERAGE
    cfg.memory.recall_min_cosine = 0.45
    cfg.memory.recall_min_coverage = 1  # a whole number is a number
    bars = memory.recall_settings()
    assert (bars.min_cosine, bars.min_coverage) == (0.45, 1.0)
    assert isinstance(bars.min_coverage, float)
    for bad in (0.19, 0.96, True, "0.5", math.nan, math.inf, None):
        cfg.memory.recall_min_cosine = bad  # type: ignore[assignment]
        assert memory.recall_settings().min_cosine == MIN_COSINE, bad
    for bad in (0.09, 1.01, False, "0.5", math.nan, -math.inf, None):
        cfg.memory.recall_min_coverage = bad  # type: ignore[assignment]
        assert memory.recall_settings().min_coverage == MIN_COVERAGE, bad
    # The bounds themselves are allowed.
    cfg.memory.recall_min_cosine, cfg.memory.recall_min_coverage = 0.2, 0.1
    assert memory.recall_settings()[-2:] == (0.2, 0.1)
    cfg.memory.recall_min_cosine, cfg.memory.recall_min_coverage = 0.95, 1.0
    assert memory.recall_settings()[-2:] == (0.95, 1.0)


async def test_a_turn_at_cosine_half_joins_the_block_only_under_a_lower_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    memory = await memory_with(tmp_path, store, cfg, cosine=0.50)
    try:
        ask = await _turn(store, "Ask")
        assert (await memory.before_turn(store, ask, FAR_QUESTION)).used == ()
        cfg.memory.recall_min_cosine = 0.45
        found = await memory.before_turn(
            store, await _turn(store, "Ask 2"), FAR_QUESTION
        )
        assert [item.key[:2] for item in found.used] == ["d:"]
        assert TURN[0] in found.block
        # The setting is read again each turn.
        cfg.memory.recall_min_cosine = 0.55
        again = await memory.before_turn(
            store, await _turn(store, "Ask 3"), FAR_QUESTION
        )
        assert again.used == ()
    finally:
        await memory.aclose()


async def test_the_search_tool_honours_the_cosine_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    memory = await memory_with(tmp_path, store, cfg, cosine=0.50)
    tools = MemoryTools(memory)
    try:
        assert await memory.search(FAR_QUESTION) == []
        assert await tools.search(query=FAR_QUESTION) == NOTHING_FOUND
        cfg.memory.recall_min_cosine = 0.45
        found = await memory.search(FAR_QUESTION)
        assert [item.key[:2] for item in found] == ["d:"]
        text = await tools.search(query=FAR_QUESTION, scope="chats")
        assert text.startswith("1. ") and TURN[0] in text
        cfg.memory.recall_min_cosine = 0.55
        assert await tools.search(query=FAR_QUESTION) == NOTHING_FOUND
    finally:
        await memory.aclose()


async def test_half_the_question_words_join_the_block_only_under_a_lower_coverage(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_records=3, recall_docs=3)
    memory = await memory_with(tmp_path, store, cfg, note=True)
    asked = f"which {HALF_QUESTION}?"
    try:
        none = await memory.before_turn(store, await _turn(store, "Ask"), asked)
        assert none.used == ()
        cfg.memory.recall_min_coverage = 0.4
        found = await memory.before_turn(store, await _turn(store, "Ask 2"), asked)
        assert sorted(item.key[:2] for item in found.used) == ["d:", "r:"]
        cfg.memory.recall_min_coverage = MIN_COVERAGE
        assert (
            await memory.before_turn(store, await _turn(store, "Ask 3"), asked)
        ).used == ()
    finally:
        await memory.aclose()


async def test_the_search_tool_honours_the_coverage_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    memory = await memory_with(tmp_path, store, cfg, note=True)
    tools = MemoryTools(memory)
    try:
        assert await memory.search(HALF_QUESTION) == []
        assert await tools.search(query=HALF_QUESTION) == NOTHING_FOUND
        cfg.memory.recall_min_coverage = 0.4
        found = await memory.search(HALF_QUESTION)
        assert sorted(item.key[:2] for item in found) == ["d:", "r:"]
        assert (await tools.search(query=HALF_QUESTION)).startswith("1. ")
        assert len((await tools.search(query=HALF_QUESTION)).splitlines()) == 2
        cfg.memory.recall_min_coverage = 0.6
        assert await tools.search(query=HALF_QUESTION) == NOTHING_FOUND
    finally:
        await memory.aclose()


async def test_the_llm_retriever_keeps_its_loose_candidates_whatever_the_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    """The model chooses among candidates that took no coverage bar; a strict
    ``recall_min_coverage`` does not remove them."""
    cfg = config(retriever="llm", recall_min_coverage=1.0)
    memory = await memory_with(tmp_path, store, cfg)
    chat = FakeChat(needle="tomato")
    memory.chat_provider = lambda: chat  # type: ignore[method-assign]
    try:
        found = await memory.search(HALF_QUESTION)
        assert [item.key[:2] for item in found] == ["d:"]
        assert chat.calls  # the model was asked about the loose candidates
    finally:
        await memory.aclose()
