"""How recall finds notes: ``[memory] retriever`` and the timings of every recall.

The fixed set of notes and turns of ``test_chat_recall`` (the #309 recall set),
with a fake embedding model and a fake chat model. ``auto`` and ``embedding``
are the hybrid; ``bm25`` is words only; ``llm`` lets words pick up to 20
candidates and the chat model choose among them, falling back to the words-only
order on a timeout, an error or a reply it cannot read, and making no embedding
call at all. Every recall reports ``retriever``, ``fallback``, its timings,
the tokens of the choice, ``candidates`` and ``used``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, Recall
from localmelo.support.memory import service as memory_service
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeChat, FakeEmbedding, teach
from localmelo.tests.memory.test_chat_recall import (
    BY_WORDS,
    NOTHING,
    QUESTIONS,
    _turn,
    build,
    recall_at_3,
)

# A word that only the right note or turn of the set holds.
NEEDLES = {
    "when is my group meeting?": "meeting",
    "where do I live?": "pittsburgh",
    "who has the robotics lab keys?": "robotics",
    "how did we fix the failing CI test?": "fixture",
    "do you remember the dentist appointment?": "dentist",
}
PARAPHRASE = "remind me of my weekly sync day"
KEYS = {
    "retriever",
    "fallback",
    "embed_ms",
    "search_ms",
    "select_ms",
    "select_tokens",
    "candidates",
    "used",
    "block_chars",
    "context",
}


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


async def made(
    tmp_path: Path,
    store: SessionStore,
    mode: str,
    embedding: FakeEmbedding | None = None,
) -> tuple[ChatMemory, Config, dict[str, str]]:
    """The recall set built under ``auto``, then switched to *mode*."""
    cfg = Config()
    memory, keys = await build(tmp_path, store, embedding, lambda: cfg)
    cfg.memory.retriever = mode
    return memory, cfg, keys


def use(memory: ChatMemory, chat: FakeChat) -> None:
    memory.chat_provider = lambda: chat  # type: ignore[method-assign]


async def test_llm_finds_what_the_model_picks_among_the_word_candidates(
    tmp_path: Path, store: SessionStore
) -> None:
    assert set(NEEDLES) == BY_WORDS
    memory, _, keys = await made(tmp_path, store, "llm")
    try:
        for question, needle in NEEDLES.items():
            chat = FakeChat(needle)
            use(memory, chat)
            recall = Recall()
            items = await memory.search(question, recall=recall)
            assert keys[QUESTIONS[question]] in [item.key for item in items[:3]]
            assert len(chat.calls) == 1 and "Query: " + question in chat.calls[0]
            assert (recall.retriever, recall.fallback) == ("llm", "none")
            assert 1 <= recall.candidates <= 20 and recall.used == len(items)
            assert recall.select_tokens == {"prompt": 321, "completion": 9}
            assert (recall.embed_ms, recall.select_ms >= 0) == (0, True)
    finally:
        await memory.aclose()


async def test_the_model_decides_not_the_word_order(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, keys = await made(tmp_path, store, "llm")
    try:
        question = "when is my group meeting?"
        use(memory, FakeChat("meeting"))
        first = await memory.search(question)
        assert [item.key for item in first] == [keys["meeting"]]
        # The slides turn shares "group" and ranks below the note by words.
        use(memory, FakeChat("slides"))
        assert [item.key for item in await memory.search(question)] == [keys["slides"]]
        none = Recall()
        use(memory, FakeChat(reply='{"ids": []}'))
        assert await memory.search(question, recall=none) == []
        assert none.fallback == "none" and none.candidates >= 2 and none.used == 0
    finally:
        await memory.aclose()


async def test_llm_sends_nothing_when_no_word_matches(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await made(tmp_path, store, "llm")
    try:
        chat = FakeChat("meeting")
        use(memory, chat)
        for question in (PARAPHRASE, *NOTHING):
            recall = Recall()
            assert await memory.search(question, recall=recall) == []
            assert (recall.candidates, recall.select_ms) == (0, 0)
        assert chat.calls == []
    finally:
        await memory.aclose()


@pytest.mark.parametrize("failure", ["timeout", "error", "unreadable"])
async def test_llm_falls_back_to_the_words_only_order(
    tmp_path: Path,
    store: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    memory, cfg, _ = await made(tmp_path, store, "bm25")
    monkeypatch.setattr(memory_service, "SELECT_TIMEOUT_S", 0.2)
    chat = {
        "timeout": FakeChat("meeting", delay_s=1.0),
        "error": FakeChat(fail=True),
        "unreadable": FakeChat(reply="I would take the first one."),
    }[failure]
    try:
        for question in NEEDLES:
            cfg.memory.retriever = "bm25"
            words = [item.key for item in await memory.search(question)]
            cfg.memory.retriever = "llm"
            use(memory, chat)
            recall = Recall()
            assert [
                i.key for i in await memory.search(question, recall=recall)
            ] == words
            assert words and recall.used == len(words)
            assert (recall.retriever, recall.fallback) == ("llm", failure)
            # A reply that arrived reports its tokens, however unreadable.
            tokens = 321 if failure == "unreadable" else 0
            assert recall.select_tokens["prompt"] == tokens
            assert recall.select_ms < 800
    finally:
        await memory.aclose()


@pytest.mark.parametrize("mode", ["llm", "bm25"])
async def test_these_two_make_no_embedding_call(
    tmp_path: Path, store: SessionStore, mode: str
) -> None:
    embedding = FakeEmbedding()
    memory, _, _ = await made(tmp_path, store, mode, embedding)
    use(memory, FakeChat("meeting"))
    try:
        embedding.calls.clear()
        typed = "I prefer tea. When is my group meeting?"
        turn = await _turn(store, "Ask")
        found = await teach(memory, store, turn, typed)
        memory.after_turn(store, turn, typed, "On Thursdays.")
        await memory.drain()
        await memory.search("where do I live?")
        assert embedding.calls == []
        assert [note.text for note in found.remembered] == ["You prefer tea."]
        assert found.recall is not None and found.recall.embed_ms == 0
        assert found.recall.retriever == mode and found.recall.used == len(found.used)
    finally:
        await memory.aclose()


@pytest.mark.parametrize("mode", ["auto", "embedding"])
async def test_auto_and_embedding_are_the_hybrid(
    tmp_path: Path, store: SessionStore, mode: str
) -> None:
    embedding = FakeEmbedding()
    memory, _, keys = await made(tmp_path, store, mode, embedding)
    try:
        assert await recall_at_3(memory, keys) == set(QUESTIONS)
        recall = Recall()
        await memory.search(PARAPHRASE, recall=recall)
        assert (recall.retriever, recall.fallback) == ("embedding", "none")
        assert recall.candidates >= 1 and recall.used >= 1 and recall.select_ms == 0
        assert recall.embed_ms >= 0 and recall.search_ms >= 0
    finally:
        await memory.aclose()


@pytest.mark.parametrize(
    ("mode", "fallback"), [("auto", "none"), ("embedding", "no-embedding")]
)
async def test_without_an_embedding_model_recall_is_words_only(
    tmp_path: Path, store: SessionStore, mode: str, fallback: str
) -> None:
    memory, _, keys = await made(tmp_path, store, mode)
    try:
        assert await recall_at_3(memory, keys) == BY_WORDS
        recall = Recall()
        await memory.search("when is my group meeting?", recall=recall)
        assert (recall.retriever, recall.fallback) == ("bm25", fallback)
    finally:
        await memory.aclose()


@pytest.mark.parametrize(("what", "fallback"), [("slow", "timeout"), ("down", "error")])
async def test_a_failing_embedding_is_named_in_the_recall(
    tmp_path: Path,
    store: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
    what: str,
    fallback: str,
) -> None:
    embedding = FakeEmbedding()
    memory, _, keys = await made(tmp_path, store, "auto", embedding)
    monkeypatch.setattr(memory_service, "EMBED_TIMEOUT_S", 0.2)
    embedding.delay_s, embedding.fail = (1.0, False) if what == "slow" else (0.0, True)
    try:
        recall = Recall()
        items = await memory.search("when is my group meeting?", recall=recall)
        assert [item.key for item in items] == [keys["meeting"]]
        assert (recall.retriever, recall.fallback) == ("embedding", fallback)
    finally:
        await memory.aclose()


async def test_every_turn_reports_its_recall_even_when_nothing_is_used(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, cfg, _ = await made(tmp_path, store, "llm")
    use(memory, FakeChat("meeting"))
    try:
        for typed in ("hi", "", "when is my group meeting?"):
            turn = await _turn(store, "Ask")
            found = await memory.before_turn(store, turn, typed)
            data = found.event()
            assert data is not None and set(data["recall"]) == KEYS
            assert data["recall"]["retriever"] == "llm"
            assert data["recall"]["used"] == len(data["used"])
            assert data["recall"]["block_chars"] == len(found.block)
            assert all(
                isinstance(data["recall"][key], int)
                for key in ("embed_ms", "search_ms", "select_ms", "candidates")
            )
        assert data is not None
        assert data["used"] and data["recall"]["select_tokens"]["prompt"] == 321
        cfg.memory.across_chats = False
        assert (await memory.before_turn(store, turn, "hi")).event() is None
    finally:
        await memory.aclose()


async def test_a_failed_query_embedding_is_not_tried_again_within_the_turn(
    tmp_path: Path, store: SessionStore
) -> None:
    embedding = FakeEmbedding()
    memory, _, keys = await made(tmp_path, store, "auto", embedding)
    try:
        embedding.calls.clear()
        embedding.fail = True
        turn = await _turn(store, "Ask")
        found = await memory.before_turn(store, turn, "when is my group meeting?")
        assert len(embedding.calls) == 1
        assert [item.key for item in found.used] == [keys["meeting"]]
        assert found.recall is not None and found.recall.fallback == "error"
    finally:
        await memory.aclose()
