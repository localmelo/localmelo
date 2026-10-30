"""Recall across chats: hybrid retrieval over records and the conversation index.

A fixed set of records and earlier turns, and fixed questions: BM25 alone finds
what shares the question's words; with an embedding model (a fake one here)
paraphrases and the other language are found too; questions about nothing
recall nothing. A slow or failing embedding falls back to BM25 for the turn.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import BLOCK_CHARS, BLOCK_HEADER, chat_label
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

HOLDER = "test-host:1:recall"

SAID = {
    "short": "I prefer short answers.",
    "meeting": "My group meeting is on Thursdays.",
    "home": "I live in Pittsburgh.",
    "sam": "Remember that Sam has the robotics lab keys.",
    "dentist": "My dentist appointment is on Nov 3.",
    "tea": "我喜欢喝绿茶。",
    "cat": "我的猫是一只橘猫。",
}
TURNS = {
    "ci": (
        "How do I fix the failing CI test in localmelo?",
        "The CI test failed because the fixture port was taken; use port 0.",
    ),
    "slides": ("Plan the group talk slides.", "Ten slides: problem, method, results."),
    "landlord": ("帮我写一封给房东的邮件", "好的，这是给房东的邮件草稿。"),
}
# Each question, and the record or turn it is about.
QUESTIONS = {
    "when is my group meeting?": "meeting",
    "where do I live?": "home",
    "who has the robotics lab keys?": "sam",
    "how did we fix the failing CI test?": "ci",
    "do you remember the dentist appointment?": "dentist",
    "remind me of my weekly sync day": "meeting",
    "我的组会是什么时候？": "meeting",
    "what do I like to drink?": "tea",
    "when do I see the dentist?": "dentist",
    "do I have a cat?": "cat",
}
# What BM25 alone can find: the questions that share their words.
BY_WORDS = {
    "when is my group meeting?",
    "where do I live?",
    "who has the robotics lab keys?",
    "how did we fix the failing CI test?",
    "do you remember the dentist appointment?",
}
NOTHING = ("hi", "write a poem about autumn", "thanks!", "今天天气怎么样？")


@pytest.fixture()
async def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


async def _turn(store: SessionStore, title: str, session_id: str | None = None):  # type: ignore[no-untyped-def]
    turn = await store.begin_turn(
        session_id,
        title,
        holder=HOLDER,
        title=title,
        new_session=NewSession() if session_id is None else None,
    )
    return turn


async def build(
    tmp_path: Path, store: SessionStore, embedding: FakeEmbedding | None
) -> tuple[ChatMemory, dict[str, str]]:
    """A memory holding SAID as records and TURNS as documents; key -> item key."""
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    if embedding is not None:
        memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    keys: dict[str, str] = {}
    for key, text in SAID.items():
        turn = await _turn(store, f"About {key}")
        found = await memory.before_turn(store, turn, text)
        (note,) = found.remembered
        keys[key] = f"r:{note.id}"
    _, index = memory.stores()
    for typed, answer in TURNS.values():
        turn = await _turn(store, typed)
        text = document_text(typed, answer)
        vec = (await embedding.embed([text]))[0] if embedding else None
        await index.add(
            Document(
                session_id=turn.session_id,
                turn=1,
                surface="gateway",
                kind="chat",
                title=typed,
                text=text,
                created_at=time.time(),
                vector=tuple(vec) if vec else None,
                embed_model=embedding.model if embedding else "",
            )
        )
    ids = {int(n) for n in await index.ids()}
    for doc in await index.fetch(ids):
        key = next(k for k, (typed, _) in TURNS.items() if typed in doc.text)
        keys[key] = f"d:{doc.id}"
    return memory, keys


async def recall_at_3(memory: ChatMemory, keys: dict[str, str]) -> set[str]:
    found = set()
    for question, key in QUESTIONS.items():
        items = await memory.search(question)
        if keys[key] in [item.key for item in items[:3]]:
            found.add(question)
    return found


async def test_recall_at_3_with_words_only(tmp_path: Path, store: SessionStore) -> None:
    memory, keys = await build(tmp_path, store, None)
    try:
        assert await recall_at_3(memory, keys) == BY_WORDS
        for question in NOTHING:
            assert await memory.search(question) == []
    finally:
        await memory.aclose()


async def test_recall_at_3_with_vectors_finds_paraphrases_and_the_other_language(
    tmp_path: Path, store: SessionStore
) -> None:
    embedding = FakeEmbedding()
    memory, keys = await build(tmp_path, store, embedding)
    try:
        assert await recall_at_3(memory, keys) == set(QUESTIONS)
        for question in NOTHING:
            assert await memory.search(question) == []
    finally:
        await memory.aclose()


async def test_the_qwen3_query_instruction_goes_on_queries_only(
    tmp_path: Path, store: SessionStore
) -> None:
    embedding = FakeEmbedding(model="Qwen3-Embedding-0.6B")
    memory, _ = await build(tmp_path, store, embedding)
    try:
        embedding.calls.clear()
        turn = await _turn(store, "Ask")
        await memory.before_turn(store, turn, "I prefer tea. When is my meeting?")
        (batch,) = embedding.calls
        assert batch[0].startswith("Instruct: ") and batch[0].endswith(
            "Query: I prefer tea. When is my meeting?"
        )
        assert batch[1:] == ["You prefer tea."]
    finally:
        await memory.aclose()


@pytest.mark.parametrize("mode", ["slow", "down"])
async def test_a_slow_or_failing_embedding_falls_back_to_words_and_logs_once(
    tmp_path: Path,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from localmelo.support.memory import service

    embedding = FakeEmbedding()
    memory, keys = await build(tmp_path, store, embedding)
    monkeypatch.setattr(service, "EMBED_TIMEOUT_S", 0.2)
    embedding.delay_s, embedding.fail = (1.0, False) if mode == "slow" else (0.0, True)
    try:
        caplog.set_level(logging.DEBUG, logger=service.__name__)
        started = time.monotonic()
        for _ in range(2):
            turn = await _turn(store, "Ask")
            found = await memory.before_turn(store, turn, "when is my group meeting?")
            assert [item.key for item in found.used] == [keys["meeting"]]
        assert time.monotonic() - started < 1.5
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "words only" in warnings[0].getMessage()
        assert memory.last_error.startswith("fake-embed: ")
    finally:
        await memory.aclose()


async def test_the_block_names_each_source_and_stays_in_bounds(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = await build(tmp_path, store, None)
    try:
        turn = await _turn(store, "Ask")
        found = await memory.before_turn(store, turn, "when is my group meeting?")
        block = found.block
        today = time.localtime()
        day = f"{time.strftime('%b', today)} {today.tm_mday}"
        assert block == (
            f"\n\n{BLOCK_HEADER}\n- the user said “My group meeting is on "
            f"Thursdays.” (chat “About meeting”, {day})"
        )
        (used,) = found.event()["used"]  # type: ignore[index]
        assert used["text"] == "Your group meeting is on Thursdays."
        assert used["source"] == f"you said it · chat “About meeting”, {day}"
        assert len(block) <= BLOCK_CHARS + 2
    finally:
        await memory.aclose()


async def test_documents_of_this_chat_are_not_recalled_and_two_per_chat_at_most(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        _, index = memory.stores()
        first = await _turn(store, "Garden")
        for n in range(4):
            await index.add(
                Document(
                    session_id=first.session_id,
                    turn=n + 1,
                    surface="gateway",
                    kind="chat",
                    title="Garden",
                    text=document_text(f"tomato garden plan {n}", "water daily"),
                    created_at=time.time(),
                )
            )
        items = await memory.search("tomato garden", docs_k=8, records_k=0)
        assert len(items) == 2
        here = await memory.search(
            "tomato garden", docs_k=8, exclude_session=first.session_id
        )
        assert here == []
        assert await memory.forget_session(first.session_id) == 4
        assert await memory.search("tomato garden", docs_k=8) == []
    finally:
        await memory.aclose()


async def test_old_documents_rank_below_new_ones(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        _, index = memory.stores()
        for age_days, title in ((90, "Old"), (1, "New")):
            turn = await _turn(store, title)
            await index.add(
                Document(
                    session_id=turn.session_id,
                    turn=1,
                    surface="gateway",
                    kind="task" if title == "New" else "chat",
                    title=title,
                    text=document_text("tomato garden plan", "water daily"),
                    created_at=time.time() - age_days * 86_400,
                )
            )
        items = await memory.search("tomato garden plan", docs_k=2, records_k=0)
        assert [item.where.split(",")[0] for item in items] == [
            "task “New”",
            "chat “Old”",
        ]
    finally:
        await memory.aclose()


def test_chat_labels() -> None:
    assert chat_label("front", "x", "gateway") == "Main chat"
    assert chat_label("chat", "Fix CI", "gateway") == "chat “Fix CI”"
    assert chat_label("chat", "Fix CI", "cli") == "terminal chat “Fix CI”"
    assert chat_label("task", "Find jobs", "gateway") == "task “Find jobs”"
    assert chat_label("chat", "", "gateway") == "a chat"
    assert chat_label("chat", "x" * 80, "gateway").endswith("…”")


async def test_search_memory_answers_with_sources_and_scopes(
    tmp_path: Path, store: SessionStore
) -> None:
    from localmelo.support.memory.service import SEARCH_MEMORY

    memory, _ = await build(tmp_path, store, None)
    try:
        (tool,) = await memory.tools().start()
        assert tool.definition is SEARCH_MEMORY and not tool.requires_approval
        assert tool.definition.annotations == {"readOnlyHint": True}
        found = str(await tool.function(query="robotics lab keys"))
        assert found.startswith(
            "1. the user said “Sam has the robotics lab keys.” (chat"
        )
        assert await tool.function(query="robotics lab keys", scope="chats") == (
            "Nothing in melo's memory matches that."
        )
        chats = str(
            await tool.function(query="fix the failing CI test", scope="chats", k=20)
        )
        assert chats.startswith("1. User: How do I fix the failing CI test")
    finally:
        await memory.aclose()


def test_a_document_keeps_no_sentence_a_note_would_refuse() -> None:
    text = document_text(
        "My wifi password is hunter2-lighthouse. Plan the garden.",
        "Your card 4111 1111 1111 1111 is noted. Water daily.",
    )
    assert "hunter2" not in text and "4111" not in text
    assert text == "User: [not kept] Plan the garden.\nmelo: [not kept] Water daily."
