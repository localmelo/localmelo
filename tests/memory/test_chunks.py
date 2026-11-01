"""Long turns in the conversation index: chunks, so the text past 2,000 characters counts.

A turn that fits is one document, as before. A longer one is several documents of
the same session and turn: chunk 0 is its head, each later chunk repeats a short
head of the user's words and goes on with the answer where the one before ended,
overlapping it. Recall holds one chunk of a turn at most, forgetting a turn takes
every chunk, a reindex re-chunks, and an index.db made before chunks still opens.
Synthetic text only.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.memory import ChatMemory, ingest
from localmelo.support.memory.index import (
    CHUNK_OVERLAP,
    DOC_CHARS,
    MAX_CHUNKS,
    ConversationIndex,
    Document,
    document_chunks,
    document_text,
    is_cut,
)
from localmelo.support.memory.recall import Item, one_per_turn, pick
from localmelo.support.memory.service import _doc_item
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

QUESTION = "What is the plan for the tomato garden?"
SENTENCE = "Sentence {n} says to water the tomato plants early in the morning. "
# A 7,000-character answer of numbered sentences.
SEVEN_K = "".join(SENTENCE.format(n=n) for n in range(200))[:7_000]
KEY = "sk-" + "a1B2c3D4" * 4


async def ingest_file(
    tmp_path: Path, data: object
) -> tuple[ingest.Report, SessionStore, ChatMemory]:
    """Ingest *data* into files under *tmp_path*; the store and memory stay open."""
    pin = ingest.Clock()
    chats = SessionStore(str(tmp_path / "sessions.db"), writer="gateway", clock=pin)
    await chats.open()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    report = await ingest.run(memory, chats, pin, ingest.parse(data), notes="none")
    return report, chats, memory


def body(chunk: str) -> str:
    """The part of a later chunk that goes on with the answer."""
    return chunk.split("\nmelo: ", 1)[1]


def source(session_id: str = "s1") -> dict[str, Any]:
    return {"session_id": session_id, "kind": "chat", "title": "Garden"}


def test_a_turn_that_fits_is_one_document_as_it_was() -> None:
    assert document_chunks("Hi", "Hello there.") == ["User: Hi\nmelo: Hello there."]
    assert document_chunks("Hi", "Hello.", title="Plan") == [
        "Task: Plan\nUser: Hi\nmelo: Hello."
    ]
    fits = "x" * (DOC_CHARS - len("User: Hi\nmelo: "))
    assert document_chunks("Hi", fits) == ["User: Hi\nmelo: " + fits]
    assert document_text("Hi", fits) == "User: Hi\nmelo: " + fits
    assert not is_cut("Hi", fits)


def test_a_7000_character_answer_is_4_chunks_that_overlap_and_repeat_the_user_head() -> (
    None
):
    chunks = document_chunks(QUESTION, SEVEN_K)
    assert len(chunks) == 4
    assert all(len(chunk) <= DOC_CHARS for chunk in chunks)
    assert chunks[0].startswith(f"User: {QUESTION}\nmelo: Sentence 0 says")
    for chunk in chunks[1:]:
        assert chunk.startswith(f"User: {QUESTION}\nmelo: ")
    # Each chunk goes on inside the one before: its first words are in that
    # chunk's last CHUNK_OVERLAP + a few characters.
    for before, after in zip(chunks, chunks[1:], strict=False):
        first = body(after)[:40]
        at = before.rfind(first)
        assert at >= 0
        assert 100 <= len(before) - at <= CHUNK_OVERLAP + 5
    # Nothing between is lost: every sentence is in some chunk.
    joined = "\n".join(chunks)
    assert all(f"Sentence {n} says" in joined for n in range(0, 105))
    assert not is_cut(QUESTION, SEVEN_K)


def test_chunks_end_at_a_paragraph_a_line_or_a_sentence_not_in_a_word() -> None:
    paragraphs = "\n\n".join(
        f"Paragraph {n} holds a few plain words about the garden bed."
        for n in range(150)
    )
    for chunk in document_chunks("q", paragraphs)[:-1]:
        assert chunk.endswith("garden bed.")
    lines = "\n".join(f"Line {n} lists one more bed of tomatoes" for n in range(300))
    for chunk in document_chunks("q", lines)[:-1]:
        assert chunk.rsplit("\n", 1)[-1].endswith("of tomatoes")
    words = ("tomato " * 2_000).strip()
    for chunk in document_chunks("q", words)[:-1]:
        assert chunk.endswith("tomato")


def test_a_long_user_message_gives_a_short_head_not_the_whole() -> None:
    typed = "Please plan the garden. " + "Beds. " * 400
    chunks = document_chunks(typed, SEVEN_K)
    assert chunks[0].startswith("User: Please plan the garden. Beds.")
    head = chunks[-1].split("\nmelo: ", 1)[0]
    assert head.startswith("User: Please plan the garden. Beds.")
    assert head.endswith("…")
    assert len(head) <= len("User: ") + 200 + 1


def test_a_user_message_longer_than_a_chunk_is_chunked_too() -> None:
    typed = "".join(SENTENCE.format(n=n) for n in range(80))[:5_000]
    chunks = document_chunks(typed, "Noted.")
    assert len(chunks) == 3
    assert all(c.startswith("User: ") for c in chunks)
    assert chunks[-1].endswith("\nmelo: Noted.")
    assert typed.strip()[-40:] in "\n".join(chunks)


def test_a_code_block_that_fits_is_never_split() -> None:
    code = "```python\n" + "\n".join(f"x{n} = {n} * 2" for n in range(60)) + "\n```"
    assert 500 < len(code) < DOC_CHARS - CHUNK_OVERLAP * 2
    # The block starts a little before the first window ends and ends after it,
    # and the only line breaks near that edge are inside it.
    before = ("A sentence of prose. " * 85).strip()  # 1,860 characters
    chunks = document_chunks(QUESTION, f"{before}\n{code}\n{SEVEN_K[:4_000]}")
    assert len(chunks) >= 3
    assert chunks[0].endswith("prose.")
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)
    assert any(code in chunk for chunk in chunks)


def test_a_code_block_that_closes_inside_the_window_ends_the_chunk() -> None:
    code = "```\nline one\n\nline two\n\nline three\n```"
    lead = ("word " * 340).strip()  # about 1,700 characters
    chunks = document_chunks("q", f"{lead}\n{code}\n{SEVEN_K}")
    assert chunks[0].endswith("line three\n```")
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)


def test_a_code_block_too_long_for_a_window_is_split_at_a_line() -> None:
    code = "```\n" + "\n".join(f"line {n} of a very long listing" for n in range(250))
    chunks = document_chunks("q", code + "\n```")
    assert 3 <= len(chunks) <= MAX_CHUNKS
    assert all(len(chunk) <= DOC_CHARS for chunk in chunks)
    assert all(
        body(chunk).splitlines()[-1].endswith("listing") or chunk is chunks[-1]
        for chunk in chunks[1:]
    )


def test_a_giant_turn_is_at_most_8_chunks_and_counted_as_cut() -> None:
    giant = "word " * 20_000
    chunks = document_chunks("q", giant)
    assert len(chunks) == MAX_CHUNKS
    assert all(len(chunk) <= DOC_CHARS for chunk in chunks)
    assert is_cut("q", giant)


def test_secrets_are_redacted_in_every_chunk_also_across_a_window_edge() -> None:
    plain = SEVEN_K
    cuts = [DOC_CHARS - 20, 3_000, 5_700, 6_900]
    for at in cuts:
        text = plain[:at] + f" my key is {KEY} " + plain[at:]
        joined = "\n".join(document_chunks(QUESTION, text))
        assert KEY not in joined
        assert KEY[:12] not in joined
        assert "[left out]" in joined  # the sentence with a key is left out
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n" + "Zm9vYmFy\n" * 600
    chunks = document_chunks(QUESTION, plain[:500] + pem)
    assert not any("Zm9vYmFy" in chunk or "MIIEow" in chunk for chunk in chunks)
    assert KEY not in "\n".join(document_chunks(f"use {KEY}", plain))


def test_a_sentence_that_looks_like_a_password_is_left_out_of_every_chunk() -> None:
    text = (
        SEVEN_K[:3_000] + " My password is hunter22 for the garden. " + SEVEN_K[3_000:]
    )
    joined = "\n".join(document_chunks(QUESTION, text))
    assert "hunter22" not in joined
    assert "[left out]" in joined


async def test_a_long_turn_is_stored_as_chunks_and_chunk_0_keeps_its_place(
    tmp_path: Path,
) -> None:
    embedding = FakeEmbedding()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    try:
        _, index = memory.stores()
        assert await memory.index_turns([(source(), 1, "q", "short", 1.0)]) == (
            1,
            len("User: q\nmelo: short"),
        )
        (before,) = await index.fetch(await index.ids())
        assert (before.turn, before.chunk, before.text) == (
            1,
            0,
            "User: q\nmelo: short",
        )
        docs, chars = await memory.index_turns(
            [(source(), 1, QUESTION, SEVEN_K, 2.0), (source(), 2, "hi", "yo", 3.0)]
        )
        chunks = document_chunks(QUESTION, SEVEN_K)
        assert (docs, chars) == (5, sum(map(len, chunks)) + len("User: hi\nmelo: yo"))
        stored = sorted(
            await index.fetch(await index.ids()), key=lambda d: (d.turn, d.chunk)
        )
        # The turn's old document is replaced, not kept beside its chunks.
        assert [(d.turn, d.chunk) for d in stored] == [
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
        ]
        assert [d.text for d in stored[:4]] == chunks
        assert all(d.vector is not None and d.embed_model for d in stored)
        assert await index.count() == 2  # turns
        assert (await memory.status())["documents"] == 2
        # A shorter version of the turn leaves no old chunk behind.
        await memory.index_turns([(source(), 1, QUESTION, "now short", 4.0)])
        assert sorted(
            (d.turn, d.chunk) for d in await index.fetch(await index.ids())
        ) == [
            (1, 0),
            (2, 0),
        ]
    finally:
        await memory.aclose()


async def test_a_batch_holds_at_most_16_texts_and_a_failed_one_leaves_words_only(
    tmp_path: Path,
) -> None:
    embedding = FakeEmbedding()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    turns = [(source(), n, QUESTION, SEVEN_K, float(n)) for n in range(1, 6)]
    try:
        docs, _ = await memory.index_turns(turns)
        assert docs == 20
        assert max(len(call) for call in embedding.calls) <= 16
        assert sum(len(call) for call in embedding.calls) == 20
        embedding.fail = True
        embedding.calls.clear()
        _, index = memory.stores()
        again, chars = await memory.index_turns(turns)
        assert (again, chars) == (20, 0)
        assert len(embedding.calls) <= 2  # the rest is not tried after a failure
        assert all(d.vector is None for d in await index.fetch(await index.ids()))
    finally:
        await memory.aclose()


async def test_only_the_first_chunk_states_what_the_user_said(tmp_path: Path) -> None:
    typed = "I live in Utrecht. " + "Please plan the garden. " * 30
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        await memory.index_turns([(source(), 1, typed, SEVEN_K, 1.0)])
        _, index = memory.stores()
        docs = sorted(await index.fetch(await index.ids()), key=lambda d: d.chunk)
        items = [_doc_item(doc) for doc in docs]
        assert items[0].said
        assert all(item.said == () for item in items[1:])
        assert len({item.key for item in items}) == len(items)
        assert {(item.session_id, item.turn) for item in items} == {("s1", 1)}
    finally:
        await memory.aclose()


async def test_recall_gives_one_chunk_of_a_turn_the_best_one(tmp_path: Path) -> None:
    middle = (
        "The rhubarb crown needs a cold winter. " * 20
        + "Zinnias attract bees. "
        + SEVEN_K[:1_200]
    )
    answer = SEVEN_K[:2_500] + middle + SEVEN_K[:2_500]
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        await memory.index_turns(
            [
                (source("s1"), 1, "garden", answer, time.time()),
                (source("s2"), 1, "bees", "Zinnias attract bees.", time.time()),
            ]
        )
        found = await memory.search("zinnias bees", exclude_session="")
        keys = [(item.session_id, item.turn) for item in found]
        assert sorted(keys) == [("s1", 1), ("s2", 1)]
        best = next(item for item in found if item.session_id == "s1")
        assert "Zinnias attract bees." in best.text
        loose = await memory.search("tomato plants morning", exclude_session="")
        assert [(i.session_id, i.turn) for i in loose] == [("s1", 1)]
    finally:
        await memory.aclose()


def test_a_list_keeps_one_chunk_of_a_turn_before_the_chats_cap() -> None:
    def doc(key: str, session: str, turn: int) -> Item:
        return Item(key, key, "", session, turn, 1.0)

    items = [
        doc("d:1", "a", 1),
        doc("d:2", "a", 1),  # a later chunk of the same turn
        doc("d:3", "a", 2),
        doc("d:4", "a", 3),
        doc("d:5", "b", 1),
        doc("d:6", "a", 1),
    ]
    assert [i.key for i in one_per_turn(items)] == ["d:1", "d:3", "d:4", "d:5"]
    # Two chunks of one turn do not use the chat's cap of two documents.
    assert [i.key for i in pick(items, records=3, docs=3)] == ["d:1", "d:3", "d:5"]


async def test_forgetting_a_turn_or_a_chat_removes_every_chunk(tmp_path: Path) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        _, index = memory.stores()
        await memory.index_turns(
            [
                (source("s1"), 1, QUESTION, SEVEN_K, 1.0),
                (source("s1"), 2, QUESTION, SEVEN_K, 2.0),
                (source("s2"), 1, QUESTION, SEVEN_K, 3.0),
            ]
        )
        assert len(await index.ids()) == 12
        assert await memory.forget_session("s1", 1) == 1  # one turn, four chunks
        left = await index.fetch(await index.ids())
        assert sorted({(d.session_id, d.turn) for d in left}) == [("s1", 2), ("s2", 1)]
        assert len(left) == 8
        assert await memory.forget_session("s1") == 1
        assert {d.session_id for d in await index.fetch(await index.ids())} == {"s2"}
        assert (await memory.forget_all())[1] == 1
        assert await index.ids() == set()
    finally:
        await memory.aclose()


async def test_a_reindex_chunks_every_turn_again_and_ingest_uses_the_same_path(
    tmp_path: Path,
) -> None:
    turns = [
        ("user", QUESTION),
        ("assistant", SEVEN_K),
        ("user", "thanks"),
        ("assistant", "You are welcome."),
    ]
    data = {
        "sessions": [
            {
                "id": "a",
                "date": "2023-05-20T13:40:00",
                "turns": [{"role": role, "text": text} for role, text in turns],
            }
        ]
    }
    report, chats, memory = await ingest_file(tmp_path, data)
    try:
        assert (report.turns, report.docs) == (2, 5)
        _, index = memory.stores()
        before = sorted(
            (d.turn, d.chunk, d.text) for d in await index.fetch(await index.ids())
        )
        # An index from before chunks holds the first 2,000 characters only.
        await index.forget_all(mark=False)
        old = document_text(QUESTION, SEVEN_K)
        await index.add(
            Document("sid", 1, "gateway", "chat", "t", old[:DOC_CHARS], 1.0)
        )
        assert await memory.reindex(chats) == (0, 2)
        after = sorted(
            (d.turn, d.chunk, d.text) for d in await index.fetch(await index.ids())
        )
        assert after == before
    finally:
        await memory.aclose()
        await chats.aclose()


async def test_an_index_made_before_chunks_opens_with_its_documents_and_ids(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "index.db")
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            turn INTEGER NOT NULL, surface TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'chat', title TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL, created_at REAL NOT NULL, vector BLOB,
            embed_model TEXT NOT NULL DEFAULT '', UNIQUE (session_id, turn));
        CREATE INDEX documents_session ON documents(session_id);
        """)
    for n in range(1, 6):
        raw.execute(
            "INSERT INTO documents (session_id, turn, text, created_at) "
            "VALUES ('old', ?, ?, 1.0)",
            (n, f"User: q{n}\nmelo: a{n}"),
        )
    raw.execute("DELETE FROM documents WHERE id IN (4, 5)")  # the sequence is at 5
    raw.commit()
    raw.close()
    index = ConversationIndex(path)
    try:
        docs = sorted(await index.fetch(await index.ids()), key=lambda d: d.id)
        assert [(d.id, d.turn, d.chunk, d.text) for d in docs] == [
            (n, n, 0, f"User: q{n}\nmelo: a{n}") for n in (1, 2, 3)
        ]
        assert await index.count() == 3
        # A second chunk of a turn fits now, and ids are not used twice.
        await index.add_many(
            [
                Document("old", 3, "", "chat", "", "head", 2.0),
                Document("old", 3, "", "chat", "", "tail", 2.0, chunk=1),
            ]
        )
        stored = await index.fetch(await index.ids())
        assert sorted((d.turn, d.chunk) for d in stored) == [
            (1, 0),
            (2, 0),
            (3, 0),
            (3, 1),
        ]
        assert min(d.id for d in stored if d.turn == 3) > 5
    finally:
        await index.aclose()
    # Opening it again does not move it twice.
    again = ConversationIndex(path)
    try:
        assert await again.count() == 3
    finally:
        await again.aclose()


async def test_delete_counts_turns_and_a_reopened_index_keeps_chunks(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "index.db")
    index = ConversationIndex(path)
    await index.add_many(
        [Document("s", 1, "", "chat", "", f"part {c}", 1.0, chunk=c) for c in range(3)]
    )
    await index.aclose()
    reopened = ConversationIndex(path)
    try:
        assert len(await reopened.ids()) == 3
        assert await reopened.delete("s", 1) == 1
        assert await reopened.ids() == set()
    finally:
        await reopened.aclose()


@pytest.mark.parametrize("size", [1, DOC_CHARS - 1, DOC_CHARS, DOC_CHARS + 1])
def test_chunk_sizes_never_pass_the_document_limit(size: int) -> None:
    chunks = document_chunks("", "w" * size)
    assert all(len(chunk) <= DOC_CHARS for chunk in chunks)
    assert len(chunks) == (1 if size + len("User: \nmelo: ") <= DOC_CHARS else 2)
