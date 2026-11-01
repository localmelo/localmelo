"""``support.memory.ingest``: past chats from a file, written as if they had happened.

Synthetic chats in English and Chinese dated 2023: the file is checked, its
messages pair into chat turns, and a run writes ordinary chat sessions through
the real store, each turn stamped with its own time, indexes every turn (words
only, or with a fake embedding model) and keeps the rule-path notes of the user's
words at the time they were said. The chat model is never called.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.support.memory import ChatMemory, ingest
from localmelo.support.memory.index import DOC_CHARS, MAX_CHUNKS, document_chunks
from localmelo.support.memory.ingest import Chat, Exchange, IngestError, Message
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

MAY_20 = datetime(2023, 5, 20, 13, 40).timestamp()
JUNE_1 = datetime(2023, 6, 1, 9, 0).timestamp()


def chat(
    label: str, date: str, *turns: tuple[str, str], **extra: Any
) -> dict[str, Any]:
    return {
        "id": label,
        "date": date,
        "turns": [{"role": role, "text": text} for role, text in turns],
        **extra,
    }


EN = chat(
    "en",
    "2023-05-20T13:40:00",
    ("user", "I live in Pittsburgh."),
    ("assistant", "Pittsburgh is a lovely city."),
    ("user", "What is a good book about trains?"),
    ("assistant", "Try a history of the railroad."),
    title="Moving",
)
ZH = chat(
    "zh",
    "2023-06-01T09:00:00",
    ("user", "我的猫是一只橘猫。"),
    ("assistant", "橘猫很可爱。"),
)


def messages(*items: tuple[str, str]) -> list[Message]:
    return [Message(role, text, float(at)) for at, (role, text) in enumerate(items)]


async def ingest_file(
    tmp_path: Path,
    data: object,
    *,
    notes: str = "rules",
    embedding: FakeEmbedding | None = None,
) -> tuple[ingest.Report, SessionStore, ChatMemory]:
    """Run an ingest of *data* into files under *tmp_path*; the store and memory stay open."""
    pin = ingest.Clock()
    chats = SessionStore(str(tmp_path / "sessions.db"), writer="gateway", clock=pin)
    await chats.open()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    if embedding is not None:
        memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    report = await ingest.run(memory, chats, pin, ingest.parse(data), notes=notes)
    return report, chats, memory


async def close(chats: SessionStore, memory: ChatMemory) -> None:
    await memory.aclose()
    await chats.aclose()


def test_dates_are_iso_or_epoch_and_a_message_takes_its_sessions_plus_its_place() -> (
    None
):
    (one,) = ingest.parse(
        {
            "sessions": [
                {
                    "id": "x",
                    "date": "2023-05-20T13:40:00Z",
                    "turns": [
                        {"role": "user", "text": "a"},
                        {"role": "assistant", "text": "b"},
                        {"role": "user", "text": "c", "date": 1700000000},
                        {"role": "assistant", "text": "d", "date": "1700000100"},
                        {"role": "user", "text": "e", "date": "2023-05-20"},
                    ],
                }
            ]
        }
    )
    start = datetime.fromisoformat("2023-05-20T13:40:00+00:00").timestamp()
    assert one.at == start
    assert [m.at for m in one.messages] == [
        start,
        start + 1,
        1700000000.0,
        1700000100.0,
        datetime(2023, 5, 20).timestamp(),  # no zone: local time
    ]


def test_a_session_without_a_date_takes_its_first_messages() -> None:
    sessions = [
        {
            "turns": [
                {"role": "user", "text": "a", "date": "2023-05-20T13:40:00"},
                {"role": "assistant", "text": "b"},
            ]
        }
    ]
    (one,) = ingest.parse({"sessions": sessions})
    assert (one.label, one.at) == ("session-1", MAY_20)
    assert one.messages[1].at == MAY_20 + 1


@pytest.mark.parametrize(
    ("data", "said"),
    [
        ([], 'a "sessions" list'),
        ({"sessions": ["x"]}, "session 1: must be an object"),
        ({"sessions": [{"id": "a"}]}, 'session a: needs a "turns" list'),
        (
            {"sessions": [{"id": "a", "date": "last week", "turns": []}]},
            "session a date: 'last week' is not an ISO 8601 date",
        ),
        ({"sessions": [{"id": "a", "turns": []}]}, "session a: has no date"),
        (
            {"sessions": [{"id": "a", "turns": [{"role": "user", "text": "x"}]}]},
            "session a: has no date",
        ),
        (
            chat("a", "2023-05-20", ("system", "x")),
            'session a, turn 1: role must be "user" or "assistant"',
        ),
        (
            {"sessions": [{"id": "a", "date": 1, "turns": [{"role": "user"}]}]},
            "session a, turn 1: text must be a string",
        ),
        (
            {"sessions": [EN, {**EN, "title": "another"}]},
            "session en: the id is used twice, with different turns",
        ),
    ],
)
def test_a_bad_file_is_refused_with_one_line_saying_why(data: Any, said: str) -> None:
    if isinstance(data, dict) and "turns" in data:
        data = {"sessions": [data]}
    with pytest.raises(IngestError, match=said.replace("[", r"\[")):
        ingest.parse(data)


def test_messages_pair_into_chat_turns() -> None:
    pairs, skipped = ingest.exchanges(
        messages(
            ("assistant", "a greeting with no question"),  # skipped
            ("user", "first"),
            ("assistant", "reply one"),
            ("assistant", "and more"),  # continues the reply before it
            ("user", "no reply"),  # another user message follows
            ("user", "second"),
            ("assistant", "reply two"),
            ("user", "  "),  # no text: skipped
            ("user", "last, unanswered"),
        )
    )
    assert skipped == 2
    assert pairs == [
        Exchange("first", "reply one\n\nand more", 1.0, 3.0),
        Exchange("no reply", "", 4.0, 4.0),
        Exchange("second", "reply two", 5.0, 6.0),
        Exchange("last, unanswered", "", 8.0, 8.0),
    ]


async def test_it_writes_chat_sessions_documents_and_notes_at_their_times(
    tmp_path: Path,
) -> None:
    report, chats, memory = await ingest_file(tmp_path, {"sessions": [ZH, EN]})
    try:
        assert (report.sessions, report.turns, report.docs, report.notes) == (
            2,
            3,
            3,
            2,
        )
        assert (report.cut, report.skipped, report.trimmed) == (0, 0, 0)
        assert report.embedded_chars == 0 and report.model == "" and report.error == ""
        sessions = {s.id: s for s in await chats.list_sessions(limit=10, kinds=None)}
        assert set(sessions) == set(report.ids.values())
        assert {s.kind for s in sessions.values()} == {"chat"}
        en = sessions[report.ids["en"]]
        assert en.title == "Moving" and en.created_at == MAY_20
        zh = sessions[report.ids["zh"]]
        assert zh.title == "我的猫是一只橘猫。" and zh.created_at == JUNE_1
        turns = await chats.list_turns(en.id)
        assert [(t.turn, t.status, t.user_text) for t in turns] == [
            (1, "completed", "I live in Pittsburgh."),
            (2, "completed", "What is a good book about trains?"),
        ]
        assert [(t.started_at, t.finished_at) for t in turns] == [
            (MAY_20, MAY_20 + 1),
            (MAY_20 + 2, MAY_20 + 3),
        ]
        assert turns[1].answer == "Try a history of the railroad."
        _, index = memory.stores()
        docs = sorted(await index.fetch(await index.ids()), key=lambda d: d.created_at)
        assert [(d.session_id, d.turn, d.created_at) for d in docs] == [
            (en.id, 1, MAY_20 + 1),
            (en.id, 2, MAY_20 + 3),
            (zh.id, 1, JUNE_1 + 1),
        ]
        assert docs[0].text == (
            "User: I live in Pittsburgh.\nmelo: Pittsburgh is a lovely city."
        )
        assert {d.vector for d in docs} == {None} and {d.kind for d in docs} == {"chat"}
        notes = sorted(await memory.records(), key=lambda n: n.created_at)
        assert [(n.text, n.created_at) for n in notes] == [
            ("You live in Pittsburgh.", MAY_20),
            ("你的猫是一只橘猫。", JUNE_1),
        ]
        assert [n.provenance["session_id"] for n in notes] == [en.id, zh.id]
        assert notes[0].provenance["chat"] == "chat “Moving”"
        assert report.seconds >= 0
    finally:
        await close(chats, memory)


async def test_notes_none_writes_no_notes_but_everything_else(tmp_path: Path) -> None:
    report, chats, memory = await ingest_file(
        tmp_path, {"sessions": [EN, ZH]}, notes="none"
    )
    try:
        assert (report.sessions, report.turns, report.docs, report.notes) == (
            2,
            3,
            3,
            0,
        )
        assert await memory.records() == []
        assert await memory.stores()[1].count() == 3
    finally:
        await close(chats, memory)
    with pytest.raises(IngestError, match="notes must be one of rules, none"):
        await ingest.run(
            ChatMemory(lambda: None), None, ingest.Clock(), [], notes="all"  # type: ignore[arg-type]
        )


async def test_a_user_message_without_a_reply_is_a_turn_with_an_empty_answer(
    tmp_path: Path,
) -> None:
    data = {
        "sessions": [
            chat(
                "a",
                "2023-05-20T13:40:00",
                ("assistant", "hello?"),
                ("user", "My dentist is Lee Park."),
                ("user", "Never mind the dentist."),
            )
        ]
    }
    report, chats, memory = await ingest_file(tmp_path, data)
    try:
        assert (report.turns, report.docs, report.skipped) == (2, 2, 1)
        (session,) = await chats.list_sessions(limit=10, kinds=None)
        turns = await chats.list_turns(session.id)
        assert [(t.user_text, t.answer, t.status) for t in turns] == [
            ("My dentist is Lee Park.", "", "completed"),
            ("Never mind the dentist.", "", "completed"),
        ]
        _, index = memory.stores()
        texts = {d.text for d in await index.fetch(await index.ids())}
        assert "User: My dentist is Lee Park.\nmelo: " in texts
    finally:
        await close(chats, memory)


async def test_sessions_are_written_oldest_first_so_a_newer_statement_wins(
    tmp_path: Path,
) -> None:
    newer = chat(
        "new",
        "2023-05-20T13:40:00",
        ("user", "I live in Pittsburgh."),
        ("assistant", "ok"),
    )
    older = chat(
        "old", "2023-01-02T10:00:00", ("user", "I live in Boston."), ("assistant", "ok")
    )
    report, chats, memory = await ingest_file(tmp_path, {"sessions": [newer, older]})
    try:
        assert list(report.ids) == ["old", "new"]
        active = [n.text for n in await memory.records()]
        assert active == ["You live in Pittsburgh."]
    finally:
        await close(chats, memory)


async def test_a_session_with_nothing_to_pair_is_not_created(tmp_path: Path) -> None:
    only_assistant = chat("a", "2023-05-20", ("assistant", "hello"))
    report, chats, memory = await ingest_file(tmp_path, {"sessions": [only_assistant]})
    try:
        assert (report.sessions, report.turns, report.skipped, report.ids) == (
            0,
            0,
            1,
            {},
        )
        assert await chats.list_sessions(limit=10, kinds=None) == []
    finally:
        await close(chats, memory)


async def test_with_an_embedding_model_documents_and_notes_get_vectors(
    tmp_path: Path,
) -> None:
    embedding = FakeEmbedding(model="fake-embed")
    report, chats, memory = await ingest_file(
        tmp_path, {"sessions": [EN, ZH]}, embedding=embedding
    )
    try:
        _, index = memory.stores()
        docs = await index.fetch(await index.ids())
        assert len(docs) == 3
        assert all(d.vector and d.embed_model == "fake-embed" for d in docs)
        assert report.model == "fake-embed" and report.error == ""
        assert report.embedded_chars == sum(len(d.text) for d in docs)
        # The three turns went to the model as one batch, then each note on its own.
        assert [len(call) for call in embedding.calls] == [3, 1, 1]
        notes = await memory.records()
        assert len(notes) == 2 and all(
            n.vector and n.embed_model == "fake-embed" for n in notes
        )
        # Recall finds an ingested turn by meaning.
        found = await memory.search("where does the user live", records_k=3, docs_k=3)
        assert any("Pittsburgh" in item.text for item in found)
    finally:
        await close(chats, memory)


async def test_a_failing_embedding_model_indexes_words_and_says_so(
    tmp_path: Path,
) -> None:
    embedding = FakeEmbedding()
    embedding.fail = True
    report, chats, memory = await ingest_file(
        tmp_path, {"sessions": [EN]}, embedding=embedding
    )
    try:
        assert (report.docs, report.embedded_chars) == (2, 0)
        assert "the embedding server is down" in report.error
        assert "failed: " in report.summary()
        _, index = memory.stores()
        assert all(d.vector is None for d in await index.fetch(await index.ids()))
    finally:
        await close(chats, memory)


async def test_a_long_turn_is_indexed_in_chunks_and_one_past_the_last_is_counted(
    tmp_path: Path,
) -> None:
    long = "word " * (DOC_CHARS // 2)
    huge = "word " * (DOC_CHARS * MAX_CHUNKS)
    data = {
        "sessions": [
            chat(
                "a",
                "2023-05-20T13:40:00",
                ("user", "short question"),
                ("assistant", "short answer"),
                ("user", "tell me everything"),
                ("assistant", long),
                ("user", "tell me even more"),
                ("assistant", huge),
            )
        ]
    }
    report, chats, memory = await ingest_file(tmp_path, data)
    try:
        assert (report.turns, report.cut) == (3, 1)
        expected = [
            document_chunks("short question", "short answer"),
            document_chunks("tell me everything", long),
            document_chunks("tell me even more", huge),
        ]
        assert [len(chunks) for chunks in expected] == [1, 3, MAX_CHUNKS]
        assert report.docs == sum(map(len, expected))
        _, index = memory.stores()
        docs = await index.fetch(await index.ids())
        assert sorted((d.turn, d.chunk) for d in docs) == [
            (turn, chunk)
            for turn, chunks in enumerate(expected, start=1)
            for chunk in range(len(chunks))
        ]
        assert max(len(d.text) for d in docs) <= DOC_CHARS
        assert await index.count() == 3  # turns, not chunks
        (session,) = await chats.list_sessions(limit=10, kinds=None)
        assert len((await chats.list_turns(session.id))[1].answer) == len(
            long
        )  # the chat keeps it all
    finally:
        await close(chats, memory)


async def test_a_user_message_over_the_stores_limit_is_trimmed_and_counted(
    tmp_path: Path,
) -> None:
    data = {
        "sessions": [
            chat(
                "a",
                "2023-05-20T13:40:00",
                ("user", "x" * (MAX_QUERY_LEN + 5)),
                ("assistant", "ok"),
            )
        ]
    }
    report, chats, memory = await ingest_file(tmp_path, data, notes="none")
    try:
        assert (report.turns, report.trimmed) == (1, 1)
        (session,) = await chats.list_sessions(limit=10, kinds=None)
        assert len((await chats.list_turns(session.id))[0].user_text) == MAX_QUERY_LEN
    finally:
        await close(chats, memory)


def test_load_reads_a_json_file_and_refuses_a_missing_or_broken_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chats.json"
    path.write_text(json.dumps({"sessions": [EN]}))
    assert [c.label for c in ingest.load(str(path))] == ["en"]
    with pytest.raises(IngestError, match="cannot read .*missing.json"):
        ingest.load(str(tmp_path / "missing.json"))
    path.write_text("{not json")
    with pytest.raises(IngestError, match="cannot read .*chats.json"):
        ingest.load(str(path))


def test_chat_and_report_shapes() -> None:
    assert Chat("a", "", 1.0, ()).messages == ()
    report = ingest.Report(sessions=1, turns=2, docs=2, notes=1, ids={"a": "b"})
    assert set(report.to_json()) == {
        "sessions",
        "turns",
        "docs",
        "notes",
        "embedded_chars",
        "cut",
        "skipped",
        "trimmed",
        "seconds",
        "model",
        "error",
        "ids",
    }
    assert report.summary().startswith(
        "Ingested 1 session, 2 turns, 2 documents (by words only) and 1 note in "
    )


def test_a_session_repeated_word_for_word_is_read_once() -> None:
    chats = ingest.parse({"sessions": [EN, ZH, EN]})
    assert [chat.label for chat in chats] == ["en", "zh"]
