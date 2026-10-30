"""Seeing and changing what melo remembers: edit, delete, forget everything, reindex.

An edit rewrites the note in place and reaches the index (the new wording is
recalled, the old one is not); a delete or a forget-all leaves no text in the
decision log or the file; a deleted note is never recalled again; a note
outlives a deleted chat; a reindex rebuilds the chat index from the chats and
gives the notes a vector of the model now set, and does not bring back what the
user forgot.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

HOLDER = "test-host:1:manage"


@pytest.fixture()
async def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path):  # type: ignore[no-untyped-def]
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def _chat(store: SessionStore, title: str, text: str, answer: str = "") -> str:
    turn = await store.begin_turn(None, text, holder=HOLDER, new_session=NewSession())
    await store.rename_session(turn.session_id, title)
    await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer=answer,
    )
    return turn.session_id


async def _say(memory: ChatMemory, store: SessionStore, sid: str, text: str):  # type: ignore[no-untyped-def]
    turn = (await store.list_turns(sid))[-1]
    return await memory.before_turn(store, turn, text)


async def _recalled(memory: ChatMemory, query: str) -> list[str]:
    return [item.text for item in await memory.search(query)]


def _with_model(memory: ChatMemory, embedding: FakeEmbedding) -> None:
    memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]


def _file_text(path: Path) -> bytes:
    """The database file and its write-ahead log, as bytes."""
    return b"".join(
        p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists()
    )


async def test_an_edit_rewrites_the_note_and_reaches_the_index(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    sid = await _chat(store, "Planning", "My dentist is Lee Park.")
    (note,) = (await _say(memory, store, sid, "My dentist is Lee Park.")).remembered
    assert await _recalled(memory, "who is my dentist?") == [
        "Your dentist is Lee Park."
    ]
    edited = await memory.edit_record(note.id, "Your dentist is Kim Soo.")
    assert edited is not None and edited.id == note.id
    assert edited.provenance["edited"] is True
    assert edited.provenance["quote"] == "Your dentist is Kim Soo."
    assert await _recalled(memory, "who is my dentist?") == ["Your dentist is Kim Soo."]
    assert [e.text for e in await memory.records()] == ["Your dentist is Kim Soo."]
    path = tmp_path / "memory" / "records.db"
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT action, text, text_sha256 FROM decisions"
        ).fetchall()
        assert conn.execute("SELECT COUNT(*) FROM long_term").fetchone() == (1,)
        assert conn.execute("SELECT edited_at > 0 FROM long_term").fetchone() == (1,)
    assert rows == [("NEW", None, ""), ("EDIT", None, "")]
    assert b"Lee Park" not in _file_text(path)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("  ", "cannot be empty"),
        ("x" * 301, "at most 300"),
        ("My password is hunter2 for the wifi", "password or key"),
    ],
)
async def test_an_edit_that_is_empty_too_long_or_a_secret_is_refused(
    memory: ChatMemory, store: SessionStore, text: str, why: str
) -> None:
    sid = await _chat(store, "Planning", "I prefer short answers.")
    (note,) = (await _say(memory, store, sid, "I prefer short answers.")).remembered
    with pytest.raises(ValueError, match=why):
        await memory.edit_record(note.id, text)
    assert [e.text for e in await memory.records()] == ["You prefer short answers."]
    assert await memory.edit_record(999, "You like tea.") is None


async def test_an_edited_note_is_embedded_with_the_model_now_set(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    _with_model(memory, FakeEmbedding())
    try:
        sid = await _chat(store, "Planning", "My group meeting is on Thursdays.")
        (note,) = (
            await _say(memory, store, sid, "My group meeting is on Thursdays.")
        ).remembered
        await memory.edit_record(note.id, "Your standup is every Friday.")
        (entry,) = await memory.records()
        assert entry.embed_model == "fake-embed" and entry.vector is not None
        # A paraphrase only the vectors can match finds the new wording.
        assert await _recalled(memory, "remind me of my weekly sync day") == [
            "Your standup is every Friday."
        ]
    finally:
        await memory.aclose()


async def test_a_deleted_note_is_never_recalled_again(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    sid = await _chat(store, "Planning", "My dentist is Lee Park.")
    (note,) = (await _say(memory, store, sid, "My dentist is Lee Park.")).remembered
    assert await _recalled(memory, "who is my dentist?")
    assert await memory.delete_record(note.id)
    assert await _recalled(memory, "who is my dentist?") == []
    assert await memory.records() == []
    with sqlite3.connect(tmp_path / "memory" / "records.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM long_term").fetchone() == (0,)


async def test_deleting_a_note_takes_the_wordings_it_replaced_along(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    sid = await _chat(store, "Planning", "I live in Boston.")
    await _say(memory, store, sid, "I live in Boston.")
    (new,) = (await _say(memory, store, sid, "I live in Pittsburgh now.")).remembered
    assert await memory.delete_record(new.id)
    path = tmp_path / "memory" / "records.db"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM long_term").fetchone() == (0,)
    assert b"Boston" not in _file_text(path) and b"Pittsburgh" not in _file_text(path)


async def test_forget_all_empties_both_files_and_blanks_the_decision_text(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    sid = await _chat(store, "Planning", "My dentist is Lee Park.", "Noted, Lee Park.")
    turn = (await store.list_turns(sid))[-1]
    await _say(memory, store, sid, "My dentist is Lee Park.")
    memory.after_turn(store, turn, "My dentist is Lee Park.", "Noted, Lee Park.")
    await memory.drain()
    assert await memory.forget_all() == (1, 1)
    assert await memory.records() == []
    assert (await memory.status())["documents"] == 0
    assert await _recalled(memory, "who is my dentist?") == []
    folder = tmp_path / "memory"
    with sqlite3.connect(folder / "records.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone() == (0,)
    for name in ("records.db", "index.db"):
        assert b"Lee Park" not in _file_text(folder / name), name


async def test_a_note_outlives_its_chat_and_can_be_forgotten_with_it(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await _chat(store, "Planning", "I prefer short answers.")
    second = await _chat(store, "Lab", "My dentist is Lee Park.")
    await _say(memory, store, first, "I prefer short answers.")
    await _say(memory, store, second, "My dentist is Lee Park.")
    assert await store.delete_session(first)
    await memory.forget_session(first)
    assert len(await memory.records()) == 2, "deleting a chat keeps its notes"
    assert await memory.forget_chat_notes(first) == 1
    assert [e.session_id for e in await memory.records()] == [second]
    assert await memory.forget_chat_notes(first) == 0


async def test_a_reindex_rebuilds_the_index_from_the_chats(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await _chat(
        store,
        "CI",
        "How do I fix the failing CI test in localmelo?",
        "The CI test failed because the fixture port was taken; use port 0.",
    )
    await _chat(store, "Nothing yet", "A question with no answer yet")
    worker = await store.begin_turn(
        None, "Run the checks", holder=HOLDER, new_session=NewSession(kind="worker")
    )
    await store.finish_turn(
        worker.session_id,
        worker.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="All checks passed.",
    )
    assert await memory.reindex(store) == (0, 1), "a worker run is not indexed"
    assert (await memory.status())["documents"] == 1
    found = await memory.search("how did we fix the failing CI test?")
    assert [item.session_id for item in found] == [sid]
    assert await memory.reindex(store) == (0, 1), "a second reindex changes nothing"
    assert (await memory.status())["documents"] == 1
    assert await store.delete_session(sid)
    assert await memory.reindex(store) == (0, 0)
    assert (await memory.status())["documents"] == 0


async def test_a_reindex_gives_the_notes_a_vector_of_the_model_now_set(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        sid = await _chat(store, "Planning", "My group meeting is on Thursdays.")
        await _say(memory, store, sid, "My group meeting is on Thursdays.")
        assert (await memory.records())[0].vector is None
        _with_model(memory, FakeEmbedding())
        assert await memory.reindex(None) == (1, 0)
        (entry,) = await memory.records()
        assert entry.embed_model == "fake-embed" and entry.vector is not None
        assert await memory.reindex(None) == (
            0,
            0,
        ), "a note with the model's vector is left"
        assert await _recalled(memory, "remind me of my weekly sync day")
    finally:
        await memory.aclose()


async def test_a_reindex_does_not_bring_back_what_the_user_forgot(
    memory: ChatMemory, store: SessionStore
) -> None:
    await _chat(store, "Old", "How do I fix the failing CI test?", "Use port 0.")
    await memory.reindex(store)
    assert (await memory.status())["documents"] == 1
    await memory.forget_all()
    assert await memory.reindex(store) == (0, 0)
    assert (await memory.status())["documents"] == 0
    await _chat(store, "New", "Where is the fixture?", "In tests/conftest.py.")
    assert await memory.reindex(store) == (0, 1), "only what came after is indexed"
