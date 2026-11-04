"""Words the person adds while melo works are indexed with that turn.

A note the run read (its ``inbox.accepted`` event) is part of the turn's words: it
is indexed after the typed words, behind a mark, in read order, by the live index
and by a rebuild alike. A note the run never read becomes the next turn's typed
text and is indexed there, once. A turn without notes is indexed as it always was.
The person rule of the typed words holds for the notes: a turn the person did not
begin holds none of their words, a worker's turn is not indexed at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import document_text
from localmelo.support.memory.service import (
    _NOTE_LEAD,
    ADDED_MARK,
    turn_words,
    with_read_notes,
)
from localmelo.support.sessions.attachments import TextAttachment, compose_user_text
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.sessions.helpers import core_event

HOLDER = "test-host:1:notes"
TYPED = "Plan my trip to Denver."
ANSWER = "Your trip is planned."
LEAD = f"{_NOTE_LEAD}\n"  # what a console chat's note reads like to the run


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def _turn(
    store: SessionStore,
    text: str = TYPED,
    *,
    read: tuple[str, ...] = (),
    unread: tuple[str, ...] = (),
    session_id: str | None = None,
    kind: str = "chat",
    title: str | None = None,
    origin: str = "",
    attachments: tuple[TextAttachment, ...] = (),
    typed_text: str | None = None,
    answer: str = ANSWER,
) -> TurnInfo:
    """A finished turn: *read* notes have an ``inbox.accepted`` event (one per step
    from step 2), *unread* ones only the ``turn.note`` event the page stores."""
    info = await store.begin_turn(
        session_id,
        text,
        holder=HOLDER,
        title=title,
        new_session=NewSession(kind=kind) if session_id is None else None,
        origin=origin,
        attachments=attachments,
        typed_text=typed_text,
    )
    events = [
        core_event(
            "inbox.accepted",
            seq=number,
            offset=number,
            text=note,
            step=number + 1,
            attempt=1,
        )
        for number, note in enumerate(read, 1)
    ]
    events += [
        {
            "v": 1,
            "type": "turn.note",
            "run_id": "",
            "seq": None,
            "ts": 1.0,
            "data": {
                "turn": info.turn,
                "turn_id": info.id,
                "note": 100 + number,
                "text": note,
            },
        }
        for number, note in enumerate(unread, 1)
    ]
    if events:
        await store.append_events(info.session_id, info.turn, events, holder=HOLDER)
    await store.finish_turn(
        info.session_id,
        info.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer=answer,
    )
    return (await store.list_turns(info.session_id))[-1]


async def _documents(memory: ChatMemory) -> list[tuple[str, int, int, str]]:
    _, index = memory.stores()
    docs = await index.fetch(await index.ids())
    return sorted((d.session_id, d.turn, d.chunk, d.text) for d in docs)


async def _index(
    memory: ChatMemory, store: SessionStore, info: TurnInfo, typed: str
) -> list[str]:
    """The live path for *info*: the texts of its documents."""
    memory.after_turn(store, info, typed, ANSWER)
    await memory.drain()
    return [
        text for sid, _, _, text in await _documents(memory) if sid == info.session_id
    ]


async def test_a_note_the_run_read_is_indexed_after_the_typed_words_with_a_mark(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store, read=(f"{LEAD}actually my flight is Tuesday",))
    assert await _index(memory, store, info, TYPED) == [
        f"User: {TYPED}\n\n{ADDED_MARK}\nactually my flight is Tuesday\nmelo: {ANSWER}"
    ]


async def test_two_notes_keep_their_read_order_whatever_their_form(
    memory: ChatMemory, store: SessionStore
) -> None:
    # A console chat's note carries the console's lead; a task's does not.
    info = await _turn(
        store,
        read=(f"{LEAD}actually my flight is Tuesday", "and I want a window seat"),
    )
    (text,) = await _index(memory, store, info, TYPED)
    assert text == (
        f"User: {TYPED}\n\n"
        f"{ADDED_MARK}\nactually my flight is Tuesday\n\n"
        f"{ADDED_MARK}\nand I want a window seat\n"
        f"melo: {ANSWER}"
    )
    assert "[The user added" not in text


async def test_a_note_the_run_never_read_is_indexed_once_as_the_next_turn(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await _turn(store, unread=("use the blue one",))
    # The run ended first: the note is the chat's next message, as typed.
    second = await _turn(store, "use the blue one", session_id=first.session_id)
    await _index(memory, store, first, TYPED)
    await _index(memory, store, second, "use the blue one")
    texts = {turn: text for _, turn, _, text in await _documents(memory)}
    assert texts[first.turn] == f"User: {TYPED}\nmelo: {ANSWER}"
    assert texts[second.turn] == f"User: use the blue one\nmelo: {ANSWER}"
    assert sum("use the blue one" in text for text in texts.values()) == 1


async def test_a_turn_without_notes_is_indexed_exactly_as_before(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store)
    assert await _index(memory, store, info, TYPED) == [document_text(TYPED, ANSWER)]
    assert document_text(TYPED, ANSWER) == f"User: {TYPED}\nmelo: {ANSWER}"
    for typed in (TYPED, "", "  two  spaces \n", "line\n\nbreak\n"):
        assert with_read_notes(typed, []) is typed
        assert with_read_notes(typed, ["", "  ", _NOTE_LEAD]) is typed
        assert await turn_words(store, info, typed) is typed


async def test_a_note_that_holds_nothing_after_the_lead_is_left_out(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store, read=(f"{LEAD}  ", "  ", f"{LEAD}my seat is 4A"))
    (text,) = await _index(memory, store, info, TYPED)
    assert text == f"User: {TYPED}\n\n{ADDED_MARK}\nmy seat is 4A\nmelo: {ANSWER}"


def test_the_picture_lines_of_a_note_are_not_the_persons_words() -> None:
    withheld = (
        "[You cannot see these pictures (this model is not sent pictures); the "
        "person can. Do not describe them; ask the person if it matters.]"
    )
    # A note of pictures only holds the stand-in for them, which is no statement.
    assert with_read_notes("typed", [f"{LEAD}[1 picture]"]) == "typed"
    assert with_read_notes("typed", [f"{LEAD}[3 pictures]"]) == "typed"
    assert with_read_notes("", [f"{LEAD}[2 pictures]\n{withheld}"]) == ""
    # The line to a model that was not sent them is left off the words.
    assert (
        with_read_notes("typed", [f"{LEAD}my seat is 4A\n{withheld}"])
        == f"typed\n\n{ADDED_MARK}\nmy seat is 4A"
    )
    # Words that only mention a picture, or hold the stand-in inside, stay.
    for words in ("[1 picture] is my seat", "see [1 picture]", "[1 picture of me]"):
        assert with_read_notes("", [f"{LEAD}{words}"]) == f"{ADDED_MARK}\n{words}"


async def test_the_picture_lines_memory_drops_are_the_ones_the_console_writes() -> None:
    notes = pytest.importorskip("localmelo.support.gateway.console.notes")
    routes = pytest.importorskip("localmelo.support.gateway.console.routes_turns")
    for count in (1, 2, 4):
        assert with_read_notes("", [f"{LEAD}{routes._marker(count, [])}"]) == ""
        line = notes.withheld_line(count)
        assert with_read_notes("", [f"{LEAD}words\n{line}"]) == f"{ADDED_MARK}\nwords"


async def test_a_note_of_pictures_only_leaves_the_turns_document_as_if_none_was_added(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store, read=(f"{LEAD}[2 pictures]",))
    assert await _index(memory, store, info, TYPED) == [document_text(TYPED, ANSWER)]


async def test_a_note_is_read_for_what_it_states_like_the_typed_words(
    memory: ChatMemory, store: SessionStore
) -> None:
    from localmelo.support.memory import records
    from localmelo.support.memory.index import typed_of

    info = await _turn(store, read=(f"{LEAD}actually my flight is Tuesday",))
    (text,) = await _index(memory, store, info, TYPED)
    assert records.said(typed_of(text)) == records.said("actually my flight is Tuesday")
    assert records.said(typed_of(text)) != ()


async def test_a_secret_in_a_note_stays_out_of_the_index(
    memory: ChatMemory, store: SessionStore
) -> None:
    info = await _turn(store, read=(f"{LEAD}my password is hunter2", "my seat is 4A"))
    (text,) = await _index(memory, store, info, TYPED)
    assert "hunter2" not in text and "my seat is 4A" in text


async def test_the_live_index_and_a_rebuild_give_the_same_documents(
    memory: ChatMemory, store: SessionStore
) -> None:
    long_answer = "The plan has many parts. " * 200  # more than one chunk
    plain = await _turn(store, "Hello there.", answer="Hi.")
    noted = await _turn(
        store,
        read=(f"{LEAD}actually my flight is Tuesday", "and I want a window seat"),
        session_id=plain.session_id,
        answer=long_answer,
    )
    trip = (TextAttachment("trip.txt", "ATTACHED BODY " * 5),)
    filed = await _turn(
        store,
        compose_user_text("Summarize this file.", trip),
        attachments=trip,
        typed_text="Summarize this file.",
        read=(f"{LEAD}keep it short",),
        session_id=plain.session_id,
    )
    other = await _turn(store, read=("my seat is 4A",))
    for info, answer in (
        (plain, "Hi."),
        (noted, long_answer),
        (filed, ANSWER),
        (other, ANSWER),
    ):
        typed = info.typed_text if info.attachments else info.user_text
        memory.after_turn(store, info, typed, answer)
    await memory.drain()
    live = await _documents(memory)
    assert len(live) > 4  # the long turn has several chunks
    assert any(f"{ADDED_MARK}\nkeep it short" in text for *_, text in live)
    assert not any("ATTACHED BODY" in text for *_, text in live)
    assert any(len(text) > 1000 and ADDED_MARK in text for *_, text in live)
    # Documents keep the id order of an index that is rebuilt; compare the text.
    assert await memory.reindex(store) == (0, 4)
    assert await _documents(memory) == live


async def test_the_person_rule_of_the_typed_words_holds_for_the_notes(
    memory: ChatMemory, store: SessionStore
) -> None:
    chat = await _turn(store, "How is the task going?")
    # A task's question to its owner chat is no words of the person's.
    asked = await _turn(
        store,
        "[task t1 · Fix it] Which file?",
        read=("a note read by that turn",),
        session_id=chat.session_id,
        origin="ask_owner",
    )
    assert await store.read_notes(asked.session_id, asked.turn) == [
        "a note read by that turn"
    ]
    assert await turn_words(store, asked, "") == ""
    assert await turn_words(store, asked, "typed") == "typed"
    memory.after_turn(store, asked, "", ANSWER)
    await memory.drain()
    assert await _documents(memory) == []
    assert await memory.reindex(store) == (0, 1)  # only the chat's own turn
    assert not any(
        "a note read by that turn" in text for *_, text in await _documents(memory)
    )


async def test_a_tasks_turn_holds_its_notes_as_a_chat_does_and_a_workers_does_not(
    memory: ChatMemory, store: SessionStore
) -> None:
    task = await _turn(
        store,
        "Count the files in my folder.",
        read=("also count the folders",),
        kind="task",
        title="Count files",
    )
    worker = await _turn(store, "Run the checks", read=("and lint",), kind="worker")
    assert await _index(memory, store, task, "Count the files in my folder.") == [
        "Task: Count files\nUser: Count the files in my folder.\n\n"
        f"{ADDED_MARK}\nalso count the folders\nmelo: {ANSWER}"
    ]
    # A worker's turn is never indexed; a rebuild agrees with the live index.
    assert await memory.reindex(store) == (0, 1)
    assert [sid for sid, *_ in await _documents(memory)] == [task.session_id]
    assert worker.session_id != task.session_id


async def test_the_lead_memory_drops_is_the_one_the_console_puts_before_a_note() -> (
    None
):
    notes = pytest.importorskip("localmelo.support.gateway.console.notes")
    assert notes.NOTE_MARK == _NOTE_LEAD


def test_the_mark_and_the_words_read_back_as_a_paragraph() -> None:
    words = with_read_notes("typed", [f"{LEAD}one", "two"])
    assert words == f"typed\n\n{ADDED_MARK}\none\n\n{ADDED_MARK}\ntwo"
    # No typed words (a message of attachments only): the notes alone.
    assert with_read_notes("  ", ["one"]) == f"{ADDED_MARK}\none"
