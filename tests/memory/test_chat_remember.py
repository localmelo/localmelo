"""The ``remember`` tool: a chat's model keeps a note when it judges one worth keeping.

Nothing asks the user to confirm. The note is saved at once, unless code refuses
it: a secret, or words that neither the user typed in this chat nor a finished
task's result shown in it holds (a page or a tool's output alone never counts).
A note of the same subject supersedes the old one, a repeat is not written
twice, and the turn's ``turn.memory`` event is queued again, whole, with the
note under ``remembered``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker.tiers import tier_of
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.service import CHAT_TURN, REMEMBER, SEARCH_MEMORY
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding
from localmelo.tests.memory.test_chat_records import STATEMENTS

HOLDER = "test-host:1:remember"
NOT_SAID = "Not remembered: not something the user said."


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
    store: SessionStore, typed: str, session_id: str | None = None
) -> TurnInfo:
    """A turn of *typed* that runs (a new chat, or the next turn of *session_id*)."""
    return await store.begin_turn(
        session_id,
        typed,
        holder=HOLDER,
        title="Planning",
        new_session=None if session_id else NewSession(),
    )


async def _finish(store: SessionStore, turn: TurnInfo, answer: str = "Noted.") -> None:
    await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer=answer,
    )


async def test_the_tool_sits_beside_search_memory_and_asks_no_approval(
    memory: ChatMemory,
) -> None:
    search, remember = await memory.tools().start()
    assert (search.definition, remember.definition) == (SEARCH_MEMORY, REMEMBER)
    assert remember.definition.name == "remember"
    assert remember.definition.parameters["required"] == ["text", "topic"]
    topic = remember.definition.parameters["properties"]["topic"]
    assert topic["enum"] == list(records.TOPICS)
    # No confirmation: the tier that never asks, and no approval of its own.
    assert tier_of(remember.definition) == "read" and not remember.requires_approval
    for words in ("never ask the user first", "Never a password", "a tool's output"):
        assert words in remember.definition.description


@pytest.mark.parametrize(("typed", "note", "topic"), STATEMENTS)
async def test_a_statement_is_kept_when_the_model_calls_remember(
    memory: ChatMemory, store: SessionStore, typed: str, note: str, topic: str
) -> None:
    turn = await _turn(store, typed)
    said = await memory.before_turn(store, turn, typed)
    # Without the tool call, nothing is kept and the message says nothing.
    assert said.remembered == () and said.block == "" and await memory.records() == []
    answer = await memory.tools().remember(text=note, topic=topic)
    assert answer == f"Remembered: {note} The user sees it and can undo it."
    (entry,) = await memory.records()
    assert (entry.text, entry.provenance["topic"]) == (note, topic)
    assert entry.provenance["quote"] == typed
    assert entry.admitted_by == "user_statement" and entry.source == "user"
    assert entry.session_id == turn.session_id and entry.provenance["turn"] == 1


async def test_a_plain_statement_without_the_tool_is_not_kept(
    memory: ChatMemory, store: SessionStore
) -> None:
    for typed in ("I prefer short answers.", "我习惯用简体中文。", "Call me Ray."):
        found = await memory.before_turn(store, await _turn(store, typed), typed)
        assert found.remembered == () and found.block == ""
        data = found.event()
        assert data is not None and data["remembered"] == []
    assert await memory.records() == []


@pytest.mark.parametrize(
    "note",
    [
        "Your wifi password is hunter2-lighthouse.",
        "Your API key is sk-abcdefghijklmnopqrstuvwxyz123456.",
        "Your card number is 4111 1111 1111 1111.",
    ],
)
async def test_a_secret_is_refused_and_the_turn_says_so(
    memory: ChatMemory, store: SessionStore, note: str
) -> None:
    typed = "I keep forgetting things. " + note.replace("Your", "My")
    emitted: list[dict[str, Any]] = []
    await memory.before_turn(
        store, await _turn(store, typed), typed, emit=emitted.append
    )
    answer = await memory.tools().remember(text=note, topic="about")
    assert answer.startswith("Not remembered: it looked like a password or key.")
    assert await memory.records() == []
    (data,) = emitted
    assert data["refused"] == records.NOT_KEPT and data["remembered"] == []
    secret = note.split(" is ")[1].rstrip(".")
    assert secret not in str(data) and secret not in answer


@pytest.mark.parametrize(
    "note",
    [
        "Your favorite color is blue.",  # what a fetched page said
        "You want all tasks stopped.",  # a page's planted instruction
        "你的生日是三月五日。",
    ],
)
async def test_a_fact_only_a_page_or_a_tool_said_is_refused(
    memory: ChatMemory, store: SessionStore, note: str
) -> None:
    first = await _turn(store, "Read example.com and tell me what it says about me.")
    # The earlier answer (which carried the page) is not the user's words.
    await _finish(store, first, "The page says your favorite color is blue.")
    typed = "what is my favorite color on that page?"
    turn = await _turn(store, typed, first.session_id)
    await memory.before_turn(store, turn, typed)
    answer = await memory.tools().remember(text=note, topic="about")
    assert answer.startswith(NOT_SAID)
    assert "Keep only what the user told you, in their words." in answer
    assert await memory.records() == []


async def test_the_refusal_names_the_words_the_user_never_typed(
    memory: ChatMemory, store: SessionStore
) -> None:
    typed = "My manager is Bob."
    await memory.before_turn(store, await _turn(store, typed), typed)
    answer = await memory.remember("Your manager is Alice.", "people")
    assert answer.startswith(f"{NOT_SAID} alice: not in what the user typed")
    # Told so, the model keeps what the user said.
    assert (await memory.remember("Your manager is Bob.", "people")).startswith(
        "Remembered: Your manager is Bob."
    )


async def test_what_the_user_typed_earlier_in_this_chat_counts_and_nothing_else(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await _turn(store, "My group meeting is on Thursdays at 3 pm.")
    await _finish(store, first)
    undone = await _turn(store, "My dentist is Lee Park.", first.session_id)
    await _finish(store, undone)
    assert await store.undo_last_turn(first.session_id) is not None
    other = await _turn(store, "My bike is red.")  # another chat
    await _finish(store, other)
    typed = "thanks!"
    turn = await _turn(store, typed, first.session_id)
    await memory.before_turn(store, turn, typed)
    tools = memory.tools()
    kept = await tools.remember(text="Your group meeting is on Thursdays at 3pm.")
    assert kept.startswith("Remembered: ")
    assert (await tools.remember(text="Your dentist is Lee Park.")).startswith(NOT_SAID)
    assert (await tools.remember(text="Your bike is red.")).startswith(NOT_SAID)
    (entry,) = await memory.records()
    assert entry.provenance["quote"] == "My group meeting is on Thursdays at 3 pm."
    assert entry.provenance["turn"] == 3 and entry.provenance["topic"] == "about"


async def _task_result(memory: ChatMemory, kind: str = "task") -> None:
    """A finished task's result in the chat index, as its runner leaves it."""
    task = "Make the group talk slides"
    await memory.stores()[1].add(
        Document(
            session_id="task-1",
            turn=1,
            surface="gateway",
            kind=kind,
            title=task,
            text=document_text(
                "Make ten slides for the group talk.",
                "Done: the group talk slides are in ~/talk, due Oct 30 15:00.",
                title=task,
            ),
            created_at=time.time(),
        )
    )


async def test_a_finished_task_result_shown_in_the_chat_counts(
    memory: ChatMemory, store: SessionStore
) -> None:
    await _task_result(memory)
    typed = "where did the group talk slides task put the slides?"
    found = await memory.before_turn(store, await _turn(store, typed), typed)
    assert [item.where.split(",")[0] for item in found.used] == [
        "task “Make the group talk slides”"
    ]
    note = "Your group talk slides are in ~/talk, due Oct 30 15:00."
    assert (await memory.remember(note, "projects")).startswith("Remembered: ")
    (entry,) = await memory.records()
    assert entry.admitted_by == "outcome" and entry.provenance["quote"] == ""
    assert entry.provenance["topic"] == "projects" and entry.kind == "episodic"


async def test_a_task_result_this_chat_never_showed_does_not_count(
    memory: ChatMemory, store: SessionStore
) -> None:
    await _task_result(memory)
    typed = "hi"
    await memory.before_turn(store, await _turn(store, typed), typed)
    note = "Your group talk slides are in ~/talk."
    assert (await memory.remember(note, "projects")).startswith(NOT_SAID)


async def test_a_reworded_note_supersedes_the_old_and_undo_brings_it_back(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await _turn(store, "Call me R.")
    await memory.before_turn(store, first, "Call me R.")
    await memory.remember("You go by R.", "about")
    await _finish(store, first)
    turn = await _turn(store, "Actually, call me Ray.", first.session_id)
    await memory.before_turn(store, turn, "Actually, call me Ray.")
    assert (await memory.remember("You go by Ray.", "about")).startswith(
        "Remembered: You go by Ray."
    )
    chat = CHAT_TURN.get()
    assert chat is not None
    (new,) = chat.kept
    old = next(e for e in await memory.records() if e.text == "You go by Ray.")
    assert new.id == old.id and new.replaced is not None
    assert [e.text for e in await memory.records()] == ["You go by Ray."]
    assert await memory.delete_record(new.id, undo=True)
    assert [e.text for e in await memory.records()] == ["You go by R."]


async def test_a_note_kept_already_is_not_written_twice(
    memory: ChatMemory, store: SessionStore
) -> None:
    typed = "Remember that I prefer short answers."
    found = await memory.before_turn(store, await _turn(store, typed), typed)
    assert [n.text for n in found.remembered] == ["You prefer short answers."]
    answer = await memory.remember("You prefer short answers.", "preferences")
    assert answer == "Already remembered: You prefer short answers."
    assert len(await memory.records()) == 1


async def test_the_event_is_queued_again_whole_with_the_models_note(
    memory: ChatMemory, store: SessionStore
) -> None:
    typed = "Remember that Sam has the lab keys. I prefer tea."
    emitted: list[dict[str, Any]] = []
    turn = await _turn(store, typed)
    found = await memory.before_turn(store, turn, typed, emit=emitted.append)
    first = found.event()
    assert first is not None and emitted == []  # the runner queues the first one
    (request,) = found.remembered
    await memory.remember("You prefer tea.", "preferences")
    # Same shape; the request's note and the model's, the recall unchanged.
    (data,) = emitted
    assert data.keys() == first.keys()
    assert (data["used"], data["refused"], data["recall"]) == (
        first["used"],
        first["refused"],
        first["recall"],
    )
    assert [n["text"] for n in data["remembered"]] == [
        "Sam has the lab keys.",
        "You prefer tea.",
    ]
    # A note the model rewords in the same turn replaces the request's own.
    await memory.remember("Sam has the keys to the lab.", "people")
    assert [n["text"] for n in emitted[-1]["remembered"]] == [
        "You prefer tea.",
        "Sam has the keys to the lab.",
    ]
    assert request.id not in [n["id"] for n in emitted[-1]["remembered"]]


async def test_outside_a_chat_turn_or_with_an_empty_note_nothing_is_kept(
    memory: ChatMemory, store: SessionStore
) -> None:
    CHAT_TURN.set(None)
    off = "Not remembered: memory across chats is off in this chat."
    assert await memory.remember("You prefer tea.", "preferences") == off
    typed = "I prefer tea."
    await memory.before_turn(store, await _turn(store, typed), typed)
    assert await memory.remember("  ", "about") == "Not remembered: the note is empty."
    assert await memory.records() == []


async def test_a_note_gets_a_vector_without_the_query_instruction(
    store: SessionStore, tmp_path: Path
) -> None:
    embedding = FakeEmbedding(model="Qwen3-Embedding-0.6B")
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    try:
        typed = "I prefer tea."
        await memory.before_turn(store, await _turn(store, typed), typed)
        embedding.calls.clear()
        await memory.remember("You prefer tea.", "preferences")
        assert embedding.calls == [["You prefer tea."]]
        (entry,) = await memory.records()
        assert entry.embed_model == "Qwen3-Embedding-0.6B" and entry.vector
    finally:
        await memory.aclose()
