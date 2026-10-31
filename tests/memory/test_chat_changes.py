"""A change the user states replaces the old note, even when the chat only asks.

The benchmark's guard cell r1__M2 (teach item F39): an active note said "Your
weekly sync is on Wednesday.", the user typed "My weekly sync is on Friday.",
and the model made no ``remember`` call and asked "Would you like me to change
it?". Wednesday stayed active and the later question got a mixed answer. After a
completed chat run in which no note was kept, code keeps each plain statement of
the typed words that gives a new value to a subject an active note holds
(``ChatMemory.keep_changes``), as a note that supersedes the old one (with
Undo), whatever ``[memory] keep_statements`` says. A statement of a new subject
is left to the model, and so is a question; a turn that kept a note, a worker,
a task's runner and memory off do nothing. End to end on the stub model, with
words only and with a fake embedding server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.service import CHAT_TURN
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.fixtures.stub_llm import StubServer
from localmelo.tests.memory.chat_memory_helpers import EmbeddingServer
from localmelo.tests.sessions.test_runner_memory import chat as runner_chat
from localmelo.tests.sessions.test_runner_memory import memory_events

WEDNESDAY = "Your weekly sync is on Wednesday."
FRIDAY = "Your weekly sync is on Friday."
ASKED = "I previously noted it as Wednesday. Would you like me to change it?"


def at(stub: StubServer, scenario: str) -> Any:
    """The stub, answering in *scenario* (a runner only needs its base URL)."""
    return SimpleNamespace(base_url=stub.base_url.replace("/v1", f"/s/{scenario}/v1"))


@pytest.fixture()
def stub() -> Iterator[StubServer]:
    remember = {
        "name": "remember",
        "arguments": {"text": "Your name is Juno.", "topic": "about"},
    }
    server = StubServer(
        scenarios={
            "noted": {"replies": [{"content": "Noted."}]},
            "asks": {"replies": [{"content": ASKED}]},
            # The model keeps one note with the tool, and says nothing of the other.
            "kept": {
                "replies": [{"tool_calls": [remember]}, {"content": "Hello Juno."}]
            },
        }
    ).start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture(params=["words", "vectors"])
async def memory(request: Any, tmp_path: Path) -> AsyncIterator[ChatMemory]:
    server = EmbeddingServer().start() if request.param == "vectors" else None
    memory = ChatMemory(
        lambda: None,
        directory=str(tmp_path / "memory"),
        embedding_url=server.base_url if server else "",
        embedding_model="Qwen3-Embedding-0.6B" if server else "",
    )
    try:
        yield memory
    finally:
        await memory.aclose()
        if server is not None:
            server.close()


async def teach(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    """The active note of the real case: "Your weekly sync is on Wednesday."."""
    await runner_chat(
        memory,
        store,
        at(stub, "noted"),
        tmp_path / "t",
        "Remember that my weekly sync is on Wednesday.",
    )
    assert [e.text for e in await memory.records()] == [WEDNESDAY]


async def test_a_stated_change_replaces_the_note_though_the_model_asked(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    await teach(memory, store, stub, tmp_path)
    (old,) = await memory.records()
    typed = "My weekly sync is on Friday."
    events = memory_events(
        await runner_chat(memory, store, at(stub, "asks"), tmp_path / "w", typed)
    )
    # Before the run's end nothing was kept; the last event shows the note, with Undo.
    assert events[0]["remembered"] == [] and events[-1]["unsaved_claim"] is False
    (kept,) = events[-1]["remembered"]
    assert kept["text"] == FRIDAY
    (entry,) = await memory.records()  # Wednesday is history, never recalled
    assert entry.text == FRIDAY and entry.id == kept["id"]
    assert entry.admitted_by == "user_statement" and entry.supersedes == old.id
    assert entry.provenance["quote"] == typed and entry.provenance["replaces"]
    assert await memory.delete_record(entry.id, undo=True)
    assert [e.id for e in await memory.records()] == [old.id]


async def test_a_statement_of_a_new_subject_writes_nothing(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    await teach(memory, store, stub, tmp_path)
    events = memory_events(
        await runner_chat(
            memory, store, at(stub, "asks"), tmp_path / "w", "My dentist is Dr. Lee."
        )
    )
    assert [e.text for e in await memory.records()] == [WEDNESDAY]
    assert events[-1]["remembered"] == []


@pytest.mark.parametrize(
    "typed",
    [
        "Is my sync on Friday?",
        "My weekly sync is not on Friday.",
        "If my weekly sync is on Friday, remind me.",
        "My weekly sync was on Friday.",
        "我的每周例会是周五吗？",
        "My weekly sync is on Wednesday.",  # no new value
    ],
)
async def test_a_question_and_a_statement_that_is_not_plain_change_nothing(
    memory: ChatMemory,
    store: SessionStore,
    stub: StubServer,
    tmp_path: Path,
    typed: str,
) -> None:
    await teach(memory, store, stub, tmp_path)
    events = memory_events(
        await runner_chat(memory, store, at(stub, "asks"), tmp_path / "w", typed)
    )
    assert [e.text for e in await memory.records()] == [WEDNESDAY]
    assert events[-1]["remembered"] == []


async def test_a_turn_that_kept_a_note_is_left_alone(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    await teach(memory, store, stub, tmp_path)
    events = memory_events(
        await runner_chat(
            memory,
            store,
            at(stub, "kept"),
            tmp_path / "w",
            "My name is Juno. My weekly sync is on Friday.",
        )
    )
    # The model's own note stands; the change it left is not chased.
    assert sorted(e.text for e in await memory.records()) == sorted(
        [WEDNESDAY, "Your name is Juno."]
    )
    assert [n["text"] for n in events[-1]["remembered"]] == ["Your name is Juno."]


async def test_only_the_turn_under_way_keeps_a_change(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    await teach(memory, store, stub, tmp_path)
    typed = "My weekly sync is on Friday."
    one = await store.begin_turn(
        None, typed, holder="test:1:changes", title=typed, new_session=NewSession()
    )
    other = await store.begin_turn(
        None, typed, holder="test:1:changes", title=typed, new_session=NewSession()
    )
    assert await memory.keep_changes(one) == []  # no turn under way
    await memory.before_turn(store, other, typed)
    assert await memory.keep_changes(one) == []  # another turn's
    turn = CHAT_TURN.get()
    assert turn is not None and turn.kept == []
    (kept,) = await memory.keep_changes(other)
    assert kept.text == FRIDAY and kept.replaced is not None
    assert [e.text for e in await memory.records()] == [FRIDAY]
    assert await memory.keep_changes(other) == []  # a note was kept now


@pytest.mark.parametrize("keep", [True, False])
async def test_it_runs_whatever_keep_statements_says(
    store: SessionStore, stub: StubServer, tmp_path: Path, keep: bool
) -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory.keep_statements = keep
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        await teach(memory, store, stub, tmp_path)
        typed = "My weekly sync is on Friday."
        await runner_chat(memory, store, at(stub, "asks"), tmp_path / "w", typed)
        assert [e.text for e in await memory.records()] == [FRIDAY]
    finally:
        await memory.aclose()


async def test_workers_task_runners_and_memory_off_do_nothing(
    store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    typed = "My weekly sync is on Friday."
    try:
        await teach(memory, store, stub, tmp_path)
        await runner_chat(
            memory, store, at(stub, "asks"), tmp_path / "w", typed, kind="worker"
        )
        task = await store.create_session(kind="task")
        await runner_chat(
            memory,
            store,
            at(stub, "asks"),
            tmp_path / "x",
            typed,
            session_id=task.id,
            index_only=True,
        )
        cfg.memory.across_chats = False
        events = memory_events(
            await runner_chat(memory, store, at(stub, "asks"), tmp_path / "o", typed)
        )
        cfg.memory.across_chats = True
        assert events == []
        assert [e.text for e in await memory.records()] == [WEDNESDAY]
    finally:
        await memory.aclose()


def test_changes_reads_the_statement_against_the_active_notes() -> None:
    from localmelo.tests.memory.test_third_party_subjects import entry

    notes = [entry(1, WEDNESDAY), entry(2, "Tom Briggs's office is in room 9C.")]
    (cand,) = records.changes("By the way, my weekly sync is on Friday.", notes)
    assert cand.text == FRIDAY and cand.admitted_by == "user_statement"
    assert records.changes("My weekly sync is on Wednesday.", notes) == []
    assert records.changes("My manager is Ana Li.", notes) == []
    assert records.changes("My weekly sync is on Friday.", []) == []
