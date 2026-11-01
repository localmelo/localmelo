"""The newer turn wins even without a note; lasting statements and the model.

Of two chat turns that state one subject with different values, the older
reads ``(older; replaced by a newer message)`` and comes after the newer one,
which recall brings in when only the older matched; this is the default and
writes no memory. By default only the model keeps a note of a plain statement,
with ``remember``, and its instructions and the tool say to keep a lasting fact
in that turn. With the developer setting ``[memory] keep_statements`` (off by
default, for comparison), code keeps after the turn a plain first-person
statement of a lasting fact that names a subject ("I live in …", "my weekly
sync is on …", "call me …", "我的…是…") that the model kept no note of, with
the same Undo; a question, a to-do, a hypothetical, a negation, the past,
someone else's opinion or a secret never. End to end on the stub model, whose
chats never call ``remember`` here, with words only and with a fake embedding
server. Look-alikes stay apart.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.recall import OLDER, OLDER_TURN, Item, newer_first
from localmelo.support.memory.service import REMEMBER
from localmelo.support.sessions.identity import NOTES_TEXT
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.fixtures.stub_llm import StubServer
from localmelo.tests.memory.chat_memory_helpers import EmbeddingServer
from localmelo.tests.memory.test_chat_newer import LOOKALIKES, PAIRS, Chat
from localmelo.tests.sessions.test_runner_memory import chat as runner_chat

OLD_LABELS = (f"({OLDER})", f"({OLDER_TURN})")


@pytest.fixture()
def stub() -> Iterator[StubServer]:
    """The chat model: it answers and never calls ``remember``."""
    server = StubServer(scenarios={"answer": {"replies": [{"content": "Noted."}]}})
    server.start()
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


def settings(*, keep: bool) -> Config:
    """The settings of a chat without an embedding backend, and the developer
    setting that keeps statements by code on or off."""
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory.keep_statements = keep
    return cfg


@contextlib.asynccontextmanager
async def chat_memory(
    kind: str, tmp_path: Path, *, keep: bool
) -> AsyncIterator[ChatMemory]:
    """Memory across chats with words only, or with a fake embedding server."""
    server = EmbeddingServer().start() if kind == "vectors" else None
    cfg = settings(keep=keep)
    memory = ChatMemory(
        lambda: cfg,
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


@pytest.fixture(params=["words", "vectors"])
async def memory(request: Any, tmp_path: Path) -> AsyncIterator[ChatMemory]:
    """With ``[memory] keep_statements`` on."""
    async with chat_memory(request.param, tmp_path, keep=True) as memory:
        yield memory


@pytest.fixture(params=["words", "vectors"])
async def default_memory(request: Any, tmp_path: Path) -> AsyncIterator[ChatMemory]:
    """As shipped: ``[memory] keep_statements`` off."""
    async with chat_memory(request.param, tmp_path, keep=False) as memory:
        yield memory


def lines_ok(lines: list[str], old: str, new: str) -> None:
    """The newer value first, and the older one on no line (it is history)."""
    assert lines and new in lines[0]
    for line in lines:
        assert old not in line


@pytest.mark.parametrize(
    ("old_typed", "_", "new_typed", "__", "asks", "old", "new"), PAIRS
)
async def test_plain_statements_become_notes_and_the_newer_value_wins(
    memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    tmp_path: Path,
    old_typed: str,
    _: str,
    new_typed: str,
    __: str,
    asks: str,
    old: str,
    new: str,
) -> None:
    chat = Chat(memory, store, stub, tmp_path)
    taught = await chat(old_typed)
    # Before the run nothing; after it, the statement as a note, with Undo.
    assert taught[0]["remembered"] == []
    (first,) = taught[-1]["remembered"]
    assert first["text"] == records.second_person(old_typed)
    taught = await chat(new_typed)
    (kept,) = taught[-1]["remembered"]
    assert kept["text"] == records.second_person(new_typed)
    (entry,) = await memory.records()
    assert entry.text == kept["text"] and entry.admitted_by == "user_statement"
    assert entry.provenance["quote"] == new_typed and entry.provenance["replaces"]

    stub.requests.clear()
    asked = await chat(asks)
    _system, lines = chat.request()
    lines_ok(lines, old, new)
    for used in asked[-1]["used"]:
        assert old not in used["text"]

    # Undo of the newer note brings the older one back.
    assert await memory.delete_record(kept["id"], undo=True)
    assert [e.text for e in await memory.records()] == [first["text"]]
    stub.requests.clear()
    await chat(asks)
    _system, lines = chat.request()
    assert any(old in line and not line.endswith(OLD_LABELS) for line in lines)


@pytest.mark.parametrize(
    ("old_typed", "_", "new_typed", "__", "asks", "old", "new"), PAIRS
)
async def test_without_notes_the_newer_turn_wins(
    default_memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    tmp_path: Path,
    old_typed: str,
    _: str,
    new_typed: str,
    __: str,
    asks: str,
    old: str,
    new: str,
) -> None:
    memory = default_memory
    chat = Chat(memory, store, stub, tmp_path)
    await chat(old_typed)
    await chat(new_typed)
    # The model kept no note and code keeps none by default: two chat turns.
    assert await memory.records() == []

    stub.requests.clear()
    asked = await chat(asks)
    _system, lines = chat.request()
    # The turn with the old value is not given to the model: it is replaced.
    lines_ok(lines, old, new)
    assert not any(old in u["text"] for u in asked[-1]["used"])

    # The search still finds it, after the newer turn, marked older.
    found = await memory.search(old)
    keys = [item.key for item in found]
    newer = next(item for item in found if new in item.text)
    stale = [item for item in found if old in item.text]
    assert stale and all(item.older == OLDER_TURN for item in stale)
    assert not newer.older and keys.index(newer.key) < keys.index(stale[0].key)


@pytest.mark.parametrize(("first", "_", "second", "__"), LOOKALIKES)
async def test_look_alikes_taught_as_statements_stay_apart(
    memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    tmp_path: Path,
    first: str,
    _: Any,
    second: str,
    __: Any,
) -> None:
    chat = Chat(memory, store, stub, tmp_path)
    await chat(first)
    await chat(second)
    notes = await memory.records()
    assert all("replaces" not in e.provenance for e in notes)
    kept = [t for t in (first, second) if records.statements(t)]
    assert len(notes) == len(kept) + sum(
        bool(records.candidates(t)[0]) for t in (first, second)
    )
    for typed in (first, second):
        assert not any(item.older for item in await memory.search(typed))


NOT_NOTES = [
    "Where do I live?",
    "我的健身教练是谁？",
    "我在哪里工作",
    "If I lived in Lisbon, I'd surf every day.",
    "Maybe my weekly sync is on Friday.",
    "如果我住在里斯本就好了。",
    "I don't live in Lisbon anymore.",
    "My name isn't Pip.",
    "我不住在里斯本了。",
    "My mom thinks my name is too plain.",
    "我妈觉得我的名字很土。",
    "I need to call my dentist tomorrow.",
    "My plan is to move to Berlin.",
    "My task is to fix the CI.",
    "My build is failing.",
    "My laptop is very slow today.",
    "My name was Pip.",
    "Your name is Melo.",
    "My wifi password is hunter2-lighthouse.",
    "I prefer tea.",
    "Remember that I live in Utrecht.",
    "My weekly sync is on Friday, right?",
]


@pytest.mark.parametrize("typed", NOT_NOTES)
def test_questions_to_dos_hypotheticals_negations_and_opinions_are_not_notes(
    typed: str,
) -> None:
    assert records.statements(typed) == []


@pytest.mark.parametrize(
    ("typed", "note"),
    [
        ("By the way, my advisor is Dr. Okafor.", "Your advisor is Dr. Okafor."),
        ("Call me Ray.", "You go by Ray."),
        ("I'm called Ray.", "You are called Ray."),
        ("对了，我的导师是周教授。", "你的导师是周教授。"),
        (
            "I live in Utrecht now. Can you find a bakery near me?",
            "You live in Utrecht now.",
        ),
        ("Now I live in Utrecht.", "You live in Utrecht."),
        ("I live in Lisbon, can you find me a flat?", "You live in Lisbon."),
        ("我住在里斯本，帮我找房子。", "你住在里斯本。"),
    ],
)
def test_a_lasting_statement_in_a_message_becomes_one_note(
    typed: str, note: str
) -> None:
    (cand,) = records.statements(typed)
    assert (cand.text, cand.admitted_by, cand.topic) == (
        note,
        "user_statement",
        "about",
    )


async def test_a_hypothetical_turn_keeps_no_note_and_says_nothing_more(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    events = await Chat(memory, store, stub, tmp_path)(
        "If I lived in Lisbon, I'd surf."
    )
    # Nothing was kept; the reply's bare "Noted." is flagged in the last event.
    assert all(e["remembered"] == [] for e in events)
    assert events[-1]["unsaved_claim"] is True and await memory.records() == []


async def test_workers_and_task_runners_keep_no_statement(
    store: SessionStore, tmp_path: Path
) -> None:
    server = StubServer(scenarios={"answer": {"replies": [{"content": "OK."}]}})
    server.start()
    cfg = settings(keep=True)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        await runner_chat(
            memory, store, server, tmp_path / "w", "I live in Lisbon.", kind="worker"
        )
        task = await store.create_session(kind="task")
        await runner_chat(
            memory,
            store,
            server,
            tmp_path / "t",
            "I live in Lisbon. Find a flat.",
            session_id=task.id,
            index_only=True,
        )
        assert await memory.records() == []
    finally:
        await memory.aclose()
        server.close()


async def test_a_statement_the_request_kept_is_not_kept_again(
    memory: ChatMemory, store: SessionStore
) -> None:
    typed = "Remember that I live in Utrecht. I live in Utrecht."
    turn = await store.begin_turn(
        None, typed, holder="test:1:statements", title=typed, new_session=NewSession()
    )
    found = await memory.before_turn(store, turn, typed)
    assert [n.text for n in found.remembered] == ["You live in Utrecht."]
    assert await memory.keep_statements(turn) == []
    assert len(await memory.records()) == 1


async def test_outside_its_own_turn_nothing_is_kept(
    memory: ChatMemory, store: SessionStore
) -> None:
    typed = "I live in Lisbon."
    one = await store.begin_turn(
        None, typed, holder="test:1:statements", title=typed, new_session=NewSession()
    )
    other = await store.begin_turn(
        None, typed, holder="test:1:statements", title=typed, new_session=NewSession()
    )
    assert await memory.keep_statements(one) == []  # no turn under way
    await memory.before_turn(store, other, typed)
    assert await memory.keep_statements(one) == []  # another turn's
    (kept,) = await memory.keep_statements(other)
    assert kept.text == "You live in Lisbon." and len(await memory.records()) == 1


def _turn(key: str, session: str, at: float, said: tuple[tuple[str, str], ...]) -> Item:
    return Item(
        key, f"User: {said[0][1]}", f"chat {session}", session, 1, at, said=said
    )


def test_newer_turns_by_subject_and_by_one_short_value_with_vectors() -> None:
    lisbon = _turn("d:1", "a", 1.0, (("home|", "i live in lisbon"),))
    utrecht = _turn("d:2", "b", 2.0, (("home|", "i live in utrecht"),))
    again = _turn("d:3", "c", 0.5, (("home|", "i live in utrecht"),))
    other = _turn("d:4", "d", 3.0, (("name|", "my name is juno"),))
    replaced = records.newer_turns([lisbon, utrecht, again, other])
    # The newest value wins; saying it again is not older; other subjects never.
    assert replaced == {("a", 1): utrecht}

    vector = np.ones(4, dtype=np.float32) / 2
    gym = (("", "i go to the gym on mondays"),)
    later = (("", "i go to the gym on fridays"),)
    laptop = (("", "i use a thinkpad x1 laptop at work"),)
    home = (("", "i use a framework 13 laptop at home"),)
    items = [
        _turn("d:5", "e", 1.0, gym),
        _turn("d:6", "f", 2.0, later),
        _turn("d:7", "g", 3.0, laptop),
        _turn("d:8", "h", 4.0, home),
    ]
    assert records.newer_turns(items) == {}  # without vectors, no subject: none
    for item in items:
        item.vector, item.embed_model = vector, "fake-embed"
    assert records.newer_turns(items) == {("e", 1): items[1]}


def test_newer_first_marks_a_turn_a_newer_turn_replaced_and_brings_it_in() -> None:
    old = _turn("d:1", "a", 1.0, (("home|", "i live in lisbon"),))
    new = _turn("d:2", "b", 2.0, (("home|", "i live in utrecht"),))
    ordered = newer_first([old], {("a", 1): new})
    assert [i.key for i in ordered] == ["d:2", "d:1"]
    assert ordered[-1].older == OLDER_TURN
    assert ordered[-1].prompt.endswith(f"(chat a) ({OLDER_TURN})")
    # The chat under way has the newer turn already: it is not brought in.
    alone = newer_first([old], {("a", 1): new}, exclude_session="b")
    assert [(i.key, i.older) for i in alone] == [("d:1", OLDER_TURN)]
    # A newer note that replaced the newer turn wins.
    note = Item("r:9", "You live in Berlin.", "chat c", "c", 1, 3.0)
    chain = newer_first([old], {("a", 1): new, ("b", 1): note})
    assert [(i.key, i.older) for i in chain] == [("r:9", ""), ("d:1", OLDER)]


async def test_by_default_code_keeps_no_statement(
    default_memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    memory = default_memory
    events = await Chat(memory, store, stub, tmp_path)("I live in Lisbon.")
    assert all(e["remembered"] == [] for e in events) and await memory.records() == []
    typed = "My name is Juno."
    turn = await store.begin_turn(
        None, typed, holder="test:1:statements", title=typed, new_session=NewSession()
    )
    await memory.before_turn(store, turn, typed)
    assert await memory.keep_statements(turn) == []
    assert await memory.records() == []


async def test_the_model_is_told_to_keep_a_lasting_statement_in_that_turn(
    default_memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    await Chat(default_memory, store, stub, tmp_path)("I live in Lisbon.")
    (request,) = stub.requests
    system = request["messages"][0]["content"]
    assert NOTES_TEXT in system
    rule = (
        "A plain statement of a lasting fact about the user (where they live, "
        "their name, a person, their schedule, a preference), such as “I live in "
        "Utrecht.”, is one: keep it with remember in that turn."
    )
    assert rule in system
    (tool,) = [
        t["function"] for t in request["tools"] if t["function"]["name"] == "remember"
    ]
    assert tool["description"] == REMEMBER.description
    for words in (
        "When the user states a lasting fact about themselves",
        "call it in that turn",
        "“I live in Utrecht.” → You live in Utrecht.",
        "never ask the user first",
    ):
        assert words in tool["description"]
