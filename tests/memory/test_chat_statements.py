"""The newer turn wins even without a note; lasting statements and the model.

Of two chat turns that state one subject with different values, the older
reads ``(older; replaced by a newer message)`` and comes after the newer one,
which recall brings in when only the older matched; this writes no memory. Only
the model keeps a note of a plain statement, with ``remember``, and its
instructions and the tool say to keep a lasting fact in that turn; code keeps none
after the turn. :func:`records.statements`, the rule that finds a plain
first-person statement of a lasting fact that names a subject ("I live in …", "my
weekly sync is on …", "call me …", "我的…是…"), is pinned here: a question, a
to-do, a hypothetical, a negation, the past, someone else's opinion or a secret is
never one. End to end on the stub model, whose chats never call ``remember``
here, with words only and with a fake embedding server.
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
from localmelo.tests.memory.test_chat_newer import PAIRS, Chat

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


def settings() -> Config:
    """The settings of a chat without an embedding backend."""
    return Config(chat_backend="vllm", embedding_backend="none")


@contextlib.asynccontextmanager
async def chat_memory(kind: str, tmp_path: Path) -> AsyncIterator[ChatMemory]:
    """Memory across chats with words only, or with a fake embedding server."""
    server = EmbeddingServer().start() if kind == "vectors" else None
    cfg = settings()
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
    async with chat_memory(request.param, tmp_path) as memory:
        yield memory


def lines_ok(lines: list[str], old: str, new: str) -> None:
    """The newer value first, and the older one on no line (it is history)."""
    assert lines and new in lines[0]
    for line in lines:
        assert old not in line


@pytest.mark.parametrize(
    ("old_typed", "_", "new_typed", "__", "asks", "old", "new"), PAIRS
)
async def test_without_notes_the_newer_turn_wins(
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
    await chat(old_typed)
    await chat(new_typed)
    # The model kept no note and code keeps none: two chat turns.
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


async def test_code_keeps_no_statement_of_a_plain_message(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    events = await Chat(memory, store, stub, tmp_path)("I live in Lisbon.")
    assert all(e["remembered"] == [] for e in events) and await memory.records() == []
    typed = "My name is Juno."
    turn = await store.begin_turn(
        None, typed, holder="test:1:statements", title=typed, new_session=NewSession()
    )
    await memory.before_turn(store, turn, typed)
    assert await memory.records() == []


async def test_the_model_is_told_to_keep_a_lasting_statement_in_that_turn(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    await Chat(memory, store, stub, tmp_path)("I live in Lisbon.")
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
