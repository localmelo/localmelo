"""When a fact is updated, the newer value wins.

A note about the same subject as an older one replaces it: the old note is kept
but never recalled, and an undo of the new note brings it back. A turn whose
fact was replaced comes back after the newer note and reads as older, and the
chat is told that the newest of disagreeing notes is true. End to end on the
stub model: chat A is told the old value, chat B the new one, chat C asks; with
words only and with a fake embedding server. Look-alikes that are not the same
subject are both kept.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.recall import BLOCK_HEADER, OLDER, Item, newer_first
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.identity import NOTES_TEXT, PINNED_TEXT
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore, StoredEvent
from localmelo.tests.fixtures.stub_llm import StubServer
from localmelo.tests.memory.chat_memory_helpers import EmbeddingServer, FakeEmbedding

# What chat A is told, the model's note, what chat B is told, its note, the
# question chat C asks, and the old and the new value (the benchmark's pairs).
PAIRS = [
    (
        "I live in Lisbon.",
        "You live in Lisbon.",
        "I live in Utrecht.",
        "The user lives in Utrecht.",
        "Where do I live?",
        "Lisbon",
        "Utrecht",
    ),
    (
        "My name is Pip.",
        "Your name is Pip.",
        "My name is Juno.",
        "The user's name is Juno.",
        "What is my name?",
        "Pip",
        "Juno",
    ),
    (
        "我在云杉科技工作。",
        "你在云杉科技工作。",
        "我在北辰数据工作。",
        "用户在北辰数据工作。",
        "我在哪里工作？",
        "云杉",
        "北辰",
    ),
    (
        "My weekly sync is on Wednesday.",
        "Your weekly sync is on Wednesday.",
        "My weekly sync is on Friday.",
        "Your weekly sync is on Friday.",
        "Which day is my weekly sync?",
        "Wednesday",
        "Friday",
    ),
    (
        "我的健身教练是阿凯。",
        "你的健身教练是阿凯。",
        "我的健身教练是小宇。",
        "你的健身教练是小宇。",
        "我的健身教练是谁？",
        "阿凯",
        "小宇",
    ),
]
# A change said as a change: the model keeps it in that turn, and it replaces the
# old note. Not a plain statement, so code never keeps it (test_chat_statements).
CHANGE = (
    "I work at Spruce Labs.",
    "You work at Spruce Labs.",
    "I changed jobs; I work at Acme now.",
    "You work at Acme.",
    "Where do I work?",
    "Spruce",
    "Acme",
)
WITH_CHANGE = [*PAIRS, CHANGE]
# Two facts that look alike and are not one subject; ``None``: kept by request.
LOOKALIKES = [
    (
        "My reading group meets on Tuesdays.",
        "Your reading group meets on Tuesdays.",
        "My group meeting is on Thursdays.",
        "Your group meeting is on Thursdays.",
    ),
    (
        "I use a ThinkPad X1 laptop at work.",
        "You use a ThinkPad X1 laptop at work.",
        "I use a Framework 13 laptop at home.",
        "You use a Framework 13 laptop at home.",
    ),
    (
        "My flight to Porto is at gate 22.",
        "Your flight to Porto is at gate 22.",
        "My flight to Faro is at gate 9.",
        "Your flight to Faro is at gate 9.",
    ),
    (
        "My parking spot at work is C14.",
        "Your parking spot at work is C14.",
        "My parking spot at the airport is L7.",
        "Your parking spot at the airport is L7.",
    ),
    (
        "我的书法课是每周三晚上八点。",
        "你的书法课是每周三晚上八点。",
        "我的游泳课是每周六早上九点。",
        "你的游泳课是每周六早上九点。",
    ),
    ("记住，我妈妈的生日是三月九号。", None, "记住，我姐姐的生日是八月二十号。", None),
]


def remember(text: str) -> dict[str, Any]:
    return {"name": "remember", "arguments": {"text": text, "topic": "about"}}


@pytest.fixture()
def stub() -> Iterator[StubServer]:
    notes = [note for pair in WITH_CHANGE for note in (pair[1], pair[3])]
    notes += [note for pair in LOOKALIKES for note in (pair[1], pair[3]) if note]
    scenarios: dict[str, dict[str, Any]] = {
        "answer": {"replies": [{"content": "Noted."}]},
        # One scenario per note: the chat model keeps it, then answers.
        **{
            f"keep{n}": {
                "replies": [{"tool_calls": [remember(note)]}, {"content": "Noted."}]
            }
            for n, note in enumerate(notes)
        },
    }
    server = StubServer(scenarios=scenarios).start()
    server.notes = notes  # type: ignore[attr-defined]
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
    """Memory across chats with words only, or with a fake embedding server."""
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


class Chat:
    """Turns in new chats on the stub model, through a chat's runner."""

    def __init__(
        self, memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
    ) -> None:
        self.memory, self.store, self.stub = memory, store, stub
        self.workspace = tmp_path / "work"
        self.workspace.mkdir(exist_ok=True)

    async def __call__(
        self, text: str, note: str | None = None
    ) -> list[dict[str, Any]]:
        """*text* in a new chat whose model keeps *note*; the turn's memory events."""
        scenario = f"keep{self.stub.notes.index(note)}" if note else "answer"
        url = self.stub.base_url.replace("/v1", f"/s/{scenario}/v1")
        agent = await build_session_agent(
            cfg=None,
            base_url=url,
            model="stub-model",
            workspace=str(self.workspace),
            options=AgentOptions(),
            identify=True,
            memory=self.memory,
        )
        events: list[StoredEvent] = []
        runner = SessionRunner(
            agent,
            self.store,
            None,
            new_session=NewSession(kind="chat"),
            on_event=events.append,
            memory=self.memory,
        )
        try:
            result = await runner.run_turn(text, title=text)
            assert result.status == "completed"
            await self.memory.drain()
        finally:
            await runner.close()
            await agent.close()
        return [e.event["data"] for e in events if e.event["type"] == "turn.memory"]

    def request(self) -> tuple[str, list[str]]:
        """The last chat request's system prompt, and its memory block's lines."""
        messages = self.stub.requests[-1]["messages"]
        user = messages[-1]["content"]
        lines = (
            user.split(f"\n\n{BLOCK_HEADER}\n", 1)[1].split("\n")
            if (BLOCK_HEADER in user)
            else []
        )
        return messages[0]["content"], lines


@pytest.mark.parametrize(
    ("old_typed", "old_note", "new_typed", "new_note", "asks", "old", "new"),
    WITH_CHANGE,
)
async def test_the_newer_value_wins_and_undo_brings_the_old_back(
    memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    tmp_path: Path,
    old_typed: str,
    old_note: str,
    new_typed: str,
    new_note: str,
    asks: str,
    old: str,
    new: str,
) -> None:
    chat = Chat(memory, store, stub, tmp_path)
    await chat(old_typed, old_note)
    taught = await chat(new_typed, new_note)
    (kept,) = taught[-1]["remembered"]
    # The old note is replaced: kept in the file, never recalled again.
    assert [e.text for e in await memory.records()] == [new_note]
    (entry,) = await memory.records()
    assert entry.supersedes is not None and entry.provenance["replaces"]

    stub.requests.clear()
    asked = await chat(asks)
    system, lines = chat.request()
    assert lines and new in lines[0]
    # The old value is history: no line of the block gives it.
    assert not any(old in line for line in lines)
    assert not any(old in used["text"] for used in asked[-1]["used"])
    # The chat is told the newest wins; the pinned quotes skip the old note.
    assert NOTES_TEXT in system and "the newest one is true" in system
    pinned = system.split(PINNED_TEXT, 1)[1]
    stated = new_typed.rsplit("; ", 1)[-1]  # the sentence that states the value
    assert stated in pinned and old_typed not in pinned

    # Undo of the replacement: the old note is current again, nothing older.
    assert await memory.delete_record(kept["id"], undo=True)
    assert [e.text for e in await memory.records()] == [old_note]
    stub.requests.clear()
    await chat(asks)
    _, lines = chat.request()
    assert any(old in line for line in lines)
    assert not any(line.endswith(f"({OLDER})") for line in lines)


@pytest.mark.parametrize(
    ("old_typed", "old_note", "new_typed", "new_note", "asks", "old", "new"),
    [pair for pair in WITH_CHANGE if pair[4].isascii()],
)
async def test_asked_about_the_past_the_older_chat_comes_back_marked_older(
    memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    tmp_path: Path,
    old_typed: str,
    old_note: str,
    new_typed: str,
    new_note: str,
    asks: str,
    old: str,
    new: str,
) -> None:
    chat = Chat(memory, store, stub, tmp_path)
    await chat(old_typed, old_note)
    await chat(new_typed, new_note)

    stub.requests.clear()
    await chat(asks)
    _, lines = chat.request()
    assert lines and not any(old in line for line in lines)

    stub.requests.clear()
    asked = await chat(f"{asks} What was it before?")
    _, lines = chat.request()
    assert any(new in line for line in lines)
    assert any(old in line and line.endswith(f"({OLDER})") for line in lines)
    assert any(old in used["text"] for used in asked[-1]["used"])


@pytest.mark.parametrize(("first", "first_note", "second", "second_note"), LOOKALIKES)
async def test_look_alikes_of_another_subject_are_both_kept(
    memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    tmp_path: Path,
    first: str,
    first_note: str | None,
    second: str,
    second_note: str | None,
) -> None:
    chat = Chat(memory, store, stub, tmp_path)
    await chat(first, first_note)
    await chat(second, second_note)
    notes = await memory.records()
    assert len(notes) == 2
    assert all("replaces" not in e.provenance for e in notes)


@pytest.mark.parametrize(
    "phrasings",
    [
        ("I live in Lisbon.", "You live in Utrecht.", "The user lives in Utrecht."),
        ("Call me Pip.", "Your name is Juno.", "The user's name is Juno."),
        ("我在云杉科技工作。", "你在北辰数据工作。", "用户在北辰数据工作。"),
        (
            "My weekly sync is on Wednesday.",
            "Your weekly sync is now on Friday.",
            "The user's weekly sync is on Friday.",
        ),
        ("我的健身教练是阿凯。", "你的健身教练现在是小宇。", "用户的健身教练是小宇。"),
    ],
)
def test_one_subject_in_the_users_words_a_records_and_the_models(
    phrasings: tuple[str, ...],
) -> None:
    subjects = {records.subject(text) for text in phrasings}
    assert len(subjects) == 1 and "" not in subjects


@pytest.mark.parametrize(("first", "_", "second", "__"), LOOKALIKES)
def test_look_alikes_name_no_subject_or_another(
    first: str, _: Any, second: str, __: Any
) -> None:
    one, other = records.subject(first), records.subject(second)
    assert not one or one != other


@pytest.mark.parametrize(
    ("old", "new", "replaced"),
    [
        (
            "Your weekly sync is on Wednesday.",
            "Your weekly sync is every Friday.",
            True,
        ),
        ("You go by Ray.", "Your name is Juno.", True),
        ("Your sister is a doctor.", "Your sister is an engineer.", True),
        ("你的导师姓林。", "你的导师姓周。", True),
        # One subject, a value of another kind: a name and a look, a name and
        # a description, the way to work and a name, a flatmate and a city.
        ("Your cat is called Biscuit.", "Your cat is grey.", False),
        ("Your landlady is Gwen Pennywhistle.", "Your landlady is very strict.", False),
        ("You go by bike to work.", "Your name is Juno.", False),
        ("You live with your sister.", "You live in Utrecht.", False),
        ("你的猫叫团子。", "你的猫是橘色的。", False),
    ],
)
async def test_a_value_of_another_kind_does_not_replace(
    tmp_path: Path, old: str, new: str, replaced: bool
) -> None:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        for note in (old, new):
            typed = note.replace("Your", "My").replace("You", "I").replace("你", "我")
            turn = await store.begin_turn(
                None,
                typed,
                holder="test:1:newer",
                title=typed,
                new_session=NewSession(),
            )
            await memory.before_turn(store, turn, typed)
            assert (await memory.remember(note, "about")).startswith("Remembered:")
        assert len(await memory.records()) == (1 if replaced else 2)
    finally:
        await memory.aclose()
        await store.aclose()


def test_value_apart_wants_one_short_value_in_a_shared_frame() -> None:
    assert records.value_apart(
        "You go to the gym on Mondays.", "You go to the gym on Fridays."
    )
    assert not records.value_apart(
        "You use a ThinkPad X1 laptop at work.",
        "You use a Framework 13 laptop at home.",
    )
    assert not records.value_apart("You prefer tea.", "You prefer coffee.")
    assert not records.value_apart("你习惯在湖边公园跑步。", "你习惯早餐吃豆浆配油条。")


@pytest.mark.parametrize(("vectors", "kept"), [(True, 1), (False, 2)])
async def test_with_vectors_a_sentence_apart_only_in_its_value_replaces(
    tmp_path: Path, vectors: bool, kept: int
) -> None:
    """Another topic, so no shared words decide: only the embedding model can."""
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    embedding = FakeEmbedding()
    if vectors:
        memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    try:
        for typed, topic in (
            ("I go to the gym on Mondays.", "about"),
            ("I go to the gym on Fridays.", "projects"),
        ):
            turn = await store.begin_turn(
                None,
                typed,
                holder="test:1:newer",
                title=typed,
                new_session=NewSession(),
            )
            await memory.before_turn(store, turn, typed)
            answer = await memory.remember(records.second_person(typed), topic)
            assert answer.startswith("Remembered:")
        assert len(await memory.records()) == kept
        assert (await memory.records())[0].text == "You go to the gym on Fridays."
    finally:
        await memory.aclose()
        await store.aclose()


async def test_a_note_reworded_in_its_own_turn_marks_no_turn_older(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        typed = "Remember that Sam has the lab keys."
        turn = await store.begin_turn(
            None, typed, holder="test:1:newer", title=typed, new_session=NewSession()
        )
        await memory.before_turn(store, turn, typed)
        await memory.remember("Sam has the keys to the lab.", "people")
        (entry,) = await memory.records()
        assert entry.supersedes is not None and "replaces" not in entry.provenance
    finally:
        await memory.aclose()
        await store.aclose()


def test_a_replaced_turn_brings_its_newer_note_in_ahead_of_it() -> None:
    note = Item("r:2", "You live in Utrecht.", "chat B, Oct 30", "b", 1, 2.0)
    note.replaces = (("a", 1),)
    turn = Item("d:1", "User: I live in Lisbon.", "chat A, Oct 30", "a", 1, 1.0)
    other = Item("d:3", "User: I like tea.", "chat D, Oct 30", "d", 1, 3.0)
    ordered = newer_first([turn, other], {("a", 1): note})
    assert [i.key for i in ordered] == ["r:2", "d:3", "d:1"]
    assert ordered[-1].older and not turn.older
    assert ordered[-1].prompt.endswith(f"(chat A, Oct 30) ({OLDER})")
    assert ordered[-1].source == f"chat A, Oct 30 · {OLDER}"
