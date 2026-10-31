"""A reply that says it noted a fact keeps it, or is counted as unsaved.

After a completed chat run, when the answer claims a save (``Remembered: X``,
``Noted — X``, ``I've noted that X``, ``记下了：X``) and the turn kept no note,
code keeps X through ``remember``'s own path: the user's words, no secret, the
newer value supersedes the old one. When X is not the user's words, or the
answer names nothing usable, nothing is kept and the turn's ``turn.memory``
event has ``unsaved_claim`` true. A turn that kept a note, a plain answer,
workers, task runners and memory off do nothing. The claims are the real ones
a guard cell of the benchmark answered with no note kept. End to end on the
stub model, with words only and with a fake embedding server.
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

PREFERENCE = (
    "I prefer that when you explain a technical topic to me you choose plain "
    "everyday words and one small example done step by step before any formal "
    "definition, and that you tell me clearly at the start of the reply whether "
    "you are sure of the answer or only fairly confident about it."
)
PREFERENCE_CLAIM = (
    "when explaining a technical topic, you want plain everyday words with one "
    "small step-by-step example before any formal definition, and a clear "
    "statement up front of whether I'm sure or only fairly confident."
)
# (typed, the reply that claims a save, the note it keeps): the benchmark's real
# claims, then an English one and a Chinese one.
CLAIMS = [
    ("My name is Juno.", "Remembered: Your name is Juno.", "Your name is Juno."),
    (
        "My flight to Porto is at gate 22.",
        "Noted — your flight to Porto is at gate 22. Safe travels!",
        "Your flight to Porto is at gate 22.",
    ),
    (
        "My flight to Faro is at gate 9.",
        "Noted — your flight to Faro is at gate 9. Have a good trip!",
        "Your flight to Faro is at gate 9.",
    ),
    (
        "My parking spot at the airport is L7.",
        "Remembered: your airport parking spot is L7 (your work spot C14 stays as it was).",
        "Your airport parking spot is L7.",
    ),
    (
        "I live in Utrecht.",
        "I've noted that you live in Utrecht.",
        "You live in Utrecht.",
    ),
    ("我住在乌得勒支。", "记下了：你住在乌得勒支。", "你住在乌得勒支。"),
    (PREFERENCE, f"Remembered: {PREFERENCE_CLAIM}", f"W{PREFERENCE_CLAIM[1:]}"),
    # A fragment is kept as it is: its words and values are the user's.
    (
        "My flight to Faro is at gate 9.",
        "Noted — gate 9 for Faro. I've also kept your earlier note about the "
        "Porto flight (gate 22), so both are on record.",
        "Gate 9 for Faro.",
    ),
]
# A claim whose X is not the user's words (a changed value, a new word that is a
# value, a paraphrase that adds a content word, a secret, nothing at all).
REFUSED = [
    ("My name is Juno.", "Remembered: Your name is Zed."),
    ("My parking spot is C14.", "Noted — your parking spot is C15."),
    (
        "My flight to Porto is at gate 22.",
        "Noted — your flight to Porto departs from gate 22.",
    ),
    ("My password is hunter2.", "Noted — your password is hunter2."),
    ("My name is Juno.", "Noted — ."),
]


def at(stub: StubServer, scenario: str) -> Any:
    """The stub, answering in *scenario* (a runner only needs its base URL)."""
    return SimpleNamespace(base_url=stub.base_url.replace("/v1", f"/s/{scenario}/v1"))


@pytest.fixture()
def stub() -> Iterator[StubServer]:
    remember = {
        "name": "remember",
        "arguments": {"text": "Your name is Juno.", "topic": "about"},
    }
    scenarios: dict[str, dict[str, Any]] = {
        "plain": {"replies": [{"content": "Hello Juno, nice to meet you."}]},
        "noted": {"replies": [{"content": "Noted."}]},
        # The model keeps the note with the tool, then claims one more fact.
        "kept": {
            "replies": [
                {"tool_calls": [remember]},
                {"content": "Noted — your name is Juno, and you like tea."},
            ]
        },
        "request": {"replies": [{"content": "Noted — you live in Utrecht."}]},
        **{f"c{n}": {"replies": [{"content": c[1]}]} for n, c in enumerate(CLAIMS)},
        **{f"r{n}": {"replies": [{"content": r[1]}]} for n, r in enumerate(REFUSED)},
        "pip": {"replies": [{"content": "Remembered: Your name is Juno."}]},
    }
    server = StubServer(scenarios=scenarios).start()
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


@pytest.mark.parametrize(("typed", "reply", "note"), CLAIMS)
async def test_a_claimed_save_becomes_a_note_with_undo(
    memory: ChatMemory,
    store: SessionStore,
    stub: StubServer,
    tmp_path: Path,
    typed: str,
    reply: str,
    note: str,
) -> None:
    n = [c[1] for c in CLAIMS].index(reply)
    events = memory_events(
        await runner_chat(memory, store, at(stub, f"c{n}"), tmp_path / "w", typed)
    )
    # Before the run's end nothing was kept; the last event holds the note.
    assert events[0]["remembered"] == [] and events[0]["unsaved_claim"] is False
    (kept,) = events[-1]["remembered"]
    assert kept["text"] == note and events[-1]["unsaved_claim"] is False
    (entry,) = await memory.records()
    assert entry.text == note and entry.admitted_by == "user_statement"
    assert entry.provenance["quote"] == typed


async def test_the_claimed_name_replaces_the_old_one(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    await runner_chat(
        memory,
        store,
        at(stub, "noted"),
        tmp_path / "a",
        "Remember that my name is Pip.",
    )
    (old,) = await memory.records()
    assert old.text == "Your name is Pip."
    events = memory_events(
        await runner_chat(
            memory, store, at(stub, "pip"), tmp_path / "b", "My name is Juno."
        )
    )
    (kept,) = events[-1]["remembered"]
    (entry,) = await memory.records()  # Pip is history, never recalled
    assert entry.text == kept["text"] == "Your name is Juno."
    assert entry.provenance["replaces"] and events[-1]["unsaved_claim"] is False


@pytest.mark.parametrize(("typed", "reply"), REFUSED)
async def test_a_claim_that_is_not_the_users_words_keeps_nothing_and_is_counted(
    memory: ChatMemory,
    store: SessionStore,
    stub: StubServer,
    tmp_path: Path,
    typed: str,
    reply: str,
) -> None:
    n = [r[1] for r in REFUSED].index(reply)
    events = memory_events(
        await runner_chat(memory, store, at(stub, f"r{n}"), tmp_path / "w", typed)
    )
    assert await memory.records() == []
    assert events[-1]["remembered"] == [] and events[-1]["unsaved_claim"] is True
    # A secret is also reported as refused, as when the tool is asked to keep it.
    assert bool(events[-1]["refused"]) == ("password" in typed)


async def test_a_turn_that_kept_a_note_is_left_alone(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    # The model's own note stands; the reply's extra claim is not chased.
    events = memory_events(
        await runner_chat(
            memory, store, at(stub, "kept"), tmp_path / "w", "My name is Juno."
        )
    )
    (entry,) = await memory.records()
    assert entry.text == "Your name is Juno."
    assert [n["text"] for n in events[-1]["remembered"]] == [entry.text]
    assert events[-1]["unsaved_claim"] is False
    # So does a note the user's own request kept.
    events = memory_events(
        await runner_chat(
            memory,
            store,
            at(stub, "request"),
            tmp_path / "x",
            "Remember that I live in Utrecht.",
        )
    )
    assert [n["text"] for n in events[-1]["remembered"]] == ["You live in Utrecht."]
    assert len(await memory.records()) == 2 and events[-1]["unsaved_claim"] is False


@pytest.mark.parametrize("scenario", ["plain", "noted"])
async def test_an_answer_that_claims_nothing_does_nothing(
    memory: ChatMemory,
    store: SessionStore,
    stub: StubServer,
    tmp_path: Path,
    scenario: str,
) -> None:
    events = memory_events(
        await runner_chat(
            memory, store, at(stub, scenario), tmp_path / "w", "My name is Juno."
        )
    )
    assert await memory.records() == []
    assert [e["unsaved_claim"] for e in events] == [False]


async def test_workers_task_runners_and_memory_off_do_nothing(
    store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    on = ChatMemory(lambda: None, directory=str(tmp_path / "on"))
    off_cfg = Config(chat_backend="vllm", embedding_backend="none")
    off_cfg.memory.across_chats = False
    off = ChatMemory(lambda: off_cfg, directory=str(tmp_path / "off"))
    claim = at(stub, "c0")
    try:
        await runner_chat(
            on, store, claim, tmp_path / "w", "My name is Juno.", kind="worker"
        )
        task = await store.create_session(kind="task")
        await runner_chat(
            on,
            store,
            claim,
            tmp_path / "t",
            "My name is Juno.",
            session_id=task.id,
            index_only=True,
        )
        assert await on.records() == []
        events = await runner_chat(
            off, store, claim, tmp_path / "o", "My name is Juno."
        )
        assert memory_events(events) == [] and await off.records() == []
    finally:
        await on.aclose()
        await off.aclose()


async def test_only_the_turn_under_way_keeps_a_claim(
    memory: ChatMemory, store: SessionStore
) -> None:
    typed = "My name is Juno."
    one = await store.begin_turn(
        None, typed, holder="test:1:claims", title=typed, new_session=NewSession()
    )
    other = await store.begin_turn(
        None, typed, holder="test:1:claims", title=typed, new_session=NewSession()
    )
    claim = "Remembered: Your name is Juno."
    assert await memory.keep_claims(one, claim) == []  # no turn under way
    await memory.before_turn(store, other, typed)
    assert await memory.keep_claims(one, claim) == []  # another turn's
    turn = CHAT_TURN.get()
    assert turn is not None and not turn.unsaved_claim
    (kept,) = await memory.keep_claims(other, claim)
    assert kept.text == "Your name is Juno." and len(await memory.records()) == 1


@pytest.mark.parametrize(
    ("answer", "fact"),
    [
        ("Remembered: Your name is Juno.", "Your name is Juno."),
        ("Remembered: your airport spot is L7 (not C14).", "Your airport spot is L7."),
        ("Noted — gate 9 for Faro. Safe travels!", "Gate 9 for Faro."),
        ("Noted: you like tea. Anything else?", "You like tea."),
        ("Noted, you like tea!", "You like tea."),
        ("noted - you like tea", "You like tea."),
        ("**Noted** — you like tea.", "You like tea."),
        ("Sure. Remembered: you like tea. Next?", "You like tea."),
        ("I've noted that you live in Utrecht.", "You live in Utrecht."),
        ("I have noted you live in Utrecht.", "You live in Utrecht."),
        ("I'll remember that you prefer short answers.", "You prefer short answers."),
        ("Got it — you live in Utrecht, noted.", "You live in Utrecht."),
        ("Got it — you live in Utrecht.\nNoted!", "You live in Utrecht."),
        ("记下了：你住在乌得勒支。以后会用。", "你住在乌得勒支。"),
        ("已记住：你的导师是周教授！", "你的导师是周教授。"),
        ("记住了，你的导师是周教授。", "你的导师是周教授。"),
        ("Noted — it costs 3.5 euros. Ok.", "It costs 3.5 euros."),
        ("Noted — .", ""),
        # The first claim counts.
        ("Noted — you like tea. Remembered: you like jam.", "You like tea."),
    ],
)
def test_claimed_reads_the_fact_a_reply_says_it_noted(answer: str, fact: str) -> None:
    assert records.claimed(answer) == fact


@pytest.mark.parametrize(
    "answer",
    [
        "Noted.",
        "Noted",
        "Hello there.",
        "As noted, tea is fine.",
        "Got it — you live in Utrecht.",
        "Remembered:",
        "I noted nothing.",
        "I'll remember to be brief.",
        "",
    ],
)
def test_claimed_is_none_when_a_reply_claims_no_save(answer: str) -> None:
    assert records.claimed(answer) is None
