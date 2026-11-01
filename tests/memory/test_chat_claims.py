"""A reply that says it noted a fact keeps it, or is counted as unsaved.

After a completed chat run, when the answer claims a save (``Remembered: X``,
``Noted — X``, ``I've noted that X``, ``记下了：X``) and the turn kept no note,
code keeps X through ``remember``'s own path: the user's words, no secret, the
newer value supersedes the old one. When X is not the user's words, or the
answer states no fact (a fragment such as "gate 9 for Faro": no verb, no "you"),
nothing is kept and the turn's ``turn.memory`` event has
``unsaved_claim`` true. So does a reply that says it saved or noted and names
nothing ("Saved.", "Noted.", "text✅ 已记录"); "Got it!" claims no save. A turn
that kept a note, a plain answer, workers, task runners and memory off do
nothing. The claims are the real ones a guard cell of the benchmark answered
with no note kept. End to end on the stub model, with words only and with a fake
embedding server.
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
    "statement up front of whether melo is sure or only fairly confident."
)
# The same claim as the model first wrote it, "I'm" for the assistant beside "you"
# for the user: refused, as the remember tool refuses it.
PREFERENCE_CLAIM_BOTH = PREFERENCE_CLAIM.replace("melo is", "I'm")
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
    # A fact about another person is a note too (the benchmark's third-party
    # claims, kept by neither the model nor code before).
    (
        "Prof. Duarte's office is in the Atrium Wing.",
        "I've noted that Prof. Duarte's office is in the Atrium Wing.",
        "Prof. Duarte's office is in the Atrium Wing.",
    ),
    (
        "Dr. Tanaka's phone number is 555-0230.",
        "I've noted that Dr. Tanaka's phone number is 555-0230.",
        "Dr. Tanaka's phone number is 555-0230.",
    ),
    (PREFERENCE, f"Remembered: {PREFERENCE_CLAIM}", f"W{PREFERENCE_CLAIM[1:]}"),
]
# A claim whose X is not the user's words (a changed value, a new word that is a
# value, a paraphrase that adds a content word, a secret) or is not written to
# the user (a fragment, the real M2 reply).
REFUSED = [
    ("My name is Juno.", "Remembered: Your name is Zed."),
    ("My parking spot is C14.", "Noted — your parking spot is C15."),
    (
        "My flight to Porto is at gate 22.",
        "Noted — your flight to Porto departs from gate 22.",
    ),
    ("My password is hunter2.", "Noted — your password is hunter2."),
    (PREFERENCE, f"Remembered: {PREFERENCE_CLAIM_BOTH}"),
    (
        "My flight to Faro is at gate 9.",
        "Noted — gate 9 for Faro. I've also kept your earlier note about the "
        "Porto flight (gate 22), so both are on record.",
    ),
]


# A reply that says it saved or noted and names nothing: the bench's real one
# first, then the ways a model says it.
BARE = [
    "✅ 已记录",
    "Got it — noted…",
    "Noted.",
    "Noted, thanks!",
    "Saved to memory.",
    "好的，已经记住了！",
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
        **{f"b{n}": {"replies": [{"content": b}]} for n, b in enumerate(BARE)},
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


async def test_an_answer_that_claims_nothing_does_nothing(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    events = memory_events(
        await runner_chat(
            memory, store, at(stub, "plain"), tmp_path / "w", "My name is Juno."
        )
    )
    assert await memory.records() == []
    assert [e["unsaved_claim"] for e in events] == [False]


@pytest.mark.parametrize("n", range(len(BARE)))
async def test_a_reply_that_says_saved_and_names_nothing_is_counted(
    memory: ChatMemory, store: SessionStore, stub: StubServer, tmp_path: Path, n: int
) -> None:
    events = memory_events(
        await runner_chat(
            memory, store, at(stub, f"b{n}"), tmp_path / "w", "My name is Juno."
        )
    )
    assert await memory.records() == []
    assert events[0]["unsaved_claim"] is False
    assert events[-1]["remembered"] == [] and events[-1]["unsaved_claim"] is True


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
        (
            "Noted — you fly from gate 9 to Faro. Safe travels!",
            "You fly from gate 9 to Faro.",
        ),
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
        ("记住了：您住在乌得勒支。", "您住在乌得勒支。"),
        ("Noted — you pay 3.5 euros. Ok.", "You pay 3.5 euros."),
        ("Noted — it is YOURS.", "It is YOURS."),
        # A statement about someone or something else is a fact to keep (remember's
        # own checks decide whether it is the user's words).
        (
            "Noted — the group meeting is on Thursdays.",
            "The group meeting is on Thursdays.",
        ),
        ("记下了：导师是周教授。", "导师是周教授。"),
        # A title's "." ends no claim.
        (
            "I've noted that Mrs. Caldwell's phone number is 555-0318.",
            "Mrs. Caldwell's phone number is 555-0318.",
        ),
        (
            "Remembered: Dr. J. Smith works at the clinic. Anything else?",
            "Dr. J. Smith works at the clinic.",
        ),
        # No statement (no verb, no possessive, not to the user), with 3 content
        # words or more: nothing usable.
        ("Noted — gate 9 for Faro. I've also kept the Porto note.", ""),
        ("Remembered: Gate 9 for Faro.", ""),
        ("Noted — it costs 3.5 euros. Ok.", ""),
        ("Noted — I'll use plain words.", ""),
        # A claim after a short interjection in the same sentence counts.
        (
            "Got it — I've noted that Dr. Kowalski's office is on Pine Street.",
            "Dr. Kowalski's office is on Pine Street.",
        ),
        (
            "OK, I'll remember that you prefer short answers.",
            "You prefer short answers.",
        ),
        ("好的，我已记住：你的导师是周教授。", "你的导师是周教授。"),
        # The first claim counts.
        ("Noted — you like tea. Remembered: you like jam.", "You like tea."),
    ],
)
def test_claimed_reads_the_fact_a_reply_says_it_noted(answer: str, fact: str) -> None:
    assert records.claimed(answer) == fact


@pytest.mark.parametrize(
    "answer",
    [
        "Noted — .",
        "Got it — thanks!",
        "As noted before, Prof. Duarte's office is in the Atrium Wing.",
        "已记住：好的。",
        "Noted — yourself is fine.",
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


# Each of these says it saved or noted and names no fact: ``""``, a claim that
# names nothing usable, not ``None``.
BARE_CLAIMS = [
    # Chinese: with or without a check mark or an end mark.
    "text✅ 已记录",
    "已记录",
    "✅ 已记录！",
    "✔️已记录。",
    "好的，已记录。",
    "记录了",
    "记录下来了。",
    "已保存",
    "保存了！",
    "已记住。",
    "记住了！",
    "记下了。",
    "已记下。",
    "已经记住",
    "我已经记住了。",
    "帮你记下了。",
    "已保存到记忆中。",
    "已保存，请查看。",
    # English.
    "Saved.",
    "Saved",
    "saved!",
    "Saved to memory",
    "Saved it to your notes.",
    "I've saved that.",
    "I have stored this.",
    "I’ve recorded it!",
    "I'll remember that.",
    "I will remember it.",
    "Got it, saved",
    "Sure! Saved. Anything else?",
    # A claim whose object is only a pronoun or a generic noun names nothing.
    "我已经记录了你的偏好。",
    "记住了你的偏好！",
    "已记录你的偏好",
    "已记录：你的偏好。",
    "I've saved your preference.",
    "I've noted your preferences.",
    "Saved that info.",
    "I'll remember your preference.",
    "Remembered: Your preferences.",
    "Noted — your preference.",
    # A bare "noted" form claims a save too.
    "Noted.",
    "Noted",
    "Noted!",
    "Got it — noted.",
    "Got it, noted…",
    "Noted, thanks!",
    "Noted — thanks a lot.",
    "Okay, noted. Anything else?",
]


@pytest.mark.parametrize("answer", BARE_CLAIMS)
def test_claimed_is_empty_for_a_save_claim_that_names_nothing(answer: str) -> None:
    assert records.claimed(answer) == ""


@pytest.mark.parametrize(
    "answer",
    [
        "Got it!",
        "Got it.",
        "Got it — thanks!",
        "Thanks!",
        # A subject before the verb, or something after it: not a claim of its own.
        "The file was saved.",
        "文件已保存。",
        "我记录了你的问题，下面是回答。",
        "I saved the file to disk.",
        "I saved your file to disk.",
        "我记录了你的问题并回答",
        "I've saved the changes to report.md.",
        "As noted, saved files stay on disk.",
        "I'll remember to check.",
        "已记录你的问题并回答",
    ],
)
def test_claimed_is_none_for_text_that_only_looks_like_a_save_claim(
    answer: str,
) -> None:
    assert records.claimed(answer) is None


@pytest.mark.parametrize(
    ("answer", "fact"),
    [
        ("已记录：你的储物柜是58号。", "你的储物柜是58号。"),
        ("好的，已保存：你的导师是周教授！", "你的导师是周教授。"),
        ("已记录，你的储物柜是58号。", "你的储物柜是58号。"),
        ("记录下来了：你住在乌得勒支。", "你住在乌得勒支。"),
        ("帮你记下了：你的航班在22号登机口。", "你的航班在22号登机口。"),
        # A bare claim before the fact does not hide it.
        ("Saved. Noted — you like tea.", "You like tea."),
    ],
)
def test_a_claim_with_a_fact_after_it_still_returns_the_fact(
    answer: str, fact: str
) -> None:
    assert records.claimed(answer) == fact


@pytest.mark.parametrize("answer", BARE_CLAIMS)
async def test_a_bare_save_claim_with_nothing_kept_is_flagged(
    memory: ChatMemory, store: SessionStore, answer: str
) -> None:
    typed = "My name is Juno."
    turn = await store.begin_turn(
        None, typed, holder="test:1:claims", title=typed, new_session=NewSession()
    )
    await memory.before_turn(store, turn, typed)
    assert await memory.keep_claims(turn, answer) == []
    chat = CHAT_TURN.get()
    assert chat is not None and chat.unsaved_claim is True
    event = chat.event()
    assert event is not None and event["unsaved_claim"] is True
    assert event["remembered"] == [] and await memory.records() == []


async def test_a_claimed_fact_in_chinese_is_kept_for_the_first_person_typed(
    memory: ChatMemory, store: SessionStore
) -> None:
    typed = "我的储物柜是58号。"
    turn = await store.begin_turn(
        None, typed, holder="test:1:claims", title=typed, new_session=NewSession()
    )
    await memory.before_turn(store, turn, typed)
    (kept,) = await memory.keep_claims(turn, "已记录：你的储物柜是58号。")
    assert kept.text == "你的储物柜是58号。"
    chat = CHAT_TURN.get()
    assert chat is not None and chat.unsaved_claim is False
    (entry,) = await memory.records()
    assert entry.admitted_by == "user_statement" and entry.provenance["quote"] == typed


async def test_a_claim_of_a_note_that_says_i_and_you_is_flagged(
    memory: ChatMemory, store: SessionStore
) -> None:
    turn = await store.begin_turn(
        None, PREFERENCE, holder="test:1:claims", title="p", new_session=NewSession()
    )
    await memory.before_turn(store, turn, PREFERENCE)
    answer = f"Remembered: {PREFERENCE_CLAIM_BOTH}"
    assert records.claimed(answer) == f"W{PREFERENCE_CLAIM_BOTH[1:]}"
    assert await memory.keep_claims(turn, answer) == []
    chat = CHAT_TURN.get()
    assert chat is not None and chat.unsaved_claim is True
    assert await memory.records() == []


# The seven third-party facts a benchmark reply claimed to have noted, with no
# note kept: (what the user typed, the reply, the note code keeps).
THIRD_PARTY = [
    (
        "Prof. Duarte's office is in the Atrium Wing.",
        "I've noted that Prof. Duarte's office is in the Atrium Wing.",
        "Prof. Duarte's office is in the Atrium Wing.",
    ),
    (
        "Mrs. Caldwell's phone number is 555-0318.",
        "I've noted that Mrs. Caldwell's phone number is 555-0318.",
        "Mrs. Caldwell's phone number is 555-0318.",
    ),
    (
        "Dr. Kowalski's office is on Pine Street.",
        "I remember that Dr. Anna Kowalski is your dentist. I've noted that her "
        "office is on Pine Street.",
        "Her office is on Pine Street.",
    ),
    (
        "Gerald's phone number is 555-0147.",
        "I've noted that Gerald's phone number is 555-0147.",
        "Gerald's phone number is 555-0147.",
    ),
    (
        "Felix's birthday is on 2 August.",
        "I've noted that Felix's birthday is on 2 August.",
        "Felix's birthday is on 2 August.",
    ),
    (
        "Dr. Tanaka's phone number is 555-0230.",
        "I've noted that Dr. Tanaka's phone number is 555-0230.",
        "Dr. Tanaka's phone number is 555-0230.",
    ),
    (
        "Dr. Ortiz's office is room 512.",
        "I've noted that Dr. Ortiz's office is in room 512.",
        "Dr. Ortiz's office is in room 512.",
    ),
]


@pytest.mark.parametrize(("typed", "answer", "fact"), THIRD_PARTY)
async def test_a_fact_about_another_person_is_claimed_and_kept(
    memory: ChatMemory, store: SessionStore, typed: str, answer: str, fact: str
) -> None:
    assert records.claimed(answer) == fact
    turn = await store.begin_turn(
        None, typed, holder="test:1:claims", title=typed, new_session=NewSession()
    )
    await memory.before_turn(store, turn, typed)
    (kept,) = await memory.keep_claims(turn, answer)
    assert kept.text == fact
    (entry,) = await memory.records()
    assert entry.admitted_by == "user_statement" and entry.provenance["quote"] == typed
