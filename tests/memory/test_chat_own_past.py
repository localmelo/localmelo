"""A chat remembers its own past, and a follow-up knows what it follows.

A chat's turns that its model's window no longer holds are recalled like
another chat's, as ``this chat``; the ones it still holds are not
(``window_start``: the turn of the window's first user message, by time stamp).
With ``[memory] context_query`` on, the message before the current one joins the
search: its words in a second BM25 list at half weight, its query vector added
at half weight. Off (the default) changes nothing.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np
import pytest

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.melo.schema import EMPTY_REPLY_NUDGE, Message
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, service
from localmelo.support.memory.index import Document
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    Item,
    in_view,
    shown_lines,
    this_chat,
)
from localmelo.support.memory.service import CHAT_TURN, window_start
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding, teach
from localmelo.tests.sessions.helpers import Clock

HOLDER = "test-host:1:own"
DENTIST = "my dentist appointment is on Nov 3"
ASKS = "when is my dentist appointment?"
FILLER = ("plan the garden", "buy tomato seeds", "paint the fence")


def today() -> str:
    now = time.localtime()
    return f"{time.strftime('%b', now)} {now.tm_mday}"


@pytest.fixture()
def clock() -> Clock:
    return Clock()


@pytest.fixture()
async def store(tmp_path: Path, clock: Clock) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway", clock=clock)
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def memory_in(
    tmp_path: Path, settings: Callable[[], Config | None] = lambda: None
) -> ChatMemory:
    return ChatMemory(settings, directory=str(tmp_path / "memory"))


async def said(
    store: SessionStore,
    session_id: str | None,
    text: str,
    *,
    memory: ChatMemory | None = None,
    answer: str = "Okay.",
) -> TurnInfo:
    """A finished turn of *text* (indexed when *memory* is given)."""
    turn = await store.begin_turn(
        session_id,
        text,
        holder=HOLDER,
        title=text if session_id is None else None,
        new_session=NewSession() if session_id is None else None,
    )
    done = await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer=answer,
    )
    if memory is not None:
        memory.after_turn(store, done, text, answer)
        await memory.drain()
    return done


async def running(store: SessionStore, session_id: str, text: str) -> TurnInfo:
    return await store.begin_turn(session_id, text, holder=HOLDER)


async def finish(store: SessionStore, turn: TurnInfo) -> TurnInfo:
    return await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="Okay.",
    )


def window_of(*turns: TurnInfo) -> list[Message]:
    """The messages the runner loads for finished turns: the turn's own times."""
    out: list[Message] = []
    for turn in turns:
        out.append(Message("user", turn.user_text, timestamp=turn.started_at))
        stop = turn.finished_at or turn.started_at
        out.append(Message("assistant", turn.answer, timestamp=stop))
    return out


async def chat_of(
    store: SessionStore, clock: Clock, texts: tuple[str, ...], memory: ChatMemory
) -> list[TurnInfo]:
    first = await said(store, None, texts[0], memory=memory)
    turns = [first]
    for text in texts[1:]:
        clock.advance(10)
        turns.append(await said(store, first.session_id, text, memory=memory))
    clock.advance(10)
    return turns


# ── Which turns the window holds ──


async def test_the_window_holds_from_the_turn_of_its_first_user_message(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path)
    turns = await chat_of(store, clock, (DENTIST, *FILLER), memory)
    now = await running(store, turns[0].session_id, ASKS)
    # Pairs of turns 3 and 4: the window starts at 3.
    assert await window_start(store, now, window_of(*turns[2:])) == 3
    # Every turn in the window: from the first turn, as before this change.
    assert await window_start(store, now, window_of(*turns)) == 1
    # Nothing in it: none of the earlier turns is shown, only this one is under way.
    assert await window_start(store, now, []) == now.turn
    await memory.aclose()


async def test_a_turn_whose_question_left_the_window_is_not_held(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path)
    turns = await chat_of(store, clock, (DENTIST, *FILLER), memory)
    now = await running(store, turns[0].session_id, ASKS)
    # Turn 2's answer is the first message left, and a system message before it
    # is no message of the user's: the window begins at turn 3's question.
    window = window_of(*turns[1:])[1:]
    assert await window_start(store, now, window) == 3
    system = [Message("system", "x", timestamp=turns[0].started_at), *window]
    assert await window_start(store, now, system) == 3
    await memory.aclose()


async def test_an_empty_reply_nudge_and_a_message_a_run_made_do_not_start_a_turn(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path)
    turns = await chat_of(store, clock, (DENTIST, *FILLER), memory)
    now = await running(store, turns[0].session_id, ASKS)
    # A nudge of turn 3 first: skipped, so the window starts with turn 4's question.
    nudge = Message("user", EMPTY_REPLY_NUDGE, timestamp=turns[2].started_at + 1)
    assert await window_start(store, now, [nudge, *window_of(turns[3])]) == 4
    # A live message is stamped after its turn began, within it: its own turn,
    # also when it is the task's message put back in front of an evicted window.
    query = Message("user", "paint the fence", timestamp=turns[2].started_at + 2)
    assert await window_start(store, now, [query, *window_of(turns[3])]) == 3
    await memory.aclose()


async def test_an_undone_turn_is_not_the_start_and_is_skipped(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path)
    turns = await chat_of(store, clock, (DENTIST, FILLER[0], FILLER[1]), memory)
    sid = turns[0].session_id
    assert (await store.undo_last_turn(sid)) is not None  # turn 3 is undone
    clock.advance(10)
    fourth = await said(store, sid, FILLER[2], memory=memory)
    now = await running(store, sid, ASKS)
    # The reloaded window holds turns 1, 2 and 4 (3 is undone).
    window = window_of(turns[0], turns[1], fourth)
    assert await window_start(store, now, window) == 1
    # A window that starts after the undone turn starts at turn 4, never at 3.
    assert await window_start(store, now, window_of(fourth)) == 4
    await memory.aclose()


async def test_a_stamp_older_than_every_turn_and_many_pages_of_turns(
    store: SessionStore,
    clock: Clock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "_TURN_PAGE", 2)
    memory = memory_in(tmp_path)
    turns = await chat_of(store, clock, (DENTIST, *FILLER, "mend the roof"), memory)
    now = await running(store, turns[0].session_id, ASKS)
    assert await window_start(store, now, window_of(*turns[1:])) == 2
    old = Message("user", "x", timestamp=turns[0].started_at - 100)
    assert await window_start(store, now, [old]) == 1
    await memory.aclose()


# ── Recall of the chat's own turns ──


async def test_this_chats_turns_that_left_the_window_are_recalled_and_the_rest_not(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path)
    turns = await chat_of(store, clock, (DENTIST, *FILLER), memory)
    now = await running(store, turns[0].session_id, ASKS)
    # Without window information the whole chat is left out, as before.
    blind = await memory.before_turn(store, now, ASKS)
    assert blind.used == () and blind.block == ""
    # The window holds turns 3 and 4: turn 1 is recalled as this chat's own.
    found = await memory.before_turn(store, now, ASKS, window=window_of(*turns[2:]))
    (item,) = found.used
    assert (item.session_id, item.turn) == (now.session_id, 1)
    assert item.where == f"this chat, {today()}"
    assert f"({item.where})" in found.block and DENTIST in found.block
    data = found.event()
    assert data is not None and data["used"][0]["source"] == f"this chat, {today()}"
    # The window holds turn 1 itself: nothing is recalled.
    again = await memory.before_turn(store, now, ASKS, window=window_of(*turns))
    assert again.used == () and again.block == ""
    await memory.aclose()


async def test_another_chat_is_recalled_as_before_and_a_note_of_this_chat_never_left_out(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path)
    other = await said(
        store, None, "I booked the dentist appointment for Nov 9", memory=memory
    )
    turns = await chat_of(store, clock, ("hello", *FILLER), memory)
    sid = turns[0].session_id
    now = await running(store, sid, ASKS)
    # The whole of this chat is in the window; the other chat still comes back.
    found = await memory.before_turn(store, now, ASKS, window=window_of(*turns))
    (item,) = found.used
    assert item.session_id == other.session_id
    assert item.where.startswith("chat “I booked") and "this chat" not in item.where
    await finish(store, now)
    clock.advance(10)
    # A note kept in this chat is recalled in it, as ever.
    request = "Remember that my dentist is Okafor"
    note = await running(store, sid, request)
    assert (await teach(memory, store, note, request)).remembered
    await finish(store, note)
    clock.advance(10)
    after = await running(store, sid, "who is my dentist?")
    found = await memory.before_turn(store, after, "who is my dentist?")
    assert any(item.is_record and "Okafor" in item.text for item in found.used)
    await memory.aclose()


async def test_search_memory_finds_this_chats_evicted_turns_but_not_the_shown_ones(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path)
    turns = await chat_of(store, clock, (DENTIST, *FILLER), memory)
    now = await running(store, turns[0].session_id, ASKS)
    search = memory.tools().search
    try:
        await memory.before_turn(store, now, "hi", window=window_of(*turns[2:]))
        reply = await search(query="dentist appointment")
        assert reply == f"1. {_prompt(DENTIST)} (this chat, {today()})"
        await memory.before_turn(store, now, "hi", window=window_of(*turns))
        assert "Nothing in melo's memory" in await search(query="dentist appointment")
        # No window known for the turn: nothing is left out, as before.
        await memory.before_turn(store, now, "hi")
        assert DENTIST in await search(query="dentist appointment")
        # Outside a chat's turn too.
        CHAT_TURN.set(None)
        assert DENTIST in await search(query="dentist appointment")
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


def _prompt(typed: str) -> str:
    return f"User: {typed} melo: Okay."


def test_in_view_this_chat_and_the_recall_lines_in_a_window() -> None:
    turn = Item("d:1", "User: a melo: b", "chat “a”, Oct 1", "s", 5, 1.0)
    note = Item("r:1", "Your name is Juno.", "chat “a”, Oct 1", "s", 5, 1.0)
    assert in_view(turn, "s", None) and in_view(turn, "s", 5) and in_view(turn, "s", 3)
    assert not in_view(turn, "s", 6) and not in_view(turn, "t", None)
    assert not in_view(note, "s", None)  # a note is never in view
    assert this_chat(turn, "s").where.startswith("this chat, ")
    assert this_chat(turn, "t") is turn and this_chat(note, "s") is note
    block_text = f"question\n\n{BLOCK_HEADER}\n- one (a)\n- two (b)"
    window = [
        Message("user", block_text),
        Message("assistant", "- three"),
        Message("user", "- four"),
    ]
    assert shown_lines(window) == {"- one (a)", "- two (b)"}


# ── A query that knows the conversation ──

QUESTION = "who is my advisor?"
FOLLOW = "and where is her office located?"


def cfg_with(context_query: bool) -> Config:
    cfg = Config()
    cfg.memory.context_query = context_query
    return cfg


def settings(context_query: bool) -> Callable[[], Config | None]:
    return lambda: cfg_with(context_query)


async def advisor_doc(store: SessionStore, memory: ChatMemory) -> None:
    """Another chat's turn that answers the question and its follow-up together,
    but not the follow-up alone: it has "office", not "located"."""
    other = await said(store, None, "my advisor")
    await memory.stores()[1].add(
        Document(
            session_id=other.session_id,
            turn=1,
            surface="gateway",
            kind="chat",
            title="Advisor",
            text="User: my advisor melo: Your advisor has an office in room 214.",
            created_at=time.time(),
        )
    )


async def follow_up(
    store: SessionStore, clock: Clock, *, gap: float = 60.0
) -> tuple[TurnInfo, TurnInfo]:
    """A chat that asked QUESTION and, *gap* seconds later, sends FOLLOW."""
    first = await said(store, None, QUESTION)
    clock.advance(gap)
    return first, await running(store, first.session_id, FOLLOW)


async def test_a_follow_up_finds_what_the_message_alone_misses_only_with_context_query(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    for setting, label in ((False, "off"), (True, "used")):
        memory = memory_in(tmp_path / label, settings(setting))
        await advisor_doc(store, memory)
        _, follow = await follow_up(store, clock)
        found = await memory.before_turn(store, follow, FOLLOW)
        assert found.recall is not None and found.recall.context == label
        data = found.event()
        assert data is not None and data["recall"]["context"] == label
        assert bool(found.used) is setting
        assert ("room 214" in found.block) is setting
        assert setting or found.block == ""
        await memory.aclose()


async def test_context_query_is_off_by_default_and_changes_nothing_when_off(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    assert Config().memory.context_query is False
    for name, memory in (
        ("none", memory_in(tmp_path / "a")),  # no settings at all
        ("table", memory_in(tmp_path / "b", settings(False))),
    ):
        await advisor_doc(store, memory)
        _, follow = await follow_up(store, clock)
        found = await memory.before_turn(store, follow, FOLLOW)
        assert found.block == "" and found.used == (), name
        assert found.recall is not None and found.recall.context == "off"
        await memory.aclose()


async def test_with_no_message_before_it_the_search_is_the_plain_one(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    blocks, contexts = [], []
    for setting in (False, True):
        memory = memory_in(tmp_path / str(setting), settings(setting))
        await advisor_doc(store, memory)
        new = await store.begin_turn(
            None, "advisor office", holder=HOLDER, new_session=NewSession()
        )
        found = await memory.before_turn(store, new, "advisor office")
        assert found.recall is not None
        blocks.append(found.block)
        contexts.append(found.recall.context)
        await finish(store, new)
        await memory.aclose()
    assert blocks[0] == blocks[1] != ""
    assert contexts == ["off", "none"]


async def test_a_message_older_than_thirty_minutes_is_a_different_moment(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path, settings(True))
    await advisor_doc(store, memory)
    _, late = await follow_up(store, clock, gap=31 * 60)
    found = await memory.before_turn(store, late, FOLLOW)
    assert found.used == () and found.recall is not None
    assert found.recall.context == "none"
    await memory.aclose()


async def test_the_message_before_is_the_latest_one_that_is_not_undone(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    memory = memory_in(tmp_path, settings(True))
    await advisor_doc(store, memory)
    first = await said(store, None, QUESTION)
    sid = first.session_id
    clock.advance(1)
    await said(store, sid, "what is for lunch?")
    undone = await store.undo_last_turn(sid)
    assert undone is not None and undone.turn == 2
    clock.advance(1)
    follow = await running(store, sid, FOLLOW)
    assert await memory._previous(store, follow) == (1, QUESTION)
    found = await memory.before_turn(store, follow, FOLLOW)
    assert found.recall is not None and found.recall.context == "used"
    assert "room 214" in found.block
    await finish(store, follow)
    # With the question undone too, there is no message to use.
    assert (await store.undo_last_turn(sid)) is not None
    assert (await store.undo_last_turn(sid)) is not None
    clock.advance(1)
    alone = await running(store, sid, FOLLOW)
    assert await memory._previous(store, alone) == (0, "")
    found = await memory.before_turn(store, alone, FOLLOW)
    assert found.used == () and found.recall is not None
    assert found.recall.context == "none"
    await memory.aclose()


class TableEmbedding(BaseEmbeddingProvider):
    """Explicit unit vectors by text; any other text points away from all of them."""

    model = "table"

    def __init__(self, table: dict[str, tuple[float, float, float]]) -> None:
        self.table = table
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [list(self.table.get(text, (0.0, 0.0, 1.0))) for text in texts]

    async def close(self) -> None:
        return None


async def test_a_follow_up_adds_half_the_previous_query_vector_with_no_extra_call(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    # The note is topic A. The question is A. The follow-up alone is 60 degrees
    # off it (cosine 0.5, under the 0.55 bar); with half the question added it
    # is 41 degrees off (0.76). No word of either is in the note.
    question, follow = "who is my landlord?", "and how much is it?"
    embedding = TableEmbedding(
        {question: (1.0, 0.0, 0.0), follow: (0.5, 3**0.5 / 2, 0.0)}
    )
    recalled: dict[bool, bool] = {}
    for setting in (False, True):
        memory = memory_in(tmp_path / str(setting), settings(setting))
        memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
        other = await said(store, None, "rent")
        await memory.stores()[1].add(
            Document(
                session_id=other.session_id,
                turn=1,
                surface="gateway",
                kind="chat",
                title="Rent",
                text="Rent is 900.",
                created_at=time.time(),
                vector=(1.0, 0.0, 0.0),
                embed_model="table",
            )
        )
        asked = await store.begin_turn(
            None, question, holder=HOLDER, new_session=NewSession()
        )
        await memory.before_turn(store, asked, question)
        await finish(store, asked)
        clock.advance(60)
        second = await running(store, asked.session_id, follow)
        embedding.calls = 0
        found = await memory.before_turn(store, second, follow)
        await finish(store, second)
        recalled[setting] = bool(found.used)
        assert embedding.calls == 1  # the follow-up's own embedding, no other call
        assert found.recall is not None
        assert found.recall.context == ("used" if setting else "off")
        await memory.aclose()
    assert recalled == {False: False, True: True}


async def test_the_cache_of_query_vectors_is_bounded_and_keeps_unit_vectors(
    store: SessionStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_PRIOR_CHATS", 2)
    memory = memory_in(tmp_path, settings(True))
    embedding = FakeEmbedding()
    memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    sessions = []
    for number in range(3):
        first = await said(store, None, f"question number {number}")
        sessions.append(first.session_id)
        turn = await running(store, first.session_id, f"follow up {number}")
        await memory.before_turn(store, turn, f"follow up {number}")
    assert list(memory._prior) == sessions[1:]  # the oldest chat's vector is gone
    number, model, vector = memory._prior[sessions[2]]
    assert (number, model) == (2, "fake-embed")
    assert np.isclose(float(np.linalg.norm(vector)), 1.0)
    # Off: nothing is kept.
    plain = memory_in(tmp_path / "off", settings(False))
    plain.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    await plain.before_turn(store, turn, "follow up 2")
    assert not plain._prior
    await memory.aclose()
    await plain.aclose()


async def test_a_line_a_recall_block_in_the_window_shows_is_not_added_again_with_context_query(
    store: SessionStore, clock: Clock, tmp_path: Path
) -> None:
    asked = "advisor office"
    for setting in (False, True):
        memory = memory_in(tmp_path / str(setting), settings(setting))
        await advisor_doc(store, memory)
        _, follow = await follow_up(store, clock)
        first = await memory.before_turn(store, follow, asked)
        assert len(first.used) == 1
        window = [Message("user", asked + first.block), Message("assistant", "ok")]
        second = await memory.before_turn(store, follow, asked, window=window)
        if setting:
            assert second.used == () and second.block == ""
        else:  # as before: it is shown again
            assert second.block == first.block
        await memory.aclose()
