"""A chat remembers its own past.

A chat's turns that its model's window no longer holds are recalled like
another chat's, as ``this chat``; the ones it still holds are not
(``window_start``: the turn of the window's first user message, by time stamp).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from localmelo.melo.schema import EMPTY_REPLY_NUDGE, TOOL_CALL_TEXT_NUDGE, Message
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, service
from localmelo.support.memory.recall import (
    Item,
    in_view,
    this_chat,
)
from localmelo.support.memory.service import CHAT_TURN, window_start
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.memory.chat_memory_helpers import teach
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
    # So is the nudge after a tool call written as text, also with a recall
    # block after it.
    for text in (TOOL_CALL_TEXT_NUDGE, TOOL_CALL_TEXT_NUDGE + "\n\n[recall]"):
        other = Message("user", text, timestamp=turns[2].started_at + 1)
        assert await window_start(store, now, [other, *window_of(turns[3])]) == 4
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


def test_in_view_and_this_chat() -> None:
    turn = Item("d:1", "User: a melo: b", "chat “a”, Oct 1", "s", 5, 1.0)
    note = Item("r:1", "Your name is Juno.", "chat “a”, Oct 1", "s", 5, 1.0)
    assert in_view(turn, "s", None) and in_view(turn, "s", 5) and in_view(turn, "s", 3)
    assert not in_view(turn, "s", 6) and not in_view(turn, "t", None)
    assert not in_view(note, "s", None)  # a note is never in view
    assert this_chat(turn, "s").where.startswith("this chat, ")
    assert this_chat(turn, "t") is turn and this_chat(note, "s") is note
