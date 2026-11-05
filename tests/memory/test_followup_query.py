"""A message with no content of its own recalls from the chat's last turns.

"go look it up", "continue" or "去查一下" says nothing to search for: on its own words,
BM25 matches whatever holds them (the person's identity turns, say), and those went
into the prompt. Such a message is searched with the user's words of the chat's last
two turns that the model's window still shows; with no such turn nothing is recalled.
Synthetic chats and notes, words only, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.schema import Message
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, service
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import gate_terms, is_contentless
from localmelo.support.memory.service import (
    _QUERY_CHARS,
    FOLLOW_UP_TURNS,
    Recall,
    TurnMemory,
    follow_up_query,
    recent_said,
)
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.sessions.helpers import Clock

HOLDER = "test-host:1:followup"

EN_FRIEND = (
    "Lena is my friend from college",
    "Lena got a new job at the bakery on Elm Street",
    "she starts as head baker next Monday",
)
EN_FRIEND_DOC = (
    "Lena got a new job at the bakery on Elm Street and starts as head baker on Monday",
    "That is great news for her.",
)
EN_IDENTITY = (
    ("can you look up my name and go over my identity details?", "Your name is Juno."),
    ("go look at my profile and look up who I am", "You are Juno, a student."),
)
ZH_FRIEND = (
    "我的朋友小林是大学同学",
    "小林在榆树街找到了面包店的新工作",
    "她下周一开始当主厨",
)
ZH_FRIEND_DOC = (
    "小林在榆树街找到了面包店的新工作，她下周一开始当主厨",
    "这对她是个好消息。",
)
ZH_IDENTITY = (
    ("请继续去查一下我的名字和身份", "你叫小明。"),
    ("继续去查一下我是谁", "你是小明，一名学生。"),
)


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


def config() -> Config:
    return Config(chat_backend="vllm", embedding_backend="none")


async def finish(
    store: SessionStore, turn: TurnInfo, answer: str = "Okay."
) -> TurnInfo:
    return await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer=answer,
    )


async def said(
    store: SessionStore,
    clock: Clock,
    session_id: str | None,
    text: str,
    *,
    origin: str = "",
) -> TurnInfo:
    """A finished turn of *text*, in a new chat when *session_id* is None."""
    clock.advance(10)
    turn = await store.begin_turn(
        session_id,
        text,
        holder=HOLDER,
        title=text if session_id is None else None,
        new_session=NewSession() if session_id is None else None,
        origin=origin,
    )
    return await finish(store, turn)


async def chat_of(
    store: SessionStore, clock: Clock, texts: Sequence[str]
) -> list[TurnInfo]:
    turns = [await said(store, clock, None, texts[0])]
    for text in texts[1:]:
        turns.append(await said(store, clock, turns[0].session_id, text))
    return turns


async def now_in(
    store: SessionStore, clock: Clock, session_id: str, typed: str
) -> TurnInfo:
    """The turn under way: the one whose message is recalled for."""
    clock.advance(10)
    return await store.begin_turn(session_id, typed, holder=HOLDER)


def window_of(*turns: TurnInfo) -> list[Message]:
    """The messages the runner loads for finished turns: the turns' own times."""
    out: list[Message] = []
    for turn in turns:
        out.append(Message("user", turn.user_text, timestamp=turn.started_at))
        out.append(Message("assistant", turn.answer, timestamp=turn.finished_at or 0.0))
    return out


async def other_chat_turn(
    memory: ChatMemory, store: SessionStore, clock: Clock, typed: str, answer: str
) -> None:
    """A turn of a chat of its own, indexed as a document."""
    turn = await said(store, clock, None, typed)
    _, index = memory.stores()
    await index.add(
        Document(
            session_id=turn.session_id,
            turn=1,
            surface="gateway",
            kind="chat",
            title=typed,
            text=document_text(typed, answer),
            created_at=time.time(),
        )
    )


async def build(
    tmp_path: Path,
    store: SessionStore,
    clock: Clock,
    cfg: Config,
    friend: Sequence[tuple[str, str]],
    identity: Sequence[tuple[str, str]],
) -> ChatMemory:
    """A memory over *cfg* holding the friend turn and the identity turns, each
    in a chat of its own, so a search can find them."""
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    for typed, answer in (*friend, *identity):
        await other_chat_turn(memory, store, clock, typed, answer)
    return memory


class Spy:
    """What the search was asked: the queries ``_find`` ranked, and the embedding calls."""

    def __init__(self, memory: ChatMemory) -> None:
        self.queries: list[str] = []
        self.embeds = 0
        find, embed = memory._find, memory._embed

        async def spy_find(query: str, *args: Any, **kwargs: Any) -> Any:
            self.queries.append(query)
            return await find(query, *args, **kwargs)

        async def spy_embed(*args: Any, **kwargs: Any) -> Any:
            self.embeds += 1
            return await embed(*args, **kwargs)

        memory._find = spy_find  # type: ignore[method-assign]
        memory._embed = spy_embed  # type: ignore[method-assign]


def texts(found: TurnMemory) -> list[str]:
    return [item.text for item in found.used]


def about(found: TurnMemory, word: str) -> bool:
    return bool(found.used) and all(word in item.text for item in found.used)


# ── What has no content of its own ──


@pytest.mark.parametrize(
    "message",
    [
        "go look it up",
        "Go look it up!",
        "look it up",
        "continue",
        "please continue",
        "do it",
        "do that",
        "go on",
        "go ahead",
        "keep going",
        "try again",
        "tell me more",
        "show me",
        "let me know",
        "do it for me",
        "look it up for me",
        "给我看看",
        "让我想想",
        "告诉我",
        "我们继续",
        "what about that one?",
        "and the second one?",
        "what about her?",
        "ok",
        "yes",
        "thanks",
        "why?",
        "去查一下",
        "继续",
        "请继续",
        "那个呢",
        "然后呢？",
        "好的",
        "帮我查一下",
        "再试一次",
        "第二个呢",
        "开始吧",
        "不用了",
        "继续 go on",
        "",
        "   ",
    ],
)
def test_a_message_that_names_nothing_has_no_content(message: str) -> None:
    assert is_contentless(message)


@pytest.mark.parametrize(
    "message",
    [
        "what is Lena's new job?",
        "我朋友叫什么名字",
        "what is my name?",
        "what's my name",
        "what do you know about me?",
        "who am I?",
        "do you remember me?",
        "tell me about myself",
        "你知道我什么",
        "我是谁",
        "我的名字",
        "where do I live?",
        "我叫什么名字",
        "do you remember the dentist appointment?",
        "Lena?",
        "go look up Lena",
        "查一下我朋友",
        "我的猫叫什么",
        "开会",
        "上海",
        "后天",
        "check file_write",
        "summarize that",
        "go look it up on 12 May",  # a number
        'go look up "Lena"',  # quoted text
        "look at https://example.com",  # a URL
        "继续写《三体》",
        "привет",  # a script it does not know is content
        "こんにちは",
    ],
)
def test_a_message_with_a_word_a_number_a_url_or_a_quote_has_content(
    message: str,
) -> None:
    assert not is_contentless(message)


def test_the_recall_bars_count_these_messages_as_content() -> None:
    # Why gate_terms is not the measure: it counts "go", "look" and "up", and a
    # Chinese run as its characters and bigrams.
    assert len(gate_terms("go look it up")) == 3
    assert len(gate_terms("继续")) == 3
    assert len(gate_terms("去查一下")) > 3


# ── The query ──


def test_the_query_is_the_message_then_the_recent_words_newest_first() -> None:
    assert follow_up_query("go look it up ", ["newest", "older"]) == (
        "go look it up\nnewest\nolder"
    )
    assert follow_up_query("go look it up", []) == ""


def test_the_query_is_bounded() -> None:
    long = "x" * (_QUERY_CHARS * 3)
    query = follow_up_query("go on", [long, "older"])
    assert len(query) == _QUERY_CHARS and query.startswith("go on\nxxx")
    assert "older" not in query


@pytest.mark.parametrize(
    ("message", "friend", "doc", "identity", "wanted"),
    [
        ("go look it up", EN_FRIEND, EN_FRIEND_DOC, EN_IDENTITY, "Lena"),
        ("去查一下", ZH_FRIEND, ZH_FRIEND_DOC, ZH_IDENTITY, "小林"),
        ("继续", ZH_FRIEND, ZH_FRIEND_DOC, ZH_IDENTITY, "小林"),
    ],
)
async def test_a_follow_up_recalls_what_the_chat_was_about_not_its_own_words(
    tmp_path: Path,
    store: SessionStore,
    clock: Clock,
    message: str,
    friend: tuple[str, ...],
    doc: tuple[str, str],
    identity: Sequence[tuple[str, str]],
    wanted: str,
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [doc], identity)
    try:
        turns = await chat_of(store, clock, friend)
        sid = turns[0].session_id
        spy = Spy(memory)
        # The last two turns' words are searched with it.
        info = await now_in(store, clock, sid, message)
        found = await memory.before_turn(store, info, message)
        assert spy.queries == [f"{message}\n{friend[2]}\n{friend[1]}"]
        assert friend[0] not in spy.queries[0]
        assert about(found, wanted), texts(found)
        assert found.recall is not None and found.recall.query_from == "recent"
        assert found.block and wanted in found.block
        assert found.event()["recall"]["query_from"] == "recent"  # type: ignore[index]
    finally:
        await memory.aclose()


async def test_melos_replies_are_not_part_of_the_query(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        first = await said(store, clock, None, EN_FRIEND[1])
        clock.advance(10)
        turn = await store.begin_turn(first.session_id, EN_FRIEND[2], holder=HOLDER)
        await finish(store, turn, "I guessed Mara is the baker's name; zebra crossing.")
        spy = Spy(memory)
        info = await now_in(store, clock, first.session_id, "go look it up")
        await memory.before_turn(store, info, "go look it up")
        assert spy.queries == [f"go look it up\n{EN_FRIEND[2]}\n{EN_FRIEND[1]}"]
        assert "Mara" not in spy.queries[0] and "zebra" not in spy.queries[0]
    finally:
        await memory.aclose()


async def test_undone_turns_and_a_tasks_questions_are_not_part_of_the_query(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        first = await said(store, clock, None, EN_FRIEND[1])
        sid = first.session_id
        await said(store, clock, sid, "tell me about my boss Ivan and his budget")
        assert await store.undo_last_turn(sid) is not None
        await said(
            store, clock, sid, "which folder is for zebra reports?", origin="ask_owner"
        )
        await said(store, clock, sid, EN_FRIEND[2])
        spy = Spy(memory)
        info = await now_in(store, clock, sid, "go look it up")
        found = await memory.before_turn(store, info, "go look it up")
        assert spy.queries == [f"go look it up\n{EN_FRIEND[2]}\n{EN_FRIEND[1]}"]
        assert about(found, "Lena")
    finally:
        await memory.aclose()


async def test_only_the_last_two_turns_are_used_and_attachments_are_not(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        turns = await chat_of(store, clock, ("alpha", "bravo", "charlie", "delta"))
        info = await now_in(store, clock, turns[0].session_id, "go on")
        assert FOLLOW_UP_TURNS == 2
        assert await recent_said(store, info, None) == ["delta", "charlie"]
        assert await recent_said(store, info, None, 3) == ["delta", "charlie", "bravo"]
        assert await recent_said(store, info, None, 9) == [
            "delta",
            "charlie",
            "bravo",
            "alpha",
        ]
        assert await recent_said(store, info, None, 0) == []
    finally:
        await memory.aclose()


async def test_many_turns_are_read_a_page_at_a_time(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    memory = ChatMemory(lambda: config(), directory=str(tmp_path / "memory"))
    try:
        turns = await chat_of(store, clock, [f"turn {n}" for n in range(1, 41)])
        sid = turns[0].session_id
        for _ in range(37):  # all but the first three turns are undone
            assert await store.undo_last_turn(sid) is not None
        info = await now_in(store, clock, sid, "go on")
        assert await recent_said(store, info, None) == ["turn 3", "turn 2"]
        assert await recent_said(store, info, 2) == ["turn 3", "turn 2"]
        assert await recent_said(store, info, 3) == ["turn 3"]
    finally:
        await memory.aclose()


async def test_a_turn_with_no_content_does_not_use_up_a_place(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        turns = await chat_of(store, clock, EN_FRIEND)
        sid = turns[0].session_id
        for text in ("continue", "ok", "go on"):
            await said(store, clock, sid, text)
        await said(store, clock, sid, "tell me more", origin="followup")
        spy = Spy(memory)
        info = await now_in(store, clock, sid, "continue")
        found = await memory.before_turn(store, info, "continue")
        assert spy.queries == [f"continue\n{EN_FRIEND[2]}\n{EN_FRIEND[1]}"]
        assert about(found, "Lena")
        assert found.recall is not None and found.recall.query_from == "recent"
        # A chat of turns that say nothing, one after the other, has nothing to follow.
        other = await chat_of(store, clock, ("continue", "ok", "go on"))
        assert (
            await recent_said(
                store, await now_in(store, clock, other[0].session_id, "yes"), None
            )
            == []
        )
    finally:
        await memory.aclose()


async def test_only_a_bounded_number_of_turns_is_looked_at(
    tmp_path: Path, store: SessionStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = ChatMemory(lambda: config(), directory=str(tmp_path / "memory"))
    try:
        monkeypatch.setattr(service, "_FOLLOW_UP_SCAN", 20)
        turns = await chat_of(store, clock, ["hello world", *["ok"] * 30])
        info = await now_in(store, clock, turns[0].session_id, "go on")
        assert await recent_said(store, info, None) == []
        monkeypatch.setattr(service, "_FOLLOW_UP_SCAN", 100)
        assert await recent_said(store, info, None) == ["hello world"]
    finally:
        await memory.aclose()


async def test_the_window_decides_which_turns_the_query_is_made_of(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        turns = await chat_of(store, clock, (*EN_FRIEND, "and she likes rye bread"))
        sid = turns[0].session_id
        spy = Spy(memory)
        # The window holds the last three turns: the last two are the query.
        info = await now_in(store, clock, sid, "go look it up")
        await memory.before_turn(
            store, info, "go look it up", window=window_of(*turns[1:])
        )
        assert (
            spy.queries[-1]
            == f"go look it up\n{turns[3].user_text}\n{turns[2].user_text}"
        )
        await finish(store, info)
        # It holds the last one and the turn that followed it, which says nothing
        # of its own and is skipped: the last one is all there is to follow.
        info = await now_in(store, clock, sid, "go look it up")
        earlier = (await store.list_turns(sid))[-2]
        assert earlier.user_text == "go look it up"
        mine = window_of(turns[3], earlier)
        await memory.before_turn(store, info, "go look it up", window=mine)
        assert spy.queries[-1] == f"go look it up\n{turns[3].user_text}"
        await finish(store, info)
        # It holds only turns that say nothing: nothing to follow.
        count = len(spy.queries)
        info = await now_in(store, clock, sid, "go look it up")
        none = await memory.before_turn(
            store, info, "go look it up", window=window_of(earlier)
        )
        assert len(spy.queries) == count and none.recall is not None
        assert none.recall.query_from == "none" and not none.block
        await finish(store, info)
        # It holds none of them: nothing to follow, nothing recalled.
        count = len(spy.queries)
        info = await now_in(store, clock, sid, "go look it up")
        found = await memory.before_turn(store, info, "go look it up", window=[])
        assert len(spy.queries) == count and found.recall is not None
        assert found.recall.query_from == "none" and not found.block
    finally:
        await memory.aclose()


# ── No earlier turn ──


@pytest.mark.parametrize(
    "message",
    [
        "我是谁",
        "你知道我什么",
        "what do you know about me?",
        "what's my name",
        "who am I?",
    ],
)
async def test_a_question_about_the_person_is_searched_as_typed_in_a_new_chat(
    tmp_path: Path, store: SessionStore, clock: Clock, message: str
) -> None:
    cfg = config()
    memory = await build(
        tmp_path,
        store,
        clock,
        cfg,
        [EN_FRIEND_DOC, ZH_FRIEND_DOC],
        EN_IDENTITY + ZH_IDENTITY,
    )
    try:
        spy = Spy(memory)
        info = await store.begin_turn(
            None, message, holder=HOLDER, title=message, new_session=NewSession()
        )
        found = await memory.before_turn(store, info, message, window=[])
        assert found.recall is not None and found.recall.query_from == "message"
        assert spy.queries == [message]  # a search ran, on the message
    finally:
        await memory.aclose()


async def test_a_question_about_the_person_after_other_talk_is_searched_as_typed(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        turns = await chat_of(store, clock, EN_FRIEND)
        spy = Spy(memory)
        for message in ("我是谁", "what do you know about me?"):
            info = await now_in(store, clock, turns[0].session_id, message)
            found = await memory.before_turn(store, info, message)
            assert found.recall is not None and found.recall.query_from == "message"
            await finish(store, info)
        assert spy.queries == ["我是谁", "what do you know about me?"]
    finally:
        await memory.aclose()


@pytest.mark.parametrize("message", ["go look it up", "去查一下", "继续", "continue"])
async def test_a_message_with_no_content_in_a_new_chat_recalls_nothing(
    tmp_path: Path, store: SessionStore, clock: Clock, message: str
) -> None:
    cfg = config()
    memory = await build(
        tmp_path,
        store,
        clock,
        cfg,
        [EN_FRIEND_DOC, ZH_FRIEND_DOC],
        EN_IDENTITY + ZH_IDENTITY,
    )
    try:
        spy = Spy(memory)
        info = await store.begin_turn(
            None, message, holder=HOLDER, title=message, new_session=NewSession()
        )
        found = await memory.before_turn(store, info, message, window=[])
        assert found.block == "" and found.used == ()
        assert found.recall is not None and found.recall.query_from == "none"
        assert (found.recall.used, found.recall.candidates) == (0, 0)
        assert spy.queries == [] and spy.embeds == 0  # no BM25 or vector search
        data = found.event()
        assert data is not None and data["recall"]["query_from"] == "none"
        assert data["used"] == []
        # Without the window (it is not known) a new chat has no turn either.
        again = await store.begin_turn(
            None, message, holder=HOLDER, title=message, new_session=NewSession()
        )
        none = await memory.before_turn(store, again, message)
        assert none.block == "" and spy.queries == []
    finally:
        await memory.aclose()


async def test_a_chat_whose_turns_are_all_undone_has_nothing_to_follow(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        first = await said(store, clock, None, EN_FRIEND[1])
        assert await store.undo_last_turn(first.session_id) is not None
        spy = Spy(memory)
        info = await now_in(store, clock, first.session_id, "go look it up")
        found = await memory.before_turn(store, info, "go look it up")
        assert found.block == "" and spy.queries == []
        assert found.recall is not None and found.recall.query_from == "none"
    finally:
        await memory.aclose()


async def test_a_failure_to_read_the_turns_searches_the_message_as_before(
    tmp_path: Path, store: SessionStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await build(tmp_path, store, clock, config(), [EN_FRIEND_DOC], EN_IDENTITY)
    try:

        async def broken(*args: Any, **kwargs: Any) -> list[str]:
            raise RuntimeError("the store is closed")

        monkeypatch.setattr(service, "recent_said", broken)
        turns = await chat_of(store, clock, EN_FRIEND)
        spy = Spy(memory)
        info = await now_in(store, clock, turns[0].session_id, "go look it up")
        found = await memory.before_turn(store, info, "go look it up")
        assert spy.queries == ["go look it up"]
        assert found.recall is not None and found.recall.query_from == "message"
    finally:
        await memory.aclose()


# ── A message with content ──


@pytest.mark.parametrize(
    ("message", "friend", "doc", "identity"),
    [
        ("what is Lena's new job?", EN_FRIEND, EN_FRIEND_DOC, EN_IDENTITY),
        ("我朋友叫什么名字", ZH_FRIEND, ZH_FRIEND_DOC, ZH_IDENTITY),
        ("what is my name?", EN_FRIEND, EN_FRIEND_DOC, EN_IDENTITY),
    ],
)
async def test_a_message_with_content_is_searched_as_it_is(
    tmp_path: Path,
    store: SessionStore,
    clock: Clock,
    message: str,
    friend: tuple[str, ...],
    doc: tuple[str, str],
    identity: Sequence[tuple[str, str]],
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [doc], identity)
    try:
        turns = await chat_of(store, clock, friend)
        sid = turns[0].session_id
        spy = Spy(memory)
        info = await now_in(store, clock, sid, message)
        found = await memory.before_turn(store, info, message)
        await finish(store, info)
        assert found.recall is not None and found.recall.query_from == "message"
        assert spy.queries == [message]  # byte for byte
        # A new chat's first message with content is searched as it is, too.
        info = await store.begin_turn(
            None, message, holder=HOLDER, title=message, new_session=NewSession()
        )
        first = await memory.before_turn(store, info, message, window=[])
        assert spy.queries[-1] == message
        assert first.recall is not None and first.recall.query_from == "message"
    finally:
        await memory.aclose()


async def test_a_long_message_is_cut_as_before(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    memory = await build(tmp_path, store, clock, config(), [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        long = "lena bakery " * 400
        spy = Spy(memory)
        info = await store.begin_turn(
            None, long, holder=HOLDER, title="long", new_session=NewSession()
        )
        await memory.before_turn(store, info, long, window=[])
        assert spy.queries == [long[:_QUERY_CHARS]]
    finally:
        await memory.aclose()


async def test_a_tasks_question_to_its_chat_is_searched_as_before(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    cfg = config()
    memory = await build(tmp_path, store, clock, cfg, [EN_FRIEND_DOC], EN_IDENTITY)
    try:
        turns = await chat_of(store, clock, EN_FRIEND)
        sid = turns[0].session_id
        clock.advance(10)
        asked = await store.begin_turn(
            sid, "go look it up", holder=HOLDER, origin="ask_owner"
        )
        spy = Spy(memory)
        found = await memory.before_turn(store, asked, "", window=window_of(*turns))
        assert spy.queries == ["go look it up"]
        assert found.recall is not None and found.recall.query_from == "message"
    finally:
        await memory.aclose()


def test_the_event_key_is_always_there() -> None:
    assert Recall().to_json()["query_from"] == "message"
    assert Recall(query_from="recent").to_json()["query_from"] == "recent"
    assert (
        service.TurnMemory(recall=Recall(query_from="none")).event()["recall"][  # type: ignore[index]
            "query_from"
        ]
        == "none"
    )
