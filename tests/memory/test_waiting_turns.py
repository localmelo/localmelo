"""The turns that wait to be encoded, and one window of them (#428).

:meth:`ChatMemory.waiting_turns` says which turns of a chat no item holds yet and an
encoder may read; :meth:`ChatMemory.encode_window` encodes the first window of them
and moves the chat's mark to its end, with the encoder's replies kept in memory only.
Real session store and memory index in a temporary folder, a stub chat model.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.config import Config, MemoryConfig
from localmelo.support.memory import ChatMemory, hippo_encode
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo
from localmelo.tests.memory.hippo_helpers import StubEncoder, item, says

HOLDER = "test-host:1:waiting"
# Long enough that two turns fill the smallest window (4000 characters) and a third does not.
PAD = "word " * 300
WINDOW = 4000


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
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(compress_at_chars=WINDOW)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def turn(
    store: SessionStore,
    session_id: str | None,
    typed: str,
    answer: str = "Noted.",
    *,
    status: str = "completed",
    origin: str = "",
) -> TurnInfo:
    """One turn of a chat, begun and then finished (left running for ``running``)."""
    info = await store.begin_turn(
        session_id,
        typed,
        holder=HOLDER,
        origin=origin,
        new_session=None if session_id else NewSession(),
    )
    if status == "running":
        return info
    return await store.finish_turn(
        info.session_id,
        info.turn,
        holder=HOLDER,
        status=status,
        cause="run",
        result=None,
        answer=answer,
    )


async def numbers(
    memory: ChatMemory, store: SessionStore, session_id: str
) -> list[int]:
    return [t.turn for t in await memory.waiting_turns(store, session_id)]


async def test_every_finished_turn_of_an_unmarked_chat_waits(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await turn(store, None, "I live in Utrecht.")
    await turn(store, first.session_id, "I like tea.")
    assert await numbers(memory, store, first.session_id) == [1, 2]
    # A chat with no turns, and a chat that does not exist, have none waiting.
    assert await numbers(memory, store, "nobody") == []


async def test_turns_up_to_the_mark_do_not_wait(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await turn(store, None, "one")
    for text in ("two", "three"):
        await turn(store, first.session_id, text)
    _, index = memory.stores()
    await index.mark_encoded(first.session_id, 2)
    assert await numbers(memory, store, first.session_id) == [3]
    await index.mark_encoded(first.session_id, 3)
    assert await numbers(memory, store, first.session_id) == []


async def test_what_the_index_would_not_hold_does_not_wait(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await turn(store, None, "one")
    sid = first.session_id
    await turn(store, sid, "an answer nobody wrote", answer="")
    await turn(store, sid, "a task asks its chat", origin="ask_owner")
    await turn(store, sid, "this one is taken back")
    assert await store.undo_last_turn(sid) is not None
    await turn(store, sid, "two")
    assert await numbers(memory, store, sid) == [1, 5]


async def test_a_turn_that_ended_without_completing_is_skipped_not_waited_for(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await turn(store, None, "one")
    sid = first.session_id
    await turn(store, sid, "this one failed", status="failed")
    await turn(store, sid, "this one was cancelled", status="cancelled")
    await turn(store, sid, "out of steps", status="budget_exhausted")
    await turn(store, sid, "five")
    assert await numbers(memory, store, sid) == [1, 5]


async def test_the_list_stops_before_a_turn_that_still_runs(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await turn(store, None, "one")
    sid = first.session_id
    await turn(store, sid, "two")
    await turn(store, sid, "this one was cancelled", status="cancelled")
    await turn(store, sid, "this one runs", status="running")
    assert await numbers(memory, store, sid) == [1, 2]


async def test_a_turn_from_before_the_user_forgot_everything_does_not_wait(
    memory: ChatMemory, store: SessionStore
) -> None:
    first = await turn(store, None, "one")
    assert await numbers(memory, store, first.session_id) == [1]
    await memory.forget_all()
    assert await numbers(memory, store, first.session_id) == []
    await turn(store, first.session_id, "two")
    assert await numbers(memory, store, first.session_id) == [2]


FACTS = [
    "The user keeps bees on a rooftop.",
    "Tomatoes grow best in the greenhouse.",
    "The marathon route crosses the old bridge.",
    "Grandmother taught piano every Sunday.",
    "The library closes early on Fridays.",
]


def reply(prompt: str) -> str:
    """One item for each exchange the prompt shows, the fact of the turn typed there."""
    shown = re.findall(r"^\[(\d+)\] .*?User: turn (\d+)", prompt, re.MULTILINE)
    return says(
        [item(f"Fact {k}", FACTS[int(k) - 1], "episodic", [int(n)]) for n, k in shown]
    )


async def padded_chat(store: SessionStore, count: int) -> str:
    sid = (await turn(store, None, "turn 1", PAD)).session_id
    for n in range(2, count + 1):
        await turn(store, sid, f"turn {n}", PAD)
    return sid


async def test_one_window_is_encoded_and_the_mark_moves_to_its_end(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await padded_chat(store, 5)
    stub = StubEncoder(reply)
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    _, index = memory.stores()
    for window, expected in enumerate(([1, 2], [3, 4], [5])):
        out = await memory.encode_window(
            store, sid, "Chat", await memory.waiting_turns(store, sid)
        )
        assert out is not None
        shown = re.findall(r"User: turn (\d+)", stub.prompts[window])
        assert shown == [str(n) for n in expected]
        assert await index.encoded_turn(sid) == expected[-1]
    assert len(stub.prompts) == 3, "one request for each window"
    assert await memory.waiting_turns(store, sid) == []
    assert await index.count_items() == 5


async def test_a_turn_longer_than_the_window_is_a_window_of_its_own(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = (await turn(store, None, "turn 1", PAD * 3)).session_id
    await turn(store, sid, "turn 2", "Noted.")
    stub = StubEncoder(reply)
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    out = await memory.encode_window(
        store, sid, "Chat", await memory.waiting_turns(store, sid)
    )
    assert out is not None
    assert len(stub.prompts) > 1, "the long turn is read in its parts"
    assert not any("User: turn 2" in prompt for prompt in stub.prompts)
    assert await memory.stores()[1].encoded_turn(sid) == 1


async def test_a_window_the_model_failed_sets_no_mark(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await padded_chat(store, 3)
    stub = StubEncoder("RAISE")
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    out = await memory.encode_window(
        store, sid, "Chat", await memory.waiting_turns(store, sid)
    )
    assert out is not None and out.encoding.failed_from == 0
    assert await memory.stores()[1].encoded_turn(sid) is None
    assert await numbers(memory, store, sid) == [1, 2, 3]


async def test_nothing_to_read_is_none(memory: ChatMemory, store: SessionStore) -> None:
    sid = (await turn(store, None, "one")).session_id
    assert await memory.encode_window(store, sid, "Chat", []) is None


async def test_the_replies_are_not_written_to_the_cache_file(
    tmp_path: Path, memory: ChatMemory, store: SessionStore
) -> None:
    sid = await padded_chat(store, 2)
    stub = StubEncoder(reply)
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    assert await memory.encode_window(
        store, sid, "Chat", await memory.waiting_turns(store, sid)
    )
    assert not (tmp_path / "memory" / "compress_cache.jsonl").exists()
    assert not list(tmp_path.rglob("*.jsonl"))


async def test_a_cache_with_no_path_keeps_its_replies_in_memory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = hippo_encode.Cache("")
    cache.put("key", {"items": []})
    assert cache.get("key") == {"items": []}
    assert hippo_encode.Cache("").get("key") is None
    assert list(tmp_path.iterdir()) == []
    # A cache with a path still writes its file, as before.
    disk = hippo_encode.Cache(str(tmp_path / "replies.jsonl"))
    disk.put("key", {"items": []})
    assert json.loads((tmp_path / "replies.jsonl").read_text())["key"] == "key"


async def test_the_turns_a_window_holds_are_the_ones_the_index_holds(
    memory: ChatMemory, store: SessionStore
) -> None:
    """The encoder reads the words and the answer of the turn, as the live index does."""
    sid = (
        await turn(store, None, "I live in Utrecht.", "Utrecht is a canal city.")
    ).session_id
    stub = StubEncoder(reply)
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    started = time.time()
    await memory.encode_window(
        store, sid, "Chat", await memory.waiting_turns(store, sid)
    )
    (prompt,) = stub.prompts
    assert "User: I live in Utrecht. / melo: Utrecht is a canal city." in prompt
    assert time.strftime("%Y-%m-%d", time.localtime(started)) in prompt


async def test_turns_from_before_a_forget_are_not_read_at_all(
    memory: ChatMemory, store: SessionStore
) -> None:
    """A forget between :meth:`waiting_turns` and :meth:`encode_window` keeps the old turns out."""
    sid = await padded_chat(store, 2)
    stub = StubEncoder(reply)
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    waiting = await memory.waiting_turns(store, sid)
    await memory.forget_all()
    assert await memory.encode_window(store, sid, "Chat", waiting) is None
    assert stub.prompts == []
    assert await memory.stores()[1].encoded_turn(sid) is None


async def test_a_chat_that_is_gone_is_not_read(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await padded_chat(store, 2)
    stub = StubEncoder(reply)
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    waiting = await memory.waiting_turns(store, sid)
    assert await store.delete_session(sid)
    assert await memory.encode_window(store, sid, "Chat", waiting) is None
    assert stub.prompts == []


async def test_a_guard_that_says_no_keeps_the_window_out(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await padded_chat(store, 2)
    stub = StubEncoder(reply)
    memory.chat_provider = lambda: stub  # type: ignore[method-assign]
    _, index = memory.stores()
    seen: list[bool] = []

    async def no() -> bool:
        seen.append(memory._lock.locked())  # a forget takes this lock: so is it held
        return False

    turns = [
        hippo_encode.Turn(n, 1000.0 * n, 1000.0 * n + 5, f"turn {n}", "Noted.")
        for n in (1, 2)
    ]
    out = await memory.compress_chat(sid, "Chat", turns, guard=no)
    assert out.forgotten and out.routed == [] and seen == [True]
    assert await index.count_items() == 0 and await index.encoded_turn(sid) is None

    # With a guard that agrees, or none, the same call stores and marks as ever.
    async def yes() -> bool:
        return True

    out = await memory.compress_chat(sid, "Chat", turns, guard=yes)
    assert not out.forgotten and await index.encoded_turn(sid) == 2
