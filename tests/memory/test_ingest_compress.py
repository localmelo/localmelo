"""``ingest.run(..., mode="compress")``: chats written as turns, then encoded into
memory items and routed (developer setting, #428).

The turns, their documents and the rule-path notes are written as ever; then each
chat is encoded window by window by the chat model and each item is routed against
the items kept, the oldest chat first. The report says, for each chat of the file,
its windows, the gate, the items stored and what became of each. Synthetic chats, a
stub chat model and temporary stores: no model server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import Config, MemoryConfig
from localmelo.support.memory import ChatMemory, ingest
from localmelo.support.memory.ingest import IngestError
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding
from localmelo.tests.memory.hippo_helpers import StubEncoder, item, says

MAY = {
    "id": "may",
    "title": "Moving",
    "date": "2023-05-20T13:40:00",
    "turns": [
        {"role": "user", "text": "I live in Pittsburgh."},
        {"role": "assistant", "text": "Pittsburgh is a lovely city."},
        {"role": "user", "text": "What is a good book about trains?"},
        {"role": "assistant", "text": "Try a history of the railroad."},
    ],
}
JUNE = {
    "id": "june",
    "title": "Moved",
    "date": "2023-06-01T09:00:00",
    "turns": [
        {"role": "user", "text": "I live in Utrecht now."},
        {"role": "assistant", "text": "Utrecht is a canal city."},
    ],
}
PITTSBURGH = "The user said they live in Pittsburgh."
UTRECHT = "The user said they now live in Utrecht."
BOOK = "The user asked for a book about trains and melo suggested a railroad history."


def reply(prompt: str) -> str:
    """What the stub encoder says of a chat, by what the prompt shows."""
    if "Pittsburgh" in prompt:
        return says(
            [
                item("Home city", PITTSBURGH, "profile", [0]),
                item("Train book", BOOK, "episodic", [1]),
            ]
        )
    if "Utrecht" in prompt:
        return says([item("Home city", UTRECHT, "profile", [0])])
    return says()


def settings(tmp_path: Path, **memory: Any) -> Config:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(
        compress_cache=str(tmp_path / "shared-cache.jsonl"), **memory
    )
    return cfg


async def compress(
    tmp_path: Path,
    data: object,
    provider: StubEncoder | None,
    *,
    sub: str = "a",
    mode: str = "compress",
    cfg: Config | None = None,
    embedding: FakeEmbedding | None = None,
) -> tuple[ingest.Report, SessionStore, ChatMemory]:
    """Ingest *data* with *provider* as the chat model into stores under *sub*."""
    folder = tmp_path / sub
    folder.mkdir(exist_ok=True)
    cfg = cfg or settings(tmp_path)
    pin = ingest.Clock()
    chats = SessionStore(str(folder / "sessions.db"), writer="gateway", clock=pin)
    await chats.open()
    memory = ChatMemory(lambda: cfg, directory=str(folder / "memory"))
    if provider is not None:
        memory.chat_provider = lambda: provider  # type: ignore[method-assign]
    else:
        memory.chat_provider = lambda: None  # type: ignore[method-assign]
    if embedding is not None:
        memory.embedding = lambda: (embedding, embedding.model)  # type: ignore[method-assign]
    try:
        report = await ingest.run(
            memory, chats, pin, ingest.parse(data), notes="rules", mode=mode
        )
    except BaseException:
        await memory.aclose()
        await chats.aclose()
        raise
    return report, chats, memory


async def close(chats: SessionStore, memory: ChatMemory) -> None:
    await memory.aclose()
    await chats.aclose()


# ── What a run writes and reports ──


async def test_the_turns_are_written_as_ever_and_the_items_follow(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(reply)
    report, chats, memory = await compress(tmp_path, {"sessions": [JUNE, MAY]}, stub)
    try:
        assert (report.sessions, report.turns, report.docs) == (2, 3, 3)
        assert report.items == 3 and report.skipped == 0
        _, index = memory.stores()
        assert await index.count() == 3 and await index.count_items() == 3
        may, june = report.ids["may"], report.ids["june"]
        # The oldest chat is routed first: the later "Utrecht" updates "Pittsburgh",
        # not the other way about, although the file lists it first.
        first, book, utrecht = await index.fetch_items([1, 2, 3])
        assert (first.text, first.kind, first.sources) == (
            PITTSBURGH,
            "profile",
            ((may, 1),),
        )
        assert (book.text, book.kind, book.sources) == (BOOK, "episodic", ((may, 2),))
        assert (utrecht.text, utrecht.sources) == (UTRECHT, ((june, 1),))
        assert (first.status, first.replaced_by) == ("replaced", 3)
        assert (book.status, utrecht.status) == ("active", "active")
        assert first.session_id == may and utrecht.session_id == june
        assert first.encoder == "1:stub-encoder" and first.embed_model == ""
        # The item's time is that of the last turn it came from.
        stored = await chats.get_session(may)
        assert stored is not None and stored.title == "Moving"
        assert first.created_at < utrecht.created_at
    finally:
        await close(chats, memory)


async def test_the_report_names_windows_gate_items_and_routes_by_file_id(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(reply)
    report, chats, memory = await compress(tmp_path, {"sessions": [JUNE, MAY]}, stub)
    try:
        data = report.to_json()
        assert data["ingest"] == "compress"
        assert (data["model_calls"], data["cache_hits"]) == (2, 0)
        assert "input_tokens" not in data and "output_tokens" not in data
        assert set(data["compress"]) == {"may", "june"}
        may = data["compress"]["may"]
        # Indexes are the exchange's place in the chat as the file gives it, from 0,
        # not the store's turn number.
        assert may["items"] == [
            {
                "id": 1,
                "title": "Home city",
                "text": PITTSBURGH,
                "kind": "profile",
                "sources": [["may", 0]],
            },
            {
                "id": 2,
                "title": "Train book",
                "text": BOOK,
                "kind": "episodic",
                "sources": [["may", 1]],
            },
        ]
        assert may["gate"] == {
            "kept": [0, 1],
            "ignored": [],
            "unmentioned": [],
            "forced": [],
        }
        (window,) = may["windows"]
        assert (window["exchanges"], window["cached"], window["failed"]) == (
            2,
            False,
            False,
        )
        assert (window["cut"], window["dropped"]) == (False, 0)
        assert window["chars"] > 0
        assert [(r["action"], r["item"], r["target"]) for r in may["routes"]] == [
            ("NEW", 1, None),
            ("NEW", 2, None),
        ]
        assert [r["title"] for r in may["routes"]] == ["Home city", "Train book"]
        (route,) = data["compress"]["june"]["routes"]
        assert (route["action"], route["item"], route["target"]) == ("UPDATE", 3, 1)
        assert route["reason"] == "same subject; update of entry 1"
        assert data["compress"]["june"]["items"][0]["sources"] == [["june", 0]]
        json.dumps(data)  # all of it is plain JSON
        assert report.summary().endswith(
            "Encoded 3 items from 2 chats (2 model calls, 0 from the cache)."
        )
    finally:
        await close(chats, memory)


async def test_a_run_of_turns_is_as_it_was_and_asks_the_model_nothing(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(reply)
    report, chats, memory = await compress(
        tmp_path, {"sessions": [MAY]}, stub, mode="turns"
    )
    try:
        assert stub.prompts == [] and report.compress is None
        data = report.to_json()
        assert "ingest" not in data and "compress" not in data
        assert "model_calls" not in data and "Encoded" not in report.summary()
        assert await memory.stores()[1].count_items() == 0
    finally:
        await close(chats, memory)


async def test_a_model_that_reports_tokens_has_them_in_the_report(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(reply, usage=(120, 30))
    report, chats, memory = await compress(tmp_path, {"sessions": [JUNE, MAY]}, stub)
    try:
        data = report.to_json()
        assert (data["input_tokens"], data["output_tokens"]) == (240, 60)
        assert data["model_calls"] == 2
    finally:
        await close(chats, memory)


async def test_each_item_is_embedded_when_an_embedding_model_is_set(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(reply)
    embedding = FakeEmbedding()
    report, chats, memory = await compress(
        tmp_path, {"sessions": [MAY]}, stub, embedding=embedding
    )
    try:
        assert report.model == "fake-embed"
        sent = [text for call in embedding.calls for text in call]
        assert f"Home city\n{PITTSBURGH}" in sent and f"Train book\n{BOOK}" in sent
        stored = await memory.stores()[1].fetch_items([1, 2])
        assert all(i.embed_model == "fake-embed" and i.vector for i in stored)
    finally:
        await close(chats, memory)


# ── The cache ──


async def test_a_second_run_over_the_same_chats_asks_the_model_nothing(
    tmp_path: Path,
) -> None:
    data = {"sessions": [JUNE, MAY]}
    first = StubEncoder(reply)
    one, chats, memory = await compress(tmp_path, data, first, sub="a")
    await close(chats, memory)
    assert one.stats.calls == 2 and len(first.prompts) == 2
    second = StubEncoder(reply)
    two, chats, memory = await compress(tmp_path, data, second, sub="b")
    try:
        assert second.prompts == []
        assert (two.stats.calls, two.stats.cache_hits) == (0, 2)
        assert two.items == one.items == 3
        assert two.to_json()["compress"]["may"]["windows"][0]["cached"] is True
        assert two.summary().endswith("(0 model calls, 2 from the cache).")
        titles = [i.text for i in await memory.stores()[1].fetch_items([1, 2, 3])]
        assert titles == [PITTSBURGH, BOOK, UTRECHT]
    finally:
        await close(chats, memory)
    assert (tmp_path / "shared-cache.jsonl").exists()


async def test_the_cache_is_in_the_memory_folder_unless_the_setting_names_one(
    tmp_path: Path,
) -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig()
    stub = StubEncoder(reply)
    _, chats, memory = await compress(
        tmp_path, {"sessions": [MAY]}, stub, cfg=cfg, sub="solo"
    )
    try:
        assert (tmp_path / "solo" / "memory" / "compress_cache.jsonl").exists()
    finally:
        await close(chats, memory)


# ── Refusals and failures ──


async def bare(
    tmp_path: Path, sub: str, provider: StubEncoder | None
) -> tuple[ingest.Clock, SessionStore, ChatMemory]:
    """Empty stores, and a chat model (or none): for a run that must not write."""
    folder = tmp_path / sub
    folder.mkdir()
    pin = ingest.Clock()
    chats = SessionStore(str(folder / "sessions.db"), writer="gateway", clock=pin)
    await chats.open()
    memory = ChatMemory(lambda: None, directory=str(folder / "memory"))
    memory.chat_provider = lambda: provider  # type: ignore[method-assign]
    return pin, chats, memory


async def test_without_a_chat_model_nothing_is_written(tmp_path: Path) -> None:
    pin, chats, memory = await bare(tmp_path, "none", None)
    try:
        with pytest.raises(IngestError) as caught:
            await ingest.run(
                memory, chats, pin, ingest.parse({"sessions": [MAY]}), mode="compress"
            )
        assert str(caught.value).startswith("ingest=compress needs a chat model: ")
        assert await chats.list_sessions(limit=9, kinds=None) == []
        assert await memory.stores()[1].count() == 0
    finally:
        await close(chats, memory)


async def test_a_mode_that_is_neither_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(reply)
    pin, chats, memory = await bare(tmp_path, "zip", stub)
    try:
        with pytest.raises(IngestError) as caught:
            await ingest.run(
                memory, chats, pin, ingest.parse({"sessions": [MAY]}), mode="zip"
            )
        assert str(caught.value) == "ingest must be one of turns, compress, got 'zip'"
        assert await chats.list_sessions(limit=9, kinds=None) == []
        assert stub.prompts == []
    finally:
        await close(chats, memory)


async def test_a_failing_model_leaves_the_turns_and_forces_the_plain_statements(
    tmp_path: Path,
) -> None:
    stub = StubEncoder("RAISE")
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY]}, stub)
    try:
        assert report.turns == 2 and report.docs == 2
        (window,) = report.to_json()["compress"]["may"]["windows"]
        assert window["failed"] is True
        # Every request was counted, the retry too.
        assert report.stats.calls == 2
        gate = report.to_json()["compress"]["may"]["gate"]
        # The forced item is the exchange's item, so the exchange counts as kept.
        assert gate == {"kept": [0], "ignored": [], "unmentioned": [1], "forced": [0]}
        (stored,) = await memory.stores()[1].fetch_items([1])
        assert (stored.kind, stored.encoder) == ("profile", "rules")
        assert "Pittsburgh" in stored.text
    finally:
        await close(chats, memory)


async def test_the_compress_run_ignores_a_held_gpu_lock(tmp_path: Path) -> None:
    lock = tmp_path / "gpu-lock"
    lock.write_text("holder: someone else\npurpose: busy\n")
    cfg = settings(tmp_path)
    cfg.scheduler.gpu_lock_file = str(lock)
    stub = StubEncoder(reply)
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY]}, stub, cfg=cfg)
    try:
        assert report.items == 2 and report.to_json()["model_calls"] == 1
        assert lock.read_text().startswith("holder: someone else")
    finally:
        await close(chats, memory)
