"""``ingest.run(..., mode="compress")``: chats written as turns, then encoded into
memory items and routed (developer setting, #428).

The turns, their documents and the rule-path notes are written as ever; then each
chat is encoded window by window by the chat model and each item is routed against
the items kept, the oldest chat first. The report says, for each chat of the file,
its windows, the gate, the items stored and what became of each. Synthetic chats, a
stub chat model and temporary stores: no model server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    LocalBackendConfig,
    MemoryConfig,
)
from localmelo.support.memory import (
    ChatMemory,
    hippo_encode,
    ingest,
    note_judge,
    service,
)
from localmelo.support.memory.index import cleaned
from localmelo.support.memory.ingest import IngestError
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding
from localmelo.tests.memory.hippo_helpers import (
    CAP,
    CUT_OFF,
    Held,
    StubEncoder,
    at,
    by_label,
    is_judge,
    item,
    judging,
    labels,
    says,
    turn,
    verdict,
)

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
# A documented test card number (it passes the Luhn check) and a made-up password.
SECRET: dict[str, Any] = {
    "id": "pay",
    "title": "Card 4242 4242 4242 4242",
    "date": "2023-05-22T10:00:00",
    "turns": [
        {
            "role": "user",
            "text": "My password is hunter2. My card is 4242 4242 4242 4242. I like tea.",
        },
        {
            "role": "assistant",
            "text": "Noted. The card 4242424242424242 is saved. Tea it is.",
        },
    ],
}
TEA = "The user said they like drinking green tea in the evening."
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


def judge(request: dict[str, Any]) -> str:
    """The judge says the June item changes the old item it is shown, and no other
    item changes or repeats anything."""
    return verdict("changes" if "Utrecht" in request["new"]["text"] else "other", 1)


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
    stub = StubEncoder(judging(reply, judge))
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
        assert first.encoder == "3:stub-encoder" and first.embed_model == ""
        # Each chat is marked encoded up to its last turn, and every turn is cited by
        # an item, so recall reads none of them as a turn any more.
        assert (await index.encoded_turn(may), await index.encoded_turn(june)) == (2, 1)
        assert await index.ids(uncited=True) == set()
        # The item's time is that of the last turn it came from.
        stored = await chats.get_session(may)
        assert stored is not None and stored.title == "Moving"
        assert first.created_at < utrecht.created_at
    finally:
        await close(chats, memory)


async def test_the_report_names_windows_gate_items_and_routes_by_file_id(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(judging(reply, judge))
    report, chats, memory = await compress(tmp_path, {"sessions": [JUNE, MAY]}, stub)
    try:
        data = report.to_json()
        assert data["ingest"] == "compress"
        # Two windows and one judge request: only the June item has a close one.
        assert (data["model_calls"], data["cache_hits"]) == (3, 0)
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
        assert (window["split"], window["failed_parts"], window["part"]) == (0, 0, None)
        assert (data["split"], data["failed_parts"]) == (0, 0)
        assert window["chars"] > 0
        assert [(r["action"], r["item"], r["target"]) for r in may["routes"]] == [
            ("NEW", 1, None),
            ("NEW", 2, None),
        ]
        assert [r["title"] for r in may["routes"]] == ["Home city", "Train book"]
        (route,) = data["compress"]["june"]["routes"]
        assert (route["action"], route["item"], route["target"]) == ("UPDATE", 3, 1)
        assert route["reason"] == "judged a change of entry 1"
        assert data["compress"]["june"]["items"][0]["sources"] == [["june", 0]]
        json.dumps(data)  # all of it is plain JSON
        assert report.summary().endswith(
            "Encoded 3 items from 2 chats (3 model calls, 0 from the cache)."
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
        assert await memory.stores()[1].encoded_turn(report.ids["may"]) is None
        data = report.to_json()
        assert "ingest" not in data and "compress" not in data
        assert "model_calls" not in data and "Encoded" not in report.summary()
        assert await memory.stores()[1].count_items() == 0
    finally:
        await close(chats, memory)


async def test_a_model_that_reports_tokens_has_them_in_the_report(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(judging(reply, judge), usage=(120, 30))
    report, chats, memory = await compress(tmp_path, {"sessions": [JUNE, MAY]}, stub)
    try:
        data = report.to_json()
        # The judge's request is one of the three, and its tokens are counted too.
        assert (data["input_tokens"], data["output_tokens"]) == (360, 90)
        assert data["model_calls"] == 3
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


# ── Secrets ──


@pytest.mark.parametrize("parallel", [1, 2])
async def test_a_password_and_a_card_number_reach_the_encoder_as_the_index_keeps_them(
    tmp_path: Path, parallel: int
) -> None:
    stub = StubEncoder(says([item("Tea", TEA, "profile", [0])]))
    cfg = settings(tmp_path, compress_parallel=parallel)
    report, chats, memory = await compress(
        tmp_path, {"sessions": [SECRET]}, stub, cfg=cfg
    )
    try:
        (prompt,) = stub.prompts
        typed, answer = (message["text"] for message in SECRET["turns"])
        assert cleaned(typed) in prompt and cleaned(answer) in prompt
        # The chat's title is cleaned the same way.
        assert prompt.startswith(f"Chat title: {cleaned(SECRET['title'])}\n\n")
        assert "hunter2" not in prompt and "4242" not in prompt
        # The same text the chat's turn is indexed with.
        index = memory.stores()[1]
        (document,) = await index.fetch(await index.ids())
        assert cleaned(typed) in document.text and cleaned(answer) in document.text
        assert report.items == 1 and report.refused == 0
    finally:
        await close(chats, memory)


async def test_an_item_that_holds_a_secret_is_not_kept_and_the_report_counts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = hippo_encode.encode

    async def leaky(provider: Any, title: str, turns: Any, **kwargs: Any) -> Any:
        """What the encoder gives, and an item the checks let a secret through."""
        encoding = await real(provider, title, turns, **kwargs)
        encoding.pieces.append(
            hippo_encode.Piece(
                title="Payment card",
                text="The user said their card number is 4111 1111 1111 1111.",
                kind="profile",
                sources=(0,),
                created_at=turns[0].answered_at,
                encoder=encoding.pieces[0].encoder,
            )
        )
        return encoding

    monkeypatch.setattr(hippo_encode, "encode", leaky)
    caplog.set_level(logging.DEBUG)
    stub = StubEncoder(says([item("Tea", TEA, "profile", [0])]))
    embedding = FakeEmbedding()
    report, chats, memory = await compress(
        tmp_path, {"sessions": [SECRET]}, stub, embedding=embedding
    )
    try:
        index = memory.stores()[1]
        # The clean item is stored exactly as ever; the other is not stored at all.
        assert report.items == 1 and report.refused == 1
        assert await index.count_items() == 1
        (stored,) = await index.fetch_items([1])
        assert (stored.title, stored.text, stored.kind, stored.status) == (
            "Tea",
            TEA,
            "profile",
            "active",
        )
        assert stored.sources == ((report.ids["pay"], 1),)
        assert stored.encoder == "3:stub-encoder" and stored.embed_model == "fake-embed"
        # A refused item does not hold the chat's mark back.
        assert await index.encoded_turn(report.ids["pay"]) == 1
        # It was never embedded or routed, and nothing of it is in a log line (4111 1111
        # 1111 1111 is another documented test card number, not the chat's).
        assert all("4111" not in text for call in embedding.calls for text in call)
        data = report.to_json()
        assert data["refused"] == 1 and data["compress"]["pay"]["refused"] == 1
        assert [r["title"] for r in data["compress"]["pay"]["routes"]] == ["Tea"]
        assert "4111" not in json.dumps(data)
        assert report.summary().endswith(
            "Encoded 1 item from 1 chat (1 model call, 0 from the cache). "
            "1 item held a secret and was not kept."
        )
        assert "4111" not in caplog.text and "Payment card" not in caplog.text
        assert any("1 held a secret" in line for line in caplog.messages)
    finally:
        await close(chats, memory)


async def test_a_run_with_no_refused_item_says_nothing_of_it_in_the_summary(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(reply)
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY]}, stub)
    try:
        assert report.refused == 0 and report.to_json()["refused"] == 0
        assert "secret" not in report.summary()
    finally:
        await close(chats, memory)


@pytest.mark.parametrize(
    ("title", "text"),
    [
        ("Card 4242 4242 4242 4242", "The user said they have a card."),
        ("Login", "The user said their password is private."),
        ("Card", "The user said their card number is 4111 1111 1111 1111."),
    ],
    ids=["title only", "words only", "value in the text"],
)
async def test_compress_chat_refuses_what_it_is_handed_and_the_report_follows(
    tmp_path: Path, title: str, text: str
) -> None:
    # The encoding comes from outside, as the ingest's parallel path hands it over.
    _, chats, memory = await compress(tmp_path, {"sessions": [MAY]}, None, mode="turns")
    try:
        turns = [turn(0, "I live in Utrecht.", "Nice."), turn(1, "Bye.", "Bye!")]
        clean = hippo_encode.Piece("Tea", TEA, "profile", (0,), 1.0, "3:x")
        secret = hippo_encode.Piece(title, text, "profile", (1,), 2.0, "rules", True)
        encoding = hippo_encode.Encoding(
            pieces=[clean, secret], kept=[0, 1], forced=[1]
        )
        done = await memory.compress_chat("s2", "T", turns, encoding=encoding)
        index = memory.stores()[1]
        assert done.refused == 1 and [p.title for p, _ in done.routed] == ["Tea"]
        assert await index.count_items() == 1
        # The report reads what became of each exchange from the items kept.
        data = ingest._chat_report("s2", done)
        assert data["refused"] == 1
        assert data["gate"] == {
            "kept": [0],
            "ignored": [],
            "unmentioned": [1],
            "forced": [],
        }
        assert [r["title"] for r in data["routes"]] == ["Tea"]
        assert text not in json.dumps(data)
        # The turn's own document is cleaned already: the mark still reaches it.
        assert await index.encoded_turn("s2") == turns[-1].turn
        assert encoding.pieces == [clean, secret] and encoding.refused == 0
    finally:
        await close(chats, memory)


async def test_the_router_refuses_what_the_gate_before_it_let_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, chats, memory = await compress(tmp_path, {"sessions": [MAY]}, None, mode="turns")
    try:
        # The gate's check is switched off here; the router keeps its own.
        monkeypatch.setattr(hippo_encode, "holds_secret", lambda title, text: False)
        card = "The user said their card number is 4111 1111 1111 1111."
        pieces = [
            hippo_encode.Piece("Card", card, "profile", (0,), 1.0, "3:x"),
            hippo_encode.Piece("Tea", TEA, "profile", (1,), 2.0, "3:x"),
        ]
        turns = [turn(0, "Hi.", "Hello."), turn(1, "Tea.", "Nice.")]
        done = await memory.compress_chat(
            "s2", "T", turns, encoding=hippo_encode.Encoding(pieces=pieces)
        )
        assert done.refused == 1 and [p.title for p, _ in done.routed] == ["Tea"]
        assert await memory.stores()[1].count_items() == 1
    finally:
        await close(chats, memory)


async def recalled_as_turns(index: Any) -> set[tuple[str, int]]:
    """The ``(chat, turn)`` of the turns no item cites: recall reads them as turns."""
    docs = await index.fetch(await index.ids(uncited=True))
    return {(doc.session_id, doc.turn) for doc in docs}


async def test_a_turn_the_encoder_left_out_stays_recalled_after_its_window(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(says([item("Home city", PITTSBURGH, "profile", [0])]))
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY]}, stub)
    try:
        index = memory.stores()[1]
        may = report.ids["may"]
        # Both exchanges were read, and only the first has an item.
        assert await index.encoded_turn(may) == 2 and await index.count_items() == 1
        assert await recalled_as_turns(index) == {(may, 2)}
    finally:
        await close(chats, memory)


async def test_a_turn_whose_only_item_was_refused_stays_recalled(
    tmp_path: Path,
) -> None:
    card = "The user said their card number is 4111 1111 1111 1111."
    stub = StubEncoder(
        says(
            [
                item("Home city", PITTSBURGH, "profile", [0]),
                item("Card", card, "profile", [1]),
            ]
        )
    )
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY]}, stub)
    try:
        index = memory.stores()[1]
        may = report.ids["may"]
        assert report.refused == 1 and await index.count_items() == 1
        assert await index.encoded_turn(may) == 2
        assert await recalled_as_turns(index) == {(may, 2)}
    finally:
        await close(chats, memory)


async def test_a_duplicate_is_cited_by_the_item_it_repeats_and_deleting_the_first_chat_loses_nothing(
    tmp_path: Path,
) -> None:
    again = {**MAY, "id": "again", "title": "Again", "date": "2023-05-27T10:00:00"}
    stub = StubEncoder(reply)
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY, again]}, stub)
    try:
        index = memory.stores()[1]
        first, second = report.ids["may"], report.ids["again"]
        # The second chat says what the first did: its items are ignored, and the
        # items they repeat cite its turns too, so no turn is recalled beside them.
        assert await index.count_items() == 2
        assert await index.encoded_turn(second) == 2
        assert [i.sources for i in await index.fetch_items([1, 2])] == [
            ((first, 1), (second, 1)),
            ((first, 2), (second, 2)),
        ]
        assert await recalled_as_turns(index) == set()
        await memory.forget_session(first)
        # Both items rested on the first chat too: they go, and the second chat's
        # turns are recalled as turns, with no mark to encode them again.
        assert await index.count_items() == 0
        assert await recalled_as_turns(index) == {(second, 1), (second, 2)}
        assert await index.encoded_turn(second) == 2
    finally:
        await close(chats, memory)


@pytest.mark.parametrize(
    ("bad", "mark"), [(None, 4), (2, 2), (0, None)], ids=["none", "third", "first"]
)
async def test_the_mark_does_not_pass_a_window_that_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad: int | None,
    mark: int | None,
) -> None:
    # One exchange to a window: four long exchanges, the window of exchange *bad*
    # fails. The mark stops before it, so its turn is read again; the items of the
    # windows that went through are kept, and the only turn recall still reads is
    # the one no item cites, the failed window's.
    data = {
        "id": "long",
        "date": "2023-05-20T13:40:00",
        "turns": [
            part
            for n in range(4)
            for part in (
                {"role": "user", "text": f"Question {n}?"},
                {"role": "assistant", "text": f"Answer {n}. " + "x" * 2500},
            )
        ],
    }

    def reply(prompt: str) -> str:
        return "RAISE" if bad in labels(prompt) else by_label(prompt)

    monkeypatch.setattr(service, "COMPRESS_AT_CHARS", 4000)  # one exchange a window
    cfg = settings(tmp_path)
    report, chats, memory = await compress(
        tmp_path, {"sessions": [data]}, StubEncoder(reply), cfg=cfg
    )
    try:
        index = memory.stores()[1]
        assert await index.encoded_turn(report.ids["long"]) == mark
        windows = report.to_json()["compress"]["long"]["windows"]
        assert [w["failed"] for w in windows] == [bad == n for n in range(4)]
        failed = {
            doc.id
            for doc in await index.fetch(await index.ids())
            if bad is not None and doc.turn == bad + 1
        }
        assert await index.ids(uncited=True) == failed
    finally:
        await close(chats, memory)


# ── The cache ──


async def test_a_second_run_over_the_same_chats_asks_the_model_nothing(
    tmp_path: Path,
) -> None:
    data = {"sessions": [JUNE, MAY]}
    first = StubEncoder(judging(reply, judge))
    one, chats, memory = await compress(tmp_path, data, first, sub="a")
    await close(chats, memory)
    assert one.stats.calls == 3 and len(first.prompts) == 3  # two windows, one judge
    second = StubEncoder(judging(reply, judge))
    two, chats, memory = await compress(tmp_path, data, second, sub="b")
    try:
        assert second.prompts == []  # neither a window nor the judge is asked again
        assert (two.stats.calls, two.stats.cache_hits) == (0, 3)
        assert two.judge == one.judge and two.judge.changes == 1
        assert two.items == one.items == 3
        assert two.to_json()["compress"]["may"]["windows"][0]["cached"] is True
        assert two.summary().endswith("(0 model calls, 3 from the cache).")
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


async def test_a_very_long_exchange_is_read_in_parts_and_a_stretch_that_runs_out_is_split(
    tmp_path: Path,
) -> None:
    long_chat = {
        "id": "long",
        "title": "Long",
        "date": "2023-07-01T09:00:00",
        "turns": [
            {"role": "user", "text": "Short question."},
            {"role": "assistant", "text": "Short answer."},
            {"role": "user", "text": "Read the whole thing."},
            {
                "role": "assistant",
                "text": "\n\n".join(chr(97 + n) * 2900 for n in range(10)),
            },
        ],
    }
    # Every part is a window of its own, at most 12000 characters; the long exchange
    # is the second of the chat, so its parts name exchange 1. A part that is more
    # than 8000 characters runs out of room, so it is split in two and read in halves.
    stub = StubEncoder(
        judging(lambda p: CUT_OFF if len(p) > 8000 else by_label(p)),
        usage=lambda p: (len(p) // 4, CAP if len(p) > 8000 else 100),
    )
    report, chats, memory = await compress(tmp_path, {"sessions": [long_chat]}, stub)
    try:
        data = report.to_json()
        rows = data["compress"]["long"]["windows"]
        assert [w["part"] for w in rows] == [None, [1, 3], [2, 3], [3, 3]]
        assert [w["exchanges"] for w in rows] == [1, 1, 1, 1]
        assert not any(w["cut"] or w["failed"] for w in rows)
        # The first two parts are over the room (about 11000 characters) and split
        # in halves; the last is the rest of the exchange and is read whole.
        assert [w["split"] for w in rows] == [0, 2, 2, 0]
        assert (data["split"], data["failed_parts"]) == (4, 0)
        # The windows' requests, and one more for each item the judge was asked about.
        assert data["judge"]["judged"] == 2
        assert (
            data["model_calls"] == 1 + (1 + 2) + (1 + 2) + 1 + data["judge"]["judged"]
        )
        assert data["compress"]["long"]["gate"]["unmentioned"] == []
        json.dumps(data)
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


# ── The chat backend reaches the encoder ──


@pytest.mark.parametrize(
    ("chat_backend", "url", "expected"),
    [
        ("vllm", None, "vllm"),
        ("ollama", None, "ollama"),
        ("mlc", None, "mlc"),
        ("openai", None, "openai"),
        ("vllm", "http://127.0.0.1:9/v1", None),  # a server named by its URL
    ],
)
async def test_the_backend_of_the_chat_model_is_what_the_encoder_is_told(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chat_backend: str,
    url: str | None,
    expected: str | None,
) -> None:
    monkeypatch.setenv("MELO_TEST_CHAT_KEY", "k")
    local = LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m")
    cfg = Config(chat_backend=chat_backend, embedding_backend="none")
    setattr(cfg, chat_backend, local)
    if chat_backend == "openai":
        cfg.openai = CloudVendorConfig(api_key_env="MELO_TEST_CHAT_KEY", chat_model="m")
    cfg.memory = MemoryConfig(compress_cache=str(tmp_path / "cache.jsonl"))
    seen: list[dict[str, Any]] = []
    real = hippo_encode.encode

    async def spy(provider: Any, title: str, turns: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return await real(StubEncoder(says()), title, turns, **kwargs)

    monkeypatch.setattr(hippo_encode, "encode", spy)
    memory = ChatMemory(
        lambda: cfg,
        directory=str(tmp_path / "memory"),
        chat_url=url,
        chat_model="m" if url else None,
    )
    try:
        await memory.compress_chat("s1", "T", [turn(0, "Hello.", "Hi.")])
        assert [kwargs["backend"] for kwargs in seen] == [expected]
    finally:
        await memory.aclose()


# ── The judge: repeats join, changes replace ──

CHOIR = "The user's choir rehearsal is on Mondays at 19:00."
CHOIR_AGAIN = "The user said their choir rehearsal is on Mondays at 19:00."
CHOIR_MOVED = "The user said the choir rehearsal moved to Wednesdays."
CHOIR_MOVED_AGAIN = "The user said the choir rehearses on Wednesdays now."
# What the person typed in each chat, and the item the encoder writes from it, under a
# different kind each time.
CHOIR_CHATS = [
    ("A", "2023-03-01", "My choir rehearses on Mondays at 19:00.", "profile", CHOIR),
    ("B", "2023-03-08", "Our choir meets every Monday night at 19:00.", "semantic", CHOIR_AGAIN),
    ("C", "2023-03-15", "Update: the choir now rehearses on Wednesdays.", "episodic", CHOIR_MOVED),
    ("D", "2023-03-22", "Reminder, choir practice is on Wednesdays.", "profile", CHOIR_MOVED_AGAIN),
]  # fmt: skip


def choir_data() -> dict[str, Any]:
    return {
        "sessions": [
            {
                "id": label,
                "title": f"Choir {label}",
                "date": date,
                "turns": [
                    {"role": "user", "text": typed},
                    {"role": "assistant", "text": "Noted."},
                ],
            }
            for label, date, typed, _, _ in CHOIR_CHATS
        ]
    }


def choir_items(prompt: str) -> str:
    """The item each chat gives, found by what the person typed."""
    for _, _, typed, kind, text in CHOIR_CHATS:
        if typed in prompt:
            return says([item("Choir", text, kind, [0])])
    return says()


def choir_judge(request: dict[str, Any]) -> str:
    """A model that follows the prompt: the second chat repeats the first, the third
    changes it, the fourth repeats the third; nothing else is related."""
    table = {
        CHOIR_AGAIN: ("same", CHOIR),
        CHOIR_MOVED: ("changes", CHOIR),
        CHOIR_MOVED_AGAIN: ("same", CHOIR_MOVED),
    }
    kind, old = table.get(request["new"]["text"], ("other", ""))
    numbers = {o["text"]: o["item"] for o in request["old"]}
    return verdict(kind, numbers[old]) if old in numbers else verdict()


async def test_a_repeat_joins_its_item_a_change_replaces_it_and_a_third_chat_joins_the_change(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(judging(choir_items, choir_judge))
    report, chats, memory = await compress(tmp_path, choir_data(), stub)
    try:
        _, index = memory.stores()
        first, moved = await index.fetch_items([1, 2])
        assert [first.text, moved.text] == [CHOIR, CHOIR_MOVED]
        assert (first.status, first.replaced_by) == ("replaced", 2)
        assert (moved.status, moved.kind) == ("active", "episodic")
        # The old item cites the turn of the chat that repeated it, the new one the
        # turn of the chat that repeated the change: no fact is shown twice.
        a, b, c, d = (report.ids[label] for label in "ABCD")
        assert first.sources == ((a, 1), (b, 1)) and moved.sources == ((c, 1), (d, 1))
        assert await index.count_items() == 2
        data = report.to_json()
        actions = [
            r["action"] for label in "ABCD" for r in data["compress"][label]["routes"]
        ]
        assert actions == ["NEW", "IGNORE", "UPDATE", "IGNORE"]
        wanted = {"judged": 3, "same": 2, "changes": 1, "unsupported": 0}
        assert data["judge"] == {**wanted, "other": 0, "failed": 0, "fallbacks": 0}
        # Each chat reports its own counts, and the report holds their sum.
        assert [data["compress"][label]["judge"]["judged"] for label in "ABCD"] == [
            0,
            1,
            1,
            1,
        ]
        assert report.stats.calls == 4 + 3  # four windows, three judge requests
    finally:
        await close(chats, memory)


@pytest.mark.parametrize("reply", [verdict(), "RAISE"], ids=["other", "no answer"])
async def test_a_model_that_says_other_or_fails_links_two_facts_that_share_a_verb(
    tmp_path: Path, reply: str
) -> None:
    data = {
        "sessions": [
            {
                "id": label,
                "title": label,
                "date": date,
                "turns": [
                    {"role": "user", "text": typed},
                    {"role": "assistant", "text": "Noted."},
                ],
            }
            for label, date, typed in (
                ("rabbits", "2023-03-01", "I live with three rabbits."),
                ("harbour", "2023-03-08", "I live in a small harbour town."),
            )
        ]
    }

    def encoder(prompt: str) -> str:
        found = "three rabbits" in prompt
        text = (
            "The user lives with three rabbits."
            if found
            else "The user lives in a small harbour town."
        )
        return says([item("Home", text, "profile", [0])])

    stub = StubEncoder(lambda p: reply if is_judge(p) else encoder(p))
    report, chats, memory = await compress(tmp_path, data, stub)
    try:
        _, index = memory.stores()
        first, second = await index.fetch_items([1, 2])
        # Linked, and neither replaced: a shared verb is not a change.
        assert (first.status, second.status) == ("active", "active")
        assert (first.related, second.related) == ((2,), (1,))
        routes = report.to_json()["compress"]["harbour"]["routes"]
        assert [(r["action"], r["target"]) for r in routes] == [("RELATED", 1)]
        assert report.judge.judged == 1 and report.judge.changes == 0
    finally:
        await close(chats, memory)


@pytest.mark.parametrize(
    "answer",
    [
        "You live in Utrecht now, as you told me.",
        "From the page: Utrecht is a canal city.",
    ],
    ids=["melo's reply", "a fetched page"],
)
async def test_a_change_whose_new_value_the_person_never_typed_links_and_does_not_replace(
    tmp_path: Path, answer: str
) -> None:
    june = {
        "id": "june",
        "title": "Later",
        "date": "2023-06-01T09:00:00",
        "turns": [
            {"role": "user", "text": "Where is that again?"},
            {"role": "assistant", "text": answer},
        ],
    }
    stub = StubEncoder(
        judging(
            lambda p: (
                says([item("Home", UTRECHT, "profile", [0])])
                if "Where is that again" in p
                else reply(p)
            ),
            lambda request: verdict("changes", 1),
        )
    )
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY, june]}, stub)
    try:
        first, *_, last = await memory.stores()[1].fetch_items([1, 2, 3])
        assert (first.status, last.status) == ("active", "active")
        (route,) = report.to_json()["compress"]["june"]["routes"]
        assert (route["action"], route["target"]) == ("RELATED", 1)
        assert report.judge.changes == 1 and report.judge.unsupported == 1
        # The request carried what the person typed, never melo's reply or the page.
        (request,) = [json.loads(p) for p in stub.prompts if is_judge(p)]
        assert request["typed"] == ["Where is that again?"]
        assert "canal" not in json.dumps(request) and "told me" not in json.dumps(
            request
        )
    finally:
        await close(chats, memory)


async def test_the_request_holds_no_secret_and_the_log_holds_no_item_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    june = {
        "id": "june",
        "title": "Later",
        "date": "2023-06-01T09:00:00",
        "turns": [
            {
                "role": "user",
                "text": "I live in Utrecht now. My password is hunter2. "
                "My card is 4242 4242 4242 4242.",
            },
            {"role": "assistant", "text": "Noted."},
        ],
    }
    caplog.set_level(logging.DEBUG)
    stub = StubEncoder(
        judging(
            lambda p: (
                says([item("Home", UTRECHT, "profile", [0])])
                if "Utrecht" in p
                else reply(p)
            ),
            lambda request: verdict("changes", 1),
        )
    )
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY, june]}, stub)
    try:
        (request,) = [p for p in stub.prompts if is_judge(p)]
        assert "hunter2" not in request and "4242" not in request
        assert report.judge.changes == 1 and report.judge.unsupported == 0
        # What memory logs (not the database driver's own lines): the counts only.
        logged = [
            r.getMessage()
            for r in caplog.records
            if r.name.startswith("localmelo.support.memory")
        ]
        assert any("judged" in line for line in logged)
        for word in ("Utrecht", "Pittsburgh", "hunter2", "4242", "live in"):
            assert not any(word in line for line in logged), word
    finally:
        await close(chats, memory)


# Two chats of one person, a place and the day each was said; compress_chat is called
# on them directly (the way the idle encoder does), a chat at a time.
OLD_TURNS = [turn(0, "I live in Pittsburgh.", "Noted.", at("2023-05-20"))]
NEW_TURNS = [turn(0, "I live in Utrecht now.", "Noted.", at("2023-06-01"))]


def home(prompt: str) -> str:
    return says(
        [item("Home", UTRECHT if "Utrecht" in prompt else PITTSBURGH, "profile")]
    )


@pytest.mark.parametrize("says_", ["same", "changes"])
async def test_a_forget_and_a_delete_do_not_wait_for_the_judge_and_a_deleted_target_loses_nothing(
    tmp_path: Path, says_: str
) -> None:
    held = Held(judging(home, lambda request: verdict(says_, 1)))
    _, chats, memory = await bare(tmp_path, "held", held)
    try:
        _, index = memory.stores()
        await memory.compress_chat("old", "Old", OLD_TURNS)
        assert await index.count_items() == 1 and not held.asked.is_set()
        task = asyncio.create_task(memory.compress_chat("new", "New", NEW_TURNS))
        await asyncio.wait_for(held.asked.wait(), 5)  # the judge is with the model
        # The lock is free: the target is deleted, and the call returns at once.
        assert await asyncio.wait_for(memory.delete_item(1), 2) is True
        held.release.set()
        out = await asyncio.wait_for(task, 5)
        # The answer rested on an item that is gone: the rules route the item, and it
        # is stored, not lost as a repeat of nothing.
        (route,) = [r for _, r in out.routed]
        assert (route.action, route.item) == ("NEW", 2)
        assert out.judge.fallbacks == 1 and out.judge.judged == 1
        assert [i.id for i in await index.active_items()] == [2]
        assert await index.encoded_turn("new") == 1
    finally:
        await close(chats, memory)


async def test_a_forget_while_the_judge_is_asked_keeps_the_chat_out(
    tmp_path: Path,
) -> None:
    held = Held(judging(home, lambda request: verdict("same", 1)))
    _, chats, memory = await bare(tmp_path, "forget", held)
    try:
        _, index = memory.stores()
        await memory.compress_chat("old", "Old", OLD_TURNS)
        since = await index.forgotten_at()

        async def unchanged() -> bool:
            return await index.forgotten_at() == since

        task = asyncio.create_task(
            memory.compress_chat("new", "New", NEW_TURNS, guard=unchanged)
        )
        await asyncio.wait_for(held.asked.wait(), 5)
        await asyncio.wait_for(memory.forget_all(), 2)  # not held up by the request
        held.release.set()
        out = await asyncio.wait_for(task, 5)
        assert out.forgotten and out.routed == []
        assert await index.count_items() == 0
        assert await index.encoded_turn("new") is None
    finally:
        await close(chats, memory)


async def test_a_repeat_a_change_and_a_repeat_in_one_call_end_with_one_current_item(
    tmp_path: Path,
) -> None:
    pieces = [
        ("My choir rehearses on Mondays at 19:00.", "profile", CHOIR),
        ("Our choir meets every Monday night at 19:00.", "semantic", CHOIR_AGAIN),
        ("Update: the choir now rehearses on Wednesdays.", "episodic", CHOIR_MOVED),
        ("Reminder, choir practice is on Wednesdays.", "profile", CHOIR_MOVED_AGAIN),
    ]
    stub = StubEncoder(
        judging(
            says([item("Choir", t, k, [n]) for n, (_, k, t) in enumerate(pieces)]),
            choir_judge,
        )
    )
    _, chats, memory = await bare(tmp_path, "one", stub)
    try:
        _, index = memory.stores()
        turns = [turn(n, typed, "Noted.") for n, (typed, _, _) in enumerate(pieces)]
        out = await memory.compress_chat("choir", "Choir", turns)
        assert [r.action for _, r in out.routed] == [
            "NEW",
            "IGNORE",
            "UPDATE",
            "IGNORE",
        ]
        (current,) = await index.active_items()
        assert current.text == CHOIR_MOVED
        assert current.sources == (("choir", 3), ("choir", 4))
        (old,) = await index.fetch_items([1])
        assert (old.status, old.sources) == ("replaced", (("choir", 1), ("choir", 2)))
        # The last request was made against what the earlier items had become: the
        # old fact, already replaced, is not among the items it is shown.
        last = [json.loads(p) for p in stub.prompts if is_judge(p)][-1]
        assert [o["text"] for o in last["old"]] == [CHOIR_MOVED]
        assert out.judge.to_json()["judged"] == 3
    finally:
        await close(chats, memory)


async def test_the_newer_fact_stays_current_when_the_older_chat_is_encoded_last(
    tmp_path: Path,
) -> None:
    stub = StubEncoder(judging(home, lambda request: verdict("changes", 1)))
    _, chats, memory = await bare(tmp_path, "late", stub)
    try:
        _, index = memory.stores()
        await memory.compress_chat("new", "New", NEW_TURNS)
        out = await memory.compress_chat("old", "Old", OLD_TURNS)
        (route,) = [r for _, r in out.routed]
        assert (route.action, route.target) == ("UPDATE", 1)
        assert "older than entry 1, kept as replaced" in route.reason
        utrecht, pittsburgh = await index.fetch_items([1, 2])
        assert (utrecht.text, utrecht.status) == (UTRECHT, "active")
        assert (pittsburgh.status, pittsburgh.replaced_by) == ("replaced", 1)
    finally:
        await close(chats, memory)


async def test_a_judge_request_that_never_answers_routes_by_the_rules_and_the_chat_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(note_judge, "JUDGE_TIMEOUT_S", 0.05)
    held = Held(home)  # no release: the judge request is never answered
    _, chats, memory = await bare(tmp_path, "slow", held)
    try:
        _, index = memory.stores()
        await memory.compress_chat("old", "Old", OLD_TURNS)
        out = await asyncio.wait_for(memory.compress_chat("new", "New", NEW_TURNS), 5)
        (route,) = [r for _, r in out.routed]
        assert route.action == "RELATED" and out.judge.failed == 1
        assert [i.id for i in await index.active_items()] == [1, 2]
        assert await index.encoded_turn("new") == 1
    finally:
        await close(chats, memory)
