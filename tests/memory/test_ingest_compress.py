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
from localmelo.support.memory import ChatMemory, hippo_encode, ingest
from localmelo.support.memory.index import cleaned
from localmelo.support.memory.ingest import IngestError
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding
from localmelo.tests.memory.hippo_helpers import (
    CAP,
    CUT_OFF,
    StubEncoder,
    by_label,
    item,
    labels,
    says,
    turn,
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


async def test_a_duplicates_turns_stay_recalled_and_deleting_the_first_chat_loses_nothing(
    tmp_path: Path,
) -> None:
    again = {**MAY, "id": "again", "title": "Again", "date": "2023-05-27T10:00:00"}
    stub = StubEncoder(reply)
    report, chats, memory = await compress(tmp_path, {"sessions": [MAY, again]}, stub)
    try:
        index = memory.stores()[1]
        first, second = report.ids["may"], report.ids["again"]
        # The second chat says what the first did: its items are ignored, so no item
        # cites its turns, which stay recalled as turns beside the first chat's items.
        assert await index.count_items() == 2
        assert await index.encoded_turn(second) == 2
        assert await recalled_as_turns(index) == {(second, 1), (second, 2)}
        await memory.forget_session(first)
        # The items rested on the first chat alone: they go, and the second chat's
        # turns, still recalled, are all that is left, with no mark to re-encode them.
        assert await index.count_items() == 0
        assert await recalled_as_turns(index) == {(second, 1), (second, 2)}
        assert await index.encoded_turn(second) == 2
    finally:
        await close(chats, memory)


@pytest.mark.parametrize(
    ("bad", "mark"), [(None, 4), (2, 2), (0, None)], ids=["none", "third", "first"]
)
async def test_the_mark_does_not_pass_a_window_that_failed(
    tmp_path: Path, bad: int | None, mark: int | None
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

    cfg = settings(tmp_path, compress_at_chars=4000)
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
        lambda p: CUT_OFF if len(p) > 8000 else by_label(p),
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
        assert data["model_calls"] == 1 + (1 + 2) + (1 + 2) + 1
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
