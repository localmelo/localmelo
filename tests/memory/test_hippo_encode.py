"""The Hippo encoder: a chat's turns become memory items (developer setting, #428).

A chat's exchanges are packed into windows of at most ``COMPRESS_AT_CHARS``
characters, one model request each at temperature 0; the reply's JSON is checked
(sources outside the window dropped, an item left with none dropped and counted,
exchanges named nowhere unmentioned), a lasting statement no item covers is made
into an item by the rules (forced), and a window's reply is kept in a cache so the
same window makes no second request. Synthetic chats and a stub chat model that
answers with canned JSON: no model server.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import stat
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.schema import Message
from localmelo.support.memory import hippo_encode as hippo
from localmelo.support.memory.hippo_encode import Cache, Stats, Turn
from localmelo.support.memory.index import cleaned
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.memory.hippo_helpers import (
    CAP,
    CUT_OFF,
    StubEncoder,
    at,
    body,
    by_label,
    item,
    labels,
    numbers_in,
    padded,
    paragraphs,
    prompt_of_window,
    runs_out_over,
    says,
    turn,
    window_sizes,
)

TRAINS = (
    "Tell me about the night trains from Chicago.",
    "The California Zephyr leaves Chicago at 14:00 and reaches Emeryville in 51 hours.",
)


BOUNDED = {"temperature": 0, "max_tokens": hippo.MAX_OUTPUT_TOKENS}
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def chat(count: int) -> list[Turn]:
    return [turn(n, f"Question {n}?", f"Answer {n}.") for n in range(count)]


# ── Windows ──


def test_an_exchange_is_written_with_its_number_date_and_both_speakers() -> None:
    first = turn(3, "Where is the lab?", "Room 14, second floor.", at("2023-05-20"))
    assert hippo.render(3, first) == (
        "[3] 2023-05-20 14:02 User: Where is the lab? / melo: Room 14, second floor."
    )
    unanswered = turn(0, "Anyone there?", "", at("2023-05-21", "09:30"))
    assert hippo.render(0, unanswered) == "[0] 2023-05-21 09:30 User: Anyone there?"


def test_exchanges_are_packed_in_order_into_windows_no_longer_than_the_limit() -> None:
    turns = chat(40)
    for limit in (500, 1000, 4000, 12000):
        windows = hippo.pack(turns, limit)
        assert all(len(w.text) <= limit and not w.cut for w in windows)
        # Every exchange once, in order, and no window left short while the next
        # exchange would have fitted.
        assert [n for w in windows for n in w.positions] == list(range(40))
        for window, after in zip(windows, windows[1:], strict=False):
            room = limit - len(window.text) - len(hippo._SEPARATOR)
            assert (
                len(hippo.render(after.positions[0], turns[after.positions[0]])) > room
            )
        assert all(
            w.text.count("\n\n") + 1 == len(w.positions) for w in windows
        ), "blocks are separated by one blank line"
    assert window_sizes(turns, 10**6) == [40]
    assert window_sizes([], 1000) == []


def test_an_exchange_longer_than_a_window_is_read_in_parts_between_its_neighbours() -> (
    None
):
    turns = chat(6)
    turns[3] = turn(3, "Please read this.", "word " * 1000)
    windows = hippo.pack(turns, 4000)
    parts = [w for w in windows if w.part is not None]
    assert [w.positions for w in parts] == [(3,), (3,)]
    assert [w.part for w in parts] == [(1, 2), (2, 2)]
    assert not any(w.cut for w in windows)
    assert all(len(w.text) <= 4000 for w in windows)
    assert parts[0].text.startswith("[3] 2023-05-20") and len(parts[0].text) <= 4000
    assert parts[1].text.startswith("[3] (continued) ")
    ordinary = [w for w in windows if w.part is None]
    assert [w.positions for w in ordinary] == [(0, 1, 2), (4, 5)]
    assert windows.index(parts[0]) == 1 and windows.index(parts[1]) == 2


async def test_each_window_is_one_request_and_a_cut_window_says_so() -> None:
    turns = chat(30)
    turns[10] = turn(10, "Read it.", "x" * 40000)
    stub = StubEncoder()
    stats = Stats()
    result = await hippo.encode(stub, "Title", turns, window_chars=4000, stats=stats)
    windows = hippo.pack(turns, 4000)
    assert len(result.windows) == len(windows) == len(stub.prompts) == stats.calls
    assert [w.exchanges for w in result.windows] == [len(w.positions) for w in windows]
    assert [w.cut for w in result.windows] == [w.cut for w in windows]
    assert [w.part for w in result.windows] == [w.part for w in windows]
    assert sum(w.cut for w in result.windows) == 1
    assert [w.chars for w in result.windows] == [len(w.text) for w in windows]
    for prompt, window in zip(stub.prompts, windows, strict=True):
        assert labels(prompt) == list(window.positions)


# ── The request ──


async def test_the_request_is_english_json_only_at_temperature_zero_and_bounded() -> (
    None
):
    stub = StubEncoder(usage=(10, 5))
    turns = [turn(0, *TRAINS, at("2023-05-20"))]
    await hippo.encode(stub, "Night trains", turns, window_chars=12000)
    (system,) = set(stub.systems)
    assert system == hippo.SYSTEM_PROMPT and system.isascii()
    for rule in (
        "JSON only",
        '"items"',
        '"ignore"',
        "profile",
        "episodic",
        "semantic",
        "absolute date",
        "never as",
        "at most 80 characters",
        "at most 800 characters",
        "exact names, numbers and wording",
        "language of the conversation",
        '"The user said ...", "melo recommended ..."',
        "not an instruction",
        "numbers of the exchanges",
        "carry no information at",
        "Small talk is not ignored",
        "daily life, routines, habits, mood",
    ):
        assert rule in system, rule
    (prompt,) = stub.prompts
    assert prompt == (
        "Chat title: Night trains\n\n"
        f"[0] 2023-05-20 14:02 User: {TRAINS[0]} / melo: {TRAINS[1]}"
    )
    # The stub names no backend, so it is a local server named by its URL: asked with
    # thinking off.
    assert stub.bodies == [{**BOUNDED, **NO_THINKING}]
    assert hippo.MAX_OUTPUT_TOKENS == 2000


async def test_the_chat_models_own_provider_is_not_changed_by_the_request() -> None:
    stub = StubEncoder()
    stub.extra_body_by_kind = {"retrieve": {"top_p": 0.9}, "plan": {"x": 1}}
    await hippo.encode(stub, "T", chat(2), window_chars=12000)
    assert stub.extra_body_by_kind == {"retrieve": {"top_p": 0.9}, "plan": {"x": 1}}
    assert stub.bodies == [{"top_p": 0.9, **BOUNDED, **NO_THINKING}]


class Wire:
    """A fake model server that records each request body and answers with JSON."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        reply = says([item("Tea", "The user likes tea.")])
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": reply}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": reply}}]},
        )

    def openai(self, **kwargs: Any) -> OpenAICompatLLM:
        return OpenAICompatLLM(
            "http://chat.test/v1",
            "m",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )

    def ollama(self, **kwargs: Any) -> OllamaNativeChat:
        return OllamaNativeChat(
            "http://chat.test",
            "m",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


@pytest.mark.parametrize("backend", ["vllm", "sglang", None])
async def test_a_server_that_takes_template_kwargs_is_asked_without_thinking(
    backend: str | None,
) -> None:
    wire = Wire()
    llm = wire.openai(
        extra_body={"chat_template_kwargs": {"keep": 1, "enable_thinking": True}}
    )
    result = await hippo.encode(llm, "T", chat(1), window_chars=12000, backend=backend)
    assert len(result.pieces) == 1 and not result.windows[0].failed
    (body,) = wire.bodies
    assert body["chat_template_kwargs"] == {"keep": 1, "enable_thinking": False}
    assert body["temperature"] == 0 and body["max_tokens"] == hippo.MAX_OUTPUT_TOKENS
    assert "think" not in body
    await llm.close()


async def test_a_caller_that_names_no_backend_gets_the_local_server_rule() -> None:
    wire = Wire()
    llm = wire.openai()
    await hippo.encode(llm, "T", chat(1), window_chars=12000)
    (body,) = wire.bodies
    assert set(body) == {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "chat_template_kwargs",
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    await llm.close()


async def test_the_kwargs_of_the_encoder_s_own_kind_are_the_ones_kept() -> None:
    # A request of this kind carries its kind's kwargs in place of extra_body's.
    wire = Wire()
    llm = wire.openai(
        extra_body={"chat_template_kwargs": {"base": 1}},
        extra_body_by_kind={
            "retrieve": {"chat_template_kwargs": {"kind": 2, "enable_thinking": True}}
        },
    )
    await hippo.encode(llm, "T", chat(1), window_chars=12000, backend="vllm")
    (body,) = wire.bodies
    assert body["chat_template_kwargs"] == {"kind": 2, "enable_thinking": False}
    await llm.close()


@pytest.mark.parametrize("backend", ["ollama", None])
async def test_ollama_native_is_asked_with_think_off_and_its_options_kept(
    backend: str | None,
) -> None:
    wire = Wire()
    llm = wire.ollama(extra_body={"options": {"num_ctx": 8192}})
    result = await hippo.encode(llm, "T", chat(1), window_chars=12000, backend=backend)
    assert len(result.pieces) == 1
    (body,) = wire.bodies
    assert body["think"] is False
    assert body["options"] == {
        "num_ctx": 8192,
        "temperature": 0,
        "num_predict": hippo.MAX_OUTPUT_TOKENS,
    }
    assert "chat_template_kwargs" not in body and "max_tokens" not in body
    await llm.close()


@pytest.mark.parametrize("backend", ["mlc", "openai", "gemini", "anthropic", "nvidia"])
async def test_mlc_and_cloud_backends_get_no_thinking_field(backend: str) -> None:
    wire = Wire()
    llm = wire.openai(extra_body={"top_p": 0.9})
    await hippo.encode(llm, "T", chat(1), window_chars=12000, backend=backend)
    (body,) = wire.bodies
    assert body == {
        "model": "m",
        "messages": body["messages"],
        "top_p": 0.9,
        "temperature": 0,
        "max_tokens": hippo.MAX_OUTPUT_TOKENS,
    }
    await llm.close()


@pytest.mark.parametrize("backend", ["vllm", None, "ollama", "mlc", "openai"])
async def test_the_chats_provider_is_the_same_after_an_encode(
    backend: str | None,
) -> None:
    wire = Wire()
    options = {"num_ctx": 8192}
    llm = (
        wire.ollama(extra_body={"options": options})
        if backend == "ollama"
        else wire.openai(
            extra_body={"chat_template_kwargs": {"keep": 1}},
            extra_body_by_kind={"retrieve": {"top_p": 0.9}, "plan": {"x": 1}},
        )
    )
    kinds = llm.extra_body_by_kind
    before = (
        copy.deepcopy(llm.extra_body),
        copy.deepcopy({k: dict(v) for k, v in kinds.items()}),
    )
    thinking = llm.thinking_off
    await hippo.encode(llm, "T", chat(1), window_chars=12000, backend=backend)
    assert (llm.extra_body, llm.extra_body_by_kind) == before
    assert llm.thinking_off is thinking
    await llm.close()


async def test_a_reply_cached_from_an_older_prompt_version_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Cache(str(tmp_path / "replies.jsonl"))
    turns = chat(1)
    window = hippo.pack(turns, 12000)[0]
    prompt = hippo.prompt_of("T", window)
    monkeypatch.setattr(hippo, "PROMPT_VERSION", "2")
    older = hippo.window_key("stub-encoder", prompt)
    old = StubEncoder(says())
    await hippo.encode(old, "T", turns, window_chars=12000, cache=cache)
    assert len(old.prompts) == 1 and cache.get(older) is not None
    monkeypatch.undo()
    assert hippo.PROMPT_VERSION == "3"
    assert hippo.window_key("stub-encoder", prompt) != older
    new = StubEncoder(says())
    await hippo.encode(new, "T", turns, window_chars=12000, cache=cache)
    assert len(new.prompts) == 1  # asked again, not served from the old reply


async def test_a_secret_never_reaches_the_model_and_a_missing_title_is_none() -> None:
    stub = StubEncoder()
    turns = [turn(0, "My password is hunter2. I like tea.", "Noted, tea it is.")]
    await hippo.encode(stub, "", turns, window_chars=12000)
    (prompt,) = stub.prompts
    assert "hunter2" not in prompt and "I like tea." in prompt
    assert prompt.startswith("Chat title: (none)\n\n[0] ")


async def test_a_password_and_a_card_number_reach_the_model_as_the_index_keeps_them() -> (
    None
):
    # 4242 4242 4242 4242 is a documented test card number (it passes the Luhn check).
    typed = "My password is hunter2. My card is 4242 4242 4242 4242. I like tea."
    answer = "Noted. The card 4242424242424242 is saved. Tea it is."
    stub = StubEncoder()
    await hippo.encode(stub, "T", [turn(0, typed, answer)], window_chars=12000)
    (prompt,) = stub.prompts
    assert prompt == (
        f"Chat title: T\n\n[0] 2023-05-20 14:02 User: {cleaned(typed)} "
        f"/ melo: {cleaned(answer)}"
    )
    assert "hunter2" not in prompt and "4242" not in prompt
    assert "I like tea." in prompt and "Tea it is." in prompt


async def test_the_token_counts_the_server_reports_are_added_up() -> None:
    stats = Stats()
    stub = StubEncoder(usage=(100, 20))
    await hippo.encode(stub, "T", chat(30), window_chars=500, stats=stats)
    assert stats.calls > 3 and stats.reported
    assert (stats.prompt_tokens, stats.completion_tokens) == (
        100 * stats.calls,
        20 * stats.calls,
    )
    silent = Stats()
    await hippo.encode(StubEncoder(), "T", chat(3), window_chars=500, stats=silent)
    assert silent.calls > 0 and not silent.reported
    assert (silent.prompt_tokens, silent.completion_tokens) == (0, 0)


# ── The reply ──


def test_the_json_is_found_in_a_fence_or_among_words_and_thinking_is_left_out() -> None:
    body = says([item("A", "The user said a thing.", "profile", [0])], [1])
    for reply in (
        body,
        f"```json\n{body}\n```",
        f"```\n{body}\n```",
        f"Here you go: {body} Hope it helps.",
        f"<think>maybe {{ not json }}</think>{body}",
    ):
        value = hippo.read_reply(reply)
        assert value is not None and value["ignore"] == [1], reply
    for reply in ("", "no json", "[1, 2]", '{"x": 1}', '{"items": 4}', "{broken"):
        assert hippo.read_reply(reply) is None, reply
    assert hippo.read_reply('{"ignore": [0]}') == {"ignore": [0]}


async def test_unreadable_json_is_asked_once_more_then_the_window_has_failed() -> None:
    stub = StubEncoder("this is not json")
    stats = Stats()
    result = await hippo.encode(stub, "T", chat(3), window_chars=12000, stats=stats)
    assert len(stub.prompts) == 2 == stats.calls
    (window,) = result.windows
    assert window.failed and not window.cached
    assert result.pieces == [] and result.kept == [] and result.ignored == []
    assert result.unmentioned == [0, 1, 2]


async def test_a_second_try_that_reads_gives_the_items_and_a_failed_call_is_retried() -> (
    None
):
    answers = iter(["garbage", says([item("Home", "The user lives in Utrecht.")])])
    stub = StubEncoder(lambda _p: next(answers))
    result = await hippo.encode(stub, "T", chat(1), window_chars=12000)
    assert len(stub.prompts) == 2 and not result.windows[0].failed
    assert [p.text for p in result.pieces] == ["The user lives in Utrecht."]
    flaky = iter(["RAISE", says([item("Home", "The user lives in Utrecht.")])])
    stub = StubEncoder(lambda _p: next(flaky))
    result = await hippo.encode(stub, "T", chat(1), window_chars=12000)
    assert len(stub.prompts) == 2 and not result.windows[0].failed
    down = StubEncoder("RAISE")
    result = await hippo.encode(down, "T", chat(1), window_chars=12000)
    assert len(down.prompts) == 2 and result.windows[0].failed


async def test_a_failed_request_names_only_the_error_class_in_the_reason_and_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider error can carry part of the server's reply, which may echo the
    request and so the user's texts: nothing of it is kept or logged."""
    caplog.set_level(logging.DEBUG)  # every logger, whatever its name

    async def send(_messages: list[Message]) -> Message:
        raise RuntimeError("server echoed: my gate is 9F")

    value, why = await hippo._ask(send, "prompt", Stats())
    assert value is None and why == "the model failed: RuntimeError"
    lines = [r.getMessage() for r in caplog.records]
    assert lines == ["Hippo encoder: the model failed: RuntimeError"] * 2
    assert not any(r.exc_info for r in caplog.records)


async def test_a_window_that_failed_does_not_stop_the_next_one() -> None:
    turns = chat(20)
    first = hippo.pack(turns, 500)[0]
    stub = StubEncoder(
        lambda prompt: (
            "garbage"
            if f"[{first.positions[0]}] " in prompt
            else says([item("Q", "The user asked something.", "episodic", [19])])
        )
    )
    result = await hippo.encode(stub, "T", turns, window_chars=500)
    assert result.windows[0].failed and not any(w.failed for w in result.windows[1:])
    assert result.pieces and set(result.unmentioned) >= set(first.positions)


# ── Validation ──


async def test_sources_outside_the_window_are_dropped_and_an_item_left_with_none_too() -> (
    None
):
    turns = chat(30)
    windows = hippo.pack(turns, 600)
    assert len(windows) > 2
    middle = windows[1].positions
    stub = StubEncoder(
        lambda prompt: (
            says(
                [
                    item(
                        "Both", "The user said both.", "episodic", [middle[0], 0, 999]
                    ),
                    item("Gone", "The user said gone.", "episodic", [0, 999, -1]),
                    item("Text", "The user said text.", "episodic", ["x", True]),  # type: ignore[list-item]
                ]
            )
            if numbers_in(prompt)[0] == middle[0]
            else says()
        )
    )
    result = await hippo.encode(stub, "T", turns, window_chars=600)
    (kept,) = result.pieces
    assert kept.title == "Both" and kept.sources == (middle[0],)
    report = result.windows[1]
    assert report.dropped == 2 and sum(w.dropped for w in result.windows) == 2


async def test_a_string_number_counts_and_a_bad_item_is_dropped_and_counted() -> None:
    reply = json.dumps(
        {
            "items": [
                {
                    "title": "A",
                    "text": "The user said a.",
                    "kind": "profile",
                    "sources": ["1", 0],
                },
                "not an item",
                {"title": "B", "text": "", "kind": "profile", "sources": [0]},
                {"title": "C", "kind": "profile", "sources": [0]},
                {"title": "D", "text": "The user said d.", "kind": "profile"},
            ],
            "ignore": ["0"],
        }
    )
    result = await hippo.encode(StubEncoder(reply), "T", chat(2), window_chars=12000)
    (piece,) = result.pieces
    assert piece.sources == (0, 1) and result.windows[0].dropped == 4
    assert result.ignored == []  # named in a source too


async def test_what_the_reply_leaves_out_is_unmentioned_and_ignore_is_ignored() -> None:
    turns = chat(6)
    reply = says([item("Q", "The user asked 1 and 2.", "episodic", [1, 2])], [0, 5, 77])
    result = await hippo.encode(StubEncoder(reply), "T", turns, window_chars=12000)
    assert result.kept == [1, 2]
    assert result.ignored == [0, 5]  # 77 is no exchange of the window
    assert result.unmentioned == [3, 4]
    assert result.forced == []
    everything = sorted(result.kept + result.ignored + result.unmentioned)
    assert everything == list(range(6))


async def test_kind_title_and_text_are_checked_and_cut() -> None:
    long_text = "The user said they like long sentences about trains. " * 30
    reply = says(
        [
            {"title": "T" * 200, "text": long_text, "kind": "PROFILE", "sources": [0]},
            {"text": "The user said something odd.", "kind": "mood", "sources": [0]},
            {"title": "x", "text": "A gift.", "kind": 5, "sources": [1]},
        ]
    )
    result = await hippo.encode(StubEncoder(reply), "T", chat(2), window_chars=12000)
    first, second, third = result.pieces
    assert first.kind == "profile" and len(first.title) <= hippo.TITLE_CHARS
    assert first.title.endswith("…")
    assert len(first.text) <= hippo.TEXT_CHARS and first.text.endswith("…")
    # An unknown kind is episodic, the kind that never replaces another item; a
    # missing title is the start of the text.
    assert (second.kind, second.title) == ("episodic", "The user said something odd.")
    assert third.kind == "episodic"
    assert hippo.TITLE_CHARS == 80 and hippo.TEXT_CHARS == 800


async def test_an_item_that_holds_a_secret_is_refused_whole_and_counted() -> None:
    # A password in the text, a card number in the title only, words with no value,
    # and a card number across the cut at TEXT_CHARS: each is refused, none is kept
    # as a remnant. 4242 4242 4242 4242 is a documented test card number.
    straddling = "word " * 158 + "4242 4242 4242 4242 and more"
    assert straddling.index("4242") < hippo.TEXT_CHARS < straddling.index("and more")
    reply = says(
        [
            item(
                "Login", "The user likes tea. The password is hunter2.", "profile", [0]
            ),
            item("Card 4242 4242 4242 4242", "The user has a card.", "profile", [1]),
            item("Secret", "The user said their password is private.", "profile", [2]),
            item("Long", straddling, "profile", [3]),
            item("Tea", "The user said they like green tea.", "profile", [4]),
        ]
    )
    result = await hippo.encode(StubEncoder(reply), "T", chat(5), window_chars=12000)
    (piece,) = result.pieces
    assert piece.title == "Tea" and piece.sources == (4,)
    assert result.refused == 4 and result.windows[0].dropped == 0
    assert result.kept == [4] and result.unmentioned == [0, 1, 2, 3]


async def test_a_refused_item_leaves_its_exchange_to_the_rules_not_to_the_secret() -> (
    None
):
    turns = [turn(0, "I live in Utrecht. My password is hunter2.", "Noted.")]
    reply = says(
        [item("Home", "The user lives in Utrecht, password hunter2.", "profile", [0])]
    )
    result = await hippo.encode(StubEncoder(reply), "T", turns, window_chars=12000)
    # The model's item is refused; the statement the rules read from the clean
    # sentence is made into an item instead.
    (forced,) = result.pieces
    assert (forced.text, forced.encoder, forced.forced) == (
        "You live in Utrecht.",
        hippo.RULES_ENCODER,
        True,
    )
    assert result.refused == 1 and result.forced == [0] and result.kept == [0]


async def test_a_chat_title_that_holds_a_secret_reaches_the_model_cleaned() -> None:
    title = "Card 4242 4242 4242 4242"
    stub = StubEncoder()
    await hippo.encode(stub, title, chat(1), window_chars=12000)
    (prompt,) = stub.prompts
    assert prompt.startswith(f"Chat title: {cleaned(title)}\n\n[0] ")
    assert "4242" not in prompt


async def test_items_come_in_time_order_stamped_with_their_newest_sources_time() -> (
    None
):
    turns = chat(8)
    reply = says(
        [
            item("Late", "The user said late.", "episodic", [6, 7]),
            item("Early", "The user said early.", "episodic", [1]),
            item("Spread", "The user said spread.", "episodic", [0, 5]),
        ]
    )
    result = await hippo.encode(StubEncoder(reply), "T", turns, window_chars=12000)
    assert [p.title for p in result.pieces] == ["Early", "Spread", "Late"]
    assert [p.created_at for p in result.pieces] == [
        turns[1].answered_at,
        turns[5].answered_at,
        turns[7].answered_at,
    ]
    assert {p.encoder for p in result.pieces} == {"3:stub-encoder"}


# ── One-shot strong signal ──


async def test_a_lasting_statement_no_item_covers_becomes_an_item_by_the_rules() -> (
    None
):
    turns = [
        turn(0, "I live in Pittsburgh.", "Nice."),
        turn(1, "Thanks, bye.", "Bye!"),
        turn(2, "What is a good book?", "Try Dune."),
    ]
    result = await hippo.encode(
        StubEncoder(says([item("Book", "melo recommended Dune.", "semantic", [2])])),
        "T",
        turns,
        window_chars=12000,
    )
    (forced,) = [p for p in result.pieces if p.forced]
    assert (forced.kind, forced.text, forced.sources) == (
        "profile",
        "You live in Pittsburgh.",
        (0,),
    )
    assert forced.encoder == hippo.RULES_ENCODER == "rules"
    assert forced.created_at == turns[0].answered_at
    assert result.forced == [0] and result.kept == [0, 2]
    assert result.unmentioned == [1]
    assert [p.title for p in result.pieces] == ["You live in Pittsburgh.", "Book"]


async def test_a_statement_an_item_covers_is_not_forced_but_an_ignored_one_is() -> None:
    turns = [
        turn(0, "I live in Pittsburgh.", "Nice."),
        turn(1, "My cat is called Pixel.", "Cute."),
    ]
    covered = says([item("Home", "The user lives in Pittsburgh.", "profile", [0])], [1])
    result = await hippo.encode(StubEncoder(covered), "T", turns, window_chars=12000)
    assert result.forced == [1] and result.ignored == []
    assert result.kept == [0, 1]
    assert sorted(p.sources for p in result.pieces) == [(0,), (1,)]
    nothing = await hippo.encode(StubEncoder(says()), "T", turns, window_chars=12000)
    assert nothing.forced == [0, 1]


async def test_a_statement_in_a_window_that_failed_is_forced_too() -> None:
    turns = [turn(0, "I live in Pittsburgh.", "Nice.")]
    result = await hippo.encode(StubEncoder("garbage"), "T", turns, window_chars=12000)
    assert result.windows[0].failed and result.forced == [0]
    assert [p.text for p in result.pieces] == ["You live in Pittsburgh."]


async def test_the_first_window_that_failed_is_where_the_mark_must_stop() -> None:
    turns = padded(4, 500)

    def broken(*bad: int) -> StubEncoder:
        def reply(prompt: str) -> str:
            return "RAISE" if set(labels(prompt)) & set(bad) else by_label(prompt)

        return StubEncoder(reply)

    # One exchange to a window, so each window is one exchange.
    assert window_sizes(turns, 700) == [1, 1, 1, 1]
    for bad, first in (((2,), 2), ((3, 2), 2), ((0,), 0)):
        result = await hippo.encode(broken(*bad), "T", turns, window_chars=700)
        assert result.failed_from == first
    whole = await hippo.encode(broken(), "T", turns, window_chars=700)
    assert whole.failed_from is None and whole.refused == 0
    # A half of a window that ran out of room counts as well: only that half failed.
    half = StubEncoder(
        lambda prompt: (
            CUT_OFF
            if len(labels(prompt)) == 4
            else "RAISE" if labels(prompt)[0] == 2 else by_label(prompt)
        ),
        usage=lambda prompt: (
            len(prompt) // 4,
            CAP if len(labels(prompt)) == 4 else 10,
        ),
    )
    split = await hippo.encode(half, "T", padded(4, 2000), window_chars=12000)
    assert split.windows[0].split == 2 and split.windows[0].failed_parts == 1
    assert split.failed_from == 2


async def test_no_statement_means_no_forced_item() -> None:
    turns = [turn(0, "Is it going to rain? I wonder if I should go.", "Maybe.")]
    result = await hippo.encode(StubEncoder(says()), "T", turns, window_chars=12000)
    assert result.pieces == [] and result.forced == [] and result.unmentioned == [0]


# ── The cache ──


async def test_a_window_asked_before_makes_no_request_and_says_cached(
    tmp_path: Path,
) -> None:
    cache = Cache(str(tmp_path / "cache" / "replies.jsonl"))
    turns = chat(25)
    reply = says([item("Q", "The user asked 3.", "episodic", [3])])
    first, second = StubEncoder(reply), StubEncoder(reply)
    ran = Stats()
    one = await hippo.encode(
        first, "T", turns, window_chars=500, cache=cache, stats=ran
    )
    assert ran.calls == len(first.prompts) > 3 and ran.cache_hits == 0
    again = Stats()
    # A fresh cache object over the same file: the replies come from disk.
    two = await hippo.encode(
        second, "T", turns, window_chars=500, cache=Cache(cache.path), stats=again
    )
    assert second.prompts == [] and again.calls == 0
    assert again.cache_hits == ran.calls
    assert all(w.cached for w in two.windows) and not any(w.cached for w in one.windows)
    assert [(p.title, p.sources) for p in two.pieces] == [
        (p.title, p.sources) for p in one.pieces
    ]
    mode = stat.S_IMODE(os.stat(cache.path).st_mode)
    assert mode == 0o600
    lines = Path(cache.path).read_text().splitlines()
    assert len(lines) == ran.calls
    assert all(set(json.loads(line)) == {"key", "reply"} for line in lines)


async def test_the_key_is_the_prompt_version_the_model_and_the_text_with_its_dates(
    tmp_path: Path,
) -> None:
    cache = Cache(str(tmp_path / "replies.jsonl"))
    turns = [turn(0, "I like tea.", "Good.", at("2023-05-20"))]
    base = StubEncoder(says())
    await hippo.encode(base, "T", turns, window_chars=12000, cache=cache)
    assert len(base.prompts) == 1

    for stub, title, chat_turns in (
        (StubEncoder(says(), model="another"), "T", turns),  # another model
        (StubEncoder(says()), "T2", turns),  # another title
        (StubEncoder(says()), "T", [turn(0, "I like tea.", "Good.", at("2023-05-21"))]),
        (StubEncoder(says()), "T", [turn(0, "I like tea!", "Good.", at("2023-05-20"))]),
    ):
        await hippo.encode(stub, title, chat_turns, window_chars=12000, cache=cache)
        assert len(stub.prompts) == 1, (stub.model, title)
    same = StubEncoder(says())
    await hippo.encode(same, "T", turns, window_chars=12000, cache=cache)
    assert same.prompts == []
    assert hippo.window_key("m", "p") == hippo.window_key("m", "p")
    assert len({hippo.window_key("m", "p"), hippo.window_key("n", "p")}) == 2
    assert hippo.window_key("m", "p") != hippo.window_key("mp", "")
    assert hippo.PROMPT_VERSION == "3"


async def test_a_window_that_failed_is_not_cached_and_a_bad_line_is_skipped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replies.jsonl"
    path.write_text('not json\n{"key": 3}\n[]\n')
    cache = Cache(str(path))
    stub = StubEncoder("garbage")
    await hippo.encode(stub, "T", chat(1), window_chars=12000, cache=cache)
    assert len(stub.prompts) == 2
    assert path.read_text() == 'not json\n{"key": 3}\n[]\n'
    good = StubEncoder(says())
    await hippo.encode(good, "T", chat(1), window_chars=12000, cache=cache)
    assert len(good.prompts) == 1 and len(path.read_text().splitlines()) == 4


async def test_one_cache_file_serves_several_chats_and_runs(tmp_path: Path) -> None:
    cache = Cache(str(tmp_path / "shared.jsonl"))
    a, b = chat(2), [turn(0, "Other chat.", "Yes.")]
    stub = StubEncoder(says())
    await hippo.encode(stub, "A", a, window_chars=12000, cache=cache)
    await hippo.encode(stub, "B", b, window_chars=12000, cache=cache)
    assert len(stub.prompts) == 2
    again = StubEncoder(says())
    for title, turns in (("A", a), ("B", b)):
        await hippo.encode(
            again, title, turns, window_chars=12000, cache=Cache(cache.path)
        )
    assert again.prompts == []


@pytest.mark.parametrize("limit", [4000, 12000, 60000])
async def test_the_window_bounds_of_the_setting_hold_for_any_chat(limit: int) -> None:
    turns = [turn(n, "q" * 300, "a" * (200 + 17 * n)) for n in range(80)]
    for window in hippo.pack(turns, limit):
        assert len(window.text) <= limit


# ── A window that runs out of room, and a very long exchange (#428) ──

# One fixed chat, as the prompt shows it in UTC, and the key of its window computed
# before windows could split: an ordinary window must keep both, byte for byte.
PINNED_BASE = 1684591320.0  # 2023-05-20 14:02:00 UTC
PINNED_PROMPT = (
    "Chat title: Tea and labs\n\n"
    "[0] 2023-05-20 14:02 User: I like green tea. / melo: Noted.\n\n"
    "[1] 2023-05-20 14:03 User: Where is the lab? / melo: Room 14, second floor.\n\n"
    "[2] 2023-05-20 14:04 User: Anyone there?"
)
PINNED_KEY = "df915b423559291dbb77e6edfea0ce0850f5b98d0594a5ed7d4cddef387d8b92"


@pytest.fixture
def utc(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


# _ran_out


def reply_of(content: str, tokens: int | None = None) -> Message:
    return Message(
        "assistant",
        content,
        usage=None if tokens is None else {"completion_tokens": tokens},
    )


def test_a_reply_ran_out_when_it_reports_the_cap_in_completion_tokens() -> None:
    for tokens in (CAP, CAP + 40, CAP - 16):
        assert hippo._ran_out(reply_of("garbage", tokens)), tokens
    for tokens in (CAP - 17, 300):
        assert not hippo._ran_out(reply_of(CUT_OFF, tokens)), tokens
    # A report that is no count (zero, or a bool) is no report: the text decides.
    assert hippo._ran_out(reply_of(CUT_OFF, 0))
    assert not hippo._ran_out(reply_of("garbage", 0))
    odd = Message("assistant", CUT_OFF, usage={"completion_tokens": True})  # type: ignore[dict-item]
    assert hippo._ran_out(odd)
    assert not hippo._ran_out(Message("assistant", "garbage"))


def test_without_a_token_count_a_json_object_never_closed_ran_out() -> None:
    assert hippo._ran_out(reply_of(CUT_OFF))
    assert hippo._ran_out(reply_of('{"items": [{"title": "A"}, {"title"'))
    assert hippo._ran_out(reply_of("```json\n" + CUT_OFF))
    assert hippo._ran_out(reply_of("<think>hm</think>\n" + CUT_OFF))
    # A brace inside a string, or an escaped quote, does not close or open anything.
    assert hippo._ran_out(reply_of('{"items": [{"text": "a } b \\" }'))
    assert not hippo._ran_out(reply_of('{"items": [{"text": "a } b"}], "ignore": []}'))
    assert not hippo._ran_out(reply_of('{"items": [], "ignore": []}'))
    assert not hippo._ran_out(reply_of('```json\n{"items": []}\n```'))
    assert not hippo._ran_out(reply_of("Sorry, I cannot do that."))
    assert not hippo._ran_out(reply_of(""))
    assert not hippo._ran_out(reply_of('Here: {"items": [{"title"'))


# Splitting a window


def test_a_window_of_several_exchanges_splits_between_exchanges_near_the_middle() -> (
    None
):
    turns = [turn(n, f"Question {n}?", "y" * (600 + 700 * (n == 5))) for n in range(10)]
    (window,) = hippo.pack(turns, 12000)
    first, second = hippo.halves(window) or pytest.fail("no halves")
    assert first.positions + second.positions == window.positions
    assert first.text + hippo._SEPARATOR + second.text == window.text
    assert not first.cut and not second.cut
    # The boundary with the least difference in characters between the two sides.
    sizes = [
        abs(
            len(hippo._SEPARATOR.join(hippo.render(n, turns[n]) for n in range(k)))
            - len(
                hippo._SEPARATOR.join(hippo.render(n, turns[n]) for n in range(k, 10))
            )
        )
        for k in range(1, 10)
    ]
    assert len(first.positions) == 1 + sizes.index(min(sizes))
    # Each half can be split again by its own exchange boundaries.
    inner = hippo.halves(second)
    assert inner is not None and inner[0].positions + inner[1].positions == (
        second.positions
    )
    assert inner[0].text + hippo._SEPARATOR + inner[1].text == second.text


def test_a_window_of_one_exchange_splits_its_text_at_a_paragraph_break() -> None:
    turns = [turn(0, "Read.", paragraphs(5, 900))]
    (window,) = hippo.pack(turns, 12000)
    first, second = hippo.halves(window) or pytest.fail("no halves")
    assert first.positions == second.positions == (0,)
    assert second.text.startswith("[0] (continued) ")
    assert first.text + second.text.removeprefix("[0] (continued) ") == window.text
    # Five paragraphs and a heading: the break nearest the middle is after the second.
    assert first.text.endswith("\n\n") and first.text.count("\n\n") == 2
    assert second.text.removeprefix("[0] (continued) ").startswith("c")


def test_without_a_paragraph_the_cut_is_at_a_sentence_end_then_whitespace_then_the_middle() -> (
    None
):
    sentences = turn(0, "Read.", "This is a sentence. " * 150)
    whitespace = turn(0, "Read.", "word " * 600)
    solid = turn(0, "Read.", "z" * 3000)
    for exchange, ends in ((sentences, ". "), (whitespace, " "), (solid, "z")):
        (window,) = hippo.pack([exchange], 12000)
        first, second = hippo.halves(window) or pytest.fail("no halves")
        rest = second.text.removeprefix("[0] (continued) ")
        assert first.text + rest == window.text
        assert first.text.endswith(ends)
        assert abs(len(first.text) - len(rest)) < 60
    (window,) = hippo.pack([solid], 12000)
    first, _ = hippo.halves(window) or pytest.fail("no halves")
    assert len(first.text) == len(window.text) // 2  # nothing to cut at: the middle
    (window,) = hippo.pack([sentences], 12000)
    first, _ = hippo.halves(window) or pytest.fail("no halves")
    assert first.text.rstrip().endswith(".")


def test_a_window_under_the_minimum_does_not_split() -> None:
    assert hippo.MIN_SPLIT_CHARS == 1500
    (small,) = hippo.pack([turn(0, "Read.", "x" * 1300)], 12000)
    assert len(small.text) < hippo.MIN_SPLIT_CHARS and hippo.halves(small) is None
    (small,) = hippo.pack(chat(5), 12000)
    assert hippo.halves(small) is None
    (enough,) = hippo.pack([turn(0, "Read.", "x" * 1500)], 12000)
    assert hippo.halves(enough) is not None
    # A window built by hand has no exchange offsets to split several exchanges by.
    assert hippo.halves(hippo.Window((0, 1), "a" * 2000)) is None


# Parts of a very long exchange


def test_one_exchange_of_30000_characters_is_three_parts_that_make_up_its_block() -> (
    None
):
    exchange = turn(0, "Read it all.", paragraphs(10, 2900))
    block = hippo.render(0, exchange)
    assert 29000 < len(block) < 30000
    windows = hippo.pack([exchange], 12000)
    assert [w.part for w in windows] == [(1, 3), (2, 3), (3, 3)]
    assert all(w.positions == (0,) and len(w.text) <= 12000 for w in windows)
    assert not any(w.cut for w in windows)
    prefix = "[0] (continued) "
    assert [w.text.startswith(prefix) for w in windows] == [False, True, True]
    assert "".join(w.text.removeprefix(prefix) for w in windows) == block
    assert windows[0].text.startswith("[0] 2023-05-20 14:02 User: Read it all.")


def test_one_exchange_of_60000_characters_is_four_parts_and_the_last_says_it_is_cut() -> (
    None
):
    exchange = turn(0, "Read it all.", "word " * 12000)
    block = hippo.render(0, exchange)
    assert len(block) > 60000
    windows = hippo.pack([exchange], 12000)
    assert len(windows) == hippo.MAX_PARTS == 4
    assert [w.part for w in windows] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [w.cut for w in windows] == [False, False, False, True]
    assert all(len(w.text) <= 12000 for w in windows)
    prefix = "[0] (continued) "
    joined = "".join(w.text.removeprefix(prefix) if w.part[0] > 1 else w.text for w in windows)  # type: ignore[index]
    assert block.startswith(joined) and len(joined) < len(block)
    assert len(windows[-1].text) == 12000  # the last part is filled to the limit


def test_an_exchange_that_fits_a_window_and_the_ones_around_it_are_not_parts() -> None:
    turns = chat(4)
    turns[2] = turn(2, "Read it.", "x" * 3000)
    windows = hippo.pack(turns, 4000)
    assert all(w.part is None and not w.cut for w in windows)
    assert [n for w in windows for n in w.positions] == [0, 1, 2, 3]


def test_a_part_is_cut_at_a_paragraph_break_when_there_is_one() -> None:
    exchange = turn(0, "Read.", paragraphs(9, 1500))
    windows = hippo.pack([exchange], 4000)
    assert len(windows) > 2
    for window in windows[:-1]:
        assert window.text.endswith("\n\n"), window.part
    for window in windows[1:]:
        rest = window.text.removeprefix("[0] (continued) ")
        assert rest[0].isalpha() and rest == rest.lstrip()
    # No break to cut at: a sentence end, else whitespace.
    prose = turn(0, "Read.", "It was a long day. " * 600)
    for window in hippo.pack([prose], 4000)[:-1]:
        assert window.text.endswith(". ")


async def test_every_part_is_a_request_with_its_own_label_and_items() -> None:
    exchange = turn(0, "Read it all.", paragraphs(6, 3500))
    stub = StubEncoder(by_label)
    result = await hippo.encode(stub, "T", [exchange], window_chars=12000)
    windows = hippo.pack([exchange], 12000)
    assert len(windows) == 2 and len(stub.prompts) == 2
    assert [body(p) for p in stub.prompts] == [w.text for w in windows]
    assert [w.part for w in result.windows] == [(1, 2), (2, 2)]
    assert [w.cut for w in result.windows] == [False, False]
    assert [p.text for p in result.pieces] == [
        "Question 0, first part.",
        "Question 0, continued part.",
    ]
    assert result.kept == [0] and result.unmentioned == []


# The reply that runs out


@pytest.mark.parametrize("report", [True, False], ids=["token count", "unclosed json"])
async def test_a_window_that_runs_out_is_split_and_each_half_is_asked_once(
    tmp_path: Path, report: bool
) -> None:
    turns = padded(12)
    (window,) = hippo.pack(turns, 12000)
    assert len(window.text) > 6000
    cache = Cache(str(tmp_path / "replies.jsonl"))
    stub = runs_out_over(len(window.text) - 1, report=report)
    stats = Stats()
    result = await hippo.encode(
        stub, "T", turns, window_chars=12000, cache=cache, stats=stats
    )
    # The whole window, then each half; never the same prompt twice.
    assert len(stub.prompts) == 3 == stats.calls and len(set(stub.prompts)) == 3
    assert stub.prompts[0].endswith(window.text)
    assert [labels(p) for p in stub.prompts[1:]] == [list(range(6)), list(range(6, 12))]
    (done,) = result.windows
    assert (done.exchanges, done.chars) == (12, len(window.text))
    assert (done.split, done.failed_parts) == (2, 0)
    assert not done.failed and not done.cached and not done.cut and done.part is None
    # The items of both halves, in the order of the chat.
    assert [p.sources for p in result.pieces] == [(n,) for n in range(12)]
    assert result.kept == list(range(12)) and result.unmentioned == []
    assert done.to_json()["split"] == 2 and done.to_json()["part"] is None
    # What came back is cached, and so is the fact that the window ran out.
    key = hippo.window_key("stub-encoder", prompt_of_window(window))
    again = Stats()
    quiet = StubEncoder(by_label)
    second = await hippo.encode(
        quiet, "T", turns, window_chars=12000, cache=Cache(cache.path), stats=again
    )
    assert quiet.prompts == [] and again.calls == 0
    assert again.cache_hits == 3  # the window's mark and each half
    assert second.windows[0].cached and second.windows[0].split == 2
    assert [(p.title, p.sources) for p in second.pieces] == [
        (p.title, p.sources) for p in result.pieces
    ]
    rows = [json.loads(line) for line in Path(cache.path).read_text().splitlines()]
    assert len(rows) == 3 and {row["key"] for row in rows} >= {hippo.split_key(key)}
    assert key not in {row["key"] for row in rows}  # no reply is kept for the window
    assert hippo.split_key(key) != key and Cache(cache.path).get(key) is None


async def test_one_exchange_that_runs_out_is_split_in_its_text_and_both_parts_are_read() -> (
    None
):
    exchange = turn(0, "Read.", paragraphs(8, 1000))
    (window,) = hippo.pack([exchange], 12000)
    stub = runs_out_over(len(window.text) - 1)
    result = await hippo.encode(stub, "T", [exchange], window_chars=12000)
    assert len(stub.prompts) == 3
    first, second = body(stub.prompts[1]), body(stub.prompts[2])
    assert first + second.removeprefix("[0] (continued) ") == window.text
    assert second.startswith("[0] (continued) ") and first.endswith("\n\n")
    assert [p.text for p in result.pieces] == [
        "Question 0, first part.",
        "Question 0, continued part.",
    ]
    assert result.windows[0].split == 2 and not result.windows[0].failed


async def test_a_reply_that_is_unreadable_but_did_not_run_out_is_asked_twice_then_fails() -> (
    None
):
    turns = padded(12)
    for usage in ((100, 300), (100, CAP - 17)):
        stub = StubEncoder(CUT_OFF, usage=usage)
        stats = Stats()
        result = await hippo.encode(stub, "T", turns, window_chars=12000, stats=stats)
        assert len(stub.prompts) == 2 == stats.calls and len(set(stub.prompts)) == 1
        (window,) = result.windows
        assert window.failed and (window.split, window.failed_parts) == (0, 0)
    prose = StubEncoder("I cannot do that.")
    result = await hippo.encode(prose, "T", turns, window_chars=12000)
    assert len(prose.prompts) == 2 and result.windows[0].failed


async def test_a_window_that_cannot_split_is_not_asked_again_and_fails() -> None:
    stub = StubEncoder(CUT_OFF, usage=(100, CAP))
    stats = Stats()
    turns = chat(3)  # far under the minimum
    result = await hippo.encode(stub, "T", turns, window_chars=12000, stats=stats)
    assert len(stub.prompts) == 1 == stats.calls
    (window,) = result.windows
    assert window.failed and window.split == 0 and window.failed_parts == 0
    assert result.pieces == [] and result.unmentioned == [0, 1, 2]


@pytest.mark.parametrize("shape", ["exchanges", "one exchange"])
async def test_a_provider_that_always_runs_out_makes_four_sub_windows_then_fails(
    shape: str,
) -> None:
    turns = (
        padded(12) if shape == "exchanges" else [turn(0, "Read.", paragraphs(7, 1000))]
    )
    stub = StubEncoder(CUT_OFF, usage=(100, CAP))
    stats = Stats()
    result = await hippo.encode(stub, "T", turns, window_chars=12000, stats=stats)
    # The window, its two halves and their two halves each; nothing is asked twice.
    assert len(stub.prompts) == 7 == stats.calls and len(set(stub.prompts)) == 7
    (window,) = result.windows
    assert window.failed and (window.split, window.failed_parts) == (4, 4)
    assert result.pieces == [] and result.kept == []
    assert window.to_json()["failed_parts"] == 4
    assert hippo.MAX_SPLIT_DEPTH == 2


async def test_a_half_that_runs_out_again_splits_once_more_and_the_rest_is_kept() -> (
    None
):
    turns = padded(12)

    def long(prompt: str) -> bool:
        """The whole window and the first half of it run out; nothing else does."""
        numbers = labels(prompt)
        return len(numbers) == 12 or (numbers[0] == 0 and len(numbers) > 3)

    stub = StubEncoder(
        lambda p: CUT_OFF if long(p) else by_label(p),
        usage=lambda p: (50, CAP if long(p) else 100),
    )
    stats = Stats()
    result = await hippo.encode(stub, "T", turns, window_chars=12000, stats=stats)
    (done,) = result.windows
    # The window, its halves (0-5 and 6-11), then the halves of the first (0-2, 3-5).
    assert [labels(p)[0] for p in stub.prompts] == [0, 0, 0, 3, 6]
    assert stats.calls == 5 and (done.split, done.failed_parts) == (3, 0)
    assert not done.failed
    assert [p.sources for p in result.pieces] == [(n,) for n in range(12)]


async def test_a_half_that_fails_fails_alone_and_the_other_is_kept() -> None:
    turns = padded(12)
    (window,) = hippo.pack(turns, 12000)

    def reply(prompt: str) -> str:
        if len(prompt) > len(window.text) - 1:
            return CUT_OFF
        return "garbage" if labels(prompt)[0] == 0 else by_label(prompt)

    stub = StubEncoder(
        reply, usage=lambda p: (50, CAP if len(p) > len(window.text) - 1 else 100)
    )
    stats = Stats()
    result = await hippo.encode(stub, "T", turns, window_chars=12000, stats=stats)
    (done,) = result.windows
    # The first half is unreadable and did not run out: asked twice, then failed.
    assert stats.calls == 4 and len(stub.prompts) == 4
    assert (done.split, done.failed_parts) == (2, 1)
    assert not done.failed
    assert [p.sources for p in result.pieces] == [(n,) for n in range(6, 12)]
    assert result.unmentioned == list(range(6))


async def test_a_chat_with_several_windows_splits_only_the_one_that_ran_out() -> None:
    turns = padded(30)
    sizes = window_sizes(turns, 6000)
    assert len(sizes) > 2
    windows = hippo.pack(turns, 6000)
    biggest = max(len(w.text) for w in windows)
    stub = runs_out_over(biggest - 1)
    ordinary = [w for w in windows if len(w.text) < biggest]
    result = await hippo.encode(stub, "T", turns, window_chars=6000)
    ran = [r for r in result.windows if r.split]
    assert ran and all(r.chars == biggest for r in ran)
    assert [r.split for r in result.windows if r.chars < biggest] == [0] * len(ordinary)
    assert [p.sources for p in result.pieces] == [(n,) for n in range(30)]
    assert not any(r.failed for r in result.windows)


# Caches stay valid


async def test_an_ordinary_window_keeps_its_prompt_and_its_cache_key(
    utc: None, tmp_path: Path
) -> None:
    turns = [
        Turn(1, PINNED_BASE, PINNED_BASE + 5, "I like green tea.", "Noted."),
        Turn(
            2,
            PINNED_BASE + 60,
            PINNED_BASE + 65,
            "Where is the lab?",
            "Room 14, second floor.",
        ),
        Turn(3, PINNED_BASE + 120, PINNED_BASE + 125, "Anyone there?", ""),
    ]
    (window,) = hippo.pack(turns, 12000)
    assert window.part is None and not window.cut
    prompt = hippo.prompt_of("Tea and labs", window)
    assert prompt == PINNED_PROMPT
    assert hippo.window_key("pinned-model", prompt) == PINNED_KEY
    assert hippo.PROMPT_VERSION == "3"
    # A cache written before splits existed is served whole: no request is made.
    cache = Cache(str(tmp_path / "old.jsonl"))
    cache.put(
        PINNED_KEY,
        json.loads(says([item("Tea", "The user likes tea.", "profile", [0])])),
    )
    stub = StubEncoder(says(), model="pinned-model")
    stats = Stats()
    result = await hippo.encode(
        stub, "Tea and labs", turns, window_chars=12000, cache=cache, stats=stats
    )
    assert stub.prompts == [] and (stats.calls, stats.cache_hits) == (0, 1)
    assert result.windows[0].cached and result.windows[0].split == 0
    assert [p.text for p in result.pieces][:1] == ["The user likes tea."]


async def test_a_cache_from_before_serves_the_windows_that_did_not_fail_and_asks_the_rest(
    tmp_path: Path,
) -> None:
    turns = padded(30)
    windows = hippo.pack(turns, 6000)
    cache = Cache(str(tmp_path / "old.jsonl"))
    failed = windows[1]
    for window in windows:
        if window is not failed:
            prompt = hippo.prompt_of("T", window)
            cache.put(
                hippo.window_key("stub-encoder", prompt),
                json.loads(by_label(prompt)),
            )
    stub = runs_out_over(len(failed.text) - 1)
    stats = Stats()
    result = await hippo.encode(
        stub, "T", turns, window_chars=6000, cache=cache, stats=stats
    )
    # Only the window that was missing is asked, and it is split.
    assert stats.cache_hits == len(windows) - 1
    assert stats.calls == 3 and all(
        labels(p)[0] >= windows[1].positions[0] for p in stub.prompts
    )
    assert [r.split for r in result.windows] == [0, 2] + [0] * (len(windows) - 2)
    assert [r.cached for r in result.windows] == [True, False] + [True] * (
        len(windows) - 2
    )
    assert not any(r.failed for r in result.windows)
