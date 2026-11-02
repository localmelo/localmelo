"""The Hippo encoder: a chat's turns become memory items (developer setting, #428).

A chat's exchanges are packed into windows of at most ``compress_at_chars``
characters, one model request each at temperature 0; the reply's JSON is checked
(sources outside the window dropped, an item left with none dropped and counted,
exchanges named nowhere unmentioned), a lasting statement no item covers is made
into an item by the rules (forced), and a window's reply is kept in a cache so the
same window makes no second request. Synthetic chats and a stub chat model that
answers with canned JSON: no model server.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from localmelo.support.memory import hippo_encode as hippo
from localmelo.support.memory.hippo_encode import Cache, Stats, Turn
from localmelo.tests.memory.hippo_helpers import (
    StubEncoder,
    at,
    item,
    numbers_in,
    says,
    turn,
    window_sizes,
)

TRAINS = (
    "Tell me about the night trains from Chicago.",
    "The California Zephyr leaves Chicago at 14:00 and reaches Emeryville in 51 hours.",
)


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


def test_an_exchange_longer_than_a_window_has_a_window_of_its_own_cut_at_its_size() -> (
    None
):
    turns = chat(6)
    turns[3] = turn(3, "Please read this.", "word " * 4000)
    windows = hippo.pack(turns, 4000)
    long = [w for w in windows if w.cut]
    assert len(long) == 1 and long[0].positions == (3,)
    assert len(long[0].text) == 4000 and long[0].text.startswith("[3] 2023-05-20")
    assert [w.positions for w in windows if not w.cut] == [(0, 1, 2), (4, 5)]
    assert windows.index(long[0]) == 1  # between its neighbours, in order


async def test_each_window_is_one_request_and_a_cut_window_says_so() -> None:
    turns = chat(30)
    turns[10] = turn(10, "Read it.", "x" * 9000)
    stub = StubEncoder()
    stats = Stats()
    result = await hippo.encode(stub, "Title", turns, window_chars=4000, stats=stats)
    windows = hippo.pack(turns, 4000)
    assert len(result.windows) == len(windows) == len(stub.prompts) == stats.calls
    assert [w.exchanges for w in result.windows] == [len(w.positions) for w in windows]
    assert [w.cut for w in result.windows] == [w.cut for w in windows]
    assert sum(w.cut for w in result.windows) == 1
    assert [w.chars for w in result.windows] == [len(w.text) for w in windows]
    for prompt, window in zip(stub.prompts, windows, strict=True):
        assert numbers_in(prompt) == list(window.positions)


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
    assert stub.bodies == [{"temperature": 0, "max_tokens": hippo.MAX_OUTPUT_TOKENS}]
    assert hippo.MAX_OUTPUT_TOKENS == 2000


async def test_the_chat_models_own_provider_is_not_changed_by_the_request() -> None:
    stub = StubEncoder()
    stub.extra_body_by_kind = {"retrieve": {"top_p": 0.9}, "plan": {"x": 1}}
    await hippo.encode(stub, "T", chat(2), window_chars=12000)
    assert stub.extra_body_by_kind == {"retrieve": {"top_p": 0.9}, "plan": {"x": 1}}
    assert stub.bodies == [
        {"top_p": 0.9, "temperature": 0, "max_tokens": hippo.MAX_OUTPUT_TOKENS}
    ]


async def test_a_secret_never_reaches_the_model_and_a_missing_title_is_none() -> None:
    stub = StubEncoder()
    turns = [turn(0, "My password is hunter2. I like tea.", "Noted, tea it is.")]
    await hippo.encode(stub, "", turns, window_chars=12000)
    (prompt,) = stub.prompts
    assert "hunter2" not in prompt and "I like tea." in prompt
    assert prompt.startswith("Chat title: (none)\n\n[0] ")


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


async def test_a_secret_the_model_wrote_is_left_out_of_the_item() -> None:
    reply = says(
        [
            item(
                "Login",
                "The user likes tea. The user said the password is hunter2.",
                "profile",
                [0],
            )
        ]
    )
    result = await hippo.encode(StubEncoder(reply), "T", chat(1), window_chars=12000)
    (piece,) = result.pieces
    assert "hunter2" not in piece.text and "likes tea" in piece.text


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
    assert {p.encoder for p in result.pieces} == {"2:stub-encoder"}


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
    assert hippo.PROMPT_VERSION == "2"


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
