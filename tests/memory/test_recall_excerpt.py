"""How much of a recalled chat turn is shown, and which part of it.

``[memory] recall_item_chars`` (default 200, the room one turn takes in the
block and in the search tool's list) and ``recall_excerpt`` (``head``, the start
of each part, or ``match``, the sentences of each part that hold most of the
question's content words). The defaults read exactly as a turn always did.
Synthetic turns, words only, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import (
    HEADS,
    TURN_CHARS,
    Item,
    Shown,
    block,
    snippet,
    turn_said,
)
from localmelo.support.memory.service import MemoryTools
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.test_recall_bars import _turn, config

QUESTION = "when does the tomato harvest peak?"
TYPED = "Tell me about the garden schedule?"
TARGET = "The tomato harvest peaks in late August after the first frost warning."
ZH_QUESTION = "我的组会是哪天？"
ZH_TARGET = "我的组会安排在每周四下午三点。"
SEP = "”; melo replied “"


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def padded(target: str, before: int, after: int, filler: str, gap: str = " ") -> str:
    """Filler sentences up to *before* characters, then *target*, then *after* more."""

    def sentences(size: int, first: int) -> str:
        out, n = "", first
        while len(out) < size:
            out += filler.format(n=n)
            n += 1
        return out

    return sentences(before, 0) + gap + target + gap + sentences(after, 1000)


def english(before: int = 1500, after: int = 600) -> str:
    return padded(TARGET, before, after, "Calm weather note {n} came and went. ")


def chinese(before: int = 1500, after: int = 500) -> str:
    return padded(ZH_TARGET, before, after, "今天的天气很好，适合出门散步{n}。", gap="")


def parts(said: str) -> tuple[str, str]:
    """The quoted user part and reply part of a turn's reading."""
    user, _, reply = said.partition(SEP)
    assert user.startswith("the user said “") and reply.endswith("”")
    return user[len("the user said “") :], reply[:-1]


def legacy_turn_said(text: str) -> str:
    """``turn_said`` before the room was a setting (200 characters, 60 for the reply)."""
    body = text.partition("\n")[2] if text.startswith("Task: ") else text
    if not body.startswith("User: "):
        return snippet(text)
    typed, marker, answer = body[len("User: ") :].partition("\nmelo: ")
    if not marker:
        return snippet(text)
    typed, answer = " ".join(typed.split()), " ".join(answer.split())
    if len(typed) + len(answer) > 200:
        typed = snippet(typed, 200 - 60 if answer else 200)
        answer = snippet(answer, 200 - len(typed)) if answer else ""
    said = [f"the user said “{typed}”"] if typed else []
    return "; ".join(said + ([f"melo replied “{answer}”"] if answer else []))


def words(size: int, tag: str) -> str:
    out, n = "", 0
    while len(out) < size:
        out += f"{tag}{n} "
        n += 1
    return out.strip()[:size].rstrip()


def test_the_defaults_read_a_turn_exactly_as_it_always_did() -> None:
    assert TURN_CHARS == 200
    assert Shown(200, frozenset()) == HEADS
    assert Shown.of() == HEADS and Shown.of(200, None) == HEADS
    assert Shown.of(200, "") == HEADS
    texts = [
        document_text(words(n, "u"), words(m, "r"))
        for n in (0, 1, 50, 120, 139, 140, 141, 200, 201, 600)
        for m in (0, 1, 59, 60, 61, 90, 200, 900)
    ]
    texts += [
        f"Task: Plan\n{texts[40]}",
        "Summary of chat “Fix CI”: the fixture port was taken. " * 8,
        "Task asked: which folder? Answer: the repo root.",
        "User: a melo: b",
        "User: question\nmelo: ",
    ]
    for text in texts:
        assert turn_said(text) == legacy_turn_said(text), text
        assert turn_said(text, HEADS) == legacy_turn_said(text), text
        assert turn_said(text, Shown.of(TURN_CHARS)) == legacy_turn_said(text), text
        # A question alone changes nothing: the head is shown unless "match" is set.
        assert turn_said(text, Shown.of(200, None)) == legacy_turn_said(text), text


def test_a_plain_snippet_takes_the_room_plus_forty() -> None:
    summary = "Summary of chat “Fix CI”: " + words(900, "s")
    assert turn_said(summary) == snippet(summary, 240) == snippet(summary)
    assert turn_said(summary, Shown.of(800)) == snippet(summary, 840)
    assert turn_said(summary, Shown.of(120)) == snippet(summary, 160)
    assert len(turn_said(summary, Shown.of(800))) == 840


def test_a_larger_room_shows_more_of_a_turn() -> None:
    text = document_text(words(500, "u"), words(900, "r"))
    small, large = turn_said(text), turn_said(text, Shown.of(800))
    assert len(large) > len(small) + 400
    user, reply = parts(large)
    assert len(user) + len(reply) <= 800 and len(user) > 200
    assert reply.endswith("…")
    # A room that holds both parts shows both whole.
    whole = turn_said(text, Shown.of(2000))
    assert "…" not in whole and whole.count("r") > 100
    # The smallest room still keeps a share of the reply.
    least = turn_said(text, Shown.of(120))
    assert (
        len(parts(least)[0]) + len(parts(least)[1]) <= 120
        and len(parts(least)[1]) >= 60
    )


def test_match_shows_the_sentence_deep_in_a_reply_with_an_ellipsis() -> None:
    text = document_text(TYPED, english())
    assert TARGET not in turn_said(text)  # the head never reaches it
    shown = Shown.of(200, QUESTION)
    said = turn_said(text, shown)
    user, reply = parts(said)
    assert user == TYPED  # the user's words fit in 40% of the room
    assert "tomato harvest peaks in late August" in reply
    assert reply.startswith("…") and reply.endswith("…")
    assert len(user) + len(reply) <= 200
    assert said.count("the user said") == 1 and said.count("melo replied") == 1
    # More room, more of what is around it, the sentence still whole.
    bigger = parts(turn_said(text, Shown.of(600, QUESTION)))[1]
    assert (
        TARGET in bigger
        and len(bigger) > len(reply) + 200
        and len(bigger) <= 600 - len(TYPED)
    )


def test_match_grows_the_best_sentence_to_its_neighbours() -> None:
    sentences = [f"Note number {n} sits here quietly." for n in range(30)]
    sentences[17] = "The tomato harvest peak is late August."
    reply = " ".join(sentences)
    room = 300
    window = parts(turn_said(document_text(TYPED, reply), Shown.of(room, QUESTION)))[1]
    assert sentences[17] in window
    assert sentences[16] in window or sentences[18] in window
    assert window.startswith("…") and window.endswith("…")
    # The window is used up, not just the one sentence.
    assert len(window) > len(TYPED) + 100
    assert len(TYPED) + len(window) <= room


def test_match_finds_the_sentence_in_chinese() -> None:
    text = document_text(ZH_QUESTION, chinese())
    assert ZH_TARGET not in turn_said(text)
    user, reply = parts(turn_said(text, Shown.of(200, ZH_QUESTION)))
    assert user == ZH_QUESTION
    assert ZH_TARGET in reply and reply.startswith("…") and reply.endswith("…")
    assert len(user) + len(reply) <= 200
    # A reply without a stop at all is cut into windows too.
    flat = "天气很好" * 300 + "我的组会安排在每周四" + "天气很好" * 300
    window = parts(
        turn_said(document_text(ZH_QUESTION, flat), Shown.of(200, ZH_QUESTION))
    )[1]
    assert "组会" in window and len(window) + len(ZH_QUESTION) <= 200


def test_with_no_match_at_all_the_head_is_shown() -> None:
    text = document_text(TYPED, english())
    for question in ("zebra yodel quokka", "what is the", "", "？？"):
        shown = Shown.of(200, question)
        assert turn_said(text, shown) == turn_said(text) == legacy_turn_said(text)
        assert turn_said(text, Shown.of(800, question)) == turn_said(
            text, Shown.of(800)
        )
    # A turn that fits is whole, match or not.
    short = document_text("Tomatoes?", "Late August.")
    assert turn_said(short, Shown.of(200, QUESTION)) == turn_said(short)
    # No reply, and no user words.
    only_user = "User: " + english() + "\nmelo: "
    said = turn_said(only_user, Shown.of(200, QUESTION))
    assert "tomato harvest peaks" in said and "melo replied" not in said
    only_reply = "User: \nmelo: " + english()
    said = turn_said(only_reply, Shown.of(200, QUESTION))
    assert "tomato harvest peaks" in said and "the user said" not in said


def test_the_users_words_stay_whole_in_forty_percent_and_are_windowed_beyond() -> None:
    reply = english(
        400, 300
    )  # with the user's words, a turn is at most 2000 characters
    room = 400
    fits = "Which tomato harvest, the one in August?"  # 40 characters
    user, answer = parts(
        turn_said(document_text(fits, reply), Shown.of(room, QUESTION))
    )
    assert user == fits and "tomato harvest peaks" in answer
    assert len(user) + len(answer) <= room
    # Words longer than 40% of the room: windowed, within their 40%.
    long_typed = padded(
        "Does the tomato harvest peak before the frost?", 700, 300, "Idle chatter {n}. "
    )
    user, answer = parts(
        turn_said(document_text(long_typed, reply), Shown.of(room, QUESTION))
    )
    assert "tomato harvest peak before the frost" in user
    assert user.startswith("…") and len(user) <= int(room * 0.4)
    assert "tomato harvest peaks" in answer and len(user) + len(answer) <= room
    # A part with no term shows its head, in its share, while the other is matched.
    quiet = words(700, "q")
    user, answer = parts(
        turn_said(document_text(quiet, reply), Shown.of(room, QUESTION))
    )
    assert user == snippet(quiet, int(room * 0.4)) and user.endswith("…")
    assert "tomato harvest peaks" in answer and len(user) + len(answer) <= room
    # The smallest room leaves the reply at least its minimum.
    user, answer = parts(
        turn_said(document_text(quiet, reply), Shown.of(120, QUESTION))
    )
    assert len(answer) >= 60 and len(user) + len(answer) <= 120


@pytest.mark.parametrize("room", [120, 200, 500, 2000])
def test_a_window_never_takes_more_than_the_room(room: int) -> None:
    for typed, reply, question in (
        (TYPED, english(), QUESTION),
        (english(300, 300), english(900, 900), QUESTION),
        (ZH_QUESTION, chinese(), ZH_QUESTION),
        (words(1000, "u"), english(), QUESTION),
    ):
        user, answer = parts(
            turn_said(document_text(typed, reply), Shown.of(room, question))
        )
        assert len(user) + len(answer) <= room, (typed[:20], room)


def test_the_page_and_the_json_keep_the_turns_text_as_it_was() -> None:
    item = Item(
        "d:1", document_text(TYPED, english()), "chat “Garden”, Oct 30", "s", 1, 1.0
    )
    assert item.to_json()["text"] == snippet(item.text)
    assert item.prompt == item.line() == item.line(show=HEADS)
    shown = Shown.of(200, QUESTION)
    assert item.line(show=shown) != item.line()
    assert "tomato harvest peaks" in item.line(show=shown)
    assert item.to_json()["text"] == snippet(
        item.text
    )  # nothing in it takes the question


def test_the_block_shows_the_window_and_still_respects_its_size_bound() -> None:
    long = Item(
        "d:1", document_text(TYPED, english()), "chat “Garden”, Oct 30", "s", 1, 1.0
    )
    text, used = block([long], chars=900, show=Shown.of(200, QUESTION))
    assert used == [long] and "tomato harvest peaks" in text and len(text) <= 900
    head, _ = block([long], chars=900)
    assert "tomato harvest peaks" not in head
    # A room bigger than the block's bound leaves the item out; the bound holds.
    wide, left = block([long], chars=900, show=Shown.of(1500, QUESTION))
    assert left == [] and wide == "" and len(wide) <= 900
    text, used = block([long], chars=4000, show=Shown.of(1500, QUESTION))
    assert used == [long] and 900 < len(text) <= 4000


async def memory_with(
    tmp_path: Path, store: SessionStore, cfg: object, reply: str
) -> ChatMemory:
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))  # type: ignore[arg-type,return-value]
    turn = await _turn(store, TYPED)
    _, index = memory.stores()
    await index.add(
        Document(
            session_id=turn.session_id,
            turn=1,
            surface="gateway",
            kind="chat",
            title=TYPED,
            text=document_text(TYPED, reply),
            created_at=time.time(),
        )
    )
    return memory


async def test_the_recall_block_follows_the_two_settings(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_chars=4000)
    memory = await memory_with(tmp_path, store, cfg, english())
    try:

        async def ask() -> str:
            found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
            assert len(found.used) == 1
            return found.block

        head = await ask()
        assert "tomato harvest peaks" not in head
        cfg.memory.recall_excerpt = "match"
        match = await ask()
        assert "tomato harvest peaks" in match and "…" in match
        assert TYPED in match and len(match) < 600
        # The room is read again each turn.
        cfg.memory.recall_item_chars = 800
        bigger = await ask()
        assert TARGET in bigger and len(bigger) > len(match) + 300
        cfg.memory.recall_excerpt = "head"
        assert (
            len(await ask()) > len(head) + 300 and "tomato harvest" not in await ask()
        )
        # The size bound of the block still holds.
        cfg.memory.recall_chars = 900
        cfg.memory.recall_item_chars = 2000
        cfg.memory.recall_excerpt = "match"
        found = await memory.before_turn(store, await _turn(store, "Ask"), QUESTION)
        assert len(found.block) <= 900 + 2 and found.used == ()
    finally:
        await memory.aclose()


async def test_the_search_tool_follows_the_two_settings_too(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    memory = await memory_with(tmp_path, store, cfg, english())
    tools = MemoryTools(memory)
    try:
        head = await tools.search(query=QUESTION)
        assert head.startswith("1. ") and "tomato harvest peaks" not in head
        cfg.memory.recall_excerpt = "match"
        match = await tools.search(query=QUESTION)
        assert "tomato harvest peaks" in match and "…" in match
        cfg.memory.recall_item_chars = 800
        assert TARGET in await tools.search(query=QUESTION)
        assert len(await tools.search(query=QUESTION)) > len(match) + 300
        # The default room and the head: the list reads as it did.
        cfg.memory.recall_item_chars = 200
        cfg.memory.recall_excerpt = "head"
        assert await tools.search(query=QUESTION) == head
        # The items found and the page's text are the same whatever the settings.
        found = await memory.search(QUESTION)
        assert [item.to_json()["text"] for item in found] == [
            snippet(item.text) for item in found
        ]
    finally:
        await memory.aclose()
