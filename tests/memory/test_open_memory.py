"""``open_memory``: the model opens the chat turns of the list it was given, whole.

Offered only with ``[memory] recall_form = "list"``; takes 1 to 5 ids such as
``t123`` from a list line and returns each turn with every chunk joined in
order, its date and chat, up to ``recall_open_chars`` (cut at a sentence end and
marked). An unknown id, a deleted chat's turn, a note's id, a turn the chat shows
already and anything else that is not a list id open nothing. The turn's memory
event names what was opened. Synthetic turns, words only, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import MAX_CHUNKS, document_chunks, is_cut
from localmelo.support.memory.recall import (
    CUT_MARK,
    OLDER_TURN,
    WHOLE,
    Item,
    cut_at_sentence,
    day,
    turn_said,
    whole_turn,
)
from localmelo.support.memory.service import (
    CHAT_TURN,
    IN_THIS_CHAT,
    NO_SUCH_TURN,
    OPEN_MEMORY,
    OPEN_USAGE,
    MemoryTools,
)
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import teach
from localmelo.tests.memory.test_recall_list import (
    DAY,
    _turn,
    add_turn,
    config,
    lines_of,
)

SENTENCE = "Sentence {n} says to water the tomato plants early in the morning. "
ZH_SENTENCE = "第{n}句话说要在清晨给番茄浇水，并且记得检查土壤的湿度。"


def numbered(size: int, sentence: str = SENTENCE) -> str:
    out, n = "", 0
    while len(out) < size:
        out += sentence.format(n=n)
        n += 1
    return out[:size]


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def body_of(reply: str) -> tuple[str, str]:
    """The heading line and the turn's reading of one opened turn."""
    heading, _, text = reply.partition("\n")
    return heading, text


# ── A turn's chunks, joined ──


@pytest.mark.parametrize("typed_size", [0, 30, 180, 400, 2500, 5200])
@pytest.mark.parametrize("answer_size", [0, 90, 1500, 2100, 4000, 7000, 12000])
@pytest.mark.parametrize("sentence", [SENTENCE, ZH_SENTENCE])
def test_the_chunks_join_back_into_the_turn(
    typed_size: int, answer_size: int, sentence: str
) -> None:
    typed, answer = numbered(typed_size, sentence), numbered(answer_size, sentence)
    chunks = document_chunks(typed, answer)
    whole = whole_turn(chunks)
    expected = f"User: {typed.strip()}\nmelo: {answer.strip()}"
    if is_cut(typed, answer):
        # What the index left out after its last chunk is not here.
        assert expected.startswith(whole) and len(chunks) == MAX_CHUNKS
        assert len(whole) > 0.7 * MAX_CHUNKS * 1800
    else:
        assert whole == expected, (typed_size, answer_size, len(chunks))


def test_a_turn_that_fits_is_its_one_chunk_and_no_chunks_is_empty() -> None:
    chunks = document_chunks("Where is it?", "Over there.")
    assert chunks == ["User: Where is it?\nmelo: Over there."]
    assert whole_turn(chunks) == chunks[0] and whole_turn([]) == ""
    assert (
        whole_turn(["Task: Plan\nUser: a\nmelo: b"]) == "Task: Plan\nUser: a\nmelo: b"
    )


def test_a_task_turn_and_a_long_user_part_are_joined_too() -> None:
    typed, answer = numbered(3000), numbered(3000)
    chunks = document_chunks(typed, answer, title="Water the beds")
    assert chunks[0].startswith("Task: Water the beds\nUser: ")
    whole = whole_turn(chunks)
    assert (
        whole == f"Task: Water the beds\nUser: {typed.strip()}\nmelo: {answer.strip()}"
    )
    reading = turn_said(whole, WHOLE)
    assert reading == (
        f"the user said “{' '.join(typed.split())}”; "
        f"melo replied “{' '.join(answer.split())}”"
    )


def test_text_cut_at_a_sentence_end_carries_the_mark_and_fits() -> None:
    text = numbered(3000)
    assert cut_at_sentence(text, 5000) == text
    for chars in (500, 777, 1500, 2999):
        cut = cut_at_sentence(text, chars)
        assert len(cut) <= chars and cut.endswith(f". {CUT_MARK}")
        assert text.startswith(cut[: -len(CUT_MARK) - 1])
    assert CUT_MARK == "… (cut)"
    # No sentence end in the second half: a word end. No space either: anywhere.
    run_on = "word " * 400
    assert cut_at_sentence(run_on, 600).endswith(f"word {CUT_MARK}")
    solid = "x" * 2000
    assert (
        cut_at_sentence(solid, 600) == "x" * (600 - len(CUT_MARK) - 1) + f" {CUT_MARK}"
    )
    zh = numbered(3000, ZH_SENTENCE)
    assert cut_at_sentence(zh, 700).endswith(f"。 {CUT_MARK}")


# ── The tool ──


async def test_it_is_offered_only_with_the_list_form(tmp_path: Path) -> None:
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        assert "open_memory" not in [
            t.definition.name for t in await MemoryTools(memory).start()
        ]
        cfg.memory.recall_form = "list"
        tools = await MemoryTools(memory).start()
        (opened,) = [t for t in tools if t.definition.name == "open_memory"]
        assert opened.definition is OPEN_MEMORY and not opened.requires_approval
        assert opened.definition.annotations == {"readOnlyHint": True}
        assert OPEN_MEMORY.parameters == {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "Ids from the list, such as t123.",
                }
            },
            "required": ["ids"],
        }
        # English only: an example in another language pulls answers toward it.
        assert OPEN_MEMORY.description.isascii()
        assert all(
            text.isascii()
            for text in (
                str(OPEN_MEMORY.parameters),
                " ".join(OPEN_MEMORY.semantic_tags),
            )
        )
    finally:
        await memory.aclose()


async def test_an_id_opens_the_whole_turn_across_its_chunks_in_order(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list", recall_open_chars=16000)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        answer = numbered(7000)
        at = time.time() - 3 * DAY
        _, ids = await add_turn(
            memory,
            store,
            "Tell me about the garden schedule? And the beds.",
            answer,
            at=at,
            title="Garden schedule",
        )
        assert len(ids) > 3
        # Any chunk's id opens the same turn: the list may name a later chunk.
        replies = [await tools.open_memory(ids=[f"t{n}"]) for n in (ids[0], ids[2])]
        (heading, text), (later, same) = (body_of(r) for r in replies)
        assert heading == f"[t{ids[0]}] {day(at)} · chat “Garden schedule”"
        assert later == f"[t{ids[2]}] {day(at)} · chat “Garden schedule”"
        assert (
            text
            == same
            == (
                "the user said “Tell me about the garden schedule? And the beds.”; "
                f"melo replied “{' '.join(answer.split())}”"
            )
        )
        # Every sentence once, in order.
        at_sentence = [text.index(f"Sentence {n} says") for n in range(100)]
        assert (
            at_sentence == sorted(at_sentence) and text.count("Sentence 99 says") == 1
        )
        # What the list shows of it is a title: the whole turn is more.
        assert len(text) > 6000
    finally:
        await memory.aclose()


async def test_recall_open_chars_cuts_a_turn_at_a_sentence_end_and_marks_it(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list", recall_open_chars=800)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        _, ids = await add_turn(memory, store, "Plan?", numbered(5000))
        _, text = body_of(await tools.open_memory(ids=[f"t{ids[0]}"]))
        assert len(text) <= 800 and text.endswith(f"morning. {CUT_MARK}")
        assert text.startswith("the user said “Plan?”; melo replied “Sentence 0 says")
        cfg.memory.recall_open_chars = 4000  # read again at each call
        _, more = body_of(await tools.open_memory(ids=[f"t{ids[0]}"]))
        assert 3000 < len(more) <= 4000 and more.endswith(f"morning. {CUT_MARK}")
        assert more.startswith(text[: -len(CUT_MARK) - 1])
        # A turn that fits is not marked.
        _, short = await add_turn(memory, store, "Where?", "Over there.")
        _, short_text = body_of(await tools.open_memory(ids=[f"t{short[0]}"]))
        assert short_text == "the user said “Where?”; melo replied “Over there.”"
    finally:
        await memory.aclose()


async def test_an_unknown_or_deleted_id_says_so_in_one_line(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        session_id, ids = await add_turn(memory, store, "Plan?", numbered(5000))
        keep, (kept,) = await add_turn(memory, store, "Garden?", "South.")
        assert await tools.open_memory(ids=["t99999"]) == f"t99999: {NO_SUCH_TURN}"
        assert "\n" not in NO_SUCH_TURN
        assert (await tools.open_memory(ids=[f"t{ids[0]}"])).startswith(f"[t{ids[0]}]")
        # The user deleted the chat: every chunk of its turn is gone.
        assert await memory.forget_session(session_id) > 0
        for number in ids:
            assert await tools.open_memory(ids=[f"t{number}"]) == (
                f"t{number}: {NO_SUCH_TURN}"
            )
        # The other chat's turn is still there.
        assert keep and (await tools.open_memory(ids=[f"t{kept}"])).startswith("[t")
    finally:
        await memory.aclose()


async def test_a_note_a_task_or_anything_else_that_is_no_list_id_opens_nothing(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        await teach(memory, store, await _turn(store, "Home"), "I live in Utrecht.")
        (note,) = await memory.records()
        _, (doc,) = await add_turn(memory, store, "Where?", "Over there.")
        for raw in (f"r{note.id}", f"r:{note.id}", f"[r{note.id}]", "R1"):
            assert await tools.open_memory(ids=[raw]) == (
                f"{raw}: that is a note, which is shown in full already."
            )
        for raw in ("task-7", "7", str(doc), "d:1", "t", "t1x", "../etc/passwd", "t-1"):
            reply = await tools.open_memory(ids=[raw])
            assert reply == (
                f"{raw}: not an id from the list; ids look like t123."
            ), raw
        # A number alone is no list id, even one that is a turn's own number.
        assert "Utrecht" not in await tools.open_memory(ids=[str(note.id)])
        # What the model typed is echoed short.
        long = "x" * 500
        assert len(await tools.open_memory(ids=[long])) < 120
    finally:
        await memory.aclose()


async def test_one_to_five_ids_at_a_time_and_a_list_or_a_string_will_do(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        ids = [
            (await add_turn(memory, store, f"Bed {n}?", f"Answer {n}."))[1][0]
            for n in range(7)
        ]
        assert await tools.open_memory(ids=[]) == OPEN_USAGE
        assert await tools.open_memory() == OPEN_USAGE
        assert await tools.open_memory(ids=[" ", ""]) == OPEN_USAGE
        assert OPEN_USAGE == "Pass 1 to 5 ids from the list, such as t123."
        one = await tools.open_memory(ids=[f"t{ids[0]}"])
        assert one.count("\n") == 1 and "Answer 0." in one
        five = await tools.open_memory(ids=[f"t{n}" for n in ids[:5]])
        parts = five.split("\n\n")
        assert len(parts) == 5
        assert [f"Answer {n}." in part for n, part in enumerate(parts)] == [True] * 5
        # A sixth is not opened, and the reply says which.
        six = await tools.open_memory(ids=[f"t{n}" for n in ids[:7]])
        assert six.split("\n\n")[:5] == parts
        assert six.split("\n\n")[5] == (
            f"Not opened (5 at most at a time): t{ids[5]}, t{ids[6]}."
        )
        assert "Answer 5." not in six
        # The same id twice opens once; the bracketed form and capital T are ids.
        twice = await tools.open_memory(
            ids=[f"t{ids[1]}", f"[T{ids[1]}]", f"t{ids[2]}"]
        )
        assert len(twice.split("\n\n")) == 2
        # A string of ids, and a single string, as a model may send them.
        text = await tools.open_memory(ids=f"t{ids[0]}, t{ids[1]} t{ids[2]}")
        assert len(text.split("\n\n")) == 3
        assert "Answer 3." in await tools.open_memory(ids=f"t{ids[3]}")
        # One bad id among good ones does not stop the others.
        mixed = await tools.open_memory(ids=["t99999", f"t{ids[4]}", "nope"])
        first, second, third = mixed.split("\n\n")
        assert first == f"t99999: {NO_SUCH_TURN}" and "Answer 4." in second
        assert third.startswith("nope: not an id")
    finally:
        await memory.aclose()


async def test_the_search_lines_open_with_their_own_ids(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        await add_turn(memory, store, "Which bed gets the sun?", "Bed two, all day.")
        (line,) = lines_of(await tools.search(query="which bed sun"))
        ident = line[1 : line.index("]")]
        reply = await tools.open_memory(ids=[ident])
        assert reply.startswith(f"[{ident}] ")
        assert reply.endswith(
            "the user said “Which bed gets the sun?”; melo replied “Bed two, all day.”"
        )
        # The id as the model may copy it from the line: with its brackets.
        assert await tools.open_memory(ids=[f"[{ident}]"]) == reply
    finally:
        await memory.aclose()


async def test_a_replaced_turn_reads_as_older_when_opened(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        now = time.time()
        _, (old,) = await add_turn(
            memory, store, "My weekly sync is on Wednesday.", "Noted.", at=now - DAY
        )
        _, (new,) = await add_turn(
            memory, store, "My weekly sync is on Friday.", "Noted.", at=now - 60
        )
        heading, _ = body_of(await tools.open_memory(ids=[f"t{old}"]))
        assert heading.endswith(f"({OLDER_TURN})")
        heading, _ = body_of(await tools.open_memory(ids=[f"t{new}"]))
        assert "older" not in heading
    finally:
        await memory.aclose()


async def test_a_turn_the_chat_shows_already_is_not_opened_and_its_own_past_reads_as_this_chat(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        session_id, (doc,) = await add_turn(
            memory, store, "Where?", "Over there.", turn=3
        )
        shown = await memory.open_turn(
            doc, exclude_session=session_id, keep_from_turn=2
        )
        assert shown == IN_THIS_CHAT and "\n" not in IN_THIS_CHAT
        # Past the window it is the chat's earlier talk, and reads so.
        past = await memory.open_turn(doc, exclude_session=session_id, keep_from_turn=4)
        assert isinstance(past, tuple) and past.item.where.startswith("this chat, ")
        assert past.text == "User: Where?\nmelo: Over there."
        # Another chat's view of it: an ordinary turn.
        other = await memory.open_turn(
            doc, exclude_session="elsewhere", keep_from_turn=0
        )
        assert isinstance(other, tuple) and other.item.where.startswith("chat “")
        # Without the window nothing is held back, as the search tool does.
        assert isinstance(await memory.open_turn(doc), tuple)
    finally:
        await memory.aclose()


async def test_a_chat_turn_that_opens_what_it_shows_is_refused_through_the_tool(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        info = await _turn(store, "Garden")
        _, (doc,) = await add_turn(
            memory, store, "Where?", "Over there.", session_id=info.session_id, turn=0
        )
        await memory.before_turn(store, info, "hi there", window=[])
        chat = CHAT_TURN.get()
        assert chat is not None and chat.keep_from_turn == info.turn
        # Turn 0 is before this turn and the window holds none of it: it is the
        # chat's own past, and opens.
        assert (await tools.open_memory(ids=[f"t{doc}"])).startswith(f"[t{doc}]")
        chat.keep_from_turn = 0  # the window shows it
        assert await tools.open_memory(ids=[f"t{doc}"]) == f"t{doc}: {IN_THIS_CHAT}"
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


# ── The turn's memory event ──


async def test_the_turns_memory_event_names_what_was_opened(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        sid, ids = await add_turn(memory, store, "Plan?", numbered(5000))
        other, (second,) = await add_turn(memory, store, "Garden?", "South.")
        emitted: list[dict[str, Any]] = []
        typed = "hi there"  # nothing here recalls automatically
        await memory.before_turn(
            store, await _turn(store, typed), typed, emit=emitted.append
        )
        chat = CHAT_TURN.get()
        assert chat is not None
        data = chat.event()
        assert data is not None and "opened" not in data and emitted == []
        await tools.open_memory(ids=[f"t{ids[0]}"])
        (data,) = emitted
        assert data["opened"] == [{"session_id": sid, "turn": 1, "id": ids[0]}]
        # The same turn by another chunk, and a bad id: nothing new, nothing sent.
        await tools.open_memory(ids=[f"t{ids[1]}", "t99999", "r1", "nope"])
        assert len(emitted) == 1
        await tools.open_memory(ids=[f"t{second}", f"t{ids[0]}"])
        assert len(emitted) == 2
        assert emitted[-1]["opened"] == [
            {"session_id": sid, "turn": 1, "id": ids[0]},
            {"session_id": other, "turn": 1, "id": second},
        ]
        assert set(emitted[-1]["opened"][0]) == {"session_id", "turn", "id"}
        # The rest of the event is the turn's as it was.
        assert emitted[-1]["used"] == [] and emitted[-1]["remembered"] == []
        assert "searched" not in emitted[-1]
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_opening_outside_a_chat_turn_changes_no_event_and_with_memory_off_says_so(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    tools = MemoryTools(memory)
    try:
        _, (doc,) = await add_turn(memory, store, "Where?", "Over there.")
        CHAT_TURN.set(None)
        assert (await tools.open_memory(ids=[f"t{doc}"])).startswith(f"[t{doc}]")
        cfg.memory.across_chats = False
        assert await tools.open_memory(ids=[f"t{doc}"]) == (
            "Memory across chats is off."
        )
    finally:
        await memory.aclose()


def test_an_item_heading_names_its_id_date_and_chat() -> None:
    at = time.time()
    item = Item("d:12", "x", f"chat “Fix CI”, {day(at)}", "s", 1, at)
    assert item.opened_heading() == f"[t12] {day(at)} · chat “Fix CI”"
    older = Item("d:12", "x", f"Main chat, {day(at)}", "s", 1, at, older=OLDER_TURN)
    assert older.opened_heading() == f"[t12] {day(at)} · Main chat ({OLDER_TURN})"
