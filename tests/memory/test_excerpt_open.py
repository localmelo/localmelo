"""A cut excerpt can be opened whole (#428).

In the excerpt form of chat turns, a line that shows less than the whole turn (the
excerpt was cut, or the document is a piece of a longer turn) starts with the turn's
id, ``[t123]``, counted in the line's room; a line that shows the whole turn, and
every note, reads as it always did. One guide line follows the excerpts when at least
one line has an id, in the block and in the search result, and ``open_memory`` is
offered. The list form and memory items are not affected. Synthetic notes and turns,
words only, no model.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import (
    BLOCK_CHARS,
    BLOCK_HEADER,
    CUT_GUIDE,
    HEADS,
    ITEM_CHARS,
    LIST_GUIDE,
    WHOLE,
    Item,
    Shown,
    day,
    snippet,
    turn_excerpt,
    turn_said,
)
from localmelo.support.memory.service import (
    CHAT_TURN,
    IN_THIS_CHAT,
    NO_SUCH_TURN,
    OPEN_MEMORY,
    REMEMBER,
    SEARCH_MEMORY,
    MemoryTools,
    RecallSettings,
    search_memory_opening,
)
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import teach
from localmelo.tests.memory.test_open_memory import numbered
from localmelo.tests.memory.test_recall_list import (
    DAY,
    NOTE,
    QUESTION,
    _turn,
    add_turn,
    ask,
    config,
)

# One chunk (a turn up to 2000 characters is one document) that an excerpt cuts.
LONG_ANSWER = numbered(1500)
SHORT_TYPED = "Which tomato garden bed gets the afternoon sun?"
CHAT_LINE = r"^\[t(\d+)\] "


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def memory_of(tmp_path: Path, **settings: object) -> tuple[ChatMemory, Config]:
    """A memory with these ``[memory]`` settings."""
    cfg = config(**settings)
    return ChatMemory(lambda: cfg, directory=str(tmp_path / "memory")), cfg


def room_used(said: str) -> int:
    """The characters of a turn's reading that are the turn's own words: what the
    user said and melo replied, without the framing words and quotes."""
    parts = re.fullmatch(
        r"(?:the user said “(?P<typed>.*?)”)?(?:; )?(?:melo replied “(?P<answer>.*)”)?",
        said,
        re.S,
    )
    assert parts is not None, said
    return len(parts.group("typed") or "") + len(parts.group("answer") or "")


def turn_lines(text: str) -> list[str]:
    """The ``- …`` lines of a block that are chat turns: not the note, the header or
    a guide."""
    return [
        line[2:]
        for line in text.split("\n")
        if line.startswith("- ")
        and NOTE not in line
        and re.search(r"\((?:this )?chat", line)
    ]


# ── The effect on one line ──


def test_excerpts_open_but_the_list_form_and_items_do_not() -> None:
    assert RecallSettings().opens_excerpts
    # The list form always offers open_memory and shows no excerpt; items are not turns.
    assert not RecallSettings(listing=True).opens_excerpts
    assert not RecallSettings(items=True).opens_excerpts
    assert Shown.of(300, None, True).ids and not Shown.of(300).ids
    assert RecallSettings().shown("tomato").ids
    assert not RecallSettings(listing=True).shown("tomato").ids


def test_turn_excerpt_says_whether_it_shows_less_than_the_whole_text() -> None:
    turn = "User: Where is it?\nmelo: Over there."
    assert turn_excerpt(turn, HEADS) == (
        "the user said “Where is it?”; melo replied “Over there.”",
        False,
    )
    # The two parts together just fit the room of 234: whole.
    fits = f"User: {'a' * 150}\nmelo: {'b' * 84}"
    assert turn_excerpt(fits, HEADS)[1] is False
    # One character more: melo's reply is cut.
    said, cut = turn_excerpt(f"{fits}b", HEADS)
    assert cut and said.endswith("…”") and "a" * 150 in said
    # The user's words are cut, and the reply has no room left: it is left out.
    said, cut = turn_excerpt(f"User: {numbered(900)}\nmelo: Over there.", HEADS)
    assert cut and said.endswith("…”") and "melo replied" not in said
    # The user's words fit, the room left for the reply is too small: left out.
    said, cut = turn_excerpt(f"User: {'a' * 230}\nmelo: Yes, it is.", HEADS)
    assert cut and "melo replied" not in said and "a" * 230 in said
    # No user words and a long reply: the reply is cut.
    said, cut = turn_excerpt("User: \nmelo: " + numbered(700), HEADS)
    assert cut and said.startswith("melo replied “") and said.endswith("…”")
    # A summary or a task's exchange is a plain snippet: cut only when it is longer.
    short = "Summary of chat “Fix CI”: the port was taken."
    assert turn_excerpt(short, HEADS) == (short, False)
    assert turn_excerpt("Summary: " + numbered(900), HEADS)[1] is True
    assert turn_excerpt("User: no reply marker here " * 20, HEADS)[1] is True
    # A matching window of a long part is a cut too.
    found = "User: Plan?\nmelo: " + numbered(800) + " The zucchini harvest peaks."
    window, cut = turn_excerpt(found, Shown.of(300, "zucchini harvest"))
    assert cut and "zucchini harvest peaks" in window
    # In a room that holds all of it, nothing is cut; turn_said is the same reading.
    long = "User: Where?\nmelo: " + numbered(900)
    assert turn_excerpt(long, WHOLE)[1] is False
    assert turn_said(long, HEADS) == turn_excerpt(long, HEADS)[0]


def test_a_reserved_id_takes_its_characters_from_the_room_of_a_cut_line() -> None:
    for chars in (120, 234, 300, 900, 2000):
        show = Shown.of(chars, None, True)
        for typed_size in (0, 40, 400, 1500):
            for answer_size in (0, 60, 700, 3000):
                text = f"User: {numbered(typed_size)}\nmelo: {numbered(answer_size)}"
                said, cut = turn_excerpt(text, show)
                if not cut:
                    continue
                for reserve in (7, 8, 9):
                    again, still = turn_excerpt(text, show, reserve)
                    # Still cut, within the room less the id, and never more text
                    # than the line without an id shows.
                    assert still
                    assert room_used(again) <= chars - reserve, (chars, typed_size)
                    assert room_used(again) <= room_used(said)
    # At a room of 234 (the head a turn had before the room grew) the user's words
    # keep their 234, as far as the room holds them: with a 7-character id they have
    # 7 less, and the reply is left out.
    text = f"User: {'a' * 300}\nmelo: {'b' * 300}"
    assert turn_excerpt(text, Shown.of(234, None, True))[0] == (
        f"the user said “{'a' * 233}…”"
    )
    assert turn_excerpt(text, Shown.of(234, None, True), 7)[0] == (
        f"the user said “{'a' * 226}…”"
    )
    # A turn with a short question keeps the same split with and without the id.
    text = f"User: {'a' * 100}\nmelo: {'b' * 300}"
    assert turn_excerpt(text, Shown.of(234, None, True))[0] == (
        f"the user said “{'a' * 100}”; melo replied “{'b' * 133}…”"
    )
    assert turn_excerpt(text, Shown.of(234, None, True), 7)[0] == (
        f"the user said “{'a' * 100}”; melo replied “{'b' * 126}…”"
    )


def test_a_line_with_an_id_reads_as_the_line_without_it_otherwise() -> None:
    def item(text: str, **more: object) -> Item:
        return Item(
            "d:42",
            text,
            "chat “Garden”, Oct 30",
            "s",
            1,
            1.0,
            **more,  # type: ignore[arg-type]
        )

    now = 1.0
    show = Shown.of(234, None, True)
    whole = item("User: Where is it?\nmelo: Over there.")
    assert whole.excerpt(now, show=show) == (whole.line(now), False)
    assert "[t42]" not in whole.line(now, show=show)
    cut = item("User: Where?\nmelo: " + numbered(900))
    said, carries = cut.excerpt(now, show=show)
    assert carries and said.startswith("[t42] the user said “Where?”; melo replied “")
    assert said.endswith("(chat “Garden”, Oct 30)")
    assert cut.line(now, show=show) == said
    # Without ids in the show it is today's line, cut or not.
    assert cut.excerpt(now, show=HEADS) == (cut.line(now), False)
    assert cut.line(now) == cut.line(now, show=HEADS) and "[t" not in cut.line(now)
    # A piece of a longer turn carries it even when it shows the whole piece.
    piece = item("User: Where…\nmelo: and then it goes on.", chunk=2)
    assert piece.excerpt(now, show=show)[0].startswith("[t42] ")
    head = item("User: Where?\nmelo: First part.", more=True)
    assert head.excerpt(now, show=show)[0].startswith("[t42] ")
    # A summary or an exchange is a turn document too: an id when its text is cut.
    long_summary = item("Summary of chat “Fix CI”: " + numbered(900))
    assert long_summary.excerpt(now, show=show)[0].startswith("[t42] Summary")
    assert not item("Summary: short.").excerpt(now, show=show)[1]
    # A note and a memory item never carry one.
    note = Item("r:3", "You stake the garden.", "chat “Stakes”, Oct 30", "s", 1, 1.0)
    assert note.excerpt(now, show=show) == (note.line(now), False)
    memory_item = Item(
        "m:7", numbered(900), "chat “Garden”, Oct 30", "s", 1, 1.0, title="Beds"
    )
    assert memory_item.excerpt(now, show=show) == (memory_item.line(now), False)
    assert not memory_item.line(now, show=show).startswith("[")


# ── The block and the tools ──


@pytest.mark.parametrize("configured", [False, True])
async def test_with_no_setting_a_cut_line_has_an_id_the_guide_and_open_memory(
    tmp_path: Path, store: SessionStore, configured: bool
) -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    memory = ChatMemory(
        (lambda: cfg) if configured else (lambda: None),
        directory=str(tmp_path / "memory"),
    )
    try:
        assert memory.recall_settings().opens_excerpts
        _, (long_id,) = await add_turn(
            memory, store, SHORT_TYPED, LONG_ANSWER, title="Long"
        )
        found = await ask(memory, store)
        (cut,) = turn_lines(found.block)
        assert cut.startswith(f"[t{long_id}] the user said “{SHORT_TYPED}”")
        assert found.block.endswith(f"\n{CUT_GUIDE}")
        reply = await MemoryTools(memory).search(query=QUESTION)
        assert f"[t{long_id}] " in reply and reply.endswith(CUT_GUIDE)
        tools = await MemoryTools(memory).start()
        assert [t.definition.name for t in tools] == [
            "search_memory",
            "remember",
            "open_memory",
        ]
        assert tools[0].definition.description.endswith(CUT_GUIDE)
        assert tools[2].definition is OPEN_MEMORY
    finally:
        await memory.aclose()


# ── On: the block ──


async def test_a_cut_turn_starts_with_its_id_and_a_whole_turn_has_none(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    try:
        await teach(memory, store, await _turn(store, "Stakes"), NOTE)
        at = time.time() - 3 * DAY
        _, (long_id,) = await add_turn(
            memory, store, SHORT_TYPED, LONG_ANSWER, at=at, title="Long"
        )
        _, (short_id,) = await add_turn(
            memory,
            store,
            "Which tomato garden bed needs stakes?",
            "Bed two needs them.",
            at=at - DAY,
            title="Short",
        )
        found = await ask(memory, store)
        cut, whole = turn_lines(found.block)
        # The id is counted in the room: the room less its 7 or more characters.
        left = ITEM_CHARS - len(f"[t{long_id}] ") - len(SHORT_TYPED)
        assert cut == (
            f"[t{long_id}] the user said “{SHORT_TYPED}”; melo replied “"
            f"{snippet(' '.join(LONG_ANSWER.split()), left)}” (chat “Long”, {day(at)})"
        )
        assert whole == (
            "the user said “Which tomato garden bed needs stakes?”; melo replied "
            f"“Bed two needs them.” (chat “Short”, {day(at - DAY)})"
        )
        assert f"[t{short_id}]" not in found.block
        # The guide follows the excerpts, once, as one line.
        assert found.block.count(CUT_GUIDE) == 1
        assert found.block.endswith(f"\n{CUT_GUIDE}") and "\n" not in CUT_GUIDE
        assert CUT_GUIDE == (
            "A line that starts with an id such as [t123] is cut; open it with "
            "open_memory when the answer may be in the rest."
        )
        assert found.block.startswith(f"\n\n{BLOCK_HEADER}\n- the user said “{NOTE}”")
        # A note never carries an id. The entries hold the exact lines the model
        # read, id included and without the dash.
        (note,) = (item for item in found.used if item.is_record)
        assert not note.shown.startswith("[") and "[t" not in note.shown
        shown = {item.key: item.shown for item in found.used if not item.is_record}
        assert shown == {f"d:{long_id}": cut, f"d:{short_id}": whole}
        entries = [e for e in found.event()["used"] if e["kind"] == "chat"]  # type: ignore[index]
        assert [e["shown"] for e in entries] == [cut, whole]
        assert entries[0]["text"].startswith("User: ")
        assert entries[0]["id"] == long_id
    finally:
        await memory.aclose()


async def test_the_guide_appears_only_when_a_line_carries_an_id(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    tools = MemoryTools(memory)
    try:
        await teach(memory, store, await _turn(store, "Stakes"), NOTE)
        await add_turn(
            memory, store, "Which tomato garden bed gets sun?", "Bed two, all day."
        )
        found = await ask(memory, store)
        assert len(turn_lines(found.block)) == 1 and "[t" not in found.block
        assert CUT_GUIDE not in found.block
        reply = await tools.search(query=QUESTION)
        assert "[t" not in reply and CUT_GUIDE not in reply
        # A note and nothing else: no guide either.
        only = await ask(memory, store, "how do I stake the tomato garden?")
        assert CUT_GUIDE not in only.block and NOTE in only.block
        # One long turn: the guide, in the block and in the result.
        await add_turn(memory, store, SHORT_TYPED, LONG_ANSWER, title="Long")
        found = await ask(memory, store)
        assert found.block.count(CUT_GUIDE) == 1
        lines = (await tools.search(query=QUESTION)).split("\n")
        assert lines[-1] == CUT_GUIDE and lines.count(CUT_GUIDE) == 1
        assert [line.split(". ")[0] for line in lines[:-1]] == [
            str(n) for n in range(1, len(lines))
        ]
    finally:
        await memory.aclose()


async def test_a_chunk_of_a_longer_turn_carries_its_id_even_when_its_text_is_short(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    try:
        filler = "".join(f"Calm weather note {n} came and went. " for n in range(120))
        answer = filler + "The zucchini harvest peaks in late August. " + filler
        _, ids = await add_turn(
            memory, store, "Tell me about the garden schedule?", answer
        )
        assert len(ids) > 2
        # A later chunk matched: its line is a piece of the turn, with the chunk's id.
        found = await ask(memory, store, "zucchini harvest")
        (line,) = turn_lines(found.block)
        match = re.match(CHAT_LINE, line)
        assert match is not None and int(match.group(1)) in ids[1:]
        assert found.block.count(CUT_GUIDE) == 1
        # The head matched, in a room that shows all of it: the turn goes on in a later
        # chunk, so the line still carries the id.
        items = memory._corpus.items.values()
        (head,) = (i for i in items if i.chunk == 0)
        assert head.more and not any(i.more for i in items if i.chunk)
        assert turn_excerpt(head.text, Shown.of(2000))[1] is False
        found = await ask(memory, store, "calm weather note")
        (line,) = turn_lines(found.block)
        assert line.startswith(f"[t{ids[0]}] the user said “Tell me about")
        assert found.block.count(CUT_GUIDE) == 1
    finally:
        await memory.aclose()


async def test_the_more_mark_follows_the_chunks_when_a_turn_is_indexed_again(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    typed = "Where is the shed key?"
    try:
        session_id, _ = await add_turn(memory, store, typed, "Over there.")

        async def heads() -> list[bool]:
            # A question with content of its own: one with none (a follow-up such as
            # "where") is searched with the chat's recent words, and a new chat has
            # none, so nothing would be searched and the corpus not filled.
            await ask(memory, store, "shed key")
            items = memory._corpus.items.values()
            return [i.more for i in items if i.is_turn and i.chunk == 0]

        assert await heads() == [False]
        # The same turn, longer: new chunks, the head knows.
        await add_turn(
            memory, store, typed, numbered(5000), session_id=session_id, turn=1
        )
        assert await heads() == [True]
        # Shorter again: one chunk, nothing more.
        await add_turn(
            memory, store, typed, "Over there.", session_id=session_id, turn=1
        )
        assert await heads() == [False]
        assert all(not i.chunk for i in memory._corpus.items.values() if i.is_turn)
    finally:
        await memory.aclose()


async def test_a_whole_turn_that_fits_the_room_has_no_id_even_when_long(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    try:
        answer = numbered(700)
        await add_turn(memory, store, "Where do the beds get sun?", answer)
        found = await ask(memory, store, "beds sun")
        (line,) = turn_lines(found.block)
        # 700 characters fit the room of 900 whole: no id, no guide, today's line.
        assert "[t" not in found.block and CUT_GUIDE not in found.block
        assert " ".join(answer.split()) in line
        (head,) = (i for i in memory._corpus.items.values() if i.is_turn)
        assert head.chunk == 0 and not head.more
    finally:
        await memory.aclose()


async def test_the_block_and_each_line_stay_within_their_room(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    try:
        await teach(memory, store, await _turn(store, "Stakes"), NOTE)
        for n in range(2):
            await add_turn(
                memory,
                store,
                f"Which tomato garden bed {n} gets the sun?",
                numbered(1500 + 700 * n),
                at=time.time() - (n + 2) * 3 * DAY,
                title=f"Bed {n}",
            )
        found = await ask(memory, store)
        text = found.block.removeprefix("\n\n")
        assert len(text) <= BLOCK_CHARS
        ided = [line for line in text.split("\n") if re.match(r"^- \[t\d+\] ", line)]
        # The guide is there exactly when an id is, and it counts in the room.
        assert len(ided) == 2 and text.count(CUT_GUIDE) == 1
        for line in ided:
            prefix = re.match(r"^- (\[t\d+\] )", line)
            assert prefix is not None
            said = line[len("- ") + len(prefix.group(1)) :].rsplit(" (", 1)[0]
            assert room_used(said) <= ITEM_CHARS - len(prefix.group(1)), line
        assert [i.shown for i in found.used if not i.is_record] == [
            line[2:]
            for line in text.split("\n")
            if line.startswith("- ") and NOTE not in line and "(chat" in line
        ]
    finally:
        await memory.aclose()


# ── On: the tools ──


async def test_open_memory_is_offered_and_search_says_what_an_id_line_is(
    tmp_path: Path,
) -> None:
    memory, cfg = memory_of(tmp_path)
    try:
        tools = await MemoryTools(memory).start()
        assert [t.definition.name for t in tools] == [
            "search_memory",
            "remember",
            "open_memory",
        ]
        search, remember, opener = tools
        assert opener.definition is OPEN_MEMORY and not opener.requires_approval
        assert opener.definition.annotations == {"readOnlyHint": True}
        assert remember.definition is REMEMBER
        assert (
            search.definition.description == f"{SEARCH_MEMORY.description} {CUT_GUIDE}"
        )
        assert search.definition.parameters == SEARCH_MEMORY.parameters
        assert search.definition.parameters is not SEARCH_MEMORY.parameters
        assert search.definition.annotations == {"readOnlyHint": True}
        assert search_memory_opening().description == search.definition.description
        assert CUT_GUIDE.isascii()  # English only
        # Memory items are not turns: nothing is offered for them. The tool list is
        # read when an agent starts, from the setting then.
        cfg.memory.recall_source = "items"
        assert [t.definition.name for t in await MemoryTools(memory).start()] == [
            "search_memory",
            "remember",
        ]
    finally:
        await memory.aclose()


async def test_an_id_from_the_block_or_the_search_opens_the_whole_turn_cut_at_recall_open_chars(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, cfg = memory_of(tmp_path, recall_open_chars=1000)
    tools = MemoryTools(memory)
    try:
        at = time.time() - 3 * DAY
        _, (doc,) = await add_turn(
            memory, store, SHORT_TYPED, LONG_ANSWER, at=at, title="Long"
        )
        found = await ask(memory, store)
        (line,) = turn_lines(found.block)
        ident = f"t{doc}"
        assert line.startswith(f"[{ident}] ")
        for given in (ident, f"[{ident}]", ident.upper()):
            reply = await tools.open_memory(ids=[given])
            heading, _, text = reply.partition("\n")
            assert heading == f"[{ident}] {day(at)} · chat “Long”"
            # More than the excerpt, up to recall_open_chars, cut at a sentence end.
            assert text.startswith(f"the user said “{SHORT_TYPED}”; melo replied “")
            assert len(text) <= 1000 and text.endswith("… (cut)")
            assert len(text) > len(line)
        cfg.memory.recall_open_chars = 4000  # read again at each call
        more = await tools.open_memory(ids=[ident])
        assert len(more) > len(reply) and not more.endswith("(cut)")
        # The tail the excerpt did not show is in the opened turn.
        assert "Sentence 20 says" in more and "Sentence 20 says" not in line
        assert " ".join(LONG_ANSWER.split()) in more
        # The search result's id opens the same turn.
        result = await tools.search(query=QUESTION)
        (searched,) = [r for r in result.split("\n") if re.match(r"\d+\. \[t", r)]
        assert searched.startswith(f"1. [{ident}] ")
        assert await tools.open_memory(ids=[ident]) == more
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_the_id_of_a_chunk_opens_the_whole_turn_and_a_turn_never_shown_is_as_before(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path, recall_open_chars=16000)
    tools = MemoryTools(memory)
    try:
        filler = "".join(f"Calm weather note {n} came and went. " for n in range(150))
        answer = filler + "The zucchini harvest peaks in late August. " + filler
        _, ids = await add_turn(memory, store, "Tell me about the garden?", answer)
        _, (other,) = await add_turn(memory, store, "Where?", "Over there.")
        found = await ask(memory, store, "zucchini harvest")
        (line,) = turn_lines(found.block)
        match = re.match(CHAT_LINE, line)
        assert match is not None and int(match.group(1)) in ids[1:]
        reply = await tools.open_memory(ids=[f"t{match.group(1)}"])
        assert reply.count("The zucchini harvest peaks") == 1  # the chunks, joined
        assert reply.partition("\n")[2].startswith("the user said “Tell me about")
        # The head's id opens the same turn.
        assert await tools.open_memory(ids=[f"t{ids[0]}"]) == reply.replace(
            f"[t{match.group(1)}]", f"[t{ids[0]}]", 1
        )
        # A turn the block never showed opens as it does for the list form; an id
        # that names nothing, a note and a word are refused as they always were.
        assert (await tools.open_memory(ids=[f"t{other}"])).startswith(f"[t{other}]")
        assert await tools.open_memory(ids=["t99999", "r:1", "nope"]) == (
            f"t99999: {NO_SUCH_TURN}\n\n"
            "r:1: that is a note, which is shown in full already.\n\n"
            "nope: not an id from the list; ids look like t123."
        )
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_the_chats_own_past_has_an_id_and_opens_until_the_window_holds_it(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    tools = MemoryTools(memory)
    try:
        info = await _turn(store, "Garden")
        _, (doc,) = await add_turn(
            memory,
            store,
            SHORT_TYPED,
            LONG_ANSWER,
            session_id=info.session_id,
            turn=0,
        )
        # The window holds none of turn 0: the chat's own past, recalled as another
        # chat's turn is and read as this chat's.
        found = await memory.before_turn(store, info, QUESTION, window=[])
        (line,) = turn_lines(found.block)
        assert line.startswith(f"[t{doc}] the user said “{SHORT_TYPED}”")
        assert "(this chat, " in line and found.block.count(CUT_GUIDE) == 1
        assert (await tools.open_memory(ids=[f"t{doc}"])).startswith(f"[t{doc}] ")
        chat = CHAT_TURN.get()
        assert chat is not None
        chat.keep_from_turn = 0  # the window shows it
        assert await tools.open_memory(ids=[f"t{doc}"]) == f"t{doc}: {IN_THIS_CHAT}"
        # A turn the window shows is not recalled at all, so it has no id to open.
        again = await memory.before_turn(store, info, QUESTION, window=None)
        assert "[t" not in again.block and CUT_GUIDE not in again.block
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_the_memory_event_names_what_was_opened_as_for_the_list_form(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path)
    tools = MemoryTools(memory)
    emitted: list[dict[str, Any]] = []
    try:
        sid, (doc,) = await add_turn(
            memory, store, SHORT_TYPED, LONG_ANSWER, title="Long"
        )
        info = await _turn(store, QUESTION)
        found = await memory.before_turn(store, info, QUESTION, emit=emitted.append)
        (line,) = turn_lines(found.block)
        assert emitted == []
        await tools.open_memory(ids=[f"t{doc}"])
        (data,) = emitted
        assert data["opened"] == [{"session_id": sid, "turn": 1, "id": doc}]
        used = [e for e in data["used"] if e["kind"] == "chat"]
        assert [e["shown"] for e in used] == [line]
        assert line.startswith(f"[t{doc}] ")
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


# ── The list form and items are as they are ──


async def test_the_list_form_has_no_excerpt_ids_and_offers_open_memory(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _ = memory_of(tmp_path, recall_form="list")
    try:
        await teach(memory, store, await _turn(store, "Stakes"), NOTE)
        await add_turn(
            memory, store, SHORT_TYPED, LONG_ANSWER, at=1_700_000_000.0, title="Long"
        )
        found = await ask(memory, store)
        tools = await MemoryTools(memory).start()
        result = await MemoryTools(memory).search(query=QUESTION)
        assert CUT_GUIDE not in found.block and CUT_GUIDE not in result
        assert LIST_GUIDE in found.block and LIST_GUIDE in result
        assert [t.definition.name for t in tools] == [
            "search_memory",
            "remember",
            "open_memory",
        ]
        assert not memory.recall_settings().opens_excerpts
    finally:
        await memory.aclose()


async def test_with_memory_items_it_changes_nothing(tmp_path: Path) -> None:
    memory, _ = memory_of(tmp_path, recall_source="items")
    try:
        assert not memory.recall_settings().opens_excerpts
        tools = await MemoryTools(memory).start()
        assert [t.definition.name for t in tools] == ["search_memory", "remember"]
        assert tools[0].definition is SEARCH_MEMORY
        assert not memory.recall_settings().shown("tomato").ids
    finally:
        await memory.aclose()
