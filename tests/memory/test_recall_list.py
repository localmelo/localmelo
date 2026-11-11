"""Recall as a list the model opens: ``[memory] recall_form = "list"``.

Notes read as they always did. A chat turn becomes one line, ``[t12] Oct 30 ·
Main chat · “title” · 0.71``, under one line that says to open the turns that
may hold the answer with ``open_memory`` (tests/memory/test_open_memory.py), in
the recall block and in the memory search tool alike. Every turn that passes the
bars is listed, ``recall_list_max`` at most, whatever the excerpt form's bounds
and a chat's cap say. The default form, ``excerpt``, is unchanged. Synthetic notes,
turns and fixed vectors, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

from localmelo.support.config import Config, MemoryConfig
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_chunks
from localmelo.support.memory.recall import (
    AUTO_RECORDS,
    BLOCK_HEADER,
    LIST_GUIDE,
    LIST_MAX,
    LIST_TITLE_CHARS,
    OLDER_TURN,
    OPEN_CHARS,
    PREMISE_GUIDE,
    Item,
    day,
    gate_terms,
    listed_title,
    meaning_of,
    pick,
    word_cut,
)
from localmelo.support.memory.service import (
    CHAT_TURN,
    OPEN_MEMORY,
    REMEMBER,
    SEARCH_MEMORY,
    MemoryTools,
    RecallSettings,
    TurnMemory,
    search_memory_listing,
)
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import teach
from localmelo.tests.memory.test_recall_bars import MODEL, FixedEmbedding, vector_at

HOLDER = "test-host:1:recall-list"
QUESTION = "what do I plant in the tomato garden?"
NOTE = "I stake the tomato garden with bamboo."
NOTHING_FOUND = "Nothing in melo's memory matches that."
DAY = 86_400.0
# The line above a list of turns: its guide, then the premise check's sentence.
GUIDE = f"{LIST_GUIDE} {PREMISE_GUIDE}"


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def config(**memory: object) -> Config:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return cfg


async def _turn(store: SessionStore, title: str):  # type: ignore[no-untyped-def]
    return await store.begin_turn(
        None, title, holder=HOLDER, title=title, new_session=NewSession()
    )


async def add_turn(
    memory: ChatMemory,
    store: SessionStore,
    typed: str,
    answer: str,
    *,
    at: float | None = None,
    cosine: float | None = None,
    session_id: str | None = None,
    turn: int = 1,
    title: str | None = None,
) -> tuple[str, list[int]]:
    """Index one turn (every chunk of it) in a chat of its own, or in *session_id*;
    the session id and the ids of the turn's documents, chunk 0 first. With
    *cosine*, each chunk has a vector at that cosine to the far question."""
    if session_id is None:
        session_id = (await _turn(store, title or typed)).session_id
    _, index = memory.stores()
    before = await index.ids()
    when = at if at is not None else time.time()
    await index.add_many(
        [
            Document(
                session_id=session_id,
                turn=turn,
                surface="gateway",
                kind="chat",
                title=title or typed,
                text=text,
                created_at=when,
                vector=tuple(vector_at(cosine)) if cosine is not None else None,
                embed_model=MODEL if cosine is not None else "",
                chunk=n,
            )
            for n, text in enumerate(document_chunks(typed, answer))
        ]
    )
    return session_id, sorted(await index.ids() - before)


def lines_of(text: str) -> list[str]:
    """The list lines of a block or a search result: ``[t12] …``."""
    return [line for line in text.split("\n") if line.startswith("[t")]


def ids_of(text: str) -> list[int]:
    return [int(line[2 : line.index("]")]) for line in lines_of(text)]


async def ask(
    memory: ChatMemory, store: SessionStore, typed: str = QUESTION
) -> TurnMemory:
    return await memory.before_turn(store, await _turn(store, "Ask"), typed)


class Filled(NamedTuple):
    """A memory with a note and three turns the question finds, words only: the
    turns are on three other days than the note, so no date shows a time of day."""

    memory: ChatMemory
    ids: list[int]
    now: float

    def at(self, n: int) -> float:
        """When turn *n* was said."""
        return self.now - (n + 1) * 3 * DAY


async def filled(tmp_path: Path, store: SessionStore, cfg: Config) -> Filled:
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    await teach(memory, store, await _turn(store, "Stakes"), NOTE)
    now = time.time()
    fill = Filled(memory, [], now)
    for n in range(3):
        _, doc_ids = await add_turn(
            memory,
            store,
            f"Which tomato garden bed {n} gets the sun?",
            f"Bed {n} faces south all day, so it gets the most sun.",
            at=fill.at(n),
        )
        fill.ids.extend(doc_ids)
    return fill


def note_line(memory: ChatMemory) -> str:
    """How the note NOTE reads to the model, with where and when it was kept."""
    (item,) = (i for i in memory._corpus.items.values() if i.is_record)
    return f"the user said “{NOTE}” (chat “Stakes”, {day(item.created_at)})"


# ── The default is unchanged ──


async def test_the_excerpt_form_is_the_default_and_reads_exactly_as_it_did(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config()
    assert cfg.memory.recall_form == "excerpt"
    assert (cfg.memory.recall_list_max, cfg.memory.recall_open_chars) == (20, 4000)
    assert (LIST_MAX, OPEN_CHARS) == (20, 4000)
    fill = await filled(tmp_path, store, cfg)
    memory = fill.memory
    try:
        settings = memory.recall_settings()
        assert settings == RecallSettings()
        assert not settings.listing
        found = await ask(memory, store)
        # A note and two turns, as the block always held them.
        said = (
            "the user said “Which tomato garden bed {n} gets the sun?”; melo "
            "replied “Bed {n} faces south all day, so it gets the most sun.” "
            "(chat “Which tomato garden bed {n} gets the sun?”, {day})"
        )
        turn = [said.format(n=n, day=day(fill.at(n))) for n in range(3)]
        assert found.block == (
            f"\n\n{BLOCK_HEADER}\n- {note_line(memory)}\n- {turn[0]}\n- {turn[1]}"
        )
        assert LIST_GUIDE not in found.block and "[t" not in found.block
        assert [item.key for item in found.used if not item.is_record] == [
            f"d:{fill.ids[0]}",
            f"d:{fill.ids[1]}",
        ]
        # The search result: numbered lines, notes and turns mixed as ranked.
        reply = await MemoryTools(memory).search(query=QUESTION)
        assert reply == "\n".join(
            f"{n}. {line}" for n, line in enumerate([note_line(memory), *turn], 1)
        )
        # Naming the default changes nothing.
        cfg.memory.recall_form = "excerpt"
        again = await ask(memory, store)
        assert again.block == found.block
        assert await MemoryTools(memory).search(query=QUESTION) == reply
    finally:
        await memory.aclose()


async def test_the_excerpt_form_offers_search_remember_and_open_memory(
    tmp_path: Path,
) -> None:
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        search, remember, opened = await memory.tools().start()
        assert search.definition.name == "search_memory"
        assert remember.definition is REMEMBER and opened.definition is OPEN_MEMORY
        assert SEARCH_MEMORY.parameters["properties"]["k"]["maximum"] == 8
        # The tool list is read when an agent starts, from the setting then.
        cfg.memory.recall_form = "list"
        names = [t.definition.name for t in await memory.tools().start()]
        assert names == ["search_memory", "remember", "open_memory"]
    finally:
        await memory.aclose()


# ── The list in the block ──


async def test_a_turn_is_one_line_and_a_note_reads_as_before(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    fill = await filled(tmp_path, store, cfg)
    memory = fill.memory
    try:
        found = await ask(memory, store)
        header, note, *rest = found.block.split("\n")[2:]
        assert header == BLOCK_HEADER
        assert note == f"- {note_line(memory)}"
        assert rest[0] == GUIDE
        assert LIST_GUIDE == (
            "Older chat turns that may hold the answer, closest first. Open the ones "
            "you need with open_memory before you answer from them; a title is not "
            "the content."
        )
        assert "\n" not in LIST_GUIDE
        title = "Which tomato garden bed {n} gets the sun?"
        assert rest[1:] == [
            f"[t{fill.ids[n]}] {day(fill.at(n))} · chat “{title.format(n=n)}” · "
            f"“{title.format(n=n)}” · words"
            for n in range(3)
        ]
        # Three turns, not the two the default holds; the note is kept.
        assert [item.is_record for item in found.used] == [True, False, False, False]
        assert found.recall is not None and found.recall.used == 4
        assert found.recall.block_chars == len(found.block)
        # The page's view of the used items is the same as for an excerpt.
        assert found.event()["used"][1]["kind"] == "chat"  # type: ignore[index]
    finally:
        await memory.aclose()


async def test_the_chat_label_is_the_main_chat_or_a_task_as_it_is_elsewhere(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        _, index = memory.stores()
        for turn, kind, title in ((1, "front", ""), (2, "task", "Water the beds")):
            await index.add(
                Document(
                    session_id=(await _turn(store, f"s{turn}")).session_id,
                    turn=turn,
                    surface="gateway",
                    kind=kind,
                    title=title,
                    text=document_chunks("Tomato garden plan?", "Plant basil.")[0],
                    created_at=time.time(),
                )
            )
        found = await ask(memory, store)
        labels = sorted(line.split(" · ")[1] for line in lines_of(found.block))
        assert labels == ["Main chat", "task “Water the beds”"]
    finally:
        await memory.aclose()


async def test_the_title_is_the_first_sentence_of_the_users_words_cut_at_a_word(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        long = (
            "Could you remind me which tomato garden beds get the afternoon sun "
            "and which ones stay in the shade all summer long? Thanks."
        )
        assert len(long.split("?")[0]) > LIST_TITLE_CHARS
        await add_turn(memory, store, long, "The east beds do.")
        await add_turn(
            memory,
            store,
            "Tomato garden plan. Also, a word on beans?",
            "Plant tomato seedlings in May.",
            title="Garden",
        )
        found = await ask(memory, store, "tomato garden beds")
        cut, short = sorted(line.split(" · ")[2] for line in lines_of(found.block))
        assert short == "“Tomato garden plan.”"
        # Cut at a word, with the ellipsis, in 80 characters.
        assert cut.startswith("“Could you remind me which tomato garden beds get the")
        words = cut[1:-2]
        assert cut.endswith("…”") and 40 < len(words) < LIST_TITLE_CHARS
        assert long.startswith(words + " ") and len(words) + 1 <= LIST_TITLE_CHARS
    finally:
        await memory.aclose()


def test_word_cut_never_passes_the_limit_and_cuts_at_a_word() -> None:
    assert word_cut("short title", 80) == "short title"
    text = "alpha beta gamma delta epsilon zeta eta theta"
    for limit in range(8, len(text)):
        cut = word_cut(text, limit)
        assert len(cut) <= limit and cut.endswith("…")
        assert text.startswith(cut[:-1]) and text[len(cut) - 1] in " "
    assert word_cut("x" * 100, 10) == "x" * 9 + "…"  # no space: cut anywhere
    assert word_cut("我" * 100, 10) == "我" * 9 + "…"


def doc(text: str, chunk: int = 0) -> Item:
    """A chat turn's document, as the index holds it."""
    return Item("d:1", text, "chat, Oct 30", "s", 1, 1.0, chunk=chunk)


def test_a_title_is_the_first_sentence_of_what_there_is() -> None:
    assert listed_title(doc("User: Where is it? Not here.\nmelo: Over there.")) == (
        "Where is it?"
    )
    assert listed_title(doc("Task: Plan\nUser: Fix the build. Now.\nmelo: Done.")) == (
        "Fix the build."
    )
    assert listed_title(doc("User: \nmelo: Nothing to say here. Really.")) == (
        "Nothing to say here."
    )
    assert listed_title(doc("Summary of chat “Fix CI”: the port was taken.")) == (
        "Summary of chat “Fix CI”: the port was taken."
    )
    assert listed_title(doc("User: 我的组会是哪天？别忘了。\nmelo: 周四。")) == (
        "我的组会是哪天？"
    )
    # A later chunk repeats the head of the user's words: the same title.
    later = doc("User: Where is it…\nmelo: and then it goes on. And on.", chunk=2)
    assert listed_title(later) == "Where is it…"
    # A chunk that starts inside very long user words: its own first sentence.
    inside = doc("User: and then the user goes on. And on.", chunk=1)
    assert listed_title(inside) == "and then the user goes on."


async def test_a_chunked_turn_is_one_line_titled_by_what_matched_in_its_chunk(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        filler = "".join(f"Calm weather note {n} came and went. " for n in range(150))
        answer = filler + "The zucchini harvest peaks in late August. " + filler
        _, ids = await add_turn(
            memory, store, "Tell me about the garden schedule?", answer
        )
        assert len(ids) > 2
        # Words in the user's part are in every chunk, so any one may be the best.
        head = await ask(memory, store, "garden schedule")
        (line,) = lines_of(head.block)
        assert ids_of(head.block)[0] in ids
        assert line.split(" · ")[2] == "“Tell me about the garden schedule?”"
        # Words only a later chunk holds find that chunk: still one line, titled
        # by the sentence of melo's reply that holds them.
        found = await ask(memory, store, "zucchini harvest")
        (line,) = lines_of(found.block)
        assert ids_of(found.block)[0] in ids[1:]
        assert line.split(" · ")[2] == "“The zucchini harvest peaks in late August.”"
        # Both chunks match a question about both; the turn is still one line.
        both = await ask(memory, store, "garden schedule zucchini harvest")
        assert len(lines_of(both.block)) == 1
    finally:
        await memory.aclose()


BOOK = "Last week I finished 'The Quiet Orchard' and loved it."
BOOK_TURN = (
    "User: I'm interested in learning more about renewable energy, can you "
    f"recommend some books? {BOOK}\nmelo: Happy to help with both."
)


def test_a_title_follows_the_query_to_the_sentence_that_matches() -> None:
    turn = doc(BOOK_TURN)
    first = "I'm interested in learning more about renewable energy, can you recommend some…"
    assert listed_title(turn) == first
    assert listed_title(turn, "what book was I reading, The Quiet Orchard?") == BOOK
    # Words of the question that name nothing (stop words, asking words) do not
    # move it; neither does a question the turn holds no word of.
    for query in ("", "what is it of the", "is the one in the", "zebra saxophone"):
        assert listed_title(turn, query) == first
    # A word stem finds its plural, as an excerpt's match does.
    assert listed_title(turn, "which orchards") == BOOK


def test_the_best_sentence_wins_and_ties_go_to_the_earliest() -> None:
    bakes = doc("User: Cy bakes. Ana bakes bread daily. Ben bakes bread daily.")
    assert listed_title(bakes, "who bakes bread") == "Ana bakes bread daily."
    assert listed_title(bakes, "bakes") == "Cy bakes."
    assert listed_title(bakes, "daily") == "Ana bakes bread daily."


def test_melos_reply_is_looked_at_only_when_the_users_words_hold_no_term() -> None:
    reply = doc(
        "User: Hello there. How are you?\nmelo: Bring a coat. "
        "The harbour opens at noon, and the harbour office opens at nine."
    )
    assert listed_title(reply, "when does the harbour open") == (
        "The harbour opens at noon, and the harbour office opens at nine."
    )
    assert listed_title(reply, "harbour") == (  # no term in what the user said
        "The harbour opens at noon, and the harbour office opens at nine."
    )
    typed = doc(
        "User: Thinking about the harbour today.\nmelo: The harbour opens at noon, "
        "and the harbour office opens at nine."
    )
    assert listed_title(typed, "harbour opens") == "Thinking about the harbour today."
    # A turn without the user's words has only the reply to look at.
    assert listed_title(
        doc("User: \nmelo: Bring a coat. Harbour at noon."), "harbour"
    ) == ("Harbour at noon.")


def test_a_long_sentence_starts_where_its_first_matching_term_is_inside_it() -> None:
    sentence = (
        "Well, "
        + "the weather was fine and nothing much happened " * 3
        + "so I planted the zucchini seeds near the east fence after lunch today and "
        "then went inside for a long cold drink and a rest."
    )
    assert len(sentence) > 2 * LIST_TITLE_CHARS
    turn = doc(f"User: {sentence} Then I slept.")
    title = listed_title(turn, "zucchini")
    window = title[1:-1]
    assert title.startswith("…") and title.endswith("…")
    assert "zucchini" in window and len(title) <= LIST_TITLE_CHARS
    assert f" {window} " in f" {sentence} "  # whole words
    # A term in the head leaves the head, cut as it always was.
    assert listed_title(turn, "weather") == listed_title(turn)
    assert listed_title(turn) == word_cut(sentence, LIST_TITLE_CHARS)
    # A term in the last words: the window ends with the sentence and fills the room.
    tail = doc("User: " + "word " * 30 + "zucchini.")
    assert listed_title(tail, "zucchini") == "…" + "word " * 14 + "zucchini."


def test_titles_follow_chinese_and_japanese_queries() -> None:
    chinese = doc("User: 今天天气很好。我的组会在周四下午。\nmelo: 好的，已经记下了。")
    assert listed_title(chinese, "我的组会是哪天") == "我的组会在周四下午。"
    assert listed_title(chinese, "你吃饭了吗") == "今天天气很好。"  # fillers only
    assert (
        listed_title(doc("User: 你好。\nmelo: 周四下午开组会。"), "组会")
        == "周四下午开组会。"
    )
    japanese = doc(
        "User: 今日はいい天気です。村上春樹の小説を読みました。\nmelo: そうですか。"
    )
    assert listed_title(japanese, "村上春樹の本") == "村上春樹の小説を読みました。"
    # A long run without spaces is cut anywhere, with the term inside.
    run = doc("User: " + "今天天气很好" * 20 + "我在学习量子力学。")
    title = listed_title(run, "量子力学")
    assert "量子力学" in title and title.startswith("…") and len(title) <= 80
    assert title.endswith("量子力学。")


def test_a_japanese_particle_is_no_term_but_a_word_that_holds_one_is() -> None:
    for particle in "のはがをにへでともかねよ":
        assert gate_terms(particle) == set()
    for particle in ("から", "まで", "より"):
        assert particle not in gate_terms(particle)
    # A word that holds a particle's character keeps its terms: the pairs, and the
    # characters that are not particles.
    assert gate_terms("はな") == {"はな", "な"}
    assert {"かば", "ばん"} <= gate_terms("かばん")
    assert gate_terms("もの") == {"もの"}  # a word made of two particle characters
    assert gate_terms("私の本を") == {"私", "本", "私の", "の本", "本を"}
    # English and Chinese are as they were.
    assert gate_terms("What is my sister's name?") == {"name", "s", "sister"}
    assert gate_terms("我的组会是哪天") == {"组", "会", "天", "组会"}


def test_a_title_ignores_the_japanese_particles_the_question_shares() -> None:
    turn = doc("User: 朝は静かでした。夕方に姉の本の話をしました。\nmelo: そうですか。")
    first = "朝は静かでした。"
    assert listed_title(turn) == first
    # のとに are all the question shares with the second sentence: no term, so the
    # title stays the first sentence.
    assert listed_title(turn, "彼の机の上に") == first
    # A word they share moves it.
    assert listed_title(turn, "姉の本") == "夕方に姉の本の話をしました。"


def test_without_a_query_or_a_match_a_title_reads_as_it_always_did() -> None:
    for text in (
        "User: Where is it? Not here.\nmelo: Over there.",
        "Task: Plan\nUser: Fix the build. Now.\nmelo: Done.",
        "User: \nmelo: Nothing to say here. Really.",
        "Summary of chat “Fix CI”: the port was taken.",
        "User: 我的组会是哪天？别忘了。\nmelo: 周四。",
        BOOK_TURN,
    ):
        for query in ("", "zebra saxophone", "of the"):
            assert listed_title(doc(text), query) == listed_title(doc(text))
    # A memory item's title is the item's own, whatever the query says.
    item = Item("m:3", "Finished a novel.", "item", "s", 1, 1.0, title="Reading list")
    assert listed_title(item, "novel finished") == "Reading list"
    assert item.listed(query="novel finished") == item.listed()
    # A summary is one sentence to look through: the query finds its sentence.
    summary = doc("Summary of chat “Plans”: the port was taken. The fix was a retry.")
    assert listed_title(summary, "retry fix") == "The fix was a retry."


async def test_the_block_and_the_search_tool_title_a_turn_by_their_own_query(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        typed = f"Thinking about my weekend plans. {BOOK}"
        await add_turn(memory, store, typed, "Nice.", title="Weekend")
        found = await ask(memory, store, "which book did I finish last week?")
        (line,) = lines_of(found.block)
        assert line.split(" · ")[2] == f"“{BOOK}”"
        assert found.used[0].shown == line  # the line the model read
        tools = MemoryTools(memory)
        # The tool's own query, not the message the turn was recalled for.
        (by_plan,) = lines_of(await tools.search(query="weekend plans"))
        assert by_plan.split(" · ")[2] == "“Thinking about my weekend plans.”"
        (by_book,) = lines_of(await tools.search(query="quiet orchard"))
        assert by_book.split(" · ")[2] == f"“{BOOK}”"
        # The default form's lines are not titles at all.
        cfg.memory.recall_form = "excerpt"
        assert "[t" not in (await ask(memory, store, "finished orchard")).block
    finally:
        await memory.aclose()


async def test_the_cap_is_recall_list_max_and_the_best_come_first(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        for n in range(30):
            await add_turn(
                memory, store, f"Tomato garden bed {n}?", f"Bed {n} faces south."
            )
        found = await ask(memory, store, "tomato garden bed")
        assert len(lines_of(found.block)) == 20 == LIST_MAX
        assert len(found.used) == 20
        for cap in (1, 7, 20):
            cfg.memory.recall_list_max = cap
            capped = await ask(memory, store, "tomato garden bed")
            assert len(lines_of(capped.block)) == min(cap, 30)
            assert [item.key for item in capped.used] == [
                f"d:{n}" for n in ids_of(capped.block)
            ]
            assert ids_of(capped.block) == ids_of(found.block)[:cap]
        # The excerpt form's own bounds (two turns, a block of 4000 characters) are
        # not the list's.
        cfg.memory.recall_list_max = 5
        assert len(lines_of((await ask(memory, store, "tomato garden bed")).block)) == 5
    finally:
        await memory.aclose()


async def test_a_chats_cap_does_not_apply_to_the_list(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        sid = (await _turn(store, "One chat")).session_id
        for n in range(1, 6):
            await add_turn(
                memory,
                store,
                f"Tomato garden bed {n}?",
                "South.",
                session_id=sid,
                turn=n,
            )
        assert len(lines_of((await ask(memory, store, "tomato garden bed")).block)) == 5
        cfg.memory.recall_form = "excerpt"  # at most two turns of one chat
        excerpt = await ask(memory, store, "tomato garden bed")
        assert excerpt.block.count("\n- ") == 2
    finally:
        await memory.aclose()


async def test_the_bars_still_drop_what_they_dropped(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        await add_turn(memory, store, "Which tomato garden bed gets sun?", "Bed one.")
        await add_turn(memory, store, "Tell me about the tomato market.", "Closed.")
        # Half of the question's four content words reach the 0.4 bar: the first
        # turn, which holds two of them, is listed and the market turn, with one, is
        # not. A quarter of them reach nothing.
        half = "tomato garden zebra yodel"
        assert len(lines_of((await ask(memory, store, half)).block)) == 1
        quarter = "tomato zebra yodel quartz"
        assert (await ask(memory, store, quarter)).block == ""
        assert (await ask(memory, store, "write a poem about autumn")).block == ""
    finally:
        await memory.aclose()


async def test_the_guidance_line_is_there_only_when_a_turn_is_listed(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        await teach(memory, store, await _turn(store, "Stakes"), NOTE)
        notes_only = await ask(memory, store)
        assert [item.is_record for item in notes_only.used] == [True]
        assert LIST_GUIDE not in notes_only.block and lines_of(notes_only.block) == []
        await add_turn(memory, store, "Which tomato garden bed gets sun?", "Bed one.")
        both = await ask(memory, store)
        assert both.block.count(LIST_GUIDE) == 1
        assert both.block.index(LIST_GUIDE) > both.block.index(f"“{NOTE}”")
        # Nothing recalled: no list and no guidance.
        nothing = await ask(memory, store, "write a poem about autumn")
        assert nothing.block == ""
        # Kept lines come first, as in the excerpt form.
        kept = await ask(memory, store, "Remember that my tea is oolong.")
        assert kept.block.split("\n")[3].startswith("melo kept from")
    finally:
        await memory.aclose()


async def test_the_notes_keep_their_own_bound_and_the_list_its_own(
    tmp_path: Path, store: SessionStore
) -> None:
    notes = [
        f"I {verb} the tomato garden {where}."
        for verb, where in (
            ("water", "every morning"),
            ("prune", "in June"),
            ("mulch", "with straw"),
            ("thin", "in April"),
        )
    ]
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        for n, said in enumerate(notes):
            await teach(memory, store, await _turn(store, f"Note {n}"), said)
        for n in range(4):
            await add_turn(memory, store, f"Tomato garden bed {n}?", "South.")
        found = await ask(memory, store, "tomato garden")
        # Four notes match: the block holds AUTO_RECORDS of them, the list its own.
        assert sum(item.is_record for item in found.used) == AUTO_RECORDS == 3
        assert len(lines_of(found.block)) == 4
    finally:
        await memory.aclose()


async def test_an_older_turn_keeps_its_mark_in_the_list(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        now = time.time()
        await add_turn(
            memory, store, "My weekly sync is on Wednesday.", "Noted.", at=now - DAY
        )
        await add_turn(
            memory, store, "My weekly sync is on Friday.", "Noted.", at=now - 2 * 60
        )
        search = await MemoryTools(memory).search(query="weekly sync")
        old, new = (
            line for line in lines_of(search) if "Wednesday" in line or "Friday" in line
        )
        assert "Friday" in old and "(older" not in old  # the newer value first
        assert f"({OLDER_TURN})" in new and "Wednesday" in new
        assert new.endswith(f"· words ({OLDER_TURN})")
        # The block gives the current value only, unless the message asks about
        # the past.
        block = (await ask(memory, store, "which day is my weekly sync?")).block
        assert "Friday" in block and "Wednesday" not in block
        past = (await ask(memory, store, "what day was my weekly sync before?")).block
        assert f"({OLDER_TURN})" in past
        assert past.index("Friday") < past.index("Wednesday")
    finally:
        await memory.aclose()


async def test_the_year_and_the_time_of_day_read_as_they_do_for_an_excerpt(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        now = time.time()
        old = now - 400 * DAY  # another year
        await add_turn(memory, store, "Tomato garden bed one?", "South.", at=old)
        await add_turn(memory, store, "Tomato garden bed two?", "East.", at=now - 7200)
        await add_turn(memory, store, "Tomato garden bed three?", "West.", at=now - 60)
        block = (await ask(memory, store, "tomato garden bed")).block
        by_title = {line.split(" · ")[2]: line for line in lines_of(block)}
        year = time.localtime(old).tm_year
        assert (
            by_title["“Tomato garden bed one?”"].split(" · ")[0].endswith(f", {year}")
        )
        # The two of one day show the time of day, so their order can be told.
        two, three = (
            by_title["“Tomato garden bed two?”"],
            by_title["“Tomato garden bed three?”"],
        )
        if time.localtime(now - 7200)[:3] == time.localtime(now - 60)[:3]:
            for line, at in ((two, now - 7200), (three, now - 60)):
                assert line.split(" · ")[0].endswith(day(at, now, timed=True)[-5:])
    finally:
        await memory.aclose()


# ── The similarity ──


async def test_a_turn_found_by_meaning_shows_its_cosine_and_one_found_by_words_says_so(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list", retriever="embedding")
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    provider = FixedEmbedding()
    memory.embedding = lambda: (provider, MODEL)  # type: ignore[method-assign]
    try:
        # The question has "zebra": its vector is [1, 0, 0]; no word of it is in a turn.
        await add_turn(memory, store, "Alpha plan?", "Meaning.", cosine=0.8)
        await add_turn(memory, store, "Beta plan?", "Meaning.", cosine=0.571)
        await add_turn(memory, store, "Gamma plan?", "Too far.", cosine=0.43)
        found = await ask(memory, store, "zebra yodel")
        assert [line.rsplit(" · ", 1)[1] for line in lines_of(found.block)] == [
            "0.80",
            "0.57",
        ]
        assert [line.split(" · ")[2] for line in lines_of(found.block)] == [
            "“Alpha plan?”",
            "“Beta plan?”",
        ]
        # Matched by words and by meaning: the cosine; by words only: "words".
        await add_turn(memory, store, "Zebra yodel class?", "Sundays.", cosine=0.9)
        await add_turn(memory, store, "Zebra yodel choir?", "Fridays.")
        mixed = await ask(memory, store, "zebra yodel")
        tail = {
            line.split(" · ")[2]: line.rsplit(" · ", 1)[1]
            for line in lines_of(mixed.block)
        }
        assert tail["“Zebra yodel class?”"] == "0.90"
        assert tail["“Zebra yodel choir?”"] == "words"
    finally:
        await memory.aclose()


def test_meaning_of_is_the_cosine_only_when_the_vector_list_would_take_the_item() -> (
    None
):
    query = np.asarray([1.0, 0.0, 0.0], np.float32)

    def turn(cosine: float | None, model: str = MODEL) -> Item:
        vector = None if cosine is None else np.asarray(vector_at(cosine), np.float32)
        return Item(
            "d:1", "x", "chat, Oct 30", "s", 1, 1.0, vector=vector, embed_model=model
        )

    assert meaning_of(turn(0.8), query, MODEL, 0.55) == pytest.approx(0.8)
    assert meaning_of(turn(0.55), query, MODEL, 0.55) == pytest.approx(0.55)
    assert meaning_of(turn(0.5), query, MODEL, 0.55) is None  # under the bar
    assert meaning_of(turn(None), query, MODEL, 0.55) is None  # no vector
    assert meaning_of(turn(0.9, "other"), query, MODEL, 0.55) is None  # other model
    assert meaning_of(turn(0.9), None, MODEL, 0.55) is None  # no question vector
    wide = Item(
        "d:1", "x", "c", "s", 1, 1.0, vector=np.ones(4, np.float32), embed_model=MODEL
    )
    assert meaning_of(wide, query, MODEL, 0.55) is None  # another size


def test_pick_with_no_chat_cap_keeps_every_turn_up_to_the_bound() -> None:
    items = [Item(f"d:{n}", "x", "chat, Oct 30", "s", n, 1.0) for n in range(6)]
    assert len(pick(items, records=0, docs=10)) == 2  # the chats' cap
    assert len(pick(items, records=0, docs=10, per_chat=None)) == 6
    assert len(pick(items, records=0, docs=4, per_chat=None)) == 4


# ── The search tool ──


async def test_the_search_result_lists_the_turns_under_the_guidance_line(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    fill = await filled(tmp_path, store, cfg)
    memory = fill.memory
    tools = MemoryTools(memory)
    try:
        reply = await tools.search(query=QUESTION)
        head, guide, *rest = reply.split("\n")
        assert head == f"1. {note_line(memory)}"
        assert guide == GUIDE
        assert [int(line[2 : line.index("]")]) for line in rest] == fill.ids
        assert all(line.endswith("· words") for line in rest)
        # Scope picks the kind, as it does for an excerpt.
        notes = await tools.search(query=QUESTION, scope="records")
        assert notes == head
        chats = await tools.search(query=QUESTION, scope="chats")
        assert chats.split("\n") == [guide, *rest]
        assert await tools.search(query="zebra saxophone") == NOTHING_FOUND
        # The excerpt form's reading of the same turns is gone from the result.
        assert "melo replied" not in reply
        # The result is as long as the settings say: k, or the list's own bound.
        for n in range(3, 30):
            await add_turn(
                memory,
                store,
                f"Tomato garden bed {n}?",
                f"Bed {n} faces south.",
                at=fill.now - n * DAY,
            )
        assert len(lines_of(await tools.search(query="tomato garden bed"))) == 20
        cfg.memory.recall_list_max = 12
        assert len(lines_of(await tools.search(query="tomato garden bed"))) == 12
        assert len(lines_of(await tools.search(query="tomato garden bed", k=4))) == 4
        assert len(lines_of(await tools.search(query="tomato garden bed", k=30))) == 12
        cfg.memory.recall_list_max = 50
        assert len(lines_of(await tools.search(query="tomato garden bed", k=30))) == 30
    finally:
        await memory.aclose()


async def test_the_search_k_bound_is_lifted_to_the_list_in_the_tool_definition(
    tmp_path: Path,
) -> None:
    cfg = config(recall_form="list", recall_list_max=35)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        search, remember, opened = await memory.tools().start()
        assert search.definition.name == "search_memory"
        assert search.definition.parameters["properties"]["k"] == {
            "type": "integer",
            "minimum": 1,
            "maximum": 35,
        }
        assert "open_memory" in search.definition.description
        assert search.definition.description.startswith(SEARCH_MEMORY.description)
        assert search.definition.annotations == {"readOnlyHint": True}
        assert not search.requires_approval and not opened.requires_approval
        assert remember.definition is REMEMBER and opened.definition is OPEN_MEMORY
        # The shared definition is not changed by making the list form's.
        assert SEARCH_MEMORY.parameters["properties"]["k"]["maximum"] == 8
        assert search_memory_listing(50).parameters["properties"]["k"]["maximum"] == 50
    finally:
        await memory.aclose()


async def test_what_the_search_lists_goes_into_the_turns_searched_key(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    fill = await filled(tmp_path, store, cfg)
    memory = fill.memory
    try:
        emitted: list[dict[str, Any]] = []
        typed = "hi there"  # nothing here recalls automatically
        await memory.before_turn(
            store, await _turn(store, typed), typed, emit=emitted.append
        )
        await MemoryTools(memory).search(query=QUESTION)
        (data,) = emitted
        assert [(item["kind"], item["id"]) for item in data["searched"]] == [
            ("record", 1),
            *(("chat", n) for n in fill.ids),
        ]
        assert "opened" not in data
        chat = CHAT_TURN.get()
        assert chat is not None and len(chat.searched) == 4
    finally:
        CHAT_TURN.set(None)
        await memory.aclose()


async def test_a_task_context_pack_and_the_plain_search_do_not_list(
    tmp_path: Path, store: SessionStore
) -> None:
    cfg = config(recall_form="list")
    fill = await filled(tmp_path, store, cfg)
    memory = fill.memory
    try:
        items = await memory.search(QUESTION, records_k=3, docs_k=2)
        assert all(item.similarity is None for item in items)
        assert sum(not item.is_record for item in items) == 2  # the chats' bound
    finally:
        await memory.aclose()


# ── The settings ──


async def test_the_list_settings_are_read_each_turn_and_a_bad_one_is_the_default(
    tmp_path: Path,
) -> None:
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        first = memory.recall_settings()
        assert (first.listing, first.list_max, first.open_chars) == (False, 20, 4000)
        cfg.memory.recall_form = "list"
        cfg.memory.recall_list_max = 7
        cfg.memory.recall_open_chars = 1200
        settings = memory.recall_settings()
        assert (settings.listing, settings.list_max, settings.open_chars) == (
            True,
            7,
            1200,
        )
        for bad in (0, 51, True, "7", 2.5):
            cfg.memory.recall_list_max = bad  # type: ignore[assignment]
            assert memory.recall_settings().list_max == 20, bad
        for bad in (499, 16001, False, "4000"):
            cfg.memory.recall_open_chars = bad  # type: ignore[assignment]
            assert memory.recall_settings().open_chars == 4000, bad
        cfg.memory.recall_form = "tree"
        assert memory.recall_settings().listing is False
    finally:
        await memory.aclose()
