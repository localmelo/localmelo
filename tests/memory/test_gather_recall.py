"""Recall looks wider for a question that counts, totals or compares.

``asks_to_gather`` finds such a question (English by words that stand alone, Chinese
by text) that also speaks of its speaker ("I", "my", "we", "我" …): a generic prompt
("how many seconds are in a day") is not one. For it the list form's vector bar
drops to ``GATHER_COSINE`` (or the recall bar when that is lower) in the recall block
and in every list-form ``search_memory``, and one sentence (``GATHER_GUIDE``) follows
the list's guide line. A question that does not gather, and the excerpt form, read
as they always did. Synthetic turns and fixed vectors, no model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import MemoryItem
from localmelo.support.memory.recall import (
    BLOCK_HEADER,
    GATHER_COSINE,
    GATHER_GUIDE,
    ITEM_LIST_GUIDE,
    LIST_GUIDE,
    MIN_COSINE,
    PREMISE_GUIDE,
    asks_to_gather,
    day,
)
from localmelo.support.memory.service import (
    CHAT_TURN,
    MemoryTools,
    Recall,
    RecallSettings,
)
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.test_followup_query import chat_of, now_in
from localmelo.tests.memory.test_hippo_recall import put
from localmelo.tests.memory.test_recall_bars import MODEL, FixedEmbedding
from localmelo.tests.memory.test_recall_list import (
    DAY,
    _turn,
    add_turn,
    ask,
    config,
    ids_of,
    lines_of,
)
from localmelo.tests.sessions.helpers import Clock

# ── The detector ──

POSITIVE = [
    # English: the words of the brief, in a message about the speaker's own life
    "How many books did I read this year?",
    "how many books did I read",
    "how much did I spend on groceries?",
    "What is the total of my donations?",
    "How much did I give in total?",
    "Altogether, how long was I in Japan?",
    "What's the combined cost of our two trips?",
    "In all, which cities did I visit?",
    "What is the number of times I went hiking?",
    "Count the doctors I saw this spring.",
    "List the restaurants we talked about.",
    "Where are all the places I stayed?",
    "Where are all my keys?",
    "Which are the different gyms I tried?",
    "What did I eat each morning?",
    "Did I call every client?",
    "Did I like both movies?",
    "I compared two phones last week, which one won?",
    "What was the difference between my two flights?",
    "Did I spend more than two hours reading?",
    "Is my commute longer than hers?",
    "How many of us went to the lake?",
    "How many times did the doctor call me?",
    "How much is mine in total?",
    "Which cities have we visited in all?",
    "I'm counting the pets: how many are there?",
    "Is it fewer than ten, as far as I know?",
    # case, spacing and the forms the words take
    "HOW MANY pets do I have?",
    "how   many pets do I have?",
    "What are my totals for May?",
    "Which lists did I make?",
    "What was the difference, and which differences did I miss?",
    # Chinese
    "我一共读了多少本书？",
    "我一共去过几个国家",
    "我去过几次日本？",
    "我养了几个宠物？",
    "我总共花了什么钱",
    "我提到过哪些餐厅？",
    "我的朋友都有谁",
    "我们一共去了几次？",
    # both languages in one message
    "我想知道 how many books",
    "how many 个朋友 did I meet",
]
NEGATIVE = [
    # generic prompts that count, total, list or compare, with nothing of the speaker
    "How many seconds are in a day?",
    "How much is the total of 12 and 30?",
    "List the planets in the solar system.",
    "Explain the difference between a mutex and a semaphore.",
    "What's the combined cost of the two trips?",
    "How does Paris compare to Rome?",
    "Was the trip less than a week?",
    "Which route was faster?",
    "What are the totals for May?",
    "What is the difference and what are the differences?",
    "Was it fewer than ten?",
    "Which cities have the most people in all of Europe?",
    "一年有多少天？",
    "列出所有的行星",
    "合计是多少",
    "所有的医生是谁",
    "每个人的爱好",
    "how many 个朋友",
    # a speaker word inside another word is not a speaker
    "How many people fit on the bus?",
    "How many meters are in a mile?",
    "Which step does the model determine first, in all?",
    # a word inside another word is not the word
    "What is my account balance?",
    "Where did I put the discount code?",
    "Which country did I visit last spring?",
    "Did I listen to that podcast yet?",
    "Is everyone coming to dinner?",
    "Did I pack everything?",
    "Is the fastener loose?",
    "Which is the longest river?",
    "Is the recounting finished?",
    "Was the countdown on?",
    "I need a compartment for the bike.",
    "Is there a totally free option?",
    "What is my combine harvester's model?",
    "What's the listing for the apartment?",
    "Where did I leave my tablet?",
    # plain questions
    "What is the capital of France?",
    "What is my dog's name?",
    "Where do I live?",
    "Remind me of my wife's birthday.",
    "What time is my dentist appointment?",
    "What did I name the cat?",
    "Which hotel did I book in Lisbon?",
    "I live in Pittsburgh now.",
    "hi there",
    "",
    "   ",
    # Chinese
    "我的狗叫什么名字？",
    "我昨天去了哪里？",
    "明天几点开会？",
    "我喜欢什么颜色？",
    "提醒我给妈妈打电话",
    "我住在哪个城市？",
    "每天早上我喝什么？",
    "他们是谁？",
]


def test_the_table_is_big_enough_and_has_both_languages() -> None:
    assert len(POSITIVE) + len(NEGATIVE) >= 30
    assert any("一" <= c <= "鿿" for text in POSITIVE for c in text)
    assert any("一" <= c <= "鿿" for text in NEGATIVE for c in text)


@pytest.mark.parametrize("message", POSITIVE)
def test_a_question_that_counts_totals_lists_or_compares_about_the_speaker_gathers(
    message: str,
) -> None:
    assert asks_to_gather(message) is True


@pytest.mark.parametrize("message", NEGATIVE)
def test_a_generic_prompt_a_plain_message_and_a_word_inside_a_word_do_not(
    message: str,
) -> None:
    assert asks_to_gather(message) is False


# ── The constants ──


def test_the_constants() -> None:
    assert GATHER_COSINE == 0.40 and MIN_COSINE == 0.55
    assert RecallSettings().wide_floor == 0.40
    assert GATHER_GUIDE == (
        "This asks for a count, total or comparison: open every listed item that "
        "may belong, search again with other words for each kind not listed yet, "
        "and count only what the items say."
    )


async def test_the_wide_bar_is_never_above_the_recall_bar_read_each_turn(
    tmp_path: Path,
) -> None:
    cfg = config()
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        assert memory.recall_settings().wide_floor == 0.40
        cfg.memory.recall_min_cosine = 0.25  # a lower recall bar is never raised
        assert memory.recall_settings().wide_floor == 0.25
        cfg.memory.recall_min_cosine = 0.60
        assert memory.recall_settings().wide_floor == 0.40
    finally:
        await memory.aclose()


# ── A memory whose evidence sits under the vector bar ──

# The far question has "zebra": its vector is [1, 0, 0], and a turn's stored vector
# is at the cosine given. Only the first turn is at or above the default 0.55.
TURNS = [
    ("My cardiologist moved my appointment.", "Moved to Friday.", 0.70),
    ("I got fitted for hearing aids.", "Good to hear it went well.", 0.52),
    ("I bought a glucose meter.", "Handy for daily checks.", 0.45),
    ("My ENT doctor said the ear is fine.", "Glad to hear it.", 0.42),
    ("The sofa arrived on Tuesday.", "Enjoy the new sofa.", 0.30),
]
GATHER = "How many zebra yodel clinics did I visit in total?"
PLAIN = "Did I go to a zebra yodel clinic?"
WORDS = "zebra yodel clinics"  # what the model might search for: no counting word
NOTHING_FOUND = "Nothing in melo's memory matches that."


@pytest.fixture()
def clock() -> Clock:
    return Clock()


@pytest.fixture()
async def store(tmp_path: Path, clock: Clock) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway", clock=clock)
    await store.open()
    try:
        yield store
    finally:
        CHAT_TURN.set(None)
        await store.aclose()


async def memory_of(
    tmp_path: Path, store: SessionStore, **memory: object
) -> tuple[ChatMemory, list[int], Config]:
    """A memory with the five turns above, a chat each, as ``t1`` to ``t5`` and
    three days apart in that order, the cosines as stored vectors, and its config."""
    memory.setdefault("retriever", "embedding")
    cfg = config(**memory)
    chat = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    provider = FixedEmbedding()
    chat.embedding = lambda: (provider, MODEL)  # type: ignore[method-assign]
    ids: list[int] = []
    for n, (typed, answer, cosine) in enumerate(TURNS):
        _, doc_ids = await add_turn(
            chat,
            store,
            typed,
            answer,
            at=time.time() - (n + 1) * 3 * DAY,
            cosine=cosine,
        )
        ids.extend(doc_ids)
    return chat, ids, cfg


def line_of(n: int, cosine: str) -> str:
    """How turn *n* (0 to 4 of TURNS) reads in the list, which has the id ``n + 1``."""
    typed = TURNS[n][0]
    when = day(time.time() - (n + 1) * 3 * DAY)
    return f"[t{n + 1}] {when} · chat “{typed}” · “{typed}” · {cosine}"


def guide(gather: bool = False) -> str:
    """The line above the list: its guide, the gather sentence for a turn that
    asks to count, total or compare, then the premise sentence."""
    return " ".join([LIST_GUIDE, *([GATHER_GUIDE] if gather else []), PREMISE_GUIDE])


def listed(*rows: tuple[int, str], gather: bool = False) -> str:
    """The block, or the search result without a note, that lists the *rows*."""
    return "\n".join([guide(gather), *(line_of(n, cosine) for n, cosine in rows)])


STRONG = (0, "0.70")
EVIDENCE = [(1, "0.52"), (2, "0.45"), (3, "0.42")]
AS_ALWAYS = f"\n\n{BLOCK_HEADER}\n"


# ── The block ──


async def test_a_gathering_question_lists_the_evidence_under_the_default_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, ids, _ = await memory_of(tmp_path, store, recall_form="list")
    try:
        found = await ask(memory, store, GATHER)
        assert found.block == AS_ALWAYS + listed(STRONG, *EVIDENCE, gather=True)
        lines = found.block.split("\n")
        assert lines[2:4] == [
            BLOCK_HEADER,
            f"{LIST_GUIDE} {GATHER_GUIDE} {PREMISE_GUIDE}",
        ]
        assert found.block.count(GATHER_GUIDE) == 1
        assert ids_of(found.block) == ids[:4]  # the far sofa (0.30) stays out
        assert found.recall is not None
        assert found.recall.gather is True and found.recall.floor == 0.40
        assert found.recall.used == 4 and found.recall.candidates == 4
        assert found.recall.block_chars == len(found.block)
        data = found.event()["recall"]  # type: ignore[index]
        assert data["gather"] is True and data["floor"] == 0.40
    finally:
        await memory.aclose()


async def test_a_plain_question_keeps_the_default_bar_and_no_sentence(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="list")
    try:
        found = await ask(memory, store, PLAIN)
        assert found.block == AS_ALWAYS + listed(STRONG)
        assert GATHER_GUIDE not in found.block
        assert found.recall is not None
        assert found.recall.gather is False and found.recall.floor == 0.55
        data = found.event()["recall"]  # type: ignore[index]
        assert data["gather"] is False and data["floor"] == 0.55
        assert data["used"] == 1
        # Nothing to search for: still no gathering, and the default bar.
        blank = await ask(memory, store, "   ")
        assert blank.recall is not None and blank.recall.query_from == "none"
        assert blank.recall.gather is False and blank.recall.floor == 0.55
    finally:
        await memory.aclose()


async def test_the_excerpt_form_has_no_wider_bar_and_no_sentence(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="excerpt")
    try:
        said = (
            "the user said “My cardiologist moved my appointment.”; melo "
            "replied “Moved to Friday.” (chat “My cardiologist moved my "
            f"appointment.”, {day(time.time() - 3 * DAY)})"
        )
        for question in (GATHER, PLAIN):
            found = await ask(memory, store, question)
            assert found.block == f"{AS_ALWAYS}- {said}"
            assert GATHER_GUIDE not in found.block
            # The detector fires on GATHER, but the excerpt form has no bar to lower.
            assert found.recall is not None and found.recall.floor == 0.55
            assert found.recall.gather is (question == GATHER)
        assert await MemoryTools(memory).search(query=WORDS) == f"1. {said}"
    finally:
        await memory.aclose()


async def test_the_wide_bar_is_never_above_the_recall_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, ids, cfg = await memory_of(tmp_path, store, recall_form="list")
    try:
        # A recall bar below the gather bar wins: the bar is never raised.
        cfg.memory.recall_min_cosine = 0.35
        low = await ask(memory, store, GATHER)
        assert ids_of(low.block) == ids[:4]
        assert low.recall is not None and low.recall.floor == 0.35
        cfg.memory.recall_min_cosine = 0.30
        all_five = await ask(memory, store, GATHER)
        assert ids_of(all_five.block) == ids  # the sofa (0.30) is at the bar
        assert all_five.recall is not None and all_five.recall.floor == 0.30
        assert lines_of(all_five.block)[-1].endswith(" · 0.30")
    finally:
        await memory.aclose()


async def test_the_list_keeps_its_own_bound_and_the_word_bar_is_unchanged(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="list")
    try:
        for n in range(25):
            await add_turn(
                memory,
                store,
                f"Visit number {n}.",
                "Noted.",
                at=time.time() - (n + 10) * DAY,
                cosine=0.50,
            )
        found = await ask(memory, store, GATHER)
        assert len(ids_of(found.block)) == 20  # recall_list_max, as before
        # A turn with some of the question's words and no vector is still left out
        # (the bar of words is unchanged).
        await add_turn(
            memory, store, "Zebra clinics are loud.", "They are.", at=time.time()
        )
        again = await ask(memory, store, GATHER)
        assert not any("Zebra clinics" in line for line in lines_of(again.block))
    finally:
        await memory.aclose()


async def test_a_follow_up_is_checked_with_the_words_it_is_searched_with(
    tmp_path: Path, store: SessionStore, clock: Clock
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="list")
    try:
        for asked, gathers, count in ((GATHER, True, 4), (PLAIN, False, 1)):
            turns = await chat_of(store, clock, ["Hello there", asked, "Okay, thanks"])
            info = await now_in(store, clock, turns[0].session_id, "go look it up")
            found = await memory.before_turn(store, info, "go look it up")
            assert found.recall is not None and found.recall.query_from == "recent"
            assert found.recall.gather is gathers
            assert found.recall.floor == (0.40 if gathers else 0.55)
            assert len(ids_of(found.block)) == count
    finally:
        await memory.aclose()


# ── The search tool ──


async def test_a_list_form_search_goes_down_to_the_wide_bar_only_under_a_gathering_turn(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, ids, _ = await memory_of(tmp_path, store, recall_form="list")
    tools = MemoryTools(memory)
    try:
        # No turn is under way: the search has no message to read, so it keeps the
        # recall bar, whatever its words.
        assert CHAT_TURN.get() is None
        assert await tools.search(query=WORDS) == listed(STRONG)
        # A plain turn: the same.
        await ask(memory, store, PLAIN)
        assert await tools.search(query=WORDS) == listed(STRONG)
        # A gathering turn: the bar is the wide bar, whatever the model searched for
        # (its words hold no counting word).
        await ask(memory, store, GATHER)
        reply = await tools.search(query=WORDS)
        assert reply == listed(STRONG, *EVIDENCE, gather=True)
        assert ids_of(reply) == ids[:4]
        assert await tools.search(query=WORDS, k=2) == listed(
            STRONG, EVIDENCE[0], gather=True
        )
        assert await tools.search(query=WORDS, scope="records") == NOTHING_FOUND
    finally:
        await memory.aclose()


async def test_the_search_of_the_excerpt_form_has_no_wide_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="excerpt")
    try:
        reply = await MemoryTools(memory).search(query=WORDS)
        assert len(reply.split("\n")) == 1 and "cardiologist" in reply
        assert len(await memory.search(WORDS, records_k=3, docs_k=5)) == 1
        # The argument of ChatMemory.search is the way to a different bar.
        low = await memory.search(WORDS, records_k=3, docs_k=5, min_cosine=0.40)
        assert len(low) == 4
    finally:
        await memory.aclose()


# ── The guide ──


async def test_the_sentence_follows_the_guide_line_for_a_gathering_turn_only(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="list")
    try:
        plain = await ask(memory, store, PLAIN)
        assert plain.block == AS_ALWAYS + listed(STRONG)
        assert GATHER_GUIDE not in plain.block
        assert plain.recall is not None and plain.recall.gather is False
    finally:
        await memory.aclose()


async def test_the_search_of_a_gathering_turn_carries_the_sentence_once(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(tmp_path, store, recall_form="list")
    tools = MemoryTools(memory)
    try:
        # The search reads the turn under way: a gathering one gets the sentence,
        # once, after the guide line, whatever the model searched for.
        await ask(memory, store, GATHER)
        reply = await tools.search(query=WORDS)
        lines = reply.split("\n")
        assert lines[0] == f"{LIST_GUIDE} {GATHER_GUIDE} {PREMISE_GUIDE}"
        assert reply.count(GATHER_GUIDE) == 1
        assert [line[:3] for line in lines[1:]] == ["[t1", "[t2", "[t3", "[t4"]
        # A plain turn, and a search outside any turn, get the guide line as before.
        await ask(memory, store, PLAIN)
        plain = await tools.search(query=WORDS)
        assert plain == listed(STRONG)  # and the default bar
        CHAT_TURN.set(None)
        assert (await tools.search(query=WORDS)).split("\n")[0] == guide()
        # With no list there is no guide line, so no sentence.
        await ask(memory, store, GATHER)
        assert await tools.search(query="saxophone") == NOTHING_FOUND
    finally:
        await memory.aclose()


async def test_the_sentence_follows_the_item_guide_of_the_items_arm(
    tmp_path: Path, store: SessionStore
) -> None:
    memory, _, _ = await memory_of(
        tmp_path, store, recall_form="list", recall_source="items"
    )
    try:
        _, index = memory.stores()
        # One item that cites the five turns, so none of them is recalled as a turn
        # (it has no vector and no words of the question, so it is not listed).
        docs = await index.fetch(await index.ids())
        await index.add_item(
            MemoryItem(
                session_id=docs[0].session_id,
                title="Notes",
                text="Five chats were kept.",
                kind="semantic",
                created_at=1.0,
                sources=tuple((doc.session_id, doc.turn) for doc in docs),
            )
        )
        session_id = (await _turn(store, "Clinics")).session_id
        number = await put(
            memory, session_id, "Hearing aid fitting", "Fitted last week.", cosine=0.45
        )
        found = await ask(memory, store, GATHER)
        lines = found.block.split("\n")
        assert lines[3] == f"{ITEM_LIST_GUIDE} {GATHER_GUIDE} {PREMISE_GUIDE}"
        assert lines[4].startswith(f"[m{number}] ") and lines[4].endswith(" · 0.45")
        reply = await MemoryTools(memory).search(query=WORDS)
        assert reply.split("\n")[0] == (
            f"{ITEM_LIST_GUIDE} {GATHER_GUIDE} {PREMISE_GUIDE}"
        )
        plain = await ask(memory, store, PLAIN)
        assert plain.block == ""  # an item at 0.45 is under the default bar
    finally:
        await memory.aclose()


# ── The recall data ──


def test_the_recall_data_always_carries_gather_and_floor() -> None:
    base = Recall().to_json()
    assert base["gather"] is False and base["floor"] == MIN_COSINE
    fired = Recall(gather=True, floor=0.4).to_json()
    assert fired["gather"] is True and fired["floor"] == 0.4
    assert {k: v for k, v in fired.items() if k not in ("gather", "floor")} == {
        k: v for k, v in base.items() if k not in ("gather", "floor")
    }
