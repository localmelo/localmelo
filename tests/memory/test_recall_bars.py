"""The recall bars: how close in meaning, and how many of the question's words.

``MIN_COSINE`` (0.45) is the least similarity a vector needs to join the vector
list and ``MIN_COVERAGE`` (0.4) the share of the question's content words an item
must hold to join the word list. Both apply to the recall block and to the memory
search tool. With ``[memory] recall_word_weight = "rare"`` the word bar is a share of
the words' weight, a word that few items hold weighing more. Synthetic notes, turns
and fixed vectors, no model.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.support.config import Config, MemoryConfig
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text
from localmelo.support.memory.recall import (
    MIN_COSINE,
    MIN_COVERAGE,
    Corpus,
    Item,
    ranked,
)
from localmelo.support.memory.service import MemoryTools
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeChat, teach

HOLDER = "test-host:1:recall-bars"
NOW = 1_000_000.0
MODEL = "fixed-embed"
NOTHING_FOUND = "Nothing in melo's memory matches that."
# No word of the turn below is in this question, so only vectors can find it.
FAR_QUESTION = "zebra yodel"
# Half of this question's four content words are in the turn and the note below.
HALF_QUESTION = "tomato garden zebra yodel"
# One of this question's four content words is in the turn and the note below.
FEW_QUESTION = "tomato zebra yodel quartz"
# A vector just above the cosine bar and one just below it.
ABOVE, BELOW = 0.47, 0.43
TURN = ("Which tomato garden bed gets the sun?", "Bed one faces south all day.")
NOTE = "I stake the tomato garden with bamboo."


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


def vector_at(cosine: float) -> list[float]:
    """A unit vector whose cosine to ``[1, 0, 0]`` is *cosine*."""
    return [cosine, math.sqrt(1.0 - cosine * cosine), 0.0]


class FixedEmbedding(BaseEmbeddingProvider):
    """The question (it holds "zebra") is ``[1, 0, 0]``; any other text is orthogonal."""

    model = MODEL

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0] if "zebra" in text else [0.0, 0.0, 1.0] for text in texts
        ]

    async def close(self) -> None:
        return None


def config(**memory: object) -> Config:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return cfg


async def _turn(store: SessionStore, title: str):  # type: ignore[no-untyped-def]
    return await store.begin_turn(
        None, title, holder=HOLDER, title=title, new_session=NewSession()
    )


async def memory_with(
    tmp_path: Path,
    store: SessionStore,
    cfg: Config,
    *,
    cosine: float | None = None,
    note: bool = False,
) -> ChatMemory:
    """A memory holding TURN as a chat turn (with a stored vector at *cosine* to the
    far question, when given) and, with *note*, NOTE as a note."""
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    if cosine is not None:
        provider = FixedEmbedding()
        memory.embedding = lambda: (provider, MODEL)  # type: ignore[method-assign]
    if note:
        await teach(memory, store, await _turn(store, "Stakes"), NOTE)
    typed, answer = TURN
    turn = await _turn(store, typed)
    _, index = memory.stores()
    await index.add(
        Document(
            session_id=turn.session_id,
            turn=1,
            surface="gateway",
            kind="chat",
            title=typed,
            text=document_text(typed, answer),
            created_at=time.time(),
            vector=tuple(vector_at(cosine)) if cosine is not None else None,
            embed_model=MODEL if cosine is not None else "",
        )
    )
    return memory


def doc_item(key: str, text: str, cosine: float | None = None) -> Item:
    vector = None if cosine is None else np.asarray(vector_at(cosine), np.float32)
    return Item(
        key,
        text,
        "chat",
        f"s{key}",
        1,
        NOW,
        vector=vector,
        embed_model=MODEL if cosine is not None else "",
    )


QUERY_VECTOR = np.asarray([1.0, 0.0, 0.0], np.float32)


def test_ranked_takes_a_vector_above_the_cosine_bar_and_leaves_one_below() -> None:
    above = doc_item("d:1", "the cat sleeps on the sofa", ABOVE)
    below = doc_item("d:2", "a dog barks in the yard", BELOW)
    corpus = Corpus()
    corpus.add(above)
    corpus.add(below)
    now = NOW + 1
    assert ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now) == [above]
    # A lower bar takes what the default leaves, the closer one first.
    low = ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.40)
    assert low == [above, below]
    # A higher bar takes away what the default keeps.
    assert (
        ranked(corpus, FAR_QUESTION, QUERY_VECTOR, MODEL, now=now, min_cosine=0.65)
        == []
    )


def test_ranked_takes_half_the_question_words_and_leaves_a_quarter() -> None:
    held = doc_item("d:1", "I stake the tomato garden with bamboo.")
    corpus = Corpus()
    corpus.add(held)
    now = NOW + 1
    assert ranked(corpus, HALF_QUESTION, None, "", now=now) == [held]
    assert ranked(corpus, FEW_QUESTION, None, "", now=now) == []
    # A stricter share takes away what the default keeps, a zero share lets all in.
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=0.6) == []
    assert ranked(corpus, FEW_QUESTION, None, "", now=now, coverage=0.0) == [held]
    # Every word of the question is a bar of 1.0.
    assert ranked(corpus, "tomato garden", None, "", now=now, coverage=1.0) == [held]
    assert ranked(corpus, HALF_QUESTION, None, "", now=now, coverage=1.0) == []


def test_ranked_leaves_japanese_particles_out_of_the_question_words() -> None:
    dog = doc_item("d:1", "User: 私の犬の名前はポチです。\nmelo: 覚えました。")
    meeting = doc_item(
        "d:2", "User: 明日の会議は午後三時からです。\nmelo: 了解しました。"
    )
    hobby = doc_item("d:3", "User: 母の趣味は庭の手入れです。\nmelo: いいですね。")
    corpus = Corpus()
    for item in (dog, meeting, hobby):
        corpus.add(item)
    now = NOW + 1
    # An ordinary question still finds its turn.
    assert ranked(corpus, "私の犬の名前は何ですか", None, "", now=now) == [dog]
    assert ranked(corpus, "明日の会議は何時からですか", None, "", now=now) == [meeting]
    # A message that is one particle names nothing, so no turn that holds it counts.
    for particle in "のはがをにへでともかねよ":
        assert ranked(corpus, particle, None, "", now=now) == []


def test_ranked_with_the_default_bars_is_ranked_with_the_constants() -> None:
    corpus = Corpus()
    for item in (
        doc_item("d:1", "I stake the tomato garden with bamboo.", 0.70),
        doc_item("d:2", "the tomato garden gets the sun", 0.40),
        doc_item("d:3", "a dog barks in the yard", 0.58),
        doc_item("d:4", "zebra yodel contest tickets", 0.20),
    ):
        corpus.add(item)
    for question in (HALF_QUESTION, FAR_QUESTION, "tomato garden"):
        assert ranked(corpus, question, QUERY_VECTOR, MODEL, now=NOW + 1) == ranked(
            corpus,
            question,
            QUERY_VECTOR,
            MODEL,
            now=NOW + 1,
            coverage=MIN_COVERAGE,
            min_cosine=MIN_COSINE,
            word_weight="count",
        )


async def test_a_turn_joins_the_block_above_the_cosine_bar_and_not_below_it(
    tmp_path: Path, store: SessionStore
) -> None:
    for cosine, joins in ((ABOVE, True), (BELOW, False)):
        memory = await memory_with(
            tmp_path / str(cosine), store, config(), cosine=cosine
        )
        try:
            found = await memory.before_turn(
                store, await _turn(store, "Ask"), FAR_QUESTION
            )
            assert [item.key[:2] for item in found.used] == (["d:"] if joins else [])
            assert (TURN[0] in found.block) is joins
        finally:
            await memory.aclose()


async def test_the_search_tool_honours_the_cosine_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    for cosine, joins in ((ABOVE, True), (BELOW, False)):
        memory = await memory_with(
            tmp_path / str(cosine), store, config(), cosine=cosine
        )
        tools = MemoryTools(memory)
        try:
            found = await memory.search(FAR_QUESTION)
            assert [item.key[:2] for item in found] == (["d:"] if joins else [])
            text = await tools.search(query=FAR_QUESTION, scope="chats")
            if joins:
                assert text.startswith("1. ") and TURN[0] in text
            else:
                assert text == NOTHING_FOUND
        finally:
            await memory.aclose()


async def test_half_the_question_words_join_the_block_and_a_quarter_does_not(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = await memory_with(tmp_path, store, config(), note=True)
    try:
        found = await memory.before_turn(
            store, await _turn(store, "Ask"), f"which {HALF_QUESTION}?"
        )
        assert sorted(item.key[:2] for item in found.used) == ["d:", "r:"]
        few = await memory.before_turn(
            store, await _turn(store, "Ask 2"), f"which {FEW_QUESTION}?"
        )
        assert few.used == ()
    finally:
        await memory.aclose()


async def test_the_search_tool_honours_the_coverage_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    memory = await memory_with(tmp_path, store, config(), note=True)
    tools = MemoryTools(memory)
    try:
        found = await memory.search(HALF_QUESTION)
        assert sorted(item.key[:2] for item in found) == ["d:", "r:"]
        assert (await tools.search(query=HALF_QUESTION)).startswith("1. ")
        assert len((await tools.search(query=HALF_QUESTION)).splitlines()) == 2
        assert await memory.search(FEW_QUESTION) == []
        assert await tools.search(query=FEW_QUESTION) == NOTHING_FOUND
    finally:
        await memory.aclose()


async def test_the_llm_retriever_keeps_its_loose_candidates_whatever_the_bar(
    tmp_path: Path, store: SessionStore
) -> None:
    """The model chooses among candidates that took no coverage bar, so a question
    that holds too few of the turn's words for the bar still has them."""
    cfg = config(retriever="llm")
    memory = await memory_with(tmp_path, store, cfg)
    chat = FakeChat(needle="tomato")
    memory.chat_provider = lambda: chat  # type: ignore[method-assign]
    try:
        found = await memory.search(FEW_QUESTION)
        assert [item.key[:2] for item in found] == ["d:"]
        assert chat.calls  # the model was asked about the loose candidates
    finally:
        await memory.aclose()


# ── recall_word_weight = "rare": a shared word weighs by how few items hold it ──

# Two content words. "group" is in nine of the ten items, "dentist" in one.
GROUP_DENTIST = "group dentist"
# Three content words: "visit" is in the eight items that hold "group" too.
THREE_WORDS = "when is my group dentist visit"


def common_words_corpus() -> tuple[Corpus, list[Item], Item, Item]:
    """Eight turns that hold "group" and "visit", one that holds only "group" and
    one that holds only "dentist"."""
    corpus = Corpus()
    both = [doc_item(f"d:{n}", f"the group visit plan number {n}") for n in range(8)]
    group_only = doc_item("d:group", "my group is loud on friday")
    dentist = doc_item("d:dentist", "the dentist is on friday")
    for item in (*both, group_only, dentist):
        corpus.add(item)
    return corpus, both, group_only, dentist


def keys_of(
    corpus: Corpus, question: str, weight: str, coverage: float = MIN_COVERAGE
) -> list[str]:
    found = ranked(
        corpus,
        question,
        None,
        "",
        now=NOW + 1,
        coverage=coverage,
        word_weight=weight,
    )
    return sorted(item.key for item in found)


def test_rare_drops_a_turn_that_shares_only_a_word_most_items_hold() -> None:
    corpus, both, group_only, dentist = common_words_corpus()
    every = sorted(item.key for item in (*both, group_only, dentist))
    # One of two words is enough for the count bar, so "group" alone passes it.
    assert keys_of(corpus, GROUP_DENTIST, "count") == every
    # The default is the count.
    assert [i.key for i in ranked(corpus, GROUP_DENTIST, None, "", now=NOW + 1)] == [
        i.key
        for i in ranked(
            corpus, GROUP_DENTIST, None, "", now=NOW + 1, word_weight="count"
        )
    ]
    # By weight "group" is under a third of the question and "dentist" the rest.
    assert keys_of(corpus, GROUP_DENTIST, "rare") == [dentist.key]


def test_rare_keeps_the_turn_with_the_rare_word_of_a_longer_question() -> None:
    corpus, both, group_only, dentist = common_words_corpus()
    # Of three words the count bar needs two: a turn with "dentist" alone is out,
    # and so is one with "group" alone; the turns with "group" and "visit" are in.
    assert keys_of(corpus, THREE_WORDS, "count") == sorted(i.key for i in both)
    # By weight "dentist" alone is more than 0.4 of the question and "group" alone
    # is not, and "group" with "visit" is just above it.
    assert keys_of(corpus, THREE_WORDS, "rare") == sorted(
        i.key for i in (*both, dentist)
    )
    # The share is of the weight: at 0.5 "group" with "visit" falls short too.
    assert keys_of(corpus, THREE_WORDS, "rare", coverage=0.5) == [dentist.key]
    assert keys_of(corpus, THREE_WORDS, "count", coverage=0.5) == sorted(
        i.key for i in both
    )


def test_rare_bar_of_one_holds_only_the_items_with_every_word() -> None:
    corpus = Corpus()
    both = doc_item("d:both", "the group dentist visit is on friday")
    corpus.add(both)
    for n in range(4):
        corpus.add(doc_item(f"d:{n}", f"the group lunch number {n}"))
    corpus.add(doc_item("d:dentist", "the dentist is on friday"))
    # A bar of exactly the whole weight is met when every word is held, whatever
    # order the weights are added in.
    assert keys_of(corpus, GROUP_DENTIST, "rare", coverage=1.0) == [both.key]
    assert keys_of(corpus, GROUP_DENTIST, "count", coverage=1.0) == [both.key]


def test_rare_still_needs_one_shared_word_when_the_bar_is_zero() -> None:
    corpus = Corpus()
    loud = doc_item("d:loud", "the group is loud")
    corpus.add(loud)
    corpus.add(doc_item("d:weather", "the weather is mild"))
    # "the" is not a content word, but the weather turn holds it, so the index finds
    # the turn; it shares no content word with the question.
    for weight in ("count", "rare"):
        assert keys_of(corpus, "the group", weight, coverage=0.0) == [loud.key]


def test_rare_acts_as_count_when_no_word_of_the_question_is_in_an_item() -> None:
    corpus, *_ = common_words_corpus()
    for question in (FAR_QUESTION, "zebra yodel quartz"):
        assert keys_of(corpus, question, "count") == []
        assert keys_of(corpus, question, "rare") == []
    assert keys_of(Corpus(), GROUP_DENTIST, "rare") == []
    # A word no item holds weighs most (df = 0), as ln(1 + N), and one that every
    # item holds weighs least.
    weights = corpus.rarity(["zebra", "group", "dentist"])
    assert weights["zebra"] == pytest.approx(math.log(1 + 10))
    assert weights["zebra"] > weights["dentist"] > weights["group"] > 0.0
    assert weights["group"] == pytest.approx(math.log(1 + 10 / (1 + 9)))


def test_the_document_counts_are_kept_until_an_item_is_added_or_removed() -> None:
    corpus, _, group_only, dentist = common_words_corpus()
    counts = corpus.doc_freq()
    assert (counts["group"], counts["dentist"], counts["zebra"]) == (9, 1, 0)
    # The same counts again, with nothing rebuilt.
    assert corpus.doc_freq() is counts
    corpus.add(doc_item("d:extra", "a dentist and a zebra"))
    after_add = corpus.doc_freq()
    assert after_add is not counts
    assert (after_add["dentist"], after_add["zebra"]) == (2, 1)
    corpus.remove(dentist.key)
    corpus.remove(group_only.key)
    after_remove = corpus.doc_freq()
    assert (after_remove["group"], after_remove["dentist"]) == (8, 1)
    # A key that is gone leaves the weights by the new item count.
    assert corpus.rarity(["group"])["group"] == pytest.approx(math.log(1 + 9 / (1 + 8)))


async def _add_turns(memory: ChatMemory, store: SessionStore, texts: list[str]) -> None:
    """Each of *texts* stored as the typed words of a chat turn of a chat of its own."""
    _, index = memory.stores()
    for text in texts:
        turn = await _turn(store, text)
        await index.add(
            Document(
                session_id=turn.session_id,
                turn=1,
                surface="gateway",
                kind="chat",
                title=text,
                text=document_text(text, "Noted."),
                created_at=time.time(),
            )
        )


GROUP_TURN = "my group is loud on friday"
DENTIST_TURN = "the dentist is on friday"


@pytest.mark.parametrize(
    ("setting", "turns_found"),
    # With the count every turn that holds "group" or "dentist" is found (the block
    # and the tool's default cap them); with the weight only the dentist turn is.
    [(None, 8), ("count", 8), ("rare", 1)],
)
async def test_the_setting_reaches_the_ranking_of_the_search_and_the_block(
    tmp_path: Path, store: SessionStore, setting: str | None, turns_found: int
) -> None:
    cfg = config() if setting is None else config(recall_word_weight=setting)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    await _add_turns(
        memory, store, [f"the group visit plan number {n}" for n in range(6)]
    )
    await _add_turns(memory, store, [GROUP_TURN, DENTIST_TURN])
    try:
        found = await memory.search(GROUP_DENTIST, docs_k=20)
        assert len(found) == turns_found
        assert any(DENTIST_TURN in item.text for item in found)
        assert any(GROUP_TURN in item.text for item in found) is (turns_found > 1)
        block = await memory.before_turn(
            store, await _turn(store, "Ask"), f"which {GROUP_DENTIST}?"
        )
        assert DENTIST_TURN in block.block
        assert len(block.used) == min(turns_found, 2)
    finally:
        await memory.aclose()


def test_the_weight_of_the_settings_now_is_count_unless_rare_is_set() -> None:
    assert ChatMemory(lambda: None).word_weight() == "count"
    assert ChatMemory(lambda: Config()).word_weight() == "count"
    for value, weight in (("rare", "rare"), ("count", "count"), ("odd", "count")):
        cfg = config(recall_word_weight=value)
        assert ChatMemory(lambda cfg=cfg: cfg).word_weight() == weight  # type: ignore[misc]
