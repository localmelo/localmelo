"""The rules every retriever follows (R3 spec, Requirements 8 and 12).

Each rule runs on the three retrievers: ``bm25``; ``llm`` with a ``send``
that records its calls; ``embedding`` with a fake provider that counts the
texts it embeds. Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.melo.memory.retrieval import (
    DOC_KINDS,
    RETRIEVER_NAMES,
    Doc,
    Retriever,
    build_retriever,
)
from localmelo.melo.memory.retrieval.bm25 import BM25Retriever
from localmelo.melo.memory.retrieval.embedding import EmbeddingRetriever
from localmelo.melo.memory.retrieval.llm import LLMRetriever
from localmelo.melo.schema import Message


class RecordingSend:
    """A ``send`` that records its calls and answers with fixed content."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[Message]] = []

    async def __call__(self, messages: list[Message]) -> Message:
        self.calls.append(messages)
        return Message("assistant", self.content)


def stub_vector(text: str) -> list[float]:
    """The stub server's embedding: 16 numbers, each lowercase token adds 1
    at ``sha256(token) % 16``."""
    vector = [0.0] * 16
    for token in text.lower().split():
        vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % 16] += 1
    return vector


class CountingEmbedding(BaseEmbeddingProvider):
    """A fake embedding provider that counts the texts it embeds."""

    model = "fake-embedding"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [stub_vector(text) for text in texts]

    async def close(self) -> None:
        pass


@dataclass
class Subject:
    """A retriever under test, and a count of the requests it sent."""

    retriever: Retriever
    requests: Callable[[], int]


# The llm retriever's answer names every note of DOCS, an unknown alias and
# a draft that is not there; the excluded ones must still never come back.
LLM_REPLY = '{"ids": ["n1", "n2", "n3", "n4", "x9", "d7"]}'


def _subject(name: str) -> Subject:
    if name == "llm":
        send = RecordingSend(LLM_REPLY)
        return Subject(build_retriever("llm", send=send), lambda: len(send.calls))
    if name == "embedding":
        provider = CountingEmbedding()
        return Subject(
            build_retriever("embedding", embedding=provider),
            lambda: len(provider.texts),
        )
    return Subject(build_retriever("bm25"), lambda: 0)


# Each doc shares at least two words with QUERY, so every retriever finds
# them all.
DOCS = [
    Doc("t:n1", "note", "the foldr argument order is reversed"),
    Doc("t:n2", "note", "foldr argument order fixed in the draft"),
    Doc("t:n3", "note", "the foldr argument order needs a test"),
    Doc("t:n4", "note", "foldr argument order: accumulator first"),
]
QUERY = "foldr argument order"
BM25_OPTIONS = ("floor_terms", "relative_cut")


@pytest.fixture(params=RETRIEVER_NAMES)
def subject(request: pytest.FixtureRequest) -> Subject:
    return _subject(request.param)


# ── Doc ─────────────────────────────────────────────────────────────────────


class TestDoc:
    @pytest.mark.parametrize("doc_id", ["", "a b", "a\tb", " a", "a\n", "a　b"])
    def test_refuses_an_empty_or_whitespace_holding_id(self, doc_id: str) -> None:
        with pytest.raises(ValueError, match="doc_id"):
            Doc(doc_id, "note", "text")

    def test_refuses_an_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="'chunk'"):
            Doc("a", "chunk", "text")

    @pytest.mark.parametrize("kind", DOC_KINDS)
    def test_takes_every_kind(self, kind: str) -> None:
        doc = Doc("t-en:n3", kind, "text")
        assert (doc.kind, doc.title, dict(doc.meta)) == (kind, "", {})


# ── search ──────────────────────────────────────────────────────────────────


class TestSearchRules:
    async def test_every_retriever_is_a_retriever(self, subject: Subject) -> None:
        assert isinstance(subject.retriever, Retriever)

    async def test_k_below_1_raises(self, subject: Subject) -> None:
        with pytest.raises(ValueError, match="k must be at least 1"):
            await subject.retriever.search(QUERY, DOCS, k=0)
        assert subject.requests() == 0

    async def test_two_docs_with_one_id_raise(self, subject: Subject) -> None:
        docs = [*DOCS, Doc("t:n2", "draft", "foldr argument order")]
        with pytest.raises(ValueError, match="'t:n2'"):
            await subject.retriever.search(QUERY, docs, k=3)
        assert subject.requests() == 0

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    async def test_a_blank_query_finds_nothing_and_sends_nothing(
        self, subject: Subject, query: str
    ) -> None:
        result = await subject.retriever.search(query, DOCS, k=3)
        assert result.hits == []
        assert result.fallback == ""
        assert result.retriever == subject.retriever.name
        assert subject.requests() == 0

    async def test_no_docs_left_after_exclude_finds_nothing_and_sends_nothing(
        self, subject: Subject
    ) -> None:
        exclude = {doc.doc_id for doc in DOCS}
        result = await subject.retriever.search(QUERY, DOCS, k=3, exclude=exclude)
        assert result.hits == []
        assert result.fallback == ""
        assert subject.requests() == 0

    async def test_no_docs_finds_nothing_and_sends_nothing(
        self, subject: Subject
    ) -> None:
        result = await subject.retriever.search(QUERY, [], k=3)
        assert result.hits == []
        assert subject.requests() == 0

    async def test_hits_stay_inside_docs_and_outside_exclude(
        self, subject: Subject
    ) -> None:
        # t:n9 is excluded but not among the docs: it is ignored.
        exclude = frozenset({"t:n1", "t:n3", "t:n9"})
        result = await subject.retriever.search(QUERY, DOCS, k=4, exclude=exclude)
        assert result.fallback == ""
        assert {hit.doc_id for hit in result.hits} == {"t:n2", "t:n4"}
        assert subject.requests() > 0 or subject.retriever.name == "bm25"

    async def test_hits_are_at_most_k_and_ranked_from_1(self, subject: Subject) -> None:
        for k in (1, 2, 3):
            result = await subject.retriever.search(QUERY, DOCS, k=k)
            assert len(result.hits) == k
            assert [hit.rank for hit in result.hits] == list(range(1, k + 1))
            ids = [hit.doc_id for hit in result.hits]
            assert len(set(ids)) == k
            assert set(ids) <= {doc.doc_id for doc in DOCS}

    async def test_ms_is_a_whole_number_of_milliseconds(self, subject: Subject) -> None:
        result = await subject.retriever.search(QUERY, DOCS, k=2)
        assert isinstance(result.ms, int)
        assert result.ms >= 0


# ── build_retriever ─────────────────────────────────────────────────────────


async def _send(messages: list[Message]) -> Message:
    return Message("assistant", '{"ids": []}')


def _bm25_of(retriever: Retriever) -> BM25Retriever:
    """*retriever* itself for bm25; the bm25 fallback of the other two."""
    if isinstance(retriever, LLMRetriever | EmbeddingRetriever):
        return retriever._fallback
    assert isinstance(retriever, BM25Retriever)
    return retriever


class TestBuildRetriever:
    def test_an_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="bm25, llm, embedding") as info:
            build_retriever("dense")
        assert "'dense'" in str(info.value)

    def test_llm_needs_send(self) -> None:
        with pytest.raises(ValueError, match="needs send"):
            build_retriever("llm")

    def test_embedding_needs_a_provider(self) -> None:
        with pytest.raises(ValueError, match="embedding provider"):
            build_retriever("embedding")

    @pytest.mark.parametrize(
        ("name", "option"),
        [
            ("bm25", "timeout_s"),
            ("bm25", "min_score"),
            ("bm25", "batch_size"),
            ("llm", "min_score"),
            ("llm", "batch_size"),
            ("embedding", "timeout_s"),
            ("bm25", "floor"),
        ],
    )
    def test_an_unknown_option_raises(self, name: str, option: str) -> None:
        with pytest.raises(ValueError, match=f"unknown option {option}"):
            build_retriever(
                name, send=_send, embedding=CountingEmbedding(), **{option: 1}
            )

    @pytest.mark.parametrize(
        ("name", "option", "value"),
        [
            ("bm25", "floor_terms", -1),
            ("bm25", "floor_terms", 1.5),
            ("bm25", "floor_terms", True),
            ("bm25", "relative_cut", -0.01),
            ("bm25", "relative_cut", 1),
            ("bm25", "relative_cut", 1.0),
            ("bm25", "relative_cut", float("nan")),
            ("llm", "floor_terms", -1),
            ("llm", "relative_cut", 1.5),
            ("llm", "timeout_s", 0),
            ("llm", "timeout_s", -1.0),
            ("llm", "timeout_s", float("nan")),
            ("embedding", "floor_terms", -2),
            ("embedding", "relative_cut", -1),
            ("embedding", "batch_size", 0),
            ("embedding", "batch_size", 2.0),
            ("embedding", "min_score", float("nan")),
            ("embedding", "min_score", "0.5"),
        ],
    )
    def test_an_out_of_range_value_raises(
        self, name: str, option: str, value: object
    ) -> None:
        with pytest.raises(ValueError, match=option):
            build_retriever(
                name, send=_send, embedding=CountingEmbedding(), **{option: value}
            )

    @pytest.mark.parametrize(
        ("name", "options"),
        [
            ("bm25", {"floor_terms": 0, "relative_cut": 0}),
            ("bm25", {"floor_terms": 5, "relative_cut": 0.99}),
            ("llm", {"timeout_s": 0.01, "floor_terms": 3, "relative_cut": 0.5}),
            ("embedding", {"min_score": -1, "batch_size": 1, "floor_terms": 0}),
        ],
    )
    def test_values_in_range_are_taken(
        self, name: str, options: dict[str, object]
    ) -> None:
        retriever = build_retriever(
            name, send=_send, embedding=CountingEmbedding(), **options
        )
        assert retriever.name == name
        for option, value in options.items():
            # The bm25 options of llm and embedding go to their fallback.
            owner = _bm25_of(retriever) if option in BM25_OPTIONS else retriever
            assert getattr(owner, option) == value

    def test_the_defaults(self) -> None:
        bm25 = build_retriever("bm25")
        assert isinstance(bm25, BM25Retriever)
        assert (bm25.floor_terms, bm25.relative_cut) == (2, 0.2)
        llm = build_retriever("llm", send=_send)
        assert isinstance(llm, LLMRetriever)
        assert llm.timeout_s == 30.0
        embedding = build_retriever("embedding", embedding=CountingEmbedding())
        assert isinstance(embedding, EmbeddingRetriever)
        assert (embedding.min_score, embedding.batch_size) == (0.0, 32)
        for retriever in (llm, embedding):
            fallback = _bm25_of(retriever)
            assert (fallback.floor_terms, fallback.relative_cut) == (2, 0.2)

    async def test_each_call_gives_a_new_instance_with_empty_caches(self) -> None:
        provider = CountingEmbedding()
        send = RecordingSend('{"ids": []}')
        for name in RETRIEVER_NAMES:
            first = build_retriever(name, send=send, embedding=provider)
            second = build_retriever(name, send=send, embedding=provider)
            assert first is not second
            assert first.name == second.name == name
            await first.search(QUERY, DOCS, k=2)

            # Whatever the first one cached, the second one starts empty.
            bm25 = _bm25_of(second)
            assert len(bm25._index) == 0
            assert bm25._texts == {}
            if isinstance(second, EmbeddingRetriever):
                assert second._vectors == {}
                embedded = len(provider.texts)
                await second.search(QUERY, DOCS, k=2)
                # The query and every doc are embedded again.
                assert len(provider.texts) - embedded == 1 + len(DOCS)

    def test_a_class_is_not_a_retriever_instance(self) -> None:
        # So evaluate() can take a class as a factory.
        for cls in (BM25Retriever, LLMRetriever, EmbeddingRetriever):
            assert not isinstance(cls, Retriever)
