"""The ``embedding`` retriever: the cache, cosine ranking, refused batches.

A fake provider looks each text up in a table of 3-dimensional vectors
chosen so that every cosine with the query is known: no network.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, ProviderError
from localmelo.melo.memory.retrieval import Doc, Hit
from localmelo.melo.memory.retrieval.bm25 import BM25Retriever
from localmelo.melo.memory.retrieval.embedding import EmbeddingRetriever

QUERY = "foldr argument order"
DOCS = [
    Doc("a", "note", "foldr argument order alpha"),
    Doc("b", "note", "foldr argument order bravo"),
    Doc("c", "note", "foldr argument order charlie"),
    Doc("d", "note", "foldr argument order delta"),
    Doc("e", "note", "foldr argument order echo"),
]
# Cosine with the query's [1, 0, 0]: a 0.6, b 1.0, c 0.8, d 0.0, e -1.0.
VECTORS: dict[str, list[float]] = {
    QUERY: [1.0, 0.0, 0.0],
    "foldr argument order alpha": [0.6, 0.8, 0.0],
    "foldr argument order bravo": [1.0, 0.0, 0.0],
    "foldr argument order charlie": [0.8, 0.6, 0.0],
    "foldr argument order delta": [0.0, 1.0, 0.0],
    "foldr argument order echo": [-1.0, 0.0, 0.0],
    # f ties with b: the same direction, twice as long.
    "foldr argument order foxtrot": [2.0, 0.0, 0.0],
    "foldr argument order zero": [0.0, 0.0, 0.0],
    "a zero query": [0.0, 0.0, 0.0],
    "another query": [0.0, 1.0, 0.0],
}
# Any other text: orthogonal to the query.
OTHER_VECTOR = [0.0, 0.0, 1.0]

Hook = Callable[[list[str], list[list[float]]], list[list[float]]]


class TableEmbedding(BaseEmbeddingProvider):
    """Embeds each text as VECTORS says (else OTHER_VECTOR), and records requests.

    ``hook``, when set, sees each request's texts and vectors and returns
    what the provider answers, or raises.
    """

    def __init__(self, model: object = "table-embedding") -> None:
        self.model = model
        self.requests: list[list[str]] = []
        self.hook: Hook | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(list(texts))
        vectors = [list(VECTORS.get(text, OTHER_VECTOR)) for text in texts]
        if self.hook is not None:
            return self.hook(texts, vectors)
        return vectors

    async def close(self) -> None:
        pass

    @property
    def texts(self) -> list[str]:
        return [text for request in self.requests for text in request]


def _ids(hits: list[Hit]) -> list[str]:
    return [hit.doc_id for hit in hits]


async def _bm25_hits(docs: list[Doc], exclude: frozenset[str]) -> list[Hit]:
    return (await BM25Retriever().search(QUERY, docs, k=10, exclude=exclude)).hits


# ── Ranking ─────────────────────────────────────────────────────────────────


async def test_cosine_order() -> None:
    docs = [
        *DOCS,
        Doc("f", "note", "foldr argument order foxtrot"),
        Doc("z", "note", "foldr argument order zero"),
    ]
    retriever = EmbeddingRetriever(TableEmbedding())
    result = await retriever.search(QUERY, docs, k=10)
    # e scores -1.0, under min_score 0.0; z has a zero norm and is skipped;
    # f ties with b and comes after it, as in docs; d scores 0.0, which is
    # not under 0.0.
    assert _ids(result.hits) == ["b", "f", "c", "a", "d"]
    assert [hit.score for hit in result.hits] == pytest.approx(
        [1.0, 1.0, 0.8, 0.6, 0.0]
    )
    assert [hit.rank for hit in result.hits] == [1, 2, 3, 4, 5]
    assert (result.retriever, result.fallback, result.usage) == ("embedding", "", {})


async def test_k_cuts_the_ranking() -> None:
    result = await EmbeddingRetriever(TableEmbedding()).search(QUERY, DOCS, k=2)
    assert _ids(result.hits) == ["b", "c"]


@pytest.mark.parametrize(
    ("min_score", "expected"),
    [
        (0.7, ["b", "c"]),
        (0.6, ["b", "c", "a"]),
        (-1.0, ["b", "c", "a", "d", "e"]),
        (1.0, ["b"]),
    ],
)
async def test_min_score(min_score: float, expected: list[str]) -> None:
    retriever = EmbeddingRetriever(TableEmbedding(), min_score=min_score)
    result = await retriever.search(QUERY, DOCS, k=10)
    assert _ids(result.hits) == expected


async def test_a_zero_query_finds_nothing() -> None:
    result = await EmbeddingRetriever(TableEmbedding()).search(
        "a zero query", DOCS, k=3
    )
    assert (result.hits, result.retriever, result.fallback) == ([], "embedding", "")


# ── The cache ───────────────────────────────────────────────────────────────


async def test_docs_are_embedded_once_in_batches() -> None:
    provider = TableEmbedding()
    retriever = EmbeddingRetriever(provider, batch_size=2)
    await retriever.search(QUERY, DOCS, k=3)
    assert provider.requests == [
        [QUERY],
        ["foldr argument order alpha", "foldr argument order bravo"],
        ["foldr argument order charlie", "foldr argument order delta"],
        ["foldr argument order echo"],
    ]
    # The query is embedded on every call; the docs only once.
    result = await retriever.search("another query", DOCS, k=3)
    assert provider.requests[4:] == [["another query"]]
    assert _ids(result.hits) == ["d", "a", "c"]


async def test_the_default_batch_is_32_texts() -> None:
    docs = [Doc(f"n{i}", "note", QUERY) for i in range(70)]
    provider = TableEmbedding()
    await EmbeddingRetriever(provider).search(QUERY, docs, k=3)
    assert [len(request) for request in provider.requests] == [1, 32, 32, 6]


async def test_excluded_docs_are_not_embedded() -> None:
    provider = TableEmbedding()
    retriever = EmbeddingRetriever(provider)
    result = await retriever.search(QUERY, DOCS, k=5, exclude={"b", "x"})
    assert "foldr argument order bravo" not in provider.texts
    assert _ids(result.hits) == ["c", "a", "d"]
    # Once no longer excluded, b is embedded, alone.
    await retriever.search(QUERY, DOCS, k=5)
    assert provider.requests[-1] == ["foldr argument order bravo"]


async def test_a_doc_whose_text_changed_is_embedded_again() -> None:
    provider = TableEmbedding()
    retriever = EmbeddingRetriever(provider)
    first = await retriever.search(
        QUERY, [Doc("a", "note", "foldr argument order echo")], k=3
    )
    assert first.hits == []  # cosine -1.0
    second = await retriever.search(
        QUERY, [Doc("a", "note", "foldr argument order bravo")], k=3
    )
    assert provider.requests[-1] == ["foldr argument order bravo"]
    assert _ids(second.hits) == ["a"]


# ── Refusals and fallbacks ──────────────────────────────────────────────────


def _drop_last(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
    return vectors[:-1]


def _one_more(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
    return [*vectors, [1.0, 0.0, 0.0]]


def _spoil(value: object) -> Hook:
    """A hook that puts *value* in the first vector of a docs batch."""

    def hook(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
        if texts != [QUERY]:
            vectors[0][1] = value  # type: ignore[call-overload]
        return vectors

    return hook


def _empty(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
    return [[] for _ in vectors]


def _ragged(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
    if texts != [QUERY]:
        vectors[-1] = [*vectors[-1], 0.0]
    return vectors


def _not_a_list(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
    return {"data": vectors}  # type: ignore[return-value]


def _down(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
    raise ProviderError("POST /embeddings failed with HTTP 503", status_code=503)


@pytest.mark.parametrize(
    "hook",
    [
        _drop_last,
        _one_more,
        _spoil(math.nan),
        _spoil(math.inf),
        _spoil("0.5"),
        _spoil(True),
        _spoil(None),
        _spoil(10**400),
        _empty,
        _ragged,
        _not_a_list,
        _down,
    ],
    ids=[
        "too-few",
        "too-many",
        "nan",
        "inf",
        "string",
        "bool",
        "none",
        "huge-int",
        "empty",
        "ragged",
        "not-a-list",
        "provider-error",
    ],
)
async def test_a_refused_batch_falls_back_to_bm25(hook: Hook) -> None:
    provider = TableEmbedding()
    provider.hook = hook
    result = await EmbeddingRetriever(provider).search(QUERY, DOCS, k=10, exclude={"a"})
    assert (result.retriever, result.fallback, result.usage) == (
        "bm25",
        "provider_error",
        {},
    )
    assert result.hits == await _bm25_hits(DOCS, frozenset({"a"}))
    assert result.hits


async def test_a_dimension_change_falls_back() -> None:
    provider = TableEmbedding()
    retriever = EmbeddingRetriever(provider)
    assert retriever.dim is None
    await retriever.search(QUERY, DOCS, k=3)
    assert retriever.dim == 3

    # Another model behind the same endpoint: 4 dimensions.
    provider.hook = lambda texts, vectors: [[*vector, 0.0] for vector in vectors]
    result = await retriever.search(QUERY, DOCS, k=3)
    assert (result.retriever, result.fallback) == ("bm25", "provider_error")
    assert retriever.dim == 3


async def test_a_refused_batch_is_not_cached_and_earlier_ones_are() -> None:
    provider = TableEmbedding()
    retriever = EmbeddingRetriever(provider, batch_size=2)

    def refuse_charlie(
        texts: list[str], vectors: list[list[float]]
    ) -> list[list[float]]:
        if "foldr argument order charlie" in texts:
            vectors[0][0] = math.nan
        return vectors

    provider.hook = refuse_charlie
    result = await retriever.search(QUERY, DOCS, k=3)
    assert result.fallback == "provider_error"
    assert len(provider.requests) == 3  # the query, a and b, then c and d

    provider.hook = None
    result = await retriever.search(QUERY, DOCS, k=3)
    assert (result.retriever, result.fallback) == ("embedding", "")
    # a and b were cached; c and d were not, and e was never reached.
    assert provider.requests[3:] == [
        [QUERY],
        ["foldr argument order charlie", "foldr argument order delta"],
        ["foldr argument order echo"],
    ]
    assert _ids(result.hits) == ["b", "c", "a"]


async def test_another_exception_propagates() -> None:
    provider = TableEmbedding()

    def broken(texts: list[str], vectors: list[list[float]]) -> list[list[float]]:
        raise RuntimeError("a bug")

    provider.hook = broken
    with pytest.raises(RuntimeError, match="a bug"):
        await EmbeddingRetriever(provider).search(QUERY, DOCS, k=3)


async def test_the_fallback_takes_the_floor_options() -> None:
    provider = TableEmbedding()
    provider.hook = _down
    docs = [Doc("a", "note", "read list_ops.py: 120 chars")]
    floored = await EmbeddingRetriever(provider).search("read list_ops.py", docs, k=3)
    assert floored.hits == []
    unfloored = await EmbeddingRetriever(provider, floor_terms=0).search(
        "read list_ops.py", docs, k=3
    )
    assert _ids(unfloored.hits) == ["a"]


# ── model and dim ───────────────────────────────────────────────────────────


class Anonymous(BaseEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0] for _ in texts]

    async def close(self) -> None:
        pass


async def test_model_and_dim() -> None:
    retriever = EmbeddingRetriever(TableEmbedding(model="qwen3-embedding"))
    assert (retriever.model, retriever.dim) == ("qwen3-embedding", None)
    await retriever.search(QUERY, DOCS, k=1)
    assert retriever.dim == 3

    anonymous = EmbeddingRetriever(Anonymous())
    assert anonymous.model == ""
    await anonymous.search(QUERY, DOCS, k=1)
    assert anonymous.dim == 2

    assert EmbeddingRetriever(TableEmbedding(model=None)).model == ""
