"""``embedding``: cosine similarity between the query's vector and each doc's.

The reference retriever. Each doc is embedded once, in batches, and its
vector cached by ``doc_id``; the query is embedded on every call. Every
vector must have the dimension of the first one accepted, so a vector of
another model or dimension never mixes with the cached ones (review E2).
A :class:`~localmelo.melo.contracts.providers.ProviderError` or a refused
batch falls back to ``bm25`` with ``fallback = "provider_error"``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Any

import numpy as np

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, ProviderError
from localmelo.melo.memory.retrieval.base import (
    Doc,
    Hit,
    RetrievalResult,
    elapsed_ms,
    is_number,
    ranked,
    search_candidates,
)
from localmelo.melo.memory.retrieval.bm25 import (
    DEFAULT_FLOOR_TERMS,
    DEFAULT_RELATIVE_CUT,
    BM25Retriever,
)

DEFAULT_MIN_SCORE = 0.0
DEFAULT_BATCH_SIZE = 32


class _RefusedBatchError(Exception):
    """The provider's vectors for a batch break the rules; none is kept."""


def _check_embedding_options(min_score: object, batch_size: object) -> None:
    """Raise :class:`ValueError` unless both options are in range."""
    if not is_number(min_score) or min_score != min_score:  # NaN
        raise ValueError(f"min_score must be a number, got {min_score!r}")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}")


class EmbeddingRetriever:
    """Ranks docs by the cosine of their vector with the query's.

    ``model`` is the provider's ``model`` attribute, or ``""``; ``dim`` is
    the dimension of the first vector accepted, ``None`` before. Docs
    whose vector has a zero norm are skipped, a query with one gives no
    hits, and scores under ``min_score`` are dropped. ``usage`` is ``{}``.
    """

    name: str

    def __init__(
        self,
        embedding: BaseEmbeddingProvider,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        floor_terms: int = DEFAULT_FLOOR_TERMS,
        relative_cut: float = DEFAULT_RELATIVE_CUT,
    ) -> None:
        _check_embedding_options(min_score, batch_size)
        self.name = "embedding"
        model = getattr(embedding, "model", "")
        self.model: str = model if isinstance(model, str) else ""
        self.dim: int | None = None
        self.min_score = min_score
        self.batch_size = batch_size
        self._provider = embedding
        self._fallback = BM25Retriever(
            floor_terms=floor_terms, relative_cut=relative_cut
        )
        # doc_id -> (the text embedded, its unit vector or None for a zero norm)
        self._vectors: dict[str, tuple[str, np.ndarray | None]] = {}

    async def search(
        self,
        query: str,
        docs: Sequence[Doc],
        *,
        k: int,
        exclude: AbstractSet[str] = frozenset(),
    ) -> RetrievalResult:
        start = time.perf_counter()
        candidates = search_candidates(query, docs, k=k, exclude=exclude)
        if not candidates:
            return RetrievalResult(hits=[], retriever=self.name, ms=elapsed_ms(start))
        try:
            [query_vector] = await self._embed([query])
            await self._embed_docs(candidates)
        except (ProviderError, _RefusedBatchError):
            answer = await self._fallback.search(query, docs, k=k, exclude=exclude)
            return RetrievalResult(
                hits=answer.hits,
                retriever=answer.retriever,
                ms=elapsed_ms(start),
                fallback="provider_error",
            )
        return RetrievalResult(
            hits=self._rank(query_vector, candidates, k),
            retriever=self.name,
            ms=elapsed_ms(start),
        )

    async def _embed_docs(self, candidates: Sequence[Doc]) -> None:
        """Embed the candidates not yet cached, or whose text changed.

        In ``docs`` order, :attr:`batch_size` at a time; each batch is
        cached once it is accepted, so a refused batch leaves the earlier
        ones cached.
        """
        pending = [
            doc
            for doc in candidates
            if doc.doc_id not in self._vectors
            or self._vectors[doc.doc_id][0] != doc.text
        ]
        for first in range(0, len(pending), self.batch_size):
            batch = pending[first : first + self.batch_size]
            vectors = await self._embed([doc.text for doc in batch])
            for doc, vector in zip(batch, vectors, strict=True):
                self._vectors[doc.doc_id] = (doc.text, vector)

    async def _embed(self, texts: list[str]) -> list[np.ndarray | None]:
        """The unit vectors of *texts*, ``None`` for a zero norm.

        Raises :class:`_RefusedBatchError` unless the provider returns one
        vector per text, each a non-empty list of finite numbers of length
        :attr:`dim` (set by the first batch accepted).
        """
        vectors: Any = await self._provider.embed(texts)
        if not isinstance(vectors, list | tuple) or len(vectors) != len(texts):
            raise _RefusedBatchError(f"expected {len(texts)} vectors")
        dim = self.dim
        for vector in vectors:
            if (
                not isinstance(vector, list | tuple)
                or not vector
                or not all(_is_finite(x) for x in vector)
            ):
                raise _RefusedBatchError(
                    "a vector is not a non-empty list of finite numbers"
                )
            if dim is None:
                dim = len(vector)
            elif len(vector) != dim:
                raise _RefusedBatchError(
                    f"a vector has {len(vector)} dimensions, not {dim}"
                )
        self.dim = dim
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        return [
            row / norm if norm > 0 else None
            for row, norm in zip(matrix, norms, strict=True)
        ]

    def _rank(
        self, query_vector: np.ndarray | None, candidates: Sequence[Doc], k: int
    ) -> list[Hit]:
        if query_vector is None:
            return []
        scored: list[tuple[float, int, str]] = []
        for position, doc in enumerate(candidates):
            vector = self._vectors[doc.doc_id][1]
            if vector is None:
                continue
            score = float(np.dot(query_vector, vector))
            if score < self.min_score:
                continue
            scored.append((score, position, doc.doc_id))
        # Best first; ties keep the order of docs.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return ranked((doc_id, score) for score, _position, doc_id in scored[:k])


def _is_finite(value: object) -> bool:
    """Whether *value* is an int or a float that is finite as a float."""
    if not is_number(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False
