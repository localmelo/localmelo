"""Switchable memory retrievers, and their offline evaluation.

Three retrievers rank memory documents (:class:`Doc`) for a query:

- ``bm25``: Okapi BM25 with a floor of matched query units and kind rules;
  no model (:mod:`.bm25`);
- ``llm``: the chat model picks ids from a catalog of the documents, with
  ``bm25`` as its fallback (:mod:`.llm`);
- ``embedding``: cosine similarity of embedding vectors, with ``bm25`` as
  its fallback (:mod:`.embedding`).

:func:`build_retriever` makes one by name, and :func:`evaluate` measures
one on labelled queries: recall@k, MRR, and cold, warm and interleaved
latency. :mod:`.corpus` reads the corpus and the labelled queries, and
turns a run's raw model calls into a corpus. Nothing here touches the
agent loop.

Experimental: the interface may change while memory recall is built on it.
"""

from __future__ import annotations

from typing import Any

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.melo.memory.retrieval.base import (
    DOC_KINDS,
    RETRIEVER_NAMES,
    Doc,
    Hit,
    RetrievalResult,
    Retriever,
    Send,
)
from localmelo.melo.memory.retrieval.bm25 import BM25Retriever
from localmelo.melo.memory.retrieval.corpus import (
    CorpusError,
    RawCall,
    doc_to_dict,
    docs_from_calls,
    load_corpus,
    load_queries,
    load_raw_calls,
)
from localmelo.melo.memory.retrieval.embedding import EmbeddingRetriever
from localmelo.melo.memory.retrieval.evaluate import (
    EvalReport,
    LabelledQuery,
    QueryResult,
    evaluate,
)
from localmelo.melo.memory.retrieval.llm import LLMRetriever, send_from_provider

__all__ = [
    "DOC_KINDS",
    "RETRIEVER_NAMES",
    "CorpusError",
    "Doc",
    "EvalReport",
    "Hit",
    "LabelledQuery",
    "QueryResult",
    "RawCall",
    "RetrievalResult",
    "Retriever",
    "Send",
    "build_retriever",
    "doc_to_dict",
    "docs_from_calls",
    "evaluate",
    "load_corpus",
    "load_queries",
    "load_raw_calls",
    "send_from_provider",
]

# The options each retriever takes.
_OPTIONS: dict[str, tuple[str, ...]] = {
    "bm25": ("floor_terms", "relative_cut"),
    "llm": ("floor_terms", "relative_cut", "timeout_s"),
    "embedding": ("floor_terms", "relative_cut", "min_score", "batch_size"),
}


def build_retriever(
    name: str,
    *,
    send: Send | None = None,
    embedding: BaseEmbeddingProvider | None = None,
    **opts: Any,
) -> Retriever:
    """A new retriever *name*, with empty caches.

    - ``bm25`` takes ``floor_terms`` (default 2, at least 0; 0 turns the
      floor off) and ``relative_cut`` (default 0.2, from 0 up to but not
      including 1);
    - ``llm`` needs *send* (see :func:`send_from_provider`) and takes
      those and ``timeout_s`` (default 30.0, above 0);
    - ``embedding`` needs *embedding* and takes those, ``min_score``
      (default 0.0) and ``batch_size`` (default 32, at least 1).

    ``floor_terms`` and ``relative_cut`` of ``llm`` and ``embedding`` apply
    to their ``bm25`` fallback. Raises :class:`ValueError` for an unknown
    name, ``llm`` without *send*, ``embedding`` without *embedding*, an
    unknown option or an out-of-range value.
    """
    if name not in RETRIEVER_NAMES:
        raise ValueError(
            f"unknown retriever {name!r}; the retrievers are "
            f"{', '.join(RETRIEVER_NAMES)}"
        )
    unknown = sorted(set(opts) - set(_OPTIONS[name]))
    if unknown:
        raise ValueError(
            f"unknown option {', '.join(unknown)} for the {name} retriever; "
            f"it takes {', '.join(_OPTIONS[name])}"
        )
    if name == "llm":
        if send is None:
            raise ValueError("the llm retriever needs send")
        return LLMRetriever(send, **opts)
    if name == "embedding":
        if embedding is None:
            raise ValueError("the embedding retriever needs an embedding provider")
        return EmbeddingRetriever(embedding, **opts)
    return BM25Retriever(**opts)
