"""Offline evaluation of a retriever on labelled queries: accuracy and latency.

For each query, :func:`evaluate` makes one **cold** call, ``repeats``
**warm** calls back to back, and, with ``interleave``, ``repeats``
**interleaved** calls, each after an untimed ``await interleave()`` (for
example a planning-shaped request, as the agent loop sends between
retrievals). Recall@k and the reciprocal rank are means over a query's
calls, then over the queries (macro); latencies are reported as
nearest-rank percentiles.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from localmelo.melo.memory.retrieval.base import (
    FALLBACK_REASONS,
    Doc,
    Hit,
    RetrievalResult,
    Retriever,
    int_step,
)


@dataclass(frozen=True)
class LabelledQuery:
    """A query, the doc ids that answer it, and the docs it may see."""

    qid: str
    query: str
    relevant: tuple[str, ...]  # full doc ids
    grades: Mapping[str, int] = field(default_factory=dict)  # echoed, not scored
    task_id: str = ""  # "": every task's docs
    before_step: int | None = None  # only docs of earlier steps
    exclude_ids: frozenset[str] = frozenset()
    source: str = ""

    def candidates(self, corpus: Sequence[Doc]) -> list[Doc]:
        """The docs of *corpus* this query searches, in corpus order.

        Those whose ``meta["task_id"]`` is :attr:`task_id` (every doc when
        it is empty) and, with :attr:`before_step`, whose ``meta["step"]``
        is less than it, so the query sees only documents that existed
        before that step. A doc without an integer step counts as step 0.
        Excluded docs are among them: a retriever gets them with
        ``exclude``.
        """
        docs: list[Doc] = []
        for doc in corpus:
            if self.task_id and doc.meta.get("task_id") != self.task_id:
                continue
            if self.before_step is not None and (int_step(doc.meta) or 0) >= (
                self.before_step
            ):
                continue
            docs.append(doc)
        return docs


@dataclass
class QueryResult:
    """The evaluation of one query, as its output line prints it."""

    qid: str
    source: str
    candidates: int  # the query's candidates, excluded ones not counted
    relevant: tuple[str, ...]  # full doc ids
    hits: list[Hit]  # the cold call's
    recall: dict[int, float]  # k -> recall@k, the mean over the query's calls
    rr: float  # the reciprocal rank, the mean over the query's calls
    ms_cold: int
    ms_warm: list[int]
    ms_interleaved: list[int]
    fallbacks: dict[str, int]  # reason -> calls, for each of FALLBACK_REASONS
    usage: dict[str, int]  # the calls' integer usage, summed by key

    def to_line(self, ks: Sequence[int]) -> dict[str, Any]:
        """The query's output line, with ``recall@<k>`` for each of *ks*.

        ``ms_warm`` and ``ms_interleaved`` are ``None`` when no such call
        ran (``repeats`` 0, or no ``interleave``), as the summary's
        percentiles are.
        """
        line: dict[str, Any] = {
            "type": "query",
            "qid": self.qid,
            "source": self.source,
            "candidates": self.candidates,
            "relevant": list(self.relevant),
            "hits": [
                {"doc_id": hit.doc_id, "score": round(hit.score, 4), "rank": hit.rank}
                for hit in self.hits
            ],
        }
        for k in ks:
            line[f"recall@{k}"] = self.recall[k]
        line["rr"] = self.rr
        line["ms_cold"] = self.ms_cold
        line["ms_warm"] = list(self.ms_warm) or None
        line["ms_interleaved"] = list(self.ms_interleaved) or None
        line["fallbacks"] = dict(self.fallbacks)
        line["usage"] = dict(self.usage)
        return line


@dataclass
class EvalReport:
    """The evaluation of a retriever over every query."""

    retriever: str  # the retriever's name
    model: str | None
    dim: int | None  # the last retriever's dim, if it has one
    ks: tuple[int, ...]
    n: int  # queries
    calls: int
    recall: dict[int, float]  # k -> the mean of the queries' recall@k
    mrr: float
    ms_cold_p50: int | None
    ms_warm_p50: int | None
    ms_warm_p95: int | None
    ms_interleaved_p50: int | None
    ms_interleaved_p95: int | None
    prompt_tokens_mean: float | None
    cached_tokens_mean: float | None
    completion_tokens_mean: float | None
    fallbacks: dict[str, int]  # reason -> calls, over every call
    queries: list[QueryResult] = field(default_factory=list)

    def summary_line(self) -> dict[str, Any]:
        """The summary output line, with ``recall@<k>`` for each of :attr:`ks`."""
        line: dict[str, Any] = {
            "type": "summary",
            "retriever": self.retriever,
            "model": self.model,
            "dim": self.dim,
            "n": self.n,
            "calls": self.calls,
        }
        for k in self.ks:
            line[f"recall@{k}"] = self.recall[k]
        line["mrr"] = self.mrr
        line["ms_cold_p50"] = self.ms_cold_p50
        line["ms_warm_p50"] = self.ms_warm_p50
        line["ms_warm_p95"] = self.ms_warm_p95
        line["ms_interleaved_p50"] = self.ms_interleaved_p50
        line["ms_interleaved_p95"] = self.ms_interleaved_p95
        line["prompt_tokens_mean"] = self.prompt_tokens_mean
        line["cached_tokens_mean"] = self.cached_tokens_mean
        line["completion_tokens_mean"] = self.completion_tokens_mean
        line["fallbacks"] = dict(self.fallbacks)
        return line


async def evaluate(
    retriever: Retriever | Callable[[], Retriever],
    *,
    corpus: Sequence[Doc],
    queries: Sequence[LabelledQuery],
    ks: Sequence[int] = (1, 3, 5),
    repeats: int = 3,
    interleave: Callable[[], Awaitable[object]] | None = None,
    model: str | None = None,
    on_query: Callable[[QueryResult], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EvalReport:
    """Evaluate *retriever* on *queries* over *corpus*.

    *retriever* is an instance, which every query shares, or a factory
    (anything that is not an instance of
    :class:`~localmelo.melo.memory.retrieval.base.Retriever`, a class
    included), which makes a new retriever with empty caches for each
    query's cold call. Each call is ``search(query, candidates,
    k=max(ks), exclude=exclude_ids)``, timed by *clock* around the
    ``await``, in whole milliseconds; the time of ``interleave`` is not
    counted. *on_query* gets each query's result once the query is done.

    Raises :class:`ValueError`, before any search, for empty *ks* or
    *queries*, a k under 1, *repeats* under 0, a query without relevant
    ids, or a relevant id that is not a candidate of its query. An
    exception from *interleave* propagates.
    """
    ks = _checked_ks(ks)
    if not queries:
        raise ValueError("there are no queries to evaluate")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 0:
        raise ValueError(f"repeats must be an integer >= 0, got {repeats!r}")
    plans = [(query, _checked_candidates(query, corpus)) for query in queries]

    # A class has the protocol's attributes too, but is a factory.
    shared: Retriever | None = None
    if isinstance(retriever, Retriever) and not isinstance(retriever, type):
        shared = retriever
    factory = cast(Callable[[], Retriever], retriever)

    k_max = max(ks)
    results: list[QueryResult] = []
    exact_recall: list[dict[int, float]] = []
    exact_rr: list[float] = []
    cold_ms: list[int] = []
    warm_ms: list[int] = []
    interleaved_ms: list[int] = []
    usages: list[dict[str, int]] = []
    fallbacks = dict.fromkeys(FALLBACK_REASONS, 0)
    last: Retriever | None = None

    for query, candidates in plans:
        current = shared if shared is not None else factory()
        last = current
        cold, ms_cold = await _timed_search(current, query, candidates, k_max, clock)
        calls = [cold]
        ms_warm: list[int] = []
        for _ in range(repeats):
            result, ms = await _timed_search(current, query, candidates, k_max, clock)
            calls.append(result)
            ms_warm.append(ms)
        ms_interleaved: list[int] = []
        if interleave is not None:
            for _ in range(repeats):
                await interleave()
                result, ms = await _timed_search(
                    current, query, candidates, k_max, clock
                )
                calls.append(result)
                ms_interleaved.append(ms)

        relevant = frozenset(query.relevant)
        recall = {
            k: _mean([_recall_at(call.hits, relevant, k) for call in calls]) for k in ks
        }
        rr = _mean([_reciprocal_rank(call.hits, relevant) for call in calls])
        query_fallbacks = dict.fromkeys(FALLBACK_REASONS, 0)
        usage: dict[str, int] = {}
        for call in calls:
            if call.fallback in query_fallbacks:
                query_fallbacks[call.fallback] += 1
                fallbacks[call.fallback] += 1
            for key, value in call.usage.items():
                usage[key] = usage.get(key, 0) + value
            usages.append(call.usage)

        query_result = QueryResult(
            qid=query.qid,
            source=query.source,
            candidates=sum(
                1 for doc in candidates if doc.doc_id not in query.exclude_ids
            ),
            relevant=tuple(query.relevant),
            hits=list(cold.hits),
            recall={k: round(value, 4) for k, value in recall.items()},
            rr=round(rr, 4),
            ms_cold=ms_cold,
            ms_warm=ms_warm,
            ms_interleaved=ms_interleaved,
            fallbacks=query_fallbacks,
            usage=usage,
        )
        results.append(query_result)
        exact_recall.append(recall)
        exact_rr.append(rr)
        cold_ms.append(ms_cold)
        warm_ms.extend(ms_warm)
        interleaved_ms.extend(ms_interleaved)
        if on_query is not None:
            on_query(query_result)

    assert last is not None  # there is at least one query
    dim = getattr(last, "dim", None)
    return EvalReport(
        retriever=last.name,
        model=model,
        dim=dim if isinstance(dim, int) and not isinstance(dim, bool) else None,
        ks=ks,
        n=len(results),
        calls=len(usages),
        recall={k: round(_mean([r[k] for r in exact_recall]), 4) for k in ks},
        mrr=round(_mean(exact_rr), 4),
        ms_cold_p50=nearest_rank(cold_ms, 50),
        ms_warm_p50=nearest_rank(warm_ms, 50),
        ms_warm_p95=nearest_rank(warm_ms, 95),
        ms_interleaved_p50=nearest_rank(interleaved_ms, 50),
        ms_interleaved_p95=nearest_rank(interleaved_ms, 95),
        prompt_tokens_mean=_token_mean(usages, "prompt_tokens"),
        cached_tokens_mean=_token_mean(usages, "cached_tokens"),
        completion_tokens_mean=_token_mean(usages, "completion_tokens"),
        fallbacks=fallbacks,
        queries=results,
    )


def nearest_rank(values: Sequence[int], percent: int) -> int | None:
    """The nearest-rank percentile: the ceil(percent / 100 × n)-th smallest.

    ``None`` when there are no values.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = -(-percent * len(ordered) // 100)  # ceil, in integers
    return ordered[max(rank, 1) - 1]


def _checked_ks(ks: Sequence[int]) -> tuple[int, ...]:
    """*ks* without repeats, in order; :class:`ValueError` if any is wrong."""
    unique = tuple(dict.fromkeys(ks))
    if not unique:
        raise ValueError("ks is empty")
    for k in unique:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"every k must be an integer >= 1, got {k!r}")
    return unique


def _checked_candidates(query: LabelledQuery, corpus: Sequence[Doc]) -> list[Doc]:
    """The candidates of *query*, once its relevant ids are checked."""
    if not query.relevant:
        raise ValueError(f"query {query.qid!r} has no relevant ids")
    candidates = query.candidates(corpus)
    ids = {doc.doc_id for doc in candidates}
    for doc_id in query.relevant:
        if doc_id not in ids:
            raise ValueError(
                f"query {query.qid!r}: relevant id {doc_id!r} is not a candidate "
                "of the query"
            )
    return candidates


async def _timed_search(
    retriever: Retriever,
    query: LabelledQuery,
    candidates: Sequence[Doc],
    k: int,
    clock: Callable[[], float],
) -> tuple[RetrievalResult, int]:
    """One search for *query*, and its time in whole milliseconds."""
    start = clock()
    result = await retriever.search(
        query.query, candidates, k=k, exclude=query.exclude_ids
    )
    return result, round((clock() - start) * 1000)


def _recall_at(hits: Sequence[Hit], relevant: frozenset[str], k: int) -> float:
    """The share of *relevant* among the first *k* hits."""
    return len(relevant.intersection(hit.doc_id for hit in hits[:k])) / len(relevant)


def _reciprocal_rank(hits: Sequence[Hit], relevant: frozenset[str]) -> float:
    """1 / the rank of the first relevant hit; 0.0 without one."""
    for rank, hit in enumerate(hits, start=1):
        if hit.doc_id in relevant:
            return 1 / rank
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _token_mean(usages: Sequence[Mapping[str, int]], key: str) -> float | None:
    """The mean of *key* over the calls whose usage has it, to 1 decimal place."""
    values = [usage[key] for usage in usages if key in usage]
    if not values:
        return None
    return round(sum(values) / len(values), 1)
