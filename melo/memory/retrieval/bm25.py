"""``bm25``: Okapi BM25 over the documents, with a floor and kind rules.

It scores with the BM25 index that ranks tools
(:mod:`localmelo.melo.memory.bm25`), unchanged, and adds:

- **the floor:** a hit must contain at least ``floor_terms`` distinct
  units of the query (:func:`floor_units`): its words and CJK bigrams,
  without the file names and paths in it and without
  :data:`FLOOR_STOPWORDS`. A step's query names the paths of its latest
  calls, which most notes of the task mention, so path matches alone must
  not let a hit through;
- **kinds:** thought chunks keep at most one hit per task step, and every
  other kind ranks above every chunk, so a strong chunk never pushes a
  note out;
- **the relative cut:** within each of those two groups, hits that score
  under ``relative_cut`` times the group's top score are dropped.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from localmelo.melo.memory.bm25 import BM25Index, tokenize
from localmelo.melo.memory.notes import path_spans
from localmelo.melo.memory.retrieval.base import (
    Doc,
    Hit,
    RetrievalResult,
    elapsed_ms,
    int_step,
    is_number,
    ranked,
    search_candidates,
)

DEFAULT_FLOOR_TERMS = 2
DEFAULT_RELATIVE_CUT = 0.2

# Query words that never count toward the floor.
FLOOR_STOPWORDS = frozenset(
    {
        # English function words.
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "let",
        "me",
        "more",
        "my",
        "no",
        "not",
        "now",
        "of",
        "on",
        "one",
        "only",
        "or",
        "our",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "up",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        # The harness's own words: note labels, observed lines and the
        # short forms of calls.
        "plan",
        "found",
        "decided",
        "hypotheses",
        "open",
        "observed",
        "outcome",
        "read",
        "wrote",
        "write",
        "ran",
        "run",
        "chars",
        "sha",
        "records",
        "ok",
        "error",
        "exit",
        "python",
        "step",
        "note",
        "draft",
        # Chinese function words.
        "一个",
        "这个",
        "那个",
        "我们",
        "你们",
        "他们",
        "然后",
        "现在",
        "就是",
        "可以",
        "没有",
        "什么",
        "怎么",
        "因为",
        "所以",
        "如果",
        "但是",
        "还是",
    }
)

# A word as the BM25 tokenizer reads it.
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def floor_units(query: str) -> frozenset[str]:
    """The units of *query* that the floor counts.

    The file names and paths of *query*
    (:func:`~localmelo.melo.memory.notes.path_spans`) are blanked out
    first. The units are then:

    - each word (``[A-Za-z0-9_]+``), lowercased, with leading and trailing
      ``_`` stripped, of 2 or more characters: the whole-word term the
      tokenizer gives first, so an identifier counts once, not once per
      part;
    - each CJK bigram the tokenizer gives; single CJK characters are not
      units;

    minus :data:`FLOOR_STOPWORDS`.
    """
    blanked = _blank_paths(query)
    units = {
        word
        for word in (
            match.group().lower().strip("_") for match in _WORD_RE.finditer(blanked)
        )
        if len(word) >= 2
    }
    # The tokenizer's terms are ASCII words, single CJK characters and CJK
    # bigrams, so a two-character term that is not ASCII is a bigram.
    units.update(
        term for term in tokenize(blanked) if len(term) == 2 and not term.isascii()
    )
    return frozenset(units - FLOOR_STOPWORDS)


def _blank_paths(text: str) -> str:
    """*text* with each file name and path replaced by a space."""
    parts: list[str] = []
    pos = 0
    for start, end in path_spans(text):
        parts.append(text[pos:start])
        parts.append(" ")
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def _check_floor_options(floor_terms: object, relative_cut: object) -> None:
    """Raise :class:`ValueError` unless both options are in range."""
    if (
        not isinstance(floor_terms, int)
        or isinstance(floor_terms, bool)
        or floor_terms < 0
    ):
        raise ValueError(f"floor_terms must be an integer >= 0, got {floor_terms!r}")
    if not is_number(relative_cut) or not 0 <= relative_cut < 1:
        raise ValueError(
            f"relative_cut must be a number from 0 up to but not including 1, "
            f"got {relative_cut!r}"
        )


class BM25Retriever:
    """Okapi BM25 with a floor of matched query units, and kind rules.

    One :class:`~localmelo.melo.memory.bm25.BM25Index` per instance; a doc
    is added on first sight, and again when its text changed. A search
    ranks only its candidates (``docs`` without ``exclude``), so IDF and the
    average length are computed over them alone. ``Hit.score`` is the BM25
    score.
    """

    name: str

    def __init__(
        self,
        *,
        floor_terms: int = DEFAULT_FLOOR_TERMS,
        relative_cut: float = DEFAULT_RELATIVE_CUT,
    ) -> None:
        _check_floor_options(floor_terms, relative_cut)
        self.name = "bm25"
        self.floor_terms = floor_terms
        self.relative_cut = relative_cut
        self._index = BM25Index()
        self._texts: dict[str, str] = {}  # doc_id -> the text indexed for it
        self._terms: dict[str, frozenset[str]] = {}  # doc_id -> its term set

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
        hits = self._rank(query, candidates, k) if candidates else []
        return RetrievalResult(hits=hits, retriever=self.name, ms=elapsed_ms(start))

    def _rank(self, query: str, candidates: Sequence[Doc], k: int) -> list[Hit]:
        units = floor_units(query) if self.floor_terms else frozenset()
        if len(units) < self.floor_terms:
            return []

        position: dict[str, int] = {}
        by_id: dict[str, Doc] = {}
        for i, doc in enumerate(candidates):
            position[doc.doc_id] = i
            by_id[doc.doc_id] = doc
            if self._texts.get(doc.doc_id) != doc.text:
                self._index.add(doc.doc_id, doc.text)
                self._texts[doc.doc_id] = doc.text
                self._terms[doc.doc_id] = frozenset(tokenize(doc.text))
        # Docs indexed by an earlier search but not candidates now take no
        # part, so they count in neither IDF nor the average length.
        absent = {doc_id for doc_id in self._texts if doc_id not in position}
        scored = self._index.search(query, len(self._texts), exclude=absent)
        # The index breaks ties by the order docs were first added; break
        # them by the order of this search's docs instead.
        scored.sort(key=lambda item: (-item[1], position[item[0]]))

        others: list[tuple[str, float]] = []
        chunks: list[tuple[str, float]] = []
        for doc_id, score in scored:
            if len(units & self._terms[doc_id]) < self.floor_terms:
                continue
            group = chunks if by_id[doc_id].kind == "thought" else others
            group.append((doc_id, score))
        chunks = _best_chunk_per_step(chunks, by_id)
        kept = self._cut(others) + self._cut(chunks)
        return ranked(kept[:k])

    def _cut(self, group: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """*group* (best first) without the hits under the relative cut."""
        if not group:
            return group
        floor = self.relative_cut * group[0][1]
        return [(doc_id, score) for doc_id, score in group if score >= floor]


def _best_chunk_per_step(
    chunks: list[tuple[str, float]], by_id: Mapping[str, Doc]
) -> list[tuple[str, float]]:
    """*chunks* (best first) with one hit per ``(task_id, step)``, the best.

    A chunk without an integer step is never merged with another.
    """
    seen: set[tuple[Any, int]] = set()
    kept: list[tuple[str, float]] = []
    for doc_id, score in chunks:
        meta = by_id[doc_id].meta
        step = int_step(meta)
        if step is not None:
            key = (meta.get("task_id"), step)
            if key in seen:
                continue
            seen.add(key)
        kept.append((doc_id, score))
    return kept
