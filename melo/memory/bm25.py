"""Okapi BM25 text search: a tokenizer and an index of documents by id.

The tool index (:mod:`localmelo.melo.memory.tools`) ranks tools with it.
It has no dependency on tools, so other text retrieval can reuse it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection

# Okapi BM25 parameters: term-frequency saturation and length normalization.
K1 = 1.2
B = 0.75

# Characters that form CJK runs. Each run gives its single characters and
# its adjacent-character bigrams, since these scripts do not put spaces
# between words.
_CJK_RANGES = (
    # CJK Unified Ideographs: extension A, the main block, and the
    # extensions in the supplementary ideographic planes.
    "㐀-䶿一-鿿\U00020000-\U0003ffff"
    # Hiragana.
    "ぁ-ゟ"
    # Katakana, without the block's two punctuation marks (U+30A0, U+30FB).
    "ァ-ヺー-ヿ"
    # Hangul: jamo, compatibility jamo and syllables.
    "ᄀ-ᇿㄱ-ㆎ가-힣"
)
_TERM_RE = re.compile(f"(?P<word>[A-Za-z0-9_]+)|(?P<cjk>[{_CJK_RANGES}]+)")


def tokenize(text: str) -> list[str]:
    """Split *text* into BM25 terms, in order.

    - A run of ASCII letters, digits and underscores is a word, lowercased.
      A word made of several parts joined by underscores also gives its
      parts: ``file_write`` gives ``file_write``, ``file`` and ``write``.
    - A run of CJK characters (CJK Unified Ideographs, Hiragana, Katakana,
      Hangul) gives its single characters, then its adjacent-character
      bigrams: ``读取文件`` gives ``读 取 文 件 读取 取文 文件``.

    Every other character separates terms.
    """
    terms: list[str] = []
    for match in _TERM_RE.finditer(text):
        word = match.group("word")
        if word is not None:
            terms.extend(_word_terms(word.lower()))
            continue
        run = match.group("cjk")
        terms.extend(run)
        terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def _word_terms(word: str) -> list[str]:
    """The terms of one lowercase word: itself, then its parts, if several."""
    parts = [part for part in word.split("_") if part]
    if len(parts) > 1:
        return [word.strip("_"), *parts]
    return parts


def idf(n_docs: int, doc_freq: int) -> float:
    """Okapi IDF of a term that *doc_freq* of *n_docs* documents contain.

    ``ln(1 + (N - n + 0.5) / (n + 0.5))``: always above 0, and the lower
    the more of the documents contain the term.
    """
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


class BM25Index:
    """Okapi BM25 search over text documents, each added under an id.

    A search ranks the documents it covers among themselves: their number,
    the number of them that contain each query term (for :func:`idf`) and
    their average length in terms are counted at search time, over those
    documents only. Each occurrence of a term in the query adds to a
    score.
    """

    def __init__(self, *, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self._frequencies: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._frequencies)

    def add(self, doc_id: str, text: str) -> None:
        """Index *text* as document *doc_id*, in place of what that id held."""
        terms = tokenize(text)
        self._frequencies[doc_id] = Counter(terms)
        self._lengths[doc_id] = len(terms)

    def remove(self, doc_id: str) -> None:
        """Remove document *doc_id*; an id that is not indexed is ignored."""
        self._frequencies.pop(doc_id, None)
        self._lengths.pop(doc_id, None)

    def search(
        self, query: str, top_k: int, *, exclude: Collection[str] = ()
    ) -> list[tuple[str, float]]:
        """Return up to *top_k* ``(doc_id, score)`` pairs for *query*, best first.

        A document with a score of 0, which contains no query term, is
        never returned. Equal scores keep the order in which the documents
        were first added. The documents named in *exclude* take no part in
        the search: the others are ranked among themselves.
        """
        if isinstance(exclude, str):
            raise TypeError("exclude must be a collection of ids, not a str")
        doc_ids = [doc_id for doc_id in self._frequencies if doc_id not in exclude]
        query_terms = tokenize(query)
        if top_k <= 0 or not doc_ids or not query_terms:
            return []

        n_docs = len(doc_ids)
        avg_length = sum(self._lengths[doc_id] for doc_id in doc_ids) / n_docs
        weights = {
            term: idf(
                n_docs,
                sum(1 for doc_id in doc_ids if term in self._frequencies[doc_id]),
            )
            for term in set(query_terms)
        }
        scored: list[tuple[str, float]] = []
        for doc_id in doc_ids:
            score = self._score(query_terms, doc_id, weights, avg_length)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _score(
        self,
        query_terms: list[str],
        doc_id: str,
        weights: dict[str, float],
        avg_length: float,
    ) -> float:
        length = self._lengths[doc_id]
        if not length:
            return 0.0
        frequencies = self._frequencies[doc_id]
        norm = self.k1 * (1 - self.b + self.b * length / avg_length)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency:
                score += weights[term] * frequency * (self.k1 + 1) / (frequency + norm)
        return score
