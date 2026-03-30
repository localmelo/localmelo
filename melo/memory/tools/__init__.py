from __future__ import annotations

import re
from collections import Counter

from localmelo.melo.schema import ToolDef


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    k1, b, avg_dl = 1.5, 0.75, 20.0
    score = 0.0
    for qt in query_tokens:
        f = tf.get(qt, 0)
        if f == 0:
            continue
        num = f * (k1 + 1)
        den = f + k1 * (1 - b + b * doc_len / avg_dl)
        score += num / den
    return score


class ToolIndex:
    """Searchable semantic index over tool definitions (BM25).

    This is the search-only component.  It does *not* serve as the
    authoritative source of executable tool definitions — that role
    belongs to :class:`ToolRegistry`.
    """

    def __init__(self) -> None:
        self._docs: dict[str, ToolDef] = {}

    def index(self, tool: ToolDef) -> None:
        self._docs[tool.name] = tool

    def remove(self, name: str) -> None:
        self._docs.pop(name, None)

    def search(self, query: str, top_k: int = 3) -> list[ToolDef]:
        if not self._docs:
            return []
        q_tokens = _tokenize(query)
        scored: list[tuple[str, float]] = []
        for name, td in self._docs.items():
            doc = f"{td.name} {td.description} {' '.join(td.semantic_tags)}"
            score = _bm25_score(q_tokens, _tokenize(doc))
            if score > 0:
                scored.append((name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._docs[name] for name, _ in scored[:top_k]]


class ToolRegistry:
    """Authoritative executable tool registry with integrated search index.

    Maintains both an exact-name lookup (the registry) and a BM25 semantic
    index for query-based discovery.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._index = ToolIndex()

    # ── Registry (authoritative lookup) ──

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool
        self._index.index(tool)

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDef]:
        return list(self._tools.values())

    # ── Index (semantic search, delegates to ToolIndex) ──

    def search(self, query: str, top_k: int = 3) -> list[ToolDef]:
        return self._index.search(query, top_k=top_k)

    @property
    def index(self) -> ToolIndex:
        """Direct access to the underlying semantic index."""
        return self._index
