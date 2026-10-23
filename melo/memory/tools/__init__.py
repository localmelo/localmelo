from __future__ import annotations

from collections.abc import Collection

from localmelo.melo.memory.bm25 import BM25Index
from localmelo.melo.schema import ToolDef


def _document(tool: ToolDef) -> str:
    """The text a tool is found by: its name, description and semantic tags.

    The tokenizer splits the name into words, so ``file_write`` also gives
    ``file`` and ``write``.
    """
    return f"{tool.name} {tool.description} {' '.join(tool.semantic_tags)}"


class ToolIndex:
    """Searchable semantic index over tool definitions (Okapi BM25).

    This is the search-only component.  It does *not* serve as the
    authoritative source of executable tool definitions — that role
    belongs to :class:`ToolRegistry`.
    """

    def __init__(self) -> None:
        self._docs: dict[str, ToolDef] = {}
        self._bm25 = BM25Index()

    def index(self, tool: ToolDef) -> None:
        self._docs[tool.name] = tool
        self._bm25.add(tool.name, _document(tool))

    def remove(self, name: str) -> None:
        self._docs.pop(name, None)
        self._bm25.remove(name)

    def search(
        self, query: str, top_k: int = 3, *, exclude: Collection[str] = ()
    ) -> list[ToolDef]:
        """Return up to *top_k* tools that match *query*, best first.

        A tool that matches no term of the query is never returned. The
        tools named in *exclude* take no part: the others are ranked among
        themselves (see :class:`~localmelo.melo.memory.bm25.BM25Index`).
        """
        return [
            self._docs[name]
            for name, _score in self._bm25.search(query, top_k, exclude=exclude)
        ]


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

    def search(
        self, query: str, top_k: int = 3, *, exclude: Collection[str] = ()
    ) -> list[ToolDef]:
        return self._index.search(query, top_k=top_k, exclude=exclude)

    @property
    def index(self) -> ToolIndex:
        """Direct access to the underlying semantic index."""
        return self._index
