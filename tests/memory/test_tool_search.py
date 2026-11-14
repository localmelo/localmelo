"""Tool search: the BM25 tool index."""

from __future__ import annotations

from localmelo.melo.memory.tools import ToolIndex, ToolRegistry
from localmelo.melo.schema import ToolDef


def _tool(name: str, description: str = "", tags: list[str] | None = None) -> ToolDef:
    return ToolDef(
        name=name, description=description, parameters={}, semantic_tags=tags or []
    )


def _names(tools: list[ToolDef]) -> list[str]:
    return [t.name for t in tools]


class TestToolIndexSearch:
    def test_a_tools_name_is_split_into_words(self) -> None:
        index = ToolIndex()
        index.index(_tool("file_write", "store text"))
        index.index(_tool("other", "unrelated"))
        for query in ("write", "file", "file_write", "save it to a file"):
            assert _names(index.search(query)) == ["file_write"], query

    def test_semantic_tags_are_searched(self) -> None:
        index = ToolIndex()
        index.index(_tool("shell_exec", "Execute a command", ["bash", "terminal"]))
        index.index(_tool("other", "unrelated"))
        assert _names(index.search("open a terminal")) == ["shell_exec"]

    def test_the_relevant_tool_ranks_first_among_ten(self) -> None:
        index = ToolIndex()
        for i in range(9):
            index.index(
                _tool(f"helper_{i}", "Do the work and return the result of the task")
            )
        index.index(_tool("transpose_text", "Transpose the rows and columns of a text"))
        query = (
            "Implement the transpose function in transpose.py so that the tests "
            "in transpose_test.py pass"
        )
        ranked = _names(index.search(query))
        assert ranked[0] == "transpose_text"
        assert len(ranked) == 3

    def test_a_chinese_query_finds_a_tool_with_a_chinese_description(self) -> None:
        index = ToolIndex()
        index.index(_tool("weather", "查询城市的天气预报"))
        index.index(_tool("translate", "把文本翻译成英文"))
        index.index(_tool("web_search", "Search the web"))
        assert _names(index.search("北京明天天气怎么样？")) == ["weather"]
        assert _names(index.search("翻译这段文字")) == ["translate"]

    def test_excluded_tools_are_left_out(self) -> None:
        index = ToolIndex()
        index.index(_tool("file_read", "Read the contents of a file"))
        index.index(_tool("note_read", "Read a note"))
        assert _names(index.search("read a file")) == ["file_read", "note_read"]
        assert _names(index.search("read a file", exclude={"file_read"})) == [
            "note_read"
        ]

    def test_the_registry_passes_exclude_on(self) -> None:
        registry = ToolRegistry()
        registry.register(_tool("file_read", "Read the contents of a file"))
        registry.register(_tool("note_read", "Read a note"))
        assert _names(registry.search("read", exclude={"note_read"})) == ["file_read"]
