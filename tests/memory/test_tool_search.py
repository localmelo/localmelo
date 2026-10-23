"""Tool search: the BM25 tool index and Hippo's tool resolution."""

from __future__ import annotations

from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.tools import ToolIndex, ToolRegistry
from localmelo.melo.schema import TOOL_SEARCH_TOP_K, ToolDef


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


def _hippo() -> Hippo:
    hippo = Hippo(embedding=None)
    for tool in (
        _tool("file_read", "Read file contents"),
        _tool("weather", "Weather forecast for a city"),
        _tool("calendar", "Add calendar events"),
        _tool("notes", "Search notes"),
    ):
        hippo.register_tool(tool)
    return hippo


class TestResolveTools:
    def test_hinted_tools_first_in_hint_order_then_the_search(self) -> None:
        tools = _hippo().resolve_tools(
            "what is the weather in Paris", hints=["notes", "calendar"]
        )
        assert _names(tools) == ["notes", "calendar", "weather"]

    def test_a_tool_both_hinted_and_found_appears_once(self) -> None:
        tools = _hippo().resolve_tools("weather in Paris", hints=["weather", "weather"])
        assert _names(tools) == ["weather"]

    def test_a_hint_that_names_no_registered_tool_is_ignored(self) -> None:
        tools = _hippo().resolve_tools("weather in Paris", hints=["no_such_tool"])
        assert _names(tools) == ["weather"]

    def test_excluded_tools_are_neither_hinted_nor_searched(self) -> None:
        tools = _hippo().resolve_tools(
            "read the weather file",
            hints=["file_read", "notes"],
            exclude={"file_read"},
        )
        assert _names(tools) == ["notes", "weather"]

    def test_the_search_adds_at_most_top_k_tools(self) -> None:
        hippo = Hippo(embedding=None)
        for i in range(TOOL_SEARCH_TOP_K + 2):
            hippo.register_tool(_tool(f"tool_{i}", "alpha"))
        assert len(hippo.resolve_tools("alpha")) == TOOL_SEARCH_TOP_K
        # Hinted tools come on top of the search's matches.
        tools = hippo.resolve_tools("alpha", hints=["tool_4"])
        assert _names(tools) == ["tool_4", "tool_0", "tool_1", "tool_2"]
