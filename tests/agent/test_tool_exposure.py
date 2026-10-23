"""Agent: tool exposure modes and the execution allowlist — no real network or model."""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker.payloads import ToolResolutionResult, ValidationResult
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import builtin_names
from localmelo.melo.executor.executor import ToolCallable
from localmelo.melo.schema import TOOL_SEARCH_TOP_K, Message, ToolCall, ToolDef

# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Pops queued responses and records every request."""

    def __init__(self, responses: list[Message] | None = None) -> None:
        self.responses: deque[Message] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return Message(role="assistant", content="(no response queued)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


class FakeEmbedding(BaseEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


async def _echo(text: str = "") -> str:
    return text


class FakeToolProvider(BaseToolProvider):
    """Provides one tool per ``name -> description``; each echoes its text."""

    def __init__(self, tools: dict[str, str]) -> None:
        self.tools = tools

    async def start(self) -> list[ProvidedTool]:
        return [
            ProvidedTool(
                definition=ToolDef(
                    name=name,
                    description=description,
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                ),
                function=_echo,
            )
            for name, description in self.tools.items()
        ]

    async def close(self) -> None:
        pass


# ── Helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _recorder(name: str, ran: list[tuple[str, str]]) -> ToolCallable:
    """A tool function that records its call and echoes its text."""

    async def run(text: str = "") -> str:
        ran.append((name, text))
        return text

    return run


def _agent(
    llm: FakeLLM,
    options: AgentOptions,
    tools: dict[str, str],
    embedding: BaseEmbeddingProvider | None = None,
) -> tuple[Agent, list[tuple[str, str]]]:
    """An agent with *tools* (name -> description) registered in that order.

    Returns the agent and the list each tool appends ``(name, text)`` to
    when it runs.
    """
    agent = Agent(llm=llm, embedding=embedding, options=options)
    ran: list[tuple[str, str]] = []
    for name, description in tools.items():
        agent.hippo.register_tool(
            ToolDef(
                name=name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )
        )
        agent.executor.register(name, _recorder(name, ran))
    return agent, ran


def _call(name: str, text: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_call=ToolCall(tool_name=name, arguments={"text": text}),
    )


def _batch(*calls: tuple[str, str]) -> Message:
    """One assistant message with a call per ``(name, text)``, in order."""
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name=n, arguments={"text": t}) for n, t in calls],
    )


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _offered(call: dict[str, Any]) -> list[str]:
    """Names of the tools one request offered, in order."""
    return [t.name for t in call["tools"] or []]


def _results(call: dict[str, Any]) -> list[str]:
    """Contents of the tool-result messages one request carried, in order."""
    return [m.content for m in call["messages"] if m.role == "tool"]


def _refused(name: str) -> str:
    """The tool-result text for a call to a tool its step did not offer."""
    return f"Error: Tool '{name}' was not offered in this step."


def _wire(call: dict[str, Any]) -> str:
    """The tools of one request, serialized byte for byte."""
    return json.dumps([dataclasses.asdict(t) for t in call["tools"]])


# Descriptions that share no word with the queries used below, so BM25
# matches none of these tools.
UNMATCHED = {
    "zeta": "last letter",
    "alpha": "first letter",
    "mid": "in between",
}


# ── tool_exposure="all" ─────────────────────────────────────────────────────


class TestAllExposure:
    @pytest.mark.asyncio
    async def test_every_tool_sorted_and_identical_on_every_step(self) -> None:
        llm = FakeLLM([_call("zeta", "1"), _call("alpha", "2"), _answer("done")])
        options = AgentOptions(tool_exposure="all", builtins=frozenset())
        agent, ran = _agent(llm, options, UNMATCHED)

        assert await agent.run("hello") == "done"
        assert len(llm.calls) == 3
        # Sorted by name, not in registration order, and only the test's own
        # tools: no built-in is registered.
        assert [_offered(call) for call in llm.calls] == [["alpha", "mid", "zeta"]] * 3
        # Byte-stable across steps, although each step added a new turn.
        assert len({_wire(call) for call in llm.calls}) == 1
        assert ran == [("zeta", "1"), ("alpha", "2")]

    @pytest.mark.asyncio
    async def test_builtins_are_offered_with_the_registered_tools(self) -> None:
        llm = FakeLLM([_answer("done")])
        agent, _ = _agent(llm, AgentOptions(tool_exposure="all"), {"echo": "echo"})

        assert await agent.run("hello") == "done"
        assert _offered(llm.calls[0]) == sorted(builtin_names() | {"echo"})

    @pytest.mark.asyncio
    async def test_skips_hints_and_bm25_but_checks_the_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = FakeLLM([_answer("done")])
        options = AgentOptions(tool_exposure="all", builtins=frozenset())
        agent, _ = _agent(llm, options, {"zeta": "z", "alpha": "a"})

        def must_not_run(*args: object, **kwargs: object) -> list[str]:
            raise AssertionError("hint extraction and BM25 run only in retrieve mode")

        monkeypatch.setattr(agent.hippo, "extract_tool_hints", must_not_run)
        monkeypatch.setattr(agent.hippo, "resolve_tools", must_not_run)

        checked: list[ToolResolutionResult] = []
        check = agent.checker.check_tool_resolution

        def spy(result: ToolResolutionResult) -> ValidationResult:
            checked.append(result)
            return check(result)

        monkeypatch.setattr(agent.checker, "check_tool_resolution", spy)

        assert await agent.run("hello") == "done"
        assert [(r.query, r.hints, r.resolved_tool_names) for r in checked] == [
            ("hello", [], ["alpha", "zeta"])
        ]

    @pytest.mark.asyncio
    async def test_long_term_recall_still_reaches_the_prompt(self) -> None:
        llm = FakeLLM([_answer("done")])
        options = AgentOptions(tool_exposure="all")
        agent, _ = _agent(llm, options, {}, embedding=FakeEmbedding())
        await agent.hippo.long.add("the user prefers metric units", [1.0, 0, 0, 0])

        assert await agent.run("hello") == "done"
        system = llm.calls[0]["messages"][0]
        assert system.role == "system"
        assert "the user prefers metric units" in system.content

    @pytest.mark.asyncio
    async def test_pre_plan_check_still_runs(self) -> None:
        llm = FakeLLM([_answer("should not reach")])
        agent, _ = _agent(llm, AgentOptions(tool_exposure="all"), {})

        # Too large on its own: no older working memory to drop helps.
        assert "Plan check failed" in await agent.run("x" * 110_000)
        assert llm.calls == []


# ── tool_exposure="retrieve": the built-ins and the extra tools ─────────────

# Shaped like the Aider exercises whose runs never got file_write before
# the built-ins were always offered.
AIDER_QUERIES = [
    "Implement the transpose function in transpose.py so that the tests in "
    "transpose_test.py pass",
    "Implement the list operations in list_ops.py so that the tests in "
    "list_ops_test.py pass",
    "Implement the scale generator in scale_generator.py so that the tests in "
    "scale_generator_test.py pass",
]

# Ten provided tools, several of whose descriptions share common words.
TEN_TOOLS = {
    "calendar_add": "Add an event to the calendar",
    "email_send": "Send an email to a contact",
    "map_route": "Find the route between two places",
    "music_play": "Play a song or a playlist",
    "notes_search": "Search the notes for a phrase",
    "stock_price": "Look up the price of a stock",
    "timer_start": "Start a timer for a number of minutes",
    "translate_text": "Translate text into another language",
    "weather_forecast": "Get the weather forecast for a city",
    "web_fetch": "Fetch the text of a web page",
}

BUILTINS = sorted(builtin_names())


def _with_providers(
    llm: FakeLLM, options: AgentOptions, tools: dict[str, str]
) -> Agent:
    """An agent whose provider supplies *tools* (name -> description)."""
    return Agent(llm=llm, options=options, tool_providers=[FakeToolProvider(tools)])


class TestRetrieveExposure:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", AIDER_QUERIES)
    async def test_an_aider_shaped_query_offers_file_write(self, query: str) -> None:
        llm = FakeLLM([_answer("done")])
        agent, _ = _agent(llm, AgentOptions(), {})

        assert await agent.run(query) == "done"
        assert "file_write" in _offered(llm.calls[0])
        # With no other tool registered, every built-in, sorted by name.
        assert _offered(llm.calls[0]) == BUILTINS

    @pytest.mark.asyncio
    async def test_only_the_selected_builtins_are_always_offered(self) -> None:
        llm = FakeLLM([_call("file_write", "x"), _answer("done")])
        options = AgentOptions(builtins=frozenset({"file_read", "shell_exec"}))
        agent, _ = _agent(llm, options, {})

        # "hello" shares no word with the name, description or tags of
        # either built-in; both are offered on every step all the same.
        result = await agent.run_task("hello")
        assert (result.status, result.result) == ("completed", "done")
        # file_write is neither registered nor offered, although the call
        # names it.
        assert [_offered(call) for call in llm.calls] == [
            ["file_read", "shell_exec"]
        ] * 2
        assert _results(llm.calls[1]) == [
            "Error: Tool not found in registry: file_write"
        ]
        # Both planning calls offered tools, so both count as main requests,
        # none as auxiliary ones.
        assert (result.n_llm, result.n_main, result.n_aux) == (2, 2, 0)

    @pytest.mark.asyncio
    async def test_without_builtins_only_the_extra_tools_are_retrieved(
        self,
    ) -> None:
        llm = FakeLLM([_answer("done")])
        tools = {"weather": "Weather forecast for a city", "calendar": "Calendar"}
        agent = _with_providers(llm, AgentOptions(builtins=frozenset()), tools)

        assert await agent.run("What will the weather be in Paris?") == "done"
        assert _offered(llm.calls[0]) == ["weather"]

    @pytest.mark.asyncio
    async def test_the_builtins_come_first_then_the_extra_tools_by_name(
        self,
    ) -> None:
        llm = FakeLLM([_answer("done")])
        agent = _with_providers(llm, AgentOptions(), TEN_TOOLS)

        assert await agent.run("What will the weather be in Paris tomorrow?") == "done"
        offered = _offered(llm.calls[0])
        assert offered[: len(BUILTINS)] == BUILTINS
        # BM25 ranks weather_forecast first; the offered list is by name.
        extra = offered[len(BUILTINS) :]
        assert "weather_forecast" in extra
        assert extra == sorted(extra)
        assert len(extra) <= TOOL_SEARCH_TOP_K
        assert len(set(offered)) == len(offered)

    @pytest.mark.asyncio
    async def test_a_chinese_query_reaches_a_tool_with_a_chinese_description(
        self,
    ) -> None:
        llm = FakeLLM([_answer("done")])
        tools = {
            "translate_cn": "把文本翻译成英文",
            "weather_cn": "查询城市的天气预报",
            "web_fetch": "Fetch the text of a web page",
        }
        agent = _with_providers(llm, AgentOptions(), tools)

        assert await agent.run("北京明天的天气怎么样？") == "done"
        assert _offered(llm.calls[0]) == [*BUILTINS, "weather_cn"]

    @pytest.mark.asyncio
    async def test_the_tools_stay_byte_identical_while_the_extra_tools_do(
        self, tmp_path: Path
    ) -> None:
        """Calls to built-ins and to an offered extra tool reorder nothing.

        Hint extraction used to move each tool the model had called to the
        front, so the tools part of the request changed from step to step.
        """
        notes = tmp_path / "notes.txt"
        notes.write_text("Paris, tomorrow", encoding="utf-8")
        llm = FakeLLM(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_call=ToolCall(
                        tool_name="file_read", arguments={"path": str(notes)}
                    ),
                ),
                Message(
                    role="assistant",
                    content="",
                    tool_call=ToolCall(
                        tool_name="python_exec", arguments={"code": "print(6 * 7)"}
                    ),
                ),
                _call("weather_forecast", "Paris"),
                _answer("done"),
            ]
        )
        agent = _with_providers(llm, AgentOptions(), TEN_TOOLS)

        assert await agent.run("What will the weather be in Paris tomorrow?") == "done"
        assert len(llm.calls) == 4
        assert len({_wire(call) for call in llm.calls}) == 1
        assert _results(llm.calls[3]) == ["Paris, tomorrow", "42", "Paris"]

    @pytest.mark.asyncio
    async def test_a_call_to_a_lower_ranked_extra_tool_reorders_nothing(
        self,
    ) -> None:
        """The extra tools are sorted by name, not in hint or rank order.

        BM25 ranks weather_forecast first for this query. A call to
        map_route makes it a hint on the next step, which in hint order
        would move it to the front of the same set of extra tools.
        """
        llm = FakeLLM([_call("map_route", "Paris"), _answer("done")])
        agent = _with_providers(llm, AgentOptions(), TEN_TOOLS)

        assert await agent.run("What will the weather be in Paris tomorrow?") == "done"
        assert len(llm.calls) == 2
        assert {"map_route", "weather_forecast"} <= set(_offered(llm.calls[0]))
        assert _wire(llm.calls[1]) == _wire(llm.calls[0])

    @pytest.mark.asyncio
    async def test_a_builtin_the_query_does_not_match_still_runs(
        self, tmp_path: Path
    ) -> None:
        """The allowlist follows the offered list: every built-in may run.

        A registered extra tool that the step did not offer is still refused.
        """
        target = tmp_path / "transpose.py"
        content = "def transpose(lines):\n    return lines\n"
        llm = FakeLLM(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="file_write",
                            arguments={"path": str(target), "content": content},
                        ),
                        ToolCall(tool_name="zeta", arguments={"text": "1"}),
                    ],
                ),
                _answer("done"),
            ]
        )
        agent, ran = _agent(llm, AgentOptions(), {"zeta": "last letter"})

        assert await agent.run(AIDER_QUERIES[0]) == "done"
        assert _offered(llm.calls[0]) == BUILTINS
        assert target.read_text(encoding="utf-8") == content
        assert _results(llm.calls[1]) == [
            f"Written {len(content)} bytes to {target}",
            _refused("zeta"),
        ]
        assert ran == []


# ── tool_exposure="retrieve": the execution allowlist ───────────────────────

# "echo" matches the query "echo one" by name (hint and BM25); "zeta" shares
# no word with either query below, so retrieval never offers it on its own.
ECHO_AND_ZETA = {"echo": "echo text back", "zeta": "last letter"}


class TestRetrieveAllowlist:
    @pytest.mark.asyncio
    async def test_tool_not_offered_is_refused_while_offered_call_runs(self) -> None:
        llm = FakeLLM([_batch(("zeta", "1"), ("echo", "2")), _answer("done")])
        agent, ran = _agent(llm, AgentOptions(builtins=frozenset()), ECHO_AND_ZETA)

        assert await agent.run("echo one") == "done"
        # zeta is registered, but this step offered only echo.
        assert _offered(llm.calls[0]) == ["echo"]
        assert ran == [("echo", "2")]
        assert _results(llm.calls[1]) == [_refused("zeta"), "2"]

    @pytest.mark.asyncio
    async def test_nothing_offered_refuses_every_call(self) -> None:
        llm = FakeLLM([_call("zeta", "1"), _answer("done")])
        options = AgentOptions(builtins=frozenset())
        agent, ran = _agent(llm, options, {"zeta": "last letter"})

        assert await agent.run("hello") == "done"
        # No tool matched, so the request carried none; the call made anyway
        # meets an empty allowlist.
        assert llm.calls[0]["tools"] is None
        assert ran == []
        assert _results(llm.calls[1]) == [_refused("zeta")]

    @pytest.mark.asyncio
    async def test_refused_tool_is_offered_on_the_next_step(self) -> None:
        """Known limit: the allowlist holds within one step only.

        The refused call and its error put the tool's name into working
        memory, so hint extraction offers it on the next step and the same
        call then runs. Since C1 this concerns only the tools other than
        the built-ins, which every step offers; hint extraction stays as
        it was for those tools. Revisit this test with the hints rework.
        """
        llm = FakeLLM([_call("zeta", "1"), _call("zeta", "2"), _answer("done")])
        agent, ran = _agent(llm, AgentOptions(builtins=frozenset()), ECHO_AND_ZETA)

        assert await agent.run("echo one") == "done"
        assert [_offered(call) for call in llm.calls] == [
            ["echo"],
            ["echo", "zeta"],
            ["echo", "zeta"],
        ]
        assert _results(llm.calls[1]) == [_refused("zeta")]
        assert ran == [("zeta", "2")]
