"""Agent: tool exposure modes and the execution allowlist — no real network or model."""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker.payloads import ToolResolutionResult, ValidationResult
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.executor import builtin_names
from localmelo.melo.executor.executor import ToolCallable
from localmelo.melo.schema import Message, ToolCall, ToolDef

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
        call then runs. Revisit this test with the hints rework (review
        finding C1).
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
