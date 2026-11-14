"""Agent: every run offers every registered tool, sorted by name — no real network or model."""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker import DEFAULT_BUILTINS
from localmelo.melo.checker.payloads import ToolResolutionResult, ValidationResult
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
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


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _offered(call: dict[str, Any]) -> list[str]:
    """Names of the tools one request offered, in order."""
    return [t.name for t in call["tools"] or []]


def _results(call: dict[str, Any]) -> list[str]:
    """Contents of the tool-result messages one request carried, in order."""
    return [m.content for m in call["messages"] if m.role == "tool"]


def _wire(call: dict[str, Any]) -> str:
    """The tools of one request, serialized byte for byte."""
    return json.dumps([dataclasses.asdict(t) for t in call["tools"]])


# Three tools, registered out of order.
UNMATCHED = {
    "zeta": "last letter",
    "alpha": "first letter",
    "mid": "in between",
}


# ── Every registered tool ───────────────────────────────────────────────────


class TestEveryToolIsOffered:
    @pytest.mark.asyncio
    async def test_every_tool_sorted_and_identical_on_every_step(self) -> None:
        llm = FakeLLM([_call("zeta", "1"), _call("alpha", "2"), _answer("done")])
        options = AgentOptions(builtins=frozenset())
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
        agent, _ = _agent(llm, AgentOptions(), {"echo": "echo"})

        assert await agent.run("hello") == "done"
        assert _offered(llm.calls[0]) == sorted(DEFAULT_BUILTINS["python"] | {"echo"})

    @pytest.mark.asyncio
    async def test_the_resolution_is_checked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = FakeLLM([_answer("done")])
        options = AgentOptions(builtins=frozenset())
        agent, _ = _agent(llm, options, {"zeta": "z", "alpha": "a"})

        checked: list[ToolResolutionResult] = []
        check = agent.checker.check_tool_resolution

        def spy(result: ToolResolutionResult) -> ValidationResult:
            checked.append(result)
            return check(result)

        monkeypatch.setattr(agent.checker, "check_tool_resolution", spy)

        assert await agent.run("hello") == "done"
        assert [(r.query, r.resolved_tool_names) for r in checked] == [
            ("hello", ["alpha", "zeta"])
        ]

    @pytest.mark.asyncio
    async def test_pre_plan_check_still_runs(self) -> None:
        llm = FakeLLM([_answer("should not reach")])
        agent, _ = _agent(llm, AgentOptions(), {})

        # Too large on its own: no older working memory to drop helps.
        assert "Plan check failed" in await agent.run("x" * 110_000)
        assert llm.calls == []


class TestSelectedBuiltins:
    @pytest.mark.asyncio
    async def test_only_the_selected_builtins_are_offered(self) -> None:
        llm = FakeLLM([_call("file_write", "x"), _answer("done")])
        options = AgentOptions(builtins=frozenset({"file_read", "shell_exec"}))
        agent, _ = _agent(llm, options, {})

        result = await agent.run_task("hello")
        assert (result.status, result.result) == ("completed", "done")
        # file_write is neither registered nor offered, although the call
        # names it.
        assert [_offered(call) for call in llm.calls] == [
            ["file_read", "shell_exec"]
        ] * 2
        assert _results(llm.calls[1]) == [
            "Error: file_write is not available in this run. "
            "Available tools: file_read, shell_exec."
        ]
        # Both planning calls offered tools, so both count as main requests,
        # none as auxiliary ones.
        assert (result.n_llm, result.n_main, result.n_aux) == (2, 2, 0)
