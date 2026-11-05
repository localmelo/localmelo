"""Refused calls don't use up a run's steps (``AgentOptions.refused_steps``).

A step whose every call a policy check refused (the tool's own check marked
it with ``refuse_call``, or the approval policy denied it) does not count
toward ``max_steps``, up to ``refused_steps`` such steps per run. A step with
one call that ran counts as before, and the default (0) changes nothing.
No real model.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tools import refuse_call
from localmelo.melo.schema import Message, ToolCall, ToolDef, ToolResult


class FakeLLM(BaseLLMProvider):
    def __init__(self, replies: list[Message]) -> None:
        self.replies: deque[Message] = deque(replies)
        self.calls = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls += 1
        if not self.replies:
            raise AssertionError("the test queued no further reply")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


def _tool(name: str) -> ToolDef:
    return ToolDef(
        name=name,
        description=name,
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


async def _note(text: str) -> str:
    """A tool whose own check refuses every call."""
    return refuse_call(f"Not kept: {text} is not something the user said.")


async def _echo(text: str) -> str:
    return text


def _call(name: str, text: str, call_id: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name=name, arguments={"text": text}, id=call_id)],
    )


def _agent(llm: BaseLLMProvider, **options: Any) -> Agent:
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(), attempt_policy="none", answer_on_stop=False, **options
        ),
    )
    for name, function in (("note", _note), ("echo", _echo)):
        agent.hippo.register_tool(_tool(name))
        agent.executor.register(name, function)
    return agent


def _refused_then_answer() -> list[Message]:
    return [
        _call("note", "a", "c1"),
        _call("note", "b", "c2"),
        _call("echo", "hello", "c3"),
        Message(role="assistant", content="Here is the answer."),
    ]


async def test_refused_steps_do_not_use_up_the_budget() -> None:
    llm = FakeLLM(_refused_then_answer())
    agent = _agent(llm, max_steps=2, refused_steps=2)
    outcome = await agent.run_task("note it, then echo hello")
    assert (outcome.status, outcome.result) == ("completed", "Here is the answer.")
    assert llm.calls == 4


async def test_without_the_allowance_refused_steps_count() -> None:
    llm = FakeLLM(_refused_then_answer())
    agent = _agent(llm, max_steps=2)
    outcome = await agent.run_task("note it, then echo hello")
    assert outcome.status == "budget_exhausted"
    assert llm.calls == 2


async def test_at_most_the_allowance_of_refused_steps_is_free() -> None:
    replies = [_call("note", str(n), f"c{n}") for n in range(5)]
    llm = FakeLLM(replies)
    agent = _agent(llm, max_steps=1, refused_steps=2)
    outcome = await agent.run_task("note it")
    assert outcome.status == "budget_exhausted"
    assert llm.calls == 3, "two free steps, then the one step of the budget"


async def test_a_step_with_a_call_that_ran_counts() -> None:
    both = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(tool_name="note", arguments={"text": "a"}, id="c1"),
            ToolCall(tool_name="echo", arguments={"text": "b"}, id="c2"),
        ],
    )
    llm = FakeLLM([both, Message(role="assistant", content="Done.")])
    agent = _agent(llm, max_steps=1, refused_steps=2)
    outcome = await agent.run_task("note and echo")
    assert outcome.status == "budget_exhausted"
    assert llm.calls == 1


class _Deny:
    """An approval policy that denies every call, as *origin* decided."""

    def __init__(self, origin: str = "policy") -> None:
        self.origin = origin

    async def decide(self, request: Any) -> ApprovalDecision:
        return ApprovalDecision(False, "not allowed", origin=self.origin)  # type: ignore[arg-type]


async def test_a_call_the_approval_policy_denied_is_a_refused_step() -> None:
    replies = [
        _call("guarded", "x", "c1"),
        Message(role="assistant", content="It was not allowed, so here is why."),
    ]
    llm = FakeLLM(replies)
    agent = _agent(llm, max_steps=1, refused_steps=2)
    agent.hippo.register_tool(_tool("guarded"))
    agent.executor.register("guarded", _echo, requires_approval=True)
    outcome = await agent.run_task("run the guarded step", approval=_Deny())
    assert outcome.status == "completed"
    assert llm.calls == 2


@pytest.mark.parametrize("origin", ["user", "timeout"])
async def test_the_persons_own_denial_counts(origin: str) -> None:
    # A step the person (or their silence) refused is theirs: asking again is
    # no free step, so the model cannot ask twice more for free.
    replies = [
        _call("guarded", "x", "c1"),
        Message(role="assistant", content="Not allowed, then."),
    ]
    llm = FakeLLM(replies)
    agent = _agent(llm, max_steps=1, refused_steps=2)
    agent.hippo.register_tool(_tool("guarded"))
    agent.executor.register("guarded", _echo, requires_approval=True)
    outcome = await agent.run_task("run the guarded step", approval=_Deny(origin))
    assert outcome.status == "budget_exhausted"
    assert llm.calls == 1


async def test_the_same_refused_call_again_counts() -> None:
    replies = [_call("note", "same", f"c{n}") for n in range(4)]
    llm = FakeLLM(replies)
    agent = _agent(llm, max_steps=1, refused_steps=2)
    outcome = await agent.run_task("note it")
    assert outcome.status == "budget_exhausted"
    assert llm.calls == 2, "the first refusal was free, its repeat was not"


def test_the_same_refused_command_with_new_words_counts() -> None:
    """The ``why`` only describes a step: rewording it does not make a fresh refusal."""
    agent = _agent(FakeLLM([]))
    run = SimpleNamespace(refused_calls=set())
    outcome = SimpleNamespace(error_category=None)

    def free(tool: str, **arguments: Any) -> bool:
        call = ToolCall(tool_name=tool, arguments=arguments, id="c")
        result = ToolResult(tool, error="Not allowed.")
        return agent._refused_free(call, result, outcome, None, ["no"], [], run)  # type: ignore[arg-type]

    assert free("shell_exec", command="make", why="One reason") is True
    assert free("shell_exec", command="make", why="Another reason") is False
    assert free("shell_exec", command="make deploy", why="One reason") is True
    assert free("python_exec", code="print(1)", why="One reason") is True
    assert free("python_exec", code="print(1)", why="Another") is False
    # Another tool's ``why`` is its own argument: a different call.
    assert free("create_issue", title="x", why="One reason") is True
    assert free("create_issue", title="x", why="Another reason") is True


def test_the_allowance_is_checked() -> None:
    options = AgentOptions(builtins=frozenset())
    assert options.refused_steps == 0
    bad: Any
    for bad in (-1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="refused_steps"):
            dataclasses.replace(options, refused_steps=bad)


def test_refuse_call_outside_a_run_only_returns_the_text() -> None:
    assert refuse_call("Not done.") == "Not done."
