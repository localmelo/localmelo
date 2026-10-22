"""How a run stops: reflection decisions and their parsing — no real model."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import FINAL_REFLECTION_NOTE, REFLECTION_PROMPT
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import MAX_REFLECTION_CHARS, Message, ToolCall, ToolDef

# ── Fakes and helpers ───────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Pops queued replies and records every request; fails when none is left."""

    def __init__(self, replies: list[Message]) -> None:
        self.replies: deque[Message] = deque(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            raise AssertionError("the test queued no further reply")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


async def _echo(text: str) -> str:
    return text


def _agent(llm: BaseLLMProvider, **options: Any) -> Agent:
    """An agent whose only tool is echo (named in every query below)."""
    agent = Agent(llm=llm, options=AgentOptions(builtins=frozenset(), **options))
    agent.hippo.register_tool(
        ToolDef(
            name="echo",
            description="echo text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    )
    agent.executor.register("echo", _echo)
    return agent


def _echo_call(text: str, content: str = "") -> Message:
    return Message(
        role="assistant",
        content=content,
        tool_call=ToolCall(tool_name="echo", arguments={"text": text}),
    )


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _decision(action: str, **fields: Any) -> dict[str, Any]:
    """A reflection decision; with ``continue`` it passes the continuation gate."""
    data: dict[str, Any] = {
        "recommended_action": action,
        "rationale": "test",
        "best_effort_result": "",
        "summary": "tried echo",
        "task_still_feasible": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "estimated_info_gain": 0.5,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.8,
        "feasibility": 0.8,
    }
    data.update(fields)
    return data


def _reflection(action: str, **fields: Any) -> Message:
    return Message(role="assistant", content=json.dumps(_decision(action, **fields)))


def _is_reflection(call: dict[str, Any]) -> bool:
    system: Message = call["messages"][0]
    return system.content.startswith(REFLECTION_PROMPT)


def _reflection_requests(llm: FakeLLM) -> list[str]:
    """The closing user message of each reflection request."""
    requests = []
    for call in llm.calls:
        if _is_reflection(call):
            assert call["messages"][-1].role == "user"
            requests.append(call["messages"][-1].content)
    return requests


# ── The reflection prompt ───────────────────────────────────────────────────


class TestReflectionPrompt:
    def test_lists_done_and_when_to_choose_it(self) -> None:
        assert '"continue" | "stop" | "decompose" | "done"' in REFLECTION_PROMPT
        assert (
            "Choose 'done' only if the task is already complete, and put the final "
            "answer in best_effort_result." in REFLECTION_PROMPT
        )


# ── done ────────────────────────────────────────────────────────────────────


class TestDone:
    @pytest.mark.asyncio
    async def test_completes_with_its_answer(self) -> None:
        llm = FakeLLM([_echo_call("a"), _reflection("done", best_effort_result="42")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == ("completed", "42", "")
        assert (result.steps, result.attempts, result.n_aux) == (1, 1, 1)

    @pytest.mark.asyncio
    async def test_without_an_answer_uses_the_latest_text(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a", content="first look"),
                _echo_call("b", content="it is 7"),
                _echo_call("c", content="  \n"),
                _reflection("done", best_effort_result=" "),
            ]
        )
        agent = _agent(llm, steps_per_attempt=3, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result) == ("completed", "it is 7")

    @pytest.mark.asyncio
    async def test_without_an_answer_or_any_text_completes_empty(self) -> None:
        llm = FakeLLM([_echo_call("a"), _reflection("done")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == ("completed", "", "")

    @pytest.mark.asyncio
    async def test_latest_text_comes_from_this_run(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a", content="an earlier run's text"),
                _answer("first"),
                _echo_call("b"),
                _reflection("done"),
            ]
        )
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        assert (await agent.run_task("echo one")).result == "first"
        result = await agent.run_task("echo two")
        assert (result.status, result.result) == ("completed", "")

    @pytest.mark.asyncio
    async def test_after_the_last_attempt_still_completes(self) -> None:
        llm = FakeLLM([_echo_call("a"), _reflection("done", best_effort_result="42")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=1)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result) == ("completed", "42")

    @pytest.mark.asyncio
    async def test_an_oversized_done_is_not_forced_to_decompose(self) -> None:
        summary = "x" * (MAX_REFLECTION_CHARS + 100)
        reply = _reflection("done", best_effort_result="42", summary=summary)
        agent = _agent(FakeLLM([_echo_call("a"), reply]), steps_per_attempt=1)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result) == ("completed", "42")

    @pytest.mark.asyncio
    async def test_the_task_record_holds_the_answer(self) -> None:
        llm = FakeLLM([_echo_call("a"), _reflection("done", best_effort_result="42")])
        agent = _agent(llm, steps_per_attempt=1)
        result = await agent.run_task("echo the answer")
        task = await agent.hippo.get_task(result.task_id)
        assert task is not None
        assert (task.status, task.result) == ("completed", "42")


# ── gate_stopped ────────────────────────────────────────────────────────────


class TestGateStopped:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["stop", "decompose"])
    async def test_error_is_the_rationale(self, action: str) -> None:
        reply = _reflection(action, rationale="no way forward", best_effort_result="7")
        agent = _agent(FakeLLM([_echo_call("a"), reply]), steps_per_attempt=1)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "7",
            "no way forward",
        )

    @pytest.mark.asyncio
    async def test_a_continue_the_gate_refuses_stops_too(self) -> None:
        reply = _reflection("continue", estimated_info_gain=0.0, rationale="meh")
        agent = _agent(FakeLLM([_echo_call("a"), reply]), steps_per_attempt=1)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.error) == ("gate_stopped", "meh")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rationale", ["", " \n "])
    async def test_a_blank_rationale_names_the_attempt(self, rationale: str) -> None:
        llm = FakeLLM(
            [
                _echo_call("a"),
                _reflection("continue"),
                _echo_call("b"),
                _reflection("stop", rationale=rationale),
            ]
        )
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.error) == (
            "gate_stopped",
            "Stopped after attempt 2",
        )
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_an_unparsed_reflection_stops_with_its_text(self) -> None:
        reply = _answer("I would rather not answer in JSON.")
        agent = _agent(FakeLLM([_echo_call("a"), reply]), steps_per_attempt=1)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "I would rather not answer in JSON.",
            "Could not parse reflection response",
        )


# ── The reflection after which no attempt can start ─────────────────────────


class TestFinalReflection:
    @pytest.mark.asyncio
    async def test_keeps_the_best_effort_after_the_last_attempt(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a"),
                _reflection("continue"),
                _echo_call("b"),
                _reflection("continue", best_effort_result="partial"),
            ]
        )
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "budget_exhausted",
            "partial",
            "Max attempts reached",
        )

    @pytest.mark.asyncio
    async def test_keeps_the_best_effort_once_the_steps_are_used_up(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a"),
                _echo_call("b"),
                _reflection("continue", best_effort_result="partial"),
            ]
        )
        agent = _agent(llm, max_steps=2, steps_per_attempt=2, max_attempts=3)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "budget_exhausted",
            "partial",
            "Max steps reached",
        )
        assert (result.steps, result.attempts) == (2, 1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["stop", "decompose"])
    async def test_a_stop_is_budget_exhausted(self, action: str) -> None:
        reply = _reflection(action, best_effort_result="partial", rationale="r")
        agent = _agent(
            FakeLLM([_echo_call("a"), reply]), steps_per_attempt=1, max_attempts=1
        )
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "budget_exhausted",
            "partial",
            "Max attempts reached",
        )

    @pytest.mark.asyncio
    async def test_only_the_final_reflection_is_told_so(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a"),
                _reflection("continue"),
                _echo_call("b"),
                _reflection("continue"),
            ]
        )
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        await agent.run_task("echo the answer")
        first, last = _reflection_requests(llm)
        assert FINAL_REFLECTION_NOTE not in first
        assert last.endswith(FINAL_REFLECTION_NOTE)


# ── Tolerant reflection parsing ─────────────────────────────────────────────

DONE = json.dumps(_decision("done", best_effort_result="42"))


class TestTolerantParsing:
    @pytest.mark.parametrize(
        "text",
        [
            DONE,
            f"```json\n{DONE}\n```",
            f"```\n{DONE}\n```",
            f"  ```JSON\n{DONE}```  ",
            f"```json {DONE}```",
            f"Here is my reflection:\n{DONE}\nThat is all.",
            f"Reflection:\n```json\n{DONE}\n```\n",
            f"Using the format {{action}}: {DONE}",
            f"An unclosed {{ brace, then {DONE}",
        ],
    )
    def test_finds_the_decision(self, text: str) -> None:
        _, decision = Agent._parse_reflection(text, 0, "budget")
        assert decision.recommended_action == "done"
        assert decision.best_effort_result == "42"

    def test_braces_inside_strings_do_not_end_the_object(self) -> None:
        data = _decision("done", best_effort_result="use {x} or } then {")
        text = f"Sure. {json.dumps(data)} Done."
        _, decision = Agent._parse_reflection(text, 0, "")
        assert decision.best_effort_result == "use {x} or } then {"

    def test_the_first_object_wins(self) -> None:
        stop = json.dumps(_decision("stop"))
        _, decision = Agent._parse_reflection(f"{DONE}\n{stop}", 0, "")
        assert decision.recommended_action == "done"

    @pytest.mark.parametrize(
        "text",
        [
            "no JSON here",
            "an unclosed { brace",
            "{not: json}",
            "```json\n```",
            "{" * 2_000,
            '{"a": ' * 2_000,
            # Nested deeper than the JSON decoder's recursion limit.
            '{"a": ' * 50_000,
        ],
    )
    def test_falls_back_to_stop(self, text: str) -> None:
        _, decision = Agent._parse_reflection(text, 0, "")
        assert decision.recommended_action == "stop"
        assert decision.rationale == "Could not parse reflection response"
        assert decision.best_effort_result == text[:500]

    def test_json_that_is_not_an_object_still_stops(self) -> None:
        _, decision = Agent._parse_reflection("```json\n[1, 2]\n```", 0, "")
        assert decision.recommended_action == "stop"
        assert decision.rationale == "Reflection response was not a JSON object"

    @pytest.mark.asyncio
    async def test_a_fenced_continue_starts_the_next_attempt(self) -> None:
        fenced = f"```json\n{json.dumps(_decision('continue'))}\n```"
        llm = FakeLLM([_echo_call("a"), _answer(fenced), _answer("42")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.attempts) == ("completed", "42", 2)
