"""AgentOptions.reflection_gate: when the next attempt starts after a
reflection, under "flags" (the default) and "utility" — no real model."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import (
    REFLECTION_PROMPT,
    REFLECTION_RETRY_NOTE,
    REFLECTION_TRANSCRIPT_INTRO,
    render_transcript,
)
from localmelo.melo.agent.agent import (
    CONTINUATION_UTILITY_THRESHOLD,
    _find_reflection,
    _holds_reflection_object,
    _should_continue,
    _utility,
)
from localmelo.melo.agent.options import ReflectionGate
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import (
    Message,
    ReflectionDecision,
    ReflectionEntry,
    ToolCall,
    ToolDef,
)

GATES: list[ReflectionGate] = ["flags", "utility"]

# The three checks of a reflection decision that both gates require.
CHECKS = ["task_still_feasible", "next_step_is_concrete", "next_step_is_novel"]

# Estimates that give a utility of -0.5: 0.0 * 0.8 * 0.8 - 0.25 - 0.25.
LOW_UTILITY = {"estimated_info_gain": 0.0, "estimated_cost": 0.25, "repeat_risk": 0.25}


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


class FakeEmbedding(BaseEmbeddingProvider):
    """Embeds every text alike and keeps each text it was given."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


async def _echo(text: str) -> str:
    return text


def _agent(
    llm: BaseLLMProvider,
    *,
    embedding: BaseEmbeddingProvider | None = None,
    **options: Any,
) -> Agent:
    """An agent whose only tool is echo (named in every query below)."""
    agent = Agent(
        llm=llm,
        embedding=embedding,
        options=AgentOptions(builtins=frozenset(), **options),
    )
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
    """A reflection decision; with ``continue`` it passes both gates."""
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


def _reflection_count(llm: FakeLLM) -> int:
    return sum(1 for call in llm.calls if _is_reflection(call))


def _kinds(llm: FakeLLM) -> list[str]:
    """Each request in order: ``plan`` or ``reflect``."""
    return ["reflect" if _is_reflection(call) else "plan" for call in llm.calls]


def _planning_systems(llm: FakeLLM) -> list[str]:
    """The system message of each planning request."""
    return [
        call["messages"][0].content for call in llm.calls if not _is_reflection(call)
    ]


def _added_reflections(
    agent: Agent, monkeypatch: pytest.MonkeyPatch
) -> list[ReflectionEntry]:
    """Every entry that the agent adds to working memory from now on."""
    added: list[ReflectionEntry] = []
    add = agent.hippo.working.add_reflection

    def add_reflection(entry: ReflectionEntry) -> None:
        added.append(entry)
        add(entry)

    monkeypatch.setattr(agent.hippo.working, "add_reflection", add_reflection)
    return added


def _passing_decision(**fields: Any) -> ReflectionDecision:
    """A ``continue`` decision whose three checks hold."""
    values: dict[str, Any] = {
        "recommended_action": "continue",
        "task_still_feasible": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
    }
    values.update(fields)
    return ReflectionDecision(**values)


# ── The gate ────────────────────────────────────────────────────────────────


class TestShouldContinue:
    def test_the_utility_comes_from_the_five_estimates(self) -> None:
        decision = _passing_decision(
            estimated_info_gain=0.5,
            feasibility=0.5,
            novelty=0.5,
            estimated_cost=0.25,
            repeat_risk=0.125,
        )
        assert _utility(decision) == 0.5 * 0.5 * 0.5 - 0.25 - 0.125

    def test_flags_ignore_the_utility(self) -> None:
        decision = _passing_decision(estimated_cost=1.0, repeat_risk=1.0)
        assert _utility(decision) == -2.0
        assert _should_continue(decision) is True
        assert _should_continue(decision, "flags") is True
        assert _should_continue(decision, "utility") is False

    def test_utility_continues_from_the_threshold_on(self) -> None:
        at = _passing_decision(estimated_info_gain=0.1, feasibility=1.0, novelty=1.0)
        assert _utility(at) == CONTINUATION_UTILITY_THRESHOLD
        assert _should_continue(at, "utility") is True
        below = _passing_decision(
            estimated_info_gain=0.1, feasibility=1.0, novelty=1.0, repeat_risk=0.01
        )
        assert _should_continue(below, "utility") is False

    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("action", ["stop", "decompose", "done"])
    def test_only_continue_continues(self, gate: ReflectionGate, action: str) -> None:
        decision = _passing_decision(
            recommended_action=action,
            estimated_info_gain=1.0,
            feasibility=1.0,
            novelty=1.0,
        )
        assert _should_continue(decision, gate) is False

    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("check", CHECKS)
    def test_every_check_must_hold(self, gate: ReflectionGate, check: str) -> None:
        decision = _passing_decision(
            estimated_info_gain=1.0, feasibility=1.0, novelty=1.0, **{check: False}
        )
        assert _should_continue(decision, gate) is False


class TestGateInTheLoop:
    @pytest.mark.asyncio
    async def test_a_low_utility_continue_starts_the_next_attempt_under_flags(
        self,
    ) -> None:
        reply = _reflection(
            "continue", rationale="run the tests", best_effort_result="7", **LOW_UTILITY
        )
        llm = FakeLLM([_echo_call("a"), reply, _answer("42")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == ("completed", "42", "")
        assert (result.steps, result.attempts) == (2, 2)
        assert (result.n_main, result.n_aux) == (2, 1)

    @pytest.mark.asyncio
    async def test_a_low_utility_continue_is_gate_stopped_under_utility(
        self,
    ) -> None:
        reply = _reflection(
            "continue", rationale="run the tests", best_effort_result="7", **LOW_UTILITY
        )
        llm = FakeLLM([_echo_call("a"), reply, _answer("42")])
        agent = _agent(
            llm, steps_per_attempt=1, max_attempts=3, reflection_gate="utility"
        )
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "7",
            "run the tests",
        )
        assert (result.steps, result.attempts) == (1, 1)
        assert (result.n_main, result.n_aux) == (1, 1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("check", CHECKS)
    async def test_a_continue_with_a_check_false_is_gate_stopped(
        self, gate: ReflectionGate, check: str
    ) -> None:
        reply = _reflection(
            "continue",
            rationale="not this way",
            best_effort_result="7",
            estimated_info_gain=1.0,
            novelty=1.0,
            feasibility=1.0,
            **{check: False},
        )
        llm = FakeLLM([_echo_call("a"), reply])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3, reflection_gate=gate)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "7",
            "not this way",
        )
        assert result.attempts == 1
        assert _reflection_count(llm) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", GATES)
    async def test_the_run_result_reports_the_gate(self, gate: ReflectionGate) -> None:
        agent = _agent(FakeLLM([_answer("42")]), reflection_gate=gate)
        result = await agent.run_task("what is the answer")
        assert result.config["reflection_gate"] == gate


# ── What counts as a reflection ─────────────────────────────────────────────

DONE = json.dumps(_decision("done", best_effort_result="42"))

# A reply recorded by the benchmark's E4 probe: a tool call written out as
# text, whose arguments decode as a JSON object without recommended_action.
TOOL_CALL_TEXT = '<tool_call>\n=python_exec({"code": "print(sum(range(10)))"})'

# A reply whose only JSON object has a recommended_action that is not valid.
INVALID_ACTION = json.dumps(_decision("proceed", rationale="go on"))


class TestFindReflection:
    @pytest.mark.parametrize(
        "text",
        [
            DONE,
            f"```json\n{DONE}\n```",
            f"Here it is: {DONE} That is all.",
            f"{TOOL_CALL_TEXT}\n{DONE}",
            f'{json.dumps({"name": "python_exec", "arguments": {"x": 1}})}\n{DONE}',
            f"{INVALID_ACTION} {DONE}",
            f"{DONE}\n{json.dumps(_decision('stop'))}",
        ],
        ids=[
            "bare",
            "fenced",
            "in_text",
            "after_a_tool_call_text",
            "after_a_tool_call_object",
            "after_an_invalid_action",
            "the_first_of_two",
        ],
    )
    def test_the_first_object_with_a_valid_action(self, text: str) -> None:
        found = _find_reflection(text)
        assert found is not None
        assert (found["recommended_action"], found["best_effort_result"]) == (
            "done",
            "42",
        )

    @pytest.mark.parametrize(
        "text",
        [
            "no JSON here",
            "```json\n```",
            TOOL_CALL_TEXT,
            INVALID_ACTION,
            json.dumps(_decision("Done")),
            json.dumps({"summary": "no action"}),
            json.dumps({"recommended_action": ["done"]}),
            # JSON that is not an object, and an object nested in another.
            json.dumps([_decision("done")]),
            json.dumps({"reflection": _decision("done")}),
            "{" * 2_000,
            '{"a": ' * 50_000,
            # A number too long for the decoder to convert.
            '{"n": ' + "9" * 5_000 + "}",
        ],
        ids=[
            "prose",
            "empty_fence",
            "tool_call_text",
            "invalid_action",
            "action_in_another_case",
            "no_action",
            "action_not_a_string",
            "json_list",
            "nested",
            "unclosed_braces",
            "too_deep",
            "huge_number",
        ],
    )
    def test_none_without_one(self, text: str) -> None:
        assert _find_reflection(text) is None

    def test_utility_still_takes_any_object(self) -> None:
        assert _holds_reflection_object(TOOL_CALL_TEXT) is True
        assert _holds_reflection_object(INVALID_ACTION) is True


def _retry_is_the_usual_one(llm: FakeLLM) -> None:
    """The reflection request was asked again in the form #31 gave it."""
    first, retry = (call["messages"] for call in llm.calls if _is_reflection(call))
    assert [m.role for m in retry] == ["system", "user"]
    assert retry[0].content == first[0].content
    turns = render_transcript(first[1:-1])
    assert retry[1].content == "\n\n".join(
        [
            f"{REFLECTION_TRANSCRIPT_INTRO}\n\n{turns}",
            first[-1].content,
            REFLECTION_RETRY_NOTE,
        ]
    )


class TestReadableReflection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first", [TOOL_CALL_TEXT, INVALID_ACTION], ids=["tool_call", "invalid"]
    )
    async def test_flags_ask_again_for_a_reply_without_a_valid_action(
        self, first: str
    ) -> None:
        llm = FakeLLM([_echo_call("a"), _answer(first), _answer(DONE)])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result) == ("completed", "42")
        assert (result.n_main, result.n_aux) == (1, 2)
        _retry_is_the_usual_one(llm)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply", [TOOL_CALL_TEXT, INVALID_ACTION], ids=["tool_call", "invalid"]
    )
    async def test_flags_find_the_same_reply_twice_unreadable(self, reply: str) -> None:
        llm = FakeLLM([_echo_call("a"), _answer(reply), _answer(reply), _answer("7")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.attempts) == ("completed", "7", 2)
        assert _kinds(llm) == ["plan", "reflect", "reflect", "plan"]

    @pytest.mark.asyncio
    async def test_flags_read_a_reflection_after_a_tool_call_object(self) -> None:
        reply = f"{TOOL_CALL_TEXT}\n{DONE}"
        llm = FakeLLM([_echo_call("a"), _answer(reply)])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result) == ("completed", "42")
        assert _reflection_count(llm) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reply", "error"),
        [
            # Read from the tool call's arguments: no action, so stop.
            (TOOL_CALL_TEXT, "Stopped after attempt 1"),
            (INVALID_ACTION, "go on"),
            # The first object wins, and it is the tool call's.
            (f"{TOOL_CALL_TEXT}\n{DONE}", "Stopped after attempt 1"),
        ],
        ids=["tool_call", "invalid", "tool_call_then_reflection"],
    )
    async def test_utility_parses_the_first_object_without_asking_again(
        self, reply: str, error: str
    ) -> None:
        llm = FakeLLM([_echo_call("a"), _answer(reply)])
        agent = _agent(
            llm, steps_per_attempt=1, max_attempts=2, reflection_gate="utility"
        )
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "",
            error,
        )
        assert _reflection_count(llm) == 1


# ── A reply that holds no reflection, also when asked again ────────────────

PROSE = "I would rather not answer in JSON."
STILL_PROSE = "Still no JSON, sorry."
UNPARSED = "Could not parse reflection response"


def _never_embedded(embedding: FakeEmbedding, *texts: str) -> bool:
    return not any(text in embedded for embedded in embedding.texts for text in texts)


class TestUnreadableReflection:
    @pytest.mark.asyncio
    async def test_flags_start_the_next_attempt_without_a_reflection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embedding = FakeEmbedding()
        llm = FakeLLM(
            [_echo_call("a"), _answer(PROSE), _answer(STILL_PROSE), _answer("42")]
        )
        agent = _agent(llm, embedding=embedding, steps_per_attempt=1, max_attempts=3)
        added = _added_reflections(agent, monkeypatch)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == ("completed", "42", "")
        assert (result.steps, result.attempts) == (2, 2)
        # The boundary sent the reflection request and its retry, and added
        # no reflection, so the next attempt planned without one.
        assert _kinds(llm) == ["plan", "reflect", "reflect", "plan"]
        assert added == []
        assert "[REFLECTION]" not in _planning_systems(llm)[-1]
        # Neither reply reached long-term memory or the run's result.
        assert embedding.texts
        assert _never_embedded(embedding, PROSE, STILL_PROSE)
        assert UNPARSED not in json.dumps(result.to_dict())

    @pytest.mark.asyncio
    async def test_flags_keep_the_previous_reflection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embedding = FakeEmbedding()
        llm = FakeLLM(
            [
                _echo_call("a"),
                _reflection("continue", summary="echo a was not it"),
                _echo_call("b"),
                _answer(PROSE),
                _answer(STILL_PROSE),
                _answer("42"),
            ]
        )
        agent = _agent(llm, embedding=embedding, steps_per_attempt=1, max_attempts=3)
        added = _added_reflections(agent, monkeypatch)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.attempts) == ("completed", "42", 3)
        # The second boundary sent exactly two requests and added nothing.
        assert _kinds(llm) == ["plan", "reflect", "plan", "reflect", "reflect", "plan"]
        assert [entry.summary for entry in added] == ["echo a was not it"]
        # The third attempt plans with the first attempt's reflection, and
        # long-term memory receives that one, never the unreadable replies.
        assert "[REFLECTION]\nAttempt 0: echo a was not it" in (
            _planning_systems(llm)[-1]
        )
        assert any("Attempt 0: echo a was not it" in t for t in embedding.texts)
        assert _never_embedded(embedding, PROSE, STILL_PROSE)
        assert UNPARSED not in json.dumps(result.to_dict())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("options", "error"),
        [
            ({"max_attempts": 1}, "Max attempts reached"),
            ({"max_steps": 1}, "Max steps reached"),
        ],
        ids=["attempts", "steps"],
    )
    async def test_flags_end_with_the_latest_text_at_the_final_boundary(
        self, options: dict[str, Any], error: str
    ) -> None:
        llm = FakeLLM(
            [_echo_call("a", content="it is 7"), _answer(PROSE), _answer(STILL_PROSE)]
        )
        agent = _agent(llm, steps_per_attempt=1, **options)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "budget_exhausted",
            "it is 7",
            error,
        )
        assert _reflection_count(llm) == 2
        assert UNPARSED not in json.dumps(result.to_dict())

    @pytest.mark.asyncio
    async def test_utility_stops_with_the_unparsed_text(self) -> None:
        embedding = FakeEmbedding()
        llm = FakeLLM(
            [_echo_call("a"), _answer(PROSE), _answer(STILL_PROSE), _answer("42")]
        )
        agent = _agent(
            llm,
            embedding=embedding,
            steps_per_attempt=1,
            max_attempts=3,
            reflection_gate="utility",
        )
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            STILL_PROSE,
            UNPARSED,
        )
        assert _kinds(llm) == ["plan", "reflect", "reflect"]
        # The unparsed text is the reflection, and long-term memory gets it.
        assert not _never_embedded(embedding, STILL_PROSE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("options", "error"),
        [
            ({"max_attempts": 1}, "Max attempts reached"),
            ({"max_steps": 1}, "Max steps reached"),
        ],
        ids=["attempts", "steps"],
    )
    async def test_utility_ends_with_the_unparsed_text_at_the_final_boundary(
        self, options: dict[str, Any], error: str
    ) -> None:
        llm = FakeLLM(
            [_echo_call("a", content="it is 7"), _answer(PROSE), _answer(STILL_PROSE)]
        )
        agent = _agent(llm, steps_per_attempt=1, reflection_gate="utility", **options)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "budget_exhausted",
            STILL_PROSE,
            error,
        )

    @pytest.mark.asyncio
    async def test_one_info_line_names_the_attempt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="localmelo.melo.agent.agent")
        llm = FakeLLM(
            [
                _echo_call("a"),
                _answer(PROSE),
                _answer(STILL_PROSE),
                _echo_call("b"),
                _answer(PROSE),
                _answer(STILL_PROSE),
            ]
        )
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.error) == (
            "budget_exhausted",
            "Max attempts reached",
        )
        records = [r for r in caplog.records if "unreadable" in r.getMessage()]
        assert [(r.levelno, r.getMessage()) for r in records] == [
            (
                logging.INFO,
                "Reflection after attempt 1 unreadable after a retry, "
                "starting attempt 2",
            ),
            (
                logging.INFO,
                "Reflection after attempt 2 unreadable after a retry, "
                "ending the run: Max attempts reached",
            ),
        ]
