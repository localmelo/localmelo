"""How a run acts on a reflection: the gate, replies that hold no reflection,
and reflections over the budget — no real model.

The helpers at the top are shared by the other reflection tests."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
from collections import deque
from collections.abc import Iterator
from typing import Any, Protocol

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import (
    ANSWER_ON_STOP_NOTE,
    BLOCKED_ANSWER_NOTE,
    PREFIX_RUN,
    REFLECTION_RETRY_GUARD_NOTE,
    REFLECTION_RETRY_NOTE,
    Chat,
)
from localmelo.melo.agent.agent import (
    REFLECTION_TRIM_ORDER,
    _find_reflection,
    _fit_reflection,
    _gate_stop_reason,
    _serialize_reflections,
)
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import (
    MAX_REFLECTION_CHARS,
    Message,
    ReflectionDecision,
    ReflectionEntry,
    TaskRecord,
    ToolCall,
    ToolDef,
)

# The three checks of a reflection decision that the gate requires. The first two
# stop a run at once; a decision that is only not novel gets the patient gate's
# allowance once (test_gate_patient.py).
STOPPING_CHECKS = ["task_still_feasible", "next_step_is_concrete"]
CHECKS = [*STOPPING_CHECKS, "next_step_is_novel"]


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


class _Recording(Protocol):
    """A fake model that keeps every request it was sent."""

    calls: list[dict[str, Any]]


@contextlib.contextmanager
def _prefix_scope(agent: Agent) -> Iterator[None]:
    """The run-scoped date that a request outside a run needs (the fixed prefix)."""
    token = agent.chat.begin_prefix()
    try:
        yield
    finally:
        PREFIX_RUN.reset(token)


def _scoped_chat(llm: BaseLLMProvider, **options: Any) -> Chat:
    """A Chat that can send a request outside a run (see :func:`_prefix_scope`)."""
    chat = Chat(llm, **options)
    chat.begin_prefix()
    return chat


def _is_reflection(call: dict[str, Any]) -> bool:
    """A reflection request: its last message asks to reflect, or is the retry note.

    The request for an answer after a stopped run quotes the reflection request
    in its evidence, so it is left out by its note.
    """
    last: str = call["messages"][-1].content
    if last.endswith((ANSWER_ON_STOP_NOTE, BLOCKED_ANSWER_NOTE)):
        return False
    return "Reflect and decide." in last or last.startswith(
        (REFLECTION_RETRY_NOTE, REFLECTION_RETRY_GUARD_NOTE)
    )


def _planning_calls(llm: _Recording) -> list[dict[str, Any]]:
    return [call for call in llm.calls if not _is_reflection(call)]


def _reflection_calls(llm: _Recording) -> list[dict[str, Any]]:
    return [call for call in llm.calls if _is_reflection(call)]


def _kinds(llm: FakeLLM) -> list[str]:
    """Each request in order: ``plan`` or ``reflect``."""
    return ["reflect" if _is_reflection(call) else "plan" for call in llm.calls]


def _said(call: dict[str, Any]) -> str:
    """Everything a request says: its messages' contents in order."""
    return "\n".join(m.content for m in call["messages"])


def _planning_texts(llm: FakeLLM) -> list[str]:
    return [_said(call) for call in _planning_calls(llm)]


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


class TestGate:
    @pytest.mark.parametrize("action", ["stop", "decompose", "done"])
    def test_only_continue_continues(self, action: str) -> None:
        decision = _passing_decision(recommended_action=action)
        assert _gate_stop_reason(decision) == "action_not_continue"
        assert _gate_stop_reason(_passing_decision()) is None

    @pytest.mark.parametrize("check", CHECKS)
    def test_every_check_must_hold(self, check: str) -> None:
        assert _gate_stop_reason(_passing_decision(**{check: False})) is not None


class TestGateInTheLoop:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("check", STOPPING_CHECKS)
    async def test_a_continue_with_a_check_false_is_gate_stopped(
        self, check: str
    ) -> None:
        reply = _reflection(
            "continue",
            rationale="not this way",
            best_effort_result="7",
            **{check: False},
        )
        llm = FakeLLM([_echo_call("a"), reply])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "7",
            "not this way",
        )
        assert result.attempts == 1
        assert len(_reflection_calls(llm)) == 1


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


def _retry_continues_the_request(llm: _Recording) -> None:
    """The reflection request was asked again on its own prefix: the request, the
    reply that went unread as an assistant message, then the retry note."""
    first, retry = (call["messages"] for call in _reflection_calls(llm))
    assert retry[: len(first)] == first
    assert [m.role for m in retry[len(first) :]] == ["assistant", "user"]
    assert retry[-1].content.startswith(REFLECTION_RETRY_GUARD_NOTE)


class TestReadableReflection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first", [TOOL_CALL_TEXT, INVALID_ACTION], ids=["tool_call", "invalid"]
    )
    async def test_a_reply_without_a_valid_action_is_asked_again(
        self, first: str
    ) -> None:
        llm = FakeLLM([_echo_call("a"), _answer(first), _answer(DONE)])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result) == ("completed", "42")
        assert (result.n_main, result.n_aux) == (1, 2)
        _retry_continues_the_request(llm)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply", [TOOL_CALL_TEXT, INVALID_ACTION], ids=["tool_call", "invalid"]
    )
    async def test_the_same_reply_twice_is_unreadable(self, reply: str) -> None:
        llm = FakeLLM([_echo_call("a"), _answer(reply), _answer(reply), _answer("7")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.attempts) == ("completed", "7", 2)
        assert _kinds(llm) == ["plan", "reflect", "reflect", "plan"]

    @pytest.mark.asyncio
    async def test_a_reflection_after_a_tool_call_object_is_read(self) -> None:
        reply = f"{TOOL_CALL_TEXT}\n{DONE}"
        llm = FakeLLM([_echo_call("a"), _answer(reply)])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result) == ("completed", "42")
        assert len(_reflection_calls(llm)) == 1


# ── A reply that holds no reflection, also when asked again ────────────────

PROSE = "I would rather not answer in JSON."
STILL_PROSE = "Still no JSON, sorry."
UNPARSED = "Could not parse reflection response"


def _never_embedded(embedding: FakeEmbedding, *texts: str) -> bool:
    return not any(text in embedded for embedded in embedding.texts for text in texts)


class TestUnreadableReflection:
    @pytest.mark.asyncio
    async def test_an_unread_reply_starts_the_next_attempt_without_a_reflection(
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
        assert "[REFLECTION]" not in _planning_texts(llm)[-1]
        # Neither reply reached long-term memory or the run's result.
        assert embedding.texts
        assert _never_embedded(embedding, PROSE, STILL_PROSE)
        assert UNPARSED not in json.dumps(result.to_dict())

    @pytest.mark.asyncio
    async def test_an_unread_reply_keeps_the_previous_reflection(
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
            _planning_texts(llm)[-1]
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
    async def test_an_unread_reply_ends_with_the_latest_text_at_the_final_boundary(
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
        assert len(_reflection_calls(llm)) == 2
        assert UNPARSED not in json.dumps(result.to_dict())

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


# ── A reflection over the budget ────────────────────────────────────────────

# How a serialized reflection starts after the first attempt.
PREFIX = "[reflection] Attempt 0: "


def _size(entry: ReflectionEntry) -> int:
    return len(_serialize_reflections([entry])[0].content)


def _lists(entry: ReflectionEntry) -> list[list[str]]:
    return [getattr(entry, name) for name in REFLECTION_TRIM_ORDER]


class TestFitReflection:
    def test_the_trim_order(self) -> None:
        assert REFLECTION_TRIM_ORDER == (
            "failed_hypotheses",
            "useful_evidence",
            "recommended_avoids",
            "next_promising_directions",
        )

    def test_an_entry_that_fits_is_returned_as_it_is(self) -> None:
        entry = ReflectionEntry(summary="short", failed_hypotheses=["a"])
        assert _fit_reflection(entry, _size(entry)) is entry

    def test_items_go_from_the_end_of_each_list_in_turn(self) -> None:
        entry = ReflectionEntry(
            summary="tried both ways",
            failed_hypotheses=["f1", "f2"],
            useful_evidence=["e1", "e2"],
            recommended_avoids=["a1", "a2"],
            next_promising_directions=["d1", "d2"],
        )
        # Each limit is the size after one more item is dropped, so the
        # trimming stops there and drops nothing more.
        steps: list[tuple[list[str], list[str], list[str], list[str]]] = [
            (["f1"], ["e1", "e2"], ["a1", "a2"], ["d1", "d2"]),
            ([], ["e1", "e2"], ["a1", "a2"], ["d1", "d2"]),
            ([], ["e1"], ["a1", "a2"], ["d1", "d2"]),
            ([], [], ["a1", "a2"], ["d1", "d2"]),
            ([], [], ["a1"], ["d1", "d2"]),
            ([], [], [], ["d1", "d2"]),
            ([], [], [], ["d1"]),
            ([], [], [], []),
        ]
        for failed, evidence, avoids, directions in steps:
            step = dataclasses.replace(
                entry,
                failed_hypotheses=failed,
                useful_evidence=evidence,
                recommended_avoids=avoids,
                next_promising_directions=directions,
            )
            fitted = _fit_reflection(entry, _size(step))
            assert _lists(fitted) == [failed, evidence, avoids, directions]
            assert fitted.summary == entry.summary
        # The entry itself is not changed.
        assert _lists(entry) == [["f1", "f2"], ["e1", "e2"], ["a1", "a2"], ["d1", "d2"]]

    @pytest.mark.parametrize(
        ("room", "summary"),
        [(12, "the quick…"), (11, "the quick…"), (10, "the quick…"), (9, "the…")],
        ids=[
            "inside_a_word",
            "after_a_space",
            "after_a_word",
            "inside_the_second_word",
        ],
    )
    def test_then_the_summary_is_cut_at_a_word_boundary(
        self, room: int, summary: str
    ) -> None:
        entry = ReflectionEntry(
            summary="the quick brown fox",
            failed_hypotheses=["f"],
            next_promising_directions=["d"],
        )
        fitted = _fit_reflection(entry, len(PREFIX) + room)
        assert (fitted.summary, _lists(fitted)) == (summary, [[], [], [], []])
        assert _size(fitted) <= len(PREFIX) + room

    def test_a_summary_of_one_word_is_cut_where_the_limit_falls(self) -> None:
        entry = ReflectionEntry(summary="x" * 3_000)
        fitted = _fit_reflection(entry, MAX_REFLECTION_CHARS)
        assert fitted.summary == "x" * (MAX_REFLECTION_CHARS - len(PREFIX) - 1) + "…"
        assert _size(fitted) == MAX_REFLECTION_CHARS

    def test_the_fields_it_does_not_serialize_are_kept(self) -> None:
        entry = ReflectionEntry(
            attempt_id=2,
            summary="so far " * 500,
            disproven_actions=["echo z"],
            unresolved_questions=["why"],
            failure_type="stuck",
            confidence=0.4,
        )
        fitted = _fit_reflection(entry, MAX_REFLECTION_CHARS)
        assert fitted.summary.endswith("so far…")
        assert dataclasses.replace(fitted, summary=entry.summary) == entry


def _bulky(action: str, **fields: Any) -> dict[str, Any]:
    """A decision whose entry serializes to about 2,500 characters."""
    return _decision(
        action,
        summary="Tried echo on every input.",
        failed_hypotheses=[f"hypothesis {i}: " + "f" * 280 for i in range(5)],
        useful_evidence=[f"evidence {i}: " + "e" * 200 for i in range(4)],
        recommended_avoids=["echo a again", "echo b again"],
        next_promising_directions=["echo c", "echo d"],
        **fields,
    )


def _parsed_size(data: dict[str, Any]) -> int:
    """The serialized size of the reflection that *data* gives after attempt 0."""
    entry, _ = Agent._reflection_from_object(data, 0, "budget")
    return _size(entry)


class TestOverBudget:
    def test_the_bulky_reflection_is_about_2500_characters(self) -> None:
        assert 2_400 <= _parsed_size(_bulky("continue")) <= 2_600

    @pytest.mark.asyncio
    async def test_a_continue_is_trimmed_and_starts_the_next_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _bulky("continue")
        llm = FakeLLM([_echo_call("a"), _answer(json.dumps(data)), _answer("42")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3)
        added = _added_reflections(agent, monkeypatch)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.attempts) == ("completed", "42", 2)
        [stored] = added
        assert _size(stored) <= MAX_REFLECTION_CHARS
        # Hypotheses from the end made the room; the rest is whole.
        kept = len(stored.failed_hypotheses)
        assert 0 < kept < len(data["failed_hypotheses"])
        assert stored.failed_hypotheses == data["failed_hypotheses"][:kept]
        assert stored.summary == data["summary"]
        for name in REFLECTION_TRIM_ORDER[1:]:
            assert getattr(stored, name) == data[name]
        # The next attempt plans with the trimmed reflection.
        assert "\n  Try: echo c; echo d" in _planning_texts(llm)[-1]

    @pytest.mark.asyncio
    async def test_the_decision_points_at_the_trimmed_entry(self) -> None:
        data = _bulky("continue", rationale="run the tests")
        agent = _agent(FakeLLM([_answer(json.dumps(data))]))
        with _prefix_scope(agent):
            reflection = await agent._do_reflect(
                TaskRecord(query="echo the answer"), 0, "budget"
            )
        assert reflection is not None
        entry, decision = reflection
        assert _size(entry) <= MAX_REFLECTION_CHARS < _parsed_size(data)
        assert decision.tried_memory is entry
        # The decision is the model's own, rationale and all.
        assert (decision.recommended_action, decision.rationale) == (
            "continue",
            "run the tests",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["stop", "decompose"])
    async def test_a_trimmed_reflection_keeps_its_rationale(self, action: str) -> None:
        data = _bulky(action, rationale="nope", best_effort_result="7")
        assert _parsed_size(data) > MAX_REFLECTION_CHARS
        llm = FakeLLM([_echo_call("a"), _answer(json.dumps(data))])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3)
        result = await agent.run_task("echo the answer")
        # The gate stops the run with the model's rationale, and no budget
        # note is added to it.
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "7",
            "nope",
        )
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_a_done_is_trimmed_and_keeps_its_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embedding = FakeEmbedding()
        data = _bulky("done", best_effort_result="42")
        llm = FakeLLM([_echo_call("a"), _answer(json.dumps(data))])
        agent = _agent(llm, embedding=embedding, steps_per_attempt=1, max_attempts=3)
        added = _added_reflections(agent, monkeypatch)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == ("completed", "42", "")
        [stored] = added
        assert _size(stored) <= MAX_REFLECTION_CHARS
        # Long-term memory receives the trimmed copy, never what it dropped.
        dropped = data["failed_hypotheses"][len(stored.failed_hypotheses) :]
        assert dropped
        assert any(
            t.startswith(f"Attempt 0: {data['summary']}") for t in embedding.texts
        )
        assert _never_embedded(embedding, *dropped)
