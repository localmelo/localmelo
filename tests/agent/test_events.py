"""Run events for an optional sink (Agent.run_task's on_event) — no real model.

Every run is scripted. Most use the custom tools below, offered on every
step (``tool_exposure="all"``), and in-memory history.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import sqlite3
import sys
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions, _events
from localmelo.melo.agent._chat import RETRY_HOOK
from localmelo.melo.contracts.approvals import (
    AllowAllPolicy,
    ApprovalDecision,
    ApprovalRequest,
)
from localmelo.melo.contracts.events import EVENT_FIELDS, AgentEvent
from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import TOOL_EXIT_CODE, ToolOutput
from localmelo.melo.schema import (
    CALL_INTERRUPTED,
    ImagePart,
    Message,
    RunResult,
    ToolCall,
    ToolDef,
)
from localmelo.tests.legacy_options import LEGACY_ANSWER_OPTIONS

# ── Fakes ───────────────────────────────────────────────────────────────────


@dataclass
class Slow:
    """A scripted reply that takes *delay_s* seconds to arrive."""

    reply: Message
    delay_s: float


Reply = Message | BaseException | Slow


class ScriptedLLM(BaseLLMProvider):
    """Answers each request with the next scripted reply; records each request.

    A scripted exception is raised instead. ``spans`` holds the
    ``perf_counter()`` start and end of each request.
    """

    def __init__(self, replies: Iterable[Reply]) -> None:
        self.replies: deque[Reply] = deque(replies)
        self.requests: list[dict[str, Any]] = []
        self.spans: list[tuple[float, float]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        started = time.perf_counter()
        self.requests.append(_request_json(messages, tools))
        if not self.replies:
            raise AssertionError("no reply is scripted for this request")
        reply = self.replies.popleft()
        try:
            if isinstance(reply, Slow):
                await asyncio.sleep(reply.delay_s)
                reply = reply.reply
            if isinstance(reply, BaseException):
                raise reply
            return reply
        finally:
            self.spans.append((started, time.perf_counter()))

    async def close(self) -> None:
        pass


class SameVectorEmbedding(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


class Recorder:
    """A sink that keeps every event; with *error*, it then raises it."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.events: list[AgentEvent] = []
        self.error = error

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)
        if self.error is not None:
            raise self.error

    @property
    def types(self) -> list[str]:
        return [event.type for event in self.events]

    def of(self, type_: str) -> list[AgentEvent]:
        return [event for event in self.events if event.type == type_]

    def notices(self) -> list[str]:
        return [event.data["kind"] for event in self.of("notice")]


async def _no_wait(delay: float) -> None:
    pass


def _one_string(name: str) -> dict[str, Any]:
    return {"type": "object", "properties": {name: {"type": "string"}}}


ECHO = ToolDef(
    name="echo", description="Echo text back", parameters=_one_string("text")
)
PROBE = ToolDef(name="probe", description="Take any arguments", parameters={})
PHOTO = ToolDef(name="photo", description="Take a photo", parameters={})
HANG = ToolDef(name="hang", description="Never return", parameters={})

PNG = ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n a small photo", "image/png")


@dataclass
class Tools:
    """The custom tools, recording what they received and saw."""

    received: list[dict[str, Any]] = field(default_factory=list)
    # TOOL_EXIT_CODE and RETRY_HOOK as each call saw them.
    slots: list[Any] = field(default_factory=list)
    hooks: list[Any] = field(default_factory=list)

    async def echo(self, text: str = "") -> str:
        self._saw({"text": text})
        return text

    async def probe(self, **arguments: Any) -> str:
        self._saw(arguments)
        return "ok"

    async def photo(self) -> ToolOutput:
        self._saw({})
        return ToolOutput(text="a photo", images=[PNG])

    async def hang(self) -> str:
        self._saw({})
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    def _saw(self, arguments: dict[str, Any]) -> None:
        self.received.append(arguments)
        self.slots.append(TOOL_EXIT_CODE.get())
        self.hooks.append(RETRY_HOOK.get())


class WeatherTools(BaseToolProvider):
    """Provides ``weather``; with *error*, start() raises it."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def start(self) -> list[ProvidedTool]:
        if self.error is not None:
            raise self.error
        return [
            ProvidedTool(
                definition=ToolDef(
                    name="weather",
                    description="Forecast for a city",
                    parameters=_one_string("city"),
                ),
                function=self._weather,
            )
        ]

    async def _weather(self, city: str = "") -> str:
        return f"Sunny in {city}"

    async def close(self) -> None:
        pass


def _agent(
    llm: BaseLLMProvider,
    *,
    tools: Tools | None = None,
    embedding: BaseEmbeddingProvider | None = None,
    providers: Sequence[BaseToolProvider] = (),
    **options: Any,
) -> Agent:
    """An agent with the custom tools, offered on every step, and no waits."""
    options.setdefault("builtins", frozenset())
    options.setdefault("tool_exposure", "all")
    for name, value in LEGACY_ANSWER_OPTIONS.items():
        options.setdefault(name, value)
    agent = Agent(
        llm=llm,
        embedding=embedding,
        options=AgentOptions(**options),
        tool_providers=providers,
    )
    tools = tools or Tools()
    for definition, function in (
        (ECHO, tools.echo),
        (PROBE, tools.probe),
        (PHOTO, tools.photo),
        (HANG, tools.hang),
    ):
        agent.hippo.register_tool(definition)
        agent.executor.register(definition.name, function)
    agent.chat.sleep = _no_wait
    return agent


# ── Replies ─────────────────────────────────────────────────────────────────


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _answer(
    text: str, *, thinking: str = "", usage: dict[str, int] | None = None
) -> Message:
    return Message(role="assistant", content=text, thinking=thinking, usage=usage)


def _calls(
    *calls: tuple[str, dict[str, Any], str],
    content: str = "",
    thinking: str = "",
    usage: dict[str, int] | None = None,
) -> Message:
    """A reply with tool calls, each ``(name, arguments, id)``."""
    return Message(
        role="assistant",
        content=content,
        thinking=thinking,
        usage=usage,
        tool_calls=[ToolCall(tool_name=n, arguments=a, id=i) for n, a, i in calls],
    )


def _echo(text: str, call_id: str = "call_1", **reply: Any) -> Message:
    return _calls(("echo", {"text": text}, call_id), **reply)


ALL_FLAGS = {
    "task_still_feasible": True,
    "next_step_is_concrete": True,
    "next_step_is_novel": True,
}


def _decision(action: str, **fields: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "summary": "Echoed once.",
        "failed_hypotheses": [],
        "useful_evidence": [],
        "recommended_avoids": [],
        "next_promising_directions": ["echo again"],
        "progress_made": True,
        "task_still_feasible": True,
        "new_information_gained": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "recommended_action": action,
        "rationale": "",
        "best_effort_result": "",
        "estimated_info_gain": 0.8,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.8,
        "feasibility": 0.8,
    }
    data.update(fields)
    return data


def _reflection(
    action: str, *, usage: dict[str, int] | None = None, **fields: Any
) -> Message:
    return _answer(json.dumps(_decision(action, **fields)), usage=usage)


PROSE = "I would rather say it in words."


def _busy() -> ProviderError:
    return ProviderError("HTTP 503: busy", status_code=503, retryable=True)


def _refused() -> ProviderError:
    return ProviderError("HTTP 400: refused", status_code=400, retryable=False)


# ── Checks ──────────────────────────────────────────────────────────────────


def _event_usage(events: Iterable[AgentEvent]) -> dict[str, int]:
    """The usage of the assistant.message and reflection events, summed."""
    total: dict[str, int] = {}
    for event in events:
        if event.type in ("assistant.message", "reflection"):
            for name, value in event.data["usage"].items():
                total[name] = total.get(name, 0) + value
    return total


async def check_guarantees(
    recorder: Recorder,
    result: RunResult,
    agent: Agent,
    *,
    usage_matches: bool = True,
    usage_extra: frozenset[str] = frozenset(),
) -> None:
    """E-1, E-2, E-4 and E-5, the pinned keys, and JSON for every event."""
    events = recorder.events
    types = recorder.types
    # E-1: run.started first, run.finished last, each once.
    assert types[0] == "run.started" and types[-1] == "run.finished"
    assert types.count("run.started") == types.count("run.finished") == 1
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    stamps = [event.ts for event in events]
    assert stamps == sorted(stamps)
    for event in events:
        assert set(event.data) == EVENT_FIELDS[event.type], event.type
        envelope = event.to_dict()
        assert envelope["v"] == 1
        json.dumps(envelope, allow_nan=False)
    # E-2: each tool.started is closed by one tool.finished with its id
    # before the next tool.started, step.started or run.finished.
    open_call: str | None = None
    for event in events:
        if event.type == "tool.started":
            assert open_call is None
            open_call = event.data["call_id"]
        elif event.type == "tool.finished":
            assert event.data["call_id"] == open_call
            open_call = None
        elif event.type in ("step.started", "run.finished"):
            assert open_call is None
    # E-4: the counts and the result.
    assert types.count("step.started") == result.steps
    assert types.count("tool.started") == result.n_tool
    assert events[-1].data["result"] == result.to_dict()
    if usage_matches:
        # *usage_extra* names keys the run result adds beside the replies'
        # usage (an explicit mode's reasoning_chars), not a counter to skip.
        expected = {
            key: value for key, value in result.usage.items() if key not in usage_extra
        }
        assert _event_usage(events) == expected
    # E-5: one run id, the task's.
    assert {event.run_id for event in events} == {result.task_id}
    assert await agent.hippo.history.get_task(result.task_id) is not None


# ── The contract of on_event ────────────────────────────────────────────────


class TestSequence:
    @pytest.mark.asyncio
    async def test_event_sequence_for_a_tool_run(self) -> None:
        llm = ScriptedLLM([_echo("hi", usage=_usage(10, 2)), _answer("hi")])
        recorder = Recorder()
        agent = _agent(llm)
        result = await agent.run_task("echo hi", on_event=recorder)
        assert result.status == "completed"
        assert recorder.types == [
            "run.started",
            "step.started",
            "assistant.message",
            "tool.started",
            "tool.finished",
            "step.started",
            "assistant.message",
            "run.finished",
        ]
        assert [event.seq for event in recorder.events] == list(range(1, 9))
        assert {event.to_dict()["v"] for event in recorder.events} == {1}
        assert {event.run_id for event in recorder.events} == {result.task_id}
        stamps = [event.ts for event in recorder.events]
        assert stamps == sorted(stamps)
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_the_data_of_a_tool_run(self) -> None:
        llm = ScriptedLLM(
            [
                _echo(
                    "hi",
                    content="Echoing.",
                    thinking="The user wants an echo.",
                    usage=_usage(10, 2),
                ),
                _answer("hi", usage={"total_tokens": 5, "cached": True}),
            ]
        )
        recorder = Recorder()
        result = await _agent(llm, max_steps=7).run_task("echo hi", on_event=recorder)
        started, step, message, tool, finished, _, answer, done = recorder.events
        assert started.data == {
            "query": "echo hi",
            "max_steps": 7,
            "workspace_root": None,
            "approval": False,
        }
        assert step.data == {"step": 1, "attempt": 1}
        assert message.data["step"] == 1
        assert message.data["content"] == "Echoing."
        assert message.data["thinking"] == "The user wants an echo."
        assert message.data["tool_calls"] == [
            {
                "id": "call_1",
                "name": "echo",
                "arguments": {"text": "hi"},
                "preview": '{"text":"hi"}',
            }
        ]
        assert message.data["usage"] == _usage(10, 2)
        assert message.data["duration_s"] >= 0
        assert tool.data == {
            "step": 1,
            "call_id": "call_1",
            "name": "echo",
            "arguments": {"text": "hi"},
            "preview": '{"text":"hi"}',
        }
        assert {k: v for k, v in finished.data.items() if k != "duration_s"} == {
            "step": 1,
            "call_id": "call_1",
            "name": "echo",
            "status": "success",
            "error_category": "none",
            "output": "hi",
            "output_chars": 2,
            "error": "",
            "images": [],
            "approval": None,
            "exit_code": None,
        }
        assert finished.data["duration_s"] >= 0
        # Only the integer fields of usage.
        assert answer.data["usage"] == {"total_tokens": 5}
        assert (answer.data["step"], answer.data["tool_calls"]) == (2, [])
        assert done.data == {"result": result.to_dict()}
        assert RunResult(**done.data["result"]) == result

    @pytest.mark.asyncio
    async def test_a_refused_call_has_its_pair(self) -> None:
        llm = ScriptedLLM(
            [
                _calls(("nope", {}, "call_1"), ("echo", {"text": "a"}, "call_2")),
                _answer("ok"),
            ]
        )
        recorder = Recorder()
        agent = _agent(llm)
        result = await agent.run_task("echo", on_event=recorder)
        finished = recorder.of("tool.finished")
        assert [(e.data["status"], e.data["error_category"]) for e in finished] == [
            ("error", "tool_not_found"),
            ("success", "none"),
        ]
        assert finished[0].data["duration_s"] == 0
        assert finished[0].data["error"] == "Tool not found in registry: nope"
        await check_guarantees(recorder, result, agent)


class TestFields:
    def test_event_fields_are_pinned(self) -> None:
        assert {
            "run.started": {"query", "max_steps", "workspace_root", "approval"},
            "inbox.accepted": {"offset", "text", "step", "attempt"},
            "step.started": {"step", "attempt"},
            "assistant.message": {
                "step",
                "content",
                "thinking",
                "tool_calls",
                "usage",
                "prefix",
                "duration_s",
            },
            "tool.started": {"step", "call_id", "name", "arguments", "preview"},
            "tool.finished": {
                "step",
                "call_id",
                "name",
                "status",
                "error_category",
                "output",
                "output_chars",
                "error",
                "duration_s",
                "images",
                "approval",
                "exit_code",
            },
            "memory": {
                "candidate_id",
                "action",
                "kind",
                "admitted_by",
                "entry_id",
            },
            "note": {
                "step",
                "attempt",
                "note_id",
                "kind",
                "source",
                "text",
                "chars",
                "drafts",
                "remember",
                "stage",
            },
            "recall": {
                "step",
                "attempt",
                "trigger",
                "retriever",
                "query",
                "hits",
                "chars",
                "ms",
                "fallback",
                "placement",
            },
            "notice": {"kind", "message", "step", "attempt", "detail"},
            "reflection.started": {"attempt", "failure_type"},
            "reflection": {
                "attempt",
                "action",
                "continue_run",
                "flags",
                "gate",
                "utility",
                "stop_reason",
                "rationale",
                "summary",
                "best_effort_result",
                "requests",
                "readable",
                "unreadable_reason",
                "repaired",
                "usage",
                "prefix",
                "duration_s",
            },
            "run.progress": {
                "level",
                "state",
                "step",
                "attempt",
                "limits",
                "phase",
                "cycle",
                "tests",
                "items",
                "fraction",
                "expected_steps",
                "past",
                "off_reason",
                "basis",
            },
            "context_budget": {
                "kind",
                "model",
                "attempt",
                "proxy_version",
                "source",
                "estimate",
                "measured_prompt_tokens",
                "input_limit",
                "over_limit",
                "image_allowance",
                "pruned",
            },
            "run.finished": {"result"},
        } == {name: set(keys) for name, keys in EVENT_FIELDS.items()}

    @pytest.mark.asyncio
    async def test_runs_emit_every_type_with_its_keys(self) -> None:
        # A zero-backoff retry, an empty reply, a used-up attempt and its
        # reflection, then an answer.
        reflect = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _busy(),
                    _echo("a"),
                    _answer(""),
                    _reflection("continue"),
                    _answer("done"),
                ]
            ),
            steps_per_attempt=2,
        )
        agent.chat.retry_delays = (0.0, 0.0)
        result = await agent.run_task("echo a", on_event=reflect, progress=True)
        assert result.status == "completed"
        await check_guarantees(reflect, result, agent)
        assert reflect.notices() == ["retry", "empty_reply", "budget"]
        # Stuck, with one attempt.
        stuck = Recorder()
        agent = _agent(
            ScriptedLLM([_echo("a", f"call_{i}") for i in range(3)]),
            attempt_policy="none",
        )
        result = await agent.run_task("echo a", on_event=stuck)
        assert result.status == "stuck"
        await check_guarantees(stuck, result, agent)
        assert stuck.notices() == ["stuck"]
        # R4's compress mode emits the note event, for the notes and the
        # attempt digest.
        compress = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a", content="<note>plan: echo a</note>"),
                    _answer("The test passed because the fix works."),
                ]
            ),
            reasoning="compress",
        )
        result = await agent.run_task("echo a", on_event=compress)
        assert result.status == "completed"
        await check_guarantees(
            compress,
            result,
            agent,
            usage_extra=frozenset(
                {"reasoning_chars", "reasoning_tokens", "cached_tokens"}
            ),
        )
        assert {event.data["kind"] for event in compress.of("note")} == {
            "step",
            "answer",
            "digest",
        }

        # R5's recall event, from a step that has a stored note to recall.
        from localmelo.melo.memory.notes import ReasoningNote
        from localmelo.melo.memory.short.note_store import NoteStoreEntry

        recalled = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a", content="<note>plan: the answer is 42</note>"),
                    _answer("done"),
                ]
            ),
            reasoning="compress",
            retriever="bm25",
            attempt_policy="none",
        )
        agent.load_notes(
            [
                NoteStoreEntry(
                    alias="n1",
                    kind="note",
                    text="plan: the answer is 42 and tests pass",
                    note=ReasoningNote(note_id="x", plan="the answer is 42"),
                    task_id="",
                )
            ]
        )
        result = await agent.run_task("echo a", on_event=recalled)
        assert result.status == "completed"
        assert recalled.of("recall")
        await agent.close()

        class OneInstruction:
            def validate(self, workspace: str | None = None) -> None:
                pass

            def poll(self) -> list[tuple[int, str]]:
                return [(6, "hello")]

            def accept(self, offset: int) -> None:
                assert offset == 6

        inbox_events = Recorder()
        agent = _agent(ScriptedLLM([_answer("done")]))
        result = await agent.run_task(
            "task", observe=inbox_events, inbox=OneInstruction()
        )
        assert result.status == "completed"
        assert inbox_events.of("inbox.accepted")
        await agent.close()
        events = (
            reflect.events
            + stuck.events
            + compress.events
            + recalled.events
            + inbox_events.events
        )
        seen = {event.type for event in events}
        # The memory event is emitted only by a store-backed admission;
        # test_memory_admission.py emits and pins it.
        # Budget events need an opted-in run; test_request_accounting pins one.
        assert seen == set(EVENT_FIELDS) - {"memory", "context_budget"}
        for event in events:
            assert set(event.data) == EVENT_FIELDS[event.type]
        details = {
            event.data["kind"]: set(event.data["detail"])
            for event in reflect.of("notice") + stuck.of("notice")
        }
        assert details == {
            "retry": {"request_attempt", "delay_s", "status_code"},
            "empty_reply": {"count"},
            "budget": set(),
            "stuck": set(),
        }

    @pytest.mark.asyncio
    async def test_notices(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [_answer(""), _echo("a"), _reflection("done", best_effort_result="a")]
            ),
            steps_per_attempt=2,
        )
        await agent.run_task("echo a", on_event=recorder)
        empty, budget = recorder.of("notice")
        assert empty.data == {
            "kind": "empty_reply",
            "message": "The model returned an empty reply; asking it to continue.",
            "step": 1,
            "attempt": 1,
            "detail": {"count": 1},
        }
        assert budget.data == {
            "kind": "budget",
            "message": "The attempt used its step budget.",
            "step": 2,
            "attempt": 1,
            "detail": {},
        }
        stuck = Recorder()
        agent = _agent(
            ScriptedLLM([_echo("a", f"call_{i}") for i in range(3)]),
            attempt_policy="none",
        )
        await agent.run_task("echo a", on_event=stuck)
        assert stuck.of("notice")[0].data == {
            "kind": "stuck",
            "message": "The same tool calls or errors repeated; the attempt ends.",
            "step": 3,
            "attempt": 1,
            "detail": {},
        }
        assert stuck.types[-2:] == ["notice", "run.finished"]


class TestCounts:
    @pytest.mark.asyncio
    async def test_counts_match_run_result(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _calls(
                        ("echo", {"text": "a"}, "call_1"),
                        ("nope", {}, "call_2"),
                        usage=_usage(10, 3),
                    ),
                    _answer("a", usage=_usage(20, 1)),
                ]
            )
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.steps, result.n_tool) == (2, 2)
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_counts_when_timeout_s_cuts_a_tool_short(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_calls(("hang", {}, "call_1"), usage=_usage(10, 3))]),
            timeout_s=0.2,
        )
        result = await agent.run_task("wait", on_event=recorder)
        assert (result.status, result.steps, result.n_tool) == ("timeout", 1, 1)
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_usage_of_a_reflection_asked_again(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a", usage=_usage(10, 3)),
                    _echo("b", "call_2", usage=_usage(30, 4)),
                    _reflection("done", best_effort_result="a", usage=_usage(50, 6)),
                ]
            ),
            steps_per_attempt=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.status, result.n_llm) == ("completed", 3)
        [reflection] = recorder.of("reflection")
        assert reflection.data["usage"] == _usage(80, 10)
        await check_guarantees(recorder, result, agent)


class TestPayloads:
    @pytest.mark.asyncio
    async def test_events_are_json_serializable(self) -> None:
        thing = object()
        tools = Tools()
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _calls(("probe", {"x": math.nan, "thing": thing}, "call_1")),
                    _answer("done"),
                ]
            ),
            tools=tools,
        )
        result = await agent.run_task("probe", on_event=recorder)
        assert result.status == "completed"
        for event in recorder.events:
            json.dumps(event.to_dict(), allow_nan=False)
        [message] = recorder.of("assistant.message")[:1]
        assert message.data["tool_calls"][0]["arguments"] == {
            "x": "nan",
            "thing": str(thing),
        }
        [started] = recorder.of("tool.started")
        assert started.data["arguments"] == {"x": "nan", "thing": str(thing)}
        # The tool gets the arguments themselves.
        assert tools.received[0]["thing"] is thing
        assert math.isnan(tools.received[0]["x"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("marked", [False, True])
    async def test_arguments_nested_past_the_recursion_limit(
        self, marked: bool
    ) -> None:
        # Arguments as a provider decodes them from a model's reply (json.loads
        # in parse_tool_arguments), nested past the recursion limit.
        depth = sys.getrecursionlimit() + 500
        try:
            arguments = json.loads('{"x": ' + "[" * depth + "]" * depth + "}")
        except RecursionError:
            pytest.skip("this Python's json cannot decode arguments this deep")
        tools = Tools()
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_calls(("probe", arguments, "call_1")), _answer("done")]),
            tools=tools,
        )
        if marked:
            agent.executor.register("probe", tools.probe, requires_approval=True)
        result = await agent.run_task(
            "probe", on_event=recorder, approval=AllowAllPolicy() if marked else None
        )
        assert (result.status, result.steps, result.n_tool) == ("completed", 2, 1)
        assert recorder.types == [
            "run.started",
            "step.started",
            "assistant.message",
            "tool.started",
            "tool.finished",
            "step.started",
            "assistant.message",
            "run.finished",
        ]
        await check_guarantees(recorder, result, agent)
        # The arguments dict is on level 1 of tool.started's data, so 99 of
        # x's lists are kept, and the one on level 101 becomes its text.
        [started] = recorder.of("tool.started")
        value = started.data["arguments"]["x"]
        for _ in range(99):
            assert isinstance(value, list) and len(value) == 1
            value = value[0]
        assert isinstance(value, str)
        [finished] = recorder.of("tool.finished")
        if marked:
            # The gate cannot copy the arguments for the policy, so the call
            # fails closed: it is refused, and its events still pair up.
            assert (finished.data["status"], tools.received) == ("blocked", [])
        else:
            assert finished.data["status"] == "success"
            assert tools.received[0]["x"] is arguments["x"]

    @pytest.mark.asyncio
    async def test_images_are_references(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_calls(("photo", {}, "call_1")), _answer("ok")]))
        result = await agent.run_task("take a photo", on_event=recorder)
        [finished] = recorder.of("tool.finished")
        assert finished.data["images"] == [
            {
                "sha256": PNG.sha256,
                "mime_type": "image/png",
                "size_bytes": PNG.size_bytes,
            }
        ]
        # The full image is on the event itself, for in-process consumers,
        # and nowhere in its JSON.
        assert finished.images == (PNG,)
        assert finished.images[0].data == PNG.data
        envelope = finished.to_dict()
        assert "images" not in envelope
        assert PNG.data not in json.dumps(envelope)
        # Working memory and history keep references only (#27).
        window = agent.hippo.working.get_window()
        assert [image.data for m in window for image in m.images] == [""]
        steps = await agent.hippo.history.get_steps(result.task_id)
        assert steps[0].tool_result is not None
        assert [image.data for image in steps[0].tool_result.images] == [""]

    @pytest.mark.asyncio
    async def test_long_text_is_clipped(self) -> None:
        long = "x" * 150_000
        clipped = "x" * 100_000 + "… [+50000 chars]"
        answer = "y" * 150_000
        tools = Tools()
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_echo(long, content=long), _answer(answer)]), tools=tools
        )
        result = await agent.run_task("echo it", on_event=recorder)
        assert result.status == "completed"
        first, last = recorder.of("assistant.message")
        assert first.data["content"] == clipped
        assert first.data["tool_calls"][0]["arguments"] == {"text": clipped}
        [started] = recorder.of("tool.started")
        assert started.data["arguments"] == {"text": clipped}
        # The executor and history get the argument whole; the window holds
        # a placeholder in its place.
        assert tools.received == [{"text": long}]
        steps = await agent.hippo.history.get_steps(result.task_id)
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.arguments == {"text": long}
        [call] = [m for m in agent.hippo.working.get_window() if m.tool_calls]
        size = len(json.dumps({"text": long}, ensure_ascii=False, sort_keys=True))
        assert call.tool_calls[0].arguments == {"_elided": f"{size} chars withheld"}
        # The output keeps its whole length in output_chars.
        [finished] = recorder.of("tool.finished")
        assert finished.data["output_chars"] == len(finished.data["output"])
        # run.finished carries the RunResult whole, a long answer included.
        assert last.data["content"] == "y" * 100_000 + "… [+50000 chars]"
        assert recorder.events[-1].data["result"]["result"] == answer
        assert recorder.events[-1].data["result"] == result.to_dict()

    @pytest.mark.asyncio
    async def test_a_long_query_is_clipped(self) -> None:
        query = "q" * 120_000
        recorder = Recorder()
        result = await _agent(ScriptedLLM([])).run_task(query, on_event=recorder)
        assert result.status == "failed"  # over the planning prompt's cap
        assert recorder.types == ["run.started", "run.finished"]
        assert recorder.events[0].data["query"] == "q" * 100_000 + "… [+20000 chars]"


class TestSinks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error", [RuntimeError("sink broke"), asyncio.CancelledError()]
    )
    async def test_raising_sink_does_not_change_the_run(
        self, error: BaseException, caplog: pytest.LogCaptureFixture
    ) -> None:
        def script() -> list[Reply]:
            return [_busy(), _echo("a", usage=_usage(5, 1)), _answer("a")]

        plain_llm = ScriptedLLM(script())
        plain = await _agent(plain_llm).run_task("echo a")
        recorder = Recorder(error)
        llm = ScriptedLLM(script())
        caplog.set_level(logging.DEBUG, logger="localmelo.melo.agent._events")
        result = await _agent(llm).run_task("echo a", on_event=recorder)
        assert _comparable(result) == _comparable(plain)
        assert llm.requests == plain_llm.requests
        assert [event.seq for event in recorder.events] == list(
            range(1, len(recorder.events) + 1)
        )
        assert recorder.types[-1] == "run.finished"
        records = [
            r for r in caplog.records if r.name == "localmelo.melo.agent._events"
        ]
        assert [r.levelno for r in records].count(logging.WARNING) == 1
        assert records[0].levelno == logging.WARNING and records[0].exc_info
        assert all(r.levelno == logging.DEBUG for r in records[1:])
        assert len(records) == len(recorder.events)

    @pytest.mark.asyncio
    async def test_a_returned_awaitable_is_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        made: list[Any] = []

        async def later(event: AgentEvent) -> None:
            made.append(event)

        def sink(event: AgentEvent) -> Any:
            coroutine = later(event)
            made.append(coroutine)
            return coroutine

        caplog.set_level(logging.WARNING, logger="localmelo.melo.agent._events")
        result = await _agent(ScriptedLLM([_answer("hi")])).run_task(
            "hi", on_event=sink
        )
        assert result.status == "completed"
        # Never awaited, and closed: no event reached later().
        assert len(made) == 4
        assert all(inspect.getcoroutinestate(item) == "CORO_CLOSED" for item in made)
        messages = [r.getMessage() for r in caplog.records]
        assert messages == [
            "on_event returned an awaitable; async sinks are not supported "
            "and the event was dropped"
        ]

    @pytest.mark.asyncio
    async def test_async_and_non_callable_sinks_are_rejected(self) -> None:
        async def async_sink(event: AgentEvent) -> None:
            pass

        class AsyncCall:
            async def __call__(self, event: AgentEvent) -> None:
                pass

        synchronous = (
            "on_event must be synchronous: put events on a queue and consume them "
            "in a task"
        )
        for sink, message in [
            (42, "on_event must be a callable that takes an AgentEvent"),
            (async_sink, synchronous),
            (AsyncCall(), synchronous),
        ]:
            llm = ScriptedLLM([_answer("hi")])
            agent = _agent(llm)
            with pytest.raises(TypeError) as caught:
                await agent.run_task("hi", on_event=sink)  # type: ignore[arg-type]
            assert str(caught.value) == message
            assert llm.requests == []
            assert agent.hippo.history._tasks == {}  # type: ignore[attr-defined]
            assert agent.hippo.working.get_window() == []

    @pytest.mark.asyncio
    async def test_a_policy_without_async_decide_is_rejected(self) -> None:
        class SyncDecide:
            def decide(self, request: ApprovalRequest) -> ApprovalDecision:
                return ApprovalDecision(True)

        for policy in (object(), SyncDecide()):
            recorder = Recorder()
            llm = ScriptedLLM([_answer("hi")])
            agent = _agent(llm)
            with pytest.raises(TypeError) as caught:
                await agent.run_task("hi", on_event=recorder, approval=policy)  # type: ignore[arg-type]
            assert str(caught.value) == (
                "approval must have an async decide(request) method"
            )
            assert (recorder.events, llm.requests) == ([], [])
            assert agent.hippo.history._tasks == {}  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_run_started_reports_a_policy(self) -> None:
        class AllowAll:
            async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
                return ApprovalDecision(True)

        recorder = Recorder()
        await _agent(ScriptedLLM([_answer("hi")])).run_task(
            "hi", on_event=recorder, approval=AllowAll()
        )
        assert recorder.events[0].data["approval"] is True

    @pytest.mark.asyncio
    async def test_the_hook_and_the_slot_belong_to_a_run_with_a_sink(self) -> None:
        tools = Tools()
        agent = _agent(
            ScriptedLLM([_echo("a"), _answer("a"), _echo("b"), _answer("b")]),
            tools=tools,
        )
        await agent.run_task("echo a")
        await agent.run_task("echo b", on_event=Recorder())
        assert tools.slots == [None, []]
        assert tools.hooks[0] is None and tools.hooks[1] is not None
        assert RETRY_HOOK.get() is None
        assert TOOL_EXIT_CODE.get() is None


class TestEveryStatus:
    """test_every_status_finishes: E-1 to E-5 for each status."""

    @pytest.mark.asyncio
    async def test_completed_through_a_reflection_done(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a", usage=_usage(10, 1)),
                    _reflection("done", best_effort_result="a", usage=_usage(40, 9)),
                ]
            ),
            steps_per_attempt=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.status, result.result) == ("completed", "a")
        assert recorder.types == [
            "run.started",
            "step.started",
            "assistant.message",
            "tool.started",
            "tool.finished",
            "notice",
            "reflection.started",
            "reflection",
            "run.finished",
        ]
        reflection = recorder.of("reflection")[0].data
        assert (reflection["action"], reflection["continue_run"]) == ("done", False)
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_failed_by_pre_plan(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([]))
        result = await agent.run_task("q" * 120_000, on_event=recorder)
        assert result.status == "failed"
        assert recorder.types == ["run.started", "run.finished"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_failed_by_post_plan(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_calls(("", {}, "call_1"), usage=_usage(5, 1))]))
        result = await agent.run_task("echo", on_event=recorder)
        assert (result.status, result.error) == (
            "failed",
            "Response check failed: Empty tool name in response",
        )
        assert recorder.types == [
            "run.started",
            "step.started",
            "assistant.message",
            "run.finished",
        ]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_failed_after_two_empty_replies(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_answer(""), _answer("", thinking="hmm")]))
        result = await agent.run_task("say it", on_event=recorder)
        assert (result.status, result.error) == ("failed", "empty replies")
        assert recorder.types == [
            "run.started",
            "step.started",
            "assistant.message",
            "notice",
            "step.started",
            "assistant.message",
            "run.finished",
        ]
        assert recorder.notices() == ["empty_reply"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_gate_stopped_by_the_gate(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a"),
                    _reflection(
                        "continue", next_step_is_novel=False, rationale="Nothing new."
                    ),
                ]
            ),
            steps_per_attempt=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.status, result.error) == ("gate_stopped", "Nothing new.")
        reflection = recorder.of("reflection")[0].data
        assert reflection["action"] == "continue"
        assert (reflection["continue_run"], reflection["readable"]) == (False, True)
        assert recorder.types[-2:] == ["reflection", "run.finished"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_gate_stopped_by_an_unreadable_reflection_under_utility(
        self,
    ) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_echo("a"), _answer(PROSE), _answer(PROSE)]),
            steps_per_attempt=1,
            reflection_gate="utility",
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "gate_stopped"
        reflection = recorder.of("reflection")[0].data
        assert (reflection["readable"], reflection["action"]) == (True, "stop")
        assert reflection["rationale"] == "Could not parse reflection response"
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_budget_exhausted(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [_echo("a"), _reflection("stop", best_effort_result="maybe a")]
            ),
            steps_per_attempt=1,
            max_attempts=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.status, result.result) == ("budget_exhausted", "maybe a")
        assert recorder.of("reflection")[0].data["continue_run"] is False
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_budget_exhausted_after_an_unreadable_final_reflection(
        self,
    ) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [_echo("a", content="Probably a."), _answer(PROSE), _answer(PROSE)]
            ),
            steps_per_attempt=1,
            max_attempts=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.status, result.result) == ("budget_exhausted", "Probably a.")
        reflection = recorder.of("reflection")[0].data
        assert (reflection["readable"], reflection["continue_run"]) == (False, False)
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_stuck(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_echo("a", f"call_{i}") for i in range(3)]),
            attempt_policy="none",
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "stuck"
        assert recorder.types[-2:] == ["notice", "run.finished"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_provider_error_from_planning(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_refused()]))
        result = await agent.run_task("echo", on_event=recorder)
        assert result.status == "provider_error"
        assert recorder.types == ["run.started", "step.started", "run.finished"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_provider_error_from_a_reflection_request(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_echo("a"), _refused()]), steps_per_attempt=1)
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "provider_error"
        assert recorder.types[-3:] == ["notice", "reflection.started", "run.finished"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_provider_error_from_the_second_reflection_request(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a", usage=_usage(10, 1)),
                    _answer(PROSE, usage=_usage(30, 3)),
                    _refused(),
                ]
            ),
            steps_per_attempt=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "provider_error"
        assert recorder.types[-3:] == ["notice", "reflection.started", "run.finished"]
        # The first reply's usage counts in the run, but no event carries it.
        await check_guarantees(recorder, result, agent, usage_matches=False)
        assert _event_usage(recorder.events) == _usage(10, 1)
        assert result.usage == _usage(40, 4)

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_calls(("hang", {}, "call_1"))]), timeout_s=0.2)
        result = await agent.run_task("wait", on_event=recorder)
        assert result.status == "timeout"
        assert recorder.types[-3:] == ["tool.started", "tool.finished", "run.finished"]
        finished = recorder.of("tool.finished")[0].data
        assert (finished["status"], finished["error_category"]) == (
            "cancelled",
            "cancelled",
        )
        assert (finished["error"], finished["output"]) == (CALL_INTERRUPTED, "")
        assert 0.1 <= finished["duration_s"] < 5.0
        assert finished["exit_code"] is None
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_crashed(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([RuntimeError("boom")]))
        result = await agent.run_task("echo", on_event=recorder)
        assert (result.status, result.error) == ("crashed", "RuntimeError: boom")
        assert recorder.types == ["run.started", "step.started", "run.finished"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_setup_error(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([]), providers=[WeatherTools(RuntimeError("no server"))]
        )
        result = await agent.run_task("weather", on_event=recorder)
        assert (result.status, result.error) == ("setup_error", "no server")
        assert recorder.types == ["run.started", "run.finished"]
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_tool_pair_closes_on_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_echo("a")]))

        def broken(payload: object) -> object:
            raise RuntimeError("check broke")

        monkeypatch.setattr(agent.checker, "check_executor_result", broken)
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.status, result.error) == ("crashed", "RuntimeError: check broke")
        assert recorder.types[-3:] == ["tool.started", "tool.finished", "run.finished"]
        finished = recorder.of("tool.finished")[0].data
        assert (finished["status"], finished["error_category"], finished["error"]) == (
            "crashed",
            "crashed",
            "RuntimeError: check broke",
        )
        assert recorder.events[-1].data["result"]["status"] == "crashed"
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_an_outside_cancellation_still_emits_run_finished(self) -> None:
        recorder = Recorder()
        tools = Tools()
        agent = _agent(ScriptedLLM([_calls(("hang", {}, "call_1"))]), tools=tools)
        running = asyncio.create_task(agent.run_task("wait", on_event=recorder))
        while not tools.received:
            await asyncio.sleep(0.01)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        # The tool pair closes, and run.finished still comes, once and last.
        assert recorder.types[-3:] == ["tool.started", "tool.finished", "run.finished"]
        assert recorder.types.count("run.finished") == 1
        assert recorder.of("tool.finished")[0].data["status"] == "cancelled"
        assert recorder.events[-1].data["result"]["status"] != "running"


class TestRetries:
    @pytest.mark.asyncio
    async def test_retry_notice(self) -> None:
        recorder = Recorder()
        agent = _agent(ScriptedLLM([_busy(), _answer("hi")]))
        result = await agent.run_task("hi", on_event=recorder)
        assert (result.status, result.llm_errors) == ("completed", 1)
        assert recorder.types == [
            "run.started",
            "step.started",
            "notice",
            "assistant.message",
            "run.finished",
        ]
        [retry] = recorder.of("notice")
        assert retry.data == {
            "kind": "retry",
            "message": "Model request failed; retrying in 1 s: HTTP 503: busy",
            "step": 1,
            "attempt": 1,
            "detail": {"request_attempt": 1, "delay_s": 1.0, "status_code": 503},
        }

    @pytest.mark.asyncio
    async def test_a_retry_notice_quotes_200_characters_of_the_error(self) -> None:
        recorder = Recorder()
        error = ProviderError("e" * 300, status_code=None, retryable=True)
        await _agent(ScriptedLLM([error, _answer("hi")])).run_task(
            "hi", on_event=recorder
        )
        [retry] = recorder.of("notice")
        assert retry.data["message"] == (
            f"Model request failed; retrying in 1 s: {'e' * 200}"
        )
        assert retry.data["detail"]["status_code"] is None


class TestReflections:
    @pytest.mark.asyncio
    async def test_a_retried_reflection_is_one_event(self) -> None:
        recorder = Recorder()
        llm = ScriptedLLM(
            [
                _echo("a", usage=_usage(10, 1)),
                # The first reflection reply is a tool call, so it is asked again.
                Slow(_echo("b", "call_2", usage=_usage(30, 3)), 0.05),
                Slow(_reflection("continue", usage=_usage(50, 5)), 0.05),
                _answer("a"),
            ]
        )
        agent = _agent(llm, steps_per_attempt=1)
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "completed"
        assert recorder.types.count("reflection.started") == 1
        [reflection] = recorder.of("reflection")
        data = reflection.data
        assert (data["requests"], data["readable"], data["action"]) == (
            2,
            True,
            "continue",
        )
        assert data["continue_run"] is True
        assert data["usage"] == _usage(80, 8)
        # Both requests fall within duration_s.
        first_started, second_ended = llm.spans[1][0], llm.spans[2][1]
        assert data["duration_s"] >= second_ended - first_started
        # n_llm and n_aux count both requests (#31).
        assert (result.n_llm, result.n_main, result.n_aux) == (4, 2, 2)
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_a_reflection_read_at_once_is_one_request(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a"),
                    _reflection(
                        "continue",
                        summary="Echoed a.",
                        rationale="Echo once more.",
                        best_effort_result="a",
                        usage=_usage(50, 5),
                    ),
                    _answer("a"),
                ]
            ),
            steps_per_attempt=1,
        )
        await agent.run_task("echo a", on_event=recorder)
        [started] = recorder.of("reflection.started")
        assert started.data == {"attempt": 1, "failure_type": "budget"}
        [reflection] = recorder.of("reflection")
        data = dict(reflection.data)
        assert data.pop("duration_s") >= 0
        assert data.pop("prefix")["against"] == {"kind": "plan", "request": 1}
        assert data == {
            "attempt": 1,
            "action": "continue",
            "continue_run": True,
            "flags": ALL_FLAGS,
            "gate": "flags",
            "utility": None,
            "stop_reason": None,
            "rationale": "Echo once more.",
            "summary": "Echoed a.",
            "best_effort_result": "a",
            "requests": 1,
            "readable": True,
            "unreadable_reason": None,
            "repaired": None,
            "usage": _usage(50, 5),
        }
        index = recorder.types.index("reflection")
        assert recorder.events[index + 1].data == {"step": 2, "attempt": 2}

    @pytest.mark.asyncio
    async def test_an_unreadable_reflection_is_reported(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a", usage=_usage(10, 1)),
                    _answer(PROSE, usage=_usage(30, 3)),
                    _answer(PROSE, usage=_usage(40, 4)),
                    _answer("a"),
                ]
            ),
            steps_per_attempt=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert (result.status, result.attempts) == ("completed", 2)
        [reflection] = recorder.of("reflection")
        data = dict(reflection.data)
        assert data.pop("duration_s") >= 0
        assert data.pop("prefix")["against"] == {"kind": "plan", "request": 1}
        assert data == {
            "attempt": 1,
            "action": None,
            "continue_run": True,
            "flags": None,
            "gate": "flags",
            "utility": None,
            "stop_reason": None,
            "rationale": None,
            "summary": None,
            "best_effort_result": None,
            "requests": 2,
            "readable": False,
            "unreadable_reason": "no_json",
            "repaired": None,
            "usage": _usage(70, 7),
        }
        # The next attempt starts.
        index = recorder.types.index("reflection")
        next_event = recorder.events[index + 1]
        assert (next_event.type, next_event.data) == (
            "step.started",
            {"step": 2, "attempt": 2},
        )
        await check_guarantees(recorder, result, agent)

    @pytest.mark.asyncio
    async def test_an_unreadable_reflection_at_the_final_boundary(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_echo("a"), _answer(PROSE), _answer(PROSE)]),
            steps_per_attempt=1,
            max_attempts=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "budget_exhausted"
        [reflection] = recorder.of("reflection")
        assert (reflection.data["readable"], reflection.data["requests"]) == (False, 2)
        assert reflection.data["continue_run"] is False
        assert recorder.types[-2:] == ["reflection", "run.finished"]
        assert recorder.events[-1].data["result"]["status"] == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_the_same_script_under_utility(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_echo("a"), _answer(PROSE), _answer(PROSE)]),
            steps_per_attempt=1,
            reflection_gate="utility",
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "gate_stopped"
        [reflection] = recorder.of("reflection")
        assert (
            reflection.data["readable"],
            reflection.data["action"],
            reflection.data["requests"],
            reflection.data["continue_run"],
        ) == (True, "stop", 2, False)

    @pytest.mark.asyncio
    async def test_a_trimmed_reflection_reports_its_entry(self) -> None:
        recorder = Recorder()
        bulky = _reflection(
            "continue",
            summary="Kept.",
            failed_hypotheses=[f"hypothesis {i}: " + "f" * 400 for i in range(6)],
        )
        agent = _agent(
            ScriptedLLM([_echo("a"), bulky, _answer("a")]), steps_per_attempt=1
        )
        result = await agent.run_task("echo a", on_event=recorder)
        assert result.status == "completed"
        [reflection] = recorder.of("reflection")
        assert (reflection.data["action"], reflection.data["continue_run"]) == (
            "continue",
            True,
        )
        assert reflection.data["summary"] == "Kept."


# ── Redaction and exit codes ────────────────────────────────────────────────

KEY = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
SECRET_TEXT = f"Use {KEY}; your verification code is 482913."
REDACTED_TEXT = "Use [redacted]; your verification code is [redacted]."


class TestRedaction:
    @pytest.mark.asyncio
    async def test_thinking_and_content_are_redacted(self) -> None:
        # The key straddles EVENT_TEXT_LIMIT: clipped first, its first 14
        # characters would stay, too few to match. Redaction runs first.
        padding = "p" * 99_960
        long_text = f"verification code 482913 {padding} {KEY} " + "q" * 20_000
        clipped = f"verification code [redacted] {padding} [redacted]… [+20001 chars]"
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _echo("a", content=SECRET_TEXT, thinking=SECRET_TEXT),
                    _answer(long_text, thinking=long_text),
                ]
            )
        )
        result = await agent.run_task("echo a", on_event=recorder)
        first, last = recorder.of("assistant.message")
        assert (first.data["content"], first.data["thinking"]) == (
            REDACTED_TEXT,
            REDACTED_TEXT,
        )
        assert (last.data["content"], last.data["thinking"]) == (clipped, clipped)
        # Working memory and the RunResult keep the text; history stores
        # the step's thought redacted, as it does every copy it keeps.
        [call] = [m for m in agent.hippo.working.get_window() if m.tool_calls]
        assert (call.content, call.thinking) == (SECRET_TEXT, SECRET_TEXT)
        steps = await agent.hippo.history.get_steps(result.task_id)
        assert steps[0].thought == REDACTED_TEXT
        assert result.result == long_text
        assert recorder.events[-1].data["result"]["result"] == long_text

    @pytest.mark.asyncio
    async def test_no_redaction_without_a_sink(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The emitter logs and drops an event whose payload raises, so the
        # fake records its calls: the raise alone would not fail the test.
        redacted: list[str] = []

        def refuse(text: str) -> tuple[str, int]:
            redacted.append(text)
            raise AssertionError("redacted without a sink")

        monkeypatch.setattr(_events, "redact_secrets", refuse)
        result = await _agent(ScriptedLLM([_answer(SECRET_TEXT)])).run_task("hi")
        assert result.result == SECRET_TEXT
        assert redacted == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("secret", [True, False], ids=["secrets", "none"])
    async def test_reflection_event_text_is_redacted(self, secret: bool) -> None:
        token = "Bearer abcdefghijklmnop1234" if secret else "Bearer token"
        key = KEY if secret else "the key"
        fields: dict[str, Any] = {
            "rationale": f"Done; {key} worked.",
            "summary": f"Called it with Authorization: {token} once.",
            "best_effort_result": f"a, with {key}",
        }
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM([_echo("a"), _reflection("done", **fields)]),
            steps_per_attempt=1,
        )
        result = await agent.run_task("echo a", on_event=recorder)
        [reflection] = recorder.of("reflection")
        expected = (
            {
                "rationale": "Done; [redacted] worked.",
                "summary": "Called it with Authorization: Bearer [redacted] once.",
                "best_effort_result": "a, with [redacted]",
            }
            if secret
            else fields
        )
        assert {name: reflection.data[name] for name in fields} == expected
        # The RunResult keeps the text as the model wrote it.
        assert result.result == fields["best_effort_result"]


class TestExitCode:
    @pytest.mark.asyncio
    async def test_exit_code_is_null_for_other_tools(self) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _calls(
                        ("shell_exec", {"command": "exit 3"}, "call_1"),
                        ("weather", {"city": "Oslo"}, "call_2"),
                    ),
                    _answer("done"),
                ]
            ),
            builtins=frozenset({"shell_exec"}),
            providers=[WeatherTools()],
        )
        result = await agent.run_task("check", on_event=recorder)
        assert result.status == "completed"
        shell, weather = recorder.of("tool.finished")
        assert (shell.data["name"], shell.data["status"]) == ("shell_exec", "success")
        assert (weather.data["name"], weather.data["output"]) == (
            "weather",
            "Sunny in Oslo",
        )
        assert shell.data["exit_code"] is None
        assert weather.data["exit_code"] is None

    @pytest.mark.asyncio
    async def test_the_last_recorded_exit_code_is_reported(self) -> None:
        async def child(text: str = "") -> str:
            codes = TOOL_EXIT_CODE.get()
            assert codes is not None
            codes.extend([1, 3])
            return "ran"

        recorder = Recorder()
        agent = _agent(ScriptedLLM([_echo("a"), _answer("done")]))
        agent.executor.register("echo", child)
        await agent.run_task("echo a", on_event=recorder)
        assert recorder.of("tool.finished")[0].data["exit_code"] == 3

    @pytest.mark.asyncio
    async def test_python_exit_code_reaches_tool_finished_without_a_root(
        self,
    ) -> None:
        recorder = Recorder()
        agent = _agent(
            ScriptedLLM(
                [
                    _calls(("python_exec", {"code": "import sys\nsys.exit(3)"}, "c1")),
                    _answer("done"),
                ]
            ),
            builtins=frozenset({"python_exec"}),
        )

        result = await agent.run_task("run python", on_event=recorder)

        assert result.status == "completed"
        finished = recorder.of("tool.finished")[0].data
        assert (finished["status"], finished["exit_code"]) == ("error", 3)
        assert finished["error"] == "Python exited with code 3"

    @pytest.mark.asyncio
    async def test_exit_code_reaches_tool_finished(self, tmp_path: Path) -> None:
        big = "head -c 60000 /dev/zero | tr '\\0' x; exit 1"
        llm = ScriptedLLM(
            [
                _calls(
                    ("shell_exec", {"command": "exit 3"}, "call_1"),
                    ("shell_exec", {"command": "exit 0"}, "call_2"),
                    ("shell_exec", {"command": "sleep 5"}, "call_3"),
                    ("python_exec", {"code": "import sys\nsys.exit(3)"}, "call_4"),
                ),
                _calls(
                    ("shell_exec", {"command": big}, "call_5"),
                    ("shell_exec", {"command": big}, "call_6"),
                ),
                _answer("done"),
            ]
        )
        recorder = Recorder()
        agent = _agent(
            llm,
            builtins=frozenset({"shell_exec", "python_exec"}),
            workspace_root=str(tmp_path),
            tool_timeout_s=1,
        )

        result = await agent.run_task("run them", on_event=recorder)

        assert result.status == "completed"
        await check_guarantees(recorder, result, agent)
        finished = {e.data["call_id"]: e.data for e in recorder.of("tool.finished")}
        reported = {
            call_id: (data["status"], data["exit_code"])
            for call_id, data in finished.items()
        }
        assert reported == {
            "call_1": ("success", 3),
            "call_2": ("success", 0),
            "call_3": ("timeout", None),  # killed by tool_timeout_s
            "call_4": ("error", 3),
            "call_5": ("success", 1),
            "call_6": ("success", 1),
        }
        assert finished["call_1"]["output"] == "[exit code 3]"
        assert finished["call_3"]["error"] == "Command timed out after 1 s"
        assert finished["call_4"]["error"] == "Python exited with code 3"
        assert finished["call_5"]["output"].startswith("[exit code 1]\nxxx")

        # The turn budget cut one 60,000-character result and withheld the
        # other; the line stays first in both, after the withheld notice.
        tool_messages = {
            m["tool_call_id"]: m["content"]
            for m in llm.requests[2]["messages"]
            if m["role"] == "tool"
        }
        withheld, cut = sorted(
            (tool_messages["call_5"], tool_messages["call_6"]),
            key=lambda content: not content.startswith("[Tool result withheld"),
        )
        notice, preview = withheld.split("\n", 1)
        assert notice.startswith("[Tool result withheld from working memory: ")
        assert notice.endswith(" First 200 characters follow.]")
        assert len(preview) == 200
        assert preview.startswith("[exit code 1]\nxxx")
        assert cut.startswith("[exit code 1]\nxxx")
        assert cut.endswith(" more characters]")
        assert "\n... [truncated: " in cut


# ── Without a sink ──────────────────────────────────────────────────────────


def _comparable(result: RunResult) -> dict[str, Any]:
    data = result.to_dict()
    del data["task_id"], data["duration_s"]
    data["config"]["memory_dir"] = "<memory_dir>"
    return data


def _request_json(
    messages: Sequence[Message], tools: Sequence[ToolDef] | None
) -> dict[str, Any]:
    return {
        "tools": None if tools is None else [tool.name for tool in tools],
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": [(c.tool_name, c.arguments, c.id) for c in m.tool_calls],
                "tool_call_id": m.tool_call_id,
            }
            for m in messages
        ],
    }


def _window_json(messages: Iterable[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": m.role,
            "content": m.content,
            "thinking": m.thinking,
            "tool_calls": [(c.tool_name, c.arguments, c.id) for c in m.tool_calls],
            "tool_call_id": m.tool_call_id,
            "images": [(i.mime_type, i.sha256, i.size_bytes, i.data) for i in m.images],
        }
        for m in messages
    ]


def _history_json(memory_dir: Path) -> dict[str, list[Any]]:
    """tasks and steps by column, without ids, timestamps and durations."""
    conn = sqlite3.connect(memory_dir / "history.db")
    try:
        tasks = conn.execute(
            "SELECT query, status, result FROM tasks ORDER BY rowid"
        ).fetchall()
        steps = conn.execute(
            "SELECT thought, tool_call_json, tool_result_json, seq FROM steps "
            "ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    kept = []
    for thought, call, outcome, seq in steps:
        outcome = json.loads(outcome)
        outcome.pop("duration_ms", None)
        kept.append([thought, json.loads(call), outcome, seq])
    return {"tasks": [list(row) for row in tasks], "steps": kept}


def _rich_script() -> list[Reply]:
    """A run through most hooks: a retry, tools, an image, an empty reply,
    a reflection asked again, and a final answer with thinking."""
    return [
        _busy(),
        _echo("a", content="Echoing.", thinking="An echo.", usage=_usage(10, 1)),
        _answer("", usage=_usage(11, 0)),
        _echo("x", "call_9", usage=_usage(30, 3)),  # a reflection reply, asked again
        _reflection("continue", usage=_usage(40, 4)),
        _calls(("photo", {}, "call_2"), usage=_usage(12, 2)),
        _answer("All done.", thinking="Finished.", usage=_usage(13, 3)),
    ]


async def _rich_run(
    memory_dir: Path, sink: Recorder | None
) -> tuple[ScriptedLLM, RunResult, Agent]:
    llm = ScriptedLLM(_rich_script())
    agent = _agent(
        llm,
        embedding=SameVectorEmbedding(),
        memory_dir=str(memory_dir),
        steps_per_attempt=2,
    )
    try:
        result = await agent.run_task("echo a, then take a photo", on_event=sink)
    finally:
        await agent.close()
    return llm, result, agent


class TestWithoutASink:
    @pytest.mark.asyncio
    async def test_no_sink_no_policy_is_unchanged(self, tmp_path: Path) -> None:
        plain_llm, plain, plain_agent = await _rich_run(tmp_path / "plain", None)
        recorder = Recorder()
        llm, result, agent = await _rich_run(tmp_path / "sink", recorder)
        assert plain.status == "completed" and recorder.events
        assert llm.requests == plain_llm.requests
        assert _comparable(result) == _comparable(plain)
        assert _window_json(agent.hippo.working.get_window()) == _window_json(
            plain_agent.hippo.working.get_window()
        )
        assert _history_json(tmp_path / "sink") == _history_json(tmp_path / "plain")

    @pytest.mark.asyncio
    async def test_no_event_is_built_without_a_sink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The emitter logs and drops an event whose payload raises, so the
        # fakes record their calls: the raise alone would not fail the test.
        built: list[str] = []

        def refusing(name: str) -> Any:
            def refuse(*args: Any, **kwargs: Any) -> Any:
                built.append(name)
                raise AssertionError("built without a sink")

            return refuse

        for name in (
            "AgentEvent",
            "json_safe_payload",
            "redact_secrets",
            "call_preview",
        ):
            monkeypatch.setattr(_events, name, refusing(name))
        _, result, _ = await _rich_run(tmp_path, None)
        assert result.status == "completed"
        assert built == []
