"""Cache telemetry in ``assistant.message`` and ``reflection``: server usage details and prefix diagnostics.

The model is a stub OpenAI-compatible server behind ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._prefix_diag import (
    MAX_FINGERPRINTS,
    best_match,
    compare,
    fingerprint,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.schema import Message, ReasoningTags, ToolCall, ToolDef
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.legacy_options import LEGACY_CACHE_OPTIONS

ECHO = ToolDef(
    name="echo",
    description="Echo text back",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)

DETAILS = {
    "prompt_tokens": 100,
    "completion_tokens": 7,
    "total_tokens": 107,
    "prompt_tokens_details": {"cached_tokens": 64},
    "completion_tokens_details": {"reasoning_tokens": 3},
}
PLAIN = {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}


def _call_reply(usage: dict[str, Any]) -> dict[str, Any]:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "echo", "arguments": '{"text": "hi"}'},
    }
    return {
        "choices": [{"message": {"content": "", "tool_calls": [call]}}],
        "usage": usage,
    }


def _final_reply(usage: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": {"content": "done"}}], "usage": usage}


def _sse(reply: dict[str, Any]) -> bytes:
    message = reply["choices"][0]["message"]
    delta = {"content": message["content"]}
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": "call_1",
                "function": message["tool_calls"][0]["function"],
            }
        ]
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    frames = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
        {"choices": [], "usage": reply["usage"]},
    ]
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
    return (body + "data: [DONE]\n\n").encode()


class Stub:
    """Replies in turn, as JSON or, for a streaming request, as SSE."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        reply = self.replies.pop(0)
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=_sse(reply),
            )
        return httpx.Response(200, json=reply)


class Display:
    reasoning_tags: ReasoningTags | None = None

    def step(self, number: int) -> None: ...
    def opened(self) -> None: ...
    def held(self) -> None: ...
    def content(self, delta: str) -> None: ...
    def closed(self) -> None: ...


async def _run(
    stub: Stub,
    *,
    extra_body: dict[str, Any] | None = None,
    on_echo: Callable[[Agent], None] | None = None,
    stream: bool = False,
    base: Mapping[str, Any] = LEGACY_CACHE_OPTIONS,
    **options: Any,
) -> list[AgentEvent]:
    llm = OpenAICompatLLM(
        "http://stub.test/v1",
        "stub-model",
        extra_body=extra_body,
        transport=httpx.MockTransport(stub),
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(), tool_exposure="all", **{**base, **options}
        ),
    )
    agent.hippo.register_tool(ECHO)

    async def echo(text: str = "") -> str:
        if on_echo is not None:
            on_echo(agent)
        return text

    agent.executor.register("echo", echo)
    events: list[AgentEvent] = []
    token = STREAM_DISPLAY.set(Display()) if stream else None
    try:
        result = await agent.run_task("echo hi", on_event=events.append)
    finally:
        if token is not None:
            STREAM_DISPLAY.reset(token)
        await agent.close()
    assert result.status == "completed", result.error
    return events


async def _plans(stub: Stub, **kwargs: Any) -> list[AgentEvent]:
    events = await _run(stub, **kwargs)
    return [e for e in events if e.type == "assistant.message"]


async def test_details_reach_the_event_but_not_the_request() -> None:
    stub = Stub([_call_reply(DETAILS), _final_reply(DETAILS)])
    first, second = await _plans(stub)
    for event in (first, second):
        assert event.data["usage"] == {
            "prompt_tokens": 100,
            "completion_tokens": 7,
            "total_tokens": 107,
            "cached_tokens": 64,
            "reasoning_tokens": 3,
        }
    assert all(
        set(body) <= {"model", "messages", "tools", "tool_choice"}
        for body in stub.bodies
    )


async def test_no_detail_the_server_did_not_send_is_invented() -> None:
    stub = Stub([_call_reply(PLAIN), _final_reply(PLAIN)])
    events = await _plans(stub)
    assert [e.data["usage"] for e in events] == [PLAIN, PLAIN]


async def test_only_the_cached_count_when_that_is_all_the_server_sent() -> None:
    usage = {**PLAIN, "prompt_tokens_details": {"cached_tokens": 0}}
    events = await _plans(Stub([_call_reply(usage), _final_reply(usage)]))
    assert events[0].data["usage"] == {**PLAIN, "cached_tokens": 0}


async def test_streaming_with_include_usage_carries_the_details() -> None:
    stub = Stub([_call_reply(DETAILS), _final_reply(DETAILS)])
    events = await _plans(
        stub, extra_body={"stream_options": {"include_usage": True}}, stream=True
    )
    assert all(body["stream"] for body in stub.bodies)
    assert [e.data["usage"]["cached_tokens"] for e in events] == [64, 64]
    assert events[1].data["usage"]["reasoning_tokens"] == 3


async def test_prefix_of_a_growing_conversation() -> None:
    stub = Stub([_call_reply(PLAIN), _final_reply(PLAIN)])
    first, second = await _plans(stub)
    assert first.data["prefix"] is None
    prefix = second.data["prefix"]
    sent_first, sent_second = (len(b["messages"]) for b in stub.bodies)
    assert prefix["messages"] == prefix["previous_messages"] == sent_first
    assert prefix["total_messages"] == sent_second > sent_first
    assert prefix["first_changed"] is None and prefix["tools_changed"] is False
    assert prefix["chars"] > 0


async def test_a_system_prompt_that_changes_names_system_as_first_changed() -> None:
    def change(agent: Agent) -> None:
        agent.chat.system_notes = ("A note that appeared after step one.",)

    stub = Stub([_call_reply(PLAIN), _final_reply(PLAIN)])
    _, second = await _plans(stub, on_echo=change)
    prefix = second.data["prefix"]
    assert prefix["first_changed"] == "system"
    assert prefix["messages"] == 0
    # The system prompt starts the same way, so some characters are shared.
    assert prefix["chars"] > 0
    assert stub.bodies[0]["messages"][0] != stub.bodies[1]["messages"][0]


def test_compare_names_the_changed_role_and_the_tools() -> None:
    system = Message(role="system", content="be brief")
    user = Message(role="user", content="hello there")
    call = ToolCall(tool_name="echo", arguments={}, id="c1")
    before = fingerprint([system, user], [ECHO])
    appended = fingerprint(
        [system, user, Message(role="assistant", tool_calls=[call])], [ECHO]
    )
    assert compare(before, appended) == {
        "messages": 2,
        "chars": len("be brief") + len("hello there"),
        "previous_messages": 2,
        "total_messages": 3,
        "first_changed": None,
        "tools_changed": False,
    }
    edited = fingerprint([system, Message(role="user", content="hello you")], [])
    result = compare(before, edited)
    assert (result["messages"], result["first_changed"]) == (1, "user")
    assert result["chars"] == len("be brief") + len("hello ")
    assert result["tools_changed"] is True


REPORT = Path(__file__).resolve().parents[2] / "scripts" / "cache_report.py"


def _report(log: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(REPORT), str(log)],
        capture_output=True,
        text=True,
        check=False,
    )


async def test_cache_report_on_a_recorded_log(tmp_path: Path) -> None:
    def change(agent: Agent) -> None:
        agent.chat.system_notes = ("A note that appeared after step one.",)

    stub = Stub(
        [
            _call_reply({**DETAILS, "prompt_tokens_details": {"cached_tokens": 0}}),
            _final_reply(DETAILS),
        ]
    )
    events = await _plans(stub, on_echo=change)
    log = tmp_path / "run.jsonl"
    lines = [{"v": 1, "type": "event", "event": e.to_dict()} for e in events] + [
        {"v": 1, "type": "result", "result": {}}
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")
    done = _report(log)
    assert done.returncode == 0, done.stderr
    rows = [line.split() for line in done.stdout.splitlines()]
    assert rows[1][:6] == ["#", "kind", "at", "prompt", "cached", "new"]
    assert rows[2] == ["1", "plan", "1", "100", "0", "100", "-", "-", "-"]
    assert rows[3][:6] == ["2", "plan", "2", "100", "64", "36"]
    assert rows[3][-3:] == ["plan", "#1", "system"]
    assert "total: 2 requests, 200 prompt tokens, 64 cached, 136 new" in done.stdout
    assert "cached share 32.0%" in done.stdout


def test_cache_report_marks_what_a_log_does_not_hold(tmp_path: Path) -> None:
    event = {
        "v": 1,
        "type": "assistant.message",
        "run_id": "r1",
        "data": {"step": 1, "usage": PLAIN},
    }
    log = tmp_path / "old.jsonl"
    log.write_text(json.dumps(event) + "\n")
    done = _report(log)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[2].split() == [
        "1",
        "plan",
        "1",
        "100",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    assert "cached share n/a" in done.stdout
    empty = tmp_path / "empty.jsonl"
    empty.write_text("{}\n")
    assert _report(empty).returncode == 1


def _reflection_reply(
    usage: dict[str, Any], *, readable: bool = True
) -> dict[str, Any]:
    decision = {
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
        "recommended_action": "continue",
        "rationale": "Once more.",
        "best_effort_result": "hi",
    }
    content = json.dumps(decision) if readable else "I would rather say it in words."
    return {"choices": [{"message": {"content": content}}], "usage": usage}


def _cached(tokens: int) -> dict[str, Any]:
    return {**PLAIN, "prompt_tokens_details": {"cached_tokens": tokens}}


async def test_a_reflection_between_attempts_carries_usage_and_prefix() -> None:
    stub = Stub(
        [
            _call_reply(_cached(0)),
            _reflection_reply(_cached(10)),
            _final_reply(_cached(40)),
        ]
    )
    events = await _run(stub, steps_per_attempt=1)
    plans = [e for e in events if e.type == "assistant.message"]
    [reflection] = [e for e in events if e.type == "reflection"]
    assert reflection.data["requests"] == 1
    assert reflection.data["usage"]["cached_tokens"] == 10
    # The reflection prompt differs from the plan's at byte 0 and offers no tools.
    prefix = reflection.data["prefix"]
    assert prefix["against"] == {"kind": "plan", "request": 1}
    assert (prefix["messages"], prefix["first_changed"]) == (0, "system")
    assert prefix["tools_changed"] is True
    # The plan after it matches the earlier plan (the same leading system text
    # and the same tools), not the reflection.
    after = plans[1].data["prefix"]
    assert after["against"] == {"kind": "plan", "request": 1}
    assert after["tools_changed"] is False
    assert after["chars"] > prefix["chars"]


async def test_a_reflection_asked_twice_sums_its_usage_and_takes_two_numbers() -> None:
    stub = Stub(
        [
            _call_reply(_cached(0)),
            _reflection_reply(_cached(10), readable=False),
            _reflection_reply(_cached(20)),
            _final_reply(_cached(40)),
        ]
    )
    events = await _run(stub, steps_per_attempt=1)
    [reflection] = [e for e in events if e.type == "reflection"]
    assert reflection.data["requests"] == 2
    assert reflection.data["usage"]["cached_tokens"] == 30
    assert reflection.data["prefix"]["against"] == {"kind": "plan", "request": 1}
    last = [e for e in events if e.type == "assistant.message"][-1]
    assert last.data["prefix"]["against"] == {"kind": "plan", "request": 1}


def test_best_match_takes_the_most_leading_messages_then_characters() -> None:
    system = Message(role="system", content="be brief")
    user = Message(role="user", content="hello there")
    other = Message(role="system", content="be brief, always")
    long = fingerprint([system, user], None)
    short = fingerprint([system], None)
    wrong = fingerprint([other, user], None)
    current = fingerprint([system, user, Message(role="assistant", content="hi")], None)
    earlier = [(1, "plan", long), (2, "reflect", short), (3, "plan", wrong)]
    match = best_match(earlier, current)
    assert match is not None
    assert match["against"] == {"kind": "plan", "request": 1}
    assert match["messages"] == 2
    # On equal messages and characters the most recent wins.
    again = best_match([(1, "plan", long), (2, "reflect", long)], current)
    assert again is not None and again["against"]["request"] == 2
    # With nothing identical, characters decide (the system texts share a start).
    differs = fingerprint([Message(role="system", content="be briefly")], None)
    unlike = fingerprint([Message(role="system", content="zzz")], None)
    third = best_match([(1, "plan", unlike), (2, "plan", differs)], long)
    assert third is not None and third["against"]["request"] == 2
    assert third["messages"] == 0 and third["chars"] == len("be brief")
    assert best_match([], current) is None


async def test_the_fingerprints_a_run_keeps_are_bounded() -> None:
    from localmelo.melo.agent._chat import Chat

    class Echoing:
        model = "stub"

        async def chat(self, messages: Any, tools: Any = None, **_: Any) -> Message:
            return Message(role="assistant", content="ok")

    chat = Chat(Echoing())  # type: ignore[arg-type]
    for i in range(MAX_FINGERPRINTS + 5):
        await chat._request(
            [Message(role="user", content=str(i))], None, kind="reflect"
        )
    assert len(chat._sent) == MAX_FINGERPRINTS
    assert chat._sent[-1][0] == MAX_FINGERPRINTS + 5
    assert chat.last_call is not None and chat.last_call.prefix is not None


def test_cache_report_has_a_row_per_request_and_totals_by_kind(tmp_path: Path) -> None:
    def event(type_: str, **data: Any) -> dict[str, Any]:
        return {"v": 1, "type": type_, "run_id": "r1", "data": data}

    plan = {"messages": 2, "chars": 50, "first_changed": None, "tools_changed": False}
    lines = [
        event("assistant.message", step=1, usage={**PLAIN, "cached_tokens": 0}),
        event(
            "reflection",
            attempt=1,
            requests=2,
            usage={**PLAIN, "cached_tokens": 50},
            prefix={
                **plan,
                "messages": 0,
                "chars": 9,
                "first_changed": "system",
                "tools_changed": True,
                "against": {"kind": "plan", "request": 1},
            },
        ),
        event(
            "assistant.message",
            step=2,
            usage={**PLAIN, "cached_tokens": 60},
            prefix={**plan, "against": {"kind": "plan", "request": 1}},
        ),
    ]
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    done = _report(log)
    assert done.returncode == 0, done.stderr
    rows = [line.split() for line in done.stdout.splitlines()]
    assert rows[3][:3] == ["2", "reflect", "a1"]
    assert rows[3][-8:] == [
        "0",
        "msgs,",
        "9",
        "chars",
        "plan",
        "#1",
        "system",
        "+tools",
    ]
    # A reflection asked twice takes two numbers: the next plan is the fourth.
    assert rows[4][:3] == ["4", "plan", "2"]
    assert "plan: 2 requests, 200 prompt tokens, 60 cached" in done.stdout
    assert "reflect: 1 requests, 100 prompt tokens, 50 cached" in done.stdout
    assert "total: 3 requests, 300 prompt tokens, 110 cached" in done.stdout


def test_cache_report_prints_its_usage_for_help_and_for_no_arguments() -> None:
    for flag in ("-h", "--help"):
        done = subprocess.run(
            [sys.executable, "-I", str(REPORT), flag],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith("Print the prompt-cache use")
        assert "usage: python scripts/cache_report.py LOG [LOG ...]" in done.stdout
    bare = subprocess.run(
        [sys.executable, "-I", str(REPORT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bare.returncode == 2
    assert bare.stdout == ""
    assert "usage: python scripts/cache_report.py LOG [LOG ...]" in bare.stderr
