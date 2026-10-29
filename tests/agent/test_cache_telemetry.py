"""Cache telemetry in ``assistant.message``: server usage details and prefix diagnostics.

The model is a stub OpenAI-compatible server behind ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._prefix_diag import compare, fingerprint
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.schema import Message, ReasoningTags, ToolCall, ToolDef
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

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
) -> list[AgentEvent]:
    llm = OpenAICompatLLM(
        "http://stub.test/v1",
        "stub-model",
        extra_body=extra_body,
        transport=httpx.MockTransport(stub),
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), tool_exposure="all"),
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
    return [e for e in events if e.type == "assistant.message"]


async def test_details_reach_the_event_but_not_the_request() -> None:
    stub = Stub([_call_reply(DETAILS), _final_reply(DETAILS)])
    first, second = await _run(stub)
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
    events = await _run(stub)
    assert [e.data["usage"] for e in events] == [PLAIN, PLAIN]


async def test_only_the_cached_count_when_that_is_all_the_server_sent() -> None:
    usage = {**PLAIN, "prompt_tokens_details": {"cached_tokens": 0}}
    events = await _run(Stub([_call_reply(usage), _final_reply(usage)]))
    assert events[0].data["usage"] == {**PLAIN, "cached_tokens": 0}


async def test_streaming_with_include_usage_carries_the_details() -> None:
    stub = Stub([_call_reply(DETAILS), _final_reply(DETAILS)])
    events = await _run(
        stub, extra_body={"stream_options": {"include_usage": True}}, stream=True
    )
    assert all(body["stream"] for body in stub.bodies)
    assert [e.data["usage"]["cached_tokens"] for e in events] == [64, 64]
    assert events[1].data["usage"]["reasoning_tokens"] == 3


async def test_prefix_of_a_growing_conversation() -> None:
    stub = Stub([_call_reply(PLAIN), _final_reply(PLAIN)])
    first, second = await _run(stub)
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
    _, second = await _run(stub, on_echo=change)
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
    events = await _run(stub, on_echo=change)
    log = tmp_path / "run.jsonl"
    lines = [{"v": 1, "type": "event", "event": e.to_dict()} for e in events] + [
        {"v": 1, "type": "result", "result": {}}
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")
    done = _report(log)
    assert done.returncode == 0, done.stderr
    rows = [line.split() for line in done.stdout.splitlines()]
    assert rows[1][:4] == ["step", "prompt", "cached", "new"]
    assert rows[2] == ["1", "100", "0", "100", "-", "-"]
    assert rows[3][:4] == ["2", "100", "64", "36"]
    assert rows[3][-1] == "system"
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
    assert done.stdout.splitlines()[2].split() == ["1", "100", "-", "-", "-", "-"]
    assert "cached share n/a" in done.stdout
    empty = tmp_path / "empty.jsonl"
    empty.write_text("{}\n")
    assert _report(empty).returncode == 1
