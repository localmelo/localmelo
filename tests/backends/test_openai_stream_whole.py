"""A server that does not stream still answers a streamed request.

``stream_chat`` reads a response that is one JSON body as an ordinary
completion, and asks once more, without streaming, when the stream ends before
it delivered anything. Either way the provider stops asking that server to
stream. A stream that breaks after part of a reply reached the caller keeps its
error. Nothing here reaches a network: every server is an ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from localmelo.melo.contracts.providers import (
    LLM_REQUEST_ATTEMPT,
    LLM_REQUEST_KIND,
    ProviderError,
    StreamFragment,
)
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.providers.llm import openai_compat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.backends.test_openai_stream import Bytes, frame

URL = "http://example.test/v1"
ASK = [Message(role="user", content="hi")]
TOOL = ToolDef(name="read", description="Read a file.", parameters={"type": "object"})

COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hi 世界",
                "reasoning_content": "think",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"x": 3}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 1},
    },
}

TEXT_ONLY: dict[str, Any] = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Plain answer."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
}

Handler = Callable[[httpx.Request], httpx.Response]


def provider(handler: Handler, **kwargs: Any) -> OpenAICompatLLM:
    return OpenAICompatLLM(
        URL, "test", transport=httpx.MockTransport(handler), **kwargs
    )


def serving(stream: Bytes, content_type: str = "application/json") -> Handler:
    """A server that answers every request with the body of *stream*."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": content_type}, stream=stream
        )

    return handler


def replying(payload: dict[str, Any]) -> Handler:
    """A server that answers every request with *payload* as JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


def fields(reply: Message | None) -> tuple[Any, ...]:
    """What a reply holds, without its timestamp and its generated ids."""
    assert reply is not None
    return (
        reply.role,
        reply.content,
        reply.thinking,
        reply.usage,
        [(call.tool_name, call.arguments) for call in reply.tool_calls],
    )


async def streamed(
    llm: OpenAICompatLLM, tools: list[ToolDef] | None = None
) -> list[StreamFragment]:
    return [fragment async for fragment in llm.stream_chat(ASK, tools)]


def sse_answer(content: str) -> bytes:
    return (
        frame({"choices": [{"index": 0, "delta": {"content": content}}]})
        + frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        + b"data: [DONE]\n\n"
    )


class Recorder:
    """A server that records its requests and answers each by *answer*."""

    def __init__(self, answer: Callable[[int, dict[str, Any]], httpx.Response]) -> None:
        self.answer = answer
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        return self.answer(len(self.bodies) - 1, body)

    @property
    def streams(self) -> list[bool]:
        """Per request, whether it asked for a stream."""
        return [body.get("stream") is True for body in self.bodies]


# ── A response that is one JSON body ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "application/json"},
        {"Content-Type": "text/event-stream"},
        {"Content-Type": "text/plain; charset=utf-8"},
        {},
    ],
    ids=["json", "event-stream", "text", "none"],
)
async def test_a_json_body_is_read_as_the_completion_chat_would_return(
    headers: dict[str, str],
) -> None:
    server = Recorder(
        lambda n, body: httpx.Response(
            200, headers=headers, content=json.dumps(COMPLETION).encode()
        )
    )
    llm = provider(server)
    try:
        whole = await llm.chat(ASK, [TOOL])
        usage, key = llm.last_usage, llm.last_reasoning_key
        llm.last_usage, llm.last_reasoning_key = None, ""
        events = await streamed(llm, [TOOL])
    finally:
        await llm.close()
    assert [event.kind for event in events] == ["reply"]
    assert fields(events[0].reply) == fields(whole)
    assert events[0].finish_reason == "tool_calls"
    assert events[0].whole is True
    assert (llm.last_usage, llm.last_reasoning_key) == (usage, key)
    assert usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "cached_tokens": 1,
    }
    assert key == "reasoning_content"
    assert server.streams == [False, True]


@pytest.mark.asyncio
async def test_every_split_of_a_json_body_reads_the_same() -> None:
    raw = b"\xef\xbb\xbf \n" + json.dumps(COMPLETION, ensure_ascii=False).encode()
    expected: Any = None
    for i in range(len(raw) + 1):
        body = Bytes([raw[:i], raw[i:]])
        llm = provider(serving(body))
        try:
            events = await streamed(llm)
        finally:
            await llm.close()
        assert [event.kind for event in events] == ["reply"], i
        assert body.closed == 1
        expected = expected or fields(events[0].reply)
        assert fields(events[0].reply) == expected
    assert expected[1] == "Hi 世界"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message,finish",
    [
        ({"role": "assistant", "content": "ok"}, "stop"),
        (COMPLETION["choices"][0]["message"], "tool_calls"),
    ],
    ids=["text", "tool calls"],
)
async def test_a_json_body_without_a_finish_reason_gets_the_usual_one(
    message: dict[str, Any], finish: str
) -> None:
    bare = {"choices": [{"message": message}]}
    llm = provider(lambda _: httpx.Response(200, json=bare))
    try:
        events = await streamed(llm)
    finally:
        await llm.close()
    assert events[0].finish_reason == finish


@pytest.mark.asyncio
async def test_after_a_whole_answer_the_server_is_not_asked_to_stream_again() -> None:
    server = Recorder(lambda n, body: httpx.Response(200, json=TEXT_ONLY))
    llm = provider(server)
    try:
        assert llm.sends_whole_replies is False
        first = await streamed(llm, [TOOL])
        assert llm.sends_whole_replies is True
        second = await streamed(llm, [TOOL])
    finally:
        await llm.close()
    assert server.streams == [True, False]
    assert [event.kind for event in first + second] == ["reply", "reply"]
    assert all(event.whole for event in first + second)
    assert second[0].reply and second[0].reply.content == "Plain answer."
    # The second request is the ordinary one: no "stream" key, same fields as chat().
    assert list(server.bodies[1]) == ["model", "messages", "tools", "tool_choice"]


@pytest.mark.asyncio
async def test_a_json_body_that_is_not_a_completion_fails_as_chat_does() -> None:
    payloads: list[dict[str, Any]] = [{"error": {"message": "no"}}, {"choices": []}]
    for payload in payloads:
        llm = provider(replying(payload))
        try:
            with pytest.raises(ProviderError) as chat_failure:
                await llm.chat(ASK)
            with pytest.raises(ProviderError) as stream_failure:
                await streamed(llm)
        finally:
            await llm.close()
        assert str(stream_failure.value) == str(chat_failure.value)
        assert not stream_failure.value.retryable
        assert llm.sends_whole_replies is False


@pytest.mark.asyncio
async def test_a_json_looking_body_that_does_not_parse_is_malformed() -> None:
    llm = provider(lambda _: httpx.Response(200, content=b'{"choices": [{"message": '))
    try:
        with pytest.raises(ProviderError, match="malformed body") as failure:
            await streamed(llm)
    finally:
        await llm.close()
    assert not failure.value.retryable
    assert llm.sends_whole_replies is False


@pytest.mark.asyncio
async def test_a_json_body_past_the_size_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_compat, "_MAX_WHOLE_BODY", 64)
    llm = provider(lambda _: httpx.Response(200, json=COMPLETION))
    try:
        with pytest.raises(ProviderError, match="size limit") as failure:
            await streamed(llm)
    finally:
        await llm.close()
    assert not failure.value.retryable


# ── A stream that ends before it delivered anything ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ended",
    [b"", b": ping\n\n", b"\n\n", b"data: [DONE]\n\n", b"event: ping\n\n"],
    ids=["empty", "comment", "blank", "done only", "event only"],
)
async def test_a_stream_that_ends_at_once_is_asked_again_without_streaming(
    ended: bytes,
) -> None:
    def answer(n: int, body: dict[str, Any]) -> httpx.Response:
        if body.get("stream"):
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=ended
            )
        return httpx.Response(200, json=COMPLETION)

    server = Recorder(answer)
    llm = provider(server)
    try:
        whole = await llm.chat(ASK, [TOOL])
        server.bodies.clear()
        llm.last_usage, llm.last_reasoning_key = None, ""
        first = await streamed(llm, [TOOL])
        usage = llm.last_usage
        second = await streamed(llm, [TOOL])
    finally:
        await llm.close()
    assert server.streams == [True, False, False]
    assert [event.kind for event in first] == ["reply"]
    assert fields(first[0].reply) == fields(whole)
    assert first[0].finish_reason == "tool_calls" and first[0].whole is True
    assert usage and usage["cached_tokens"] == 1
    assert llm.last_reasoning_key == "reasoning_content"
    assert llm.sends_whole_replies is True
    assert second[0].whole is True and fields(second[0].reply) == fields(whole)


@pytest.mark.asyncio
async def test_the_second_request_is_the_ordinary_request_of_the_same_attempt() -> None:
    def answer(n: int, body: dict[str, Any]) -> httpx.Response:
        if body.get("stream"):
            return httpx.Response(200, content=b"")
        return httpx.Response(200, json=TEXT_ONLY)

    server = Recorder(answer)
    llm = provider(
        server, extra_body={"temperature": 0}, extra_body_by_kind={"plan": {"k": 1}}
    )
    attempt = LLM_REQUEST_ATTEMPT.set(3)
    kind = LLM_REQUEST_KIND.set("plan")
    try:
        await streamed(llm, [TOOL])
        await llm.chat(ASK, [TOOL])
    finally:
        LLM_REQUEST_KIND.reset(kind)
        LLM_REQUEST_ATTEMPT.reset(attempt)
        await llm.close()
    asked, again, plain = server.bodies
    assert asked["stream"] is True and "stream" not in again
    assert again == plain and list(again) == list(plain)  # as chat() sends it
    assert again["k"] == 1  # the call kind's extra fields stay
    for headers in server.headers:
        assert headers["X-Localmelo-Attempt"] == "3"
        assert headers["X-Localmelo-Call-Kind"] == "plan"


@pytest.mark.asyncio
async def test_a_stream_cut_before_any_data_is_asked_again() -> None:
    class Cut(Bytes):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            raise httpx.RemoteProtocolError("peer closed connection")
            yield b""

    def answer(n: int, body: dict[str, Any]) -> httpx.Response:
        if body.get("stream"):
            return httpx.Response(200, stream=Cut([]))
        return httpx.Response(200, json=TEXT_ONLY)

    server = Recorder(answer)
    llm = provider(server)
    try:
        events = await streamed(llm)
    finally:
        await llm.close()
    assert server.streams == [True, False]
    assert events[0].reply and events[0].reply.content == "Plain answer."


@pytest.mark.asyncio
async def test_a_json_body_cut_half_way_is_asked_again() -> None:
    class Cut(Bytes):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"choices": [{"message": '
            raise httpx.RemoteProtocolError("peer closed connection")

    def answer(n: int, body: dict[str, Any]) -> httpx.Response:
        if body.get("stream"):
            return httpx.Response(200, stream=Cut([]))
        return httpx.Response(200, json=TEXT_ONLY)

    server = Recorder(answer)
    llm = provider(server)
    try:
        events = await streamed(llm)
    finally:
        await llm.close()
    assert server.streams == [True, False]
    assert events[0].reply and events[0].reply.content == "Plain answer."
    assert events[0].whole is True


@pytest.mark.asyncio
async def test_a_failed_second_request_raises_its_error_and_keeps_streaming() -> None:
    def answer(n: int, body: dict[str, Any]) -> httpx.Response:
        if body.get("stream"):
            return httpx.Response(200, content=b"")
        return httpx.Response(500, text="the model crashed")

    server = Recorder(answer)
    llm = provider(server)
    try:
        with pytest.raises(ProviderError) as failure:
            await streamed(llm)
        assert llm.sends_whole_replies is False
        with pytest.raises(ProviderError):
            await streamed(llm)
    finally:
        await llm.close()
    assert failure.value.status_code == 500 and "the model crashed" in str(
        failure.value
    )
    assert server.streams == [True, False, True, False]


@pytest.mark.asyncio
async def test_a_server_that_streams_is_still_asked_to_stream() -> None:
    server = Recorder(
        lambda n, body: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_answer("Hello."),
        )
    )
    llm = provider(server)
    try:
        first = await streamed(llm)
        second = await streamed(llm)
    finally:
        await llm.close()
    assert server.streams == [True, True]
    assert [event.kind for event in first] == ["content", "reply"]
    assert not any(event.whole for event in first + second)
    assert llm.sends_whole_replies is False


# ── What keeps its error ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delivered",
    [
        frame({"choices": [{"index": 0, "delta": {"content": "Partial line."}}]}),
        frame({"choices": [{"index": 0, "delta": {"reasoning_content": "hm"}}]}),
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "a", "function": {"name": "read"}}
                            ]
                        },
                    }
                ]
            }
        ),
        frame({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}),
    ],
    ids=["content", "reasoning", "tool call", "usage"],
)
async def test_a_stream_that_breaks_after_part_of_a_reply_keeps_its_error(
    delivered: bytes,
) -> None:
    server = Recorder(
        lambda n, body: httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=delivered
        )
    )
    llm = provider(server)
    seen: list[StreamFragment] = []
    try:
        with pytest.raises(ProviderError) as failure:
            async for fragment in llm.stream_chat(ASK):
                seen.append(fragment)
    finally:
        await llm.close()
    assert str(failure.value) == "Streaming response ended before completion"
    assert not failure.value.retryable
    assert seen and all(fragment.reply is None for fragment in seen)
    assert server.streams == [True]  # not asked again
    assert llm.sends_whole_replies is False


@pytest.mark.asyncio
async def test_a_body_cut_after_part_of_a_reply_keeps_its_error() -> None:
    class Cut(Bytes):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield frame({"choices": [{"index": 0, "delta": {"content": "Part"}}]})
            raise httpx.RemoteProtocolError("peer closed connection")

    server = Recorder(lambda n, body: httpx.Response(200, stream=Cut([])))
    llm = provider(server)
    try:
        with pytest.raises(ProviderError) as failure:
            await streamed(llm)
    finally:
        await llm.close()
    assert str(failure.value) == "Streaming request failed: RemoteProtocolError"
    assert not failure.value.retryable
    assert server.streams == [True]


@pytest.mark.asyncio
async def test_a_server_that_cannot_be_reached_is_asked_once() -> None:
    calls: list[int] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("All connection attempts failed")

    llm = provider(refuse)
    try:
        with pytest.raises(ProviderError, match="cannot reach") as failure:
            await streamed(llm)
    finally:
        await llm.close()
    assert failure.value.retryable and calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("timeout"), httpx.ReadError("reset")],
    ids=["timeout", "reset"],
)
async def test_a_failure_before_any_data_is_not_asked_again(
    error: httpx.TransportError,
) -> None:
    class Broken(Bytes):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            raise error
            yield b""

    server = Recorder(lambda n, body: httpx.Response(200, stream=Broken([])))
    llm = provider(server)
    try:
        with pytest.raises(ProviderError) as failure:
            await streamed(llm)
    finally:
        await llm.close()
    assert failure.value.retryable
    assert server.streams == [True]
    assert llm.sends_whole_replies is False


@pytest.mark.asyncio
async def test_an_error_status_is_not_asked_again() -> None:
    server = Recorder(lambda n, body: httpx.Response(503, text="busy"))
    llm = provider(server)
    try:
        with pytest.raises(ProviderError) as failure:
            await streamed(llm)
    finally:
        await llm.close()
    assert failure.value.status_code == 503 and failure.value.retryable
    assert server.streams == [True]


@pytest.mark.asyncio
async def test_a_stream_of_garbage_is_not_asked_again() -> None:
    server = Recorder(
        lambda n, body: httpx.Response(200, content=b"data: not-json\n\n")
    )
    llm = provider(server)
    try:
        with pytest.raises(ProviderError, match="Malformed SSE JSON"):
            await streamed(llm)
    finally:
        await llm.close()
    assert server.streams == [True]


# ── chat() is the request it always was ──


@pytest.mark.asyncio
async def test_chat_sends_the_same_body_as_before() -> None:
    server = Recorder(lambda n, body: httpx.Response(200, json=TEXT_ONLY))
    llm = provider(
        server, extra_body={"temperature": 0}, extra_body_by_kind={"plan": {"k": 1}}
    )
    kind = LLM_REQUEST_KIND.set("plan")
    try:
        await llm.chat(ASK, [TOOL])
        LLM_REQUEST_KIND.reset(kind)
        await llm.chat(ASK, [TOOL])
        await llm.chat(ASK)
    finally:
        await llm.close()
    planned, with_tools, without = server.bodies
    assert list(planned) == [
        "temperature",
        "k",
        "model",
        "messages",
        "tools",
        "tool_choice",
    ]
    assert list(with_tools) == [
        "temperature",
        "model",
        "messages",
        "tools",
        "tool_choice",
    ]
    assert list(without) == ["temperature", "model", "messages"]
    assert not any("stream" in body for body in server.bodies)
