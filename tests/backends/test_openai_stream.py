from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.providers.llm._sse import sse_data
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


class Bytes(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed += 1


def frame(obj: object) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


def fixture() -> bytes:
    return (
        b": ping\r\n\r\n"
        + frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hi 世界", "reasoning_content": "think"},
                        "finish_reason": None,
                    }
                ]
            }
        )
        + frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "read", "arguments": '{"x":'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        + frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": " 3}"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        + frame({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}})
        + b"data: [DONE]\n\n"
    )


async def collect(chunks: list[bytes]) -> tuple[list, Bytes, dict]:
    body = Bytes(chunks)
    request: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        request.update(json.loads(req.content))
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=body
        )

    provider = OpenAICompatLLM(
        "http://example.test/v1", "test", transport=httpx.MockTransport(handler)
    )
    try:
        events = [
            event
            async for event in provider.stream_chat(
                [Message(role="user", content="hi")]
            )
        ]
        assert provider.last_usage == {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        }
        assert provider.last_reasoning_key == "reasoning_content"
        return events, body, request
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_stream_fixture_every_split() -> None:
    raw = fixture()
    for i in range(len(raw) + 1):
        events, body, request = await collect([raw[:i], raw[i:]])
        assert request["stream"] is True
        assert body.closed == 1
        assert [event.kind for event in events] == [
            "content",
            "reasoning",
            "tool",
            "tool",
            "usage",
            "reply",
        ]
        reply = events[-1].reply
        assert reply and reply.content == "Hi 世界" and reply.thinking == "think"
        assert reply.tool_calls[0].arguments == {"x": 3}
        assert events[-1].finish_reason == "tool_calls"
    await collect([bytes([byte]) for byte in raw])


@pytest.mark.asyncio
async def test_multiline_comments_and_cr() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for byte in b"\xef\xbb\xbf:comment\rdata: first\rdata: second\r\r":
            yield bytes([byte])

    assert [value async for value in sse_data(chunks())] == ["first\nsecond"]


@pytest.mark.asyncio
async def test_incomplete_stream_and_bad_json() -> None:
    for raw in (fixture().removesuffix(b"data: [DONE]\n\n"), b"data: not-json\n\n"):
        stream = Bytes([raw])
        provider = OpenAICompatLLM(
            "http://example.test/v1",
            "test",
            transport=httpx.MockTransport(
                lambda _, stream=stream: httpx.Response(200, stream=stream)
            ),
        )
        try:
            with pytest.raises(ProviderError):
                _ = [event async for event in provider.stream_chat([])]
            assert stream.closed == 1
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_early_close_releases_response() -> None:
    stream = Bytes([fixture()])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(
            lambda _, stream=stream: httpx.Response(200, stream=stream)
        ),
    )
    try:
        events = provider.stream_chat([])
        assert (await anext(events)).kind == "content"
        await events.aclose()
        assert stream.closed == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_cancellation_closes_response() -> None:
    entered = asyncio.Event()

    class Waiting(Bytes):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await asyncio.Event().wait()
            yield b""

    stream = Waiting([])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)),
    )

    async def consume() -> None:
        async for _ in provider.stream_chat([]):
            pass

    try:
        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments,finish",
    [
        ('{"x":', "tool_calls"),
        ("[]", "tool_calls"),
        ('{"x":1}', "length"),
        ('{"x":NaN}', "tool_calls"),
    ],
)
async def test_incomplete_tools_are_never_returned(arguments: str, finish: str) -> None:
    raw = (
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "a",
                                    "function": {
                                        "name": "read",
                                        "arguments": arguments,
                                    },
                                }
                            ]
                        },
                        "finish_reason": finish,
                    }
                ]
            }
        )
        + b"data: [DONE]\n\n"
    )
    stream = Bytes([raw])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)),
    )
    seen = []
    try:
        with pytest.raises(ProviderError):
            async for event in provider.stream_chat([]):
                seen.append(event)
        assert all(event.reply is None for event in seen)
        assert stream.closed == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_a_refused_stream_names_the_server() -> None:
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(_refuse),
    )
    try:
        with pytest.raises(ProviderError) as failure:
            _ = [event async for event in provider.stream_chat([])]
        assert failure.value.retryable
        assert str(failure.value) == (
            "cannot reach the model server at http://example.test/v1. "
            "Start it, or run melo to set one up. "
            "(ConnectError: All connection attempts failed)"
        )
    finally:
        await provider.close()


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("All connection attempts failed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("timeout"), httpx.ReadError("disconnect")]
)
async def test_transport_failure_closes_stream(error: httpx.TransportError) -> None:
    class Broken(Bytes):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield frame({"choices": [{"delta": {"content": "partial"}}]})
            raise error

    stream = Broken([])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)),
    )
    try:
        with pytest.raises(ProviderError) as failure:
            _ = [event async for event in provider.stream_chat([])]
        assert not failure.value.retryable
        assert stream.closed == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_http_errors_and_invalid_utf8() -> None:
    body = Bytes([b'{"error":{"code":"context_length_exceeded"}}'])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(lambda _: httpx.Response(400, stream=body)),
    )
    try:
        with pytest.raises(ProviderError) as failure:
            _ = [part async for part in provider.stream_chat([])]
        assert failure.value.context_overflow and not failure.value.retryable
        assert body.closed == 1
    finally:
        await provider.close()
    body = Bytes([b"data: \xff\n\n"])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body)),
    )
    try:
        with pytest.raises(ProviderError, match="UTF-8"):
            _ = [part async for part in provider.stream_chat([])]
        assert body.closed == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_unsupported_tool_type_is_rejected() -> None:
    raw = frame(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "other"}]}}]}
    )
    body = Bytes([raw])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body)),
    )
    try:
        with pytest.raises(ProviderError, match="tool type"):
            _ = [part async for part in provider.stream_chat([])]
        assert body.closed == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_empty_data_lines_count_toward_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.providers.llm import _sse

    monkeypatch.setattr(_sse, "_MAX_EVENT_CHARS", 8)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"data:\n" * 9

    with pytest.raises(ProviderError, match="size limit"):
        _ = [value async for value in sse_data(chunks())]


@pytest.mark.asyncio
async def test_error_body_is_bounded() -> None:
    class Many(Bytes):
        consumed = 0

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for _ in range(1000):
                self.consumed += 1
                yield b"x" * 8192

    body = Many([])
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(lambda _: httpx.Response(500, stream=body)),
    )
    try:
        with pytest.raises(ProviderError):
            _ = [part async for part in provider.stream_chat([])]
        assert body.consumed <= 9 and body.closed == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_interleaved_tools_and_repeated_identifiers() -> None:
    def chunk(calls: list[dict], finish: str | None = None) -> bytes:
        return frame(
            {"choices": [{"delta": {"tool_calls": calls}, "finish_reason": finish}]}
        )

    raw = (
        chunk(
            [
                {
                    "index": 0,
                    "id": "a",
                    "function": {"name": "read", "arguments": '{"x":'},
                },
                {"index": 1, "id": "b", "function": {"name": "noop"}},
            ]
        )
        + chunk(
            [
                {"index": 1, "id": "b", "function": {"name": "noop", "arguments": " "}},
                {
                    "index": 0,
                    "id": "a",
                    "function": {"name": "read", "arguments": "3}"},
                },
            ],
            "tool_calls",
        )
        + b"data: [DONE]\n\n"
    )
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=Bytes([raw]))
        ),
    )
    try:
        events = [event async for event in provider.stream_chat([])]
        reply = events[-1].reply
        assert reply is not None
        assert [
            (call.id, call.tool_name, call.arguments) for call in reply.tool_calls
        ] == [
            ("a", "read", {"x": 3}),
            ("b", "noop", {}),
        ]
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        frame({"error": {"message": "failed"}}),
        frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        + b"data: [DONE]\n\n",
        b'data: {"choices":',
    ],
)
async def test_error_frames_and_incomplete_calls(raw: bytes) -> None:
    provider = OpenAICompatLLM(
        "http://example.test/v1",
        "test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=Bytes([raw]))
        ),
    )
    try:
        with pytest.raises(ProviderError):
            _ = [event async for event in provider.stream_chat([])]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_usage_opt_in_and_compatible_final_frames() -> None:
    requests: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        requests.append(body)
        raw = frame({"choices": [{"finish_reason": "stop"}]})
        if body.get("stream_options", {}).get("include_usage"):
            raw += frame({"usage": {"prompt_tokens": 2, "completion_tokens": 3}})
        return httpx.Response(200, stream=Bytes([raw + b"data: [DONE]\n\n"]))

    for options in ({}, {"stream_options": {"include_usage": True}}):
        provider = OpenAICompatLLM(
            "http://example.test/v1",
            "test",
            extra_body=options,
            transport=httpx.MockTransport(handler),
        )
        try:
            events = [event async for event in provider.stream_chat([])]
            reply = events[-1].reply
            assert reply is not None
            assert (reply.usage is not None) == bool(options)
        finally:
            await provider.close()
    assert "stream_options" not in requests[0]
    assert requests[1]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_non_streaming_fallback() -> None:
    class Provider(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            return Message(role="assistant", content="whole")

        async def close(self) -> None:
            pass

    result = [part async for part in Provider().stream_chat([])]
    assert len(result) == 1 and result[0].reply and result[0].reply.content == "whole"
