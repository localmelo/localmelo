"""OllamaNativeChat tool-call wire format over a fake HTTP transport."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from localmelo.melo.schema import Message, ToolCall, ToolResult
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat


class FakeOllama:
    """Records ``/api/chat`` bodies and replies with queued messages."""

    def __init__(self, *messages: dict[str, Any]) -> None:
        self.replies: deque[dict[str, Any]] = deque(messages)
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.requests.append(json.loads(request.content))
        message = (
            self.replies.popleft()
            if self.replies
            else {"role": "assistant", "content": "ok"}
        )
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": message,
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        )

    def llm(self, **kwargs: Any) -> OllamaNativeChat:
        return OllamaNativeChat(
            base_url="http://ollama.test",
            model="qwen3:8b",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


class TestOllamaRequestSerialization:
    @pytest.mark.asyncio
    async def test_tool_turn_is_replayed_natively(self) -> None:
        server = FakeOllama()
        llm = server.llm()
        call_a = ToolCall(tool_name="echo", arguments={"text": "a"}, id="call_a")
        call_b = ToolCall(tool_name="file_read", arguments={"path": "x"}, id="call_b")
        thinking = "I will echo and read."
        history = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                # What OllamaNativeChat itself returns: thinking inlined.
                content=f"<think>\n{thinking}\n</think>\n\n",
                thinking=thinking,
                tool_calls=[call_a, call_b],
            ),
            Message(
                role="tool",
                content="a",
                tool_call_id="call_a",
                tool_result=ToolResult(tool_name="echo", output="a"),
            ),
            # No tool_result: the name is looked up through the call id.
            Message(role="tool", content="data", tool_call_id="call_b"),
        ]
        await llm.chat(history)

        assert server.paths == ["/api/chat"]
        body = server.requests[0]
        assert body["stream"] is False
        assert body["think"] is True
        assert body["messages"] == [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "thinking": thinking,
                "tool_calls": [
                    {"function": {"name": "echo", "arguments": {"text": "a"}}},
                    {"function": {"name": "file_read", "arguments": {"path": "x"}}},
                ],
            },
            {"role": "tool", "content": "a", "tool_name": "echo"},
            {"role": "tool", "content": "data", "tool_name": "file_read"},
        ]
        await llm.close()

    @pytest.mark.asyncio
    async def test_plain_messages_unchanged(self) -> None:
        server = FakeOllama()
        llm = server.llm()
        await llm.chat(
            [
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ]
        )
        assert server.requests[0]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        await llm.close()


class TestOllamaResponseParsing:
    @pytest.mark.asyncio
    async def test_all_tool_calls_parsed(self) -> None:
        server = FakeOllama(
            {
                "role": "assistant",
                "content": "",
                "thinking": "two calls",
                "tool_calls": [
                    {"function": {"name": "echo", "arguments": {"text": "1"}}},
                    {
                        "id": "call_x",
                        "function": {"name": "echo", "arguments": '{"text": "2"}'},
                    },
                ],
            }
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")])
        assert [tc.arguments for tc in reply.tool_calls] == [
            {"text": "1"},
            {"text": "2"},
        ]
        assert reply.tool_calls[0].id.startswith("call_")
        assert reply.tool_calls[1].id == "call_x"
        assert reply.thinking == "two calls"
        # Existing behavior: thinking is also inlined into content.
        assert reply.content.startswith("<think>")
        await llm.close()

    @pytest.mark.asyncio
    async def test_reply_round_trips_into_next_request(self) -> None:
        server = FakeOllama(
            {
                "role": "assistant",
                "content": "",
                "thinking": "use echo",
                "tool_calls": [
                    {"function": {"name": "echo", "arguments": {"text": "hi"}}}
                ],
            },
            {"role": "assistant", "content": "hi"},
        )
        llm = server.llm()
        history = [Message(role="user", content="say hi")]
        reply = await llm.chat(history)
        history.append(reply)
        history.append(
            Message(role="tool", content="hi", tool_call_id=reply.tool_calls[0].id)
        )
        await llm.chat(history)
        replayed = server.requests[1]["messages"]
        assert replayed[1] == {
            "role": "assistant",
            "content": "",
            "thinking": "use echo",
            "tool_calls": [{"function": {"name": "echo", "arguments": {"text": "hi"}}}],
        }
        assert replayed[2] == {"role": "tool", "content": "hi", "tool_name": "echo"}
        await llm.close()
