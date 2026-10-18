"""OpenAICompatLLM wire-format tests over a fake HTTP transport (no network)."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from localmelo.melo.schema import Message, ToolCall, ToolDef, ToolResult
from localmelo.support.providers.llm._protocol import RAW_ARGUMENTS_KEY
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

# ── Fake transport ─────────────────────────────────────────────────────────


class FakeServer:
    """Records request bodies and replies with queued ``message`` objects."""

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
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        return httpx.Response(200, json=body)

    def llm(self, **kwargs: Any) -> OpenAICompatLLM:
        return OpenAICompatLLM(
            base_url="http://omlx.test/v1",
            model="Qwen3.8-27B-8bit",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


def _tool_call_entry(call_id: str, name: str, args: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


ECHO = ToolDef(
    name="echo",
    description="echo text",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)


# ── Request serialization ──────────────────────────────────────────────────


class TestRequestSerialization:
    @pytest.mark.asyncio
    async def test_plain_messages_only_role_and_content(self) -> None:
        server = FakeServer()
        llm = server.llm()
        await llm.chat(
            [
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello", thinking="hmm"),
            ]
        )
        assert server.paths == ["/v1/chat/completions"]
        assert server.requests[0]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        await llm.close()

    @pytest.mark.asyncio
    async def test_multi_turn_tool_conversation(self) -> None:
        """Assistant tool calls and tool results are sent in OpenAI form."""
        server = FakeServer()
        llm = server.llm()
        first = ToolCall(tool_name="echo", arguments={"text": "a"}, id="call_1")
        second = ToolCall(tool_name="echo", arguments={"text": "b"}, id="call_2")
        third = ToolCall(tool_name="file_read", arguments={"path": "x"}, id="call_3")
        history = [
            Message(role="system", content="sys"),
            Message(role="user", content="do things"),
            Message(
                role="assistant",
                content="",
                tool_calls=[first, second],
                thinking="internal reasoning",
                usage={"total_tokens": 9},
            ),
            Message(
                role="tool",
                content="a",
                tool_call_id="call_1",
                tool_result=ToolResult(tool_name="echo", output="a"),
            ),
            Message(role="tool", content="b", tool_call_id="call_2"),
            Message(role="assistant", content="now reading", tool_call=third),
            Message(role="tool", content="Error: missing", tool_call_id="call_3"),
        ]
        await llm.chat(history, tools=[ECHO])

        sent = server.requests[0]["messages"]
        assert sent == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do things"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call_entry("call_1", "echo", '{"text": "a"}'),
                    _tool_call_entry("call_2", "echo", '{"text": "b"}'),
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            {
                "role": "assistant",
                "content": "now reading",
                "tool_calls": [
                    _tool_call_entry("call_3", "file_read", '{"path": "x"}')
                ],
            },
            {"role": "tool", "tool_call_id": "call_3", "content": "Error: missing"},
        ]
        await llm.close()

    @pytest.mark.asyncio
    async def test_every_tool_result_answers_a_preceding_call(self) -> None:
        server = FakeServer()
        llm = server.llm()
        calls = [
            ToolCall(tool_name="echo", arguments={"text": str(i)}) for i in range(3)
        ]
        history = [
            Message(role="user", content="q"),
            Message(role="assistant", tool_calls=calls),
            *[
                Message(role="tool", content=str(i), tool_call_id=tc.id)
                for i, tc in enumerate(calls)
            ],
        ]
        await llm.chat(history)
        sent = server.requests[0]["messages"]
        call_ids = [tc["id"] for tc in sent[1]["tool_calls"]]
        result_ids = [m["tool_call_id"] for m in sent[2:]]
        assert call_ids == result_ids == [tc.id for tc in calls]
        await llm.close()

    @pytest.mark.asyncio
    async def test_arguments_are_json_strings_and_keep_unicode(self) -> None:
        server = FakeServer()
        llm = server.llm()
        cjk = "\u4f60\u597d\uff0c\u4e16\u754c"  # non-ASCII sample text
        args = {"path": "chapter-1.md", "content": cjk, "n": 3, "ok": True}
        await llm.chat(
            [
                Message(role="user", content="write it"),
                Message(
                    role="assistant",
                    tool_call=ToolCall(tool_name="file_write", arguments=args),
                ),
                Message(role="tool", content="done", tool_call_id="x"),
            ]
        )
        encoded = server.requests[0]["messages"][1]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert isinstance(encoded, str)
        assert cjk in encoded  # not \u-escaped inside the arguments string
        assert json.loads(encoded) == args
        await llm.close()

    @pytest.mark.asyncio
    async def test_tools_and_tool_choice_only_with_tools(self) -> None:
        server = FakeServer()
        llm = server.llm()
        await llm.chat([Message(role="user", content="a")], tools=[ECHO])
        await llm.chat([Message(role="user", content="b")])
        with_tools, without_tools = server.requests
        assert with_tools["tool_choice"] == "auto"
        assert with_tools["tools"][0]["function"]["name"] == "echo"
        assert "tools" not in without_tools
        assert "tool_choice" not in without_tools
        await llm.close()

    @pytest.mark.asyncio
    async def test_tool_message_without_id_keeps_legacy_shape(self) -> None:
        server = FakeServer()
        llm = server.llm()
        await llm.chat([Message(role="tool", content="[echo] hi")])
        assert server.requests[0]["messages"] == [
            {"role": "tool", "content": "[echo] hi"}
        ]
        await llm.close()


# ── Response parsing ───────────────────────────────────────────────────────


class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_all_tool_calls_parsed_with_ids(self) -> None:
        server = FakeServer(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call_entry("call_a", "echo", '{"text": "1"}'),
                    _tool_call_entry("call_b", "echo", '{"text": "2"}'),
                    _tool_call_entry("call_c", "file_read", '{"path": "f"}'),
                ],
            }
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")], tools=[ECHO])
        assert reply.content == ""
        assert [(tc.id, tc.tool_name, tc.arguments) for tc in reply.tool_calls] == [
            ("call_a", "echo", {"text": "1"}),
            ("call_b", "echo", {"text": "2"}),
            ("call_c", "file_read", {"path": "f"}),
        ]
        assert reply.tool_call is reply.tool_calls[0]
        assert reply.usage == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        await llm.close()

    @pytest.mark.asyncio
    async def test_missing_and_duplicate_ids_are_replaced(self) -> None:
        server = FakeServer(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "a", "arguments": "{}"}},
                    _tool_call_entry("", "b", "{}"),
                    _tool_call_entry("call_dup", "c", "{}"),
                    _tool_call_entry("call_dup", "d", "{}"),
                ],
            }
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")])
        ids = [tc.id for tc in reply.tool_calls]
        assert [tc.tool_name for tc in reply.tool_calls] == ["a", "b", "c", "d"]
        assert all(ids)
        assert len(set(ids)) == 4
        assert ids[2] == "call_dup"
        await llm.close()

    @pytest.mark.asyncio
    async def test_argument_variants(self) -> None:
        server = FakeServer(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call_entry("c1", "empty", ""),
                    {"id": "c2", "function": {"name": "missing"}},
                    {"id": "c3", "function": {"name": "obj", "arguments": {"k": 1}}},
                    _tool_call_entry("c4", "bad", "{not json"),
                    _tool_call_entry("c5", "list", "[1, 2]"),
                ],
            }
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")])
        args = {tc.tool_name: tc.arguments for tc in reply.tool_calls}
        assert args["empty"] == {}
        assert args["missing"] == {}
        assert args["obj"] == {"k": 1}
        # Not a JSON object: kept raw instead of crashing the run.
        assert args["bad"] == {RAW_ARGUMENTS_KEY: "{not json"}
        assert args["list"] == {RAW_ARGUMENTS_KEY: "[1, 2]"}
        await llm.close()

    @pytest.mark.asyncio
    async def test_call_without_function_name_is_kept_unnamed(self) -> None:
        server = FakeServer(
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")])
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].tool_name == ""
        await llm.close()

    @pytest.mark.asyncio
    async def test_reasoning_content_fills_thinking(self) -> None:
        server = FakeServer(
            {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "6 times 7 is 42.",
            }
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="6*7?")])
        assert reply.thinking == "6 times 7 is 42."
        # The answer text is left as the server sent it.
        assert reply.content == "The answer is 42."
        await llm.close()

    @pytest.mark.asyncio
    async def test_reasoning_field_fallback(self) -> None:
        server = FakeServer(
            {"role": "assistant", "content": "ok", "reasoning": "short thought"}
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")])
        assert reply.thinking == "short thought"
        await llm.close()

    @pytest.mark.asyncio
    async def test_reasoning_with_tool_calls(self) -> None:
        server = FakeServer(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should echo.",
                "tool_calls": [_tool_call_entry("call_r", "echo", '{"text": "x"}')],
            }
        )
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")], tools=[ECHO])
        assert reply.thinking == "I should echo."
        assert reply.tool_calls[0].id == "call_r"
        await llm.close()

    @pytest.mark.asyncio
    async def test_no_reasoning_means_empty_thinking(self) -> None:
        server = FakeServer({"role": "assistant", "content": "plain"})
        llm = server.llm()
        reply = await llm.chat([Message(role="user", content="q")])
        assert reply.thinking == ""
        assert reply.tool_calls == []
        assert reply.tool_call is None
        await llm.close()


# ── Round trip ─────────────────────────────────────────────────────────────


class TestToolCallIdRoundTrip:
    @pytest.mark.asyncio
    async def test_server_ids_come_back_on_the_next_request(self) -> None:
        """Ids the server assigned are replayed on the call and its result."""
        server = FakeServer(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "two lookups",
                "tool_calls": [
                    _tool_call_entry("chatcmpl-tool-1", "echo", '{"text": "a"}'),
                    _tool_call_entry("chatcmpl-tool-2", "echo", '{"text": "b"}'),
                ],
            },
            {"role": "assistant", "content": "a b"},
        )
        llm = server.llm()
        history = [Message(role="user", content="echo a and b")]
        reply = await llm.chat(history, tools=[ECHO])
        history.append(reply)
        for tc in reply.tool_calls:
            history.append(
                Message(
                    role="tool",
                    content=tc.arguments["text"],
                    tool_call_id=tc.id,
                )
            )
        final = await llm.chat(history, tools=[ECHO])
        assert final.content == "a b"

        second = server.requests[1]["messages"]
        assert second[1] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call_entry("chatcmpl-tool-1", "echo", '{"text": "a"}'),
                _tool_call_entry("chatcmpl-tool-2", "echo", '{"text": "b"}'),
            ],
        }
        assert second[2] == {
            "role": "tool",
            "tool_call_id": "chatcmpl-tool-1",
            "content": "a",
        }
        assert second[3] == {
            "role": "tool",
            "tool_call_id": "chatcmpl-tool-2",
            "content": "b",
        }
        # Reasoning stays local: it is not part of the OpenAI request format.
        assert "reasoning_content" not in json.dumps(second)
        await llm.close()
