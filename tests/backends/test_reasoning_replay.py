"""R2's provider reasoning capability and wire form — no network.

The pinned "thinking stays local" tests in ``test_openai_compat_llm.py`` and
``test_tool_protocol.py`` are the default (``reasoning_field=None``) case and
stay unchanged; these cover the explicit keys and Ollama's fixed field.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from localmelo.melo.schema import Message, ToolCall
from localmelo.support.backends.local.ollama import OllamaBackend
from localmelo.support.backends.openai_compat import build_openai_chat
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


class Server:
    """Records OpenAI request bodies and answers with scripted messages."""

    def __init__(self, *messages: dict[str, Any]) -> None:
        self.replies: deque[dict[str, Any]] = deque(messages)
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        message = (
            self.replies.popleft()
            if self.replies
            else {"role": "assistant", "content": "ok"}
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )

    def llm(self, **kwargs: Any) -> OpenAICompatLLM:
        return OpenAICompatLLM(
            base_url="http://stub.test/v1",
            model="stub-model",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


class OllamaServer:
    """Records Ollama native request bodies and answers with scripted messages."""

    def __init__(self, *messages: dict[str, Any]) -> None:
        self.replies: deque[dict[str, Any]] = deque(messages)
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        message = (
            self.replies.popleft()
            if self.replies
            else {"role": "assistant", "content": "ok"}
        )
        return httpx.Response(
            200, json={"message": message, "prompt_eval_count": 3, "eval_count": 2}
        )

    def llm(self, **kwargs: Any) -> OllamaNativeChat:
        return OllamaNativeChat(
            base_url="http://stub.test",
            model="stub-model",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


def _call(call_id: str = "call_1") -> ToolCall:
    return ToolCall(tool_name="echo", arguments={"text": "a"}, id=call_id)


@pytest.mark.asyncio
async def test_reasoning_content_is_the_only_key_sent() -> None:
    server = Server()
    llm = server.llm(reasoning_field="reasoning_content")
    await llm.chat(
        [
            Message(role="assistant", content="plain", thinking="plain thought"),
            Message(
                role="assistant",
                content="",
                tool_calls=[_call()],
                thinking="tool thought",
            ),
        ]
    )
    messages = server.bodies[0]["messages"]
    assert messages[0] == {
        "role": "assistant",
        "content": "plain",
        "reasoning_content": "plain thought",
    }
    assert messages[1]["reasoning_content"] == "tool thought"
    assert "reasoning" not in messages[0] and "reasoning" not in messages[1]
    await llm.close()


@pytest.mark.asyncio
async def test_reasoning_is_the_only_key_sent() -> None:
    server = Server()
    llm = server.llm(reasoning_field="reasoning")
    await llm.chat([Message(role="assistant", content="a", thinking="t")])
    message = server.bodies[0]["messages"][0]
    assert message["reasoning"] == "t"
    assert "reasoning_content" not in message
    await llm.close()


@pytest.mark.asyncio
async def test_auto_follows_the_key_the_last_reply_used() -> None:
    server = Server(
        {"role": "assistant", "content": "a", "reasoning_content": "from content"},
        {"role": "assistant", "content": "b", "reasoning": "from reasoning"},
    )
    llm = server.llm(reasoning_field="auto")
    await llm.chat([Message(role="user", content="q")])
    await llm.chat([Message(role="assistant", content="a", thinking="think a")])
    await llm.chat([Message(role="assistant", content="b", thinking="think b")])
    assert "reasoning_content" not in server.bodies[0]["messages"][0]
    assert server.bodies[1]["messages"][0]["reasoning_content"] == "think a"
    assert "reasoning" not in server.bodies[1]["messages"][0]
    assert server.bodies[2]["messages"][0]["reasoning"] == "think b"
    assert "reasoning_content" not in server.bodies[2]["messages"][0]
    await llm.close()


@pytest.mark.asyncio
async def test_auto_sends_no_key_before_a_reply_or_for_empty_thinking() -> None:
    server = Server({"role": "assistant", "content": "a"})
    llm = server.llm(reasoning_field="auto")
    await llm.chat(
        [
            Message(role="assistant", content="empty", thinking=""),
            Message(role="assistant", content="has", thinking="t"),
        ]
    )
    first, second = server.bodies[0]["messages"]
    assert "reasoning_content" not in first and "reasoning" not in first
    assert "reasoning_content" not in second and "reasoning" not in second
    await llm.close()


@pytest.mark.asyncio
async def test_default_sends_no_reasoning_key() -> None:
    server = Server()
    llm = server.llm()
    await llm.chat([Message(role="assistant", content="a", thinking="t")])
    assert "reasoning_content" not in server.bodies[0]["messages"][0]
    assert "reasoning" not in server.bodies[0]["messages"][0]
    await llm.close()


def test_an_unknown_reasoning_field_raises() -> None:
    with pytest.raises(ValueError, match="reasoning_field"):
        OpenAICompatLLM(
            base_url="http://stub.test/v1", model="m", reasoning_field="thinking"
        )


@pytest.mark.asyncio
async def test_ollama_sends_thinking_and_removes_the_inline_copy() -> None:
    server = OllamaServer()
    llm = server.llm()
    assert llm.reasoning_field == "thinking"
    await llm.chat(
        [
            Message(
                role="assistant", content="<think>\nt\n</think>\n\nanswer", thinking="t"
            )
        ]
    )
    message = server.bodies[0]["messages"][0]
    assert message["content"] == "answer"
    assert message["thinking"] == "t"
    await llm.chat([Message(role="assistant", content="plain")])
    assert "thinking" not in server.bodies[1]["messages"][0]
    await llm.close()


@pytest.mark.asyncio
async def test_last_usage_and_last_reasoning_key_are_kept_whole() -> None:
    server = Server(
        {
            "role": "assistant",
            "content": "a",
            "reasoning_content": "t",
        }
    )

    # A handler whose usage reports the details R2's JSON line exposes.
    def handler(request: httpx.Request) -> httpx.Response:
        server.bodies.append(json.loads(request.content))
        server.replies.popleft()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "a",
                            "reasoning_content": "t",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": 6},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            },
        )

    llm = OpenAICompatLLM(
        base_url="http://stub.test/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        reasoning_field="auto",
    )
    await llm.chat([Message(role="user", content="q")])
    assert llm.last_reasoning_key == "reasoning_content"
    assert llm.last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "cached_tokens": 6,
        "reasoning_tokens": 3,
    }
    await llm.close()


@pytest.mark.asyncio
async def test_the_backend_builders_pass_the_two_keys_through() -> None:
    provider = build_openai_chat(
        "http://stub.test/v1",
        "m",
        reasoning_field="reasoning_content",
        requires_reasoning_replay=True,
    )
    assert isinstance(provider, OpenAICompatLLM)
    assert provider.reasoning_field == "reasoning_content"
    assert provider.requires_reasoning_replay is True
    await provider.close()

    cfg = Config(
        chat_backend="ollama",
        embedding_backend="none",
        ollama=LocalBackendConfig(
            chat_url="http://stub.test",
            chat_model="m",
            requires_reasoning_replay=True,
        ),
    )
    ollama = OllamaBackend().build_chat_provider(cfg)
    assert isinstance(ollama, OllamaNativeChat)
    assert ollama.reasoning_field == "thinking"
    assert ollama.requires_reasoning_replay is True
    await ollama.close()
