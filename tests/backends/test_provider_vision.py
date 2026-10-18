"""Multimodal request bodies for the chat providers (fake HTTP, no server)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from localmelo.melo.schema import ImagePart, Message, ToolCall, ToolResult
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

SHOT = ImagePart.from_bytes(b"screenshot bytes", "image/png")
SHOT2 = ImagePart.from_bytes(b"second screenshot", "image/jpeg")


class Recorder:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.reply)


async def _openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    rec = Recorder({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    llm = OpenAICompatLLM(
        base_url="http://omlx.test/v1",
        model="m",
        transport=httpx.MockTransport(rec.handler),
    )
    await llm.chat(messages)
    await llm.close()
    result: list[dict[str, Any]] = rec.bodies[0]["messages"]
    return result


def _tool_turn(
    *results: tuple[str, str, list[ImagePart]],
) -> list[Message]:
    calls = [ToolCall(tool_name=name, id=call_id) for call_id, name, _ in results]
    turn = [Message(role="assistant", tool_calls=calls)]
    for call_id, name, images in results:
        turn.append(
            Message(
                role="tool",
                content=f"{name} done",
                tool_call_id=call_id,
                images=images,
                tool_result=ToolResult(tool_name=name, images=images),
            )
        )
    return turn


class TestOpenAIUserImages:
    @pytest.mark.asyncio
    async def test_user_message_becomes_content_parts(self) -> None:
        sent = await _openai_messages(
            [Message(role="user", content="what is this?", images=[SHOT])]
        )
        assert sent == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": SHOT.data_url}},
                ],
            }
        ]
        assert sent[0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )

    @pytest.mark.asyncio
    async def test_image_only_user_message(self) -> None:
        sent = await _openai_messages([Message(role="user", images=[SHOT, SHOT2])])
        assert [p["type"] for p in sent[0]["content"]] == ["image_url", "image_url"]

    @pytest.mark.asyncio
    async def test_reference_only_images_are_not_sent(self) -> None:
        sent = await _openai_messages(
            [Message(role="user", content="old", images=[SHOT.as_reference()])]
        )
        assert sent == [{"role": "user", "content": "old"}]


class TestOpenAIToolImages:
    @pytest.mark.asyncio
    async def test_tool_images_follow_in_a_user_message(self) -> None:
        sent = await _openai_messages(
            [
                Message(role="user", content="take a screenshot"),
                *_tool_turn(("call_s", "mcp__cua__screenshot", [SHOT])),
            ]
        )
        assert sent[2] == {
            "role": "tool",
            "tool_call_id": "call_s",
            "content": "mcp__cua__screenshot done",
        }
        assert sent[3]["role"] == "user"
        label, image = sent[3]["content"]
        assert label["type"] == "text"
        assert "call_s" in label["text"]
        assert "mcp__cua__screenshot" in label["text"]
        assert SHOT.ref in label["text"]
        assert image == {"type": "image_url", "image_url": {"url": SHOT.data_url}}
        assert len(sent) == 4

    @pytest.mark.asyncio
    async def test_images_wait_until_all_tool_results_are_sent(self) -> None:
        """No user message may sit between tool results of one turn."""
        sent = await _openai_messages(
            [
                Message(role="user", content="q"),
                *_tool_turn(
                    ("call_a", "shot", [SHOT]),
                    ("call_b", "echo", []),
                    ("call_c", "shot", [SHOT2]),
                ),
                Message(role="assistant", content="done"),
            ]
        )
        roles = [m["role"] for m in sent]
        assert roles == [
            "user",
            "assistant",
            "tool",
            "tool",
            "tool",
            "user",
            "user",
            "assistant",
        ]
        assert [m["tool_call_id"] for m in sent[2:5]] == ["call_a", "call_b", "call_c"]
        assert "call_a" in sent[5]["content"][0]["text"]
        assert "call_c" in sent[6]["content"][0]["text"]
        assert sent[6]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )

    @pytest.mark.asyncio
    async def test_tool_message_itself_stays_text_only(self) -> None:
        sent = await _openai_messages(_tool_turn(("call_s", "shot", [SHOT])))
        tool = next(m for m in sent if m["role"] == "tool")
        assert isinstance(tool["content"], str)

    @pytest.mark.asyncio
    async def test_reference_only_tool_images_add_nothing(self) -> None:
        sent = await _openai_messages(
            _tool_turn(("call_s", "shot", [SHOT.as_reference()]))
        )
        assert [m["role"] for m in sent] == ["assistant", "tool"]

    @pytest.mark.asyncio
    async def test_images_on_other_roles_are_not_sent(self) -> None:
        sent = await _openai_messages(
            [
                Message(role="system", content="sys", images=[SHOT]),
                Message(role="assistant", content="a", images=[SHOT]),
            ]
        )
        assert sent == [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "a"},
        ]


class TestOllamaImages:
    @pytest.mark.asyncio
    async def test_native_images_field(self) -> None:
        rec = Recorder({"message": {"role": "assistant", "content": "ok"}})
        llm = OllamaNativeChat(
            base_url="http://ollama.test",
            model="qwen3-vl",
            transport=httpx.MockTransport(rec.handler),
        )
        await llm.chat(
            [
                Message(role="user", content="look", images=[SHOT]),
                *_tool_turn(("call_s", "shot", [SHOT2])),
            ]
        )
        await llm.close()
        sent = rec.bodies[0]["messages"]
        assert sent[0] == {"role": "user", "content": "look", "images": [SHOT.data]}
        assert sent[2] == {"role": "tool", "content": "shot done", "tool_name": "shot"}
        assert sent[3]["role"] == "user"
        assert sent[3]["images"] == [SHOT2.data]
        assert "call_s" in sent[3]["content"]
