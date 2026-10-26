"""The window codec: what a checkpoint keeps, redacts and drops."""

from __future__ import annotations

import json
import logging

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import ImagePart, Message, ToolCall, ToolDef, ToolResult
from localmelo.support.sessions.window import (
    WINDOW_CODEC_VERSION,
    decode_window,
    encode_window,
    window_thinking_since,
)

SECRET = "sk-" + "0123456789abcdef" * 2


class NoModel(BaseLLMProvider):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        raise AssertionError("no request is made")

    async def close(self) -> None:
        pass


def a_window() -> list[Message]:
    image = ImagePart.from_bytes(b"\x89PNG screenshot", "image/png")
    call = ToolCall(tool_name="file_read", arguments={"path": "notes.txt"}, id="call_1")
    return [
        Message(role="user", content="Read my notes", timestamp=100.0),
        Message(
            role="assistant",
            content="Let me look.",
            thinking="the notes are in notes.txt",
            tool_calls=[call],
            timestamp=101.0,
            usage={"prompt_tokens": 12, "completion_tokens": 3},
        ),
        Message(
            role="tool",
            content="Tool file_read output:\nbuy milk [1 image]",
            tool_call_id="call_1",
            tool_result=ToolResult(
                tool_name="file_read",
                output="buy milk",
                error="",
                duration_ms=4.5,
                images=[image],
            ),
            images=[image.as_reference()],
            timestamp=102.0,
        ),
        Message(role="assistant", content="Buy milk.", timestamp=103.0),
    ]


def test_a_round_trip_keeps_every_field_a_request_needs() -> None:
    window = a_window()
    decoded = decode_window(encode_window(window))
    assert decoded is not None and len(decoded) == len(window)
    for before, after in zip(window, decoded, strict=True):
        assert after.role == before.role
        assert after.content == before.content
        assert after.thinking == before.thinking
        assert after.tool_call_id == before.tool_call_id
        assert after.timestamp == before.timestamp
        assert after.usage == before.usage
        assert [(c.id, c.tool_name, c.arguments) for c in after.tool_calls] == [
            (c.id, c.tool_name, c.arguments) for c in before.tool_calls
        ]
        assert [(i.mime_type, i.sha256, i.size_bytes) for i in after.images] == [
            (i.mime_type, i.sha256, i.size_bytes) for i in before.images
        ]
    result = decoded[2].tool_result
    assert result is not None
    assert (result.tool_name, result.error, result.duration_ms) == (
        "file_read",
        "",
        4.5,
    )
    # No tool output, and images by reference only.
    assert result.output == ""
    assert result.images[0].data == "" and result.images[0].sha256
    assert result.images[0].size_bytes == len(b"\x89PNG screenshot")
    text = encode_window(window)
    assert 'buy milk"' not in text  # the output is not stored
    assert window[2].tool_result.images[0].data not in text  # type: ignore[union-attr]
    data = json.loads(text)
    assert data["v"] == WINDOW_CODEC_VERSION == 1
    assert set(data["messages"][1]) == {
        "role",
        "content",
        "thinking",
        "tool_calls",
        "tool_call_id",
        "tool_result",
        "images",
        "timestamp",
        "usage",
    }


def test_assistant_text_is_redacted_and_user_text_kept() -> None:
    window = [
        Message(role="user", content=f"my key is {SECRET}"),
        Message(
            role="assistant",
            content=f"Stored {SECRET}.",
            thinking=f"the key {SECRET} goes in the file",
        ),
    ]
    decoded = decode_window(encode_window(window))
    assert decoded is not None
    assert decoded[0].content == f"my key is {SECRET}"
    assert decoded[1].content == "Stored [redacted]."
    assert decoded[1].thinking == "the key [redacted] goes in the file"
    assert window[1].content == f"Stored {SECRET}."  # the live window is unchanged


@pytest.mark.parametrize(
    ("content", "stripped"),
    [
        ("<think>plan</think>Answer.", "Answer."),
        ("plan only the template opened</think>Answer.", "Answer."),
        ("  <think>cut off while thinking", ""),
        ("No block here.", "No block here."),
    ],
)
def test_thinking_older_than_the_cutoff_is_cleared(content: str, stripped: str) -> None:
    window = [
        Message(role="user", content="q", timestamp=10.0),
        Message(role="assistant", content=content, thinking="old", timestamp=10.0),
        Message(role="user", content="q2", timestamp=50.0),
        Message(role="assistant", content=content, thinking="new", timestamp=50.0),
    ]
    decoded = decode_window(encode_window(window, thinking_cutoff=20.0))
    assert decoded is not None and len(decoded) == 4  # no message is dropped
    assert (decoded[1].thinking, decoded[1].content) == ("", stripped)
    assert (decoded[3].thinking, decoded[3].content) == ("new", content)
    assert window_thinking_since(window, thinking_cutoff=20.0) == 50.0
    assert window_thinking_since(window) == 10.0
    assert window_thinking_since(window[:2], thinking_cutoff=20.0) is None


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"v": 2, "messages": []}),
        json.dumps({"messages": []}),
        "{not json",
        json.dumps({"v": 1, "messages": [{"role": "user", "content": 3}]}),
        json.dumps({"v": 1, "messages": "x"}),
        json.dumps([1, 2]),
    ],
)
def test_another_version_or_a_broken_row_decodes_to_none(text: str) -> None:
    assert decode_window(text) is None


def test_a_decoded_window_loads_without_a_warning(
    tmp_path: object, caplog: pytest.LogCaptureFixture
) -> None:
    agent = Agent(llm=NoModel(), options=AgentOptions(record_answers=True))
    decoded = decode_window(encode_window(a_window()))
    assert decoded is not None
    caplog.set_level(logging.WARNING)
    assert agent.load_conversation(decoded) == 4
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [m.role for m in agent.conversation()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
