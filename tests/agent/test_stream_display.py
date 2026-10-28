"""Chat's opt-in stream display seam: release, fallback, retry and cancel."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence

import pytest

from localmelo.melo.agent._chat import Chat
from localmelo.melo.contracts.providers import (
    BaseLLMProvider,
    ProviderError,
    StreamFragment,
)
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.schema import Message, ReasoningTags, ToolCall, ToolDef


class StreamLLM(BaseLLMProvider):
    """One scripted stream per request; an exception item raises in-stream."""

    def __init__(self, scripts: Sequence[Sequence[object]]) -> None:
        self.scripts = list(scripts)
        self.chat_calls = 0
        self.stream_calls = 0
        self.closed = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.chat_calls += 1
        return Message(role="assistant", content="complete")

    async def stream_chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> AsyncGenerator[StreamFragment, None]:
        self.stream_calls += 1
        script = self.scripts.pop(0)
        try:
            for item in script:
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, Message):
                    yield StreamFragment(kind="reply", reply=item)
                elif isinstance(item, StreamFragment):
                    yield item
                elif callable(item):
                    item()  # a probe that sees the display mid-stream
                else:
                    yield StreamFragment(kind="content", text=str(item))
        finally:
            self.closed += 1

    async def close(self) -> None:
        pass


class ChatOnlyLLM(BaseLLMProvider):
    """No streaming override: the contract's complete-reply fallback."""

    def __init__(self) -> None:
        self.chat_calls = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.chat_calls += 1
        return Message(role="assistant", content="complete reply")

    async def close(self) -> None:
        pass


class Display:
    def __init__(self, hint: ReasoningTags | None = None) -> None:
        self.reasoning_tags = hint
        self.events: list[tuple[str, str]] = []

    def opened(self) -> None:
        self.events.append(("opened", ""))

    def held(self) -> None:
        self.events.append(("held", ""))

    def content(self, delta: str) -> None:
        self.events.append(("content", delta))

    def closed(self) -> None:
        self.events.append(("closed", ""))

    def text(self) -> str:
        return "".join(value for kind, value in self.events if kind == "content")

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]


async def _drain(display: Display) -> str:
    return display.text()


async def test_a_declared_stream_releases_lines_before_the_reply() -> None:
    display = Display("none")
    seen: list[str] = []
    llm = StreamLLM(
        [
            [
                "Hello\n",
                lambda: seen.append(display.text()),
                "wor",
                "ld\n",
                Message(role="assistant", content="Hello\nworld\n"),
            ]
        ]
    )
    token = STREAM_DISPLAY.set(display)
    try:
        reply = await Chat(llm).plan_step([], [], [], "hi")
    finally:
        STREAM_DISPLAY.reset(token)
    assert reply.content == "Hello\nworld\n"
    assert seen == ["Hello\n"]  # released before the reply fragment
    assert display.text() == "Hello\nworld\n"
    assert display.kinds()[0] == "opened" and display.kinds()[-1] == "closed"
    assert llm.chat_calls == 0


async def test_an_undeclared_hint_releases_nothing_until_finish() -> None:
    display = Display(None)
    seen: list[str] = []
    llm = StreamLLM(
        [
            [
                "Hello\n",
                lambda: seen.append(display.text()),
                Message(role="assistant", content="Hello\n"),
            ]
        ]
    )
    token = STREAM_DISPLAY.set(display)
    try:
        await Chat(llm).plan_step([], [], [], "hi")
    finally:
        STREAM_DISPLAY.reset(token)
    assert seen == [""]
    assert display.text() == "Hello\n"


async def test_separate_reasoning_never_reaches_content() -> None:
    display = Display("none")
    llm = StreamLLM(
        [
            [
                StreamFragment(kind="reasoning", text="secret thought"),
                StreamFragment(kind="content", text="answer\n"),
                Message(
                    role="assistant", content="answer\n", thinking="secret thought"
                ),
            ]
        ]
    )
    token = STREAM_DISPLAY.set(display)
    try:
        await Chat(llm).plan_step([], [], [], "hi")
    finally:
        STREAM_DISPLAY.reset(token)
    assert display.text() == "answer\n"
    assert "secret thought" not in display.text()
    assert "held" in display.kinds()


async def test_no_display_keeps_the_complete_reply_path() -> None:
    llm = StreamLLM([[Message(role="assistant", content="unused")]])
    reply = await Chat(llm).plan_step([], [], [], "hi")
    assert reply.content == "complete"
    assert llm.chat_calls == 1 and llm.stream_calls == 0


async def test_a_provider_without_streaming_falls_back_silently() -> None:
    display = Display("none")
    llm = ChatOnlyLLM()
    token = STREAM_DISPLAY.set(display)
    try:
        reply = await Chat(llm).plan_step([], [], [], "hi")
    finally:
        STREAM_DISPLAY.reset(token)
    assert reply.content == "complete reply"
    assert await _drain(display) == ""
    assert display.kinds() == ["opened", "closed"]


async def test_split_closing_tag_never_releases_reasoning() -> None:
    display = Display("closing_only")
    content = "<think>secret plan</think>visible\n"
    llm = StreamLLM(
        [
            [
                "<thi",
                "nk>secret plan</thi",
                "nk>visible\n",
                Message(role="assistant", content=content),
            ]
        ]
    )
    token = STREAM_DISPLAY.set(display)
    try:
        await Chat(llm).plan_step([], [], [], "hi")
    finally:
        STREAM_DISPLAY.reset(token)
    released = await _drain(display)
    assert released == "visible\n"
    assert "secret" not in released


async def test_a_bearer_token_split_across_lines_is_never_released() -> None:
    display = Display("none")
    token_text = "abcdefghijklmnopqrstuvwxyz0123456789"
    seen: list[str] = []
    content = f"Here: Bearer\n  {token_text}\nDone\n"
    llm = StreamLLM(
        [
            [
                "Here: Bearer\n",
                "  ",
                token_text,
                "\nDone\n",
                lambda: seen.append(display.text()),
                Message(role="assistant", content=content),
            ]
        ]
    )
    token = STREAM_DISPLAY.set(display)
    try:
        await Chat(llm).plan_step([], [], [], "hi")
    finally:
        STREAM_DISPLAY.reset(token)
    released = display.text()
    assert seen and token_text not in seen[0] and "[redacted]" in seen[0]
    assert token_text not in released
    assert "[redacted]" in released and "Done\n" in released


async def test_a_retry_after_visible_text_opens_a_new_stream() -> None:
    display = Display("none")
    llm = StreamLLM(
        [
            ["partial\n", ProviderError("reset", retryable=True)],
            ["final\n", Message(role="assistant", content="final\n")],
        ]
    )
    token = STREAM_DISPLAY.set(display)
    try:
        reply = await Chat(llm, retry_delays=[0], sleep=_no_sleep).plan_step(
            [], [], [], "hi"
        )
    finally:
        STREAM_DISPLAY.reset(token)
    assert reply.content == "final\n"
    assert display.kinds() == [
        "opened",
        "content",
        "closed",
        "opened",
        "content",
        "closed",
    ]
    assert await _drain(display) == "partial\nfinal\n"
    assert llm.closed == 2


async def test_cancel_mid_stream_closes_the_upstream_and_the_display() -> None:
    class Waiting(StreamLLM):
        async def stream_chat(self, messages, tools=None):  # type: ignore[no-untyped-def]
            self.stream_calls += 1
            try:
                yield StreamFragment(kind="content", text="start\n")
                await asyncio.Event().wait()
            finally:
                self.closed += 1

    display = Display("none")
    llm = Waiting([])
    token = STREAM_DISPLAY.set(display)
    task = asyncio.create_task(Chat(llm).plan_step([], [], [], "hi"))
    try:
        while "content" not in display.kinds():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        STREAM_DISPLAY.reset(token)
    assert llm.closed == 1
    assert display.kinds()[-1] == "closed"


async def test_a_tool_call_preamble_is_released_before_the_complete_reply() -> None:
    call = ToolCall(tool_name="echo", arguments={"text": "x"})
    display = Display("none")
    llm = StreamLLM(
        [
            [
                "Let me check\n",
                Message(role="assistant", content="Let me check\n", tool_call=call),
            ]
        ]
    )
    token = STREAM_DISPLAY.set(display)
    try:
        reply = await Chat(llm).plan_step([], [], [], "hi")
    finally:
        STREAM_DISPLAY.reset(token)
    assert reply.tool_calls and reply.tool_calls[0].tool_name == "echo"
    assert await _drain(display) == "Let me check\n"


async def _no_sleep(seconds: float) -> None:
    return None
