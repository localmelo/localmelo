"""A broken model reply is not shown as the answer (#355).

Two kinds were seen in a run of one-step chat replies: a reply that starts with
a spelled-out ``<thinking>...</thinking>`` block, which was shown to the person
as the answer, and an answer of dots only. The first is thinking, as a
``<think>`` block is; the second is an empty reply, so the model is asked once
more. Real answers, short ones included, stay as they are.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker.stream_filter import StreamFilter
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    Message,
    ReasoningTags,
    RunResult,
    ToolDef,
    has_content,
    split_leading_thinking,
    split_thinking,
    unfinished_as_thinking,
    unfinished_thinking,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.window import has_think_block, without_think_block
from localmelo.tests.fixtures.stub_llm import StubServer

# ── The rules, on their own ──

BLOCK = "<thinking>The user works at Acme.\nI will keep it short.</thinking>"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A leading, closed block is the thinking, in any letter case.
        (
            f"{BLOCK}\nDone.",
            ("The user works at Acme.\nI will keep it short.", "Done."),
        ),
        ("  \n<THINKING>plan</Thinking>Done.", ("plan", "Done.")),
        ("<thinking></thinking>Done.", ("", "Done.")),
        ("<thinking>plan</thinking>", ("plan", "")),
        # Not a leading, closed block: the text stays as it is.
        ("Done. <thinking>plan</thinking>", ("", "Done. <thinking>plan</thinking>")),
        ("Use <thinking> to open a block.", ("", "Use <thinking> to open a block.")),
        ("<thinking>never closed", ("", "<thinking>never closed")),
        ("<thinking>never closed. Done.", ("", "<thinking>never closed. Done.")),
        # A <think> block reads as before.
        ("<think>plan</think>Done.", ("plan", "Done.")),
    ],
)
def test_a_leading_closed_thinking_block_is_the_thinking(
    text: str, expected: tuple[str, str]
) -> None:
    assert split_thinking(text) == expected


def test_the_answer_keeps_its_whitespace_on_request() -> None:
    thinking, answer = split_thinking(
        "<thinking>plan</thinking>\n    code()\n", strip_answer=False
    )
    assert (thinking, answer) == ("plan", "\n    code()\n")


@pytest.mark.parametrize(
    ("tags", "text", "expected"),
    [
        (
            "explicit",
            f"{BLOCK}Done.",
            ("The user works at Acme.\nI will keep it short.", "Done."),
        ),
        (
            "explicit",
            "Done. <thinking>plan</thinking>",
            ("", "Done. <thinking>plan</thinking>"),
        ),
        ("explicit", "<thinking>never closed", ("", "<thinking>never closed")),
        ("none", f"{BLOCK}Done.", ("", f"{BLOCK}Done.")),
    ],
)
def test_a_declared_framing_reads_the_block_the_same_way(
    tags: Any, text: str, expected: tuple[str, str]
) -> None:
    assert split_thinking(text, reasoning_tags=tags) == expected


def test_the_leading_block_helper() -> None:
    assert split_leading_thinking(" <thinking>a</thinking> b") == ("a", " b")
    assert split_leading_thinking("b <thinking>a</thinking>") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("...", False),
        ("…", False),
        (" . . . ", False),
        ("--", False),
        ("?!", False),
        ("👍", False),
        ("```\n```", False),
        ("", False),
        ("OK", True),
        ("好", True),
        ("👍 done", True),
        ("1", True),
        ("٣", True),
        ("Привет", True),
        ("```py\nx = 1\n```", True),
    ],
)
def test_a_text_needs_a_letter_or_a_digit(text: str, expected: bool) -> None:
    assert has_content(text) is expected


def test_a_closed_block_with_no_answer_is_unfinished_only_with_reasoning_tokens() -> (
    None
):
    assert unfinished_thinking(BLOCK, "", framing=None, reasoning_tokens=5)
    assert not unfinished_thinking(BLOCK, "", framing=None, reasoning_tokens=0)
    assert not unfinished_thinking(
        f"{BLOCK}Done.", "", framing=None, reasoning_tokens=5
    )
    # A mention in the middle of an answer, and tags declared literal, are answers.
    assert not unfinished_thinking("Use <thinking> here.", "", framing=None)
    assert not unfinished_thinking(BLOCK, "", framing="none", reasoning_tokens=5)
    assert (
        unfinished_as_thinking(BLOCK, "")
        == "The user works at Acme.\nI will keep it short."
    )


def test_the_store_clears_a_leading_thinking_block_like_a_think_block() -> None:
    assert has_think_block(f"{BLOCK}Done.")
    assert without_think_block(f"{BLOCK}Done.") == "Done."
    assert not has_think_block("Done. <thinking>plan</thinking>")
    assert not has_think_block("<thinking>never closed")


# ── The streamed reply ──


def _shown(text: str, hint: Any, size: int) -> tuple[str, str, str]:
    """What the display gets of *text* streamed *size* characters at a time.

    The result is the thinking, what was shown before the stream ended, and
    all that was shown.
    """
    stream = StreamFilter(hint)
    during = ""
    for start in range(0, len(text), size):
        during += stream.feed(text[start : start + size])[1]
    thinking, tail = stream.finish()
    return thinking, during, during + tail


@pytest.mark.parametrize("hint", [None, "explicit"])
@pytest.mark.parametrize("size", [1, 3, 1000])
def test_the_block_never_shows_while_it_streams(hint: Any, size: int) -> None:
    thinking, during, content = _shown(f"  {BLOCK}\nDone.", hint, size)
    assert thinking == "The user works at Acme.\nI will keep it short."
    assert content.strip() == "Done."
    assert "Acme" not in during and "<thinking" not in during.lower()


@pytest.mark.parametrize("hint", [None, "explicit"])
@pytest.mark.parametrize("size", [1, 4, 1000])
@pytest.mark.parametrize(
    "text",
    [
        "Done. <thinking>plan</thinking>",
        "Use <thinking> to open a block.",
        "<thinking>never closed. Done.",
        "<thinker> is not a block.",
        "<b>Done.</b>",
    ],
)
def test_other_text_with_the_tag_streams_as_it_is(
    hint: Any, size: int, text: str
) -> None:
    thinking, _, content = _shown(text, hint, size)
    assert (thinking, content) == ("", text)


def test_a_declared_literal_framing_keeps_the_block() -> None:
    assert _shown(f"{BLOCK}Done.", "none", 5)[::2] == ("", f"{BLOCK}Done.")


# ── A run ──


class _Scripted(BaseLLMProvider):
    """A model that plays the replies it is given, one per request."""

    def __init__(self, *replies: str) -> None:
        self.replies = deque(replies)
        self.requests = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests += 1
        return Message(role="assistant", content=self.replies.popleft())

    async def close(self) -> None:
        pass


async def _run(
    *replies: str, reasoning: str = "legacy"
) -> tuple[RunResult, list[AgentEvent], _Scripted]:
    llm = _Scripted(*replies)
    events: list[AgentEvent] = []
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            attempt_policy="none", retriever="off", reasoning=reasoning  # type: ignore[arg-type]
        ),
    )
    try:
        result = await agent.run_task("Say something", on_event=events.append)
    finally:
        await agent.close()
    return result, events, llm


def _notices(events: list[AgentEvent]) -> list[str]:
    return [e.data["kind"] for e in events if e.type == "notice"]


@pytest.mark.parametrize("dots", ["...", "…", " . . . ", "-", "?!", "👍"])
async def test_an_answer_with_no_letter_or_digit_is_asked_again_once(dots: str) -> None:
    result, events, llm = await _run(dots, "Here is the real answer.")
    assert (result.status, result.result) == ("completed", "Here is the real answer.")
    assert llm.requests == 2
    assert _notices(events) == ["empty_reply"]


async def test_two_such_answers_in_a_row_end_the_run() -> None:
    result, events, llm = await _run("...", "…")
    assert (result.status, result.error, result.result) == (
        "failed",
        "empty replies",
        "",
    )
    assert llm.requests == 2


async def test_the_dots_are_not_kept_as_the_latest_text() -> None:
    llm = _Scripted("...", "Real.")
    agent = Agent(llm=llm, options=AgentOptions(attempt_policy="none", retriever="off"))
    try:
        result = await agent.run_task("Say something")
        window = [m.content for m in agent.conversation()]
    finally:
        await agent.close()
    assert result.result == "Real."
    assert EMPTY_REPLY_LABEL in window
    assert "..." not in window


@pytest.mark.parametrize(
    "answer",
    [
        "OK",
        "好",
        "👍 done",
        "1",
        "Yes.",
        "Use `<thinking>` for a block, or <thinking> alone.",
        "<thinking>never closed. The answer is 4.",
        "```python\nprint('hi')\n```",
        "Done. <thinking>plan</thinking>",
    ],
)
async def test_real_answers_are_kept_as_they_are(answer: str) -> None:
    result, events, llm = await _run(answer)
    assert (result.status, result.result) == ("completed", answer)
    assert llm.requests == 1
    assert _notices(events) == []


@pytest.mark.parametrize("reasoning", ["legacy", "drop"])
async def test_a_leading_thinking_block_is_not_the_answer(reasoning: str) -> None:
    result, events, llm = await _run(f"{BLOCK}\nUpdated.", reasoning=reasoning)
    assert result.status == "completed"
    assert split_thinking(result.result)[1] == "Updated."
    assert llm.requests == 1
    reply = next(e for e in events if e.type == "assistant.message")
    if reasoning == "drop":
        assert reply.data["content"] == "Updated."
        assert reply.data["thinking"].startswith("The user works at Acme.")


async def test_a_reply_of_only_a_thinking_block_is_asked_again() -> None:
    result, events, llm = await _run(BLOCK, "Updated.", reasoning="drop")
    assert (result.status, result.result) == ("completed", "Updated.")
    assert llm.requests == 2
    assert _notices(events) == ["empty_reply"]


async def test_a_thinking_block_then_dots_is_asked_again() -> None:
    result, events, llm = await _run(f"{BLOCK}\n...", "Updated.", reasoning="drop")
    assert (result.status, result.result) == ("completed", "Updated.")
    assert _notices(events) == ["empty_reply"]


# ── The streamed planning reply of a run ──


class _Display:
    def __init__(self, hint: ReasoningTags | None) -> None:
        self.reasoning_tags = hint
        self.shown: list[str] = []

    def step(self, number: int) -> None:
        pass

    def opened(self) -> None:
        pass

    def held(self) -> None:
        pass

    def content(self, delta: str) -> None:
        self.shown.append(delta)

    def closed(self) -> None:
        pass


@pytest.mark.parametrize("hint", [None, "explicit"])
async def test_a_streamed_block_is_never_released_to_the_screen(
    hint: ReasoningTags | None, tmp_path: Path
) -> None:
    chunks = ["<thin", "king>The user works", " at Acme.</thinking>", "\nUpdated."]
    server = StubServer(
        {
            "spelled": {
                "replies": [{"content": "".join(chunks), "stream": {"chunks": chunks}}]
            }
        }
    ).start()
    server.default = "spelled"
    display = _Display(hint)
    token = STREAM_DISPLAY.set(display)
    try:
        agent = Agent(
            llm=OpenAICompatLLM(base_url=server.base_url, model="stub-model"),
            options=AgentOptions(
                attempt_policy="none", retriever="off", workspace_root=str(tmp_path)
            ),
        )
        try:
            result = await agent.run_task("Say something")
        finally:
            await agent.close()
    finally:
        STREAM_DISPLAY.reset(token)
        server.close()
    assert split_thinking(result.result)[1] == "Updated."
    shown = "".join(display.shown)
    assert shown.strip() == "Updated."
