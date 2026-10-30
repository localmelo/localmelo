"""A reply that is only the model's unfinished thinking is never the answer (#278).

The stub model server plays the replies: one that opens thinking and never
closes it, one that closes it, a server that found no end mark and returned
the thinking as content, and a reply with no answer once its thinking is set
aside. The unfinished reply is stored as thinking, the model is asked once
more, and a second unfinished reply ends the run with a plain error.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.schema import (
    UNFINISHED_ANSWER_ERROR,
    ReasoningTags,
    RunResult,
    split_thinking,
    unfinished_as_thinking,
    unfinished_thinking,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.fixtures.stub_llm import SCENARIOS_DIR, StubServer, load_scenarios

GREETING = "The user just said hello. I'll keep it simple and friendly."
ECHOED = "The workspace is empty. The user just said hello again, so it is only a greeting. I'll keep it simple and friendly."

# ── The rule, on its own ──


@pytest.mark.parametrize(
    ("content", "thinking", "framing", "tokens", "unfinished"),
    [
        # A leading <think> that never closes.
        ("<think>still going", "", None, 0, True),
        ("  <THINK>still going", "", "explicit", 0, True),
        ("<think>still going", "", "none", 0, False),
        ("<think>done</think>Hello!", "", None, 0, False),
        # No answer once thinking is set aside: only with reasoning tokens.
        ("", "A thought.", None, 12, True),
        ("<think>done</think>", "", None, 3, True),
        ("", "A thought.", None, 0, False),
        ("", "A thought.", None, None, False),
        ("", "", None, 0, False),
        # The server repeated its reasoning as content (a stream only).
        ("The same  words.", "The same words.", None, 0, True),
        ("Hello!", "A greeting.", "field", 0, False),
        # A framing that ends thinking with a mark, and no mark.
        (GREETING, "", "field", 0, True),
        (GREETING, "", "field", None, False),  # no count: the server says nothing
        (GREETING, "", "closing_only", 0, True),
        ("thought</think>Hello!", "", "closing_only", 0, False),
        (GREETING, "", None, 0, False),
        (GREETING, "", "explicit", 0, False),
        (GREETING, "", "none", 0, False),
    ],
)
def test_the_rule_reads_facts_of_the_reply(
    content: str,
    thinking: str,
    framing: str | None,
    tokens: int | None,
    unfinished: bool,
) -> None:
    assert (
        unfinished_thinking(
            content, thinking, framing=framing, reasoning_tokens=tokens, streamed=True
        )
        is unfinished
    )


def test_only_a_streamed_reply_that_repeats_its_reasoning_is_unfinished() -> None:
    # oMLX sends unfinished thinking again as content only at the end of a stream.
    words = ("The same words.", "The same words.")
    assert unfinished_thinking(*words, framing=None, streamed=True)
    assert not unfinished_thinking(*words, framing=None, streamed=False)


def test_the_thinking_of_an_unfinished_reply_keeps_each_text_once() -> None:
    assert unfinished_as_thinking("<think>A plan.", "") == "A plan."
    assert unfinished_as_thinking(ECHOED, ECHOED) == ECHOED
    assert unfinished_as_thinking(GREETING, "Earlier.") == f"Earlier.\n\n{GREETING}"
    assert unfinished_as_thinking("<think>A.</think>", "") == "A."


# ── A run on the stub ──


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer(load_scenarios([SCENARIOS_DIR])).start()
    try:
        yield server
    finally:
        server.close()


async def _run(
    stub: StubServer, scenario: str, tmp_path: Path, **options: Any
) -> tuple[RunResult, list[AgentEvent]]:
    stub.default = scenario
    events: list[AgentEvent] = []
    agent = Agent(
        llm=OpenAICompatLLM(base_url=stub.base_url, model="stub-model"),
        options=AgentOptions(
            builtins=frozenset({"list_files"}),
            attempt_policy="none",
            tool_exposure="all",
            retriever="off",
            max_steps=6,
            workspace_root=str(tmp_path),
            **options,
        ),
    )
    try:
        result = await agent.run_task("Say hello", on_event=events.append)
    finally:
        await agent.close()
    return result, events


def _replies(events: list[AgentEvent]) -> list[tuple[str, str]]:
    return [
        (event.data["content"], event.data["thinking"])
        for event in events
        if event.type == "assistant.message"
    ]


def _notices(events: list[AgentEvent]) -> list[str]:
    return [event.data["kind"] for event in events if event.type == "notice"]


@pytest.mark.parametrize("reasoning", ["legacy", "drop"])
async def test_an_unclosed_think_is_thinking_and_the_model_answers_again(
    stub: StubServer, tmp_path: Path, reasoning: str
) -> None:
    result, events = await _run(stub, "unfinished_open", tmp_path, reasoning=reasoning)
    assert result.status == "completed"
    assert split_thinking(result.result)[1] == "Hello! What can I do for you?"
    assert (result.steps, result.n_llm) == (2, 2)
    assert _replies(events)[0] == ("", GREETING)
    assert _notices(events) == ["unfinished_answer"]


async def test_a_second_unfinished_reply_ends_the_run_without_an_answer(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "unfinished_twice", tmp_path)
    assert (result.status, result.error, result.result) == (
        "failed",
        UNFINISHED_ANSWER_ERROR,
        "",
    )
    assert result.n_llm == 2  # asked once more, no more
    assert _replies(events) == [("", GREETING), ("", GREETING)]
    assert _notices(events) == ["unfinished_answer"]


async def test_thinking_a_server_returned_as_content_is_never_the_answer(
    stub: StubServer, tmp_path: Path
) -> None:
    # As in the report: step 1 reasons in its own field and lists files, step
    # 2 comes back as content with no reasoning (the server found no end mark).
    result, events = await _run(stub, "unfinished_server", tmp_path)
    assert result.status == "completed"
    assert result.result == "Hello! The folder is empty. What would you like to make?"
    assert result.n_llm == 3
    assert _replies(events)[1] == ("", ECHOED)
    assert ECHOED not in result.result


async def test_reasoning_tokens_with_no_answer_ask_for_the_answer(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "unfinished_empty", tmp_path)
    assert (result.status, result.result) == (
        "completed",
        "Hello! What can I do for you?",
    )
    assert _notices(events) == ["unfinished_answer"]
    assert result.n_llm == 2


@pytest.mark.parametrize("scenario", ["answer", "list_files"])
async def test_normal_replies_are_unchanged(
    stub: StubServer, tmp_path: Path, scenario: str
) -> None:
    result, events = await _run(stub, scenario, tmp_path)
    assert result.status == "completed"
    assert "unfinished_answer" not in _notices(events)
    assert result.result == _replies(events)[-1][0]


# ── The streamed planning reply ──


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


async def test_a_held_unfinished_reply_is_never_released_to_the_screen(
    stub: StubServer, tmp_path: Path
) -> None:
    display = _Display(None)  # the conservative rule holds content to the end
    token = STREAM_DISPLAY.set(display)
    try:
        result, _ = await _run(stub, "unfinished_server", tmp_path)
    finally:
        STREAM_DISPLAY.reset(token)
    assert result.result == "Hello! The folder is empty. What would you like to make?"
    assert ECHOED not in "".join(display.shown)


async def test_a_declared_closing_mark_finds_an_unfinished_first_reply(
    stub: StubServer, tmp_path: Path
) -> None:
    stub.scenarios["closing"] = {
        "sequential": True,
        "replies": [{"content": GREETING}, {"content": "A greeting.</think>Hello!"}],
    }
    display = _Display("closing_only")
    token = STREAM_DISPLAY.set(display)
    try:
        result, events = await _run(stub, "closing", tmp_path)
    finally:
        STREAM_DISPLAY.reset(token)
    assert split_thinking(result.result)[1] == "Hello!"
    assert _replies(events)[0] == ("", GREETING)
    assert display.shown == ["Hello!"]
