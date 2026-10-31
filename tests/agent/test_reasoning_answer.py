"""An answer sent as reasoning is kept, and a failed turn does not sink the next (#357).

A model that was told not to think (``enable_thinking = false``) has no
thinking to send, so words a server puts in its reasoning field, or a plain
reply after a reply that did reason, are its answer. With thinking on, the
rules of #278 stand. A turn that ended without an answer leaves a label in
working memory and in the saved turn; the model reads a user-side note in its
place, so it cannot copy the label.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import _unanswered_as_notes
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import thinking_disabled
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.schema import (
    INTERRUPT_LABEL,
    NO_ANSWER_LABEL,
    UNANSWERED_NOTE,
    Message,
    ReasoningTags,
    RunResult,
    answer_from_thinking,
    is_no_answer_label,
    unfinished_thinking,
)
from localmelo.support.gateway.chat_gate import GatedLLM
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.fixtures.stub_llm import SCENARIOS_DIR, StubServer, load_scenarios

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
ECHOED = (
    "The workspace is empty. The user just said hello again, so it is only a "
    "greeting. I'll keep it simple and friendly."
)
FOLDER = "The folder is empty."

# ── The rules, on their own ──


@pytest.mark.parametrize(
    ("body", "off"),
    [
        (NO_THINKING, True),
        ({"think": False}, True),
        ({"chat_template_kwargs": {"enable_thinking": True}}, False),
        ({"think": True}, False),
        ({"chat_template_kwargs": {"other": 1}}, False),
        ({"temperature": 0}, False),
        ({}, False),
    ],
)
def test_a_request_body_says_whether_the_model_is_told_not_to_think(
    body: dict[str, Any], off: bool
) -> None:
    assert thinking_disabled(body) is off


def test_providers_know_from_their_request_fields() -> None:
    assert OpenAICompatLLM("http://x/v1", "m", extra_body=NO_THINKING).thinking_off
    assert not OpenAICompatLLM("http://x/v1", "m").thinking_off
    assert not OpenAICompatLLM(
        "http://x/v1", "m", extra_body={"chat_template_kwargs": {"enable_thinking": 1}}
    ).thinking_off
    planned = OpenAICompatLLM(
        "http://x/v1", "m", extra_body_by_kind={"plan": NO_THINKING}
    )
    assert planned.thinking_off
    assert OllamaNativeChat("http://x", "m", extra_body={"think": False}).thinking_off
    assert not OllamaNativeChat("http://x", "m").thinking_off


async def test_a_gated_provider_copies_the_flag() -> None:
    inner = OpenAICompatLLM("http://x/v1", "m", extra_body=NO_THINKING)
    assert GatedLLM(inner, client=None).thinking_off  # type: ignore[arg-type]
    plain = OpenAICompatLLM("http://x/v1", "m")
    assert not GatedLLM(plain, client=None).thinking_off  # type: ignore[arg-type]
    await inner.close()
    await plain.close()


@pytest.mark.parametrize(
    ("content", "thinking", "tokens", "words"),
    [
        # The recorded shape: no answer text, the words in the reasoning field.
        ("", "汤要炖20分钟。", 0, "汤要炖20分钟。"),
        ("", "汤要炖20分钟。", None, "汤要炖20分钟。"),
        (" \n", FOLDER, 0, FOLDER),
        ("<think>The folder is empty.</think>", "", 0, FOLDER),
        # The server counted reasoning: the model did think (#278 stands).
        ("", FOLDER, 12, ""),
        # An answer, nothing, or thinking that never closed.
        ("Fine.", FOLDER, 0, ""),
        ("", "", 0, ""),
        ("<think>still going", "", 0, ""),
    ],
)
def test_words_sent_as_thinking_are_the_answer_when_thinking_is_off(
    content: str, thinking: str, tokens: int | None, words: str
) -> None:
    assert answer_from_thinking(content, thinking, reasoning_tokens=tokens) == words


@pytest.mark.parametrize(
    ("content", "thinking", "framing", "tokens", "streamed", "on", "off"),
    [
        # A framing seen on an earlier reply, then a plain reply (the report).
        (ECHOED, "", "field", 0, False, True, False),
        (ECHOED, "", "closing_only", 0, False, True, False),
        # The server repeated its reasoning as content at the end of a stream.
        ("The same  words.", "The same words.", None, 0, True, True, False),
        # What a model that does not think cannot explain stays unfinished.
        ("<think>still going", "", None, 0, False, True, True),
        ("", FOLDER, None, 12, False, True, True),
        ("<think>done</think>", "", None, 3, False, True, True),
    ],
)
def test_thinking_off_keeps_the_facts_and_drops_the_guesses(
    content: str,
    thinking: str,
    framing: str | None,
    tokens: int | None,
    streamed: bool,
    on: bool,
    off: bool,
) -> None:
    kwargs: dict[str, Any] = {
        "framing": framing,
        "reasoning_tokens": tokens,
        "streamed": streamed,
    }
    assert unfinished_thinking(content, thinking, **kwargs) is on
    assert unfinished_thinking(content, thinking, thinking_off=True, **kwargs) is off


def test_the_labels_are_recognised_with_any_status() -> None:
    assert is_no_answer_label(INTERRUPT_LABEL)
    assert is_no_answer_label(NO_ANSWER_LABEL.format(status="failed"))
    assert is_no_answer_label(NO_ANSWER_LABEL.format(status="budget_exhausted"))
    assert not is_no_answer_label("[no answer: the run ended with status")
    assert not is_no_answer_label("Sure.\n[no answer: the run ended with status x]")
    assert not is_no_answer_label("[empty reply]")


def test_a_label_in_a_request_is_a_user_side_note_and_the_roles_alternate() -> None:
    label = NO_ANSWER_LABEL.format(status="failed")
    window = [
        Message(role="user", content="first"),
        Message(role="assistant", content=label),
        Message(role="user", content="second"),
        Message(role="assistant", content=INTERRUPT_LABEL),
        Message(role="user", content="third"),
        Message(role="assistant", content="Fine."),
    ]
    sent = _unanswered_as_notes(window)
    assert [(m.role, m.content) for m in sent] == [
        ("user", f"first\n\n{UNANSWERED_NOTE}\n\nsecond\n\n{UNANSWERED_NOTE}\n\nthird"),
        ("assistant", "Fine."),
    ]
    assert window[1].content == label  # the window itself is not changed
    clean = [Message(role="user", content="a"), Message(role="assistant", content="b")]
    assert _unanswered_as_notes(clean) is clean


# ── A run on the stub ──


class _Display:
    reasoning_tags: ReasoningTags | None = None

    def __init__(self) -> None:
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


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer(load_scenarios([SCENARIOS_DIR])).start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(params=["whole", "streamed"])
def streaming(request: pytest.FixtureRequest) -> Iterator[_Display | None]:
    display = _Display() if request.param == "streamed" else None
    token = STREAM_DISPLAY.set(display)
    try:
        yield display
    finally:
        STREAM_DISPLAY.reset(token)


def _agent(
    stub: StubServer, tmp_path: Path, *, thinking: bool, **options: Any
) -> Agent:
    return Agent(
        llm=OpenAICompatLLM(
            base_url=stub.base_url,
            model="stub-model",
            extra_body=None if thinking else NO_THINKING,
        ),
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


async def _run(
    stub: StubServer, scenario: str, tmp_path: Path, *, thinking: bool
) -> tuple[RunResult, list[AgentEvent]]:
    stub.default = scenario
    events: list[AgentEvent] = []
    agent = _agent(stub, tmp_path, thinking=thinking)
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


async def test_an_answer_sent_as_reasoning_is_the_answer_when_thinking_is_off(
    stub: StubServer, tmp_path: Path, streaming: _Display | None
) -> None:
    result, events = await _run(stub, "thinking_off_field", tmp_path, thinking=False)
    assert (result.status, result.result, result.n_llm) == ("completed", FOLDER, 2)
    assert _replies(events)[-1] == (FOLDER, "")
    assert _notices(events) == []


async def test_the_same_reply_with_thinking_on_is_not_an_answer(
    stub: StubServer, tmp_path: Path, streaming: _Display | None
) -> None:
    result, events = await _run(stub, "thinking_off_field", tmp_path, thinking=True)
    assert result.result == "Hello! The folder is empty. What would you like to make?"
    assert result.n_llm == 3  # the reply was empty, and the model was asked again
    assert _replies(events)[1] == ("", FOLDER)
    assert _notices(events) == ["empty_reply"]


async def test_a_plain_reply_after_a_reasoned_one_is_the_answer_when_thinking_is_off(
    stub: StubServer, tmp_path: Path, streaming: _Display | None
) -> None:
    # The report: step 1 reasons in its own field and lists files, step 2 is
    # plain content with no reasoning tokens, and used to be "unfinished".
    result, events = await _run(stub, "unfinished_server", tmp_path, thinking=False)
    assert (result.status, result.result, result.n_llm) == ("completed", ECHOED, 2)
    assert _notices(events) == []
    if streaming is not None:
        assert "".join(streaming.shown) == ECHOED


async def test_with_thinking_on_the_same_reply_is_still_unfinished(
    stub: StubServer, tmp_path: Path, streaming: _Display | None
) -> None:
    result, events = await _run(stub, "unfinished_server", tmp_path, thinking=True)
    assert result.n_llm == 3
    assert ECHOED not in result.result
    assert _notices(events) == ["unfinished_answer"]
    if streaming is not None:
        assert ECHOED not in "".join(streaming.shown)


async def test_counted_reasoning_with_no_answer_is_unfinished_even_when_thinking_is_off(
    stub: StubServer, tmp_path: Path, streaming: _Display | None
) -> None:
    result, events = await _run(stub, "unfinished_empty", tmp_path, thinking=False)
    assert result.result == "Hello! What can I do for you?"
    assert _notices(events) == ["unfinished_answer"]


# ── A failed turn and the next request ──


def _roles(body: dict[str, Any]) -> list[str]:
    return [m["role"] for m in body["messages"] if m["role"] != "system"]


async def _two_turns(
    stub: StubServer, tmp_path: Path, first: dict[str, Any]
) -> tuple[Agent, RunResult, list[dict[str, Any]]]:
    stub.scenarios["two"] = {
        "sequential": True,
        "replies": [first, first, {"content": "Fine, the second."}],
    }
    stub.default = "two"
    agent = _agent(stub, tmp_path, thinking=True, record_answers=True)
    await agent.run_task("First question")
    seen = len(stub.requests)
    second = await agent.run_task("Second question")
    return agent, second, stub.requests[seen:]


@pytest.mark.parametrize("first", [{"content": ""}, {"content": "<think>never ends"}])
async def test_the_next_request_after_a_failed_turn_has_no_label(
    stub: StubServer, tmp_path: Path, streaming: _Display | None, first: dict[str, Any]
) -> None:
    agent, second, requests = await _two_turns(stub, tmp_path, first)
    try:
        assert second.result == "Fine, the second."
        label = NO_ANSWER_LABEL.format(status="failed")
        window = agent.conversation()
        # Working memory, which the saved turn and the console read, keeps it.
        assert [m.content for m in window if is_no_answer_label(m.content)] == [label]
        assert window[-1].content == "Fine, the second."
        assert len(requests) == 1
        body = requests[0]
        sent = [m for m in body["messages"] if m["role"] != "system"]
        assert not any(is_no_answer_label(str(m.get("content"))) for m in sent)
        assert "[no answer" not in str(body["messages"])
        assert sent[-1]["content"].endswith(f"{UNANSWERED_NOTE}\n\nSecond question")
        roles = _roles(body)
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False))
    finally:
        await agent.close()


async def test_a_chat_whose_first_reply_reasoned_keeps_answering_when_thinking_is_off(
    stub: StubServer, tmp_path: Path, streaming: _Display | None
) -> None:
    # The report: once a reply had reasoned, every later plain reply failed.
    def usage(reasoning: int) -> dict[str, Any]:
        return {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        }

    stub.scenarios["chat"] = {
        "sequential": True,
        "replies": [
            {"reasoning": "A thought.", "content": "First.", "usage": usage(3)},
            {"content": "Second.", "usage": usage(0)},
            {"content": "Third.", "usage": usage(0)},
            {"content": "Fourth.", "usage": usage(0)},
        ],
    }
    stub.default = "chat"
    agent = _agent(stub, tmp_path, thinking=False, record_answers=True)
    try:
        results = [
            await agent.run_task(query) for query in ("one", "two", "three", "four")
        ]
    finally:
        await agent.close()
    assert [(r.status, r.error) for r in results] == [("completed", "")] * 4
    assert [r.result for r in results] == ["First.", "Second.", "Third.", "Fourth."]
