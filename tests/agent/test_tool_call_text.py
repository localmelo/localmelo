"""A tool call written as text is not an answer.

A local model's server can swallow a malformed tool call and leave its first
argument's name (``text``), or the call's markup, in the reply's content with
no real tool call. Such a reply is not the final answer: the model is asked
again like after an empty reply, and a run that keeps doing it fails honestly.
Real answers, one-word ones included, stay as they are.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.short import group_turns
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    TOOL_CALL_TEXT_ERROR,
    TOOL_CALL_TEXT_LABEL,
    TOOL_CALL_TEXT_NUDGE,
    Message,
    RunResult,
    ToolCall,
    ToolDef,
    broken_tool_call,
    is_reply_label,
    is_reply_nudge,
    starts_with_reply_label,
    starts_with_reply_nudge,
)
from localmelo.tests.legacy_options import LEGACY_ANSWER_OPTIONS

REMEMBER = ToolDef(
    name="remember",
    description="Save a note",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}, "topic": {"type": "string"}},
        "required": ["text"],
    },
)
SEARCH = ToolDef(
    name="search_files",
    description="Find files",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
    },
)
# A parameter name too short to be told from a plain word or letter.
TINY = ToolDef(
    name="pick",
    description="Pick one",
    parameters={"type": "object", "properties": {"n": {}, "id": {}}},
)

# The XML a local model's tool format writes, exactly.
XML_CALL = (
    "<tool_call>\n<function=remember>\n<parameter=text>\n"
    "My manager is Priya Shah.\n</parameter>\n<parameter=topic>\n"
    "preferences\n</parameter>\n</function>\n</tool_call>"
)

# ── The detector, on its own ──


@pytest.mark.parametrize(
    "text",
    [
        # What a local model's server left in the content.
        "text",
        "  text\n",
        "text✅ 已记录",
        "text=My manager is Priya Shah., topic=people",
        'text<function_calls><function invoke name="remember">…',
        # A parameter name followed by what a call leaves behind.
        "topic",
        "topic=people",
        'text:"My manager is Priya Shah."',
        "text:My manager is Priya Shah.",
        "text：已记录",
        "text<",
        "query✓",
        "path",
        # Tool-call markup, in any letter case, anywhere in the answer.
        '<function_calls>\n<invoke name="remember">',
        "Saving it now. <FUNCTION_CALLS>",
        '<invoke name="remember"><parameter name="text">x</parameter>',
        '<Invoke  Name = "remember">',
        "<function invoke remember>",
        '<tool_call>\n{"name": "remember"}\n</tool_call>',
        "<function=remember><parameter=text>x</parameter></function>",
        # The XML shape of a tool format: whole, partial, or never closed.
        XML_CALL,
        XML_CALL.replace("</tool_call>", ""),
        XML_CALL.split("</parameter>")[0],
        "<tool_call>\n<function=remember>\n<parameter=text>\nMy manager",
        "<tool_call>\n<function=rem",
        "<tool_call>",
        "<function=remember>\n<parameter=text>\nx\n</parameter>\n</function>",
        "<parameter=topic>\npreferences\n</parameter>",
        "Saving that now.\n\n" + XML_CALL,
        "<think>save it</think>\n" + XML_CALL,
        "<arg_key>text</arg_key><arg_value>x</arg_value>",
        "已记录。<tool_call>",
        # Reasoning is split off first; what is left is judged.
        "<think>I will save it.</think>text",
        "<thinking>I will save it.</thinking>\ntext=Priya",
        # Markup outside code, next to code that discusses it.
        "Use `<tool_call>` to call.\n<tool_call>{}",
    ],
)
def test_a_tool_call_written_as_text_is_broken(text: str) -> None:
    assert broken_tool_call(text, [REMEMBER, SEARCH]) is True


@pytest.mark.parametrize(
    "text",
    [
        # Real answers, short ones included.
        "Done.",
        "Thursday",
        "Yes",
        "42",
        "好的，已经记下了。",
        "已记录：你的经理是 Priya Shah。",
        "我已经把 text 这个参数写进去了。",
        "Noted: your manager is Priya Shah.",
        # A parameter name in the text, but not a call's remnant.
        "text is short",
        "Text messages are free.",
        "text messages are free.",
        "text-based answers work",
        "text.",
        "text, then more",
        "text 已记录",
        "text: here is the answer",
        "texts",
        "text_id is the field",
        "topic of the day",
        "Topic: people",
        "pathology is the study of disease",
        "query?",
        "The path is /tmp/a, the text is ready.",
        # A name too short to count as a parameter name.
        "n",
        "id",
        # Nothing to judge.
        "",
        "   ",
        "...",
        # Markup discussed in code is not a call.
        "Use `<function_calls>` to call a tool.",
        "The format is ``<invoke name=...>`` in the prompt.",
        '```xml\n<function_calls>\n<invoke name="remember">\n```',
        "Here is the format:\n```\n<tool_call>{}</tool_call>\n```\nThat is all.",
        "~~~\n<parameter=text>\n~~~",
        "The format is:\n```xml\n" + XML_CALL + "\n```\nThat is all.",
        "```\n" + XML_CALL.split("</parameter>")[0],
        "Wrap it in `<tool_call>`, then `<function=name>` and `<parameter=key>`.",
        "Here is how:\n```xml\n<function_calls>\nnever closed",
        "`text`",
        "`text=1` is how it looks.",
        # Markup-like text that is not a call.
        "Use <b>bold</b>.",
        "A <function> is a block of code.",
        "The <parameter> element is optional.",
        # Reasoning that mentions a call is not the answer.
        "<think>I will call <function_calls> now.</think>Done.",
        "<thinking>text</thinking>Done.",
    ],
)
def test_a_real_answer_is_not_broken(text: str) -> None:
    assert broken_tool_call(text, [REMEMBER, SEARCH, TINY]) is False


def test_a_parameter_name_counts_only_for_a_tool_that_is_offered() -> None:
    assert broken_tool_call("text", [REMEMBER]) is True
    assert broken_tool_call("text", [SEARCH]) is False
    assert broken_tool_call("text", []) is False
    assert broken_tool_call("text") is False
    # The name is case-sensitive.
    assert broken_tool_call("Text", [REMEMBER]) is False
    assert broken_tool_call("TEXT", [REMEMBER]) is False
    # Markup needs no tool at all.
    assert broken_tool_call("<function_calls>", []) is True


def test_a_tool_without_a_schema_of_parameters_has_no_names() -> None:
    bare = ToolDef(name="probe", description="Take any arguments", parameters={})
    odd = ToolDef(
        name="odd", description="Odd", parameters={"properties": ["text", "topic"]}
    )
    assert broken_tool_call("text", [bare, odd]) is False


# ── The window keeps a label and its nudge together ──


def test_the_texts_of_the_label_and_the_nudge() -> None:
    assert TOOL_CALL_TEXT_LABEL == "[tool call written as text]"
    assert TOOL_CALL_TEXT_NUDGE == (
        "Your last reply wrote a tool call as text, so it did not run. Call the "
        "tool through the tool interface, or give your final answer in plain words."
    )
    assert TOOL_CALL_TEXT_ERROR == "tool call written as text"


def test_the_helpers_know_both_labels_and_both_nudges() -> None:
    for nudge in (EMPTY_REPLY_NUDGE, TOOL_CALL_TEXT_NUDGE):
        assert is_reply_nudge(nudge)
        assert starts_with_reply_nudge(nudge)
        assert starts_with_reply_nudge(f"{nudge}\n\n[RECALL] a note")
        assert not is_reply_nudge(f"{nudge}\n\n[RECALL] a note")
    assert not is_reply_nudge("My manager is Priya.")
    assert not starts_with_reply_nudge(f"{TOOL_CALL_TEXT_NUDGE} More")
    for label in (EMPTY_REPLY_LABEL, TOOL_CALL_TEXT_LABEL):
        assert is_reply_label(label)
        assert is_reply_label(f"{label}\nnote", with_note=True)
        assert not is_reply_label(f"{label}\nnote")
        assert starts_with_reply_label(f"{label}\nnote")
    assert not is_reply_label("[empty]")


@pytest.mark.parametrize("label", [EMPTY_REPLY_LABEL, TOOL_CALL_TEXT_LABEL])
def test_a_label_owns_the_nudge_after_it(label: str) -> None:
    nudge = TOOL_CALL_TEXT_NUDGE if label == TOOL_CALL_TEXT_LABEL else EMPTY_REPLY_NUDGE
    messages = [
        Message(role="user", content="query"),
        Message(role="assistant", content=label),
        Message(role="user", content=nudge),
        Message(role="assistant", content=f"{label}\nnote"),
        Message(role="user", content=f"{nudge}\n\n[RECALL] x"),
    ]
    assert [len(unit) for unit in group_turns(messages)] == [1, 2, 1, 1]
    assert [len(unit) for unit in group_turns(messages, empty_prefix=True)] == [1, 2, 2]


# ── A run ──


class _Scripted(BaseLLMProvider):
    """A model that plays the replies it is given, one per request."""

    def __init__(self, *replies: str | Message) -> None:
        self.replies = deque(replies)
        self.requests: list[list[Message]] = []
        self.offered: list[list[str]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        self.offered.append([tool.name for tool in tools or []])
        reply = self.replies.popleft()
        if isinstance(reply, str):
            return Message(role="assistant", content=reply)
        return reply

    async def close(self) -> None:
        pass


def _remember_call(call_id: str = "call_1", *, content: str = "") -> Message:
    return Message(
        role="assistant",
        content=content,
        tool_calls=[
            ToolCall(
                tool_name="remember",
                arguments={"text": "My manager is Priya Shah.", "topic": "people"},
                id=call_id,
            )
        ],
    )


def _decision(action: str) -> Message:
    flags = {
        "task_still_feasible": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
    }
    data = {
        "summary": "Tried to save a note.",
        "failed_hypotheses": [],
        "useful_evidence": [],
        "recommended_avoids": [],
        "next_promising_directions": [],
        "progress_made": False,
        "new_information_gained": False,
        "recommended_action": action,
        "rationale": "",
        "best_effort_result": "",
        "estimated_info_gain": 0.1,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.1,
        "feasibility": 0.5,
        **flags,
    }
    return Message(role="assistant", content=json.dumps(data))


class _Memory:
    """The ``remember`` tool: records what it was called with."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def remember(self, text: str = "", topic: str = "") -> str:
        self.calls.append({"text": text, "topic": topic})
        return "Saved."


def _agent(llm: _Scripted, **options: Any) -> tuple[Agent, _Memory]:
    options.setdefault("builtins", frozenset())
    options.setdefault("tool_exposure", "all")
    options.setdefault("retriever", "off")
    options.setdefault("attempt_policy", "none")
    for name, value in LEGACY_ANSWER_OPTIONS.items():
        options.setdefault(name, value)
    agent = Agent(llm=llm, options=AgentOptions(**options))
    memory = _Memory()
    agent.hippo.register_tool(REMEMBER)
    agent.executor.register("remember", memory.remember)
    return agent, memory


async def _run(
    *replies: str | Message, **options: Any
) -> tuple[RunResult, list[AgentEvent], _Scripted, _Memory]:
    llm = _Scripted(*replies)
    agent, memory = _agent(llm, **options)
    events: list[AgentEvent] = []
    try:
        result = await agent.run_task(
            "My manager is Priya Shah.", on_event=events.append
        )
    finally:
        await agent.close()
    return result, events, llm, memory


def _notices(events: list[AgentEvent]) -> list[str]:
    return [e.data["kind"] for e in events if e.type == "notice"]


@pytest.mark.parametrize(
    "broken",
    [
        XML_CALL,
        XML_CALL.split("</parameter>")[0],
        "text",
        "text✅ 已记录",
        "text=My manager is Priya Shah., topic=people",
        '<function_calls><invoke name="remember"><parameter name="text">x',
    ],
)
async def test_a_broken_call_is_asked_again_and_the_real_call_runs(
    broken: str,
) -> None:
    result, events, llm, memory = await _run(
        broken, _remember_call(), "Noted: your manager is Priya Shah."
    )
    assert (result.status, result.result) == (
        "completed",
        "Noted: your manager is Priya Shah.",
    )
    assert memory.calls == [{"text": "My manager is Priya Shah.", "topic": "people"}]
    assert llm.requests and len(llm.requests) == 3
    assert _notices(events) == ["tool_call_as_text"]
    # The step's event still records what the model sent: that is history.
    first = next(e for e in events if e.type == "assistant.message")
    assert first.data["content"] == broken and first.data["tool_calls"] == []
    # The model is told what went wrong, with the roles alternating.
    second = [(m.role, m.content) for m in llm.requests[1][1:]]
    assert second == [
        ("user", "My manager is Priya Shah."),
        ("assistant", TOOL_CALL_TEXT_LABEL),
        ("user", TOOL_CALL_TEXT_NUDGE),
    ]
    assert broken not in [content for _, content in second]


async def test_the_notice_names_the_row_and_the_step() -> None:
    _, events, _, _ = await _run("text", _remember_call(), "Done.")
    notice = next(e for e in events if e.type == "notice")
    assert notice.data == {
        "kind": "tool_call_as_text",
        "message": (
            "The model wrote a tool call as text, so it did not run; asking it again."
        ),
        "step": 1,
        "attempt": 1,
        "detail": {"count": 1},
    }


async def test_a_run_that_keeps_writing_the_call_fails_honestly() -> None:
    result, events, llm, memory = await _run("text", "text✅ 已记录")
    assert (result.status, result.error, result.result) == (
        "failed",
        TOOL_CALL_TEXT_ERROR,
        "",
    )
    assert "text" not in result.result and memory.calls == []
    assert len(llm.requests) == 2
    assert _notices(events) == ["tool_call_as_text"]


async def test_the_broken_text_is_not_the_runs_answer_under_the_default_policy() -> (
    None
):
    llm = _Scripted("text", "text=Priya")
    agent, _ = _agent(llm, attempt_policy="reflect")
    try:
        outcome = await agent.run_task("My manager is Priya Shah.")
    finally:
        await agent.close()
    assert (outcome.status, outcome.error, outcome.result) == (
        "failed",
        TOOL_CALL_TEXT_ERROR,
        "",
    )


async def test_run_says_the_error_and_not_the_text() -> None:
    llm = _Scripted("text", "text")
    agent, _ = _agent(llm)
    try:
        reply = await agent.run("My manager is Priya Shah.")
    finally:
        await agent.close()
    assert reply == "[failed] tool call written as text"


async def test_a_broken_call_is_not_the_latest_text_for_a_done_reflection() -> None:
    # A "done" without an answer falls back to the latest text the model
    # wrote, and the broken call is not an answer the model wrote.
    llm = _Scripted("text", _decision("done"))
    agent, _ = _agent(
        llm, attempt_policy="reflect", steps_per_attempt=1, max_attempts=2
    )
    try:
        outcome = await agent.run_task("My manager is Priya Shah.")
    finally:
        await agent.close()
    assert outcome.result != "text"
    assert (outcome.status, outcome.result) == ("completed", "")


async def test_text_beside_a_real_call_is_not_judged() -> None:
    result, events, _, memory = await _run(_remember_call(content="text"), "Noted.")
    assert (result.status, result.result) == ("completed", "Noted.")
    assert len(memory.calls) == 1
    assert _notices(events) == []


async def test_a_broken_call_after_a_real_one_still_gets_one_more_try() -> None:
    result, events, llm, memory = await _run(
        _remember_call("call_1"), "text", _remember_call("call_2"), "Noted."
    )
    assert (result.status, result.result) == ("completed", "Noted.")
    assert len(memory.calls) == 2
    assert _notices(events) == ["tool_call_as_text"]


async def test_a_real_call_ends_the_row() -> None:
    # Two broken replies that a real call separates are not two in a row.
    result, events, _, _ = await _run(
        "text", _remember_call("call_1"), "text", _remember_call("call_2"), "Noted."
    )
    assert (result.status, result.result) == ("completed", "Noted.")
    assert _notices(events) == ["tool_call_as_text", "tool_call_as_text"]


@pytest.mark.parametrize(
    "replies",
    [("...", "text"), ("text", "..."), ("", "text✅"), ("text", "")],
)
async def test_a_row_of_an_empty_and_a_broken_reply_fails_with_the_honest_error(
    replies: tuple[str, str],
) -> None:
    result, _, _, _ = await _run(*replies)
    assert (result.status, result.error, result.result) == (
        "failed",
        TOOL_CALL_TEXT_ERROR,
        "",
    )


async def test_two_empty_replies_keep_their_own_error() -> None:
    result, events, _, _ = await _run("...", "")
    assert (result.status, result.error) == ("failed", "empty replies")
    assert _notices(events) == ["empty_reply"]


@pytest.mark.parametrize(
    "answer",
    [
        "Done.",
        "Thursday",
        "OK",
        "text is short",
        "Text messages are free.",
        "topic of the day",
        "Use `<function_calls>` for a call.",
        '```xml\n<function_calls>\n<invoke name="remember">\n```',
        "好的，已记录。",
    ],
)
async def test_a_real_answer_ends_the_run(answer: str) -> None:
    result, events, llm, memory = await _run(answer)
    assert (result.status, result.result) == ("completed", answer)
    assert len(llm.requests) == 1 and memory.calls == []
    assert _notices(events) == []


async def test_the_broken_text_does_not_reach_the_window_or_the_history() -> None:
    llm = _Scripted("text✅ 已记录", _remember_call(), "Noted.")
    agent, _ = _agent(llm)
    try:
        result = await agent.run_task("My manager is Priya Shah.")
        window = [m.content for m in agent.conversation()]
        task = await agent.hippo.get_task(result.task_id)
    finally:
        await agent.close()
    assert result.status == "completed"
    assert TOOL_CALL_TEXT_LABEL in window
    assert "text✅ 已记录" not in window
    assert task is not None
    recorded = json.dumps([s.tool_call.tool_name for s in task.steps if s.tool_call])
    assert recorded == '["remember"]'


async def test_the_compress_modes_keep_the_text_out_of_the_note() -> None:
    broken = Message(
        role="assistant",
        content="text✅ Priya Shah is the manager",
        thinking="I should save that the user's manager is Priya Shah.",
    )
    llm = _Scripted(broken, _remember_call(), "Noted.")
    agent, _ = _agent(llm, reasoning="compress")
    seen: list[str] = []
    original = agent._reply_note

    def spy(response: Message, results: Any, **kwargs: Any) -> Any:
        seen.append(response.content)
        return original(response, results, **kwargs)

    agent._reply_note = spy  # type: ignore[method-assign]
    try:
        result = await agent.run_task("My manager is Priya Shah.")
    finally:
        await agent.close()
    assert result.status == "completed"
    # The first reply's note is made from its thinking, never from the broken text.
    assert seen[0] == ""
    label = next(
        m for m in llm.requests[1] if m.role == "assistant" and not m.tool_calls
    )
    assert label.content.startswith(TOOL_CALL_TEXT_LABEL)
    assert "✅" not in label.content
    nudge = llm.requests[1][-1]
    assert starts_with_reply_nudge(nudge.content)
    assert nudge.content.startswith(TOOL_CALL_TEXT_NUDGE)
