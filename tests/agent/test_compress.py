"""R4's compress modes through the agent loop: notes, drafts, reflection."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.agent import ASSISTANT_TEXT_SHARE
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.notes import (
    RECALL_START,
)
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    Message,
    ToolCall,
    ToolDef,
    ToolResult,
)

CALL_ID = "call_1"


class FakeLLM(BaseLLMProvider):
    """Answers with queued replies and keeps every request it is sent."""

    reasoning_field = "auto"
    requires_reasoning_replay = False

    def __init__(self, *replies: Message) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[list[Message]] = []
        self.offers: list[list[str]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(messages)
        self.offers.append(sorted(tool.name for tool in tools or ()))
        if not self.replies:
            raise AssertionError("no reply scripted")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


class ReplayRequired(FakeLLM):
    reasoning_field = "reasoning_content"
    requires_reasoning_replay = True


def _echo() -> ToolDef:
    return ToolDef(
        name="echo",
        description="echo text back",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


async def _run_echo(text: str) -> str:
    return f"echo: {text}"


def _agent(llm: BaseLLMProvider, options: AgentOptions, **kwargs: Any) -> Agent:
    agent = Agent(llm=llm, options=options, **kwargs)
    agent.hippo.register_tool(_echo())
    agent.executor.register("echo", _run_echo)
    return agent


def _options(**kwargs: Any) -> AgentOptions:
    kwargs.setdefault("builtins", frozenset())
    kwargs.setdefault("attempt_policy", "none")
    kwargs.setdefault("tool_exposure", "all")
    return AgentOptions(**kwargs)


def _call_reply(
    content: str = "",
    *,
    thinking: str = "",
    arguments: dict[str, Any] | None = None,
) -> Message:
    return Message(
        role="assistant",
        content=content,
        thinking=thinking,
        tool_calls=[
            ToolCall(
                tool_name="echo",
                arguments=arguments or {"text": "a"},
                id=CALL_ID,
            )
        ],
    )


def _answer(content: str = "done", *, thinking: str = "") -> Message:
    return Message(role="assistant", content=content, thinking=thinking)


def _system(request: list[Message]) -> str:
    return next(message.content for message in request if message.role == "system")


def _assistant(request: list[Message]) -> list[Message]:
    return [message for message in request if message.role == "assistant"]


# ── The modes and the instruction ──


@pytest.mark.asyncio
async def test_compress_adds_the_instruction_and_stores_notes() -> None:
    server = FakeLLM(
        _call_reply("<note>plan: echo a</note>"),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(server, _options(reasoning="compress"))
    result = await agent.run_task("q")
    await agent.close()

    assert result.status == "completed"
    assert "write a short note" in _system(server.requests[0])
    assert "write a short note" not in _system(
        [Message(role="system", content="legacy")]
    )
    assert result.memory_stats is not None
    assert result.memory_stats["notes"] == 2
    assert result.memory_stats["notes_model"] == 1
    assert result.memory_stats["notes_heuristic"] == 1
    assert result.memory_stats["digests"] == 1
    assert result.config["reasoning"] == "compress"
    assert result.config["note_writer"] == "auto"
    assert [entry.kind for entry in agent.notes()] == ["note", "note", "digest"]


@pytest.mark.asyncio
async def test_legacy_and_drop_and_replay_keep_the_old_prompt() -> None:
    for mode in ("legacy", "drop", "replay"):
        server = FakeLLM(_answer())
        agent = _agent(server, _options(reasoning=mode))
        await agent.run_task("q")
        await agent.close()
        assert "write a short note" not in _system(server.requests[0])
        assert agent.notes() == []


@pytest.mark.asyncio
async def test_the_instruction_is_stable_across_steps() -> None:
    server = FakeLLM(_call_reply(), _answer())
    agent = _agent(server, _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    prompts = [_system(request) for request in server.requests]
    assert len(prompts) == 2
    assert prompts[0] == prompts[1]


@pytest.mark.asyncio
async def test_compress_replay_instruction_names_the_kept_turns() -> None:
    server = FakeLLM(_answer())
    agent = _agent(
        server,
        _options(reasoning="compress+replay", replay_turns=2),
    )
    await agent.run_task("q")
    await agent.close()
    assert "last 2 steps" in _system(server.requests[0])


@pytest.mark.asyncio
async def test_a_replay_required_provider_runs_compress_as_compress_replay() -> None:
    server = ReplayRequired(_call_reply(thinking="draft"), _answer())
    agent = _agent(server, _options(reasoning="compress"))
    result = await agent.run_task("q")
    await agent.close()
    assert result.config["reasoning"] == "compress+replay"
    assert any(
        message.role == "assistant" and message.thinking == "draft"
        for message in server.requests[1]
    )


# ── Placement ──


@pytest.mark.asyncio
async def test_a_model_note_stays_in_the_window_copy() -> None:
    content = "I will echo a.\n<note>plan: echo a</note>"
    server = FakeLLM(_call_reply(content), _answer())
    agent = _agent(server, _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    window = agent.conversation()
    assistant = [m for m in window if m.role == "assistant"][0]
    assert assistant.content == content
    assert assistant.thinking == ""


@pytest.mark.asyncio
async def test_a_reply_without_text_gets_the_checked_note_as_its_copy() -> None:
    server = FakeLLM(
        _call_reply(
            thinking="Let me read list_ops.py. The test fails because of the bug."
        ),
        _answer(),
    )
    agent = _agent(server, _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    assistant = [m for m in agent.conversation() if m.role == "assistant"][0]
    assert assistant.content.startswith("plan: ")
    assert "<note>" not in assistant.content


@pytest.mark.asyncio
async def test_a_short_text_gets_the_note_as_its_second_paragraph() -> None:
    server = FakeLLM(_call_reply("Let me read list_ops.py."), _answer())
    agent = _agent(server, _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    assistant = [m for m in agent.conversation() if m.role == "assistant"][0]
    assert assistant.content.startswith("Let me read list_ops.py.\n\n")
    assert len(assistant.content) <= ASSISTANT_TEXT_SHARE


@pytest.mark.asyncio
async def test_long_text_keeps_the_note_store_only() -> None:
    long_text = (
        "Let me read list_ops.py. The test fails because of the bug. " * 40
    ).strip()
    assert len(long_text) > 2_000
    server = FakeLLM(_call_reply(long_text), _answer())
    agent = _agent(server, _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    assistant = [m for m in agent.conversation() if m.role == "assistant"][0]
    assert assistant.content == long_text.strip()
    assert any(entry.kind == "note" for entry in agent.notes())


@pytest.mark.asyncio
async def test_the_answer_excludes_the_note_block() -> None:
    server = FakeLLM(_answer("<note>plan: answer</note>\nThe answer is 42."))
    agent = _agent(server, _options(reasoning="compress"))
    result = await agent.run_task("q")
    await agent.close()
    assert result.result == "The answer is 42."
    assert result.memory_stats is not None
    assert result.memory_stats["notes"] == 1


@pytest.mark.asyncio
async def test_the_empty_label_carries_the_note_and_keeps_its_nudge() -> None:
    server = FakeLLM(
        Message(role="assistant", content="<note>plan: think</note>"),
        _answer(),
    )
    agent = _agent(server, _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    window = agent.conversation()
    label = next(
        message for message in window if message.content.startswith(EMPTY_REPLY_LABEL)
    )
    assert label.content.startswith(EMPTY_REPLY_LABEL + "\nplan: ")
    nudge = window[window.index(label) + 1]
    assert nudge.content == EMPTY_REPLY_NUDGE


# ── Drafts and pinning ──


def _write_reply(path: str = "list_ops.py", content: str = "print(1)") -> Message:
    return Message(
        role="assistant",
        content=f"writing {path}\n```\n{content}\n```",
        tool_calls=[
            ToolCall(
                tool_name="file_write",
                arguments={"path": path, "content": content},
                id=CALL_ID,
            )
        ],
    )


@pytest.mark.asyncio
async def test_a_pending_targeted_draft_is_pinned_once(tmp_path: Any) -> None:
    server = FakeLLM(
        Message(
            role="assistant",
            content="I will write list_ops.py.\n```\nprint(1)\n```",
            tool_calls=[
                ToolCall(
                    tool_name="echo",
                    arguments={"text": "a"},
                    id=CALL_ID,
                )
            ],
        ),
        _answer(),
    )
    agent = _agent(
        server,
        _options(reasoning="compress", workspace_root=str(tmp_path)),
    )
    await agent.run_task("q")
    await agent.close()
    tool_message = next(
        message for message in agent.conversation() if message.role == "tool"
    )
    assert RECALL_START in tool_message.content
    drafts = [entry for entry in agent.notes() if entry.kind == "draft"]
    assert len(drafts) == 1
    assert drafts[0].draft is not None and drafts[0].draft.status == "pending"


@pytest.mark.asyncio
async def test_an_untargeted_draft_is_never_pinned(tmp_path: Any) -> None:
    server = FakeLLM(
        Message(
            role="assistant",
            content="```\nprint(1)\n```",
            tool_calls=[
                ToolCall(tool_name="echo", arguments={"text": "a"}, id=CALL_ID)
            ],
        ),
        _answer(),
    )
    agent = _agent(
        server,
        _options(reasoning="compress", workspace_root=str(tmp_path)),
    )
    await agent.run_task("q")
    await agent.close()
    assert RECALL_START not in next(
        message.content for message in agent.conversation() if message.role == "tool"
    )


@pytest.mark.asyncio
async def test_a_same_reply_write_marks_the_draft_written_and_never_pins(
    tmp_path: Any,
) -> None:
    server = FakeLLM(_write_reply(), _answer())
    agent = Agent(
        llm=server,
        options=_options(
            reasoning="compress",
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_write"}),
        ),
    )
    await agent.run_task("q")
    await agent.close()
    drafts = [entry for entry in agent.notes() if entry.kind == "draft"]
    assert len(drafts) == 1
    assert drafts[0].draft is not None and drafts[0].draft.status == "written"
    assert RECALL_START not in next(
        message.content for message in agent.conversation() if message.role == "tool"
    )


# ── The reflection ──


def _reflection(action: str = "continue", **extra: Any) -> Message:
    data = {
        "recommended_action": action,
        "rationale": "test",
        "summary": "tried it",
        "task_still_feasible": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
    }
    data.update(extra)
    return Message(
        role="assistant",
        content=json.dumps(data),
        thinking="Let me check list_ops.py. The test passed because the fix works.",
    )


@pytest.mark.asyncio
async def test_the_reflection_request_carries_notes_and_used_notes() -> None:
    server = FakeLLM(
        _call_reply("<note>plan: echo a</note>"),
        _reflection("continue", used_notes=["n1", "n99", 3]),
        _answer(),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            attempt_policy="reflect",
            steps_per_attempt=1,
            max_attempts=2,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()
    reflection_request = server.requests[1]
    assert "[localmelo notes:" in reflection_request[-1].content
    assert "[end of localmelo notes]" in reflection_request[-1].content
    assert "used_notes" in _system([reflection_request[0]])
    assert result.memory_stats is not None
    # One digest per attempt: the reflection ended attempt 1, and the
    # answer ended attempt 2.
    assert result.memory_stats["digests"] == 2
    assert result.memory_stats["rehearsed"]["cited"] == 1
    first = next(entry for entry in agent.notes() if entry.alias == "n1")
    assert first.uses == 1
    reflections = [entry for entry in agent.notes() if entry.note is not None]
    assert any(
        entry.note is not None and entry.note.kind == "reflection"
        for entry in reflections
    )


@pytest.mark.asyncio
async def test_note_writer_heuristic_ignores_the_model_note() -> None:
    server = FakeLLM(
        _call_reply(
            "<note>plan: the model's plan; found: the model's fact</note>",
            thinking="Let me read list_ops.py. The test fails because of the bug.",
        ),
        _answer(),
    )
    agent = _agent(server, _options(reasoning="compress", note_writer="heuristic"))
    await agent.run_task("q")
    await agent.close()
    step_note = next(
        entry.note
        for entry in agent.notes()
        if entry.kind == "note" and entry.note is not None and entry.note.kind == "step"
    )
    assert step_note.source == "heuristic"
    assert "the model's plan" not in step_note.plan
    assistant = [m for m in agent.conversation() if m.role == "assistant"][0]
    assert "<note>" not in assistant.content
    assert assistant.content.startswith("plan: ")


@pytest.mark.asyncio
async def test_a_cancelled_tool_turn_still_stores_and_places_its_note() -> None:
    from localmelo.melo.agent.agent import _InflightStep, _RunState

    server = FakeLLM()
    agent = _agent(server, _options(reasoning="compress"))
    response = _call_reply("<note>plan: echo a</note>")
    step = _InflightStep(
        response=response,
        rest="",
        has_model_note=True,
        attempt=1,
    )
    step.results.append(ToolResult(tool_name="echo", output="echo: a"))
    step.cut_before.append(0)
    run = _RunState(task_id="t" * 32)
    run.steps = 1
    run.attempts = 1
    agent._close_interrupted_step(step, "", run)
    await agent.close()

    note = next(entry for entry in agent.notes() if entry.kind == "note")
    assert note.note is not None and note.note.kind == "step"
    assistant = [m for m in agent.conversation() if m.role == "assistant"][0]
    assert assistant.content == response.content
