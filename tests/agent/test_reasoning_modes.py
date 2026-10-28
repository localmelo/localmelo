"""R2's reasoning modes through the agent loop — scripted HTTP, no model."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.agent import _as_sent
from localmelo.melo.checker import Checker
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.memory.short import WorkingMemory
from localmelo.melo.schema import EMPTY_REPLY_LABEL, Message, ToolCall, ToolDef
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

CALL_ID = "call_1"


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
    return text


class Scripted:
    """Answers each request with the next reply and keeps every request body."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies: deque[dict[str, Any]] = deque(replies)
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError("no reply scripted")
        reply = self.replies.popleft()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": reply["message"]}],
                "usage": reply.get(
                    "usage",
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                ),
            },
        )

    def llm(self, **kwargs: Any) -> OpenAICompatLLM:
        return OpenAICompatLLM(
            base_url="http://stub.test/v1",
            model="stub-model",
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


class OllamaScripted:
    """Answers native ``/api/chat`` requests with scripted messages."""

    def __init__(self, *messages: dict[str, Any]) -> None:
        self.replies: deque[dict[str, Any]] = deque(messages)
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError("no reply scripted")
        return httpx.Response(
            200,
            json={
                "message": self.replies.popleft(),
                "prompt_eval_count": 10,
                "eval_count": 2,
            },
        )

    def llm(self) -> OllamaNativeChat:
        return OllamaNativeChat(
            base_url="http://stub.test",
            model="stub-model",
            transport=httpx.MockTransport(self.handler),
        )


def _reply(
    content: str = "",
    *,
    thinking: str = "",
    calls: int = 0,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking:
        message["reasoning_content"] = thinking
    if calls:
        message["tool_calls"] = [
            {
                "id": CALL_ID,
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": json.dumps({"text": "a"}),
                },
            }
        ]
    return {"message": message, "usage": usage}


def _reflection(action: str = "continue") -> dict[str, Any]:
    return _reply(
        json.dumps(
            {
                "recommended_action": action,
                "rationale": "test",
                "summary": "tried echo",
                "task_still_feasible": True,
                "next_step_is_concrete": True,
                "next_step_is_novel": True,
            }
        ),
        thinking="reflection thought",
    )


def _options(**kwargs: Any) -> AgentOptions:
    kwargs.setdefault("builtins", frozenset())
    return AgentOptions(**kwargs)


def _agent(
    llm: BaseLLMProvider,
    options: AgentOptions,
    *,
    embedding: BaseEmbeddingProvider | None = None,
) -> Agent:
    agent = Agent(llm=llm, embedding=embedding, options=options)
    agent.hippo.register_tool(_echo())
    agent.executor.register("echo", _run_echo)
    return agent


def _assistant(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in body["messages"] if m["role"] == "assistant"]


def _has_reasoning(message: dict[str, Any]) -> bool:
    return "reasoning_content" in message or "reasoning" in message


@pytest.mark.asyncio
async def test_drop_equals_legacy_on_the_wire() -> None:
    script = [
        _reply("", thinking="draft one", calls=1),
        _reply("", thinking="draft two"),
        _reflection(),
        _reply("done", thinking="draft four"),
    ]
    legacy_server = Scripted(*script)
    legacy_llm = legacy_server.llm()
    legacy = _agent(
        legacy_llm,
        _options(
            reasoning="legacy",
            attempt_policy="reflect",
            steps_per_attempt=2,
            max_attempts=2,
        ),
    )
    legacy_result = await legacy.run_task("q")
    await legacy.close()

    drop_server = Scripted(*script)
    drop_llm = drop_server.llm(reasoning_field="auto")
    drop = _agent(
        drop_llm,
        _options(
            reasoning="drop",
            attempt_policy="reflect",
            steps_per_attempt=2,
            max_attempts=2,
        ),
    )
    drop_result = await drop.run_task("q")
    await drop.close()

    assert legacy_result.status == drop_result.status == "completed"
    assert legacy_server.bodies == drop_server.bodies
    assert "memory_stats" not in legacy_result.to_dict()
    assert drop_result.memory_stats == {
        "replayed_chars": 0,
        "replay_skipped": 0,
        "replay_stripped": 0,
    }
    assert drop_result.config["reasoning"] == "drop"
    assert "reasoning_chars" in drop_result.usage


@pytest.mark.asyncio
async def test_auto_sends_exactly_one_key_on_every_later_request() -> None:
    server = Scripted(
        _reply("", thinking="first", calls=1),
        _reply("done", thinking="second"),
    )
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="replay"))
    await agent.run_task("q")
    await agent.close()

    for body in server.bodies[1:]:
        for message in _assistant(body):
            if message.get("content") or message.get("tool_calls"):
                assert _has_reasoning(message)
                assert len({"reasoning_content", "reasoning"} & set(message)) == 1


@pytest.mark.asyncio
async def test_a_turn_over_the_cap_is_skipped_and_counted() -> None:
    huge = "x" * 50_000
    server = Scripted(
        _reply("", thinking=huge, calls=1),
        _reply("done"),
    )
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="replay"))
    result = await agent.run_task("q")
    await agent.close()

    assert result.memory_stats is not None
    assert result.memory_stats["replay_skipped"] == 1
    tool_turn = _assistant(server.bodies[1])[0]
    assert tool_turn["tool_calls"]
    assert "reasoning_content" not in tool_turn
    assert "reasoning" not in tool_turn


@pytest.mark.asyncio
async def test_replay_required_evicts_whole_units_and_never_cuts() -> None:
    old = Message(
        role="assistant",
        content="old",
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "a"}, id="old")],
        thinking="x" * 60_000,
    )
    new = Message(
        role="assistant",
        content="new",
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "a"}, id="new")],
        thinking="y" * 60_000,
    )
    server = Scripted(_reply("done"))
    llm = server.llm(
        reasoning_field="reasoning_content", requires_reasoning_replay=True
    )
    agent = _agent(
        llm,
        _options(reasoning="replay", replay_scope="window"),
    )
    agent.load_conversation(
        [
            old,
            Message(role="tool", content="r1", tool_call_id="old"),
            new,
            Message(role="tool", content="r2", tool_call_id="new"),
        ]
    )
    result = await agent.run_task("q")
    await agent.close()

    assert result.memory_stats is not None
    assert result.memory_stats["replay_skipped"] == 0
    assert result.memory_stats["replay_stripped"] == 0
    sent = _assistant(server.bodies[0])
    assert [m["content"] for m in sent] == ["new"]
    assert sent[0]["reasoning_content"] == "y" * 60_000


@pytest.mark.asyncio
async def test_replay_strips_the_oldest_thinking_before_dropping_a_unit() -> None:
    old = Message(
        role="assistant",
        content="old",
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "a"}, id="old")],
        thinking="x" * 39_000,
    )
    new = Message(
        role="assistant",
        content="new",
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "a"}, id="new")],
        thinking="y" * 39_000,
    )
    server = Scripted(_reply("done"))
    llm = server.llm(reasoning_field="reasoning_content")
    agent = _agent(llm, _options(reasoning="replay", replay_scope="window"))
    agent.load_conversation(
        [
            old,
            Message(role="tool", content="r1", tool_call_id="old"),
            new,
            Message(role="tool", content="z" * 30_000, tool_call_id="new"),
        ]
    )
    result = await agent.run_task("q")
    await agent.close()

    assert result.memory_stats is not None
    assert result.memory_stats["replay_stripped"] == 1
    assert result.memory_stats["replay_skipped"] == 0
    sent = _assistant(server.bodies[0])
    assert [m["content"] for m in sent] == ["old", "new"]
    assert "reasoning_content" not in sent[0]
    assert sent[1]["reasoning_content"] == "y" * 39_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turns", "expected"), [(2, [3, 4]), (1, [4]), (None, [1, 2, 3, 4])]
)
async def test_replay_turns_keeps_the_newest_k(
    turns: int | None, expected: list[int]
) -> None:
    history = [
        Message(role="assistant", content=f"a{i}", thinking=f"t{i}")
        for i in range(1, 5)
    ]
    server = Scripted(_reply("done"))
    llm = server.llm(reasoning_field="reasoning_content")
    agent = _agent(
        llm,
        _options(reasoning="replay", replay_scope="window", replay_turns=turns),
    )
    agent.load_conversation(history)
    await agent.run_task("q")
    await agent.close()

    sent = _assistant(server.bodies[0])
    assert [m["content"] for m in sent] == ["a1", "a2", "a3", "a4"]
    replayed = [i + 1 for i, m in enumerate(sent) if _has_reasoning(m)]
    assert replayed == expected


@pytest.mark.asyncio
async def test_replay_scope_task_keeps_earlier_tasks_out() -> None:
    old = Message(role="assistant", content="old", thinking="old thought")
    script = [
        _reply("", thinking="new thought", calls=1),
        _reply("done", thinking="last"),
    ]

    task_server = Scripted(*script)
    task_llm = task_server.llm(reasoning_field="auto")
    task_agent = _agent(task_llm, _options(reasoning="replay"))
    task_agent.load_conversation([old])
    await task_agent.run_task("q")
    await task_agent.close()
    assert not _has_reasoning(_assistant(task_server.bodies[0])[0])
    current = next(m for m in _assistant(task_server.bodies[1]) if m.get("tool_calls"))
    assert current["reasoning_content"] == "new thought"

    window_server = Scripted(*script)
    window_llm = window_server.llm(reasoning_field="reasoning_content")
    window_agent = _agent(
        window_llm, _options(reasoning="replay", replay_scope="window")
    )
    window_agent.load_conversation([old])
    await window_agent.run_task("q")
    await window_agent.close()
    assert _assistant(window_server.bodies[0])[0]["reasoning_content"] == "old thought"


def test_an_evicted_task_query_makes_every_remaining_turn_the_tasks() -> None:
    window = [
        Message(role="assistant", content="old", thinking="old thought"),
        Message(role="user", content="later"),
    ]
    options = AgentOptions(reasoning="replay")
    sent = _as_sent(window, options, task_query=Message(role="user", content="gone"))
    assert sent[0].thinking == "old thought"


@pytest.mark.asyncio
async def test_the_receipt_split_cleans_content_and_keeps_thinking() -> None:
    server = Scripted(
        _reply("<think>draft</think>answer", calls=1),
        _reply("<think>last</think>done"),
    )
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="replay", record_answers=True))
    result = await agent.run_task("q")
    conversation = agent.conversation()
    await agent.close()

    tool_turn = next(m for m in conversation if m.tool_calls)
    assert (tool_turn.content, tool_turn.thinking) == ("answer", "draft")
    assert result.result == "done"
    assert result.usage["reasoning_chars"] == len("draft") + len("last")


@pytest.mark.asyncio
async def test_ollama_drop_sends_no_thinking_and_no_inline_copy() -> None:
    server = OllamaScripted(
        {
            "role": "assistant",
            "content": "",
            "thinking": "draft",
            "tool_calls": [
                {
                    "function": {
                        "name": "echo",
                        "arguments": {"text": "a"},
                    }
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    )
    agent = _agent(server.llm(), _options(reasoning="drop"))
    await agent.run_task("q")
    await agent.close()
    turn = next(m for m in server.bodies[1]["messages"] if m["role"] == "assistant")
    assert "thinking" not in turn
    assert turn["content"] == ""


@pytest.mark.asyncio
async def test_legacy_keeps_the_inline_block_in_content() -> None:
    server = Scripted(_reply("<think>draft</think>answer"))
    llm = server.llm()
    agent = _agent(llm, _options(reasoning="legacy", record_answers=True))
    result = await agent.run_task("q")
    await agent.close()
    assert result.result == "<think>draft</think>answer"


@pytest.mark.asyncio
async def test_the_empty_reply_label_keeps_its_thinking_for_replay() -> None:
    script = [_reply("", thinking="empty thought"), _reply("done", thinking="last")]

    replay_server = Scripted(*script)
    replay_llm = replay_server.llm(reasoning_field="auto")
    replay_agent = _agent(replay_llm, _options(reasoning="replay"))
    await replay_agent.run_task("q")
    label = next(
        m for m in replay_agent.conversation() if m.content == EMPTY_REPLY_LABEL
    )
    await replay_agent.close()
    assert label.thinking == "empty thought"
    assert (
        _assistant(replay_server.bodies[1])[0]["reasoning_content"] == "empty thought"
    )

    legacy_server = Scripted(*script)
    legacy_llm = legacy_server.llm()
    legacy_agent = _agent(legacy_llm, _options(reasoning="legacy"))
    await legacy_agent.run_task("q")
    await legacy_agent.close()
    assert not _has_reasoning(_assistant(legacy_server.bodies[1])[0])

    drop_server = Scripted(*script)
    drop_llm = drop_server.llm(reasoning_field="auto")
    drop_agent = _agent(drop_llm, _options(reasoning="drop"))
    await drop_agent.run_task("q")
    label = next(m for m in drop_agent.conversation() if m.content == EMPTY_REPLY_LABEL)
    await drop_agent.close()
    assert label.thinking == "empty thought"
    assert not _has_reasoning(_assistant(drop_server.bodies[1])[0])


@pytest.mark.asyncio
async def test_the_size_cut_turn_keeps_its_thinking_in_replay() -> None:
    script = [_reply("a" * 60_000, thinking="cut thought", calls=1), _reply("done")]
    replay_server = Scripted(*script)
    replay_llm = replay_server.llm(reasoning_field="auto")
    replay_agent = _agent(replay_llm, _options(reasoning="replay"))
    await replay_agent.run_task("q")
    cut = next(m for m in replay_agent.conversation() if m.tool_calls)
    await replay_agent.close()
    assert (cut.content, cut.thinking) == ("", "cut thought")

    legacy_server = Scripted(*script)
    legacy_llm = legacy_server.llm()
    legacy_agent = _agent(legacy_llm, _options(reasoning="legacy"))
    await legacy_agent.run_task("q")
    cut = next(m for m in legacy_agent.conversation() if m.tool_calls)
    await legacy_agent.close()
    assert (cut.content, cut.thinking) == ("", "")


@pytest.mark.asyncio
async def test_the_reflection_request_is_fitted_to_the_cap() -> None:
    async def noop(text: str) -> str:
        return text

    agent = _agent(
        Scripted(_reply("done")).llm(reasoning_field="auto"),
        _options(reasoning="replay"),
    )
    systems = [Message(role="system", content="sys")]
    earliest = Message(
        role="assistant",
        content="a",
        tool_calls=[ToolCall(tool_name="echo", arguments={}, id="a")],
    )
    middle = Message(
        role="assistant",
        content="b",
        tool_calls=[ToolCall(tool_name="echo", arguments={}, id="b")],
    )
    newest = Message(
        role="assistant",
        content="c",
        tool_calls=[ToolCall(tool_name="echo", arguments={}, id="c")],
    )
    messages = [
        *systems,
        earliest,
        Message(role="tool", content="x" * 60_000, tool_call_id="a"),
        middle,
        Message(role="tool", content="y" * 50_000, tool_call_id="b"),
        newest,
        Message(role="tool", content="z" * 1_000, tool_call_id="c"),
        Message(role="user", content="reflect"),
    ]
    fitted = await agent._fit_reflection_messages(messages, None)
    await agent.close()
    assert fitted is not None
    assert fitted[0] is systems[0]
    assert fitted[-1] is messages[-1]
    assert fitted[1] is middle
    assert len(json.dumps([m.content for m in fitted])) < 100_000

    too_big = [
        systems[0],
        newest,
        Message(role="tool", content="q" * 120_000, tool_call_id="c"),
        Message(role="user", content="reflect"),
    ]
    agent = _agent(
        Scripted(_reply("done")).llm(reasoning_field="auto"),
        _options(reasoning="replay"),
    )
    assert await agent._fit_reflection_messages(too_big, None) is None
    await agent.close()


@pytest.mark.asyncio
async def test_a_reflection_too_large_to_fit_is_not_sent() -> None:
    server = Scripted(_reply("done"))
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="replay"))
    huge = Message(
        role="assistant",
        content="a",
        tool_calls=[ToolCall(tool_name="echo", arguments={}, id="a")],
    )
    short = [
        huge,
        Message(role="tool", content="x" * 120_000, tool_call_id="a"),
    ]

    async def fit(messages: list[Message]) -> list[Message] | None:
        return await agent._fit_reflection_messages(messages, None)

    answer = await agent.chat.reflect(short=short, query="q", attempt_id=0, fit=fit)
    assert answer.content == ""
    assert server.bodies == []
    await agent.close()


class CountingEmbedding(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


class RecordingLLM(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.reasoning_field = "auto"

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append(list(messages))
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "recall"), [("legacy", True), ("drop", False), ("replay", False)]
)
async def test_recall_is_read_only_in_legacy(mode: str, recall: bool) -> None:
    embedding = CountingEmbedding()
    llm = RecordingLLM()
    agent = _agent(
        llm,
        _options(reasoning=mode),
        embedding=embedding,
    )
    await agent.hippo.long.add("a fact", [1.0, 0.0, 0.0, 0.0])
    await agent.run_task("q")
    await agent.close()
    prompt = llm.calls[0][0].content
    assert ("[RECALL]" in prompt) is recall
    assert embedding.calls == (1 if recall else 0)


@pytest.mark.asyncio
async def test_pre_plan_counts_thinking_only_when_asked() -> None:
    checker = Checker()
    window = [
        Message(role="assistant", content="", thinking="x" * 100_001),
        Message(role="user", content="q"),
    ]
    assert (await checker.pre_plan(window)).allowed
    refused = await checker.pre_plan(window, count_thinking=True)
    assert not refused.allowed
    assert refused.reason.startswith("Prompt too large")
    # A non-assistant message never contributes thinking.
    assert (
        await checker.pre_plan(
            [Message(role="user", content="q", thinking="x" * 100_001)],
            count_thinking=True,
        )
    ).allowed


def test_strip_thinking_keeps_the_newest_unit() -> None:
    working = WorkingMemory(max_len=10)
    first = Message(role="assistant", content="a", thinking="one")
    second = Message(role="assistant", content="b", thinking="two")
    working.append(first)
    working.append(second)
    assert working.strip_thinking(5) == 1
    assert working.get_window()[0].thinking == ""
    assert working.get_window()[1].thinking == "two"


@pytest.mark.asyncio
async def test_replay_counters_match_what_the_requests_carried() -> None:
    server = Scripted(
        _reply("", thinking="a" * 1_000, calls=1),
        _reply("done", thinking="b" * 2_000),
    )
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="replay"))
    result = await agent.run_task("q")
    await agent.close()
    carried = sum(
        len(message["reasoning_content"])
        for body in server.bodies
        for message in body["messages"]
        if "reasoning_content" in message
    )
    assert result.memory_stats is not None
    assert result.memory_stats["replayed_chars"] == carried
