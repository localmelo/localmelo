"""R2's reasoning modes through the agent loop — scripted HTTP, no model."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.agent import _replay_plan, _RunState
from localmelo.melo.agent.options import reasoning_refusal
from localmelo.melo.checker import Checker
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    CheckResult,
    Message,
    TaskRecord,
    ToolCall,
    ToolDef,
)
from localmelo.support.backends import providers_for
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.agent.test_reflection_gate import _prefix_scope

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
    kwargs.setdefault("retriever", "off")
    return AgentOptions(**kwargs)


def test_agent_options_validate_the_reasoning_fields() -> None:
    for removed in ("summarize", "drop", "replay"):
        with pytest.raises(ValueError, match="reasoning must be one of"):
            AgentOptions(reasoning=removed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="note_writer must be one of"):
        AgentOptions(note_writer="mystery")  # type: ignore[arg-type]
    assert AgentOptions(reasoning="compress").note_writer == "auto"
    assert AgentOptions(reasoning="compress+replay").note_writer == "auto"
    with pytest.raises(ValueError, match="replay_scope must be one of"):
        AgentOptions(replay_scope="session")  # type: ignore[arg-type]


def test_reasoning_refusal_is_a_pure_capability_check() -> None:
    assert reasoning_refusal("legacy", None, True) is None
    assert reasoning_refusal("compress+replay", "auto", False) is None
    assert "need a reasoning field" in (
        reasoning_refusal("compress+replay", None, False) or ""
    )
    assert reasoning_refusal("compress", None, False) is None
    assert reasoning_refusal("compress", None, True) is None


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
async def test_auto_sends_exactly_one_key_on_every_later_request() -> None:
    server = Scripted(
        _reply("", thinking="first", calls=1),
        _reply("done", thinking="second"),
    )
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="compress+replay"))
    await agent.run_task("q")
    await agent.close()

    for body in server.bodies[1:]:
        for message in _assistant(body):
            if message.get("content") or message.get("tool_calls"):
                assert message["reasoning_content"] == "first"
                assert "reasoning" not in message


@pytest.mark.asyncio
async def test_a_turn_over_the_cap_is_skipped_and_counted() -> None:
    huge = "x" * 50_000
    server = Scripted(
        _reply("", thinking=huge, calls=1),
        _reply("done"),
    )
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="compress+replay"))
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
        _options(reasoning="compress+replay", replay_scope="window"),
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
    # The turns came before the task: the fit drops the oldest whole, down to
    # 70% of the cap, and the one it keeps carries all of its reasoning.
    [sent] = _assistant(server.bodies[0])
    assert (sent["content"], sent["reasoning_content"]) == ("new", "y" * 60_000)


@pytest.mark.asyncio
async def test_replay_strips_the_oldest_thinking_before_dropping_a_unit() -> None:
    turns = []
    for name, mark in (("a", "x"), ("b", "y"), ("c", "z")):
        turns += [
            Message(
                role="assistant",
                content=name,
                tool_calls=[ToolCall(tool_name="echo", arguments={}, id=name)],
                thinking=mark * 35_000,
            ),
            Message(role="tool", content="r", tool_call_id=name),
        ]
    server = Scripted(_reply("done"))
    llm = server.llm(reasoning_field="reasoning_content")
    agent = _agent(llm, _options(reasoning="compress+replay", replay_scope="window"))
    agent.load_conversation(turns)
    result = await agent.run_task("q")
    await agent.close()

    assert result.memory_stats is not None
    # Down to 70% of the cap: the newest turn keeps its reasoning, no turn goes.
    assert result.memory_stats["replay_stripped"] == 2
    assert result.memory_stats["replay_skipped"] == 0
    assert result.memory_stats["replayed_chars"] == 35_000
    sent = _assistant(server.bodies[0])
    assert [m["content"] for m in sent] == ["a", "b", "c"]
    assert not _has_reasoning(sent[0]) and not _has_reasoning(sent[1])
    assert sent[2]["reasoning_content"] == "z" * 35_000


@pytest.mark.asyncio
async def test_replay_scope_task_keeps_earlier_tasks_out() -> None:
    old = Message(role="assistant", content="old", thinking="old thought")
    script = [
        _reply("", thinking="new thought", calls=1),
        _reply("done", thinking="last"),
    ]

    task_server = Scripted(*script)
    task_llm = task_server.llm(reasoning_field="auto")
    task_agent = _agent(task_llm, _options(reasoning="compress+replay"))
    task_agent.load_conversation([old])
    await task_agent.run_task("q")
    await task_agent.close()
    assert not _has_reasoning(_assistant(task_server.bodies[0])[0])
    current = next(m for m in _assistant(task_server.bodies[1]) if m.get("tool_calls"))
    assert current["reasoning_content"] == "new thought"

    window_server = Scripted(*script)
    window_llm = window_server.llm(reasoning_field="reasoning_content")
    window_agent = _agent(
        window_llm, _options(reasoning="compress+replay", replay_scope="window")
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
    options = AgentOptions(reasoning="compress+replay")
    sent, _ = _replay_plan(
        window, options, task_query=Message(role="user", content="gone")
    )
    assert sent[0].thinking == "old thought"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["compress", "compress+replay"])
async def test_the_receipt_split_cleans_content_and_keeps_thinking(mode: str) -> None:
    server = Scripted(
        _reply("<think>draft</think>answer", calls=1),
        _reply("<think>last</think>done"),
    )
    llm = server.llm(reasoning_field="reasoning_content")
    agent = _agent(llm, _options(reasoning=mode, record_answers=True))
    result = await agent.run_task("q")
    conversation = agent.conversation()
    await agent.close()

    tool_turn = next(m for m in conversation if m.tool_calls)
    assert tool_turn.content.startswith("answer")
    assert "<think>" not in tool_turn.content
    assert tool_turn.thinking == ("draft" if mode == "compress+replay" else "")
    assert result.result == "done"
    assert result.usage["reasoning_chars"] == len("draft") + len("last")
    current = next(m for m in _assistant(server.bodies[1]) if m.get("tool_calls"))
    if mode == "compress+replay":
        assert current["reasoning_content"] == "draft"
    else:
        assert not _has_reasoning(current)


@pytest.mark.asyncio
async def test_ollama_compress_sends_no_thinking_and_no_inline_copy() -> None:
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
    agent = _agent(server.llm(), _options(reasoning="compress"))
    await agent.run_task("q")
    await agent.close()
    turn = next(m for m in server.bodies[1]["messages"] if m["role"] == "assistant")
    assert "thinking" not in turn
    assert "<think>" not in turn["content"] and "draft" not in turn["content"]


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
    replay_agent = _agent(replay_llm, _options(reasoning="compress+replay"))
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

    compress_server = Scripted(*script)
    compress_llm = compress_server.llm(reasoning_field="auto")
    compress_agent = _agent(compress_llm, _options(reasoning="compress"))
    await compress_agent.run_task("q")
    label = next(
        m for m in compress_agent.conversation() if m.content == EMPTY_REPLY_LABEL
    )
    await compress_agent.close()
    assert label.thinking == ""  # a compress window carries no thinking
    assert not _has_reasoning(_assistant(compress_server.bodies[1])[0])


@pytest.mark.asyncio
async def test_the_size_cut_turn_keeps_its_thinking_in_replay() -> None:
    script = [_reply("a" * 60_000, thinking="cut thought", calls=1), _reply("done")]
    replay_server = Scripted(*script)
    replay_llm = replay_server.llm(reasoning_field="auto")
    replay_agent = _agent(replay_llm, _options(reasoning="compress+replay"))
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


def _unit(call_id: str, output: str, *, thinking: str = "") -> list[Message]:
    return [
        Message(
            role="assistant",
            content=f"a{call_id}",
            tool_calls=[ToolCall(tool_name="echo", arguments={}, id=call_id)],
            thinking=thinking,
        ),
        Message(role="tool", content=output, tool_call_id=call_id),
    ]


@pytest.mark.asyncio
async def test_do_reflect_fits_the_request_it_sends() -> None:
    server = Scripted(_reflection("stop"))
    agent = _agent(
        server.llm(reasoning_field="auto"),
        _options(reasoning="compress+replay"),
    )
    agent.load_conversation(
        [
            *_unit("a", "x" * 60_000),
            *_unit("b", "y" * 50_000),
            *_unit("c", "z" * 1_000),
        ]
    )
    task = TaskRecord(query="q")
    run = _RunState(task_id=task.task_id)
    with _prefix_scope(agent):
        reflection = await agent._do_reflect(task, 0, "budget", run=run)
    await agent.close()

    assert reflection is not None
    assert len(server.bodies) == 1
    body = server.bodies[0]
    assert "Original task:" in body["messages"][-1]["content"]
    contents = [message.get("content", "") for message in body["messages"]]
    assert sum(len(text) for text in contents) < 100_000
    # Through the planning request's fit: the oldest result is a preview.
    [old] = [text for text in contents if text.startswith("x")]
    assert old.endswith("characters omitted]")
    assert any("y" * 50_000 in text for text in contents)
    assert run.replay_skipped == 0


@pytest.mark.asyncio
async def test_do_reflect_counts_an_over_cap_turn_it_actually_sends() -> None:
    server = Scripted(_reflection("stop"))
    agent = _agent(
        server.llm(reasoning_field="auto"),
        _options(reasoning="compress+replay"),
    )
    agent.load_conversation(
        [
            *_unit("a", "small", thinking="t" * 50_000),
            *_unit("b", "small"),
        ]
    )
    task = TaskRecord(query="q")
    run = _RunState(task_id=task.task_id)
    with _prefix_scope(agent):
        reflection = await agent._do_reflect(task, 0, "budget", run=run)
    await agent.close()

    assert reflection is not None
    assert len(server.bodies) == 1
    assert run.replay_skipped == 1
    assert run.replayed_chars == 0


@pytest.mark.asyncio
async def test_do_reflect_leaves_out_a_tail_that_cannot_fit() -> None:
    server = Scripted(_reflection("stop"))
    agent = _agent(
        server.llm(reasoning_field="auto"),
        _options(reasoning="compress+replay"),
    )
    # A failed check is never shortened: it goes whole, the last thing to go.
    failed = "Traceback (most recent call last):\n" + "x" * 120_000
    agent.load_conversation([*_unit("a", failed, thinking="t" * 50_000)])
    task = TaskRecord(query="q")
    run = _RunState(task_id=task.task_id)
    with _prefix_scope(agent):
        reflection = await agent._do_reflect(task, 0, "budget", run=run)
    await agent.close()

    assert reflection is not None
    [body] = server.bodies
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert run.replay_skipped == 0  # a unit the fit leaves out is not a skip


class CountingEmbedding(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_the_agent_refuses_an_impossible_pair() -> None:
    without_field = OpenAICompatLLM(
        base_url="http://stub.test/v1",
        model="m",
        transport=httpx.MockTransport(Scripted().handler),
    )
    with pytest.raises(ValueError, match="need a reasoning field"):
        Agent(llm=without_field, options=_options(reasoning="compress+replay"))
    await without_field.close()


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
@pytest.mark.parametrize("mode", ["legacy", "compress+replay"])
async def test_no_mode_embeds_without_a_retriever(mode: str) -> None:
    embedding = CountingEmbedding()
    llm = RecordingLLM()
    agent = _agent(
        llm,
        _options(reasoning=mode),
        embedding=embedding,
    )
    await agent.run_task("q")
    await agent.close()
    assert embedding.calls == 0


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


@pytest.mark.asyncio
async def test_replay_counters_match_what_the_requests_carried() -> None:
    server = Scripted(
        _reply("", thinking="a" * 1_000, calls=1),
        _reply("done", thinking="b" * 2_000),
    )
    llm = server.llm(reasoning_field="auto")
    agent = _agent(llm, _options(reasoning="compress+replay"))
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


@pytest.mark.asyncio
async def test_a_replay_required_provider_replays_the_whole_window() -> None:
    # The task scope would leave the two loaded turns out; a server that
    # requires replay must get them, so it is exempt.
    old = Message(role="assistant", content="old", thinking="old thought")
    older = Message(role="assistant", content="older", thinking="older thought")
    server = Scripted(
        _reply("", thinking="new thought", calls=1),
        _reply("done"),
    )
    llm = server.llm(
        reasoning_field="reasoning_content", requires_reasoning_replay=True
    )
    agent = _agent(
        llm,
        _options(reasoning="compress+replay"),
    )
    agent.load_conversation([old, older])
    await agent.run_task("q")
    await agent.close()

    sent = _assistant(server.bodies[0])
    assert [m["content"] for m in sent] == ["old", "older"]
    assert sent[0]["reasoning_content"] == "old thought"
    assert sent[1]["reasoning_content"] == "older thought"


@pytest.mark.asyncio
async def test_a_replay_required_answer_keeps_its_thinking() -> None:
    server = Scripted(
        _reply("first answer", thinking="answer thought"),
        _reply("second answer"),
    )
    llm = server.llm(
        reasoning_field="reasoning_content", requires_reasoning_replay=True
    )
    agent = _agent(
        llm,
        _options(reasoning="compress+replay", record_answers=True),
    )
    await agent.run_task("first")
    await agent.run_task("second")
    await agent.close()

    sent = _assistant(server.bodies[1])
    assert [m["content"] for m in sent] == ["first answer"]
    assert sent[0]["reasoning_content"] == "answer thought"


@pytest.mark.asyncio
async def test_config_providers_clear_the_section_field_outside_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.backends.openai_family import VllmBackend
    from localmelo.support.config import Config, LocalBackendConfig

    server = Scripted(_reply("", thinking="plan thought", calls=1), _reply("done"))

    def build(self: VllmBackend, cfg: Any) -> OpenAICompatLLM:
        return server.llm(
            reasoning_field=cfg.vllm.reasoning_field or None,
            requires_reasoning_replay=cfg.vllm.requires_reasoning_replay,
        )

    monkeypatch.setattr(VllmBackend, "build_chat_provider", build)
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(
            chat_url="http://stub.test/v1",
            chat_model="stub-model",
            reasoning_field="reasoning_content",
        ),
    )
    llm, embedding = providers_for(cfg)
    agent = Agent(llm=llm, embedding=embedding)
    agent.hippo.register_tool(_echo())
    agent.executor.register("echo", _run_echo)
    result = await agent.run_task("q")
    await agent.close()

    assert result.status == "completed"
    assert agent._llm.reasoning_field is None
    assert not _has_reasoning(_assistant(server.bodies[1])[0])


@pytest.mark.asyncio
async def test_config_providers_keep_the_section_field_in_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.backends.openai_family import VllmBackend
    from localmelo.support.config import Config, LocalBackendConfig

    server = Scripted(_reply("", thinking="plan thought", calls=1), _reply("done"))

    def build(self: VllmBackend, cfg: Any) -> OpenAICompatLLM:
        return server.llm(
            reasoning_field=cfg.vllm.reasoning_field or None,
            requires_reasoning_replay=cfg.vllm.requires_reasoning_replay,
        )

    monkeypatch.setattr(VllmBackend, "build_chat_provider", build)
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(
            chat_url="http://stub.test/v1",
            chat_model="stub-model",
            reasoning_field="reasoning_content",
        ),
    )
    llm, embedding = providers_for(cfg, "compress+replay")
    agent = Agent(
        llm=llm, embedding=embedding, options=_options(reasoning="compress+replay")
    )
    agent.hippo.register_tool(_echo())
    agent.executor.register("echo", _run_echo)
    result = await agent.run_task("q")
    await agent.close()

    assert result.status == "completed"
    assert agent._llm.reasoning_field == "reasoning_content"
    assert _assistant(server.bodies[1])[0]["reasoning_content"] == "plan thought"


@pytest.mark.asyncio
async def test_a_reflection_the_prompt_check_refuses_reads_as_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Checker.pre_plan

    async def refuse(self: Checker, messages: list[Message], **kw: Any) -> Any:
        if "Reflect and decide." in messages[-1].content:
            return CheckResult(allowed=False, reason="Prompt too large: 1 chars")
        return await original(self, messages, **kw)

    monkeypatch.setattr(Checker, "pre_plan", refuse)
    server = Scripted(_reply("", thinking="plan thought", calls=1), _reply("done"))
    agent = _agent(
        server.llm(reasoning_field="auto"),
        _options(
            reasoning="compress+replay",
            answer_on_stop=False,  # the request it adds is not what this counts
            max_steps=5,
            steps_per_attempt=1,
            max_attempts=2,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()

    assert result.status == "completed"
    assert len(server.bodies) == 2
    # The refused request is never sent: the second attempt starts.
    assert result.steps == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["compress", "compress+replay"])
async def test_the_explicit_modes_write_no_raw_step_summary(mode: str) -> None:
    embedding = CountingEmbedding()
    server = Scripted(
        _reply("", thinking="draft", calls=1),
        _reply("done"),
    )
    agent = _agent(
        server.llm(reasoning_field="auto"),
        _options(reasoning=mode),
        embedding=embedding,
    )
    await agent.run_task("q")
    await agent.close()
    # R5's recall may embed the query when the plan is on; what the explicit
    # modes must not do is write the raw step summary to long-term memory.
    assert len(agent.hippo.long._entries) == 0
