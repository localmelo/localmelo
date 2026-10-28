from __future__ import annotations

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import Chat
from localmelo.melo.agent._request_accounting import (
    IMAGE_TOKENS,
    ContextBudget,
    request_proxy,
)
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.schema import ImagePart, Message, ToolCall, ToolDef


class FakeLLM(BaseLLMProvider):
    model = "model-a"
    reasoning_field = "reasoning"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []

    async def chat(self, messages, tools=None):
        self.sent.append((messages, tools))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return Message(role="assistant", content="ok", usage=reply)

    async def close(self):
        pass


@pytest.mark.parametrize("values", [(0, 1, 1), (10, 0, 1), (10, 1, True), (2, 1, 1)])
def test_budget_validation(values):
    with pytest.raises(ValueError):
        ContextBudget(*values)


def test_proxy_counts_utf8_schema_replay_and_images():
    base = [Message(role="assistant", content="a")]
    plain, _ = request_proxy(base, [], replay=True)
    rich = [
        Message(
            role="assistant",
            content="中文🙂",
            thinking="replayed reasoning",
            tool_calls=[ToolCall("search", {"query": "evidence"}, id="id")],
        ),
        Message(
            role="user", content="see", images=[ImagePart.from_bytes(b"x", "image/png")]
        ),
    ]
    tools = [ToolDef("search", "description", {"type": "object"})]
    value, allowance = request_proxy(rich, tools, replay=True)
    no_replay, _ = request_proxy(rich, tools, replay=False)
    assert value > plain + IMAGE_TOKENS
    assert value > no_replay
    assert allowance == IMAGE_TOKENS
    rich[1].images = [rich[1].images[0].as_reference()]
    assert request_proxy(rich, tools, replay=True)[1] == 0


async def test_opt_in_calibration_kind_model_and_reset():
    llm = FakeLLM(
        [
            {"prompt_tokens": 9000},
            {"prompt_tokens": 0},
            {"prompt_tokens": 12000},
            None,
            None,
            None,
        ]
    )
    chat = Chat(llm, context_budget=ContextBudget(100, 10, 10))

    async def plan():
        await chat.plan_step(
            [], [Message(role="user", content="question")], [], "question"
        )

    await plan()
    await plan()
    await chat.reflect([], "question", 1)
    await plan()
    llm.model = "model-b"
    await plan()
    records = chat.stats.context_budget
    assert records is not None
    assert [r["source"] for r in records] == [
        "proxy",
        "calibrated",
        "proxy",
        "proxy",
        "proxy",
    ]
    assert records[0]["measured_prompt_tokens"] == 9000
    assert records[1]["measured_prompt_tokens"] is None
    assert records[0]["over_limit"]
    assert records[0]["input_limit"] == 80
    assert len(llm.sent) == 5  # over-limit requests are still sent
    chat.reset_stats()
    await plan()
    assert chat.stats.context_budget[0]["source"] == "proxy"


async def test_failed_attempts_and_other_kinds():
    async def no_sleep(_):
        pass

    llm = FakeLLM(
        [
            ProviderError("busy", retryable=True),
            {"prompt_tokens": 200},
            {"prompt_tokens": 100},
        ]
    )
    chat = Chat(
        llm,
        retry_delays=(0,),
        sleep=no_sleep,
        context_budget=ContextBudget(1000, 100, 100),
    )
    emitted = []
    chat.on_context_budget = emitted.append
    await chat.plan_step([], [Message(role="user", content="q")], [], "q")
    await chat.estimate_steps("q")
    records = chat.stats.context_budget
    assert emitted == records
    assert emitted[0]["measured_prompt_tokens"] is None
    assert emitted[1]["measured_prompt_tokens"] == 200
    assert [r["attempt"] for r in records] == [1, 2]
    assert records[0]["measured_prompt_tokens"] is None
    assert records[1]["measured_prompt_tokens"] == 200


async def test_default_disabled():
    chat = Chat(FakeLLM([None]))
    await chat.plan_step([], [Message(role="user", content="q")], [], "q")
    assert chat.stats.context_budget is None


async def test_agent_reports_budget_in_result_and_event():
    agent = Agent(
        llm=FakeLLM([{"prompt_tokens": 9999}]),
        options=AgentOptions(
            context_window=100,
            output_reserve=10,
            safety_margin=10,
            memory=False,
            builtins=frozenset(),
        ),
    )
    events = []
    try:
        result = await agent.run_task("answer", on_event=events.append)
    finally:
        await agent.close()
    records = result.to_dict()["context_budget"]
    assert len(records) == 1 and records[0]["over_limit"]
    assert records[0]["measured_prompt_tokens"] == 9999
    assert [dict(e.data) for e in events if e.type == "context_budget"] == records
    assert events[-1].to_dict()["data"]["result"]["context_budget"] == records


@pytest.mark.parametrize("values", [(1, None, None), (10, 5, 5), (True, 1, 1)])
def test_agent_options_reject_invalid_budget(values):
    with pytest.raises(ValueError):
        AgentOptions(
            context_window=values[0], output_reserve=values[1], safety_margin=values[2]
        )
