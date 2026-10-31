"""AgentOptions.max_calls and the answer to a refused request (answer_on_stop).

``max_calls`` makes a run count its model requests and stop before the one that would
pass the limit; with ``answer_on_stop`` the last call is kept for the answer. A
request the server refuses for a budget or rate limit is answered too, when one more
request is allowed. Both are developer options, off by default. No real model.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.agent._call_budget import CallBudgetError, is_budget_refusal
from localmelo.melo.agent._chat import ANSWER_ON_STOP_NOTE, Chat
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.schema import Message, RunResult
from localmelo.tests.agent.test_reflection_gate import (
    FakeLLM,
    _agent,
    _answer,
    _echo_call,
    _reflection,
)

TASK = (
    "Use echo to find the person.\nEnd your reply with one line: FINAL ANSWER: <answer>"
)
BUDGET_BODY = (
    'POST /chat/completions failed with HTTP 400: {"error": {"code": '
    '"budget_exhausted", "message": "budget_exhausted: the run has '
    'used its 30 steps"}}'
)


def _is_answer(call: dict[str, Any]) -> bool:
    return ANSWER_ON_STOP_NOTE in call["messages"][-1].content


def _tools(call: dict[str, Any]) -> bool:
    return bool(call["tools"])


class RefusingLLM(FakeLLM):
    """Replies as FakeLLM does, but refuses requests *at* to *through* (1-based) with *error*."""

    def __init__(
        self,
        replies: list[Message],
        *,
        at: int,
        error: ProviderError,
        through: int | None = None,
    ):
        super().__init__(replies)
        self.at = at
        self.through = at if through is None else through
        self.error = error
        self.refusals = 0

    async def chat(self, messages: list[Message], tools: Any = None) -> Message:
        if self.at <= len(self.calls) + 1 <= self.through:
            self.calls.append({"messages": list(messages), "tools": tools})
            self.refusals += 1
            raise self.error
        return await super().chat(messages, tools)


# ── max_calls ───────────────────────────────────────────────────────────────


class TestMaxCalls:
    @pytest.mark.asyncio
    async def test_five_calls_stop_after_four_and_answer_on_the_fifth(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a", "looking"),
                _echo_call("b"),
                _echo_call("c"),
                _echo_call("d"),
            ]
            + [_answer("FINAL ANSWER: 42")]
        )
        agent = _agent(llm, max_calls=5, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert len(llm.calls) == 5
        assert [_tools(call) for call in llm.calls] == [True] * 4 + [False]
        assert _is_answer(llm.calls[-1]) and not any(map(_is_answer, llm.calls[:4]))
        assert (result.status, result.result) == (
            "budget_exhausted",
            "FINAL ANSWER: 42",
        )
        assert result.answered_on_stop is True
        assert result.to_dict()["answered_on_stop"] is True
        assert result.error.startswith("Call budget reached: 4 model requests sent")
        assert (result.steps, result.n_llm, result.llm_errors) == (4, 5, 0)
        assert result.config["max_calls"] == 5
        assert result.config["answer_on_stop"] is True

    @pytest.mark.asyncio
    async def test_the_stop_is_not_reached_by_a_run_that_finishes_first(self) -> None:
        llm = FakeLLM([_echo_call("a"), _answer("FINAL ANSWER: 7")])
        agent = _agent(llm, max_calls=5, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert (result.status, result.result) == ("completed", "FINAL ANSWER: 7")
        assert result.answered_on_stop is False
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_a_reflection_counts_as_a_call(self) -> None:
        # plan, plan, reflection (3 calls), plan (4th), then the stop; answer is call 6.
        llm = FakeLLM(
            [
                _echo_call("a"),
                _echo_call("b"),
                _reflection("continue"),
                _echo_call("c"),
                _echo_call("d"),
                _answer("FINAL ANSWER: 9"),
            ]
        )
        agent = _agent(llm, max_calls=6, answer_on_stop=True, steps_per_attempt=2)
        result = await agent.run_task(TASK)
        assert len(llm.calls) == 6
        assert _is_answer(llm.calls[-1])
        assert (result.status, result.result) == ("budget_exhausted", "FINAL ANSWER: 9")
        assert (result.n_main, result.n_aux, result.attempts) == (4, 2, 2)

    @pytest.mark.asyncio
    async def test_a_reflection_that_would_pass_the_limit_is_not_sent(self) -> None:
        llm = FakeLLM([_echo_call("a"), _echo_call("b"), _answer("FINAL ANSWER: 3")])
        agent = _agent(llm, max_calls=3, answer_on_stop=True, steps_per_attempt=2)
        result = await agent.run_task(TASK)
        # Two plans, then the boundary reflection would be call 3: the answer's.
        assert [_tools(call) for call in llm.calls] == [True, True, False]
        assert (result.status, result.result) == ("budget_exhausted", "FINAL ANSWER: 3")
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_without_answer_on_stop_the_run_may_send_max_calls_requests(
        self,
    ) -> None:
        llm = FakeLLM([_echo_call("a", "partial"), _echo_call("b"), _echo_call("c")])
        agent = _agent(llm, max_calls=3)
        result = await agent.run_task(TASK)
        assert len(llm.calls) == 3
        assert result.status == "budget_exhausted"
        assert result.error.startswith("Call budget reached: 3 model requests sent")
        # No answer request; the latest text the model wrote stands in.
        assert (result.result, result.answered_on_stop) == ("partial", False)
        assert "answer_on_stop" not in result.config

    @pytest.mark.asyncio
    async def test_a_failed_answer_request_leaves_the_best_effort(self) -> None:
        llm = FakeLLM([_echo_call("a", "partial"), _echo_call("b"), _echo_call("c")])
        agent = _agent(
            llm, max_calls=4, answer_on_stop=True
        )  # the answer finds no reply
        result = await agent.run_task(TASK)
        assert len(llm.calls) == 4 and _is_answer(llm.calls[-1])
        assert result.status == "budget_exhausted"
        assert (result.result, result.answered_on_stop) == ("partial", False)

    @pytest.mark.asyncio
    async def test_a_run_with_no_text_ends_with_an_empty_result(self) -> None:
        llm = FakeLLM([_echo_call("a"), _echo_call("b")])
        agent = _agent(llm, max_calls=2)
        result = await agent.run_task(TASK)
        assert (result.status, result.result) == ("budget_exhausted", "")

    @pytest.mark.asyncio
    async def test_a_retried_request_counts_each_attempt(self) -> None:
        llm = RefusingLLM(
            [_echo_call("a"), _answer("FINAL ANSWER: 5")],
            at=2,
            error=ProviderError("busy", status_code=503, retryable=True),
        )
        agent = _agent(llm, max_calls=3, answer_on_stop=True)
        agent.chat.retry_delays = (0.0, 0.0)
        result = await agent.run_task(TASK)
        # Call 1 planned, call 2 failed: a retry would be call 3, the answer's.
        assert len(llm.calls) == 3 and _is_answer(llm.calls[-1])
        assert (result.status, result.result) == ("budget_exhausted", "FINAL ANSWER: 5")
        assert (result.n_llm, result.llm_errors) == (2, 1)
        assert result.error.startswith("Call budget reached: 2 model requests sent")

    @pytest.mark.asyncio
    async def test_a_retry_inside_the_limit_still_happens(self) -> None:
        llm = RefusingLLM(
            [_echo_call("a"), _answer("FINAL ANSWER: 5")],
            at=2,
            error=ProviderError("busy", status_code=503, retryable=True),
        )
        agent = _agent(llm, max_calls=6, answer_on_stop=True)
        agent.chat.retry_delays = (0.0,)
        result = await agent.run_task(TASK)
        assert (result.status, result.result) == ("completed", "FINAL ANSWER: 5")
        assert len(llm.calls) == 3

    @pytest.mark.asyncio
    async def test_the_single_attempt_policy_stops_too(self) -> None:
        llm = FakeLLM([_echo_call("a"), _echo_call("b"), _answer("FINAL ANSWER: 1")])
        agent = _agent(llm, max_calls=3, answer_on_stop=True, attempt_policy="none")
        result = await agent.run_task(TASK)
        assert (result.status, result.result) == ("budget_exhausted", "FINAL ANSWER: 1")
        assert (len(llm.calls), result.steps) == (3, 2)

    @pytest.mark.asyncio
    async def test_unset_changes_nothing(self) -> None:
        def script() -> list[Message]:
            return [
                _echo_call("a"),
                _reflection("continue"),
                _answer("FINAL ANSWER: 4"),
            ]

        def sent(llm: FakeLLM) -> list[Any]:
            return [
                ([(m.role, m.content, m.tool_calls) for m in c["messages"]], c["tools"])
                for c in llm.calls
            ]

        def record(result: RunResult) -> dict[str, Any]:
            data = result.to_dict()
            for name in ("duration_s", "task_id", "usage"):
                del data[name]
            return data

        plain, explicit = FakeLLM(script()), FakeLLM(script())
        first = await _agent(plain, steps_per_attempt=1).run_task(TASK)
        second = await _agent(explicit, steps_per_attempt=1, max_calls=None).run_task(
            TASK
        )
        assert AgentOptions().max_calls is None
        assert first.status == "completed" and len(plain.calls) == 3
        assert sent(plain) == sent(explicit)
        assert record(first) == record(second)
        assert "max_calls" not in first.config and "early_stop" not in record(first)

    def test_the_stop_is_decided_before_the_request(self) -> None:
        chat = Chat(FakeLLM([]), max_calls=4, reserve_answer=True)
        chat.stats.n_llm = 2
        chat.stats.llm_errors = 1
        with pytest.raises(CallBudgetError, match="3 model requests sent"):
            chat.check_call_budget()
        chat.check_call_budget(closing=True)  # the reserved call
        chat.stats.llm_errors = 2
        with pytest.raises(CallBudgetError):
            chat.check_call_budget(closing=True)
        Chat(FakeLLM([])).check_call_budget()  # no limit

    def test_the_value_is_validated(self) -> None:
        assert AgentOptions(max_calls=2).max_calls == 2
        for bad in (1, 0, -3, True, "5", 2.5):
            with pytest.raises(ValueError, match="max_calls"):
                AgentOptions(max_calls=bad)  # type: ignore[arg-type]


# ── A refused request ───────────────────────────────────────────────────────


class TestRefusal:
    @pytest.mark.asyncio
    async def test_a_budget_refusal_gets_one_answer_request(self) -> None:
        llm = RefusingLLM(
            [_echo_call("a", "partial"), _echo_call("b"), _answer("FINAL ANSWER: 8")],
            at=3,
            error=ProviderError(BUDGET_BODY, status_code=400),
        )
        agent = _agent(llm, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert [_tools(call) for call in llm.calls] == [True, True, True, False]
        assert _is_answer(llm.calls[-1]) and llm.refusals == 1
        # The status and error stay the server's; the result is the answer.
        assert result.status == "provider_error"
        assert "budget_exhausted" in result.error
        assert (result.result, result.answered_on_stop) == ("FINAL ANSWER: 8", True)
        assert (result.n_llm, result.llm_errors, result.n_aux) == (3, 1, 1)

    @pytest.mark.asyncio
    async def test_a_429_gets_one_answer_request_after_its_retries(self) -> None:
        llm = RefusingLLM(
            [_echo_call("a"), _answer("FINAL ANSWER: 2")],
            at=2,
            error=ProviderError("slow down", status_code=429, retryable=True),
        )
        agent = _agent(llm, answer_on_stop=True)
        agent.chat.retry_delays = ()  # no retry: the 429 ends the planning request
        result = await agent.run_task(TASK)
        assert len(llm.calls) == 3 and _is_answer(llm.calls[-1])
        assert (result.status, result.result) == ("provider_error", "FINAL ANSWER: 2")
        assert result.answered_on_stop is True

    @pytest.mark.asyncio
    async def test_a_429_that_the_retries_do_not_clear_gets_one_answer_request(
        self,
    ) -> None:
        async def no_sleep(_: float) -> None:
            return None

        llm = RefusingLLM(
            [_echo_call("a"), _answer("FINAL ANSWER: 2")],
            at=2,
            through=4,  # the request and both of its retries
            error=ProviderError("slow down", status_code=429, retryable=True),
        )
        agent = _agent(llm, answer_on_stop=True, max_calls=30)
        agent.chat.sleep = no_sleep
        result = await agent.run_task(TASK)
        assert (len(llm.calls), llm.refusals) == (5, 3)
        assert [_tools(call) for call in llm.calls] == [True] * 4 + [False]
        assert (result.status, result.result) == ("provider_error", "FINAL ANSWER: 2")
        assert (result.answered_on_stop, result.llm_errors) == (True, 3)

    @pytest.mark.asyncio
    async def test_the_reserved_call_is_there_for_the_answer_after_a_refusal(
        self,
    ) -> None:
        llm = RefusingLLM(
            [_echo_call("a"), _answer("FINAL ANSWER: 6")],
            at=2,
            error=ProviderError(BUDGET_BODY, status_code=400),
        )
        agent = _agent(llm, answer_on_stop=True, max_calls=3)
        result = await agent.run_task(TASK)
        # Call 2 was refused; the answer is call 3 of 3.
        assert len(llm.calls) == 3 and _is_answer(llm.calls[-1])
        assert result.status == "provider_error"
        assert (result.result, result.answered_on_stop) == ("FINAL ANSWER: 6", True)
        assert result.n_llm + result.llm_errors == 3

    @pytest.mark.asyncio
    async def test_retries_of_a_429_use_up_the_limit_not_the_reserve(self) -> None:
        llm = RefusingLLM(
            [_echo_call("a"), _answer("FINAL ANSWER: 6")],
            at=2,
            through=3,
            error=ProviderError("slow down", status_code=429, retryable=True),
        )
        agent = _agent(llm, answer_on_stop=True, max_calls=4)
        agent.chat.retry_delays = (0.0, 0.0)
        result = await agent.run_task(TASK)
        # Calls 2 and 3 are refused, the retry that would be call 4 is not sent;
        # call 4 is the answer's.
        assert len(llm.calls) == 4 and llm.refusals == 2
        assert _is_answer(llm.calls[-1])
        assert result.status == "budget_exhausted"
        assert (result.result, result.answered_on_stop) == ("FINAL ANSWER: 6", True)

    @pytest.mark.asyncio
    async def test_an_answer_request_that_is_refused_too_leaves_the_best_effort(
        self,
    ) -> None:
        class Refusing(FakeLLM):
            async def chat(self, messages: list[Message], tools: Any = None) -> Message:
                if len(self.calls) >= 1:
                    self.calls.append({"messages": list(messages), "tools": tools})
                    raise ProviderError(BUDGET_BODY, status_code=400)
                return await super().chat(messages, tools)

        llm = Refusing([_echo_call("a", "partial")])
        agent = _agent(llm, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert [_tools(call) for call in llm.calls] == [True, True, False]
        assert (result.status, result.result) == ("provider_error", "partial")
        assert result.answered_on_stop is False
        assert "budget_exhausted" in result.error

    @pytest.mark.asyncio
    async def test_a_refusal_with_no_text_at_all_ends_empty(self) -> None:
        llm = RefusingLLM(
            [],
            at=1,
            error=ProviderError(BUDGET_BODY, status_code=400),
        )
        agent = _agent(llm, answer_on_stop=True)
        agent.chat.retry_delays = ()
        result = await agent.run_task(TASK)
        assert (result.status, result.result) == ("provider_error", "")

    @pytest.mark.asyncio
    async def test_another_error_is_not_answered(self) -> None:
        llm = RefusingLLM(
            [_echo_call("a")],
            at=2,
            error=ProviderError("server gone", status_code=400),
        )
        agent = _agent(llm, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert len(llm.calls) == 2
        assert (result.status, result.result, result.answered_on_stop) == (
            "provider_error",
            "",
            False,
        )

    @pytest.mark.asyncio
    async def test_off_by_default_a_refusal_ends_as_before(self) -> None:
        llm = RefusingLLM(
            [_echo_call("a", "partial")],
            at=2,
            error=ProviderError(BUDGET_BODY, status_code=400),
        )
        result = await _agent(llm).run_task(TASK)
        assert len(llm.calls) == 2
        assert (result.status, result.result) == ("provider_error", "")
        assert "answered_on_stop" not in result.to_dict()

    @pytest.mark.parametrize(
        ("error", "refusal"),
        [
            (ProviderError("x", status_code=429, retryable=True), True),
            (ProviderError(BUDGET_BODY, status_code=400), True),
            (ProviderError("POST failed with HTTP 400: budget exceeded"), True),
            (ProviderError("the run has used its budget"), True),
            (ProviderError("HTTP 400: Rate limit reached for requests"), True),
            (ProviderError("POST failed with HTTP 400: model not found"), False),
            (ProviderError("POST failed with HTTP 500: boom", status_code=500), False),
            (ProviderError("server gone"), False),
            (ProviderError("context_length_exceeded"), False),
        ],
    )
    def test_what_counts_as_a_refusal(
        self, error: ProviderError, refusal: bool
    ) -> None:
        assert is_budget_refusal(error) is refusal


def test_the_options_dataclass_still_freezes() -> None:
    options = AgentOptions(max_calls=5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.max_calls = 6  # type: ignore[misc]
