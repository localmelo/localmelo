"""The closing line a task asks for: detected in the task text, restored by one request.

``answer_on_stop`` (on by default) adds the line when a run's answer lacks it. No
real model.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.melo.agent._answer_format import (
    closing_line_of,
    has_closing_line,
    requested_closing_label,
)
from localmelo.melo.agent._chat import CLOSING_LINE_NOTE
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.schema import Message
from localmelo.tests.agent.test_reflection_gate import (
    FakeLLM,
    _agent,
    _answer,
    _echo_call,
    _reflection,
)

TASK = (
    "Who ran the 1st Marais-Viljoen expedition? Use echo to look it up.\n"
    "End your reply with one line: FINAL ANSWER: <answer>"
)
PLAIN = "Use echo to look up who ran the expedition."
NOTE = CLOSING_LINE_NOTE.format(label="FINAL ANSWER:")


def _closing_requests(llm: FakeLLM) -> list[dict[str, Any]]:
    return [call for call in llm.calls if NOTE in call["messages"][-1].content]


# ── Finding the line in the task's text ─────────────────────────────────────


@pytest.mark.parametrize(
    ("task", "label"),
    [
        (TASK, "FINAL ANSWER:"),
        ("Name the capital. Your last line must start with 'Answer:'.", "Answer:"),
        ("Reply with `FINAL ANSWER: <name>` on the last line.", "FINAL ANSWER:"),
        ('Finish your answer with the line "Final Answer: X".', "Final Answer:"),
        ("Give the total. End with a line like RESULT: <number>", "RESULT:"),
        ("Summarize it, then close with `DONE: <one sentence>`.", "DONE:"),
        ("Answer format: FINAL ANSWER: <short answer>", "FINAL ANSWER:"),
        ("请回答。最后一行以“最终答案：”开头。", "最终答案："),
        ("Put it in a line of the form 'Answer: <x>'.", "Answer:"),
        ("Think, then reply in the format Final Answer: <x>", "Final Answer:"),
    ],
)
def test_a_label_the_task_names_as_its_closing_line(task: str, label: str) -> None:
    assert requested_closing_label(task) == label


@pytest.mark.parametrize(
    "task",
    [
        "What is 2 + 2?",
        PLAIN,
        "Q: capital of France? Answer: Paris. Now solve 2 + 2.",
        "Example: Question: 2+2? Answer: 4",
        "IMPORTANT: end your reply with a short summary.",
        "Don't forget the format: be brief and polite.",
        "Write the final answer: a name, as plain text.",
        "NOTE: the last line of the file is blank.",
        "Use the 'Answer:' field of the form below.",
        "",
    ],
)
def test_a_task_with_no_such_line_asks_for_nothing(task: str) -> None:
    assert requested_closing_label(task) is None


def test_the_last_label_wins() -> None:
    task = "End with 'Answer: <x>'. Actually, end with the line FINAL ANSWER: <x>"
    assert requested_closing_label(task) == "FINAL ANSWER:"


@pytest.mark.parametrize(
    ("answer", "has"),
    [
        ("Ada\nFINAL ANSWER: Ada", True),
        ("FINAL ANSWER: Ada\nthanks", True),
        ("**FINAL ANSWER:** Ada", True),
        ("> FINAL ANSWER: Ada", True),
        ("Ada. The FINAL ANSWER: is Ada", False),
        ("Final answer: Ada", False),
        ("Ada", False),
        ("", False),
    ],
)
def test_a_line_that_starts_with_the_label(answer: str, has: bool) -> None:
    assert has_closing_line(answer, "FINAL ANSWER:") is has


def test_the_colon_may_be_full_width() -> None:
    assert has_closing_line("结论\n最终答案: 42", "最终答案：")
    assert has_closing_line("最终答案：42", "最终答案:")
    assert closing_line_of("最终答案: 42", "最终答案：") == "最终答案: 42"


def test_the_closing_line_of_a_reply() -> None:
    reply = "Sure.\nFINAL ANSWER: Marais-Viljoen\n"
    assert closing_line_of(reply, "FINAL ANSWER:") == "FINAL ANSWER: Marais-Viljoen"
    assert closing_line_of("**FINAL ANSWER: 7**", "FINAL ANSWER:") == "FINAL ANSWER: 7"
    assert closing_line_of("Marais-Viljoen", "FINAL ANSWER:") is None
    assert closing_line_of("", "FINAL ANSWER:") is None


# ── The request ─────────────────────────────────────────────────────────────


class TestRestoredLine:
    @pytest.mark.asyncio
    async def test_the_right_content_without_the_line_gets_it(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("Marais-Viljoen"),
                _answer("The expedition was led by Marais-Viljoen."),
                _answer("FINAL ANSWER: Marais-Viljoen"),
            ]
        )
        result = await _agent(llm, answer_on_stop=True).run_task(TASK)
        assert result.status == "completed"
        assert result.result == (
            "The expedition was led by Marais-Viljoen.\n" "FINAL ANSWER: Marais-Viljoen"
        )
        assert result.format_fixed is True
        assert result.to_dict()["format_fixed"] is True
        assert result.answered_on_stop is False
        # One request more, auxiliary, without tools.
        assert (result.n_main, result.n_aux, result.n_llm) == (2, 1, 3)
        [call] = _closing_requests(llm)
        assert call is llm.calls[-1]
        assert not call["tools"]
        text = call["messages"][-1].content
        assert text.startswith("Original task: " + TASK)
        assert "The expedition was led by Marais-Viljoen." in text
        assert text.endswith(NOTE)

    @pytest.mark.asyncio
    async def test_the_event_of_the_finished_run_carries_the_key(self) -> None:
        events: list[Any] = []
        llm = FakeLLM([_answer("Ada"), _answer("FINAL ANSWER: Ada")])
        result = await _agent(llm, answer_on_stop=True).run_task(
            TASK, on_event=events.append
        )
        [finished] = [e for e in events if e.type == "run.finished"]
        assert finished.data["result"]["format_fixed"] is True
        assert finished.data["result"]["result"] == result.result

    @pytest.mark.asyncio
    async def test_a_stopped_run_is_answered_and_then_given_the_line(self) -> None:
        stale = {"next_step_is_novel": False}
        llm = FakeLLM(
            [
                _echo_call("page text"),
                _reflection("continue", **stale),
                _answer("It was Marais-Viljoen."),  # the answer on stop
                _answer("FINAL ANSWER: Marais-Viljoen"),  # the closing line
            ]
        )
        agent = _agent(
            llm,
            steps_per_attempt=1,
            max_attempts=3,
            reflection_gate="flags",
            answer_on_stop=True,
        )
        result = await agent.run_task(TASK)
        assert result.status == "gate_stopped"
        assert result.result == "It was Marais-Viljoen.\nFINAL ANSWER: Marais-Viljoen"
        assert (result.answered_on_stop, result.format_fixed) == (True, True)
        assert len(llm.calls) == 4

    @pytest.mark.asyncio
    async def test_an_answer_with_the_line_is_left_alone(self) -> None:
        llm = FakeLLM([_answer("Ada\nFINAL ANSWER: Ada")])
        result = await _agent(llm, answer_on_stop=True).run_task(TASK)
        assert result.result == "Ada\nFINAL ANSWER: Ada"
        assert result.format_fixed is False
        assert "format_fixed" not in result.to_dict()
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_a_task_with_no_format_is_untouched(self) -> None:
        llm = FakeLLM([_answer("Ada")])
        result = await _agent(llm, answer_on_stop=True).run_task(PLAIN)
        assert (result.status, result.result) == ("completed", "Ada")
        assert result.format_fixed is False
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_an_empty_answer_is_not_asked_about(self) -> None:
        llm = FakeLLM([_answer("   ")])
        result = await _agent(llm, answer_on_stop=True).run_task(TASK)
        assert result.format_fixed is False
        assert len(_closing_requests(llm)) == 0

    @pytest.mark.asyncio
    async def test_off_without_answer_on_stop(self) -> None:
        llm = FakeLLM([_answer("Ada")])
        result = await _agent(llm, answer_on_stop=False).run_task(TASK)
        assert (result.result, result.format_fixed) == ("Ada", False)
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_a_think_block_stays_and_the_line_follows_the_answer(self) -> None:
        llm = FakeLLM([_answer("<think>hm</think>Ada"), _answer("FINAL ANSWER: Ada")])
        result = await _agent(llm, answer_on_stop=True).run_task(TASK)
        assert result.result.endswith("Ada\nFINAL ANSWER: Ada")
        assert result.format_fixed is True
        [call] = _closing_requests(llm)
        assert "hm" not in call["messages"][-1].content  # the thinking is not sent

    @pytest.mark.asyncio
    async def test_the_default_options_ask_for_the_line(self) -> None:
        # No answer_on_stop given: the default is on.
        from localmelo.melo.agent import Agent, AgentOptions

        llm = FakeLLM([_answer("Ada"), _answer("FINAL ANSWER: Ada")])
        agent = Agent(llm=llm, options=AgentOptions(builtins=frozenset()))
        result = await agent.run_task(TASK)
        assert (result.result, result.format_fixed) == ("Ada\nFINAL ANSWER: Ada", True)


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_request_that_raises_keeps_the_answer(self) -> None:
        class Failing(FakeLLM):
            async def chat(self, messages: list[Message], tools: Any = None) -> Message:
                if self.calls:  # the second request is the closing line's
                    self.calls.append({"messages": messages, "tools": tools})
                    raise ProviderError("server gone", status_code=400)
                return await super().chat(messages, tools)

        llm = Failing([_answer("Ada")])
        result = await _agent(llm, answer_on_stop=True).run_task(TASK)
        assert (result.status, result.result) == ("completed", "Ada")
        assert (result.format_fixed, result.llm_errors) == (False, 1)

    @pytest.mark.asyncio
    async def test_a_reply_without_the_line_keeps_the_answer(self) -> None:
        llm = FakeLLM([_answer("Ada"), _answer("I am not sure how to put it.")])
        result = await _agent(llm, answer_on_stop=True).run_task(TASK)
        assert (result.result, result.format_fixed) == ("Ada", False)
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_a_blank_reply_keeps_the_answer(self) -> None:
        llm = FakeLLM([_answer("Ada"), _answer("  ")])
        result = await _agent(llm, answer_on_stop=True).run_task(TASK)
        assert (result.result, result.format_fixed) == ("Ada", False)

    @pytest.mark.asyncio
    async def test_no_call_left_keeps_the_answer(self) -> None:
        # max_calls=2 keeps the last call for the answer on stop; the closing line
        # would be a third request.
        llm = FakeLLM([_echo_call("a"), _answer("Ada")])
        result = await _agent(llm, answer_on_stop=True, max_calls=2).run_task(TASK)
        assert result.status == "budget_exhausted"
        assert (result.result, result.answered_on_stop) == ("Ada", True)
        assert result.format_fixed is False
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_other_statuses_are_not_asked_about(self) -> None:
        crashed = FakeLLM([])  # the first request fails, so the run does not complete
        result = await _agent(crashed, answer_on_stop=True).run_task(TASK)
        assert result.status != "completed"
        assert result.format_fixed is False
        assert len(crashed.calls) == 1
