"""reflection_gate="patient", answer_on_stop and the reflection list caps.

The gate table, the once-per-run allowance, the extra request for an answer when a
run stops without one, and the caps on a reflection's lists. ``patient`` and
``answer_on_stop`` are the defaults now; these tests build their agents with the
earlier values (``tests/legacy_options.py``) and turn each on by name. No real model.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.agent._chat import (
    ANSWER_EVIDENCE_CHARS,
    ANSWER_ON_STOP_NOTE,
    LEAN_REFLECTION_PROMPT,
    REFLECTION_PROMPT,
)
from localmelo.melo.agent.agent import (
    REFLECTION_ITEM_CHARS,
    REFLECTION_LIST_ITEMS,
    _cap_reflection_lists,
    _gate_stop_reason,
    _should_continue,
    _stop_reason,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.schema import Message, ReflectionEntry
from localmelo.tests.agent.test_reflection_gate import (
    FakeLLM,
    _added_reflections,
    _agent,
    _answer,
    _echo_call,
    _passing_decision,
    _reflection,
)

TASK = (
    "Use echo to find the person.\nEnd your reply with one line: FINAL ANSWER: <answer>"
)
# The same task without the closing line: the format request (see
# test_answer_format.py) asks nothing about it.
PLAIN_TASK = "Use echo to find the person."
STALE = {"next_step_is_novel": False}  # a continue that is only not novel
# The cache options of a plain run today (the shared helpers send the earlier ones).
DEFAULT_CACHE: dict[str, Any] = {
    "fixed_prefix": True,
    "window_eviction": "chunk",
    "reflect_tool_guard": True,
    "cache_layout": "v3",
    "window_messages": 64,
}


def _is_answer_request(call: dict[str, Any]) -> bool:
    return ANSWER_ON_STOP_NOTE in call["messages"][-1].content


def _answer_requests(llm: FakeLLM) -> list[dict[str, Any]]:
    return [call for call in llm.calls if _is_answer_request(call)]


# ── The gate ────────────────────────────────────────────────────────────────


class TestGateTable:
    @pytest.mark.parametrize(
        ("feasible", "concrete", "novel"),
        list(itertools.product([True, False], repeat=3)),
    )
    @pytest.mark.parametrize("allowance", [True, False])
    def test_patient_differs_from_flags_only_on_a_not_novel_continue(
        self, feasible: bool, concrete: bool, novel: bool, allowance: bool
    ) -> None:
        decision = _passing_decision(
            task_still_feasible=feasible,
            next_step_is_concrete=concrete,
            next_step_is_novel=novel,
        )
        flags = _gate_stop_reason(decision, "flags")
        patient = _gate_stop_reason(decision, "patient", patient=allowance)
        if feasible and concrete and not novel and allowance:
            assert (flags, patient) == ("not_novel", None)
        else:
            assert patient == flags

    def test_without_an_allowance_patient_is_flags(self) -> None:
        decision = _passing_decision(next_step_is_novel=False)
        assert _should_continue(decision, "patient") is False
        assert _gate_stop_reason(decision, "patient") == "not_novel"

    def test_the_allowance_does_not_help_the_other_gates(self) -> None:
        decision = _passing_decision(next_step_is_novel=False)
        assert _gate_stop_reason(decision, "flags", patient=True) == "not_novel"
        assert _gate_stop_reason(decision, "utility", patient=True) == "not_novel"

    @pytest.mark.parametrize("action", ["stop", "decompose", "done"])
    def test_only_a_continue_is_helped(self, action: str) -> None:
        decision = _passing_decision(
            recommended_action=action, next_step_is_novel=False
        )
        assert _gate_stop_reason(decision, "patient", patient=True) == (
            "action_not_continue"
        )

    def test_the_stop_reason_follows_the_allowance(self) -> None:
        decision = _passing_decision(next_step_is_novel=False)
        assert _stop_reason(decision, "patient", "", patient=True) is None
        assert _stop_reason(decision, "patient", "") == "not_novel"
        # A budget boundary and an unreadable reply stop whatever the allowance.
        assert _stop_reason(decision, "patient", "Max steps reached", patient=True) == (
            "step_budget_used"
        )
        assert _stop_reason(None, "patient", "Max steps reached", patient=True) == (
            "unreadable_reply"
        )


class TestOncePerRun:
    @staticmethod
    async def _run(
        replies: list[Message], **options: Any
    ) -> tuple[Any, list[Any], FakeLLM]:
        events: list[AgentEvent] = []
        llm = FakeLLM(replies)
        agent = _agent(llm, steps_per_attempt=1, max_attempts=5, **options)
        result = await agent.run_task("echo the answer", on_event=events.append)
        return result, [e.data for e in events if e.type == "reflection"], llm

    @pytest.mark.asyncio
    async def test_flags_stop_at_the_first_not_novel_continue(self) -> None:
        result, [event], _ = await self._run(
            [_echo_call("a"), _reflection("continue", rationale="again", **STALE)]
        )
        assert (result.status, result.attempts) == ("gate_stopped", 1)
        assert event["stop_reason"] == "not_novel"

    @pytest.mark.asyncio
    async def test_patient_starts_one_more_attempt_with_the_avoids(self) -> None:
        reflection = _reflection(
            "continue", recommended_avoids=["do not click the banner"], **STALE
        )
        result, [event], llm = await self._run(
            [_echo_call("a"), reflection, _answer("42")], reflection_gate="patient"
        )
        assert (result.status, result.result, result.attempts) == ("completed", "42", 2)
        assert (event["gate"], event["continue_run"], event["stop_reason"]) == (
            "patient",
            True,
            None,
        )
        assert event["flags"]["next_step_is_novel"] is False
        # The next attempt's planning request carries the avoids, as under flags.
        assert "do not click the banner" in llm.calls[-1]["messages"][0].content

    @pytest.mark.asyncio
    async def test_a_second_not_novel_continue_stops_as_flags_do(self) -> None:
        result, events, _ = await self._run(
            [
                _echo_call("a"),
                _reflection("continue", **STALE),
                _echo_call("b"),
                _reflection("continue", rationale="still stuck", **STALE),
            ],
            reflection_gate="patient",
        )
        assert (result.status, result.attempts, result.error) == (
            "gate_stopped",
            2,
            "still stuck",
        )
        assert [(e["continue_run"], e["stop_reason"]) for e in events] == [
            (True, None),
            (False, "not_novel"),
        ]

    @pytest.mark.asyncio
    async def test_novel_continues_do_not_spend_the_allowance(self) -> None:
        result, events, _ = await self._run(
            [
                _echo_call("a"),
                _reflection("continue"),
                _echo_call("b"),
                _reflection("continue", **STALE),
                _echo_call("c"),
                _reflection("continue", **STALE),
            ],
            reflection_gate="patient",
        )
        assert (result.status, result.attempts) == ("gate_stopped", 3)
        assert [e["continue_run"] for e in events] == [True, True, False]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("check", ["task_still_feasible", "next_step_is_concrete"])
    async def test_the_allowance_covers_only_novelty(self, check: str) -> None:
        result, [event], _ = await self._run(
            [_echo_call("a"), _reflection("continue", **{check: False, **STALE})],
            reflection_gate="patient",
        )
        assert result.status == "gate_stopped"
        assert event["stop_reason"] != "not_novel"

    @pytest.mark.asyncio
    async def test_patient_reads_a_reflection_as_flags_do(self) -> None:
        # No estimates are needed: a low utility does not stop it.
        reply = _reflection(
            "continue", estimated_info_gain=0.0, estimated_cost=1.0, repeat_risk=1.0
        )
        result, _, _ = await self._run(
            [_echo_call("a"), reply, _answer("42")], reflection_gate="patient"
        )
        assert (result.status, result.attempts) == ("completed", 2)


class TestPatientReadsLikeFlags:
    @pytest.mark.parametrize(
        ("gate", "lean"),
        [("flags", True), ("patient", True), ("utility", False)],
    )
    def test_the_lean_reflection_prompt_goes_with_the_flag_style_gates(
        self, gate: Any, lean: bool
    ) -> None:
        agent = _agent(FakeLLM([]), lean_output=True, reflection_gate=gate)
        expected = LEAN_REFLECTION_PROMPT if lean else REFLECTION_PROMPT
        assert agent.chat.reflection_prompt == expected


# ── An answer when a run stops ──────────────────────────────────────────────


async def _stopped(
    *replies: Message, query: str = TASK, **options: Any
) -> tuple[Any, FakeLLM]:
    """A run whose first reflection stops it without an answer."""
    llm = FakeLLM(
        [_echo_call("page text about Lovelace"), _reflection("continue", **STALE)]
        + list(replies)
    )
    agent = _agent(llm, steps_per_attempt=1, max_attempts=3, **options)
    return await agent.run_task(query), llm


class TestAnswerOnStop:
    @pytest.mark.asyncio
    async def test_a_gate_stopped_run_gets_its_answer(self) -> None:
        reply = _answer("Ada Lovelace\nFINAL ANSWER: Ada Lovelace")
        result, llm = await _stopped(reply, answer_on_stop=True)
        assert result.status == "gate_stopped"
        assert result.result == "Ada Lovelace\nFINAL ANSWER: Ada Lovelace"
        assert result.answered_on_stop is True
        assert result.to_dict()["answered_on_stop"] is True
        assert result.config["answer_on_stop"] is True
        assert result.error == "test"  # the reflection's rationale, as before
        # One request more, auxiliary: two aux (reflection, answer), one main.
        assert (result.n_main, result.n_aux, result.n_llm) == (1, 2, 3)
        assert len(llm.calls) == 3

    @pytest.mark.asyncio
    async def test_the_request_has_no_tools_and_the_evidence(self) -> None:
        _, llm = await _stopped(_answer("FINAL ANSWER: x"), answer_on_stop=True)
        [call] = _answer_requests(llm)
        assert not call["tools"]
        text = call["messages"][-1].content
        assert call is llm.calls[-1]
        assert text.startswith("Original task: " + TASK)  # the format line stays
        assert "page text about Lovelace" in text  # the window
        assert "tried echo" in text  # the latest reflection's summary
        assert text.endswith(ANSWER_ON_STOP_NOTE)
        assert "unknown" in ANSWER_ON_STOP_NOTE

    @pytest.mark.asyncio
    async def test_the_default_layout_shows_the_reflection_once(self) -> None:
        # fixed prefix, cache layout v3: the reflection is already in the window.
        llm = FakeLLM([_echo_call("a"), _reflection("continue", **STALE), _answer("7")])
        agent = _agent(llm, steps_per_attempt=1, answer_on_stop=True, **DEFAULT_CACHE)
        result = await agent.run_task("echo")
        assert (result.status, result.result) == ("gate_stopped", "7")
        [call] = _answer_requests(llm)
        text = call["messages"][-1].content
        assert text.count("tried echo") == 1  # not again as the exchange's reply
        assert REFLECTION_PROMPT not in text  # nor the exchange's request
        assert call["messages"][0].role == "system"

    @pytest.mark.asyncio
    async def test_unknown_passes_through(self) -> None:
        result, _ = await _stopped(
            _answer("FINAL ANSWER: unknown"), answer_on_stop=True
        )
        assert (result.status, result.result, result.answered_on_stop) == (
            "gate_stopped",
            "FINAL ANSWER: unknown",
            True,
        )

    @pytest.mark.asyncio
    async def test_a_budget_exhausted_run_gets_its_answer(self) -> None:
        llm = FakeLLM(
            [_echo_call("a"), _reflection("continue"), _answer("FINAL ANSWER: 42")]
        )
        agent = _agent(llm, steps_per_attempt=1, max_attempts=1, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert (result.status, result.error) == (
            "budget_exhausted",
            "Max attempts reached",
        )
        assert (result.result, result.answered_on_stop) == ("FINAL ANSWER: 42", True)
        assert len(_answer_requests(llm)) == 1

    @pytest.mark.asyncio
    async def test_a_single_attempt_run_that_runs_out_of_steps_gets_one_too(
        self,
    ) -> None:
        llm = FakeLLM([_echo_call("a"), _echo_call("b"), _answer("FINAL ANSWER: 1")])
        agent = _agent(llm, attempt_policy="none", max_steps=2, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert (result.status, result.result) == ("budget_exhausted", "FINAL ANSWER: 1")
        assert result.answered_on_stop is True

    @pytest.mark.asyncio
    async def test_a_think_block_is_not_part_of_the_answer(self) -> None:
        result, _ = await _stopped(
            _answer("<think>maybe</think>FINAL ANSWER: 5"), answer_on_stop=True
        )
        assert result.result == "FINAL ANSWER: 5"

    @pytest.mark.asyncio
    async def test_no_request_when_the_run_has_an_answer(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a"),
                _reflection("continue", best_effort_result="7", **STALE),
            ]
        )
        agent = _agent(llm, steps_per_attempt=1, answer_on_stop=True)
        result = await agent.run_task(PLAIN_TASK)
        assert (result.status, result.result) == ("gate_stopped", "7")
        assert result.answered_on_stop is False
        assert "answered_on_stop" not in result.to_dict()
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_no_request_for_other_statuses(self) -> None:
        llm = FakeLLM([_answer("42")])
        agent = _agent(llm, answer_on_stop=True)
        result = await agent.run_task(PLAIN_TASK)
        assert (result.status, result.result) == ("completed", "42")
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_off_when_turned_off(self) -> None:
        result, llm = await _stopped(_answer("never asked"), answer_on_stop=False)
        assert (result.status, result.result) == ("gate_stopped", "")
        assert "answered_on_stop" not in result.to_dict()
        assert "answer_on_stop" not in result.config
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_a_request_that_raises_leaves_the_run_as_it_was(self) -> None:
        failing = FakeLLM(
            [_echo_call("a"), _reflection("continue", **STALE)]  # then: nothing queued
        )
        agent = _agent(failing, steps_per_attempt=1, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert (result.status, result.result, result.answered_on_stop) == (
            "gate_stopped",
            "",
            False,
        )

    @pytest.mark.asyncio
    async def test_a_provider_error_does_not_replace_the_status(self) -> None:
        class Failing(FakeLLM):
            async def chat(self, messages: list[Message], tools: Any = None) -> Message:
                if len(self.calls) >= 2:  # the third request is the answer's
                    self.calls.append({"messages": messages, "tools": tools})
                    raise ProviderError("server gone", status_code=400)
                return await super().chat(messages, tools)

        llm = Failing([_echo_call("a"), _reflection("continue", **STALE)])
        agent = _agent(llm, steps_per_attempt=1, answer_on_stop=True)
        result = await agent.run_task(TASK)
        assert (result.status, result.result) == ("gate_stopped", "")
        assert (result.llm_errors, result.answered_on_stop) == (1, False)

    @pytest.mark.asyncio
    async def test_a_blank_reply_is_no_answer(self) -> None:
        result, _ = await _stopped(_answer("   "), answer_on_stop=True)
        assert (result.result, result.answered_on_stop) == ("", False)

    @pytest.mark.asyncio
    async def test_a_long_window_is_cut_to_its_newest_part(self) -> None:
        big = "old " * (ANSWER_EVIDENCE_CHARS // 4 + 500)
        llm = FakeLLM(
            [
                _echo_call(big),
                _echo_call("newest evidence"),
                _reflection("continue", **STALE),
                _answer("FINAL ANSWER: 3"),
            ]
        )
        agent = _agent(llm, steps_per_attempt=2, answer_on_stop=True)
        result = await agent.run_task(TASK)
        text = _answer_requests(llm)[0]["messages"][-1].content
        assert result.result == "FINAL ANSWER: 3"
        assert "newest evidence" in text
        assert "[earlier turns left out]" in text
        assert len(text) < ANSWER_EVIDENCE_CHARS + 4_000


# ── The caps ────────────────────────────────────────────────────────────────


def _entry(**lists: Any) -> ReflectionEntry:
    return dataclasses.replace(ReflectionEntry(attempt_id=1, summary="s"), **lists)


class TestCaps:
    def test_a_list_keeps_its_first_eight_items(self) -> None:
        items = [f"item {n}" for n in range(12)]
        capped = _cap_reflection_lists(_entry(useful_evidence=items))
        assert capped.useful_evidence == items[:REFLECTION_LIST_ITEMS]
        assert REFLECTION_LIST_ITEMS == 8

    def test_an_item_is_cut_to_300_characters(self) -> None:
        long = "word " * 200
        [item] = _cap_reflection_lists(
            _entry(failed_hypotheses=[long])
        ).failed_hypotheses
        assert len(item) <= REFLECTION_ITEM_CHARS == 300
        assert item.endswith("…")
        assert long.startswith(item[:-1])

    def test_the_other_fields_and_the_entry_itself_are_kept(self) -> None:
        entry = _entry(
            useful_evidence=["e"] * 9,
            next_promising_directions=["d"] * 12,
            unresolved_questions=["q" * 400],
        )
        capped = _cap_reflection_lists(entry)
        assert capped.next_promising_directions == ["d"] * 12
        assert capped.unresolved_questions == ["q" * 400]
        assert capped.summary == "s"
        assert len(entry.useful_evidence) == 9  # not changed in place

    def test_a_list_within_the_caps_is_unchanged(self) -> None:
        entry = _entry(recommended_avoids=["a", "b"])
        assert _cap_reflection_lists(entry) == entry

    @staticmethod
    def _dump() -> dict[str, Any]:
        return {
            "useful_evidence": [f"thought {n} " + "x" * 280 for n in range(40)],
            "recommended_avoids": ["a" * 900, "b"],
            "failed_hypotheses": [f"h{n}" for n in range(20)],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "options", [{"reflection_gate": "patient"}, {"answer_on_stop": True}]
    )
    async def test_a_stored_reflection_is_capped_when_an_option_is_on(
        self, options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = FakeLLM(
            [_echo_call("a"), _reflection("continue", **self._dump(), **STALE)]
        )
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2, **options)
        added = _added_reflections(agent, monkeypatch)
        await agent.run_task("echo")
        [entry] = added
        assert 0 < len(entry.useful_evidence) <= REFLECTION_LIST_ITEMS
        assert len(entry.failed_hypotheses) <= REFLECTION_LIST_ITEMS
        assert all(
            len(item) <= REFLECTION_ITEM_CHARS for item in entry.recommended_avoids
        )
        assert entry.recommended_avoids[0].startswith("a")

    @pytest.mark.asyncio
    async def test_the_default_keeps_a_reflection_as_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dump = {"failed_hypotheses": [f"h{n}" for n in range(12)]}
        llm = FakeLLM([_echo_call("a"), _reflection("continue", **dump, **STALE)])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=2)
        added = _added_reflections(agent, monkeypatch)
        await agent.run_task("echo")
        assert added[0].failed_hypotheses == dump["failed_hypotheses"]  # 12 items

    @pytest.mark.asyncio
    async def test_the_model_sees_only_the_capped_lists(self) -> None:
        dump = {"failed_hypotheses": [f"hypothesis-{n}" for n in range(12)]}
        replies = [_echo_call("a"), _reflection("continue", **dump), _answer("42")]
        llm = FakeLLM(replies)
        agent = _agent(
            llm, steps_per_attempt=1, max_attempts=2, reflection_gate="patient"
        )
        await agent.run_task("echo")
        system = llm.calls[-1]["messages"][0].content
        assert "hypothesis-7" in system
        assert "hypothesis-8" not in system


class TestOptionValues:
    def test_the_gate_names_are_validated(self) -> None:
        assert AgentOptions(reflection_gate="patient").reflection_gate == "patient"
        with pytest.raises(ValueError, match="reflection_gate"):
            AgentOptions(reflection_gate="impatient")  # type: ignore[arg-type]

    def test_the_defaults_are_the_patient_gate_and_the_answer_request(self) -> None:
        options = AgentOptions()
        assert (options.reflection_gate, options.answer_on_stop) == ("patient", True)

    def test_answer_on_stop_must_be_a_bool(self) -> None:
        with pytest.raises(ValueError, match="answer_on_stop"):
            AgentOptions(answer_on_stop="on")  # type: ignore[arg-type]
