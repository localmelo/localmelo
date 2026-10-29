"""Brief runs only (``AgentOptions.brief_check``): the commands on the brief's
``Check:`` lines run before the run ends ``completed``, and a failing one sends
the model back to work, at most twice."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from localmelo.melo.agent._brief_gate import (
    CHECK_FAILED_LINE,
    MAX_CHECK_RETRIES,
    BriefCheckResult,
    check_commands,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.executor import Executor
from localmelo.melo.schema import Message
from localmelo.tests.agent.test_reflection_gate import (
    FakeLLM,
    _agent,
    _answer,
    _echo_call,
    _reflection,
)

QUERY = "echo the answer\nCheck: pytest -q\n"
OUTPUT = "\n".join(f"line {n}" for n in range(1, 4))


def _result(command: str, exit_code: int | None, **fields: Any) -> BriefCheckResult:
    return BriefCheckResult(command, exit_code, 0.25, OUTPUT, **fields)


class Checks:
    """A ``brief_check`` that answers with *exit_codes* in turn, the last one on."""

    def __init__(self, *exit_codes: int | None, refused: str = "") -> None:
        self.exit_codes = list(exit_codes)
        self.refused = refused
        self.commands: list[str] = []

    async def __call__(self, executor: Executor, command: str) -> BriefCheckResult:
        self.commands.append(command)
        code = (
            self.exit_codes.pop(0) if len(self.exit_codes) > 1 else self.exit_codes[0]
        )
        return _result(command, code, refused=self.refused)


async def _run(
    replies: list[Any], check: Any, query: str = QUERY, **options: Any
) -> tuple[Any, FakeLLM, list[AgentEvent]]:
    events: list[AgentEvent] = []
    llm = FakeLLM(replies)
    agent = _agent(llm, brief_check=check, **options)
    result = await agent.run_task(query, on_event=events.append)
    return result, llm, events


def _notices(events: list[AgentEvent], kind: str) -> list[Mapping[str, Any]]:
    return [e.data for e in events if e.type == "notice" and e.data["kind"] == kind]


def _bodies(llm: FakeLLM) -> list[list[tuple[str, str]]]:
    return [[(m.role, m.content) for m in call["messages"]] for call in llm.calls]


def _last_user(llm: FakeLLM, request: int) -> str:
    messages: list[Message] = llm.calls[request]["messages"]
    return next(m.content for m in reversed(messages) if m.role == "user")


def test_the_check_lines_are_parsed() -> None:
    brief = (
        "Fix it.\nCheck: uv run pytest tests/a -q\n  Check: indented\n"
        "check: lower case\nCheck:\nCheck:   make lint  \nSee Check: inline\n"
    )
    assert check_commands(brief) == ["uv run pytest tests/a -q", "make lint"]
    assert check_commands("no checks here") == []


class TestTheGate:
    @pytest.mark.asyncio
    async def test_a_failing_check_sends_the_model_back_to_work(self) -> None:
        check = Checks(1, 0)
        result, llm, events = await _run([_answer("done"), _answer("fixed")], check)
        assert (result.status, result.result) == ("completed", "fixed")
        assert check.commands == ["pytest -q", "pytest -q"]
        assert len(llm.calls) == 2
        told = _last_user(llm, 1)
        assert told == (
            f"Command: pytest -q\nExit code: 1\nLast 60 lines of output:\n{OUTPUT}"
            f"\n\n{CHECK_FAILED_LINE}"
        )
        assert CHECK_FAILED_LINE == (
            "The task's check failed. Fix what it shows, run it again, then finish."
        )
        roles = [m.role for m in llm.calls[1]["messages"]]
        assert roles[-3:] == ["user", "assistant", "user"]  # the answer, then the news
        first, second = _notices(events, "brief_check")
        assert (first["detail"]["round"], first["detail"]["exit_code"]) == (1, 1)
        assert (second["detail"]["round"], second["detail"]["exit_code"]) == (2, 0)
        assert first["detail"]["command"] == "pytest -q"
        assert first["detail"]["duration_s"] == 0.25
        assert not _notices(events, "brief_check_failed")

    @pytest.mark.asyncio
    async def test_a_passing_check_changes_nothing(self) -> None:
        check = Checks(0)
        result, llm, events = await _run([_answer("done")], check)
        assert (result.status, result.result) == ("completed", "done")
        assert len(llm.calls) == 1 and len(check.commands) == 1
        [notice] = _notices(events, "brief_check")
        assert notice["message"] == "The task's check passed."

    @pytest.mark.asyncio
    async def test_a_check_that_keeps_failing_ends_completed_after_two_retries(
        self,
    ) -> None:
        check = Checks(2)
        replies = [_answer(f"try {n}") for n in range(MAX_CHECK_RETRIES + 1)]
        result, llm, events = await _run(replies, check)
        assert (result.status, result.result) == ("completed", "try 2")
        assert len(llm.calls) == 3 and len(check.commands) == 3
        [failed] = _notices(events, "brief_check_failed")
        assert failed["detail"] == {"retries": 2}
        assert [n["detail"]["round"] for n in _notices(events, "brief_check")] == [
            1,
            2,
            3,
        ]

    @pytest.mark.asyncio
    async def test_every_check_line_runs_and_the_first_failure_stops_the_gate(
        self,
    ) -> None:
        check = Checks(0, 1, 0, 0, 0)
        query = "echo it\nCheck: one\nCheck: two\nCheck: three\n"
        result, llm, _events = await _run(
            [_answer("done"), _answer("fixed")], check, query
        )
        assert result.status == "completed"
        # Round 1 stops at "two"; round 2 runs all three.
        assert check.commands == ["one", "two", "one", "two", "three"]
        assert "Command: two\nExit code: 1" in _last_user(llm, 1)

    @pytest.mark.asyncio
    async def test_a_refused_check_is_reported_and_not_retried(self) -> None:
        check = Checks(None, refused="Blocked: shell_exec is not available")
        result, llm, events = await _run([_answer("done")], check)
        assert (result.status, len(llm.calls)) == ("completed", 1)
        [notice] = _notices(events, "brief_check")
        assert notice["message"] == "The task's check was refused and did not run."
        assert notice["detail"]["exit_code"] is None
        assert notice["detail"]["refused"] == "Blocked: shell_exec is not available"
        [failed] = _notices(events, "brief_check_failed")
        assert failed["detail"] == {"retries": 0}

    @pytest.mark.asyncio
    async def test_a_check_with_no_exit_status_is_told_as_none(self) -> None:
        check = Checks(None, 0)
        _result_, llm, _events = await _run([_answer("done"), _answer("ok")], check)
        assert "Exit code: none" in _last_user(llm, 1)


class TestWithoutTheGate:
    @pytest.mark.asyncio
    async def test_a_brief_without_a_check_line_is_as_before(self) -> None:
        check = Checks(1)
        result, llm, events = await _run(
            [_answer("done")], check, "echo the answer, no check"
        )
        assert (result.status, len(llm.calls), check.commands) == ("completed", 1, [])
        assert not _notices(events, "brief_check")

    @pytest.mark.asyncio
    async def test_a_run_without_the_option_sends_identical_requests(self) -> None:
        passing = Checks(0)
        with_gate = await _run([_answer("done")], passing)
        without = await _run([_answer("done")], None)
        assert [_bodies(llm) for llm in (with_gate[1], without[1])] == [
            _bodies(without[1])
        ] * 2
        assert (with_gate[0].status, without[0].status) == ("completed", "completed")
        assert not _notices(without[2], "brief_check")


class TestAtTheBoundary:
    @pytest.mark.asyncio
    async def test_no_step_left_ends_the_run_without_a_retry(self) -> None:
        check = Checks(1, 0)
        result, llm, events = await _run(
            [_answer("done")], check, attempt_policy="none", max_steps=1
        )
        assert (result.status, len(llm.calls)) == ("completed", 1)
        assert len(_notices(events, "brief_check_failed")) == 1

    @pytest.mark.asyncio
    async def test_a_done_reflection_with_a_failing_check_starts_another_attempt(
        self,
    ) -> None:
        check = Checks(1, 0)
        result, llm, events = await _run(
            [
                _echo_call("a"),
                _reflection("done", best_effort_result="all good"),
                _answer("fixed"),
            ],
            check,
            steps_per_attempt=1,
            max_attempts=3,
        )
        assert (result.status, result.result) == ("completed", "fixed")
        assert (result.attempts, check.commands) == (2, ["pytest -q", "pytest -q"])
        [reflection] = [e.data for e in events if e.type == "reflection"]
        assert reflection["continue_run"] is True
        assert reflection["stop_reason"] is None
        assert _last_user(llm, 2).endswith(CHECK_FAILED_LINE)

    @pytest.mark.asyncio
    async def test_a_done_reflection_in_the_last_attempt_ends_the_run(self) -> None:
        check = Checks(1)
        result, llm, events = await _run(
            [_echo_call("a"), _reflection("done", best_effort_result="all good")],
            check,
            steps_per_attempt=1,
            max_attempts=1,
        )
        assert (result.status, result.result) == ("completed", "all good")
        assert len(_notices(events, "brief_check_failed")) == 1
        [reflection] = [e.data for e in events if e.type == "reflection"]
        assert reflection["continue_run"] is False


def _moving(llm: FakeLLM) -> Any:
    """A probe whose digest changes with every planning request."""
    return lambda workspace: f"after {len(llm.calls)} requests"


async def _deadline_run(
    replies: list[Any], check: Any, *, moving: bool = True, **options: Any
) -> tuple[Any, FakeLLM, list[AgentEvent]]:
    events: list[AgentEvent] = []
    llm = FakeLLM(replies)
    probe = _moving(llm) if moving else (lambda workspace: "same")
    agent = _agent(llm, brief_check=check, brief_probe=probe, **options)
    result = await agent.run_task(QUERY, on_event=events.append)
    return result, llm, events


class TestTheDeadlineCheck:
    @pytest.mark.asyncio
    async def test_a_failing_check_two_steps_from_the_end_can_be_fixed(self) -> None:
        check = Checks(1, 0)
        result, llm, events = await _deadline_run(
            [_echo_call("a"), _echo_call("b"), _answer("fixed"), _answer("unused")],
            check,
            attempt_policy="none",
            max_steps=4,
        )
        assert (result.status, result.result) == ("completed", "fixed")
        assert len(llm.calls) == 3
        # The failure message is the user message of the planning request
        # that has two steps left, not the answer's.
        assert _last_user(llm, 2).endswith(CHECK_FAILED_LINE)
        assert "Exit code: 1" in _last_user(llm, 2)
        assert not any(CHECK_FAILED_LINE in c for c in _flat(llm, 1))
        assert check.commands == ["pytest -q", "pytest -q"]  # the gate runs again
        assert [n["detail"]["exit_code"] for n in _notices(events, "brief_check")] == [
            1,
            0,
        ]
        assert not _notices(events, "brief_check_failed")

    @pytest.mark.asyncio
    async def test_a_pass_changes_nothing(self) -> None:
        check = Checks(0)
        result, llm, _events = await _deadline_run(
            [_echo_call("a"), _echo_call("b"), _answer("done")],
            check,
            attempt_policy="none",
            max_steps=4,
        )
        assert (result.status, len(llm.calls)) == ("completed", 3)
        assert not any(
            CHECK_FAILED_LINE in c for call in range(3) for c in _flat(llm, call)
        )
        assert len(check.commands) == 2  # the deadline check, then the gate

    @pytest.mark.asyncio
    async def test_a_failure_uses_one_retry(self) -> None:
        check = Checks(1)
        result, llm, events = await _deadline_run(
            [
                _echo_call("a"),
                _echo_call("b"),
                _echo_call("c"),
                _answer("x"),
                _answer("y"),
            ],
            check,
            attempt_policy="none",
            max_steps=5,
        )
        # Deadline failure (retry 1), the gate's failure (retry 2), then it gives up.
        assert (result.status, len(llm.calls)) == ("completed", 5)
        [failed] = _notices(events, "brief_check_failed")
        assert failed["detail"] == {"retries": MAX_CHECK_RETRIES}

    @pytest.mark.asyncio
    async def test_nothing_changed_since_the_last_check_means_no_check(self) -> None:
        check = Checks(0)
        result, llm, _events = await _deadline_run(
            [_echo_call("a"), _echo_call("b"), _answer("done")],
            check,
            moving=False,
            attempt_policy="none",
            max_steps=4,
        )
        assert (result.status, len(llm.calls)) == ("completed", 3)
        assert check.commands == ["pytest -q"]  # the gate alone

    @pytest.mark.asyncio
    async def test_it_runs_once_per_attempt(self) -> None:
        check = Checks(0)
        await _deadline_run(
            [_echo_call("a"), _echo_call("b"), _echo_call("c"), _answer("done")],
            check,
            attempt_policy="none",
            max_steps=5,
        )
        assert len(check.commands) == 2

    @pytest.mark.asyncio
    async def test_an_attempt_that_is_not_the_last_has_none(self) -> None:
        check = Checks(0)
        replies = [_echo_call("a"), _echo_call("b"), _answer("done")]
        await _deadline_run(replies, check, steps_per_attempt=3, max_attempts=2)
        assert check.commands == ["pytest -q"]

    @pytest.mark.asyncio
    async def test_the_last_attempt_of_several_has_one(self) -> None:
        check = Checks(1, 0)
        replies = [
            _echo_call("a"),
            _echo_call("b"),
            _echo_call("c"),
            _reflection("continue"),
            _echo_call("d"),
            _echo_call("e"),
            _answer("fixed"),
        ]
        result, llm, _events = await _deadline_run(
            replies, check, steps_per_attempt=3, max_attempts=2
        )
        assert result.status == "completed"
        assert check.commands[:1] == ["pytest -q"]
        assert any(CHECK_FAILED_LINE in c for c in _flat(llm, len(llm.calls) - 1))

    @pytest.mark.asyncio
    async def test_a_brief_without_check_lines_runs_none(self) -> None:
        check = Checks(1)
        events: list[AgentEvent] = []
        llm = FakeLLM([_echo_call("a"), _echo_call("b"), _answer("done")])
        agent = _agent(
            llm,
            brief_check=check,
            brief_probe=_moving(llm),
            attempt_policy="none",
            max_steps=4,
        )
        result = await agent.run_task("echo it", on_event=events.append)
        assert (result.status, check.commands) == ("completed", [])

    @pytest.mark.asyncio
    async def test_a_run_without_brief_probe_runs_none(self) -> None:
        check = Checks(1, 0)
        result, llm, _events = await _run(
            [_echo_call("a"), _echo_call("b"), _answer("done"), _answer("fixed")],
            check,
            attempt_policy="none",
            max_steps=4,
        )
        assert result.status == "completed"
        assert len(check.commands) == 2 and len(llm.calls) == 4


def _flat(llm: FakeLLM, request: int) -> list[str]:
    return [m.content for m in llm.calls[request]["messages"]]


class TestTheOutputTail:
    @pytest.mark.asyncio
    async def test_the_notice_carries_the_last_twenty_lines(self) -> None:
        long = "\n".join(f"row {n}" for n in range(100))
        check = _Output(long)
        _result_, _llm, events = await _run([_answer("done"), _answer("ok")], check)
        [notice, _] = _notices(events, "brief_check")
        tail = notice["detail"]["output_tail"]
        assert tail.splitlines() == [f"row {n}" for n in range(80, 100)]

    def test_the_tail_is_bounded_in_characters(self) -> None:
        wide = BriefCheckResult("c", 1, 0.1, "\n".join("x" * 1000 for _ in range(10)))
        assert len(wide.output_tail) == 4000
        assert BriefCheckResult("c", 1, 0.1).output_tail == ""


class _Output:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = 0

    async def __call__(self, executor: Executor, command: str) -> BriefCheckResult:
        self.calls += 1
        return BriefCheckResult(command, 1 if self.calls == 1 else 0, 0.1, self.output)
