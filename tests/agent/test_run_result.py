"""RunResult and Agent.run_task — no real network or model."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.executor import builtin_names
from localmelo.melo.memory.history import History
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import (
    MAX_AGENT_STEPS,
    MAX_ATTEMPTS,
    STEPS_PER_ATTEMPT,
    Message,
    RunResult,
    TaskRecord,
    ToolCall,
    ToolDef,
)

# The fields in the order of the Phase 1a spec (B4). The CLI's JSON output
# prints RunResult.to_dict() and relies on this order.
SPEC_FIELDS = [
    "task_id",
    "status",
    "result",
    "error",
    "steps",
    "attempts",
    "n_llm",
    "n_main",
    "n_aux",
    "n_tool",
    "llm_errors",
    "usage",
    "duration_s",
    "config",
]


class TestRunResult:
    def test_fields_are_in_spec_order(self) -> None:
        assert [f.name for f in dataclasses.fields(RunResult)] == SPEC_FIELDS

    def test_defaults(self) -> None:
        assert RunResult(task_id="t1", status="completed").to_dict() == {
            "task_id": "t1",
            "status": "completed",
            "result": "",
            "error": "",
            "steps": 0,
            "attempts": 0,
            "n_llm": 0,
            "n_main": 0,
            "n_aux": 0,
            "n_tool": 0,
            "llm_errors": 0,
            "usage": {},
            "duration_s": 0.0,
            "config": {},
        }

    def test_to_dict_keeps_field_order_and_round_trips_through_json(self) -> None:
        result = RunResult(
            task_id="t2",
            status="budget_exhausted",
            error="Max steps reached",
            steps=3,
            attempts=2,
            n_llm=4,
            n_main=3,
            n_aux=1,
            n_tool=3,
            usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            duration_s=1.25,
            config={
                "max_steps": 3,
                "builtins": ["file_read"],
                "memory_dir": None,
                "timeout_s": None,
            },
        )
        data = result.to_dict()
        assert list(data) == SPEC_FIELDS
        assert json.loads(json.dumps(data)) == data

    def test_to_dict_is_a_deep_copy(self) -> None:
        result = RunResult(
            task_id="t3",
            status="completed",
            usage={"total_tokens": 1},
            config={"tools": ["echo"]},
        )
        data = result.to_dict()
        data["usage"]["total_tokens"] = 99
        data["config"]["tools"].append("other")
        assert result.usage == {"total_tokens": 1}
        assert result.config == {"tools": ["echo"]}

    def test_usage_and_config_are_not_shared_between_results(self) -> None:
        first = RunResult(task_id="a", status="completed")
        second = RunResult(task_id="b", status="completed")
        first.usage["total_tokens"] = 1
        first.config["memory"] = True
        assert second.usage == {}
        assert second.config == {}


# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Pops queued replies and records every request.

    A queued exception is raised instead of returned.
    """

    def __init__(self, replies: list[Message | Exception] | None = None) -> None:
        self.replies: deque[Message | Exception] = deque(replies or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            return Message(role="assistant", content="(no response queued)")
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def close(self) -> None:
        pass


class BlockingLLM(FakeLLM):
    """Returns its queued replies, then waits in chat() until cancelled."""

    def __init__(self, replies: list[Message | Exception] | None = None) -> None:
        super().__init__(replies)
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if self.replies:
            return await super().chat(messages, tools)
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")


class FakeEmbedding(BaseEmbeddingProvider):
    """Embeds every text alike; fails for texts that start with *fail_prefix*.

    The failure raises *error* (by default a ConnectionError), or with
    *hang* the request never answers.
    """

    def __init__(
        self,
        fail_prefix: str | None = None,
        *,
        error: Exception | None = None,
        hang: bool = False,
    ) -> None:
        self.calls = 0
        self.fail_prefix = fail_prefix
        self.error = (
            error if error is not None else ConnectionError("embedding server down")
        )
        self.hang = hang

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_prefix and any(t.startswith(self.fail_prefix) for t in texts):
            if self.hang:
                await asyncio.Event().wait()
            raise self.error
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


class FailingHistory(History):
    """In-memory history whose save_task fails from the *fail_from*-th call on."""

    def __init__(self, fail_from: int) -> None:
        super().__init__()
        self.fail_from = fail_from
        self.saves = 0

    async def save_task(self, task: TaskRecord) -> None:
        self.saves += 1
        if self.saves >= self.fail_from:
            raise OSError("disk full")
        await super().save_task(task)


# ── Helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


async def _echo(text: str) -> str:
    return text


def _options(**kwargs: Any) -> AgentOptions:
    """Options without built-ins, so the test's echo tool is the only tool."""
    return AgentOptions(builtins=frozenset(), **kwargs)


def _agent(
    llm: BaseLLMProvider,
    options: AgentOptions | None = None,
    embedding: BaseEmbeddingProvider | None = None,
) -> Agent:
    agent = Agent(llm=llm, embedding=embedding, options=options or _options())
    agent.hippo.register_tool(
        ToolDef(
            name="echo",
            description="echo text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    )
    agent.executor.register("echo", _echo)
    return agent


def _echo_call(text: str, usage: dict[str, int] | None = None) -> Message:
    return Message(
        role="assistant",
        content=f"echo {text}",
        tool_call=ToolCall(tool_name="echo", arguments={"text": text}),
        usage=usage,
    )


def _answer(text: str, usage: dict[str, int] | None = None) -> Message:
    return Message(role="assistant", content=text, usage=usage)


def _reflection(
    action: str, best_effort: str = "", usage: dict[str, int] | None = None
) -> Message:
    """A reflection reply; with ``continue`` it passes the continuation gate."""
    data = {
        "recommended_action": action,
        "rationale": "test",
        "best_effort_result": best_effort,
        "summary": "tried echo",
        "task_still_feasible": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "estimated_info_gain": 0.5,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.8,
        "feasibility": 0.8,
    }
    return Message(role="assistant", content=json.dumps(data), usage=usage)


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _only_task(agent: Agent) -> TaskRecord:
    tasks = list(agent.hippo.history._tasks.values())
    assert len(tasks) == 1
    return tasks[0]


def _promoted_reflections(agent: Agent) -> list[str]:
    return [
        entry.text
        for entry in agent.hippo.long._entries
        if entry.metadata.get("type") == "reflection"
    ]


# ── Statuses ────────────────────────────────────────────────────────────────


class TestStatuses:
    @pytest.mark.asyncio
    async def test_completed(self) -> None:
        agent = _agent(FakeLLM([_answer("42")]))
        result = await agent.run_task("q")
        assert (result.status, result.result, result.error) == ("completed", "42", "")
        assert (result.steps, result.attempts) == (1, 1)
        task = _only_task(agent)
        assert (task.task_id, task.status, task.result) == (
            result.task_id,
            "completed",
            "42",
        )

    @pytest.mark.asyncio
    async def test_failed_when_the_pre_plan_check_refuses(self) -> None:
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm)
        agent.hippo.working.append(Message(role="system", content="x" * 110_000))
        result = await agent.run_task("q")
        assert (result.status, result.result) == ("failed", "")
        assert result.error.startswith("Plan check failed: ")
        # The check refused the step before its planning call was made.
        assert llm.calls == []
        assert (result.steps, result.attempts, result.n_llm) == (0, 1, 0)
        assert _only_task(agent).result == result.error

    @pytest.mark.asyncio
    async def test_failed_when_the_response_check_refuses(self) -> None:
        bad = Message(role="assistant", tool_call=ToolCall(tool_name=""))
        result = await _agent(FakeLLM([bad])).run_task("q")
        assert (result.status, result.result) == ("failed", "")
        assert result.error.startswith("Response check failed: ")
        assert (result.steps, result.n_llm, result.n_tool) == (1, 1, 0)

    @pytest.mark.asyncio
    async def test_failed_when_the_reflection_gate_stops(self) -> None:
        llm = FakeLLM([_echo_call("a"), _reflection("stop", best_effort="partial")])
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2))
        result = await agent.run_task("echo q")
        assert (result.status, result.result) == ("failed", "partial")
        assert result.error == "Stopped after attempt 0: test"
        assert _only_task(agent).result == "partial"

    @pytest.mark.asyncio
    async def test_gate_stop_without_a_best_effort_reports_the_error(self) -> None:
        llm = FakeLLM([_echo_call("a"), _reflection("stop")])
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2))
        assert await agent.run("echo q") == "[failed] Stopped after attempt 0: test"
        assert _only_task(agent).result == "Stopped after attempt 0: test"

    @pytest.mark.asyncio
    async def test_budget_exhausted_by_max_steps(self) -> None:
        llm = FakeLLM([_echo_call(str(i)) for i in range(5)])
        agent = _agent(llm, _options(attempt_policy="none", max_steps=3))
        result = await agent.run_task("echo q")
        assert (result.status, result.result, result.error) == (
            "budget_exhausted",
            "",
            "Max steps reached",
        )
        assert (result.steps, result.attempts, result.n_tool) == (3, 1, 3)
        assert _only_task(agent).result == "Max steps reached"

    @pytest.mark.asyncio
    async def test_budget_exhausted_by_max_attempts(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a"),
                _reflection("continue"),
                _echo_call("b"),
                _reflection("continue"),
            ]
        )
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2))
        result = await agent.run_task("echo q")
        assert (result.status, result.result, result.error) == (
            "budget_exhausted",
            "",
            "Max attempts reached",
        )
        assert (result.steps, result.attempts) == (2, 2)

    @pytest.mark.asyncio
    async def test_an_attempt_without_steps_left_does_not_start(self) -> None:
        llm = FakeLLM([_echo_call("a"), _echo_call("b"), _reflection("continue")])
        options = _options(max_steps=2, steps_per_attempt=2, max_attempts=3)
        result = await _agent(llm, options).run_task("echo q")
        assert (result.status, result.error) == (
            "budget_exhausted",
            "Max steps reached",
        )
        assert (result.steps, result.attempts) == (2, 1)

    @pytest.mark.asyncio
    async def test_stuck(self) -> None:
        llm = FakeLLM([_echo_call("same") for _ in range(5)])
        agent = _agent(llm, _options(attempt_policy="none", max_steps=10))
        result = await agent.run_task("echo q")
        assert (result.status, result.result) == ("stuck", "")
        assert result.error == "Stuck: the same tool calls or errors repeated"
        assert (result.steps, result.attempts) == (3, 1)

    @pytest.mark.asyncio
    async def test_crashed_when_the_llm_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _agent(FakeLLM([RuntimeError("boom")]))
        with caplog.at_level(logging.ERROR, logger="localmelo.melo.agent.agent"):
            result = await agent.run_task("q")
        assert (result.status, result.result, result.error) == (
            "crashed",
            "",
            "RuntimeError: boom",
        )
        # The planning call was made and raised.
        assert (result.steps, result.n_llm, result.llm_errors) == (1, 0, 1)
        task = _only_task(agent)
        assert (task.status, task.result) == ("crashed", "RuntimeError: boom")
        [record] = [r for r in caplog.records if "crashed" in r.getMessage()]
        assert record.exc_info is not None

    @pytest.mark.asyncio
    async def test_run_shows_the_status_and_error_of_a_crashed_run(self) -> None:
        agent = _agent(FakeLLM([RuntimeError("boom")]))
        assert await agent.run("q") == "[crashed] RuntimeError: boom"

    @pytest.mark.asyncio
    async def test_crash_without_a_message_names_the_exception_type(self) -> None:
        result = await _agent(FakeLLM([RuntimeError()])).run_task("q")
        assert (result.status, result.error) == ("crashed", "RuntimeError")


# ── The finally block ───────────────────────────────────────────────────────


class TestFinally:
    @pytest.mark.asyncio
    async def test_cancelled_run_propagates_and_is_saved_as_crashed(self) -> None:
        emb = FakeEmbedding()
        llm = BlockingLLM([_echo_call("a"), _reflection("continue")])
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2), emb)
        running = asyncio.create_task(agent.run_task("echo q"))
        await llm.blocked.wait()
        assert agent.hippo.working.get_reflections()  # attempt 1 is planning

        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        task = _only_task(agent)
        assert (task.status, task.result) == ("crashed", "CancelledError")
        assert agent.hippo.working.get_reflections() == []
        assert _promoted_reflections(agent) == ["Attempt 0: tried echo"]

    @pytest.mark.asyncio
    async def test_save_failure_is_logged_not_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _agent(FakeLLM([_answer("ok")]))
        agent.hippo.history = FailingHistory(fail_from=2)
        result = await agent.run_task("q")
        assert (result.status, result.result) == ("completed", "ok")
        assert "Could not save the final state of task" in caplog.text

    @pytest.mark.asyncio
    async def test_save_failure_does_not_replace_the_crash_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _agent(FakeLLM([RuntimeError("boom")]))
        agent.hippo.history = FailingHistory(fail_from=2)
        result = await agent.run_task("q")
        assert (result.status, result.error) == ("crashed", "RuntimeError: boom")
        assert "Could not save the final state of task" in caplog.text

    @pytest.mark.asyncio
    async def test_failing_first_save_crashes_the_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm)
        agent.hippo.history = FailingHistory(fail_from=1)
        result = await agent.run_task("q")
        assert (result.status, result.error) == ("crashed", "OSError: disk full")
        assert llm.calls == []
        assert (result.steps, result.attempts) == (0, 0)
        assert "Could not save the final state of task" in caplog.text

    @pytest.mark.asyncio
    async def test_crashed_run_is_saved_to_sqlite(self, tmp_path: Path) -> None:
        options = _options(memory_dir=str(tmp_path))
        agent = _agent(FakeLLM([ConnectionError("refused")]), options)
        result = await agent.run_task("q")
        await agent.close()

        history = SqliteHistory(tmp_path / "history.db")
        try:
            task = await history.get_task(result.task_id)
        finally:
            await history.aclose()
        assert task is not None
        assert (task.status, task.result) == ("crashed", "ConnectionError: refused")

    @pytest.mark.asyncio
    async def test_reflections_are_promoted_and_cleared_after_a_crash(self) -> None:
        emb = FakeEmbedding()
        llm = FakeLLM([_echo_call("a"), _reflection("continue"), RuntimeError("boom")])
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2), emb)
        result = await agent.run_task("echo q")
        assert result.status == "crashed"
        assert agent.hippo.working.get_reflections() == []
        assert _promoted_reflections(agent) == ["Attempt 0: tried echo"]

    @pytest.mark.asyncio
    async def test_memory_off_clears_reflections_without_promoting(self) -> None:
        emb = FakeEmbedding()
        llm = FakeLLM([_echo_call("a"), _reflection("continue"), RuntimeError("boom")])
        options = _options(steps_per_attempt=1, max_attempts=2, memory=False)
        agent = _agent(llm, options, emb)
        result = await agent.run_task("echo q")
        assert result.status == "crashed"
        assert agent.hippo.working.get_reflections() == []
        assert agent.hippo.long._entries == []
        assert emb.calls == 0

    @pytest.mark.asyncio
    async def test_promotion_failure_is_logged_and_reflections_cleared(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Promoted reflections read "Attempt <n>: ..."; nothing else does.
        emb = FakeEmbedding(fail_prefix="Attempt ")
        llm = FakeLLM([_echo_call("a"), _reflection("continue"), _answer("done")])
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2), emb)
        result = await agent.run_task("echo q")
        assert (result.status, result.result) == ("completed", "done")
        assert "Could not promote the reflections of task" in caplog.text
        assert agent.hippo.working.get_reflections() == []
        assert _only_task(agent).status == "completed"

    @pytest.mark.asyncio
    async def test_a_provider_error_during_promotion_logs_no_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        down = ProviderError("POST /embeddings failed: ConnectError", retryable=True)
        emb = FakeEmbedding(fail_prefix="Attempt ", error=down)
        refused = ProviderError(
            "POST /chat/completions failed with HTTP 400: no",
            status_code=400,
            retryable=False,
        )
        llm = FakeLLM([_echo_call("a"), _reflection("continue"), refused])
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2), emb)
        with caplog.at_level(logging.DEBUG):
            result = await agent.run_task("echo q")
        assert (result.status, result.error) == ("provider_error", str(refused))
        [record] = [r for r in caplog.records if "Could not promote" in r.getMessage()]
        assert record.levelno == logging.INFO
        assert record.getMessage().endswith(f": {down}")
        # A failed request, not a crash: nothing logged with a traceback.
        assert [r for r in caplog.records if r.exc_info] == []
        assert agent.hippo.working.get_reflections() == []
        assert _only_task(agent).status == "provider_error"

    @pytest.mark.asyncio
    async def test_a_promotion_without_an_answer_is_cut_short(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("localmelo.melo.agent.agent.PROMOTION_TIMEOUT_S", 0.05)
        emb = FakeEmbedding(fail_prefix="Attempt ", hang=True)
        # Attempt 0 reflects; attempt 1 waits for the model until timeout_s.
        llm = BlockingLLM([_echo_call("a"), _reflection("continue")])
        options = _options(steps_per_attempt=1, max_attempts=2, timeout_s=0.2)
        agent = _agent(llm, options, emb)
        with caplog.at_level(logging.DEBUG):
            result = await asyncio.wait_for(agent.run_task("echo q"), 5)
        assert (result.status, result.error) == ("timeout", "Timed out after 0.2 s")
        # timeout_s, then at most the promotion's bound.
        assert result.duration_s < 2.0
        [record] = [r for r in caplog.records if "Could not promote" in r.getMessage()]
        assert record.levelno == logging.INFO
        assert record.getMessage().endswith(": no answer within 0.05 s")
        assert [r for r in caplog.records if r.exc_info] == []
        assert agent.hippo.working.get_reflections() == []
        assert _only_task(agent).status == "timeout"


# ── Counters ────────────────────────────────────────────────────────────────


class TestCounters:
    @pytest.mark.asyncio
    async def test_counts_of_a_run_with_a_reflection(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a", usage=_usage(10, 2)),
                _reflection("continue", usage=_usage(5, 3)),
                _answer("done", usage=_usage(20, 1)),
            ]
        )
        agent = _agent(llm, _options(steps_per_attempt=1, max_attempts=2))
        result = await agent.run_task("echo q")
        assert result.status == "completed"
        assert (result.steps, result.attempts, result.n_tool) == (2, 2, 1)
        # Two planning calls offered the echo tool; the reflection offered none.
        assert (result.n_llm, result.n_main, result.n_aux) == (3, 2, 1)
        assert result.llm_errors == 0
        assert result.usage == _usage(35, 6)
        assert 0.0 <= result.duration_s < 60.0

    @pytest.mark.asyncio
    async def test_a_planning_call_without_tools_counts_as_aux(self) -> None:
        llm = FakeLLM([_answer("hi")])
        agent = Agent(llm=llm, options=_options())  # no tool registered at all
        result = await agent.run_task("q")
        assert llm.calls[0]["tools"] is None
        assert (result.steps, result.n_llm, result.n_main, result.n_aux) == (
            1,
            1,
            0,
            1,
        )

    @pytest.mark.asyncio
    async def test_n_tool_counts_every_call_the_executor_handled(self) -> None:
        both = Message(
            role="assistant",
            tool_calls=[
                ToolCall(tool_name="echo", arguments={"text": "a"}),
                ToolCall(tool_name="no_such_tool", arguments={}),
            ],
        )
        agent = _agent(FakeLLM([both, _answer("done")]))
        result = await agent.run_task("echo q")
        assert (result.status, result.steps, result.n_tool) == ("completed", 2, 2)

    @pytest.mark.asyncio
    async def test_usage_sums_integer_fields_only(self) -> None:
        odd: dict[str, Any] = {"total_tokens": 3, "cached": True, "ratio": 0.5}
        llm = FakeLLM(
            [
                _echo_call("a", usage=odd),
                _answer("done", usage=None),
            ]
        )
        result = await _agent(llm).run_task("echo q")
        assert result.usage == {"total_tokens": 3}

    @pytest.mark.asyncio
    async def test_counters_cover_one_run(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a", usage=_usage(10, 2)),
                _answer("first", usage=_usage(20, 1)),
                _answer("second", usage=_usage(7, 1)),
            ]
        )
        agent = _agent(llm)
        await agent.run_task("echo q")
        result = await agent.run_task("echo again")
        assert result.result == "second"
        assert (result.steps, result.attempts, result.n_tool) == (1, 1, 0)
        assert (result.n_llm, result.llm_errors) == (1, 0)
        assert result.usage == _usage(7, 1)


# ── Configuration ───────────────────────────────────────────────────────────

CONFIG_KEYS = [
    "max_steps",
    "attempt_policy",
    "steps_per_attempt",
    "max_attempts",
    "tool_exposure",
    "builtins",
    "memory",
    "memory_dir",
    "timeout_s",
    "tools",
]


class TestConfig:
    @pytest.mark.asyncio
    async def test_default_configuration(self) -> None:
        agent = Agent(llm=FakeLLM([_answer("ok")]))
        result = await agent.run_task("q")
        assert list(result.config) == CONFIG_KEYS
        assert result.config == {
            "max_steps": MAX_AGENT_STEPS,
            "attempt_policy": "reflect",
            "steps_per_attempt": STEPS_PER_ATTEMPT,
            "max_attempts": MAX_ATTEMPTS,
            "tool_exposure": "retrieve",
            "builtins": sorted(builtin_names()),
            "memory": True,
            "memory_dir": None,
            "timeout_s": None,
            "tools": sorted(builtin_names()),
        }

    @pytest.mark.asyncio
    async def test_effective_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        options = AgentOptions(
            max_steps=4,
            attempt_policy="none",
            steps_per_attempt=2,
            max_attempts=1,
            tool_exposure="all",
            builtins=frozenset({"python_exec", "file_read"}),
            memory=False,
            memory_dir="~/memory",
            timeout_s=2.5,
        )
        agent = _agent(FakeLLM([_answer("ok")]), options)
        result = await agent.run_task("q")
        await agent.close()
        assert result.config == {
            "max_steps": 4,
            "attempt_policy": "none",
            "steps_per_attempt": 2,
            "max_attempts": 1,
            "tool_exposure": "all",
            "builtins": ["file_read", "python_exec"],
            "memory": False,
            # The resolved directory, not the option as given.
            "memory_dir": str(tmp_path / "memory"),
            "timeout_s": 2.5,
            # Built-in and other registered tools alike.
            "tools": ["echo", "file_read", "python_exec"],
        }

    @pytest.mark.asyncio
    async def test_memory_dir_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
        monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(tmp_path / "env"))
        agent = Agent(llm=FakeLLM([_answer("ok")]))
        result = await agent.run_task("q")
        await agent.close()
        assert result.config["memory_dir"] == str(tmp_path / "env")

    @pytest.mark.asyncio
    async def test_run_result_is_json(self) -> None:
        result = await _agent(FakeLLM([_answer("ok")])).run_task("q")
        data = result.to_dict()
        assert list(data) == SPEC_FIELDS
        assert json.loads(json.dumps(data)) == data


# ── Public surface ──────────────────────────────────────────────────────────


def test_exported_from_the_package_root() -> None:
    import localmelo

    assert localmelo.Agent is Agent
    assert localmelo.AgentOptions is AgentOptions
    assert localmelo.RunResult is RunResult
    assert {"Agent", "AgentOptions", "RunResult"} <= set(localmelo.__all__)
