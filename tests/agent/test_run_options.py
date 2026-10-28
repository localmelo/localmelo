"""Agent: run options and tool providers — no real network or model."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import REFLECTION_PROMPT
from localmelo.melo.agent.agent import _resolve_memory_dir
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import builtin_names
from localmelo.melo.executor.models import ErrorCategory, ExecutionRequest
from localmelo.melo.memory.history import History
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import Message, TaskRecord, ToolCall, ToolDef

# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Pops queued responses and records every request."""

    def __init__(self, responses: list[Message] | None = None) -> None:
        self.responses: deque[Message] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return Message(role="assistant", content="(no response queued)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


class FakeEmbedding(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]

    async def close(self) -> None:
        self.closed = True


class FakeToolProvider(BaseToolProvider):
    def __init__(self) -> None:
        self.starts = 0

    async def start(self) -> list[ProvidedTool]:
        self.starts += 1
        return []

    async def close(self) -> None:
        pass


# ── Helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


async def _echo(text: str) -> str:
    return text


def _echo_agent(
    llm: FakeLLM,
    options: AgentOptions,
    embedding: BaseEmbeddingProvider | None = None,
) -> Agent:
    agent = Agent(llm=llm, embedding=embedding, options=options)
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


def _echo_call(text: str) -> Message:
    return Message(
        role="assistant",
        content=f"echo {text}",
        tool_call=ToolCall(tool_name="echo", arguments={"text": text}),
    )


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _reflection(action: str, best_effort: str = "") -> Message:
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
    return Message(role="assistant", content=json.dumps(data))


def _is_reflection(call: dict[str, Any]) -> bool:
    return bool(call["messages"]) and call["messages"][0].content.startswith(
        REFLECTION_PROMPT
    )


def _planning_calls(llm: FakeLLM) -> list[dict[str, Any]]:
    return [c for c in llm.calls if not _is_reflection(c)]


def _reflection_calls(llm: FakeLLM) -> list[dict[str, Any]]:
    return [c for c in llm.calls if _is_reflection(c)]


def _only_task(agent: Agent) -> TaskRecord:
    tasks = list(agent.hippo.history._tasks.values())
    assert len(tasks) == 1
    return tasks[0]


def _tool_names(agent: Agent) -> set[str]:
    return {t.name for t in agent.hippo.list_tools()}


# ── Construction ────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_options(self) -> None:
        agent = Agent(llm=FakeLLM())
        assert agent.options == AgentOptions()
        assert agent._tool_providers == ()

    def test_options_are_kept(self) -> None:
        options = AgentOptions(max_steps=7, attempt_policy="none")
        assert Agent(llm=FakeLLM(), options=options).options is options

    def test_tool_providers_are_stored_not_started(self) -> None:
        first, second = FakeToolProvider(), FakeToolProvider()
        agent = Agent(llm=FakeLLM(), tool_providers=[first, second])
        assert agent._tool_providers == (first, second)
        assert first.starts == 0
        assert second.starts == 0


# ── Built-in selection ──────────────────────────────────────────────────────


class TestBuiltinSelection:
    def test_all_builtins_by_default(self) -> None:
        agent = Agent(llm=FakeLLM())
        assert _tool_names(agent) == builtin_names()
        assert set(agent.executor._callables) == builtin_names()

    def test_direct_mode_keeps_every_builtin_in_p13a(self) -> None:
        agent = Agent(llm=FakeLLM(), options=AgentOptions(mode="direct"))
        assert _tool_names(agent) == builtin_names()

    def test_the_gateway_mode_registers_no_shell(self) -> None:
        agent = Agent(llm=FakeLLM(), options=AgentOptions(mode="gateway"))
        expected = builtin_names() - {"shell_exec"}
        assert _tool_names(agent) == expected
        assert set(agent.executor._callables) == expected
        config = agent._run_config()
        assert config["builtins"] == sorted(expected)
        assert config["tools"] == sorted(expected)

    def test_an_explicit_set_wins_over_the_mode(self) -> None:
        agent = Agent(
            llm=FakeLLM(),
            options=AgentOptions(mode="gateway", builtins=frozenset({"shell_exec"})),
        )
        assert _tool_names(agent) == {"shell_exec"}

    @pytest.mark.parametrize(
        "selected",
        [
            frozenset(),
            frozenset({"file_read"}),
            frozenset({"python_exec", "shell_exec"}),
        ],
    )
    def test_only_selected_builtins_are_registered(
        self, selected: frozenset[str]
    ) -> None:
        agent = Agent(llm=FakeLLM(), options=AgentOptions(builtins=selected))
        assert _tool_names(agent) == selected
        assert set(agent.executor._callables) == selected

    @pytest.mark.asyncio
    async def test_unselected_builtin_cannot_run(self) -> None:
        agent = Agent(llm=FakeLLM(), options=AgentOptions(builtins=frozenset()))
        outcome = await agent.executor.execute_structured(
            ExecutionRequest(tool_name="shell_exec", arguments={"command": "echo hi"})
        )
        assert outcome.error_category == ErrorCategory.TOOL_NOT_FOUND


# ── memory option ───────────────────────────────────────────────────────────


def _continue_then_answer() -> FakeLLM:
    """One tool step, a reflection that continues, then a final answer."""
    return FakeLLM([_echo_call("a"), _reflection("continue"), _answer("answer")])


class TestMemoryOption:
    @pytest.mark.asyncio
    async def test_memory_on_writes_and_promotes(self) -> None:
        """Control for the next test: the same run uses long-term memory."""
        emb = FakeEmbedding()
        options = AgentOptions(steps_per_attempt=1, max_attempts=2)
        agent = _echo_agent(_continue_then_answer(), options, embedding=emb)
        assert await agent.run("echo q") == "answer"
        assert emb.calls > 0
        assert agent.hippo.long._entries

    @pytest.mark.asyncio
    async def test_memory_off_skips_long_term_and_promotion(self) -> None:
        emb = FakeEmbedding()
        options = AgentOptions(steps_per_attempt=1, max_attempts=2, memory=False)
        agent = _echo_agent(_continue_then_answer(), options, embedding=emb)
        assert agent.hippo.embedding is None
        assert agent._embedding is emb

        assert await agent.run("echo q") == "answer"
        # No recall, no long-term writes, no reflection promotion.
        assert emb.calls == 0
        assert agent.hippo.long._entries == []
        # The reflection was still cleared at the end of the run.
        assert agent.hippo.working.get_reflections() == []

        await agent.close()
        assert emb.closed


# ── Persistence ─────────────────────────────────────────────────────────────


class TestPersistence:
    def test_in_memory_without_option_or_env(self) -> None:
        agent = Agent(llm=FakeLLM(), embedding=FakeEmbedding())
        assert agent._memory_dir is None
        assert type(agent.hippo.history) is History
        assert type(agent.hippo.long) is LongTerm

    def test_empty_env_flag_keeps_memory_in_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "")
        assert _resolve_memory_dir(AgentOptions()) is None

    @pytest.mark.asyncio
    async def test_memory_dir_option_enables_sqlite(self, tmp_path: Path) -> None:
        mem = tmp_path / "memory"
        agent = Agent(
            llm=FakeLLM([_answer("ok")]),
            embedding=FakeEmbedding(),
            options=AgentOptions(memory_dir=str(mem)),
        )
        assert agent._memory_dir == str(mem)
        assert isinstance(agent.hippo.history, SqliteHistory)
        assert isinstance(agent.hippo.long, SqliteLongTerm)
        try:
            assert await agent.run("q") == "ok"
        finally:
            await agent.close()
        assert (mem / "history.db").exists()
        assert (mem / "long_term.db").exists()

    @pytest.mark.asyncio
    async def test_memory_dir_option_wins_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
        monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(tmp_path / "from-env"))
        option_dir = tmp_path / "from-option"
        agent = Agent(llm=FakeLLM(), options=AgentOptions(memory_dir=str(option_dir)))
        assert agent._memory_dir == str(option_dir)
        assert option_dir.is_dir()
        assert not (tmp_path / "from-env").exists()
        await agent.close()

    @pytest.mark.asyncio
    async def test_memory_dir_option_expands_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        agent = Agent(llm=FakeLLM(), options=AgentOptions(memory_dir="~/memory"))
        assert agent._memory_dir == str(tmp_path / "memory")
        assert (tmp_path / "memory").is_dir()
        await agent.close()

    @pytest.mark.asyncio
    async def test_env_opt_in_uses_env_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
        monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(tmp_path / "env"))
        agent = Agent(llm=FakeLLM())
        assert agent._memory_dir == str(tmp_path / "env")
        assert isinstance(agent.hippo.history, SqliteHistory)
        # No embedding provider: long-term memory stays in process.
        assert type(agent.hippo.long) is LongTerm
        await agent.close()

    def test_env_dir_is_used_as_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
        monkeypatch.setenv("LOCALMELO_MEMORY_DIR", "~/as-given")
        assert _resolve_memory_dir(AgentOptions()) == "~/as-given"

    def test_env_opt_in_defaults_to_the_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
        assert _resolve_memory_dir(AgentOptions()) == str(
            tmp_path / ".cache" / "localmelo" / "memory"
        )

    @pytest.mark.asyncio
    async def test_memory_off_persists_history_only(self, tmp_path: Path) -> None:
        mem = tmp_path / "memory"
        agent = Agent(
            llm=FakeLLM([_answer("ok")]),
            embedding=FakeEmbedding(),
            options=AgentOptions(memory=False, memory_dir=str(mem)),
        )
        assert isinstance(agent.hippo.history, SqliteHistory)
        assert type(agent.hippo.long) is LongTerm
        try:
            assert await agent.run("q") == "ok"
        finally:
            await agent.close()
        assert (mem / "history.db").exists()
        assert not (mem / "long_term.db").exists()


# ── attempt_policy="none" ───────────────────────────────────────────────────


class TestNoReflectionPolicy:
    @pytest.mark.asyncio
    async def test_direct_answer_completes(self) -> None:
        llm = FakeLLM([_answer("42")])
        agent = _echo_agent(llm, AgentOptions(attempt_policy="none"))
        assert await agent.run("q") == "42"
        task = _only_task(agent)
        assert task.status == "completed"
        assert task.attempts_completed == 0

    @pytest.mark.asyncio
    async def test_budget_is_max_steps_and_nothing_reflects(self) -> None:
        llm = FakeLLM([_echo_call(str(i)) for i in range(6)])
        # steps_per_attempt and max_attempts only apply to "reflect".
        options = AgentOptions(
            attempt_policy="none", max_steps=4, steps_per_attempt=1, max_attempts=1
        )
        agent = _echo_agent(llm, options)
        assert await agent.run("echo q") == "[budget_exhausted] Max steps reached"
        assert len(llm.calls) == 4
        assert _reflection_calls(llm) == []
        task = _only_task(agent)
        assert task.status == "budget_exhausted"
        assert task.result == "Max steps reached"
        assert task.attempts_completed == 1

    @pytest.mark.asyncio
    async def test_stuck_fails_without_reflection(self) -> None:
        llm = FakeLLM([_echo_call("same") for _ in range(6)])
        agent = _echo_agent(llm, AgentOptions(attempt_policy="none", max_steps=10))
        result = await agent.run("echo q")
        assert result == "[stuck] Stuck: the same tool calls or errors repeated"
        assert len(llm.calls) == 3
        assert _reflection_calls(llm) == []
        assert _only_task(agent).status == "stuck"


# ── attempt_policy="reflect" ────────────────────────────────────────────────


class TestReflectPolicyBudgets:
    @pytest.mark.asyncio
    async def test_max_steps_caps_planning_calls_across_attempts(self) -> None:
        llm = FakeLLM(
            [
                _echo_call("a"),
                _echo_call("b"),
                _reflection("continue"),
                _echo_call("c"),
                _reflection("continue"),
                _echo_call("d"),
                _echo_call("e"),
            ]
        )
        options = AgentOptions(max_steps=3, steps_per_attempt=2, max_attempts=5)
        agent = _echo_agent(llm, options)
        assert await agent.run("echo q") == "[budget_exhausted] Max steps reached"
        # Attempt 0 uses 2 steps, attempt 1 the 1 step left; no attempt 2.
        assert len(_planning_calls(llm)) == 3
        assert len(_reflection_calls(llm)) == 2
        assert _only_task(agent).status == "budget_exhausted"
