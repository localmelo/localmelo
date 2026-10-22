"""Agent.begin_exit: a run cut short because the process exits.

No real network, model or process: the tool provider records what
happens to it, and the history records each save.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any

import pytest

import localmelo.melo.agent.agent as agent_mod
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.memory.history import History
from localmelo.melo.schema import Message, TaskRecord, ToolCall, ToolDef

# ── Fakes ───────────────────────────────────────────────────────────────────


class BlockingLLM(BaseLLMProvider):
    """Returns its queued replies, then waits in chat() until cancelled."""

    def __init__(self, replies: list[Message] | None = None) -> None:
        self.replies: deque[Message] = deque(replies or [])
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if self.replies:
            return self.replies.popleft()
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def close(self) -> None:
        pass


class ExitProvider(BaseToolProvider):
    """Records each start, close and kill in *events*.

    With a *start_gate*, start() waits for it; with a *close_gate*, close()
    does, so a test can make either hang until it sets the gate.
    """

    def __init__(
        self,
        events: list[str],
        *,
        start_gate: asyncio.Event | None = None,
        close_gate: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.start_gate = start_gate
        self.close_gate = close_gate
        self.starting = asyncio.Event()
        self.kills = 0

    async def start(self) -> list[ProvidedTool]:
        self.events.append("start")
        self.starting.set()
        if self.start_gate is not None:
            await self.start_gate.wait()
        return []

    async def close(self) -> None:
        self.events.append("close")
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.events.append("closed")

    def kill(self) -> None:
        self.kills += 1
        self.events.append("kill")


class RecordingHistory(History):
    """In-memory history that records each save_task() in *events*."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def save_task(self, task: TaskRecord) -> None:
        self.events.append(f"save {task.status}")
        await super().save_task(task)


class FakeEmbedding(BaseEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


# ── Helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _agent(
    llm: BaseLLMProvider,
    events: list[str],
    provider: BaseToolProvider | None = None,
    *,
    embedding: BaseEmbeddingProvider | None = None,
    **options: Any,
) -> Agent:
    agent = Agent(
        llm=llm,
        embedding=embedding,
        options=AgentOptions(builtins=frozenset(), tool_exposure="all", **options),
        tool_providers=[provider] if provider is not None else [],
    )
    agent.hippo.history = RecordingHistory(events)
    return agent


def _only_task(agent: Agent) -> TaskRecord:
    tasks = list(agent.hippo.history._tasks.values())
    assert len(tasks) == 1
    return tasks[0]


async def _exit_run(
    agent: Agent, running: asyncio.Task[Any], within_s: float, reason: str
) -> float:
    """begin_exit() and cancel *running*; return the seconds until it ended."""
    started = time.monotonic()
    agent.begin_exit(reason, within_s=within_s)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    return time.monotonic() - started


# ── The run ─────────────────────────────────────────────────────────────────


class TestExitingRun:
    @pytest.mark.asyncio
    async def test_the_providers_are_closed_before_the_task_is_saved(self) -> None:
        events: list[str] = []
        llm = BlockingLLM()
        agent = _agent(llm, events, ExitProvider(events))
        running = asyncio.create_task(agent.run_task("q"))
        await llm.blocked.wait()

        await _exit_run(agent, running, 5.0, "Cancelled by SIGTERM")
        # A leaked provider process outlives this one; a lost row only costs
        # a record. So the providers come first, then the save.
        assert events == ["save running", "start", "close", "closed", "save crashed"]
        task = _only_task(agent)
        assert (task.status, task.result) == ("crashed", "Cancelled by SIGTERM")
        await agent.close()

    @pytest.mark.asyncio
    async def test_the_first_exit_sets_the_reason(self) -> None:
        events: list[str] = []
        llm = BlockingLLM()
        agent = _agent(llm, events)
        running = asyncio.create_task(agent.run_task("q"))
        await llm.blocked.wait()

        agent.begin_exit("Cancelled by SIGHUP", within_s=5.0)
        await _exit_run(agent, running, 0.0, "Cancelled by SIGTERM")
        assert _only_task(agent).result == "Cancelled by SIGHUP"

    @pytest.mark.asyncio
    async def test_a_provider_that_does_not_close_in_time_is_killed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent_mod, "EXIT_SAVE_RESERVE_S", 0.1)
        events: list[str] = []
        hang = asyncio.Event()
        provider = ExitProvider(events, close_gate=hang)
        llm = BlockingLLM()
        agent = _agent(llm, events, provider)
        running = asyncio.create_task(agent.run_task("q"))
        await llm.blocked.wait()

        elapsed = await _exit_run(agent, running, 0.3, "Cancelled by SIGTERM")
        # Closing waited until 0.1 s before the deadline, then killed the
        # provider, so the hanging close() did not prevent the save.
        assert 0.15 <= elapsed < 1.0
        assert events == ["save running", "start", "close", "kill", "save crashed"]
        assert _only_task(agent).status == "crashed"

        # close() keeps to the same deadline, which has passed: it kills
        # again rather than wait.
        await asyncio.wait_for(agent.close(), 1.0)
        assert provider.kills == 2
        hang.set()  # let the abandoned close() calls end
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_an_exit_during_the_provider_setup_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent_mod, "EXIT_SAVE_RESERVE_S", 0.1)
        events: list[str] = []
        never, hang = asyncio.Event(), asyncio.Event()
        provider = ExitProvider(events, start_gate=never, close_gate=hang)
        agent = _agent(BlockingLLM(), events, provider)
        running = asyncio.create_task(agent.run_task("q"))
        await provider.starting.wait()

        elapsed = await _exit_run(agent, running, 0.3, "Cancelled by SIGTERM")
        # The failed setup's clean-up stops waiting at the same limit, and
        # the run's own clean-up has no time left: it kills at once.
        assert elapsed < 1.0
        assert events == [
            "save running",
            "start",
            "close",
            "kill",
            "close",
            "kill",
            "save crashed",
        ]
        task = _only_task(agent)
        assert (task.status, task.result) == ("crashed", "Cancelled by SIGTERM")
        hang.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_the_reflections_are_cleared_without_being_promoted(
        self,
    ) -> None:
        reflection = {
            "recommended_action": "continue",
            "rationale": "test",
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
        # Attempt 0 runs out of its one step and reflects; attempt 1 blocks.
        llm = BlockingLLM(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
                ),
                Message(role="assistant", content=json.dumps(reflection)),
            ]
        )
        events: list[str] = []
        agent = _agent(
            llm,
            events,
            embedding=FakeEmbedding(),
            steps_per_attempt=1,
            max_attempts=2,
        )
        agent.hippo.register_tool(
            ToolDef(name="echo", description="echo text", parameters={})
        )

        async def echo(text: str) -> str:
            return text

        agent.executor.register("echo", echo)
        running = asyncio.create_task(agent.run_task("echo a"))
        await llm.blocked.wait()
        assert agent.hippo.working.get_reflections()

        await _exit_run(agent, running, 5.0, "Cancelled by SIGTERM")
        # Promotion would call the embedding server, which the exit has no
        # time for.
        assert agent.hippo.working.get_reflections() == []
        assert [
            entry.text
            for entry in agent.hippo.long._entries
            if entry.metadata.get("type") == "reflection"
        ] == []

    @pytest.mark.asyncio
    async def test_no_provider_starts_once_an_exit_began(self) -> None:
        events: list[str] = []
        provider = ExitProvider(events)
        agent = _agent(BlockingLLM(), events, provider)
        agent.begin_exit("Cancelled by SIGTERM", within_s=5.0)

        result = await agent.run_task("q")
        assert (result.status, result.error) == (
            "setup_error",
            "Cannot start the tool providers: the agent is closed",
        )
        assert "start" not in events


class TestExitingClose:
    @pytest.mark.asyncio
    async def test_close_kills_a_provider_that_does_not_close_in_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent_mod, "EXIT_SAVE_RESERVE_S", 0.1)
        events: list[str] = []
        hang = asyncio.Event()
        provider = ExitProvider(events, close_gate=hang)
        agent = _agent(BlockingLLM(), events, provider)

        started = time.monotonic()
        agent.begin_exit("Cancelled by SIGTERM", within_s=0.3)
        await asyncio.wait_for(agent.close(), 1.0)
        assert 0.15 <= time.monotonic() - started < 1.0
        assert events == ["close", "kill"]
        hang.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_close_does_not_wait_for_a_setup_past_the_deadline(self) -> None:
        events: list[str] = []
        never = asyncio.Event()
        provider = ExitProvider(events, start_gate=never)
        agent = _agent(BlockingLLM(), events, provider)
        # A run in another task is still starting the provider.
        running = asyncio.create_task(agent.run_task("q"))
        await provider.starting.wait()

        agent.begin_exit("Cancelled by SIGTERM", within_s=0.2)
        await asyncio.wait_for(agent.close(), 1.0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
