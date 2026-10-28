"""Agent: tool providers are set up at the first run — no real network or model."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Sequence
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker import DEFAULT_BUILTINS
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tool_capture import TOOL_ENTRY
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool, ToolFunction
from localmelo.melo.schema import Message, TaskRecord, ToolCall, ToolDef

# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Pops queued replies and records every request."""

    def __init__(self, replies: list[Message] | None = None) -> None:
        self.replies: deque[Message] = deque(replies or [])
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            return Message(role="assistant", content="(no response queued)")
        return self.replies.popleft()

    async def close(self) -> None:
        self.closed = True


def _tool_function(tool: str, delay_s: float) -> ToolFunction:
    async def run(text: str = "") -> str:
        await asyncio.sleep(delay_s)
        return f"{tool}: {text}"

    return run


class FakeToolProvider(BaseToolProvider):
    """Provides tools that answer ``"<tool>: <text>"`` after *delay_s*.

    *error* is raised by start() and *close_error* by close(). With a
    *gate*, start() waits until the gate is set, and with a *close_gate*,
    close() does. With *stop_s*, a start() cancelled while it waits for
    its gate takes *stop_s* seconds to stop what it started, as an MCP
    provider stops the servers it spawned, records ``stopped <name>`` and
    then gives up. ``starting`` is set once start() is called, ``closed``
    once close() is; ``closes_done`` counts the close() calls that ran to
    their end. *events* records each start and close in order, and may be
    shared between providers.
    """

    def __init__(
        self,
        name: str,
        tools: Sequence[str] = (),
        *,
        timeout_ms: float | None = None,
        delay_s: float = 0.0,
        error: Exception | None = None,
        close_error: Exception | None = None,
        gate: asyncio.Event | None = None,
        close_gate: asyncio.Event | None = None,
        stop_s: float | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.name = name
        self.tools = list(tools)
        self.timeout_ms = timeout_ms
        self.delay_s = delay_s
        self.error = error
        self.close_error = close_error
        self.gate = gate
        self.close_gate = close_gate
        self.stop_s = stop_s
        self.events = events if events is not None else []
        self.starting = asyncio.Event()
        self.closed = asyncio.Event()
        self.starts = 0
        self.closes = 0
        self.closes_done = 0
        # start() returned its tools and close() was not called since.
        self.running = False

    async def start(self) -> list[ProvidedTool]:
        self.starts += 1
        self.events.append(f"start {self.name}")
        self.starting.set()
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                if self.stop_s is not None:
                    await asyncio.sleep(self.stop_s)
                    self.events.append(f"stopped {self.name}")
                raise
        if self.error is not None:
            raise self.error
        self.running = True
        return [
            ProvidedTool(
                definition=ToolDef(
                    name=tool,
                    description=f"{tool} tool",
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                ),
                function=_tool_function(tool, self.delay_s),
                timeout_ms=self.timeout_ms,
            )
            for tool in self.tools
        ]

    async def close(self) -> None:
        self.closes += 1
        self.running = False
        self.events.append(f"close {self.name}")
        self.closed.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closes_done += 1
        if self.close_error is not None:
            raise self.close_error


# ── Helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _agent(
    llm: FakeLLM,
    providers: Sequence[BaseToolProvider],
    options: AgentOptions | None = None,
) -> Agent:
    return Agent(llm=llm, options=options, tool_providers=providers)


def _provided_only(**kwargs: Any) -> AgentOptions:
    """No built-ins and every tool offered on every step."""
    return AgentOptions(builtins=frozenset(), tool_exposure="all", **kwargs)


def _call(name: str, text: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_call=ToolCall(tool_name=name, arguments={"text": text}),
    )


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _offered(call: dict[str, Any]) -> list[str]:
    """Names of the tools one request offered, in order."""
    return [t.name for t in call["tools"] or []]


def _results(call: dict[str, Any]) -> list[str]:
    """Contents of the tool-result messages one request carried, in order."""
    return [m.content for m in call["messages"] if m.role == "tool"]


def _tasks(agent: Agent) -> list[TaskRecord]:
    return list(agent.hippo.history._tasks.values())


# ── Setup ───────────────────────────────────────────────────────────────────


class TestSetup:
    @pytest.mark.asyncio
    async def test_providers_start_once_at_the_first_run(self) -> None:
        events: list[str] = []
        providers = [
            FakeToolProvider(name, [f"{name}_tool"], events=events) for name in "ab"
        ]
        agent = _agent(FakeLLM([_answer("one"), _answer("two")]), providers)
        assert events == []  # constructing the agent starts nothing

        first = await agent.run_task("q1")
        second = await agent.run_task("q2")
        assert (first.status, second.status) == ("completed", "completed")
        assert events == ["start a", "start b"]

        await agent.close()
        assert events == ["start a", "start b", "close a", "close b"]

    @pytest.mark.asyncio
    async def test_provider_start_marks_capture_incomplete_first(self) -> None:
        events: list[str] = []
        provider = FakeToolProvider("p", ["tool"], events=events)
        agent = _agent(FakeLLM([_answer("ok")]), [provider])

        async def capture() -> None:
            events.append("capture")

        token = TOOL_ENTRY.set(capture)
        try:
            await agent.start()
        finally:
            TOOL_ENTRY.reset(token)
            await agent.close()
        # A provider can spawn a child (an MCP server) before any tool call,
        # so the execution is incomplete before its start().
        assert events == ["capture", "start p", "close p"]

    @pytest.mark.asyncio
    async def test_provided_tools_are_registered_next_to_the_builtins(self) -> None:
        timed = FakeToolProvider("a", ["slow_tool"], timeout_ms=1500.0)
        untimed = FakeToolProvider("b", ["quick_tool"])
        agent = _agent(FakeLLM([_answer("ok")]), [timed, untimed])
        result = await agent.run_task("q")

        provided = {"slow_tool", "quick_tool"}
        assert {t.name for t in agent.hippo.list_tools()} == DEFAULT_BUILTINS[
            "python"
        ] | provided
        assert provided <= set(agent.executor._callables)
        # A provider's timeout_ms is registered with the executor; without
        # one, the executor's default applies.
        assert agent.executor._timeouts_ms.get("slow_tool") == 1500.0
        assert "quick_tool" not in agent.executor._timeouts_ms
        # config.tools lists every registered tool; builtins only built-ins.
        assert result.config["tools"] == sorted(DEFAULT_BUILTINS["python"] | provided)
        assert result.config["builtins"] == sorted(DEFAULT_BUILTINS["python"])

    @pytest.mark.asyncio
    async def test_a_provided_tool_runs_through_the_agent(self) -> None:
        llm = FakeLLM([_call("lookup", "x"), _answer("done")])
        provider = FakeToolProvider("p", ["lookup", "fetch"])
        agent = _agent(llm, [provider], _provided_only())
        result = await agent.run_task("hello")

        assert (result.status, result.result, result.n_tool) == ("completed", "done", 1)
        # Every planning request offered exactly the provided tools, sorted.
        assert [_offered(call) for call in llm.calls] == [["fetch", "lookup"]] * 2
        assert _results(llm.calls[1]) == ["lookup: x"]
        assert result.config["tools"] == ["fetch", "lookup"]
        assert result.config["builtins"] == []

    @pytest.mark.asyncio
    async def test_retrieve_mode_offers_a_provided_tool_the_query_names(
        self,
    ) -> None:
        llm = FakeLLM([_call("lookup", "x"), _answer("done")])
        provider = FakeToolProvider("p", ["lookup"])
        agent = _agent(llm, [provider], AgentOptions(builtins=frozenset()))
        assert await agent.run("use lookup on x") == "done"
        assert _offered(llm.calls[0]) == ["lookup"]
        assert _results(llm.calls[1]) == ["lookup: x"]

    @pytest.mark.asyncio
    async def test_the_providers_timeout_bounds_each_call(self) -> None:
        llm = FakeLLM([_call("slow", "x"), _answer("done")])
        provider = FakeToolProvider("p", ["slow"], timeout_ms=100.0, delay_s=10.0)
        agent = _agent(llm, [provider], _provided_only())
        assert await agent.run("hello") == "done"
        assert _results(llm.calls[1]) == ["Error: Execution timed out after 0.1s"]

    @pytest.mark.asyncio
    async def test_concurrent_setups_start_each_provider_once(self) -> None:
        gate = asyncio.Event()
        provider = FakeToolProvider("a", ["a_tool"], gate=gate)
        agent = _agent(FakeLLM(), [provider])
        first = asyncio.create_task(agent._ensure_tools())
        second = asyncio.create_task(agent._ensure_tools())
        await provider.starting.wait()
        gate.set()
        await asyncio.gather(first, second)
        assert provider.starts == 1
        assert agent.hippo.get_tool("a_tool") is not None
        await agent.close()

    @pytest.mark.asyncio
    async def test_an_agent_without_providers_needs_no_setup(self) -> None:
        agent = Agent(llm=FakeLLM([_answer("ok")]))
        await agent.close()
        # Closing it first does not fail the run: there is nothing to start.
        result = await agent.run_task("q")
        assert (result.status, result.result) == ("completed", "ok")


# ── Setup failures ──────────────────────────────────────────────────────────


class TestSetupErrors:
    @pytest.mark.asyncio
    async def test_a_name_taken_by_a_builtin(self) -> None:
        llm = FakeLLM([_answer("never sent")])
        provider = FakeToolProvider("p", ["mine", "file_read"])
        agent = _agent(llm, [provider])
        builtin = agent.hippo.get_tool("file_read")
        builtin_fn = agent.executor._callables["file_read"]

        result = await agent.run_task("q")
        error = "Provided tool 'file_read' collides with a registered tool"
        assert (result.status, result.result, result.error) == (
            "setup_error",
            "",
            error,
        )
        assert (result.steps, result.attempts, result.n_llm, result.n_tool) == (
            0,
            0,
            0,
            0,
        )
        assert llm.calls == []
        # Nothing was registered or replaced, and the provider is closed.
        assert agent.hippo.get_tool("file_read") is builtin
        assert agent.executor._callables["file_read"] is builtin_fn
        assert agent.hippo.get_tool("mine") is None
        assert result.config["tools"] == sorted(DEFAULT_BUILTINS["python"])
        assert (provider.starts, provider.closes, provider.running) == (1, 1, False)
        # The task row is saved with the status; the query never reached
        # working memory.
        [task] = _tasks(agent)
        assert (task.task_id, task.status, task.result) == (
            result.task_id,
            "setup_error",
            error,
        )
        assert agent.hippo.working.get_window() == []

    @pytest.mark.asyncio
    async def test_a_name_taken_by_a_tool_registered_on_the_agent(self) -> None:
        provider = FakeToolProvider("p", ["echo"])
        agent = _agent(FakeLLM(), [provider], AgentOptions(builtins=frozenset()))
        agent.hippo.register_tool(
            ToolDef(name="echo", description="echo text back", parameters={})
        )
        result = await agent.run_task("q")
        assert (result.status, result.error) == (
            "setup_error",
            "Provided tool 'echo' collides with a registered tool",
        )
        assert provider.closes == 1

    @pytest.mark.asyncio
    async def test_a_name_two_providers_share(self) -> None:
        events: list[str] = []
        first = FakeToolProvider("a", ["shared", "only_a"], events=events)
        second = FakeToolProvider("b", ["shared"], events=events)
        agent = _agent(FakeLLM(), [first, second], AgentOptions(builtins=frozenset()))
        result = await agent.run_task("q")
        assert (result.status, result.error) == (
            "setup_error",
            "More than one provided tool is named 'shared'",
        )
        assert events == ["start a", "start b", "close a", "close b"]
        assert agent.hippo.list_tools() == []

    @pytest.mark.asyncio
    async def test_ensure_tools_raises_value_error_for_a_taken_name(self) -> None:
        agent = _agent(FakeLLM(), [FakeToolProvider("a", ["file_read"])])
        with pytest.raises(ValueError, match="'file_read' collides"):
            await agent._ensure_tools()

    @pytest.mark.asyncio
    async def test_a_provider_that_fails_to_start(self) -> None:
        events: list[str] = []
        providers = [
            FakeToolProvider("a", ["a_tool"], events=events),
            FakeToolProvider("b", error=ConnectionError("b is down"), events=events),
            FakeToolProvider("c", ["c_tool"], events=events),
        ]
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm, providers)
        result = await agent.run_task("q")
        assert (result.status, result.error) == ("setup_error", "b is down")
        # a started and is closed again, and so is b, whose start() was
        # called; c was never started.
        assert events == ["start a", "start b", "close a", "close b"]
        assert agent.hippo.get_tool("a_tool") is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_a_start_failure_without_a_message_names_its_type(self) -> None:
        agent = _agent(FakeLLM(), [FakeToolProvider("a", error=OSError())])
        result = await agent.run_task("q")
        assert (result.status, result.error) == ("setup_error", "OSError")

    @pytest.mark.asyncio
    async def test_run_shows_the_status_and_the_error(self) -> None:
        provider = FakeToolProvider("a", error=RuntimeError("no server"))
        agent = _agent(FakeLLM(), [provider])
        assert await agent.run("q") == "[setup_error] no server"

    @pytest.mark.asyncio
    async def test_setup_is_not_retried(self) -> None:
        provider = FakeToolProvider("a", ["a_tool"], error=RuntimeError("no server"))
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm, [provider])
        await agent.run_task("q1")
        provider.error = None  # even once the cause is gone

        second = await agent.run_task("q2")
        assert (second.status, second.error) == (
            "setup_error",
            "Tool setup failed in an earlier run: no server",
        )
        assert provider.starts == 1
        assert llm.calls == []
        assert [task.status for task in _tasks(agent)] == ["setup_error"] * 2

    @pytest.mark.asyncio
    async def test_a_close_failure_during_the_clean_up_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = FakeToolProvider("a", ["a_tool"], close_error=RuntimeError("stuck"))
        second = FakeToolProvider("b", error=RuntimeError("no server"))
        agent = _agent(FakeLLM(), [first, second])
        result = await agent.run_task("q")
        # The run reports why setup failed, not the failure to clean up.
        assert (result.status, result.error) == ("setup_error", "no server")
        assert (first.closes, second.closes) == (1, 1)
        assert "Could not close tool provider FakeToolProvider" in caplog.text

    @pytest.mark.asyncio
    async def test_a_cancellation_during_the_clean_up_waits_for_it(self) -> None:
        gate = asyncio.Event()
        slow = FakeToolProvider("a", ["a_tool"], close_gate=gate)
        failing = FakeToolProvider("b", error=ConnectionError("b is down"))
        agent = _agent(FakeLLM(), [slow, failing])
        running = asyncio.create_task(agent.run_task("q"))
        await slow.closed.wait()  # b failed to start; a is being closed

        running.cancel()
        await asyncio.sleep(0.05)
        # The cancellation does not interrupt a's close() ...
        assert not running.done()
        assert (slow.closes_done, failing.closes) == (0, 0)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        # ... and propagates once every started provider is closed.
        assert (slow.closes_done, failing.closes_done) == (1, 1)
        [task] = _tasks(agent)
        assert (task.status, task.result) == ("crashed", "CancelledError")


# ── timeout_s ───────────────────────────────────────────────────────────────


class TestSetupTimeout:
    @pytest.mark.asyncio
    async def test_a_setup_still_running_at_timeout_s_ends_as_timeout(self) -> None:
        provider = FakeToolProvider("a", ["a_tool"], gate=asyncio.Event())
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm, [provider], AgentOptions(timeout_s=0.05))

        # The gate never opens: without timeout_s the setup would wait for
        # ever, so the outer bound turns that into a failure.
        result = await asyncio.wait_for(agent.run_task("q"), 10)
        assert (result.status, result.error) == ("timeout", "Timed out after 0.05 s")
        # The interrupted setup closed the provider and registered nothing.
        assert (provider.starts, provider.closes) == (1, 1)
        assert agent.hippo.get_tool("a_tool") is None
        assert llm.calls == []

        later = await agent.run_task("q2")
        assert (later.status, later.error) == (
            "setup_error",
            "Tool setup failed in an earlier run: CancelledError",
        )
        assert provider.starts == 1

    @pytest.mark.asyncio
    async def test_a_timeout_during_the_clean_up_does_not_cut_it_short(
        self,
    ) -> None:
        events: list[str] = []
        gate = asyncio.Event()
        slow = FakeToolProvider("a", ["a_tool"], close_gate=gate, events=events)
        failing = FakeToolProvider(
            "b", error=ConnectionError("b is down"), events=events
        )
        unused = FakeToolProvider("c", ["c_tool"], events=events)
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm, [slow, failing, unused], AgentOptions(timeout_s=0.05))
        running = asyncio.create_task(agent.run_task("q"))
        await slow.closed.wait()  # b failed to start; a is being closed

        # This sleep starts after the deadline was set, so the deadline
        # expires first, while a's close() is still waiting.
        await asyncio.sleep(0.1)
        assert not running.done()
        assert (slow.closes_done, failing.closes) == (0, 0)
        gate.set()
        result = await asyncio.wait_for(running, 10)
        # The deadline expired before the setup was over, clean-up included.
        assert (result.status, result.error) == ("timeout", "Timed out after 0.05 s")
        # a's close() ran to its end and b, whose start() was called, was
        # closed after it; c never started, so it was not closed.
        assert events == ["start a", "start b", "close a", "close b"]
        assert (slow.closes_done, failing.closes_done, unused.closes) == (1, 1, 0)
        assert agent.hippo.get_tool("a_tool") is None
        assert llm.calls == []

        # Later runs name the failure that started the clean-up.
        later = await agent.run_task("q2")
        assert (later.status, later.error) == (
            "setup_error",
            "Tool setup failed in an earlier run: b is down",
        )
        assert [task.status for task in _tasks(agent)] == ["timeout", "setup_error"]

    @pytest.mark.asyncio
    async def test_the_clean_up_of_a_start_it_cancels_is_waited_for(self) -> None:
        # timeout_s cancels start(), which then takes 0.3 s to stop what it
        # started. close() returns at once, as an MCP provider's can while
        # its start() still stops a server, so closing the provider does
        # not wait for that: only waiting for start() itself does.
        events: list[str] = []
        provider = FakeToolProvider(
            "a", ["a_tool"], gate=asyncio.Event(), stop_s=0.3, events=events
        )
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm, [provider], AgentOptions(timeout_s=0.1))

        result = await asyncio.wait_for(agent.run_task("q"), 10)
        assert (result.status, result.error) == ("timeout", "Timed out after 0.1 s")
        # The run returned only once start() had stopped, and closed the
        # provider after that.
        assert events == ["start a", "stopped a", "close a"]
        assert agent.hippo.get_tool("a_tool") is None
        assert llm.calls == []

        await agent.close()
        assert events == ["start a", "stopped a", "close a", "close a"]


# ── close() ─────────────────────────────────────────────────────────────────


class TestClose:
    @pytest.mark.asyncio
    async def test_close_closes_every_provider(self) -> None:
        events: list[str] = []
        providers = [
            FakeToolProvider(name, [f"{name}_tool"], events=events) for name in "ab"
        ]
        llm = FakeLLM([_answer("ok")])
        agent = _agent(llm, providers)
        await agent.run_task("q")

        await agent.close()
        assert events == ["start a", "start b", "close a", "close b"]
        assert not any(provider.running for provider in providers)
        assert llm.closed
        await agent.close()  # again: harmless
        assert [provider.closes for provider in providers] == [2, 2]

    @pytest.mark.asyncio
    async def test_close_before_any_run_closes_the_providers(self) -> None:
        provider = FakeToolProvider("a", ["a_tool"])
        agent = _agent(FakeLLM(), [provider])
        await agent.close()
        assert (provider.starts, provider.closes) == (0, 1)

    @pytest.mark.asyncio
    async def test_a_provider_that_fails_to_close_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        failing = FakeToolProvider("a", close_error=RuntimeError("stuck"))
        other = FakeToolProvider("b")
        llm = FakeLLM()
        agent = _agent(llm, [failing, other])
        with caplog.at_level(logging.WARNING, logger="localmelo.melo.agent.agent"):
            await agent.close()
        # The other provider and the model provider are still closed.
        assert other.closes == 1
        assert llm.closed
        [record] = [
            r
            for r in caplog.records
            if "Could not close tool provider" in r.getMessage()
        ]
        # One line, no traceback: closing a provider is not a crash.
        assert record.levelno == logging.WARNING
        assert record.exc_info is None
        assert record.getMessage() == (
            "Could not close tool provider FakeToolProvider: RuntimeError: stuck"
        )

    @pytest.mark.asyncio
    async def test_a_cancelled_close_still_closes_every_provider(self) -> None:
        gate = asyncio.Event()
        slow = FakeToolProvider("a", close_gate=gate)
        other = FakeToolProvider("b")
        llm = FakeLLM()
        agent = _agent(llm, [slow, other])
        closing = asyncio.create_task(agent.close())
        await slow.closed.wait()

        closing.cancel()
        await asyncio.sleep(0.05)
        # The cancellation waits until every provider is closed ...
        assert not closing.done()
        assert other.closes == 0
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert (slow.closes_done, other.closes_done) == (1, 1)
        # ... and the model provider is still closed.
        assert llm.closed

    @pytest.mark.asyncio
    async def test_no_provider_starts_after_close(self) -> None:
        provider = FakeToolProvider("a", ["a_tool"])
        llm = FakeLLM([_answer("never sent")])
        agent = _agent(llm, [provider])
        await agent.close()
        result = await agent.run_task("q")
        assert (result.status, result.error) == (
            "setup_error",
            "Cannot start the tool providers: the agent is closed",
        )
        assert provider.starts == 0
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_close_waits_for_a_setup_still_running(self) -> None:
        gate = asyncio.Event()
        provider = FakeToolProvider("a", ["a_tool"], gate=gate)
        agent = _agent(FakeLLM([_answer("never sent")]), [provider])
        running = asyncio.create_task(agent.run_task("q"))
        await provider.starting.wait()

        closing = asyncio.create_task(agent.close())
        await provider.closed.wait()
        # This provider's start() does not give up when it is closed, so
        # close() waits for the setup ...
        assert not closing.done()
        gate.set()
        await closing
        # ... which saw close() begin and closed the provider again.
        assert (provider.starts, provider.closes, provider.running) == (1, 2, False)
        result = await running
        assert (result.status, result.error) == (
            "setup_error",
            "Cannot start the tool providers: the agent is closed",
        )
        assert agent.hippo.get_tool("a_tool") is None
