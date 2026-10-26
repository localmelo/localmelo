"""Working memory across runs, and the Agent's tool API — no real model.

``record_answers`` keeps each run's answer in working memory;
``conversation()`` and ``load_conversation()`` read and replace the window
between runs; ``start()`` sets up the tool providers ahead of the first run;
``list_tools()`` lists the tools and ``run_tool()`` runs one call outside a
run. Every run is scripted, with in-memory history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

import pytest

import localmelo.melo.agent.agent as agent_mod
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker.payloads import MemoryWritePayload, ValidationResult
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    DenyAllPolicy,
    denial_text,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.memory.short import group_turns
from localmelo.melo.schema import (
    ANSWER_NOT_KEPT_LABEL,
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    INTERRUPT_LABEL,
    NO_ANSWER_LABEL,
    SESSION_TEXT_CHARS,
    SHORT_TERM_MAX,
    Message,
    ReflectionEntry,
    ToolCall,
    ToolDef,
    ToolInfo,
    answer_text,
    clip_session_text,
    split_thinking,
)

# ── Fakes ───────────────────────────────────────────────────────────────────


class ScriptedLLM(BaseLLMProvider):
    """Answers with each scripted reply in turn, then waits until cancelled.

    ``requests`` keeps the messages of every request; ``blocked`` is set
    once a request waits for a reply that is not scripted.
    """

    def __init__(self, replies: Iterable[Message] = ()) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[list[Message]] = []
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        if self.replies:
            return self.replies.popleft()
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def close(self) -> None:
        pass


class Tools:
    """The custom tool; ``calls`` keeps the text of each call that ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def echo(self, text: str = "") -> str:
        self.calls.append(text)
        return text


ECHO = ToolDef(
    name="echo",
    description="Echo text back",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)
WEATHER = ToolDef(
    name="weather",
    description="Forecast for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


class WeatherProvider(BaseToolProvider):
    """Provides ``weather``, which requires approval; counts its starts.

    With *error*, start() raises it.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.starts = 0

    async def start(self) -> list[ProvidedTool]:
        self.starts += 1
        if self.error is not None:
            raise self.error
        return [
            ProvidedTool(
                definition=WEATHER, function=self._weather, requires_approval=True
            )
        ]

    async def _weather(self, city: str = "") -> str:
        return f"Sunny in {city}"

    async def close(self) -> None:
        pass


class RecordingPolicy:
    """Allows every call and keeps each request."""

    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision(True)


class Recorder:
    """A sink that keeps every event."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)


async def _no_wait(delay: float) -> None:
    pass


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _agent(
    llm: BaseLLMProvider,
    *,
    tools: Tools | None = None,
    providers: Sequence[BaseToolProvider] = (),
    **options: Any,
) -> Agent:
    """An agent with the custom tool, offered on every step, and no waits."""
    options.setdefault("builtins", frozenset())
    options.setdefault("tool_exposure", "all")
    agent = Agent(llm=llm, options=AgentOptions(**options), tool_providers=providers)
    agent.hippo.register_tool(ECHO)
    agent.executor.register("echo", (tools or Tools()).echo)
    agent.chat.sleep = _no_wait
    return agent


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _calls(*calls: tuple[str, dict[str, Any], str], content: str = "") -> Message:
    """A reply with tool calls, each ``(name, arguments, id)``."""
    return Message(
        role="assistant",
        content=content,
        tool_calls=[ToolCall(tool_name=n, arguments=a, id=i) for n, a, i in calls],
    )


def _result(call_id: str, text: str = "ok") -> Message:
    return Message(role="tool", content=text, tool_call_id=call_id)


def _echo_call(call_id: str = "call_1") -> Message:
    return _calls(("echo", {"text": "a"}, call_id))


def _window(agent: Agent) -> list[tuple[str, str]]:
    return [(msg.role, msg.content) for msg in agent.hippo.working.get_window()]


def _shape(messages: Iterable[Message]) -> list[tuple[str, str, list[str], str]]:
    return [
        (m.role, m.content, [c.id for c in m.tool_calls], m.tool_call_id)
        for m in messages
    ]


async def _stop(running: asyncio.Task[Any]) -> None:
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


# ── record_answers ──────────────────────────────────────────────────────────


class TestRecordAnswers:
    @pytest.mark.asyncio
    async def test_off_appends_nothing(self) -> None:
        agent = _agent(ScriptedLLM([_echo_call(), _answer("Hello.")]))

        result = await agent.run_task("hi")

        assert result.status == "completed"
        assert [role for role, _ in _window(agent)] == ["user", "assistant", "tool"]

    @pytest.mark.asyncio
    async def test_on_appends_the_answer(self) -> None:
        agent = _agent(
            ScriptedLLM([_echo_call(), _answer("Hello.")]), record_answers=True
        )

        result = await agent.run_task("hi")

        assert result.result == "Hello."
        window = agent.hippo.working.get_window()
        assert [msg.role for msg in window] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert (window[-1].content, window[-1].tool_calls) == ("Hello.", [])

    @pytest.mark.asyncio
    async def test_the_next_request_alternates_user_answer_user(self) -> None:
        llm = ScriptedLLM([_answer("Hello."), _answer("Fine.")])
        agent = _agent(llm, record_answers=True)

        await agent.run_task("hi")
        await agent.run_task("how are you?")

        assert [(m.role, m.content) for m in llm.requests[1][1:]] == [
            ("user", "hi"),
            ("assistant", "Hello."),
            ("user", "how are you?"),
        ]

    @pytest.mark.asyncio
    async def test_a_cancelled_run_keeps_the_interrupt_label(self) -> None:
        llm = ScriptedLLM()
        agent = _agent(llm, record_answers=True)
        running = asyncio.create_task(agent.run_task("hi", on_event=Recorder()))
        await asyncio.wait_for(llm.blocked.wait(), 5.0)

        await _stop(running)

        assert _window(agent) == [("user", "hi"), ("assistant", INTERRUPT_LABEL)]

    @pytest.mark.asyncio
    async def test_a_run_without_an_answer_keeps_a_status_label(self) -> None:
        agent = _agent(ScriptedLLM([_answer(""), _answer("")]), record_answers=True)

        result = await agent.run_task("hi")

        assert (result.status, result.error) == ("failed", "empty replies")
        assert _window(agent) == [
            ("user", "hi"),
            ("assistant", EMPTY_REPLY_LABEL),
            ("user", EMPTY_REPLY_NUDGE),
            ("assistant", "[no answer: the run ended with status failed]"),
        ]

    @pytest.mark.asyncio
    async def test_a_whitespace_only_result_keeps_a_status_label(self) -> None:
        reflection = {
            "summary": "Echoed once.",
            "recommended_action": "stop",
            "best_effort_result": "   ",
        }
        agent = _agent(
            ScriptedLLM([_echo_call(), _answer(json.dumps(reflection))]),
            record_answers=True,
            max_attempts=1,
            steps_per_attempt=1,
        )

        result = await agent.run_task("hi")

        assert (result.status, result.result) == ("budget_exhausted", "   ")
        assert _window(agent)[-1] == (
            "assistant",
            "[no answer: the run ended with status budget_exhausted]",
        )

    @pytest.mark.asyncio
    async def test_a_long_answer_is_clipped_with_the_marker(self) -> None:
        agent = _agent(ScriptedLLM([_answer("x" * 60_000)]), record_answers=True)

        result = await agent.run_task("hi")

        assert result.result == "x" * 60_000
        assert _window(agent)[-1] == (
            "assistant",
            "x" * 8_000 + "\n... [52000 more characters not shown]",
        )

    @pytest.mark.asyncio
    async def test_nothing_is_appended_when_the_query_never_reached_the_window(
        self,
    ) -> None:
        agent = _agent(
            ScriptedLLM(),
            providers=[WeatherProvider(RuntimeError("no server"))],
            record_answers=True,
        )

        result = await agent.run_task("hi")

        assert result.status == "setup_error"
        assert _window(agent) == []

    @pytest.mark.asyncio
    async def test_an_answer_the_check_refuses_is_replaced_by_a_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(ScriptedLLM([_answer("Hello.")]), record_answers=True)

        def refuse(payload: MemoryWritePayload) -> ValidationResult:
            return ValidationResult(allowed=False, reason="no room")

        monkeypatch.setattr(agent.checker, "check_memory_write", refuse)

        result = await agent.run_task("hi")

        assert result.result == "Hello."
        assert _window(agent)[-1] == (
            "assistant",
            ANSWER_NOT_KEPT_LABEL.format(reason="no room"),
        )
        assert _window(agent)[-1][1] == ("[answer not kept in working memory: no room]")

    @pytest.mark.asyncio
    async def test_a_failure_to_keep_the_answer_changes_nothing_else(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _agent(ScriptedLLM([_answer("Hello.")]), record_answers=True)

        def broken(payload: MemoryWritePayload) -> ValidationResult:
            raise RuntimeError("checker broke")

        monkeypatch.setattr(agent.checker, "check_memory_write", broken)

        with caplog.at_level(logging.ERROR, logger=agent_mod.__name__):
            result = await agent.run_task("hi")

        assert (result.status, result.result) == ("completed", "Hello.")
        task = await agent.hippo.history.get_task(result.task_id)
        assert task is not None and task.status == "completed"
        assert _window(agent) == [("user", "hi")]
        [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert record.getMessage() == (
            f"Could not keep the answer of task {result.task_id} in working memory"
        )


class TestLongAnswers:
    @pytest.mark.asyncio
    async def test_a_fourth_turn_still_plans_after_three_long_answers(self) -> None:
        answers = ["a" * 40_000, "b" * 40_000, "c" * 40_000, "Short."]
        llm = ScriptedLLM([_answer(text) for text in answers])
        agent = _agent(llm, record_answers=True)

        results = [await agent.run_task(f"turn {n}") for n in range(1, 5)]

        assert [r.status for r in results] == ["completed"] * 4
        assert results[3].result == "Short."
        marker = "\n... [32000 more characters not shown]"
        assert [(m.role, m.content) for m in llm.requests[3][1:]] == [
            ("user", "turn 1"),
            ("assistant", "a" * SESSION_TEXT_CHARS + marker),
            ("user", "turn 2"),
            ("assistant", "b" * SESSION_TEXT_CHARS + marker),
            ("user", "turn 3"),
            ("assistant", "c" * SESSION_TEXT_CHARS + marker),
            ("user", "turn 4"),
        ]


# ── Thinking blocks ─────────────────────────────────────────────────────────


class TestThinkingBlocks:
    @pytest.mark.parametrize(
        ("reply", "kept"),
        [
            # Ollama's provider puts its reasoning in front of the content so.
            ("<think>\nI should greet.\n</think>\n\nHello.", "Hello."),
            ("<THINK>I should greet.</Think>\nHello.", "Hello."),
            (
                "<think>I should greet, but the output was cut",
                "[no answer: the run ended with status completed]",
            ),
            (
                "<think>\nOnly thinking.\n</think>\n\n",
                "[no answer: the run ended with status completed]",
            ),
            ("I should greet.\n</think>\n\nHello.", "Hello."),
        ],
        ids=["ollama_prefix", "upper_case", "unclosed", "only_a_block", "bare_closing"],
    )
    @pytest.mark.asyncio
    async def test_the_kept_answer_leaves_the_block_out(
        self, reply: str, kept: str
    ) -> None:
        agent = _agent(ScriptedLLM([_answer(reply)]), record_answers=True)

        result = await agent.run_task("hi")

        assert result.result == reply  # the RunResult keeps the block
        assert _window(agent)[-1] == ("assistant", kept)

    @pytest.mark.parametrize(
        ("text", "split"),
        [
            ("Hello.", ("", "Hello.")),
            ("  Hello.\n", ("", "Hello.")),
            ("<think>\nGreet.\n</think>\n\nHello.", ("Greet.", "Hello.")),
            ("<ThInK>Greet.</tHiNk>Hello.", ("Greet.", "Hello.")),
            ("Before <think>Greet.</think> after.", ("Greet.", "Before  after.")),
            (
                "<think>one</think> middle <think>two</think>",
                ("one", "middle <think>two</think>"),
            ),
            ("\n  <think>Cut off while", ("Cut off while", "")),
            ("Hello <think>never closed", ("", "Hello <think>never closed")),
            # The template opened the block in the prompt: only </think> is left.
            ("Greet first.\n</think>\n\nHello.", ("Greet first.", "Hello.")),
            ("Greet.</THINK>Hello.", ("Greet.", "Hello.")),
            ("Only thinking.</think>", ("Only thinking.", "")),
            ("<think></think>", ("", "")),
            ("", ("", "")),
        ],
    )
    def test_split_thinking(self, text: str, split: tuple[str, str]) -> None:
        assert split_thinking(text) == split


class TestAnswerHelpers:
    def test_answer_text(self) -> None:
        assert answer_text("completed", "Hello.") == "Hello."
        assert answer_text("crashed", " partial ") == " partial "
        assert answer_text("cancelled", "") == INTERRUPT_LABEL
        assert answer_text("cancelled", " \n") == INTERRUPT_LABEL
        assert answer_text("timeout", "  ") == NO_ANSWER_LABEL.format(status="timeout")
        assert INTERRUPT_LABEL == "[interrupt: no assistant output for this turn]"
        assert NO_ANSWER_LABEL.format(status="failed") == (
            "[no answer: the run ended with status failed]"
        )

    def test_clip_session_text(self) -> None:
        assert SESSION_TEXT_CHARS == 8_000
        fits = "y" * SESSION_TEXT_CHARS
        assert clip_session_text(fits) == fits
        assert clip_session_text(fits + "zz") == (
            fits + "\n... [2 more characters not shown]"
        )


# ── conversation() and load_conversation() ──────────────────────────────────


class TestLoadConversation:
    def test_whole_units_are_loaded(self) -> None:
        agent = _agent(ScriptedLLM())
        messages = [
            Message(role="system", content="Be brief."),
            Message(role="user", content="q"),
            _calls(("echo", {}, "c1"), ("echo", {}, "c2")),
            _result("c2"),
            _result("c1"),
            _answer("a"),
            _answer(EMPTY_REPLY_LABEL),
            Message(role="user", content=EMPTY_REPLY_NUDGE),
        ]

        assert agent.load_conversation(messages) == len(messages)
        assert _shape(agent.hippo.working.get_window()) == _shape(messages)

    def test_an_incomplete_unit_and_an_orphan_result_are_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _agent(ScriptedLLM())
        messages = [
            Message(role="user", content="q"),
            _calls(("echo", {}, "c1"), ("echo", {}, "c2")),
            _result("c1"),  # c2 has no result
            Message(role="user", content="next"),
            _result("c9"),  # no call
            _calls(("echo", {}, "c3")),
            _result("c3"),
            _result("c3"),  # answered twice
            _calls(("echo", {}, "c4")),  # no result at all
            Message(role="developer", content="unknown role"),
            _calls(("echo", {}, "c5")),
            _result("c5"),
        ]

        with caplog.at_level(logging.WARNING, logger=agent_mod.__name__):
            kept = agent.load_conversation(messages)

        assert kept == 4
        assert _shape(agent.hippo.working.get_window()) == _shape(
            [messages[0], messages[3], messages[10], messages[11]]
        )
        warnings = [r.getMessage() for r in caplog.records]
        assert warnings == [
            "Dropped 2 message(s) from the loaded conversation: a tool call whose "
            "results do not answer each call once",
            "Dropped 1 message(s) from the loaded conversation: a tool result "
            "without its call",
            "Dropped 3 message(s) from the loaded conversation: a tool call whose "
            "results do not answer each call once",
            "Dropped 1 message(s) from the loaded conversation: a tool call whose "
            "results do not answer each call once",
            "Dropped 1 message(s) from the loaded conversation: a message with the "
            "role 'developer'",
        ]
        assert all(r.levelno == logging.WARNING for r in caplog.records)

    def test_an_empty_reply_label_keeps_its_nudge(self) -> None:
        agent = _agent(ScriptedLLM())
        nudge = Message(role="user", content=EMPTY_REPLY_NUDGE)
        messages = [
            Message(role="user", content="q"),
            _answer(EMPTY_REPLY_LABEL),
            nudge,
        ]

        agent.load_conversation(messages)

        units = group_turns(agent.hippo.working.get_window())
        assert [[m.content for m in unit] for unit in units] == [
            ["q"],
            [EMPTY_REPLY_LABEL, EMPTY_REPLY_NUDGE],
        ]

    def test_the_oldest_messages_are_evicted(self) -> None:
        agent = _agent(ScriptedLLM())
        messages = [
            Message(role="user" if n % 2 == 0 else "assistant", content=str(n))
            for n in range(SHORT_TERM_MAX + 5)
        ]

        assert agent.load_conversation(messages) == SHORT_TERM_MAX
        assert [content for _, content in _window(agent)] == [
            str(n) for n in range(5, SHORT_TERM_MAX + 5)
        ]

    @pytest.mark.asyncio
    async def test_an_empty_list_clears_the_conversation(self) -> None:
        agent = _agent(ScriptedLLM([_answer("Hello.")]), record_answers=True)
        await agent.run_task("hi")
        agent.hippo.working.add_reflection(ReflectionEntry(summary="tried"))

        assert agent.load_conversation([]) == 0
        assert _window(agent) == []
        assert agent.hippo.working.get_reflections() == []

    @pytest.mark.asyncio
    async def test_a_loaded_conversation_reaches_the_next_request(self) -> None:
        llm = ScriptedLLM([_answer("Now.")])
        agent = _agent(llm)
        agent.load_conversation(
            [Message(role="user", content="earlier"), _answer("Then.")]
        )

        await agent.run_task("now")

        assert [(m.role, m.content) for m in llm.requests[0][1:]] == [
            ("user", "earlier"),
            ("assistant", "Then."),
            ("user", "now"),
        ]

    @pytest.mark.asyncio
    async def test_it_is_refused_during_a_run(self) -> None:
        llm = ScriptedLLM()
        agent = _agent(llm)
        running = asyncio.create_task(agent.run_task("hi"))
        await asyncio.wait_for(llm.blocked.wait(), 5.0)

        with pytest.raises(RuntimeError) as excinfo:
            agent.load_conversation([])

        assert str(excinfo.value) == "Cannot load a conversation while a run is active"
        assert _window(agent) == [("user", "hi")]
        await _stop(running)
        assert agent.load_conversation([]) == 0

    def test_a_non_message_is_refused_before_anything_is_cleared(self) -> None:
        agent = _agent(ScriptedLLM())
        agent.load_conversation([Message(role="user", content="q")])

        with pytest.raises(TypeError):
            agent.load_conversation([{"role": "user", "content": "x"}])  # type: ignore[list-item]

        assert _window(agent) == [("user", "q")]

    def test_messages_are_copied_both_ways(self) -> None:
        agent = _agent(ScriptedLLM())
        original = Message(role="user", content="q")
        agent.load_conversation([original])
        original.content = "changed"

        copy = agent.conversation()
        copy[0].content = "changed too"
        copy.append(_answer("extra"))

        assert _window(agent) == [("user", "q")]


# ── start(), list_tools() and run_tool() ────────────────────────────────────


class TestStartAndTools:
    @pytest.mark.asyncio
    async def test_start_starts_the_providers_once(self) -> None:
        provider = WeatherProvider()
        agent = _agent(ScriptedLLM([_answer("Hello.")]), providers=[provider])

        await agent.start()
        await agent.start()
        result = await agent.run_task("hi")

        assert result.status == "completed"
        assert provider.starts == 1

    @pytest.mark.asyncio
    async def test_start_raises_the_setup_failure(self) -> None:
        provider = WeatherProvider(RuntimeError("no server"))
        agent = _agent(ScriptedLLM(), providers=[provider])

        with pytest.raises(RuntimeError, match="^no server$"):
            await agent.start()
        with pytest.raises(RuntimeError) as excinfo:
            await agent.start()

        assert str(excinfo.value) == "Tool setup failed in an earlier run: no server"
        assert provider.starts == 1

    @pytest.mark.asyncio
    async def test_list_tools_reports_sources_and_marks(self) -> None:
        agent = _agent(ScriptedLLM(), providers=[WeatherProvider()], builtins=None)

        before = [tool.name for tool in agent.list_tools()]
        await agent.start()
        tools = agent.list_tools()

        assert "weather" not in before
        assert [tool.name for tool in tools] == sorted(before + ["weather"])
        summary = {tool.name: (tool.source, tool.requires_approval) for tool in tools}
        assert summary == {
            "echo": ("custom", False),
            "file_read": ("builtin", False),
            "file_write": ("builtin", True),
            "python_exec": ("builtin", True),
            "shell_exec": ("builtin", True),
            "weather": ("provided", True),
        }
        [weather] = [tool for tool in tools if tool.name == "weather"]
        assert weather == ToolInfo(
            name="weather",
            description="Forecast for a city",
            parameters=WEATHER.parameters,
            source="provided",
            requires_approval=True,
        )
        # A copy: changing it does not change the registered tool.
        weather.parameters["properties"].clear()
        assert agent.hippo.get_tool("weather") == WEATHER
        assert WEATHER.parameters["properties"] == {"city": {"type": "string"}}

    @pytest.mark.asyncio
    async def test_run_tool_returns_the_output(self) -> None:
        tools = Tools()
        agent = _agent(ScriptedLLM(), tools=tools)

        result = await agent.run_tool("echo", {"text": "hi"})

        assert (result.tool_name, result.output, result.error) == ("echo", "hi", "")
        assert tools.calls == ["hi"]

    @pytest.mark.asyncio
    async def test_run_tool_starts_the_providers(self) -> None:
        provider = WeatherProvider()
        agent = _agent(ScriptedLLM(), providers=[provider])

        result = await agent.run_tool("weather", {"city": "Oslo"})

        assert result.output == "Sunny in Oslo"
        assert provider.starts == 1

    @pytest.mark.asyncio
    async def test_run_tool_gives_the_denial_text_under_a_denying_policy(
        self,
    ) -> None:
        tools = Tools()
        agent = _agent(ScriptedLLM(), tools=tools)
        agent.executor.register("echo", tools.echo, requires_approval=True)

        result = await agent.run_tool(
            "echo", {"text": "hi"}, approval=DenyAllPolicy("not now.")
        )

        assert result.output == ""
        assert result.error == denial_text(
            ApprovalDecision(False, "not now.", origin="policy")
        )
        assert result.error == (
            "The approval policy denied this call: not now. Do not run it again "
            "or reach the same result another way unless the user asks."
        )
        assert tools.calls == []

    @pytest.mark.asyncio
    async def test_the_request_has_no_run_id_and_a_fresh_call_id(self) -> None:
        tools = Tools()
        agent = _agent(ScriptedLLM(), tools=tools)
        agent.executor.register("echo", tools.echo, requires_approval=True)
        policy = RecordingPolicy()

        await agent.run_tool("echo", {"text": "a"}, approval=policy)
        await agent.run_tool("echo", {"text": "b"}, approval=policy)

        first, second = policy.requests
        assert (first.run_id, second.run_id) == ("", "")
        assert re.fullmatch(r"call_[0-9a-f]{24}", first.call_id)
        assert first.call_id != second.call_id
        assert first.request_id != second.request_id
        assert dict(first.arguments) == {"text": "a"}
        assert tools.calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_run_tool_writes_no_memory_or_history(self) -> None:
        agent = _agent(ScriptedLLM())
        agent.load_conversation([Message(role="user", content="q")])

        await agent.run_tool("echo", {"text": "hi"})

        assert _window(agent) == [("user", "q")]
        assert agent.hippo.history._tasks == {}  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_run_tool_reports_a_refusal_as_its_error(self) -> None:
        agent = _agent(ScriptedLLM())

        result = await agent.run_tool("nope", {})

        assert (result.output, result.error) == ("", "Tool not found in registry: nope")

    @pytest.mark.asyncio
    async def test_run_tool_is_refused_during_a_run(self) -> None:
        llm = ScriptedLLM()
        tools = Tools()
        agent = _agent(llm, tools=tools)
        running = asyncio.create_task(agent.run_task("hi"))
        await asyncio.wait_for(llm.blocked.wait(), 5.0)

        with pytest.raises(RuntimeError) as excinfo:
            await agent.run_tool("echo", {"text": "hi"})

        assert str(excinfo.value) == "Cannot run a tool while a run is active"
        assert tools.calls == []
        await _stop(running)

    @pytest.mark.asyncio
    async def test_run_tool_checks_the_policy_type(self) -> None:
        agent = _agent(ScriptedLLM())

        with pytest.raises(TypeError):
            await agent.run_tool("echo", {}, approval=object())  # type: ignore[arg-type]
