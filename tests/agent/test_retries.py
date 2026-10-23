"""Chat's retries and the provider_error status — no real network or model."""

from __future__ import annotations

import asyncio
import logging
from collections import deque

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import RETRY_DELAYS_S, Chat
from localmelo.melo.contracts.providers import (
    LLM_REQUEST_ATTEMPT,
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.schema import Message, TaskRecord, ToolCall, ToolDef

REFUSED = 'POST /chat/completions failed with HTTP 400: {"error": "bench_budget"}'

ECHO = ToolDef(
    name="echo",
    description="echo text back",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


def _busy(status: int = 503) -> ProviderError:
    return ProviderError(
        f"POST /chat/completions failed with HTTP {status}: busy",
        status_code=status,
        retryable=True,
    )


def _refused() -> ProviderError:
    return ProviderError(REFUSED, status_code=400, retryable=False)


def _answer(text: str, usage: dict[str, int] | None = None) -> Message:
    return Message(role="assistant", content=text, usage=usage)


def _echo_call(text: str) -> Message:
    return Message(
        role="assistant",
        tool_call=ToolCall(tool_name="echo", arguments={"text": text}),
    )


class ScriptedLLM(BaseLLMProvider):
    """Pops queued replies, raising queued exceptions.

    Records ``LLM_REQUEST_ATTEMPT`` as each request sees it.
    """

    def __init__(self, replies: list[Message | BaseException]) -> None:
        self.replies: deque[Message | BaseException] = deque(replies)
        self.attempts: list[int] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.attempts.append(LLM_REQUEST_ATTEMPT.get())
        if not self.replies:
            return _answer("(no response queued)")
        reply = self.replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def close(self) -> None:
        pass


class BlockingLLM(ScriptedLLM):
    """Returns its queued replies, then waits in chat() until cancelled."""

    def __init__(self, replies: list[Message | BaseException] | None = None) -> None:
        super().__init__(replies or [])
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if self.replies:
            return await super().chat(messages, tools)
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")


class FailingEmbedding(BaseEmbeddingProvider):
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise self.error

    async def close(self) -> None:
        pass


class Sleeps:
    """A sleep that returns at once and records each delay."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


async def _echo(text: str) -> str:
    return text


def _agent(llm: BaseLLMProvider, options: AgentOptions | None = None) -> Agent:
    """An agent whose only tool is echo, with a sleep that returns at once."""
    agent = Agent(llm=llm, options=options or AgentOptions(builtins=frozenset()))
    agent.hippo.register_tool(ECHO)
    agent.executor.register("echo", _echo)
    agent.chat.sleep = Sleeps()
    return agent


def _delays(agent: Agent) -> list[float]:
    """The retry delays the agent's Chat waited out (see _agent)."""
    sleeps = agent.chat.sleep
    assert isinstance(sleeps, Sleeps)
    return sleeps.delays


def _only_task(agent: Agent) -> TaskRecord:
    tasks = list(agent.hippo.history._tasks.values())
    assert len(tasks) == 1
    return tasks[0]


def _user(text: str = "q") -> list[Message]:
    return [Message(role="user", content=text)]


# ── Chat ────────────────────────────────────────────────────────────────────


class TestChatRetries:
    def test_two_retries_after_1_s_and_4_s_by_default(self) -> None:
        chat = Chat(ScriptedLLM([]))
        assert chat.retry_delays == RETRY_DELAYS_S == (1.0, 4.0)
        assert chat.sleep is asyncio.sleep

    @pytest.mark.asyncio
    async def test_a_retryable_error_is_retried_until_a_response(self) -> None:
        usage = {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
        llm = ScriptedLLM([_busy(), _busy(429), _answer("ok", usage)])
        sleeps = Sleeps()
        chat = Chat(llm, sleep=sleeps)
        reply = await chat.plan_step(context=[], short=_user(), tools=[ECHO], query="q")
        assert reply.content == "ok"
        assert llm.attempts == [1, 2, 3]
        assert sleeps.delays == [1.0, 4.0]
        stats = chat.stats
        assert (stats.n_llm, stats.n_main, stats.n_aux) == (1, 1, 0)
        assert stats.llm_errors == 2
        assert stats.usage == usage

    @pytest.mark.asyncio
    async def test_after_two_retries_the_last_error_propagates(self) -> None:
        llm = ScriptedLLM([_busy(502), _busy(503), _busy(504), _answer("never")])
        sleeps = Sleeps()
        chat = Chat(llm, sleep=sleeps)
        with pytest.raises(ProviderError) as excinfo:
            await chat.reflect(short=_user(), query="q", attempt_id=0)
        assert excinfo.value.status_code == 504
        assert llm.attempts == [1, 2, 3]
        assert sleeps.delays == [1.0, 4.0]
        assert (chat.stats.n_llm, chat.stats.llm_errors) == (0, 3)

    @pytest.mark.asyncio
    async def test_a_non_retryable_error_is_not_retried(self) -> None:
        llm = ScriptedLLM([_refused(), _answer("never")])
        sleeps = Sleeps()
        chat = Chat(llm, sleep=sleeps)
        with pytest.raises(ProviderError, match="HTTP 400"):
            await chat._request(_user(), tools=None)
        assert (llm.attempts, sleeps.delays) == ([1], [])
        assert (chat.stats.n_llm, chat.stats.llm_errors) == (0, 1)

    @pytest.mark.asyncio
    async def test_other_exceptions_are_not_retried(self) -> None:
        llm = ScriptedLLM([ConnectionError("refused"), _answer("never")])
        sleeps = Sleeps()
        chat = Chat(llm, sleep=sleeps)
        with pytest.raises(ConnectionError):
            await chat._request(_user(), tools=None)
        assert (llm.attempts, sleeps.delays) == ([1], [])
        assert chat.stats.llm_errors == 1

    @pytest.mark.asyncio
    async def test_a_custom_schedule_sets_the_number_of_retries(self) -> None:
        llm = ScriptedLLM([_busy(), _busy(), _answer("never")])
        sleeps = Sleeps()
        chat = Chat(llm, retry_delays=[0.5], sleep=sleeps)
        with pytest.raises(ProviderError):
            await chat._request(_user(), tools=None)
        assert (llm.attempts, sleeps.delays) == ([1, 2], [0.5])

    @pytest.mark.asyncio
    async def test_the_attempt_number_is_back_to_1_after_a_request(self) -> None:
        chat = Chat(ScriptedLLM([_busy(), _answer("a"), _busy(), _refused()]))
        chat.sleep = Sleeps()
        await chat._request(_user(), tools=None)
        assert LLM_REQUEST_ATTEMPT.get() == 1
        with pytest.raises(ProviderError):
            await chat._request(_user(), tools=None)
        assert LLM_REQUEST_ATTEMPT.get() == 1

    @pytest.mark.asyncio
    async def test_a_request_cut_short_by_a_cancellation_counts_as_an_error(
        self,
    ) -> None:
        llm = BlockingLLM()
        chat = Chat(llm)
        request = asyncio.create_task(chat._request(_user(), tools=None))
        await llm.blocked.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert (chat.stats.n_llm, chat.stats.llm_errors) == (0, 1)

    @pytest.mark.asyncio
    async def test_a_cancellation_while_waiting_to_retry_sends_nothing_more(
        self,
    ) -> None:
        llm = ScriptedLLM([_busy(), _answer("never")])
        waiting = asyncio.Event()

        async def sleep(delay: float) -> None:
            waiting.set()
            await asyncio.Event().wait()

        chat = Chat(llm, sleep=sleep)
        request = asyncio.create_task(chat._request(_user(), tools=None))
        await waiting.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert llm.attempts == [1]
        assert (chat.stats.n_llm, chat.stats.llm_errors) == (0, 1)


# ── The provider_error status ───────────────────────────────────────────────


class TestProviderErrorStatus:
    @pytest.mark.asyncio
    async def test_a_refused_request_ends_the_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = _agent(ScriptedLLM([_refused()]))
        with caplog.at_level(logging.DEBUG):
            result = await agent.run_task("q")
        assert (result.status, result.result, result.error) == (
            "provider_error",
            "",
            REFUSED,
        )
        assert (result.steps, result.attempts) == (1, 1)
        assert (result.n_llm, result.llm_errors) == (0, 1)
        task = _only_task(agent)
        assert (task.status, task.result) == ("provider_error", REFUSED)
        # Reported, not a crash: nothing logged with a traceback.
        assert [r for r in caplog.records if r.exc_info] == []

    @pytest.mark.asyncio
    async def test_run_shows_the_status_and_the_error(self) -> None:
        agent = _agent(ScriptedLLM([_refused()]))
        assert await agent.run("q") == f"[provider_error] {REFUSED}"

    @pytest.mark.asyncio
    async def test_the_run_ends_after_two_retries(self) -> None:
        llm = ScriptedLLM([_busy(), _busy(), _busy(), _answer("never")])
        agent = _agent(llm)
        result = await agent.run_task("q")
        assert result.status == "provider_error"
        assert result.error == "POST /chat/completions failed with HTTP 503: busy"
        assert llm.attempts == [1, 2, 3]
        assert _delays(agent) == [1.0, 4.0]
        assert (result.steps, result.n_llm, result.llm_errors) == (1, 0, 3)

    @pytest.mark.asyncio
    async def test_a_retry_that_gets_a_response_continues_the_run(self) -> None:
        llm = ScriptedLLM([_busy(), _answer("42")])
        result = await _agent(llm).run_task("q")
        assert (result.status, result.result) == ("completed", "42")
        assert (result.steps, result.n_llm, result.llm_errors) == (1, 1, 1)

    @pytest.mark.asyncio
    async def test_a_failed_reflection_request_ends_the_run(self) -> None:
        llm = ScriptedLLM([_echo_call("a"), _refused()])
        options = AgentOptions(builtins=frozenset(), steps_per_attempt=1)
        result = await _agent(llm, options).run_task("echo q")
        assert (result.status, result.error) == ("provider_error", REFUSED)
        assert (result.steps, result.n_tool) == (1, 1)
        assert (result.n_llm, result.n_main, result.n_aux) == (1, 1, 0)
        assert result.llm_errors == 1

    @pytest.mark.asyncio
    async def test_a_failed_embedding_request_ends_the_run(self) -> None:
        error = ProviderError(
            "POST /embeddings failed: ConnectError: refused", retryable=True
        )
        llm = ScriptedLLM([_answer("never")])
        agent = Agent(
            llm=llm,
            embedding=FailingEmbedding(error),
            options=AgentOptions(builtins=frozenset()),
        )
        result = await agent.run_task("q")
        assert (result.status, result.error) == ("provider_error", str(error))
        # Retrieval embeds the query before the first planning call.
        assert (result.steps, llm.attempts) == (0, [])
        assert _only_task(agent).status == "provider_error"

    @pytest.mark.asyncio
    async def test_a_provider_error_without_a_message_names_its_type(self) -> None:
        result = await _agent(ScriptedLLM([ProviderError("")])).run_task("q")
        assert (result.status, result.error) == ("provider_error", "ProviderError")

    @pytest.mark.asyncio
    async def test_other_exceptions_still_crash_without_a_retry(self) -> None:
        llm = ScriptedLLM([ConnectionError("refused"), _answer("never")])
        agent = _agent(llm)
        result = await agent.run_task("q")
        assert (result.status, result.error) == (
            "crashed",
            "ConnectionError: refused",
        )
        assert llm.attempts == [1]
        assert _delays(agent) == []
