"""A failing model server, through the real OpenAI-compatible provider.

The server is an ``httpx.MockTransport``: no network.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import RunResult
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

ANSWER = {
    "choices": [{"message": {"role": "assistant", "content": "42"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


class FakeServer:
    """Answers each request with the next status or exception of a script.

    200 is a completion; another status carries an error body. Records the
    ``X-Localmelo-Attempt`` header of every request.
    """

    def __init__(self, *script: int | Exception) -> None:
        self.script: deque[int | Exception] = deque(script)
        self.attempts: list[str | None] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.attempts.append(request.headers.get("X-Localmelo-Attempt"))
        step = self.script.popleft() if self.script else 200
        if isinstance(step, Exception):
            raise step
        if step == 200:
            return httpx.Response(200, json=ANSWER)
        return httpx.Response(step, json={"error": "bench_budget_exhausted"})


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


async def _run(server: FakeServer, memory_dir: Path) -> tuple[RunResult, Sleeps]:
    llm = OpenAICompatLLM(
        base_url="http://llm.test/v1",
        model="m",
        transport=httpx.MockTransport(server.handler),
    )
    options = AgentOptions(builtins=frozenset(), memory_dir=str(memory_dir))
    agent = Agent(llm=llm, options=options)
    sleeps = Sleeps()
    agent.chat.sleep = sleeps
    try:
        return await agent.run_task("What is 6*7?"), sleeps
    finally:
        await agent.close()


async def _saved(memory_dir: Path, task_id: str) -> tuple[str, str]:
    history = SqliteHistory(memory_dir / "history.db")
    try:
        task = await history.get_task(task_id)
    finally:
        await history.aclose()
    assert task is not None
    return task.status, task.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script", "error"),
    [
        ((400,), 'failed with HTTP 400: {"error":"bench_budget_exhausted"}'),
        ((500,), 'failed with HTTP 500: {"error":"bench_budget_exhausted"}'),
    ],
)
async def test_an_error_status_ends_the_run_without_a_retry(
    tmp_path: Path, script: tuple[int, ...], error: str
) -> None:
    server = FakeServer(*script)
    result, sleeps = await _run(server, tmp_path)
    assert result.status == "provider_error"
    assert result.error == f"POST /chat/completions {error}"
    assert server.attempts == ["1"]
    assert sleeps.delays == []
    assert (result.n_llm, result.llm_errors) == (0, 1)
    assert await _saved(tmp_path, result.task_id) == ("provider_error", result.error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (httpx.ReadTimeout("timed out"), "ReadTimeout: timed out"),
        (httpx.ConnectError("All connection attempts failed"), "ConnectError: All"),
        (503, "HTTP 503"),
    ],
)
async def test_a_retryable_failure_ends_the_run_after_two_retries(
    tmp_path: Path, failure: int | Exception, error: str
) -> None:
    server = FakeServer(failure, failure, failure)
    result, sleeps = await _run(server, tmp_path)
    assert result.status == "provider_error"
    assert error in result.error
    assert server.attempts == ["1", "2", "3"]
    assert sleeps.delays == [1.0, 4.0]
    assert (result.n_llm, result.llm_errors) == (0, 3)
    assert await _saved(tmp_path, result.task_id) == ("provider_error", result.error)


@pytest.mark.asyncio
async def test_a_retry_that_gets_an_answer_completes_the_run(tmp_path: Path) -> None:
    server = FakeServer(httpx.ConnectError("refused"), 502, 200)
    result, sleeps = await _run(server, tmp_path)
    assert (result.status, result.result) == ("completed", "42")
    assert server.attempts == ["1", "2", "3"]
    assert sleeps.delays == [1.0, 4.0]
    assert (result.n_llm, result.llm_errors) == (1, 2)
    assert result.usage == ANSWER["usage"]
    assert await _saved(tmp_path, result.task_id) == ("completed", "42")
