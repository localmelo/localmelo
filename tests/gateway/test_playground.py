"""Workspace session lifecycle tests without a model server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi import FastAPI  # noqa: E402

from localmelo.melo.agent import Agent  # noqa: E402
from localmelo.melo.contracts.providers import BaseLLMProvider  # noqa: E402
from localmelo.melo.memory.history.sqlite import SqliteHistory  # noqa: E402
from localmelo.melo.schema import (  # noqa: E402
    INTERRUPT_LABEL,
    NO_ANSWER_LABEL,
    Message,
    TaskRecord,
    ToolDef,
)
from localmelo.support.gateway.chat_gate import GatedLLM  # noqa: E402
from localmelo.support.gateway.playground import (  # noqa: E402
    SessionBusyError,
    SmokePlayground,
)
from localmelo.support.gateway.webapp import mount  # noqa: E402


class FakeLLM(BaseLLMProvider):
    def __init__(self) -> None:
        self.closed = False

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content="Hello")

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
async def manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[SmokePlayground]:
    manager = SmokePlayground(memory_dir=tmp_path)
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: FakeLLM())
    yield manager
    await manager.close_all()


def outcome(result: str) -> SimpleNamespace:
    """What a fake run_task returns: only the fields the workspace reads."""
    return SimpleNamespace(result=result, status="completed", error="")


async def create(manager: SmokePlayground) -> str:
    payload = await manager.create_session(
        chat_url="http://127.0.0.1:8000/v1", chat_model="Qwen-27B", adapter="omlx"
    )
    return str(payload["session_id"])


async def start_blocked(
    manager: SmokePlayground, sid: str
) -> tuple[asyncio.Task[dict[str, Any]], asyncio.Event]:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def run_task(query: str, **kwargs: Any) -> SimpleNamespace:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return outcome("unreachable")

    with patch.object(manager._sessions[sid].agent, "run_task", run_task):
        task = asyncio.create_task(manager.run(sid, "Long task"))
        await asyncio.wait_for(started.wait(), timeout=1)
    return task, stopped


async def test_transcript_can_be_recovered_after_reload(
    manager: SmokePlayground,
) -> None:
    sid = await create(manager)
    with patch.object(
        manager._sessions[sid].agent,
        "run_task",
        AsyncMock(return_value=outcome("<think>Check</think>42")),
    ):
        await manager.run(sid, "  What is six times seven?  ")
    listed = manager.list_sessions()
    assert listed[0]["title"] == "What is six times seven?"
    assert listed[0]["adapter"] == "omlx"
    assert "messages" not in listed[0]
    detail = manager.get_session(sid)
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["answer"] == "42"
    assert detail["messages"][1]["thinking"] == "Check"
    assert not detail["is_busy"]
    detail["messages"].clear()
    assert len(manager.get_session(sid)["messages"]) == 2


async def test_cancel_stops_agent_and_keeps_session_usable(
    manager: SmokePlayground,
) -> None:
    sid = await create(manager)
    task, stopped = await start_blocked(manager, sid)
    assert manager.get_session(sid)["is_busy"]
    assert await manager.cancel(sid)
    assert (await task)["cancelled"]
    assert stopped.is_set()
    assert not manager.get_session(sid)["is_busy"]
    assert manager.get_session(sid)["messages"][-1] == {
        "role": "note",
        "content": "Response stopped.",
    }
    assert not await manager.cancel(sid)
    with patch.object(
        manager._sessions[sid].agent,
        "run_task",
        AsyncMock(return_value=outcome("Recovered")),
    ):
        assert (await manager.run(sid, "Try again"))["answer"] == "Recovered"


async def test_busy_session_rejects_run_and_delete(manager: SmokePlayground) -> None:
    sid = await create(manager)
    task, stopped = await start_blocked(manager, sid)
    with pytest.raises(SessionBusyError):
        await manager.run(sid, "Duplicate")
    with pytest.raises(SessionBusyError):
        await manager.close(sid)
    assert not stopped.is_set()
    assert len(manager.get_session(sid)["messages"]) == 1
    await manager.cancel(sid)
    await task
    assert await manager.close(sid)


@pytest.mark.parametrize(
    "failure", [RuntimeError("backend unavailable"), TimeoutError()]
)
async def test_failed_runs_leave_recoverable_transcript(
    manager: SmokePlayground, failure: Exception
) -> None:
    sid = await create(manager)
    with (
        patch.object(
            manager._sessions[sid].agent, "run_task", AsyncMock(side_effect=failure)
        ),
        pytest.raises(type(failure)),
    ):
        await manager.run(sid, "Please help")
    detail = manager.get_session(sid)
    assert not detail["is_busy"]
    assert [m["role"] for m in detail["messages"]] == ["user", "note"]
    assert detail["messages"][-1]["content"]


async def test_request_cancellation_releases_session(manager: SmokePlayground) -> None:
    sid = await create(manager)
    task, stopped = await start_blocked(manager, sid)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert not manager.get_session(sid)["is_busy"]
    assert manager.get_session(sid)["messages"][-1]["role"] == "note"


async def test_all_busy_capacity_is_enforced(manager: SmokePlayground) -> None:
    manager._max = 1
    sid = await create(manager)
    task, _ = await start_blocked(manager, sid)
    with pytest.raises(SessionBusyError):
        await create(manager)
    assert len(manager.list_sessions()) == 1
    await manager.cancel(sid)
    await task


async def test_concurrent_creation_does_not_exceed_capacity(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager._max = 1
    seed = manager._seed_memories

    async def slow_seed(agent: Agent, scenario_id: str) -> dict[str, Any] | None:
        await asyncio.sleep(0.01)
        return await seed(agent, scenario_id)

    monkeypatch.setattr(manager, "_seed_memories", slow_seed)
    await asyncio.gather(create(manager), create(manager), create(manager))
    assert len(manager.list_sessions()) == 1


async def test_invalid_creation_does_not_evict_existing_chat(
    manager: SmokePlayground,
) -> None:
    manager._max = 1
    sid = await create(manager)
    with pytest.raises(ValueError):
        await manager.create_session(chat_url="", chat_model="")
    assert manager.get_session(sid)


async def test_closing_one_chat_preserves_other_sqlite_connection(
    manager: SmokePlayground,
) -> None:
    first = await create(manager)
    second = await create(manager)
    first_history = manager._sessions[first].agent.hippo.history
    second_history = manager._sessions[second].agent.hippo.history
    assert isinstance(first_history, SqliteHistory)
    assert isinstance(second_history, SqliteHistory)
    await first_history.count_tasks()
    await second_history.count_tasks()
    connection = second_history._conn
    assert first_history is not second_history
    assert first_history._db_path == second_history._db_path
    await manager.close(first)
    assert second_history._conn is connection
    assert await second_history.count_tasks() == 0


async def test_shutdown_cancels_work_before_closing_provider(
    manager: SmokePlayground,
) -> None:
    sid = await create(manager)
    llm = manager._sessions[sid].agent._llm
    # The gate wraps the builder's fake; close() still reaches it.
    assert isinstance(llm, GatedLLM)
    assert isinstance(llm.inner, FakeLLM)
    task, stopped = await start_blocked(manager, sid)
    await manager.close_all()
    assert (await task)["cancelled"]
    assert stopped.is_set()
    assert llm.inner.closed
    assert manager.list_sessions() == []


async def test_discovery_fetches_shared_endpoint_once(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = AsyncMock(return_value=["Qwen-27B", "embed"])
    monkeypatch.setattr(manager, "_fetch_models", fetch)
    result = await manager.discover(chat_url="127.0.0.1:8000/v1", adapter="omlx")
    assert result["chat_models"] == result["embedding_models"]
    fetch.assert_awaited_once_with("http://127.0.0.1:8000/v1")


async def test_seed_failure_closes_new_provider(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLM()
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: llm)
    monkeypatch.setattr(
        manager, "_seed_memories", AsyncMock(side_effect=RuntimeError("seed failed"))
    )
    with pytest.raises(RuntimeError, match="seed failed"):
        await create(manager)
    assert llm.closed
    assert manager.list_sessions() == []


async def test_failed_replacement_preserves_existing_chat(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager._max = 1
    sid = await create(manager)
    existing_provider = manager._sessions[sid].agent._llm
    assert isinstance(existing_provider, GatedLLM)
    assert isinstance(existing_provider.inner, FakeLLM)
    new_provider = FakeLLM()
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: new_provider)
    monkeypatch.setattr(
        manager, "_seed_memories", AsyncMock(side_effect=RuntimeError("seed failed"))
    )

    with pytest.raises(RuntimeError, match="seed failed"):
        await create(manager)

    assert [session["session_id"] for session in manager.list_sessions()] == [sid]
    assert not existing_provider.closed
    assert new_provider.closed


async def test_replacement_preserves_chat_that_becomes_busy(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager._max = 1
    sid = await create(manager)
    new_provider = FakeLLM()
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: new_provider)
    seeding = asyncio.Event()
    finish_seed = asyncio.Event()

    async def delayed_seed(agent: Agent, scenario_id: str) -> dict[str, Any] | None:
        seeding.set()
        await finish_seed.wait()
        return None

    monkeypatch.setattr(manager, "_seed_memories", delayed_seed)
    replacement = asyncio.create_task(create(manager))
    await asyncio.wait_for(seeding.wait(), timeout=1)
    run, stopped = await start_blocked(manager, sid)
    finish_seed.set()

    with pytest.raises(SessionBusyError):
        await replacement

    assert [session["session_id"] for session in manager.list_sessions()] == [sid]
    assert not stopped.is_set()
    assert new_provider.closed
    await manager.cancel(sid)
    await run


@pytest.fixture()
async def client(manager: SmokePlayground) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.state.smoke_playground = manager
    mount(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def test_http_transcript_recovery_and_delete(
    manager: SmokePlayground, client: httpx.AsyncClient
) -> None:
    response = await client.post(
        "/v1/playground/session",
        json={"chat_url": "http://localhost:8000/v1", "chat_model": "Qwen-27B"},
    )
    assert response.status_code == 200
    sid = response.json()["session_id"]
    with patch.object(
        manager._sessions[sid].agent,
        "run_task",
        AsyncMock(return_value=outcome("Hello")),
    ):
        assert (
            await client.post(f"/v1/playground/session/{sid}/run", json={"query": "Hi"})
        ).status_code == 200
    assert len((await client.get("/v1/playground/sessions")).json()["sessions"]) == 1
    assert (
        len((await client.get(f"/v1/playground/session/{sid}")).json()["messages"]) == 2
    )
    assert (await client.delete(f"/v1/playground/session/{sid}")).status_code == 200
    assert (await client.get(f"/v1/playground/session/{sid}")).status_code == 404


async def test_http_cancel_and_busy_conflicts(
    manager: SmokePlayground, client: httpx.AsyncClient
) -> None:
    sid = await create(manager)
    task, _ = await start_blocked(manager, sid)
    assert (await client.delete(f"/v1/playground/session/{sid}")).status_code == 409
    assert (
        await client.post(f"/v1/playground/session/{sid}/run", json={"query": "again"})
    ).status_code == 409
    response = await client.post(f"/v1/playground/session/{sid}/cancel")
    assert response.json() == {"cancelled": True, "session_id": sid}
    assert (await task)["cancelled"]
    assert (await client.post(f"/v1/playground/session/{sid}/cancel")).json()[
        "cancelled"
    ] is False
    assert (
        await client.post("/v1/playground/session/missing/cancel")
    ).status_code == 404


@pytest.mark.parametrize("value", [None, 42, [], {}])
async def test_http_rejects_non_string_queries(
    manager: SmokePlayground, client: httpx.AsyncClient, value: Any
) -> None:
    sid = await create(manager)
    response = await client.post(
        f"/v1/playground/session/{sid}/run", json={"query": value}
    )
    assert response.status_code == 400
    assert manager.get_session(sid)["messages"] == []


async def test_timeout_cancels_provider_work(manager: SmokePlayground) -> None:
    sid = await create(manager)
    stopped = asyncio.Event()

    async def blocked(query: str, **kwargs: Any) -> SimpleNamespace:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return outcome("unreachable")

    with (
        patch.object(manager._sessions[sid].agent, "run_task", blocked),
        pytest.raises(TimeoutError),
    ):
        await manager.run(sid, "Slow request", timeout=0.01)
    assert stopped.is_set()
    assert not manager.get_session(sid)["is_busy"]
    assert manager.get_session(sid)["messages"][-1]["content"] == "Response timed out."


async def test_embedding_model_uses_chat_url_when_url_omitted(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = Mock(return_value=None)
    monkeypatch.setattr(manager, "_build_embedding_provider", build)
    result = await manager.create_session(
        chat_url="localhost:8000/v1", chat_model="Qwen-27B", embedding_model="embed"
    )
    build.assert_called_once_with("http://localhost:8000/v1", "embed")
    assert result["embedding_url"] == "http://localhost:8000/v1"


class RecordingLLM(BaseLLMProvider):
    """Answers "Answer 1", "Answer 2", ... and keeps each request's messages."""

    def __init__(self, block_on: int = 0) -> None:
        self.requests: list[list[Message]] = []
        self.block_on = block_on
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        if len(self.requests) == self.block_on:
            self.blocked.set()
            await asyncio.Event().wait()
        return Message(role="assistant", content=f"Answer {len(self.requests)}")

    async def close(self) -> None:
        pass


def turns(messages: list[Message]) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in messages if m.role != "system"]


async def test_a_follow_up_sees_the_previous_answer_once(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = RecordingLLM()
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: llm)
    sid = await create(manager)
    assert (await manager.run(sid, "First question"))["answer"] == "Answer 1"
    assert (await manager.run(sid, "Follow-up"))["answer"] == "Answer 2"
    assert turns(llm.requests[1]) == [
        ("user", "First question"),
        ("assistant", "Answer 1"),
        ("user", "Follow-up"),
    ]


async def test_a_stop_closes_the_turn_with_the_interrupt_label(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = RecordingLLM(block_on=1)
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: llm)
    sid = await create(manager)
    task = asyncio.create_task(manager.run(sid, "Long question"))
    await asyncio.wait_for(llm.blocked.wait(), timeout=5)
    assert await manager.cancel(sid)
    assert (await task)["cancelled"]
    window = manager._sessions[sid].agent.hippo.working.get_window()
    assert turns(window) == [("user", "Long question"), ("assistant", INTERRUPT_LABEL)]
    assert (await manager.run(sid, "Try again"))["answer"] == "Answer 2"
    assert turns(llm.requests[1]) == [
        ("user", "Long question"),
        ("assistant", INTERRUPT_LABEL),
        ("user", "Try again"),
    ]


@pytest.mark.parametrize(
    ("result", "thinking", "answer"),
    [
        ("<think>Check</think>42", "Check", "42"),
        # The reply of a template that opens the block in the prompt.
        ("Check</think>42", "Check", "42"),
        # Cut off while the model thought: the label that memory keeps.
        ("<think>Check", "Check", NO_ANSWER_LABEL.format(status="completed")),
    ],
)
async def test_the_transcript_splits_the_answer_as_memory_does(
    manager: SmokePlayground, result: str, thinking: str, answer: str
) -> None:
    sid = await create(manager)
    with patch.object(
        manager._sessions[sid].agent,
        "run_task",
        AsyncMock(return_value=outcome(result)),
    ):
        response = await manager.run(sid, "What is six times seven?")
    assert (response["thinking"], response["answer"]) == (thinking, answer)
    message = manager.get_session(sid)["messages"][-1]
    assert (message["thinking"], message["answer"]) == (thinking, answer)


async def test_a_stop_during_the_final_save_keeps_the_answer(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = RecordingLLM()
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: llm)
    sid = await create(manager)
    hippo = manager._sessions[sid].agent.hippo
    promote = hippo.promote_reflections
    promoting = asyncio.Event()

    async def stalled_promotion(task_id: str) -> None:
        # As on an embedding server that stopped answering.
        promoting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(hippo, "promote_reflections", stalled_promotion)
    task = asyncio.create_task(manager.run(sid, "Invent a code"))
    await asyncio.wait_for(promoting.wait(), timeout=5)
    assert await manager.cancel(sid)
    response = await task
    assert not response["cancelled"]
    assert response["answer"] == "Answer 1"
    messages = manager.get_session(sid)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    monkeypatch.setattr(hippo, "promote_reflections", promote)
    assert (await manager.run(sid, "Repeat it"))["answer"] == "Answer 2"
    assert turns(llm.requests[1]) == [
        ("user", "Invent a code"),
        ("assistant", "Answer 1"),
        ("user", "Repeat it"),
    ]


def hold_cancelled_saves(
    agent: Agent, monkeypatch: pytest.MonkeyPatch
) -> tuple[asyncio.Event, asyncio.Event, list[tuple[str, str]]]:
    """Hold the save of each cancelled task until *release* is set.

    Returns ``(saving, release, saved)``: *saving* is set when a cancelled
    task's save is held, and *saved* lists the status and result of each
    save once it is done.
    """
    saving, release = asyncio.Event(), asyncio.Event()
    saved: list[tuple[str, str]] = []
    save_task = agent.hippo.save_task

    async def held_save(task: TaskRecord) -> None:
        if task.status == "cancelled":
            saving.set()
            await release.wait()
        await save_task(task)
        saved.append((task.status, task.result))

    monkeypatch.setattr(agent.hippo, "save_task", held_save)
    return saving, release, saved


@pytest.mark.parametrize("second", ["stop", "dropped request"])
async def test_the_chat_stays_busy_until_a_stopped_run_is_saved(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch, second: str
) -> None:
    llm = RecordingLLM(block_on=1)
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: llm)
    sid = await create(manager)
    saving, release, saved = hold_cancelled_saves(
        manager._sessions[sid].agent, monkeypatch
    )
    task = asyncio.create_task(manager.run(sid, "Long question"))
    await asyncio.wait_for(llm.blocked.wait(), timeout=5)
    stops = [asyncio.create_task(manager.cancel(sid))]
    await asyncio.wait_for(saving.wait(), timeout=5)
    # A Stop from a second tab, or the request of the run going away, during
    # the save: neither may let a next run start before the save is done.
    if second == "stop":
        stops.append(asyncio.create_task(manager.cancel(sid)))
    else:
        task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done() and not any(stop.done() for stop in stops)
    assert manager.get_session(sid)["is_busy"]
    release.set()
    assert await asyncio.gather(*stops) == [True] * len(stops)
    assert saved[-1][0] == "cancelled"
    if second == "stop":
        assert (await task)["cancelled"]
    else:
        with pytest.raises(asyncio.CancelledError):
            await task
    assert manager.get_session(sid)["messages"][-1] == {
        "role": "note",
        "content": "Response stopped.",
    }
    assert (await manager.run(sid, "Try again"))["answer"] == "Answer 2"


async def test_a_timeout_saves_the_run_as_timed_out(
    manager: SmokePlayground, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = RecordingLLM(block_on=1)
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: llm)
    sid = await create(manager)
    agent = manager._sessions[sid].agent
    _, release, saved = hold_cancelled_saves(agent, monkeypatch)
    release.set()
    with pytest.raises(TimeoutError):
        await manager.run(sid, "Long question", timeout=0.05)
    assert saved[-1] == ("cancelled", "Timed out after 0.05 s")
    assert manager.get_session(sid)["messages"][-1]["content"] == "Response timed out."
    window = agent.hippo.working.get_window()
    assert turns(window) == [("user", "Long question"), ("assistant", INTERRUPT_LABEL)]
