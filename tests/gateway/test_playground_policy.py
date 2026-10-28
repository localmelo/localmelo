"""The gate wired into the workspace: payload, timeout, stop and routes (R3-R5)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi import FastAPI  # noqa: E402

from localmelo.melo.contracts.providers import BaseLLMProvider  # noqa: E402
from localmelo.melo.schema import Message, RunResult, ToolDef  # noqa: E402
from localmelo.support.config import (  # noqa: E402
    Config,
    GatewayConfig,
    LocalBackendConfig,
    SchedulerConfig,
)
from localmelo.support.gateway import webapp  # noqa: E402
from localmelo.support.gateway.chat_gate import GatedLLM  # noqa: E402
from localmelo.support.gateway.playground import SmokePlayground  # noqa: E402
from localmelo.support.scheduler import REASONS  # noqa: E402
from localmelo.support.scheduler.sim.trace import read_trace  # noqa: E402

CHAT_URL = "http://127.0.0.1:8000/v1"


class FakeLLM(BaseLLMProvider):
    def __init__(self, delay: float = 0.0) -> None:
        self.model = "Qwen-27B"
        self.last_reasoning_key = ""
        self.last_usage: dict[str, int] | None = {
            "prompt_tokens": 12,
            "completion_tokens": 4,
        }
        self.delay = delay
        self.requests = 0
        self.closed = False

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return Message(role="assistant", content="Hello")

    async def close(self) -> None:
        self.closed = True


def make_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm: FakeLLM | None = None,
    **kwargs: Any,
) -> tuple[SmokePlayground, FakeLLM]:
    manager = SmokePlayground(memory_dir=tmp_path, **kwargs)
    provider = llm if llm is not None else FakeLLM()
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: provider)
    return manager, provider


def admitted(call: Any) -> bool:
    return bool(call.future is not None and call.future.done())


async def create(manager: SmokePlayground) -> str:
    payload = await manager.create_session(
        chat_url=CHAT_URL, chat_model="Qwen-27B", adapter="omlx"
    )
    return str(payload["session_id"])


def client_for(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def make_app(manager: SmokePlayground) -> FastAPI:
    app = FastAPI()
    app.state.smoke_playground = manager
    webapp.mount(app)
    return app


async def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    """Await *predicate* up to *timeout* seconds, sleeping in small steps."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


def release_later(gate: Any, call: Any, seconds: float) -> asyncio.Task[None]:
    async def release() -> None:
        await asyncio.sleep(seconds)
        gate.finish(call)

    return asyncio.create_task(release())


# ── The payload (R16, R28) ──


async def test_the_session_payload_publishes_level_and_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="fcfs", n_srv=2)
    try:
        session_id = await create(manager)
        payload = manager.get_session(session_id)
        assert payload["level"] == 2
        assert payload["held"] == ""
        holder = manager._gate.submit("holder")
        held = manager._gate.submit(session_id)
        assert not held.future.done()  # type: ignore[union-attr]
        assert manager.get_session(session_id)["held"] in REASONS
        manager._gate.finish(holder)
        # Freeing the slot admits the call, so its reason is gone at once.
        assert manager.get_session(session_id)["held"] == ""
        manager._gate.finish(held)
        assert manager.get_session(session_id)["held"] == ""
    finally:
        await manager.close_all()


async def test_off_leaves_the_page_payload_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="off")
    try:
        session_id = await create(manager)
        payload = manager.get_session(session_id)
        assert payload["held"] == ""
        assert payload["level"] == 2
        assert set(payload) == {
            "session_id",
            "adapter",
            "chat_url",
            "chat_model",
            "embedding_url",
            "embedding_model",
            "scenario_id",
            "scenario",
            "created_at",
            "last_active",
            "title",
            "is_busy",
            "level",
            "held",
            "messages",
        }
    finally:
        await manager.close_all()


# ── The 600 s clock (R17) ──


async def test_a_hold_does_not_trip_the_run_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="fcfs", n_srv=1)
    try:
        session_id = await create(manager)
        holder = manager._gate.submit("background")
        release = release_later(manager._gate, holder, 0.4)
        try:
            response = await manager.run(session_id, "hello", timeout=0.2)
        finally:
            await release
        assert response["answer"] == "Hello"
        # The call really was held, so the timeout path was exercised.
        assert manager._gate.held_seconds(session_id) >= 0.3
    finally:
        await manager.close_all()


async def test_a_hold_that_starts_near_the_timeout_still_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="fcfs", n_srv=1)
    try:
        session_id = await create(manager)

        async def run_task(query: str, **kwargs: Any) -> RunResult:
            await asyncio.sleep(0.18)
            call = manager._gate.submit(session_id)
            await manager._gate.acquire(call)
            manager._gate.finish(call, "ok")
            return RunResult(task_id="t", status="completed", result="held then done")

        monkeypatch.setattr(manager._sessions[session_id].agent, "run_task", run_task)
        holder: list[Any] = []

        async def fill_then_release() -> None:
            await asyncio.sleep(0.15)
            holder.append(manager._gate.submit("background"))
            await asyncio.sleep(0.3)
            manager._gate.finish(holder[0])

        filler = asyncio.create_task(fill_then_release())
        try:
            response = await manager.run(session_id, "hello", timeout=0.2)
        finally:
            await filler
        assert response["answer"] == "held then done"
        assert manager._gate.held_seconds(session_id) >= 0.1
    finally:
        await manager.close_all()


class ClockBox:
    """A monotonic clock the test advances by hand, in microseconds."""

    def __init__(self) -> None:
        self.mono_us = 1_000_000
        self.wall_us = 1_700_000_000_000_000

    def __call__(self) -> Any:
        from localmelo.support.scheduler import Clock

        return Clock(self.mono_us, self.wall_us)

    def advance(self, seconds: float) -> None:
        self.mono_us += round(seconds * 1_000_000)
        self.wall_us += round(seconds * 1_000_000)


async def test_a_hold_wakes_when_the_focus_window_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="fcfs", n_srv=8, n_fg=1)
    clock = ClockBox()
    manager._gate._clock = clock
    manager._hold_wake_s = 0.02
    gate = manager._gate
    try:
        gate.focus("focused")
        holder = gate.submit("background")
        assert admitted(holder)
        started = gate.turn_start("waiting")
        held = gate.submit("waiting")
        assert not admitted(held)
        task = asyncio.create_task(gate.acquire(held))
        waiter = asyncio.create_task(
            manager._wait_for_turn("waiting", task, 60.0, started)
        )
        await asyncio.sleep(0.05)
        assert gate.held("waiting") in REASONS
        clock.advance(31.0)  # the typing heartbeat is now stale
        await asyncio.wait_for(waiter, 2.0)
        assert admitted(held)
        assert gate.held("waiting") == ""
        gate.finish(held)
        gate.finish(holder)
    finally:
        await manager.close_all()


async def test_the_timeout_still_fires_when_nothing_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(
        tmp_path, monkeypatch, policy="fcfs", n_srv=8, llm=FakeLLM(delay=0.3)
    )
    app = make_app(manager)
    real_run = manager.run

    async def run_short(session: str, query: str, **kwargs: Any) -> dict[str, Any]:
        kwargs["timeout"] = 0.1
        return await real_run(session, query, **kwargs)

    monkeypatch.setattr(manager, "run", run_short)
    try:
        session_id = await create(manager)
        async with client_for(app) as client:
            response = await client.post(
                f"/v1/playground/session/{session_id}/run", json={"query": "slow"}
            )
        assert response.status_code == 504
        assert "timed out" in response.json()["error"].lower()
    finally:
        await manager.close_all()


# ── Stop (R21) ──


async def test_a_stop_during_a_hold_ends_the_turn_and_keeps_the_chat_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(
        tmp_path, monkeypatch, policy="fcfs", n_srv=1, call_log_dir=tmp_path
    )
    try:
        session_id = await create(manager)
        holder = manager._gate.submit("background")
        run = asyncio.create_task(manager.run(session_id, "first"))
        await wait_until(lambda: manager._gate.held(session_id) != "")
        assert await manager.cancel(session_id)
        assert (await run)["cancelled"]
        assert manager._gate.held(session_id) == ""
        manager._gate.finish(holder)
        response = await manager.run(session_id, "second")
        assert response["answer"] == "Hello"
        assert manager._gate.call_log_path is not None
        traces = read_trace(manager._gate.call_log_path)
        statuses = [call.status for trace in traces for call in trace.calls]
        assert "refused" in statuses
    finally:
        await manager.close_all()


# ── The routes (R4, R26, R27) ──


async def test_the_level_route_sets_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="fcfs", n_srv=8)
    app = make_app(manager)
    try:
        session_id = await create(manager)
        async with client_for(app) as client:
            path = f"/v1/playground/session/{session_id}/level"
            for level in (1, 2, 3, 4):
                response = await client.post(path, json={"level": level})
                assert response.status_code == 200, response.text
                assert response.json() == {"session_id": session_id, "level": level}
                assert manager.get_session(session_id)["level"] == level
            for bad in (0, 5, 2.0, True, "2", None):
                response = await client.post(path, json={"level": bad})
                assert response.status_code == 400, bad
            missing = await client.post(
                "/v1/playground/session/unknown/level", json={"level": 2}
            )
            assert missing.status_code == 404
    finally:
        await manager.close_all()


async def test_the_routes_are_accepted_under_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="off")
    app = make_app(manager)
    try:
        session_id = await create(manager)
        async with client_for(app) as client:
            level = await client.post(
                f"/v1/playground/session/{session_id}/level", json={"level": 4}
            )
            assert level.status_code == 200
            assert level.json() == {"session_id": session_id, "level": 4}
            presence = await client.post(
                "/v1/playground/presence", json={"session_id": session_id}
            )
            assert presence.status_code == 200
            assert presence.json() == {"ok": True}
        payload = manager.get_session(session_id)
        assert payload["level"] == 4
        assert payload["held"] == ""
    finally:
        await manager.close_all()


async def test_the_presence_route_moves_the_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="fcfs", n_srv=8)
    app = make_app(manager)
    try:
        first = await create(manager)
        second = await create(manager)
        async with client_for(app) as client:
            response = await client.post(
                "/v1/playground/presence", json={"session_id": second}
            )
            assert response.status_code == 200
            assert response.json() == {"ok": True}
            bad = await client.post("/v1/playground/presence", json={"session_id": 3})
            assert bad.status_code == 400
            missing = await client.post(
                "/v1/playground/presence", json={"session_id": "unknown"}
            )
            assert missing.status_code == 404
        in_second = manager._gate.submit(second)
        in_first = manager._gate.submit(first)
        assert in_second.meta.session_kind == "front"
        assert in_first.meta.session_kind == "chat"
        manager._gate.finish(in_second)
        manager._gate.finish(in_first)
    finally:
        await manager.close_all()


# ── The configured playground (R3, R4) ──


def _scheduler_cfg(call_log_dir: Path) -> Config:
    return Config(
        chat_backend="mlc",
        embedding_backend="mlc",
        mlc=LocalBackendConfig(
            chat_url=CHAT_URL,
            chat_model="Qwen-27B",
            embedding_url="",
            embedding_model="",
        ),
        gateway=GatewayConfig(port=8401, host="127.0.0.1"),
        scheduler=SchedulerConfig(policy="fcfs", call_log_dir=str(call_log_dir)),
    )


async def test_the_playground_reads_the_scheduler_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support.gateway import playground

    monkeypatch.setattr(playground, "_DEFAULT_MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(
        SmokePlayground, "_build_llm_provider", lambda self, *args: FakeLLM()
    )
    app = FastAPI()
    webapp.mount(app, _scheduler_cfg(tmp_path / "scheduler"))
    manager = app.state.smoke_playground
    try:
        assert manager._gate.policy == "fcfs"
        session_id = await create(manager)
        assert isinstance(manager._sessions[session_id].agent._llm, GatedLLM)
        response = await manager.run(session_id, "hello")
        assert response["answer"] == "Hello"
        path = manager._gate.call_log_path
        assert path is not None and path.is_file()
        assert path.is_relative_to(tmp_path / "scheduler" / "runs")
    finally:
        await manager.close_all()


async def test_a_closed_chat_is_forgotten_from_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _ = make_manager(tmp_path, monkeypatch, policy="fcfs", n_srv=2)
    try:
        session_id = await create(manager)
        assert manager.set_level(session_id, 3)["level"] == 3
        assert (await manager.run(session_id, "hello"))["answer"] == "Hello"
        assert any(t.task_id == session_id for t in manager._gate._state.tasks)
        assert await manager.close(session_id)
        assert not any(t.task_id == session_id for t in manager._gate._state.tasks)
        assert manager._gate.level(session_id) == 2, "the level map is dropped too"
    finally:
        await manager.close_all()


async def test_call_log_false_disables_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support.gateway import playground

    monkeypatch.setattr(playground, "_DEFAULT_MEMORY_DIR", tmp_path / "memory")
    cfg = _scheduler_cfg(tmp_path / "scheduler")
    cfg.scheduler.call_log = False
    app = FastAPI()
    webapp.mount(app, cfg)
    manager = app.state.smoke_playground
    try:
        assert manager._gate.call_log_path is None
        assert not (tmp_path / "scheduler").exists()
    finally:
        await manager.close_all()


async def test_the_call_log_is_under_the_configured_dir_and_is_pruned(
    tmp_path: Path,
) -> None:
    from localmelo.support.gateway.chat_gate import ChatGate

    runs = tmp_path / "runs"
    old = runs / "20200101T000000"
    old.mkdir(parents=True)
    os.utime(old, (time.time() - 31 * 86400, time.time() - 31 * 86400))
    keep = runs / "not-a-run"
    keep.mkdir()
    os.utime(keep, (time.time() - 31 * 86400, time.time() - 31 * 86400))
    gate = ChatGate(policy="off", call_log_dir=tmp_path, retention_days=30)
    try:
        assert not old.exists()
        assert keep.exists(), "Only timestamp-directories are pruned"
        path = gate.call_log_path
        assert path is not None
        assert path.parent.parent == runs
        assert path.parent.name[:2] == "20"
        assert (path.stat().st_mode & 0o777) == 0o600
        assert (path.parent.stat().st_mode & 0o777) == 0o700
        call = gate.submit("a")
        gate.finish(call)
        assert path.read_text(encoding="utf-8").strip()
    finally:
        await gate.close()
