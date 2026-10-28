"""Tests for the gateway session manager.

Uses a fake Agent to avoid real model dependencies.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway.session import (
    Session,
    SessionBusyError,
    SessionManager,
)

# ── Fake Agent ──


class FakeAgent:
    """Minimal Agent stand-in for testing."""

    def __init__(self) -> None:
        self.closed = False
        self.run_count = 0
        self._run_delay = 0.0

    async def run(self, query: str) -> str:
        self.run_count += 1
        if self._run_delay:
            await asyncio.sleep(self._run_delay)
        return f"echo: {query}"

    async def close(self) -> None:
        self.closed = True


class FakeSessionManager(SessionManager):
    """SessionManager that produces FakeAgents instead of real ones."""

    def _create_agent(self) -> FakeAgent:  # type: ignore[override]
        return FakeAgent()


# ── Session tests ──


async def test_queued_start_blocks_close_and_eviction() -> None:
    mgr = FakeSessionManager(idle_ttl=0)
    session = await mgr.create("queued")
    await mgr._creation_lock.acquire()
    task = asyncio.create_task(mgr.get_or_create("queued"))
    await asyncio.sleep(0)
    try:
        with pytest.raises(SessionBusyError):
            await mgr.close("queued")
        await mgr._evict_idle()
        assert mgr.get("queued") is session
    finally:
        mgr._creation_lock.release()
    assert await task is session
    await mgr.close("queued")


async def test_first_queued_start_blocks_close() -> None:
    mgr = FakeSessionManager()
    await mgr._creation_lock.acquire()
    task = asyncio.create_task(mgr.get_or_create("new"))
    await asyncio.sleep(0)
    try:
        with pytest.raises(SessionBusyError):
            await mgr.close("new")
        assert mgr.get("new") is None
    finally:
        mgr._creation_lock.release()
    session = await task
    assert mgr.get("new") is session
    assert await mgr.close("new")


async def test_close_removes_before_awaiting_agent() -> None:
    mgr = FakeSessionManager()
    session = await mgr.create("closing")
    entered, release = asyncio.Event(), asyncio.Event()

    async def close() -> None:
        entered.set()
        await release.wait()

    session.agent.close = close  # type: ignore[method-assign]
    first = asyncio.create_task(mgr.close("closing"))
    await entered.wait()
    assert not await mgr.close("closing")
    with pytest.raises(SessionBusyError):
        await mgr.get_or_create("closing")
    with pytest.raises(SessionBusyError):
        await mgr.create("closing")
    release.set()
    assert await first


# ── Session tests ──


class TestSession:
    def test_touch_updates_last_active(self) -> None:
        s = Session(agent=FakeAgent())  # type: ignore[arg-type]
        old = s.last_active
        time.sleep(0.01)
        s.touch()
        assert s.last_active > old

    def test_idle_seconds(self) -> None:
        s = Session(agent=FakeAgent())  # type: ignore[arg-type]
        assert s.idle_seconds >= 0
        assert s.idle_seconds < 1.0

    def test_is_busy_default_false(self) -> None:
        s = Session(agent=FakeAgent())  # type: ignore[arg-type]
        assert not s.is_busy


# ── SessionManager tests ──


class TestSessionManager:
    @pytest.fixture()
    def mgr(self) -> FakeSessionManager:
        return FakeSessionManager(max_sessions=3, idle_ttl=0.05)

    @pytest.mark.asyncio
    async def test_create_and_get(self, mgr: FakeSessionManager) -> None:
        s = await mgr.create("test-1")
        assert s.session_id == "test-1"
        assert mgr.get("test-1") is s

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mgr: FakeSessionManager) -> None:
        assert mgr.get("nope") is None

    @pytest.mark.asyncio
    async def test_get_or_create_reuses(self, mgr: FakeSessionManager) -> None:
        s1 = await mgr.get_or_create("abc")
        s2 = await mgr.get_or_create("abc")
        assert s1 is s2

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, mgr: FakeSessionManager) -> None:
        s1 = await mgr.get_or_create("a")
        s2 = await mgr.get_or_create("b")
        assert s1.session_id != s2.session_id

    @pytest.mark.asyncio
    async def test_close_session(self, mgr: FakeSessionManager) -> None:
        s = await mgr.create("x")
        agent: FakeAgent = s.agent  # type: ignore[assignment]
        assert await mgr.close("x")
        assert agent.closed
        assert mgr.get("x") is None

    @pytest.mark.asyncio
    async def test_close_nonexistent(self, mgr: FakeSessionManager) -> None:
        assert not await mgr.close("nope")

    @pytest.mark.asyncio
    async def test_close_busy_session_refuses_and_leaves_it(
        self, mgr: FakeSessionManager
    ) -> None:
        s = await mgr.create("busy")
        agent: FakeAgent = s.agent  # type: ignore[assignment]
        await s._lock.acquire()
        try:
            with pytest.raises(SessionBusyError, match="busy"):
                await mgr.close("busy")
            assert not agent.closed
            assert mgr.get("busy") is s
            assert s.status == "idle"
        finally:
            s._lock.release()
        assert await mgr.close("busy")
        assert agent.closed

    @pytest.mark.asyncio
    async def test_list_sessions(self, mgr: FakeSessionManager) -> None:
        await mgr.create("s1")
        await mgr.create("s2")
        listing = mgr.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert ids == {"s1", "s2"}

    @pytest.mark.asyncio
    async def test_list_excludes_closed(self, mgr: FakeSessionManager) -> None:
        await mgr.create("s1")
        await mgr.create("s2")
        await mgr.close("s1")
        listing = mgr.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert ids == {"s2"}

    @pytest.mark.asyncio
    async def test_idle_eviction(self, mgr: FakeSessionManager) -> None:
        s = await mgr.create("old")
        agent: FakeAgent = s.agent  # type: ignore[assignment]
        # Backdate last_active to trigger TTL
        s.last_active = time.time() - 1.0
        # Creating a new session triggers eviction
        await mgr.create("new")
        assert mgr.get("old") is None
        assert agent.closed

    @pytest.mark.asyncio
    async def test_max_eviction(self, mgr: FakeSessionManager) -> None:
        # Fill to max (3)
        await mgr.create("a")
        await mgr.create("b")
        await mgr.create("c")
        # Backdate 'a' so it's oldest
        mgr.get("a").last_active = time.time() - 100  # type: ignore[union-attr]
        # 4th session triggers oldest eviction
        await mgr.create("d")
        assert mgr.get("a") is None
        assert mgr.get("d") is not None

    @pytest.mark.asyncio
    async def test_busy_session_not_evicted(self, mgr: FakeSessionManager) -> None:
        s = await mgr.create("busy")
        s.last_active = time.time() - 1.0  # past TTL
        # Simulate busy by acquiring lock
        await s._lock.acquire()
        try:
            await mgr._evict_idle()
            # Should NOT be evicted because lock is held
            assert mgr.get("busy") is not None
        finally:
            s._lock.release()

    @pytest.mark.asyncio
    async def test_close_all(self, mgr: FakeSessionManager) -> None:
        s1 = await mgr.create("a")
        s2 = await mgr.create("b")
        agent1: FakeAgent = s1.agent  # type: ignore[assignment]
        agent2: FakeAgent = s2.agent  # type: ignore[assignment]
        await mgr.close_all()
        assert agent1.closed
        assert agent2.closed
        assert mgr.list_sessions() == []

    @pytest.mark.asyncio
    async def test_close_all_closes_busy_sessions(
        self, mgr: FakeSessionManager
    ) -> None:
        """Shutdown closes every session, a running task's included."""
        s = await mgr.create("busy")
        agent: FakeAgent = s.agent  # type: ignore[assignment]
        await s._lock.acquire()
        try:
            await mgr.close_all()
        finally:
            s._lock.release()
        assert agent.closed
        assert mgr.list_sessions() == []


# ── Per-session serialization test ──


class TestTheAgentsASessionBuilds:
    @pytest.mark.asyncio
    async def test_a_gateway_session_registers_no_shell_exec(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="none",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:9/v1", chat_model="stub-model"
            ),
        )
        manager = SessionManager(config=cfg)
        agent = manager._create_agent()
        try:
            names = {tool.name for tool in agent.list_tools()}  # type: ignore[attr-defined]
            assert "shell_exec" not in names
            assert {"file_read", "file_write", "python_exec"} <= names
        finally:
            await agent.close()


class TestSessionSerialization:
    @pytest.mark.asyncio
    async def test_concurrent_runs_serialize(self) -> None:
        """Two concurrent runs on the same session should not overlap."""
        mgr = FakeSessionManager(max_sessions=10, idle_ttl=3600.0)
        s = await mgr.create("serial")
        agent: FakeAgent = s.agent  # type: ignore[assignment]
        agent._run_delay = 0.05

        order: list[str] = []

        async def run_task(label: str) -> None:
            async with s._lock:
                order.append(f"{label}-start")
                await agent.run("test")
                order.append(f"{label}-end")

        await asyncio.gather(run_task("A"), run_task("B"))

        # Serialized: first task must fully complete before second starts
        assert order[0].endswith("-start")
        assert order[1].endswith("-end")
        assert order[2].endswith("-start")
        assert order[3].endswith("-end")
        # Both ran
        assert agent.run_count == 2

    @pytest.mark.asyncio
    async def test_is_busy_during_lock(self) -> None:
        s = Session(agent=FakeAgent())  # type: ignore[arg-type]
        assert not s.is_busy
        await s._lock.acquire()
        assert s.is_busy
        s._lock.release()
        assert not s.is_busy
