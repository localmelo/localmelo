"""The console's registry: open, keep, idle close, delete and shutdown."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from localmelo.melo.schema import Message
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.errors import ApiError
from localmelo.support.gateway.console.registry import ConsoleSessions, SettingsSource
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStoreError

from .conftest import FakeAgent, FakeBuilder, make_cfg, open_session, seed_turn


def registry(
    console: Console,
    builder: FakeBuilder,
    *,
    max_open: int = 8,
    idle_close_s: float = 900.0,
    mcp_config: str | None = None,
) -> ConsoleSessions:
    settings: SettingsSource = (
        cast(SettingsSource, lambda: SimpleNamespace(mcp_config=mcp_config))
        if mcp_config
        else make_cfg
    )
    return ConsoleSessions(
        console.store,
        console.broker,
        settings,
        builder=builder,
        max_open=max_open,
        idle_close_s=idle_close_s,
    )


async def test_open_builds_and_resumes(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    entry = await open_session(console.sessions, turn.session_id)
    assert builder.calls
    agent = cast(FakeAgent, entry.agent)
    assert agent.workspace_root == workspace
    assert entry.resumed is not None
    # The pairs arm loaded the turn's user text and answer.
    assert [message.role for message in agent.conversation()] == [
        "user",
        "assistant",
    ]


async def test_keep_then_reopen_loads_the_conversation(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    entry = await open_session(console.sessions, turn.session_id)
    kept = [Message(role="user", content="kept")]
    first = cast(FakeAgent, entry.agent)
    first.messages = kept
    await console.sessions.keep(turn.session_id)
    assert entry.agent is None
    assert first.closed
    assert entry.kept == kept

    reopened = await open_session(console.sessions, turn.session_id)
    assert reopened is entry
    second = cast(FakeAgent, reopened.agent)
    assert second is not first
    assert second.conversation() == kept
    assert len(builder.calls) == 2


async def test_the_cap_closes_the_least_recently_used(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    first = await seed_turn(console.store, workspace, title="one")
    second = await seed_turn(console.store, workspace, title="two")
    reg = registry(console, builder, max_open=1)
    await open_session(reg, first.session_id)
    first_agent = cast(FakeAgent, reg.get(first.session_id).agent)  # type: ignore[union-attr]
    await open_session(reg, second.session_id)
    assert reg.get(first.session_id).agent is None  # type: ignore[union-attr]
    assert first_agent.closed
    assert reg.get(first.session_id).kept is not None  # type: ignore[union-attr]
    assert reg.get(second.session_id).agent is not None  # type: ignore[union-attr]


async def test_the_idle_sweep_keeps_the_conversation(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    reg = registry(console, builder, idle_close_s=0)
    entry = await open_session(reg, turn.session_id)
    # Only an Agent a builder really gave MCP tools is swept; the flag is
    # what that builder will report when MCP lands (PR 3+).
    entry.built_with_mcp = True
    cast(FakeAgent, entry.agent).messages = [Message(role="user", content="idle")]
    entry.last_used = 0
    assert await reg.close_idle() == 1
    assert entry.agent is None
    assert entry.kept is not None


async def test_an_mcp_config_alone_does_not_mark_the_agent(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    reg = registry(console, builder, idle_close_s=0, mcp_config="/tmp/mcp.json")
    entry = await open_session(reg, turn.session_id)
    # default_builder does not read mcp_config and builds no MCP tools, so
    # the sweep must not close this Agent as if it held MCP processes.
    assert entry.built_with_mcp is False
    entry.last_used = 0
    assert await reg.close_idle() == 0
    assert entry.agent is not None


async def test_the_idle_sweep_leaves_an_agent_without_mcp(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    reg = registry(console, builder, idle_close_s=0)
    entry = await open_session(reg, turn.session_id)
    entry.last_used = 0
    assert await reg.close_idle() == 0
    assert entry.agent is not None


async def test_settings_changed_closes_idle_agents(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    first = await seed_turn(console.store, workspace, title="one")
    second = await seed_turn(console.store, workspace, title="two")
    reg = registry(console, builder)
    one = await open_session(reg, first.session_id)
    two = await open_session(reg, second.session_id)
    await reg.settings_changed()
    assert one.agent is None and two.agent is None
    assert one.kept is not None and two.kept is not None


async def test_delete_removes_the_row_and_closes_the_agent(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    reg = registry(console, builder)
    entry = await open_session(reg, turn.session_id)
    agent = cast(FakeAgent, entry.agent)
    await reg.delete(turn.session_id)
    assert reg.get(turn.session_id) is None
    assert agent.closed
    assert await console.store.get_session(turn.session_id) is None


async def test_open_returns_the_same_entry_without_a_rebuild(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    first = await open_session(console.sessions, turn.session_id)
    second = await open_session(console.sessions, turn.session_id)
    assert second is first
    assert len(builder.calls) == 1


async def test_delete_a_running_session_is_409(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace, status="running")
    reg = registry(console, builder)
    await open_session(reg, turn.session_id)
    lock = reg.lock(turn.session_id)
    with pytest.raises(ApiError) as info:
        await reg.delete(turn.session_id)
    assert info.value.status_code == 409
    assert info.value.code == "session_busy"
    assert reg.get(turn.session_id) is None
    # The row survives, so its lock must survive as the same object.
    assert reg.lock(turn.session_id) is lock


async def test_delete_an_unknown_session_is_404(console: Console) -> None:
    with pytest.raises(ApiError) as info:
        await console.sessions.delete("0123456789ab")
    assert info.value.status_code == 404


async def test_close_all_closes_every_agent(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    first = await seed_turn(console.store, workspace, title="one")
    second = await seed_turn(console.store, workspace, title="two")
    reg = registry(console, builder)
    one = await open_session(reg, first.session_id)
    two = await open_session(reg, second.session_id)
    agents = [cast(FakeAgent, one.agent), cast(FakeAgent, two.agent)]
    await reg.close_all()
    assert reg.entries() == []
    for agent in agents:
        assert agent.closed
        assert agent.exited is not None


async def test_open_requires_the_session_lock(console: Console, workspace: str) -> None:
    turn = await seed_turn(console.store, workspace)
    with pytest.raises(RuntimeError, match="lock held"):
        await console.sessions.open(turn.session_id)


async def test_open_refuses_a_lock_another_task_holds(
    console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    holding = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        async with console.sessions.lock(turn.session_id):
            holding.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await holding.wait()
    try:
        # A lock another task holds is not the caller's to open under.
        with pytest.raises(RuntimeError, match="lock held"):
            await console.sessions.open(turn.session_id)
    finally:
        release.set()
        await holder


async def test_open_maps_a_store_read_failure(
    console: Console, workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    turn = await seed_turn(console.store, workspace)

    async def broken(session_id: str) -> Any:
        raise SessionStoreError("read failed")

    monkeypatch.setattr(console.store, "get_session", broken)
    with pytest.raises(ApiError) as info:
        await open_session(console.sessions, turn.session_id)
    assert info.value.status_code == 503
    assert info.value.code == "store_unavailable"


async def test_open_maps_a_resume_failure(
    console: Console,
    builder: FakeBuilder,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = await seed_turn(console.store, workspace)

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise SessionStoreError("resume failed")

    monkeypatch.setattr(console.store, "list_turns", broken)
    with pytest.raises(ApiError) as info:
        await open_session(console.sessions, turn.session_id)
    assert info.value.status_code == 503
    assert console.sessions.get(turn.session_id) is None
    assert len(builder.calls) == 1


async def test_open_under_its_own_lock_does_not_deadlock(
    console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    async with console.sessions.lock(turn.session_id):
        entry = await asyncio.wait_for(
            console.sessions.open(turn.session_id), timeout=1
        )
    assert entry.session_id == turn.session_id


async def test_close_all_closes_the_registry(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    reg = registry(console, builder)
    await open_session(reg, turn.session_id)
    await reg.close_all()
    with pytest.raises(RuntimeError):
        await open_session(reg, turn.session_id)


async def test_open_during_a_close_all_load_raises_and_closes(
    console: Console,
    builder: FakeBuilder,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = await seed_turn(console.store, workspace)
    loaded = asyncio.Event()
    release = asyncio.Event()
    resume = SessionRunner.resume

    async def blocked(self: SessionRunner) -> Any:
        loaded.set()
        await release.wait()
        return await resume(self)

    monkeypatch.setattr(SessionRunner, "resume", blocked)
    opening = asyncio.create_task(open_session(console.sessions, turn.session_id))
    await loaded.wait()
    await console.sessions.close_all()
    release.set()
    with pytest.raises(RuntimeError):
        await opening
    assert builder.agents
    assert all(agent.closed for agent in builder.agents)
    assert console.sessions.get(turn.session_id) is None


async def test_a_failed_keep_leaves_no_agent_to_reuse(
    console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    reg = registry(console, FakeBuilder(close_fails=True))
    entry = await open_session(reg, turn.session_id)
    assert entry.agent is not None
    with pytest.raises(RuntimeError):
        await reg.keep(turn.session_id)
    assert entry.agent is None
    assert entry.kept is not None


async def test_a_builder_that_fails_is_502(
    console: Console, builder: FakeBuilder, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    builder.fail = True
    with pytest.raises(ApiError) as info:
        await open_session(console.sessions, turn.session_id)
    assert info.value.status_code == 502
    assert info.value.code == "agent_start_failed"
    assert "no model server" in info.value.message
    assert console.sessions.get(turn.session_id) is None


async def test_a_missing_workspace_is_400(console: Console, tmp_path: Path) -> None:
    turn = await seed_turn(console.store, str(tmp_path / "gone"))
    with pytest.raises(ApiError) as info:
        await open_session(console.sessions, turn.session_id)
    assert info.value.status_code == 400
    assert info.value.code == "invalid_workspace"
    assert "no longer exists" in info.value.message


async def test_an_unknown_session_is_404(console: Console) -> None:
    with pytest.raises(ApiError) as info:
        await open_session(console.sessions, "0123456789ab")
    assert info.value.status_code == 404


async def test_a_session_without_a_workspace_gets_the_default(
    console: Console, home: Path, workspace: str
) -> None:
    turn = await console.store.begin_turn(
        None, "hello", holder="h:1:x", new_session=NewSession()
    )
    entry = await open_session(console.sessions, turn.session_id)
    assert entry.agent is not None
    info = await console.store.get_session(turn.session_id)
    assert info is not None
    default = home / "localmelo-workspace"
    assert info.workspace == str(default)
    assert default.is_dir()
    assert workspace != str(default)


async def test_configured_gateway_workspace_is_used_by_create_and_open(
    console: Console, builder: FakeBuilder, home: Path
) -> None:
    configured = home / "configured-ws"
    console.cfg.gateway.workspace = "~/configured-ws"

    created, _ = await console.sessions.create(None)
    created_info = await console.store.get_session(created.session_id)
    assert created_info is not None
    assert created_info.workspace == str(configured.resolve())
    assert configured.is_dir()

    updated = home / "updated-ws"
    console.cfg.gateway.workspace = str(updated)
    turn = await console.store.begin_turn(
        None, "hello", holder="h:1:x", new_session=NewSession()
    )
    entry = await open_session(console.sessions, turn.session_id)
    assert entry.agent is not None
    opened_info = await console.store.get_session(turn.session_id)
    assert opened_info is not None
    assert opened_info.workspace == str(updated.resolve())
    assert updated.is_dir()


async def test_a_create_in_flight_at_shutdown_registers_nothing(
    console: Console, builder: FakeBuilder, tmp_path: Path
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Slow:
        """A builder the test holds until the shutdown has run."""

        async def __call__(
            self, settings: Any, workspace: str, mcp_config: str | None, session_id: str
        ) -> Any:
            entered.set()
            await release.wait()
            return await builder(settings, workspace, mcp_config, session_id)

    reg = ConsoleSessions(console.store, console.broker, make_cfg, builder=Slow())
    creating = asyncio.create_task(reg.create(str(tmp_path)))
    await entered.wait()
    await reg.close_all()
    release.set()
    with pytest.raises(RuntimeError):
        await creating
    assert reg.entries() == []
    assert builder.agents and all(agent.closed for agent in builder.agents)
    assert await console.store.list_sessions() == []


async def test_create_keeps_the_open_cap(
    console: Console, builder: FakeBuilder, tmp_path: Path
) -> None:
    reg = registry(console, builder, max_open=1)
    first, _ = await reg.create(str(tmp_path))
    second, _ = await reg.create(str(tmp_path))
    assert reg.get(first.session_id).agent is None  # type: ignore[union-attr]
    assert reg.get(second.session_id).agent is not None  # type: ignore[union-attr]


async def test_a_finished_create_keeps_no_key_lock(
    console: Console, builder: FakeBuilder, tmp_path: Path
) -> None:
    reg = registry(console, builder)
    await reg.create(str(tmp_path), client_request_id="8d2f6c1e")
    # A lock that stayed would hold one object per key for the process.
    assert reg._create_locks == {}


async def test_a_workspace_move_whose_save_fails_changes_nothing(
    console: Console,
    builder: FakeBuilder,
    workspace: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = await seed_turn(console.store, workspace)
    reg = registry(console, builder)
    entry = await open_session(reg, turn.session_id)
    old = cast(FakeAgent, entry.agent)
    other = tmp_path / "other"
    other.mkdir()

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise SessionStoreError("write failed")

    monkeypatch.setattr(console.store, "update_session", broken)
    with pytest.raises(SessionStoreError):
        await reg.change_workspace(turn.session_id, str(other))
    info = await console.store.get_session(turn.session_id)
    assert info is not None and info.workspace == workspace
    assert entry.agent is old
    assert old.workspace_root == workspace
    assert not old.closed
    # The Agent it built for the new folder went away again.
    assert builder.agents[-1] is not old and builder.agents[-1].closed
