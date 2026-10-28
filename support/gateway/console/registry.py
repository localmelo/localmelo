"""The console's open sessions: build an Agent, keep it, close it, resume it.

One :class:`OpenSession` holds everything that outlives an Agent: the
runner, the approval policy and its grants, the kept conversation, and the
resume result the next ``202`` reports. :class:`ConsoleSessions` opens a
session on demand, closes the least recently used idle Agent when too many
are open, closes idle Agents on a sweep, and closes everything on shutdown.

No turn starts here in this PR: ``create``, ``start_turn``, ``retry`` and
``change_workspace`` are PR 3. :meth:`ConsoleSessions.open` is what the
turn routes will call under the session's lock: the caller holds
:meth:`ConsoleSessions.lock` and ``open`` refuses a call without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import Message
from localmelo.support.sessions import display
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.approvals import ApprovalBroker, RememberingPolicy
from localmelo.support.sessions.runner import (
    DEFAULT_RESUME_MODE,
    ResumeInfo,
    SessionRunner,
)
from localmelo.support.sessions.store import (
    SessionBusyError,
    SessionStore,
    SessionStoreError,
    process_holder,
)

from .errors import ApiError

if TYPE_CHECKING:
    from localmelo.support.config import Config

logger = logging.getLogger(__name__)

# Seconds an idle Agent is kept before the sweep closes it.
IDLE_CLOSE_S: Final = 900.0
# Agents open at once; opening one more closes the least recently used idle
# Agent of any kind first.
MAX_OPEN: Final = 8
# Seconds the shutdown may take to close every runner.
CLOSE_ALL_TIMEOUT_S: Final = 10.0
# Of that, the reserve an Agent's begin_exit leaves for its own save.
EXIT_SAVE_RESERVE_S: Final = 1.0
# Seconds between idle sweeps.
SWEEP_INTERVAL_S: Final = 60.0
# Where a session whose row has no workspace starts, until the settings
# gain their [gateway] workspace key (PR 4).
DEFAULT_WORKSPACE: Final = "~/localmelo-workspace"

SettingsSource = Callable[[], "Config"]
AgentBuilder = Callable[["Config", str, str | None], Awaitable[Agent]]


async def default_builder(
    settings: Config, workspace: str, mcp_config: str | None
) -> Agent:
    """A started Agent for *workspace*, from the shared session builder.

    MCP is deferred to U4, so *mcp_config* is not read yet; a caller that
    needs it passes its own builder in :class:`ConsoleOptions`.
    """

    return await build_session_agent(
        cfg=settings,
        base_url=None,
        model=None,
        workspace=workspace,
        options=AgentOptions(),
    )


@dataclass(eq=False)
class OpenSession:
    """One session the console keeps open.

    ``agent`` is ``None`` after a close; then ``kept`` holds the
    conversation it had, and :meth:`ConsoleSessions.open` loads it into the
    next Agent. ``resumed`` is the last ``resume()``'s result, for the next
    ``202``; the route that reports it clears it.
    """

    session_id: str
    runner: SessionRunner
    policy: RememberingPolicy
    granted_at: dict[str, float]
    agent: Agent | None
    kept: list[Message] | None = None
    mcp_error: str = ""
    built_with_mcp: bool = False
    resumed: ResumeInfo | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = 0.0


class ConsoleSessions:
    """The console's sessions, opened on demand over one gateway store.

    *settings* is read again on every open, so a settings change reaches
    the next Agent. *builder* returns a started Agent and must close it and
    raise on failure; *idle_close_s* and *max_open* are the bounds of
    :data:`IDLE_CLOSE_S` and :data:`MAX_OPEN`. *clock* replaces time in
    tests.
    """

    def __init__(
        self,
        store: SessionStore,
        broker: ApprovalBroker,
        settings: SettingsSource,
        *,
        builder: AgentBuilder | None = None,
        idle_close_s: float = IDLE_CLOSE_S,
        max_open: int = MAX_OPEN,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if store.writer != "gateway":
            raise ValueError("the console opens the gateway's store")
        self._store = store
        self._broker = broker
        self._settings = settings
        self._builder: AgentBuilder = builder or default_builder
        self._idle_close_s = idle_close_s
        self._max_open = max_open
        self._clock = clock
        self._default_workspace = os.path.expanduser(DEFAULT_WORKSPACE)
        self._holder = process_holder("gateway")
        self._sessions: dict[str, OpenSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    # ── Reading ──

    @property
    def store(self) -> SessionStore:
        return self._store

    def get(self, session_id: str) -> OpenSession | None:
        return self._sessions.get(session_id)

    def entries(self) -> list[OpenSession]:
        return list(self._sessions.values())

    def lock(self, session_id: str) -> asyncio.Lock:
        """The lock that serializes this session's opens and turns."""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def find_turn(self, turn_id: str) -> OpenSession | None:
        """The open session whose running turn is *turn_id*, or ``None``."""
        for entry in self._sessions.values():
            current = entry.runner.current_turn
            if current is not None and current.id == turn_id:
                return entry
        return None

    # ── Open and keep ──

    async def open(self, session_id: str) -> OpenSession:
        """Open *session_id*, building and resuming its Agent when needed.

        The caller holds :meth:`lock` for *session_id*: the turn routes will
        open a session under it, and the registry never takes it twice. A
        session already open with an Agent is returned; a session whose
        Agent was kept gets a new Agent and its conversation; a session not
        loaded yet gets a runner, a policy and a resume. A workspace that no
        longer exists gives 400, a builder that fails 502, and an unknown
        session 404.
        """
        if self._closed:
            raise RuntimeError("The console's sessions are closed")
        if not self.lock(session_id).locked():
            raise RuntimeError(
                "ConsoleSessions.open() must be called with the session's lock held"
            )
        return await self._open_locked(session_id)

    async def _open_locked(self, session_id: str) -> OpenSession:
        if self._closed:
            raise RuntimeError("The console's sessions are closed")
        entry = self._sessions.get(session_id)
        if entry is not None and entry.agent is not None:
            entry.last_used = self._clock()
            return entry

        try:
            info = await self._store.get_session(session_id)
        except SessionStoreError as exc:
            raise ApiError(
                503, "store_unavailable", display.home_label(str(exc))
            ) from exc
        if info is None:
            raise ApiError(404, "not_found", "Session not found.")
        workspace = info.workspace
        if workspace is None:
            workspace = self._default_workspace
            try:
                os.makedirs(workspace, mode=0o700, exist_ok=True)
                await self._store.update_session(session_id, workspace=workspace)
            except SessionStoreError as exc:
                raise ApiError(503, "store_unavailable", str(exc)) from exc
            except OSError as exc:
                raise ApiError(
                    400,
                    "invalid_workspace",
                    f"Could not create the default workspace "
                    f"{display.home_label(workspace)}: {exc}",
                ) from exc
        if not os.path.isdir(workspace):
            raise ApiError(
                400,
                "invalid_workspace",
                f"The workspace {display.home_label(workspace)} no longer exists. "
                "Change it in the header.",
            )

        await self._make_room(session_id)
        if self._closed:
            raise RuntimeError("The console's sessions are closed")
        is_new = entry is None
        settings = self._settings()
        mcp_config = getattr(settings, "mcp_config", "") or None
        try:
            agent = await self._builder(settings, workspace, mcp_config)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ApiError(
                502,
                "agent_start_failed",
                str(exc) or type(exc).__name__,
            ) from exc

        if self._closed:
            with contextlib.suppress(Exception):
                await agent.close()
            raise RuntimeError("The console's sessions are closed")
        if entry is None:
            runner = SessionRunner(
                agent,
                self._store,
                session_id,
                holder=self._holder,
                resume_mode=DEFAULT_RESUME_MODE,
            )
            policy = RememberingPolicy(
                ask=runner.observe(self._broker.asker(session_id))
            )
            entry = OpenSession(
                session_id=session_id,
                runner=runner,
                policy=policy,
                granted_at={},
                agent=agent,
                built_with_mcp=mcp_config is not None,
                lock=self.lock(session_id),
                last_used=self._clock(),
            )
            self._sessions[session_id] = entry
        else:
            try:
                entry.runner.replace_agent(agent)
            except SessionBusyError as exc:
                with contextlib.suppress(Exception):
                    await agent.close()
                raise ApiError(409, "session_busy", str(exc)) from exc
            entry.agent = agent
            entry.built_with_mcp = mcp_config is not None
            entry.last_used = self._clock()

        try:
            await self._load(entry)
        except BaseException as exc:
            # The registry must not keep a half-opened Agent. A new session
            # is dropped; a session that was merely paused keeps its runner,
            # policy and kept conversation for the next open.
            if entry.agent is agent:
                entry.agent = None
            with contextlib.suppress(Exception):
                await agent.close()
            if is_new:
                self._sessions.pop(session_id, None)
            if isinstance(exc, SessionStoreError):
                raise ApiError(
                    503, "store_unavailable", display.home_label(str(exc))
                ) from exc
            raise
        return entry

    async def _load(self, entry: OpenSession) -> None:
        """Load the kept conversation, or resume from the store."""
        assert entry.agent is not None
        entry.resumed = None
        if entry.kept is not None:
            kept = entry.kept
            entry.agent.load_conversation(kept)
            entry.kept = None
        else:
            await entry.runner.resume()
            entry.resumed = entry.runner.last_resume

    async def keep(self, session_id: str) -> None:
        """Close *session_id*'s Agent, keeping its conversation.

        409 while a turn runs; nothing happens for a session that is not
        open.
        """
        async with self.lock(session_id):
            entry = self._sessions.get(session_id)
            if entry is None:
                return
            await self._keep_locked(entry)

    async def _keep_locked(self, entry: OpenSession) -> None:
        if entry.agent is None:
            return
        if entry.runner.busy:
            raise ApiError(
                409,
                "session_busy",
                "A turn is already running in this session.",
            )
        agent = entry.agent
        entry.kept = list(agent.conversation())
        try:
            await agent.close()
        finally:
            # Even a failed close must not hand the same Agent back to the
            # next open: the kept conversation is what it loads.
            entry.agent = None
        entry.last_used = self._clock()

    async def _make_room(self, session_id: str) -> None:
        """Close the least recently used idle Agent when at the cap."""
        if self._max_open <= 0:
            return
        while (
            sum(1 for entry in self._sessions.values() if entry.agent is not None)
            >= self._max_open
        ):
            candidates = [
                entry
                for entry in self._sessions.values()
                if entry.agent is not None
                and not entry.runner.busy
                and not entry.lock.locked()
                and entry.session_id != session_id
            ]
            if not candidates:
                return
            oldest = min(candidates, key=lambda entry: entry.last_used)
            async with oldest.lock:
                if oldest.agent is not None and not oldest.runner.busy:
                    await self._keep_locked(oldest)

    # ── Sweeps ──

    async def close_idle(self) -> int:
        """Close every idle Agent older than ``idle_close_s``; return how many.

        An Agent with a turn running or an approval pending is left alone.
        """
        now = self._clock()
        closed = 0
        for entry in list(self._sessions.values()):
            if entry.agent is None or entry.runner.busy:
                continue
            # The sweep frees MCP servers' processes; an Agent without MCP
            # is kept and only the open cap closes it.
            if not entry.built_with_mcp:
                continue
            if self._broker.pending(entry.session_id):
                continue
            if now - entry.last_used < self._idle_close_s:
                continue
            async with entry.lock:
                if (
                    entry.agent is not None
                    and not entry.runner.busy
                    and not self._broker.pending(entry.session_id)
                ):
                    await self._keep_locked(entry)
                    closed += 1
        return closed

    async def settings_changed(self) -> None:
        """Close every idle Agent at once, keeping each conversation.

        A busy Agent is closed after its turn; PR 3 adds that path.
        """
        for entry in list(self._sessions.values()):
            if entry.agent is None or entry.runner.busy:
                continue
            async with entry.lock:
                if (
                    entry.agent is not None
                    and not entry.runner.busy
                    and not self._broker.pending(entry.session_id)
                ):
                    await self._keep_locked(entry)

    # ── Delete and shutdown ──

    async def delete(self, session_id: str) -> None:
        """Close the session's runner and delete its row; 404 when unknown."""
        removed = False
        try:
            async with self.lock(session_id):
                entry = self._sessions.get(session_id)
                if entry is not None:
                    await entry.runner.close(cause="closed")
                    if entry.agent is not None:
                        agent, entry.agent = entry.agent, None
                        with contextlib.suppress(Exception):
                            await agent.close()
                    self._sessions.pop(session_id, None)
                try:
                    deleted = await self._store.delete_session(session_id)
                except SessionBusyError as exc:
                    self._sessions.pop(session_id, None)
                    raise ApiError(409, "session_busy", str(exc)) from exc
                if not deleted:
                    raise ApiError(404, "not_found", "Session not found.")
                removed = True
        finally:
            # Only a row that is really gone may lose its lock: a 409/503
            # leaves the row, and the same lock must still serialize it.
            if removed:
                self._locks.pop(session_id, None)

    async def close_all(self) -> None:
        """Stop every turn and close every Agent within the shutdown budget."""
        self._closed = True
        entries = list(self._sessions.values())
        deadline = max(CLOSE_ALL_TIMEOUT_S - EXIT_SAVE_RESERVE_S, 0.1)
        for entry in entries:
            if entry.agent is not None:
                with contextlib.suppress(Exception):
                    entry.agent.begin_exit("Session closed", within_s=deadline)
        try:
            async with asyncio.timeout(CLOSE_ALL_TIMEOUT_S):
                await asyncio.gather(
                    *(entry.runner.close(cause="shutdown") for entry in entries),
                    return_exceptions=True,
                )
        except TimeoutError:
            logger.warning(
                "Closing the console's runners took over %g s", CLOSE_ALL_TIMEOUT_S
            )
        self._broker.withdraw_all(reason="The gateway is stopping")
        for entry in entries:
            if entry.agent is None:
                continue
            with contextlib.suppress(Exception):
                entry.kept = list(entry.agent.conversation())
            with contextlib.suppress(Exception):
                await entry.agent.close()
            entry.agent = None
        self._sessions.clear()
