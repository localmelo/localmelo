"""The console's open sessions: build an Agent, keep it, close it, resume it.

One :class:`OpenSession` holds everything that outlives an Agent: the
runner, the approval policy and its grants, the kept conversation, and the
resume result the next ``202`` reports. :class:`ConsoleSessions` opens a
session on demand, closes the least recently used idle Agent when too many
are open, closes idle Agents on a sweep, and closes everything on shutdown.

:meth:`ConsoleSessions.create`, :meth:`ConsoleSessions.start_turn`,
:meth:`ConsoleSessions.retry` and :meth:`ConsoleSessions.change_workspace`
are the writes an API v2 turn makes; each takes the session's lock, and
``open`` refuses a call without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.review import offers_grant
from localmelo.melo.schema import Message
from localmelo.support.config import gateway_workspace_problem
from localmelo.support.gateway.chat_gate import ChatGate, GatedEmbedding, GatedLLM
from localmelo.support.gateway.i18n import t
from localmelo.support.gateway.task_policy import DEFAULT_WORKSPACE
from localmelo.support.sessions import display
from localmelo.support.sessions.agents import (
    build_session_agent,
    configured_session_options,
)
from localmelo.support.sessions.approvals import (
    ApprovalBroker,
    AskFunction,
    RememberingPolicy,
)
from localmelo.support.sessions.runner import (
    DEFAULT_RESUME_MODE,
    ResumeInfo,
    SessionRunner,
)
from localmelo.support.sessions.store import (
    SessionBusyError,
    SessionInfo,
    SessionStore,
    SessionStoreError,
    TurnInfo,
    process_holder,
)
from localmelo.support.sessions.submissions import check_submission

from .errors import ApiError

# What a Stop from the console says, and the cause it records.
STOP_REASON = "Stopped from the web console"

if TYPE_CHECKING:
    from localmelo.support.config import Config
    from localmelo.support.memory import ChatMemory

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

SettingsSource = Callable[[], "Config"]
# A builder gets the settings, the workspace, the MCP config and the
# session id the Agent is made for: the scheduler gate tags its calls with
# that id, so it must exist before the first model call.
AgentBuilder = Callable[["Config", str, str | None, str], Awaitable[Agent]]


class SessionLock(asyncio.Lock):
    """An :class:`asyncio.Lock` that remembers which task holds it.

    :meth:`ConsoleSessions.open` needs to know that its caller holds the
    lock, not only that someone does; a lock held by another task is not
    the caller's to open under.
    """

    owner: asyncio.Task[object] | None = None

    def __init__(self) -> None:
        super().__init__()
        self.pending = 0

    async def acquire(self) -> Literal[True]:
        self.pending += 1
        try:
            await super().acquire()
            self.owner = asyncio.current_task()
            return True
        finally:
            self.pending -= 1

    def release(self) -> None:
        self.owner = None
        super().release()


async def default_builder(
    settings: Config,
    workspace: str,
    mcp_config: str | None,
    session_id: str,
    gate: ChatGate | None = None,
    memory: ChatMemory | None = None,
) -> Agent:
    """A started Agent for *workspace*, from the shared session builder.

    MCP is deferred to U4, so *mcp_config* is not read yet; a caller that
    needs it passes its own builder in :class:`ConsoleOptions`. With *gate*,
    the providers are wrapped in :class:`GatedLLM` and
    :class:`GatedEmbedding` for *session_id*, so every call this Agent makes
    waits for its slot like a workspace chat's. *memory* gives the chat
    memory across chats (see :func:`build_session_agent`).
    """

    llm = None if gate is None else gate.client(session_id)
    return await build_session_agent(
        cfg=settings,
        base_url=None,
        model=None,
        workspace=workspace,
        options=configured_session_options(settings, AgentOptions(mode="gateway")),
        browser_driver=settings.browser.driver,
        wrap_llm=None if llm is None else (lambda inner: GatedLLM(inner, llm)),
        wrap_embedding=(
            None if llm is None else (lambda inner: GatedEmbedding(inner, llm))
        ),
        identify=True,
        memory=memory,
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
    lock: SessionLock = field(default_factory=SessionLock)
    last_used: float = 0.0
    # The main chat, which works in the default folder as it is now.
    main: bool = False


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
        front_builder: AgentBuilder | None = None,
        gate: ChatGate | None = None,
        memory: ChatMemory | None = None,
        idle_close_s: float = IDLE_CLOSE_S,
        max_open: int = MAX_OPEN,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if store.writer != "gateway":
            raise ValueError("the console opens the gateway's store")
        self._store = store
        self._broker = broker
        self._settings = settings
        self._gate = gate
        self._memory = memory
        self._builder: AgentBuilder = builder or self._default_builder
        # The main chat's own Agent (the front); None builds it as any chat.
        self._front_builder = front_builder
        self._idle_close_s = idle_close_s
        self._max_open = max_open
        self._clock = clock
        self._holder = process_holder("gateway")
        self._sessions: dict[str, OpenSession] = {}
        self._locks: dict[str, SessionLock] = {}
        # One lock per client_request_id while a keyed create runs, so a
        # retried create makes one session.
        self._create_locks: dict[str, asyncio.Lock] = {}
        # Stops that arrived for a turn a create had not registered yet.
        self._deferred_cancels: set[str] = set()
        self._closed = False

    # ── Reading ──

    async def _default_builder(
        self, settings: Config, workspace: str, mcp_config: str | None, session_id: str
    ) -> Agent:
        return await default_builder(
            settings,
            workspace,
            mcp_config,
            session_id,
            gate=self._gate,
            memory=self._memory,
        )

    @property
    def store(self) -> SessionStore:
        return self._store

    def get(self, session_id: str) -> OpenSession | None:
        return self._sessions.get(session_id)

    def entries(self) -> list[OpenSession]:
        return list(self._sessions.values())

    def lock(self, session_id: str) -> SessionLock:
        """The lock that serializes this session's opens and turns."""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = SessionLock()
            self._locks[session_id] = lock
        return lock

    def defer_cancel(self, turn_id: str) -> None:
        """Ask for a turn to stop once it is registered.

        A cancel by key can find a turn accepted while its create is still
        building it; the create applies this when it registers. A turn that
        never registers leaves its id here, so the set is bounded.
        """
        if len(self._deferred_cancels) >= 64:
            self._deferred_cancels.clear()
        self._deferred_cancels.add(turn_id)

    def _apply_deferred_cancel(self, entry: OpenSession, turn_id: str) -> None:
        if turn_id not in self._deferred_cancels:
            return
        self._deferred_cancels.discard(turn_id)
        entry.runner.cancel(STOP_REASON, cause="user_stop", turn_id=turn_id)

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
        if self.lock(session_id).owner is not asyncio.current_task():
            raise RuntimeError(
                "ConsoleSessions.open() must be called with the session's lock held"
            )
        return await self._open_locked(session_id)

    async def _open_locked(self, session_id: str) -> OpenSession:
        if self._closed:
            raise RuntimeError("The console's sessions are closed")
        entry = self._sessions.get(session_id)
        if entry is not None and entry.agent is not None:
            root = entry.agent.workspace_root
            assert root is not None
            if entry.main and not entry.runner.busy and root != self._main_folder():
                # The default folder changed: the main chat follows it with
                # a new Agent on the same conversation (after a running turn).
                await self._keep_locked(entry)
            else:
                # A folder deleted while the Agent was open is still the 400
                # a closed Agent's session gets: the turn must not start.
                self.check_workspace(root)
                entry.last_used = self._clock()
                return entry

        try:
            info = await self._store.get_session(session_id)
        except SessionStoreError as exc:
            raise ApiError(
                503, "store_unavailable", display.home_label(str(exc))
            ) from exc
        if info is None:
            raise ApiError(404, "not_found", t("errors.sessionNotFound"))
        if info.kind == "task":
            raise ApiError(409, "task_session_read_only", t("errors.taskAsChat"))
        workspace = info.workspace
        if info.kind == "front":
            # The main chat keeps no folder of its own: it works in the
            # default folder as the settings name it now (#321). Its row
            # only mirrors that folder for the page.
            workspace = self._main_folder()
            if info.workspace != workspace:
                try:
                    await self._store.update_session(session_id, workspace=workspace)
                except SessionStoreError as exc:
                    raise ApiError(503, "store_unavailable", str(exc)) from exc
        elif workspace is None:
            workspace = self._default_workspace()
            try:
                os.makedirs(workspace, mode=0o700, exist_ok=True)
                # The real path, as a create stores it: the row and the
                # Agent's workspace_root agree from the start.
                workspace = os.path.realpath(workspace)
                await self._store.update_session(session_id, workspace=workspace)
            except SessionStoreError as exc:
                raise ApiError(503, "store_unavailable", str(exc)) from exc
            except OSError as exc:
                raise ApiError(
                    400,
                    "invalid_workspace",
                    t(
                        "errors.defaultWorkspace",
                        path=display.home_label(workspace),
                        problem=exc,
                    ),
                ) from exc
        self.check_workspace(workspace)

        await self._make_room(session_id)
        if self._closed:
            raise RuntimeError("The console's sessions are closed")
        is_new = entry is None
        settings = self._settings()
        mcp_config = getattr(settings, "mcp_config", "") or None
        agent = await self._build_agent(
            settings, workspace, mcp_config, session_id, front=info.kind == "front"
        )

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
                progress=True,
                pictures=True,
                memory=self._memory,
            )
            granted_at: dict[str, float] = {}
            policy = RememberingPolicy(
                ask=runner.observe(self._timed_ask(session_id, granted_at))
            )
            entry = OpenSession(
                session_id=session_id,
                runner=runner,
                policy=policy,
                granted_at=granted_at,
                agent=agent,
                # MCP is deferred (decision 5 B) and default_builder never
                # reads mcp_config, so no console Agent has MCP tools yet:
                # the idle sweep must not close it as if it did (2.10).
                built_with_mcp=False,
                lock=self.lock(session_id),
                last_used=self._clock(),
                main=info.kind == "front",
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
            entry.built_with_mcp = False
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
        if self._closed:
            # close_all() may have run while _load() awaited. It closed this
            # Agent and cleared the registry, so never hand the entry back.
            if entry.agent is agent:
                entry.agent = None
            with contextlib.suppress(Exception):
                await agent.close()
            if is_new:
                self._sessions.pop(session_id, None)
            raise RuntimeError("The console's sessions are closed")
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

    def forget_grant(self, session_id: str, tool_name: str | None = None) -> None:
        """Take back one session grant, or every grant with ``None``.

        The grant lives in the policy and its start time in the registry, so
        both go; a session that is not open has neither.
        """
        entry = self._sessions.get(session_id)
        if entry is None:
            return
        entry.policy.forget(tool_name)
        if tool_name is None:
            entry.granted_at.clear()
        else:
            entry.granted_at.pop(tool_name, None)

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
                t("errors.turnRunning"),
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

    def _timed_ask(
        self,
        session_id: str | Callable[[], str | None],
        granted_at: dict[str, float],
    ) -> AskFunction:
        """The broker's ask, recording when an allowed-with-remember grant began.

        *session_id* may be a callable, for a session whose first turn has
        not created it yet: the id is read when a request asks.
        """

        async def timed(request: ApprovalRequest) -> ApprovalDecision:
            current = session_id() if callable(session_id) else session_id
            if current is None:
                return ApprovalDecision(
                    False, "The session is not open yet.", origin="policy"
                )
            decision = await self._broker.asker(current)(request)
            if (
                isinstance(decision, ApprovalDecision)
                and decision.allow
                and decision.remember
                and offers_grant(request.review)
            ):
                granted_at[request.tool_name] = self._clock()
            return decision

        return timed

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
                and not entry.lock.pending
                and not self._broker.pending(entry.session_id)
                and entry.session_id != session_id
            ]
            if not candidates:
                return
            oldest = min(candidates, key=lambda entry: entry.last_used)
            async with oldest.lock:
                if (
                    self._sessions.get(oldest.session_id) is oldest
                    and oldest.agent is not None
                    and not oldest.runner.busy
                    and not oldest.lock.pending
                    and not self._broker.pending(oldest.session_id)
                ):
                    await self._keep_locked(oldest)
                else:
                    return

    # ── Turns ──

    async def _drop_unopened(self, session_id: str) -> None:
        """Delete a row whose Agent or first turn never came up, best effort."""
        with contextlib.suppress(Exception):
            await self._store.delete_session(session_id)

    async def _build_agent(
        self,
        settings: Config,
        workspace: str,
        mcp_config: str | None,
        session_id: str,
        *,
        front: bool = False,
    ) -> Agent:
        """The builder's Agent (the front's for the main chat), a failed start mapped as 2.2 says."""
        builder = (self._front_builder if front else None) or self._builder
        try:
            return await builder(settings, workspace, mcp_config, session_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ApiError(
                502,
                "agent_start_failed",
                str(exc) or type(exc).__name__,
            ) from exc

    def _settings_model(self) -> str:
        """The chat model the settings name, as ``session.model`` records it."""
        settings = self._settings()
        backend = getattr(settings, "chat_backend", "") or ""
        section = getattr(settings, backend, None) if backend else None
        return getattr(section, "chat_model", "") or ""

    def check_workspace(self, workspace: str) -> None:
        """400 ``invalid_workspace`` unless *workspace* is still a directory."""
        if not os.path.isdir(workspace):
            raise ApiError(
                400,
                "invalid_workspace",
                t("errors.workspaceGone", path=display.home_label(workspace)),
            )

    def _default_workspace(self) -> str:
        """The configured ``[gateway] workspace``, or :data:`DEFAULT_WORKSPACE`."""
        settings = self._settings()
        gateway = getattr(settings, "gateway", None)
        raw = getattr(gateway, "workspace", "") if gateway is not None else ""
        if gateway_workspace_problem(raw, allow_empty=False) is None:
            return os.path.expanduser(raw)
        return os.path.expanduser(DEFAULT_WORKSPACE)

    def _main_folder(self) -> str:
        """The default folder's real path, made when it is missing (the main chat's)."""
        path = self._default_workspace()
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ApiError(
                400,
                "invalid_workspace",
                t(
                    "errors.defaultWorkspace",
                    path=display.home_label(path),
                    problem=exc,
                ),
            ) from exc
        return os.path.realpath(path)

    def _resolve_workspace(self, workspace: str | None) -> str:
        """The folder a new session gets: the default, or a checked path.

        The real path is what the row keeps, so every stored workspace is
        the Agent's ``workspace_root``: a link or a ``..`` path cannot make
        a session that a later open reads as a different folder.
        """
        if workspace is None:
            path = self._default_workspace()
            try:
                os.makedirs(path, mode=0o700, exist_ok=True)
            except OSError as exc:
                raise ApiError(
                    400,
                    "invalid_workspace",
                    t(
                        "errors.defaultWorkspace",
                        path=display.home_label(path),
                        problem=exc,
                    ),
                ) from exc
            return os.path.realpath(path)
        if not isinstance(workspace, str) or not (
            os.path.isabs(workspace) or workspace.startswith("~/")
        ):
            raise ApiError(
                400,
                "invalid_workspace",
                t("errors.workspacePath"),
            )
        path = os.path.expanduser(workspace)
        if not os.path.isdir(path):
            raise ApiError(
                400,
                "invalid_workspace",
                t("errors.notDirectory", path=display.home_label(path)),
            )
        return os.path.realpath(path)

    def _create_lock(self, client_request_id: str | None) -> asyncio.Lock | None:
        """The lock of *client_request_id*'s create, or ``None`` for an unkeyed one."""
        if client_request_id is None:
            return None
        lock = self._create_locks.get(client_request_id)
        if lock is None:
            lock = asyncio.Lock()
            self._create_locks[client_request_id] = lock
        return lock

    def _release_create_lock(
        self, client_request_id: str | None, lock: asyncio.Lock | None
    ) -> None:
        """Forget a finished create's lock; a waiter keeps its reference.

        Two creates that race a dropped lock cannot both make a session: the
        store checks the key again in ``begin_turn``'s transaction.
        """
        if client_request_id is not None and lock is not None and not lock.locked():
            self._create_locks.pop(client_request_id, None)

    async def create(
        self,
        workspace: str | None,
        first: tuple[str, str] | None = None,
        *,
        client_request_id: str | None = None,
    ) -> tuple[OpenSession, TurnInfo | None]:
        """Create a session, usually with its first turn (``POST /api/sessions``).

        *first* is ``(typed, composed)``: the typed text titles the session,
        the composed text is the turn's. *workspace* is the folder, or the
        default when ``None``. With *client_request_id*, a repeat of an
        accepted key raises :class:`SubmissionRepeated` with its turn and
        makes no Agent; a raised error after the Agent was built closes it
        and registers nothing.
        """
        lock = self._create_lock(client_request_id)
        try:
            async with lock or contextlib.nullcontext():
                return await self._create_locked(workspace, first, client_request_id)
        finally:
            # Outside the block: the lock this task took is free again, so
            # the entry can go. A waiter keeps the object it already has.
            self._release_create_lock(client_request_id, lock)

    async def _create_locked(
        self,
        workspace: str | None,
        first: tuple[str, str] | None,
        client_request_id: str | None,
    ) -> tuple[OpenSession, TurnInfo | None]:
        if self._closed:
            raise RuntimeError("The console's sessions are closed")
        typed, composed = (None, None) if first is None else first
        if client_request_id is not None:
            # The key decides before the directory or the Agent, so a repeat
            # costs neither, and a rejected key writes no row.
            found = await self._store.get_submission(None, client_request_id)
            check_submission(found, composed or "")
        path = self._resolve_workspace(workspace)
        settings = self._settings()
        model = self._settings_model()
        # A create opens an Agent too, so the cap counts it like an open.
        await self._make_room("")
        # The row comes before the Agent: the scheduler gate tags every call
        # with the session id, so the id must exist before the first one.
        info = await self._store.create_session(model=model, workspace=path)
        # Under the session's lock, so a turn started on the new row by
        # another client waits until this create has registered it.
        async with self.lock(info.id):
            return await self._fill_new(
                info,
                settings,
                path,
                typed,
                composed,
                client_request_id,
                has_turn=first is not None,
            )

    async def _fill_new(
        self,
        info: SessionInfo,
        settings: Config,
        path: str,
        typed: str | None,
        composed: str | None,
        client_request_id: str | None,
        *,
        has_turn: bool,
    ) -> tuple[OpenSession, TurnInfo | None]:
        """Build the Agent of a new row, start its first turn and register it."""
        try:
            # MCP is deferred (U3 decision 5 B): the builder gets no config.
            agent = await self._build_agent(settings, path, None, info.id)
        except BaseException:
            await self._drop_unopened(info.id)
            raise
        granted_at: dict[str, float] = {}
        runner = SessionRunner(
            agent,
            self._store,
            info.id,
            holder=self._holder,
            resume_mode=DEFAULT_RESUME_MODE,
            progress=True,
            pictures=True,
            memory=self._memory,
            # A create's key stays where a create looks it up, even though
            # the row was made before the Agent.
            create_scope_key=client_request_id is not None,
        )
        policy = RememberingPolicy(
            ask=runner.observe(self._timed_ask(info.id, granted_at))
        )
        turn: TurnInfo | None = None
        if has_turn:
            # The first turn is a turn of this chat, so the gate hears it as
            # the page's later sends make it heard: the hold clock restarts
            # and this chat becomes the lane's focus before its first call.
            self._note_turn_start(info.id)
            try:
                turn = await runner.start_turn(
                    composed or "",
                    approval=policy,
                    title=typed,
                    client_request_id=client_request_id,
                )
            except BaseException:
                # The row exists already, and a first turn that failed or
                # was cancelled may have started its tasks: no client has
                # this id, so end the runner and drop the row before the
                # Agent goes.
                await self._abandon_create(runner, agent)
                raise
        session_id = info.id
        if self._closed:
            # close_all() ran while the row was made: register nothing, end
            # a turn that started, and drop the row the shutdown never saw.
            with contextlib.suppress(Exception):
                await runner.close(cause="shutdown")
            with contextlib.suppress(Exception):
                await agent.close()
            await self._drop_unopened(session_id)
            raise RuntimeError("The console's sessions are closed")
        entry = OpenSession(
            session_id=session_id,
            runner=runner,
            policy=policy,
            granted_at=granted_at,
            agent=agent,
            built_with_mcp=False,
            lock=self.lock(session_id),
            last_used=self._clock(),
        )
        self._sessions[session_id] = entry
        if turn is not None:
            self._apply_deferred_cancel(entry, turn.id)
        return entry, turn

    async def _abandon_create(self, runner: SessionRunner, agent: Agent) -> None:
        """Undo a create whose first turn never returned.

        The create made the row before the Agent, and ``start_turn`` may
        have started the turn's tasks before it raised, while no client
        ever received the id. The cancel that brought us here keeps
        arriving, so the cleanup runs in
        its own task: shielded from this one, it ends the runner, deletes
        the row and closes the Agent.
        """

        async def cleanup() -> None:
            if runner.session_id is not None:
                with contextlib.suppress(Exception):
                    await runner.close(cause="shutdown")
                with contextlib.suppress(Exception):
                    await self._store.delete_session(runner.session_id)
            with contextlib.suppress(Exception):
                await agent.close()

        task = asyncio.create_task(cleanup())
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(task)

    async def start_turn(
        self,
        session_id: str,
        typed: str,
        composed: str,
        *,
        client_request_id: str | None = None,
    ) -> tuple[TurnInfo, ResumeInfo | None]:
        """Start a turn in *session_id*; return it and the open's resume.

        The resume is the ``ResumeInfo`` of an open this call made, for the
        route's 202, else ``None``. The key is checked before the session is
        opened, so a repeat never builds an Agent.
        """
        async with self.lock(session_id):
            if client_request_id is not None:
                found = await self._store.get_submission(session_id, client_request_id)
                check_submission(found, composed)
            entry = await self.open(session_id)
            resumed = entry.resumed
            entry.resumed = None
            await self._reconcile(entry)
            self._note_turn_start(session_id)
            turn = await entry.runner.start_turn(
                composed,
                approval=entry.policy,
                title=typed,
                client_request_id=client_request_id,
            )
            entry.last_used = self._clock()
            return turn, resumed

    async def retry(
        self, session_id: str, turn_id: str
    ) -> tuple[TurnInfo, ResumeInfo | None]:
        """Undo the newest finished turn and run its text again (``/retry``).

        409 ``session_busy`` while a turn runs, and 409 ``not_latest_turn``
        for a turn that is not the newest finished one. The resume is the one
        ``undo()`` ended with.
        """
        async with self.lock(session_id):
            entry = await self.open(session_id)
            entry.resumed = None
            if entry.runner.busy:
                raise ApiError(
                    409,
                    "session_busy",
                    t("errors.turnRunning"),
                )
            latest = await self._latest_finished_turn(session_id)
            if latest is None or latest.id != turn_id:
                raise ApiError(
                    409,
                    "not_latest_turn",
                    t("errors.retryNewest"),
                )
            undone = await entry.runner.undo()
            if undone is None:
                raise ApiError(
                    409,
                    "not_latest_turn",
                    t("errors.retryNewest"),
                )
            resumed = entry.runner.last_resume
            await self._reconcile(entry)
            self._note_turn_start(session_id)
            turn = await entry.runner.start_turn(
                undone.user_text, approval=entry.policy
            )
            entry.last_used = self._clock()
            return turn, resumed

    async def change_workspace(self, session_id: str, workspace: str) -> SessionInfo:
        """Move *session_id* to *workspace*, with a new Agent on its conversation.

        409 ``session_busy`` while a turn runs; a bad folder is 400
        ``invalid_workspace`` and closes the new Agent, leaving the row as it
        was. With no open Agent, only the row and the grants change.
        """
        path = self._resolve_workspace(workspace)
        async with self.lock(session_id):
            current = await self._store.get_session(session_id)
            if current is not None and current.kind == "task":
                raise ApiError(
                    409,
                    "task_session_read_only",
                    t("errors.taskWorkspace"),
                )
            if current is not None and current.kind == "front":
                raise ApiError(
                    409,
                    "main_chat_folder",
                    t("errors.mainFolder"),
                )
            entry = self._sessions.get(session_id)
            if entry is None or entry.agent is None:
                info = await self._store.update_session(session_id, workspace=path)
                if entry is not None:
                    entry.policy.forget()
                    entry.granted_at.clear()
                return info
            if entry.runner.busy:
                raise ApiError(
                    409,
                    "session_busy",
                    t("errors.turnRunning"),
                )
            settings = self._settings()
            old_root = entry.agent.workspace_root
            agent = await self._build_agent(
                settings,
                path,
                None,
                session_id,
                front=current is not None and current.kind == "front",
            )
            try:
                agent.load_conversation(entry.agent.conversation())
                # The row names the new folder before the swap, so a store
                # failure leaves the old Agent and the old row together, as
                # the request found them.
                info = await self._store.update_session(
                    session_id, workspace=agent.workspace_root
                )
                try:
                    entry.runner.replace_agent(agent)
                except BaseException:
                    with contextlib.suppress(Exception):
                        await self._store.update_session(session_id, workspace=old_root)
                    raise
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    await agent.close()
                if isinstance(exc, SessionBusyError):
                    raise ApiError(
                        409,
                        "session_busy",
                        t("errors.turnRunning"),
                    ) from exc
                raise
            old, entry.agent = entry.agent, agent
            with contextlib.suppress(Exception):
                await old.close()
            entry.policy.forget()
            entry.granted_at.clear()
            entry.last_used = self._clock()
            return info

    def _note_turn_start(self, session_id: str) -> None:
        """The gate's turn facts, as the page's chats set them (S2).

        The hold clock restarts for the new turn and this chat becomes the
        lane's focus. A console built without a gate (a unit test) has none.
        """
        if self._gate is None:
            return
        self._gate.turn_start(session_id)
        self._gate.focus(session_id)

    async def _reconcile(self, entry: OpenSession) -> None:
        """Record the settings' chat model on the session, when it changed."""
        model = self._settings_model()
        if not model:
            return
        info = await self._store.get_session(entry.session_id)
        if info is not None and info.model != model:
            await self._store.update_session(entry.session_id, model=model)

    async def _latest_finished_turn(self, session_id: str) -> TurnInfo | None:
        """The newest turn that is finished and not undone, or ``None``."""
        turns = await self._store.list_turns(session_id, limit=2)
        for turn in reversed(turns):
            if not turn.undone and turn.status != "running":
                return turn
        return None

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
                    self._sessions.get(entry.session_id) is entry
                    and entry.agent is not None
                    and not entry.runner.busy
                    and not entry.lock.pending
                    and self._clock() - entry.last_used >= self._idle_close_s
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
                    self._sessions.get(entry.session_id) is entry
                    and entry.agent is not None
                    and not entry.runner.busy
                    and not entry.lock.pending
                    and not self._broker.pending(entry.session_id)
                ):
                    await self._keep_locked(entry)

    # ── Delete and shutdown ──

    async def delete(self, session_id: str, *, forget_notes: bool = False) -> None:
        """Close the session's runner and delete its row; 404 when unknown.

        The chat's index entries go with it; its notes stay (a remembered fact
        outlives the moment it was learned) unless *forget_notes*.
        """
        lock = self.lock(session_id)
        # Do not wait behind an opening or queued start: it has already claimed
        # the session, even if the runner has not registered a turn yet.
        if lock.locked() or lock.pending:
            raise ApiError(409, "session_busy", t("errors.turnRunning"))
        async with lock:
            info = await self._store.get_session(session_id)
            if info is not None and info.kind == "task":
                raise ApiError(
                    409,
                    "task_session_read_only",
                    t("errors.taskDelete"),
                )
            if info is not None and info.kind == "front":
                raise ApiError(
                    409,
                    "main_chat",
                    t("errors.mainDelete"),
                )
            entry = self._sessions.get(session_id)
            if entry is not None and (
                entry.runner.busy or self._broker.pending(session_id)
            ):
                raise ApiError(409, "session_busy", t("errors.turnRunning"))
            try:
                deleted = await self._store.delete_session(session_id)
            except SessionBusyError as exc:
                raise ApiError(409, "session_busy", t("errors.turnRunning")) from exc
            if not deleted:
                raise ApiError(404, "not_found", t("errors.sessionNotFound"))
            if self._memory is not None:
                try:
                    await self._memory.forget_session(session_id)
                    if forget_notes:
                        await self._memory.forget_chat_notes(session_id)
                except Exception:
                    logger.warning("Could not forget a deleted chat", exc_info=True)
            if entry is not None:
                await entry.runner.close(cause="closed")
                if entry.agent is not None:
                    agent, entry.agent = entry.agent, None
                    with contextlib.suppress(Exception):
                        await agent.close()
                self._sessions.pop(session_id, None)
        # The lock stays: a turn that queued on it before the delete holds
        # this object, and ``open`` checks that the lock it is under is the
        # session's. The dict keeps one small lock per session this process
        # has opened; a session id is never reused.

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
