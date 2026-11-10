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
import copy
import dataclasses
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts import work_folder
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    Withdrawal,
)
from localmelo.melo.contracts.inbox import InboxSource
from localmelo.melo.contracts.tools import BaseToolProvider
from localmelo.melo.schema import TASK_EXTENSIONS, ImagePart, Message
from localmelo.support.browser import configured_driver
from localmelo.support.config import SavedFolders, gateway_workspace_problem
from localmelo.support.gateway.auto_blocks import BlockError
from localmelo.support.gateway.chat_gate import ChatGate, GatedEmbedding, GatedLLM
from localmelo.support.gateway.i18n import t
from localmelo.support.gateway.task_policy import (
    DEFAULT_WORKSPACE,
    folder_problem,
    worker_tool_allowlist,
)
from localmelo.support.sessions import display
from localmelo.support.sessions.agents import (
    build_session_agent,
    configured_session_options,
)
from localmelo.support.sessions.approvals import (
    ApprovalBroker,
    AskFunction,
    CardWatch,
    ChatMode,
    ChatPolicy,
    chat_mode,
)
from localmelo.support.sessions.attachments import (
    TextAttachment,
    compose_user_text,
    words_before_files,
)
from localmelo.support.sessions.grants import (
    STORED_KINDS,
    Grant,
    GrantBook,
    SessionGrants,
)
from localmelo.support.sessions.live import LiveReply, Publish, declared_framing
from localmelo.support.sessions.runner import (
    DEFAULT_RESUME_MODE,
    BeforeTurn,
    ResumeInfo,
    ResumeLoader,
    SessionRunner,
)
from localmelo.support.sessions.store import (
    SessionBusyError,
    SessionInfo,
    SessionStore,
    SessionStoreError,
    TurnInfo,
    check_attachment_submission,
    process_holder,
)
from localmelo.support.sessions.submissions import check_submission

from .errors import ApiError, store_unavailable

# What a Stop from the console says, and the cause it records.
STOP_REASON = "Stopped from the web console"

if TYPE_CHECKING:
    from localmelo.support.config import Config
    from localmelo.support.memory import ChatMemory

logger = logging.getLogger(__name__)

# Agents open at once; opening one more closes the least recently used idle
# Agent of any kind first.
MAX_OPEN: Final = 8
# Seconds the shutdown may take to close every runner.
CLOSE_ALL_TIMEOUT_S: Final = 10.0
# Of that, the reserve an Agent's begin_exit leaves for its own save.
EXIT_SAVE_RESERVE_S: Final = 1.0
# Seconds between the console's sweeps (the timer that writes the main chat's cards).
SWEEP_INTERVAL_S: Final = 60.0

# What a chat has in a folder no task may work in: it reads, and changes nothing.
READING_BUILTINS: Final = frozenset({"file_read", "git_status", "list_files"})

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


def chat_options(settings: Config, workspace: str) -> AgentOptions:
    """A chat's options, main or side: the person's work runs in the chat (#571).

    It has the file tools and Python of a person's task, written under the same
    confinement, and that work's step bound: the steps of a chat's turn are the
    steps of the work. There is no reflection: a chat's turn goes on or ends, and
    the person is there to say what comes next. A folder no task may work in
    (:func:`~localmelo.support.gateway.task_policy.folder_problem`) stays readable
    and nothing in it is written or run: the chat opens, and says why.
    """
    options = configured_session_options(
        settings,
        AgentOptions(
            mode="gateway",
            builtins=(
                READING_BUILTINS
                if folder_problem(workspace, settings)
                else worker_tool_allowlist("user").builtins
            ),
            confine_writes=settings.agent.confine_writes,
            max_extensions=TASK_EXTENSIONS,
        ),
    )
    # A legacy Agent has no use for the option; the other modes use it for their
    # own recall.
    return dataclasses.replace(
        options,
        attempt_policy="none",
        memory=options.memory and options.reasoning != "legacy",
    )


async def default_builder(
    settings: Config,
    workspace: str,
    mcp_config: str | None,
    session_id: str,
    gate: ChatGate | None = None,
    memory: ChatMemory | None = None,
    tools: Sequence[BaseToolProvider] = (),
    agent_class: type[Agent] | None = None,
) -> Agent:
    """A chat's started Agent, from the shared session builder: the one composition of every chat.

    MCP is deferred to U4, so *mcp_config* is not read yet; a caller that
    needs it passes its own builder in :class:`ConsoleOptions`. With *gate*,
    the providers are wrapped in :class:`GatedLLM` and
    :class:`GatedEmbedding` for *session_id*, so every call this Agent makes
    waits for its slot like a workspace chat's. *memory* gives the chat
    memory across chats (see :func:`build_session_agent`); *tools* and
    *agent_class* give a chat its task tools and its Agent class. It has
    ``web_fetch`` unless ``[tools] web_fetch`` is off, ``web_search`` when
    ``[web] search`` is on, and the browser the settings name.
    """

    llm = None if gate is None else gate.client(session_id)
    return await build_session_agent(
        cfg=settings,
        base_url=None,
        model=None,
        workspace=workspace,
        options=chat_options(settings, workspace),
        browser_driver=configured_driver(settings),
        # Its pages stay in melo's browser, as a task's do: another app's window
        # is the person's to see, and they may not be at the screen.
        desktop=False,
        web_fetch=settings.tools.web_fetch is True,
        web_search=settings.web.search is True,
        wrap_llm=None if llm is None else (lambda inner: GatedLLM(inner, llm)),
        wrap_embedding=(
            None if llm is None else (lambda inner: GatedEmbedding(inner, llm))
        ),
        identify=True,
        memory=memory,
        tools=tools,
        agent_class=agent_class,
        chat_id=session_id,
    )


def _with_chat_model(settings: Config, model: str) -> Config:
    """*settings* asking for *model*: the same object when its chat backend names it already.

    Otherwise a copy, so the Agent asks for exactly the model that was read
    (an empty one is no choice: the build fails as it does without).
    """
    backend = getattr(settings, "chat_backend", "") or ""
    section = getattr(settings, backend, None)
    if not model or section is None or getattr(section, "chat_model", model) == model:
        return settings
    pinned = copy.deepcopy(settings)
    getattr(pinned, backend).chat_model = model
    return pinned


@dataclass(frozen=True)
class TurnExtras:
    """What a message carries besides its words: files read as text, and pictures."""

    attachments: tuple[TextAttachment, ...] = ()
    images: tuple[ImagePart, ...] = ()


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
    policy: ChatPolicy
    agent: Agent | None
    kept: list[Message] | None = None
    mcp_error: str = ""
    resumed: ResumeInfo | None = None
    lock: SessionLock = field(default_factory=SessionLock)
    last_used: float = 0.0
    # The main chat, which works in the default folder as it is now.
    main: bool = False
    # The settings change its Agent was built after (ConsoleSessions' count):
    # an older Agent is built again at the chat's next open.
    settings_epoch: int = 0
    # The chat model its Agent asks for ("" when the settings name none). A
    # backend that serves its own model (melo_serve) changes what it serves
    # with no edit of the settings, so the chat is built again at its next
    # open when the settings name another one now.
    model: str = ""

    @property
    def grants(self) -> SessionGrants:
        """The grants that hold in this chat (its own, and the main chat's)."""
        return self.policy.grants


class ConsoleSessions:
    """The console's sessions, opened on demand over one gateway store.

    *settings* is read again on every open, so a settings change reaches
    the next Agent. *builder* returns a started Agent and must close it and
    raise on failure; *max_open* is the bound of :data:`MAX_OPEN`. *clock*
    replaces time in tests.
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
        max_open: int = MAX_OPEN,
        clock: Callable[[], float] = time.time,
        viewers: Callable[[str], int] | None = None,
        inbox_for: Callable[[str], InboxSource] | None = None,
        grants: GrantBook | None = None,
        live: Publish | None = None,
    ) -> None:
        if store.writer != "gateway":
            raise ValueError("the console opens the gateway's store")
        self._store = store
        self._broker = broker
        # How many surfaces show a chat to a person now (the hub's viewers,
        # #465); None: nobody ever does, so Auto blocks as before.
        self._viewers = viewers
        # Where a chat's reply goes as it is written (the hub's live); None:
        # every reply arrives whole.
        self._publish_live = live
        # The run inbox of a chat: the words the person adds while it works.
        self.inbox_for = inbox_for
        # Every chat's grants, in the store's table: what the person allows
        # in the main chat counts in every chat and task.
        self.book = grants if grants is not None else GrantBook()
        # The folders "Always allow this folder" keeps, read from the settings file (#405).
        self.saved_folders = SavedFolders()
        self._settings = settings
        self._gate = gate
        self._memory = memory
        self._builder: AgentBuilder = builder or self._default_builder
        # The main chat's own Agent (the front); None builds it as any chat.
        self._front_builder = front_builder
        self._max_open = max_open
        self._clock = clock
        self._holder = process_holder("gateway")
        self._sessions: dict[str, OpenSession] = {}
        self._locks: dict[str, SessionLock] = {}
        # Each chat's mode as last committed or read: what a task of the chat
        # decides by (#576), set in the same step as the store's write.
        self._modes: dict[str, ChatMode] = {}
        # Each Allow once's watch: it drops the allowance its turn left unused.
        self._allowance_watches: set[asyncio.Task[None]] = set()
        # One lock per client_request_id while a keyed create runs, so a
        # retried create makes one session.
        self._create_locks: dict[str, asyncio.Lock] = {}
        # Stops that arrived for a turn a create had not registered yet.
        self._deferred_cancels: set[str] = set()
        # How many times the settings changed: an Agent built before the last
        # change (one that was busy then) is built again at its next open.
        self._settings_epoch = 0
        self._closed = False
        # Waits, before a turn calls the model, for a switch of the served
        # model (the console's ServeSwitch.hold); True when it waited.
        self.hold: Callable[[str], Awaitable[bool]] | None = None
        # Told each Agent a chat gets (a rebuilt one included): the console follows its browser there.
        self.on_agent: Callable[[str, Agent], None] | None = None

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

    def _inbox(self, session_id: str) -> InboxSource | None:
        return None if self.inbox_for is None else self.inbox_for(session_id)

    def get(self, session_id: str) -> OpenSession | None:
        return self._sessions.get(session_id)

    def grants(self, session_id: str) -> frozenset[str]:
        """The names chat *session_id* holds a grant of now, open or not (the main chat's too)."""
        return self.book.allowed_for(session_id)

    def folder_grants(self, session_id: str) -> tuple[str, ...]:
        """The folders outside the workspace chat *session_id* allowed for reading (#405)."""
        return self.book.folders_for(session_id)

    def grants_view(self, info: SessionInfo) -> SessionGrants:
        """The grants of the session *info*: the main chat's count in every chat.

        A task's own view holds only what was allowed for it: its chat's
        grants reach it through its chat (#369).
        """
        kind = STORED_KINDS.get(info.kind, "side")
        return self.book.view(info.id, kind, inherit=kind != "task")

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
            served = self._settings_model()
            if not entry.runner.busy and (
                (entry.main and root != self._main_folder())
                or (
                    (
                        entry.settings_epoch != self._settings_epoch
                        or (served and served != entry.model)
                    )
                    and not self._broker.pending(session_id)
                )
            ):
                # The default folder changed: the main chat follows it with
                # a new Agent on the same conversation (after a running turn).
                # So does a chat whose turn ran when the settings changed (a
                # browser chosen in Settings), or that was built on another
                # model than the settings name now (melo_serve switched):
                # its next turn gets them.
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
            raise store_unavailable(exc) from exc
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
                    raise store_unavailable(exc) from exc
        elif workspace is None:
            workspace = self._default_workspace()
            try:
                os.makedirs(workspace, mode=0o700, exist_ok=True)
                # The real path, as a create stores it: the row and the
                # Agent's workspace_root agree from the start.
                workspace = os.path.realpath(workspace)
                await self._store.update_session(session_id, workspace=workspace)
            except SessionStoreError as exc:
                raise store_unavailable(exc) from exc
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
        epoch = self._settings_epoch
        settings = self._settings()
        mcp_config = getattr(settings, "mcp_config", "") or None
        agent, model = await self._build_agent(
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
                inbox=self._inbox(session_id),
                before_turn=self._before_turn(session_id),
                live=self._live(session_id),
            )
            policy = ChatPolicy(
                runner.observe(self._timed_ask(session_id)),
                self.mode_of(info),
                saved=self.saved_folders,
                watch=self.card_watch(session_id),
                grants=self.grants_view(info),
            )
            entry = OpenSession(
                session_id=session_id,
                runner=runner,
                policy=policy,
                agent=agent,
                lock=self.lock(session_id),
                last_used=self._clock(),
                main=info.kind == "front",
                settings_epoch=epoch,
                model=model,
            )
            self._sessions[session_id] = entry
        else:
            entry.policy.mode = self.mode_of(info)
            try:
                entry.runner.replace_agent(agent)
            except SessionBusyError as exc:
                with contextlib.suppress(Exception):
                    await agent.close()
                raise ApiError(409, "session_busy", str(exc)) from exc
            entry.agent = agent
            entry.settings_epoch = epoch
            entry.model = model
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
                raise store_unavailable(exc) from exc
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

    def _before_turn(self, session_id: str) -> BeforeTurn:
        """What a chat's turn does before it calls the model: wait for a model switch.

        A turn that waited runs on a new Agent when the switch changed the
        model the settings name (melo_serve's): the conversation moves to it,
        as at a new open, and the old Agent closes.
        """

        async def before(reload: ResumeLoader) -> None:
            if self.hold is None or not await self.hold(session_id):
                return
            entry = self._sessions.get(session_id)
            served = self._settings_model()
            if entry is None or entry.agent is None or served in ("", entry.model):
                return
            old = entry.agent
            root = old.workspace_root
            assert root is not None
            agent, model = await self._build_agent(
                self._settings(), root, None, session_id, front=entry.main
            )
            agent.load_conversation(old.conversation())
            entry.runner.adopt(agent)
            entry.agent, entry.model = agent, model
            with contextlib.suppress(Exception):
                await old.close()

        return before

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

    async def forget_grant(
        self, session_id: str, tool_name: str | None = None
    ) -> list[Grant]:
        """Take back one grant that holds in *session_id*, or every grant it gave.

        Open or not: the grants live in the table. A grant of the main chat
        that holds here is taken back everywhere by its name; with ``None``
        only the chat's own go (in the main chat, every global one). The
        checks stop using them at once; this waits for the rows, and a write
        that fails raises :class:`SessionStoreError`.
        """
        return await self.book.forget(session_id, tool_name)

    async def forget_folder(self, session_id: str, folder: str) -> list[Grant]:
        """Take back a folder's grant that holds in *session_id*; the grants taken back.

        A main chat's grant of the folder holds everywhere, so it goes
        everywhere. A write that fails raises :class:`SessionStoreError`.
        """
        return await self.book.forget(session_id, folder, folders=True)

    async def _forget_moved(self, session_id: str) -> None:
        """A chat in another folder starts without its own grants (the main chat's still hold)."""
        try:
            await self.forget_grant(session_id)
        except SessionStoreError:
            # The checks stopped using them; only the file still lists them.
            logger.warning(
                "Could not save that %s's grants were taken back", session_id
            )

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

    def card_watch(self, session_id: str) -> CardWatch:
        """Who sees the card Auto shows in *session_id*, and how long it waits (#465).

        The chat's viewers (a console, the app or the terminal following its
        stream as one); none with ``[approvals] auto_asks`` off, so Auto
        blocks as before. A card nobody answers is ended through the broker.
        """

        async def viewers() -> int:
            if self._viewers is None or not self._settings().approvals.auto_asks:
                return 0
            return self._viewers(session_id)

        return CardWatch(viewers, self._broker.ends)

    def _timed_ask(
        self,
        session_id: str | Callable[[], str | None],
    ) -> AskFunction:
        """The broker's ask for the chat's card.

        *session_id* may be a callable, for a session whose first turn has
        not created it yet: the id is read when a request asks. The answer
        is the person's for this call; what they allow for the chat is the
        chat policy's grant, kept in the table, and nothing else is kept.
        """

        async def timed(request: ApprovalRequest) -> ApprovalDecision:
            current = session_id() if callable(session_id) else session_id
            if current is None:
                return Withdrawal(
                    False, "The session is not open yet.", origin="policy"
                )
            return await self._broker.asker(current)(request)

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
    ) -> tuple[Agent, str]:
        """The builder's Agent (the front's for the main chat) and the chat model it asks for.

        The model is read once, and the Agent is built on it: a backend that
        serves its own model (melo_serve) could answer another one to the
        builder's own read. A failed start is mapped as 2.2 says.
        """
        builder = (self._front_builder if front else None) or self._builder
        model = self._settings_model(settings)
        try:
            agent = await builder(
                _with_chat_model(settings, model), workspace, mcp_config, session_id
            )
            if self.on_agent is not None:
                self.on_agent(session_id, agent)
            return agent, model
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ApiError(
                502,
                "agent_start_failed",
                str(exc) or type(exc).__name__,
            ) from exc

    def new_chat_mode(self) -> ChatMode:
        """The mode a new chat starts in: ``[approvals] mode``, else Ask."""
        approvals = getattr(self._settings(), "approvals", None)
        return chat_mode({"approvals": getattr(approvals, "mode", "ask")})

    def mode_of(self, info: SessionInfo) -> ChatMode:
        """A chat's mode: its own, or for the main chat the new chats' until picked."""
        default = self.new_chat_mode() if info.kind == "front" else "ask"
        return chat_mode(info.meta, default)

    async def set_mode(self, session_id: str, mode: ChatMode) -> SessionInfo:
        """Switch a chat to Ask or Auto: stored, and from its next step.

        A call already waiting for an answer keeps waiting; Auto's
        allowances and Ask's grants are kept for a later switch back.
        """
        async with self.lock(session_id):
            info = await self._store.update_session(
                session_id, meta_update={"approvals": mode}
            )
            # Nothing is awaited from the commit to here: a reader that sees
            # the store's new mode finds this one too.
            self._modes[session_id] = mode
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.policy.mode = mode
            return info

    async def current_mode(self, session_id: str) -> ChatMode:
        """A chat's mode as the store has it now; :meth:`mode_now` gives it from here on.

        A switch that commits while this reads is newer than what was read,
        and stays. A chat that is gone has Ask.
        """
        seen = self._modes.get(session_id)
        info = await self._store.get_session(session_id)
        if info is None:
            self._modes.pop(session_id, None)
            return "ask"
        if self._modes.get(session_id) == seen:
            self._modes[session_id] = self.mode_of(info)
        return self._modes[session_id]

    def mode_now(self, session_id: str | None) -> ChatMode:
        """A chat's mode as last committed or read, without waiting; Ask for a chat not read yet."""
        return self._modes.get(session_id or "", "ask")

    def allowances(self, session_id: str) -> dict[str, float]:
        """The steps allowed once in this chat since the gateway started: key -> until."""
        entry = self._sessions.get(session_id)
        return {} if entry is None else dict(entry.policy.auto.given)

    def allowance_live(self, session_id: str, key: str) -> bool:
        """Whether the step *key* has an allowance in this chat not used or run out yet."""
        entry = self._sessions.get(session_id)
        return entry is not None and entry.policy.auto.allowed_once(key)

    async def allow_once(
        self,
        session_id: str,
        key: str,
        categories: list[str],
        text: str,
        tool: str = "",
    ) -> tuple[TurnInfo, float]:
        """Allow the step *key* once (Auto, #339) and send *text* as the person's message.

        409 ``session_busy`` while a turn runs, so the allowance and the
        message go together; the message starts the turn that retries it.
        :class:`BlockError` ``already_allowed`` while an allowance of *key*
        waits. When the turn ends without using the allowance, it is
        dropped. Returns the turn and when the allowance runs out.
        """
        async with self.lock(session_id):
            entry = await self.open(session_id)
            if entry.runner.busy:
                raise ApiError(409, "session_busy", t("errors.turnRunning"))
            auto = entry.policy.auto
            if auto.allowed_once(key):
                raise BlockError("already_allowed")
            entry.resumed = None
            auto.allow_once(key, categories, tool)
            until = auto.given[key]
            self._note_turn_start(session_id)
            try:
                turn = await entry.runner.start_turn(text, approval=entry.policy)
            except BaseException:
                auto.withdraw(key)
                raise
            entry.last_used = self._clock()
            watch = asyncio.create_task(self._drop_unused(entry, key))
            self._allowance_watches.add(watch)
            watch.add_done_callback(self._allowance_watches.discard)
            return turn, until

    async def _drop_unused(self, entry: OpenSession, key: str) -> None:
        """When the turn Allow once started ends, drop its allowance if unused."""
        with contextlib.suppress(Exception):
            await entry.runner.wait()
        entry.policy.auto.withdraw(key)

    def _live(self, session_id: str) -> Callable[[TurnInfo], LiveReply] | None:
        """The live display of each turn of chat *session_id*, or None.

        The framing is the one ``[cli] reasoning_tags`` declares for the
        chat's model, read at each turn.
        """
        publish = self._publish_live
        if publish is None:
            return None

        def display(info: TurnInfo) -> LiveReply:
            entry = self._sessions.get(session_id)
            model = entry.model if entry is not None else self._settings_model()
            cli = getattr(self._settings(), "cli", None)
            tags = getattr(cli, "reasoning_tags", None) or {}
            return LiveReply(
                info, publish, reasoning_tags=declared_framing(tags, model)
            )

        return display

    def _settings_model(self, settings: Config | None = None) -> str:
        """The chat model the settings name, as ``session.model`` records it."""
        settings = settings or self._settings()
        backend = getattr(settings, "chat_backend", "") or ""
        if not backend:
            return ""
        from localmelo.support.backends.registry import in_use

        return in_use(settings, backend)[1]

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

    @contextlib.asynccontextmanager
    async def keyed(self, client_request_id: str | None) -> AsyncIterator[None]:
        """Run one create of *client_request_id* at a time, so a repeat finds what the first made."""
        lock = self._create_lock(client_request_id)
        try:
            async with lock or contextlib.nullcontext():
                yield
        finally:
            self._release_create_lock(client_request_id, lock)

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
        mode: ChatMode | None = None,
        admit: Callable[[], Awaitable[None]] | None = None,
        extras: TurnExtras | None = None,
        session_id: str | None = None,
    ) -> tuple[OpenSession, TurnInfo | None]:
        """Create a session, usually with its first turn (``POST /api/sessions``).

        *first* is ``(typed, composed)``: the typed text titles the session,
        the composed text is the turn's; *extras* are its files and pictures.
        *workspace* is the folder, or the
        default when ``None``. *mode* is the chat's Ask or Auto, else the
        settings' mode for new chats. With *client_request_id*, a repeat of
        an accepted key raises :class:`SubmissionRepeated` with its turn and
        makes no Agent; a raised error after the Agent was built closes it
        and registers nothing. *session_id* names the new chat instead of a
        random id, when the id is itself a receipt.
        """
        lock = self._create_lock(client_request_id)
        try:
            async with lock or contextlib.nullcontext():
                return await self._create_locked(
                    workspace,
                    first,
                    client_request_id,
                    mode or self.new_chat_mode(),
                    admit,
                    extras,
                    session_id,
                )
        finally:
            # Outside the block: the lock this task took is free again, so
            # the entry can go. A waiter keeps the object it already has.
            self._release_create_lock(client_request_id, lock)

    async def _create_locked(
        self,
        workspace: str | None,
        first: tuple[str, str] | None,
        client_request_id: str | None,
        mode: ChatMode = "ask",
        admit: Callable[[], Awaitable[None]] | None = None,
        extras: TurnExtras | None = None,
        session_id: str | None = None,
    ) -> tuple[OpenSession, TurnInfo | None]:
        if self._closed:
            raise RuntimeError("The console's sessions are closed")
        extras = extras or TurnExtras()
        typed, composed = (None, None) if first is None else first
        if client_request_id is not None:
            # The key decides before the directory or the Agent, so a repeat
            # costs neither, and a rejected key writes no row.
            found = await self._store.get_submission(None, client_request_id)
            check_attachment_submission(
                found, extras.attachments, composed if extras.attachments else None
            )
            check_submission(
                found, compose_user_text(composed or "", extras.attachments)
            )
            if admit is not None:
                await admit()
        path = self._resolve_workspace(workspace)
        epoch = self._settings_epoch
        settings = self._settings()
        model = self._settings_model()
        # A create opens an Agent too, so the cap counts it like an open.
        await self._make_room("")
        # The row comes before the Agent: the scheduler gate tags every call
        # with the session id, so the id must exist before the first one.
        info = await self._store.create_session(
            model=model,
            workspace=path,
            meta={"approvals": mode},
            session_id=session_id,
        )
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
                extras=extras,
                epoch=epoch,
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
        extras: TurnExtras | None = None,
        epoch: int = 0,
    ) -> tuple[OpenSession, TurnInfo | None]:
        """Build the Agent of a new row, start its first turn and register it.

        *epoch* is the settings change *settings* were read after.
        """
        extras = extras or TurnExtras()
        try:
            # MCP is deferred (U3 decision 5 B): the builder gets no config.
            agent, model = await self._build_agent(settings, path, None, info.id)
        except BaseException:
            await self._drop_unopened(info.id)
            raise
        runner = SessionRunner(
            agent,
            self._store,
            info.id,
            holder=self._holder,
            resume_mode=DEFAULT_RESUME_MODE,
            progress=True,
            pictures=True,
            memory=self._memory,
            inbox=self._inbox(info.id),
            # A create's key stays where a create looks it up, even though
            # the row was made before the Agent.
            create_scope_key=client_request_id is not None,
            before_turn=self._before_turn(info.id),
            live=self._live(info.id),
        )
        policy = ChatPolicy(
            runner.observe(self._timed_ask(info.id)),
            chat_mode(info.meta),
            saved=self.saved_folders,
            watch=self.card_watch(info.id),
            grants=self.grants_view(info),
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
                    attachments=extras.attachments,
                    images=extras.images,
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
            agent=agent,
            lock=self.lock(session_id),
            last_used=self._clock(),
            settings_epoch=epoch,
            model=model,
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
        extras: TurnExtras | None = None,
        words: str | None = None,
    ) -> tuple[TurnInfo, ResumeInfo | None]:
        """Start a turn in *session_id*; return it and the open's resume.

        *extras* are the files and pictures the message carries. *words* are
        the person's own words when *composed* holds more than they typed (an
        older queued message has its files in its text): the run reads these
        as theirs, never a file's text (#626).

        The resume is the ``ResumeInfo`` of an open this call made, for the
        route's 202, else ``None``. The key is checked before the session is
        opened, so a repeat never builds an Agent.
        """
        extras = extras or TurnExtras()
        async with self.lock(session_id):
            if client_request_id is not None:
                found = await self._store.get_submission(session_id, client_request_id)
                check_attachment_submission(
                    found, extras.attachments, composed if extras.attachments else None
                )
                check_submission(found, compose_user_text(composed, extras.attachments))
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
                attachments=extras.attachments,
                words=words,
                images=extras.images,
            )
            entry.last_used = self._clock()
            return turn, resumed

    async def _undo_newest(
        self, session_id: str, turn_id: str, words: str, *, retry: bool = False
    ) -> tuple[OpenSession, TurnInfo]:
        """Undo the newest finished turn (under the session's lock), then resume.

        409 ``session_busy`` while a turn runs, and 409 ``not_latest_turn``
        (with the catalog's *words*) for a turn that is not the newest
        finished one. With *retry*, a turn the person did not type (a task's
        question) is refused: its words are not theirs to send.
        """
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
            raise ApiError(409, "not_latest_turn", t(words))
        if retry and latest.origin:
            raise ApiError(409, "not_retryable", t("errors.retryAsk"))
        undone = await entry.runner.undo()
        if undone is None:
            raise ApiError(409, "not_latest_turn", t(words))
        return entry, undone

    async def start_internal(self, session_id: str, text: str, origin: str) -> TurnInfo:
        """Begin a turn the person did not type (a task's question, #369) in an idle chat.

        :class:`SessionBusyError` while a turn runs. The turn has the chat's
        mode; the chat does not take the scheduler's focus for it.
        """
        async with self.lock(session_id):
            entry = await self.open(session_id)
            if entry.runner.busy:
                raise SessionBusyError(f"Session {session_id} is running a turn")
            entry.resumed = None
            if self._gate is not None:
                self._gate.turn_start(session_id)
            turn = await entry.runner.start_turn(
                text, approval=entry.policy, origin=origin
            )
            entry.last_used = self._clock()
            return turn

    async def retry(
        self, session_id: str, turn_id: str
    ) -> tuple[TurnInfo, ResumeInfo | None]:
        """Undo the newest finished turn and run its text again (``/retry``).

        409 ``session_busy`` while a turn runs, and 409 ``not_latest_turn``
        for a turn that is not the newest finished one. The resume is the one
        ``undo()`` ended with.
        """
        async with self.lock(session_id):
            entry, undone = await self._undo_newest(
                session_id, turn_id, "errors.retryNewest", retry=True
            )
            resumed = entry.runner.last_resume
            await self._reconcile(entry)
            self._note_turn_start(session_id)
            # The same message again, as it was sent: its typed words and its files
            # apart, and the pictures they added (#626). A turn stored before files
            # were kept apart has them in its text: its words are what comes before
            # the first of them, never a file's text, however many times it is tried
            # again.
            pictures = await self._store.turn_images(session_id, undone.turn)
            text, words = undone.user_text, None
            if undone.attachments:
                text = undone.typed_text
            else:
                head = words_before_files(text)
                words = None if head == text else head
            turn = await entry.runner.start_turn(
                text,
                approval=entry.policy,
                attachments=undone.attachments or (),
                words=words,
                images=pictures,
            )
            entry.last_used = self._clock()
            return turn, resumed

    async def undo(self, session_id: str, turn_id: str) -> TurnInfo:
        """Forget the newest finished turn (``/undo``); its file changes stay.

        The errors are :meth:`retry`'s. The turn comes back as the store
        held it when it was undone.
        """
        async with self.lock(session_id):
            entry, undone = await self._undo_newest(
                session_id, turn_id, "errors.undoNewest"
            )
            await self._reconcile(entry)
            entry.last_used = self._clock()
            return undone

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
                # A chat in another folder starts without its grants (the main
                # chat's still hold), open or not.
                await self._forget_moved(session_id)
                return info
            if entry.runner.busy:
                raise ApiError(
                    409,
                    "session_busy",
                    t("errors.turnRunning"),
                )
            epoch = self._settings_epoch
            settings = self._settings()
            old_root = entry.agent.workspace_root
            agent, model = await self._build_agent(
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
            entry.settings_epoch = epoch
            entry.model = model
            with contextlib.suppress(Exception):
                await old.close()
            await self._forget_moved(session_id)
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
        """Record on the session the chat model its Agent asks for, when it changed."""
        model = entry.model
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

    # ── Settings changes ──

    async def settings_changed(self) -> None:
        """Close every idle Agent at once, keeping each conversation.

        An Agent that is busy (a turn runs, a question waits, an open is
        queued) keeps the settings it was built with until its turn ends: the
        chat's next open builds it again (:meth:`_open_locked`).
        """
        self._settings_epoch += 1
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
            # Its grants' rows went with it (ON DELETE CASCADE); the checks forget them now.
            self.book.forget_session(session_id)
            self._modes.pop(session_id, None)
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
            # Its work folder (the files its pages handed over) goes with it (#551).
            await asyncio.to_thread(work_folder.remove, session_id)
        # The lock stays: a turn that queued on it before the delete holds
        # this object, and ``open`` checks that the lock it is under is the
        # session's. The dict keeps one small lock per session this process
        # has opened; a session id is never reused.

    async def close_all(self) -> None:
        """Stop every turn and close every Agent within the shutdown budget."""
        self._closed = True
        for watch in list(self._allowance_watches):
            watch.cancel()
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
        await self.book.flush()  # a refused take-back has one more try
