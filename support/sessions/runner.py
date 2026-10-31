"""Run a session's turns on an Agent, store their events and resume them.

:class:`SessionRunner` puts one session's turns through one Agent, one at a
time. A turn begins with a committed row (``SessionStore.begin_turn``), runs
as an interactive run (``Agent.run_task`` with a sink), has its events
stored in batches by a writer task while a lease task keeps the turn's
lease, and ends exactly once with ``SessionStore.finish_turn``, which also
stores the window checkpoint. A Stop, a close, a shutdown or a takeover
ends the run ``cancelled``, with a RunResult and a closed window, and
records what ended the turn as its cause.

:meth:`SessionRunner.resume` loads a stored conversation into the Agent by
one of two arms: ``pairs`` rebuilds the window from each turn's user text
and answer, ``window`` restores the checkpoint and adds the later turns as
pairs.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import time
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, cast, get_args

from localmelo.melo.agent import Agent
from localmelo.melo.agent.agent import CANCELLED_ERROR, FINALIZE_TIMEOUT_S
from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
from localmelo.melo.contracts.events import (
    EVENT_SCHEMA_VERSION,
    AgentEvent,
    json_safe_payload,
)
from localmelo.melo.contracts.inbox import InboxSource
from localmelo.melo.schema import (
    EMPTY_REPLY_NUDGE,
    ImagePart,
    Message,
    RunResult,
    answer_text,
    clip_session_text,
    split_thinking,
)
from localmelo.support.sessions import pictures as turn_pictures
from localmelo.support.sessions.approvals import AskFunction
from localmelo.support.sessions.attachments import TextAttachment, compose_user_text
from localmelo.support.sessions.store import (
    APPROVAL_EVENT_TYPES,
    CAUSES,
    CLOSED_ERROR,
    LEASE_S,
    MEMORY_EVENT_TYPE,
    PICTURES_EVENT_TYPE,
    RESERVED_EVENT_PREFIXES,
    RESERVED_EVENT_TYPE_RE,
    SESSION_EVENT_FIELDS,
    TAKEOVER_ERROR,
    WRITER_BATCH_MAX,
    Cause,
    NewSession,
    SessionBusyError,
    SessionStore,
    SessionStoreError,
    StoredEvent,
    TurnInfo,
    UnknownSessionError,
    check_attachment_submission,
    process_holder,
    redacted_event,
)
from localmelo.support.sessions.submissions import check_submission
from localmelo.support.sessions.window import WINDOW_CODEC_VERSION, decode_window
from localmelo.support.sessions.worker import redact_worker, sensitive_read

if TYPE_CHECKING:
    from localmelo.support.memory import ChatMemory

logger = logging.getLogger(__name__)

EventCallback = Callable[[StoredEvent], Awaitable[None] | None]
ResumeMode = Literal["pairs", "window"]
ResumeLoader = Callable[[], Awaitable[int]]
BeforeTurn = Callable[[ResumeLoader], Awaitable[None]]

# Seconds an awaitable on_event result may take; a slower one is cut off.
SINK_TIMEOUT_S: Final = 5.0
# WRITER_BATCH_MAX, the events of one append_events transaction, is the
# store's, imported above.
RESUME_MAX_TURNS: Final = 10
# Characters of user text and answers that a pairs resume loads at most.
RESUME_BUDGET_CHARS: Final = 32_000
DEFAULT_RESUME_MODE: Final[ResumeMode] = "pairs"
# Delays before each retry of a CLI turn's finish_turn.
CLI_FINISH_RETRY_DELAYS_S: Final = (0.5, 1.0, 2.0)
# The delay before the one retry of a gateway turn's finish_turn.
GATEWAY_FINISH_RETRY_DELAYS_S: Final = (1.0,)
# The error of a turn that another terminal took over; recovery writes it too.
TAKEOVER_REASON: Final = TAKEOVER_ERROR

# Seconds between the lease task's refreshes, whatever the run is doing.
_LEASE_REFRESH_S: Final = LEASE_S / 3
_RESUME_MODES: Final = frozenset(get_args(ResumeMode))
# Causes that only the store or a run gives, never a cancel.
_NOT_A_STOP: Final = frozenset({"run", "crash"})
# What the writer finds on the queue: an event, a flush marker, the end.
_STOP: Final = ("stop",)
# The finish_turn arguments of each CLI turn whose end no try could store, by
# store and session. This process's own lease never lapses while it lives, so
# every runner of the session tries the end again before the session's next
# turn or undo (in the gateway's file the next process's open recovers it).
_LOST_ENDS: weakref.WeakKeyDictionary[SessionStore, dict[str, dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True)
class ResumeInfo:
    """What the last :meth:`SessionRunner.resume` loaded."""

    mode: ResumeMode  # the arm that loaded the window
    fell_back: bool  # True when "window" was asked for and pairs were loaded
    turns: int  # user queries the load kept (EMPTY_REPLY_NUDGE not counted)
    checkpoint_turns: int  # of those, the window checkpoint's (0 for "pairs")
    messages: int  # load_conversation's return value
    chars: int  # characters of content loaded


@dataclass(eq=False)
class _Turn:
    """One turn under way, from ``begin_turn`` to the end of its step 7."""

    turn: TurnInfo
    agent: Agent
    text: str
    approval: ApprovalPolicy | None
    # What the person typed, without attachments: memory reads only this.
    typed: str = ""
    # The run's routine name, run_task's ``routine=`` (UR-N).
    routine: str | None = None
    queue: asyncio.Queue[tuple[Any, ...]] = field(default_factory=asyncio.Queue)
    # The task that runs the turn: the runner's own (start_turn) or the
    # caller's (run_turn).
    task: asyncio.Task[Any] | None = None
    # Whether _drive began in that task; a task cancelled before its first
    # step never runs its coroutine.
    started: bool = False
    # What a cancel by the runner recorded: the first one counts.
    cause: Cause | None = None
    reason: str = ""
    # task.cancel() calls by the runner, which it takes back with uncancel().
    runner_cancels: int = 0
    closing: bool = False
    # The run, as the sink saw it.
    saw_started: bool = False
    run_id: str = ""
    outcome: RunResult | None = None
    # The RunResult of a turn that ended without run.finished (step 7.4).
    fallback: RunResult | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    # The checked text of the last digest note event of this turn (R4).
    last_note: str = ""
    # How memory's recall went for this turn (``turn.memory``'s ``recall``).
    recall: dict[str, Any] | None = None
    # Whether the writer still takes events; it stops at step 7.2.
    accepting: bool = True
    writer: asyncio.Task[None] | None = None
    lease: asyncio.Task[None] | None = None
    finish: asyncio.Future[None] | None = None
    # Set once step 7 has ended.
    done: asyncio.Event = field(default_factory=asyncio.Event)


def _check_agent(agent: Agent) -> None:
    if not agent.options.record_answers:
        raise ValueError(
            "SessionRunner needs an Agent built with AgentOptions(record_answers=True)"
        )
    if agent.workspace_root is None:
        raise ValueError("SessionRunner needs an Agent built with a workspace_root")


def _check_cause(cause: object) -> Cause:
    if cause not in CAUSES or cause in _NOT_A_STOP:
        stops = ", ".join(sorted(CAUSES - _NOT_A_STOP))
        raise ValueError(f"cause must be one of {stops}, got {cause!r}")
    return cast(Cause, cause)


def _exception_text(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


async def _settle(future: asyncio.Future[Any]) -> None:
    """Wait until *future* is done, through cancellations of this task.

    The caller re-raises the cancellation once it returns: the work in
    *future*, such as a committed turn row, must not be left half known.
    """
    while not future.done():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(future)


class SessionRunner:
    """Runs one session's turns on *agent*, one at a time, and stores them.

    *session_id* names an existing session; with ``None`` and
    *new_session*, the first turn creates the session in its own
    transaction. *on_event* gets each stored event, with its durable id,
    after it is committed; an awaitable result is awaited for at most
    :data:`SINK_TIMEOUT_S`. *holder* names this process in leases
    (:func:`~localmelo.support.sessions.store.process_holder` by default).
    *resume_mode* picks the arm :meth:`resume` loads. *progress* asks each
    turn's run for ``run.progress`` events (``Agent.run_task``'s
    ``progress=``); the flag stays on the runner, :meth:`replace_agent`
    included. *before_turn*, when set, is awaited after acquiring the turn's
    lease and before calling the model. It receives a reload coroutine
    factory for synchronizing conversation under that lease; do not retain it.
    With *pictures*, each turn looks at the Agent's folder before and after its
    run and stores the picture files it made or changed, and those its answer
    names, as one ``turn.pictures`` event (:mod:`~.pictures`). With *memory*,
    each turn of a chat recalls what matches its typed text and keeps what it
    asks to remember (a ``turn.memory`` event, queued again whole when the
    model keeps a note with ``remember``, or when, after a completed run with
    ``[memory] keep_statements`` on, code keeps a lasting fact the user stated
    that no note kept), and a completed turn is indexed; *index_only* (a
    task's runner) only indexes.

    The Agent needs ``AgentOptions(record_answers=True)`` and a
    ``workspace_root``, and the caller awaits ``agent.start()`` before it
    passes the Agent here.
    """

    def __init__(
        self,
        agent: Agent,
        store: SessionStore,
        session_id: str | None,
        *,
        new_session: NewSession | None = None,
        on_event: EventCallback | None = None,
        holder: str | None = None,
        resume_mode: ResumeMode = DEFAULT_RESUME_MODE,
        progress: bool = False,
        before_turn: BeforeTurn | None = None,
        create_scope_key: bool = False,
        observe: Callable[[AgentEvent], object] | None = None,
        inbox: InboxSource | None = None,
        pictures: bool = False,
        memory: ChatMemory | None = None,
        index_only: bool = False,
    ) -> None:
        self._worker = new_session is not None and new_session.kind == "worker"
        if not self._worker:
            _check_agent(agent)
        if resume_mode not in _RESUME_MODES:
            raise ValueError(
                f"resume_mode must be 'pairs' or 'window', got {resume_mode!r}"
            )
        if (session_id is None) == (new_session is None):
            raise ValueError(
                "SessionRunner takes a session_id or a new_session, not both"
            )
        if on_event is not None and not callable(on_event):
            raise TypeError("on_event must be callable")
        self._agent = agent
        self._store = store
        self._session_id = session_id
        self._new_session = new_session
        self._on_event = on_event
        self._observe = observe
        self._inbox = inbox
        self._pictures = pictures
        self._memory = None if self._worker else memory
        self._index_only = index_only
        self._sensitive_reads: dict[str, str] = {}
        self._holder = holder or process_holder(store.writer)
        self._resume_mode: ResumeMode = resume_mode
        # Whether each turn's run is asked for run.progress (UR-N); kept
        # for every turn, replace_agent's included.
        self._progress = progress
        self._before_turn = before_turn
        # One-shot: the key of the session's first turn belongs to a create
        # request, so it is stored where a create looks it up (the console's
        # create makes its session row before the Agent, so the gate knows
        # the id before the first model call).
        self._create_scope_key = create_scope_key
        self._active: _Turn | None = None
        self._last: _Turn | None = None
        # A begin_turn under way, before its turn exists.
        self._begin: asyncio.Future[_Turn] | None = None
        # A turn whose run's finalizer outlived step 7.1's wait.
        self._finishing: _Turn | None = None
        # Held while undo() or resume() loads the conversation; a turn waits
        # for it, so that it runs on the conversation they leave.
        self._loading = asyncio.Lock()
        self._last_resume: ResumeInfo | None = None
        self._closed = False
        self._close_reason = "Session closed"
        self._close_cause: Cause = "closed"

    # ── State ──

    @property
    def session_id(self) -> str | None:
        """The session's id; ``None`` until a new session's first turn has begun."""
        return self._session_id

    @property
    def busy(self) -> bool:
        """Whether a turn runs, begins, or still finishes its run."""
        return (
            self._active is not None
            or self._begin is not None
            or self._finishing is not None
        )

    @property
    def run_ended(self) -> bool:
        """Whether the running turn's run has ended; the turn may still finish.

        True from the moment the run sends ``run.finished`` (before that
        event is stored and forwarded) or returns.
        """
        turn = self._active
        return turn is not None and (turn.finished.is_set() or turn.finish is not None)

    @property
    def current_turn(self) -> TurnInfo | None:
        return None if self._active is None else self._active.turn

    @property
    def last_resume(self) -> ResumeInfo | None:
        return self._last_resume

    def _name(self) -> str:
        return self._session_id or "(new)"

    def _check_idle(self) -> None:
        if self._active is not None or self._begin is not None:
            raise SessionBusyError(f"Session {self._name()} is already running a turn")
        if self._finishing is not None:
            raise SessionBusyError(
                f"Session {self._name()} is still finishing its last turn"
            )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("The session runner is closed")

    def replace_agent(self, agent: Agent) -> None:
        """Run the next turns on *agent*, which the caller has started.

        The runner, its policy and their grants stay. The caller moves the
        conversation and closes the old Agent.
        """
        _check_agent(agent)
        self._check_idle()
        self._agent = agent

    # ── Turns ──

    async def run_turn(
        self,
        text: str,
        *,
        approval: ApprovalPolicy | None = None,
        title: str | None = None,
        routine: str | None = None,
        attachments: Sequence[TextAttachment] = (),
    ) -> RunResult:
        """Run a turn of *text* in this task and return its RunResult.

        *title* is the typed text, for the session's first title. *routine*
        names the run's routine (``Agent.run_task``'s ``routine=``), passed
        only when given. A turn that this runner stopped (:meth:`cancel`,
        :meth:`close`, a takeover) returns the ``cancelled`` RunResult of
        its ``run.finished``, or, when it was stopped before its run
        began, a ``cancelled`` RunResult with the stop's reason and no
        counters. The runner takes back the cancellations it caused, so the
        task is no longer cancelling. A cancellation from outside the runner
        propagates once the turn has ended, or at once when it comes while
        the turn ends, which then ends on its own.
        """
        turn = await self._start(
            text, approval, title, None, routine, tuple(attachments), own_task=False
        )
        if turn.finish is not None:  # stopped before its run began
            task = asyncio.current_task()
            assert task is not None
            return self._outcome(turn, task, None, None)
        return await self._drive(turn)

    async def start_turn(
        self,
        text: str,
        *,
        approval: ApprovalPolicy | None = None,
        title: str | None = None,
        client_request_id: str | None = None,
        routine: str | None = None,
        attachments: Sequence[TextAttachment] = (),
    ) -> TurnInfo:
        """Begin a turn of *text* and return it once its row exists.

        The turn runs on in a task the runner owns; :meth:`wait` returns its
        RunResult. *routine* names the run's routine (``Agent.run_task``'s
        ``routine=``), passed only when given. With *client_request_id*, a
        key already accepted raises ``SubmissionRepeated`` with the stored
        turn and starts nothing, a withdrawn one ``SubmissionWithdrawn``,
        and one used for another text ``SubmissionConflict``.
        """
        turn = await self._start(
            text,
            approval,
            title,
            client_request_id,
            routine,
            tuple(attachments),
            own_task=True,
        )
        return turn.turn

    async def wait(self) -> RunResult | None:
        """Wait for the last turn to end and return its RunResult.

        That is the RunResult of its ``run.finished``, or, for a turn that
        ended without one, the one :meth:`run_turn` would return; ``None``
        when no turn has begun.
        """
        turn = self._last
        if turn is None:
            return None
        await turn.done.wait()
        return turn.outcome or turn.fallback

    async def _start(
        self,
        text: str,
        approval: ApprovalPolicy | None,
        title: str | None,
        client_request_id: str | None,
        routine: str | None,
        attachments: tuple[TextAttachment, ...],
        *,
        own_task: bool,
    ) -> _Turn:
        self._check_open()
        typed_text = text
        if attachments:
            if self._worker:
                raise ValueError("worker sessions do not support attachments")
            text = compose_user_text(text, attachments)
        if self._worker and self._last is not None:
            raise ValueError("worker sessions have only one turn")
        # 0. The key decides before the busy check, so a repeat of the
        # running turn gets that turn.
        if client_request_id is not None:
            await self._check_key(text, client_request_id, attachments, typed_text)
        # 1. Checks.
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be blank")
        if approval is not None and not (
            isinstance(approval, ApprovalPolicy)
            and inspect.iscoroutinefunction(approval.decide)
        ):
            raise TypeError("approval must have an async decide(request) method")
        while True:
            self._check_open()
            begin = self._begin
            if begin is not None and client_request_id is not None:
                # A begin_turn under way may accept this very key: once it
                # has ended, the key decides again.
                await asyncio.wait({begin})
                await self._check_key(text, client_request_id, attachments, typed_text)
            elif self._loading.locked():
                # An undo or a resume loads the conversation: the turn runs
                # on the conversation it leaves.
                async with self._loading:
                    pass
            else:
                break
        self._check_idle()
        # 2. The turn row, in a task of its own: once begin_turn commits, its
        # turn must get a runner even when this task is cancelled meanwhile.
        task = asyncio.current_task()
        assert task is not None
        caller = None if own_task else task
        begin = asyncio.ensure_future(
            self._begin_turn(
                text,
                approval,
                title,
                client_request_id,
                routine,
                caller,
                attachments,
                typed_text,
            )
        )
        self._begin = begin
        try:
            return await asyncio.shield(begin)
        except asyncio.CancelledError as exc:
            await _settle(begin)
            if begin.cancelled() or begin.exception() is not None or own_task:
                raise
            # This task was to run the turn, and it is being cancelled: by
            # the runner (a Stop, a close) or from outside. The turn ends at
            # once, without a run.
            turn = begin.result()
            await _settle(self._start_finish(turn, exc, cancelled=True))
            if task.cancelling() > turn.runner_cancels:
                _uncancel(task, turn.runner_cancels)
                raise
            return turn  # run_turn returns its RunResult

    async def _check_key(
        self,
        text: str,
        client_request_id: str,
        attachments: tuple[TextAttachment, ...],
        typed_text: str,
    ) -> None:
        """Step 0: raise what ``begin_turn`` step 1 would for this key."""
        found = await self._store.get_submission(self._session_id, client_request_id)
        check_attachment_submission(found, attachments, typed_text)
        check_submission(found, text)

    async def _begin_turn(
        self,
        text: str,
        approval: ApprovalPolicy | None,
        title: str | None,
        client_request_id: str | None,
        routine: str | None,
        caller: asyncio.Task[Any] | None,
        attachments: tuple[TextAttachment, ...],
        typed_text: str,
    ) -> _Turn:
        try:
            # The last turn's end must be stored before the next turn: in a
            # CLI file, this process's own lease never lapses while it lives.
            await self._record_lost_end()
            info = await self._store.begin_turn(
                self._session_id,
                redact_worker(text) if self._worker else text,
                holder=self._holder,
                title=redact_worker(title) if self._worker else title,
                client_request_id=client_request_id,
                new_session=self._new_session if self._session_id is None else None,
                create_scope=self._create_scope_key,
                attachments=attachments,
                typed_text=typed_text if attachments else None,
            )
            self._create_scope_key = False
        finally:
            self._begin = None
        # From here to the return, no await: a cancel by key that finds the
        # submission accepted finds its turn.
        self._session_id = info.session_id
        self._new_session = None
        turn = _Turn(
            turn=info,
            agent=self._agent,
            text=text,
            approval=approval,
            routine=routine,
            typed=typed_text,
        )
        turn.writer = asyncio.create_task(self._write_events(turn))
        turn.lease = asyncio.create_task(self._keep_lease(turn))
        self._active = turn
        self._last = turn
        if caller is None:
            turn.task = asyncio.create_task(self._drive(turn))
            turn.task.add_done_callback(functools.partial(self._turn_task_done, turn))
        else:
            turn.task = caller
        if self._closed:
            turn.closing = True
            self._cancel_turn(turn, self._close_reason, self._close_cause)
        return turn

    def _turn_task_done(self, turn: _Turn, task: asyncio.Task[Any]) -> None:
        if not turn.started:
            # Cancelled before its first step: the coroutine never ran.
            self._start_finish(turn, None, cancelled=True)
        if not task.cancelled():
            task.exception()  # retrieved: wait() reports the outcome

    def _sink(self, turn: _Turn) -> Callable[[AgentEvent], None]:
        """The run's synchronous sink: it queues each event for the writer."""

        def sink(event: AgentEvent) -> None:
            if self._observe is not None:
                try:
                    self._observe(event)
                except Exception:
                    logger.exception("Worker observer failed on a %s event", event.type)
            try:
                if event.type == "run.started":
                    turn.saw_started = True
                    turn.run_id = event.run_id
                elif event.type == "run.finished":
                    turn.outcome = RunResult(**event.data["result"])
                    turn.finished.set()
                    if self._finishing is turn:
                        self._finishing = None
                if event.type == "note":
                    data = event.data
                    if data.get("kind") == "digest" and isinstance(
                        data.get("text"), str
                    ):
                        turn.last_note = data["text"]
                if turn.accepting:
                    payload = event.to_dict()
                    images = event.images
                    if self._worker:
                        payload = redact_worker(payload)
                        data = payload.get("data", {})
                        if event.type == "tool.started":
                            path = sensitive_read(
                                str(event.data.get("name", "")),
                                event.data.get("arguments"),
                                turn.agent.workspace_root,
                            )
                            if path:
                                self._sensitive_reads[str(data.get("call_id", ""))] = (
                                    path
                                )
                        elif event.type == "tool.finished":
                            path = self._sensitive_reads.pop(
                                str(data.get("call_id", "")), None
                            )
                            if path:
                                data["output"] = (
                                    f"[Sensitive file content omitted: {path}]"
                                )
                                data["output_chars"] = len(data["output"])
                                data["error"] = ""
                                data["images"] = []
                                images = ()
                        elif event.type == "run.finished" and isinstance(
                            data.get("result"), dict
                        ):
                            data["result"].pop("config", None)
                    turn.queue.put_nowait(("event", payload, images))
            except Exception:
                logger.exception("Could not queue a %s event", event.type)

        return sink

    async def _drive(self, turn: _Turn) -> RunResult:
        """Steps 6 and 7 of a turn, in the task that runs it."""
        turn.started = True
        task = asyncio.current_task()
        assert task is not None
        run_exc: BaseException | None = None
        returned: RunResult | None = None
        before: turn_pictures.Snapshot | None = None
        try:
            if self._before_turn is not None:
                await self._before_turn(
                    functools.partial(
                        self._resume,
                        RESUME_MAX_TURNS,
                        reserved=True,
                    )
                )
            kwargs: dict[str, Any] = {
                "observe" if self._worker else "on_event": self._sink(turn),
                "approval": turn.approval,
                "progress": self._progress,
            }
            if turn.routine is not None:
                kwargs["routine"] = turn.routine
            if self._inbox is not None:
                kwargs["inbox"] = self._inbox
            text = turn.text
            if self._memory is not None and not self._index_only:
                text = await self._recall(turn)
            before = await self._scan_folder(turn)
            returned = await turn.agent.run_task(text, **kwargs)
        except BaseException as exc:
            run_exc = exc
        if before is not None:
            await self._note_pictures(turn, before, returned)
        if returned is not None and returned.status == "completed":
            await self._keep_statements(turn)
        cancelled = (
            isinstance(run_exc, asyncio.CancelledError) and task.cancelling() > 0
        )
        finish = self._start_finish(turn, run_exc, cancelled=cancelled)
        while not finish.done():
            try:
                await asyncio.shield(finish)
            except asyncio.CancelledError:
                if task.cancelling() > turn.runner_cancels:
                    break  # cancelled from outside again: step 7 goes on alone
        return self._outcome(turn, task, run_exc, returned)

    async def _recall(self, turn: _Turn) -> str:
        """The turn's text with what memory recalls for it; queues ``turn.memory``.

        A note the model keeps during the run queues the event again, whole.
        """
        assert self._memory is not None
        info = turn.turn

        def emit(data: Mapping[str, Any]) -> None:
            ids = {"turn": info.turn, "turn_id": info.id}
            self._enqueue(turn, MEMORY_EVENT_TYPE, {**ids, **data})

        try:
            found = await self._memory.before_turn(
                self._store, info, turn.typed, emit=emit
            )
        except Exception:
            logger.warning("Memory across chats failed for this turn", exc_info=True)
            return turn.text
        fits = len(turn.text) + len(found.block) <= MAX_QUERY_LEN
        if found.recall is not None and not fits:
            found.recall.block_chars = 0  # a block too long for the message is left out
        data = found.event()
        if data is not None:
            turn.recall = data["recall"]
            emit(data)
        if found.block and fits:
            return turn.text + found.block
        return turn.text

    async def _keep_statements(self, turn: _Turn) -> None:
        """Keep the lasting facts the user stated that the model kept no note of.

        It queues ``turn.memory`` again, whole, when it keeps one. A cancel
        does not stop a write under way.
        """
        if self._memory is None or self._index_only:
            return
        try:
            await asyncio.shield(self._memory.keep_statements(turn.turn))
        except asyncio.CancelledError:
            return  # the turn's own cancel counter still ends the run as cancelled
        except Exception:
            logger.warning("Could not keep what the user stated", exc_info=True)

    async def _scan_folder(self, turn: _Turn) -> turn_pictures.Snapshot | None:
        """The picture files in the Agent's folder, or ``None`` when pictures are off."""
        root = turn.agent.workspace_root
        if not self._pictures or root is None:
            return None
        try:
            return await asyncio.to_thread(turn_pictures.scan, root)
        except Exception:
            logger.warning("Could not look at the folder for pictures", exc_info=True)
            return None

    async def _note_pictures(
        self, turn: _Turn, before: turn_pictures.Snapshot, returned: RunResult | None
    ) -> None:
        """Queue ``turn.pictures``: what the run made or changed, and what its answer names."""
        root = turn.agent.workspace_root
        outcome = turn.outcome or returned
        answer = split_thinking(outcome.result)[1] if outcome is not None else ""
        assert root is not None
        try:
            found = await asyncio.to_thread(self._pictures_of, root, before, answer)
        except asyncio.CancelledError:
            return  # the turn's own cancel counter still ends the run as cancelled
        except Exception:
            logger.warning("Could not list the pictures of a turn", exc_info=True)
            return
        if found:
            self._enqueue(
                turn,
                PICTURES_EVENT_TYPE,
                {"turn": turn.turn.turn, "turn_id": turn.turn.id, "pictures": found},
            )

    @staticmethod
    def _pictures_of(
        root: str, before: turn_pictures.Snapshot, answer: str
    ) -> list[dict[str, Any]]:
        made = turn_pictures.new_pictures(before, turn_pictures.scan(root))
        named = turn_pictures.mentioned(root, answer, {item["path"] for item in made})
        return (made + named)[: turn_pictures.MAX_LISTED]

    def _outcome(
        self,
        turn: _Turn,
        task: asyncio.Task[Any],
        run_exc: BaseException | None,
        returned: RunResult | None,
    ) -> RunResult:
        """Step 7.6 in *task*, which ran the turn: its RunResult, or a re-raise.

        The runner takes back the cancellations it caused (:meth:`cancel`,
        :meth:`close`, the lease task). A ``CancelledError`` propagates only
        while the task is still being cancelled from outside the runner; one
        that a component raised while nothing cancelled the task ended the
        run ``crashed``, and its RunResult is returned.
        """
        _uncancel(task, turn.runner_cancels)
        if run_exc is not None and not isinstance(run_exc, asyncio.CancelledError):
            raise run_exc
        if task.cancelling() > 0:
            if isinstance(run_exc, asyncio.CancelledError):
                raise run_exc
            raise asyncio.CancelledError()
        if returned is not None:
            return returned
        if turn.outcome is not None:
            return turn.outcome
        if turn.fallback is not None:
            return turn.fallback
        # Step 7 failed before it took the turn's status: the end is unknown.
        return RunResult(
            task_id=turn.run_id,
            status="crashed",
            error="The turn ended without a RunResult",
        )

    def _start_finish(
        self, turn: _Turn, run_exc: BaseException | None, *, cancelled: bool
    ) -> asyncio.Future[None]:
        if turn.finish is None:
            turn.finish = asyncio.ensure_future(self._finish(turn, run_exc, cancelled))
        return turn.finish

    async def _finish(
        self, turn: _Turn, run_exc: BaseException | None, cancelled: bool
    ) -> None:
        """Step 7: wait for run.finished, drain, record the end, stop the lease."""
        try:
            # 1. run.finished, which a finalizer may send after a second
            # cancellation.
            if turn.saw_started and not turn.finished.is_set():
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(FINALIZE_TIMEOUT_S + 1):
                        await turn.finished.wait()
            # 2. The writer stores and forwards what is queued.
            turn.accepting = False
            turn.queue.put_nowait(_STOP)
            if turn.writer is not None:
                await asyncio.wait({turn.writer})
            # 3. and 4. The window and the end.
            await self._record_end(turn, run_exc, cancelled)
            outcome = turn.outcome
            if self._memory is not None and outcome and outcome.status == "completed":
                answer = split_thinking(outcome.result)[1]
                try:
                    self._memory.after_turn(self._store, turn.turn, turn.typed, answer)
                except Exception:
                    logger.warning("Could not index a turn for memory", exc_info=True)
        except Exception:
            logger.exception(
                "Could not end turn %d of session %s",
                turn.turn.turn,
                turn.turn.session_id,
            )
        finally:
            # 5. The lease task stops; nothing here awaits, so a second
            # cancellation cannot skip the end of step 7.
            if turn.lease is not None:
                turn.lease.cancel()
            if turn.saw_started and not turn.finished.is_set():
                self._finishing = turn  # until the sink sees run.finished
            if self._active is turn:
                self._active = None
            turn.done.set()

    async def _record_end(
        self, turn: _Turn, run_exc: BaseException | None, cancelled: bool
    ) -> None:
        outcome = turn.outcome
        info = turn.turn
        window: list[Message] | None = None
        result: dict[str, Any] | None = None
        if outcome is not None:
            # _record_answer ran before the finalizer or the final save, so
            # the window ends with this turn's answer or INTERRUPT_LABEL.
            window = None if self._worker else turn.agent.conversation()
            status = outcome.status
            result = outcome.to_dict()
            answer = answer_text(status, outcome.result)
            error = outcome.error
            cause = self._cause(turn) if status == "cancelled" else "run"
        elif cancelled or turn.cause is not None:
            status, answer = "cancelled", answer_text("cancelled", "")
            cause = self._cause(turn)
            error = turn.reason
            if not error and isinstance(run_exc, asyncio.CancelledError):
                message = run_exc.args[0] if run_exc.args else None
                error = message if isinstance(message, str) else ""
            error = error or CANCELLED_ERROR
        else:
            status, answer, cause = "crashed", answer_text("crashed", ""), "run"
            error = (
                _exception_text(run_exc)
                if run_exc is not None
                else "The run ended without run.finished"
            )
        if outcome is None:
            turn.fallback = RunResult(task_id=turn.run_id, status=status, error=error)
            # The result column carries the reason too: a caller that reads the
            # saved turn (instead of turn.finished) must see why it ended.
            result = turn.fallback.to_dict()
        if turn.recall is not None and result is not None:
            result["memory_recall"] = turn.recall  # per-turn timings, beside duration_s
        end: dict[str, Any] = {
            "session_id": info.session_id,
            "turn": info.turn,
            "holder": self._holder,
            "status": status,
            "cause": cause,
            "result": result,
            "answer": answer,
            "window": window,
            "error": error,
            "note": turn.last_note,
        }
        if self._worker:
            for key in ("result", "answer", "error", "note"):
                end[key] = redact_worker(end[key])
            if isinstance(end["result"], dict):
                end["result"].pop("config", None)
        delays: Sequence[float] = GATEWAY_FINISH_RETRY_DELAYS_S
        if self._store.writer == "cli":
            closing = turn.closing or cause == "shutdown"
            delays = () if closing else CLI_FINISH_RETRY_DELAYS_S
        for attempt in range(len(delays) + 1):
            # The committed turn.finished (and any withdrawn approval), for
            # the sink: the renderer takes a cancelled turn's cause from it.
            stored: list[StoredEvent] = []
            try:
                await self._store.finish_turn(**end, on_stored=stored.append)
                for event in stored:
                    await self._forward(event)
                return
            except Exception as exc:
                final = attempt == len(delays) or _cannot_retry(exc)
                if final:
                    kept = self._store.writer == "cli" and not _cannot_retry(exc)
                    if kept:
                        _LOST_ENDS.setdefault(self._store, {})[info.session_id] = end
                    logger.error(
                        "Could not record the end of turn %d of session %s; %s: %s",
                        info.turn,
                        info.session_id,
                        (
                            "it is tried again before the session's next turn"
                            if kept
                            else "the next open recovers it"
                        ),
                        _exception_text(exc),
                    )
                    return
                logger.warning(
                    "Could not record the end of turn %d of session %s; trying again "
                    "in %g s: %s",
                    info.turn,
                    info.session_id,
                    delays[attempt],
                    _exception_text(exc),
                )
            await asyncio.sleep(delays[attempt])

    async def _record_lost_end(self) -> None:
        """Try once more to store the end of this session's turn that was lost.

        A CLI turn whose end no try of :meth:`_record_end` stored stays
        ``running`` until the lease rule frees it, which never happens while
        this process lives. A failure is logged, and the end is tried again
        next time.
        """
        lost = _LOST_ENDS.get(self._store)
        if lost is None or self._session_id is None:
            return
        end = lost.pop(self._session_id, None)
        if end is None:
            return
        try:
            await self._store.finish_turn(**end)
        except Exception as exc:
            if not _cannot_retry(exc):
                lost.setdefault(self._session_id, end)
            logger.warning(
                "Could not record the end of turn %d of session %s: %s",
                end["turn"],
                end["session_id"],
                _exception_text(exc),
            )

    def _cause(self, turn: _Turn) -> Cause:
        if turn.cause is not None:
            return turn.cause
        logger.warning(
            "Turn %d of session %s was cancelled from outside its runner; its cause "
            "is recorded as shutdown",
            turn.turn.turn,
            turn.turn.session_id,
        )
        return "shutdown"

    # ── The writer and the lease ──

    async def _write_events(self, turn: _Turn) -> None:
        """Store queued events in batches, then forward each; it never dies."""
        queue = turn.queue
        while True:
            item: tuple[Any, ...] | None = await queue.get()
            batch: list[tuple[dict[str, Any], tuple[ImagePart, ...]]] = []
            while item is not None and item[0] == "event":
                batch.append((item[1], item[2]))
                if len(batch) >= WRITER_BATCH_MAX or queue.empty():
                    item = None
                else:
                    item = queue.get_nowait()
            if batch:
                await self._store_batch(turn, batch)
            if item is None:
                continue
            if item[0] == "flush":
                marker: asyncio.Future[None] = item[1]
                if not marker.done():
                    marker.set_result(None)
                continue
            return  # _STOP

    async def _store_batch(
        self, turn: _Turn, batch: list[tuple[dict[str, Any], tuple[ImagePart, ...]]]
    ) -> None:
        info = turn.turn
        events = [event for event, _ in batch]
        images = [image for _, parts in batch for image in parts]
        stored: list[StoredEvent] = []
        try:
            stored = await self._store.append_events(
                info.session_id, info.turn, events, holder=self._holder, images=images
            )
        except asyncio.CancelledError:
            if _being_cancelled():
                raise
            logger.warning(
                "Storing events of session %s raised CancelledError", info.session_id
            )
        except Exception as exc:
            logger.warning(
                "Could not store %d event(s) of turn %d of session %s: %s",
                len(events),
                info.turn,
                info.session_id,
                _exception_text(exc),
            )
        if not stored:
            stored = [
                StoredEvent(None, info.session_id, info.turn, redacted_event(event))
                for event in events
            ]
        for event in stored:
            await self._forward(event)

    async def _forward(self, event: StoredEvent) -> None:
        callback = self._on_event
        if callback is None:
            return
        try:
            returned = callback(event)
            if inspect.isawaitable(returned):
                async with asyncio.timeout(SINK_TIMEOUT_S):
                    await returned
        except TimeoutError:
            logger.warning(
                "on_event took more than %g s on a %s event; it was cut off",
                SINK_TIMEOUT_S,
                event.event.get("type"),
            )
        except asyncio.CancelledError:
            if _being_cancelled():
                raise
            logger.warning("on_event raised CancelledError", exc_info=True)
        except Exception:
            logger.exception("on_event failed on a %s event", event.event.get("type"))

    async def _keep_lease(self, turn: _Turn) -> None:
        """Refresh the lease every LEASE_S / 3, whatever the run is doing."""
        info = turn.turn
        while True:
            await asyncio.sleep(_LEASE_REFRESH_S)
            try:
                kept = await self._store.refresh_lease(
                    info.session_id, holder=self._holder
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not refresh the lease of session %s; trying again at the "
                    "next tick: %s",
                    info.session_id,
                    _exception_text(exc),
                )
                continue
            if not kept:
                logger.warning(
                    "Session %s was taken over by another process; stopping turn %d",
                    info.session_id,
                    info.turn,
                )
                self._cancel_turn(turn, TAKEOVER_REASON, "takeover")
                return

    # ── Stopping ──

    def cancel(
        self,
        reason: str = "Cancelled by the user",
        *,
        cause: Cause = "user_stop",
        turn_id: str | None = None,
    ) -> bool:
        """Stop the running turn with *reason*, recording *cause*.

        With *turn_id*, only when the running turn has that id. Returns
        ``False`` when it stopped nothing, so a late Stop, or a cancel by
        key, never stops the next turn.
        """
        cause = _check_cause(cause)
        turn = self._active
        if turn is None or (turn_id is not None and turn.turn.id != turn_id):
            return False
        return self._cancel_turn(turn, reason, cause)

    def cancel_for_park(self, turn_id: str) -> bool:
        """Cancel exactly this turn for park only if it is not already stopping.

        The check and cancel are synchronous so a park fence can arbitrate
        against Stop without yielding to the event loop.
        """
        turn = self._active
        if (
            not turn_id
            or turn is None
            or turn.turn.id != turn_id
            or turn.cause is not None
            or turn.task is None
            or turn.task.cancelling()
        ):
            return False
        return self._cancel_turn(turn, "Parked by scheduler", "park")

    def _cancel_turn(self, turn: _Turn, reason: str, cause: Cause) -> bool:
        if turn.finish is not None:
            return False  # the run has ended: its status and cause stand
        task = turn.task
        if task is None or task.done():
            return False
        if turn.cause is None:
            turn.cause, turn.reason = cause, reason
        if task.cancel(reason):
            turn.runner_cancels += 1
            return True
        return False

    async def close(
        self, *, reason: str = "Session closed", cause: Cause = "closed"
    ) -> None:
        """Stop a running turn with *reason* and *cause*, wait for it, stop the tasks.

        Waits for the turn for at most ``FINALIZE_TIMEOUT_S + 2`` seconds.
        The caller closes the Agent.
        """
        cause = _check_cause(cause)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + FINALIZE_TIMEOUT_S + 2
        self._closed = True
        self._close_reason, self._close_cause = reason, cause
        begin = self._begin
        if begin is not None:
            await asyncio.wait({begin}, timeout=max(deadline - loop.time(), 0.0))
        turn = self._active
        if turn is None:
            return
        turn.closing = True
        self._cancel_turn(turn, reason, cause)
        waiter = asyncio.ensure_future(turn.done.wait())
        try:
            await asyncio.wait({waiter}, timeout=max(deadline - loop.time(), 0.0))
        finally:
            waiter.cancel()
        if not turn.done.is_set():
            logger.error(
                "Turn %d of session %s did not end within %g s of the close",
                turn.turn.turn,
                turn.turn.session_id,
                FINALIZE_TIMEOUT_S + 2,
            )
            if turn.lease is not None:
                turn.lease.cancel()

    # ── Approvals and published events ──

    def observe(self, ask: AskFunction) -> AskFunction:
        """Wrap *ask* so that approvals show in the session's stream.

        During a turn the wrapper flushes the writer, publishes
        ``approval.requested``, awaits *ask* and publishes
        ``approval.resolved``: ``withdrawn`` on a cancellation, ``denied``
        with origin ``error`` on a failure. Outside a turn it only calls
        *ask*.
        """

        async def observed(request: ApprovalRequest) -> ApprovalDecision:
            turn = self._active
            if turn is None or not turn.accepting:
                return await ask(request)
            await self.flush()
            self._enqueue(
                turn,
                "approval.requested",
                {
                    "request_id": request.request_id,
                    "call_id": request.call_id,
                    "tool_name": request.tool_name,
                    "arguments": request.arguments,
                    "preview": request.preview,
                    "workspace_root": request.workspace_root,
                },
            )

            def resolved(
                outcome: str, origin: str, reason: str, remember: bool
            ) -> None:
                self._enqueue(
                    turn,
                    "approval.resolved",
                    {
                        "request_id": request.request_id,
                        "call_id": request.call_id,
                        "tool_name": request.tool_name,
                        "outcome": outcome,
                        "origin": origin,
                        "reason": reason,
                        "remember": remember,
                    },
                )

            try:
                decision = await ask(request)
            except asyncio.CancelledError as exc:
                if _being_cancelled():
                    resolved("withdrawn", "withdrawn", turn.reason, False)
                else:
                    resolved("denied", "error", _exception_text(exc), False)
                raise
            except BaseException as exc:
                resolved("denied", "error", _exception_text(exc), False)
                raise
            if isinstance(decision, ApprovalDecision):
                resolved(
                    "allowed" if decision.allow else "denied",
                    decision.origin,
                    decision.reason,
                    decision.remember,
                )
            else:
                resolved(
                    "denied",
                    "error",
                    f"the ask function returned {type(decision).__name__}",
                    False,
                )
            return decision

        return observed

    def publish(self, type: str, data: Mapping[str, Any]) -> None:
        """Queue one session event for the current turn.

        *type* is ``approval.requested`` or ``approval.resolved`` (with
        their keys), or, in the gateway's file only, a type with a reserved
        prefix. *data* passes ``json_safe_payload``. ``RuntimeError`` outside
        a turn.
        """
        if type in APPROVAL_EVENT_TYPES:
            if set(data) != SESSION_EVENT_FIELDS[type]:
                raise ValueError(
                    f"a {type} event has the keys {sorted(SESSION_EVENT_FIELDS[type])}"
                )
        elif (
            isinstance(type, str)
            and type.startswith(RESERVED_EVENT_PREFIXES)
            and (RESERVED_EVENT_TYPE_RE.fullmatch(type))
        ):
            if self._store.writer != "gateway":
                raise ValueError(f"{type} events need the gateway's store")
        else:
            raise ValueError(
                f"publish takes approval.* or reserved types, got {type!r}"
            )
        turn = self._active
        if turn is None or not turn.accepting:
            raise RuntimeError("No turn is running")
        self._enqueue(turn, type, data)

    def _enqueue(self, turn: _Turn, type_: str, data: Mapping[str, Any]) -> None:
        if not turn.accepting:
            return
        event = {
            "v": EVENT_SCHEMA_VERSION,
            "type": type_,
            "run_id": "",
            "seq": None,
            "ts": time.time(),
            "data": json_safe_payload(data),
        }
        turn.queue.put_nowait(("event", event, ()))

    async def flush(self) -> None:
        """Wait until the events queued so far are stored and forwarded."""
        turn = self._active
        if turn is None or not turn.accepting:
            return
        marker: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        turn.queue.put_nowait(("flush", marker))
        await asyncio.shield(marker)

    # ── Resume ──

    async def resume(self, *, max_turns: int = RESUME_MAX_TURNS) -> int:
        """Load the stored conversation into the Agent; return the turns kept.

        ``resume_mode`` picks the arm: ``pairs`` loads the newest
        *max_turns* finished, not undone turns as user and answer pairs,
        within :data:`RESUME_BUDGET_CHARS`; ``window`` restores the window
        checkpoint and adds the later turns as pairs, and falls back to
        ``pairs`` without a usable checkpoint. :attr:`last_resume` says what
        was loaded. :class:`SessionBusyError` during a turn.
        """
        if (
            isinstance(max_turns, bool)
            or not isinstance(max_turns, int)
            or max_turns < 0
        ):
            raise ValueError(f"max_turns must be an int >= 0, got {max_turns!r}")
        self._check_idle()
        async with self._loading:
            return await self._resume(max_turns)

    async def _resume(self, max_turns: int, *, reserved: bool = False) -> int:
        """:meth:`resume`'s load, or the before-turn load under this runner's lease."""
        if not reserved:
            self._check_idle()
        checkpoint: list[Message] = []
        tail: list[Message] = []
        fell_back = False
        session_id = self._session_id
        if session_id is not None:
            loaded = None
            if self._resume_mode == "window":
                loaded = await self._window_arm(session_id, max_turns)
                fell_back = loaded is None
            if loaded is None:
                tail = await self._pairs(session_id, max_turns)
            else:
                checkpoint, tail = loaded
        if session_id is not None and await self._store.get_session(session_id) is None:
            raise UnknownSessionError(f"Unknown session: {session_id}")
        if not reserved:
            self._check_idle()
        try:
            kept = self._agent.load_conversation(checkpoint + tail)
        except RuntimeError:
            raise SessionBusyError(
                f"Session {self._name()} is still finishing its last turn"
            ) from None
        window = self._agent.conversation()
        turns = sum(_is_query(msg) for msg in window)
        # Eviction drops the oldest units first, so what stays is a suffix
        # of what was loaded, and the tail is its end.
        from_checkpoint = max(len(window) - len(tail), 0)
        checkpoint_turns = sum(_is_query(msg) for msg in window[:from_checkpoint])
        self._last_resume = ResumeInfo(
            mode="pairs" if fell_back or not checkpoint else self._resume_mode,
            fell_back=fell_back,
            turns=turns,
            checkpoint_turns=checkpoint_turns if checkpoint else 0,
            messages=kept,
            chars=sum(len(msg.content) for msg in window),
        )
        return turns

    async def _eligible_turns(
        self, session_id: str, max_turns: int, *, above: int = 0
    ) -> list[TurnInfo]:
        """The newest *max_turns* finished, not undone turns numbered above *above*.

        Newest first.
        """
        eligible: list[TurnInfo] = []
        before: int | None = None
        page_size = max(2 * max_turns, 20)
        while len(eligible) < max_turns:
            page = await self._store.list_turns(
                session_id, limit=page_size, before=before
            )
            for turn in reversed(page):
                if turn.turn <= above:
                    return eligible
                if turn.status == "running" or turn.undone:
                    continue
                eligible.append(turn)
                if len(eligible) == max_turns:
                    return eligible
            if len(page) < page_size:
                break
            before = page[0].turn
        return eligible

    async def _pairs(
        self, session_id: str, max_turns: int, *, above: int = 0
    ) -> list[Message]:
        """The ``pairs`` arm: user text and answer of each turn, oldest first."""
        pairs: list[tuple[TurnInfo, str, str]] = []
        total = 0
        for turn in await self._eligible_turns(session_id, max_turns, above=above):
            user = (
                turn.user_text
                if turn.attachments
                else clip_session_text(turn.user_text)
            )
            # As _record_answer derives the live answer, from the stored copy.
            answer = clip_session_text(
                answer_text(turn.status, split_thinking(turn.answer)[1])
            )
            if total + len(user) + len(answer) > RESUME_BUDGET_CHARS:
                if turn.attachments:
                    continue
                break
            total += len(user) + len(answer)
            pairs.append((turn, user, answer))
        messages: list[Message] = []
        for turn, user, answer in reversed(pairs):
            finished_at = (
                turn.finished_at if turn.finished_at is not None else turn.started_at
            )
            messages.append(
                Message(role="user", content=user, timestamp=turn.started_at)
            )
            messages.append(
                Message(role="assistant", content=answer, timestamp=finished_at)
            )
        return messages

    async def _window_arm(
        self, session_id: str, max_turns: int
    ) -> tuple[list[Message], list[Message]] | None:
        """The checkpoint and the later turns as pairs; ``None`` when unusable."""
        stored = await self._store.get_window(session_id)
        if stored is None or stored.version != WINDOW_CODEC_VERSION:
            return None
        messages = decode_window(stored.data)
        if messages is None:
            return None
        found = await self._store.list_turns(
            session_id, limit=1, before=stored.turn + 1
        )
        if not found or found[0].turn != stored.turn:
            return None
        if found[0].status == "running" or found[0].undone:
            return None
        tail = await self._pairs(session_id, max_turns, above=stored.turn)
        return messages, tail

    async def undo(self) -> TurnInfo | None:
        """Undo the newest finished turn, then resume; return it, or ``None``.

        A turn started meanwhile waits until the resume has loaded the
        conversation without the undone turn. The undo and its resume go on
        when this task is cancelled, and a resume that fails once the undo is
        stored empties the Agent's window, so no later request holds the
        undone turn; :meth:`resume` loads it again.
        """
        self._check_idle()
        async with self._loading:
            self._check_idle()
            if self._session_id is None:
                return None
            work = asyncio.ensure_future(self._undo(self._session_id))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                await _settle(work)
                if not work.cancelled():
                    work.exception()  # retrieved: the cancellation propagates
                raise

    async def _undo(self, session_id: str) -> TurnInfo | None:
        """:meth:`undo`'s work, under ``_loading``, in a task of its own."""
        await self._record_lost_end()
        undone = await self._store.undo_last_turn(session_id)
        if self._memory is not None and undone is not None:
            with contextlib.suppress(Exception):
                await self._memory.forget_session(session_id, undone.turn)
        try:
            await self._resume(RESUME_MAX_TURNS)
        except BaseException:
            with contextlib.suppress(RuntimeError):  # a finalizer still runs
                self._agent.load_conversation([])
            raise
        return undone


def _cannot_retry(exc: Exception) -> bool:
    """Whether a failed ``finish_turn`` would fail the same way every time."""
    return isinstance(exc, ValueError) or (
        isinstance(exc, SessionStoreError) and str(exc) == CLOSED_ERROR
    )


def _uncancel(task: asyncio.Task[Any], times: int) -> None:
    for _ in range(min(times, task.cancelling())):
        task.uncancel()


def _is_query(msg: Message) -> bool:
    return msg.role == "user" and msg.content != EMPTY_REPLY_NUDGE


def _being_cancelled() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
