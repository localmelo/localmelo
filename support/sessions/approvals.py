"""Approval policies for the surfaces: session grants, and a broker for the web.

:class:`RememberingPolicy` asks a person, through an ask function, and
keeps "allow this tool for the session" as a grant of the tool's name,
and "allow this folder for this chat" as a grant of a folder outside the
workspace (#405), which only reads use; "always" keeps the folder in the
settings (:class:`SavedFolders`).
:class:`ApprovalBroker` gives each session an ask function that waits for
an answer from elsewhere, such as the web console's approval route.
:class:`ApprovalLedger` keeps the approvals the person gave in each chat,
and :class:`OwnedTaskPolicy` lets a task of that chat use them (#369).
Grants, recorded approvals and pending requests live in memory only: a
restart forgets them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from localmelo.melo.contracts.approvals import (
    ALLOWED_ONCE,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    AutoPolicy,
    ReviewMode,
    find_auto,
    find_folder_asker,
    review_mode,
)
from localmelo.melo.contracts.folders import covers
from localmelo.melo.contracts.review import (
    RULED_TOOLS,
    ActionReview,
    covered_by_grant,
    offers_grant,
    step_key,
)

logger = logging.getLogger(__name__)

AskFunction = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]

# Answered request ids the broker remembers, so that a second answer reads
# already_resolved rather than unknown.
RESOLVED_IDS_KEPT: Final = 256


# The reasons of a read that a folder grant let run without asking (#405).
FOLDER_FOR_CHAT: Final = "folder allowed for this chat"
FOLDER_SAVED: Final = "folder in the extra folders"


class SavedFolders(Protocol):
    """The folders "Always allow this folder" keeps (``[workspace] extra_folders``)."""

    def folders(self) -> Sequence[str]:
        """The folders kept now, as absolute paths."""
        ...

    def add(self, folder: str) -> bool:
        """Keep one more folder; whether it was new. Raises when it cannot be kept."""
        ...


def _covered(folders: Iterable[str], path: str) -> bool:
    return any(covers(folder, path) for folder in folders)


class RememberingPolicy:
    """An approval policy that asks through *ask* and keeps session grants.

    A routine call of a tool granted for the session runs without asking
    (``ApprovalDecision(True, origin="session")``); any other call is
    asked about, and an answer that allows with ``remember`` grants the
    tool's name. A grant covers routine calls only: a high-impact or
    undecided call (see :mod:`localmelo.melo.contracts.review`) is asked
    about every time, and a ``remember`` on a high-impact one is dropped,
    so it allows that call and grants nothing. Under a grant the agent's
    gate first lets the user's model review an undecided call
    (:meth:`silent_for`); one it finds routine then runs on the grant.
    Grants are per tool name and per policy object, and never stored.

    With *folders*, it is also asked about a file tool's path outside the
    workspace (:meth:`decide_folder`, #405); *saved* holds the folders kept
    by "always". A policy without *folders* is never asked about one (a
    task the person did not start), so such a path is refused.
    """

    def __init__(
        self,
        ask: AskFunction,
        *,
        folders: bool = False,
        saved: SavedFolders | None = None,
    ) -> None:
        self._ask = ask
        self._granted: set[str] = set()
        self.asks_folders = folders
        self._saved = saved
        # The folders allowed for reading in this chat, as real paths.
        self._folders: list[str] = []

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(self._granted)

    @property
    def allowed_folders(self) -> tuple[str, ...]:
        """The folders outside the workspace allowed for reading in this chat."""
        return tuple(self._folders)

    def forget(self, tool_name: str | None = None) -> None:
        """Take back the grant of *tool_name*, or every grant with ``None``.

        ``None`` takes back the chat's folder grants too.
        """
        if tool_name is None:
            self._granted.clear()
            self._folders.clear()
        else:
            self._granted.discard(tool_name)

    def forget_folder(self, folder: str) -> bool:
        """Take back the chat's grant of *folder*; whether there was one."""
        if folder not in self._folders:
            return False
        self._folders.remove(folder)
        return True

    def covered(self, path: str) -> str:
        """What lets the read of *path* run unasked: :data:`FOLDER_FOR_CHAT`, :data:`FOLDER_SAVED` or ``""``."""
        if _covered(self._folders, path):
            return FOLDER_FOR_CHAT
        if self._saved is not None:
            try:
                saved = self._saved.folders()
            except Exception:
                logger.warning("Could not read the extra folders", exc_info=True)
                saved = ()
            if _covered(saved, path):
                return FOLDER_SAVED
        return ""

    async def decide_folder(self, request: ApprovalRequest) -> ApprovalDecision:
        """Decide a call on a path outside the workspace (#405).

        A read in a folder allowed for this chat, or kept in the extra
        folders, runs unasked; anything else asks the person. A write asks
        every time, and an answer for it grants nothing; so does an answer
        for a folder that may not be granted (the home folder). "This chat"
        (``remember``) keeps the folder for the chat; "always" also keeps it
        in the settings, and a failure to save it still allows the call.
        """
        outside = request.outside
        if outside is None:
            raise ValueError("decide_folder takes a request with outside set")
        if outside.access == "read":
            found = self.covered(outside.path)
            if found == FOLDER_FOR_CHAT:
                return ApprovalDecision(True, found, origin="session")
            if found:
                return ApprovalDecision(True, found, origin="policy")
        decision = await self._ask(request)
        if not isinstance(decision, ApprovalDecision):
            return decision
        if not outside.grantable:
            decision = dataclasses.replace(decision, remember=False, always=False)
        if decision.allow and decision.always:
            if self._saved is not None:
                try:
                    await asyncio.to_thread(self._saved.add, outside.folder)
                except Exception:
                    logger.warning(
                        "Could not keep %s in the extra folders",
                        outside.folder,
                        exc_info=True,
                    )
            else:
                decision = dataclasses.replace(decision, always=False)
        if (
            decision.allow
            and (decision.remember or decision.always)
            and outside.folder not in self._folders
        ):
            self._folders.append(outside.folder)
        return decision

    def silent_for(self, request: ApprovalRequest) -> bool:
        """Whether a grant would let *request* run unasked: the gate then lets
        the user's model review an undecided call first (#279 §3.3)."""
        return request.tool_name in self._granted

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.outside is not None:
            # A folder outside the workspace is only ever decided by decide_folder.
            return ApprovalDecision(
                False,
                "a folder outside the workspace needs its own question",
                origin="policy",
            )
        if request.tool_name in self._granted and covered_by_grant(request.review):
            return ApprovalDecision(True, origin="session")
        decision = await self._ask(request)
        if isinstance(decision, ApprovalDecision) and not offers_grant(request.review):
            decision = dataclasses.replace(decision, remember=False)
        if (
            isinstance(decision, ApprovalDecision)
            and decision.allow
            and decision.remember
        ):
            self._granted.add(request.tool_name)
        return decision


ChatMode = Literal["ask", "auto"]


def chat_mode(meta: object, default: ChatMode = "ask") -> ChatMode:
    """A session's mode from its ``meta`` (``approvals``), else *default*.

    The main chat has *default* (the settings' mode for new chats) until
    the person picks one; any other session without one is ``ask``.
    """
    found = meta.get("approvals") if isinstance(meta, dict) else None
    if found in ("ask", "auto"):
        return "auto" if found == "auto" else "ask"
    return "auto" if default == "auto" else "ask"


class ChatPolicy:
    """One chat's policy: Ask (a :class:`RememberingPolicy`) or Auto (#339).

    :attr:`mode` is read at every decision, so a switch applies from the
    chat's next step; a call already waiting for a person keeps waiting.
    The Ask policy keeps its grants and the :class:`AutoPolicy` its
    allowances across switches. ``inner`` is the policy of the mode now,
    for :func:`~localmelo.melo.contracts.approvals.find_auto`.

    The person is asked about a folder outside the workspace in either
    mode (:meth:`decide_folder`): auto mode never decides it.
    """

    def __init__(
        self,
        ask: AskFunction,
        mode: ChatMode = "ask",
        *,
        saved: SavedFolders | None = None,
    ) -> None:
        self.mode: ChatMode = mode
        self.ask = RememberingPolicy(ask, folders=True, saved=saved)
        self.auto = AutoPolicy()
        self.asks_folders = True

    @property
    def inner(self) -> ApprovalPolicy:
        return self.auto if self.mode == "auto" else self.ask

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self.ask.allowed_tools

    @property
    def allowed_folders(self) -> tuple[str, ...]:
        return self.ask.allowed_folders

    def forget(self, tool_name: str | None = None) -> None:
        """Take back one grant, or every grant and folder (allowances stay)."""
        self.ask.forget(tool_name)

    def forget_folder(self, folder: str) -> bool:
        return self.ask.forget_folder(folder)

    def review_mode(self, request: ApprovalRequest) -> ReviewMode:
        return "auto" if self.mode == "auto" else review_mode(self.ask, request)

    async def decide_folder(self, request: ApprovalRequest) -> ApprovalDecision:
        return await self.ask.decide_folder(request)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return await self.inner.decide(request)


@dataclass
class _Pending:
    session_id: str
    request: ApprovalRequest
    answer: asyncio.Future[ApprovalDecision]


class ApprovalBroker:
    """Pending approval requests of every session, answered by request id.

    ``asker(session_id)`` gives a session's policy its ask function. Without
    *timeout_s* a request waits for an answer without a limit; with it,
    the request is denied with origin ``timeout`` when no answer comes in
    time, so an abandoned tab cannot block a turn for ever.
    """

    def __init__(self, *, timeout_s: float | None = None) -> None:
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int | float)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(
                f"timeout_s must be None or a finite number > 0, got {timeout_s!r}"
            )
        self._timeout_s = timeout_s
        self._pending: OrderedDict[str, _Pending] = OrderedDict()
        self._resolved: deque[str] = deque(maxlen=RESOLVED_IDS_KEPT)
        self._resolved_set: set[str] = set()

    def asker(self, session_id: str) -> AskFunction:
        """The ask function of *session_id*'s policy.

        It registers the request before its first await, then waits for
        :meth:`answer`. A cancellation removes the request and propagates.
        """

        async def ask(request: ApprovalRequest) -> ApprovalDecision:
            return await self.begin(session_id, request)

        return ask

    def begin(
        self, session_id: str, request: ApprovalRequest
    ) -> asyncio.Task[ApprovalDecision]:
        """Register synchronously and return a cancellable answer waiter."""
        answer: asyncio.Future[ApprovalDecision] = (
            asyncio.get_running_loop().create_future()
        )
        entry = _Pending(session_id, request, answer)
        self._pending[request.request_id] = entry

        async def wait() -> ApprovalDecision:
            if self._timeout_s is None:
                return await answer
            try:
                async with asyncio.timeout(self._timeout_s):
                    return await answer
            except TimeoutError:
                if answer.done() and not answer.cancelled():
                    return answer.result()
                return ApprovalDecision(
                    False, f"No answer within {self._timeout_s:g} s", origin="timeout"
                )

        task = asyncio.create_task(wait())

        def cleanup(_: asyncio.Task[ApprovalDecision]) -> None:
            if self._pending.get(request.request_id) is entry:
                self._pending.pop(request.request_id)
                self._remember(request.request_id)

        task.add_done_callback(cleanup)
        return task

    def _remember(self, request_id: str) -> None:
        if len(self._resolved) == self._resolved.maxlen:
            self._resolved_set.discard(self._resolved[0])
        self._resolved.append(request_id)
        self._resolved_set.add(request_id)

    def request(self, request_id: str) -> ApprovalRequest | None:
        """The pending request *request_id*, or ``None`` once it is answered."""
        entry = self._pending.get(request_id)
        return entry.request if entry is not None and not entry.answer.done() else None

    def session_of(self, request_id: str) -> str | None:
        """The session whose request *request_id* waits for an answer, else ``None``."""
        entry = self._pending.get(request_id)
        return (
            entry.session_id if entry is not None and not entry.answer.done() else None
        )

    def pending(self, session_id: str) -> list[ApprovalRequest]:
        """The requests of *session_id* that wait for an answer, oldest first."""
        return [
            entry.request
            for entry in self._pending.values()
            if entry.session_id == session_id and not entry.answer.done()
        ]

    def answer(
        self, request_id: str, decision: ApprovalDecision
    ) -> Literal["resolved", "already_resolved", "unknown"]:
        """Answer *request_id*; the first answer wins."""
        if not isinstance(decision, ApprovalDecision):
            raise TypeError("answer takes an ApprovalDecision")
        entry = self._pending.get(request_id)
        if entry is not None and not entry.answer.done():
            entry.answer.set_result(decision)
            return "resolved"
        if entry is not None or request_id in self._resolved_set:
            return "already_resolved"
        return "unknown"

    def withdraw_all(self, session_id: str | None = None, *, reason: str) -> int:
        """Deny the pending requests, of one session or of all; return how many.

        Each is denied with origin ``policy`` and *reason*.
        """
        denied = 0
        for entry in list(self._pending.values()):
            if session_id is not None and entry.session_id != session_id:
                continue
            if not entry.answer.done():
                entry.answer.set_result(
                    ApprovalDecision(False, reason, origin="policy")
                )
                denied += 1
        return denied


# ── The approvals the person gave, per chat (#369) ──

# Recorded approvals a chat keeps, and steps that ran on one, at most.
LEDGER_KEPT: Final = 200
USES_KEPT: Final = 500


def reusable(tool_name: str, review: ActionReview | None) -> bool:
    """Whether an approval of a call of *tool_name* reviewed as *review* may run the same step again.

    Only a call of a tool melo's rules read (:data:`RULED_TOOLS`) that the
    rules themselves found routine, in no category. Never a high-impact one
    (#279: ``self`` and the click-only categories included); never an
    undecided one (melo could not see what it runs: a file too big or not
    read, a command built at run time); never one a model settled; never a
    tool the rules do not read (a network or MCP tool is routine to them
    unread); never a call nothing reviewed. Those ask every time.
    """
    return (
        tool_name in RULED_TOOLS
        and review is not None
        and review.verdict == "routine"
        and review.by == "rules"
        and not review.categories
    )


def _folder(workspace_root: str | None) -> str:
    return os.path.realpath(workspace_root) if workspace_root else ""


def request_step(request: ApprovalRequest) -> str:
    """The step key of *request*, as auto mode keys it: the exact call, folder and files read."""
    review = request.review
    inputs = review.inputs if review is not None else ()
    return step_key(
        request.tool_name, request.arguments, request.workspace_root, inputs
    )


@dataclass(frozen=True)
class RecordedApproval:
    """One step the person allowed in a chat or in one of its tasks."""

    entry_id: str
    chat_id: str
    tool: str
    step_key: str
    folder: str
    verdict: str
    categories: tuple[str, ...]
    preview: str
    at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.entry_id,
            "tool": self.tool,
            "folder": self.folder,
            "verdict": self.verdict,
            "categories": list(self.categories),
            "preview": self.preview,
            "at": self.at,
        }


@dataclass(frozen=True)
class ApprovalUse:
    """A task's step that ran on a recorded approval, without asking."""

    entry_id: str
    chat_id: str
    task_id: str
    request_id: str
    preview: str
    approved_at: float
    at: float


class ApprovalLedger:
    """The approvals the person gave in each chat: "already approved" by code.

    An entry is the person's Allow of one exact step (its tool, its
    :func:`request_step` key and its folder) in a chat or in a task the chat
    owns. :meth:`match` finds an entry for a request of the same tool, key
    and folder when both reviews are routine by the rules (:func:`reusable`);
    the model never decides what matches. Memory only, like grants.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._entries: dict[str, OrderedDict[str, RecordedApproval]] = {}
        self._uses: deque[ApprovalUse] = deque(maxlen=USES_KEPT)
        # The tools chat *id* holds a grant of now; the console sets it.
        self.grants_of: Callable[[str], frozenset[str]] = lambda chat_id: frozenset()
        # The folders outside the workspace chat *id* allowed for reading (#405).
        self.folders_of: Callable[[str], tuple[str, ...]] = lambda chat_id: ()

    def record(self, chat_id: str, request: ApprovalRequest) -> RecordedApproval | None:
        """Keep the person's Allow of *request*; ``None`` when it may never be used again.

        A call on a path outside the workspace is never kept (#405): its
        folder grant is the chat's, and a write asks every time.
        """
        review = request.review
        if (
            review is None
            or request.outside is not None
            or not reusable(request.tool_name, review)
        ):
            return None
        key = request_step(request)
        entries = self._entries.setdefault(chat_id, OrderedDict())
        for old in [e.entry_id for e in entries.values() if e.step_key == key]:
            del entries[old]
        entry = RecordedApproval(
            uuid.uuid4().hex[:12],
            chat_id,
            request.tool_name,
            key,
            _folder(request.workspace_root),
            review.verdict,
            review.categories,
            request.preview,
            self._clock(),
        )
        entries[entry.entry_id] = entry
        while len(entries) > LEDGER_KEPT:
            entries.popitem(last=False)
        return entry

    def match(self, chat_id: str, request: ApprovalRequest) -> RecordedApproval | None:
        """The entry of chat *chat_id* that allowed exactly *request*'s step, else ``None``."""
        if request.outside is not None or not reusable(
            request.tool_name, request.review
        ):
            return None
        key, folder = request_step(request), _folder(request.workspace_root)
        for entry in self._entries.get(chat_id, {}).values():
            same = (entry.tool, entry.step_key, entry.folder) == (
                request.tool_name,
                key,
                folder,
            )
            if same and entry.verdict != "high" and not entry.categories:
                return entry
        return None

    def used(
        self, entry: RecordedApproval, task_id: str, request: ApprovalRequest
    ) -> ApprovalUse:
        """Note that task *task_id* ran *request* on *entry*."""
        use = ApprovalUse(
            entry.entry_id,
            entry.chat_id,
            task_id,
            request.request_id,
            request.preview,
            entry.at,
            self._clock(),
        )
        self._uses.append(use)
        return use

    def entries(self, chat_id: str) -> list[RecordedApproval]:
        """Chat *chat_id*'s recorded approvals, newest first."""
        return list(reversed(self._entries.get(chat_id, {}).values()))

    def revoke(self, chat_id: str, entry_id: str) -> bool:
        """Take back one recorded approval; whether there was one. Steps that ran stay run."""
        entries = self._entries.get(chat_id)
        return entries is not None and entries.pop(entry_id, None) is not None

    def drop(self, chat_id: str) -> None:
        """Forget a deleted chat's recorded approvals."""
        self._entries.pop(chat_id, None)

    def uses_of(self, task_id: str) -> list[dict[str, Any]]:
        """The steps of *task_id* that ran on a recorded approval; ``live`` while it stands."""
        return [
            {
                "entry_id": use.entry_id,
                "chat_id": use.chat_id,
                "request_id": use.request_id,
                "preview": use.preview,
                "approved_at": use.approved_at,
                "at": use.at,
                "live": use.entry_id in self._entries.get(use.chat_id, {}),
            }
            for use in self._uses
            if use.task_id == task_id
        ]


class OwnedTaskPolicy:
    """A task's policy: its chat's grants and recorded approvals, and *inner*.

    In Ask, before *inner* asks the person, a routine call of a tool the
    owner chat holds a grant of runs as under a session grant, and a call
    whose exact step the person allowed in that chat or one of its tasks
    runs with origin ``earlier``; *on_use* is told. When *inner* asks the
    person and they allow, the approval is recorded in the chat that owns
    the task when they answer (nothing, when that chat went meanwhile). In
    Auto, *inner* decides first (it asks nobody); only a step it blocked
    could run on a recorded approval, and it blocks none that is reusable.
    Auto's Allow once is never recorded: it stays one exact step, once.
    *owner* gives the chat that owns the task now, and *started* when a
    chat began. A task the person did not start is never wrapped (its
    approvals are never remembered).

    Grants and recorded approvals are used only after :meth:`review_window`
    read the owner chat's start, and only while that chat still owns the
    task: the task's steps are then reviewed with every file changed since
    that chat began, so code an earlier task of the chat planted is read
    before the chat's approval covers a step. Without that start, the
    person is asked.

    A path outside the workspace (#405) is asked about through *folders*
    (the task's asking policy, also in Auto, which asks nobody), or the
    folder layer of *inner*: a read in a folder its owner chat allowed
    runs unasked.
    """

    def __init__(
        self,
        inner: ApprovalPolicy,
        ledger: ApprovalLedger,
        task_id: str,
        owner: Callable[[], Awaitable[str | None]],
        on_use: Callable[[ApprovalUse], Awaitable[None]] | None = None,
        *,
        started: Callable[[str], Awaitable[float | None]] | None = None,
        folders: ApprovalPolicy | None = None,
    ) -> None:
        self.inner = inner
        self._folders = folders
        self._ledger = ledger
        self._task_id = task_id
        self._owner = owner
        self._on_use = on_use
        self._started = started
        # The owner chat as last read, for review_mode (which cannot await).
        self._chat: str | None = None
        # The chat whose start the review window begins at, and that start.
        self._window_chat: str | None = None
        self._window: float | None = None

    def _auto(self) -> bool:
        return find_auto(self.inner) is not None

    def _folder_asker(self) -> Any:
        found = find_folder_asker(self._folders)
        return found if found is not None else find_folder_asker(self.inner)

    @property
    def asks_folders(self) -> bool:
        return self._folder_asker() is not None

    async def decide_folder(self, request: ApprovalRequest) -> ApprovalDecision:
        """A read in a folder the owner chat allowed runs; anything else asks (#405)."""
        outside = request.outside
        asker = self._folder_asker()
        if outside is None or asker is None:
            raise ValueError("decide_folder needs an outside request and an asker")
        if outside.access == "read":
            chat = await self._owner()
            if chat and _covered(self._ledger.folders_of(chat), outside.path):
                return ApprovalDecision(True, FOLDER_FOR_CHAT, origin="session")
        decision: ApprovalDecision = await asker.decide_folder(request)
        return decision

    async def review_window(self) -> float | None:
        """When the task's review window starts: its owner chat's start.

        ``None`` (Auto, which uses neither, or a start that cannot be read):
        the task's steps are reviewed from the task's own start, and its
        chat's grants and approvals are not used.
        """
        self._window_chat = self._window = None
        if self._auto() or self._started is None:
            return None
        chat = await self._owner()
        if not chat:
            return None
        try:
            started = await self._started(chat)
        except Exception:
            logger.warning("Could not read when chat %s began", chat, exc_info=True)
            return None
        if started is None or not math.isfinite(started):
            return None
        self._window_chat, self._window = chat, started
        return started

    def _reviewed_since(self, chat: str | None) -> bool:
        """Whether the steps were reviewed from *chat*'s start."""
        return bool(chat) and self._window is not None and chat == self._window_chat

    def _granted(self, tool_name: str) -> bool:
        chat = self._chat
        return (
            self._reviewed_since(chat)
            and not self._auto()
            and tool_name in self._ledger.grants_of(chat or "")
        )

    def review_mode(self, request: ApprovalRequest) -> ReviewMode:
        mode = review_mode(self.inner, request)
        return "silent" if mode == "ask" and self._granted(request.tool_name) else mode

    async def _earlier(
        self, chat: str, request: ApprovalRequest
    ) -> ApprovalDecision | None:
        """The step runs on the person's recorded approval of it, else ``None``."""
        entry = self._ledger.match(chat, request)
        if entry is None:
            return None
        use = self._ledger.used(entry, self._task_id, request)
        if self._on_use is not None:
            try:
                await self._on_use(use)
            except Exception:
                logger.warning("Could not show a used approval", exc_info=True)
        return ApprovalDecision(
            True, f"earlier approval {entry.entry_id}", origin="earlier"
        )

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self._chat = chat = await self._owner()
        if self._auto():
            # Auto asks nobody: what it runs (a routine step, an Allow once)
            # runs as it would. It blocks only steps that are not routine,
            # which match no approval: kept so the person outranks Auto if
            # that ever changes.
            decision = await self.inner.decide(request)
            if not chat or decision.allow or decision.origin != "auto":
                return decision
            if not self._reviewed_since(chat):
                return decision
            return await self._earlier(chat, request) or decision
        if chat and self._reviewed_since(chat):
            if self._granted(request.tool_name) and covered_by_grant(request.review):
                return ApprovalDecision(True, origin="session")
            earlier = await self._earlier(chat, request)
            if earlier is not None:
                return earlier
        decision = await self.inner.decide(request)
        if (
            chat
            and isinstance(decision, ApprovalDecision)
            and decision.allow
            and decision.origin == "user"
            and decision.reason != ALLOWED_ONCE
            # The chat that asked still owns the task: a deleted chat's Allow is not the main chat's.
            and await self._owner() == chat
        ):
            self._ledger.record(chat, request)
        return decision
