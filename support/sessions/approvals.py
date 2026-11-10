"""Approval policies for the surfaces: session grants, and a broker for the web.

:class:`RememberingPolicy` asks a person, through an ask function, and
keeps "allow this tool for the session" as a grant of the tool's name,
and "allow this folder for this chat" as a grant of a folder outside the
workspace (#405), which only reads use; "always" keeps the folder in the
settings (:class:`SavedFolders`).
:class:`ApprovalBroker` gives each session an ask function that waits for
an answer from elsewhere, such as the web console's approval route.
:class:`OwnedTaskPolicy` lets a task use the grants of the chat that owns
it (#369).
A policy's grants live in a :class:`~localmelo.support.sessions.grants.GrantBook`
(one table of every grant, kept in the session store when the book has one,
so they outlive a restart; what the person allows in the main chat counts
everywhere). It is the only place a kept answer lives: an Allow once is
never kept. Pending requests live in memory only: a restart forgets them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from localmelo.melo.contracts.approvals import (
    AUTO_ASK_WAIT_S,
    LEFT_UNANSWERED,
    NO_ANSWER,
    STAYS_FOR_CHAT,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    AskWatch,
    AutoPolicy,
    Chosen,
    ReviewMode,
    Withdrawal,
    find_auto,
    find_folder_asker,
    grant_offered,
    holds_grant,
    review_mode,
    revoke_site,
    site_grant,
    stays_for_chat,
    window_grant,
)
from localmelo.melo.contracts.folders import covers
from localmelo.melo.contracts.review import covered_by_grant
from localmelo.support.sessions.grants import GrantBook, SessionGrants

logger = logging.getLogger(__name__)

AskFunction = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]

# Answered request ids the broker remembers, so that a second answer reads
# already_resolved rather than unknown.
RESOLVED_IDS_KEPT: Final = 256


# The reason of a site question that runs because Auto let the chat use the
# site earlier (#465).
PASSED_IN_AUTO: Final = "allowed in auto mode earlier in this chat"

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


def kept_answer(
    grants: SessionGrants, request: ApprovalRequest
) -> ApprovalDecision | None:
    """What a chat's kept answers decide about *request*: an allow, or ``None`` (ask).

    The one rule for a chat and for the tasks it owns. The person's grant of
    the question (or of another host of its site) covers a routine call, a
    site Auto stopped only as not asked for, and an app's window. A site Auto
    let the chat read covers its plain reads only, never a click, an address
    that can carry data, a card-only kind or a step Auto stopped
    (:func:`~localmelo.melo.contracts.approvals.stays_for_chat`).
    """
    if grants.covers(request.tool_name) and (
        covered_by_grant(request.review) or site_grant(request) or window_grant(request)
    ):
        return ApprovalDecision(True, origin="session")
    if (
        request.tool_name in grants.passed()
        and covered_by_grant(request.review)
        and stays_for_chat(request)
    ):
        return ApprovalDecision(True, PASSED_IN_AUTO, origin="session")
    return None


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
    In auto mode it is the policy that asks about a step Auto stopped
    (#465): a site Auto stopped only as not asked for may be granted for
    the chat, and a grant of that site question covers it after
    (:func:`~localmelo.melo.contracts.approvals.site_grant`). A person's grant
    of a site question covers the site's other hosts too
    (:meth:`~localmelo.support.sessions.grants.SessionGrants.covers`, #577). A
    site Auto let the chat read (:meth:`passed_in_auto`) stays allowed for the chat, on its exact origin:
    in Ask its plain reads run unasked, as on a grant, but never a click, an
    address that can carry data, a card-only kind or a step Auto's review
    stops (:func:`~localmelo.melo.contracts.approvals.stays_for_chat`).
    Both are listed in :attr:`allowed_tools` and taken back by
    :meth:`forget`.
    The grants live in *grants*, a session's view of a
    :class:`~localmelo.support.sessions.grants.GrantBook`: in the main chat
    what the person allows counts everywhere, elsewhere in that session
    only, and the book keeps them across a restart when it has a store.
    Without *grants*, they are this policy object's own and never stored.

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
        grants: SessionGrants | None = None,
    ) -> None:
        self._ask = ask
        # The grants that hold in this chat, and where new ones go.
        self.grants = grants if grants is not None else GrantBook().view("", "side")
        self.asks_folders = folders
        self._saved = saved

    @property
    def allowed_tools(self) -> frozenset[str]:
        """The names granted for the chat, and the sites Auto let it use."""
        return self.grants.held()

    def granted(self, name: str) -> bool:
        """Whether the question *name* is allowed for this chat (``window-read:<app>``, a site, a tool)."""
        return name in self.allowed_tools

    def passed_in_auto(self, tool_name: str) -> None:
        """Keep a site read auto mode let run: it stays allowed for the chat (#465).

        Only a question about opening or reading a site (:data:`STAYS_FOR_CHAT`)
        is kept, as Auto's (``auto``): for this chat only, even the main chat;
        any other name is ignored.
        """
        if tool_name.startswith(STAYS_FOR_CHAT):
            self.grants.grant_now(tool_name, by="auto")

    @property
    def allowed_folders(self) -> tuple[str, ...]:
        """The folders outside the workspace allowed for reading in this chat."""
        return self.grants.folders()

    def forget(self, tool_name: str | None = None) -> None:
        """Take back the grant of *tool_name*, or every grant with ``None``.

        ``None`` takes back the chat's folder grants too. A grant from the
        main chat that holds here is taken back everywhere by its name, and
        stays when every grant of a side chat is taken back.
        """
        self.grants.forget_now(tool_name)

    def forget_folder(self, folder: str) -> bool:
        """Take back the chat's grant of *folder*; whether there was one."""
        return self.grants.forget_folder_now(folder)

    async def take_back_site(self, key: str, by: str | None = None) -> bool:
        """A refusal of a site: what this chat kept for it goes (#577). Whether that is saved."""
        return await self.grants.take_back_site(key, by)

    def covered(self, path: str) -> str:
        """What lets the read of *path* run unasked: :data:`FOLDER_FOR_CHAT`, :data:`FOLDER_SAVED` or ``""``."""
        if _covered(self.grants.folders(), path):
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
            and outside.folder not in self.grants.folders()
        ):
            await self.grants.grant(
                outside.folder, kind="folder", request=request, decision=decision
            )
        return decision

    def silent_for(self, request: ApprovalRequest) -> bool:
        """Whether a grant would let *request* run unasked: the gate then lets
        the user's model review an undecided call first (#279 §3.3)."""
        return self.grants.covers(request.tool_name)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.outside is not None:
            # A folder outside the workspace is only ever decided by decide_folder.
            return ApprovalDecision(
                False,
                "a folder outside the workspace needs its own question",
                origin="policy",
            )
        kept = kept_answer(self.grants, request)
        if kept is not None:
            return kept
        decision = await self._ask(request)
        if isinstance(decision, ApprovalDecision) and not grant_offered(request):
            decision = dataclasses.replace(decision, remember=False)
        if (
            isinstance(decision, ApprovalDecision)
            and decision.allow
            and decision.remember
        ):
            await self.grants.grant(
                request.tool_name, request=request, decision=decision
            )
        return decision


# How often a card auto mode shows checks that someone still sees it, and how
# long nobody may have seen it (from the first check that found nobody) before
# it counts as closed: a reload of the page or a reconnect of its stream takes
# less (#465).
CARD_POLL_S: Final = 1.0
CARD_LEFT_S: Final = 5.0


class CardWatch:
    """Who sees the card auto mode shows, and how long it waits (#465).

    *viewers* counts the surfaces that show the card now, read again at each
    check (a task's chat may change while its card waits): on the gateway,
    the chat's event streams a console, the app or the terminal opened as a
    viewer (``watch=true``); 0 with ``[approvals] auto_asks`` off. Auto mode
    asks only while one is open (:meth:`watched`); with none it blocks the
    step as before. A card waits at most :data:`AUTO_ASK_WAIT_S`; when nobody
    answers in that time, or every viewer is gone for :data:`CARD_LEFT_S`,
    *end* answers it as auto mode's Don't allow, a :class:`Chosen` with origin
    ``timeout`` and its code (``unanswered`` ``time`` with ``wait_s``, or
    ``left``; :data:`NO_ANSWER`, :data:`LEFT_UNANSWERED` as its text). *end*
    takes the request id and the decision and says whether the card was up
    (answered now or before); it is the answer of the broker or of the
    terminal's panel, so the card ends as an answered one. *clock* and
    *sleep* are a test's fake time.
    """

    def __init__(
        self,
        viewers: Callable[[], Awaitable[int]],
        end: Callable[[str, ApprovalDecision], bool],
        *,
        wait_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self._viewers = viewers
        self._end = end
        self.wait_s = AUTO_ASK_WAIT_S if wait_s is None else wait_s
        self._clock = clock
        self._sleep = sleep

    async def watched(self) -> bool:
        try:
            return await self._viewers() > 0
        except Exception:
            logger.warning("Could not count who sees the chat", exc_info=True)
            return False

    async def ask(
        self, request: ApprovalRequest, asking: ApprovalPolicy
    ) -> ApprovalDecision:
        expiry = asyncio.ensure_future(self._expire(request))
        try:
            return await asking.decide(request)
        finally:
            expiry.cancel()
            await asyncio.gather(expiry, return_exceptions=True)

    async def _expire(self, request: ApprovalRequest) -> None:
        """Answer the card of *request* once nobody answered or nobody sees it."""
        start = self._clock()
        gone: float | None = None  # when a check first found nobody
        decision = Chosen(
            False, NO_ANSWER, origin="timeout", unanswered="time", wait_s=self.wait_s
        )
        while True:
            await self._sleep(CARD_POLL_S)
            now = self._clock()
            if await self.watched():
                gone = None
            else:
                gone = now if gone is None else gone
                if now - gone >= CARD_LEFT_S:
                    decision = Chosen(
                        False, LEFT_UNANSWERED, origin="timeout", unanswered="left"
                    )
                    break
            if now - start >= self.wait_s:
                break
        # A card not up yet (its event is still being written) is tried again.
        while not self._end(request.request_id, decision):
            await self._sleep(CARD_POLL_S)


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


class ModePolicy:
    """The Ask policy or the Auto policy, whichever :attr:`mode` names now (#339, #576).

    :attr:`mode` is read at every decision, so a switch applies from the
    next step; a call already waiting for a person keeps waiting. Both
    policies keep their own state (Ask's grants, Auto's allowances) across
    switches. ``inner`` is the policy of the mode now, for
    :func:`~localmelo.melo.contracts.approvals.find_auto`. *mode* is a mode,
    or a function that gives it: a task's policy reads its chat's mode that
    way, so the task has no mode of its own to leave behind (#576).
    """

    def __init__(
        self,
        ask: ApprovalPolicy,
        auto: AutoPolicy,
        mode: ChatMode | Callable[[], ChatMode] = "ask",
    ) -> None:
        self._mode = mode
        self.ask = ask
        self.auto = auto

    @property
    def mode(self) -> ChatMode:
        return self._mode() if callable(self._mode) else self._mode

    @mode.setter
    def mode(self, value: ChatMode) -> None:
        self._mode = value

    @property
    def inner(self) -> ApprovalPolicy:
        return self.auto if self.mode == "auto" else self.ask

    def review_mode(self, request: ApprovalRequest) -> ReviewMode:
        return review_mode(self.inner, request)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return await self.inner.decide(request)


class ChatPolicy(ModePolicy):
    """One chat's policy: Ask (a :class:`RememberingPolicy`) or Auto (#339).

    The Ask policy keeps its grants and the :class:`AutoPolicy` its
    allowances across switches (:class:`ModePolicy`).

    The person is asked about a folder outside the workspace in either
    mode (:meth:`decide_folder`): auto mode never decides it.
    """

    ask: RememberingPolicy

    def __init__(
        self,
        ask: AskFunction,
        mode: ChatMode = "ask",
        *,
        saved: SavedFolders | None = None,
        watch: AskWatch | None = None,
        grants: SessionGrants | None = None,
    ) -> None:
        asking = RememberingPolicy(ask, folders=True, saved=saved, grants=grants)
        # A step Auto stops that the person may allow asks them on the
        # chat's card while *watch* says someone sees the chat, and an
        # answer for the chat is a grant of the Ask policy (#465).
        super().__init__(asking, AutoPolicy(asking=asking, watch=watch), mode)
        self.asks_folders = True

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self.ask.allowed_tools

    @property
    def allowed_folders(self) -> tuple[str, ...]:
        return self.ask.allowed_folders

    @property
    def grants(self) -> SessionGrants:
        return self.ask.grants

    def granted(self, name: str) -> bool:
        return self.ask.granted(name)

    def forget(self, tool_name: str | None = None) -> None:
        """Take back one grant, or every grant and folder (allowances stay)."""
        self.ask.forget(tool_name)

    def forget_folder(self, folder: str) -> bool:
        return self.ask.forget_folder(folder)

    async def decide_folder(self, request: ApprovalRequest) -> ApprovalDecision:
        return await self.ask.decide_folder(request)


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

    def ends(self, request_id: str, decision: ApprovalDecision) -> bool:
        """Answer *request_id* for a :class:`CardWatch`; whether its card was up.

        A card answered before counts as up: its answer stays.
        """
        return self.answer(request_id, decision) != "unknown"

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
                # Withdrawn unanswered: nobody denied the step itself.
                entry.answer.set_result(Withdrawal(False, reason, origin="policy"))
                denied += 1
        return denied


# ── A task's policy: its chat's grants (#369) ──


class OwnedTaskPolicy:
    """A task's policy: the grants of the chat that owns it, and *inner*.

    In Ask, before *inner* asks the person, a routine call of a tool the
    owner chat holds a grant of (its own, or the main chat's) runs as under
    a session grant, and so does a site question the person kept there. An
    Allow once is never kept: each call the grants do not cover asks. In
    Auto, *inner* decides (it asks nobody) and the chat's grants are not
    used. *book* holds every grant; *owner* gives the chat that owns the
    task now, and *started* when a chat began. A task the person did not
    start is never wrapped.

    The chat's grants are used only after :meth:`review_window` read the
    owner chat's start, and only while that chat still owns the task: the
    task's steps are then reviewed with every file changed since that chat
    began, so code an earlier task of the chat planted is read before the
    chat's grant covers a step. Without that start, the person is asked.

    A path outside the workspace (#405) is asked about through *folders*
    (the task's asking policy, also in Auto, which asks nobody), or the
    folder layer of *inner*: a read in a folder its owner chat allowed
    runs unasked.
    """

    def __init__(
        self,
        inner: ApprovalPolicy,
        book: GrantBook,
        owner: Callable[[], Awaitable[str | None]],
        *,
        started: Callable[[str], Awaitable[float | None]] | None = None,
        folders: ApprovalPolicy | None = None,
    ) -> None:
        self.inner = inner
        self._folders = folders
        self._book = book
        self._owner = owner
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
            if chat and _covered(self._book.folders_for(chat), outside.path):
                return ApprovalDecision(True, FOLDER_FOR_CHAT, origin="session")
        decision: ApprovalDecision = await asker.decide_folder(request)
        return decision

    async def review_window(self) -> float | None:
        """When the task's review window starts: its owner chat's start.

        ``None`` (Auto, which uses no grant of the chat, or a start that
        cannot be read): the task's steps are reviewed from the task's own
        start, and its chat's grants are not used.
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

    def _chat_grants(self) -> SessionGrants | None:
        """The grants of the chat that owns the task, read as that chat reads them.

        ``None`` in Auto, and unless the steps were reviewed from that chat's start.
        """
        chat = self._chat or ""
        if self._auto() or not self._reviewed_since(chat):
            return None
        return self._book.view(chat, "side")

    def review_mode(self, request: ApprovalRequest) -> ReviewMode:
        mode = review_mode(self.inner, request)
        grants = self._chat_grants()
        # As in the chat (RememberingPolicy.silent_for): the person's grant only.
        silent = grants is not None and grants.covers(request.tool_name)
        return "silent" if mode == "ask" and silent else mode

    def granted(self, name: str) -> bool:
        """Whether the question *name* is allowed: by the task's own grants, or its chat's."""
        grants = self._chat_grants()
        return (grants is not None and name in grants.held()) or holds_grant(
            self.inner, name
        )

    async def take_back_site(self, key: str, by: str | None = None) -> bool:
        """A refusal of a site: what the task kept goes, and what it holds from its chat (#577).

        Both at once, so a write held up in one does not leave the other
        holding. Whether all of that is saved.
        """
        taken: list[Awaitable[bool]] = [revoke_site(self.inner, key, by)]
        if self._chat:
            taken.append(self._book.take_back_site(self._chat, key, by))
        return all(await asyncio.gather(*taken))

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self._chat = await self._owner()
        grants = self._chat_grants()
        kept = None if grants is None else kept_answer(grants, request)
        return kept if kept is not None else await self.inner.decide(request)
