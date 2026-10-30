"""Approval policies for the surfaces: session grants, and a broker for the web.

:class:`RememberingPolicy` asks a person, through an ask function, and
keeps "allow this tool for the session" as a grant of the tool's name.
:class:`ApprovalBroker` gives each session an ask function that waits for
an answer from elsewhere, such as the web console's approval route.
Grants and pending requests live in memory only: a restart forgets both.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Literal

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
from localmelo.melo.contracts.review import covered_by_grant, offers_grant

AskFunction = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]

# Answered request ids the broker remembers, so that a second answer reads
# already_resolved rather than unknown.
RESOLVED_IDS_KEPT: Final = 256


class RememberingPolicy:
    """An approval policy that asks through *ask* and keeps session grants.

    A routine call of a tool granted for the session runs without asking
    (``ApprovalDecision(True, origin="session")``); any other call is
    asked about, and an answer that allows with ``remember`` grants the
    tool's name. A grant covers routine calls only: a high-impact or
    undecided call (see :mod:`localmelo.melo.contracts.review`) is asked
    about every time, and a ``remember`` on a high-impact one is dropped,
    so it allows that call and grants nothing. Grants are per tool name and
    per policy object, and never stored.
    """

    def __init__(self, ask: AskFunction) -> None:
        self._ask = ask
        self._granted: set[str] = set()

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(self._granted)

    def forget(self, tool_name: str | None = None) -> None:
        """Take back the grant of *tool_name*, or every grant with ``None``."""
        if tool_name is None:
            self._granted.clear()
        else:
            self._granted.discard(tool_name)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
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


class RoutineOnlyPolicy:
    """Runs routine calls unasked and leaves every other call to *inner*.

    The terminal's auto mode: a routine call (or one nothing reviewed) is
    allowed with origin ``policy``; a high-impact or undecided one goes to
    *inner*, which asks.
    """

    def __init__(self, inner: ApprovalPolicy) -> None:
        self.inner = inner

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if covered_by_grant(request.review):
            return ApprovalDecision(True, origin="policy")
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
