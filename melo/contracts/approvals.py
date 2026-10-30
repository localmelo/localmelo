"""Contract for asking an approval policy before a marked tool runs.

The executor consults the policy just before a tool marked as needing
approval would run, after every automatic check has passed. The agent
builds the :class:`ApprovalRequest`; the policy answers with an
:class:`ApprovalDecision`, and a denial reaches the model as
:func:`denial_text`.

Rules for a policy's :meth:`ApprovalPolicy.decide`:

- It may take as long as a person needs.
- When the run is cancelled while it decides, it receives the
  ``CancelledError``, must withdraw its prompt, and must re-raise. A
  withdrawn prompt is never a denial: the call does not run because the
  run stopped.
- The call is denied, and the run continues, when ``decide`` raises an
  exception, raises a ``CancelledError`` while the run is not being
  cancelled, or returns something other than an :class:`ApprovalDecision`.

This module imports only the standard library, :mod:`localmelo.melo.schema`
and :mod:`localmelo.melo.contracts.review`, which import no other localmelo
module, so the executor can use it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, get_args, runtime_checkable

from localmelo.melo.contracts.review import ActionReview, because
from localmelo.melo.schema import ToolTier

# Who decided: the person asked ("user"), a grant for the rest of the
# session ("session"), a fixed policy ("policy"), or no answer in time
# ("timeout").
ApprovalOrigin = Literal["user", "session", "policy", "timeout"]

# Ends the text of a denial by a person or a policy. Without it, models
# tend to reissue a denied call or reach its result another way.
DENIAL_ENDING: Final = (
    "Do not run it again or reach the same result another way unless the user asks."
)

# The text of a call that did not run because the policy failed; {error} is
# "<Type>: <message>".
APPROVAL_FAILED: Final = "Approval failed, so the call did not run: {error}"

# After the review clause of a high-impact call: a person's denial, and a
# policy's.
ASK_INSTEAD: Final = (
    "If the task needs it, say in your answer what for and ask the user."
)
WITHOUT_IT: Final = (
    "Finish what you can without it, and say in your answer what you would need."
)

# Characters of a denial's reason that its text keeps.
DENIAL_REASON_CHARS: Final = 500

_ORIGINS: Final = frozenset(get_args(ApprovalOrigin))


@dataclass(frozen=True)
class ApprovalRequest:
    """One call that waits for approval.

    ``request_id`` is ``uuid4().hex``, minted per request, so it is unique
    and cannot be guessed. ``run_id`` is the run's task id (``""`` for a
    call outside a run), and ``call_id`` the tool call's id. ``arguments``
    is a deep copy of exactly what will run, file paths resolved;
    ``preview`` is its one-line summary; ``workspace_root`` the agent's
    root, or ``None``; ``tier`` the tool's permission tier, or ``None``
    when a caller built the request itself. An agent always fills it, and
    ``review``: what the checker's rules found in the call (#279). A
    session grant covers only a call whose review is routine (or ``None``).
    """

    request_id: str
    run_id: str
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    preview: str
    workspace_root: str | None
    tier: ToolTier | None = None
    review: ActionReview | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """A policy's answer to an :class:`ApprovalRequest`.

    ``reason`` is given to the model on a denial (see :func:`denial_text`).
    ``remember`` asks to allow the tool for the rest of the session: the
    run reports it, and the surface's policy keeps the grant. ``origin``
    says who decided (see :data:`ApprovalOrigin`).

    A field of the wrong type raises ``TypeError``. ``ValueError`` is
    raised for an unknown origin, and for the pairs no policy may give:
    ``remember`` on a denial, and ``origin="session"`` on a denial (a
    session grant only allows) or ``origin="timeout"`` on an allow (a
    timeout only denies).
    """

    allow: bool
    reason: str = ""
    remember: bool = False
    origin: ApprovalOrigin = "user"

    def __post_init__(self) -> None:
        for name, expected in (
            ("allow", bool),
            ("reason", str),
            ("remember", bool),
            ("origin", str),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected):
                raise TypeError(
                    f"ApprovalDecision.{name} must be {expected.__name__}, "
                    f"got {value!r}"
                )
        if self.origin not in _ORIGINS:
            raise ValueError(
                "ApprovalDecision.origin must be one of "
                f"{', '.join(get_args(ApprovalOrigin))}, got {self.origin!r}"
            )
        if self.remember and not self.allow:
            raise ValueError("ApprovalDecision.remember needs allow=True")
        if self.origin == "session" and not self.allow:
            raise ValueError(
                "ApprovalDecision.origin 'session' needs allow=True: "
                "a session grant only allows"
            )
        if self.origin == "timeout" and self.allow:
            raise ValueError(
                "ApprovalDecision.origin 'timeout' needs allow=False: "
                "a timeout only denies"
            )


@runtime_checkable
class ApprovalPolicy(Protocol):
    """Decides whether a marked tool call may run (see the module docstring)."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


# Set around each tool call of a run that has a policy: asks that policy about
# something the tool found out while it ran, such as the site a redirect led
# to. The arguments are the tool name the policy sees (a grant of the run
# covers that name), the arguments to show and a one-line preview. It raises
# like the policy does; unset (no policy) means nobody is asked.
ToolAsk = Callable[[str, Mapping[str, Any], str], Awaitable[ApprovalDecision]]
TOOL_ASK: ContextVar[ToolAsk | None] = ContextVar("localmelo_tool_ask", default=None)


class AllowAllPolicy:
    """Allows every call."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True, origin="policy")


class DenyAllPolicy:
    """Denies every call, with *reason* for the model."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(False, self.reason, origin="policy")


def denial_text(decision: ApprovalDecision, review: ActionReview | None = None) -> str:
    """The text the model reads for a call that *decision* denied.

    A reason is stripped of surrounding whitespace and of a final ``.``,
    and keeps at most :data:`DENIAL_REASON_CHARS` characters. A timeout's
    text ignores the reason. For a high-impact *review*, the text says
    what the call does and what to do instead. ``ValueError`` if
    *decision* allows the call.
    """
    if decision.allow:
        raise ValueError("denial_text needs a decision that denies the call")
    if decision.origin == "timeout":
        return (
            "The call was not approved in time, so it did not run. Silence is "
            "not consent: do not retry it without asking the user."
        )
    who = "The user" if decision.origin == "user" else "The approval policy"
    reason = decision.reason.strip()
    if reason.endswith("."):
        reason = reason[:-1].rstrip()
    reason = reason[:DENIAL_REASON_CHARS]
    text = (
        f"{who} denied this call: {reason}." if reason else f"{who} denied this call."
    )
    if review is None or review.verdict != "high" or not review.reasons:
        return f"{text} {DENIAL_ENDING}"
    if decision.origin == "user":
        return f"{text} melo asked because it {because(review)}. {DENIAL_ENDING} {ASK_INSTEAD}"
    return f"{text} It {because(review)}. {DENIAL_ENDING} {WITHOUT_IT}"
