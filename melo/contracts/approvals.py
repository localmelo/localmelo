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

This module imports only the standard library, so the executor can use it
without an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, get_args, runtime_checkable

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
    root, or ``None``.
    """

    request_id: str
    run_id: str
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    preview: str
    workspace_root: str | None


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


def denial_text(decision: ApprovalDecision) -> str:
    """The text the model reads for a call that *decision* denied.

    A reason is stripped of surrounding whitespace and of a final ``.``,
    and keeps at most :data:`DENIAL_REASON_CHARS` characters. A timeout's
    text ignores the reason. ``ValueError`` if *decision* allows the call.
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
    if not reason:
        return f"{who} denied this call. {DENIAL_ENDING}"
    return f"{who} denied this call: {reason}. {DENIAL_ENDING}"
