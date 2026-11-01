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

A file tool's call on a path outside the workspace (#405) carries an
:class:`~localmelo.melo.contracts.folders.OutsideAccess`; only a policy
layer with ``decide_folder`` and ``asks_folders`` (see
:func:`find_folder_asker`) is asked about it, never ``decide``.

This module imports only the standard library, :mod:`localmelo.melo.schema`,
:mod:`localmelo.melo.contracts.review` and
:mod:`localmelo.melo.contracts.folders`, which import no other localmelo
module, so the executor can use it without an import cycle.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, get_args, runtime_checkable

from localmelo.melo.contracts.folders import OutsideAccess
from localmelo.melo.contracts.review import (
    CATEGORY_WORDS,
    ActionReview,
    because,
    block_words,
    loose_keys,
    step_key,
)
from localmelo.melo.schema import ToolTier

# Who decided: the person asked ("user"), a grant for the rest of the
# session ("session"), a fixed policy ("policy"), no answer in time
# ("timeout"), auto mode's review ("auto", #339): it runs a routine call
# and blocks the rest, and a denial's reason is then the whole note the
# model reads; or the person's earlier approval of the exact same step,
# found by code in the chat that owns a task ("earlier").
ApprovalOrigin = Literal["user", "session", "policy", "timeout", "auto", "earlier"]
# How a policy wants a call reviewed before it decides: "ask" (a person
# will be asked, or nothing runs unasked: the rules alone), "silent" (it
# would allow the call without a person, as under a session grant: the
# user's model reviews an undecided call first), or "auto" (auto mode).
ReviewMode = Literal["ask", "silent", "auto"]

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
    ``outside`` is set for a file tool's call on a path outside the
    workspace (#405): the person is asked about the folder, and a tool
    grant never covers it.
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
    outside: OutsideAccess | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """A policy's answer to an :class:`ApprovalRequest`.

    ``reason`` is given to the model on a denial (see :func:`denial_text`).
    ``remember`` asks to allow the tool for the rest of the session: the
    run reports it, and the surface's policy keeps the grant. For a folder
    outside the workspace (#405) it allows the folder for the chat, and
    ``always`` keeps the folder in the settings' extra folders as well.
    ``origin`` says who decided (see :data:`ApprovalOrigin`).

    A field of the wrong type raises ``TypeError``. ``ValueError`` is
    raised for an unknown origin, and for the pairs no policy may give:
    ``remember`` or ``always`` on a denial, ``origin="session"`` or
    ``"earlier"`` on a denial (a grant and an earlier approval only allow),
    and ``origin="timeout"`` on an allow (a timeout only denies).
    """

    allow: bool
    reason: str = ""
    remember: bool = False
    origin: ApprovalOrigin = "user"
    always: bool = False

    def __post_init__(self) -> None:
        for name, expected in (
            ("allow", bool),
            ("reason", str),
            ("remember", bool),
            ("origin", str),
            ("always", bool),
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
        if self.always and not self.allow:
            raise ValueError("ApprovalDecision.always needs allow=True")
        if self.origin in ("session", "earlier") and not self.allow:
            raise ValueError(
                f"ApprovalDecision.origin {self.origin!r} needs allow=True: "
                "a grant or an earlier approval only allows"
            )
        if self.origin == "timeout" and self.allow:
            raise ValueError(
                "ApprovalDecision.origin 'timeout' needs allow=False: "
                "a timeout only denies"
            )


class Withdrawal(ApprovalDecision):
    """A refusal that is nobody's denial of the step (#411).

    The question was withdrawn unanswered (the gateway stops) or could not be
    asked (the chat is not open yet): the call does not run, but nothing
    blocked the step itself, so a task its chat continues may try it again.
    Always a denial.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.allow:
            raise ValueError("a Withdrawal never allows")


@runtime_checkable
class ApprovalPolicy(Protocol):
    """Decides whether a marked tool call may run (see the module docstring)."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


# Set around each tool call of a run that has a policy: asks that policy about
# something the tool found out while it ran, such as the site a redirect led
# to. The arguments are the tool name the policy sees (a grant of the run
# covers that name), the arguments to show and a one-line preview. It raises
# like the policy does; unset (no policy) means nobody is asked. The agent's
# also answers ``names_host(host)``: whether the user's own words name a site.
ToolAsk = Callable[[str, Mapping[str, Any], str], Awaitable[ApprovalDecision]]
TOOL_ASK: ContextVar[ToolAsk | None] = ContextVar("localmelo_tool_ask", default=None)


def review_mode(policy: object, request: ApprovalRequest) -> ReviewMode:
    """How *policy* wants *request* reviewed (see :data:`ReviewMode`).

    A policy says so with an optional ``review_mode(request)`` method, or
    ``silent_for(request)`` (true: ``silent``); a wrapper with an ``inner``
    policy and neither is looked through. Anything else is ``ask``.
    """
    layer, depth = policy, 0
    while layer is not None and depth < 8:
        method = getattr(layer, "review_mode", None)
        if callable(method):
            mode = method(request)
            return mode if mode in ("ask", "silent", "auto") else "ask"
        silent = getattr(layer, "silent_for", None)
        if callable(silent):
            return "silent" if silent(request) is True else "ask"
        layer, depth = getattr(layer, "inner", None), depth + 1
    return "ask"


def find_folder_asker(policy: object) -> Any:
    """The layer of *policy* that may ask a person about a folder outside the workspace.

    That is the first layer, *policy* or one behind its ``inner``
    wrappers, with an async ``decide_folder(request)`` method and an
    ``asks_folders`` attribute that is ``True``; ``None`` when there is
    none. A policy without one is never asked about such a folder: the
    call is refused as it always was. A layer whose ``asks_folders`` is not
    ``True`` is looked through.
    """
    layer, depth = policy, 0
    while layer is not None and depth < 8:
        if callable(getattr(layer, "decide_folder", None)) and (
            getattr(layer, "asks_folders", False) is True
        ):
            return layer
        layer, depth = getattr(layer, "inner", None), depth + 1
    return None


def find_auto(policy: object) -> AutoPolicy | None:
    """The :class:`AutoPolicy` in *policy* or behind its ``inner`` wrappers."""
    layer, depth = policy, 0
    while layer is not None and depth < 8:
        if isinstance(layer, AutoPolicy):
            return layer
        layer, depth = getattr(layer, "inner", None), depth + 1
    return None


# How long an "Allow once" of a blocked step stays good, in seconds.
ALLOW_ONCE_S: Final = 30 * 60
# The reason of the decision that runs a step on its allowance.
ALLOWED_ONCE: Final = "allowed once"


class AutoPolicy:
    """Auto mode (#339): nobody is asked; a call that is not routine is blocked.

    The agent's gate reviews each call first (the rules, then the user's
    model for an undecided call, or for a high-impact one the user's words
    may cover), so this policy reads only the verdict: ``routine`` runs
    (origin ``auto``), anything else is denied with origin ``auto`` and the
    category in a few words as its reason. A call nothing reviewed is
    blocked too.

    :meth:`allow_once` keeps an allowance for one exact step
    (:func:`~localmelo.melo.contracts.review.step_key`, the files it read
    included) for :data:`ALLOW_ONCE_S` of wall-clock time; the next call of
    that step runs once, with origin ``user``, when its review finds no
    category the allowance did not cover. Allowances live in this object
    only; :attr:`given` keeps when each one given runs out, also once it was
    used or ran out, so a surface can tell a step allowed before.
    """

    def __init__(
        self, *, task: bool = False, clock: Callable[[], float] = time.time
    ) -> None:
        # A saved task's policy: a blocked step is offered on the task's page.
        self.task = task
        self._clock = clock
        # key -> (until, categories, tool) of each allowance not used yet.
        self._allowed: dict[str, tuple[float, frozenset[str], str]] = {}
        self.given: dict[str, float] = {}

    def review_mode(self, request: ApprovalRequest) -> ReviewMode:
        return "auto"

    def allow_once(self, key: str, categories: Iterable[str], tool: str = "") -> None:
        """Allow the step *key* once, for the next :data:`ALLOW_ONCE_S` seconds.

        *categories* are those of the blocked step the user saw: a run of
        it whose review finds any other is not allowed by this. *tool* is
        the step's tool, so only a block of that tool hears of it.
        """
        until = self._clock() + ALLOW_ONCE_S
        self._allowed[key] = (until, frozenset(categories), tool)
        self.given[key] = until

    def withdraw(self, key: str) -> bool:
        """Drop the allowance of *key* when it was not used; whether there was one.

        Its :attr:`given` entry goes too, so the step can be allowed again.
        """
        if self._allowed.pop(key, None) is None:
            return False
        self.given.pop(key, None)
        return True

    def allowed_once(self, key: str) -> bool:
        """Whether *key* has an allowance that has not run out (it is not used)."""
        kept = self._allowed.get(key)
        if kept is not None and kept[0] <= self._clock():
            del self._allowed[key]
            return False
        return kept is not None

    def has_allowances(self, tool: str | None = None) -> bool:
        """Whether a step (of *tool*, when given) is allowed once and not used or run out."""
        return any(
            self.allowed_once(key)
            for key, kept in list(self._allowed.items())
            if tool is None or kept[2] in ("", tool)
        )

    def forget(self) -> None:
        """Drop every allowance."""
        self._allowed.clear()

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        review = request.review
        key = step_key(
            request.tool_name,
            request.arguments,
            request.workspace_root,
            review.inputs if review is not None else (),
        )
        if (
            review is not None
            and self.allowed_once(key)
            and "self" not in review.categories
            and set(review.categories) <= self._allowed[key][1]
        ):
            del self._allowed[key]
            return ApprovalDecision(True, ALLOWED_ONCE, origin="user")
        if review is None:
            return ApprovalDecision(False, CATEGORY_WORDS["unreviewed"], origin="auto")
        if review.verdict == "routine":
            return ApprovalDecision(True, origin="auto")
        return ApprovalDecision(
            False, block_words(review) or "needs your approval", origin="auto"
        )


# Why a call is refused that was blocked before in the same task (#411);
# {who} is who blocked it: "Auto mode", "the user" or "melo's rules".
BLOCKED_EARLIER: Final = (
    "{who} blocked this exact step earlier in this task, so it does not run "
    "again; find another way that does not need it"
)
# A step alike (the same command spelt another way, the same install): refused too.
BLOCKED_ALIKE: Final = (
    "{who} blocked a step like this one earlier in this task (the same command "
    "or install), so it does not run again; find another way that does not need it"
)


def call_key(
    tool_name: str, arguments: Mapping[str, Any], workspace_root: str | None
) -> str:
    """What makes two calls the same step: the tool, the folder and the call's
    normalized arguments (Python as its syntax tree, a command as its text)."""
    return step_key(tool_name, arguments, workspace_root)


class EarlierBlocks:
    """A policy that refuses again, before any review, a step blocked earlier (#411).

    *blocks* maps a :func:`call_key` to ``(who, step_key)``: who blocked it
    and, for a block auto mode kept, its step key. *loose* maps the same
    call keys to their :func:`~localmelo.melo.contracts.review.loose_keys`,
    so a call alike (the same command spelt another way, the same install)
    is refused too. The agent's gate asks :meth:`match` first; :meth:`decide`
    is the *inner* policy's. A block the person allowed once since (its step
    key has an allowance in the inner :class:`AutoPolicy`) is not refused,
    nor is a call alike to it. It never allows anything.
    """

    def __init__(
        self,
        inner: ApprovalPolicy,
        blocks: Mapping[str, tuple[str, str]],
        loose: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.inner = inner
        self._blocks = dict(blocks)
        self._loose = {key: frozenset(keys) for key, keys in (loose or {}).items()}

    def _lifted(self, key: str) -> bool:
        auto = find_auto(self.inner)
        return (
            bool(key)
            and auto is not None
            and (auto.allowed_once(key) or key in auto.given)
        )

    def match(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        workspace_root: str | None,
    ) -> tuple[str, bool] | None:
        """Who blocked this call, or one alike, earlier, and whether it was this exact call."""
        exact = call_key(tool_name, arguments, workspace_root)
        found = self._blocks.get(exact)
        if found is not None and not self._lifted(found[1]):
            return found[0], True
        alike = set(loose_keys(tool_name, arguments))
        if not alike:
            return None
        for key, keys in self._loose.items():
            kept = self._blocks.get(key)
            if kept is not None and keys & alike and not self._lifted(kept[1]):
                return kept[0], False
        return None

    def blocked(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        workspace_root: str | None,
    ) -> str | None:
        """Who blocked this call, or one alike, earlier, or ``None`` when it may be asked."""
        found = self.match(tool_name, arguments, workspace_root)
        return None if found is None else found[0]

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return await self.inner.decide(request)


def find_earlier(policy: object) -> EarlierBlocks | None:
    """The :class:`EarlierBlocks` in *policy* or behind its ``inner`` wrappers."""
    layer, depth = policy, 0
    while layer is not None and depth < 8:
        if isinstance(layer, EarlierBlocks):
            return layer
        layer, depth = getattr(layer, "inner", None), depth + 1
    return None


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
    if decision.origin == "auto":
        # The gate wrote the whole note (every note has a line break, see
        # block_note); a bare policy's short reason gets the plain block.
        text = decision.reason.strip()
        return text if "\n" in text else block_note(review, words=text)
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


# What the model reads for a step auto mode blocked (#339). ``{it}`` is
# "It installs x; creates y." (or why the review failed); ``{off_goal}`` is
# " The user did not ask for it." when the model said so; ``{allow}`` says
# how the user can let it run.
BLOCK_NOTE: Final = (
    "Auto mode blocked this step, so it did not run. {it}{off_goal}\n"
    "In your answer, tell the user in plain words:\n"
    "- what you wanted to do, and why melo blocked it;\n"
    "- why the task needed it;\n"
    "- other ways to reach the goal: without this step, or with something "
    "the user can do;\n"
    "- {allow}\n"
    "Do not try to reach the same result another way (another command, a "
    "script, another tool): the user chose auto mode so that such steps do "
    "not run without them. Go on with any work that does not need it."
)
ALLOW_IN_CHAT: Final = (
    "that they can allow this step once from the chat, or say so in their "
    "next message."
)
ALLOW_IN_TASK: Final = (
    "that they can allow it once from the task's page; the task goes on " "without it."
)
ALLOW_NEVER: Final = (
    "that this step cannot be allowed: melo never changes its own runtime, "
    "settings or keys. Offer a way that works inside {folder}."
)
# A second block of the same kind in one turn: the run stops and answers.
BLOCK_STOP: Final = (
    "Auto mode blocked this step too: {it_lower}\nStop working on this. Your "
    "next reply is your answer to the user: explain what was blocked and why, "
    "why the task needed it, and the other ways, as above."
)
# A call of a turn auto mode already stopped: it does not run either.
STOPPED_NOTE: Final = (
    "Auto mode stopped this turn after blocking a step, so this step did not "
    "run.\nYour next reply is your answer to the user: explain what was "
    "blocked and why, why the task needed it, and the other ways."
)
# A retry of an allowed step that changed: the allowance named this one.
ALLOWED_EXACTLY: Final = (
    " The user allowed one exact step; this one differs. Run exactly the step "
    "that was blocked, or explain."
)


def _it(review: ActionReview | None, words: str = "") -> str:
    if review is not None and review.categories[:1] == ("unreviewed",):
        return "It could not be reviewed, so it was blocked."
    if review is not None and review.reasons:
        return f"It {because(review)}."
    if words:
        return f"It {words}."
    return "It needs the user's approval."


def block_note(
    review: ActionReview | None,
    *,
    words: str = "",
    task: bool = False,
    folder: str = "the folder",
    second: bool = False,
    changed: bool = False,
) -> str:
    """The text the model reads for a step auto mode blocked.

    *task*: the run is a saved task, whose page offers the allowance;
    *folder*: the workspace, for a step that can never be allowed; *second*:
    a second block of the same kind in this turn, which stops the run;
    *changed*: the user allowed a step and this retry differs from it.
    """
    it = _it(review, words)
    if second:
        return BLOCK_STOP.format(it_lower=it[:1].lower() + it[1:])
    off_goal = (
        " The user did not ask for it."
        if review is not None
        and review.model is not None
        and review.model.fits_request == "no"
        else ""
    )
    if review is not None and "self" in review.categories:
        allow = ALLOW_NEVER.format(folder=folder)
    else:
        allow = ALLOW_IN_TASK if task else ALLOW_IN_CHAT
    note = BLOCK_NOTE.format(it=it, off_goal=off_goal, allow=allow)
    return note + ALLOWED_EXACTLY if changed else note
