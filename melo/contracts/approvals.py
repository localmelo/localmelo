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

The agent asks every policy through :func:`ask`, which applies these rules.

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

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    Final,
    Literal,
    Protocol,
    TypeGuard,
    get_args,
    runtime_checkable,
)

from localmelo.melo.contracts.folders import OutsideAccess
from localmelo.melo.contracts.review import (
    CARD_ONLY,
    CATEGORY_WORDS,
    NAMED_CAN_ASK,
    SITE_KEY,
    UNREADABLE_SITE,
    WORDS_CAN_ALLOW,
    ActionReview,
    because,
    block_words,
    host_site,
    loose_keys,
    offers_grant,
    site_host,
    site_key_of,
    site_keys,
    step_key,
    url_site,
)
from localmelo.melo.schema import ToolTier

# Who decided: the person asked ("user"), a grant for the rest of the
# session ("session"), a fixed policy ("policy"), no answer in time
# ("timeout"), auto mode's review ("auto", #339): it runs a routine call
# and blocks the rest, and a denial's reason is then the whole note the
# model reads; or the person's words, sent while the card waited
# ("words"): they end the wait as a refusal and never allow anything.
# The person's Allow once is never kept: no origin means "allowed before".
ApprovalOrigin = Literal["user", "session", "policy", "timeout", "auto", "words"]
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

# What the model reads when the person's message ended the wait for an answer
# (origin ``words``): the call did not run and nothing was allowed.
WORDS_NOTE: Final = (
    "The user sent a message while melo waited for their answer, so the call "
    "did not run and nothing was allowed. Read their message. If the task "
    "still needs the call, run it again: melo asks them again."
)

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
# Why a card auto mode showed went unanswered (#465): none, no answer in time,
# or every viewer left first.
UNANSWERED_CODES: Final = frozenset({"", "time", "left"})


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
    grant never covers it. ``person_cjk`` says whether the person's own recent
    words have Chinese, Japanese or Korean characters (``None``: no words are
    known): a card shows the model's reason only in that script (#576).
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
    person_cjk: bool | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """A policy's answer to an :class:`ApprovalRequest`.

    ``reason`` is given to the model on a denial (see :func:`denial_text`).
    ``remember`` asks to allow the tool for the rest of the session: the
    run reports it, and the surface's policy keeps the grant. For a folder
    outside the workspace (#405) it allows the folder for the chat, and
    ``always`` keeps the folder in the settings' extra folders as well.
    ``origin`` says who decided (see :data:`ApprovalOrigin`). ``notice`` is
    a sentence the model reads after a denial, for the person.

    A field of the wrong type raises ``TypeError``. ``ValueError`` is
    raised for an unknown origin, and for the pairs no policy may give:
    ``remember`` or ``always`` on a denial, ``origin="session"`` on a
    denial (a grant only allows), and ``origin="timeout"`` or ``"words"``
    on an allow (a timeout and the person's words only deny).
    """

    allow: bool
    reason: str = ""
    remember: bool = False
    origin: ApprovalOrigin = "user"
    always: bool = False
    notice: str = ""

    def __post_init__(self) -> None:
        for name, expected in (
            ("allow", bool),
            ("reason", str),
            ("remember", bool),
            ("origin", str),
            ("always", bool),
            ("notice", str),
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
        if self.origin == "session" and not self.allow:
            raise ValueError(
                "ApprovalDecision.origin 'session' needs allow=True: "
                "a grant only allows"
            )
        if self.origin == "timeout" and self.allow:
            raise ValueError(
                "ApprovalDecision.origin 'timeout' needs allow=False: "
                "a timeout only denies"
            )
        if self.origin == "words" and self.allow:
            raise ValueError(
                "ApprovalDecision.origin 'words' needs allow=False: "
                "the person's words only deny"
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


@dataclass(frozen=True)
class Chosen(ApprovalDecision):
    """The person's answer to the card auto mode shows in place of a block (#465).

    Allow once, allow on this site for this chat (``remember``), or Don't
    allow with the person's optional note as ``reason``: origin ``user``. A
    card nobody answered is no answer, not a no: origin ``timeout``, with the
    human text of what happened as ``reason`` and its code as ``unanswered``:
    ``time`` (no answer in ``wait_s`` seconds, :data:`AUTO_ASK_WAIT_S`) or
    ``left`` (every viewer left first). :func:`denial_text` tells the model what
    happened: after a Don't allow, not to try the step again in this turn;
    after no answer, to go on with the rest and try it again in their next
    turn, when melo asks again. ``again`` is set on the refusal of a step
    declined, or left unanswered, earlier in the turn: nobody was asked this
    time (a card for the same step does not wait again in the same turn).
    """

    again: bool = False
    unanswered: str = ""
    wait_s: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.again, bool):
            raise TypeError(f"Chosen.again must be bool, got {self.again!r}")
        if self.origin not in ("user", "timeout"):
            raise ValueError("a Chosen answer is the person's: origin 'user'")
        if self.origin == "timeout" and self.allow:
            raise ValueError("a card nobody answered never allows")
        if self.again and self.allow:
            raise ValueError("Chosen.again needs allow=False")
        if self.unanswered not in UNANSWERED_CODES or (
            bool(self.unanswered) != (self.origin == "timeout")
        ):
            raise ValueError(
                "Chosen.unanswered is 'time' or 'left' for origin 'timeout' only, "
                f"got {self.unanswered!r}"
            )
        if self.wait_s is not None and self.unanswered != "time":
            raise ValueError("Chosen.wait_s goes with unanswered='time'")

    @classmethod
    def of(cls, decision: ApprovalDecision) -> Chosen:
        """*decision*, a person's answer or a timeout, as the answer to auto mode's card."""
        return cls(
            decision.allow,
            decision.reason,
            decision.remember,
            decision.origin,
            decision.always,
            unanswered="time" if decision.origin == "timeout" else "",
        )

    def record(self) -> dict[str, Any]:
        """What a call's approval record keeps of a card nobody answered: its code and wait."""
        if not self.unanswered:
            return {}
        if self.wait_s is None:
            return {"unanswered": self.unanswered}
        return {"unanswered": self.unanswered, "wait_s": self.wait_s}


@runtime_checkable
class ApprovalPolicy(Protocol):
    """Decides whether a marked tool call may run (see the module docstring)."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


def is_cancellation(exc: BaseException) -> TypeGuard[asyncio.CancelledError]:
    """Whether *exc* is a cancellation of the current task.

    That is a ``CancelledError`` while something cancels the task: its
    ``cancelling()`` count is above 0, as after ``task.cancel()``, a
    scope such as ``asyncio.timeout`` that cancels it, or the run's own
    ``timeout_s`` until the deadline turns the cancellation into a
    ``TimeoutError``. A ``CancelledError`` that a component raises while
    nothing cancels the task, for example a tool function's, is that
    component's failure.
    """
    if not isinstance(exc, asyncio.CancelledError):
        return False
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


async def ask(
    decide: Callable[[ApprovalRequest], Awaitable[object]],
    request: ApprovalRequest,
    *,
    record: Callable[[str, BaseException | None], None],
) -> ApprovalDecision:
    """``await decide(request)``, which must give an :class:`ApprovalDecision`.

    The one place a policy is asked, so that every site fails the same way,
    by the rules of the module docstring. *record* is told what came of a
    question that got no decision, and the caller refuses the call:

    - ``("withdrawn", None)``: the run was cancelled while it decided; the
      ``CancelledError`` propagates.
    - ``("error", exc)``: *exc* is raised. That is the exception *decide*
      raised, a ``RuntimeError`` for a ``CancelledError`` while nothing
      cancels the run, or a ``TypeError`` for a return that is no decision.
    """
    try:
        decided: object = await decide(request)
    except asyncio.CancelledError as exc:
        if is_cancellation(exc):
            record("withdrawn", None)
            raise
        spurious = RuntimeError(
            "the approval policy raised CancelledError while the run was not "
            "cancelled"
        )
        record("error", spurious)
        raise spurious from None
    except BaseException as exc:
        record("error", exc)
        raise
    if not isinstance(decided, ApprovalDecision):
        no_decision = TypeError(
            f"the approval policy returned {type(decided).__name__}, "
            "not an ApprovalDecision"
        )
        record("error", no_decision)
        raise no_decision
    return decided


# The question the browser provider asks before it presses a button in a window of
# another app (``desktop-act:<app>``). Like the window question below, only the person
# answers it, in Ask and in Auto: no model sees the window or the press.
DESKTOP_ACT: Final = "desktop-act:"

# Questions a tool asks itself whose answer is never kept for the chat, whatever a
# client sends: each call asks again. Sending a form is one (``browser-submit:<site>``)
# and so is typing (``browser-type:<site>``): a grant would cover every later form
# or text, whatever it holds and wherever it goes. A click at a point on a secret
# field, a consent or terms control, or inside a box from another site
# (``browser-point:<site>``) is one too, and so is a press in another app's window
# that appeared after melo's own act (``desktop-act:<app>``), and OK to a
# ``confirm`` box a page put up while melo acted (``browser-dialog:<site>``), and
# so is attaching a file to a page's upload field (``browser-attach:<site>``): a
# grant would cover every later file, whatever it holds.
NEVER_GRANTED: Final = (
    "browser-submit:",
    "browser-type:",
    "browser-attach:",
    "browser-point:",
    DESKTOP_ACT,
    "browser-dialog:",
    "takeover:",
)

# The tools melo ships that ask their own questions (per site, and for every text they
# type or form they send), so the gate puts no second, empty question about the call.
# Two things must hold for a call to skip the gate: its name is on this fixed list in
# melo, and the tool came from melo's own browser provider, which says so with
# ``ProvidedTool.asks_itself``. A tool's annotation never counts (an MCP server cannot
# claim to ask for itself), and neither does a name alone: a tool that library code
# registers as ``browser_type`` keeps its approval.
SELF_ASKING_TOOLS: Final = frozenset(
    {
        "browser_click",
        "browser_type",
        "browser_select",
        "browser_click_point",
        "browser_attach",
        "window_press",
    }
)


# The question the browser provider asks when another app puts up a window right after
# melo's own act and the page did not ask for that app (``window-read:<bundle id>``). Only
# the person answers it, in Ask and in Auto: no model sees the window, so none reviews it.
# "Allow <App> in this chat" grants this name for the chat (the per-chat grant of
# :class:`~localmelo.support.sessions.approvals.RememberingPolicy`).
WINDOW_READ: Final = "window-read:"
# The card that asks the person to take over a page of melo's browser (a sign-in,
# a code, a QR code, a human check): only the person answers it, never a review.
TAKEOVER: Final = "takeover:"
# The question the browser provider asks before it presses inside a human check
# (#583), only where code proves the press is in one: a frame the browser itself
# names a known check provider's. In Ask and in Auto only the person answers it,
# and "Always in this chat" keeps it. Its name is ``browser-check:<page origin>``
# (:func:`check_scope`), and a kept answer covers such checks on every site.
HUMAN_CHECK: Final = "browser-check:"
# An origin in the one spelling ``sites.origin_of`` gives it: an http(s) scheme, a
# host in lower-case ASCII with no dot at its end (an IPv6 address in brackets),
# and a port only when it is not the scheme's own.
_ORIGIN: Final = re.compile(
    r"(https?)://"
    r"([a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)*"
    r"|\[[0-9a-f:.]+\])"
    r"(?::([1-9][0-9]{0,4}))?"
)
# The questions only the person answers whose answer a chat may keep: another app's
# window, and a human check.
PERSON_KEPT: Final = (WINDOW_READ, HUMAN_CHECK)


def check_question(page: str) -> str:
    """The name of the human-check question on the page of origin *page*."""
    return f"{HUMAN_CHECK}{page}"


def check_scope(name: str) -> str | None:
    """The page origin the human-check question *name* is asked on, spelt once.

    ``None`` for any other name, and for one whose origin is not spelt the one
    way the browser provider writes it (upper case, a dot at the end of the
    host, the scheme's own port written out, another host form): such a name is
    no check question, and nothing it keeps covers anything.
    """
    if not name.startswith(HUMAN_CHECK):
        return None
    return spelt_origin(name[len(HUMAN_CHECK) :])


def spelt_origin(page: str) -> str | None:
    """*page* when it is an origin spelt the one way the browser provider writes it, else ``None``.

    Upper case, a dot at the end of the host, the scheme's own port written
    out or another host form is no origin here (:data:`_ORIGIN`).
    """
    found = _ORIGIN.fullmatch(page)
    if found is None:
        return None
    scheme, host, port = found.groups()
    bare = host[1:-1] if host.startswith("[") else host
    if site_host(bare) != bare or (bare != host) != (":" in bare):
        return None
    if port is not None and (int(port) > 65535 or int(port) == WEB_PORTS[scheme]):
        return None
    return page


def check_covers(kept: str, name: str) -> bool:
    """Whether a kept answer to the human-check question *kept* covers the question *name* (#583).

    Both are human checks of a known provider, on any two sites: a kept
    answer covers every such check. Nothing else.
    """
    return check_scope(kept) is not None and check_scope(name) is not None


def kept_check(request: ApprovalRequest) -> bool:
    """Whether *request* is a human check a kept answer may cover (:func:`person_grant` of :data:`HUMAN_CHECK`).

    The one question an answer the chat kept settles in Auto with nobody
    watching, and the one a task or a scheduled task takes from its chat
    without the person's review (#583).
    """
    return request.tool_name.startswith(HUMAN_CHECK) and person_grant(request)


def window_question(request: ApprovalRequest) -> bool:
    """Whether *request* is the question whether melo may see another app's window."""
    return request.tool_name.startswith(WINDOW_READ) and request.outside is None


def person_grant(request: ApprovalRequest) -> bool:
    """Whether a chat's kept answer to a question only the person answers covers *request*.

    Another app's window (``window-read:<app>``) and a human check
    (:data:`HUMAN_CHECK`, a name :func:`check_scope` reads), as asked in Ask (no
    review) or as Auto shows them to the person (a ``high`` review of the one
    category ``opaque``): nothing else is covered.
    """
    review = request.review
    return (
        request.outside is None
        and request.tool_name.startswith(PERSON_KEPT)
        and not never_kept(request)
        and (review is None or review.categories == ("opaque",))
    )


def never_granted(tool_name: str) -> bool:
    """Whether an answer to the question *tool_name* names may never be kept for a chat."""
    return tool_name.startswith(NEVER_GRANTED)


def never_kept(request: ApprovalRequest) -> bool:
    """Whether no answer to *request* may ever be kept, by the person or by Auto.

    The one list the offer (:func:`grant_offered`), Auto's keep
    (:func:`stays_for_chat`) and the grants table's check read: a question of
    :data:`NEVER_GRANTED`, a human check's whose name :func:`check_scope` does
    not read, or a step of a card-only kind (:data:`CARD_ONLY`: paying, an
    account's settings, opening an app).
    """
    name, review = request.tool_name, request.review
    return (
        never_granted(name)
        or (name.startswith(HUMAN_CHECK) and check_scope(name) is None)
        or (review is not None and bool(set(review.categories) & CARD_ONLY))
    )


# The site questions whose answer on auto mode's card may be kept for the chat
# (#465): opening or reading a site, and clicking or choosing on it. Typing and
# sending a form never (:data:`NEVER_GRANTED`).
SITE_GRANTS: Final = ("browser:", "browser-act:")


def whole_site(name: str) -> str | None:
    """The site a kept answer to the question *name* covers: ``example.com`` (#577).

    ``None`` for a question that is not one of :data:`SITE_GRANTS`, and for an
    origin :func:`~localmelo.melo.contracts.review.host_site` does not read as
    one site (the origin stays exact: a port that is not the scheme's own has
    one written, which no plain name holds). An origin is spelt once, so a host
    a second reading would change (a dot left at its end) is no origin's.
    """
    if not name.startswith(SITE_GRANTS):
        return None
    scheme, found, host = name.partition(":")[2].partition("://")
    if not found or scheme not in ("http", "https") or site_host(host) != host:
        return None
    return host_site(host)


def site_covers(kept: str, name: str) -> bool:
    """Whether a kept answer to the question *kept* covers the question *name* on another host (#577).

    The same kind of question, the same scheme and the same :func:`whole_site`.
    The stored name stays as it was given: hosts are matched here, at lookup.
    """
    site = whole_site(name)
    return (
        site is not None
        and site == whole_site(kept)
        and name.partition("://")[0] == kept.partition("://")[0]
    )


def kept_covers(kept: str, name: str) -> bool:
    """Whether the person's kept answer to the question *kept* covers the question *name* of another name.

    Another host of its site (:func:`site_covers`, #577) or another human check
    (:func:`check_covers`, #583); a name of its own is covered by its own row.
    """
    return site_covers(kept, name) or check_covers(kept, name)


def site_grant(request: ApprovalRequest) -> bool:
    """Whether a chat's grant of *request*'s question may cover it in auto mode (#465).

    A question about a site (:data:`SITE_GRANTS`) that auto mode's review
    stopped only because the request did not ask for it: its categories are
    ``beyond_request`` and nothing else, with nothing the review could not
    tell. A grant of the question's name ("this kind of step on this site")
    then lets it run for the rest of the chat. A step with any high-impact
    category is never covered (#279), nor typing or a form sent.
    """
    review = request.review
    return (
        request.outside is None
        and request.tool_name.startswith(SITE_GRANTS)
        and not never_granted(request.tool_name)
        and bool(request.arguments.get("site"))
        and review is not None
        and review.verdict == "high"
        and review.categories == ("beyond_request",)
        and not review.doubts
    )


# The site questions auto mode's run keeps allowed for the chat (#465): opening
# or reading a site, never clicking or choosing on it. Nobody reviews a step in
# Ask, so a click kept from Auto would let any button of the site run there with
# no card ("Pay" too): Ask asks about every click of a site the person did not
# allow themselves.
STAYS_FOR_CHAT: Final = ("browser:",)
# What a point question (``browser-point:<site>``) says in its ``how`` when the
# point is typed into rather than clicked: the one value a card and the heading
# read as typing, as the browser provider asks it (#576).
POINT_TYPING: Final = "type into"
# The port a web address has when it names none, by scheme.
WEB_PORTS: Final = {"http": 80, "https": 443}


def plain_address(url: object) -> bool:
    """Whether a site question's *url* is a plain address: no query, fragment or
    user name, and shown whole. Any other can carry a secret or more data out."""
    return (
        isinstance(url, str)
        and bool(url.strip())
        and not url.endswith("…")
        and not any(mark in url for mark in "?#@")
    )


def stays_for_chat(request: ApprovalRequest) -> bool:
    """Whether auto mode's run of *request* keeps its question allowed for the chat (#465).

    A question about opening or reading a site (:data:`STAYS_FOR_CHAT`) with a
    ``site`` argument and, when it carries the address it reads (web_fetch's
    ``url``), a plain one (:func:`plain_address`): a site Auto let the chat
    read stays allowed for the rest of the chat, in Ask too. Never a click or
    a choice, typing, a form sent, a click at a point, a path outside the
    workspace, or a step of a card-only kind (:data:`CARD_ONLY`): nothing
    implicit covers what the person must allow themselves.
    """
    return (
        request.outside is None
        and request.tool_name.startswith(STAYS_FOR_CHAT)
        and not never_kept(request)
        and bool(request.arguments.get("site"))
        and ("url" not in request.arguments or plain_address(request.arguments["url"]))
    )


def grant_offered(request: ApprovalRequest) -> bool:
    """Whether an answer to *request* may be kept for the chat ("Allow for this chat").

    A routine or unreviewed call of a tool whose answers may be kept
    (:func:`~localmelo.melo.contracts.review.offers_grant`), or a site auto
    mode stopped only as not asked for (:func:`site_grant`). Never a
    high-impact call, anything :func:`never_kept` names, or a path outside the
    workspace (whose folder has a grant of its own, #405).
    """
    if request.outside is not None or never_kept(request):
        return False
    return offers_grant(request.review) or site_grant(request) or person_grant(request)


def kept_offered(request: ApprovalRequest) -> bool:
    """Whether a kept answer is offered for *request*: never for what
    :func:`never_kept` names; a folder outside the workspace by its own rule
    (``outside.grantable``, #405), else :func:`grant_offered`.
    """
    outside = request.outside
    return not never_kept(request) and (
        outside.grantable if outside is not None else grant_offered(request)
    )


# Set around each tool call of a run that has a policy: asks that policy about
# something the tool found out while it ran, such as the site a redirect led
# to. The arguments are the tool name the policy sees (a grant of the run
# covers that name), the arguments to show and a one-line preview. It fails
# as a call's approval does (:func:`ask`): the policy's exception is raised, a
# ``CancelledError`` while the run is not cancelled and a return that is no
# decision are raised as an error; unset (no policy) means nobody is asked.
# The agent's also answers ``names_host(host)``: whether the user's own words
# name a site, and ``refused_earlier(name, arguments, preview)``: the refusal
# of a question blocked earlier in the task (#431), which a tool checks before
# it asks; ``refused_unasked(name, arguments, preview)``: that, or one on a site
# refused in this run (#577), which a tool checks before it lets something pass
# with no question; and ``granted(name)``: whether the run's chat holds a grant
# of the question *name* (:func:`holds_grant`).
ToolAsk = Callable[[str, Mapping[str, Any], str], Awaitable[ApprovalDecision]]
TOOL_ASK: ContextVar[ToolAsk | None] = ContextVar("localmelo_tool_ask", default=None)
# Whether the run of the running tool may act (``Agent.run_task(acts=)``): what a
# tool reaches through another tool, a search through the browser, reads it.
RUN_ACTS: ContextVar[bool] = ContextVar("localmelo_run_acts", default=True)


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


def _grants_layer(policy: object, method: str) -> Any:
    """*method* of the first layer that has it: *policy*, or one behind its ``inner`` wrappers or an Auto policy's ``asking``."""
    layer, depth = policy, 0
    while layer is not None and depth < 8:
        found = getattr(layer, method, None)
        if callable(found):
            return found
        below = getattr(layer, "inner", None)
        layer, depth = (
            below if below is not None else getattr(layer, "asking", None)
        ), depth + 1
    return None


def holds_grant(policy: object, name: str) -> bool:
    """Whether *policy* holds a grant of the question *name* for the chat.

    The first layer with a ``granted(name)`` method answers. A tool that keeps
    no question of its own asks this before it uses something the person
    allowed for the chat, such as another app's window
    (``window-read:<app>``). A question of :data:`NEVER_GRANTED` is never held.
    """
    check = None if never_granted(name) else _grants_layer(policy, "granted")
    return check is not None and check(name) is True


# What the model reads when the file did not take a person's refusal (#577).
REFUSAL_NOT_SAVED: Final = (
    "Tell the person: this refusal holds for this run, but it could not be "
    "saved. melo may ask again, and after a restart the answer they kept may "
    "hold again until they take it back in Settings, under Permissions."
)


async def revoke_site(policy: object, key: str, by: str | None = None) -> bool:
    """Take back what the chat or task of *policy* keeps about the site *key* (#577).

    *key* is the ``ask:site:<site>`` of a refusal. A person's refusal takes
    every answer kept there, on either scheme, so a later turn asks again and
    no refusal has to be remembered. With *by* only that decider's go.
    Whether it is saved: the answers stop holding here either way.
    """
    take_back = _grants_layer(policy, "take_back_site")
    return True if take_back is None else await take_back(key, by)


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


def allowance_key(request: ApprovalRequest) -> str:
    """The key an Allow once of *request* is kept under, and looked up by (:class:`AutoPolicy`).

    The exact step: its tool, or the name of the question its tool asked
    (``browser-type:<origin>``), every argument (for a question, the whole
    text and where it goes), the folder and the files its review read.
    """
    review = request.review
    return step_key(
        request.tool_name,
        request.arguments,
        request.workspace_root,
        review.inputs if review is not None else (),
    )


class Allowance(ApprovalDecision):
    """The decision that runs one step on the person's Allow once of it in auto mode.

    Its type, not its text, says how the call was decided: the gate runs it
    as auto mode's step (confined), and nothing keeps it. Always the
    person's allow (origin ``user``), and never ``remember`` or ``always``:
    an allowance is used once and goes.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.allow or self.origin != "user":
            raise ValueError("an Allowance is the person's allow: origin 'user'")
        if self.remember or self.always:
            raise ValueError("an Allowance is used once and never kept")


# The site questions that hold the whole step: the text typed, the form sent and
# where it goes. Each is reviewed by itself, and its verdict is never kept for
# the next; only such a question shows a person all it does, so only such a one
# may be allowed once after Auto blocked it (#617), or kept for a scheduled task
# (#619). Each question's name, and the kind of step its ``access`` says.
WHOLE_STEPS: Final = {"browser-type:": "type", "browser-submit:": "submit"}


def whole_step(arguments: Mapping[str, Any]) -> bool:
    """Whether a site question with *arguments* holds all its step does (:data:`WHOLE_STEPS`)."""
    return arguments.get("access") in WHOLE_STEPS.values()


def kept_site(name: str, arguments: Mapping[str, Any]) -> tuple[str, str] | None:
    """The site and kind a scheduled task's kept answer to the question *name* holds (#619), or ``None``.

    The site is the origin the question is asked on (scheme, host and port),
    read from its name and spelt once (:func:`spelt_origin`); the kind is
    ``type`` or ``submit`` (:data:`WHOLE_STEPS`), and *arguments* say the same
    kind. A form sent has a site only when the address it goes to (``to``) is
    on that same origin: one sent elsewhere, or where the page decides
    (``to`` empty), has none.
    """
    kind = next((k for p, k in WHOLE_STEPS.items() if name.startswith(p)), None)
    site = None if kind is None else spelt_origin(name.partition(":")[2])
    if kind is None or site is None or arguments.get("access") != kind:
        return None
    to = arguments.get("to")
    if kind == "submit" and not (
        isinstance(to, str) and (to == site or to.startswith(site + "/"))
    ):
        return None
    return site, kind


def kept_review(review: ActionReview | None) -> bool:
    """Whether a scheduled task's kept answer may stand for the person's answer after *review* (#619).

    The model reviewed the step, said it fits the request (``fits_request``
    ``yes``; ``user_asked`` is not read: a scheduled run has nobody asking),
    and every category the review names is ``send_out`` ("Send data from this
    Mac?"): the one question a kept answer answers, the posting rules' stops
    for a repeating task and for a text the person neither wrote nor asked to
    send among them. A step filed under anything else (passwords or keys,
    money, an account's settings, something the person did not ask for), one
    the model found off the request or was unsure of, one that could not be
    reviewed, and a routine one (it needs no answer) are not.
    """
    return (
        review is not None
        and review.verdict == "high"
        and review.by == "model"
        and review.model is not None
        and review.model.fits_request == "yes"
        and bool(review.categories)
        and set(review.categories) == {"send_out"}
    )


def keep_holds(
    name: str,
    arguments: Mapping[str, Any],
    review: ActionReview | None,
    *,
    personal: bool,
) -> tuple[str, str] | None:
    """Whether a scheduled task's kept answer would cover this very step: the site and kind it holds, or ``None`` (#619).

    The one predicate the agent's gate (a step that runs on a kept answer) and
    the offer (a block one may be made from, and the route that makes it)
    both use: the review (:func:`kept_review`), the site and kind
    (:func:`kept_site`), and no detail of the person's: *personal* is whether
    melo's own detector finds their phone number, street address or real name
    in what the step sends, whatever the review said.
    """
    if personal or not kept_review(review):
        return None
    return kept_site(name, arguments)


@dataclass(frozen=True)
class ScheduledKeep:
    """What one scheduled task's runs may type and send without asking (#619).

    *chore_id* is the scheduled task; *sites* maps each kept site (an origin)
    to the kinds kept there (``type``, ``submit``). Trusted code reads it from
    the scheduled task's store when one of its runs starts; nothing a run
    says changes it.
    """

    chore_id: str
    sites: Mapping[str, frozenset[str]]

    def keeps(self, site: str, kind: str) -> bool:
        """Whether this keeps the kind of step *kind* on *site* (as :func:`keep_holds` gives them)."""
        return kind in self.sites.get(site, ())


# The reason of the decision that runs a step on a scheduled task's kept answer.
KEPT_FOR_SCHEDULE: Final = "allowed for this scheduled task"


class Kept(Allowance):
    """The decision that runs a typing or sending step on its scheduled task's kept answer (#619).

    The person's own answer, kept by them for that scheduled task and site
    (origin ``user``): like an :class:`Allowance` it lifts nothing else on
    the site, and a site refused in the run still stands over it.
    """


# Auto mode (#465): the categories of a stopped step a person may allow from a
# card: those the user's words can allow, a step melo could not check, a step
# whose review failed (it timed out, could not be read or ran out of room: the
# person decides it instead of the step stopping without a word), and the kinds
# only a person allows (paying, an account's settings). Credentials and code
# from the internet count too when every finding is about a key or a script the
# person's own words name (:data:`NAMED_CAN_ASK`, set by code in the run). A step
# with any other category (system settings, an app driven by a script,
# something that runs later, localmelo itself) stays blocked, as before.
ASKS_IN_AUTO: Final = WORDS_CAN_ALLOW | {"opaque", "unreviewed"} | CARD_ONLY


def asks_in_auto(review: ActionReview | None) -> bool:
    """Whether auto mode asks a person about a step it stopped, when one can be asked (#465).

    A step whose review is not routine and whose every category is in
    :data:`ASKS_IN_AUTO`, or one the person's words name
    (``review.named``, only :data:`NAMED_CAN_ASK` categories count). A step
    nothing reviewed, or one without a category (undecided and never
    settled), is not asked about.
    """
    return (
        review is not None
        and review.verdict != "routine"
        and bool(review.categories)
        and set(review.categories) <= ASKS_IN_AUTO | set(review.named) & NAMED_CAN_ASK
    )


# How long the card auto mode shows waits for the person (#465): a card nobody
# answers in this time did not run its step, and is not a no: the step waits for
# their answer (:data:`UNANSWERED_NOTE`). The one place this time is set.
AUTO_ASK_WAIT_S: Final = 60.0


def wait_words(seconds: float) -> str:
    """*seconds* in words: ``1 minute``, ``2 minutes``, ``45 seconds``."""
    if seconds >= 60 and seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes} minute" + ("" if minutes == 1 else "s")
    return f"{seconds:g} second" + ("" if seconds == 1 else "s")


# The human text of a card nobody answered in time, and of one whose every
# viewer left before an answer (each is no answer yet, not a Don't allow); code reads
# ``Chosen.unanswered`` ("time", "left"), never these.
NO_ANSWER: Final = f"no answer within {wait_words(AUTO_ASK_WAIT_S)}"
LEFT_UNANSWERED: Final = "the chat was closed before an answer"


class AskWatch(Protocol):
    """Who sees the card auto mode shows, and how long it waits (#465).

    Auto mode asks only while :meth:`watched` (a console, the app or the
    terminal shows the chat); with nobody watching, the step is blocked as
    before. :meth:`ask` asks through *asking* and ends a card nobody answers
    as a refusal with origin ``timeout``: the step did not run and waits.
    """

    async def watched(self) -> bool:
        """Whether a person can see a card now."""
        ...

    async def ask(
        self, request: ApprovalRequest, asking: ApprovalPolicy
    ) -> ApprovalDecision:
        """``asking.decide(request)``, ended as a ``timeout`` deny when nobody answers."""
        ...


class AutoPolicy:
    """Auto mode (#339): a routine call runs; a call that is not routine is stopped.

    The agent's gate reviews each call first (the rules, then the user's
    model for an undecided call, or for a high-impact one the user's words
    may cover), so this policy reads only the verdict: ``routine`` runs
    (origin ``auto``). A stopped call that a person may allow
    (:func:`asks_in_auto`) goes to *asking*, the policy that asks a person
    (the chat's card, #465), and the answer comes back as :class:`Chosen`.
    First, for a human check (:func:`kept_check`), *asking*'s optional
    ``kept(request)`` says what the chat kept: an allow there runs the press
    whether or not anybody watches (#583). Every other question goes on below.
    With a *watch*, it asks only while a person sees the chat, and a card
    nobody answers did not run its step and is not a no (:data:`AUTO_ASK_WAIT_S`). Without
    *asking* (nobody can be asked: a schedule, a benchmark), with nobody
    watching, and for any other stopped call, it is denied with origin
    ``auto`` and the category in a few words as its reason. A call nothing
    reviewed is blocked too. A site read it runs (:func:`stays_for_chat`) is
    told to *asking* through its optional ``passed_in_auto(name)``, so the
    site's plain reads stay allowed for the chat in Ask too; a click never.

    :meth:`allow_once` keeps an allowance for one exact step
    (:func:`~localmelo.melo.contracts.review.step_key`, the files it read
    included) for :data:`ALLOW_ONCE_S` of wall-clock time; the next call of
    that step runs once, with origin ``user``, when its review finds no
    category the allowance did not cover. Allowances live in this object
    only; :attr:`given` keeps when each one given runs out, also once it was
    used or ran out, so a surface can tell a step allowed before.

    :attr:`kept` is what a scheduled task's run may type and send without
    asking (:class:`ScheduledKeep`, #619): set by trusted code for a run of
    that scheduled task only, and read by the agent's gate for a typing or
    sending question after its review; ``None`` everywhere else.
    """

    def __init__(
        self,
        *,
        task: bool = False,
        clock: Callable[[], float] = time.time,
        asking: ApprovalPolicy | None = None,
        watch: AskWatch | None = None,
    ) -> None:
        # A saved task's policy: a blocked step is offered while its job is unfinished.
        self.task = task
        self._clock = clock
        # The policy that asks a person about a stopped step (#465); None
        # when nobody can be asked, so the step is blocked.
        self.asking = asking
        # Who sees the card and how long it waits; None: always, unbounded.
        self.watch = watch
        # key -> (until, categories, tool) of each allowance not used yet.
        self._allowed: dict[str, tuple[float, frozenset[str], str]] = {}
        self.given: dict[str, float] = {}
        self.kept: ScheduledKeep | None = None

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
        key = allowance_key(request)
        if (
            review is not None
            and self.allowed_once(key)
            and "self" not in review.categories
            and set(review.categories) <= self._allowed[key][1]
        ):
            del self._allowed[key]
            return Allowance(True, ALLOWED_ONCE, origin="user")
        if review is None:
            return ApprovalDecision(False, CATEGORY_WORDS["unreviewed"], origin="auto")
        if review.verdict == "routine":
            if stays_for_chat(request):
                # A site Auto let this chat read stays allowed for the chat
                # (#465): the asking policy keeps it, for Ask. Never a click.
                keep = getattr(self.asking, "passed_in_auto", None)
                if callable(keep):
                    keep(request.tool_name)
            return ApprovalDecision(True, origin="auto")
        asking, watch = self.asking, self.watch
        kept = getattr(asking, "kept", None) if kept_check(request) else None
        found = kept(request) if callable(kept) else None
        if isinstance(found, ApprovalDecision) and found.allow:
            # A human check the chat kept is pressed alike whether anybody watches
            # (#583): the person's answer holds while they are away too. Every
            # other kept answer waits for somebody watching, as before.
            return found
        if (
            asking is not None
            and asks_in_auto(review)
            and (watch is None or await watch.watched())
        ):
            # The person chooses on the card instead of a block (#465). A
            # grant of the chat that covers the step answers with origin
            # "session"; a withdrawn card stays what it is. A card nobody
            # answered did not run: no answer, not a no.
            decided = await (
                asking.decide(request) if watch is None else watch.ask(request, asking)
            )
            if type(decided) is ApprovalDecision and (
                decided.origin == "user"
                or (decided.origin == "timeout" and not decided.allow)
            ):
                return Chosen.of(decided)
            return decided
        if asking is None and self.task and request.tool_name.startswith(HUMAN_CHECK):
            # A task nobody can be asked for (a chore, one melo or another agent
            # started) never shows a card, even with its chat open: the model reads
            # what is true instead of "keep the chat open" (#583).
            return ApprovalDecision(False, TASK_ASKS_NOBODY, origin="auto")
        return ApprovalDecision(
            False, block_words(review) or "needs your approval", origin="auto"
        )


# Why a site question is refused that something other than the person would
# have run: a kept answer, Auto, a page approved before (#577).
SITE_REFUSED: Final = (
    "This site was refused earlier in this task, and only the user's own answer "
    "stands over that, so this does not run; find another way that does not "
    "need it"
)
# Why a call is refused that was blocked before in the same task (#411);
# {who} is who blocked it: "Auto mode", "the user" or "melo's rules".
BLOCKED_EARLIER: Final = (
    "{who} blocked this exact step earlier in this task, so it does not run "
    "again; find another way that does not need it"
)
# A step alike (the same command spelt another way, the same install): refused too.
BLOCKED_ALIKE: Final = (
    "{who} blocked a step like this one earlier in this task (the same command, "
    "install or site), so it does not run again; find another way that does not "
    "need it"
)
# Why a call of the job whose effect is unknown asks the person (#571): an
# earlier run of the same call was cut off, so it may have done its work.
UNKNOWN_RESULT: Final = (
    "An earlier run of this exact step was cut off before its result was known, "
    "so it may have done its work already; only the person decides that it runs "
    "again."
)


def call_key(
    tool_name: str, arguments: Mapping[str, Any], workspace_root: str | None
) -> str:
    """What makes two calls the same step: the tool, the folder and the call's
    normalized arguments (Python as its syntax tree, a command as its text)."""
    return step_key(tool_name, arguments, workspace_root)


def question_key(name: str) -> str:
    """The loose key of a question a running tool asked (``TOOL_ASK``) under *name*.

    A tool asks under a name of its own choosing, such as ``browser:<origin>``
    for a site, and a grant of that name covers every question with it; so a
    refusal does too: a later question under the same name (another page of a
    refused site, by either tool) is a step alike (#431).
    """
    return f"ask:{name}"


def site_key(arguments: Mapping[str, Any]) -> str | None:
    """The loose key of a question about a site: ``ask:site:<site>`` (#431, #577).

    The host of the question's http(s) ``origin`` argument, in ASCII (IDNA),
    lower case and without one final dot, read as one site
    (:func:`~localmelo.melo.contracts.review.site_key_of`): a refused site
    stays refused on its other hosts and whatever scheme, port or spelling a
    later address gives it. A host not read as one site keys itself. None when
    the question names no such origin.
    """
    origin = arguments.get("origin")
    host = url_site(origin) if isinstance(origin, str) else None
    return None if host is None else site_key_of(host)


def question_site_key(name: str) -> str | None:
    """The :func:`site_key` of the question *name* (``browser:<origin>``, ``browser-act:<origin>``…), or None for one on no site."""
    return site_key({"origin": name.partition(":")[2]})


class EarlierBlocks:
    """A policy that refuses again, before any review, a step blocked earlier (#411).

    *blocks* maps a :func:`call_key` to ``(who, step_key)``: who blocked it
    and, for a block auto mode kept, its step key. *loose* maps the same
    call keys to their :func:`~localmelo.melo.contracts.review.loose_keys`,
    so a call alike (the same command spelt another way, the same install)
    is refused too. The agent's gate asks :meth:`match` first, and so does
    a running tool's own question (a site web_fetch or the browser would
    open, #431); :meth:`decide` is the *inner* policy's. A block the person
    allowed once since (its step key has an allowance in the inner
    :class:`AutoPolicy`) is not refused, nor is a call alike to it. It never
    allows anything.

    A site refused earlier (a block with an ``ask:site:<site>`` key, kept
    from a tool's own site question, read again as one site when it was kept
    under its host before #577) stays refused for a ``shell_exec`` or
    ``python_exec`` step that names an address on it (#433,
    :func:`~localmelo.melo.contracts.review.site_keys`), and one naming an
    address whose encoded host does not decode is refused while any site is.

    *fresh* are the :func:`call_key` of calls of the job whose effect is
    unknown (#571), each as the call's start kept it (``run_key``). None is
    refused: the agent's gate asks the person about each, whatever its tool's
    tier, the mode, a grant or an earlier answer say, and only their answer
    runs it, that once (see ``_ApprovalGate``).
    """

    def __init__(
        self,
        inner: ApprovalPolicy,
        blocks: Mapping[str, tuple[str, str]],
        loose: Mapping[str, Sequence[str]] | None = None,
        fresh: Iterable[str] = (),
    ) -> None:
        self.inner = inner
        self.fresh = frozenset(fresh)
        self._blocks = dict(blocks)
        self._loose = {
            key: frozenset(
                site_key_of(k[len(SITE_KEY) :]) if k.startswith(SITE_KEY) else k
                for k in keys
            )
            for key, keys in (loose or {}).items()
        }

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
        also: Iterable[str] = (),
    ) -> tuple[str, bool] | None:
        """Who blocked this call, or one alike, earlier, and whether it was this exact call.

        *also* are further loose keys of the call: a tool's own question
        adds its :func:`question_key`.
        """
        exact = call_key(tool_name, arguments, workspace_root)
        found = self._blocks.get(exact)
        if found is not None and not self._lifted(found[1]):
            return found[0], True
        alike = {*loose_keys(tool_name, arguments), *site_keys(tool_name, arguments)}
        alike |= set(also)
        if not alike:
            return None
        unreadable = UNREADABLE_SITE in alike
        for key, keys in self._loose.items():
            kept = self._blocks.get(key)
            if kept is None or self._lifted(kept[1]):
                continue
            if keys & alike or (
                unreadable and any(k.startswith(SITE_KEY) for k in keys)
            ):
                return kept[0], False
        return None

    def refused_sites(self) -> list[str]:
        """The hosts of the sites refused earlier and not allowed since, oldest first."""
        hosts: dict[str, None] = {}
        for key, keys in self._loose.items():
            kept = self._blocks.get(key)
            if kept is None or self._lifted(kept[1]):
                continue
            for name in sorted(keys):
                if name.startswith(SITE_KEY):
                    hosts[name[len(SITE_KEY) :]] = None
        return list(hosts)

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


def is_refusal(decision: ApprovalDecision) -> bool:
    """Whether *decision* refuses the step: a person's Deny, an auto block, a policy's refusal.

    Not a question nobody answered, a person's message that ended the wait, or
    a withdrawn one: nobody said no to the step.
    """
    return (
        not decision.allow
        and decision.origin in ("user", "auto", "policy")
        and not isinstance(decision, Withdrawal)
    )


def denial_text(decision: ApprovalDecision, review: ActionReview | None = None) -> str:
    """The text the model reads for a call that *decision* denied.

    The person's Don't allow on auto mode's card (a :class:`Chosen`) reads
    :func:`chosen_note`. Otherwise a reason is stripped of surrounding
    whitespace and of a final ``.``,
    and keeps at most :data:`DENIAL_REASON_CHARS` characters. A timeout's
    text ignores the reason. For a high-impact *review*, the text says
    what the call does and what to do instead. ``ValueError`` if
    *decision* allows the call.
    """
    if decision.allow:
        raise ValueError("denial_text needs a decision that denies the call")
    if isinstance(decision, Chosen):
        # Don't allow on auto mode's card (#465): what the person chose.
        return chosen_note(decision)
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
    if decision.origin == "words":
        return WORDS_NOTE
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
# how the user can let it run. melo asks the person itself, on a card: the
# model never asks them to type a permission (#465).
BLOCK_NOTE: Final = (
    "Auto mode blocked this step, so it did not run. {it}{off_goal}\n"
    "In your answer, tell the user in plain words:\n"
    "- what you wanted to do, and why melo blocked it;\n"
    "- why the task needed it;\n"
    "- other ways to reach the goal: without this step, or with something "
    "the user can do;\n"
    "- {allow}\n"
    "Never ask the user to type a permission: when a step needs them, melo "
    "asks them itself on a card, and when they want it done, you just try "
    "the step again.\n"
    "Do not try to reach the same result another way (another command, a "
    "script, another tool): the user chose auto mode so that such steps do "
    "not run without them. Go on with any work that does not need it."
)
ALLOW_IN_CHAT: Final = (
    "that they can allow this step once from the chat, or switch the chat to Ask "
    "so that melo asks them first."
)
# The same for a task nobody can be asked for (a scheduled run, a task melo or
# another agent started, #619): the person decides on the task's page later, so
# the model is not told to try the step again, nor to ask for a permission.
# {decide} is what the person can do there (:data:`DECIDE_ON_PAGE`).
UNATTENDED_NOTE: Final = (
    "Auto mode blocked this step, so it did not run. {it}{off_goal}\n"
    "Nobody can be asked during this run: it runs on a schedule, or melo or "
    "another agent started it. {decide}\n"
    "Do not try this step again in this run, and do not reach the same result "
    "another way (another command, a script, another tool). Go on with the work "
    "that does not need it. A task that cannot finish without it ends with "
    "Ending: blocked. In your answer, say in plain words what was blocked and why "
    "the task needed it.\n"
    "Never ask the user to type a permission."
)
# By whether the person can allow the step from the task's page.
DECIDE_ON_PAGE: Final = {
    True: "The user will see this step on the task's page and can allow it there.",
    False: (
        "The user will see on the task's page that it was blocked; it cannot be "
        "allowed from the task."
    ),
}
# A step a person may allow on auto mode's card that was blocked because nobody
# could be asked now (no one had the chat open, or the run has nobody to ask).
ALLOW_WHEN_OPEN: Final = (
    "that this step needs them and nobody could be asked now: they can allow "
    "this step once from the chat, switch the chat to Ask, or keep the chat open "
    "so that melo asks them on a card the next time."
)
# The same two for a step the person cannot allow once from the chat (a question
# that does not hold its whole step, a text holding a secret, a site refused).
ASK_WHEN_OPEN: Final = (
    "that this step needs them and nobody could be asked now: they can keep the "
    "chat open so that melo asks them on a card the next time, or switch the chat "
    "to Ask."
)
BLOCKED_IN_CHAT: Final = (
    "that this step stays blocked in Auto: they can do it themselves, change what "
    "they ask for, or switch the chat to Ask so that melo asks them first."
)
# A question only the person can answer (another app's window, a press in one, a page to
# take over) that Auto refused because nobody has the chat open: the model reads the
# truth, so it does not make up a reason.
NOBODY_TO_ASK: Final = (
    "Auto mode did not run this step: only the user can answer it, and nobody has "
    "the chat open to ask.\n"
    "In your answer, tell the user in plain words:\n"
    "- what you wanted to do, and that only they can answer it;\n"
    "- that they can keep the chat open so that melo asks them on a card the "
    "next time, or switch the chat to Ask.\n"
    "That is the only reason. Do not give any other reason (a network problem, a "
    "fault, a limit) and do not guess one.\n"
    "Do not try to reach the same result another way."
)

# The same for a human check in a task nobody can be asked for in Auto (a scheduled
# task, one melo or another agent started): no card comes, whoever has the chat
# open, so the model reads what the person can do instead (#583).
TASK_ASKS_NOBODY: Final = (
    "Auto mode did not run this step: only the user can answer it, and this task "
    "runs in Auto with nobody to ask (it runs on a schedule, or melo or another "
    "agent started it), so no card is shown to them.\n"
    "In your answer, tell the user in plain words:\n"
    "- what you wanted to do, and that only they can answer it;\n"
    "- that a scheduled task follows its chat: they can open that chat and "
    "choose Always in this chat on a human check's card there, or run this task "
    "in Ask.\n"
    "That is the only reason. Do not give any other reason (a network problem, a "
    "fault, a limit) and do not guess one.\n"
    "Do not try to reach the same result another way."
)
# A worker is one-time: once its job finished, the step is new work, not an allowance.
ALLOW_IN_TASK: Final = (
    "that the task goes on without it, and that once it has finished this step is "
    "new work they ask the chat for. A task that cannot finish without the step "
    "ends with Ending: blocked instead, and then they can allow it once. Tried "
    "again before they do, it is reviewed again and blocked; once they allow it, "
    "melo tells you which step, and you run exactly that step again."
)
# A step of a task that the person cannot allow from the task (a text a review
# filed under passwords or keys, a site refused in the task).
BLOCKED_IN_TASK: Final = (
    "that the task goes on without it, and that this step cannot be allowed "
    "from the task: tried again, it is reviewed again and blocked. If it is "
    "needed, they can ask for it in a chat, where they decide on it themselves."
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


# What the model reads when the person chose Don't allow on auto mode's card
# (#465); {note} is " Their note: ..." when they wrote one.
DONT_ALLOW_NOTE: Final = (
    "The user chose Don't allow for this step, so it did not run.{note}\n"
    "Do not try it again in this turn or reach the same result another way: "
    "melo will not ask them again. Go on with any work that does not need it, "
    "and say in your answer what you could not do."
)
# The same step, or one alike, tried again in the turn: refused without asking.
DECLINED_AGAIN: Final = (
    "The user chose Don't allow for this step earlier in this turn, so it did not "
    "run and they were not asked again.{note}\n"
    "Do not try it again in this turn or reach the same result another way. Go on "
    "with any work that does not need it, and say in your answer what you could "
    "not do."
)


# A card nobody answered (#465), and the same step tried again in the turn;
# {reason} is what happened (:data:`NO_ANSWER`, :data:`LEFT_UNANSWERED`). No
# answer is not a no: the person may be away. The step waits, the model goes on
# with the rest of the work, and melo asks again when they are back (the next
# turn: a new run asks about the step again).
UNANSWERED_NOTE: Final = (
    "The user has not answered the card for this step yet ({reason}), so it did "
    "not run. That is not a no: they may be away.\n"
    "Go on with the parts of the task that do not need it, and say in your answer "
    "which part waits for their answer. melo asks them again when they are back: "
    "try the step again in their next turn."
)
UNANSWERED_AGAIN: Final = (
    "The user has not answered the card for this step yet ({reason}), so it did "
    "not run, and melo did not ask again in this turn.\n"
    "Go on with the parts of the task that do not need it, and say in your answer "
    "which part waits for their answer. melo asks them again when they are back: "
    "try the step again in their next turn."
)


def chosen_note(decision: Chosen) -> str:
    """What the model reads for a Don't allow on auto mode's card (#465).

    That the person chose it, their note (whitespace folded, at most
    :data:`DENIAL_REASON_CHARS` characters) when they wrote one, and not to
    try the step again in this turn; for a step declined earlier in the turn
    (``again``), that they were not asked again. A card nobody answered
    (origin ``timeout``) says so, with what happened.
    """
    reason = " ".join(decision.reason.split())[:DENIAL_REASON_CHARS]
    if decision.origin == "timeout":
        template = UNANSWERED_AGAIN if decision.again else UNANSWERED_NOTE
        return template.format(reason=reason or NO_ANSWER)
    note = ""
    if reason:
        note = f" Their note: {reason}"
        if not reason.endswith((".", "!", "?")):
            note += "."
    return (DECLINED_AGAIN if decision.again else DONT_ALLOW_NOTE).format(note=note)


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
    allowable: bool = True,
    unattended: bool = False,
) -> str:
    """The text the model reads for a step auto mode blocked.

    *task*: the run is a saved task, offered the allowance while its job is
    unfinished; *allowable*: the person can allow the step once (in a task or
    a chat), else the note says what they can do instead;
    *folder*: the workspace, for a step that can never be allowed; *second*:
    a second block of the same kind in this turn, which stops the run;
    *changed*: the user allowed a step and this retry differs from it;
    *unattended*: a task nobody can be asked for (:data:`UNATTENDED_NOTE`).
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
    elif task and unattended:
        note = UNATTENDED_NOTE.format(
            it=it, off_goal=off_goal, decide=DECIDE_ON_PAGE[allowable]
        )
        return note + ALLOWED_EXACTLY if changed else note
    elif task:
        allow = ALLOW_IN_TASK if allowable else BLOCKED_IN_TASK
    elif asks_in_auto(review):
        # A kind the card asks about was blocked only because nobody could be
        # asked now (#465); Allow once on the block only where it has a key.
        allow = ALLOW_WHEN_OPEN if allowable else ASK_WHEN_OPEN
    else:
        allow = ALLOW_IN_CHAT if allowable else BLOCKED_IN_CHAT
    note = BLOCK_NOTE.format(it=it, off_goal=off_goal, allow=allow)
    return note + ALLOWED_EXACTLY if changed else note
