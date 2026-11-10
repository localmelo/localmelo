"""The approval gate of a run: what asks the policy before a call runs.

:class:`_ApprovalGate` is the executor's gate for one call of a run with a
policy; :class:`_ToolAsk` is what a running tool (the browser tools) uses to
ask the same policy its own questions. Both take the run's state, the policy
and the checker, never the Agent, and keep what came of a question in the run
(``run.reviews``) and in the record of the call.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.agent._step_review import (
    StepReviewer,
    attach_review,
    holds_secret,
    plain_page_key,
)
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.checker.review.gist import step_gist
from localmelo.melo.checker.review.posting import is_message, personal_data, sent_texts
from localmelo.melo.contracts.approvals import (
    BLOCKED_ALIKE,
    BLOCKED_EARLIER,
    DESKTOP_ACT,
    HUMAN_CHECK,
    NOBODY_TO_ASK,
    REFUSAL_NOT_SAVED,
    SITE_REFUSED,
    STOPPED_NOTE,
    TAKEOVER,
    TASK_ASKS_NOBODY,
    UNKNOWN_RESULT,
    WINDOW_READ,
    Allowance,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    AutoPolicy,
    Chosen,
    ReviewMode,
    allowance_key,
    ask,
    asks_in_auto,
    block_note,
    call_key,
    find_auto,
    find_earlier,
    find_folder_asker,
    holds_grant,
    is_refusal,
    question_key,
    question_site_key,
    review_mode,
    revoke_site,
    site_covers,
    site_key,
)
from localmelo.melo.contracts.folders import (
    NAMED_FOLDER,
    OutsideAccess,
    attach_refusal,
    canonical_path,
    covers,
    home_label,
)
from localmelo.melo.contracts.review import (
    CATEGORY_WORDS,
    RULED_TOOLS,
    SELF_DENIAL,
    SITE_KEY,
    TITLES,
    ActionReview,
    address_keys,
    block_words,
    loose_keys,
    reaches_unread_site,
    step_key,
)
from localmelo.melo.executor import AutoConfinement, call_preview
from localmelo.melo.executor.models import ExecutionRequest
from localmelo.melo.schema import ToolCall, ToolTier, exception_text

if TYPE_CHECKING:
    from localmelo.melo.agent.agent import _RunState


class _ApprovalGate:
    """The executor's approval gate for one call of a run with a policy.

    The executor calls it just before a tool that requires approval would
    run (see :meth:`Executor.execute_structured`). It builds the
    :class:`ApprovalRequest` from the request that would run, asks the
    run's policy, and keeps what came of it in :attr:`asked` for the
    call's ``tool.finished`` event. While the review and the policy decide,
    ``run.waiting_call_id`` names the call, so that a cancellation there
    closes the call as not started (see :meth:`Agent._on_cancelled`).

    With a *checker*, the request carries the call's review (#279), kept in
    :attr:`review` for the denial text; a call that would change localmelo
    itself (the ``self`` category) is denied without asking the policy.
    With a *reviewer*, the user's model settles a call the rules left open
    when the policy would let it run without a person (a session grant, or
    auto mode; see :mod:`localmelo.melo.agent._step_review`). In auto mode
    (#339) a denial is a block: the model reads the note of
    :func:`~localmelo.melo.contracts.approvals.block_note`, and a second
    block of the same kind sets ``run.reviews.stop``; the step's later
    calls are then blocked unreviewed. A tool the rules do not read
    (:data:`~localmelo.melo.checker.review.RULED_TOOLS`) that is not
    read-only is undecided in auto mode, so the user's model reviews it,
    unless the person allowed that tool for the chat (#583).
    While a site is refused in the task (#433), a ``shell_exec`` or
    ``python_exec`` call the rules find routine that reaches a host melo
    cannot read (:func:`~localmelo.melo.contracts.review.reaches_unread_site`)
    is undecided too when it would run without a person, so the user's
    model reviews it, told which sites were refused.

    A file tool's call on a path outside the workspace (#405) is asked of
    the policy's folder layer only
    (:func:`~localmelo.melo.contracts.approvals.find_folder_asker`), and
    only in a run the person started: auto mode never decides it and no
    tool grant covers it (see :meth:`asks_folders` and :meth:`_folder`).

    A call of the job whose effect is unknown (#571, ``EarlierBlocks.fresh``,
    :attr:`fresh`) is asked about whatever its tool (``ExecutionRequest.ask``),
    a self-asking one too, and reviewed as only a person settles it: high
    impact, of code melo cannot check (:func:`_fresh_review`). No grant, earlier
    answer or model review lets it run, and Auto shows it on its card; the
    person's Allow runs it that once.

    The run's cancellation propagates, and the prompt counts as withdrawn.
    The policy fails closed otherwise: an exception it raises, a
    ``CancelledError`` while nothing cancels the run, and a return value
    other than an :class:`ApprovalDecision` each reach the executor as an
    exception, and the executor refuses the call.
    """

    def __init__(
        self,
        policy: ApprovalPolicy,
        run: _RunState,
        call: ToolCall,
        workspace_root: str | None,
        tier: ToolTier,
        checker: Checker | None = None,
        reviewer: StepReviewer | None = None,
        destructive: bool = False,
        confinement: AutoConfinement | None = None,
        asks_itself: bool = False,
    ) -> None:
        self._policy = policy
        self._run = run
        self._call = call
        self._workspace_root = workspace_root
        self._tier = tier
        self._checker = checker
        self._reviewer = reviewer
        self._destructive = destructive
        self._confinement = confinement
        self._asks_itself = asks_itself
        self.review: ActionReview | None = None
        # The approval of tool.finished: request_id, allowed, reason,
        # remember and origin, once the policy is asked; None until then.
        self.asked: dict[str, Any] | None = None
        self._mode: ReviewMode = "ask"
        self._key = ""
        self._allowance = False
        # A call of the job whose effect is unknown, by the key its start keeps (#571).
        blocks = find_earlier(policy)
        self.fresh = (
            blocks is not None
            and bool(blocks.fresh)
            and call_key(call.tool_name, call.arguments, workspace_root) in blocks.fresh
        )

    def asks_folders(self, outside: OutsideAccess) -> bool:
        """Whether a person may be asked about *outside* (#405).

        Only in a run the person started, and only when the policy has a
        layer that asks about folders; otherwise the executor refuses the
        path as it always did.
        """
        return self._run.from_person and find_folder_asker(self._policy) is not None

    async def __call__(self, request: ExecutionRequest) -> ApprovalDecision:
        earlier = self._earlier(request)
        if earlier is not None:
            return earlier
        if self._asks_itself and not self.fresh:
            # One of melo's own tools that ask their questions themselves (the
            # browser tools that act: per site, for every text typed and every
            # form sent) is not asked a second, empty question here about its
            # call. Only the names melo lists count, never a tool's annotation.
            return ApprovalDecision(True, origin="policy")
        if request.outside is not None:
            return await self._folder(request, request.outside)
        run = self._run
        if self._checker is not None:
            # In a thread: a runner's review walks the workspace.
            self.review = await asyncio.to_thread(
                self._checker.review_action,
                ActionReviewPayload(
                    request.tool_name,
                    dict(request.arguments),
                    self._workspace_root,
                    since=None if self._reviewer is None else self._reviewer.since,
                    destructive=self._destructive,
                ),
            )
        if self._reviewer is not None and self.review is not None:
            # A key or a script the person's words name asks on a card (#547).
            self.review = self._reviewer.name(self.review)
        if self.fresh:
            self.review = _fresh_review(self.review)
        asking = self._asking(request, review=self.review)
        preview = asking.preview
        self._mode = review_mode(self._policy, asking)
        auto = find_auto(self._policy) if self._mode == "auto" else None
        if (
            self._mode == "auto"
            and self.review is not None
            and self.review.verdict == "routine"
            and request.tool_name not in RULED_TOOLS
            and self._tier != "read"
            # A tool the person allowed for the chat runs its routine calls as in
            # Ask: no forced review (#583).
            and not holds_grant(self._policy, request.tool_name)
        ):
            self.review = ActionReview(
                "undecided",
                (),
                (f"Runs {request.tool_name}, a tool melo's rules do not read.",),
            )
            asking = dataclasses.replace(asking, review=self.review)
        refused = _refused_sites(self._policy, run)
        if (
            self._mode != "ask"
            and refused
            and self.review is not None
            and self.review.verdict == "routine"
            and reaches_unread_site(request.tool_name, request.arguments)
        ):
            # A site is refused in this task: a step that reaches a host melo
            # cannot read is reviewed, not run as routine (#433).
            self.review = dataclasses.replace(
                self.review, verdict="undecided", reasons=(UNREAD_SITE,)
            )
            asking = dataclasses.replace(asking, review=self.review)
        run.reviews.count(self.review)
        if self.review is not None and (
            self._mode != "ask" or self.review.verdict != "routine"
        ):
            self._key = step_key(
                request.tool_name,
                request.arguments,
                self._workspace_root,
                self.review.inputs,
            )
        if self._mode == "auto" and run.reviews.stop:
            # The turn already stopped: the step's later calls do not run.
            self._record(asking, origin="auto", reason="the turn stopped")
            return ApprovalDecision(False, STOPPED_NOTE, origin="auto")
        if self._mode == "auto":
            again = run.reviews.declined_again(
                [
                    call_key(
                        request.tool_name, request.arguments, self._workspace_root
                    ),
                    *loose_keys(request.tool_name, request.arguments),
                ]
            )
            if again is not None:
                # This step, or one alike, was declined on the card earlier in
                # this turn (the person's Don't allow, or no answer): nobody is
                # asked again (#465).
                self._record(asking, origin=again.origin, reason=again.reason)
                assert self.asked is not None
                self.asked["declined"] = True
                self.asked.update(again.record())
                return again
        if self.review is not None and "self" in self.review.categories:
            # No model reviews it and no policy is asked.
            self._counted(request.tool_name, preview)
            if self._mode == "auto":
                words = CATEGORY_WORDS["self"]
                self._record(asking, origin="auto", reason=words, denial=True)
                return ApprovalDecision(
                    False, self._block(asking, auto, words), origin="auto"
                )
            self._record(asking, origin="policy", reason=SELF_DENIAL, denial=True)
            return ApprovalDecision(False, SELF_DENIAL, origin="policy")

        async def settle(asked: ApprovalRequest) -> ApprovalDecision:
            # The review is asked in the same guard as the policy: a failure of
            # either refuses the call alike.
            if (
                self._reviewer is not None
                and self.review is not None
                and not self.fresh
                and not (auto is not None and auto.allowed_once(self._key))
            ):
                self.review = await self._reviewer.review(
                    run.reviews,
                    tool_name=request.tool_name,
                    arguments=dict(request.arguments),
                    preview=preview,
                    review=self.review,
                    mode=self._mode,
                    key=self._key,
                    workspace=self._workspace_root,
                    refused_sites=refused,
                )
                asked = dataclasses.replace(asked, review=self.review)
            self._counted(request.tool_name, preview)
            return await self._policy.decide(asked)

        decided = await self._ask(settle, asking)
        # Who decided, not the mode the call began in: a chat may switch
        # while the call is reviewed. Auto's card nobody answered is auto
        # mode's decision (#465).
        allowance = isinstance(decided, Allowance)
        unanswered = isinstance(decided, Chosen) and decided.origin == "timeout"
        if decided.origin == "auto" or allowance or unanswered:
            self._mode = "auto"
            auto = find_auto(self._policy) or auto
        elif self._mode == "auto":
            self._mode = "ask"
        self._allowance = decided.allow and allowance
        if decided.allow and self._confinement is not None:
            # A call auto mode let run starts confined; one a person allowed
            # does not.
            self._confinement.on = self._mode == "auto"
            if self._confinement.on:
                await asyncio.to_thread(self._confinement.ready)
            # An allowed step may write the paths its review found in it, as
            # allowed: its sandbox lets them through, so "Allow" takes effect
            # instead of the operating system refusing the write (#547).
            self._confinement.allow(() if self.review is None else self.review.paths)
        self._record(
            asking,
            origin=decided.origin,
            allowed=decided.allow,
            reason=decided.reason,
            remember=decided.remember,
            # A person's Deny, an auto block, a policy's refusal of the step
            # itself; never a question withdrawn or one that could not be asked.
            denial=is_refusal(decided),
        )
        if isinstance(decided, Chosen) and not decided.allow:
            # Don't allow on the card, or no answer, holds for the rest of the
            # turn (#465); a card nobody answered keeps why, for the surfaces.
            assert self.asked is not None
            self.asked.update(decided.record())
            run.reviews.decline(
                [
                    call_key(
                        request.tool_name, request.arguments, self._workspace_root
                    ),
                    *loose_keys(
                        request.tool_name,
                        request.arguments,
                        () if self.review is None else self.review.categories,
                    ),
                ],
                decided,
            )
        if not decided.allow and decided.origin == "auto":
            return dataclasses.replace(
                decided, reason=self._block(asking, auto, decided.reason)
            )
        return decided

    def _earlier(self, request: ExecutionRequest) -> ApprovalDecision | None:
        """A refusal of a step blocked earlier in this task, before any review (#411).

        Only a policy with :class:`~localmelo.melo.contracts.approvals.EarlierBlocks`
        in it refuses: a task the chat continued on its own. The record says
        who blocked the step first (``approval.earlier``).
        """
        blocks = find_earlier(self._policy)
        found = (
            None
            if blocks is None
            else blocks.match(
                request.tool_name, request.arguments, self._workspace_root
            )
        )
        if found is None:
            return None
        who, exact = found
        asking = self._asking(request, outside=request.outside)
        reason = (BLOCKED_EARLIER if exact else BLOCKED_ALIKE).format(who=who)
        self._record(asking, origin="policy", reason=reason)
        assert self.asked is not None
        self.asked["earlier"] = who
        return ApprovalDecision(False, reason, origin="policy")

    async def _folder(
        self, request: ExecutionRequest, outside: OutsideAccess
    ) -> ApprovalDecision:
        """Ask the person about a call on a path outside the workspace (#405).

        The policy's folder layer decides; its answer may be a grant of the
        folder for the chat or in the settings, which only a read uses. A
        write is reviewed by the rules first and is high-impact: a write
        into localmelo itself is denied without asking. Failures and the
        run's cancellation go as they go for any other call.
        """
        run = self._run
        self._mode = "ask"
        if outside.access == "write":
            review = None
            if self._checker is not None:
                review = await asyncio.to_thread(
                    self._checker.review_action,
                    ActionReviewPayload(
                        request.tool_name,
                        dict(request.arguments),
                        self._workspace_root,
                    ),
                )
            self.review = _outside_write(review, outside)
            run.reviews.count(self.review)
        asking = self._asking(request, review=self.review, outside=outside)
        asker = find_folder_asker(self._policy)
        if asker is None or not run.from_person:
            # asks_folders said yes, so only a policy changed meanwhile.
            self._record(asking, origin="policy", reason=NO_FOLDER_ASKER)
            return ApprovalDecision(False, NO_FOLDER_ASKER, origin="policy")
        if self.review is not None and "self" in self.review.categories:
            self._record(asking, origin="policy", reason=SELF_DENIAL)
            return ApprovalDecision(False, SELF_DENIAL, origin="policy")
        if (
            self._reviewer is not None
            and review_mode(self._policy, asking) == "auto"
            and (self.review is None or _only_outside_write(self.review))
            and self._reviewer.names_folder(outside.path)
        ):
            # In Auto, a folder the person wrote in their own words needs no
            # card (#547), for a read or a plain write there. A write the rules
            # also call something else (a launch agent, a shell startup file, a
            # key) or could not read keeps its card: naming a folder is not
            # asking for that.
            decided: ApprovalDecision = ApprovalDecision(
                True, NAMED_FOLDER, origin="user"
            )
        else:
            decided = await self._ask(asker.decide_folder, asking)
        if decided.allow and decided.origin == "auto":
            # Only a person, a grant of the folder or the settings allow it.
            refused = ApprovalDecision(False, NO_FOLDER_ASKER, origin="policy")
            self._record(asking, origin="policy", reason=NO_FOLDER_ASKER)
            return refused
        self._record(
            asking,
            origin=decided.origin,
            allowed=decided.allow,
            reason=decided.reason,
            remember=decided.remember,
        )
        assert self.asked is not None
        self.asked["folder"] = {**outside.to_json(), "scope": _folder_scope(decided)}
        return decided

    def _asking(
        self,
        request: ExecutionRequest,
        *,
        review: ActionReview | None = None,
        outside: OutsideAccess | None = None,
    ) -> ApprovalRequest:
        """The request to the policy for the call *request* would run."""
        return ApprovalRequest(
            request_id=uuid.uuid4().hex,
            run_id=self._run.task_id,
            call_id=self._call.id,
            tool_name=request.tool_name,
            arguments=copy.deepcopy(request.arguments),
            preview=call_preview(request.tool_name, request.arguments),
            workspace_root=self._workspace_root,
            tier=self._tier,
            review=review,
            outside=outside,
            person_cjk=None if self._reviewer is None else self._reviewer.person_cjk(),
        )

    async def _ask(
        self,
        decide: Callable[[ApprovalRequest], Awaitable[object]],
        asking: ApprovalRequest,
    ) -> ApprovalDecision:
        """:func:`ask` with the call named in ``run.waiting_call_id`` meanwhile."""
        run = self._run
        run.waiting_call_id = self._call.id
        decided = await ask(
            decide, asking, record=functools.partial(self._failed, asking)
        )
        run.waiting_call_id = ""
        return decided

    def _failed(
        self, asking: ApprovalRequest, origin: str, error: BaseException | None
    ) -> None:
        """Keep a question that got no decision (see :func:`ask`)."""
        if error is not None:
            # A withdrawn question keeps the name: the call did not start.
            self._run.waiting_call_id = ""
        reason = "" if error is None else exception_text(error)
        self._record(asking, origin=origin, reason=reason)

    def _counted(self, tool_name: str, preview: str) -> None:
        """Count the call among those that a later model review is told came before it."""
        self._run.reviews.calls.append(
            (
                tool_name,
                preview,
                "routine" if self.review is None else self.review.verdict,
            )
        )

    def _block(
        self, asking: ApprovalRequest, auto: AutoPolicy | None, words: str = ""
    ) -> str:
        """Keep an auto-mode block and return the note the model reads."""
        task = auto is not None and auto.task
        changed = auto is not None and auto.has_allowances(asking.tool_name)
        if self._reviewer is None:
            return block_note(self.review, words=words, task=task, changed=changed)
        return self._reviewer.blocked(
            self._run.reviews,
            review=self.review,
            key=self._key,
            call_id=self._call.id,
            preview=asking.preview,
            words=words or block_words(self.review),
            task=task,
            workspace=self._workspace_root,
            changed=changed,
        )

    def _record(
        self,
        asking: ApprovalRequest,
        *,
        origin: str,
        allowed: bool = False,
        reason: str = "",
        remember: bool = False,
        denial: bool = False,
    ) -> None:
        """Keep the outcome of *asking*.

        *origin* is the decision's, ``"error"`` when the policy failed (the
        reason is then the failure), or ``"withdrawn"`` when the run was
        cancelled while the policy decided. In auto mode, and for a call
        that is not plainly routine, the record also holds the ``mode``, the
        ``review`` and the ``step_key``; an allowance used is ``"once"``.
        *denial*: the step itself was denied (a person's Deny, an auto block,
        a policy's refusal of the step); the record then keeps its
        ``call_key`` and ``loose_keys``, so a task its chat continues does not
        run it, or a step alike, again (#411).
        """
        self.asked = {
            "request_id": asking.request_id,
            "allowed": allowed,
            "reason": reason,
            "remember": remember,
            "origin": origin,
        }
        review = self.review
        if self._mode == "auto":
            self.asked["mode"] = "auto"
        if review is not None and (
            self._mode == "auto" or review.verdict != "routine" or review.by == "model"
        ):
            shown = review.to_json()
            # The sentence the approval card starts with (#454).
            gist = step_gist(
                asking.tool_name, asking.arguments, review, cjk=asking.person_cjk
            )
            if gist is not None:
                shown["gist"] = gist.to_json()
            self.asked["review"] = shown
            if self._key:
                self.asked["step_key"] = self._key
        if self._allowance:
            self.asked["allowance"] = "once"
        if denial and not allowed:
            # What makes a later call the same step, or one alike: a task the
            # chat continues on its own never runs it again (#411).
            self.asked["call_key"] = call_key(
                asking.tool_name, asking.arguments, self._workspace_root
            )
            self.asked["loose_keys"] = loose_keys(
                asking.tool_name,
                asking.arguments,
                () if review is None else review.categories,
            )


def _fresh_review(review: ActionReview | None) -> ActionReview:
    """*review* of a call whose effect is unknown, as only a person settles it (#571).

    High impact, with ``opaque`` (code melo cannot check, which words and the
    model never allow, and Auto asks about on its card) beside what the rules
    found, and why first.
    """
    if review is None:
        return ActionReview("high", ("opaque",), (UNKNOWN_RESULT,))
    categories = review.categories
    return dataclasses.replace(
        review,
        verdict="high",
        categories=categories if "opaque" in categories else (*categories, "opaque"),
        reasons=(UNKNOWN_RESULT, *review.reasons),
    )


# Why a step is reviewed that the rules found routine: it reaches a host melo
# cannot read while a site is refused in the task (#433).
UNREAD_SITE: Final = (
    "Reaches a host melo cannot read, while a site is refused in this task."
)

# Why a path outside the workspace was refused when no person could be
# asked after all (the policy changed while the call was checked).
NO_FOLDER_ASKER: Final = "no one can allow a folder outside the workspace in this run"


def _outside_write(review: ActionReview | None, outside: OutsideAccess) -> ActionReview:
    """*review* of a write outside the workspace: high-impact, ``outside_write`` included.

    The rules leave the category out of a file tool's review, since the
    workspace fence used to place every write; here it is outside.
    """
    found = review or ActionReview("routine")
    categories = [*found.categories]
    if "outside_write" not in categories:
        categories.append("outside_write")
    order = list(TITLES)
    categories.sort(key=lambda name: order.index(name) if name in order else len(order))
    reasons = (
        *found.reasons,
        f"Writes {home_label(outside.path)}, outside the workspace.",
    )
    return dataclasses.replace(
        found, verdict="high", categories=tuple(categories), reasons=reasons[:3]
    )


def _only_outside_write(review: ActionReview) -> bool:
    """Whether *review* of a write outside the workspace finds nothing but that: no other category, no doubt."""
    return review.categories == ("outside_write",) and not review.doubts


def _folder_scope(decision: ApprovalDecision) -> str:
    """What allowed a call outside the workspace: once, chat, always, named (the person wrote the folder), or "" (denied)."""
    if not decision.allow:
        return ""
    if decision.reason == NAMED_FOLDER:
        return "named"
    if decision.origin == "session":
        return "chat"
    if decision.origin == "policy":
        return "always"
    if decision.always:
        return "always"
    return "chat" if decision.remember else "once"


# The words of auto mode's review of a site question, by what the tool wants to
# do there: open it (no ``access``), click or choose on a page, click at a
# point, type on it, or send a form from it. The step is judged on the goal of
# the chat (#465), so nothing here says the request did not name it.
_SITE_DOUBT = {
    "": "Opens {site}.",
    "read": "Opens {site}.",
    "act": "Clicks or chooses on {site}.",
    "point": "Clicks at a point on {site}.",
    "submit": "Sends a form from {site}, which needs its own review.",
    "dialog": "Answers OK to a question the page on {site} asks.",
}
# A page asked to open another app (an address of the app's kind): only a
# person's answer lets melo open it, in auto mode too.
_HANDOFF = "Opens {app}, an app on this Mac, because a page on {site} asked for it."
# The questions only the person answers, in Ask and in Auto alike: no model sees them, so no
# review decides them. In Auto each is stopped for a card with these words (nobody else sees
# them): a window another app put up, a press in one, a page that needs the person, and a
# press inside a human check (#583).
_PERSON_WORDS = {
    WINDOW_READ: "Another app put up a window right after a step; only you decide whether melo sees it.",
    DESKTOP_ACT: "Pressing a button in another app's window is yours to decide, every time.",
    TAKEOVER: "melo needs you on this page (a sign-in or a check only you can do); only you can take it over.",
    HUMAN_CHECK: "A human check on this page: only you decide whether melo presses it.",
}
_TYPING_DOUBT = (
    "Types text on {site}, which needs its own review: the text is the model's own."
)
# A click on a site the person's words name: still reviewed, by what it is on.
_NAMED_SITE = " The user's words name this site."
# What the text holds that code found of the person (#547).
_PERSONAL = " The text holds the user's {kinds}."
# A click whose target melo cannot tell (#465): the person is asked.
_UNKNOWN_TARGET = "Clicks on {site} where melo cannot tell what it clicks."
# The site questions that carry no value of melo's (#465): a click or a choice
# on a page, a click at a point, an OK to a box. Only these can turn a model's
# ``credentials`` into no category; a read (its address can hold a key), a
# search, typing, a form sent, or any access not listed never does.
_VALUE_FREE_ACCESS: Final = frozenset({"act", "point", "dialog"})
# The site questions that hold the whole step: the text typed, the form sent and
# where it goes. Each is reviewed by itself, and its verdict is never kept for
# the next; only such a question shows a person all it does, so only such a one
# may be allowed once after Auto blocked it.
_WHOLE_ACCESS: Final = frozenset({"type", "submit"})


def whole_step(arguments: Mapping[str, Any]) -> bool:
    """Whether a site question with *arguments* holds all its step does (:data:`_WHOLE_ACCESS`)."""
    return arguments.get("access") in _WHOLE_ACCESS


def _personal(
    reviewer: StepReviewer, tool_name: str, arguments: Mapping[str, Any]
) -> str:
    """What of the person's own the text a step sends holds, in words (``phone number and address``), or ``""``."""
    found = personal_data(
        "\n".join(sent_texts(tool_name, arguments)), reviewer.person_names
    )
    return " and ".join(found)


def _site_doubt(access: str, site: str, preview: str) -> str:
    """What a running tool's question does, for its review.

    A question of a kind this table does not know (another tool's) is
    described by its preview.
    """
    if access == "type":
        return _TYPING_DOUBT.format(site=site)
    if access in _SITE_DOUBT:
        return _SITE_DOUBT[access].format(site=site)
    said = " ".join(preview.split())[:200].rstrip(".")
    return (
        f"{said[:1].upper()}{said[1:]}." if said else _SITE_DOUBT[""].format(site=site)
    )


class _ToolAsk:
    """What a running tool uses to ask the run's approval policy (``TOOL_ASK``).

    The request is for a name and arguments the tool chooses, not for the
    tool's own call, so a policy that keeps grants by name keeps one per name.
    It has the call's id, so a surface shows the question with the call. The
    run's cancellation propagates, and the prompt counts as withdrawn. The
    policy fails closed as it does for a call (:func:`ask`), and a question that
    got no decision is kept in *refused* with its ``origin`` (``withdrawn`` or
    ``error``).

    In auto mode (#339) a site the browser would open is reviewed first:
    routine when the user's words name it, otherwise the user's model asks
    whether it serves the request (cached per site for the chat), and a
    block is kept as the gate keeps one.

    A question refused is a blocked step like any other (#431): it is kept
    in *refused* (the call's ``tool_asks``) with its ``call_key`` and its
    ``loose_keys`` (:func:`question_key` and, for a site, :func:`site_key`
    among them), and a run its chat continued on its own refuses the same
    question, one under the same name, or one about the same host however
    its address is spelt, before any review or prompt (:class:`EarlierBlocks`).
    A refusal covers the whole site (:func:`site_key`) and beats everything but
    the person's own newer answer (#577): the person's refusal takes back what
    the chat keeps for the site (:func:`revoke_site`), and whatever else would
    run a question on a site refused in this task (a kept answer that holds
    from elsewhere, Auto, a page approved before) is refused
    (:data:`SITE_REFUSED`). The person's fresh Allow lifts it for what they
    allowed: that question, or all of the site on a kept answer.
    """

    def __init__(
        self,
        policy: ApprovalPolicy,
        run: _RunState,
        call: ToolCall,
        workspace_root: str | None,
        reviewer: StepReviewer | None = None,
        refused: list[dict[str, Any]] | None = None,
        sees_images: Callable[[], bool] = lambda: False,
    ) -> None:
        self._policy = policy
        self._run = run
        self._call = call
        self._workspace_root = workspace_root
        self._reviewer = reviewer
        self._sees_images = sees_images
        self.refused: list[dict[str, Any]] = [] if refused is None else refused

    def names_host(self, host: str) -> bool:
        """Whether the user's recent messages name *host* itself (``www.`` aside)."""
        return self._reviewer is not None and self._reviewer.names_host(host)

    def sign_in_value(self, handle: str, site: str) -> str | None:
        """The password *handle* stands for in a password box on *site*, or None (#547).

        Only for a password the person gave in their newest message, on a site
        their own words name; see :meth:`StepReviewer.sign_in_value`.
        """
        reviewer = self._reviewer
        return None if reviewer is None else reviewer.sign_in_value(handle, site)

    def from_person(self) -> bool:
        """Whether the run's words are the person's: their own message or a chore they saved, not a follow-up or a task's own (#551, #570).

        The browser works through a picture check itself only in such a run.
        """
        return self._run.from_person

    def sees_pictures(self) -> bool:
        """Whether the model reads the pictures tools return (#551): the run's own answer, never a page's."""
        return self._sees_images()

    def attachable(self, path: str) -> str | None:
        """Who vouches for the file *path* as one to attach to a page: ``folder`` or ``named``; None when nobody does (#551).

        ``folder``: it lies in the chat's folder (the workspace). ``named``:
        the person's own words name this file, by its path or its specific name
        (see :meth:`StepReviewer.names_file`); a folder they named vouches for
        nothing in it. A place never offered, or a file that looks like a key,
        is neither, however the person worded it. The chat's work folder is the
        browser's own to vouch for; a page's text, a tool's result and the
        model's own words name nothing. A file nobody vouches for is asked
        about, by name, whoever suggested it.
        """
        real = canonical_path(path)
        if attach_refusal(real) is not None:
            return None
        root = self._workspace_root
        if root and covers(canonical_path(root), real):
            return "folder"
        reviewer = self._reviewer
        if reviewer is not None and reviewer.names_file(real):
            return "named"
        return None

    def asks_to_share(self) -> bool | None:
        """Whether the person's own words speak of sharing or making something public; None when nothing reads them (#547)."""
        return None if self._reviewer is None else self._reviewer.asks_to_share()

    def written_sites(self) -> frozenset[tuple[str, int]]:
        """The servers on this computer the person's own words write, as ``(address, port)``; none when nothing reads them or the run is over (#547).

        The browser keeps this reader for the pages it lets reach those
        servers, so the end of the run ends what they may reach.
        """
        if self._reviewer is None or not self._run.running:
            return frozenset()
        return self._reviewer.written_sites()

    def signs_in(self, site: str) -> bool:
        """Whether melo has sign-in details of the person's to try on *site*."""
        return self._reviewer is not None and self._reviewer.signs_in(site)

    def holds_sign_in(self) -> bool:
        """Whether a password the person gave in this run can be typed in it (#571):
        work that moves elsewhere would hold it spent."""
        return self._reviewer is not None and self._reviewer.holds_sign_in()

    def granted(self, name: str) -> bool:
        """Whether the run's chat holds a grant of the question *name* (``window-read:<app>``)."""
        return holds_grant(self._policy, name)

    def refused_earlier(
        self, tool_name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision | None:
        """The refusal of a question blocked earlier in this task, or one alike, or None.

        Asked before anything lets the question through without asking, such
        as a site the user's words name: a block stands over that (#431).
        """
        return self._earlier(self._request(tool_name, arguments, preview))

    def refused_unasked(
        self, tool_name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision | None:
        """:meth:`refused_earlier`, or a refusal of a site in this run: for what passes with no question (#577).

        Asked before a page approved earlier is read again and before a site
        the user's words name is read. Where a question comes next, the person
        decides instead.
        """
        request = self._request(tool_name, arguments, preview)
        return self._earlier(request) or self._site_refusal(request)

    def _request(
        self, tool_name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=uuid.uuid4().hex,
            run_id=self._run.task_id,
            call_id=self._call.id,
            tool_name=tool_name,
            arguments=copy.deepcopy(dict(arguments)),
            preview=preview,
            workspace_root=self._workspace_root,
            tier="network",
        )

    async def _ask(
        self,
        decide: Callable[[ApprovalRequest], Awaitable[object]],
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        """:func:`ask`; a question that got no decision is kept in *refused*."""
        return await ask(
            decide, request, record=functools.partial(self._failed, request)
        )

    def _failed(
        self, request: ApprovalRequest, origin: str, error: BaseException | None
    ) -> None:
        """Keep a question that got no decision (see :func:`ask`).

        Without its preview, which can name a window of another app.
        """
        self.refused.append(
            {"request_id": request.request_id, "tool": request.tool_name,
             "allowed": False, "origin": origin,
             "reason": "" if error is None else exception_text(error)}
        )  # fmt: skip

    async def _window(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        preview: str,
        words: str,
    ) -> ApprovalDecision:
        """A question only the person answers: whether melo may see another app's window, or press a button in one.

        Nothing about the window (its app, its title) or the press goes to any
        model: not to the review model, which never sees this question, and not
        into a blocked step's record or a later review's prompt. In Auto the policy
        is handed a review that only a card can settle (the one category
        ``opaque``, which words and the model never allow), so it asks the
        person like a card-only kind, or denies when nobody is watching (the reason
        is :data:`NOBODY_TO_ASK`, not the category's words). An
        answer of "Allow <App> in this chat" is the policy's grant of this
        name (``window-read:<bundle id>``). A press (``desktop-act:<app>``) is
        asked every time and never kept.

        The card that asks the person to take over a page of melo's browser
        (``takeover:<site>``) goes the same way: only the person can sign in or
        pass a check, so no review decides it, and so does a press inside a
        human check (``browser-check:<site>``, #583), whose "Always in this
        chat" the policy keeps. *words* are the card's own, per kind
        (:data:`_PERSON_WORDS`).
        """
        request = self._request(tool_name, arguments, preview)
        if review_mode(self._policy, request) == "auto":
            if self._run.reviews.stop:
                return ApprovalDecision(False, STOPPED_NOTE, origin="auto")
            request = dataclasses.replace(
                request,
                review=ActionReview("high", ("opaque",), (words,)),
            )
        decided = await self._ask(self._policy.decide, request)
        if not decided.allow and decided.origin == "auto":
            # Auto refused it only because nobody can be asked: say that, not the
            # category of code melo cannot check that the fixed review carries. A
            # task that asks nobody says so itself (TASK_ASKS_NOBODY, #583).
            told = decided.reason if decided.reason == TASK_ASKS_NOBODY else None
            return dataclasses.replace(decided, reason=told or NOBODY_TO_ASK)
        self._ended_by_words(request, decided)
        return decided

    def _ended_by_words(
        self, request: ApprovalRequest, decided: ApprovalDecision, preview: str = ""
    ) -> None:
        """Record a question the person's message ended: for the call's ``tool_asks``, so the step shows it.

        Nobody's denial, so no call key and no block: the same step may be
        asked again. A question of a window or a page has no *preview* (it can
        name another app's window).
        """
        if decided.allow or decided.origin != "words":
            return
        self.refused.append(
            {"request_id": request.request_id, "tool": request.tool_name,
             "allowed": False, "origin": "words", "reason": decided.reason,
             **({"preview": preview} if preview else {})}
        )  # fmt: skip

    async def __call__(
        self, tool_name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        for prefix, words in _PERSON_WORDS.items():
            if tool_name.startswith(prefix):
                return await self._window(tool_name, arguments, preview, words)
        request = self._request(tool_name, arguments, preview)
        key = allowance_key(request)
        held = find_auto(self._policy)
        # The person allowed this exact question once: a block alike to it (another
        # text refused on the same site) does not hold it back; nothing else lifts.
        allowed = held is not None and held.allowed_once(key)
        earlier = None if allowed else self._earlier(request)
        if earlier is not None:
            return earlier
        reviewer = self._reviewer
        auto = review_mode(self._policy, request) == "auto" and reviewer is not None
        if auto and reviewer is not None and self._run.reviews.stop:
            # The turn already stopped: no site opens either.
            return ApprovalDecision(False, STOPPED_NOTE, origin="auto")
        declined = _declined_keys(tool_name, request.arguments, self._workspace_root)
        again = self._run.reviews.declined_again(declined) if auto else None
        if again is not None:
            # The same question was declined on the card earlier in this turn
            # (the person's Don't allow, or no answer): nobody is asked again
            # (#465). A repeat is no new refusal: it keeps no call key.
            self.refused.append(
                {"request_id": request.request_id, "tool": tool_name,
                 "allowed": False, "origin": again.origin,
                 "reason": again.reason, "declined": True, "preview": preview,
                 **again.record()}
            )  # fmt: skip
            return again

        async def settle(asked: ApprovalRequest) -> ApprovalDecision:
            # The review is asked in the same guard as the policy, as the
            # gate does: a failure of either refuses the question alike.
            nonlocal request
            if arguments.get("access") == "attach":
                # A file sent to a site (#551): decided by code alone, a card in
                # Ask and unless the person's words ask for the send.
                request = dataclasses.replace(
                    asked,
                    review=attach_review(
                        arguments,
                        asked=auto
                        and reviewer is not None
                        and reviewer.asks_to_attach(),
                    ),
                )
                return await self._policy.decide(request)
            if auto and reviewer is not None:
                site = str(arguments.get("site") or "")
                access = str(arguments.get("access") or "")
                # Each step is judged on the goal of the chat (#465). A click or a
                # choice names what it is on, so each target is reviewed by itself
                # (once a chat); typed text is the model's own data and a form sent
                # from a page leaves the computer: each is reviewed by itself, even
                # on a site you named, and the verdict of one is never kept for the
                # next.
                whole = whole_step(arguments)
                typing = access == "type"
                reading = access in ("", "read")
                if arguments.get("target") == "unknown":
                    # melo cannot tell what the click lands on: nobody can judge it
                    # on the goal, so the person is asked (#465).
                    review = ActionReview(
                        "high", ("opaque",), (_UNKNOWN_TARGET.format(site=site),)
                    )
                elif access == "handoff":
                    # Another app comes to the front: the person's choice, always.
                    app = " ".join(str(arguments.get("app") or "an app").split())[:80]
                    review = ActionReview(
                        "high", ("open_app",), (_HANDOFF.format(app=app, site=site),)
                    )
                elif reading and reviewer.names_host(site):
                    review = ActionReview(
                        "routine", (), (f"Opens {site}, which you named.",)
                    )
                elif (
                    typing
                    and reviewer.names_host(site)
                    and not holds_secret(arguments, preview)
                    and (
                        reviewer.searching(arguments)
                        or not is_message([str(arguments.get("text") or "")])
                    )
                    and not _personal(reviewer, tool_name, arguments)
                ):
                    # The rules judge it: a field's worth of text (not a message
                    # by the one reading the posting check has, or a search the
                    # person asked for, of any length) with no secret in it and
                    # nothing of the person's own, typed on a site the person
                    # wrote themselves. Nothing leaves the page until a form is
                    # sent, and that is reviewed on its own, so a long form does
                    # not use up the turn's reviews (#547). A message is a post:
                    # it is reviewed whole, with all its words, before any of it
                    # is typed.
                    review = ActionReview(
                        "routine", (), (f"Types on {site}, which you named.",)
                    )
                else:
                    doubt = _site_doubt(access, site, preview)
                    if site and not reading and reviewer.names_host(site):
                        doubt += _NAMED_SITE
                    if kinds := _personal(reviewer, tool_name, arguments):
                        doubt += _PERSONAL.format(kinds=kinds)
                    undecided = ActionReview("undecided", (), (doubt,))
                    # The person allowed this exact question once (the text, the
                    # site, the box): no model reviews it again, as the gate does
                    # for a call allowed once; the policy runs it on the allowance.
                    review = (
                        undecided
                        if allowed
                        else await reviewer.review(
                            self._run.reviews,
                            tool_name=tool_name,
                            arguments=dict(request.arguments),
                            preview=preview,
                            review=undecided,
                            mode="auto",
                            key=f"{key}|{self._call.id}" if whole else key,
                            workspace=self._workspace_root,
                            # One routine verdict reads the site's plain pages.
                            shared=(
                                ""
                                if whole
                                else plain_page_key(
                                    tool_name, request.arguments, self._workspace_root
                                )
                            ),
                            refused_sites=_refused_sites(self._policy, self._run),
                            # A click, a choice or a point carries no value: a click on
                            # a sign-in button is no credential use (#465). A read's
                            # address can hold a key, so its review stands.
                            secret_free=access in _VALUE_FREE_ACCESS,
                        )
                    )
                request = dataclasses.replace(asked, review=review)
            return await self._policy.decide(request)

        decided = await self._ask(settle, request)
        site = question_site_key(tool_name)
        if decided.allow and site is not None:
            if decided.origin == "user" and not isinstance(decided, Allowance):
                # The person's own answer here; an Allow once given for one exact
                # step lifts nothing else on the site.
                self._run.reviews.allowed_sites.setdefault(site, []).append(
                    (tool_name, decided.remember)
                )
            elif (refusal := self._site_refusal(request)) is not None:
                # Only the person's newer word stands over a refusal (#577). What
                # Auto kept on the way to this answer goes with it.
                await self._take_back(site, "auto")
                return refusal
        if isinstance(decided, Allowance):
            # The question runs on the person's Allow once: the record says so,
            # and the step is shown as allowed and used.
            self.refused.append(
                {"request_id": request.request_id, "tool": tool_name,
                 "allowed": True, "origin": "user", "allowance": "once",
                 "step_key": allowance_key(request)}
            )  # fmt: skip
        self._ended_by_words(request, decided, preview)
        if is_refusal(decided):
            # A task its chat continues on its own never asks it again.
            decided = await self._keep(request, decided)
        if isinstance(decided, Chosen) and not decided.allow:
            # Don't allow on the card, or no answer, holds for the rest of the
            # turn (#465).
            self._run.reviews.decline(declined, decided)
            if decided.origin == "timeout":
                # Nobody's denial of the step (no call key), but the call's
                # tool_asks say why it did not run, for the surfaces.
                self.refused.append(
                    {"request_id": request.request_id, "tool": tool_name,
                     "allowed": False, "origin": "timeout",
                     "reason": decided.reason, "preview": preview,
                     **decided.record()}
                )  # fmt: skip
        if (
            auto
            and reviewer is not None
            and not decided.allow
            and decided.origin == "auto"
        ):
            policy = find_auto(self._policy)
            note = reviewer.blocked(
                self._run.reviews,
                review=request.review,
                key=key,
                call_id=self._call.id,
                preview=preview,
                words=decided.reason,
                task=policy is not None and policy.task,
                workspace=self._workspace_root,
                changed=policy is not None
                and policy.has_allowances(self._call.tool_name),
                allowable=bool(self.refused and self.refused[-1].get("step_key")),
            )
            return dataclasses.replace(decided, reason=note)
        return decided

    def _earlier(self, request: ApprovalRequest) -> ApprovalDecision | None:
        """A refusal of a question refused earlier in this task, or one alike, before any review."""
        blocks = find_earlier(self._policy)
        name = request.tool_name
        found = (
            None
            if blocks is None
            else blocks.match(
                name,
                request.arguments,
                self._workspace_root,
                # An address nested in its path (a reader, an archive) too:
                # only to refuse, never kept with a refusal.
                also=(
                    *_question_keys(name, request.arguments),
                    *address_keys(str(request.arguments.get("url") or "")),
                ),
            )
        )
        if found is None:
            return None
        who, exact = found
        reason = (BLOCKED_EARLIER if exact else BLOCKED_ALIKE).format(who=who)
        # A refusal of a repeat is no new block: it keeps no call key.
        self.refused.append(
            {"request_id": request.request_id, "tool": name, "allowed": False,
             "origin": "policy", "reason": reason, "earlier": who,
             "preview": request.preview}
        )  # fmt: skip
        return ApprovalDecision(False, reason, origin="policy")

    def _site_refusal(self, request: ApprovalRequest) -> ApprovalDecision | None:
        """The refusal of *request*, on a site refused in this task and not allowed since by the person, or None (#577)."""
        name = request.tool_name
        site = question_site_key(name)
        if site is None or site[len(SITE_KEY) :] not in _refused_sites(
            self._policy, self._run
        ):
            return None
        allowed = self._run.reviews.allowed_sites.get(site, ())
        if any(
            given == name or (kept and site_covers(given, name))
            for given, kept in allowed
        ):
            return None
        self.refused.append(
            {"request_id": request.request_id, "tool": name, "allowed": False,
             "origin": "policy", "reason": SITE_REFUSED,
             "preview": request.preview}
        )  # fmt: skip
        return ApprovalDecision(False, SITE_REFUSED, origin="policy")

    async def _keep(
        self, request: ApprovalRequest, decided: ApprovalDecision
    ) -> ApprovalDecision:
        """Keep a refused question as a blocked step: its call key and keys alike.

        A refused site joins the run's refused sites, which a later review
        is told of (#433), and the person's refusal takes back what the chat
        keeps for it (#577). All of that stands before the take-back is
        written; the decision comes back with a notice if the file refused it.
        """
        name, arguments, review = request.tool_name, request.arguments, request.review
        categories = () if review is None else review.categories
        site = site_key(arguments)
        reviews = self._run.reviews
        # A site refused earlier in this task (before this refusal, which may
        # refuse its own site) keeps its steps from an Allow once.
        on_refused = (question_site_key(name) or "")[len(SITE_KEY) :] in (
            _refused_sites(self._policy, self._run)
        )
        if site is not None:
            if site[len(SITE_KEY) :] not in reviews.refused_sites:
                reviews.refused_sites.append(site[len(SITE_KEY) :])
            reviews.allowed_sites.pop(site, None)  # the newest word stands
        self.refused.append(
            {"request_id": request.request_id, "tool": name, "allowed": False,
             "origin": decided.origin, "reason": decided.reason,
             "preview": request.preview,
             "call_key": call_key(name, arguments, self._workspace_root),
             "loose_keys": [*loose_keys(name, arguments, categories),
                            *_question_keys(name, arguments)]}
        )  # fmt: skip
        if decided.origin == "auto" and review is not None:
            # Auto's block, kept as the gate keeps one, with what the review found.
            # Allow once may run it only when the question holds the whole step
            # (a text typed, a form sent: the card shows all of it), a chat's card
            # would have asked about it, its site was not refused and it holds no
            # secret: then the key and the question whole, for the person to see.
            # Any other keeps no text.
            self.refused[-1].update(mode="auto", review=review.to_json())
            if (
                whole_step(arguments)
                and asks_in_auto(review)
                and not on_refused
                # A text that holds a password or a key by melo's own detectors
                # is kept nowhere, whatever the review filed it under.
                and not holds_secret(arguments, request.preview)
            ):
                self.refused[-1].update(
                    step_key=allowance_key(request), arguments=copy.deepcopy(arguments)
                )
        # An automatic block leaves the person's word: it can come from nobody
        # being there to ask.
        if site is None or decided.origin != "user":
            return decided
        if await self._take_back(site):
            return decided
        return dataclasses.replace(decided, notice=REFUSAL_NOT_SAVED)

    async def _take_back(self, site: str, by: str | None = None) -> bool:
        """:func:`revoke_site` to its end, whatever stops the run: Stop ends only the wait (#577)."""
        task = asyncio.ensure_future(revoke_site(self._policy, site, by))
        _TAKE_BACKS.add(task)
        task.add_done_callback(_TAKE_BACKS.discard)
        return await asyncio.shield(task)


# Take-backs under way: a run stopped meanwhile holds them no more (#577).
_TAKE_BACKS: set[asyncio.Future[bool]] = set()


def _refused_sites(policy: ApprovalPolicy, run: _RunState) -> tuple[str, ...]:
    """The hosts of the sites refused in this task (#433): earlier runs' first, then this run's."""
    blocks = find_earlier(policy)
    found = dict.fromkeys(() if blocks is None else blocks.refused_sites())
    found.update(dict.fromkeys(run.reviews.refused_sites))
    return tuple(found)


def _question_keys(name: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """The loose keys of a running tool's question: its name, and its site's host."""
    site = site_key(arguments)
    return (question_key(name),) if site is None else (question_key(name), site)


def _declined_keys(
    name: str, arguments: Mapping[str, Any], workspace_root: str | None
) -> tuple[str, ...]:
    """The keys a Don't allow of a running tool's question holds for the turn (#465).

    A click or a choice that names what it is on holds for that exact
    question: Don't allow on "Pay" does not refuse the check-in button. Any
    other (typing, a form sent, a click melo cannot tell) holds for every
    question under its name and site.
    """
    if (
        arguments.get("on")
        and arguments.get("target") != "unknown"
        and arguments.get("access") not in ("type", "submit")
    ):
        return (call_key(name, arguments, workspace_root),)
    return _question_keys(name, arguments)
