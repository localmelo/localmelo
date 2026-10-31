"""The model review of steps, and auto mode's blocks (#279 §3.3, #339).

:class:`StepReviewer` belongs to one Agent, so one chat: it keeps the
user's latest messages, the reviews it settled (by step key) and the steps
auto mode blocked. :class:`RunReviews` is one run's share: the reviews it
sent, its calls and its blocks, and the summary ``run.finished`` carries.

When the model is asked:

- an undecided call that would run without a person (a session grant, or
  auto mode);
- in auto mode, a high-impact call whose categories the user's words can
  allow (:data:`~localmelo.melo.contracts.review.WORDS_CAN_ALLOW`), with
  nothing the rules could not tell, when a user message came after a block
  of the same kind or names something the call acts on.

It is never asked about a routine call or a ``self`` one, nor when a person
will be asked anyway. A review that times out, fails or cannot be read, the
turn's seventh, and a call too long to review all leave the call high
(fail closed).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

from localmelo.melo.checker.review.model import (
    EARLIER_REQUESTS,
    MAX_ACTION_CHARS,
    MAX_REQUEST_CHARS,
    REVIEW_TIMEOUT_S,
    REVIEWS_PER_TURN,
    SYSTEM_PROMPT,
    ReviewInputs,
    action_text,
    own_words,
    parse_reply,
    shares_a_name,
    step_names,
    user_message,
)
from localmelo.melo.contracts.approvals import ReviewMode, block_note
from localmelo.melo.contracts.review import (
    CATEGORY_WORDS,
    MODEL_CAN_ALLOW,
    TITLES,
    WORDS_CAN_ALLOW,
    ActionReview,
    ModelAnswer,
    because,
)
from localmelo.melo.schema import Message

if TYPE_CHECKING:
    from localmelo.melo.agent._chat import Chat

# Blocks in one turn that stop it: a second of the same kind, or this many.
BLOCKS_PER_TURN: Final = 3
# Reviews of a runner (pytest…) that runs changed files melo does not read,
# a turn: apart from the turn's other reviews, so test cycles leave them.
RUNNER_REVIEWS_PER_TURN: Final = 3
# Blocks of a chat kept for the prompt and the "go ahead" check.
BLOCKS_KEPT: Final = 20

Purpose = Literal["undecided", "consent"]
_HOST: Final = re.compile(
    r"(?<![a-z0-9_.-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})"
)


def _bare_host(text: str) -> str:
    """*text* (a host or a URL) as a lower-case host, without ``www.`` or a port."""
    host = text.strip().lower()
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0].split("@")[-1].split(":", 1)[0].rstrip(".")
    return host.removeprefix("www.")


@dataclass
class _Block:
    category: str
    key: str
    marker: int  # the user message count when it was blocked
    title: str
    preview: str


@dataclass
class RunReviews:
    """One run's reviews: what ``run.finished`` reports, and the stop rule's state."""

    used: int = 0
    runner_used: int = 0
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    blocks: list[tuple[str, str]] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {"routine": 0, "high": 0, "undecided": 0, "model": 0}
    )
    blocked: list[dict[str, str]] = field(default_factory=list)
    # A second block of the same kind: the run stops and answers.
    stop: bool = False

    def count(self, review: ActionReview | None) -> None:
        if review is not None:
            self.counts[review.verdict] = self.counts.get(review.verdict, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "blocked": [dict(b) for b in self.blocked]}


def _title(review: ActionReview) -> str:
    first = review.categories[0] if review.categories else "beyond_request"
    return TITLES.get(first, "Run this?").rstrip("?").format(folder="the folder")


class StepReviewer:
    """Asks the user's model about steps the rules left open; keeps the chat's blocks."""

    def __init__(
        self,
        chat: Callable[[], Chat],
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        # The Agent's Chat, read at each review (a test may replace it).
        self._chat = chat
        self._clock = clock
        self._wall = wall
        # When the chat began: the rules count a file changed after it as
        # written in this chat. A session runner sets the stored start.
        self.since = wall()
        self.reset()

    def reset(self, *, new_chat: bool = False) -> None:
        """Forget the messages, reviews and blocks kept (a reload, an undo).

        With *new_chat*, the chat also begins now (``since``).
        """
        self._requests: deque[str] = deque(maxlen=EARLIER_REQUESTS + 1)
        self._marker = 0
        self._cache: dict[str, tuple[ActionReview, int, Purpose]] = {}
        self._blocks: deque[_Block] = deque(maxlen=BLOCKS_KEPT)
        if new_chat:
            self.since = self._wall()

    # ── The user's words ──

    def user_message(self, text: str) -> None:
        """A message of the user's own: a run's request, or one it read mid-run.

        Recalled notes and earlier chats appended to it are left out: they
        are not the user's words, and an earlier answer could argue a call.
        """
        text = own_words(text)
        # A message counts even with no words (an attachment): a consent or a
        # block of the last message does not carry over to it.
        self._marker += 1
        if text.strip():
            self._requests.append(text[:MAX_REQUEST_CHARS])

    @property
    def requests(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def names_host(self, host: str) -> bool:
        """Whether a user message names *host*, or a site *host* belongs to.

        Whole host names only (``www.`` aside): ``docs.example.com`` is
        named by ``example.com``, but ``t.co`` is not by ``microsoft.com``.
        """
        bare = _bare_host(host)
        if not bare:
            return False
        named = {
            _bare_host(found)
            for text in self._requests
            for found in _HOST.findall(text.lower())
        }
        return any(bare == name or bare.endswith("." + name) for name in named)

    # ── Reviews ──

    def _purpose(
        self, review: ActionReview, mode: ReviewMode, wanted: set[str]
    ) -> Purpose | None:
        if mode == "ask" or review.verdict == "routine" or "self" in review.categories:
            return None
        if review.verdict == "undecided":
            return "undecided"
        if (
            mode != "auto"
            or review.doubts  # words never allow what melo could not read
            or not set(review.categories) <= WORDS_CAN_ALLOW
        ):
            return None
        after_block = any(
            block.category in review.categories and block.marker < self._marker
            for block in self._blocks
        )
        if after_block or shares_a_name(wanted, self._requests):
            return "consent"
        return None

    async def review(
        self,
        run: RunReviews,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        preview: str,
        review: ActionReview,
        mode: ReviewMode,
        key: str,
        workspace: str | None,
    ) -> ActionReview:
        """*review* settled by the model when the rules leave it open, else as it is.

        A settled review is kept by step key: one that lets an undecided
        call run is used again in this chat; any other only until the
        user's next message, since new words can change the answer.
        """
        action = action_text(tool_name, arguments)
        purpose = self._purpose(review, mode, step_names(tool_name, arguments))
        if purpose is None:
            return review
        # A runner's review holds while the files it could not read stay as
        # they are.
        key = key + "|" + "|".join(review.changed) if review.changed else key
        cached = self._cache.get(key)
        if cached is not None and (
            cached[1] == self._marker
            or cached[2] == "undecided"
            and cached[0].verdict == "routine"
        ):
            return cached[0]
        runner = bool(review.changed)
        used = run.runner_used if runner else run.used
        if len(action) > MAX_ACTION_CHARS:
            answer = ModelAnswer(error="the call is too long to review")
        elif used >= (RUNNER_REVIEWS_PER_TURN if runner else REVIEWS_PER_TURN):
            answer = ModelAnswer(error="this turn's reviews are used up")
        else:
            if runner:
                run.runner_used += 1
            else:
                run.used += 1
            run.counts["model"] += 1
            inputs = ReviewInputs(
                workspace=workspace or "",
                requests=self._requests,
                tool=tool_name,
                action=action,
                findings=because(review) if review.reasons else "",
                calls=run.calls,
                blocked=[(b.title, b.preview) for b in self._blocks],
            )
            answer = await self._ask(inputs)
        settled = _settle(review, answer, purpose)
        if not answer.error:
            self._cache[key] = (settled, self._marker, purpose)
        return settled

    async def _ask(self, inputs: ReviewInputs) -> ModelAnswer:
        """One review request; any failure is an answer with an ``error``."""
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_message(inputs)),
        ]
        started = self._clock()

        def ms() -> int:
            return int((self._clock() - started) * 1000)

        try:
            async with asyncio.timeout(REVIEW_TIMEOUT_S):
                reply = await self._chat()._request(
                    messages, None, kind="review", retry=False
                )
        except TimeoutError:
            return ModelAnswer(error="the review timed out", ms=ms())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the review fails closed, the run goes on
            return ModelAnswer(
                error=f"the review failed ({type(exc).__name__})", ms=ms()
            )
        answer = parse_reply(reply.content)
        if answer is None:
            return ModelAnswer(error="its answer could not be read", ms=ms())
        return dataclasses.replace(answer, ms=ms())

    # ── Blocks ──

    def blocked(
        self,
        run: RunReviews,
        *,
        review: ActionReview | None,
        key: str,
        call_id: str,
        preview: str,
        words: str = "",
        task: bool = False,
        workspace: str | None = None,
        changed: bool = False,
    ) -> str:
        """Keep a block of auto mode and return the note the model reads.

        A second block of the same step or category in this run, or the
        :data:`BLOCKS_PER_TURN`-th, sets ``run.stop``: the note then says to
        stop and answer.
        """
        reviewed = review if review is not None else ActionReview("high")
        first = reviewed.categories[0] if reviewed.categories else "beyond_request"
        second = any(k == key or c == first for k, c in run.blocks)
        run.blocks.append((key, first))
        if second or len(run.blocks) >= BLOCKS_PER_TURN:
            run.stop = True
        title = _title(reviewed)
        self._blocks.append(_Block(first, key, self._marker, title, preview))
        run.blocked.append({"call_id": call_id, "title": title, "step_key": key})
        folder = os.path.basename(os.path.normpath(workspace)) if workspace else ""
        return block_note(
            review,
            words=words,
            task=task,
            folder=folder or "the folder",
            second=run.stop,
            changed=changed,
        )


def _model_reason(category: str) -> str:
    """melo's own sentence for what the model found: its free text is only kept."""
    if category == "beyond_request":
        return "Does something the request did not ask for."
    if category == "opaque":
        return "Does something melo could not check."
    return f"Your model's review says it {CATEGORY_WORDS[category]}."


def _settle(
    review: ActionReview, answer: ModelAnswer, purpose: Purpose
) -> ActionReview:
    """The review once the model answered (or failed to).

    A call runs only on a category of :data:`MODEL_CAN_ALLOW`: an answer
    that names a category only a click can allow keeps it blocked, and an
    undecided call the model files under a category words can allow runs
    only when the user's words ask for it.
    """
    allows = answer.category in MODEL_CAN_ALLOW
    if purpose == "consent":
        if answer.user_asked == "yes" and answer.fits_request == "yes" and allows:
            # The user's words allow it: it runs, its categories kept.
            return dataclasses.replace(
                review, verdict="routine", by="model", model=answer
            )
        return dataclasses.replace(review, by="model", model=answer)
    if answer.error:
        return dataclasses.replace(
            review,
            verdict="high",
            categories=("unreviewed",),
            reasons=(f"melo could not review this call: {answer.error}.",),
            by="model",
            model=answer,
        )
    plain = answer.category in ("none", "") or answer.user_asked == "yes"
    if (
        answer.verdict == "routine"
        and answer.fits_request == "yes"
        and allows
        and plain
    ):
        return dataclasses.replace(review, verdict="routine", by="model", model=answer)
    category = "opaque" if answer.category == "unclear" else answer.category
    if category not in TITLES or category in ("self", "none", "unreviewed"):
        category = "beyond_request" if answer.fits_request == "no" else "opaque"
    return dataclasses.replace(
        review,
        verdict="high",
        categories=(category,),
        reasons=(_model_reason(category),),
        by="model",
        model=answer,
    )


def answer_on_block(title: str) -> str:
    """The answer of a run auto mode stopped when the model wrote none."""
    what = (
        f"a step this needs ({title[:1].lower() + title[1:]})"
        if title
        else "a step this needs"
    )
    return (
        f"I stopped here: auto mode blocked {what}. You can allow it once from "
        "the chat, or tell me to go ahead in your next message."
    )
