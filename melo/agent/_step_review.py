"""The model review of steps, and auto mode's blocks (#279 §3.3, #339).

:class:`StepReviewer` belongs to one Agent, so one chat: it keeps the
user's latest messages, the reviews it settled (by step key) and the steps
auto mode blocked. :class:`RunReviews` is one run's share: the reviews it
sent, its calls and its blocks, and the summary ``run.finished`` carries.

When the model is asked:

- an undecided call that would run without a person (a session grant, or
  auto mode);
- in auto mode, a high-impact call whose categories the user's words can
  allow (:data:`~localmelo.melo.contracts.review.WORDS_CAN_ALLOW`), when a
  user message came after a block of the same kind or names something the
  call acts on.

It is never asked about a routine call or a ``self`` one, nor when a person
will be asked anyway. A review that times out, fails or cannot be read, the
turn's seventh, and a call too long to review all leave the call high
(fail closed).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
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
    user_message,
)
from localmelo.melo.contracts.approvals import ReviewMode, block_note
from localmelo.melo.contracts.review import (
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
# Blocks of a chat kept for the prompt and the "go ahead" check.
BLOCKS_KEPT: Final = 20

Purpose = Literal["undecided", "consent"]


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
        self, chat: Callable[[], Chat], *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        # The Agent's Chat, read at each review (a test may replace it).
        self._chat = chat
        self._clock = clock
        self._requests: deque[str] = deque(maxlen=EARLIER_REQUESTS + 1)
        self._marker = 0
        self._cache: dict[str, tuple[ActionReview, int]] = {}
        self._blocks: deque[_Block] = deque(maxlen=BLOCKS_KEPT)

    # ── The user's words ──

    def user_message(self, text: str) -> None:
        """A message of the user's own: a run's request, or one it read mid-run.

        Recalled notes and earlier chats appended to it are left out: they
        are not the user's words, and an earlier answer could argue a call.
        """
        text = own_words(text)
        if text.strip():
            self._marker += 1
            self._requests.append(text[:MAX_REQUEST_CHARS])

    @property
    def requests(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def names_host(self, host: str) -> bool:
        """Whether a user message names *host* (``www.`` aside)."""
        bare = host.lower().removeprefix("www.")
        return bool(bare) and any(bare in text.lower() for text in self._requests)

    # ── Reviews ──

    def _purpose(
        self, review: ActionReview, mode: ReviewMode, action: str
    ) -> Purpose | None:
        if mode == "ask" or review.verdict == "routine" or "self" in review.categories:
            return None
        if review.verdict == "undecided":
            return "undecided"
        if mode != "auto" or not set(review.categories) <= WORDS_CAN_ALLOW:
            return None
        after_block = any(
            block.category in review.categories and block.marker < self._marker
            for block in self._blocks
        )
        if after_block or shares_a_name(action, self._requests):
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
        """*review* settled by the model when the rules leave it open, else as it is."""
        action = action_text(tool_name, arguments)
        purpose = self._purpose(review, mode, action)
        if purpose is None:
            return review
        cached = self._cache.get(key)
        if cached is not None and (
            cached[0].verdict == "routine" or cached[1] == self._marker
        ):
            return cached[0]
        if len(action) > MAX_ACTION_CHARS:
            answer = ModelAnswer(error="the call is too long to review")
        elif run.used >= REVIEWS_PER_TURN:
            answer = ModelAnswer(error="this turn's reviews are used up")
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
            self._cache[key] = (settled, self._marker)
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


def _settle(
    review: ActionReview, answer: ModelAnswer, purpose: Purpose
) -> ActionReview:
    """The review once the model answered (or failed to)."""
    if purpose == "consent":
        if answer.user_asked == "yes" and answer.fits_request == "yes":
            # The user's words allow it: it runs, its categories kept.
            return dataclasses.replace(
                review, verdict="routine", by="model", model=answer
            )
        return dataclasses.replace(review, by="model", model=answer)
    if answer.error:
        return ActionReview(
            "high",
            ("unreviewed",),
            (f"melo could not review this call: {answer.error}.",),
            review.paths,
            by="model",
            model=answer,
        )
    if answer.verdict == "routine" and answer.fits_request == "yes":
        return dataclasses.replace(review, verdict="routine", by="model", model=answer)
    category = "opaque" if answer.category == "unclear" else answer.category
    if category not in TITLES or category == "self":
        category = "beyond_request" if answer.fits_request == "no" else "opaque"
    reason = answer.reason or (
        "Does something the request did not ask for."
        if category == "beyond_request"
        else "Does something melo could not check."
    )
    return ActionReview(
        "high", (category,), (reason,), review.paths, by="model", model=answer
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
