"""The terminal UI's opt-in display seam for streamed planning replies.

A display is set for one turn in the task that runs it (a context variable,
like the agent's retry and accounting hooks), so direct runs, the gateway and
every other caller keep using the provider's complete reply. The deltas a
display receives are provisional, already redacted release output: they are
for the screen only and are never stored or replayed.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Protocol

from localmelo.melo.schema import ReasoningTags


class StreamDisplay(Protocol):
    """What a turn's live region implements for one streamed planning reply.

    ``reasoning_tags`` is the per-model framing declaration the release filter
    uses, or ``None`` for the conservative rule. ``step`` says which planning
    step the next attempts belong to (the agent calls it as the step starts,
    ahead of the stored ``step.started`` event); ``opened`` starts one attempt;
    a retry opens again. ``held`` says text arrived but none is releasable yet;
    ``content`` carries a safe delta; ``closed`` ends the attempt, once.
    """

    reasoning_tags: ReasoningTags | None

    def step(self, number: int) -> None: ...

    def opened(self) -> None: ...

    def held(self) -> None: ...

    def content(self, delta: str) -> None: ...

    def closed(self) -> None: ...


# Set for one interactive turn only, around the call that runs the agent.
STREAM_DISPLAY: ContextVar[StreamDisplay | None] = ContextVar(
    "STREAM_DISPLAY", default=None
)
