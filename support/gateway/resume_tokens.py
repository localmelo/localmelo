"""In-memory, single-use Resume confirmation tokens.

A token authorizes one accepted Resume transition. It is bound to the task,
its source execution, the evidence identity and the CAS generation that a
preview showed, so it cannot be replayed after Stop or after any transition
that changes the task's shared park identity. Nothing here is persisted: a
restart re-runs recovery and asks the user to look at the evidence again.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

TOKEN_TTL_S = 300.0


@dataclass(frozen=True)
class ResumeBinding:
    """What a confirmation token authorizes, as previewed."""

    task_id: str
    execution_id: str | None
    evidence_key: str
    generation: int | None
    pending_operation_id: str | None


@dataclass
class _Grant:
    binding: ResumeBinding
    expires_at: float


class ResumeTokens:
    """Issue, check and consume single-use grants without durable state."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_s: float = TOKEN_TTL_S,
    ) -> None:
        if not isinstance(ttl_s, int | float) or ttl_s <= 0:
            raise ValueError("token lifetime must be positive")
        self._clock = clock
        self._ttl_s = float(ttl_s)
        self._grants: dict[str, _Grant] = {}

    def issue(self, binding: ResumeBinding) -> str:
        self._prune()
        token = secrets.token_urlsafe(32)
        self._grants[token] = _Grant(binding, self._clock() + self._ttl_s)
        return token

    def valid(self, token: str, binding: ResumeBinding) -> bool:
        """Whether *token* still authorizes exactly *binding*; does not consume."""
        grant = self._grants.get(token)
        if grant is None:
            return False
        if grant.expires_at <= self._clock():
            del self._grants[token]
            return False
        return grant.binding == binding

    def binding_for(self, token: str) -> ResumeBinding | None:
        """The binding a still-pending token carries, or ``None`` when spent."""
        grant = self._grants.get(token)
        if grant is None:
            return None
        if grant.expires_at <= self._clock():
            del self._grants[token]
            return None
        return grant.binding

    def consume(self, token: str, binding: ResumeBinding) -> bool:
        """Consume *token* if it still authorizes *binding*.

        Callers check :meth:`valid` immediately before the transition; this
        final step compares the binding only so a token that expires during
        that commit cannot fail an already accepted transition.
        """
        grant = self._grants.get(token)
        if grant is None or grant.binding != binding:
            return False
        del self._grants[token]
        return True

    def invalidate(self, task_id: str) -> int:
        """Drop every pending grant for *task_id*; return how many were dropped."""
        tokens = [
            token
            for token, grant in self._grants.items()
            if grant.binding.task_id == task_id
        ]
        for token in tokens:
            del self._grants[token]
        return len(tokens)

    def pending(self, task_id: str) -> int:
        """Count unexpired grants for *task_id*; a test may assert invalidation."""
        self._prune()
        return sum(
            1 for grant in self._grants.values() if grant.binding.task_id == task_id
        )

    def _prune(self) -> None:
        now = self._clock()
        stale = [
            token for token, grant in self._grants.items() if grant.expires_at <= now
        ]
        for token in stale:
            del self._grants[token]
