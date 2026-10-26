"""The simulated model server: slots, processor sharing with Θ(n), and a prefix cache.

The simulator needs a server whose speed depends on how many calls share
it (design 8.1): at most ``n_srv`` calls run, more wait in the server's
own first-come queue (as oMLX does, design 1.2), and the running calls
share Θ(n) = n^α of throughput evenly. Prefill and decode share that rate
until a later probe models prefill apart (spec Requirement 46; the known
gap is in docs/scheduler.md). Completions are events tagged with a
generation, so a rate change makes earlier end events stale instead of
moving them.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from localmelo.support.scheduler.costs import Rates, alone_service_s


@dataclass(frozen=True)
class ServerParams:
    """The server model's settings; ``None`` takes the value from ``Params``.

    ``theta_alpha`` is 0.4, fitted to probe P-1's Θ(n)/Θ(1) on oMLX
    (1.5-2.2 for n = 2-8); 0.5 and 1.0 are the sensitivity arms.
    ``foreign_calls`` permanently hold that many slots and count in Θ:
    clients the gate cannot see (design 9).
    """

    n_srv: int | None = None
    rates: Rates | None = None
    theta_alpha: float = 0.4
    cache: Literal["paged_lru", "single_prefix"] = "paged_lru"
    cache_tasks: int = 3
    shared_prefix_tokens: int = 0
    foreign_calls: int = 0

    def __post_init__(self) -> None:
        if self.n_srv is not None and self.n_srv < 1:
            raise ValueError("n_srv: must be at least 1")
        if not (math.isfinite(self.theta_alpha) and self.theta_alpha >= 0):
            raise ValueError("theta_alpha: must be finite and at least 0")
        if self.cache not in ("paged_lru", "single_prefix"):
            raise ValueError(f"cache: {self.cache!r} is not a cache model")
        for name in ("cache_tasks", "shared_prefix_tokens", "foreign_calls"):
            if getattr(self, name) < (1 if name == "cache_tasks" else 0):
                raise ValueError(f"{name}: out of range")


@dataclass
class Job:
    """A call on the server; ``key`` names its prefix (a task id, or the front).

    ``w`` is its remaining work in seconds alone; ``h`` the cached tokens
    it found at its start.
    """

    call_id: str
    key: str
    prompt: int
    warm: int
    completion: int
    start_us: int = -1
    h: int = 0
    w: float = 0.0
    gen: int = 0


# An end event to push: (time, call id, generation).
End = tuple[int, str, int]


class Server:
    """Running calls, the server's queue and the cache, advanced by events."""

    def __init__(self, n_srv: int, rates: Rates, sp: ServerParams) -> None:
        self.n_srv, self.rates, self.sp = n_srv, rates, sp
        self.theta = tuple(float(n) ** sp.theta_alpha for n in range(n_srv + 1))
        self.running: dict[str, Job] = {}
        self.queue: deque[Job] = deque()
        self.cache: list[str] = []  # most recently ended first
        self.last_us = 0
        self.busy_us = 0

    def _n(self) -> int:
        return len(self.running) + self.sp.foreign_calls

    def progress(self, now: int) -> None:
        """Every running call's work up to *now* at the current rate."""
        n, dt = self._n(), now - self.last_us
        if self.running and dt > 0:
            self.busy_us += dt
            for j in self.running.values():  # each call alone: order is immaterial
                j.w = max(0.0, j.w - dt * self.theta[n] / n / 1e6)
        self.last_us = max(self.last_us, now)

    def _fill(self, now: int, n_old: int) -> tuple[list[End], list[Job]]:
        """Start queued calls on free slots; new end events when n changed."""
        started: list[Job] = []
        while self.queue and self._n() < self.n_srv:
            j = self.queue.popleft()
            cached = j.key in self.cache
            j.start_us = now
            j.h = j.warm if cached else min(j.prompt, self.sp.shared_prefix_tokens)
            j.w = alone_service_s(j.prompt, j.h, j.completion, self.rates)
            self.running[j.call_id] = j
            started.append(j)
        n = self._n()
        moved = list(self.running.values()) if n != n_old else started
        ends: list[End] = []
        for j in sorted(moved, key=lambda j: (j.start_us, j.call_id)):
            j.gen += 1
            ends.append(
                (now + math.ceil(j.w * n / self.theta[n] * 1e6), j.call_id, j.gen)
            )
        return ends, started

    def submit(self, now: int, jobs: Sequence[Job]) -> tuple[list[End], list[Job]]:
        """Queue *jobs* in order and start what fits."""
        self.progress(now)
        n_old = self._n()
        self.queue.extend(jobs)
        return self._fill(now, n_old)

    def end(
        self, now: int, events: Sequence[tuple[str, int]]
    ) -> tuple[list[Job], list[End], list[Job]]:
        """The jobs that *events* end, in order; stale or repeated events are ignored.

        Every end of one instant is applied before any rate changes, so
        calls with equal ends finish together, start-first.
        """
        self.progress(now)
        n_old = self._n()
        ended = []
        for call_id, gen in events:
            j = self.running.get(call_id)
            if j is None or j.gen != gen:
                continue
            del self.running[call_id]
            self.cache = [j.key] + [k for k in self.cache if k != j.key]
            keep = self.sp.cache_tasks if self.sp.cache == "paged_lru" else 1
            del self.cache[keep:]
            ended.append(j)
        if not ended:
            return [], [], []
        return (ended, *self._fill(now, n_old))

    def evict(self, key: str) -> None:
        """Drop *key*'s prefix from the cache (a park)."""
        self.cache = [k for k in self.cache if k != key]
