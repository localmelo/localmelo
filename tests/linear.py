"""A time bound for the scans that read a page or a file: linear, not quadratic.

``assert_linear(make, run)`` builds a text of 50 KB and one of 200 KB with *make* and
times *run* on each, alternating, the best of five each: a busy moment then slows both
sizes alike instead of one. The scan must take under half a second on the small one and
under two seconds on the large one, and 4 times as much text may take at most 8 times
as long (with a little slack for a clock that cannot see so small a time). A failed
measurement is taken once more: a machine busy with other work can still spoil one, but
a scan that is quadratic in the text takes 16 times as long, and many seconds, each time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

SMALL = 50_000
LARGE = 200_000
ROUNDS = 5


def _timed(run: Callable[[], object]) -> float:
    start = time.perf_counter()
    run()
    return time.perf_counter() - start


def measure(small: T, large: T, run: Callable[[T], Any]) -> tuple[float, float]:
    """The best time of *run* on *small* and on *large*, measured in turn."""
    fast = slow = float("inf")
    for _ in range(ROUNDS):
        fast = min(fast, _timed(lambda: run(small)))
        slow = min(slow, _timed(lambda: run(large)))
        if slow > 2.0:  # clearly too slow: no need to wait for more rounds
            break
    return fast, slow


def linear(fast: float, slow: float) -> bool:
    return fast < 0.5 and slow < 2.0 and slow <= 8 * fast + 0.05


def assert_linear(make: Callable[[int], T], run: Callable[[T], Any]) -> None:
    small, large = make(SMALL), make(LARGE)
    fast, slow = measure(small, large, run)
    if not linear(fast, slow):
        fast, slow = measure(small, large, run)
    assert linear(fast, slow), (fast, slow)
