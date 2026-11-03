"""A time bound for the scans that read a page or a file: linear, not quadratic.

``assert_linear(make, run)`` builds a text of 50 KB and one of 200 KB with *make* and
times *run* on each, the best of three. The scan must take under half a second on the
small one and under two seconds on the large one, and 4 times as much text may take at
most 8 times as long (with a little slack for a clock that cannot see so small a time).
A scan that is quadratic in the text takes 16 times as long, and many seconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

SMALL = 50_000
LARGE = 200_000


def best(run: Callable[[], object]) -> float:
    times: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
        if times[0] > 0.5:  # clearly too slow: no need to wait for two more
            break
    return min(times)


def assert_linear(make: Callable[[int], T], run: Callable[[T], Any]) -> None:
    small, large = make(SMALL), make(LARGE)
    fast = best(lambda: run(small))
    slow = best(lambda: run(large))
    assert fast < 0.5 and slow < 2.0, (fast, slow)
    assert slow <= 8 * fast + 0.05, (fast, slow)
