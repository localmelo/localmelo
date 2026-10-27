"""The medoid's work is bounded: a run's end never blocks the event loop long.

Learning runs inside ``asyncio.timeout``, which cannot stop pure Python
CPU. The medoid compares each pair of episodes once, at most
MEDOID_MAX_SEGMENTS segments of each, and past MEDOID_MAX_CELLS compared
token pairs it is not computed: the new version keeps its parent's
skeleton, or a first version takes the latest completed episode's.
"""

from __future__ import annotations

import time
import uuid

from localmelo.melo.memory.procedures.normalize import key_for_query
from localmelo.melo.memory.procedures.store import MemoryProcedures, RunFacts
from localmelo.melo.memory.procedures.template import (
    DEFAULT_PARAMS,
    MEDOID_MAX_CELLS,
    MEDOID_MAX_SEGMENTS,
    Episode,
    build_version,
    edit_distance,
    medoid,
    medoid_cells,
)
from localmelo.melo.memory.procedures.trace import Segment

KEY = key_for_query("rebuild the index", "/work")


def segments(n: int, shift: int = 0) -> tuple[Segment, ...]:
    """*n* segments whose targets change at each one, so none merge."""
    return tuple(
        Segment(
            "RW"[k % 2],
            f"Read f{(k + shift) % 37}.py",
            f"f{(k + shift) % 37}.py",
            (1, 1),
        )
        for k in range(n)
    )


def episode(n: int, shift: int = 0, task_id: str | None = None) -> Episode:
    return Episode(
        task_id=task_id or uuid.uuid4().hex,
        routine_id="r",
        template_id="r:t",
        model="m",
        harness_version="x",
        match="exact",
        level=0,
        template_version=None,
        status="completed",
        steps=5,
        attempts=1,
        n_tool=n,
        duration_s=1.0,
        first_pass_step=None,
        cycles=0,
        segments=segments(n, shift),
        shadow_bar_error=None,
        created_at=0.0,
    )


def test_the_bounds() -> None:
    assert (MEDOID_MAX_SEGMENTS, MEDOID_MAX_CELLS) == (200, 250_000)


def test_the_work_counts_each_pair_once_and_capped_lengths() -> None:
    small = [episode(10), episode(20), episode(30)]
    assert medoid_cells(small) == 10 * 20 + 10 * 30 + 20 * 30
    big = [episode(1000), episode(5)]
    assert medoid_cells(big) == MEDOID_MAX_SEGMENTS * 5


def test_the_medoid_is_unchanged_for_small_episodes() -> None:
    eps = [episode(6), episode(7, shift=1), episode(6), episode(9, shift=3)]
    tokens = [[(s.cls, s.target) for s in e.segments] for e in eps]
    totals = [
        sum(edit_distance(tokens[i], tokens[j]) for j in range(len(eps)) if j != i)
        for i in range(len(eps))
    ]
    best = max(i for i, t in enumerate(totals) if t == min(totals))  # ties: latest
    assert medoid(eps) is eps[best]


def test_eight_long_episodes_are_compared_quickly() -> None:
    eps = [episode(1000, shift=k) for k in range(8)]
    started = time.perf_counter()
    medoid(eps)
    assert time.perf_counter() - started < 0.5


def test_past_the_budget_a_version_keeps_its_parent_s_skeleton() -> None:
    small = [episode(4, task_id="a"), episode(4, task_id="b")]
    parent = build_version(
        template_id="r:t",
        routine_id="r",
        model="m",
        options={},
        window=small,
        first_task_id="a",
        parent=None,
        params=DEFAULT_PARAMS,
        harness_version="x",
        created_at=0.0,
    )
    window = [episode(1000, shift=k) for k in range(8)]
    assert medoid_cells([e for e in window if e.completed]) > MEDOID_MAX_CELLS
    started = time.perf_counter()
    child = build_version(
        template_id="r:t",
        routine_id="r",
        model="m",
        options={},
        window=window,
        first_task_id="a",
        parent=parent,
        params=DEFAULT_PARAMS,
        harness_version="x",
        created_at=1.0,
    )
    assert time.perf_counter() - started < 0.5
    assert (child.skeleton, child.medoid) == (parent.skeleton, parent.medoid)
    assert child.stats.n == 8  # the statistics still come from the window


def test_past_the_budget_a_first_version_takes_the_latest_episode() -> None:
    window = [episode(1000, shift=k, task_id=f"t{k}") for k in range(8)]
    first = build_version(
        template_id="r:t",
        routine_id="r",
        model="m",
        options={},
        window=window,
        first_task_id="t0",
        parent=None,
        params=DEFAULT_PARAMS,
        harness_version="x",
        created_at=0.0,
    )
    assert first.medoid == "t7"


async def test_learning_long_runs_stays_far_below_the_bound() -> None:
    store = MemoryProcedures()
    took: list[float] = []
    for k in range(9):
        facts = RunFacts(
            task_id=uuid.uuid4().hex,
            key=KEY,
            model="m",
            options={},
            status="completed",
            steps=5,
            segments=segments(1000, shift=k),
        )
        started = time.perf_counter()
        await store.learn(facts)
        took.append(time.perf_counter() - started)
    assert max(took) < 0.5
