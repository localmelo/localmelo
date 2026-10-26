"""Workloads are drawn from seeded sha256 substreams and follow the design's mix.

Requirements 49 and 50: a workload is identical for a seed and differs
across seeds; W1 has the design's level mix, one away hour, create turns
for user tasks and a 12 h horizon; W3 has four P1 tasks with overlapping
deadlines; substream seeds come from sha256, never ``hash()``.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from localmelo.support.scheduler.model import us
from localmelo.support.scheduler.sim import SEEDS, make_workload
from localmelo.support.scheduler.sim.workloads import (
    WorkloadName,
    _substream_seed,
    synthetic_templates,
)

NAMES: tuple[WorkloadName, ...] = ("W1", "W2", "W2c", "W3", "W5")
H = 3600.0


def test_each_workload_is_identical_for_a_seed_and_differs_across_seeds() -> None:
    for name in NAMES:
        a = make_workload(name, 1)
        assert a == make_workload(name, 1)
        assert a != make_workload(name, 2)
        assert [t.task_id for t in a.tasks][:1] == [f"{name}-1-000"]
        assert all(t.level == t.gold_level and t.size == "" for t in a.tasks)
        assert a.deps == ()
    # A tool-time pool is an input: sorted before use, so its order is moot.
    pool = synthetic_templates(1, n=3, tool_pool=[1.0, 2.0, 3.0])
    assert pool == synthetic_templates(1, n=3, tool_pool=[3.0, 2.0, 1.0])


def test_w1_follows_the_design_mix_presence_and_horizon() -> None:
    levels: Counter[int] = Counter()
    for seed in SEEDS:
        wl = make_workload("W1", seed)
        assert wl.horizon_us == us(12 * H) and len(wl.tasks) == 24
        ((start, end),) = wl.presence
        assert us(2 * H) <= start <= us(5 * H) and end - start == us(H)
        creates = {at for at, kind in wl.front_messages if kind == "create"}
        assert all(not start <= at < end for at, _ in wl.front_messages)
        assert all(at < us(8 * H) for at, kind in wl.front_messages if kind == "answer")
        for t in wl.tasks:
            levels[t.gold_level] += 1
            assert (t.origin == "agent") == (t.gold_level == 4)
            assert (t.arrival_us in creates) == (t.origin == "user")
            if t.gold_level == 1:
                left = (t.deadline_wall_us or 0) - wl.wall_start_us - t.arrival_us
                assert us(H) - 1 <= left <= us(4 * H) + 1
            else:
                assert t.deadline_wall_us is None
    total = sum(levels.values())
    for level, share in ((2, 70), (1, 15), (3, 10), (4, 5)):
        assert abs(100 * levels[level] / total - share) <= 6, (level, levels)


def test_w3_has_four_p1_tasks_with_overlapping_deadlines() -> None:
    for seed in (1, 2, 3):
        wl = make_workload("W3", seed)
        assert wl.horizon_us == us(6 * H) and wl.front_messages == ()
        p1 = [t for t in wl.tasks if t.gold_level == 1]
        assert len(p1) == 4 and all(t.arrival_us <= us(600.0) for t in p1)
        slack = sorted(
            (t.deadline_wall_us or 0) - wl.wall_start_us - t.arrival_us for t in p1
        )
        assert slack == [us(60.0 * m) for m in (60, 75, 90, 105)]
        ends = sorted((t.deadline_wall_us or 0) for t in p1)
        starts = sorted(t.arrival_us + wl.wall_start_us for t in p1)
        assert starts[-1] < ends[0]  # every P1 is live before the first is due
        rest = Counter(t.gold_level for t in wl.tasks if t.gold_level != 1)
        assert rest == {2: 8, 3: 4}
        assert all(t.arrival_us <= us(1800.0) for t in wl.tasks)


def test_substream_seeds_come_from_sha256_not_hash() -> None:
    digest = hashlib.sha256(b"7:arrivals").digest()
    assert _substream_seed(7, "arrivals") == int.from_bytes(digest[:8], "big")
    seeds = {_substream_seed(7, s) for s in ("arrivals", "levels", "front", "presence")}
    assert len(seeds) == 4
    assert _substream_seed(7, "arrivals") != _substream_seed(8, "arrivals")
