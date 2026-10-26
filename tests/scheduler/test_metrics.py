"""Metrics weight completion by gold level and judge starvation within a class.

Requirement 51 and decision D30: WCT weights by gold level and counts an
unfinished task at the horizon; percentiles use the nearest rank; empty
and zero cases give ``None`` or floored values; the starvation ratio
counts only waits without higher-class work (booked by the engine);
deadline misses and tardiness count late and unfinished tasks.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.support.scheduler import params_for_arm
from localmelo.support.scheduler.model import us
from localmelo.support.scheduler.sim import Workload, compute_metrics, run
from localmelo.support.scheduler.sim.engine import (
    ReplyRecord,
    SimResult,
    TaskRecord,
    params_json,
)
from localmelo.support.scheduler.sim.metrics import percentile
from localmelo.support.scheduler.sim.workloads import WALL_START_US
from localmelo.tests.scheduler.helpers import flat_template, wtask

HORIZON = us(1000.0)


def rec(
    task_id: str, level: int, arrival_s: float, finish_s: float | None, **kw: Any
) -> TaskRecord:
    values: dict[str, Any] = {
        "origin": "user",
        "kind": "task",
        "first_admit_us": us(arrival_s),
        "status": "done" if finish_s is not None else "running",
        "deadline_wall_us": None,
        "service_s": 10.0,
        "alone_s": 10.0,
        "wait_own_s": 0.0,
        "wait_total_s": 0.0,
        "parks": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "reprefill_tokens": 0,
    }
    values.update(kw)
    finish = None if finish_s is None else us(finish_s)
    return TaskRecord(
        task_id, level, level, arrival_us=us(arrival_s), finish_us=finish, **values
    )


def result(tasks: Any = (), replies: Any = (), **kw: Any) -> SimResult:
    values: dict[str, Any] = {
        "decisions": 0,
        "log_sha256": "",
        "log": (),
        "alarms": (),
        "busy_us": 0,
        "yields": 0,
        "foreign_lock_task_calls": 0,
        "params_json": params_json(params_for_arm("las2")),
    }
    values.update(kw)
    horizon = values.pop("horizon", HORIZON)
    return SimResult(
        "test",
        1,
        "las2",
        horizon,
        WALL_START_US,
        tuple(tasks),
        tuple(replies),
        **values,
    )


def test_wct_weights_by_gold_level_and_counts_unfinished_tasks_at_the_horizon() -> None:
    m = compute_metrics(
        result(
            [rec("a", 1, 0.0, 100.0), rec("b", 3, 0.0, 300.0), rec("c", 3, 600.0, None)]
        )
    )
    # (16·100 + 1·300 + 1·(1000 − 600)) / (16 + 1 + 1)
    assert m.wct_s == pytest.approx((1600 + 300 + 400) / 18)
    assert m.finished_by_class == (0, 1, 0, 1, 0)
    assert m.tasks_by_class == (0, 1, 0, 2, 0)
    assert m.completion_by_class == (None, 1.0, None, 0.5, None)
    assert m.throughput_per_h[3] == pytest.approx(3.6)
    odd = compute_metrics(result([rec("x", 5, 0.0, 10.0), rec("y", 2, 0.0, 10.0)]))
    assert odd.wct_s == pytest.approx(10.0)  # P5 is outside the weighted classes


def test_percentiles_use_the_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0) == 1.0  # index max(0, 0 − 1)
    assert percentile(values, 95) == 95.0  # p·n = 9500, a whole multiple of 100
    assert percentile(values, 95.5) == 96.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([], 50) is None
    replies = [
        ReplyRecord(0, us(s), us(s), "answer", False) for s in (1.0, 2.0, 3.0, 4.0)
    ]
    m = compute_metrics(result(replies=replies))
    assert m.reply_whole_s == (2.0, 4.0, 4.0)
    assert m.reply_whole_us == (us(1.0), us(2.0), us(3.0), us(4.0))


def test_empty_and_zero_cases_give_none_or_floored_values() -> None:
    m = compute_metrics(result())
    assert (m.wct_s, m.slowdown_mean, m.starvation_max) == (None, None, None)
    assert (m.cache_hit_share, m.reprefill_per_finished, m.by_code_share) == (None,) * 3
    assert m.reply_whole_s == (None, None, None) and m.deadline_miss_rate is None
    empty = compute_metrics(result(horizon=0))  # no hours: rates are 0, not a crash
    assert empty.throughput_per_h == (0.0,) * 5 and empty.parks_per_h == 0.0
    tiny = compute_metrics(
        result(
            [
                rec("a", 2, 0.0, 40.0, alone_s=0.0),
                rec("b", 2, 100.0, None, first_admit_us=None),
            ]
        )
    )
    assert tiny.slowdown_p95 == pytest.approx(900.0 / 20.0)  # b: 900 s over s0 = 20 s
    assert tiny.slowdown_mean == pytest.approx((40.0 / 20.0 + 900.0 / 20.0) / 2)
    assert tiny.start_delay_p95_s[2] == pytest.approx(900.0)  # never admitted
    assert tiny.start_delay_met == (None, True, None)
    assert tiny.model_loads_per_h == 0.0


def test_starvation_counts_only_waits_without_higher_class_work() -> None:
    m = compute_metrics(
        result(
            [
                rec(
                    "a",
                    3,
                    0.0,
                    500.0,
                    wait_own_s=60.0,
                    wait_total_s=900.0,
                    service_s=20.0,
                )
            ]
        )
    )
    assert m.starvation_max == pytest.approx((60.0 + 20.0) / (20.0 + 20.0))
    assert m.starvation_total_max == pytest.approx((900.0 + 20.0) / (20.0 + 20.0))
    # The engine books own-class waits: a P3 behind P2 work waits for none
    # of them; a P3 behind another P3 does.
    one = flat_template("one", 3, 1320)
    wl = Workload(
        "starve",
        0,
        us(600.0),
        WALL_START_US,
        (wtask("p2", 0.0, 2, one), wtask("p3", 1.0, 3, one), wtask("q3", 1.0, 3, one)),
    )
    r = run(wl, "class", params_for_arm("class", n_a=1))
    by_id = {t.task_id: t for t in r.tasks}
    assert by_id["p3"].wait_total_s > 0 and by_id["p3"].wait_own_s == 0  # behind P2
    assert by_id["q3"].wait_total_s > by_id["q3"].wait_own_s > 0  # also behind p3


def test_deadline_misses_and_tardiness() -> None:
    def due(s: float) -> int:
        return WALL_START_US + us(s)

    m = compute_metrics(
        result(
            [
                rec("on_time", 1, 0.0, 100.0, deadline_wall_us=due(200.0)),
                rec("late", 1, 0.0, 300.0, deadline_wall_us=due(200.0)),
                rec("unfinished", 1, 0.0, None, deadline_wall_us=due(1500.0)),
                rec("none", 2, 0.0, 900.0),
            ]
        )
    )
    assert m.deadline_miss_rate == pytest.approx(2 / 3)
    assert m.tardiness_mean_s == pytest.approx(
        100.0 / 3
    )  # 0 + 100 + 0 (due after the horizon)
