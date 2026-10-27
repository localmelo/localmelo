"""The off reason is the first thing that took a run off its template.

A stuck notice at step k ends the attempt, and the reflection it causes
starts at the same step: the reason stays ``stuck``, at that event and
after, never turning into ``retrying``. An earlier reflection stays the
first reason when the run gets stuck later.
"""

from __future__ import annotations

from localmelo.melo.memory.procedures.template import (
    DEFAULT_PARAMS,
    SkeletonItem,
    Stats,
    TemplateView,
)
from localmelo.melo.memory.procedures.trace import Trace
from localmelo.melo.memory.procedures.tracker import bar_series


def view() -> TemplateView:
    return TemplateView(
        level=2,
        reason="",
        match="exact",
        routine_id="r",
        template_id="r:t",
        version=2,
        model="m",
        stats=Stats(
            n=2,
            steps=(8, 8),
            median=8,
            lo=8,
            hi=8,
            spread=0.0,
            cv=0.0,
            wall_s=(1.0, 1.0, 1.0),
            attempts_share=0.0,
            gate=True,
        ),
        skeleton=(SkeletonItem(("R",), "a.py", "Read a.py"),),
        params=DEFAULT_PARAMS,
    )


def test_a_stuck_notice_stays_the_reason_after_its_reflection() -> None:
    trace = Trace(clock=lambda: 0.0)
    for k in (1, 2, 3):
        trace.step(k, 1)
    trace.stuck()
    assert bar_series(view(), trace)[1] == "stuck"
    trace.reflection()  # the reflection the stuck notice caused
    assert bar_series(view(), trace)[1] == "stuck"
    trace.step(4, 2)
    assert bar_series(view(), trace)[1] == "stuck"


def test_an_earlier_reflection_stays_the_reason() -> None:
    trace = Trace(clock=lambda: 0.0)
    for k in (1, 2, 3):
        trace.step(k, 1)
    trace.reflection()
    trace.step(4, 2)
    assert bar_series(view(), trace)[1] == "retrying"
    for k in (5, 6):
        trace.step(k, 2)
    trace.stuck()
    assert bar_series(view(), trace)[1] == "retrying"
    trace.step(7, 3)
    assert bar_series(view(), trace)[1] == "retrying"
