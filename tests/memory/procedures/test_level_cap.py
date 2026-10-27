"""Level 2 is off (decision L-19): a run shows at most level 1.

The gate still gives level 2 (template.decide_level) and the replay still
measures it; only what a running task shows is capped, by one constant.
"""

from __future__ import annotations

import pytest

from localmelo.melo.memory.procedures import tracker
from localmelo.melo.memory.procedures.template import (
    DEFAULT_PARAMS,
    SkeletonItem,
    Stats,
    TemplateView,
)
from localmelo.melo.memory.procedures.trace import Trace
from localmelo.melo.memory.procedures.tracker import shown_view, track


def view(level: int) -> TemplateView:
    return TemplateView(
        level=level,
        reason="",
        match="exact",
        routine_id="r",
        template_id="r:abc",
        version=2,
        model="m",
        stats=Stats(
            n=2,
            steps=(4, 4),
            median=4,
            lo=4,
            hi=4,
            spread=0.0,
            cv=0.0,
            wall_s=(1.0, 1.0, 1.0),
            attempts_share=0.0,
            tail_after_pass=1,
            gate=True,
            gate_reason="",
        ),
        skeleton=(SkeletonItem(("R",), "a.py", "Read a.py"),),
        params=DEFAULT_PARAMS,
    )


def test_level_2_ships_off() -> None:
    assert tracker.LEVEL2_ENABLED is False


def test_level_2_is_shown_at_level_1_while_off() -> None:
    shown = shown_view(view(2))
    assert shown is not None
    assert (shown.level, shown.reason) == (1, "level 2 is off")
    assert shown.stats == view(2).stats and shown.skeleton == view(2).skeleton


@pytest.mark.parametrize("enabled", [False, True])
def test_lower_levels_and_no_view_are_unchanged(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", enabled)
    assert shown_view(None) is None
    for level in (0, 1):
        assert shown_view(view(level)) == view(level)


def test_level_2_is_shown_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracker, "LEVEL2_ENABLED", True)
    assert shown_view(view(2)) == view(2)


def test_a_capped_run_never_carries_a_bar() -> None:
    trace = Trace(clock=lambda: 0.0)
    for k in range(1, 10):  # past every run in the window
        trace.step(k, 1)
        trace.tool("file_read", {"path": "/a.py"}, "success", "")
    for ended in (None, "completed"):
        state = track(shown_view(view(2)), trace, ended=ended)
        assert state.level == 1
        assert (state.fraction, state.expected_steps, state.off_reason) == (
            None,
            None,
            None,
        )
        assert state.state == ("running" if ended is None else "ended")
