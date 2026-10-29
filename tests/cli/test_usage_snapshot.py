"""The usage snapshot of /status, on synthetic event sequences (#176 SL1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from localmelo.support.usage_snapshot import (
    TaskCounts,
    UsageSnapshot,
    status_segments,
    usage_lines,
)

T0 = 1_000.0


def message(
    prompt: int | None,
    cached: int | None,
    ts: float = T0,
    kind: str = "assistant.message",
) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if cached is not None:
        usage["cached_tokens"] = cached
    return {"type": kind, "ts": ts, "data": {"usage": usage}}


def budget(measured: int | None, limit: int | None) -> dict[str, Any]:
    data = {"measured_prompt_tokens": measured, "input_limit": limit}
    return {"type": "context_budget", "ts": T0, "data": data}


def steps(payload: Mapping[str, Any]) -> list[str]:
    return [f"level {payload['level']} · step {payload['step']}", "past 3 runs"]


def render(
    events: list[dict[str, Any]],
    *,
    local: bool = True,
    now: float = T0 + 30,
    tasks: TaskCounts | None = None,
) -> list[str]:
    snap = UsageSnapshot()
    for event in events:
        snap.observe(event)
    return usage_lines(snap, local=local, now=now, steps=steps, tasks=tasks)


def test_no_requests_yet_says_why_for_every_segment() -> None:
    assert render([]) == [
        "cache      no requests yet",
        "context    no context budget measured yet",
        "steps      no progress facts yet",
        "tasks      no task store in this session",
    ]


def test_local_backend_shows_shares_and_idle_time_without_a_countdown() -> None:
    lines = render([message(1000, 0), message(2000, 1500, ts=T0 + 10)], now=T0 + 75)
    assert lines[0] == (
        "cache      last request 75% cached (1.5k of 2.0k prompt tokens)"
        " · session 50% over 2 requests"
        " · idle 1m05s"
    )


def test_a_server_without_cached_tokens_shows_dashes_not_zero() -> None:
    line = render([message(1000, None)])[0]
    assert "last request --: the server did not report cached tokens" in line
    assert "0%" not in line and "session" not in line


def test_the_average_counts_only_requests_that_reported_cached_tokens() -> None:
    line = render([message(1000, 500), message(1000, 1000), message(9000, None)])[0]
    assert "last request --" in line
    assert "session 75% over 2 requests" in line


def test_reflection_requests_count_like_planning_requests() -> None:
    line = render([message(400, 100, kind="reflection")])[0]
    assert "last request 25% cached" in line


def test_cloud_backend_shows_cache_and_no_countdown_or_idle_time() -> None:
    line = render([message(1000, 800)], local=False)[0]
    assert "last request 80% cached" in line
    assert "idle" not in line
    assert "cache lifetime not tracked yet" in line


def test_cached_tokens_never_exceed_the_prompt() -> None:
    assert "100% cached" in render([message(100, 900)])[0]


def test_context_is_the_last_measured_prompt_over_the_input_limit() -> None:
    lines = render([budget(None, 8192), budget(3000, 8192), budget(None, 8192)])
    assert lines[1] == (
        "context    3.0k of 8.1k input tokens (37%) at the last measured request"
    )
    assert "no input limit known" in render([budget(3000, None)])[1]


def test_a_run_in_progress_and_the_last_run() -> None:
    running = [
        {"type": "run.started", "ts": T0, "data": {}},
        {"type": "run.progress", "ts": T0, "data": {"level": 1, "step": 4}},
    ]
    lines = render(running)
    assert lines[2:4] == [
        "steps      running · level 1 · step 4",
        "           past 3 runs",
    ]
    finished = running + [{"type": "run.finished", "ts": T0, "data": {}}]
    assert render(finished)[2] == "steps      last run · level 1 · step 4"
    assert render(running[:1])[2] == "steps      run has no facts yet"


def test_tasks_with_a_store() -> None:
    counts = TaskCounts(running=1, queued=3, held=2, top_level=1)
    assert render([], tasks=counts)[-1] == (
        "tasks      1 running · 3 queued · 2 held · top priority P1"
    )
    assert render([], tasks=TaskCounts(0, 0, 0))[-1] == (
        "tasks      0 running · 0 queued · 0 held"
    )


def test_malformed_events_are_ignored() -> None:
    snap = UsageSnapshot()
    for event in (
        {"type": "assistant.message", "data": "x"},
        {"type": "assistant.message", "data": {"usage": [1]}},
        message(True, 1),  # type: ignore[arg-type]
        message(0, 0),
        {"type": "run.progress"},
    ):
        snap.observe(event)
    assert snap.requests == 0 and snap.progress is None


def test_hide_empty_leaves_out_the_rows_with_nothing_to_say() -> None:
    snap = UsageSnapshot()
    assert usage_lines(snap, local=True, now=T0, steps=steps, hide_empty=True) == []
    snap.observe(message(1000, 500))
    assert usage_lines(snap, local=True, now=T0, steps=steps, hide_empty=True) == [
        "cache      last request 50% cached (500 of 1.0k prompt tokens) · "
        "session 50% over 1 request · idle 0s"
    ]
    assert usage_lines(snap, local=True, now=T0, steps=steps)[0].startswith("cache ")
    assert len(usage_lines(snap, local=True, now=T0, steps=steps)) == 4


# ── The status row (#176 SL2) ──


def row(
    events: list[dict[str, Any]], *, local: bool = True, now: float = T0 + 30
) -> list[str]:
    snap = UsageSnapshot()
    for event in events:
        snap.observe(event)
    return status_segments(snap, local=local, now=now)


def progress(**data: Any) -> dict[str, Any]:
    return {"type": "run.progress", "ts": T0, "data": {"level": 0, **data}}


def test_row_has_no_segment_without_data() -> None:
    assert row([]) == []


def test_cache_segment_shows_the_last_requests_share_and_idle_for_a_local_server() -> (
    None
):
    events = [message(1000, 100), message(2000, 1660)]
    assert row(events) == ["cache 83%"]
    assert row(events, now=T0 + 190) == ["cache 83% idle 3m"]
    assert row(events, now=T0 + 3 * 3600 + 120) == ["cache 83% idle 3h02m"]
    assert row(events, local=False, now=T0 + 190) == ["cache 83%"]


def test_cache_segment_is_dashes_when_the_server_reports_no_cached_tokens() -> None:
    assert row([message(1000, None)]) == ["cache --"]


def test_ctx_segment_needs_a_measurement_and_shows_the_limit_when_known() -> None:
    assert row([budget(None, 32768)]) == []
    assert row([budget(12345, 32768)]) == ["ctx 12k/32k"]
    assert row([budget(900, None)]) == ["ctx 900"]


def test_steps_segment_is_level_zero_facts_only_while_a_run_is_active() -> None:
    started = {"type": "run.started", "ts": T0, "data": {}}
    finished = {"type": "run.finished", "ts": T0, "data": {}}
    events = [started, progress(step=4, attempt=2, phase="fix", past={"runs": 3})]
    assert row(events) == ["step 4 attempt 2 phase fix"]
    assert row([started, progress(step=1)]) == ["step 1"]
    assert row([*events, finished]) == []
    for text in row(events):
        assert "%" not in text and "/" not in text


def test_tasks_segment_only_with_a_task_store() -> None:
    snap = UsageSnapshot()
    assert status_segments(snap, local=True, now=T0) == []
    counts = TaskCounts(running=1, queued=2, held=0)
    assert status_segments(snap, local=True, now=T0, tasks=counts) == [
        "tasks 1 running 2 queued"
    ]
    held = TaskCounts(running=0, queued=0, held=1)
    assert status_segments(snap, local=True, now=T0, tasks=held) == [
        "tasks 0 running 0 queued 1 held"
    ]
