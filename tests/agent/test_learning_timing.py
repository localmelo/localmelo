"""What learning costs a run, on a procedures.db that holds 1,000 episodes.

Section 10: the run-end write (from ``run.finished`` to ``run_task``'s
return) under 50 ms at p95, and the run-start lookup (from ``run.started``
to the first ``run.progress``) under 20 ms. Timed here, never logged: the
p95 values are recorded as test properties (``--junitxml``).
"""

from __future__ import annotations

import functools
import statistics
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.procedures.normalize import key_for_query, length_options
from localmelo.melo.memory.procedures.store import RunFacts, SqliteProcedures
from localmelo.tests.agent.procedural_helpers import (
    MODEL,
    QUERY,
    ScriptedLLM,
    fix_run,
    make_agent,
    passing_tools,
    past_trace,
    query_key,
)

RUNS = 40


def _mark(marks: dict[str, float], event: AgentEvent) -> None:
    """When run.started, the first run.progress and run.finished came."""
    if event.type == "run.started":
        marks["started"] = time.perf_counter()
    elif event.type == "run.progress" and "progress" not in marks:
        marks["progress"] = time.perf_counter()
    elif event.type == "run.finished":
        marks["finished"] = time.perf_counter()


def p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[-1]


async def _fill(store: SqliteProcedures, facts: Callable[[int], RunFacts]) -> None:
    for i in range(1000):
        await store.learn(facts(i))


async def test_learning_and_lookup_are_fast_with_1000_episodes(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    agent = make_agent(
        ScriptedLLM(fix_run() * RUNS), tools=passing_tools(), memory_dir=str(tmp_path)
    )
    store = agent.hippo.procedures
    assert isinstance(store, SqliteProcedures)
    options = length_options(agent._run_config())
    place_key = query_key(agent)
    other_keys = [key_for_query(f"task number {i}", place_key.place) for i in range(99)]

    def facts(i: int) -> RunFacts:
        # One routine in ten is the timed one, so its window is full.
        key = place_key if i % 10 == 0 else other_keys[i % 99]
        steps = 4 + i % 3
        return RunFacts.from_trace(
            past_trace(steps),
            task_id=uuid.uuid4().hex,
            key=key,
            model=MODEL,
            options=options,
            status="completed",
            steps=steps,
            attempts=1,
            duration_s=30.0,
        )

    writes: list[float] = []
    looks: list[float] = []
    try:
        await _fill(store, facts)
        for _ in range(RUNS):
            marks: dict[str, float] = {}
            result = await agent.run_task(
                QUERY, observe=functools.partial(_mark, marks), progress=True
            )
            returned = time.perf_counter()
            assert result.status == "completed"
            writes.append(returned - marks["finished"])
            looks.append(marks["progress"] - marks["started"])
        rows = await store.export_rows()
    finally:
        await agent.close()
    assert sum(row["table"] == "episodes" for row in rows) == 1000 + RUNS
    record_property("run_end_write_p95_ms", round(p95(writes) * 1000, 2))
    record_property("lookup_p95_ms", round(p95(looks) * 1000, 2))
    assert p95(writes) < 0.050
    assert p95(looks) < 0.020


def test_p95_is_the_95th_percentile() -> None:
    assert p95([float(i) for i in range(1, 101)]) == pytest.approx(95.95)
