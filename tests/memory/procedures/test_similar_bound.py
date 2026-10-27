"""A new routine is compared with its place's latest routines only.

Every run of a new task looks for a similar routine in its folder, at its
end and, with progress, at its start. Over a store that keeps every
routine, comparing with all of them would cost each such run more as
the store grows (about 17 µs a routine): the candidates are the place's
latest :data:`SIMILAR_MAX_ROUTINES` routines.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.procedures import store as store_module
from localmelo.melo.memory.procedures.normalize import (
    SIMILAR_MAX_ROUTINES,
    find_similar,
    key_for_query,
)
from localmelo.melo.memory.procedures.store import (
    MemoryProcedures,
    Procedures,
    RunFacts,
    SqliteProcedures,
)

PLACE = "/work/project"


@pytest.fixture(params=["sqlite", "memory"])
async def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[Procedures]:
    made: Procedures = (
        SqliteProcedures(tmp_path / "procedures.db")
        if request.param == "sqlite"
        else MemoryProcedures()
    )
    try:
        yield made
    finally:
        await made.aclose()


def facts(query: str, place: str = PLACE, t: float = 0.0) -> RunFacts:
    return RunFacts(
        task_id=uuid.uuid4().hex,
        key=key_for_query(query, place),
        model="m",
        options={},
        status="failed",
        steps=1,
        created_at=t,
    )


def test_the_bound() -> None:
    assert SIMILAR_MAX_ROUTINES == 1000


async def test_only_the_latest_routines_of_the_place_are_candidates(
    store: Procedures, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_module, "SIMILAR_MAX_ROUTINES", 4)
    for i in range(6):
        await store.learn(facts(f"task number {i}", t=float(i)))
    await store.learn(facts("elsewhere", place="/other", t=10.0))
    seen: list[list[str]] = []

    def spy(query_text: str, candidates: Sequence[tuple[str, str]]) -> Any:
        seen.append([text for _, text in candidates])
        return find_similar(query_text, candidates)

    monkeypatch.setattr(store_module, "find_similar", spy)
    await store.learn(facts("a new task", t=11.0))
    await store.lookup(key_for_query("another task", PLACE), model="m", options={})
    # At the end of the run, before its own routine is added; then at a
    # later run's start, with that routine among the latest.
    assert seen == [
        [f"task number {i}" for i in range(2, 6)],
        [f"task number {i}" for i in range(3, 6)] + ["a new task"],
    ]


async def test_a_similar_routine_is_found_among_the_latest(
    store: Procedures, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_module, "SIMILAR_MAX_ROUTINES", 6)
    old = "summarize the unread mail into the daily notes file"
    await store.learn(facts(old, t=0.0))
    for i in range(6):
        await store.learn(facts(f"unrelated chore {i} {i * 7}", t=1.0 + i))
    # The old routine is past the bound: no candidate, no match.
    result = await store.learn(facts(old + " please", t=20.0))
    assert result.routine.similar_to is None
    # The routine just added is among the latest: it is matched.
    again = await store.learn(facts(old + " please now", t=21.0))
    assert again.routine.similar_to == result.routine.routine_id
