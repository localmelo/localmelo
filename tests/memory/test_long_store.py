"""The typed long store: entries, text search, decisions and forgetting."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.long.sqlite import SqliteLongTerm


def _v(*values: float) -> list[float]:
    return list(values)


async def _apply(
    store: LongTerm,
    text: str,
    *,
    kind: str = "profile",
    source: str = "user",
    admitted_by: str = "user_statement",
    action: str = "NEW",
    target_id: int | None = None,
    task_id: str = "t1",
    embedding: list[float] | None = None,
    keep_text: bool = False,
    source_ids: list[str] | None = None,
    provenance: dict[str, object] | None = None,
) -> tuple[int | None, str]:
    return await store.apply(
        text=text,
        kind=kind,
        source=source,
        admitted_by=admitted_by,
        provenance=provenance or {"source_ids": source_ids or []},
        task_id=task_id,
        session_id="s1",
        created_at=time.time(),
        embedding=embedding,
        action=action,
        target_id=target_id,
        reason="test",
        source_ids=source_ids or [],
        keep_text=keep_text,
        candidate_id="c1",
    )


@pytest.mark.parametrize("persistent", [False, True])
async def test_task_filtered_search_uses_the_full_corpus_scores(
    tmp_path: Path, persistent: bool
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db") if persistent else LongTerm()
    try:
        for i in range(22):
            await _apply(
                store,
                "transpose transpose matrix " + str(i),
                task_id=f"other{i}",
                embedding=[1.0, 0.0],
            )
        await _apply(
            store,
            "transpose matrix",
            task_id="same",
            provenance={"task_key": "key"},
            embedding=[0.8, 0.6],
        )
        all_text = await store.search_text("transpose matrix", top_k=100)
        filtered = await store.search_text("transpose matrix", top_k=1, task_key="key")
        assert len(filtered) == 1 and filtered[0][2]["task_id"] == "same"
        assert filtered[0][1] == next(
            row[1] for row in all_text if row[2]["task_id"] == "same"
        )
        vectors = await store.search_embedding(
            [1.0, 0.0], top_k=1, task_key="key", exclude_task_ids={"excluded"}
        )
        assert len(vectors) == 1 and vectors[0][2]["task_id"] == "same"
    finally:
        if isinstance(store, SqliteLongTerm):
            await store.aclose()
    if persistent:
        reopened = SqliteLongTerm(tmp_path / "long_term.db")
        try:
            rows = await reopened.search_text(
                "transpose matrix", top_k=1, task_key="key"
            )
            assert len(rows) == 1 and rows[0][2]["provenance"]["task_key"] == "key"
        finally:
            await reopened.aclose()


async def test_in_memory_filtered_search_indexes_its_rows() -> None:
    store = SqliteLongTerm(":memory:")
    try:
        await _apply(store, "Berlin is the capital", kind="semantic")
        rows = await store.search_text("Berlin", kinds=["semantic"])
        assert [text for text, _score, _meta in rows] == ["Berlin is the capital"]
    finally:
        await store.aclose()


async def test_add_keeps_its_signature_and_writes_the_legacy_defaults(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("a step summary", [1.0, 0.0], {"step_id": "s1"})
        entries = await store.active_entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.kind == "episodic"
        assert entry.source == "step_summary"
        assert entry.status == "active"
        assert entry.admitted_by == ""
        assert entry.task_id == ""
        assert entry.vector == (1.0, 0.0)
        assert entry.created_at > 0
    finally:
        await store.aclose()


async def test_apply_writes_every_typed_column(tmp_path: Path) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        entry_id, action = await _apply(
            store,
            "my birthday is 12 May",
            kind="profile",
            source="user",
            source_ids=["r1"],
            provenance={"source_ids": ["r1"], "origin_task_ids": ["t0"]},
        )
        assert action == "NEW" and entry_id is not None
        entry = (await store.active_entries())[0]
        assert entry.text == "my birthday is 12 May"
        assert entry.kind == "profile"
        assert entry.source == "user"
        assert entry.status == "active"
        assert entry.admitted_by == "user_statement"
        assert entry.provenance["origin_task_ids"] == ["t0"]
        assert entry.task_id == "t1"
        assert entry.session_id == "s1"
        assert entry.uses == 0
        assert entry.vector is None
        assert entry.embed_dim == 0
    finally:
        await store.aclose()


async def test_a_text_only_entry_is_skipped_by_cosine_and_found_by_bm25(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("vector row about cats", [1.0, 0.0])
        await _apply(store, "my cat is called Miso", kind="profile")
        scored = await store.search([1.0, 0.0], top_k=5)
        assert [text for text, _score, _meta in scored] == ["vector row about cats"]
        found = await store.search_text("Miso", top_k=5)
        assert [text for text, _score, _meta in found] == ["my cat is called Miso"]
    finally:
        await store.aclose()


async def test_the_bm25_index_is_rebuilt_after_an_apply_and_a_forget(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        assert await store.search_text("Berlin", top_k=5) == []
        entry_id, _ = await _apply(store, "I live in Berlin", kind="profile")
        assert [text for text, _s, _m in await store.search_text("Berlin")] == [
            "I live in Berlin"
        ]
        await store.forget("t1")
        assert await store.search_text("Berlin", top_k=5) == []
        assert entry_id is not None
    finally:
        await store.aclose()


async def test_mixed_dimension_vectors_are_skipped(tmp_path: Path) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("two", [1.0, 0.0])
        await store.add("three", [1.0, 0.0, 0.0])
        assert [text for text, _s, _m in await store.search([1.0, 0.0, 0.0])] == [
            "three"
        ]
        assert [text for text, _s, _m in await store.search([1.0, 0.0])] == ["two"]
    finally:
        await store.aclose()


async def test_500_active_entries_rank_under_five_milliseconds(tmp_path: Path) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        for index in range(500):
            await store.add(f"entry {index} about cats and dogs", [1.0, index / 500])
        await store.search_text("cats", top_k=20)  # builds the index once
        started = time.perf_counter()
        rows = await store.search_text("cats", top_k=20)
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert len(rows) == 20
        assert elapsed_ms < 5, f"BM25 searched 500 entries in {elapsed_ms:.1f} ms"
    finally:
        await store.aclose()


async def test_count_entries_counts_every_row(tmp_path: Path) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("one", [1.0, 0.0])
        await _apply(store, "two words that are long enough", kind="episodic")
        assert await store.count_entries() == 2
    finally:
        await store.aclose()


async def test_forget_removes_the_task_s_entries_and_decisions(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("a legacy step summary", [1.0, 0.0], {"step_id": "s1"})
        await _apply(store, "I live in Berlin", kind="profile", task_id="t1")
        await _apply(store, "I live in Paris", kind="profile", task_id="t2")
        await store.forget("t1")
        assert {entry.text for entry in await store.active_entries()} == {
            "a legacy step summary",
            "I live in Paris",
        }
        assert {row["task_id"] for row in await store.decisions()} == {"t2"}
    finally:
        await store.aclose()


async def test_forget_all_clears_both_tables(tmp_path: Path) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("a legacy step summary", [1.0, 0.0])
        await _apply(store, "I live in Berlin", kind="profile")
        await store.forget_all()
        assert await store.count_entries() == 0
        assert await store.decisions() == []
    finally:
        await store.aclose()


async def test_the_file_and_its_directory_are_private(tmp_path: Path) -> None:
    directory = tmp_path / "memory"
    directory.mkdir()
    store = SqliteLongTerm(directory / "long_term.db")
    try:
        await store.add("one", [1.0, 0.0])
        file_mode = stat.S_IMODE(os.stat(directory / "long_term.db").st_mode)
        assert file_mode == 0o600
    finally:
        await store.aclose()


async def test_decisions_keep_the_hash_always_and_the_text_only_with_the_switch(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await _apply(store, "I live in Berlin", keep_text=False, source_ids=["r1"])
        await _apply(store, "I live in Paris", keep_text=True, source_ids=["r2"])
        rows = await store.decisions()
        assert rows[0]["text"] is None
        assert rows[1]["text"] == "I live in Paris"
        assert rows[0]["text_sha256"]
        assert json.loads(json.dumps(rows[0]["source_ids"])) == ["r1"]
        assert rows[0]["task_id"] == "t1"
    finally:
        await store.aclose()


async def test_apply_rolls_back_both_the_entry_and_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("one", [1.0, 0.0])
        conn = await store._ensure_ready()
        real_execute = conn.execute

        async def fail_on_decisions(sql: str, *args: object) -> object:
            if "INSERT INTO decisions" in sql:
                raise sqlite3.OperationalError("injected failure")
            return await real_execute(sql, *args)

        monkeypatch.setattr(conn, "execute", fail_on_decisions)
        with pytest.raises(sqlite3.OperationalError):
            await _apply(store, "I live in Berlin", kind="profile")
        monkeypatch.setattr(conn, "execute", real_execute)
        assert await store.count_entries() == 1
        assert await store.decisions() == []
    finally:
        await store.aclose()


async def test_the_in_memory_store_answers_the_same_typed_calls() -> None:
    store = LongTerm()
    await store.add("a step summary", [1.0, 0.0], {"task_id": "t1"})
    await _apply(store, "I live in Berlin", kind="profile", task_id="t2")
    assert [entry.kind for entry in await store.active_entries()] == [
        "episodic",
        "profile",
    ]
    assert [text for text, _s, _m in await store.search_text("Berlin")] == [
        "I live in Berlin"
    ]
    assert (await store.decisions())[0]["action"] == "NEW"
    await store.forget("t1")
    assert [entry.text for entry in await store.active_entries()] == [
        "I live in Berlin"
    ]
    await asyncio.sleep(0)


async def test_update_supersedes_and_a_missing_target_becomes_new(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        first, _ = await _apply(store, "I live in Berlin", kind="profile")
        assert first is not None
        second, action = await _apply(
            store, "I live in Paris", kind="profile", action="UPDATE", target_id=first
        )
        assert action == "UPDATE" and second is not None
        active = {entry.id: entry for entry in await store.active_entries()}
        assert list(active) == [second]
        assert active[second].supersedes == first
        assert active[second].links == (first,)
        assert (await store.decisions())[1]["action"] == "UPDATE"

        third, action = await _apply(
            store, "I live in Rome", kind="profile", action="UPDATE", target_id=first
        )
        assert action == "NEW"
        assert (await store.active_entries())[-1].id == third
        assert (await store.decisions())[-1]["target_id"] is None
    finally:
        await store.aclose()


async def test_both_stores_make_the_same_decisions_and_answer_the_same_searches(
    tmp_path: Path,
) -> None:
    writes: list[dict[str, Any]] = [
        {"text": "alpha one", "embedding": [1.0, 0.0]},
        {"text": "alpha two", "embedding": [0.0, 1.0], "admitted_by": "rehearsal"},
        # A duplicate is ignored and bumps the use count of what it matched.
        {"text": "alpha two", "action": "IGNORE", "target_id": 2},
        {"text": "alpha one", "action": "IGNORE", "target_id": 99},
        # An update supersedes its target; a second one finds it superseded
        # and a third finds no target: both become new entries, unlinked.
        {"text": "alpha uno", "action": "UPDATE", "target_id": 1},
        {"text": "alpha eins", "action": "UPDATE", "target_id": 1},
        {"text": "alpha ein", "action": "UPDATE", "target_id": 99},
        # Links point at a row that exists; a missing target stays in the
        # decision row with no link.
        {"text": "alpha near", "action": "RELATED", "target_id": 2},
        {"text": "alpha clash", "action": "CONFLICT", "target_id": 2},
        {"text": "alpha lost", "action": "RELATED", "target_id": 99},
        {"text": "alpha gone", "action": "CONFLICT", "target_id": 99},
        {"text": "alpha long", "reason": "x" * 600},
        {
            "text": "alpha think",
            "kind": "reflection",
            "source": "reflection",
            "provenance": {"origin_task_ids": ["t8", "t9"]},
        },
    ]

    async def run(store: LongTerm) -> tuple[Any, ...]:
        outcomes = []
        for index, write in enumerate(writes):
            params: dict[str, Any] = {
                "kind": "semantic",
                "source": "user",
                "admitted_by": "outcome",
                "provenance": {},
                "task_id": f"t{index}",
                "session_id": "s1",
                "created_at": 100.0 + index,
                "embedding": None,
                "action": "NEW",
                "target_id": None,
                "reason": "test",
                "source_ids": ["r1"],
                "keep_text": True,
                "candidate_id": f"c{index}",
                **write,
            }
            outcomes.append(await store.apply(**params))
        decisions = [
            {key: value for key, value in row.items() if key != "timestamp"}
            for row in await store.decisions()
        ]
        entries = [
            replace(entry, last_used_at=float(entry.last_used_at > 0))
            for entry in await store.active_entries()
        ]
        searches = []
        for query in ("alpha", "alpha one", "think"):
            for kinds in (None, ("semantic",)):
                for exclude in ((), ("t0", "t1")):
                    kwargs: dict[str, Any] = {
                        "kinds": kinds,
                        "exclude_task_ids": exclude,
                    }
                    for rows in (
                        await store.search([1.0, 0.0], 20, **kwargs),
                        await store.search_text(query, 20, **kwargs),
                    ):
                        searches.append(
                            [
                                (text, round(score, 5), {**meta, "last_used_at": 0})
                                for text, score, meta in rows
                            ]
                        )
        return outcomes, decisions, entries, searches

    ram = LongTerm()
    sqlite = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        expected = await run(ram)
        assert await run(sqlite) == expected
    finally:
        await sqlite.aclose()
    outcomes, decisions, entries, _searches = expected
    assert " ".join(action for _id, action in outcomes) == (
        "NEW NEW IGNORE IGNORE UPDATE NEW NEW RELATED CONFLICT RELATED CONFLICT "
        "NEW NEW"
    )
    by_text = {entry.text: entry for entry in entries}
    assert by_text["alpha uno"].supersedes == 1 and by_text["alpha uno"].links == (1,)
    assert by_text["alpha eins"].links == () and by_text["alpha ein"].links == ()
    assert by_text["alpha near"].links == (2,) and by_text["alpha clash"].links == (2,)
    assert by_text["alpha lost"].links == () and by_text["alpha gone"].links == ()
    assert by_text["alpha two"].uses == 1 and "alpha one" not in by_text
    assert [row["target_id"] for row in decisions[2:7]] == [2, 99, 1, None, None]
    assert [row["target_id"] for row in decisions[9:11]] == [99, 99]
    assert len(decisions[11]["reason"]) == 500


async def test_search_returns_the_metadata_merged_with_the_typed_columns(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("cats", [1.0, 0.0], {"step_id": "s1", "custom": 1})
        _text, _score, meta = (await store.search([1.0, 0.0], top_k=1))[0]
        assert meta["step_id"] == "s1"
        assert meta["custom"] == 1
        assert meta["kind"] == "episodic"
        assert meta["source"] == "step_summary"
        assert meta["status"] == "active"
        assert meta["origin_task_ids"] == []
    finally:
        await store.aclose()


def test_a_vector_row_may_hold_a_tuple_stored_as_float32(tmp_path: Path) -> None:
    # The store's BLOB is float32; a caller reading an entry back gets the
    # same values, not the original float64 list.
    async def run() -> None:
        store = SqliteLongTerm(tmp_path / "long_term.db")
        try:
            await store.add("one", [0.1, 0.2])
            entry = (await store.active_entries())[0]
            assert entry.vector is not None
            assert entry.vector == tuple(
                float(v) for v in np.asarray([0.1, 0.2], dtype=np.float32)
            )
        finally:
            await store.aclose()

    asyncio.run(run())


async def test_two_legacy_rows_with_equal_scores_keep_their_id_order(
    tmp_path: Path,
) -> None:
    store = SqliteLongTerm(tmp_path / "long_term.db")
    try:
        await store.add("first in id order", [1.0, 0.0])
        await store.add("second in id order", [1.0, 0.0])
        rows = await store.search([1.0, 0.0], top_k=5)
        assert [text for text, _score, _meta in rows] == [
            "first in id order",
            "second in id order",
        ]
    finally:
        await store.aclose()
