"""Typed decisions are atomic and preserve older assertions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localmelo.melo.memory.assertions import parse_assertion
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.memory.recall import (
    docs_from_entries,
    hit_from_doc,
    render_recall_block,
)


async def apply(
    store: LongTerm, text: str, observed_at: float, candidate: str, task: str = "task"
) -> tuple[int | None, str]:
    assertion = parse_assertion(text)
    assert assertion is not None
    return await store.apply(
        text=text,
        kind="semantic",
        source="user",
        admitted_by="user_statement",
        provenance={"assertion": {**assertion.to_dict(), "observed_at": observed_at}},
        task_id=task,
        session_id="",
        created_at=observed_at,
        embedding=None,
        action="NEW",
        target_id=None,
        reason="typed assertion",
        source_ids=[],
        keep_text=False,
        candidate_id=candidate,
        typed_assertion=True,
    )


@pytest.mark.parametrize("sqlite", [False, True])
async def test_atomic_typed_updates_and_conflicts(tmp_path: Path, sqlite: bool) -> None:
    store: LongTerm = SqliteLongTerm(tmp_path / "long.db") if sqlite else LongTerm()
    try:
        first, action = await apply(store, "Mina lives in Paris.", 10, "a")
        assert first is not None and action == "NEW"
        second, action = await apply(store, "Mina now lives in Rome.", 20, "b")
        assert second is not None and action == "UPDATE"
        active = await store.active_entries()
        assert len(active) == 1 and active[0].supersedes == first
        history = await store.history_entries("task")
        assert [entry.id for entry in history] == [second, first]
        assert [entry.status for entry in history] == ["active", "superseded"]
        assert await store.history_entries("") == []
        third, action = await apply(store, "Mina lives in Oslo.", 30, "c")
        assert third is not None and action == "CONFLICT"
        assert len(await store.active_entries()) == 2
        assert (await apply(store, "Mina now lives in Berlin.", 5, "d"))[
            1
        ] == "CONFLICT"
    finally:
        if isinstance(store, SqliteLongTerm):
            await store.aclose()


@pytest.mark.parametrize("sqlite", [False, True])
async def test_history_search_labels_superseded_without_changing_default(
    tmp_path: Path, sqlite: bool
) -> None:
    store: LongTerm = SqliteLongTerm(tmp_path / "long.db") if sqlite else LongTerm()
    try:
        await apply(store, "Mina lives in Paris.", 10, "a")
        await apply(store, "Mina now lives in Rome.", 20, "b")
        assert len(await store.search_text("Mina lives", top_k=10)) == 1
        rows = await store.search_text("Mina lives", top_k=10, include_history=True)
        docs = docs_from_entries(rows, include_history=True)
        assert len(docs) == 2
        docs.sort(key=lambda doc: float(doc.meta["created_at"]), reverse=True)
        block = render_recall_block(
            [hit_from_doc(doc, score=1, rank=i + 1) for i, doc in enumerate(docs)],
            trigger="start",
            retriever="bm25",
            placement="message",
        )
        assert block.index("Rome") < block.index("Paris")
        assert "superseded" in block
        assert len(docs_from_entries(rows)) == 1
    finally:
        if isinstance(store, SqliteLongTerm):
            await store.aclose()


async def test_history_reopens_and_forget_removes_superseded(tmp_path: Path) -> None:
    path = tmp_path / "long.db"
    store = SqliteLongTerm(path)
    await apply(store, "Mina lives in Paris.", 10, "a")
    await apply(store, "Mina now lives in Rome.", 20, "b")
    await store.aclose()
    reopened = SqliteLongTerm(path)
    try:
        history = await reopened.history_entries("task")
        assert [row.text for row in history] == [
            "Mina now lives in Rome.",
            "Mina lives in Paris.",
        ]
        assert [row.status for row in history] == ["active", "superseded"]
        await reopened.forget("task")
        assert await reopened.history_entries("task") == []
    finally:
        await reopened.aclose()


async def test_ram_assertion_provenance_is_defensively_copied() -> None:
    store = LongTerm()
    assertion = parse_assertion("Mina lives in Paris.")
    assert assertion is not None
    provenance = {"assertion": assertion.to_dict()}
    await store.apply(
        text=assertion.original,
        kind="semantic",
        source="user",
        admitted_by="user_statement",
        provenance=provenance,
        task_id="task",
        session_id="",
        created_at=10,
        embedding=None,
        action="NEW",
        target_id=None,
        reason="typed assertion",
        source_ids=[],
        keep_text=False,
        candidate_id="a",
        typed_assertion=True,
    )
    provenance["assertion"]["value_text"] = "tampered"
    entry = (await store.active_entries())[0]
    assert entry.provenance["assertion"]["value_text"] == "Paris"
    entry.provenance["assertion"]["value_text"] = "tampered again"
    assert (await store.active_entries())[0].provenance["assertion"][
        "value_text"
    ] == "Paris"


async def test_concurrent_sqlite_writers_compare_inside_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long.db"
    first, second = SqliteLongTerm(path), SqliteLongTerm(path)
    try:
        await apply(first, "Mina lives in Paris.", 10, "a")
        results = await asyncio.gather(
            apply(first, "Mina now lives in Rome.", 20, "b"),
            apply(second, "Mina now lives in Oslo.", 21, "c"),
        )
        assert all(action in ("UPDATE", "CONFLICT") for _, action in results)
        active = await first.active_entries()
        assert any(entry.text == "Mina now lives in Oslo." for entry in active)
    finally:
        await first.aclose()
        await second.aclose()
