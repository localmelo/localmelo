"""R6-lite's start recall: active entries of other tasks, kinds, BM25."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.long import LongTerm

from .test_compress import FakeLLM, _agent, _answer, _options


async def _seed(agent: Any, rows: list[tuple[str, str, str, str]]) -> None:
    for index, (text, kind, source, task_id) in enumerate(rows, start=1):
        await agent.hippo.long.apply(
            text=text,
            kind=kind,
            source=source,
            admitted_by="rehearsal",
            provenance={"task_text": "an earlier task", "source_ids": []},
            task_id=task_id,
            session_id="",
            created_at=float(index),
            embedding=None,
            action="NEW",
            target_id=None,
            reason="seed",
            source_ids=[],
            keep_text=False,
            candidate_id=f"seed-{index}",
        )


def _prompt(server: FakeLLM) -> str:
    return "\n".join(
        message.content for message in server.requests[0] if message.content
    )


@pytest.mark.asyncio
async def test_start_recalls_active_entries_of_other_tasks(tmp_path: Path) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            retriever="bm25",
            max_steps=1,
        ),
    )
    await _seed(
        agent,
        [
            ("the deployment checklist is owned by Alice", "semantic", "user", "t-a"),
            ("the deployment runbook lives in the wiki", "episodic", "digest", "t-b"),
            ("I prefer the deployment on Fridays", "profile", "user", "t-c"),
            ("the deployment failed on its first try", "reflection", "reflection", "d"),
        ],
    )
    result = await agent.run_task("deployment checklist for the release")
    await agent.close()
    assert result.status == "completed"
    prompt = _prompt(server)
    assert "earlier tasks" in prompt
    # R5's start trigger keeps two long hits.
    assert "Alice" in prompt
    assert "runbook" in prompt or "Fridays" in prompt
    assert result.memory_stats is not None
    assert result.memory_stats["recalls"] == 1


@pytest.mark.asyncio
async def test_a_superseded_entry_is_never_recalled(tmp_path: Path) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            retriever="bm25",
            max_steps=1,
        ),
    )
    await _seed(
        agent,
        [("the deployment checklist is owned by Alice", "semantic", "user", "t-a")],
    )
    first = await agent.hippo.long.active_entries("semantic")
    await agent.hippo.long.apply(
        text="the deployment checklist is owned by Bob",
        kind="semantic",
        source="user",
        admitted_by="user_statement",
        provenance={"task_text": "an earlier task", "source_ids": []},
        task_id="t-a",
        session_id="",
        created_at=9.0,
        embedding=None,
        action="UPDATE",
        target_id=first[0].id,
        reason="update",
        source_ids=[],
        keep_text=False,
        candidate_id="seed-2",
    )
    await agent.run_task("deployment checklist for the release")
    await agent.close()
    prompt = _prompt(server)
    assert "Bob" in prompt
    assert "Alice" not in prompt


@pytest.mark.asyncio
async def test_long_not_in_recall_kinds_sends_no_long_request(tmp_path: Path) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            retriever="bm25",
            recall_kinds=frozenset({"note"}),
            max_steps=1,
        ),
    )
    await _seed(
        agent,
        [("the deployment checklist is owned by Alice", "semantic", "user", "t-a")],
    )
    await agent.run_task("deployment checklist for the release")
    await agent.close()
    assert "earlier tasks" not in _prompt(server)


@pytest.mark.asyncio
async def test_a_text_only_store_recalls_with_bm25_and_keeps_origin_ids(
    tmp_path: Path,
) -> None:
    events: list[AgentEvent] = []
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            retriever="bm25",
            max_steps=1,
        ),
    )
    await _seed(
        agent,
        [("the deployment checklist is owned by Alice", "semantic", "user", "t-a")],
    )
    await agent.run_task("deployment checklist for the release", on_event=events.append)
    await agent.close()
    recall = next(event for event in events if event.type == "recall")
    assert recall.data["trigger"] == "start"
    assert recall.data["hits"][0]["doc_id"] == "long:1"
    assert recall.data["hits"][0]["origin_task_ids"] == ["t-a"]


@pytest.mark.asyncio
async def test_memory_false_has_no_long_leg(tmp_path: Path) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            memory=False,
            retriever="bm25",
            max_steps=1,
        ),
    )
    await agent.run_task("deployment checklist for the release")
    await agent.close()
    assert "earlier tasks" not in _prompt(server)


def test_docs_from_entries_uses_the_entry_id_not_a_text_hash() -> None:
    from localmelo.melo.memory.recall import docs_from_entries

    rows = [
        ("same memory", 0.9, {"id": 4, "task_id": "t-a", "created_at": 1.0}),
        ("same memory", 0.8, {"id": 5, "task_id": "t-b", "created_at": 2.0}),
    ]
    docs = docs_from_entries(rows)
    assert [doc.doc_id for doc in docs] == ["long:4", "long:5"]
    assert [doc.meta["store_kind"] for doc in docs] == ["", ""]


def test_docs_from_entries_drops_a_non_active_row() -> None:
    from localmelo.melo.memory.recall import docs_from_entries

    rows = [
        ("hidden", 0.9, {"id": 1, "status": "superseded", "task_id": "t-a"}),
        ("shown", 0.8, {"id": 2, "status": "active", "task_id": "t-b"}),
    ]
    assert [doc.text for doc in docs_from_entries(rows)] == ["shown"]


@pytest.mark.asyncio
async def test_the_in_memory_long_term_searches_text() -> None:
    store = LongTerm()
    await store.add("the deployment checklist is owned by Alice", [1.0, 0.0])
    rows = await store.search_text("checklist")
    assert [text for text, _score, _meta in rows] == [
        "the deployment checklist is owned by Alice"
    ]


@pytest.mark.asyncio
async def test_an_embedding_agent_still_recalls_text_only_entries() -> None:
    from localmelo.melo.memory.coordinator import Hippo

    class FakeEmbedding:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _text in texts]

    store = LongTerm()
    await store.apply(
        text="the deployment checklist is owned by Alice",
        kind="semantic",
        source="user",
        admitted_by="user_statement",
        provenance={"source_ids": []},
        task_id="t-a",
        session_id="",
        created_at=1.0,
        embedding=None,
        action="NEW",
        target_id=None,
        reason="new",
        source_ids=[],
        keep_text=False,
        candidate_id="c1",
    )
    hippo = Hippo(embedding=FakeEmbedding(), long=store)  # type: ignore[arg-type]
    rows = await hippo.recall_long("deployment checklist")
    assert [text for text, _score, _meta in rows] == [
        "the deployment checklist is owned by Alice"
    ]
