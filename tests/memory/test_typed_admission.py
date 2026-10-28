"""Admission verifies source records before typed updates."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.hippo.signals import user_candidates
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.memory.recall import docs_from_entries
from localmelo.melo.schema import CandidateItem, MemoryCandidate, TaskRecord


def candidate(
    name: str, text: str, evidence: str | None, time: float, task: str = "task"
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=name,
        source="reasoning_note",
        text=text,
        remember_by="rule",
        task_id=task,
        created_at=time,
        remember=True,
        items=(CandidateItem(text, (name,), evidence=evidence),),
    )


def record(
    name: str,
    text: str,
    timestamp: str = "2026-04-11T16:50:00-04:00",
    field: str = "timestamp",
) -> str:
    return json.dumps({"id": name, "text": text, field: timestamp})


def test_short_user_facts_are_opt_in_and_not_questions() -> None:
    text = "Mina lives in Paris. Does Mina live in Rome?"
    default = user_candidates(text, task_id="task")
    opted_in = user_candidates(text, task_id="task", typed_assertions=True)
    assert not any(item.text == "Mina lives in Paris." for item in default)
    assert [item.text for item in opted_in] == ["Mina lives in Paris."]


async def test_short_user_fact_requires_matching_saved_task() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    await hippo.save_task(TaskRecord(query="Mina lives in Paris.", task_id="task"))
    item = user_candidates(
        "Mina lives in Paris.", task_id="task", typed_assertions=True
    )[0]
    hippo.propose(item)
    decision = (await hippo.admit(item.candidate_id, "user_statement"))[0]
    assert decision.action == "NEW"
    assert (await store.active_entries())[0].provenance["assertion"][
        "polarity"
    ] == "positive"


async def test_agent_admits_short_user_fact_to_ram() -> None:
    from localmelo.melo.schema import Message
    from localmelo.tests.agent.test_recall_loop import FakeLLM, _agent, _options

    agent = _agent(
        FakeLLM(Message("assistant", "Noted.")),
        _options(reasoning="compress", retriever="bm25", typed_assertions=True),
    )
    try:
        result = await agent.run_task("Mina lives in Paris.")
        assert result.status == "completed"
        entries = await agent.hippo.long.active_entries()
        assert any(entry.text == "Mina lives in Paris." for entry in entries)
    finally:
        await agent.close()


async def test_agent_user_correction_supersedes_in_message_order() -> None:
    from localmelo.melo.schema import Message
    from localmelo.tests.agent.test_recall_loop import FakeLLM, _agent, _options

    agent = _agent(
        FakeLLM(Message("assistant", "Noted.")),
        _options(reasoning="compress", retriever="bm25", typed_assertions=True),
    )
    try:
        await agent.run_task("Mina lives in Paris. Mina now lives in Rome.")
        active = await agent.hippo.long.active_entries()
        assert [entry.text for entry in active] == ["Mina now lives in Rome."]
        decisions = [row["action"] for row in await agent.hippo.long.decisions()]
        assert decisions == ["NEW", "UPDATE"]
    finally:
        await agent.close()


@pytest.mark.parametrize("sqlite", [False, True])
async def test_the_gate_drops_only_an_exact_duplicate(
    tmp_path: Path, sqlite: bool
) -> None:
    store: LongTerm = SqliteLongTerm(tmp_path / "long.db") if sqlite else LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    exact = "Quarterly planning review is scheduled for 2026-05-04.\nfiles: plan.md"
    shared = [f"token{index}" for index in range(9)]
    try:
        for name in ("a", "b"):
            hippo.propose(MemoryCandidate(name, "digest", exact, "rule", "task"))
            decision = (await hippo.admit(name, "rehearsal"))[0]
        assert decision.action == "IGNORE"
        assert len(await store.active_entries()) == 1
        actions = []
        near = (
            " ".join([*shared, "alpha", "\nfiles: plan.md"]),
            " ".join([*shared, "beta", "\nfiles: plan.md"]),
        )
        for name, text in zip(("c", "d"), near, strict=True):
            hippo.propose(MemoryCandidate(name, "digest", text, "rule", "task"))
            actions.append((await hippo.admit(name, "rehearsal"))[0].action)
        assert actions == ["NEW", "NEW"]
        assert len(await store.active_entries()) == 3
    finally:
        if isinstance(store, SqliteLongTerm):
            await store.aclose()


@pytest.mark.parametrize("sqlite", [False, True])
async def test_a_same_key_digest_competes_with_its_task_digest(
    tmp_path: Path, sqlite: bool
) -> None:
    store: LongTerm = SqliteLongTerm(tmp_path / "long.db") if sqlite else LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    exact = "Quarterly planning review moved to 2026-06-01.\nfiles: plan.md"
    key = "same-task-key"
    try:
        for name in ("a", "b"):
            hippo.propose(
                MemoryCandidate(
                    name,
                    "digest",
                    exact,
                    "rule",
                    "task",
                    provenance={"task_key": key},
                )
            )
            decision = (await hippo.admit(name, "rehearsal"))[0]
        assert decision.action == "IGNORE"
        assert len(await store.active_entries()) == 1
    finally:
        if isinstance(store, SqliteLongTerm):
            await store.aclose()


async def test_date_field_is_recorded_as_source_time() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    text = "Doors moved to 7 tonight, the support act cancelled!"
    stamp = "2026-05-02T19:10:00-04:00"
    hippo.propose(candidate("a", text, record("a", text, stamp, "date"), 10))
    assert (await hippo.admit("a", "rehearsal"))[0].action == "NEW"
    assertion = (await store.active_entries())[0].provenance["assertion"]
    assert assertion["effective_date"] == "2026-05-02"
    assert assertion["observed_at"] == datetime.fromisoformat(stamp).timestamp()
    assert assertion["observed_date"] == "2026-05-02"


async def test_recall_title_uses_the_source_date_not_the_machine_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from datetime import date

    from localmelo.melo.memory.assertions import parse_assertion
    from localmelo.melo.memory.recall import docs_from_entries

    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    try:
        text = "Doors moved to 7 tonight, the support act cancelled!"
        assertion = parse_assertion(text, source_date=date(2026, 5, 2))
        assert assertion is not None
        observed = datetime.fromisoformat("2026-05-02T23:30:00-04:00").timestamp()
        assertion_data = {
            **assertion.to_dict(),
            "observed_at": observed,
            "observed_date": "2026-05-02",
        }
        meta = {
            "id": 1,
            "kind": "semantic",
            "source": "reasoning_note",
            "status": "active",
            "task_id": "task",
            "created_at": observed,
            "provenance": {"assertion": assertion_data},
        }
        docs = docs_from_entries([(text, 1.0, meta)], include_history=True)
        assert docs[0].title.startswith("2026-05-02")
    finally:
        monkeypatch.undo()
        time.tzset()


async def test_verified_typed_updates_and_unverified_abstention() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    old = "Mina lives in Paris."
    update = "Mina now lives in Rome."
    old_id = None
    for name, text, timestamp in [("a", old, 10), ("b", update, 20)]:
        source_time = f"2026-04-11T16:{timestamp:02}:00-04:00"
        hippo.propose(candidate(name, text, record(name, text, source_time), timestamp))
        decisions = await hippo.admit(name, "rehearsal")
        assert decisions[0].action == ("NEW" if name == "a" else "UPDATE")
        if name == "a":
            old_id = decisions[0].entry_id
            assert old_id is not None
        else:
            assert decisions[0].target_id == old_id
            assert decisions[0].reason == (await store.decisions())[-1]["reason"]
    active = await store.active_entries()
    assert len(active) == 1 and active[0].text == update
    assert active[0].provenance["assertion"]["original"] == update
    forged = "Mina now lives in Oslo."
    hippo.propose(candidate("c", forged, record("c", "Someone claimed " + forged), 30))
    decisions = await hippo.admit("c", "rehearsal")
    assert decisions[0].action != "UPDATE"
    assert any(entry.text == update for entry in await store.active_entries())


async def test_late_ingestion_does_not_order_older_source_evidence() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    old = "Mina lives in Paris."
    update = "Mina now lives in Rome."
    hippo.propose(candidate("a", old, record("a", old, "2026-04-11T16:30:00Z"), 10))
    assert (await hippo.admit("a", "rehearsal"))[0].action == "NEW"
    hippo.propose(
        candidate("b", update, record("b", update, "2026-04-11T16:00:00Z"), 20)
    )
    assert (await hippo.admit("b", "rehearsal"))[0].action == "CONFLICT"
    assert len(await store.active_entries()) == 2


async def test_t4_reader_sees_newer_unlinked_doors_first(tmp_path: Path) -> None:
    from localmelo.melo.agent.agent import _RunState
    from localmelo.tests.agent.test_recall_loop import FakeLLM, _agent, _options

    agent = _agent(
        FakeLLM(),
        _options(
            reasoning="compress",
            retriever="bm25",
            typed_assertions=True,
            memory_dir=str(tmp_path),
        ),
    )
    first = "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11, doors at 20:00."
    later = "Doors pushed to 9 tonight, the opener cancelled!"
    try:
        for name, text, observed in [("F1", first, 10), ("F3", later, 20)]:
            stamp = f"2026-04-11T16:{observed:02}:00-04:00"
            agent.hippo.propose(
                candidate(name, text, record(name, text, stamp), observed)
            )
            assert (await agent.hippo.admit(name, "rehearsal"))[0].action == "NEW"
        _, block, _ = await agent._run_recall(
            "start", "What time are the doors?", _RunState(task_id="reader")
        )
        assert block.index(later) < block.index(first)
        assert "unlinked assertion" in block and "effective 2026-04-11" in block
    finally:
        await agent.close()


async def test_user_and_source_entries_keep_wall_time_for_recall_order() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    await hippo.save_task(TaskRecord(query="Mina lives in Paris.", task_id="facts"))
    user_text = "Mina lives in Paris."
    hippo.propose(
        MemoryCandidate(
            "u",
            "user",
            user_text,
            "rule",
            "facts",
            created_at=1_790_000_000.0,
            remember=True,
        )
    )
    assert (await hippo.admit("u", "user_statement"))[0].action == "NEW"
    source_text = "Mina moved to Rome."
    hippo.propose(
        candidate(
            "f1",
            source_text,
            record("f1", source_text, "2026-05-02T19:10:00-04:00", "date"),
            1_790_000_100.0,
            "facts",
        )
    )
    assert (await hippo.admit("f1", "rehearsal"))[0].action == "CONFLICT"
    rows = await store.search_text("Mina", top_k=10, include_history=True)
    docs = docs_from_entries(rows, include_history=True)
    times = {doc.text: doc.meta["assertion_order_time"] for doc in docs}
    assert times[user_text] > times[source_text]


@pytest.mark.parametrize("field", ["timestamp", "date"])
async def test_t4_unlinked_evidence_is_kept_without_false_link(field: str) -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    old = "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11, doors at 20:00."
    new = "Doors pushed to 9 tonight, the opener cancelled!"
    for name, text, observed in [("F1", old, 10), ("F3", new, 20)]:
        hippo.propose(candidate(name, text, record(name, text, field=field), observed))
        assert (await hippo.admit(name, "rehearsal"))[0].action == "NEW"
    entries = await store.active_entries()
    assert len(entries) == 2
    latest = next(entry for entry in entries if entry.text == new)
    assert latest.provenance["assertion"]["subject_key"] is None
    assert latest.provenance["assertion"]["effective_date"] == "2026-04-11"
    assert latest.provenance["assertion"]["value_text"] == "21:00"
