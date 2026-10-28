"""Admission verifies source records before typed updates."""

from __future__ import annotations

import json
from pathlib import Path

from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.hippo.signals import user_candidates
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.schema import CandidateItem, MemoryCandidate, TaskRecord


def candidate(
    name: str, text: str, evidence: str | None, time: float
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=name,
        source="reasoning_note",
        text=text,
        remember_by="rule",
        task_id="task",
        created_at=time,
        remember=True,
        items=(CandidateItem(text, (name,), evidence=evidence),),
    )


def record(name: str, text: str, timestamp: str = "2026-04-11T16:50:00-04:00") -> str:
    return json.dumps({"id": name, "text": text, "timestamp": timestamp})


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


async def test_verified_typed_updates_and_unverified_abstention() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    old = "Mina lives in Paris."
    update = "Mina now lives in Rome."
    old_id = None
    for name, text, timestamp in [("a", old, 10), ("b", update, 20)]:
        hippo.propose(candidate(name, text, record(name, text), timestamp))
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
            agent.hippo.propose(candidate(name, text, record(name, text), observed))
            assert (await agent.hippo.admit(name, "rehearsal"))[0].action == "NEW"
        _, block, _ = await agent._run_recall(
            "start", "What time are the doors?", _RunState(task_id="reader")
        )
        assert block.index(later) < block.index(first)
        assert "unlinked assertion" in block and "effective 2026-04-11" in block
    finally:
        await agent.close()


async def test_t4_unlinked_evidence_is_kept_without_false_link() -> None:
    store = LongTerm()
    hippo = Hippo(long=store, typed_assertions=True)
    old = "Static Bloom concert at the Paradise Lot on Saturday 2026-04-11, doors at 20:00."
    new = "Doors pushed to 9 tonight, the opener cancelled!"
    for name, text, observed in [("F1", old, 10), ("F3", new, 20)]:
        hippo.propose(candidate(name, text, record(name, text), observed))
        assert (await hippo.admit(name, "rehearsal"))[0].action == "NEW"
    entries = await store.active_entries()
    assert len(entries) == 2
    latest = next(entry for entry in entries if entry.text == new)
    assert latest.provenance["assertion"]["subject_key"] is None
    assert latest.provenance["assertion"]["effective_date"] == "2026-04-11"
    assert latest.provenance["assertion"]["value_text"] == "21:00"
