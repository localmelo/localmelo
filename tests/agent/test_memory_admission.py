"""R6-lite's admission through the agent loop: signals, flush, stats, events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.long.sqlite import (
    SqliteLongTerm,
    read_export_decisions,
    read_export_entries,
)
from localmelo.melo.schema import Message, ToolCall

from .test_compress import FakeLLM, _agent, _answer, _options

CALL_ID = "call_1"


def _note_reply(text: str) -> Message:
    return Message(
        role="assistant",
        content=f"<note>found: {text}</note>",
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "a"}, id=CALL_ID)],
    )


def _call(tool: str, arguments: dict[str, Any], call_id: str = CALL_ID) -> Message:
    return Message(
        role="assistant",
        tool_calls=[ToolCall(tool_name=tool, arguments=arguments, id=call_id)],
    )


async def _entries(directory: Path) -> list[dict[str, Any]]:
    path = directory / "memory" / "long_term.db"
    if not path.is_file():
        return []
    return read_export_entries(str(path))


@pytest.mark.asyncio
async def test_a_rehearsed_note_is_written_once_through_the_loop(
    tmp_path: Path,
) -> None:
    (tmp_path / "list_ops.py").write_text("def append(a, b):\n    return a + b\n")
    server = FakeLLM(
        _note_reply("list_ops.py returns the wrong list from its append helper"),
        _call("file_read", {"path": "list_ops.py"}),
        _answer("done"),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            max_steps=3,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()

    assert result.memory_stats is not None
    assert result.memory_stats["admitted"]["rehearsal"] == 1
    assert result.memory_stats["written"]["episodic"] == 1
    entries = await _entries(tmp_path)
    assert [entry["text"] for entry in entries] == [
        "list_ops.py returns the wrong list from its append helper"
    ]
    assert entries[0]["admitted_by"] == "rehearsal"
    decisions = read_export_decisions(str(tmp_path / "memory" / "long_term.db"))
    assert len(decisions) == 1
    assert decisions[0]["action"] == "NEW"


@pytest.mark.asyncio
async def test_a_request_that_states_no_fact_admits_notes_and_the_digest(
    tmp_path: Path,
) -> None:
    server = FakeLLM(
        _note_reply("the migration uses PostgreSQL 16 for the analytics store"),
        _answer("done"),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            max_steps=2,
        ),
    )
    result = await agent.run_task("Remember this for the next migration run, please.")
    await agent.close()

    assert result.memory_stats is not None
    assert result.memory_stats["admitted"]["user_request"] >= 1
    entries = await _entries(tmp_path)
    assert len(entries) == 2
    assert {entry["source"] for entry in entries} == {"reasoning_note", "digest"}
    assert {entry["admitted_by"] for entry in entries} == {"user_request"}
    assert result.memory_stats["written"]["reflection"] == 0


@pytest.mark.asyncio
async def test_a_stated_fact_is_admitted_at_the_start_without_a_model_call(
    tmp_path: Path,
) -> None:
    server = FakeLLM(_answer("noted"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            max_steps=1,
        ),
    )
    result = await agent.run_task(
        "Please remember that my birthday is 12 May 1990 and I like green tea."
    )
    await agent.close()

    entries = await _entries(tmp_path)
    assert [entry["text"] for entry in entries] == [
        "Please remember that my birthday is 12 May 1990 and I like green tea."
    ]
    assert entries[0]["kind"] == "profile"
    assert entries[0]["admitted_by"] == "user_statement"
    assert entries[0]["provenance"]["source_ids"] == []
    assert result.memory_stats is not None
    assert result.memory_stats["admitted"]["user_statement"] == 1


@pytest.mark.asyncio
async def test_admit_on_outcome_writes_the_digest_with_its_terminal_status(
    tmp_path: Path,
) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            max_steps=1,
            admit_on_outcome=True,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()

    assert result.status == "completed"
    entries = await _entries(tmp_path)
    assert [entry["source"] for entry in entries] == ["digest"]
    assert entries[0]["admitted_by"] == "outcome"
    assert entries[0]["provenance"]["final_status"] == "completed"
    assert result.memory_stats is not None
    assert result.memory_stats["admitted"]["outcome"] == 1


@pytest.mark.asyncio
async def test_the_memory_event_has_its_five_fields(tmp_path: Path) -> None:
    events: list[tuple[str, Any]] = []

    def on_event(event: AgentEvent) -> None:
        if event.type == "memory":
            events.append((event.type, event.data))

    (tmp_path / "list_ops.py").write_text("def append(a, b):\n    return a + b\n")
    server = FakeLLM(
        _note_reply("list_ops.py returns the wrong list from its append helper"),
        _call("file_read", {"path": "list_ops.py"}),
        _answer("done"),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            max_steps=3,
        ),
    )
    await agent.run_task("q", on_event=on_event)
    await agent.close()
    assert len(events) == 1
    assert set(events[0][1]) == {
        "candidate_id",
        "action",
        "kind",
        "admitted_by",
        "entry_id",
    }


@pytest.mark.asyncio
async def test_keep_decision_text_stores_the_checked_text(tmp_path: Path) -> None:
    # T53R finding 10: no end-to-end test covered --keep-decision-text.
    (tmp_path / "list_ops.py").write_text("def append(a, b):\n    return a + b\n")
    server = FakeLLM(
        _note_reply("list_ops.py returns the wrong list from its append helper"),
        _call("file_read", {"path": "list_ops.py"}),
        _answer("done"),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            max_steps=3,
            keep_decision_text=True,
        ),
    )
    await agent.run_task("q")
    await agent.close()
    entries = await _entries(tmp_path)
    decisions = read_export_decisions(str(tmp_path / "memory" / "long_term.db"))
    assert decisions[0]["text"] == entries[0]["text"]
    assert decisions[0]["text_sha256"]


@pytest.mark.asyncio
async def test_a_meta_sentence_makes_no_profile_entry(tmp_path: Path) -> None:
    # T53R finding 6, end to end: a task-shaped prompt is not a fact.
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            max_steps=1,
        ),
    )
    await agent.run_task("My task is to fix the build. My question is how do I deploy?")
    await agent.close()
    entries = await _entries(tmp_path)
    assert all(entry["kind"] != "profile" for entry in entries)
    assert all(entry["admitted_by"] != "user_statement" for entry in entries)


@pytest.mark.asyncio
async def test_a_tool_result_never_becomes_a_user_fact(tmp_path: Path) -> None:
    (tmp_path / "profile.json").write_text(
        json.dumps({"record_id": "r1", "note": "my birthday is 12 May 1990"}) + "\n"
    )
    server = FakeLLM(
        _note_reply("profile.json says my birthday is 12 May 1990 and I like tea"),
        _call("file_read", {"path": "profile.json"}),
        _answer("done"),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            max_steps=3,
        ),
    )
    await agent.run_task("read the profile file")
    await agent.close()
    entries = await _entries(tmp_path)
    assert entries
    assert all(entry["kind"] != "profile" for entry in entries)


@pytest.mark.asyncio
async def test_an_aider_shaped_query_makes_no_profile_entry(tmp_path: Path) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            max_steps=1,
        ),
    )
    await agent.run_task(
        "Fix the failing test in list_ops.py and run pytest until it passes"
    )
    await agent.close()
    assert not (tmp_path / "memory" / "long_term.db").exists() or all(
        entry["kind"] != "profile" for entry in await _entries(tmp_path)
    )


@pytest.mark.asyncio
async def test_a_secret_in_a_note_item_is_checked_before_it_is_staged(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.py").write_text("def append(a, b):\n    return a + b\n")
    key = "sk-" + "a" * 24
    server = FakeLLM(
        _note_reply(f"notes.py holds the key {key} in its append helper"),
        _call("file_read", {"path": "notes.py"}),
        _answer("done"),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            max_steps=3,
        ),
    )
    await agent.run_task("q")
    staged = agent.hippo.staged.all()
    await agent.close()
    assert staged
    for candidate in staged:
        assert key not in candidate.text
        for item in candidate.items:
            assert key not in item.text


@pytest.mark.asyncio
async def test_the_store_can_be_reopened_after_the_run(tmp_path: Path) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            max_steps=1,
            admit_on_outcome=True,
        ),
    )
    await agent.run_task("q")
    await agent.close()

    store = SqliteLongTerm(tmp_path / "memory" / "long_term.db")
    try:
        assert await store.count_entries() == 1
        assert len(await store.decisions()) == 1
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_memory_false_still_writes_and_recalls_nothing(tmp_path: Path) -> None:
    server = FakeLLM(_answer("done"))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            memory=False,
            max_steps=1,
        ),
    )
    await agent.run_task("Remember my birthday is 12 May 1990 and I like tea.")
    await agent.close()
    assert not (tmp_path / "memory" / "long_term.db").exists()


@pytest.mark.asyncio
async def test_twenty_runs_through_the_agent_keep_the_growth_bound(
    tmp_path: Path,
) -> None:
    # T53R finding 10: the stability slice opened the store directly, never
    # through 20 Agent sessions. The spec's D11 bound: at most 3 entries per
    # run on average, and no run more than 8.
    memory = tmp_path / "memory"
    counts: list[int] = []
    for run in range(20):
        server = FakeLLM(_answer("done"))
        agent = _agent(
            server,
            _options(
                reasoning="compress",
                memory_dir=str(memory),
                max_steps=1,
            ),
        )
        await agent.run_task(f"Remember that my project codename is Bluebird{run}.")
        await agent.close()
        counts.append(len(await _entries(tmp_path)))
    deltas = [counts[0], *(counts[i] - counts[i - 1] for i in range(1, len(counts)))]
    assert max(deltas) <= 8
    assert sum(deltas) / len(deltas) <= 3
