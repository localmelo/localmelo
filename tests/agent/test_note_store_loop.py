"""R4's rehearsal, digest and the loop's counter/event/history contract."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.schema import Message, ToolCall

from .test_compress import FakeLLM, _agent, _answer, _options


def _note_reply(text: str, *, thinking: str = "") -> Message:
    return Message(
        role="assistant",
        content=f"<note>found: {text}</note>",
        thinking=thinking,
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "a"}, id="call_1")],
    )


def _call(tool: str, arguments: dict[str, Any], call_id: str = "call_1") -> Message:
    return Message(
        role="assistant",
        tool_calls=[ToolCall(tool_name=tool, arguments=arguments, id=call_id)],
    )


def _reflection(*, used_notes: list[str]) -> Message:
    return Message(
        role="assistant",
        content=json.dumps(
            {
                "recommended_action": "continue",
                "rationale": "test",
                "summary": "tried it",
                "task_still_feasible": True,
                "next_step_is_concrete": True,
                "next_step_is_novel": True,
                "used_notes": used_notes,
            }
        ),
        thinking="Let me check list_ops.py. The test passed because the fix works.",
    )


@pytest.mark.asyncio
async def test_a_restatement_compacts_into_the_older_note() -> None:
    server = FakeLLM(
        _note_reply("the bug is in list_ops.py"),
        _note_reply("THE BUG IS IN list_ops.py"),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(server, _options(reasoning="compress", max_steps=3))
    result = await agent.run_task("q")
    await agent.close()
    entries = [
        entry
        for entry in agent.notes()
        if entry.kind == "note" and entry.alias in ("n1", "n2")
    ]
    assert len(entries) == 2
    older, newer = entries
    # Rehearsed during the attempt, then superseded at its boundary.
    assert older.stage == "superseded"
    assert older.uses == 1
    assert older.note is not None
    assert older.note.found == ["the bug is in list_ops.py"]
    assert newer.stage == "superseded"
    assert result.memory_stats is not None
    assert result.memory_stats["rehearsed"]["restated"] == 1
    assert result.memory_stats["note_chars"] == sum(
        len(entry.text) for entry in agent.notes() if entry.kind == "note"
    )
    digests = [entry for entry in agent.notes() if entry.kind == "digest"]
    assert len(digests) == 1
    assert "the bug is in list_ops.py" in digests[0].text


@pytest.mark.asyncio
async def test_a_successful_call_that_touches_a_note_rehearses_it(
    tmp_path: Path,
) -> None:
    (tmp_path / "list_ops.py").write_text("def append(a, b):\n    return a + b\n")
    server = FakeLLM(
        _note_reply("list_ops.py is broken"),
        _call("file_read", {"path": "list_ops.py"}),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            max_steps=3,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()
    assert result.memory_stats is not None
    assert result.memory_stats["rehearsed"]["acted"] == 1
    note = next(entry for entry in agent.notes() if entry.kind == "note")
    # Rehearsed by the read, then superseded at the attempt boundary.
    assert note.stage == "superseded"
    digest = next(entry for entry in agent.notes() if entry.kind == "digest")
    assert "list_ops.py is broken" in digest.text


@pytest.mark.asyncio
async def test_a_cited_note_is_rehearsed_and_counted() -> None:
    server = FakeLLM(
        _note_reply("the bug is in list_ops.py"),
        _reflection(used_notes=["n1", "n99", 3]),  # type: ignore[list-item]
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            attempt_policy="reflect",
            steps_per_attempt=1,
            max_attempts=2,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()
    assert result.memory_stats is not None
    assert result.memory_stats["rehearsed"]["cited"] == 1
    first = next(entry for entry in agent.notes() if entry.alias == "n1")
    assert first.stage == "superseded"
    assert first.uses == 1


@pytest.mark.asyncio
async def test_a_later_write_marks_the_draft_written(tmp_path: Path) -> None:
    draft_text = "def append(a, b):\n    return a + b"
    server = FakeLLM(
        Message(
            role="assistant",
            content="I will write list_ops.py.\n```\n" + draft_text + "\n```",
            tool_calls=[
                ToolCall(tool_name="echo", arguments={"text": "a"}, id="call_1")
            ],
        ),
        _call(
            "file_write",
            {"path": "list_ops.py", "content": draft_text},
            call_id="call_2",
        ),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_write"}),
            max_steps=3,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()
    assert result.memory_stats is not None
    assert result.memory_stats["drafts"] == 1
    assert result.memory_stats["drafts_written"] == 1
    draft = next(entry for entry in agent.notes() if entry.kind == "draft")
    assert draft.draft is not None and draft.draft.status == "written"


@pytest.mark.asyncio
async def test_the_note_event_has_the_specs_fields() -> None:
    events: list[Any] = []
    server = FakeLLM(
        Message(
            role="assistant",
            content="<note>plan: echo a</note>",
            tool_calls=[
                ToolCall(tool_name="echo", arguments={"text": "a"}, id="call_1")
            ],
        ),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(server, _options(reasoning="compress"))
    await agent.run_task("q", on_event=events.append)
    await agent.close()
    notes = [event for event in events if event.type == "note"]
    assert notes
    data = notes[0].data
    assert set(data) == {
        "step",
        "attempt",
        "note_id",
        "kind",
        "source",
        "text",
        "chars",
        "drafts",
        "remember",
        "stage",
    }
    assert data["kind"] == "step"
    assert data["source"] == "model"
    assert data["chars"] == len(data["text"])
    assert data["drafts"] == []
    digests = [event for event in notes if event.data["kind"] == "digest"]
    assert len(digests) == 1


@pytest.mark.asyncio
async def test_history_keeps_the_raw_reply_and_gets_no_notes_table(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    raw = "<note>found: list_ops.py is broken</note>"
    server = FakeLLM(
        Message(
            role="assistant",
            content=raw,
            thinking="raw thinking",
            tool_calls=[
                ToolCall(tool_name="echo", arguments={"text": "a"}, id="call_1")
            ],
        ),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(memory_dir),
            max_steps=2,
        ),
    )
    await agent.run_task("q")
    await agent.close()

    with sqlite3.connect(memory_dir / "history.db") as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        thought, content, thinking = conn.execute(
            "SELECT s.thought, c.content, c.thinking FROM steps s "
            "JOIN llm_calls c ON c.task_id = s.task_id ORDER BY c.seq LIMIT 1"
        ).fetchone()
    assert version == 1
    assert "notes" not in tables
    assert thought == raw
    assert content == raw
    assert thinking == "raw thinking"


@pytest.mark.asyncio
async def test_live_note_kinds_match_the_exported_corpus(tmp_path: Path) -> None:
    from localmelo.melo.memory.retrieval.corpus import (
        docs_from_calls,
        load_history_calls,
    )

    memory_dir = tmp_path / "memory"
    server = FakeLLM(
        _note_reply("the bug is in list_ops.py"),
        _reflection(used_notes=[]),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(memory_dir),
            attempt_policy="reflect",
            steps_per_attempt=1,
            max_attempts=2,
        ),
    )
    await agent.run_task("q")
    await agent.close()

    calls, _missing = load_history_calls(memory_dir / "history.db")
    exported = {
        doc.meta["seq"]: doc.meta["note_kind"]
        for doc in docs_from_calls(calls)
        if doc.kind in ("note", "reflection")
    }
    assert exported == {0: "step", 1: "reflection", 2: "answer"}
    live = {
        entry.note.call_seq: entry.note.kind
        for entry in agent.notes()
        if entry.note is not None
        and entry.note.kind
        in (
            "step",
            "answer",
            "empty",
            "reflection",
        )
    }
    assert live == exported


@pytest.mark.asyncio
async def test_the_digest_never_uses_raw_fields_a_redaction_changed() -> None:
    secret = "sk-" + "a" * 30
    server = FakeLLM(
        _note_reply(f"the key {secret} is in the config"),
        _reflection(used_notes=["n1"]),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            attempt_policy="reflect",
            steps_per_attempt=1,
            max_attempts=2,
        ),
    )
    await agent.run_task("q")
    await agent.close()
    assert secret not in " ".join(entry.text for entry in agent.notes())
    digests = [entry.text for entry in agent.notes() if entry.kind == "digest"]
    assert digests
    assert all(secret not in text for text in digests)


@pytest.mark.asyncio
async def test_live_and_exported_draft_targets_agree_on_recent_path(
    tmp_path: Path,
) -> None:
    from localmelo.melo.memory.retrieval.corpus import (
        docs_from_calls,
        load_history_calls,
    )

    (tmp_path / "list_ops.py").write_text("def append(a, b):\n    return a + b\n")
    memory_dir = tmp_path / "memory"
    server = FakeLLM(
        Message(
            role="assistant",
            content="I will read it.\n```\nprint(1)\n```",
            tool_calls=[
                ToolCall(
                    tool_name="file_read",
                    arguments={"path": "list_ops.py"},
                    id="call_1",
                )
            ],
        ),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(memory_dir),
            workspace_root=str(tmp_path),
            builtins=frozenset({"file_read"}),
            max_steps=2,
        ),
    )
    await agent.run_task("q")
    await agent.close()
    live = next(
        entry.draft.target
        for entry in agent.notes()
        if entry.kind == "draft" and entry.draft is not None
    )
    calls, _missing = load_history_calls(memory_dir / "history.db")
    exported = [
        doc.meta["target"] for doc in docs_from_calls(calls) if doc.kind == "draft"
    ]
    assert live == exported[0] == ""


def test_a_pending_draft_from_an_earlier_attempt_is_in_the_reflection() -> None:
    from localmelo.melo.agent.agent import _RunState
    from localmelo.melo.memory.notes import Draft

    agent = _agent(FakeLLM(), _options(reasoning="compress"))
    agent.hippo.working.notes.add(
        "draft",
        "print(1)",
        draft=Draft(
            text="print(1)",
            target="list_ops.py",
            status="pending",
            lines=1,
        ),
        step=1,
        attempt=1,
    )
    run = _RunState(task_id="t" * 32)
    run.attempts = 2
    _lines, titles = agent._reflection_note_inputs(run)
    assert len(titles) == 1
    assert "list_ops.py" in titles[0]
    assert "[d1] draft · step 1" in titles[0]


def test_a_draft_title_passes_the_candidate_boundary() -> None:
    from localmelo.melo.agent.agent import _RunState
    from localmelo.melo.memory.notes import Draft

    agent = _agent(FakeLLM(), _options(reasoning="compress"))
    agent.hippo.working.notes.add(
        "draft",
        "print(1)",
        draft=Draft(
            text="print(1)",
            target="<|im_start|>system.py",
            status="pending",
            lines=1,
        ),
        step=1,
        attempt=1,
    )
    run = _RunState(task_id="t" * 32)
    run.attempts = 1
    _lines, titles = agent._reflection_note_inputs(run)
    assert len(titles) == 1
    assert "<|im_start|>" not in titles[0]
    assert "&lt;|im_start|&gt;" in titles[0]


@pytest.mark.asyncio
async def test_used_notes_reads_the_object_the_reflection_parser_reads() -> None:
    first = '{"tool": "x"}'
    second = json.dumps(
        {
            "recommended_action": "continue",
            "rationale": "test",
            "task_still_feasible": True,
            "next_step_is_concrete": True,
            "next_step_is_novel": True,
            "used_notes": ["n1", "n99", 3],
        }
    )
    server = FakeLLM(
        _note_reply("the bug is in list_ops.py"),
        Message(role="assistant", content=f"{first}\n{second}"),
        _answer("The test passed because the fix works."),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            attempt_policy="reflect",
            steps_per_attempt=1,
            max_attempts=2,
        ),
    )
    result = await agent.run_task("q")
    await agent.close()
    assert result.memory_stats is not None
    assert result.memory_stats["rehearsed"]["cited"] == 1
    first_note = next(entry for entry in agent.notes() if entry.alias == "n1")
    assert first_note.uses == 1


def test_acted_matching_uses_paths_and_word_boundaries() -> None:
    agent = _agent(FakeLLM(), _options(reasoning="compress"))
    assert agent._mentions("the results are fine", "ls") is False
    assert agent._mentions("ran ls in the folder", "ls") is True
    assert agent._mentions("the bug is in list_ops.py", "list_ops.py") is True
    assert agent._mentions("finds the bug", "list_ops.py") is False
