"""R4's rehearsal, digest and the loop's counter/event/history contract."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.schema import Message, ToolCall, ToolDef, ToolResult
from localmelo.tests.legacy_options import LEGACY_CACHE_OPTIONS

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
        _note_reply("THE BUG IS IN list_ops.py\ndecided: use the fix"),
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
    assert older.note.decided == ["use the fix"]
    assert newer.stage == "superseded"
    assert result.memory_stats is not None
    assert result.memory_stats["rehearsed"]["restated"] == 1
    # Per run: the restatement grew n1 after the run counted its first text.
    growth = len("\ndecided: use the fix")
    assert result.memory_stats["note_chars"] == (
        sum(len(entry.text) for entry in agent.notes() if entry.kind == "note") - growth
    )
    digests = [entry for entry in agent.notes() if entry.kind == "digest"]
    assert len(digests) == 1
    assert "the bug is in list_ops.py" in digests[0].text


@pytest.mark.asyncio
async def test_a_restatement_merges_the_new_ids_and_drops_its_candidate(
    tmp_path: Path,
) -> None:
    # Review finding 8: the merged note keeps the new note's records, and the
    # superseded note's candidate cannot be admitted on its own.
    first = Message(
        role="assistant",
        content="<note>found: the bug is in list_ops.py</note>",
        tool_calls=[
            ToolCall(
                tool_name="raw",
                arguments={"text": '{"id": "r1", "name": "Alice"}'},
                id="call_1",
            )
        ],
    )
    second = Message(
        role="assistant",
        content=(
            "<note>found: THE BUG IS IN list_ops.py\n"
            "decided: the team should use the fix after the review</note>"
        ),
        tool_calls=[
            ToolCall(
                tool_name="raw",
                arguments={"text": '{"id": "r2", "name": "Bob"}'},
                id="call_1",
            )
        ],
    )
    server = FakeLLM(first, second, _answer("The test passed because the fix works."))
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            memory_dir=str(tmp_path / "memory"),
            max_steps=3,
        ),
    )

    async def _raw(text: str) -> str:
        return text

    agent.hippo.register_tool(
        ToolDef(
            name="raw",
            description="return text unchanged",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    )
    agent.executor.register("raw", _raw)
    try:
        await agent.run_task("q")
        entries = [
            entry
            for entry in agent.notes()
            if entry.kind == "note" and entry.alias in ("n1", "n2")
        ]
        older, newer = entries
        assert older.note is not None and newer.note is not None
        assert older.note.source_ids == ["r1", "r2"]
        assert {record_id for record_id, _text in older.note.source_records} == {
            "r1",
            "r2",
        }
        staged = {candidate.candidate_id for candidate in agent.hippo.staged.all()}
        assert older.note.note_id in staged
        assert newer.note.note_id not in staged
        # Both merged items are distinct traces; neither supersedes the other.
        active = await agent.hippo.long.active_entries("episodic")
        assert {entry.text for entry in active} == {
            "the bug is in list_ops.py",
            "the team should use the fix after the review",
        }
    finally:
        await agent.close()


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
async def test_an_attempts_thought_chunks_are_superseded_with_its_notes() -> None:
    # A later attempt's boundary supersedes the earlier attempt's
    # thinking chunks as well as its notes, so a recalled chunk is marked.
    server = FakeLLM(
        _note_reply(
            "the bug is in list_ops.py",
            thinking="First attempt: read list_ops.py to find the bug.",
        ),
        _reflection(used_notes=["n1"]),
        _note_reply(
            "the bug is in list_ops.py",
            thinking="Second attempt: reread list_ops.py after the reflection.",
        ),
        _reflection(used_notes=["n2"]),
    )
    agent = _agent(
        server,
        _options(
            reasoning="compress",
            attempt_policy="reflect",
            steps_per_attempt=1,
            max_attempts=2,
            recall_kinds=frozenset({"note", "draft", "thought"}),
        ),
    )
    await agent.run_task("q")
    await agent.close()
    thoughts = [entry for entry in agent.notes() if entry.kind == "thought"]
    assert [entry.alias for entry in thoughts] == ["t1", "t2"]
    assert [entry.attempt for entry in thoughts] == [1, 2]
    assert [entry.stage for entry in thoughts] == ["superseded", "superseded"]
    notes = [entry for entry in agent.notes() if entry.kind == "note"]
    assert notes and all(entry.stage == "superseded" for entry in notes)


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
    assert version == 2
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
async def test_a_thinking_less_reflection_gets_a_note_from_its_summary() -> None:
    server = FakeLLM(
        _note_reply("the bug is in list_ops.py"),
        Message(
            role="assistant",
            content=json.dumps(
                {
                    "recommended_action": "continue",
                    "rationale": "the test still fails",
                    "summary": "the fix works",
                    "task_still_feasible": True,
                    "next_step_is_concrete": True,
                    "next_step_is_novel": True,
                    "used_notes": [],
                }
            ),
        ),
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
    assert result.memory_stats["candidates_refused"] == 0
    reflection = next(
        entry
        for entry in agent.notes()
        if entry.note is not None and entry.note.kind == "reflection"
    )
    assert reflection.text == "found: the fix works"


@pytest.mark.asyncio
async def test_every_document_crosses_the_boundary_once() -> None:
    reply = Message(
        role="assistant",
        content=(
            "I will write list_ops.py.\n```\nprint(1)\n```\n"
            "<note>found: list_ops.py holds it</note>"
        ),
        tool_calls=[ToolCall(tool_name="echo", arguments={"text": "a"}, id="call_1")],
    )
    server = FakeLLM(
        reply,
        _reflection(used_notes=[]),
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
    calls: list[tuple[str, str]] = []
    original = agent.checker.check_memory_candidate

    def spy(payload: Any) -> Any:
        calls.append((payload.source, payload.text))
        return original(payload)

    agent.checker.check_memory_candidate = spy  # type: ignore[method-assign]
    result = await agent.run_task("q")
    await agent.close()
    assert result.memory_stats is not None
    assert result.memory_stats["candidates"] == len(calls)
    assert result.memory_stats["candidates_refused"] == 0
    by_source: dict[str, list[str]] = {}
    for source, text in calls:
        by_source.setdefault(source, []).append(text)
    drafts = [entry for entry in agent.notes() if entry.kind == "draft"]
    assert len(drafts) == 1
    assert by_source["draft"].count(drafts[0].text) == 1
    # The second draft document is the pinned block, not a second copy.
    assert any(
        text != drafts[0].text and drafts[0].text in text for text in by_source["draft"]
    )
    for entry in (item for item in agent.notes() if item.kind == "note"):
        source = (
            "reflection"
            if entry.note is not None and entry.note.kind == "reflection"
            else "reasoning_note"
        )
        assert by_source[source].count(entry.text) == 1
    for entry in (item for item in agent.notes() if item.kind == "digest"):
        assert by_source["digest"].count(entry.text) == 1


@pytest.mark.asyncio
async def test_the_reflection_block_keeps_twelve_notes_of_four_hundred_chars() -> None:
    from localmelo.melo.agent.agent import (
        REFLECTION_MAX_NOTES,
        REFLECTION_NOTE_CHARS,
        _RunState,
    )
    from localmelo.melo.memory.notes import ReasoningNote

    agent = _agent(FakeLLM(), _options(reasoning="compress"))
    for index in range(REFLECTION_MAX_NOTES + 3):
        agent.hippo.working.notes.add(
            "note",
            "n" * 500,
            note=ReasoningNote(plan=f"note {index}"),
            attempt=1,
        )
    lines, _titles = agent._reflection_note_inputs(_RunState(attempts=1))
    await agent.close()
    assert len(lines) == REFLECTION_MAX_NOTES
    assert all(line.endswith("n" * REFLECTION_NOTE_CHARS) for line in lines)


@pytest.mark.asyncio
async def test_a_reflection_refused_for_its_notes_block_sends_nothing() -> None:
    from localmelo.melo.agent.agent import _RunState
    from localmelo.melo.memory.notes import NOTES_START, ReasoningNote
    from localmelo.melo.schema import CheckResult, TaskRecord

    server = FakeLLM()
    agent = _agent(server, _options(reasoning="compress", **LEGACY_CACHE_OPTIONS))
    agent.hippo.working.notes.add(
        "note",
        "found: a fact",
        note=ReasoningNote(found=["a fact"]),
        attempt=1,
    )
    seen: list[bool] = []
    original = agent.checker.pre_plan

    async def pre_plan(messages: Any, **kwargs: Any) -> CheckResult:
        has_notes = any(NOTES_START in message.content for message in messages)
        seen.append(has_notes)
        if has_notes:
            return CheckResult(allowed=False, reason="Prompt too large: 100001 chars")
        return await original(messages, **kwargs)

    agent.checker.pre_plan = pre_plan  # type: ignore[method-assign]
    result = await agent._do_reflect(
        TaskRecord(query="q"),
        attempt_id=1,
        failure_type="",
        run=_RunState(attempts=1),
    )
    await agent.close()
    assert result is None
    assert seen == [True]
    assert server.requests == []


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


@pytest.mark.asyncio
async def test_live_and_exported_draft_targets_use_the_recent_path(
    tmp_path: Path,
) -> None:
    from localmelo.melo.memory.retrieval.corpus import (
        docs_from_calls,
        load_history_calls,
    )

    (tmp_path / "list_ops.py").write_text("def append(a, b):\n    return a + b\n")
    (tmp_path / "README.md").write_text("readme\n")
    memory_dir = tmp_path / "memory"
    server = FakeLLM(
        Message(
            role="assistant",
            content="I will read it.",
            tool_calls=[
                ToolCall(
                    tool_name="file_read",
                    arguments={"path": "list_ops.py"},
                    id="call_1",
                )
            ],
        ),
        Message(
            role="assistant",
            content="Now the code.\n```\nprint(2)\n```",
            tool_calls=[
                ToolCall(
                    tool_name="file_read",
                    arguments={"path": "README.md"},
                    id="call_2",
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
            max_steps=3,
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
    assert live == exported[0] == "list_ops.py"


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


def test_a_shell_success_needs_a_known_zero_exit() -> None:
    agent = _agent(FakeLLM(), _options(reasoning="compress"))
    call = ToolCall(tool_name="shell_exec", arguments={"command": "false"})
    empty = ToolResult(tool_name="shell_exec", output="")
    assert agent._call_succeeded(call, empty, []) is False
    assert agent._call_succeeded(call, empty, [1]) is False
    assert agent._call_succeeded(call, empty, [0]) is True
    failed = ToolResult(tool_name="shell_exec", output="[exit code 3]\nboom")
    assert agent._call_succeeded(call, failed, [3]) is False
    file_call = ToolCall(tool_name="file_read", arguments={"path": "a.py"})
    assert (
        agent._call_succeeded(
            file_call, ToolResult(tool_name="file_read", output="x"), []
        )
        is True
    )
