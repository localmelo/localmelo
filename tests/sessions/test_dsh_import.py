from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from localmelo.support.sessions.dsh_import import import_dsh
from localmelo.support.sessions.store import SessionStore


def write_log(path: Path, records: list[dict], *, invalid: bool = False) -> None:
    text = "\n".join(json.dumps(record) for record in records) + "\n"
    if invalid:
        text += "not json\n"
    path.write_text(text)


@pytest.mark.asyncio
async def test_import_mapping_idempotence_redaction_and_retention(
    tmp_path: Path,
) -> None:
    now = [100.0]
    store = SessionStore(
        tmp_path / "sessions.db",
        writer="cli",
        clock=lambda: now[0],
        thinking_retention_days=1,
    )
    await store.open()
    second_store = SessionStore(
        tmp_path / "sessions.db",
        writer="cli",
        clock=lambda: now[0],
        thinking_retention_days=1,
    )
    await second_store.open()
    key = "sk-" + "a" * 30
    path = tmp_path / "log.jsonl"
    write_log(
        path,
        [
            {"type": "thinking", "text": "private reasoning " + key},
            {
                "type": "tool_call",
                "callId": "c1",
                "tool": "read",
                "input": {"path": "notes.json"},
            },
            {
                "type": "tool_result",
                "callId": "c1",
                "status": "completed",
                "result": "output " + key,
            },
            {
                "type": "tool_call",
                "callId": "c2",
                "tool": "read",
                "input": {"path": ".env"},
            },
            {
                "type": "tool_result",
                "callId": "c2",
                "status": "completed",
                "result": "ordinary-looking private value",
            },
            {
                "type": "some_step_record",
                "step": 1,
                "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
            },
            {"type": "thinking", "text": "second thought"},
            {
                "type": "some_step_record",
                "step": 2,
                "usage": {"inputTokens": 20, "outputTokens": 5, "totalTokens": 25},
            },
            {"type": "text", "text": "answer"},
            {"type": "final", "text": "answer"},
        ],
        invalid=True,
    )
    first, second = await asyncio.gather(
        import_dsh(path, title="Review", store=store),
        import_dsh(path, title="Other", store=second_store),
    )
    assert first.session_id == second.session_id
    assert sorted([first.imported, second.imported]) == [False, True]
    assert first.skipped == second.skipped == 3
    info = await store.get_session(first.session_id)
    assert info is not None and info.kind == "worker"
    assert info.meta["source"] == "dsh"
    assert info.meta["tokens"] == 37
    assert info.meta["usage"]["completion_tokens"] == 7
    turns = await store.list_turns(first.session_id)
    assert len(turns) == 1 and turns[0].status == "completed"
    assert turns[0].answer == "answer"
    assert turns[0].result is not None
    assert turns[0].result["usage"]["completion_tokens"] == 7
    events = await store.list_events(first.session_id, limit=1000)
    data = [e.event for e in events]
    serialized = json.dumps(data)
    assert key not in serialized and "ordinary-looking private value" not in serialized
    starts = [e["data"] for e in data if e["type"] == "tool.started"]
    ends = [e["data"] for e in data if e["type"] == "tool.finished"]
    assert (
        [e["call_id"] for e in starts] == [e["call_id"] for e in ends] == ["c1", "c2"]
    )
    assert ends[0]["name"] == "read" and ends[0]["status"] == "success"
    assert ends[0]["output"] == "output [redacted]"
    assert ".env" in ends[1]["output"]
    messages = [e["data"] for e in data if e["type"] == "assistant.message"]
    assert sum(e["usage"].get("completion_tokens", 0) for e in messages) == 7
    assert any(e["thinking"] for e in messages)
    if shutil.which("node"):
        reducer = Path(__file__).parents[2] / "support/gateway/static/turns.mjs"
        script = """
const {newView, applyEvent} = await import(process.argv[1]);
let input = ''; for await (const chunk of process.stdin) input += chunk;
const view = newView('test');
for (const event of JSON.parse(input)) applyEvent(view, event);
const turn = view.turns[0];
if (turn.status !== 'completed' || turn.answer !== 'answer' || turn.outputTokens !== 7 ||
    turn.items.filter(item => item.kind === 'tool' && item.status === 'success').length !== 2 ||
    !turn.items.some(item => item.kind === 'thinking')) process.exit(1);
"""
        frames = [
            {**event.event, "id": event.id, "turn": event.turn} for event in events
        ]
        subprocess.run(
            ["node", "--input-type=module", "-e", script, reducer.as_uri()],
            input=json.dumps(frames),
            text=True,
            check=True,
            capture_output=True,
        )
    now[0] += 86401
    await store.apply_retention()
    events = await store.list_events(first.session_id, limit=1000)
    assert all(
        not e.event["data"]["thinking"]
        for e in events
        if e.event["type"] == "assistant.message"
    )
    await second_store.aclose()
    await store.aclose()


@pytest.mark.asyncio
async def test_import_transaction_rolls_back_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    path = tmp_path / "log.jsonl"
    write_log(path, [{"type": "final", "text": "done"}])
    original = store._before_commit

    def fail(label: str) -> None:
        if label == "import_dsh":
            raise RuntimeError("injected")
        original(label)

    monkeypatch.setattr(store, "_before_commit", fail)
    with pytest.raises(RuntimeError, match="injected"):
        await import_dsh(path, store=store)
    assert await store.list_sessions(kinds=("worker",)) == []
    monkeypatch.setattr(store, "_before_commit", original)
    assert (await import_dsh(path, store=store)).imported
    await store.aclose()


@pytest.mark.asyncio
async def test_missing_call_id_cannot_leak_sensitive_read(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    path = tmp_path / "log.jsonl"
    write_log(
        path,
        [
            {"type": "tool_call", "tool": "read", "input": {"path": ".env"}},
            {
                "type": "tool_result",
                "status": "success",
                "result": "private-fixture-value",
            },
            {"type": "final", "text": "done"},
        ],
    )
    imported = await import_dsh(path, store=store)
    events = await store.list_events(imported.session_id, limit=100)
    assert "private-fixture-value" not in json.dumps([event.event for event in events])
    await store.aclose()


@pytest.mark.asyncio
async def test_scrubbed_real_log_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/dsh/dsh-worker.jsonl"
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    first = await import_dsh(fixture, title="Fixture", store=store)
    second = await import_dsh(fixture, store=store)
    assert first.imported and not second.imported
    assert (
        first.session_id == second.session_id and first.skipped == second.skipped == 0
    )
    records = [json.loads(line) for line in fixture.read_text().splitlines()]
    assert len(records) == 32
    assert any("usage" in record for record in records)
    turns = await store.list_turns(first.session_id)
    assert len(turns) == 1 and turns[0].answer == "Synthetic answer"
    expected_output = sum(
        record["usage"]["outputTokens"] for record in records if "usage" in record
    )
    assert turns[0].result is not None
    assert turns[0].result["usage"]["completion_tokens"] == expected_output
    events = await store.list_events(first.session_id, limit=100)
    data = [event.event for event in events]
    starts = {
        event["data"]["call_id"] for event in data if event["type"] == "tool.started"
    }
    finished = next(event["data"] for event in data if event["type"] == "tool.finished")
    assert finished["call_id"] in starts
    assert finished["output"] == "Synthetic result"
    assert any(
        event["data"].get("thinking") == "Synthetic reasoning"
        for event in data
        if event["type"] == "assistant.message"
    )
    if shutil.which("node"):
        reducer = Path(__file__).parents[2] / "support/gateway/static/turns.mjs"
        script = """
const {newView, applyEvent} = await import(process.argv[1]);
let input = ''; for await (const chunk of process.stdin) input += chunk;
const view = newView('test');
for (const event of JSON.parse(input)) applyEvent(view, event);
const turn = view.turns[0];
if (!turn.items.some(item => item.kind === 'assistant_text' &&
    item.text === 'Synthetic pre-tool text') || turn.outputTokens !== Number(process.argv[2]))
    process.exit(1);
"""
        frames = [
            {**event.event, "id": event.id, "turn": event.turn} for event in events
        ]
        subprocess.run(
            [
                "node",
                "--input-type=module",
                "-e",
                script,
                reducer.as_uri(),
                str(expected_output),
            ],
            input=json.dumps(frames),
            text=True,
            check=True,
            capture_output=True,
        )
    await store.aclose()


@pytest.mark.asyncio
async def test_session_workspace_git_and_invalid_line(tmp_path: Path) -> None:
    workspace = tmp_path / ("sk-" + "a" * 30)
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(workspace), *args], text=True
        ).strip()

    path = tmp_path / "log.jsonl"
    write_log(
        path,
        [
            {"type": "session", "cwd": str(workspace)},
            {"type": "session", "cwd": "/ignored"},
            {"type": "status", "step": 1, "phase": "step_end"},
            {"type": "final", "text": "done"},
        ],
        invalid=True,
    )
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    imported = await import_dsh(path, store=store)
    assert imported.skipped == 1
    info = await store.get_session(imported.session_id)
    assert info is not None
    assert info.workspace == str(tmp_path / "[redacted]")
    assert info.meta["git_branch"] == git("rev-parse", "--abbrev-ref", "HEAD")
    assert info.meta["git_head"] == git("rev-parse", "HEAD")
    await store.aclose()


@pytest.mark.asyncio
async def test_missing_final_is_crashed_and_unknown_tool_is_safe(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    path = tmp_path / "log.jsonl"
    write_log(
        path,
        [
            {"type": []},
            {
                "type": "tool_result",
                "callId": "missing",
                "status": "error",
                "result": "bad",
            },
        ],
    )
    imported = await import_dsh(path, store=store)
    assert imported.skipped == 1
    turns = await store.list_turns(imported.session_id)
    assert turns[0].status == "crashed"
    events = await store.list_events(imported.session_id, limit=100)
    result = next(e.event["data"] for e in events if e.event["type"] == "tool.finished")
    assert result["call_id"] == "missing" and result["name"] == "unknown"
    await store.aclose()
