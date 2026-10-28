"""Direct worker recording preserves run behavior and redacts stored records."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support.sessions.store import SessionStore
from localmelo.support.sessions.worker import run_worker

from .helpers import ScriptedLLM, answer, tool_call


async def test_worker_run_records_one_redacted_turn(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    secret = "sk-" + "f00dfeed12" * 3
    llm = ScriptedLLM([answer(f"Result {secret}")])
    agent = Agent(
        llm=llm, options=AgentOptions(workspace_root=str(tmp_path), memory=False)
    )
    result = await run_worker(agent, f"Check {secret}", title="", parent_run="parent-1")
    assert result.status == "completed"
    await agent.close()
    store = SessionStore(writer="cli")
    sessions = await store.list_sessions(kinds=("worker",))
    assert len(sessions) == 1
    session = sessions[0]
    assert session.meta["parent_run"] == "parent-1"
    assert session.meta["status"] == "completed"
    turns = await store.list_turns(session.id)
    events = await store.list_events(session.id)
    assert len(turns) == 1 and turns[0].status == "completed"
    assert {"run.started", "step.started", "assistant.message", "run.finished"} <= {
        e.event["type"] for e in events
    }
    assert secret not in json.dumps(
        [session.meta, session.title, turns[0].user_text, [e.event for e in events]]
    )
    assert all(
        "config" not in e.event.get("data", {}).get("result", {})
        for e in events
        if e.event["type"] == "run.finished"
    )
    if shutil.which("node"):
        reducer = Path(__file__).parents[2] / "support/gateway/static/turns.mjs"
        script = """
const {newView, applyEvent} = await import(process.argv[1]);
let input = ''; for await (const chunk of process.stdin) input += chunk;
const view = newView('test');
for (const event of JSON.parse(input)) applyEvent(view, event);
if (view.turns.length !== 1 || view.turns[0].status !== 'completed') process.exit(1);
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
    await store.aclose()


async def test_saved_and_unsaved_runs_send_the_same_requests(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    options = AgentOptions(workspace_root=str(tmp_path), memory=False)
    first = ScriptedLLM([answer("Done")])
    agent = Agent(llm=first, options=options)
    await agent.run_task("Do the task")
    await agent.close()
    assert not (tmp_path / ".cache/localmelo/cli/sessions.db").exists()
    second = ScriptedLLM([answer("Done")])
    agent = Agent(llm=second, options=options)
    await run_worker(agent, "Do the task", title="", parent_run=None)
    await agent.close()

    def wire(messages):
        return [
            [
                (m.role, m.content, m.tool_calls, m.tool_call_id, m.thinking)
                for m in batch
            ]
            for batch in messages
        ]

    assert wire(first.requests) == wire(second.requests)


async def test_worker_git_and_url_metadata_are_minimal(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "--no-verify",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    llm = ScriptedLLM([answer("Done")])
    llm.model = "example"
    llm._http = SimpleNamespace(  # type: ignore[attr-defined]
        base_url="https://user:password@example.invalid/private?secret=value"
    )
    agent = Agent(
        llm=llm, options=AgentOptions(workspace_root=str(tmp_path), memory=False)
    )
    await run_worker(agent, "Task", title="", parent_run=None)
    await agent.close()
    store = SessionStore(writer="cli")
    info = (await store.list_sessions(kinds=("worker",)))[0]
    assert info.meta["base_url_host"] == "example.invalid"
    assert info.meta["git_head"] and info.meta["git_branch"]
    assert "password" not in json.dumps(info.meta)
    assert "private" not in json.dumps(info.meta)
    await store.aclose()


async def test_sensitive_file_read_is_not_stored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "settings.json").write_text("fixture-private-content")
    llm = ScriptedLLM(
        [tool_call("file_read", {"path": "settings.json"}), answer("Done")]
    )
    agent = Agent(
        llm=llm, options=AgentOptions(workspace_root=str(tmp_path), memory=False)
    )
    result = await run_worker(agent, "Read the file", title="test", parent_run=None)
    assert result.status == "completed"
    await agent.close()
    store = SessionStore(writer="cli")
    session = (await store.list_sessions(kinds=("worker",)))[0]
    events = await store.list_events(session.id)
    output = [
        e.event["data"]["output"] for e in events if e.event["type"] == "tool.finished"
    ]
    assert len(output) == 1 and "settings.json" in output[0]
    assert "fixture-private-content" not in json.dumps([e.event for e in events])
    await store.aclose()
