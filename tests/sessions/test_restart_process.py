"""Crash recovery across a real process boundary, which never replays work.

The deterministic core of the resume A/B harness (scripts/resume_ab.py):
a child process (restart_child.py) runs a gateway store and a runner on a
scripted model and ends with ``os._exit(73)`` at each crash point of the
spec's table, C-1 to C-6, without running a single finally block. The
parent then opens the file and checks the invariants I-0 to I-7, the
statuses and causes, the submission lookup, that a repeat of the key runs
nothing, and that a client resuming from its last event id receives
recovery's events. It started from the old gateway's process-boundary
test (bb7e910:tests/gateway/test_recovery_process.py), keeping its
technique: the child skips every clean-up, the parent reopens the files.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.schema import NO_ANSWER_LABEL
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    CLOSED_ERROR,
    RECOVERED_APPROVAL_REASON,
    STALE_ERROR,
    SessionStore,
)
from localmelo.support.sessions.submissions import SubmissionRepeated

from .helpers import (
    ScriptedLLM,
    answer,
    check_invariants,
    rows,
    started_agent,
    workspace,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")

CHILD = Path(__file__).with_name("restart_child.py")
KEY = "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07"
TEXT = "Crash during this request"


async def crash(*args: str) -> dict[str, Any]:
    """Run the child to its crash point; return what it printed."""
    env = dict(os.environ)
    env.pop("LOCALMELO_PERSIST_MEMORY", None)
    env.pop("LOCALMELO_MEMORY_DIR", None)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        str(CHILD),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=60)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 73, err.decode() + out.decode()
    data: dict[str, Any] = json.loads(out.decode().strip().splitlines()[-1])
    return data


async def repeat_runs_nothing(
    tmp_path: Path, store: SessionStore, session_id: str, turn_id: str
) -> None:
    """A repeat of the accepted key returns the turn and never runs it again."""
    llm = ScriptedLLM()
    agent = await started_agent(tmp_path / "ws", llm)
    runner = SessionRunner(agent, store, session_id)
    with pytest.raises(SubmissionRepeated) as repeated:
        await runner.start_turn(TEXT, client_request_id=KEY)
    assert repeated.value.turn.id == turn_id
    assert llm.requests == []  # the model sees no request
    await agent.close()


CRASH_POINTS = {
    # point: (scenario, status, cause)
    "C-1": ("c1", "crashed", "crash"),
    "C-2": ("c2", "crashed", "crash"),
    "C-2 with an approval": ("c2_approval", "crashed", "crash"),
    "C-3": ("c3", "completed", "run"),
    "C-3 cancelled": ("c3_cancelled", "cancelled", "crash"),
    "C-4": ("c4", "completed", "run"),
    "C-5 in append_events": ("c5_append", "crashed", "crash"),
    "C-5 in finish_turn": ("c5_finish", "completed", "run"),
}


@pytest.mark.parametrize("point", sorted(CRASH_POINTS))
async def test_a_process_death_at_each_crash_point_is_recovered(
    tmp_path: Path, point: str
) -> None:
    scenario, status, cause = CRASH_POINTS[point]
    path = tmp_path / "sessions.db"
    root = workspace(tmp_path / "ws")
    seen = await crash(scenario, str(path), root)
    store = SessionStore(path, writer="gateway")
    await store.open()  # recovery runs before open() returns
    check_invariants(path, "gateway")
    sid, turn_id = seen["session_id"], seen["turn_id"]
    (turn,) = await store.list_turns(sid)
    assert (turn.id, turn.status, turn.cause) == (turn_id, status, cause)
    if status == "completed":
        assert turn.answer == "AKITA-RECOVERY-527"
        assert turn.result is not None and turn.result["status"] == "completed"
    if point in ("C-1", "C-5 in append_events"):
        assert turn.run_id == ""  # run.started never reached the file
    submission = await store.get_submission(sid, KEY)
    assert submission is not None and submission.state == "accepted"
    assert (submission.turn_id, submission.status, submission.cause) == (
        turn_id,
        status,
        cause,
    )
    # A client that reconnects with its last event id gets recovery's events.
    replay = await store.list_events(sid, after_id=seen["last_id"])
    types = [e.event["type"] for e in replay]
    if point == "C-4":
        assert types == []  # complete: nothing to recover
    else:
        assert types[-1] == "turn.finished"
        finished = replay[-1].event["data"]
        assert (finished["status"], finished["cause"]) == (status, cause)
        if status == "crashed":
            assert finished["error"] == STALE_ERROR
            assert turn.answer == NO_ANSWER_LABEL.format(status="crashed")
    if point == "C-2 with an approval":
        resolved = [
            e.event["data"] for e in replay if e.event["type"] == "approval.resolved"
        ]
        assert len(resolved) == 1
        assert resolved[0]["outcome"] == "withdrawn"
        assert resolved[0]["reason"] == RECOVERED_APPROVAL_REASON
    await repeat_runs_nothing(tmp_path, store, sid, turn_id)
    check_invariants(path, "gateway")
    await store.aclose()


async def test_a_death_before_acceptance_leaves_nothing_and_a_retry_runs_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    seen = await crash("c5_begin", str(path), workspace(tmp_path / "ws"))
    store = SessionStore(path, writer="gateway")
    sid = seen["session_id"]
    assert await store.list_turns(sid) == []
    assert await store.get_submission(sid, KEY) is None  # unknown: nothing ran
    llm = ScriptedLLM([answer("ran once")])
    agent = await started_agent(tmp_path / "ws", llm)
    runner = SessionRunner(agent, store, sid)
    await runner.start_turn(TEXT, client_request_id=KEY)
    result = await runner.wait()
    assert result is not None and result.result == "ran once"
    assert len(llm.requests) == 1
    with pytest.raises(SubmissionRepeated):
        await runner.start_turn(TEXT, client_request_id=KEY)
    assert len(llm.requests) == 1
    check_invariants(path, "gateway")
    await agent.close()
    await store.aclose()


async def test_a_death_during_recovery_is_finished_by_the_next_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    made = await crash("c6_make", str(path))
    halfway = await crash("c6_recover", str(path))
    assert halfway == {"recovered": 1}
    # The child died inside the second turn's recovery: the first stays done.
    assert sorted(row[0] for row in rows(path, "SELECT status FROM turns")) == [
        "crashed",
        "running",
    ]
    last = rows(path, "SELECT MAX(id) FROM events")[0][0]
    store = SessionStore(path, writer="gateway")
    await store.open()
    check_invariants(path, "gateway")
    replayed = [
        e.event["type"]
        for sid in made["session_ids"]
        for e in await store.list_events(sid, after_id=last)
    ]
    assert replayed == ["turn.finished"]  # only the rest was recovered
    for sid in made["session_ids"]:
        (turn,) = await store.list_turns(sid)
        assert (turn.status, turn.cause) == ("crashed", "crash")
        finished = [
            e
            for e in await store.list_events(sid)
            if e.event["type"] == "turn.finished"
        ]
        assert len(finished) == 1  # recovery is idempotent per turn
    await store.aclose()


async def test_a_shutdown_records_shutdown_and_a_late_end_is_left_to_recovery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    seen = await crash("shutdown", str(path), workspace(tmp_path / "ws"))
    assert seen["late_error"] == CLOSED_ERROR
    store = SessionStore(path, writer="gateway")
    (turn,) = await store.list_turns(seen["session_id"])
    assert (turn.status, turn.cause) == ("cancelled", "shutdown")
    assert turn.result is not None and turn.result["error"] == "Session closed"
    assert await store.get_window(seen["session_id"]) is not None
    (late,) = await store.list_turns(seen["late_session_id"])
    assert (late.status, late.cause) == ("crashed", "crash")  # recovered at this open
    check_invariants(path, "gateway")
    await store.aclose()


async def test_both_resume_arms_after_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    root = workspace(tmp_path / "ws")
    seen = await crash("arms", str(path), root)
    sid = seen["session_id"]
    copy = tmp_path / "pairs.db"
    copy.write_bytes(path.read_bytes())
    for wal in ("-wal", "-shm"):
        if Path(f"{path}{wal}").exists():
            Path(f"{copy}{wal}").write_bytes(Path(f"{path}{wal}").read_bytes())
    results: dict[str, Any] = {}
    for mode, file in (("window", path), ("pairs", copy)):
        store = SessionStore(file, writer="gateway")
        llm = ScriptedLLM([answer("turn 3")])
        agent = await started_agent(tmp_path / "ws", llm)
        runner = SessionRunner(agent, store, sid, resume_mode=mode)  # type: ignore[arg-type]
        assert await runner.resume() == 2
        info = runner.last_resume
        assert info is not None
        results[mode] = (info.turns, info.checkpoint_turns)
        assert (await runner.run_turn("Turn three")).status == "completed"
        request = llm.requests[0]
        roles = [m.role for m in request if m.role != "system"]
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False)), roles
        # Turn 2 as its pair, right before turn 3.
        talk = [(m.role, m.content) for m in request if m.role in ("user", "assistant")]
        assert talk[-3:] == [
            ("user", "Now wait for ever"),
            ("assistant", NO_ANSWER_LABEL.format(status="crashed")),
            ("user", "Turn three"),
        ]
        # Each tool message answers the call before it.
        for before, message in zip(request, request[1:], strict=False):
            if message.role == "tool":
                assert before.tool_calls
                assert message.tool_call_id == before.tool_calls[0].id
        has_tool_turn = any(m.role == "tool" for m in request)
        if mode == "window":
            assert has_tool_turn
            tool = next(m for m in request if m.role == "tool")
            assert "buy milk" in tool.content  # turn 1's tool result
            assert any(
                m.tool_calls and m.tool_calls[0].tool_name == "file_read"
                for m in request
            )
        else:
            assert not has_tool_turn
        check_invariants(file, "gateway")
        await agent.close()
        await store.aclose()
    assert results == {"window": (2, 1), "pairs": (2, 0)}
