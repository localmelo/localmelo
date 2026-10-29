"""Shared helpers for the session tests: a clock, events and the invariants."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import sqlite3
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import EVENT_FIELDS
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.sessions.store import CAUSES, RUN_STATUSES
from localmelo.support.sessions.window import WINDOW_CODEC_VERSION

DAY = 86_400.0
T0 = 1_761_000_000.0


class Clock:
    """An injected clock that moves only when told to."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def core_event(
    type_: str, *, run_id: str = "run-1", seq: int = 1, ts: float = T0, **data: Any
) -> dict[str, Any]:
    """A core event as ``AgentEvent.to_dict()`` gives it, with every pinned key."""
    defaults: dict[str, dict[str, Any]] = {
        "run.started": {
            "query": "q",
            "max_steps": 3,
            "workspace_root": None,
            "approval": False,
        },
        "step.started": {"step": 1, "attempt": 1},
        "assistant.message": {
            "step": 1,
            "content": "",
            "thinking": "",
            "tool_calls": [],
            "usage": {},
            "prefix": None,
            "duration_s": 0.1,
        },
        "tool.started": {
            "step": 1,
            "call_id": "call_1",
            "name": "file_read",
            "arguments": {"path": "a.txt"},
            "preview": "a.txt",
        },
        "tool.finished": {
            "step": 1,
            "call_id": "call_1",
            "name": "file_read",
            "status": "success",
            "error_category": "none",
            "output": "x",
            "output_chars": 1,
            "error": "",
            "duration_s": 0.01,
            "images": [],
            "approval": None,
            "exit_code": None,
        },
        "reflection": {
            "attempt": 1,
            "action": "continue",
            "continue_run": True,
            "flags": {
                "task_still_feasible": True,
                "next_step_is_concrete": True,
                "next_step_is_novel": True,
            },
            "gate": "flags",
            "utility": None,
            "stop_reason": None,
            "rationale": "",
            "summary": "",
            "best_effort_result": "",
            "requests": 1,
            "readable": True,
            "unreadable_reason": None,
            "usage": {},
            "prefix": None,
            "duration_s": 0.1,
        },
        "run.finished": {"result": run_result()},
    }
    payload = dict(defaults[type_])
    payload.update(data)
    assert set(payload) == EVENT_FIELDS[type_]
    return {
        "v": 1,
        "type": type_,
        "run_id": run_id,
        "seq": seq,
        "ts": ts,
        "data": payload,
    }


def run_result(
    status: str = "completed", result: str = "done", error: str = "", **extra: Any
) -> dict[str, Any]:
    """A ``RunResult.to_dict()``."""
    data: dict[str, Any] = {
        "task_id": "run-1",
        "status": status,
        "result": result,
        "error": error,
        "steps": 1,
        "attempts": 1,
        "n_llm": 1,
        "n_main": 1,
        "n_aux": 0,
        "n_tool": 0,
        "llm_errors": 0,
        "usage": {},
        "duration_s": 0.5,
        "config": {},
    }
    data.update(extra)
    return data


def approval_event(type_: str, request_id: str, **data: Any) -> dict[str, Any]:
    """An ``approval.*`` session event as a runner publishes it."""
    if type_ == "approval.requested":
        payload: dict[str, Any] = {
            "request_id": request_id,
            "call_id": "call_1",
            "tool_name": "shell_exec",
            "arguments": {"command": "ls"},
            "preview": "ls",
            "workspace_root": "/tmp",
        }
    else:
        payload = {
            "request_id": request_id,
            "call_id": "call_1",
            "tool_name": "shell_exec",
            "outcome": "allowed",
            "origin": "user",
            "reason": "",
            "remember": False,
        }
    payload.update(data)
    return {"v": 1, "type": type_, "run_id": "", "seq": None, "ts": T0, "data": payload}


@contextmanager
def read_only(path: str | os.PathLike[str]) -> Iterator[sqlite3.Connection]:
    """A read-only sqlite3 connection to *path*, for checking rows."""
    conn = sqlite3.connect(f"file:{os.fspath(path)}?mode=ro", uri=True)
    try:
        yield conn
    finally:
        conn.close()


def rows(path: str | os.PathLike[str], sql: str, *args: Any) -> list[Any]:
    with read_only(path) as conn:
        return conn.execute(sql, args).fetchall()


def events_of(
    path: str | os.PathLike[str], session_id: str
) -> list[tuple[int, Any, str]]:
    """``(id, turn, type)`` of a session's events, in id order."""
    return [
        (row[0], row[1], row[2])
        for row in rows(
            path,
            "SELECT id, turn, type FROM events WHERE session_id = ? ORDER BY id",
            session_id,
        )
    ]


def check_invariants(path: str | os.PathLike[str], writer: str) -> None:
    """Assert I-0 to I-7 (and S-3) on the file at *path*."""
    with read_only(path) as conn:
        # I-0: one writer marker, this writer's.
        markers = conn.execute(
            "SELECT value FROM store_info WHERE key = 'writer'"
        ).fetchall()
        assert markers == [(writer,)]
        sessions = [row[0] for row in conn.execute("SELECT id FROM sessions")]
        for session_id in sessions:
            turns = conn.execute(
                "SELECT turn, id, status, cause, finished_at, result, undone, user_text "
                "FROM turns WHERE session_id = ? ORDER BY turn",
                (session_id,),
            ).fetchall()
            # I-2: numbers 1..N.
            assert [row[0] for row in turns] == list(range(1, len(turns) + 1))
            # I-4: at most one running turn.
            assert sum(row[2] == "running" for row in turns) <= 1
            for number, turn_id, status, cause, finished_at, result, _, _ in turns:
                events = conn.execute(
                    "SELECT id, type, event FROM events WHERE session_id = ? AND turn = ? "
                    "ORDER BY id",
                    (session_id, number),
                ).fetchall()
                types = [row[1] for row in events]
                # I-1: one turn.started, and one turn.finished once not running,
                # with nothing of the turn after it.
                assert types.count("turn.started") == 1, (session_id, number, types)
                assert types[0] == "turn.started"
                if status == "running":
                    assert "turn.finished" not in types
                    # I-7.
                    assert cause == "" and finished_at is None and result is None
                else:
                    assert types.count("turn.finished") == 1, types
                    assert types[-1] == "turn.finished", types
                    finished = json.loads(events[-1][2])["data"]
                    assert finished["status"] == status
                    assert finished["cause"] == cause
                    assert finished["turn_id"] == turn_id
                    # I-7.
                    assert status in RUN_STATUSES, status
                    assert cause in CAUSES, cause
                    assert finished_at is not None
                    # S-3: every request answered once.
                    requested: dict[str, int] = {}
                    for _, type_, text in events:
                        data = json.loads(text)["data"]
                        if type_ == "approval.requested":
                            requested.setdefault(data["request_id"], 0)
                        elif type_ == "approval.resolved":
                            requested[data["request_id"]] = (
                                requested.get(data["request_id"], 0) + 1
                            )
                    assert all(count == 1 for count in requested.values()), requested
            # I-6: a window row's turn is finished and not undone.
            for turn, version in conn.execute(
                "SELECT turn, version FROM windows WHERE session_id = ?", (session_id,)
            ):
                status, undone = conn.execute(
                    "SELECT status, undone FROM turns WHERE session_id = ? AND turn = ?",
                    (session_id, turn),
                ).fetchone()
                assert status != "running" and not undone
                assert version <= WINDOW_CODEC_VERSION
        # I-3: one submission per key; accepted ones have a matching turn.
        for state, session_id, turn, digest in conn.execute(
            "SELECT state, session_id, turn, text_sha256 FROM submissions"
        ):
            if state == "accepted":
                (text,) = conn.execute(
                    "SELECT user_text FROM turns WHERE session_id = ? AND turn = ?",
                    (session_id, turn),
                ).fetchone()
                assert hashlib.sha256(text.encode()).hexdigest() == digest
            else:
                assert state == "withdrawn" and turn is None
        # I-5: ids only grow: the sequence is at or above the largest id.
        largest = conn.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]
        sequence = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM sqlite_sequence WHERE name = 'events'"
        ).fetchone()[0]
        assert sequence >= largest


class ScriptedLLM(BaseLLMProvider):
    """Answers with each scripted reply in turn, then waits until cancelled.

    ``requests`` keeps the messages of every request; ``blocked`` is set
    once a request waits for a reply that is not scripted.
    """

    def __init__(self, replies: Iterable[Message] = ()) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[list[Message]] = []
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(copy.deepcopy(messages))
        if self.replies:
            return self.replies.popleft()
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def close(self) -> None:
        pass


def answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name=name, arguments=arguments, id=call_id)],
    )


def workspace(root: Path) -> str:
    """A workspace directory with a notes file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.txt").write_text("buy milk\n")
    return str(root)


async def started_agent(root: Path, llm: BaseLLMProvider, **options: Any) -> Agent:
    """A started Agent with a workspace and record_answers, in-memory history."""
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            workspace_root=workspace(root),
            record_answers=True,
            **options,
        ),
    )
    await agent.start()
    return agent
