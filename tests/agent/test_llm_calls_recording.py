"""An Agent with a memory directory and the llm_calls table of history (R1).

Every memory directory is under ``tmp_path``; no test touches
``~/.cache/localmelo``. The environment is set with ``monkeypatch``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent import agent as agent_module
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import Message, ToolDef

SQLITE_HELPERS_LOGGER = "localmelo.melo.memory._sqlite"


class FakeLLM(BaseLLMProvider):
    """Pops queued replies and records every request."""

    def __init__(self, replies: list[Message] | None = None) -> None:
        self.replies: deque[Message] = deque(replies or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            return Message(role="assistant", content="(no reply queued)")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


# ── The memory directory's mode ─────────────────────────────────────────────


async def test_the_bench_directory_is_created_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bench sets these and never passes through _make_memory_dir.
    memory_dir = tmp_path / "run" / "mem"
    monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(memory_dir))
    agent = Agent(llm=FakeLLM())
    await agent.close()
    assert _mode(memory_dir) == 0o700


async def test_an_options_directory_is_created_0700(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    agent = Agent(llm=FakeLLM(), options=AgentOptions(memory_dir=str(memory_dir)))
    await agent.close()
    assert _mode(memory_dir) == 0o700


async def test_the_default_directory_is_tightened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    default = tmp_path / "default"
    default.mkdir()
    default.chmod(0o755)
    monkeypatch.setattr(agent_module, "DEFAULT_MEMORY_DIR", str(default))
    monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
    caplog.set_level(logging.DEBUG, logger=SQLITE_HELPERS_LOGGER)
    agent = Agent(llm=FakeLLM())
    await agent.close()
    assert _mode(default) == 0o700
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, f"Restricted {default} to mode 0700")
    ]


@pytest.mark.parametrize("named_by", ["environment", "options"])
async def test_a_chosen_directory_keeps_its_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named_by: str
) -> None:
    # Also when it names the default directory.
    default = tmp_path / "default"
    default.mkdir()
    default.chmod(0o755)
    monkeypatch.setattr(agent_module, "DEFAULT_MEMORY_DIR", str(default))
    options = None
    if named_by == "environment":
        monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
        monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(default))
    else:
        options = AgentOptions(memory_dir=str(default))
    agent = Agent(llm=FakeLLM(), options=options)
    await agent.close()
    assert _mode(default) == 0o755
    assert os.path.isdir(default)


# ── Retention through the Agent ─────────────────────────────────────────────


async def _write_old_call(path: Path, age_days: float) -> None:
    """One llm_calls row with thinking, *age_days* old, by a separate history."""
    history = SqliteHistory(path)
    try:
        conn = await history._ensure_ready()
        await conn.execute(
            """INSERT INTO llm_calls (task_id, seq, kind, step, attempt, outcome,
               content, thinking, reasoning_key, timestamp)
               VALUES ('old', 0, 'plan', 1, 1, 'answer', 'Paris.', 'Recall it.',
                       'reasoning_content', ?)""",
            (time.time() - age_days * 86400,),
        )
        await conn.commit()
    finally:
        await history.aclose()


def _old_thinking(path: Path) -> str:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        row = connection.execute(
            "SELECT thinking FROM llm_calls WHERE task_id = 'old'"
        ).fetchone()
    return str(row[0])


@pytest.mark.parametrize(("value", "kept"), [("7", ""), (None, "Recall it.")])
async def test_the_retention_variable_reaches_the_agents_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None, kept: str
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    await _write_old_call(memory_dir / "history.db", age_days=8)
    if value is not None:
        monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", value)
    monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(memory_dir))
    agent = Agent(llm=FakeLLM([Message(role="assistant", content="Paris.")]))
    try:
        result = await agent.run_task("What is the capital of France?")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert _old_thinking(memory_dir / "history.db") == kept


async def test_a_bad_retention_value_raises_before_the_directory_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_dir = tmp_path / "memory"
    monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "abc")
    with pytest.raises(ValueError, match="LOCALMELO_THINKING_RETENTION_DAYS"):
        Agent(llm=FakeLLM(), options=AgentOptions(memory_dir=str(memory_dir)))
    assert not memory_dir.exists()
    # Without persistence the variable is not read.
    agent = Agent(llm=FakeLLM())
    await agent.close()
