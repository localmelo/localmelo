"""The web workspace's shared history keeps reasoning as long as melo's (R1).

Requires fastapi (gateway extra). Each playground keeps its memory under
``tmp_path``.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from localmelo.melo.contracts.providers import BaseLLMProvider  # noqa: E402
from localmelo.melo.memory.history.sqlite import SqliteHistory  # noqa: E402
from localmelo.melo.schema import Message, ToolDef  # noqa: E402
from localmelo.support.gateway.playground import SmokePlayground  # noqa: E402


class FakeLLM(BaseLLMProvider):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content="Hello", thinking="Say hello.")

    async def close(self) -> None:
        pass


async def _write_old_call(path: Path, age_days: float) -> None:
    """One llm_calls row with thinking, *age_days* old, by a separate history."""
    history = SqliteHistory(path)
    try:
        conn = await history._ensure_ready()
        await conn.execute(
            """INSERT INTO llm_calls (task_id, seq, kind, step, attempt, outcome,
               content, thinking, reasoning_key, timestamp)
               VALUES ('old', 0, 'plan', 1, 1, 'answer', 'Hi.', 'Greet them.',
                       'reasoning_content', ?)""",
            (time.time() - age_days * 86400,),
        )
        await conn.commit()
    finally:
        await history.aclose()


def _query(path: Path, sql: str) -> list[tuple[object, ...]]:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        return connection.execute(sql).fetchall()


@pytest.fixture()
async def make_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[[], SmokePlayground]]:
    managers: list[SmokePlayground] = []

    def make() -> SmokePlayground:
        manager = SmokePlayground(memory_dir=tmp_path)
        monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: FakeLLM())
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        await manager.close_all()


@pytest.mark.parametrize(("value", "kept"), [("7", ""), (None, "Greet them.")])
async def test_a_new_chat_clears_old_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_manager: Callable[[], SmokePlayground],
    value: str | None,
    kept: str,
) -> None:
    path = tmp_path / "history.db"
    await _write_old_call(path, age_days=8)
    if value is not None:
        monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", value)
    manager = make_manager()
    payload = await manager.create_session(
        chat_url="http://127.0.0.1:8000/v1", chat_model="Qwen-27B", adapter="omlx"
    )
    await manager.run(str(payload["session_id"]), "Say hello.")

    assert _query(path, "SELECT thinking FROM llm_calls WHERE task_id = 'old'") == [
        (kept,)
    ]
