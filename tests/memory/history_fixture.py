"""A fixture ``history.db``: tasks and steps through SqliteHistory, plus llm_calls.

At ``c295ce2`` localmelo records no model calls, and ``history.db`` holds
only ``tasks`` and ``steps``. R1 adds the ``llm_calls`` table and the code
that writes it. Until then this helper creates the table from the plan's v1
sketch (``.melo/reasoning_memory_plan.md``, lines 258-271) and writes its
rows itself. R1's migration replaces this copy of the DDL, and whichever of
R1 and R3 merges second replaces the rows with ones R1's writer records
(R3 spec, Requirement 15).

With ``llm_calls=False`` the file is one written before R1: since R1,
``SqliteHistory`` creates ``llm_calls`` and sets ``user_version`` 1, so the
helper drops that table and sets ``user_version`` back to 0 once the
history is closed.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.retrieval import RawCall
from localmelo.melo.schema import StepRecord, TaskRecord

# The plan's v1 sketch of llm_calls, copied as it stands; R1's migration
# replaces it.
LLM_CALLS_V1 = """
CREATE TABLE IF NOT EXISTS llm_calls (
  task_id TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  step INTEGER, attempt INTEGER,
  outcome TEXT NOT NULL,
  content TEXT NOT NULL DEFAULT '', thinking TEXT NOT NULL DEFAULT '',
  reasoning_key TEXT NOT NULL DEFAULT '',
  step_ids_json TEXT,
  redactions INTEGER NOT NULL DEFAULT 0,
  tool_calls_json TEXT, usage_json TEXT, duration_s REAL, timestamp REAL NOT NULL,
  PRIMARY KEY (task_id, seq));
"""

_INSERT = """
INSERT INTO llm_calls (task_id, seq, kind, step, attempt, outcome, content,
                       thinking, step_ids_json, tool_calls_json, timestamp)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

StepIds = dict[tuple[str, int], list[str]]


async def write_history(
    path: str | Path, calls: Sequence[RawCall], *, llm_calls: bool = True
) -> StepIds:
    """Write *calls* to a new ``history.db`` at *path*, as a recording run would.

    Each task gets a ``tasks`` row. Each tool call with a result gets a
    ``steps`` row through :meth:`SqliteHistory.add_step`, with the reply's
    content as the first row's thought, as the agent writes them. With
    *llm_calls*, each call also gets an ``llm_calls`` row:
    ``tool_calls_json`` lists its calls as ``{"tool_name", "arguments",
    "id"}`` (``NULL`` without any), and ``step_ids_json`` its ``steps``
    rows in call order (``NULL`` without any).

    Returns the step ids written for each call, by ``(task_id, seq)``.
    Every connection is closed on return, so no ``-wal`` or ``-shm`` file
    is left behind.
    """
    history = SqliteHistory(path)
    step_ids: StepIds = {}
    try:
        for task_id in dict.fromkeys(call.task_id for call in calls):
            await history.save_task(
                TaskRecord(query=f"fixture task {task_id}", task_id=task_id)
            )
        for call in calls:
            ids: list[str] = []
            pairs = zip(call.tool_calls, call.tool_results, strict=False)
            for index, (tool_call, result) in enumerate(pairs):
                step = StepRecord(
                    thought=call.content if index == 0 else "",
                    tool_call=tool_call,
                    tool_result=result,
                    timestamp=call.timestamp,
                )
                await history.add_step(call.task_id, step)
                ids.append(step.step_id)
            step_ids[(call.task_id, call.seq)] = ids
    finally:
        await history.aclose()

    if not llm_calls:
        # A file written before R1: no llm_calls table, user_version 0.
        connection = sqlite3.connect(path)
        try:
            connection.execute("DROP TABLE llm_calls")
            connection.execute("PRAGMA user_version = 0")
            connection.commit()
        finally:
            connection.close()
    if llm_calls:
        connection = sqlite3.connect(path)
        try:
            connection.executescript(LLM_CALLS_V1)
            connection.executemany(
                _INSERT,
                [_row(call, step_ids[(call.task_id, call.seq)]) for call in calls],
            )
            connection.commit()
        finally:
            connection.close()
    return step_ids


def write_history_sync(
    path: str | Path, calls: Sequence[RawCall], *, llm_calls: bool = True
) -> StepIds:
    """:func:`write_history`, for tests that run outside an event loop."""
    return asyncio.run(write_history(path, calls, llm_calls=llm_calls))


def _row(call: RawCall, step_ids: list[str]) -> tuple[object, ...]:
    tool_calls_json = (
        json.dumps(
            [
                {"tool_name": c.tool_name, "arguments": c.arguments, "id": c.id}
                for c in call.tool_calls
            ]
        )
        if call.tool_calls
        else None
    )
    return (
        call.task_id,
        call.seq,
        call.kind,
        call.step,
        call.attempt,
        call.outcome,
        call.content,
        call.thinking,
        json.dumps(step_ids) if step_ids else None,
        tool_calls_json,
        call.timestamp,
    )
