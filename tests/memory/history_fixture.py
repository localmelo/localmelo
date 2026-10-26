"""A fixture ``history.db``: tasks, steps and llm_calls through SqliteHistory.

The helper writes a run's model calls as a recording run would, through
R1's writer: each task's ``tasks`` row, each tool call's ``steps`` row with
:meth:`SqliteHistory.add_step`, each call's ``llm_calls`` row with
:meth:`SqliteHistory.add_llm_call`, and the call's step ids with
:meth:`SqliteHistory.link_llm_call_steps`. Its history's clock reads 0, so
the retention of thinking never clears the fixture's rows, whose
timestamps are far in the past.

With ``llm_calls=False`` the file is one written before R1: since R1,
``SqliteHistory`` creates ``llm_calls`` and sets ``user_version`` 1, so the
helper drops that table and sets ``user_version`` back to 0 once the
history is closed.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.retrieval import RawCall
from localmelo.melo.schema import LLMCallRecord, StepRecord, TaskRecord

StepIds = dict[tuple[str, int], list[str]]


async def write_history(
    path: str | Path, calls: Sequence[RawCall], *, llm_calls: bool = True
) -> StepIds:
    """Write *calls* to a new ``history.db`` at *path*, as a recording run would.

    Each task gets a ``tasks`` row. Each tool call with a result gets a
    ``steps`` row through :meth:`SqliteHistory.add_step`, with the reply's
    content as the first row's thought, as the agent writes them. With
    *llm_calls*, each call also gets an ``llm_calls`` row through
    :meth:`SqliteHistory.add_llm_call`: ``tool_calls_json`` lists its calls
    as ``{"tool_name", "arguments", "id"}`` (``NULL`` without any), and
    :meth:`SqliteHistory.link_llm_call_steps` lists its ``steps`` rows in
    call order in ``step_ids_json`` (``NULL`` without any). The rows have
    no reasoning key, redactions, usage or duration.

    Returns the step ids written for each call, by ``(task_id, seq)``.
    Every connection is closed on return, so no ``-wal`` or ``-shm`` file
    is left behind.
    """
    history = SqliteHistory(path, clock=lambda: 0.0)
    step_ids: StepIds = {}
    try:
        for task_id in dict.fromkeys(call.task_id for call in calls):
            await history.save_task(
                TaskRecord(query=f"fixture task {task_id}", task_id=task_id)
            )
        for call in calls:
            if llm_calls:
                await history.add_llm_call(_record(call))
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
            if llm_calls:
                await history.link_llm_call_steps(call.task_id, call.seq, ids)
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
    return step_ids


def write_history_sync(
    path: str | Path, calls: Sequence[RawCall], *, llm_calls: bool = True
) -> StepIds:
    """:func:`write_history`, for tests that run outside an event loop."""
    return asyncio.run(write_history(path, calls, llm_calls=llm_calls))


def _record(call: RawCall) -> LLMCallRecord:
    """The ``llm_calls`` row of *call*, as R1's writer takes it."""
    return LLMCallRecord(
        task_id=call.task_id,
        seq=call.seq,
        kind=call.kind,
        step=call.step,
        attempt=call.attempt,
        outcome=call.outcome,
        content=call.content,
        thinking=call.thinking,
        reasoning_key="",
        redactions=0,
        tool_calls=tuple(call.tool_calls),
        usage=None,
        duration_s=0.0,
        timestamp=call.timestamp,
    )
