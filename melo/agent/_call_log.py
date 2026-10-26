"""The per-run recorder of model calls in history's ``llm_calls`` table.

Internal to the agent: ``Agent.run_task`` sets a :class:`CallLog` as the
run's call recorder whenever its history can keep model calls (see
:class:`CallHistory`), and ``Chat`` hands it every response. It records;
it never changes a request or a decision of the run, and a failure of the
history is logged, never raised.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.schema import LLMCallRecord, Message

logger = logging.getLogger(__name__)


@runtime_checkable
class CallHistory(Protocol):
    """A history that keeps model calls, such as ``SqliteHistory``."""

    async def add_llm_call(self, record: LLMCallRecord) -> None: ...

    async def link_llm_call_steps(
        self, task_id: str, seq: int, step_ids: Sequence[str]
    ) -> None: ...


class RunCounters(Protocol):
    """The step and attempt counters of a run (``_RunState`` in the agent)."""

    @property
    def steps(self) -> int: ...

    @property
    def attempts(self) -> int: ...


def _exception_text(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _cancelling() -> bool:
    """Whether the current task is being cancelled (or there is no task)."""
    task = asyncio.current_task()
    return task is None or task.cancelling() > 0


def _outcome(kind: str, response: Message) -> str:
    """The shape of a ``plan`` reply, by the run loop's rule; ``aux`` otherwise."""
    if kind != "plan":
        return "aux"
    if response.tool_calls:
        return "tools"
    return "answer" if response.content.strip() else "empty"


class CallLog:
    """Records the model calls of one run of task *task_id* in *history*.

    Each response becomes one row (see :meth:`record`), numbered from 0
    in the order the responses arrived, by a counter of its own that
    advances also when a write fails. :meth:`link_steps` lists the
    ``steps`` rows that the latest planning response produced on its row.

    A failure of the history is never the run's failure: each is logged
    on this module's logger, the run's first at WARNING and later ones at
    DEBUG. A cancellation of the run propagates, and so does any other
    ``BaseException``; a ``CancelledError`` that the history raises while
    nothing cancels the task is the history's failure, and is logged.
    """

    def __init__(self, history: CallHistory, task_id: str, run: RunCounters) -> None:
        self._history = history
        self._task_id = task_id
        self._run = run
        self._seq = 0
        # The seq of the latest plan response, written or not.
        self._latest_plan: int | None = None
        self._failures = 0

    async def record(
        self,
        kind: str,
        response: Message,
        *,
        reasoning_key: str,
        usage: Mapping[str, int] | None,
        duration_s: float,
        timestamp: float,
    ) -> None:
        """Write *response*, of call kind *kind*, as the run's next row.

        ``step`` and ``attempt`` are the run's counters now, when the
        response has arrived. ``content`` and ``thinking`` pass
        :func:`~localmelo.melo.checker.redaction.redact_secrets`, and
        ``redactions`` counts both. *reasoning_key* is kept only when the
        response has thinking. A ``plan`` response becomes the latest
        planning row before it is written, whether or not the write
        succeeds.
        """
        seq = self._seq
        self._seq += 1
        if kind == "plan":
            self._latest_plan = seq
        try:
            content, content_spans = redact_secrets(response.content)
            thinking, thinking_spans = redact_secrets(response.thinking)
            record = LLMCallRecord(
                task_id=self._task_id,
                seq=seq,
                kind=kind,
                step=self._run.steps,
                attempt=self._run.attempts,
                outcome=_outcome(kind, response),
                content=content,
                thinking=thinking,
                reasoning_key=reasoning_key if response.thinking else "",
                redactions=content_spans + thinking_spans,
                tool_calls=tuple(response.tool_calls),
                usage=None if usage is None else dict(usage),
                duration_s=duration_s,
                timestamp=timestamp,
            )
            await self._history.add_llm_call(record)
        except asyncio.CancelledError as exc:
            if _cancelling():
                raise
            self._log_failure(
                f"Could not record model call {seq} of task {self._task_id} "
                f"in history: {_exception_text(exc)}"
            )
        except Exception as exc:
            self._log_failure(
                f"Could not record model call {seq} of task {self._task_id} "
                f"in history: {_exception_text(exc)}"
            )

    async def link_steps(self, step_ids: Sequence[str]) -> None:
        """List *step_ids*, in order, on the latest planning row.

        Nothing happens before the first planning response or for an empty
        list. A planning row that was never written is updated nowhere: the
        ids never move onto an earlier row.
        """
        seq = self._latest_plan
        if seq is None or not step_ids:
            return
        try:
            await self._history.link_llm_call_steps(self._task_id, seq, step_ids)
        except asyncio.CancelledError as exc:
            if _cancelling():
                raise
            self._log_failure(
                f"Could not link model call {seq} of task {self._task_id} "
                f"to its steps in history: {_exception_text(exc)}"
            )
        except Exception as exc:
            self._log_failure(
                f"Could not link model call {seq} of task {self._task_id} "
                f"to its steps in history: {_exception_text(exc)}"
            )

    def _log_failure(self, message: str) -> None:
        self._failures += 1
        level = logging.WARNING if self._failures == 1 else logging.DEBUG
        logger.log(level, "%s", message)
