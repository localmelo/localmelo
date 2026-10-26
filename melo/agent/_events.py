"""The events of one run, for the caller's optional sink (see ``Agent.run_task``).

Internal to the agent: :class:`_Emitter` builds each event of the catalogue
in :mod:`localmelo.melo.contracts.events` at its hook in the run loop and
hands it to the sink.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.events import AgentEvent, EventSink, json_safe_payload
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.executor.models import ExecutionOutcome
from localmelo.melo.executor.preview import call_preview
from localmelo.melo.schema import (
    CALL_INTERRUPTED,
    ImagePart,
    Message,
    ReflectionDecision,
    ReflectionEntry,
    RunResult,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

# The message of each notice kind but retry, whose message names its delay
# and error.
NOTICE_MESSAGES: Mapping[str, str] = {
    "stuck": "The same tool calls or errors repeated; the attempt ends.",
    "budget": "The attempt used its step budget.",
    "empty_reply": "The model returned an empty reply; asking it to continue.",
}

# Characters of a failed request's error that a retry notice quotes.
RETRY_ERROR_CHARS = 200


def check_sink(on_event: object) -> None:
    """Raise ``TypeError`` unless *on_event* is ``None`` or a synchronous callable.

    A coroutine function, or an object whose ``__call__`` is one, is
    refused: the run never awaits its sink.
    """
    if on_event is None:
        return
    if not callable(on_event):
        raise TypeError("on_event must be a callable that takes an AgentEvent")
    if inspect.iscoroutinefunction(on_event) or inspect.iscoroutinefunction(
        getattr(on_event, "__call__", None)  # noqa: B004 – reads __call__ itself
    ):
        raise TypeError(
            "on_event must be synchronous: put events on a queue and consume "
            "them in a task (see localmelo.support.sessions.SessionRunner)"
        )


class _Emitter:
    """Builds the events of one run and hands each to the run's sink.

    Without a sink, every method returns at once: the run builds no event
    and no payload. With one, each event gets the next ``seq`` and is
    passed to the sink synchronously, in order, from the task that runs
    the run:

    - A sink that raises an ``Exception`` or a ``CancelledError`` loses
      that event, and the run goes on; a synchronous sink cannot receive the
      run's cancellation, so its ``CancelledError`` is its own failure. The
      first failure of a run is logged at WARNING with a traceback, later
      ones at DEBUG. Any other ``BaseException`` propagates.
    - An event whose payload cannot be built is lost the same way, without
      using a ``seq``.
    - A returned awaitable is not awaited, and a coroutine is closed; one
      WARNING per run says so.
    - There is no buffer and no timeout: a slow sink holds up the run.

    The emitter also keeps the run's position: the step and attempt of the
    latest ``step.started``, which the step's other events carry, and the
    clocks for the durations it reports.
    """

    def __init__(self, sink: EventSink | None = None, run_id: str = "") -> None:
        self._sink = sink
        self._run_id = run_id
        self._seq = 0
        self._step = 0
        self._attempt = 0
        # perf_counter() values: after step.started, after tool.started and
        # after reflection.started.
        self._planning_since = 0.0
        self._tool_since = 0.0
        self._reflecting_since = 0.0
        # The replies of the reflection under way (see replies).
        self._replies: list[Message] | None = None
        self._sink_failed = False
        self._warned_awaitable = False

    @property
    def active(self) -> bool:
        """Whether the run has a sink."""
        return self._sink is not None

    # ── The catalogue ──

    def run_started(
        self,
        query: str,
        *,
        max_steps: int,
        workspace_root: str | None,
        approval: bool,
    ) -> None:
        if self._sink is None:
            return
        self._emit(
            "run.started",
            lambda: json_safe_payload(
                {
                    "query": query,
                    "max_steps": max_steps,
                    "workspace_root": workspace_root,
                    "approval": approval,
                }
            ),
        )

    def step_started(self, step: int, attempt: int) -> None:
        if self._sink is None:
            return
        self._step, self._attempt = step, attempt
        self._emit(
            "step.started",
            lambda: json_safe_payload({"step": step, "attempt": attempt}),
        )
        self._planning_since = time.perf_counter()

    def assistant_message(self, response: Message) -> None:
        """The planning reply of the current step.

        Its ``thinking`` and ``content`` pass :func:`redact_secrets`, on the
        whole text and before the payload clips them; *response* itself
        keeps them as they are. ``duration_s`` is the time since
        ``step.started``, which comes right before the planning request:
        its retries and their delays included.
        """
        if self._sink is None:
            return
        duration_s = time.perf_counter() - self._planning_since
        step = self._step

        def build() -> dict[str, Any]:
            return json_safe_payload(
                {
                    "step": step,
                    "content": redact_secrets(response.content)[0],
                    "thinking": redact_secrets(response.thinking)[0],
                    "tool_calls": [_call_entry(call) for call in response.tool_calls],
                    "usage": _usage([response]),
                    "duration_s": duration_s,
                }
            )

        self._emit("assistant.message", build)

    def tool_started(self, call: ToolCall) -> None:
        if self._sink is None:
            return
        step = self._step
        self._emit(
            "tool.started",
            lambda: json_safe_payload(
                {
                    "step": step,
                    "call_id": call.id,
                    "name": call.tool_name,
                    "arguments": call.arguments,
                    "preview": call_preview(call.tool_name, call.arguments),
                }
            ),
        )
        self._tool_since = time.perf_counter()

    def tool_finished(
        self,
        call: ToolCall,
        outcome: ExecutionOutcome,
        result: ToolResult,
        *,
        exit_codes: Sequence[int] | None,
    ) -> None:
        """The checked *result* of *call*, whose executor *outcome* gives the status.

        *exit_codes* are those the call's tool recorded in
        ``TOOL_EXIT_CODE``; the last one is the event's ``exit_code``.
        """
        if self._sink is None:
            return
        step = self._step
        self._emit(
            "tool.finished",
            lambda: json_safe_payload(
                {
                    "step": step,
                    "call_id": call.id,
                    "name": call.tool_name,
                    "status": outcome.status.value,
                    "error_category": outcome.error_category.value,
                    "output": result.output,
                    "output_chars": len(result.output),
                    "error": result.error,
                    "duration_s": result.duration_ms / 1000,
                    "images": [_image_entry(image) for image in result.images],
                    "approval": None,
                    "exit_code": exit_codes[-1] if exit_codes else None,
                }
            ),
            images=tuple(result.images),
        )

    def tool_stopped(
        self,
        call: ToolCall,
        exc: BaseException,
        *,
        exit_codes: Sequence[int] | None,
    ) -> None:
        """*call* ended in *exc* instead of a result.

        A ``CancelledError`` (a cancellation, or the run's ``timeout_s``)
        gives the status ``cancelled``; any other exception, which ends the
        run, ``crashed``. ``duration_s`` is the time since ``tool.started``.
        """
        if self._sink is None:
            return
        step = self._step
        duration_s = time.perf_counter() - self._tool_since
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "crashed"
        error = CALL_INTERRUPTED if status == "cancelled" else _error_text(exc)
        self._emit(
            "tool.finished",
            lambda: json_safe_payload(
                {
                    "step": step,
                    "call_id": call.id,
                    "name": call.tool_name,
                    "status": status,
                    "error_category": status,
                    "output": "",
                    "output_chars": 0,
                    "error": error,
                    "duration_s": duration_s,
                    "images": [],
                    "approval": None,
                    "exit_code": exit_codes[-1] if exit_codes else None,
                }
            ),
        )

    def stuck(self) -> None:
        if self._sink is None:
            return
        self._notice("stuck", NOTICE_MESSAGES["stuck"], {})

    def budget_used(self) -> None:
        if self._sink is None:
            return
        self._notice("budget", NOTICE_MESSAGES["budget"], {})

    def empty_reply(self, count: int) -> None:
        """*count* is the number of empty replies in a row."""
        if self._sink is None:
            return
        self._notice("empty_reply", NOTICE_MESSAGES["empty_reply"], {"count": count})

    def retry(self, request_attempt: int, delay: float, error: ProviderError) -> None:
        """A model request failed and is sent again after *delay* seconds.

        The run sets this as the ``RETRY_HOOK`` of its requests.
        """
        if self._sink is None:
            return
        message = (
            f"Model request failed; retrying in {delay:g} s: "
            f"{str(error)[:RETRY_ERROR_CHARS]}"
        )
        detail = {
            "request_attempt": request_attempt,
            "delay_s": delay,
            "status_code": error.status_code,
        }
        self._notice("retry", message, detail)

    def reflection_started(self, failure_type: str) -> None:
        if self._sink is None:
            return
        attempt = self._attempt
        self._emit(
            "reflection.started",
            lambda: json_safe_payload(
                {"attempt": attempt, "failure_type": failure_type}
            ),
        )
        self._replies = []
        self._reflecting_since = time.perf_counter()

    @property
    def replies(self) -> list[Message] | None:
        """The list that the reflection under way puts its replies in.

        ``None`` without a sink: the run then keeps no replies.
        """
        return self._replies

    def reflection(
        self,
        reflected: tuple[ReflectionEntry, ReflectionDecision] | None,
        *,
        continue_run: bool,
    ) -> None:
        """The outcome of the reflection under way.

        *reflected* is what ``Agent._do_reflect`` returned: ``None`` for a
        reply that holds no reflection after it was asked again, whose
        decision fields are then ``None``. ``requests``, ``usage`` and
        ``duration_s`` cover the replies in :attr:`replies`, from
        ``reflection.started`` on.
        """
        if self._sink is None:
            return
        attempt = self._attempt
        replies = list(self._replies or ())
        duration_s = time.perf_counter() - self._reflecting_since
        action: str | None = None
        rationale: str | None = None
        summary: str | None = None
        best_effort: str | None = None
        if reflected is not None:
            entry, decision = reflected
            action = decision.recommended_action
            rationale = decision.rationale
            summary = entry.summary
            best_effort = decision.best_effort_result
        self._emit(
            "reflection",
            lambda: json_safe_payload(
                {
                    "attempt": attempt,
                    "action": action,
                    "continue_run": continue_run,
                    "rationale": rationale,
                    "summary": summary,
                    "best_effort_result": best_effort,
                    "requests": len(replies),
                    "readable": reflected is not None,
                    "usage": _usage(replies),
                    "duration_s": duration_s,
                }
            ),
        )

    def run_finished(self, result: RunResult) -> None:
        """The run's last event, with ``result.to_dict()`` whole.

        A RunResult holds only JSON values, and nothing in it is clipped, so
        ``RunResult(**event.data["result"])`` rebuilds *result*.
        """
        if self._sink is None:
            return
        self._emit("run.finished", lambda: {"result": result.to_dict()})

    # ── Delivery ──

    def _notice(self, kind: str, message: str, detail: dict[str, Any]) -> None:
        step, attempt = self._step, self._attempt
        self._emit(
            "notice",
            lambda: json_safe_payload(
                {
                    "kind": kind,
                    "message": message,
                    "step": step,
                    "attempt": attempt,
                    "detail": detail,
                }
            ),
        )

    def _emit(
        self,
        type_: str,
        build: Callable[[], dict[str, Any]],
        *,
        images: tuple[ImagePart, ...] = (),
    ) -> None:
        sink = self._sink
        if sink is None:
            return
        try:
            data = build()
        except Exception:
            self._failed("Could not build a %s event; it is lost", type_)
            return
        self._seq += 1
        event = AgentEvent(
            type=type_,
            run_id=self._run_id,
            seq=self._seq,
            ts=time.time(),
            data=data,
            images=images,
        )
        try:
            returned = sink(event)
        except (Exception, asyncio.CancelledError):
            self._failed("on_event raised on %s event %d; it is lost", type_, event.seq)
            return
        if inspect.isawaitable(returned):
            if inspect.iscoroutine(returned):
                returned.close()
            if not self._warned_awaitable:
                self._warned_awaitable = True
                logger.warning(
                    "on_event returned an awaitable; async sinks are not "
                    "supported and the event was dropped"
                )

    def _failed(self, message: str, *args: object) -> None:
        """Log a lost event: the run's first loss at WARNING, later ones at DEBUG."""
        level = logging.DEBUG if self._sink_failed else logging.WARNING
        self._sink_failed = True
        logger.log(level, message, *args, exc_info=True)


def _call_entry(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.tool_name,
        "arguments": call.arguments,
        "preview": call_preview(call.tool_name, call.arguments),
    }


def _image_entry(image: ImagePart) -> dict[str, Any]:
    return {
        "sha256": image.sha256,
        "mime_type": image.mime_type,
        "size_bytes": image.size_bytes,
    }


def _usage(replies: Sequence[Message]) -> dict[str, int]:
    """The integer usage fields of *replies*, summed by name.

    The same sum as ``RequestStats.add_usage``, so the events' usage adds up
    to the run's.
    """
    total: dict[str, int] = {}
    for reply in replies:
        for name, value in (reply.usage or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                total[name] = total.get(name, 0) + value
    return total


def _error_text(exc: BaseException) -> str:
    """``"<type>: <message>"``, or the type alone when the message is empty."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
