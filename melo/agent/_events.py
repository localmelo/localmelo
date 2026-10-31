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
from typing import TYPE_CHECKING, Any, NamedTuple

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.events import AgentEvent, EventSink, json_safe_payload
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.executor.models import ExecutionOutcome
from localmelo.melo.executor.preview import call_preview
from localmelo.melo.schema import (
    CALL_INTERRUPTED,
    CALL_NOT_STARTED,
    ImagePart,
    Message,
    ReflectionDecision,
    ReflectionEntry,
    RunResult,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from localmelo.melo.agent._brief_gate import BriefCheckResult
    from localmelo.melo.agent._chat import CallTelemetry
    from localmelo.melo.memory.procedures.tracker import ProgressState
    from localmelo.melo.memory.recall import RecallRecord

logger = logging.getLogger(__name__)

# The message of each notice kind but retry, whose message names its delay
# and error.
NOTICE_MESSAGES: Mapping[str, str] = {
    "stuck": "The same tool calls or errors repeated; the attempt ends.",
    "budget": "The attempt used its step budget.",
    "empty_reply": "The model returned an empty reply; asking it to continue.",
    "unfinished_answer": (
        "The model's reply ended inside its thinking; asking it for its answer."
    ),
    "no_change": "Nothing has changed in the workspace yet.",
    "brief_items": "Some numbered items of the brief have no edit yet.",
    "brief_check_failed": (
        "The task's check still fails; the run ends completed with failing tests."
    ),
}

# The message of a brief_check notice, by how the check went.
BRIEF_CHECK_MESSAGES = {
    "passed": "The task's check passed.",
    "failed": "The task's check failed.",
    "refused": "The task's check was refused and did not run.",
}

# Characters of a failed request's error that a retry notice quotes.
RETRY_ERROR_CHARS = 200


def check_sink(on_event: object, name: str = "on_event") -> None:
    """Raise ``TypeError`` unless *on_event* is ``None`` or a synchronous callable.

    A coroutine function, or an object whose ``__call__`` is one, is
    refused: the run never awaits its sink. *name* is the keyword the
    sink was given as, for the message.
    """
    if on_event is None:
        return
    if not callable(on_event):
        raise TypeError(f"{name} must be a callable that takes an AgentEvent")
    if inspect.iscoroutinefunction(on_event) or inspect.iscoroutinefunction(
        getattr(on_event, "__call__", None)  # noqa: B004 – reads __call__ itself
    ):
        raise TypeError(
            f"{name} must be synchronous: put events on a queue and consume "
            "them in a task"
        )


class GateVerdict(NamedTuple):
    """The gate in force after a reflection, and why the run stopped."""

    gate: str
    stop_reason: str | None
    utility: float | None


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

    *interactive* says whether the sink follows the run as its caller
    does (``on_event``), which makes a cancellation end the run
    ``cancelled``; an observer's sink (``observe``) is not interactive.
    *progress* says whether the caller asked for ``run.progress`` events.
    """

    def __init__(
        self,
        sink: EventSink | None = None,
        run_id: str = "",
        *,
        interactive: bool = True,
        progress: bool = False,
    ) -> None:
        self._sink = sink
        self._run_id = run_id
        self._interactive = interactive
        self._progress = progress
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
        self._unreadable_reason: str | None = None
        self._repaired: str | None = None
        self._calls: list[CallTelemetry] = []
        self._sink_failed = False
        self._warned_awaitable = False

    @property
    def active(self) -> bool:
        """Whether the run has a sink, a caller's or an observer's."""
        return self._sink is not None

    @property
    def interactive(self) -> bool:
        """Whether the run has a sink that follows it as its caller does."""
        return self._sink is not None and self._interactive

    @property
    def reports_progress(self) -> bool:
        """Whether the run has a sink and its caller asked for progress."""
        return self._sink is not None and self._progress

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

    def context_budget(self, record: dict[str, Any]) -> None:
        if self._sink is not None:
            self._emit("context_budget", lambda: json_safe_payload(record))

    def inbox_accepted(self, offset: int, text: str, step: int, attempt: int) -> None:
        if self._sink is not None:
            self._emit(
                "inbox.accepted",
                lambda: json_safe_payload(
                    {
                        "offset": offset,
                        "text": redact_secrets(text)[0],
                        "step": step,
                        "attempt": attempt,
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

    def assistant_message(
        self, response: Message, call: CallTelemetry | None = None
    ) -> None:
        """The planning reply of the current step.

        *call* is what the request reported beyond the reply: its extra
        usage fields join ``usage`` and its prefix diagnostics are the
        ``prefix`` object (``None`` for the run's first planning request).

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
                    "usage": {**_usage([response]), **(call.details if call else {})},
                    "prefix": call.prefix if call else None,
                    "duration_s": duration_s,
                }
            )

        self._emit("assistant.message", build)

    def note(
        self,
        *,
        step: int,
        attempt: int,
        note_id: str,
        kind: str,
        source: str,
        text: str,
        chars: int,
        drafts: Sequence[str],
        remember: bool,
        stage: str,
    ) -> None:
        """A note entered the working-memory note store.

        *text* is the checked text (already redacted and sanitized), so it
        passes no further redaction; *chars* is its length and *drafts*
        the aliases of the reply's drafts. The CLI ignores the event until
        U2 draws a plan line.
        """
        if self._sink is None:
            return
        self._emit(
            "note",
            lambda: json_safe_payload(
                {
                    "step": step,
                    "attempt": attempt,
                    "note_id": note_id,
                    "kind": kind,
                    "source": source,
                    "text": text,
                    "chars": chars,
                    "drafts": list(drafts),
                    "remember": remember,
                    "stage": stage,
                }
            ),
        )

    def memory(
        self,
        *,
        candidate_id: str,
        action: str,
        kind: str,
        admitted_by: str,
        entry_id: int | None,
    ) -> None:
        """One admission decision was committed to the decision log.

        No text and no ids beyond the entry id: the decision log is the
        audit surface, and the event is its summary. The CLI ignores it.
        """
        if self._sink is None:
            return
        self._emit(
            "memory",
            lambda: json_safe_payload(
                {
                    "candidate_id": candidate_id,
                    "action": action,
                    "kind": kind,
                    "admitted_by": admitted_by,
                    "entry_id": entry_id,
                }
            ),
        )

    def recall(self, record: RecallRecord) -> None:
        """A recall block entered a request (R5).

        The query and hit ids are already redacted, and the record holds no
        item text (see :func:`~localmelo.melo.memory.recall.event_fields`).
        The CLI ignores the event until U2 draws a recall line.
        """
        if self._sink is None:
            return
        from localmelo.melo.memory.recall import event_fields

        self._emit("recall", lambda: json_safe_payload(event_fields(record)))

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
        approval: Mapping[str, Any] | None = None,
    ) -> None:
        """The checked *result* of *call*, whose executor *outcome* gives the status.

        *exit_codes* are those the call's tool recorded in
        ``TOOL_EXIT_CODE``; the last one is the event's ``exit_code``.
        *approval* is what came of asking the approval policy about the
        call (``request_id``, ``allowed``, ``reason``, ``remember`` and
        ``origin``), or ``None`` when it was not asked.
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
                    "approval": None if approval is None else dict(approval),
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
        cancelled: bool,
        exit_codes: Sequence[int] | None,
        not_started: bool = False,
        approval: Mapping[str, Any] | None = None,
    ) -> None:
        """*call* ended in *exc* instead of a result.

        *cancelled* says that *exc* is the run's cancellation (a caller's,
        or the run's ``timeout_s``), which gives the status ``cancelled``
        and the error ``CALL_INTERRUPTED``, or ``CALL_NOT_STARTED`` when
        *not_started* says that the approval policy was still deciding
        about the call. Any other exception, which ends the run, gives
        ``crashed``, also a ``CancelledError`` that something raised while
        nothing cancelled the run. ``duration_s`` is the time since
        ``tool.started``. *approval* is as for :meth:`tool_finished`.
        """
        if self._sink is None:
            return
        step = self._step
        duration_s = time.perf_counter() - self._tool_since
        status = "cancelled" if cancelled else "crashed"
        if not cancelled:
            error = _error_text(exc)
        elif not_started:
            error = CALL_NOT_STARTED
        else:
            error = CALL_INTERRUPTED
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
                    "approval": None if approval is None else dict(approval),
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

    def unfinished_answer(self) -> None:
        """A reply was only unfinished thinking, and the model is asked again."""
        if self._sink is None:
            return
        self._notice("unfinished_answer", NOTICE_MESSAGES["unfinished_answer"], {})

    def no_change(self, scope: str | None = None) -> None:
        """A brief run found the workspace unchanged: at a boundary, or with
        *scope* ``"step"`` after steps of reading."""
        if self._sink is None:
            return
        detail = {} if scope is None else {"scope": scope}
        self._notice("no_change", NOTICE_MESSAGES["no_change"], detail)

    def brief_items(self, scope: str, marks: dict[str, str]) -> None:
        """A brief run named the items no edit has touched; *marks* maps each
        item's number to its mark and *scope* is ``"reflection"`` or ``"deadline"``."""
        if self._sink is None:
            return
        self._notice(
            "brief_items",
            NOTICE_MESSAGES["brief_items"],
            {"scope": scope, "marks": marks},
        )

    def brief_check(self, result: BriefCheckResult, round_: int) -> None:
        """A brief run's gate ran one check; *round_* counts the gate's runs."""
        if self._sink is None:
            return
        state = "refused" if result.refused else "passed" if result.passed else "failed"
        self._notice(
            "brief_check",
            BRIEF_CHECK_MESSAGES[state],
            {
                "round": round_,
                "command": result.command,
                "exit_code": result.exit_code,
                "duration_s": result.duration_s,
                "refused": result.refused,
                "output_tail": result.output_tail,
            },
        )

    def brief_check_failed(self, retries: int) -> None:
        """The gate gives up: the run ends ``completed`` with a failing check."""
        if self._sink is None:
            return
        self._notice(
            "brief_check_failed",
            NOTICE_MESSAGES["brief_check_failed"],
            {"retries": retries},
        )

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
        self._calls = []
        self._unreadable_reason = None
        self._repaired = None
        self._reflecting_since = time.perf_counter()

    @property
    def replies(self) -> list[Message] | None:
        """The list that the reflection under way puts its replies in.

        ``None`` without a sink: the run then keeps no replies.
        """
        return self._replies

    def reflection_call(self, call: CallTelemetry | None) -> None:
        """Keep what a request of the reflection under way reported."""
        if call is not None and self._replies is not None:
            self._calls.append(call)

    def reflection_unreadable(self, reason: str) -> None:
        """Keep why the first reply of the reflection under way held no reflection."""
        self._unreadable_reason = reason

    def reflection_repaired(self, repair: str | None) -> None:
        """Keep the repair that made the reply of the reflection under way readable."""
        self._repaired = repair

    def reflection(
        self,
        reflected: tuple[ReflectionEntry, ReflectionDecision] | None,
        *,
        continue_run: bool,
        verdict: GateVerdict,
    ) -> None:
        """The outcome of the reflection under way.

        *reflected* is what ``Agent._do_reflect`` returned: ``None`` for a
        reply that holds no reflection after it was asked again, whose
        decision fields are then ``None``. ``requests``, ``usage`` and
        ``duration_s`` cover the replies in :attr:`replies`, from
        ``reflection.started`` on; the extra usage fields of the requests
        (see :meth:`reflection_call`) are summed into ``usage`` and the first
        request's prefix diagnostics are ``prefix``. *verdict* is the gate in
        force and why the run stopped, if it did. ``unreadable_reason`` is why
        the first reply held no reflection: ``tool_call`` (it has tool calls
        and no reflection), ``no_json`` or ``invalid``; ``None`` when it
        read. The retry that follows can still make ``readable`` true.
        ``repaired`` names the one slip of the JSON that was mended to read
        the reply that gave the reflection (``unmatched_closer``,
        ``trailing_comma``, ``control_char``, ``invalid_escape`` or
        ``unclosed_array``); ``None`` when none was.
        """
        if self._sink is None:
            return
        attempt = self._attempt
        replies = list(self._replies or ())
        usage = _usage(replies)
        for call in self._calls:
            for name, value in call.details.items():
                usage[name] = usage.get(name, 0) + value
        prefix = self._calls[0].prefix if self._calls else None
        later = [call.prefix for call in self._calls[1:] if call.prefix is not None]
        if prefix is not None and later:
            # Like usage, the novelty covers every request of the reflection.
            prefix = {
                **prefix,
                **{
                    key: sum(p[key] for p in [prefix, *later])
                    for key in ("fresh_chars", "request_chars")
                },
            }
        duration_s = time.perf_counter() - self._reflecting_since
        action: str | None = None
        rationale: str | None = None
        summary: str | None = None
        best_effort: str | None = None
        flags: dict[str, bool] | None = None
        if reflected is not None:
            entry, decision = reflected
            action = decision.recommended_action
            rationale = decision.rationale
            summary = entry.summary
            best_effort = decision.best_effort_result
            flags = {
                "task_still_feasible": decision.task_still_feasible,
                "next_step_is_concrete": decision.next_step_is_concrete,
                "next_step_is_novel": decision.next_step_is_novel,
            }
        self._emit(
            "reflection",
            lambda: json_safe_payload(
                {
                    "attempt": attempt,
                    "action": action,
                    "continue_run": continue_run,
                    "flags": flags,
                    "gate": verdict.gate,
                    "utility": verdict.utility,
                    "stop_reason": verdict.stop_reason,
                    "rationale": _redacted(rationale),
                    "summary": _redacted(summary),
                    "best_effort_result": _redacted(best_effort),
                    "requests": len(replies),
                    "readable": reflected is not None,
                    "unreadable_reason": self._unreadable_reason,
                    "repaired": self._repaired,
                    "usage": usage,
                    "prefix": prefix,
                    "duration_s": duration_s,
                }
            ),
        )

    def progress(self, state: ProgressState) -> None:
        """Where the run stands: a ``run.progress`` event with *state*'s data.

        Only when the caller asked for progress (see
        :attr:`reports_progress`); the run computes *state* only then.
        """
        if self._sink is None or not self._progress:
            return
        self._emit("run.progress", lambda: json_safe_payload(state.as_dict()))

    def run_finished(
        self, result: RunResult, reviews: dict[str, Any] | None = None
    ) -> None:
        """The run's last event, with ``result.to_dict()`` whole.

        A RunResult holds only JSON values, and nothing in it is clipped, so
        ``RunResult(**event.data["result"])`` rebuilds *result*. ``reviews``
        is ``None`` for a run without an approval policy; with one, the calls
        the rules found routine, high and undecided, the model reviews sent,
        and the steps auto mode blocked (``blocked``: call id, title and step
        key each, #339).
        """
        if self._sink is None:
            return
        self._emit(
            "run.finished",
            lambda: {
                "result": result.to_dict(),
                "reviews": None if reviews is None else json_safe_payload(reviews),
            },
        )

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


def _redacted(text: str | None) -> str | None:
    """*text* through :func:`redact_secrets`; ``None`` stays ``None``."""
    return None if text is None else redact_secrets(text)[0]


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
