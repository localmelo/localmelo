"""Build a bounded, evidence-only resumption card from committed events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory.procedures.mask import mask_credentials
from localmelo.melo.schema import broken_tool_call
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.sessions.store import StoredEvent

MAX_CARD_CHARS = 2_000
# How every card begins: a turn whose text starts so is melo's card, not words
# the person typed.
CARD_START = "Where stopped: "


def _text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    value = mask_credentials(redact_secrets(value)[0])
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def build_park_card(
    spec: TaskSpec | Mapping[str, Any], events: Iterable[StoredEvent]
) -> str:
    """Return a deterministic card from a task spec and ordered stored events.

    Only events with a persisted ID are used. Tool arguments and outputs are
    not generally redacted by the event store, so only a file tool's path is
    extracted and redacted here; all other arguments and outputs are omitted.
    Missing or malformed fields are treated as unknown, never as success.
    """
    if isinstance(spec, TaskSpec):
        spec = spec.to_dict()
    if not isinstance(spec, Mapping):
        spec = {}
    objective = _text(spec.get("objective"), 350) or "Unknown"
    done = spec.get("done_when")
    items = [_text(item, 100) for item in done[:5]] if isinstance(done, list) else []
    items = [item for item in items if item]
    tools: list[str] = []
    pending: dict[tuple[str, str], str] = {}
    reflection = reply = ""
    for stored in events if isinstance(events, Iterable) else ():
        if not isinstance(stored, StoredEvent) or stored.id is None:
            continue
        event = stored.event
        if not isinstance(event, Mapping) or not isinstance(event.get("data"), Mapping):
            continue
        data = event["data"]
        kind = event.get("type")
        if kind == "assistant.message":
            said = data.get("content")
            # A tool call written as text is not what the task last said.
            if not (isinstance(said, str) and broken_tool_call(said)):
                reply = _text(said, 250) or reply
        elif kind == "reflection":
            reflection = _text(data.get("summary"), 250)
        elif kind in ("tool.started", "tool.finished"):
            name = _text(data.get("name"), 60) or "unknown tool"
            call_id = data.get("call_id")
            run_id = event.get("run_id")
            key = (
                (run_id, call_id)
                if isinstance(run_id, str) and isinstance(call_id, str) and call_id
                else None
            )
            if kind == "tool.started":
                if key:
                    pending[key] = name
                args = data.get("arguments")
                path = (
                    _text(args.get("path"), 120)
                    if name in ("file_read", "file_write") and isinstance(args, Mapping)
                    else ""
                )
                tools.append(f"{name} started" + (f" (path: {path})" if path else ""))
            else:
                if key:
                    pending.pop(key, None)
                status = _text(data.get("status"), 40) or "unknown"
                code = data.get("exit_code")
                suffix = (
                    f", exit code {code}"
                    if isinstance(code, int) and not isinstance(code, bool)
                    else ""
                )
                tools.append(f"{name}: {status}{suffix}")
    stopped = _text("; ".join(tools[-4:]), 400) or "No recorded tool activity"
    if pending:
        stopped += "; tool call without a recorded finish"
    left = _text("; ".join(items), 300) or "Completion criteria not recorded"
    card = (
        f"{CARD_START}{stopped}\n"
        f"Objective: {objective}\n"
        f"Last reflection (reported): {reflection or 'Not recorded'}\n"
        f"Last reply (reported): {reply or 'Not recorded'}\n"
        f"What is left: {left} (completion unverified)\n"
        "Next action: If resuming, inspect the recorded evidence and current state "
        "before deciding what to do.\n"
        "Cautions: Work may have been interrupted; check for side effects before "
        "repeating any action. Recorded reports are not proof of completion."
    )
    return card[:MAX_CARD_CHARS]
