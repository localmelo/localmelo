"""Redact a terminal chat on read, as a worker run is redacted when it is written.

The terminal stores a chat's events unredacted, ``!`` shell output included, so
the console passes everything of a terminal chat through the same steps the
worker recorder applies at write time: secrets in every string
(:func:`redact_worker`), the output of a read of a secret file replaced by a
note, and the run configuration left out. A worker run is stored that way
already and is never passed through here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from localmelo.support.sessions.attachments import TextAttachment
from localmelo.support.sessions.store import SessionInfo, StoredEvent, TurnInfo
from localmelo.support.sessions.worker import redact_worker, sensitive_read


def sensitive_path(started: Mapping[str, Any], workspace: str | None) -> str | None:
    """The secret file a stored ``tool.started`` event reads, redacted, or ``None``."""
    data = started.get("data")
    if started.get("type") != "tool.started" or not isinstance(data, Mapping):
        return None
    return sensitive_read(str(data.get("name", "")), data.get("arguments"), workspace)


def redact_event(event: StoredEvent, secret_reads: Mapping[str, str]) -> StoredEvent:
    """*event* redacted; *secret_reads* maps a call id to the secret file it read."""
    payload = redact_worker(event.event)
    data = payload.get("data")
    if isinstance(data, dict):
        if payload.get("type") == "tool.finished":
            path = secret_reads.get(str(data.get("call_id", "")))
            if path:
                data["output"] = f"[Sensitive file content omitted: {path}]"
                data["output_chars"] = len(data["output"])
                data["error"] = ""
                data["images"] = []
        elif payload.get("type") == "run.finished" and isinstance(
            data.get("result"), dict
        ):
            data["result"].pop("config", None)
    return replace(event, event=payload)


def redact_turn(turn: TurnInfo) -> TurnInfo:
    result = redact_worker(turn.result)
    if isinstance(result, dict):
        result.pop("config", None)
    attachments = turn.attachments
    return replace(
        turn,
        user_text=redact_worker(turn.user_text),
        typed_text=redact_worker(turn.typed_text),
        answer=redact_worker(turn.answer),
        result=result,
        attachments=(
            None
            if attachments is None
            else tuple(
                TextAttachment(redact_worker(item.name), redact_worker(item.text))
                for item in attachments
            )
        ),
    )


def redact_info(info: SessionInfo) -> SessionInfo:
    return replace(
        info,
        title=redact_worker(info.title),
        model=redact_worker(info.model),
        workspace=redact_worker(info.workspace),
        meta=redact_worker(info.meta),
    )
