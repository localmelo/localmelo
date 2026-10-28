"""The JSON shapes of API v2: sessions, turns, events, approvals, submissions.

The shapes here are the ones ``docs/api.md`` documents. Every function is
pure: it takes a store record and returns a JSON-ready ``dict``. Labels go
through :mod:`localmelo.support.sessions.display`, so the console and the
terminal word a path the same way.
"""

from __future__ import annotations

from typing import Any

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.melo.contracts.events import json_safe_payload
from localmelo.support.sessions import display
from localmelo.support.sessions.store import (
    SessionInfo,
    StoredEvent,
    SubmissionInfo,
    TurnInfo,
)


def workspace_label(workspace: str | None) -> str:
    """The home-labelled workspace, or ``""`` for a row without one."""
    return "" if workspace is None else display.home_label(workspace)


def session_json(
    info: SessionInfo,
    *,
    pending_approvals: int = 0,
    running_here: bool | None = None,
) -> dict[str, Any]:
    """One session as ``GET /api/sessions`` and ``GET /api/sessions/{id}`` give it."""
    busy = info.busy
    return {
        "id": info.id,
        "kind": info.kind,
        "title": info.title,
        "created_at": info.created_at,
        "updated_at": info.updated_at,
        "model": info.model,
        "workspace": info.workspace,
        "workspace_label": workspace_label(info.workspace),
        "busy": busy,
        "running_here": busy if running_here is None else running_here,
        "pending_approvals": pending_approvals,
    }


def event_json(event: StoredEvent) -> dict[str, Any]:
    """One stored event exactly as its SSE frame's ``data`` reads."""
    return {
        "id": event.id,
        "session_id": event.session_id,
        "turn": event.turn,
        **event.event,
    }


def turn_json(
    turn: TurnInfo,
    *,
    running_here: bool | None = None,
    events: list[StoredEvent] | None = None,
) -> dict[str, Any]:
    """One turn, without its holder and session id, plus ``running_here``."""
    data: dict[str, Any] = {
        "id": turn.id,
        "turn": turn.turn,
        "run_id": turn.run_id,
        "user_text": turn.user_text,
        "status": turn.status,
        "cause": turn.cause,
        "answer": turn.answer,
        "result": turn.result,
        "undone": turn.undone,
        "started_at": turn.started_at,
        "finished_at": turn.finished_at,
        "client_request_id": turn.client_request_id,
        "running_here": (
            turn.status == "running" if running_here is None else running_here
        ),
    }
    if events is not None:
        data["events"] = [event_json(event) for event in events]
    return data


def approval_json(request: ApprovalRequest) -> dict[str, Any]:
    """One pending approval, as the console's card reads it.

    ``target`` stays ``None`` until the shared ``approval_target`` helper
    lands; API v2 only adds keys, so a later value is additive.
    """
    root = request.workspace_root
    return {
        "request_id": request.request_id,
        "call_id": request.call_id,
        "tool_name": request.tool_name,
        "arguments": json_safe_payload(dict(request.arguments)),
        "preview": request.preview,
        "workspace_root": root,
        "workspace_label": "" if root is None else display.home_label(root),
        "target": None,
    }


def submission_json(
    submission: SubmissionInfo | None, client_request_id: str
) -> dict[str, Any]:
    """One submission by its key; an unknown key reads ``state: "unknown"``.

    Only the key and the state are known for an unknown key; every other
    field is ``None``, as API v2 documents.
    """
    if submission is None:
        return {
            "client_request_id": client_request_id,
            "state": "unknown",
            "session_id": None,
            "turn_id": None,
            "status": None,
            "cause": None,
            "turn": None,
            "error": "",
            "created_at": None,
        }
    turn = submission.turn
    return {
        "client_request_id": submission.client_request_id,
        "state": submission.state,
        "session_id": submission.session_id,
        "turn_id": submission.turn_id,
        "status": submission.status,
        "cause": submission.cause,
        "turn": None if turn is None else turn_json(turn),
        "error": submission.error,
        "created_at": submission.created_at,
    }
