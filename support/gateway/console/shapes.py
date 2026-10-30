"""The JSON shapes of API v2: sessions, turns, events, approvals, submissions.

The shapes here are the ones ``docs/api.md`` documents. Every function is
pure: it takes a store record and returns a JSON-ready ``dict``. Labels go
through :mod:`localmelo.support.sessions.display`, so the console and the
terminal word a path the same way.
"""

from __future__ import annotations

from typing import Any

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.melo.contracts.events import json_safe_payload
from localmelo.melo.contracts.review import ActionReview, review_title
from localmelo.melo.schema import Message, ToolInfo
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
    level: int = 2,
    held: str = "",
) -> dict[str, Any]:
    """One session as ``GET /api/sessions`` and ``GET /api/sessions/{id}`` give it.

    *level* and *held* are the scheduler's view of the chat (S2): the chip's
    P1-P4 and why a model call waits, from the one gate both surfaces use.
    ``main`` is true for the main chat (kind ``front``); every console chat
    of kind ``chat`` is one of its side chats.
    """
    busy = info.busy
    return {
        "id": info.id,
        "kind": info.kind,
        "main": info.kind == "front",
        "title": info.title,
        "created_at": info.created_at,
        "updated_at": info.updated_at,
        "model": info.model,
        "workspace": info.workspace,
        "workspace_label": workspace_label(info.workspace),
        "busy": busy,
        "running_here": busy if running_here is None else running_here,
        "pending_approvals": pending_approvals,
        "level": level,
        "held": held,
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


def _arguments_complete(original: dict[str, Any], safe: dict[str, Any]) -> bool:
    """Whether sanitizing preserved every JSON argument without conversion."""
    pairs: list[tuple[Any, Any]] = [(original, safe)]
    while pairs:
        left, right = pairs.pop()
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            if len(left) != len(right) or any(
                type(key) is not str or key not in right for key in left
            ):
                return False
            pairs.extend((value, right[key]) for key, value in left.items())
        elif type(left) is list:
            if len(left) != len(right):
                return False
            pairs.extend(zip(left, right, strict=True))
        elif type(left) not in (str, int, float, bool, type(None)) or left != right:
            return False
    return True


def _review_json(
    review: ActionReview | None, root: str | None
) -> dict[str, Any] | None:
    """The review as the card reads it: the verdict, the reasons and the title.

    ``title`` is the card's question for a high-impact call and ``""`` for
    any other. A reason goes through the secret redaction.
    """
    if review is None:
        return None
    return {
        "verdict": review.verdict,
        "title": review_title(review, display.folder_name(root)),
        "reasons": [redact_secrets(reason)[0] for reason in review.reasons],
    }


def approval_json(request: ApprovalRequest) -> dict[str, Any]:
    """One pending approval, as the console's card reads it.

    ``target`` is the shared ``approval_target`` shape for a ``file_write``
    request, read per response, and ``None`` for every other tool.
    ``review`` is what the checker's rules found, or ``None`` (#279).
    """
    root = request.workspace_root
    arguments = dict(request.arguments)
    safe_arguments = json_safe_payload(arguments)
    return {
        "request_id": request.request_id,
        "call_id": request.call_id,
        "tool_name": request.tool_name,
        "arguments": safe_arguments,
        "arguments_complete": _arguments_complete(arguments, safe_arguments),
        "preview": request.preview,
        "workspace_root": root,
        "workspace_label": "" if root is None else display.home_label(root),
        "target": display.approval_target(
            {
                "tool_name": request.tool_name,
                "arguments": request.arguments,
                "workspace_root": root,
            }
        ),
        "review": _review_json(request.review, root),
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


def memory_json(message: Message, *, limit: int) -> dict[str, Any]:
    """One working-memory row: role, text, tool names and image count.

    The text is what the conversation holds: a tool's output can carry a
    key, and a row the browser reads passes the same redaction the stored
    events do.
    """
    text = message.content
    if message.role == "assistant":
        # The page shows what was said, not how the model got there.
        _, text = display.split_thinking(text)
    text, _ = redact_secrets(text)
    if len(text) > limit:
        text = f"{text[:limit]}… (+{len(text) - limit} chars)"
    return {
        "role": message.role,
        "text": text,
        "tool_calls": [call.tool_name for call in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "images": len(message.images),
    }


def tool_json(tool: ToolInfo, granted: bool) -> dict[str, Any]:
    """One tool of a session, with whether this session allows it."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": json_safe_payload(dict(tool.parameters)),
        "source": tool.source,
        "requires_approval": tool.requires_approval,
        "granted": granted,
    }


def run_json(turn: TurnInfo) -> dict[str, Any]:
    """One finished turn's RunResult without its answer text."""
    result = turn.result
    if isinstance(result, dict):
        result = {key: value for key, value in result.items() if key != "result"}
    return {
        "turn": turn.turn,
        "turn_id": turn.id,
        "status": turn.status,
        "started_at": turn.started_at,
        "finished_at": turn.finished_at,
        "result": result,
    }
