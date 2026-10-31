"""A chat made for a task: ``POST /api/sessions`` with ``turn.as_task``.

The request is a submission of the create scope, like any message: the key is checked
against the store's submissions before the chat is made (a cancelled key is
``submission_withdrawn``, a key another message was accepted under is a conflict, and the
other way round: a create of a message under a key a task was made for is a conflict), and
what it made is found again from the task, which carries the key in a column of its own
(``create_key``, apart from the ``request_key`` of a message to a chat). So a lookup says
``accepted`` with the chat and the task, a cancel stops the task instead of storing a
withdrawal, and a repeat answers the chat and echo the first made. The chat is the task's
owner as things stand: if it was deleted and its task handed to the main chat, the main
chat is the answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from localmelo.support.gateway.front_rules import route
from localmelo.support.gateway.i18n import t
from localmelo.support.gateway.task_store import TaskRecord
from localmelo.support.sessions.store import SessionInfo

from .cards import card_json
from .errors import ApiError
from .front import ECHO_EVENT

if TYPE_CHECKING:
    from . import Console


def conflict() -> ApiError:
    """A create-scope key used for another message: 409 ``submission_conflict``."""
    return ApiError(409, "submission_conflict", t("errors.messageIdUsed"))


def lock_name(client_request_id: str | None) -> str | None:
    """The name a create for a task and its cancel take turns under (NUL cannot be in a key)."""
    return None if client_request_id is None else "\0task\0" + client_request_id


async def created(
    console: Console, client_request_id: str, text: str | None = None
) -> tuple[TaskRecord, SessionInfo] | None:
    """The task a create made and the chat that owns it now, or ``None`` when the key made none.

    With *text*, a key made for another message is 409 ``submission_conflict``.
    """
    record = await console.workers.tasks.by_create_key(client_request_id)
    if record is None:
        return None
    owner = await console.effective_owner(record.front_id)
    info = await console.store.get_session(owner) if owner else None
    if info is None:
        raise conflict()
    if (
        text is not None
        and record.spec.get("objective") != route(text, as_task=True).text.strip()
    ):
        raise conflict()
    return record, info


async def started_echo(
    console: Console, record: TaskRecord, info: SessionInfo, text: str
) -> dict[str, Any]:
    """The ``started`` echo of the task in its chat; written from the task when the gateway stopped before it was (or the chat it was in has gone)."""
    found = await console.store.events_of_type(info.id, ECHO_EVENT, limit=200)
    for event in found:
        data = event.event.get("data") or {}
        if data.get("task_id") == record.task_id and data.get("echo") == "started":
            return card_json(event)
    return await console.front_for(info.id).restore_started(record, show=text)


def submission_of(
    record: TaskRecord, info: SessionInfo, client_request_id: str
) -> dict[str, Any]:
    """The submission as ``GET /api/submissions/{key}`` reads it: accepted, with the chat and the task."""
    return {
        "client_request_id": client_request_id,
        "state": "accepted",
        "session_id": info.id,
        "turn_id": None,
        "status": None,
        "cause": None,
        "turn": None,
        "error": "",
        "created_at": record.created_at,
        "task_id": record.task_id,
    }
