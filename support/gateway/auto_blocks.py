"""The steps auto mode blocked in a chat or a task, and their Allow once (#339).

A block is the ``tool.finished`` record of a call auto mode did not run:
its ``approval`` has origin ``auto``, ``allowed`` false, the call's
``review`` and its ``step_key``. Allow once finds the block by the
approval's ``request_id`` in that session's records (a client never sends a
key), allows that key once in the session's :class:`AutoPolicy`, and sends
the person's message that asks melo to go ahead. A block whose step ran on
an allowance, or has one waiting, is not allowed again.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from localmelo.melo.contracts.review import TITLES
from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.store import SessionStore


class BlockError(LookupError):
    """No block to allow: *code* is ``not_found``, ``not_allowable`` or ``already_allowed``."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

    @property
    def status(self) -> int:
        """The HTTP status the API answers with."""
        return 404 if self.code == "not_found" else 409

    @property
    def message(self) -> str:
        """What the API says, in the console's language."""
        if self.code == "not_found":
            return t("errors.blockNotFound")
        if self.code == "already_allowed":
            return t("errors.blockAllowedAlready")
        return t("errors.blockNotAllowable")


def _approval(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    approval = data.get("approval") if isinstance(data, dict) else None
    return approval if isinstance(approval, dict) else {}


def _categories(approval: dict[str, Any]) -> list[str]:
    review = approval.get("review")
    found = review.get("categories") if isinstance(review, dict) else None
    return [str(c) for c in found] if isinstance(found, list) else []


def allowable(approval: dict[str, Any]) -> tuple[str, list[str]] | None:
    """A block's step key and categories when Allow once may run it, else ``None``.

    Never for a step that changes localmelo itself, and never for a call a
    stopped turn did not review (it has no categories of its own).
    """
    key = approval.get("step_key")
    categories = [c for c in _categories(approval) if c in TITLES]
    if (
        approval.get("origin") != "auto"
        or approval.get("allowed") is not False
        or not isinstance(key, str)
        or not key
        or not categories
        or "self" in categories
    ):
        return None
    return key, categories


async def block_to_allow(
    store: SessionStore,
    session_id: str,
    request_id: str,
    live: Callable[[str], bool] = lambda key: False,
) -> tuple[str, list[str], str]:
    """The step key, categories and tool of the block *request_id* Allow once may run.

    :class:`BlockError` ``not_found`` for no such block in *session_id*,
    ``not_allowable`` for one that can never be allowed (:func:`allowable`),
    and ``already_allowed`` once the step ran on an allowance after it, or
    while *live* says its key has one not used yet.
    """
    found = None
    for stored in reversed(await store.auto_records(session_id, limit=200)):
        approval = _approval(stored.event)
        if (
            approval.get("request_id") == request_id
            and approval.get("allowed") is False
        ):
            found = stored
            break
    if found is None:
        raise BlockError("not_found")
    approval = _approval(found.event)
    allowed = allowable(approval)
    if allowed is None:
        raise BlockError("not_allowable")
    key, categories = allowed
    block = next(
        (
            b
            for b in await blocked_steps(store, session_id)
            if b["request_id"] == request_id
        ),
        None,
    )
    if live(key) or (block is not None and block["used_at"] is not None):
        raise BlockError("already_allowed")
    raw = found.event.get("data")
    tool = str(raw.get("name") or "") if isinstance(raw, dict) else ""
    return key, categories, tool


def step_words(categories: list[str]) -> str:
    """What the step does, in the page's language: "install software"."""
    title = t(f"auto.title.{categories[0]}") if categories else ""
    return title[:1].lower() + title[1:]


def go_ahead_text(categories: list[str]) -> str:
    """The person's message Allow once sends (decision 7)."""
    return t("auto.goAhead", what=step_words(categories))


async def blocked_steps(store: SessionStore, session_id: str) -> list[dict[str, Any]]:
    """The steps blocked in *session_id*, newest last, each with its record.

    ``used_at`` is when the allowed step ran (a later call of the same key
    allowed once), else ``None``.
    """
    records = await store.auto_records(session_id, limit=200)
    out: list[dict[str, Any]] = []
    for stored in records:
        event, approval = stored.event, _approval(stored.event)
        raw = event.get("data")
        data: dict[str, Any] = raw if isinstance(raw, dict) else {}
        if approval.get("allowance") == "once":
            for block in out:
                if block["step_key"] == approval.get("step_key") and not block.get(
                    "used_at"
                ):
                    block["used_at"] = event.get("ts")
            continue
        if approval.get("allowed") is not False:
            continue
        review = approval.get("review")
        out.append(
            {
                "request_id": approval.get("request_id"),
                "call_id": data.get("call_id"),
                "name": data.get("name"),
                "step_key": approval.get("step_key"),
                "reason": approval.get("reason", ""),
                "review": review if isinstance(review, dict) else None,
                "ts": event.get("ts"),
                "turn": stored.turn,
                "can_allow": allowable(approval) is not None,
                "used_at": None,
                "preview": "",
            }
        )
    # The step as the chat shows it: its tool.started preview.
    for turn in sorted({b["turn"] for b in out if b["turn"] is not None}):
        events = await store.list_events(session_id, turns=(turn, turn))
        previews = {
            e.event["data"].get("call_id"): str(e.event["data"].get("preview") or "")
            for e in events
            if e.event.get("type") == "tool.started"
            and isinstance(e.event.get("data"), dict)
        }
        for block in out:
            if block["turn"] == turn:
                block["preview"] = previews.get(block["call_id"], "")
    return out
