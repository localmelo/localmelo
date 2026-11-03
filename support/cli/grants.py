"""``/grants`` in the terminal: what you allowed, where it holds, and Revoke.

In the main chat and its side chats the list is the gateway's
(``GET /api/grants``): what you allowed in the main chat counts everywhere
and comes first, then each chat's own; ``/grants revoke N`` takes back the
N-th (``DELETE /api/grants/{id}``). The console's ``/grants`` and its
Settings → Permissions number them the same way.

In the terminal's own chat it lists what holds here: this chat's own grants
and the main chat's. Only the main chat or the console takes back one of
the main chat's.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from localmelo.support.cli.main_chat import GatewayError
from localmelo.support.sessions.grants import listed
from localmelo.support.sessions.store import SessionStoreError

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession

USAGE: Final = "✗ Use /grants or /grants revoke N."
NONE: Final = (
    "No grants. When melo asks and you allow something for the chat, it shows here."
)
RULE: Final = (
    "What you allow in the main chat counts everywhere; in a side chat, only there."
)
FROM_MAIN: Final = (
    "✗ That one was allowed in the main chat. Take it back there with /grants, "
    "or in the console's Settings → Permissions."
)


def _when(at: object) -> str:
    if isinstance(at, bool) or not isinstance(at, int | float) or at <= 0:
        return ""
    return time.strftime("%b %d %H:%M", time.localtime(at))


def grant_line(number: int, row: Mapping[str, Any]) -> str:
    """One numbered line: ``2. Open example.com · Trip · you · Nov 03 14:05``."""
    where = (
        "everywhere"
        if row.get("scope") == "global"
        else str(row.get("chat_title") or "")
    )
    who = "Auto" if row.get("by") == "auto" else "you"
    parts = [
        str(row.get("label") or row.get("key") or ""),
        where,
        who,
        _when(row.get("granted_at")),
    ]
    return f"{number}. " + " · ".join(part for part in parts if part)


def _revoke_number(arguments: str) -> int | None:
    """``0`` for a plain ``/grants``, N for ``revoke N``; ``None`` for anything else."""
    parts = arguments.split()
    if not parts:
        return 0
    if len(parts) == 2 and parts[0].lower() == "revoke" and parts[1].isdigit():
        number = int(parts[1])
        return number if number > 0 else None
    return None


async def _remote_rows(session: CliSession) -> list[dict[str, Any]]:
    assert session.remote is not None
    body = await session.remote.client.call("GET", "/api/grants")
    rows = body.get("grants")
    return (
        [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    )


async def _local_rows(session: CliSession) -> list[dict[str, Any]]:
    await session.grants_ready()
    view = session.approvals.grants
    if view is None:
        return []
    title = session.title or "this chat"
    return [
        {
            **grant.to_json(),
            "chat_title": "Main chat" if grant.scope == "global" else title,
            "here": session.grant_book.get(grant.id) is not None,
        }
        for grant in listed(view.book.visible(view.session_id or ""))
    ]


async def command(session: CliSession, arguments: str) -> None:
    """``/grants`` and ``/grants revoke N``."""
    out = session.out
    number = _revoke_number(arguments)
    if number is None:
        out.print(session._text(USAGE, "error"))
        return
    try:
        rows = await (_remote_rows(session) if session.remote else _local_rows(session))
    except GatewayError as exc:
        out.print(session._text(f"✗ Could not read the grants: {exc}", "error"))
        return
    if number == 0:
        if not rows:
            out.print(session._text(NONE, "muted"))
            return
        for index, row in enumerate(rows, start=1):
            out.print(session._text(grant_line(index, row)))
        out.print(session._text(f"{RULE} /grants revoke N takes one back.", "muted"))
        return
    if number > len(rows):
        out.print(
            session._text(f"✗ There is no grant {number}; /grants lists them.", "error")
        )
        return
    row = rows[number - 1]
    grant_id = str(row.get("id") or "")
    if session.remote is not None:
        try:
            await session.remote.client.call("DELETE", f"/api/grants/{grant_id}")
        except GatewayError as exc:
            out.print(session._text(f"✗ Could not revoke it: {exc}", "error"))
            return
    elif not row.get("here"):
        out.print(session._text(FROM_MAIN, "error"))
        return
    else:
        try:
            await session.grant_book.revoke(grant_id)
        except SessionStoreError as exc:
            # The check stopped using it already; only the file kept it.
            out.print(session._text(f"⚠ Revoked here, but not saved: {exc}", "warn"))
            return
    out.print(session._text(f"Revoked: {row.get('label') or row.get('key')}.", "ok"))
