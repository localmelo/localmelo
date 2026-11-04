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

import os
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from localmelo.support.cli.approvals import grant_listed, grant_site
from localmelo.support.cli.main_chat import GatewayError
from localmelo.support.sessions.display import home_label
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


FROM_MAIN_FORGET: Final = (
    "✗ {name} was allowed in the main chat, so it counts everywhere and still holds "
    "here. Take it back with /grants in the main chat, or in the console's "
    "Settings → Permissions."
)
STILL_FROM_MAIN: Final = (
    "Allowed in the main chat, these still hold here: {names}. Take them back with "
    "/grants in the main chat, or in the console's Settings → Permissions."
)


def _names(name: str, held: frozenset[str]) -> set[str]:
    """The grants in *held* that ``/approvals forget NAME`` means: the name, or a site's."""
    site = name.lower().removeprefix("www.")
    return {
        grant
        for grant in held
        if grant == name or grant_site(grant).removeprefix("www.") == site
    }


async def forget(session: CliSession, name: str | None) -> None:
    """``/approvals forget [NAME]`` in the terminal's own chat.

    Takes back this chat's own grants (of NAME, or all) and says plainly when
    one of the main chat's still holds: those are taken back in the main chat
    or in the console. A revoke the file could not save is said too.
    """
    out, approvals = session.out, session.approvals
    approvals.forget(name)
    try:
        await session.grant_book.settle()
    except SessionStoreError as exc:
        out.print(session._text(f"⚠ Taken back here, but not saved: {exc}", "warn"))
    main = session.grant_book.outside()
    from_main = {g.key for g in main if g.kind != "folder"}
    main_folders = {g.key for g in main if g.kind == "folder"}
    still = approvals.allowed_tools & from_main
    still_folders = [f for f in approvals.allowed_folders if f in main_folders]
    if name is not None:
        if _names(name, frozenset(still)) or _folder_named(name, still_folders):
            out.print(session._text(FROM_MAIN_FORGET.format(name=name), "error"))
        return
    if still or still_folders:
        names = ", ".join(
            [grant_listed(grant) for grant in sorted(still)]
            + [home_label(folder) for folder in still_folders]
        )
        out.print(session._text(STILL_FROM_MAIN.format(names=names), "muted"))


def _folder_named(name: str, folders: list[str]) -> bool:
    """Whether *name* (a path, ``~/…`` or the folder as listed) is one of *folders*."""
    path = os.path.realpath(os.path.expanduser(name)).rstrip("/")
    return any(
        name in (folder, home_label(folder)) or path == folder for folder in folders
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
