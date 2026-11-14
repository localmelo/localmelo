"""``/model`` once a hosted service has a key: the one list of every model that can be used now.

The list is the page's (``GET /api/chat-models``): the local server's models and each
service's that has a key, the one in use marked. Choosing one switches the chat model
for the next message, with no key typed again, no test and no restart; ``/model NAME``
(or ``/model service:NAME``) does it without the list. The gateway that runs on this
Mac does the work, as for melo_serve's switch, so every surface sees the same choice.
With no hosted key, or no gateway, the command is the one it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from localmelo.support import config, model_choices
from localmelo.support.cli import serve_model
from localmelo.support.cli.picker import ModelRow

if TYPE_CHECKING:
    from localmelo.support.cli.main_chat import GatewayClient
    from localmelo.support.cli.session import CliSession

_JOIN = " · "


@dataclass(frozen=True)
class Row:
    """One model of the list: the group it belongs to, and the line that shows it."""

    group: str
    model: str
    label: str
    names: tuple[str, ...]


def applies(session: CliSession) -> bool:
    """Whether a hosted service has a key, so ``/model`` lists every server and service.

    melo_serve keeps its own switch (:mod:`.serve_model`), and a session that was not
    started from the saved config has no hosted service to follow. With no hosted key
    nothing changes: ``/model`` is the one of the model server.
    """
    return (
        not serve_model.applies(session)
        and session.factory.configured
        and bool(model_choices.keyed(config.load()))
    )


def rows_of(groups: list[dict[str, Any]]) -> list[Row]:
    """The selectable rows, in the list's order; a group whose listing failed has none."""
    rows = []
    for group in groups:
        if group.get("problem"):
            continue
        for model in [""] if group.get("serves") else group.get("models", []):
            label = f"{group['label']}{_JOIN}{model}" if model else str(group["label"])
            known = (
                (model, f"{group['id']}:{model}", label)
                if model
                else (str(group["id"]), label)
            )
            rows.append(Row(str(group["id"]), str(model), label, known))
    return rows


def find(rows: list[Row], typed: str) -> Row | None:
    """The row *typed* names: its model, ``service:model``, its line or a server's name."""
    wanted = typed.strip().casefold()
    return next(
        (row for row in rows if wanted in (name.casefold() for name in row.names)),
        None,
    )


def _in_use(row: Row, now: dict[str, Any]) -> bool:
    return row.group == now.get("group") and row.model in (now.get("model"), "")


async def command(session: CliSession, arguments: str) -> bool:
    """``/model`` and ``/model NAME``; False when no gateway runs, so the old command goes on."""
    client, opened = await serve_model._gateway(session)
    if client is None:
        return False
    try:
        await _switch(session, client, arguments)
    finally:
        if opened:
            await client.aclose()
    return True


async def _switch(session: CliSession, client: GatewayClient, arguments: str) -> None:
    body = await client.call("GET", "/api/chat-models")
    groups = [group for group in body.get("groups", []) if isinstance(group, dict)]
    now = body.get("current") or {}
    for group in groups:
        if group.get("problem"):
            session.out.print(
                session._text(f"⚠ {group['label']}: {group['problem']}", "warn")
            )
    rows = rows_of(groups)
    if arguments:
        found = find(rows, arguments)
        if found is None:
            session.out.print(
                session._text(
                    f"✗ No listed model is called '{arguments}'. /model lists them.",
                    "error",
                )
            )
            return
    else:
        if session.app is None or not rows:
            return
        current = next((row.label for row in rows if _in_use(row, now)), "")
        picked = await session.app.choose_model(
            [ModelRow(row.label) for row in rows],
            where="Every server and service with a key",
            current=current,
        )
        if picked is None:
            session.out.print(
                session._text(
                    f"Cancelled. Still using {current or 'the same model'}.", "muted"
                )
            )
            return
        found = next(row for row in rows if row.label == picked)
    if _in_use(found, now):
        session.out.print(session._text(f"Still using {found.label}.", "muted"))
        return
    await client.call(
        "POST",
        "/api/chat-models/switch",
        json={"group": found.group, "model": found.model},
    )
    session.out.print(
        session._text(f"Model: {found.label}. Saved as the default.", "ok")
    )
    await _follow(session, found)


async def _follow(session: CliSession, row: Row) -> None:
    """Move the terminal's row, and its own chat, to what the switch saved.

    The main chat's Agent is the gateway's: it builds it at the next message.
    """
    session.factory.follow_saved(config.load())
    if session.remote is not None:
        return
    try:
        await session._carry_over(await session.factory.build())
    except Exception as exc:
        session.out.print(
            session._text(
                f"✗ Could not start {row.model or row.label}: {exc}. The choice is saved.",
                "error",
            )
        )
