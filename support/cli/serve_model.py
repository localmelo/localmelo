"""``/model`` in a chat whose model server is melo_serve: its chat models, and one switch.

``/model`` lists the chat models melo_serve offers, as the console and the app
list them (:func:`~localmelo.support.melo_serve.pick_words` words each row);
enter switches. ``/model <name>`` switches with no list. Every chat on this Mac
follows the switch.

The switch goes through the gateway when one runs on this Mac (the main chat's,
or one its endpoint file names), with the token the main chat's client already
uses: the gateway waits for its chats' turns first, and the terminal reads the
switch's progress from its health. With no gateway, the terminal asks melo_serve
itself. Esc stops waiting: a wait in the gateway is cancelled, and a load that
has started goes on in melo_serve.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Final

from rich.text import Text

from localmelo.support import gateway_endpoint, melo_serve, task_cli
from localmelo.support.cli.main_chat import GatewayClient, GatewayError
from localmelo.support.words import t_in

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession

# Seconds between two reads of the gateway's switch.
POLL_S: Final = 0.5
# Health reads in a row that may fail before the terminal stops following.
MISSED_READS: Final = 5
ACTIVE: Final = ("waiting", "loading")
# A switch whose end the terminal cannot read: nothing says it failed.
UNKNOWN: Final = "unknown"
STOP_KEY: Final = "esc stop waiting"


def applies(session: CliSession) -> bool:
    """Whether the chat's model server is melo_serve, whose switch every chat follows."""
    return session.factory.backend.key == "melo_serve"


def find(picks: list[melo_serve.Pick], typed: str) -> melo_serve.Pick | None:
    """The pick *typed* names: its short name, id, name or served model, any case."""
    wanted = typed.strip().casefold()
    for pick in picks:
        if wanted in (
            n.casefold() for n in (pick.short, pick.id, pick.name, pick.model)
        ):
            return pick
    return None


async def command(session: CliSession, arguments: str) -> None:
    """``/model`` and ``/model NAME`` for melo_serve."""
    try:
        picks = await melo_serve.picks()
    except melo_serve.MeloServeError as exc:
        session.out.print(session._text(f"✗ {exc}", "error"))
        return
    current = next((pick for pick in picks if pick.current), None)
    still = current.name if current else session.factory.backend.model
    if arguments:
        found = find(picks, arguments)
        if found is None:
            names = ", ".join(pick.short or pick.id for pick in picks)
            session.out.print(
                session._text(
                    f"✗ melo_serve offers no model called '{arguments}' ({names}).",
                    "error",
                )
            )
            return
    elif session.app is None:
        return
    else:
        found = await session.app.choose_pick(picks, foot=t_in("en", "serve.foot"))
        if found is None:
            session.out.print(
                session._text(f"Cancelled. Still using {still}.", "muted")
            )
            return
    if found.current:
        session.out.print(session._text(f"Still using {found.name}.", "muted"))
    elif not found.loadable:
        why = melo_serve.pick_words(found)[0]
        session.out.print(session._text(f"✗ {found.name} cannot load: {why}.", "error"))
    else:
        await _switch(session, found, still)


async def _gateway(session: CliSession) -> tuple[GatewayClient | None, bool]:
    """The gateway to switch through, and whether this call opened it (and closes it).

    The main chat's; else one that runs on this Mac, reached with the gateway's
    own token; None when none runs or its token cannot be read.
    """
    if session.remote is not None:
        return session.remote.client, False
    found = await asyncio.to_thread(gateway_endpoint.live)
    if found is None:
        return None, False
    try:
        url = task_cli.address(found.url)  # the token goes to this Mac only
        headers = task_cli.client_headers(url)
    except (ValueError, task_cli.ClientError):
        return None, False
    return GatewayClient(url, headers=headers), True


def _show(session: CliSession, pick: melo_serve.Pick, switch: dict[str, Any]) -> None:
    """The activity line: what the switch waits for, what it may take, and Esc."""
    waiting = switch.get("waiting_for")
    if switch.get("state") == "waiting" and isinstance(waiting, int) and waiting:
        session.activity = t_in("en", "serve.waiting", n=waiting)
        session.activity_note = STOP_KEY
    else:
        session.activity = f"Loading {pick.name}"
        about = f"about {round(pick.load_s)} s · " if pick.load_s else ""
        session.activity_note = about + STOP_KEY
    if session.app is not None:
        session.app.invalidate()


async def _through(
    session: CliSession,
    client: GatewayClient,
    pick: melo_serve.Pick,
    seen: dict[str, Any],
) -> dict[str, Any]:
    """Start the switch in the gateway and follow it on its health until it ends."""
    body = await client.call("POST", "/api/serve/switch", json={"pick": pick.id})
    started = body.get("switch")
    switch: dict[str, Any] = started if isinstance(started, dict) else {}
    number = switch.get("id")
    missed = 0
    while True:
        seen.update(switch)
        _show(session, pick, switch)
        if switch.get("state") not in ACTIVE:
            return switch
        await asyncio.sleep(POLL_S)
        try:
            found = (await client.call("GET", "/api/health")).get("switch")
        except GatewayError:
            missed += 1
            if missed == MISSED_READS:
                return {"state": UNKNOWN}
            continue
        missed = 0
        if not isinstance(found, dict) or found.get("id") != number:
            return {"state": UNKNOWN}
        switch = found


async def _switch(session: CliSession, pick: melo_serve.Pick, still: str) -> None:
    """Switch melo_serve to *pick*, then show the model every chat now uses.

    The session is "switching" from the moment ``/model`` was sent, so no message
    can start a turn under it; the command's end leaves that state.
    """
    client, opened = await _gateway(session)
    seen: dict[str, Any] = {"state": "waiting" if client else "loading"}
    try:
        if client is None:
            _show(session, pick, seen)
            snap = await asyncio.to_thread(melo_serve.snapshot)
            await melo_serve.switch(pick.id, expected_generation=snap.generation)
            ended: dict[str, Any] = {"state": "done"}
        else:
            ended = await _through(session, client, pick, seen)
    except asyncio.CancelledError:
        await _stopped(session, client, pick, seen, still)
        return
    except (melo_serve.MeloServeError, GatewayError) as exc:
        if isinstance(exc, GatewayError) and not exc.code:
            _unknown(session)  # no answer: the gateway may have started it
        else:
            session.out.print(session._text(f"✗ {exc}. Still using {still}.", "error"))
        return
    finally:
        session.activity, session.activity_note = None, ""
        if opened and client is not None:
            await client.aclose()
    await _ended(session, pick, ended, still)


def _unknown(session: CliSession) -> None:
    """The gateway stopped answering: the switch may go on, so nothing is claimed."""
    session.out.print(
        session._text(
            "✗ The gateway stopped answering. The switch may still be running; "
            "/model shows what is served.",
            "error",
        )
    )


async def _stopped(
    session: CliSession,
    client: GatewayClient | None,
    pick: melo_serve.Pick,
    seen: dict[str, Any],
    still: str,
) -> None:
    """Esc: a wait in the gateway is cancelled; a load goes on in melo_serve."""
    if client is not None and seen.get("state") == "waiting":
        with contextlib.suppress(GatewayError):
            await client.call("POST", "/api/serve/switch/cancel")
            session.out.print(
                session._text(f"Switch cancelled. Still using {still}.", "muted")
            )
            return
    session.out.print(
        session._text(
            f"Stopped waiting. {pick.name} goes on loading in melo_serve; "
            "every chat follows once it is ready.",
            "muted",
        )
    )


async def _ended(
    session: CliSession, pick: melo_serve.Pick, ended: dict[str, Any], still: str
) -> None:
    state = ended.get("state")
    if state == UNKNOWN:
        _unknown(session)
        return
    if state == "cancelled":
        session.out.print(
            session._text(f"The switch was cancelled. Still using {still}.", "muted")
        )
        return
    if state != "done":
        error = str(ended.get("error") or "the switch did not finish").rstrip(".")
        session.out.print(session._text(f"✗ {error}. Still using {still}.", "error"))
        return
    if session.remote is None:
        # The terminal's own chat moves to the new model now, as at its next turn:
        # still "switching" until it has, and past Esc, as CliSession._switch is.
        session._work = None
        try:
            await session._carry_over(await session.factory.build())
        except Exception as exc:
            session._start_failed(pick.model or pick.name, exc)
            return
        finally:
            session._state = "idle"
    else:
        session.factory.show_served()
    line = Text()
    line.append_text(session._text(session.renderer.symbols.ok, "ok"))
    line.append_text(session._text(f" Switched to {pick.name}"))
    if pick.backend_name:
        line.append_text(session._text(f" · {pick.backend_name}", "muted"))
    session.out.print(line)
