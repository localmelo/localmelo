"""Bring in what other assistants keep about you: the first run's offer, ``melo memory import`` and ``/memory import``.

A scan lists the files under the home folder that other assistants keep and writes nothing. The person then
chooses from a list: bring the notes in, show what the files say, or not now. Nothing is brought in without
that choice: an arrow-key list on a terminal, a typed number when stdin is piped, and no answer (end of input,
esc, ctrl+c) is "Not now". The first run offers once per computer (``ChatMemory.import_pending``), only to a
person at a terminal; a direct run never reaches it. Files show as paths, with a few sample lines and what was
left out; nothing else of a file is printed. The notes are the ones the web console and the app use.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from localmelo.support import config, onboard
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.importer import ImportPlan

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession

log = logging.getLogger(__name__)

TITLE = "melo found notes on this Mac from other assistants"
NONE = "No notes from other assistants were found on this Mac."
KNOWN = "Nothing new: melo already has what those files say."
OFF = (
    "Memory across chats is off: melo config set memory.across_chats true turns it on."
)
# The words of the web console's card for what a file left out.
LEFT_OUT = {
    "secret": "secrets",
    "third_party": "other people's details",
    "wrong_owner": "lines about someone else",
    "duplicate": "repeats",
    "mechanics": "how the other tool works",
}


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def usable(plan: ImportPlan) -> list[dict[str, Any]]:
    """The files with a note to bring in."""
    return [source for source in plan.sources if source["items_kept"]]


def menu(sources: list[dict[str, Any]]) -> list[str]:
    """The rows of the choice: bring in, show what they say, not now."""
    notes = sum(source["items_kept"] for source in sources)
    return [
        f"Bring in {_count(notes, 'note')} from {_count(len(sources), 'file')}",
        "Show what they say",
        "Not now",
    ]


def describe(plan: ImportPlan) -> str:
    """Each file with its path, notes, sample lines and what was left out."""
    lines: list[str] = []
    for source in plan.sources:
        kept = _count(source["items_kept"], "note")
        lines.append(f"{onboard._home_path(source['path'])} · {kept}")
        lines += [f"    “{line}”" for line in source["sample"]]
        left = {k: n for k, n in source["filtered"].items() if n}
        if left:
            reasons = ", ".join(LEFT_OUT.get(k, k) for k in left)
            lines.append(f"    {sum(left.values())} left out: {reasons}")
    return "\n".join(lines)


def done(written: int, batch_id: str) -> str:
    if not written:
        return "Nothing new: melo already knows these."
    return (
        f"Brought in {_count(written, 'note')}. melo memory list shows them; "
        f"melo memory import --undo {batch_id} takes them back."
    )


async def _scan() -> ImportPlan | None:
    memory = ChatMemory(config.load)
    try:
        return await memory.import_scan() if memory.enabled else None
    finally:
        await memory.aclose()


async def _bring(plan: ImportPlan, source_ids: list[str]) -> int:
    memory = ChatMemory(config.load)
    try:
        return len((await memory.import_apply(plan, source_ids))["written_ids"])
    finally:
        await memory.aclose()


async def _undo(batch_id: str) -> int:
    memory = ChatMemory(config.load)
    try:
        return await memory.undo_import(batch_id)
    finally:
        await memory.aclose()


def undo(batch_id: str) -> int:
    """``melo memory import --undo``: delete a batch's notes; the exit code."""
    removed = asyncio.run(_undo(batch_id))
    if not removed:
        print(f"melo: no notes are left from the import {batch_id}")
        return 1
    print(f"Took back {_count(removed, 'note')}.")
    return 0


def _ask(rows: list[str]) -> int:
    """The row chosen (from 0), outside an event loop: a list on a terminal, a typed number when piped."""
    try:
        return onboard._ask_choice(TITLE, rows, default=1) - 1
    except (EOFError, KeyboardInterrupt):
        return len(rows) - 1


def offer(*, quiet: bool = False) -> int:
    """Scan, then ask until the person brings the notes in or says not now; the exit code.

    *quiet* (the first run) says nothing when there is nothing to offer.
    """
    plan = asyncio.run(_scan())
    sources = usable(plan) if plan is not None else []
    if plan is None or not sources:
        if not quiet:
            print(OFF if plan is None else KNOWN if plan.sources else NONE)
        return 0
    rows = menu(sources)
    while True:
        pick = _ask(rows)
        if pick == 1:
            print(describe(plan))
        elif pick == 0:
            ids = [source["id"] for source in sources]
            print(done(asyncio.run(_bring(plan, ids)), plan.batch_id))
            return 0
        else:
            ChatMemory(config.load).import_offered("declined")
            return 0


def first_run() -> None:
    """Offer once, before the chat opens, to a person at a terminal; it never stops melo."""
    try:
        if ChatMemory(config.load).import_pending():
            offer(quiet=True)
    except Exception:  # a broken memory folder must not keep the person out of melo
        log.info("The offer to bring in notes failed.", exc_info=True)


async def bring_in(session: CliSession) -> None:
    """``/memory import`` in a chat: the same choice, in melo's own screen."""
    app = session.app
    assert app is not None
    memory = ChatMemory(config.load)
    try:
        plan = await memory.import_scan() if memory.enabled else None
        sources = usable(plan) if plan is not None else []
        if plan is None or not sources:
            session.out.print(
                session._text(OFF if plan is None else KNOWN if plan.sources else NONE)
            )
            return
        while True:
            pick = await app.choose_from(TITLE, menu(sources))
            if pick == 1:
                session.out.print(session._text(describe(plan)))
            elif pick == 0:
                ids = [source["id"] for source in sources]
                written = len((await memory.import_apply(plan, ids))["written_ids"])
                session.out.print(session._text(done(written, plan.batch_id)))
                return
            else:
                memory.import_offered("declined")
                return
    finally:
        await memory.aclose()
