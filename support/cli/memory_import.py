"""Bring in what other assistants keep about you: the first run's offer, ``melo memory import`` and ``/memory import``.

A scan lists the files under the home folder that other assistants keep and writes nothing. The person then
chooses from a list: bring the notes in, show what the files say, or not now. Nothing is brought in without
that choice: an arrow-key list on a terminal (it starts on "Show what they say"), a typed number when stdin
is piped. No answer (end of input, esc, ctrl+c) is not a choice either: it leaves, brings nothing in and
records nothing, so the offer is made again; only the "Not now" row ends it. The first run offers once per
computer (``ChatMemory.import_pending``), only to a person at a terminal; a direct run never reaches it. Files
show as paths, with a few sample lines and what was left out, cleaned of terminal controls; nothing else of a
file is printed. The notes are the ones the web console and the app use.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from localmelo.support import config, onboard
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.importer import ImportPlan
from localmelo.support.sessions.display import clean_output

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession

log = logging.getLogger(__name__)

TITLE = "melo found notes on this Mac from other assistants"
NONE = "No notes from other assistants were found on this Mac."
KNOWN = "Nothing new: melo already has what those files say."
CANCELLED = "Nothing brought in."
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
    "capped": "over the limit",
}


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _reasons(filtered: dict[str, int]) -> str:
    """The words for the reasons that left lines out, in the order the scan lists them."""
    return ", ".join(
        LEFT_OUT.get(reason, reason) for reason, n in filtered.items() if n
    )


def _tilde(path: str) -> str:
    """*path* with the home folder as ``~``; the home is written as it resolves, as the scan's paths are."""
    for home in dict.fromkeys((str(Path.home().resolve()), str(Path.home()))):
        if path == home or path.startswith(home + os.sep):
            return "~" + path[len(home) :]
    return path


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


def empty(plan: ImportPlan | None) -> str:
    """What to say when no file holds a note to bring in (*plan* None: memory is off).

    "Nothing new" is for files with lines melo already has. When every line was left out for another reason
    (secrets, other people's details), that is said instead, so the person is not told something untrue.
    """
    if plan is None:
        return OFF
    left: dict[str, int] = {}
    for source in plan.sources:
        for reason, n in source["filtered"].items():
            left[reason] = left.get(reason, 0) + n
    left = {reason: n for reason, n in left.items() if n}
    if "duplicate" in left:
        return KNOWN
    if left:
        return f"Nothing to bring in: the lines were left out ({_reasons(left)})."
    return NONE


def describe(plan: ImportPlan) -> str:
    """Each file with its path, notes, sample lines and what was left out, as a terminal would show them."""
    lines: list[str] = []
    for source in plan.sources:
        kept = _count(source["items_kept"], "note")
        lines.append(clean_output(f"{_tilde(source['path'])} · {kept}"))
        lines += [f"    “{clean_output(line)}”" for line in source["sample"]]
        left = {k: n for k, n in source["filtered"].items() if n}
        if left:
            lines.append(f"    {sum(left.values())} left out: {_reasons(left)}")
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


async def _undo(batch_id: str) -> tuple[int, int]:
    """How many notes of the batch were deleted, and how many the person had edited, which stay."""
    memory = ChatMemory(config.load)
    try:
        removed = await memory.undo_import(batch_id)
        return removed, await memory.import_kept(batch_id)
    finally:
        await memory.aclose()


def undo(batch_id: str) -> int:
    """``melo memory import --undo``: delete a batch's notes; the exit code."""
    removed, kept = asyncio.run(_undo(batch_id))
    if not removed and not kept:
        print(f"melo: no notes are left from the import {batch_id}")
        return 1
    took = (
        f"Took back {_count(removed, 'note')}." if removed else "Nothing to take back."
    )
    print(f"{took} Kept {_count(kept, 'note')} you edited." if kept else took)
    return 0


def _ask(rows: list[str], default: int) -> int:
    """The row chosen (from 0), outside an event loop: a list on a terminal, a typed number when piped.

    No answer is no choice: end of input and ctrl+d raise ``EOFError``, ctrl+c and esc ``KeyboardInterrupt``.
    """
    return onboard._ask_choice(TITLE, rows, default=default + 1) - 1


def offer(*, quiet: bool = False) -> int:
    """Scan, then ask until the person brings the notes in or says not now; the exit code.

    *quiet* (the first run) says nothing when there is nothing to offer. The list starts on "Show what they
    say", and on "Bring in" once the files were shown. A question left unanswered raises ``EOFError`` or
    ``KeyboardInterrupt`` with nothing brought in and nothing recorded.
    """
    plan = asyncio.run(_scan())
    sources = usable(plan) if plan is not None else []
    if plan is None or not sources:
        if not quiet:
            print(empty(plan))
        return 0
    rows = menu(sources)
    default = 1
    while True:
        pick = _ask(rows, default)
        if pick == 1:
            print(describe(plan))
            default = 0
        elif pick == 0:
            ids = [source["id"] for source in sources]
            print(done(asyncio.run(_bring(plan, ids)), plan.batch_id))
            return 0
        else:
            ChatMemory(config.load).import_offered("declined")
            return 0


def command(undo_batch: str | None = None) -> int:
    """``melo memory import [--undo BATCH]``: the exit code, 130 for ctrl+c or esc, as setup ends."""
    if undo_batch:
        return undo(undo_batch)
    try:
        return offer()
    except EOFError:
        print(f"\n{CANCELLED}")
        return 0
    except KeyboardInterrupt:
        print(f"\n{CANCELLED}")
        return 130


def first_run() -> None:
    """Offer once, before the chat opens, to a person at a terminal; it never stops melo.

    ``EOFError`` and ``KeyboardInterrupt`` (no answer) pass through: the caller ends melo as a cancelled
    setup ends, and the offer is made again next time.
    """
    try:
        if ChatMemory(config.load).import_pending():
            offer(quiet=True)
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:  # a broken memory folder must not keep the person out of melo
        log.info("The offer to bring in notes failed.", exc_info=True)


async def bring_in(session: CliSession) -> None:
    """``/memory import`` in a chat: the same choice, in melo's own screen. Esc leaves without recording."""
    app = session.app
    assert app is not None
    memory = ChatMemory(config.load)
    try:
        plan = await memory.import_scan() if memory.enabled else None
        sources = usable(plan) if plan is not None else []
        if plan is None or not sources:
            session.out.print(session._text(empty(plan)))
            return
        default = 1
        while True:
            pick = await app.choose_from(TITLE, menu(sources), default=default)
            if pick is None:
                return
            if pick == 1:
                session.out.print(session._text(describe(plan)))
                default = 0
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
