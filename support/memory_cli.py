"""``melo memory``: the notes melo keeps, from the command line.

``list`` prints them newest first; ``forget [ID]`` forgets one note, or every note
and the index of every chat, after a question; ``status`` says what is on and how
much is kept; ``reindex`` rebuilds the chat index and gives every note a vector of
the embedding model now set; ``import`` offers the notes other assistants keep
(``--undo BATCH`` takes a batch back). The files are shared with the web console and the app.
``sleep [VARIANT]`` (developer mode only) tidies the notes by the ``[memory] sleep``
variant and prints what it changed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence

from localmelo.melo.memory.long import StoredEntry
from localmelo.support import command_words, config, onboard
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.recall import origin
from localmelo.support.sessions.store import (
    SessionStore,
    SessionStoreError,
    StoreOwnedError,
    default_path,
)

VERBS = ("list", "forget", "status", "reindex", "import", "sleep")


def note_origin(entry: StoredEntry) -> str:
    """Where a note came from, as the chat named it when the note was written."""
    label = str(entry.provenance.get("chat") or "a chat")
    return origin(label, False, entry.created_at)


def _confirm(question: str, keep: str, forget: str) -> bool:
    """Ask *question* as a list (typed number when piped); only the second answer forgets."""
    try:
        return onboard._ask_choice(question, [keep, forget], default=1) == 2
    except (EOFError, KeyboardInterrupt):
        return False


async def _list(count: int) -> int:
    memory = ChatMemory(config.load)
    try:
        entries = await memory.records()
    finally:
        await memory.aclose()
    if not entries:
        print(records.EMPTY)
        return 0
    for entry in entries[: count or None]:
        topic = records.TOPIC_LABELS.get(str(entry.provenance.get("topic")), "")
        print(f"{entry.id:>4}  {topic:<11}  {entry.text}")
        edited = " · edited" if entry.provenance.get("edited") else ""
        print(f"      {note_origin(entry)}{edited}")
    return 0


async def _held() -> tuple[list[StoredEntry], int]:
    """The notes, newest first, and how many chat turns the index holds."""
    memory = ChatMemory(config.load)
    try:
        return await memory.records(), (await memory.status())["documents"]
    finally:
        await memory.aclose()


async def _delete(note_id: int | None) -> None:
    """Forget one note, or (without *note_id*) every note and the chat index."""
    memory = ChatMemory(config.load)
    try:
        if note_id is None:
            await memory.forget_all()
        else:
            await memory.delete_record(note_id)
    finally:
        await memory.aclose()


def _forget(note_id: int | None, yes: bool) -> int:
    # The question is asked outside the event loop: a terminal list runs its own.
    entries, turns = asyncio.run(_held())
    if note_id is not None:
        entry = next((e for e in entries if e.id == note_id), None)
        if entry is None:
            print(f"melo: there is no note {note_id}", file=sys.stderr)
            return 1
        question, keep, forget = f"Forget “{entry.text}”?", "Keep it", "Forget it"
    else:
        question = (
            f"Forget {len(entries)} note{'s' if len(entries) != 1 else ''} and melo's "
            f"search index of {turns} chat turn{'s' if turns != 1 else ''}? "
            "Your chats stay."
        )
        keep, forget = "Keep everything", "Forget everything"
    if not yes and not _confirm(question, keep, forget):
        print("Nothing forgotten.")
        return 1
    asyncio.run(_delete(note_id))
    print(
        f"Forgot note {note_id}."
        if note_id is not None
        else "Forgot everything melo remembered. Your chats stay."
    )
    return 0


async def _status() -> int:
    memory = ChatMemory(config.load)
    try:
        status = await memory.status()
    finally:
        await memory.aclose()
    model = status["embedding_model"]
    rows = [
        ("memory", "on, across chats" if status["on"] else "off"),
        ("notes", str(status["records"])),
        ("chats", f"{status['documents']} turns indexed"),
        (
            "search",
            f"by meaning with {model}, and by words" if model else "by words only",
        ),
        ("folder", onboard._home_path(memory.directory)),
    ]
    if status["last_error"]:
        rows.append(("last error", status["last_error"]))
    for name, value in rows:
        print(f"{name:<11} {value}")
    return 0


async def _open_chats() -> SessionStore | int | None:
    """The chat file (``None`` when there is none), or the exit code when it cannot be read."""
    if not os.path.isfile(default_path("gateway")):
        return None
    chats = SessionStore(writer="gateway")
    try:
        await chats.open()
    except StoreOwnedError:
        print(
            "melo: the console or the app is running and owns the chats; "
            "stop it, then run this again.",
            file=sys.stderr,
        )
    except SessionStoreError as exc:
        print(f"melo: cannot read the chats: {exc}", file=sys.stderr)
    else:
        return chats
    await chats.aclose()
    return 1


async def _reindex() -> int:
    memory = ChatMemory(config.load)
    chats = await _open_chats()
    try:
        if isinstance(chats, int):
            return chats
        notes, turns = await memory.reindex(chats)
        _, model = memory.embedding()
        how = (
            f"with {model}" if model else "by words only (no embedding model is set up)"
        )
        print(f"Indexed {turns} chat turns and gave {notes} notes a vector, {how}.")
        return 0
    finally:
        if isinstance(chats, SessionStore):
            await chats.aclose()
        await memory.aclose()


async def _sleep(variant: str | None) -> int:
    memory = ChatMemory(config.load)
    chats = await _open_chats()
    try:
        if isinstance(chats, int):
            return chats
        try:
            report = await memory.sleep(variant, chats=chats)
        except NotImplementedError as exc:
            print(f"melo: {exc}", file=sys.stderr)
            return 1
        found = len(report.changes)
        print(
            f"Sleep {report.variant}: {found} change{'s' if found != 1 else ''}, "
            f"{report.model_calls} model calls, run {report.run_id}"
            + (f" ({report.aborted})" if report.aborted else "")
            + (" (more are left for the next run)" if report.capped else "")
        )
        for change in report.changes:
            before = ",".join(str(i) for i in change.before_ids) or "-"
            print(f"  {change.op}  {before} -> {change.after_id}  {change.text}")
        return 0
    finally:
        if isinstance(chats, SessionStore):
            await chats.aclose()
        await memory.aclose()


def main(argv: Sequence[str], prog: str = "melo memory") -> int:
    rest = list(argv)
    if rest and not rest[0].startswith("-") and rest[0] not in VERBS:
        command_words.refuse("memory", rest)
    parser = argparse.ArgumentParser(
        prog=prog, description="See and change the notes melo keeps."
    )
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="VERB")
    verbs.add_parser("list", help="print the notes, newest first").add_argument(
        "-n", type=int, default=0, metavar="N", help="only the newest N"
    )
    forget = verbs.add_parser(
        "forget", help="forget one note, or everything after a question"
    )
    forget.add_argument("id", nargs="?", type=int, help="the note's number (see list)")
    forget.add_argument("-y", "--yes", action="store_true", help="do not ask")
    verbs.add_parser("status", help="what is on and how much is kept")
    verbs.add_parser("reindex", help="rebuild the chat index from the chats")
    bring = verbs.add_parser(
        "import", help="bring in the notes other assistants keep, after a question"
    )
    bring.add_argument(
        "-u", "--undo", metavar="BATCH", help="take back the notes of an earlier import"
    )
    try:
        developer = config.load().ui.mode == "developer"
    except Exception:  # a broken config file must not stop the other verbs
        developer = False
    if developer:
        tidy = verbs.add_parser("sleep", help="tidy the notes (developer mode)")
        tidy.add_argument("variant", nargs="?", choices=config.MEMORY_SLEEP)
    elif rest[:1] == ["sleep"]:
        print('melo: "memory sleep" needs [ui] mode = "developer"', file=sys.stderr)
        return 2
    args = parser.parse_args(rest)
    if args.verb == "list":
        return asyncio.run(_list(args.n))
    if args.verb == "forget":
        return _forget(args.id, args.yes)
    if args.verb == "status":
        return asyncio.run(_status())
    if args.verb == "sleep":
        return asyncio.run(_sleep(args.variant))
    if args.verb == "import":
        from localmelo.support.cli import memory_import

        return memory_import.command(args.undo)
    return asyncio.run(_reindex())
