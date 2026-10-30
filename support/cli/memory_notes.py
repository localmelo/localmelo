"""``/memory``: the notes melo keeps, as lists in the terminal.

A list of the notes (newest first; type to filter), then a list of what to do
with the one chosen: go back, edit it, delete it. The last row of the first list
forgets everything, after a list that names what goes and keeps everything first.
The files are the ones the web console and the app use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from localmelo.melo.memory.long import StoredEntry
from localmelo.support import config
from localmelo.support.cli.picker import FORGET_ROW, NoteRow
from localmelo.support.cli.text import cut_cells
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory_cli import note_origin

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession


def rows(entries: list[StoredEntry]) -> list[NoteRow]:
    return [
        NoteRow(
            entry.id,
            entry.text,
            records.TOPIC_LABELS.get(str(entry.provenance.get("topic")), ""),
            note_origin(entry),
        )
        for entry in entries
    ]


async def manage(session: CliSession) -> None:
    """Pick a note and edit or delete it, until the list is closed."""
    app = session.app
    assert app is not None
    memory = ChatMemory(config.load)
    try:
        while True:
            entries = await memory.records()
            if not entries:
                session.out.print(session._text(records.EMPTY))
                return
            note = await app.choose_note(rows(entries))
            if note is None:
                return
            if note is FORGET_ROW:
                await _forget_all(session, memory, len(entries))
            else:
                await _one(session, memory, note)
    finally:
        await memory.aclose()


async def _one(session: CliSession, memory: ChatMemory, note: NoteRow) -> None:
    app = session.app
    assert app is not None
    ellipsis = session.renderer.symbols.ellipsis
    shown = cut_cells(" ".join(note.text.split()), 60, ellipsis)
    answer = await app.choose_from(f"“{shown}”", ["Back", "Edit it", "Delete it"])
    if answer == 1:
        text = await app.ask("New wording", note.text)
        if text is None or not text.strip() or text.strip() == note.text:
            return
        try:
            await memory.edit_record(note.id, text)
        except ValueError as exc:
            session.out.print(session._text(str(exc)))
        else:
            session.out.print(session._text(f"Saved: {text.strip()}"))
    elif answer == 2:
        await memory.delete_record(note.id)
        session.out.print(session._text(f"Forgot: {shown}"))


async def _forget_all(session: CliSession, memory: ChatMemory, notes: int) -> None:
    app = session.app
    assert app is not None
    turns = (await memory.status())["documents"]
    question = (
        f"Forget {notes} note{'s' if notes != 1 else ''} and melo's search index "
        f"of {turns} chat turn{'s' if turns != 1 else ''}? Your chats stay."
    )
    if await app.choose_from(question, ["Keep everything", "Forget everything"]) == 1:
        await memory.forget_all()
        session.out.print(session._text("Forgot everything melo remembered."))
