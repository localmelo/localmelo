"""API v2 memory across chats: the notes melo keeps, and the switch that turns them on.

The notes are what the user said about themselves (see
:mod:`localmelo.support.memory`). ``GET /api/memory/records`` lists them newest
first (``?session_id=`` keeps one chat's), each with where it came from: a note
outlives a deleted chat, and says so. ``PATCH /api/memory/records/{id}`` rewords
one, ``DELETE /api/memory/records/{id}`` removes it and its vector and blanks the
text of its decision rows (with ``?undo=true``, a turn's "Undo", the note it
replaced is current again), and ``DELETE /api/memory/records?all=true`` forgets
every note and the index of every chat. ``GET /api/memory/import/scan`` lists
what other assistants' files hold (no writes), ``POST /api/memory/import`` writes
the chosen sources of a scan as notes (the ``memory.import`` event is in the
response: memory has no session channel) and ``DELETE /api/memory/import/{batch_id}``
takes the batch back (``kept`` counts the notes the person had reworded, which stay; only
present when above 0); ``GET /api/memory/import/offer`` says whether the offer to do it
was never made (the first run's card) and ``DELETE`` on it records "Not now" and answers
whether the offer is still open (it is when the record could not be written).
``PUT /api/memory`` switches
``[memory] across_chats`` and ``POST /api/memory/reindex`` rebuilds the index
from the chats. In developer mode (``[ui] mode = "developer"``; anywhere else
404), ``POST /api/memory/sleep`` runs one sleep (:mod:`localmelo.support.memory.sleep`)
and answers its report, and ``DELETE /api/memory/sleep/{run_id}`` undoes that run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from localmelo.melo.memory.long import StoredEntry
from localmelo.support import config
from localmelo.support.gateway.i18n import current, t
from localmelo.support.memory.recall import chat_label, chat_title, day

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console


class EditBody(BaseModel):
    """Request body of ``PATCH /api/memory/records/{id}``."""

    model_config = ConfigDict(extra="forbid")

    text: StrictStr


class SleepBody(BaseModel):
    """Request body of ``POST /api/memory/sleep``: the variant, else ``[memory] sleep``."""

    model_config = ConfigDict(extra="forbid")

    variant: StrictStr | None = None
    budget: StrictInt | None = Field(default=None, ge=1, le=100)


class ImportBody(BaseModel):
    """Request body of ``POST /api/memory/import``."""

    model_config = ConfigDict(extra="forbid")

    batch_id: StrictStr
    source_ids: list[StrictStr]


class SwitchBody(BaseModel):
    """Request body of ``PUT /api/memory``."""

    model_config = ConfigDict(extra="forbid")

    on: StrictBool


async def _chats(console: Console, ids: set[str]) -> dict[str, Any]:
    """The chats that still exist among *ids* (session id to its info)."""
    found: dict[str, Any] = {}
    if console.store_state() != "ok":
        return found
    for session_id in ids:
        info = await console.store.get_session(session_id)
        if info is not None:
            found[session_id] = info
    return found


def _on(at: float) -> str:
    """The day of *at* in the reader's language."""
    when = time.localtime(at)
    month = time.strftime("%b", when) if current() == "en" else when.tm_mon
    return t("memoryRows.day", month=month, day=when.tm_mday)


def _from_words(info: Any, label: str, gone: bool, at: float, surface: str) -> str:
    """Where a note came from, in the reader's language; in English, :func:`origin`'s words."""
    on = _on(at)
    if gone:
        return t("memoryRows.fromDeleted", day=on)
    if info is None:
        return t("memoryRows.from", chat=label, day=on)
    title = chat_title(info.title)
    kind = {"front": "main", "task": "task", "front_bg": "mainTask"}.get(
        info.kind, "terminal" if surface == "cli" and info.kind == "chat" else "chat"
    )
    chat = (
        t("memoryRows.chat.main")
        if kind == "main"
        else (
            t(f"memoryRows.chat.{kind}", title=title)
            if title
            else t(f"memoryRows.chat.{kind}Untitled")
        )
    )
    return t("memoryRows.from", chat=chat, day=on)


def _from_file(prov: dict[str, Any], at: float) -> str:
    """Where an imported note came from: its file's name and the day."""
    name = Path(str(prov.get("path") or "")).name or str(prov.get("chat") or "")
    return t("memoryRows.fromFile", name=name, day=_on(at))


def _row(
    entry: StoredEntry, console: Console, chats: dict[str, Any] | None
) -> dict[str, Any]:
    """A note as the page and the clients read it."""
    prov = entry.provenance
    surface = console.memory.surface
    label = str(prov.get("chat") or chat_label("chat", "", surface))
    gone = (
        chats is not None and bool(entry.session_id) and entry.session_id not in chats
    )
    info = chats.get(entry.session_id) if chats else None
    if info is not None:
        label = chat_label(info.kind, info.title, surface)
    imported = prov.get("source") == "import"
    return {
        "id": entry.id,
        "text": entry.text,
        "topic": str(prov.get("topic") or ""),
        "quote": str(prov.get("quote") or ""),
        "source": f"{'imported from' if imported else 'you said it ·'} {label}, "
        f"{day(entry.created_at)}",
        "imported": imported,
        "batch_id": str(prov.get("batch_id") or ""),
        "from": (
            _from_file(prov, entry.created_at)
            if imported
            else _from_words(info, label, gone, entry.created_at, surface)
        ),
        "chat_gone": gone,
        "edited": bool(prov.get("edited")),
        "session_id": entry.session_id,
        "created_at": entry.created_at,
    }


async def _identity_changed(console: Console, pinned: str) -> None:
    """Rebuild idle chats' Agents when the notes their instructions pin changed.

    A chat's instructions list the user's About-you and Preferences notes in
    their words when its Agent is built; a note that was reworded or deleted
    must not stay in a chat that is already open.
    """
    if await console.memory.pinned_quotes() != pinned:
        await console.sessions.settings_changed()


def _developer(console: Console) -> None:
    """404 unless developer mode (or the console's ``dev_routes``) is on."""
    if console.cfg.ui.mode != "developer" and not console.options.dev_routes:
        raise ApiError(404, "not_found", t("errors.notFound"))


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the memory routes on *app*."""

    @app.get("/api/memory/records")
    async def memory_records(session_id: str = Query("")) -> dict[str, Any]:
        memory = console.memory
        entries = [
            entry
            for entry in await memory.records()
            if not session_id or entry.session_id == session_id
        ]
        chats = (
            await _chats(console, {e.session_id for e in entries}) if entries else {}
        )
        known = console.store_state() == "ok"
        status = await memory.status()
        return {
            "on": memory.enabled,
            "records": [_row(e, console, chats if known else None) for e in entries],
            "documents": status["documents"],
            "embedding_model": status["embedding_model"],
            "retriever": status["retriever"],
            "last_error": status["last_error"],
        }

    @app.patch("/api/memory/records/{record_id}")
    async def memory_record_edit(record_id: int, body: EditBody) -> dict[str, Any]:
        pinned = await console.memory.pinned_quotes()
        try:
            entry = await console.memory.edit_record(record_id, body.text)
        except ValueError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from exc
        if entry is None:
            raise ApiError(404, "not_found", t("errors.noteGone"))
        await _identity_changed(console, pinned)
        chats = await _chats(console, {entry.session_id})
        return _row(entry, console, chats if console.store_state() == "ok" else None)

    @app.delete("/api/memory/records/{record_id}")
    async def memory_record_delete(
        record_id: int, undo: bool = Query(False)
    ) -> Response:
        pinned = await console.memory.pinned_quotes()
        if not await console.memory.delete_record(record_id, undo=undo):
            raise ApiError(404, "not_found", t("errors.noteGone"))
        await _identity_changed(console, pinned)
        return Response(status_code=204)

    @app.delete("/api/memory/records")
    async def memory_forget_all(
        everything: bool = Query(False, alias="all")
    ) -> dict[str, int]:
        if not everything:
            raise ApiError(400, "invalid_request", t("errors.forgetAllFlag"))
        pinned = await console.memory.pinned_quotes()
        notes, turns = await console.memory.forget_all()
        await _identity_changed(console, pinned)
        return {"notes": notes, "turns": turns}

    @app.get("/api/memory/import/scan")
    async def memory_import_scan() -> dict[str, Any]:
        # The home folder, as the scan's paths spell it, lets the page write a path as `~/…`.
        return {
            **(await console.memory.import_scan()).to_json(),
            "home": str(Path.home().resolve()),
        }

    @app.get("/api/memory/import/offer")
    async def memory_import_offer() -> dict[str, bool]:
        return {"pending": console.memory.import_pending()}

    @app.delete("/api/memory/import/offer")
    async def memory_import_decline() -> dict[str, bool]:
        console.memory.import_offered("declined")
        # Still pending when the record could not be written: the page must not say "Not now" was kept.
        return {"pending": console.memory.import_pending()}

    @app.post("/api/memory/import")
    async def memory_import(body: ImportBody) -> dict[str, Any]:
        plan = console.memory.import_plan(body.batch_id)
        if plan is None:
            raise ApiError(404, "not_found", t("errors.importScanGone"))
        try:
            return await console.memory.import_apply(plan, body.source_ids)
        except ValueError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from exc

    @app.delete("/api/memory/import/{batch_id}")
    async def memory_import_undo(batch_id: str) -> dict[str, Any]:
        removed = await console.memory.undo_import(batch_id)
        # Notes the person reworded since stay: they are theirs now.
        kept = await console.memory.import_kept(batch_id)
        if not removed and not kept:
            raise ApiError(404, "not_found", t("errors.importGone"))
        return {
            "batch_id": batch_id,
            "removed": removed,
            **({"kept": kept} if kept else {}),
        }

    @app.put("/api/memory")
    async def memory_switch(body: SwitchBody) -> dict[str, bool]:
        saved = config.load()
        saved.memory.across_chats = body.on
        config.save(saved)
        console.cfg.memory.across_chats = body.on
        await console.sessions.settings_changed()
        return {"on": body.on}

    @app.post("/api/memory/reindex")
    async def memory_reindex() -> dict[str, int]:
        chats = console.store if console.store_state() == "ok" else None
        notes, turns = await console.memory.reindex(chats)
        return {"notes": notes, "turns": turns}

    @app.post("/api/memory/sleep")
    async def memory_sleep(body: SleepBody | None = None) -> dict[str, Any]:
        _developer(console)
        variant = body.variant if body else None
        if variant is not None and variant not in config.MEMORY_SLEEP:
            raise ApiError(
                400,
                "invalid_request",
                t("errors.sleepVariant", choices=", ".join(config.MEMORY_SLEEP)),
            )
        chats = console.store if console.store_state() == "ok" else None
        pinned = await console.memory.pinned_quotes()
        try:
            report = await console.memory.sleep(
                variant,
                chats=chats,
                budget=(
                    body.budget if body and console.cfg.ui.mode == "developer" else None
                ),
            )
        except NotImplementedError as exc:
            raise ApiError(501, "not_built", str(exc)) from exc
        await _identity_changed(console, pinned)
        return report.to_json()

    @app.delete("/api/memory/sleep/{run_id}")
    async def memory_sleep_undo(run_id: str) -> dict[str, Any]:
        _developer(console)
        pinned = await console.memory.pinned_quotes()
        undone = await console.memory.undo_sleep(run_id)
        if undone is None:
            raise ApiError(404, "not_found", t("errors.sleepGone"))
        await _identity_changed(console, pinned)
        return undone


__all__: list[str] = []
