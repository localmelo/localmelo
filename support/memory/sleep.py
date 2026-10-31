"""Sleep: tidy the notes while melo is idle, by rules (a developer setting).

In-turn writing leaves defects only a later pass can see. ``[memory] sleep``
(``off`` by default; ``rules`` is built, ``model`` and ``model+tags`` are not
yet) replays the chat turns since the last sleep against the notes, with no
model call, and makes at most :data:`MAX_CHANGES` changes, each one of:

- ``O1`` merge duplicates: of two notes of one subject, or close in meaning and
  wording (the older one only says what the newer says), the older is
  superseded by the newer;
- ``O2`` missed supersede: a newer typed statement of a note's subject with
  another value is written, and supersedes the note;
- ``O3`` recover lost facts: a plain, lasting statement of a turn that no note
  of the turn keeps is written (turns flagged ``unsaved_claim`` first).

Every write passes the checks a note passes (:func:`records.unsupported`,
:func:`records.wrong_owner`, :func:`records.sensitive`) on the user's own typed
words and is ``admitted_by="sleep"`` with provenance
``{source: "sleep", run_id, session_id, turn}`` (the turn it read). Nothing is deleted:
the older note is superseded. A note the user reworded is left alone unless a
newer typed statement of its subject contradicts it. A watermark
(``meta.sleep_until`` of ``index.db``) keeps a second run from redoing a turn;
a run that hit the cap leaves it, and the next run goes on. :func:`undo` puts a
run back exactly: what it superseded is current again, what it wrote is gone.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from localmelo.melo.memory.hippo.policy import content_tokens, overlap
from localmelo.melo.memory.long import StoredEntry
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import new_record_id
from localmelo.support.memory import records
from localmelo.support.memory.index import ConversationIndex
from localmelo.support.memory.recall import Item, unit

if TYPE_CHECKING:
    from localmelo.support.sessions.store import SessionStore

logger = logging.getLogger(__name__)

VARIANTS: Final = ("off", "rules")
MAX_CHANGES: Final = 20
# Of the newest notes, how many are compared for duplicates.
MAX_NOTES: Final = 1_000
DUPLICATE_COSINE: Final = 0.9


@dataclass
class Change:
    """One change: ``before_ids`` are the notes it superseded, ``after_id`` the
    note that replaced them (written by sleep, or for ``O1`` the one kept)."""

    op: str
    before_ids: list[int]
    after_id: int | None
    text: str
    source_turn: list[Any] | None = None


@dataclass
class SleepReport:
    """What one run did. *aborted* says why a run did nothing or stopped early;
    *capped* that more changes were left for the next run."""

    run_id: str
    variant: str
    started: float
    ended: float
    model_calls: int = 0
    changes: list[Change] = field(default_factory=list)
    aborted: str = ""
    capped: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SleepTurn:
    """A finished chat turn memory was on for: what the user typed, the notes
    its ``turn.memory`` event listed, and whether a reply claimed a save."""

    session_id: str
    turn: int
    started_at: float
    finished_at: float
    typed: str
    remembered: tuple[records.Written, ...] = ()
    unsaved_claim: bool = False


async def sleep_turns(chats: SessionStore, since: float) -> list[SleepTurn]:
    """The chat turns that finished after *since* with a ``turn.memory`` event
    (memory was on): completed, not undone, not a task's or a worker's."""
    from localmelo.support.sessions.store import MEMORY_EVENT_TYPE

    out: list[SleepTurn] = []
    for session in await chats.list_sessions(limit=100_000, kinds=("chat", "front")):
        if session.updated_at <= since:
            continue
        events = await chats.events_of_type(
            session.id, MEMORY_EVENT_TYPE, limit=100_000
        )
        seen = {e.turn: e.event.get("data", {}) for e in events if e.turn is not None}
        for turn in await chats.list_turns(session.id):
            data = seen.get(turn.turn)
            if (
                data is None
                or turn.status != "completed"
                or turn.undone
                or (turn.finished_at or 0) <= since
            ):
                continue
            notes = tuple(
                records.Written(
                    int(n.get("id") or 0), str(n.get("text") or ""), str(n.get("topic"))
                )
                for n in data.get("remembered") or []
                if isinstance(n, dict)
            )
            typed = turn.typed_text if turn.attachments else turn.user_text
            out.append(
                SleepTurn(
                    session.id,
                    turn.turn,
                    turn.started_at,
                    turn.finished_at or 0,
                    typed,
                    notes,
                    bool(data.get("unsaved_claim")),
                )
            )
    return out


def _edited(note: StoredEntry) -> bool:
    return bool(note.provenance.get("edited"))


def _same(new: StoredEntry, old: StoredEntry) -> bool:
    """Whether *old* adds nothing to *new*: one subject, or close in meaning and
    wording with every value of *old* in *new*. Other subjects never merge."""
    mine, theirs = records.subject_key(new.text), records.subject_key(old.text)
    if mine and mine == theirs:
        return True
    if records.subject(new.text) and records.subject(old.text):
        return False
    a, b = unit(new.vector), unit(old.vector)
    return (
        a is not None
        and b is not None
        and bool(new.embed_model)
        and new.embed_model == old.embed_model
        and a.shape == b.shape
        and float(np.dot(a, b)) >= DUPLICATE_COSINE
        and overlap(content_tokens(new.text), content_tokens(old.text))
        >= records.SUPERSEDE_OVERLAP
        and not records.unsupported(old.text, [new.text])
    )


def duplicates(notes: Sequence[StoredEntry]) -> list[tuple[StoredEntry, list[int]]]:
    """O1: each note to keep and the older notes it makes redundant, newest first.

    A note the user reworded is never merged away.
    """
    newest = sorted(notes, key=lambda n: (n.created_at, n.id), reverse=True)
    gone: set[int] = set()
    out: list[tuple[StoredEntry, list[int]]] = []
    for at, keep in enumerate(newest[:MAX_NOTES]):
        if keep.id in gone:
            continue
        older = [
            old.id
            for old in newest[at + 1 : MAX_NOTES]
            if old.id not in gone and not _edited(old) and _same(keep, old)
        ]
        if older:
            gone.update(older)
            out.append((keep, older))
    return out


def recoverable(
    turns: Sequence[SleepTurn], notes: Sequence[StoredEntry]
) -> list[tuple[SleepTurn, records.Candidate]]:
    """O2 and O3: the plain statements of *turns* that no note of the turn keeps.

    A turn a newer one replaced (:func:`records.newer_turns`) states nothing
    new. Flagged turns (``unsaved_claim``) come first, then the newest.
    """
    items = [
        Item(
            f"d:{n}",
            t.typed,
            "",
            t.session_id,
            t.turn,
            t.started_at,
            said=records.said(t.typed),
        )
        for n, t in enumerate(turns)
    ]
    replaced = records.newer_turns(items)
    out: list[tuple[SleepTurn, records.Candidate]] = []
    for turn in sorted(turns, key=lambda t: (not t.unsaved_claim, -t.started_at)):
        if (turn.session_id, turn.turn) in replaced:
            continue
        kept = [
            *turn.remembered,
            *(
                records.Written(n.id, n.text, "")
                for n in notes
                if n.session_id == turn.session_id
                and n.provenance.get("turn") == turn.turn
            ),
        ]
        out.extend(
            (turn, cand)
            for cand in records.statements(turn.typed)
            if not records.covered(cand, kept)
        )
    return out


def _allowed(
    turn: SleepTurn, cand: records.Candidate, active: Sequence[StoredEntry]
) -> bool:
    """Whether writing *cand*, a statement of *turn*, is safe and needed.

    It must pass the checks of a note (the user's words, the owner, no secret).
    A note of its subject changes only when the turn is newer and says another
    value of that subject.
    """
    if (
        records.sensitive(cand.text)
        or records.unsupported(cand.text, [turn.typed])
        or records.wrong_owner(cand.text, cand.quote)
    ):
        return False
    action, target = records.decide(cand.text, active, kind=cand.kind)
    if action == "IGNORE":
        return False
    old = next((n for n in active if n.id == target), None)
    if action == "NEW" or old is None:
        return True
    # A note changes only when the user's newer words state its subject again
    # with another value (also a note the user reworded).
    key = records.subject_key(cand.text)
    return (
        key != ""
        and key == records.subject_key(old.text)
        and old.created_at < turn.started_at
        and bool(records.unsupported(old.text, [cand.text]))
    )


async def run(
    store: SqliteLongTerm,
    index: ConversationIndex,
    turns: Sequence[SleepTurn],
    *,
    variant: str,
    now: float | None = None,
    until: float = 0.0,
) -> SleepReport:
    """One sleep over the notes and the *turns* that finished after *until*.

    Raises ``NotImplementedError`` for a variant not built yet. The report is
    kept in ``index.db`` (ids only) for :func:`undo`; ``variant`` ``off`` does
    nothing. A failure keeps the changes made, ends the run and is the report's
    ``aborted``; the watermark stays.
    """
    started = time.time() if now is None else now
    report = SleepReport(new_record_id()[:12], variant, started, started)
    if variant == "off":
        report.aborted = "sleep is off"
        return report
    if variant not in VARIANTS:
        raise NotImplementedError(f"sleep variant {variant!r} is not built yet")
    try:
        await _tidy(store, index, report, [t for t in turns if t.finished_at > until])
    except Exception as exc:  # keep what is done, and say why it stopped
        logger.warning(
            "Sleep stopped after %d changes", len(report.changes), exc_info=True
        )
        report.aborted = f"{type(exc).__name__}: {exc}"
    report.ended = time.time() if now is None else now
    if report.changes:
        steps: list[dict[str, Any]] = [
            {"op": c.op, "before": c.before_ids, "after": c.after_id}
            for c in report.changes
        ]
        await index.save_run(report.run_id, started, steps)
    return report


async def _tidy(
    store: SqliteLongTerm,
    index: ConversationIndex,
    report: SleepReport,
    fresh: Sequence[SleepTurn],
) -> None:
    """The changes of one run, added to *report* as they are made."""
    for keep, older in duplicates(await store.active_entries()):
        if len(report.changes) >= MAX_CHANGES:
            report.capped = True
            return
        if done := await store.set_status(
            older,
            "superseded",
            keeper=keep.id,
            admitted_by="sleep",
            reason=f"merged into {keep.id} by sleep {report.run_id}",
        ):
            report.changes.append(Change("O1", done, keep.id, keep.text))
    for turn, cand in recoverable(fresh, await store.active_entries()):
        if not _allowed(turn, cand, await store.active_entries()):
            continue
        if len(report.changes) >= MAX_CHANGES:
            report.capped = True
            return
        place = [turn.session_id, turn.turn]
        source = {"session_id": turn.session_id, "source": "sleep", "turn": turn.turn}
        found = records.Candidate(cand.text, cand.quote, cand.topic, "sleep")
        for note in await records.write(
            store,
            [found],
            [None],
            source={**source, "run_id": report.run_id},
            embed_model="",
            origin="sleep",
            created_at=turn.started_at,
        ):
            op, before = (
                ("O3", []) if note.replaced is None else ("O2", [note.replaced])
            )
            report.changes.append(Change(op, before, note.id, note.text, place))
    if fresh:
        await index.set_meta("sleep_until", max(t.finished_at for t in fresh))


async def undo(
    store: SqliteLongTerm, index: ConversationIndex, run_id: str
) -> dict[str, Any] | None:
    """Put a sleep run back: the notes it superseded are current again and the
    notes it wrote are gone. ``None`` for a run unknown or already undone.

    A note the run wrote that a later note replaced, or the user reworded, is
    left as it is (``kept``), so undo never brings back a value that lost to a
    newer one.
    """
    changes = await index.load_run(run_id)
    if changes is None:
        return None
    restored = removed = kept = 0
    for change in reversed(changes):
        after, before = int(change["after"] or 0), [int(i) for i in change["before"]]
        live = {n.id: n for n in await store.active_entries()}
        if after not in live or (change["op"] != "O1" and _edited(live[after])):
            kept += 1  # a later note replaced it, or the user reworded it
        elif change["op"] == "O1":
            reason = f"sleep {run_id} undone"
            done = await store.set_status(
                before, "active", keeper=after, admitted_by="sleep", reason=reason
            )
            restored += len(done)
        else:
            await store.delete_entry(after, restore=True)
            removed += 1
            restored += len(before)
    await index.drop_run(run_id)
    return {"run_id": run_id, "restored": restored, "removed": removed, "kept": kept}
