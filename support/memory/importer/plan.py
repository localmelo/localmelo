"""Scan the sources into a plan, write what a plan's sources hold, take a batch back.

A scan writes nothing. A line becomes a note with :func:`records.second_person`
(its language kept), topic by :func:`rules.topic_for`, and provenance that names
the file, line and heading it came from. A rescan skips lines it knows
(``line_sha``); a line whose wording changed in the same place supersedes the
note of its old wording. Notes of the user's own, and notes of another source,
are never superseded: a line that says what one says is a duplicate.

A source holds at most :data:`PER_SOURCE` notes. A folder of note files fills it
round by round (:func:`_by_rounds`): the first note of every file (its summary),
then the second of every file, so a long file cannot crowd the others out. What
passed every check but did not fit is counted as ``capped``.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from localmelo.melo.memory.hippo.policy import content_tokens, normalise, overlap
from localmelo.melo.memory.long import StoredEntry
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import new_record_id
from localmelo.support.memory import records
from localmelo.support.memory.importer.rules import own_email, reject, topic_for
from localmelo.support.memory.importer.sources import (
    Source,
    Statement,
    discover,
    read_source,
)
from localmelo.support.memory.importer.voice import to_you

PER_SOURCE = 60
PER_BATCH = 300
FILTERS = ("secret", "third_party", "wrong_owner", "duplicate", "mechanics", "capped")


@dataclass
class Note:
    """One note to write; *target* is the id of the note of its old wording."""

    text: str
    topic: str
    provenance: dict[str, Any]
    target: int | None = None


@dataclass
class ImportPlan:
    """What a scan found: *sources* as the API shows them, their *notes*, and
    the sources already written (*applied*)."""

    batch_id: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, list[Note]] = field(default_factory=dict)
    partial: bool = False
    applied: set[str] = field(default_factory=set)

    def to_json(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "sources": self.sources,
            "partial": self.partial,
        }


def line_sha(text: str) -> str:
    return hashlib.sha1(normalise(text).encode()).hexdigest()[:16]


def _changed(
    st: Any,
    tokens: set[str],
    olds: Sequence[StoredEntry],
    present: set[tuple[str, str]],
    taken: set[int],
) -> StoredEntry | None:
    """The imported note of the old wording of *st*: same file and heading, its line
    gone from the file, most of its words kept."""
    for old in olds:
        p = old.provenance
        if (
            (p.get("path"), p.get("heading")) == (st.path, st.heading)
            and (st.path, p.get("line_sha")) not in present
            and old.id not in taken
            and overlap(tokens, content_tokens(old.text)) >= records.SUPERSEDE_OVERLAP
        ):
            return old
    return None


def _by_assistant(src: Source) -> bool:
    """Whether the assistant wrote *src* (its note files, its memory database, its
    own Hermes notes), not the user (instruction files, Hermes ``USER.md``)."""
    return src.kind in ("memory", "sqlite") or (
        src.kind == "hermes" and Path(src.path).stem != "USER"
    )


def _by_rounds(
    statements: Sequence[Statement], by_file: bool, notes: Sequence[Note]
) -> Iterator[Statement]:
    """*statements* in rounds: each file (in the order read) hands over statements
    until one becomes a note in *notes*, then the next file has its turn.

    Without *by_file* the source is one file, read in order.
    """
    files: dict[str, deque[Statement]] = {}
    for st in statements:
        files.setdefault(st.path if by_file else "", deque()).append(st)
    queues = list(files.values())
    while queues:
        for queue in queues:
            before = len(notes)
            while queue:
                yield queue.popleft()
                if len(notes) > before:
                    break
        queues = [queue for queue in queues if queue]


def build_plan(
    home: Path, active: Sequence[StoredEntry], *, timeout_s: float
) -> ImportPlan:
    """Scan *home* against the notes *active* now; ``partial`` when time ran out."""
    home = home.resolve()
    deadline = time.monotonic() + timeout_s
    plan = ImportPlan(uuid.uuid4().hex[:12])
    email = own_email(home)
    olds = [e for e in active if e.provenance.get("source") == "import"]
    known = {e.provenance.get("line_sha") for e in olds}
    taken: set[int] = set()
    kept: list[tuple[str, set[str]]] = []
    seen: set[str] = set()
    for src in discover(home):
        if time.monotonic() > deadline:
            plan.partial = True
            break
        statements = list(read_source(src, home, seen))
        present = {(st.path, line_sha(st.text)) for st in statements}
        counts = dict.fromkeys(FILTERS, 0)
        notes: list[Note] = []
        over: list[tuple[str, set[str]]] = []  # passed every check, no room left
        found = 0
        for st in _by_rounds(statements, src.kind == "memory", notes):
            if st.kind in ("project", "reference") and not src.workspace:
                continue  # its folder is gone
            if time.monotonic() > deadline:
                plan.partial = True
                break
            found += 1
            sha = line_sha(st.text)
            if reason := reject(st.text, email):
                counts[reason] += 1
                continue
            if _by_assistant(src):  # "I" is the assistant, "the user" is the person
                text = records.finished(to_you(st.text))
            elif "@" in st.text:  # "me@host" must stay an address: no swap
                text = records.finished(st.text)
            else:
                text = records.second_person(st.text)
            if records.wrong_owner(text, st.text):
                counts["wrong_owner"] += 1
                continue
            topic = topic_for(st)
            kind = records.TOPIC_KINDS[topic]
            tokens = content_tokens(text)
            old = _changed(st, tokens, olds, present, taken)
            if old is None and (
                sha in known
                or records.decide(text, active, kind=kind)[0] != "NEW"
                or any(
                    k == kind and overlap(tokens, t) >= records.SUPERSEDE_OVERLAP
                    for k, t in (*kept, *over)
                )
            ):
                counts["duplicate"] += 1
                continue
            if len(notes) >= PER_SOURCE:
                counts["capped"] += 1
                over.append((kind, tokens))
                continue
            prov: dict[str, Any] = {
                "source": "import",
                "harness": src.harness,
                "path": st.path,
                "line": st.line,
                "heading": st.heading,
                "batch_id": plan.batch_id,
                "line_sha": sha,
                "topic": topic,
                "quote": st.text[: records.MAX_RECORD_CHARS],
                "chat": f"the file {Path(st.path).name}",
            }
            if topic == "projects":
                prov["workspace"] = src.workspace
            notes.append(Note(text, topic, prov, old.id if old else None))
            kept.append((kind, tokens))
            if old:
                taken.add(old.id)
        if found:
            plan.notes[src.id] = notes
            plan.sources.append(
                {
                    "id": src.id,
                    "harness": src.harness,
                    "path": src.path,
                    "scope": src.scope,
                    "workspace": src.workspace,
                    "items_found": found,
                    "items_kept": len(notes),
                    "filtered": counts,
                    "sample": [n.text for n in notes[:3]],
                }
            )
    return plan


def select(plan: ImportPlan, source_ids: Sequence[str]) -> list[tuple[str, Note]]:
    """The notes of the chosen sources not written yet, at most :data:`PER_BATCH`."""
    for source_id in source_ids:
        if source_id not in plan.notes:
            raise ValueError(f"Unknown source {source_id!r}: scan again.")
    chosen = [
        (sid, note)
        for sid in dict.fromkeys(source_ids)
        if sid not in plan.applied
        for note in plan.notes[sid]
    ]
    return chosen[:PER_BATCH]


async def write(
    store: SqliteLongTerm,
    chosen: Sequence[tuple[str, Note]],
    vectors: Sequence[list[float] | None],
    model: str,
) -> list[int | None]:
    """Write each note; its id, or ``None`` for one the store holds already."""
    have = {normalise(e.text) for e in await store.active_entries()}
    ids: list[int | None] = []
    for (_, note), vector in zip(chosen, vectors, strict=True):
        if normalise(note.text) in have:
            ids.append(None)
            continue
        embed_model = model if vector is not None else ""
        entry_id, _ = await store.apply(
            text=note.text,
            kind=records.TOPIC_KINDS[note.topic],
            source="user",
            admitted_by="import",
            provenance={**note.provenance, "embed_model": embed_model},
            task_id="",
            session_id="",
            created_at=time.time(),
            embedding=vector,
            action="UPDATE" if note.target else "NEW",
            target_id=note.target,
            reason="an imported line changed" if note.target else "imported",
            source_ids=(),
            keep_text=False,
            candidate_id=new_record_id(),
            record={
                "topic": note.topic,
                "quote": note.provenance["quote"],
                "embed_model": embed_model,
            },
        )
        ids.append(entry_id)
        have.add(normalise(note.text))
    return ids
