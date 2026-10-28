"""The working-memory note store: RAM-only notes, drafts and digests.

R4 keeps every document the reasoning-note loop produces here, beside the
conversation window: one entry per note, draft, attempt digest (and, from
R5, thought chunk), each with the alias a prompt names it by. The store is
never written to ``sessions.db`` or ``history.db``; it lives as long as
the Agent's working memory.

This module imports nothing from ``melo.agent`` or ``support/``.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence

from localmelo.melo.memory.notes import (
    RECALL_END,
    RECALL_START,
    Draft,
    ReasoningNote,
    same_path,
)
from localmelo.melo.schema import Message

logger = logging.getLogger(__name__)

# Caps of one Agent's store: documents, and characters of their texts.
NOTE_STORE_MAX_ENTRIES = 300
NOTE_STORE_MAX_CHARS = 300_000

# The alias prefix of each entry kind.
_ALIAS_PREFIX = {"note": "n", "draft": "d", "digest": "g", "thought": "t"}

_RECALL_SPAN_RE = re.compile(
    re.escape(RECALL_START) + r".*?" + re.escape(RECALL_END), re.DOTALL
)
_NOTE_BLOCK_RE = re.compile(r"<note>.*?(?:</note>|\Z)", re.IGNORECASE | re.DOTALL)


@dataclasses.dataclass
class NoteStoreEntry:
    """One stored document: its checked text and the metadata around it."""

    alias: str  # n1, n2, …; d1, …; t1, …; g1, … (per kind)
    kind: str  # note | draft | digest | thought (thought: R5)
    text: str  # the checked rendered text (the boundary's output)
    stage: str = "fresh"  # fresh | rehearsed | superseded
    # The identity/meta the prompts, digest and (later) Hippo need:
    note: ReasoningNote | None = None  # note, digest and reflection entries
    draft: Draft | None = None  # draft entries
    step: int = 0
    attempt: int = 0
    remember: bool = False
    remember_by: str = ""
    uses: int = 0
    last_used_at: float = 0.0
    created_at: float = 0.0


class NoteStore:
    """The Agent's notes, drafts and digests, in RAM, with bounded size.

    At most :data:`NOTE_STORE_MAX_ENTRIES` documents and
    :data:`NOTE_STORE_MAX_CHARS` characters of text. When a new document
    would pass a cap, entries are evicted until it fits: thought chunks
    first (R5), then drafts that are not pending, then fresh notes, then
    rehearsed notes and digests, least recently used first within each
    group. A pending draft is never evicted. When only pending drafts are
    left and the cap is still passed, the new entry is not added, the
    addition is counted in :attr:`refused`, and the reason is logged at
    DEBUG.

    Aliases are assigned per kind in creation order and never reused
    within a store.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        max_entries: int = NOTE_STORE_MAX_ENTRIES,
        max_chars: int = NOTE_STORE_MAX_CHARS,
    ) -> None:
        self._clock = clock
        self._max_entries = max_entries
        self._max_chars = max_chars
        self._entries: list[NoteStoreEntry] = []
        self._counters: dict[str, int] = {}
        # Additions the caps refused (see the class docstring).
        self.refused = 0

    # ── Reading ──

    def entries(self) -> list[NoteStoreEntry]:
        """Deep copies of every entry, in creation order."""
        return copy.deepcopy(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def chars(self) -> int:
        """The characters of every stored text."""
        return sum(len(entry.text) for entry in self._entries)

    def get(self, alias: str) -> NoteStoreEntry | None:
        """The entry named *alias*, or ``None``."""
        for entry in self._entries:
            if entry.alias == alias:
                return entry
        return None

    def of_kind(self, kind: str) -> list[NoteStoreEntry]:
        """The entries of *kind*, in creation order."""
        return [entry for entry in self._entries if entry.kind == kind]

    def texts(self) -> list[str]:
        """The checked text of each entry, in creation order."""
        return [entry.text for entry in self._entries]

    # ── Writing ──

    def add(
        self,
        kind: str,
        text: str,
        *,
        note: ReasoningNote | None = None,
        draft: Draft | None = None,
        stage: str = "fresh",
        step: int = 0,
        attempt: int = 0,
        remember: bool = False,
        remember_by: str = "",
    ) -> NoteStoreEntry | None:
        """Store one document; ``None`` when the caps refuse it.

        The caller has already run the document through the candidate
        boundary; *text* is its checked output.
        """
        if kind not in _ALIAS_PREFIX:
            raise ValueError(f"unknown note-store kind: {kind!r}")
        if not text.strip():
            return None
        self._evict_until_fits(len(text))
        if len(self._entries) + 1 > self._max_entries or (
            self.chars() + len(text) > self._max_chars
        ):
            self.refused += 1
            logger.debug(
                "Refused a %s of %d chars: the note store holds %d documents "
                "and %d chars, with only pending drafts left to evict",
                kind,
                len(text),
                len(self._entries),
                self.chars(),
            )
            return None
        now = self._clock()
        prefix = _ALIAS_PREFIX[kind]
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        entry = NoteStoreEntry(
            alias=f"{prefix}{self._counters[prefix]}",
            kind=kind,
            text=text,
            stage=stage,
            note=note,
            draft=draft,
            step=step,
            attempt=attempt,
            remember=remember,
            remember_by=remember_by,
            created_at=now,
            last_used_at=now,
        )
        self._entries.append(entry)
        return entry

    def touch(
        self, alias: str, *, use: bool = False, stage: str | None = None
    ) -> NoteStoreEntry | None:
        """Mark *alias* used now; with *use*, count a use; with *stage*, set it."""
        entry = self.get(alias)
        if entry is None:
            return None
        entry.last_used_at = self._clock()
        if use:
            entry.uses += 1
        if stage is not None:
            entry.stage = stage
        return entry

    def replace_text(
        self, alias: str, text: str, note: ReasoningNote | None = None
    ) -> bool:
        """Replace the checked text (and note) of *alias*.

        ``False`` for an unknown alias, or when the replacement would pass
        a cap even after evicting in the usual order; the entry is then
        left as it was. A growing replacement can evict other entries,
        never the entry itself.
        """
        entry = self.get(alias)
        if entry is None:
            return False
        old_text, old_note = entry.text, entry.note
        entry.text = text
        if note is not None:
            entry.note = note
        self._evict_until_fits(0, extra_entries=0, exclude=alias)
        if len(self._entries) + 0 > self._max_entries or (
            self.chars() > self._max_chars
        ):
            entry.text, entry.note = old_text, old_note
            self.refused += 1
            logger.debug(
                "Refused a restatement of %s: the store would pass its caps",
                alias,
            )
            return False
        return True

    def supersede_attempt(self, attempt: int) -> int:
        """Mark the attempt's notes superseded; returns how many changed."""
        changed = 0
        for entry in self._entries:
            if (
                entry.kind == "note"
                and entry.attempt == attempt
                and entry.stage != "superseded"
            ):
                entry.stage = "superseded"
                changed += 1
        return changed

    def supersede_drafts(self, target: str, *, keep: str = "") -> int:
        """Mark older pending drafts for *target* superseded.

        *keep* names the alias of the newer draft that supersedes them.
        Returns how many drafts changed.
        """
        changed = 0
        for entry in self._entries:
            if (
                entry.kind == "draft"
                and entry.alias != keep
                and entry.draft is not None
                and entry.draft.status == "pending"
                and same_path(entry.draft.target, target)
            ):
                entry.draft.status = "superseded"
                entry.stage = "superseded"
                changed += 1
        return changed

    def clear(self) -> None:
        """Remove every entry and reset the alias counters."""
        self._entries.clear()
        self._counters.clear()
        self.refused = 0

    def load(self, entries: Sequence[NoteStoreEntry]) -> int:
        """Replace the store with copies of *entries*; returns how many fit.

        The store is cleared first. An entry keeps its alias; the counters
        move past every alias loaded, so a later addition never reuses one.
        The caps still apply, in the same eviction order.
        """
        self.clear()
        kept = 0
        seen: set[str] = set()
        for entry in entries:
            copied = copy.deepcopy(entry)
            if not copied.alias or copied.alias in seen:
                self.refused += 1
                logger.debug(
                    "Refused a loaded %s entry: its alias %r is empty or used",
                    copied.kind,
                    copied.alias,
                )
                continue
            seen.add(copied.alias)
            self._remember_alias(copied.alias)
            if len(self._entries) + 1 > self._max_entries or (
                self.chars() + len(copied.text) > self._max_chars
            ):
                self.refused += 1
                continue
            self._entries.append(copied)
            kept += 1
        return kept

    # ── Eviction ──

    def _remember_alias(self, alias: str) -> None:
        """Move the counter of *alias*'s kind past its number."""
        for prefix in _ALIAS_PREFIX.values():
            if not alias.startswith(prefix):
                continue
            number = alias[len(prefix) :]
            if number.isdigit():
                self._counters[prefix] = max(self._counters.get(prefix, 0), int(number))
            return

    def _evict_groups(self) -> Iterable[list[NoteStoreEntry]]:
        """The eviction groups, in order (see the class docstring)."""
        thoughts = [e for e in self._entries if e.kind == "thought"]
        retired_drafts = [
            e
            for e in self._entries
            if e.kind == "draft" and (e.draft is None or e.draft.status != "pending")
        ]
        fresh_notes = [
            e for e in self._entries if e.kind == "note" and e.stage == "fresh"
        ]
        rehearsed = [
            e
            for e in self._entries
            if e.kind == "digest" or (e.kind == "note" and e.stage != "fresh")
        ]
        return thoughts, retired_drafts, fresh_notes, rehearsed

    def _evict_until_fits(
        self, incoming: int, *, extra_entries: int = 1, exclude: str = ""
    ) -> None:
        for group in self._evict_groups():
            if len(self._entries) + extra_entries <= self._max_entries and (
                self.chars() + incoming <= self._max_chars
            ):
                return
            for entry in sorted(group, key=self._lru_key):
                if entry.alias == exclude:
                    continue
                self._entries.remove(entry)
                if len(self._entries) + extra_entries <= self._max_entries and (
                    self.chars() + incoming <= self._max_chars
                ):
                    return

    @staticmethod
    def _lru_key(entry: NoteStoreEntry) -> tuple[float, float]:
        return (entry.last_used_at or entry.created_at, entry.created_at)


def strip_note_spans(
    messages: Sequence[Message], notes: Iterable[str]
) -> list[Message]:
    """*messages* with stored notes and recall spans removed, for tool hints.

    In a compress-mode step the tool-hint scan must not read the harness's
    own note text as a hint: a note that names a tool would otherwise
    offer it. Each stored note's checked text and every raw ``<note>``
    block are removed from an assistant message's content, and a tool
    message loses the text between
    :data:`~localmelo.melo.memory.notes.RECALL_START` and
    :data:`~localmelo.melo.memory.notes.RECALL_END`. Messages the scan
    would not change are returned as they are.
    """
    note_texts = [text for text in notes if text]
    stripped: list[Message] = []
    for message in messages:
        content = message.content
        if message.role == "assistant":
            content = _NOTE_BLOCK_RE.sub("", content)
            for text in note_texts:
                content = content.replace(text, "")
            content = content.strip()
        elif message.role == "tool":
            content = _RECALL_SPAN_RE.sub("", content).strip()
        if content == message.content:
            stripped.append(message)
        else:
            stripped.append(dataclasses.replace(message, content=content))
    return stripped
