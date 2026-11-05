"""The reasoning notes of the agent loop (the compress modes).

:class:`_NotesLoop` is a mixin of :class:`~localmelo.melo.agent.agent.Agent`: the
methods that turn a model reply into a note, check and store the notes and drafts,
stage what a run offers long-term memory, and write the digest a step leaves
behind. They read three attributes that ``Agent.__init__`` sets (``options``,
``hippo`` and ``checker``) and keep no state of their own; the counters they
change belong to the run state they are passed.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from localmelo.melo.agent._chat import CALL_RECORDER
from localmelo.melo.agent._reflection import _coerce_str, _read_reflection
from localmelo.melo.agent.options import AgentOptions
from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import MemoryCandidatePayload
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.hippo import MemoryDecision
from localmelo.melo.memory.notes import (
    NOTE_MAX_DECIDED,
    NOTE_MAX_FOUND,
    NOTE_SHOWN_CHARS,
    Draft,
    ReasoningNote,
    candidate_items,
    candidate_text,
    digest_note,
    draft_title,
    extract_draft,
    mark_drafts_written,
    note_from_reply,
    note_title,
    path_spans,
    reflection_candidate_text,
    render_digest,
    render_draft_block,
    render_note,
    same_path,
    source_records,
)
from localmelo.melo.memory.recall_gate import digest_has_evidence, digest_task_key
from localmelo.melo.memory.short.note_store import NoteStoreEntry
from localmelo.melo.schema import (
    ASSISTANT_TEXT_SHARE,
    CandidateItem,
    MemoryCandidate,
    Message,
    ReflectionEntry,
    ToolCall,
    ToolResult,
    new_record_id,
)

if TYPE_CHECKING:
    from localmelo.melo.agent.agent import _InflightStep, _RunState

logger = logging.getLogger(__name__)


# A note text after this many characters is cut in the reflection's block.
REFLECTION_NOTE_CHARS = NOTE_SHOWN_CHARS

# Notes of this attempt the reflection request carries at most.
REFLECTION_MAX_NOTES = 12


class _NotesLoop:
    """The note methods of :class:`~localmelo.melo.agent.agent.Agent`."""

    options: AgentOptions
    hippo: Hippo
    checker: Checker

    @property
    def _notes_mode(self) -> bool:
        """Whether the run compresses its replies into notes."""
        return self.options.reasoning in ("compress", "compress+replay")

    @staticmethod
    def _last_call_seq() -> int:
        """The ``llm_calls.seq`` of the reply that just arrived, or 0."""
        recorder = CALL_RECORDER.get()
        return int(getattr(recorder, "last_seq", 0) or 0)

    def _reply_note(
        self,
        response: Message,
        results: Sequence[ToolResult],
        *,
        kind: str,
        task_id: str,
        run: _RunState,
    ) -> tuple[ReasoningNote, str]:
        """Build the note of *response* and the content without its note block."""
        window_ids = {
            message.tool_call_id
            for message in self.hippo.working.get_window()
            if message.role == "tool" and message.tool_call_id
        }
        window_records, task_records = run.source_log.earlier(window_ids)
        note, rest = note_from_reply(
            response.content,
            response.thinking,
            response.tool_calls,
            results,
            kind=kind,
            writer=self.options.note_writer,
            window_records=window_records,
            task_records=task_records,
        )
        note = dataclasses.replace(
            note,
            note_id=new_record_id(),
            task_id=task_id,
            step=run.steps,
            attempt=run.attempts,
            call_seq=self._last_call_seq(),
            created_at=time.time(),
        )
        return note, rest

    def _check_candidate(
        self, text: str, source: str, run: _RunState, *, target: str = "candidate"
    ) -> str | None:
        """Run one document through the candidate boundary; its checked text."""
        run.candidates += 1
        check = self.checker.check_memory_candidate(
            MemoryCandidatePayload(text=text, source=source, target=target)
        )
        if not check.allowed:
            run.candidates_refused += 1
            logger.debug(
                "Dropped a %s candidate of task %s: %s",
                source,
                run.task_id,
                check.reason,
            )
            return None
        payload = check.sanitized_payload
        run.redactions += int(payload.metadata.get("redactions", 0) or 0)
        return str(payload.text)

    def _checked_candidate_text(
        self, text: str, source: str, run: _RunState
    ) -> str | None:
        """A candidate text through the boundary, counted as one boundary check.

        The note or draft it comes from was counted when it entered its own
        store; this pass is the one R6-lite requires before a candidate is
        staged, so the policy never compares raw text. ``candidates`` counts
        boundary checks, not documents (review finding 11).
        """
        run.candidates += 1
        check = self.checker.check_memory_candidate(
            MemoryCandidatePayload(text=text, source=source, target="candidate")
        )
        if not check.allowed:
            run.candidates_refused += 1
            return None
        payload = check.sanitized_payload
        run.redactions += int((payload.metadata or {}).get("redactions", 0) or 0)
        return str(payload.text)

    def _memory_provenance(
        self,
        run: _RunState,
        *,
        note: ReasoningNote | None = None,
        calls: Sequence[ToolCall] = (),
    ) -> dict[str, Any]:
        """The provenance a candidate records about the turn that made it."""
        paths: list[str] = []
        commands: list[str] = []
        for call in calls:
            for name in self._touched_names(call):
                bucket = paths if path_spans(name) else commands
                if name not in bucket:
                    bucket.append(name)
        source_ids = list(note.source_ids) if note is not None else []
        return {
            "task_id": run.task_id,
            "session_id": "",
            "step": run.steps,
            "attempt": run.attempts,
            "call_seq": self._last_call_seq(),
            "task_text": run.task_text[:300],
            "paths": paths,
            "commands": commands,
            "source_ids": source_ids,
            "note_id": note.note_id if note is not None else "",
        }

    def _propose_note_candidate(
        self,
        note: ReasoningNote,
        run: _RunState,
        calls: Sequence[ToolCall] = (),
    ) -> None:
        """Stage a rememberable note's items for long-term memory."""
        if not note.remember or not (note.found or note.decided):
            return
        text = self._checked_candidate_text(candidate_text(note), "reasoning_note", run)
        if text is None:
            return
        items: list[CandidateItem] = []
        for item in candidate_items(note):
            checked = self._checked_candidate_text(item.text, "reasoning_note", run)
            if checked is not None:
                items.append(CandidateItem(checked, item.source_ids))
        candidate = MemoryCandidate(
            candidate_id=note.note_id,
            source="reasoning_note",
            text=text,
            remember_by=note.remember_by or "rule",
            task_id=run.task_id,
            session_id="",
            provenance=self._memory_provenance(run, note=note, calls=calls),
            created_at=note.created_at,
            remember=True,
            items=tuple(items),
        )
        self.hippo.propose(candidate)

    def _drain_decisions(
        self, decisions: Sequence[MemoryDecision], run: _RunState
    ) -> None:
        """Count the decisions an admission made and emit one event each."""
        for decision in decisions:
            if decision.refused:
                run.candidates_refused += 1
                continue
            run.redactions += decision.redactions
            run.admitted[decision.admitted_by] = (
                run.admitted.get(decision.admitted_by, 0) + 1
            )
            if decision.action == "IGNORE":
                run.ignored += 1
            else:
                run.written[decision.kind] = run.written.get(decision.kind, 0) + 1
            run.events.memory(
                candidate_id=decision.candidate_id,
                action=decision.action,
                kind=decision.kind,
                admitted_by=decision.admitted_by,
                entry_id=decision.entry_id,
            )

    def _propose_reflection(
        self, entry: ReflectionEntry, run: _RunState, *, note_id: str = ""
    ) -> None:
        """Stage the final reflection, with its next directions and open questions."""
        candidate_id = note_id or new_record_id()
        text = self._checked_candidate_text(
            reflection_candidate_text(entry), "reflection", run
        )
        if text is None:
            return
        provenance = self._memory_provenance(run, note=None)
        provenance["note_id"] = candidate_id
        candidate = MemoryCandidate(
            candidate_id=candidate_id,
            source="reflection",
            text=text,
            remember_by="rule",
            task_id=run.task_id,
            session_id="",
            provenance=provenance,
            created_at=time.time(),
            remember=True,
        )
        run.final_reflection = candidate
        self.hippo.propose(candidate)

    def _store_note(
        self,
        note: ReasoningNote,
        run: _RunState,
        *,
        source: str,
        drafts: Sequence[str] = (),
        calls: Sequence[ToolCall] = (),
    ) -> NoteStoreEntry | None:
        """Check and store one note; emit its ``note`` event."""
        text = render_note(note, executor=True)
        checked = self._check_candidate(text, source, run)
        if checked is None:
            return None
        entry = self.hippo.working.notes.add(
            "note",
            checked,
            note=note,
            stage=note.stage,
            step=note.step,
            attempt=note.attempt,
            remember=note.remember,
            remember_by=note.remember_by,
            task_id=note.task_id,
            title=note_title(note),
        )
        if entry is None:
            run.candidates_refused += 1
            return None
        run.notes += 1
        run.note_chars += len(checked)
        if note.source == "model":
            run.notes_model += 1
        elif note.source == "heuristic":
            run.notes_heuristic += 1
        if note.remember and note.remember_by == "model":
            run.remember_model += 1
        elif note.remember and note.remember_by == "heuristic":
            run.remember_heuristic += 1
        run.events.note(
            step=note.step,
            attempt=note.attempt,
            note_id=note.note_id,
            kind=note.kind,
            source=note.source,
            text=checked,
            chars=len(checked),
            drafts=list(drafts),
            remember=note.remember,
            stage=entry.stage,
        )
        if source == "reasoning_note":
            self._propose_note_candidate(note, run, calls)
        return entry

    def _store_draft(self, draft: Draft, run: _RunState) -> NoteStoreEntry | None:
        """Check and store one draft, superseding older pending ones."""
        checked = self._check_candidate(draft.text, "draft", run)
        if checked is None:
            return None
        entry = self.hippo.working.notes.add(
            "draft",
            checked,
            draft=draft,
            step=draft.step,
            attempt=run.attempts,
            task_id=draft.task_id,
            title=draft_title(draft),
        )
        if entry is None:
            run.candidates_refused += 1
            return None
        run.drafts += 1
        self.hippo.working.notes.supersede_drafts(draft.target, keep=entry.alias)
        return entry

    async def _prepare_note_turn(
        self,
        response: Message,
        step: _InflightStep,
        pairs: Sequence[tuple[ToolCall, ToolResult]],
        run: _RunState,
    ) -> tuple[str, str, NoteStoreEntry | None, NoteStoreEntry | None]:
        """Build and store a tool-call reply's draft and note; place them.

        Returns the working-memory content and thinking of the turn, and
        the stored note and draft entries (either may be ``None`` when the
        boundary or the store refuses it). The draft is marked written
        first, for this reply's writes and any earlier pending draft they
        cover, so a draft written in the same reply is never pinned.
        """
        store = self.hippo.working.notes
        pending = [
            entry.draft
            for entry in store.of_kind("draft")
            if entry.draft is not None and entry.draft.status == "pending"
        ]
        run.drafts_written += mark_drafts_written(pending, pairs)
        draft_entry = None
        draft = extract_draft(step.rest, recent_path=step.recent_path)
        if draft is not None:
            draft = dataclasses.replace(
                draft,
                draft_id=new_record_id(),
                task_id=run.task_id,
                step=run.steps,
                call_seq=self._last_call_seq(),
                created_at=time.time(),
            )
            run.drafts_written += mark_drafts_written([draft], pairs)
            draft_entry = self._store_draft(draft, run)
        note, _rest = self._reply_note(
            response,
            [result for _call, result in pairs],
            kind="step",
            task_id=run.task_id,
            run=run,
        )
        aliases = [] if draft_entry is None else [draft_entry.alias]
        note_entry = self._store_note(
            note,
            run,
            source="reasoning_note",
            drafts=aliases,
            calls=[call for call, _result in pairs],
        )
        # The note is built: these observations are earlier results for the
        # next reply's note, never this one's.
        for call, result in pairs:
            run.source_log.add(call.id, source_records([(call, result)]))
        if note_entry is not None:
            await self._restate(note, run)
        note_text = "" if note_entry is None else note_entry.text
        if step.has_model_note:
            content = response.content
        elif not step.rest.strip():
            content = note_text
        elif note_text and len(step.rest) + 2 + len(note_text) <= ASSISTANT_TEXT_SHARE:
            content = f"{step.rest}\n\n{note_text}"
        else:
            content = step.rest
        thinking = "" if self.options.reasoning == "compress" else response.thinking
        return content, thinking, note_entry, draft_entry

    def _pin_pending_draft(
        self,
        tool_msgs: list[Message],
        draft_entry: NoteStoreEntry | None,
        run: _RunState,
    ) -> bool:
        """Append the turn's pending targeted draft to its last tool result.

        Returns whether the draft was placed, so the step block skips it.
        """
        if (
            draft_entry is None
            or draft_entry.draft is None
            or draft_entry.draft.status != "pending"
            or not draft_entry.draft.target
            or not tool_msgs
        ):
            return False
        block = render_draft_block(
            draft_entry.alias,
            draft_entry.step,
            draft_entry.draft,
            text=draft_entry.text,
        )
        checked = self._check_candidate(block, "draft", run)
        if checked:
            last = tool_msgs[-1]
            tool_msgs[-1] = dataclasses.replace(
                last, content=f"{last.content}\n{checked}"
            )
            return True
        return False

    @staticmethod
    def _checked_note_copy(entry: NoteStoreEntry) -> ReasoningNote:
        """*entry*'s note with only the fields its checked text still holds.

        The digest is built from the checked text the store keeps, never
        from raw fields a redaction or a cap changed: an item whose raw
        text is not in ``entry.text`` is left out, and the rest is a copy
        that the digest may read.
        """
        note = entry.note
        assert note is not None
        text = entry.text

        def kept(items: Sequence[str]) -> list[str]:
            return [item for item in items if item and item in text]

        return dataclasses.replace(
            note,
            plan="",
            found=kept(note.found),
            decided=kept(note.decided),
            open_questions=kept(note.open_questions),
            observed=kept(note.observed),
            outcome=note.outcome if note.outcome and note.outcome in text else "",
            stage="rehearsed",
        )

    def _build_digest(
        self, query: str, outcome: str, run: _RunState, task_id: str
    ) -> str:
        """Build, check and store the digest of the attempt that just ended."""
        notes = [
            self._checked_note_copy(entry)
            for entry in self.hippo.working.notes.of_kind("note")
            if entry.attempt == run.attempts
            and entry.stage == "rehearsed"
            and entry.note is not None
        ]
        note = digest_note(
            query,
            outcome or run.status,
            notes,
            task_id=task_id,
            attempt=run.attempts,
            step=run.steps,
            calls=run.attempt_calls,
            root=run.place or "",
        )
        note = dataclasses.replace(
            note,
            note_id=new_record_id(),
            call_seq=self._last_call_seq(),
            created_at=time.time(),
        )
        checked = self._check_candidate(render_digest(note), "digest", run)
        if checked is None:
            return ""
        entry = self.hippo.working.notes.add(
            "digest",
            checked,
            note=note,
            step=note.step,
            attempt=note.attempt,
            remember=True,
            remember_by="rule",
            task_id=task_id,
            title=note_title(note),
        )
        if entry is None:
            run.candidates_refused += 1
            return ""
        run.digests += 1
        run.events.note(
            step=note.step,
            attempt=note.attempt,
            note_id=note.note_id,
            kind="digest",
            source=note.source,
            text=checked,
            chars=len(checked),
            drafts=[],
            remember=True,
            stage=entry.stage,
        )
        provenance = self._memory_provenance(run, note=note)
        # Key the digest by the run's task, the same input recall hashes: the
        # checked text may have escaped a role token or cut a long plan, and a
        # key read from it would no longer match this run's task.
        provenance["task_key"] = digest_task_key(query, run.place or "")
        candidate = MemoryCandidate(
            candidate_id=note.note_id,
            source="digest",
            text=checked,
            remember_by="rule",
            task_id=task_id,
            session_id="",
            provenance=provenance,
            created_at=note.created_at,
            remember=True,
        )
        if digest_has_evidence(checked):
            # Keep the run's last evidence-bearing attempt. A final attempt
            # without evidence must not drop an episode whose earlier attempt
            # wrote a file or passed a check.
            run.final_digest = candidate
            self.hippo.propose(candidate)
        return checked

    async def _restate(self, note: ReasoningNote, run: _RunState) -> bool:
        """Compact *note*'s new items into an older note with the same facts.

        A later note whose ``found`` or ``decided`` items match an existing
        note's by normalized text restates it: the new items join the older
        note (deduplicated, caps kept), its rendered text passes the
        boundary again, and its stage becomes ``rehearsed``. Returns
        whether a restatement happened.
        """
        items = [*note.found, *note.decided]
        if not items:
            return False
        new_keys = {" ".join(item.split()).lower() for item in items}
        for entry in self.hippo.working.notes.of_kind("note"):
            older = entry.note
            if older is None or older is note or entry.stage == "superseded":
                continue
            if not any(
                " ".join(item.split()).lower() in new_keys
                for item in (*older.found, *older.decided)
            ):
                continue
            found = list(older.found)
            decided = list(older.decided)
            old_keys = {" ".join(item.split()).lower() for item in (*found, *decided)}
            for item in note.found:
                key = " ".join(item.split()).lower()
                if key not in old_keys and len(found) < NOTE_MAX_FOUND:
                    found.append(item)
                    old_keys.add(key)
            for item in note.decided:
                key = " ".join(item.split()).lower()
                if key not in old_keys and len(decided) < NOTE_MAX_DECIDED:
                    decided.append(item)
                    old_keys.add(key)
            source_ids = list(older.source_ids)
            for source_id in note.source_ids:
                if source_id not in source_ids:
                    source_ids.append(source_id)
            source_records = list(older.source_records)
            seen = {record_id for record_id, _text in source_records}
            for record in note.source_records:
                if record[0] not in seen:
                    source_records.append(record)
                    seen.add(record[0])
            remember = older.remember or note.remember
            remember_by = older.remember_by if older.remember else note.remember_by
            updated = dataclasses.replace(
                older,
                found=found,
                decided=decided,
                source_ids=source_ids,
                source_records=source_records,
                remember=remember,
                remember_by=remember_by,
                stage="rehearsed",
                uses=older.uses + 1,
                last_used_at=time.time(),
            )
            checked = self._check_candidate(
                render_note(updated, executor=True), "reasoning_note", run
            )
            if checked is None:
                return False
            if not self.hippo.working.notes.replace_text(entry.alias, checked, updated):
                return False
            self.hippo.working.notes.touch(entry.alias, use=True, stage="rehearsed")
            run.rehearsed["restated"] += 1
            # The new note's candidate is superseded by the merged one; a
            # request would otherwise be able to admit both (review finding 8).
            self.hippo.staged.drop(note.note_id)
            self._propose_note_candidate(updated, run)
            decisions = await self.hippo.admit(updated.note_id, "rehearsal")
            self._drain_decisions(decisions, run)
            return True
        return False

    @staticmethod
    def _call_succeeded(
        call: ToolCall, result: ToolResult, exit_codes: Sequence[int]
    ) -> bool:
        """Whether a call returned without an error and with exit status 0.

        *exit_codes* are the statuses the tool recorded in ``TOOL_EXIT_CODE``
        (see :meth:`_do_execute`). A rooted ``shell_exec`` prefixes a
        non-zero status in its text; an unrooted one records none, so its
        status is unknown there and the call is not counted as a success:
        a rehearsal signal must not come from a command that may have
        failed.
        """
        if result.error.strip():
            return False
        if call.tool_name != "shell_exec":
            return True
        if result.output.startswith("[exit code "):
            return result.output.startswith("[exit code 0]")
        return bool(exit_codes) and exit_codes[-1] == 0

    @staticmethod
    def _touched_names(call: ToolCall) -> list[str]:
        """The paths and command words a successful call acted on."""
        names: list[str] = []
        if call.tool_name in ("file_read", "file_write", "file_edit"):
            path = call.arguments.get("path")
            if isinstance(path, str) and path.strip():
                names.append(path)
                names.extend(path[start:end] for start, end in path_spans(path))
        elif call.tool_name == "shell_exec":
            command = call.arguments.get("command")
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            if isinstance(command, str) and command.strip():
                names.append(command)
                names.extend(command[start:end] for start, end in path_spans(command))
                head = command.split()[0]
                if head:
                    names.append(head)
        return [name for name in names if len(name) >= 3]

    @staticmethod
    def _mentions(text: str, name: str) -> bool:
        """Whether *text* names *name* as a path or a whole word.

        A path or file name is compared with :func:`same_path` against
        the paths :func:`~localmelo.melo.memory.notes.path_spans` finds in
        *text*; any other name (a command word) must stand between
        non-word characters, so ``ls`` does not match ``results``.
        """
        if path_spans(name):
            return any(
                same_path(text[start:end], name) for start, end in path_spans(text)
            )
        start = 0
        while True:
            index = text.find(name, start)
            if index < 0:
                return False
            before = text[index - 1] if index else ""
            after = text[index + len(name) : index + len(name) + 1]
            if not (before.isalnum() or before == "_") and not (
                after.isalnum() or after == "_"
            ):
                return True
            start = index + 1

    async def _rehearse_acted(
        self,
        call: ToolCall,
        result: ToolResult,
        run: _RunState,
        exit_codes: Sequence[int] = (),
    ) -> None:
        """Mark the notes a successful call acted on as rehearsed."""
        if not self._call_succeeded(call, result, exit_codes):
            return
        touched = self._touched_names(call)
        if not touched:
            return
        for entry in self.hippo.working.notes.of_kind("note"):
            note = entry.note
            if note is None or entry.stage == "superseded":
                continue
            text = " ".join((*note.found, *note.decided))
            if entry.stage == "rehearsed":
                continue
            if not any(self._mentions(text, name) for name in touched):
                continue
            note.stage = "rehearsed"
            self.hippo.working.notes.touch(entry.alias, stage="rehearsed")
            run.rehearsed["acted"] += 1
            decisions = await self.hippo.admit(note.note_id, "rehearsal")
            self._drain_decisions(decisions, run)

    async def _cite_notes(self, aliases: Sequence[str], run: _RunState) -> None:
        """Mark the notes a reflection cited in ``used_notes`` as rehearsed."""
        for alias in aliases:
            entry = self.hippo.working.notes.get(alias)
            if entry is None or entry.kind != "note" or entry.stage == "superseded":
                continue
            if entry.note is not None:
                entry.note.stage = "rehearsed"
            self.hippo.working.notes.touch(alias, use=True, stage="rehearsed")
            run.rehearsed["cited"] += 1
            if entry.note is not None:
                decisions = await self.hippo.admit(entry.note.note_id, "rehearsal")
                self._drain_decisions(decisions, run)

    @staticmethod
    def _used_notes(content: str) -> list[str]:
        """The string aliases of the reflection reply's ``used_notes``.

        The object is the one the gate's parser reads: the first object with a
        valid ``recommended_action`` (see :func:`_read_reflection`). A non-list
        or non-string entries are ignored.
        """
        found = _read_reflection(_coerce_str(content))
        if found is None:
            return []
        names = found[0].get("used_notes")
        if not isinstance(names, list):
            return []
        return [name for name in names if isinstance(name, str) and name]

    def _reflection_note_inputs(self, run: _RunState) -> tuple[list[str], list[str]]:
        """The notes lines and draft-title lines a reflection request carries."""
        notes = [
            entry
            for entry in self.hippo.working.notes.of_kind("note")
            if entry.attempt == run.attempts
            and entry.stage != "superseded"
            and entry.note is not None
        ][-REFLECTION_MAX_NOTES:]
        lines = [
            f"[{entry.alias}] note · step {entry.step} · "
            f"{entry.text[:REFLECTION_NOTE_CHARS]}"
            for entry in notes
        ]
        # A pending draft is store-global: an earlier attempt's draft that
        # was never written is still one the next step may need.
        drafts = [
            entry
            for entry in self.hippo.working.notes.of_kind("draft")
            if entry.draft is not None and entry.draft.status == "pending"
        ]
        titles: list[str] = []
        for entry in drafts:
            if entry.draft is None:
                continue
            title = (
                f"[{entry.alias}] draft · step {entry.step} · "
                f"{draft_title(dataclasses.replace(entry.draft, text=entry.text))}"
            )
            # The target is raw metadata; the line reaches a prompt only
            # after the candidate boundary.
            checked = self._check_candidate(title, "draft", run)
            if checked:
                titles.append(checked)
        return lines, titles
