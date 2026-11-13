"""The recall of the agent loop (the retriever plan of the explicit modes).

:class:`_RecallLoop` is a mixin of :class:`~localmelo.melo.agent.agent.Agent`: the
methods that find the documents a trigger may recall (the task's notes, the other
tasks' long-term entries), rank them with the plan's retriever, check and fit the
block, and answer the ``recall`` tool. It reads the attributes that
``Agent.__init__`` sets (``options``, ``hippo``, ``checker``, ``chat``, ``_plan``,
``_retrievers`` and ``_effective_kinds``) and keeps no state of its own; the
counters it changes belong to the run state it is passed.
"""

from __future__ import annotations

import contextvars
import dataclasses
import logging
import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from localmelo.melo.agent._chat import Chat
from localmelo.melo.agent._notes_loop import _NotesLoop
from localmelo.melo.agent.options import retrieval_refusal
from localmelo.melo.checker.payloads import MemoryCandidatePayload
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.contracts.redaction import redact_secrets
from localmelo.melo.memory.notes import thought_chunks
from localmelo.melo.memory.recall import (
    LONG_POOL,
    RECALL_BOUNDARY_CHARS,
    RECALL_END,
    RECALL_K_MAX,
    RECALL_LIVE_CHARS,
    RECALL_LONG_K,
    RECALL_THOUGHT_PER_TASK,
    RECALL_TURN_CHARS,
    RecallHit,
    RecallPlan,
    RecallRecord,
    build_step_query,
    compact_record,
    docs_from_entries,
    docs_from_notes,
    fit_recall_block,
    hit_from_doc,
    recall_span_chars,
    recall_spans,
    record_to_dict,
    relation_counts,
)
from localmelo.melo.memory.recall_gate import (
    digest_task_key,
    gate_long_docs,
    is_task_digest,
    same_task,
)
from localmelo.melo.memory.retrieval import build_retriever
from localmelo.melo.memory.retrieval.base import Doc, RetrievalResult, Retriever
from localmelo.melo.memory.short.note_store import NoteStoreEntry
from localmelo.melo.schema import Message, ToolCall, shorten

if TYPE_CHECKING:
    from localmelo.melo.agent.agent import _RunState

logger = logging.getLogger(__name__)


# The run a ``recall`` tool call belongs to, set by run_task beside
# CALL_RECORDER. The tool's handler reads it; outside a run it is None and
# the handler answers in one line.
RECALL_RUN: contextvars.ContextVar[_RunState | None] = contextvars.ContextVar(
    "RECALL_RUN", default=None
)


def _checked_title(text: str) -> str:
    """The first sentence of checked *text*, at most 120 characters."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    for separator in (". ", "。", "! ", "? "):
        head, found, _tail = first.partition(separator)
        if found and head.strip():
            return shorten(head.strip() + separator.strip(), 120)
    return shorten(first, 120)


class _RecallLoop(_NotesLoop):
    """The recall methods of :class:`~localmelo.melo.agent.agent.Agent`."""

    chat: Chat
    _plan: RecallPlan
    _retrievers: dict[str, Retriever]
    _effective_kinds: frozenset[str]

    def _retrieval_refusal(self) -> str | None:
        return retrieval_refusal(
            self.options.retriever,
            self._plan,
            self.hippo.embedding is not None,
            self.options.reasoning,
            memory=self.options.memory,
            placement=self.options.recall_placement,
            recall_at=self.options.recall_at or (),
            builtins=self.options.builtins,
        )

    def _get_retriever(self, name: str) -> Retriever:
        """The Agent's one retriever called *name*, built on first use."""
        retriever = self._retrievers.get(name)
        if retriever is not None:
            return retriever
        options: dict[str, Any] = {}
        if name == "embedding":
            options["min_score"] = self.options.recall_min_score
        retriever = build_retriever(
            name,
            send=self.chat.retrieve,
            embedding=self.hippo.embedding,
            **options,
        )
        self._retrievers[name] = retriever
        return retriever

    async def _long_docs(self, query: str, run: _RunState) -> list[Doc]:
        """The other tasks' active long entries that may answer *query*, aliased."""
        if "long" not in self._effective_kinds:
            return []
        try:
            rows = await self.hippo.recall_long(
                query,
                exclude_task_ids={run.task_id},
                pool=LONG_POOL,
                task_key=digest_task_key(run.task_text, run.place or ""),
                min_score=self.options.recall_min_score,
            )
        except ProviderError as exc:
            run.retrieve_errors += 1
            logger.info("Long-term recall of task %s failed: %s", run.task_id, exc)
            return []
        except (IndexError, ValueError) as exc:
            run.retrieve_errors += 1
            logger.info("Long-term recall of task %s failed: %s", run.task_id, exc)
            return []
        # The cosine floor applies only to an embedding score; BM25 has no
        # absolute floor, and docs_from_entries reads each row's retriever.
        docs = docs_from_entries(
            rows,
            exclude_task_ids={run.task_id},
            min_score=self.options.recall_min_score,
        )
        key = digest_task_key(run.task_text, run.place or "")
        docs = gate_long_docs(
            docs,
            key,
            ratio=self.options.cross_task_ratio,
            cap=self.options.cross_task_cap,
            cold=self.options.cross_task_cold,
        )[:RECALL_LONG_K]
        docs = [
            dataclasses.replace(
                doc,
                meta={
                    **doc.meta,
                    "task_relation": (
                        "same"
                        if same_task(doc, key)
                        else "cross" if is_task_digest(doc) else "unscoped"
                    ),
                },
            )
            for doc in docs
        ]
        # Same-task digests first; within each group the creation-time order.
        docs.sort(
            key=lambda doc: (
                doc.meta["task_relation"] != "same",
                float(doc.meta.get("created_at") or 0.0),
                doc.doc_id,
            )
        )
        return [
            dataclasses.replace(doc, meta={**doc.meta, "alias": f"m{index}"})
            for index, doc in enumerate(docs, start=1)
        ]

    def _hits_from_result(
        self,
        result: RetrievalResult,
        docs: Sequence[Doc],
        *,
        k: int,
    ) -> list[RecallHit]:
        """The task-local hits of *result*, at most *k*, in rank order."""
        by_id = {doc.doc_id: doc for doc in docs}
        hits: list[RecallHit] = []
        for item in result.hits:
            doc = by_id.get(item.doc_id)
            if doc is None:
                continue
            hits.append(hit_from_doc(doc, score=item.score, rank=len(hits) + 1))
            if len(hits) == k:
                break
        return hits

    def _merge_llm_hits(
        self,
        result: RetrievalResult,
        docs: Sequence[Doc],
        *,
        k: int,
        long_ids: set[str],
    ) -> list[RecallHit]:
        """The llm hits split into at most *k* task and 2 long, in its order."""
        by_id = {doc.doc_id: doc for doc in docs}
        hits: list[RecallHit] = []
        task_kept = 0
        long_kept = 0
        for item in result.hits:
            doc = by_id.get(item.doc_id)
            if doc is None:
                continue
            if doc.doc_id in long_ids:
                if long_kept >= RECALL_LONG_K:
                    continue
                long_kept += 1
                score = float(doc.meta.get("score") or item.score)
            else:
                if task_kept >= k:
                    continue
                task_kept += 1
                score = item.score
            hits.append(hit_from_doc(doc, score=score, rank=len(hits) + 1))
        return hits

    async def _run_recall(
        self,
        trigger: str,
        query: str,
        run: _RunState,
        *,
        boundary: bool = False,
        include_superseded: bool = False,
        k: int | None = None,
        placement: str = "",
        exclude: Iterable[str] = (),
    ) -> tuple[RecallRecord | None, str, list[RecallHit]]:
        """Run one recall trigger and return its record, block and hits.

        ``([], "")`` when the trigger is off, when nothing is a candidate,
        or when the block cannot be made to fit; a refused block counts
        ``recall_refused`` and returns nothing.
        """
        name = getattr(self._plan, trigger, None)
        if name is None:
            return None, "", []
        started = time.perf_counter()
        # The query can carry a tool call's command or path, so it is a
        # request path P4 must redact too: before an llm or embedding
        # retriever sees it (bm25 ranks on the raw text; decision 1).
        redacted_query, query_spans = redact_secrets(query)
        search_query = redacted_query if name in ("llm", "embedding") else query
        limit_k = (
            self.options.recall_k if k is None else max(1, min(int(k), RECALL_K_MAX))
        )
        excluded = set(exclude) | set(run.pending_shown)
        task_docs: list[Doc] = []
        if trigger != "start":
            task_docs = docs_from_notes(
                self.hippo.working.notes.entries(),
                task_id=run.task_id,
                kinds=self._effective_kinds,
                exclude_aliases=excluded,
                include_superseded=include_superseded,
            )
        long_docs = await self._long_docs(redacted_query, run)
        if long_docs:
            shown = "\n".join(
                [*recall_spans(self.hippo.working.get_window()), run.boundary_block]
            )
            long_docs = [
                doc
                for doc in long_docs
                if hit_from_doc(doc, score=0.0, rank=1).text not in shown
            ]
        if not task_docs and not long_docs:
            run.recall_empty += 1
            return None, "", []
        hits: list[RecallHit] = []
        retriever = name
        fallback = ""
        if name == "llm":
            docs = [*task_docs, *long_docs]
            long_ids = {doc.doc_id for doc in long_docs}
            result = await self._get_retriever("llm").search(
                search_query,
                docs,
                k=limit_k + min(len(long_docs), RECALL_LONG_K),
            )
            if search_query.strip():
                run.aux_retrieve_calls += 1
            self._count_retrieve_usage(result, run)
            retriever, fallback = result.retriever, result.fallback
            hits = self._merge_llm_hits(result, docs, k=limit_k, long_ids=long_ids)
        else:
            if task_docs:
                result = await self._get_retriever(name).search(
                    search_query, task_docs, k=limit_k
                )
                retriever, fallback = result.retriever, result.fallback
                self._count_retrieve_usage(result, run)
                hits = self._hits_from_result(result, task_docs, k=limit_k)
            for doc in long_docs:
                score = float(doc.meta.get("score") or 0.0)
                # The floor is a cosine floor; a BM25-scored row has none.
                if (
                    doc.meta.get("retrieval") != "bm25"
                    and score < self.options.recall_min_score - 1e-6
                ):
                    continue
                hits.append(hit_from_doc(doc, score=score, rank=len(hits) + 1))
        hits.sort(key=lambda hit: hit.task_relation != "same")
        hits = [
            dataclasses.replace(hit, rank=index) for index, hit in enumerate(hits, 1)
        ]
        if fallback:
            run.recall_fallbacks += 1
            run.retrieve_errors += 1
        if trigger == "start":
            hits = [hit for hit in hits if hit.kind == "long"][:RECALL_LONG_K]
        if not hits:
            run.recall_empty += 1
            return None, "", []
        candidates = len(task_docs) + len(long_docs)
        checked_hits: list[RecallHit] = []
        for hit in hits:
            text = hit.text
            checked = text
            if hit.kind == "long":
                checked_text = self._check_candidate(
                    text, "recall", run, target="prompt"
                )
                if checked_text is None:
                    continue
                checked = checked_text
            checked_hits.append(
                dataclasses.replace(
                    hit, text=checked, chars=len(checked), rank=len(checked_hits) + 1
                )
            )
        if not checked_hits:
            return None, "", []
        if boundary:
            limit = RECALL_BOUNDARY_CHARS
        else:
            live = RECALL_LIVE_CHARS - recall_span_chars(
                self.hippo.working.get_window()
            )
            limit = min(RECALL_TURN_CHARS, max(live, 0))
        if limit <= 0:
            run.recall_skipped += 1
            return None, "", []
        # The block's bytes do not carry the latency, so fitting is stable;
        # the record below keeps it for recall_json, the event and stats.
        ms = round((time.perf_counter() - started) * 1000)
        kept, block = fit_recall_block(
            checked_hits,
            trigger=trigger,
            retriever=retriever,
            placement=placement,
            limit=limit,
        )
        if not kept or not block:
            run.recall_skipped += 1
            return None, "", []
        checked_block = self._check_recall_block(block, run)
        if checked_block is None:
            return None, "", []
        run.redactions += query_spans
        record_hits = [
            dataclasses.replace(
                hit,
                doc_id=redact_secrets(hit.doc_id)[0],
                alias=redact_secrets(hit.alias)[0],
            )
            for hit in kept
        ]
        record = RecallRecord(
            trigger=trigger,
            placement=placement,
            retriever=retriever,
            query=redacted_query,
            k=limit_k,
            candidates=candidates,
            hits=record_hits,
            chars=len(checked_block),
            ms=ms,
            fallback=fallback,
            boundary=boundary,
            step=run.steps,
            attempt=run.attempts,
        )
        run.pending_recalls.append(record_to_dict(record))
        run.pending_recall_records.append(record)
        return record, checked_block, record_hits

    @staticmethod
    def _count_retrieve_usage(result: RetrievalResult, run: _RunState) -> None:
        if not result.usage:
            return
        run.retrieve_prompt_tokens += int(result.usage.get("prompt_tokens", 0) or 0)
        run.retrieve_completion_tokens += int(
            result.usage.get("completion_tokens", 0) or 0
        )

    @staticmethod
    def _commit_recalls(records: Sequence[RecallRecord], run: _RunState) -> None:
        """Count, log and emit the recall records a request has just carried."""
        for record in records:
            run.recalls += 1
            run.recalls_by_trigger[record.trigger] = (
                run.recalls_by_trigger.get(record.trigger, 0) + 1
            )
            run.recall_hits += len(record.hits)
            counts = relation_counts(record.hits)
            run.recall_same_task += counts["same_task"]
            run.recall_cross_task += counts["cross_task"]
            run.recall_task_notes += counts["task_notes"]
            run.recall_chars += record.chars
            run.recall_ms += record.ms
            run.recall_log.append(compact_record(record))
            run.events.recall(record)

    def _check_recall_block(self, text: str, run: _RunState) -> str | None:
        """Run one whole recall block through the candidate boundary."""
        run.candidates += 1
        check = self.checker.check_memory_candidate(
            MemoryCandidatePayload(text=text, source="recall", target="prompt")
        )
        if not check.allowed:
            run.candidates_refused += 1
            run.recall_refused += 1
            logger.debug(
                "Refused a recall block of task %s: %s", run.task_id, check.reason
            )
            return None
        payload = check.sanitized_payload
        run.redactions += int(payload.metadata.get("redactions", 0) or 0)
        checked = str(payload.text)
        if not checked.endswith(RECALL_END):
            run.recall_refused += 1
            logger.debug(
                "Refused a recall block of task %s: the boundary cut its "
                "closing delimiter",
                run.task_id,
            )
            return None
        return checked

    async def _step_recall(
        self,
        run: _RunState,
        note_entry: NoteStoreEntry | None,
        calls: Sequence[ToolCall],
        *,
        fallback: str,
        exclude: Iterable[str] = (),
    ) -> tuple[str, list[RecallHit]]:
        """The step trigger's block for the next planning call, and its hits.

        A note from an earlier attempt is ``superseded`` at that attempt's
        boundary, but its turn can still be evicted from the window, and
        R5's step trigger is what makes an evicted turn reachable again;
        step recall therefore ranks superseded notes too. The window's
        aliases still exclude what the request already shows.
        """
        if self._plan.step is None:
            return "", []
        note_text = "" if note_entry is None else note_entry.text
        outcome = ""
        if note_entry is not None and note_entry.note is not None:
            outcome = note_entry.note.outcome
        query = build_step_query(note_text, calls, fallback=fallback, outcome=outcome)
        _record, block, hits = await self._run_recall(
            "step",
            query,
            run,
            placement=self.options.recall_placement,
            exclude=exclude,
            include_superseded=True,
        )
        if block:
            run.pending_shown.update(hit.alias for hit in hits if hit.kind != "long")
        return block, hits

    async def _store_thought_chunks(
        self, response: Message, run: _RunState
    ) -> list[str]:
        """Store up to the task's remaining thought chunks from *response*.

        Returns the aliases of the chunks stored, so the caller can mark
        them shown with the reply's turn.
        """
        if "thought" not in self._effective_kinds or not self._notes_mode:
            return []
        stored = sum(
            1
            for entry in self.hippo.working.notes.of_kind("thought")
            if entry.task_id == run.task_id
        )
        remaining = RECALL_THOUGHT_PER_TASK - max(run.thought_chunks, stored)
        if remaining <= 0 or not response.thinking:
            return []
        aliases: list[str] = []
        for chunk in thought_chunks(response.thinking)[:remaining]:
            checked = self._check_candidate(chunk.text, "thought", run)
            if checked is None:
                continue
            entry = self.hippo.working.notes.add(
                "thought",
                checked,
                step=run.steps,
                attempt=run.attempts,
                task_id=run.task_id,
                title=_checked_title(checked),
            )
            if entry is not None:
                run.thought_chunks += 1
                aliases.append(entry.alias)
        return aliases

    async def _recall_tool(self, query: str, k: int | None = None) -> str:
        """The ``recall`` tool: rank the task's documents for *query*."""
        run = RECALL_RUN.get()
        if run is None:
            return "recall is only available inside a run"
        if not isinstance(query, str) or not query.strip():
            return "recall needs a non-empty query"
        if isinstance(k, bool) or not isinstance(k, int):
            k = self.options.recall_k
        refused_before = run.recall_refused
        _record, block, hits = await self._run_recall(
            "tool",
            query,
            run,
            include_superseded=True,
            k=k,
            placement="tool-result",
            exclude=self.hippo.working.shown_aliases(),
        )
        if not block:
            if run.recall_refused > refused_before:
                return "recall refused: the block did not pass the boundary"
            return "recall found nothing for that query"
        run.pending_shown.update(hit.alias for hit in hits if hit.kind != "long")
        return block
