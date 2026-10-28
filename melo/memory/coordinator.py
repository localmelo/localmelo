from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import replace
from typing import Any

from localmelo.melo.checker.checker import Checker
from localmelo.melo.checker.payloads import MemoryCandidatePayload
from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.melo.memory.hippo import (
    SIGNALS,
    CandidateItem,
    HippoPolicy,
    MemoryCandidate,
    MemoryDecision,
    StagedCandidates,
    classify_kind,
)
from localmelo.melo.memory.history import History
from localmelo.melo.memory.long import LongTerm, LongTermSchemaError
from localmelo.melo.memory.procedures.store import MemoryProcedures, Procedures
from localmelo.melo.memory.recall_gate import digest_has_evidence
from localmelo.melo.memory.short import WorkingMemory
from localmelo.melo.memory.tools import ToolRegistry
from localmelo.melo.schema import (
    LONG_TERM_TOP_K,
    SHORT_TERM_MAX,
    TOOL_SEARCH_TOP_K,
    Message,
    ReflectionEntry,
    StepRecord,
    TaskRecord,
    ToolDef,
)

logger = logging.getLogger(__name__)


def _exception_text(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _redacted_text(text: str) -> str:
    """*text* with the secrets of :func:`redact_secrets` replaced."""
    return redact_secrets(text)[0]


# Provenance is stored beside the entry, so every text it carries is bounded
# and redacted like the entry text (P4, T53R finding 2).
PROVENANCE_TEXT_CHARS = 300
PROVENANCE_LIST_MAX = 20


def _bounded_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """*provenance* with its texts redacted and bounded, for the store.

    The policy compares the raw ``task_text`` first; this runs just before
    the write, so a candidate's route (the Agent or ``--route``) cannot
    leave a secret in ``provenance_json``.
    """
    clean = dict(provenance)
    task_text, _task_redactions = redact_secrets(str(provenance.get("task_text") or ""))
    clean["task_text"] = task_text[:PROVENANCE_TEXT_CHARS]
    for key in ("paths", "commands"):
        values = provenance.get(key) or ()
        if isinstance(values, (list, tuple)):
            values = values[:PROVENANCE_LIST_MAX]
        else:
            values = ()
        clean[key] = [
            redact_secrets(value)[0][:PROVENANCE_TEXT_CHARS]
            for value in values
            if isinstance(value, str)
        ]
    return clean


def _redacted_value(value: Any) -> Any:
    """A tool argument with the secrets of its text replaced.

    A call's arguments are a dict of JSON values; every string in it, at any
    depth, is redacted, and every other value stays as it is. The walk keeps
    its own stack, so arguments nested past the recursion limit (which a
    provider decodes from a reply all the same) do not raise
    ``RecursionError``.
    """
    if isinstance(value, str):
        return _redacted_text(value)
    if not isinstance(value, (dict, list)):
        return value
    root: Any = {} if isinstance(value, dict) else []
    todo: list[tuple[Any, Any]] = [(value, root)]
    while todo:
        source, target = todo.pop()
        pairs = source.items() if isinstance(source, dict) else enumerate(source)
        for name, item in pairs:
            if isinstance(item, str):
                redacted: Any = _redacted_text(item)
            elif isinstance(item, (dict, list)):
                redacted = {} if isinstance(item, dict) else []
                todo.append((item, redacted))
            else:
                redacted = item
            if isinstance(target, dict):
                target[name] = redacted
            else:
                target.append(redacted)
    return root


def _redacted_step(step: StepRecord) -> StepRecord:
    """The copy of *step* that history stores, with its secrets redacted."""
    call = step.tool_call
    if call is not None:
        call = replace(call, arguments=_redacted_value(call.arguments))
    result = step.tool_result
    if result is not None:
        result = replace(
            result,
            output=_redacted_text(result.output),
            error=_redacted_text(result.error),
        )
    return replace(
        step,
        thought=_redacted_text(step.thought),
        tool_call=call,
        tool_result=result,
    )


def _redacted_task(task: TaskRecord) -> TaskRecord:
    """The copy of *task* that history stores, with its texts redacted.

    The result is redacted like the query: an answer often quotes the code
    or the key a tool returned.
    """
    return replace(
        task,
        query=_redacted_text(task.query),
        result=_redacted_text(task.result),
    )


class Hippo:
    """Memory coordinator: working memory, long-term, history, procedures and tools.

    When ``embedding`` is None, long-term memory operations are silently
    skipped and the agent runs with working-memory context only.

    Optional ``history``, ``long`` and ``procedures`` parameters allow
    injecting persistent backends (e.g. :class:`SqliteHistory`,
    :class:`SqliteLongTerm`, :class:`SqliteProcedures`). When omitted the
    in-memory defaults are used. ``procedures`` is the procedural store,
    the step templates that runs learn at their end; nothing in it reaches
    a prompt.
    """

    def __init__(
        self,
        embedding: BaseEmbeddingProvider | None = None,
        short_term_max: int = SHORT_TERM_MAX,
        long_term_top_k: int = LONG_TERM_TOP_K,
        tool_search_top_k: int = TOOL_SEARCH_TOP_K,
        *,
        history: History | None = None,
        long: LongTerm | None = None,
        procedures: Procedures | None = None,
        short_term_empty_prefix: bool = False,
        memory: bool = True,
        admit_on_outcome: bool = False,
        keep_decision_text: bool = False,
    ) -> None:
        self.embedding = embedding
        self.memory = memory
        self.admit_on_outcome = admit_on_outcome
        self.keep_decision_text = keep_decision_text
        # R6-lite's typed write path exists only for a real long store: an
        # injected backend, never the in-memory fallback of a legacy run.
        self._store_enabled = long is not None and memory
        self._store_broken = False
        self._store_error_logged = False
        self.staged = StagedCandidates()
        self.policy = HippoPolicy()
        self._checker = Checker()
        self._long_term_top_k = long_term_top_k
        self._tool_search_top_k = tool_search_top_k
        self.history = history if history is not None else History()
        self.working = WorkingMemory(
            max_len=short_term_max, empty_prefix=short_term_empty_prefix
        )
        self.long = long if long is not None else LongTerm()
        self.procedures: Procedures = (
            procedures if procedures is not None else MemoryProcedures()
        )
        self.tools = ToolRegistry()

    @property
    def short(self) -> WorkingMemory:
        """Backward-compat alias for ``self.working``."""
        return self.working

    def close(self) -> None:
        """Close persistent backends if they support it.

        Prefer :meth:`aclose` for SQLite-backed memory: the sync shim only
        drops references and lets GC release file descriptors. This stays
        for callers that don't have an event loop available.
        """
        for backend in (self.history, self.long, self.procedures):
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                close_fn()

    async def aclose(self) -> None:
        """Async close path for backends that hold open async resources.

        Falls back to sync :meth:`close` semantics for backends that only
        expose ``close()``.
        """
        for backend in (self.history, self.long, self.procedures):
            aclose_fn = getattr(backend, "aclose", None)
            if callable(aclose_fn):
                await aclose_fn()
                continue
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                close_fn()

    # ── Tool management ──

    def register_tool(self, tool: ToolDef) -> None:
        self.tools.register(tool)

    def get_tool(self, name: str) -> ToolDef | None:
        return self.tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        return self.tools.list_all()

    # ── Task management ──

    async def save_task(self, task: TaskRecord) -> None:
        """Record *task* in history, with its texts redacted.

        The copy that history keeps has the secrets of :func:`redact_secrets`
        replaced, in its query and its result; *task* itself is untouched,
        because a run still reads it.
        """
        await self.history.save_task(_redacted_task(task))

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return await self.history.get_task(task_id)

    # ── Step recording ──

    async def store_step(self, task_id: str, step: StepRecord) -> str:
        """Record step in history and return a summary string.

        History keeps a copy with the secrets of :func:`redact_secrets`
        replaced, in its thought, tool arguments, tool output and error;
        *step* itself is untouched, because working memory and the model
        still read it. The summary is built from the stored copy, so the
        text a caller passes to :meth:`memorize` is redacted too.

        Does NOT write to short/long memory.  The caller is responsible
        for routing the summary through ``Checker.check_memory_write``
        and then calling ``memorize()`` if allowed.
        """
        stored = _redacted_step(step)
        await self.history.add_step(task_id, stored)

        summary = stored.thought
        if stored.tool_call:
            summary += f" [called {stored.tool_call.tool_name}]"
        if stored.tool_result:
            output = stored.tool_result.output[:200]
            summary += f" -> {output}"

        return summary

    async def memorize(
        self,
        text: str,
        role: str = "assistant",
        metadata: dict[str, Any] | None = None,
        *,
        to_working: bool = True,
    ) -> None:
        """Write text to working memory and (if embedding available) long-term memory.

        Pass ``to_working=False`` to write long-term memory only. The agent
        does this for step summaries, because working memory already holds
        the real tool-call turn for the current task.

        Long-term memory keeps *text* with the secrets of
        :func:`redact_secrets` replaced. Working memory keeps it as it is:
        it is what the model reads.
        """
        if to_working:
            self.working.append(Message(role=role, content=text))

        if self.embedding and not self._store_broken:
            stored = _redacted_text(text)
            try:
                emb = await self.embedding.embed([stored])
                await self.long.add(
                    text=stored, embedding=emb[0], metadata=metadata or {}
                )
            except LongTermSchemaError as exc:
                self._disable_store(exc)

    # ── R6-lite: admission ──

    def propose(self, candidate: MemoryCandidate) -> None:
        """Stage *candidate*; a no-op without a long store."""
        if not self._store_ready():
            return
        self.staged.propose(candidate)

    async def admit(self, candidate_id: str, signal: str) -> list[MemoryDecision]:
        """Route one staged candidate's items through the boundary and policy."""
        if signal not in SIGNALS:
            raise ValueError(f"Unknown admission signal: {signal!r}")
        if not self._store_ready():
            return []
        candidate = self.staged.get(candidate_id)
        if candidate is None or candidate.processed:
            return []
        items = candidate.items or (CandidateItem(candidate.text),)
        decisions: list[MemoryDecision] = []
        for index, item in enumerate(items):
            item_provenance = {
                **candidate.provenance,
                "candidate_id": candidate.candidate_id,
                "item_index": index,
            }
            task_text = str(candidate.provenance.get("task_text") or "")
            kind = classify_kind(
                item.text, source=candidate.source, task_text=task_text
            )
            check = self._checker.check_memory_candidate(
                MemoryCandidatePayload(
                    text=item.text,
                    source=candidate.source,
                    target="long",
                    kind=kind,
                )
            )
            if not check.allowed:
                decisions.append(
                    MemoryDecision(
                        candidate_id=candidate.candidate_id,
                        action="",
                        kind=kind,
                        admitted_by=signal,
                        reason=check.reason,
                        task_id=candidate.task_id,
                        refused=True,
                    )
                )
                continue
            payload = check.sanitized_payload
            checked = str(payload.text)
            if candidate.source == "digest" and not digest_has_evidence(checked):
                decisions.append(
                    MemoryDecision(
                        candidate_id=candidate.candidate_id,
                        action="",
                        kind=kind,
                        admitted_by=signal,
                        reason="digest has no file or verification evidence",
                        task_id=candidate.task_id,
                        refused=True,
                    )
                )
                continue
            redactions = int((payload.metadata or {}).get("redactions", 0) or 0)
            source_ids = (
                tuple(candidate.provenance.get("source_ids") or ())
                if item.source_ids is None
                else item.source_ids
            )
            try:
                active = await self.long.active_entries(kind)
            except LongTermSchemaError as exc:
                self._disable_store(exc)
                return decisions
            except Exception as exc:  # a locked or damaged store
                logger.warning(
                    "Could not read long-term memory for task %s: %s",
                    candidate.task_id,
                    _exception_text(exc),
                )
                continue
            if candidate.source == "digest":
                key = item_provenance.get("task_key")
                active = [
                    entry
                    for entry in active
                    if key
                    and entry.source == "digest"
                    and entry.provenance.get("task_key") == key
                ]
            try:
                decision = self.policy.decide(
                    checked,
                    kind=kind,
                    source=candidate.source,
                    provenance=item_provenance,
                    active=active,
                    admitted_by=signal,
                )
            except Exception as exc:  # malformed stored data or a policy bug
                logger.warning("Could not route a memory candidate: %s", exc)
                continue
            try:
                entry_id, action = await self.long.apply(
                    text=checked,
                    kind=kind,
                    source=candidate.source,
                    admitted_by=signal,
                    provenance=_bounded_provenance(
                        {**item_provenance, "source_ids": list(source_ids)}
                    ),
                    task_id=candidate.task_id,
                    session_id=candidate.session_id,
                    created_at=candidate.created_at,
                    embedding=None,
                    action=decision.action,
                    target_id=decision.target_id,
                    reason=decision.reason,
                    source_ids=source_ids,
                    keep_text=self.keep_decision_text,
                    candidate_id=candidate.candidate_id,
                )
            except LongTermSchemaError as exc:
                self._disable_store(exc)
                return decisions
            except Exception as exc:  # a write failure is logged, not raised
                logger.warning(
                    "Could not write a memory candidate of task %s: %s",
                    candidate.task_id,
                    _exception_text(exc),
                )
                continue
            decisions.append(
                MemoryDecision(
                    candidate_id=candidate.candidate_id,
                    action=action,
                    kind=kind,
                    admitted_by=signal,
                    entry_id=entry_id,
                    target_id=decision.target_id,
                    reason=decision.reason,
                    source_ids=source_ids,
                    task_id=candidate.task_id,
                    redactions=redactions,
                )
            )
        written = next(
            (item.entry_id for item in decisions if item.entry_id is not None), None
        )
        candidate.processed = True
        candidate.entry_id = written
        self.staged.update(candidate)
        return decisions

    async def flush(
        self,
        *,
        run_request: bool,
        digest: MemoryCandidate | None,
        final_reflection: MemoryCandidate | None,
        task_id: str = "",
    ) -> list[MemoryDecision]:
        """Admit the run's candidates at its end; never raises.

        *task_id* scopes the run's candidates in a long-lived Agent: without
        it a candidate of an earlier turn stays staged and a later request
        would admit it. It is optional so a direct Hippo test can omit it.
        """
        try:
            return await self._flush(
                run_request=run_request,
                digest=digest,
                final_reflection=final_reflection,
                task_id=task_id,
            )
        except Exception as exc:  # a flush never replaces the run's result
            logger.warning("Could not flush the memory candidates: %s", exc)
            return []

    async def _flush(
        self,
        *,
        run_request: bool,
        digest: MemoryCandidate | None,
        final_reflection: MemoryCandidate | None,
        task_id: str = "",
    ) -> list[MemoryDecision]:
        if not self._store_ready():
            return []
        scope = task_id or (digest.task_id if digest is not None else "")
        selected: list[tuple[MemoryCandidate, str]] = []
        if run_request:
            for candidate in self.staged.all():
                if scope and candidate.task_id != scope:
                    continue
                if not candidate.remember or candidate.processed:
                    continue
                if candidate.source == "digest":
                    continue  # earlier attempt digests stay out
                if (
                    final_reflection is not None
                    and candidate.candidate_id == final_reflection.candidate_id
                ):
                    continue
                selected.append((candidate, "user_request"))
            if digest is not None:
                selected.append((digest, "user_request"))
        elif self.admit_on_outcome and digest is not None:
            selected.append((digest, "outcome"))
        decisions: list[MemoryDecision] = []
        for candidate, signal in selected:
            if self.staged.get(candidate.candidate_id) is None:
                self.staged.propose(candidate)
            decisions.extend(await self.admit(candidate.candidate_id, signal))
        return decisions

    def _store_ready(self) -> bool:
        return self.memory and self._store_enabled and not self._store_broken

    def _disable_store(self, exc: Exception) -> None:
        """Stop using a store a newer localmelo wrote, with one WARNING."""
        if not self._store_error_logged:
            logger.warning(
                "Long-term memory is off for this run: %s", _exception_text(exc)
            )
            self._store_error_logged = True
        self._store_enabled = False
        self._store_broken = True

    async def recall_long(
        self,
        query: str,
        *,
        exclude_task_ids: Collection[str] = (),
        kinds: Collection[str] | None = None,
        pool: int = 20,
        task_key: str = "",
        min_score: float = 0.0,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Up to *pool* active entries, by embedding or by BM25."""
        if not self.memory or self._store_broken:
            return []
        try:
            if await self.long.count_entries() == 0:
                return []
            # Each row is tagged with the retriever that scored it, so the
            # cosine floor never drops a BM25 row (T53R finding 3).
            if self.embedding is not None:
                vectors = await self.embedding.embed([query])
                raw_rows = await self.long.search_embedding(
                    vectors[0],
                    top_k=pool,
                    kinds=kinds,
                    exclude_task_ids=exclude_task_ids,
                )
                rows = [
                    (text, score, {**meta, "retrieval": "embedding"})
                    for text, score, meta in raw_rows
                ]
                # R6-lite's typed writes are text-only, so the embedding
                # search cannot see them; the BM25 rows fill the pool.
                text_rows = await self.long.search_text(
                    query,
                    top_k=pool,
                    kinds=kinds,
                    exclude_task_ids=exclude_task_ids,
                )
                seen = {row[2].get("id") for row in raw_rows}
                rows.extend(
                    (text, score, {**meta, "retrieval": "bm25"})
                    for text, score, meta in text_rows
                    if meta.get("id") not in seen
                )
                rows = rows[:pool]
            else:
                rows = [
                    (text, score, {**meta, "retrieval": "bm25"})
                    for text, score, meta in await self.long.search_text(
                        query,
                        top_k=pool,
                        kinds=kinds,
                        exclude_task_ids=exclude_task_ids,
                    )
                ]
            same_rows: list[tuple[str, float, dict[str, Any]]] = []
            if task_key:
                if self.embedding is not None:
                    same_rows.extend(
                        (text, score, {**meta, "retrieval": "embedding"})
                        for text, score, meta in await self.long.search_embedding(
                            vectors[0],
                            top_k=2,
                            kinds=kinds,
                            exclude_task_ids=exclude_task_ids,
                            task_key=task_key,
                        )
                        if score >= min_score - 1e-6
                    )
                same_rows.extend(
                    (text, score, {**meta, "retrieval": "bm25"})
                    for text, score, meta in await self.long.search_text(
                        query,
                        top_k=2,
                        kinds=kinds,
                        exclude_task_ids=exclude_task_ids,
                        task_key=task_key,
                    )
                )
                ids = {row[2].get("id") for row in same_rows}
                rows = same_rows + [row for row in rows if row[2].get("id") not in ids]
                # A vector row and its text row can describe the same entry.
                unique: dict[Any, tuple[str, float, dict[str, Any]]] = {}
                for row in rows:
                    unique.setdefault(row[2].get("id"), row)
                rows = list(unique.values())
            return [
                (text, score, {**meta, "retrieval_rank": rank})
                for rank, (text, score, meta) in enumerate(rows, start=1)
            ]
        except LongTermSchemaError as exc:
            self._disable_store(exc)
            return []

    # ── Forgetting ──

    async def forget(self, task_id: str) -> None:
        """Forget one task: its history rows and its long-term entries.

        History loses the task's ``tasks``, ``steps`` and ``llm_calls``
        rows; long-term memory loses the entries whose metadata names
        *task_id* as their task, which today are the reflections of its
        runs. A step summary names only its step, so it stays in long-term
        memory until :meth:`forget_all`. Working memory (the current
        conversation) and procedural memory are untouched.
        """
        await self.history.forget(task_id)
        await self.long.forget(task_id)
        self.staged.drop_task(task_id)

    async def forget_all(self) -> None:
        """Forget all of history and long-term memory.

        Working memory (the current conversation), procedural memory
        (``melo-procedures forget`` clears that) and the session store are
        separate and stay.
        """
        await self.history.forget_all()
        await self.long.forget_all()
        self.staged.clear()

    # ── Reflection promotion ──

    async def promote_reflections(self, task_id: str) -> None:
        """Promote reflection entries to long-term memory and clear them.

        Called when a run ends, whatever its status.
        """
        reflections = self.working.get_reflections()
        if not reflections or not self.embedding:
            self.working.clear_reflections()
            return

        for entry in reflections:
            text = _redacted_text(self._format_reflection_for_long(entry))
            if text.strip() and not self._store_broken:
                try:
                    emb = await self.embedding.embed([text])
                    await self.long.add(
                        text=text,
                        embedding=emb[0],
                        metadata={
                            "task_id": task_id,
                            "attempt_id": entry.attempt_id,
                            "type": "reflection",
                        },
                    )
                except LongTermSchemaError as exc:
                    self._disable_store(exc)
        self.working.clear_reflections()

    @staticmethod
    def _format_reflection_for_long(entry: ReflectionEntry) -> str:
        parts = [f"Attempt {entry.attempt_id}: {entry.summary}"]
        if entry.failed_hypotheses:
            parts.append(f"Failed: {'; '.join(entry.failed_hypotheses)}")
        if entry.useful_evidence:
            parts.append(f"Evidence: {'; '.join(entry.useful_evidence)}")
        if entry.recommended_avoids:
            parts.append(f"Avoid: {'; '.join(entry.recommended_avoids)}")
        return " | ".join(parts)

    # ── Level 1: Context Retrieval (embedding search only) ──

    async def retrieve_context(self, query: str) -> list[Message]:
        """Return long-term memory messages relevant to the query.

        Only searches long-term memory via embedding similarity.
        Does NOT include the short-term window — the caller assembles
        the full prompt from retrieval results + short-term separately.
        Does NOT extract tool hints — that is a separate stage.
        """
        if not self.embedding:
            return []

        query_emb = await self.embedding.embed([query])
        try:
            long_results = await self.long.search(
                query_emb[0], top_k=self._long_term_top_k
            )
        except LongTermSchemaError as exc:
            # A newer localmelo wrote this file: one WARNING, then the run
            # continues without long-term memory (Requirement 4, T53R
            # finding 5). The legacy path had no catch at all.
            self._disable_store(exc)
            return []
        return [
            Message(role="system", content=f"[memory] {text}")
            for text, _score, _meta in long_results
        ]

    # ── Tool hint extraction (the model's tool calls only) ──

    def extract_tool_hints(self, messages: list[Message]) -> list[str]:
        """Return the names of the tools called in *messages*, in order.

        Only structured tool calls count. A message that merely names a
        tool — user text, a tool result, a memory or a reflection — adds
        nothing, so a run's own text cannot expose a tool to the model.
        This is a read-only scan — no tools are resolved or executed.
        """
        tool_hints: list[str] = []
        for msg in messages:
            for tc in msg.tool_calls:
                if tc.tool_name and tc.tool_name not in tool_hints:
                    tool_hints.append(tc.tool_name)
        return tool_hints

    # ── Level 2: Tool Resolution (BM25 over registry) ──

    def resolve_tools(
        self,
        query: str,
        hints: list[str] | None = None,
        *,
        exclude: Collection[str] = (),
    ) -> list[ToolDef]:
        """Return the registered tools that *hints* name, then BM25's matches.

        The hinted tools come first, in hint order. Then come the best
        ``tool_search_top_k`` matches of a BM25 search over *query*, less
        the ones hinted already. The tools named in *exclude* take no part:
        no hint resolves to them, and the search ranks the other tools
        among themselves.
        """
        results: dict[str, ToolDef] = {}

        for name in hints or []:
            if name in exclude:
                continue
            td = self.tools.get(name)
            if td:
                results[td.name] = td

        for td in self.tools.search(
            query, top_k=self._tool_search_top_k, exclude=exclude
        ):
            if td.name not in results:
                results[td.name] = td

        return list(results.values())
