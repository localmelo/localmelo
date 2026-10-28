from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.melo.memory.history import History
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.procedures.store import MemoryProcedures, Procedures
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
    ) -> None:
        self.embedding = embedding
        self._long_term_top_k = long_term_top_k
        self._tool_search_top_k = tool_search_top_k
        self.history = history if history is not None else History()
        self.working = WorkingMemory(max_len=short_term_max)
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
        await self.history.save_task(task)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return await self.history.get_task(task_id)

    # ── Step recording ──

    async def store_step(self, task_id: str, step: StepRecord) -> str:
        """Record step in history and return a summary string.

        Does NOT write to short/long memory.  The caller is responsible
        for routing the summary through ``Checker.check_memory_write``
        and then calling ``memorize()`` if allowed.
        """
        await self.history.add_step(task_id, step)

        summary = step.thought
        if step.tool_call:
            summary += f" [called {step.tool_call.tool_name}]"
        if step.tool_result:
            output = step.tool_result.output[:200]
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
        """
        if to_working:
            self.working.append(Message(role=role, content=text))

        if self.embedding:
            emb = await self.embedding.embed([text])
            await self.long.add(text=text, embedding=emb[0], metadata=metadata or {})

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

    async def forget_all(self) -> None:
        """Forget all of history and long-term memory.

        Working memory (the current conversation), procedural memory
        (``melo-procedures forget`` clears that) and the session store are
        separate and stay.
        """
        await self.history.forget_all()
        await self.long.forget_all()

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
            text = self._format_reflection_for_long(entry)
            if text.strip():
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
        long_results = await self.long.search(query_emb[0], top_k=self._long_term_top_k)
        return [
            Message(role="system", content=f"[memory] {text}")
            for text, _score, _meta in long_results
        ]

    # ── Tool hint extraction (pure text scan, no execution) ──

    def extract_tool_hints(self, messages: list[Message]) -> list[str]:
        """Extract tool-name hints from messages.

        Collects the names of structured tool calls, then scans contents
        for registered tool names and ``[called X]`` patterns. This is a
        read-only scan — no tools are resolved or executed.
        """
        tool_hints: list[str] = []
        all_tool_names = {t.name for t in self.tools.list_all()}
        for msg in messages:
            for tc in msg.tool_calls:
                if tc.tool_name and tc.tool_name not in tool_hints:
                    tool_hints.append(tc.tool_name)
            for name in all_tool_names:
                if name in msg.content and name not in tool_hints:
                    tool_hints.append(name)
            for match in re.findall(r"\[called (\w+)\]", msg.content):
                if match not in tool_hints:
                    tool_hints.append(match)
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
