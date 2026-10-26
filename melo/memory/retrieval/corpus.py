"""Corpora, labelled queries and raw model calls, as JSON lines.

An offline evaluation reads two files, and an export writes the first:

- a **corpus** line is one document (:func:`load_corpus`,
  :func:`doc_to_dict`);
- a **labelled-query** line is one query, the ids of the documents that
  answer it and the documents it may see (:func:`load_queries`).

A corpus comes from a run's model calls. A **raw-call** line is one call, as
the benchmark records it (:func:`load_raw_calls`), and a run's
``history.db`` holds the same calls once localmelo records them in its
``llm_calls`` table (:func:`load_history_calls`). :func:`docs_from_calls`
turns them into documents: a note for each planning call, a draft of its
code blocks, a reflection for each reflection call whose reply holds one,
and thought chunks of the reasoning.

Each loader skips blank lines and raises :class:`CorpusError`, with the
1-based line number, for anything else that is wrong. Unknown keys are
ignored.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from localmelo.melo.checker.sanitize import sanitize_assistant_text
from localmelo.melo.contracts.providers import LLM_REQUEST_KINDS
from localmelo.melo.memory.notes import (
    DRAFT_WRITTEN_SHARE,
    NOTE_ITEM_CHARS,
    NOTE_PLAN_CHARS,
    TITLE_CHARS,
    Draft,
    ReasoningNote,
    _collapse,
    _shorten,
    draft_coverage,
    extract_draft,
    note_from_reply,
    note_title,
    render_note,
    same_path,
    thought_chunks,
)
from localmelo.melo.memory.retrieval.base import DOC_KINDS, Doc, int_step
from localmelo.melo.memory.retrieval.evaluate import LabelledQuery
from localmelo.melo.memory.retrieval.llm import _strip_code_fences
from localmelo.melo.schema import RAW_ARGUMENTS_KEY, ToolCall, ToolResult

# The call kinds, shared with the agent that records them in llm_calls.
CALL_KINDS = LLM_REQUEST_KINDS

# The decisions a reflection may recommend: a copy of REFLECTION_ACTIONS in
# melo/agent/agent.py (see _find_reflection below).
REFLECTION_ACTIONS = ("continue", "stop", "decompose", "done")
CALL_OUTCOMES = ("tools", "answer", "empty", "aux")

# The note kind of a planning call, by its outcome.
_NOTE_KINDS = {"tools": "step", "answer": "answer", "empty": "empty"}

# The keys a labelled query's scope may have.
_SCOPE_KEYS = ("task_id", "before_step", "exclude_ids")

# The first bytes of every SQLite database file.
SQLITE_HEADER = b"SQLite format 3\x00"

# The llm_calls columns that load_history_calls reads, by name.
HISTORY_CALL_COLUMNS = (
    "task_id",
    "seq",
    "kind",
    "step",
    "attempt",
    "outcome",
    "content",
    "thinking",
    "tool_calls_json",
    "step_ids_json",
    "timestamp",
)


class CorpusError(ValueError):
    """An input line, or a history row, that breaks the rules of its format.

    ``line`` is the 1-based number of the input line at fault, or ``None``
    when no line applies.
    """

    def __init__(self, message: str, *, line: int | None = None) -> None:
        super().__init__(message)
        self.line = line


class NoCallsTableError(CorpusError):
    """A history database without the ``llm_calls`` table.

    localmelo wrote such databases before it recorded its model calls.
    Their ``steps`` table cannot stand in: it has one row per executed
    tool call, with the reply's text only on the first, no reasoning, and
    no row for answers, empty replies or reflections.
    """


@dataclass(frozen=True)
class RawCall:
    """One model call of a run, as a raw-call line holds it."""

    task_id: str
    seq: int
    kind: str = "plan"  # plan | reflect | retrieve | estimate
    step: int = 0
    attempt: int = 1
    content: str = ""
    thinking: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    # Paired with tool_calls by index, and never more numerous: a call
    # without a result never ran.
    tool_results: tuple[ToolResult, ...] = ()
    outcome: str = ""  # tools | answer | empty | aux
    timestamp: float = 0.0


# ── Loaders ──


def load_corpus(lines: Iterable[str]) -> list[Doc]:
    """The documents of corpus *lines*, one JSON object per line.

    ``{"doc_id": "t-en:n3", "kind": "note", "title": "…", "text": "…",
    "meta": {"task_id": "t-en", "step": 2, …}}``. Required: ``doc_id``
    (unique in the file, without whitespace), ``kind`` (one of
    :data:`~localmelo.melo.memory.retrieval.base.DOC_KINDS`) and ``text``
    (a non-empty string). Optional: ``title`` (a string) and ``meta`` (an
    object, whose ``task_id`` is a string and ``step`` an integer >= 0
    when present).
    """
    docs: list[Doc] = []
    first_line: dict[str, int] = {}
    for number, obj in _json_objects(lines):
        try:
            doc = _corpus_doc(obj)
            if doc.doc_id in first_line:
                raise CorpusError(
                    f"doc_id {doc.doc_id!r} is repeated "
                    f"(first on line {first_line[doc.doc_id]})"
                )
        except CorpusError as exc:
            raise CorpusError(str(exc), line=number) from None
        first_line[doc.doc_id] = number
        docs.append(doc)
    return docs


def load_queries(lines: Iterable[str], corpus: Sequence[Doc]) -> list[LabelledQuery]:
    """The labelled queries of *lines*, checked against *corpus*.

    ``{"qid": "q1", "query": "…", "scope": {"task_id": "t-en",
    "before_step": 9, "exclude_ids": ["n7"]}, "relevant": ["d5", "n3"],
    "grades": {"d5": 2}, "source": "gold"}``. Required: ``qid`` (unique),
    ``query`` (not blank) and ``relevant`` (a non-empty list of ids).
    Optional: ``scope``, with only those three keys (``before_step`` an
    integer >= 1, and only with ``task_id``); ``grades`` (keys among
    ``relevant``, integer values >= 1); ``source`` (a string).

    In ``relevant``, ``grades`` and ``exclude_ids``, an id without ``:``
    is short for ``<scope.task_id>:<id>``, and an error without a
    ``task_id``. Every relevant id must be in *corpus* and a candidate of
    its query (see
    :meth:`~localmelo.melo.memory.retrieval.evaluate.LabelledQuery.candidates`),
    and none may be in ``exclude_ids``. A relevant id given twice counts
    once.
    """
    docs = {doc.doc_id: doc for doc in corpus}
    queries: list[LabelledQuery] = []
    first_line: dict[str, int] = {}
    for number, obj in _json_objects(lines):
        try:
            query = _labelled_query(obj, corpus, docs)
            if query.qid in first_line:
                raise CorpusError(
                    f"qid {query.qid!r} is repeated "
                    f"(first on line {first_line[query.qid]})"
                )
        except CorpusError as exc:
            raise CorpusError(str(exc), line=number) from None
        first_line[query.qid] = number
        queries.append(query)
    return queries


def load_raw_calls(lines: Iterable[str]) -> list[RawCall]:
    """The model calls of raw-call *lines*, in line order.

    ``{"task_id": "t-en", "seq": 3, "kind": "plan", "step": 2,
    "attempt": 1, "content": "", "thinking": "…", "tool_calls":
    [{"tool_name": "file_read", "arguments": {"path": "list_ops.py"},
    "id": "call_1"}], "tool_results": [{"tool_name": "file_read",
    "output": "…", "error": "", "duration_ms": 3.1}], "outcome": "tools",
    "timestamp": 1761200000.0}``.

    Required: ``task_id`` (a non-empty string without whitespace) and
    ``seq`` (an integer >= 0, unique per task). The rest defaults:
    ``kind`` ``"plan"`` (or ``reflect``, ``retrieve``, ``estimate``),
    ``step`` 0, ``attempt`` 1 (``null`` reads as the default for both, as
    in the history's nullable columns), ``content`` and ``thinking``
    ``""``, no calls or results, ``timestamp`` 0.0. ``outcome`` is derived
    when absent: on a plan call, ``tools`` with calls, else ``answer`` with
    non-blank content, else ``empty``; ``aux`` on the other kinds. A given
    outcome must be one of those four, and on a plan call one of the first
    three.

    Calls and results have the shapes of the history's
    ``steps.tool_call_json`` and ``steps.tool_result_json``. ``arguments``
    is an object, or a string holding one, as OpenAI bodies carry it; a
    string that does not decode to an object becomes
    ``{"_raw_arguments": <text>}``. A result's ``tool_name`` defaults to
    its call's. Results pair with calls by index and are never more
    numerous.
    """
    calls: list[RawCall] = []
    first_line: dict[tuple[str, int], int] = {}
    for number, obj in _json_objects(lines):
        try:
            call = _raw_call(obj)
            key = (call.task_id, call.seq)
            if key in first_line:
                raise CorpusError(
                    f"seq {call.seq} of task {call.task_id!r} is repeated "
                    f"(first on line {first_line[key]})"
                )
        except CorpusError as exc:
            raise CorpusError(str(exc), line=number) from None
        first_line[key] = number
        calls.append(call)
    return calls


def doc_to_dict(doc: Doc) -> dict[str, Any]:
    """*doc* as the object of a corpus line.

    Its keys are ``doc_id``, ``kind``, ``title``, ``text`` and ``meta``,
    in this order.
    """
    return {
        "doc_id": doc.doc_id,
        "kind": doc.kind,
        "title": doc.title,
        "text": doc.text,
        "meta": dict(doc.meta),
    }


# ── Documents from calls ──


def docs_from_calls(calls: Iterable[RawCall]) -> list[Doc]:
    """The documents of a run's model calls.

    Tasks go in the order of their first call in *calls*, and each task's
    calls by ``seq``. Per task, ``recent_path`` is the ``path`` of the
    latest ``file_read`` or ``file_write`` of an earlier call whose result
    has no error.

    - A ``plan`` call gives a note, ``<task_id>:n<seq>``: the
      :func:`~localmelo.melo.memory.notes.note_from_reply` of the call
      (its kind ``step``, ``answer`` or ``empty`` from the outcome),
      rendered with :func:`~localmelo.melo.memory.notes.render_note` and
      titled with :func:`~localmelo.melo.memory.notes.note_title`. It then
      gives a draft, ``<task_id>:d<seq>``, of the code blocks in the
      reasoning and in the content without its ``<note>`` blocks (target:
      a file named before them, else ``recent_path``), titled
      ``<target or "no target">, <lines> lines: <first code line>``.
    - A ``reflect`` call gives a reflection, ``<task_id>:r<seq>``, built
      as a note of kind ``reflection``, when its reply holds one (see
      :func:`_find_reflection`), and no draft.
    - Both then give the thought chunks of the reasoning,
      ``<task_id>:t<seq>.<i>`` with i from 1.
    - ``retrieve`` and ``estimate`` calls give nothing.

    A new draft makes the task's older pending drafts for the same target
    ``superseded``. After each call, a ``file_write`` without an error
    marks the pending drafts for its ``path`` ``written`` when its
    ``content`` holds :data:`~localmelo.melo.memory.notes.DRAFT_WRITTEN_SHARE`
    of a draft's lines (the same call's draft included). Targets and paths
    meet by :func:`~localmelo.melo.memory.notes.same_path`.

    Every text and title passes
    :func:`~localmelo.melo.checker.sanitize.sanitize_assistant_text`, and
    a document whose text is then blank is left out. ``meta`` holds
    ``task_id``, ``step``, ``attempt``, ``seq`` and ``created_at`` (the
    call's ``timestamp``); notes and reflections add ``source``,
    ``note_kind``, ``remember`` and ``remember_by``; drafts add
    ``target``, ``status`` (at the end of the task), ``lines``, ``blocks``
    and ``cut``; chunks add ``chunk``. Documents go by task, then call,
    then note (or reflection), draft and chunks.

    Raises :class:`CorpusError` for two calls of a task with one ``seq``.
    """
    tasks: dict[str, list[RawCall]] = {}
    for call in calls:
        tasks.setdefault(call.task_id, []).append(call)
    docs: list[Doc] = []
    for task_calls in tasks.values():
        docs.extend(_task_docs(sorted(task_calls, key=lambda call: call.seq)))
    return docs


def _task_docs(calls: Sequence[RawCall]) -> list[Doc]:
    """The documents of one task's calls, which are in ``seq`` order."""
    for earlier, later in zip(calls, calls[1:], strict=False):
        if earlier.seq == later.seq:
            raise CorpusError(
                f"task {later.task_id!r} has two calls with seq {later.seq}"
            )
    # A draft's doc waits for the end of the task, which settles its status.
    entries: list[Doc | tuple[str, Draft, dict[str, Any]]] = []
    drafts: list[Draft] = []
    recent_path = ""
    for call in calls:
        meta: dict[str, Any] = {
            "task_id": call.task_id,
            "step": call.step,
            "attempt": call.attempt,
            "seq": call.seq,
            "created_at": call.timestamp,
        }
        if call.kind in ("plan", "reflect"):
            reflection = call.kind == "reflect"
            found = _find_reflection(call.content) if reflection else None
            note, rest = note_from_reply(
                call.content,
                call.thinking,
                call.tool_calls,
                call.tool_results,
                kind=(
                    "reflection" if reflection else _NOTE_KINDS.get(_outcome(call), "")
                ),
            )
            if found is not None:
                # The reply is JSON: its own fields read better than the
                # prose heuristic run over its syntax.
                note = _reflection_note(found)
            note_doc = _doc(
                f"{call.task_id}:{'r' if reflection else 'n'}{call.seq}",
                "reflection" if reflection else "note",
                render_note(note),
                note_title(note),
                {
                    **meta,
                    "source": note.source,
                    "note_kind": note.kind,
                    "remember": note.remember,
                    "remember_by": note.remember_by,
                },
            )
            # The agent adds no reflection for a reflect reply without one,
            # so a doc for it would be a reflection the run never had. The
            # call's thought chunks stay.
            if note_doc is not None and (not reflection or found is not None):
                entries.append(note_doc)
            if not reflection:
                draft = extract_draft(
                    call.thinking + "\n\n" + rest, recent_path=recent_path
                )
                if draft is not None:
                    for older in drafts:
                        if older.status == "pending" and same_path(
                            older.target, draft.target
                        ):
                            older.status = "superseded"
                    drafts.append(draft)
                    entries.append((f"{call.task_id}:d{call.seq}", draft, meta))
            for index, chunk in enumerate(thought_chunks(call.thinking), start=1):
                chunk_doc = _doc(
                    f"{call.task_id}:t{call.seq}.{index}",
                    "thought",
                    chunk.text,
                    chunk.title,
                    {**meta, "chunk": index},
                )
                if chunk_doc is not None:
                    entries.append(chunk_doc)
        _mark_written(call, drafts)
        recent_path = _recent_path(call, recent_path)

    docs: list[Doc] = []
    for entry in entries:
        if isinstance(entry, Doc):
            docs.append(entry)
            continue
        doc_id, draft, meta = entry
        draft_doc = _doc(
            doc_id,
            "draft",
            draft.text,
            _draft_title(draft),
            {
                **meta,
                "target": draft.target,
                "status": draft.status,
                "lines": draft.lines,
                "blocks": draft.blocks,
                "cut": draft.cut,
            },
        )
        if draft_doc is not None:
            docs.append(draft_doc)
    return docs


def _doc(
    doc_id: str, kind: str, text: str, title: str, meta: dict[str, Any]
) -> Doc | None:
    """A sanitized document; ``None`` when its text is blank."""
    text = sanitize_assistant_text(text)
    if not text.strip():
        return None
    return Doc(
        doc_id=doc_id,
        kind=kind,
        text=text,
        title=sanitize_assistant_text(title),
        meta=meta,
    )


def _draft_title(draft: Draft) -> str:
    """``<target or "no target">, <lines> lines: <first code line>``, shortened."""
    first_line = next(
        (line.strip() for line in draft.text.splitlines() if line.strip()), ""
    )
    return _shorten(
        f"{draft.target or 'no target'}, {draft.lines} lines: {first_line}",
        TITLE_CHARS,
    )


def _executed(call: RawCall) -> Iterator[tuple[ToolCall, ToolResult]]:
    """The tool calls of *call* that ran without an error, with their results."""
    for tool_call, result in zip(call.tool_calls, call.tool_results, strict=False):
        if not result.error.strip():
            yield tool_call, result


def _mark_written(call: RawCall, drafts: Sequence[Draft]) -> None:
    """Mark the pending drafts that a ``file_write`` of *call* wrote."""
    for tool_call, _result in _executed(call):
        if tool_call.tool_name != "file_write":
            continue
        path = tool_call.arguments.get("path")
        content = tool_call.arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        for draft in drafts:
            if (
                draft.status == "pending"
                and same_path(draft.target, path)
                and draft_coverage(draft.text, content) >= DRAFT_WRITTEN_SHARE
            ):
                draft.status = "written"


def _recent_path(call: RawCall, recent_path: str) -> str:
    """The path of the latest file read or write of *call*, else *recent_path*."""
    for tool_call, _result in _executed(call):
        if tool_call.tool_name in ("file_read", "file_write"):
            path = tool_call.arguments.get("path")
            if isinstance(path, str) and path.strip():
                recent_path = path
    return recent_path


# ── Reflections ──
#
# The test of a reflect reply: copies of _json_values and _find_reflection in
# melo/agent/agent.py (lines 105-193 at c295ce2), with REFLECTION_ACTIONS
# above and _strip_code_fences from llm.py, itself a copy. melo/memory does
# not import melo.agent; a parity test holds the copies to the originals
# until R5 moves the test to one place.


def _json_values(text: str) -> Iterator[object]:
    """The JSON values of a reflect reply, in the order they are read.

    A Markdown code fence around the reply is stripped first. When the
    rest is JSON as a whole, that value is the only one. Otherwise each
    balanced ``{...}`` in it that parses as a JSON object is one. After a
    ``{`` that does not start one, the search goes on from where the
    decoder gave up, and after an object, from its end. Nesting deeper
    than the decoder's recursion limit ends the search.
    """
    body = _strip_code_fences(text)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        pass
    else:
        yield value
        return
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError as exc:
            start = body.find("{", max(exc.pos, start + 1))
            continue
        except RecursionError:
            return
        yield value
        start = body.find("{", end)


def _find_reflection(text: str) -> dict[str, Any] | None:
    """The reflection a reflect reply holds, or ``None``.

    That is the first JSON value of the reply (see :func:`_json_values`)
    that is an object whose ``recommended_action``, as a string, is one of
    :data:`REFLECTION_ACTIONS`. An object without one is passed over for
    the objects after it, but an object nested inside it is not looked at.
    JSON that is not an object holds no reflection.
    """
    try:
        for value in _json_values(text):
            if not isinstance(value, dict):
                continue
            action = value.get("recommended_action")
            if ("" if action is None else str(action)) in REFLECTION_ACTIONS:
                return value
    except ValueError:  # a number the decoder cannot convert ends the search
        pass
    return None


def _reflection_note(reflection: Mapping[str, Any]) -> ReasoningNote:
    """The note of a reflect reply that holds *reflection*, from its fields.

    ``plan`` is ``<recommended_action>: <summary>``; ``found`` takes
    ``useful_evidence`` (at most 4 items); ``decided`` takes
    ``next_promising_directions`` as ``next: …`` (at most 3);
    ``hypotheses`` takes ``failed_hypotheses`` as ``refuted: …``, then
    ``recommended_avoids`` as ``avoid: …`` (at most 3 together). Items are
    whitespace-collapsed and shortened like a note's; a field that is not
    a list, or an item that is not a string, is left out.
    """

    def items(key: str, prefix: str = "") -> list[str]:
        value = reflection.get(key)
        if not isinstance(value, list):
            return []
        texts = (_collapse(item) for item in value if isinstance(item, str))
        return [_shorten(prefix + text, NOTE_ITEM_CHARS) for text in texts if text]

    # _find_reflection admits only a recommended_action in REFLECTION_ACTIONS.
    plan = str(reflection.get("recommended_action"))
    summary = reflection.get("summary")
    if isinstance(summary, str) and _collapse(summary):
        plan = f"{plan}: {_collapse(summary)}"
    return ReasoningNote(
        kind="reflection",
        source="model",
        plan=_shorten(plan, NOTE_PLAN_CHARS),
        found=items("useful_evidence")[:4],
        decided=items("next_promising_directions", "next: ")[:3],
        hypotheses=[
            *items("failed_hypotheses", "refuted: "),
            *items("recommended_avoids", "avoid: "),
        ][:3],
    )


def _outcome(call: RawCall) -> str:
    """The call's outcome, derived when the call has none."""
    return call.outcome or _derived_outcome(call.kind, call.content, call.tool_calls)


def _derived_outcome(kind: str, content: str, tool_calls: Sequence[ToolCall]) -> str:
    """``tools``, ``answer`` or ``empty`` for a plan call; ``aux`` otherwise."""
    if kind != "plan":
        return "aux"
    if tool_calls:
        return "tools"
    return "answer" if content.strip() else "empty"


# ── A run's history.db ──


def load_history_calls(path: str | os.PathLike[str]) -> tuple[list[RawCall], int]:
    """The model calls recorded in the ``history.db`` at *path*.

    Returns the calls, tasks in the order of their first ``timestamp`` and
    calls by ``seq``, and the number of step ids that gave no result: those
    of ``step_ids_json`` with no ``steps`` row, or whose row matches none
    of the call's tool calls.

    The database is opened read-only through a URI, with ``immutable=1``
    when ``<path>-wal`` does not exist: every committed row is then in the
    main file, and reading it creates no ``-wal`` or ``-shm`` file, even
    in a directory the user cannot write. With a ``-wal`` file (a run
    still writing, or one killed before its checkpoint), plain
    ``mode=ro`` reads a consistent snapshot through it and may create a
    missing ``-shm``. The database is never migrated, and nothing is
    written.

    Each ``llm_calls`` row becomes a :class:`RawCall` under the raw-call
    rules (see :func:`load_raw_calls`); a ``NULL`` ``step`` or ``attempt``
    reads as 0 and 1, and a ``content`` that starts with the ``<think>``
    copy of ``thinking`` that Ollama's provider adds is read without it.
    The calls come from ``tool_calls_json`` (``NULL``: none), and the
    results from the ``steps`` rows that ``step_ids_json`` names: each
    call takes the first of them, not yet taken, whose ``tool_call_json``
    has the call's ``id``. The agent writes those rows in call order, so
    the results run in call order up to the first call without one, as the
    calls' pairing by index needs.

    Raises :class:`NoCallsTableError` without an ``llm_calls`` table, and
    :class:`CorpusError` for a missing column, a row that breaks the
    rules (``llm_calls row <task_id>/<seq>: <problem>``) or a database
    that cannot be read.
    """
    database = Path(path).resolve()
    uri = database.as_uri() + "?mode=ro"
    if not Path(f"{database}-wal").exists():
        uri += "&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise CorpusError(f"cannot open the database: {exc}") from None
    try:
        return _history_calls(connection)
    except sqlite3.Error as exc:
        raise CorpusError(f"cannot read the database: {exc}") from None
    finally:
        connection.close()


def _history_calls(connection: sqlite3.Connection) -> tuple[list[RawCall], int]:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "llm_calls" not in tables:
        raise NoCallsTableError("the database has no llm_calls table")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(llm_calls)")}
    missing = [name for name in HISTORY_CALL_COLUMNS if name not in columns]
    if missing:
        raise CorpusError(f"llm_calls has no column {', '.join(missing)}")

    rows = connection.execute(
        f"SELECT {', '.join(HISTORY_CALL_COLUMNS)} FROM llm_calls "
        "ORDER BY task_id, seq"
    ).fetchall()
    calls: list[RawCall] = []
    seen: set[tuple[str, int]] = set()
    skipped = 0
    for row in rows:
        values = dict(zip(HISTORY_CALL_COLUMNS, row, strict=True))
        try:
            call, row_skipped = _history_call(connection, values)
            if (call.task_id, call.seq) in seen:
                raise CorpusError(f"seq {call.seq} of the task is repeated")
        except CorpusError as exc:
            raise CorpusError(
                f"llm_calls row {values['task_id']}/{values['seq']}: {exc}"
            ) from None
        seen.add((call.task_id, call.seq))
        calls.append(call)
        skipped += row_skipped

    first: dict[str, float] = {}
    for call in calls:
        first[call.task_id] = min(
            first.get(call.task_id, call.timestamp), call.timestamp
        )
    calls.sort(key=lambda call: (first[call.task_id], call.task_id, call.seq))
    return calls, skipped


def _history_call(
    connection: sqlite3.Connection, values: Mapping[str, Any]
) -> tuple[RawCall, int]:
    """The raw call of one ``llm_calls`` row, and its step ids that gave nothing."""
    tool_calls = _json_column(values["tool_calls_json"], "tool_calls_json", [])
    step_ids = _json_column(values["step_ids_json"], "step_ids_json", [])
    if not isinstance(step_ids, list) or not all(
        isinstance(step_id, str) for step_id in step_ids
    ):
        raise CorpusError("step_ids_json must be a JSON list of step ids")
    results, skipped = _linked_results(connection, tool_calls, step_ids)
    call = _raw_call(
        {
            "task_id": values["task_id"],
            "seq": values["seq"],
            "kind": values["kind"],
            "step": values["step"],
            "attempt": values["attempt"],
            "content": _without_ollama_copy(values["content"], values["thinking"]),
            "thinking": values["thinking"],
            "tool_calls": tool_calls,
            "tool_results": results,
            "outcome": values["outcome"],
            "timestamp": values["timestamp"],
        }
    )
    return call, skipped


def _without_ollama_copy(content: object, thinking: object) -> object:
    """*content* without the copy of *thinking* that Ollama's provider adds.

    When *thinking* is not empty and *content* starts with ``<think>``, a
    newline, *thinking*, a newline, ``</think>`` and a blank line, the
    content is the text after that prefix, as the provider reads it when it
    sends the turn back. Anything else, a value that is not text included,
    is returned as it is.
    """
    if isinstance(content, str) and isinstance(thinking, str) and thinking:
        prefix = f"<think>\n{thinking}\n</think>\n\n"
        if content.startswith(prefix):
            return content[len(prefix) :]
    return content


def _linked_results(
    connection: sqlite3.Connection, tool_calls: object, step_ids: Sequence[str]
) -> tuple[list[Any], int]:
    """The results of *tool_calls*, from the ``steps`` rows of *step_ids*.

    Returns them in call order, up to the first call that takes no row,
    and the number of step ids that no call took.
    """
    # One entry per listed step id: the id of its row's tool call and the
    # row's result, or None for a row that cannot give one.
    linked: list[tuple[str, Any] | None] = []
    for step_id in step_ids:
        row = connection.execute(
            "SELECT tool_call_json, tool_result_json FROM steps WHERE step_id = ?",
            (step_id,),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            linked.append(None)
            continue
        recorded = _json_column(row[0], f"steps row {step_id}: tool_call_json")
        result = _json_column(row[1], f"steps row {step_id}: tool_result_json")
        call_id = recorded.get("id") if isinstance(recorded, dict) else None
        linked.append((call_id, result) if isinstance(call_id, str) else None)

    taken = [False] * len(linked)
    results: list[Any] = []
    complete = True
    for item in tool_calls if isinstance(tool_calls, list) else []:
        call_id = item.get("id") if isinstance(item, dict) else None
        position = next(
            (
                i
                for i, entry in enumerate(linked)
                if entry is not None and not taken[i] and entry[0] == call_id
            ),
            None,
        )
        if position is None:
            complete = False
            continue
        taken[position] = True
        entry = linked[position]
        if complete and entry is not None:
            results.append(entry[1])
    return results, taken.count(False)


def _json_column(text: object, name: str, default: Any = None) -> Any:
    """The JSON value of a text column; *default* for ``NULL``."""
    if text is None:
        return default
    if not isinstance(text, str):
        raise CorpusError(f"{name} is not JSON text")
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        raise CorpusError(f"{name} is not JSON") from None
    surrogate = _unpaired_surrogate(value)
    if surrogate:
        raise CorpusError(f"{name}: {surrogate}")
    return value


# ── Line rules ──


def _json_objects(lines: Iterable[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """The 1-based number and JSON object of each non-blank line."""
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(
                f"not JSON: {exc.msg} at column {exc.colno}", line=number
            ) from None
        except (ValueError, RecursionError):
            raise CorpusError("not JSON", line=number) from None
        if not isinstance(value, dict):
            raise CorpusError("not a JSON object", line=number)
        surrogate = _unpaired_surrogate(value)
        if surrogate:
            raise CorpusError(surrogate, line=number)
        yield number, value


# One half of a UTF-16 surrogate pair. JSON's \uXXXX escapes can name one
# alone (a writer that cut a string between the two halves), and json.loads
# then gives a string that UTF-8 cannot encode, nor melo-retrieve print.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _unpaired_surrogate(value: object) -> str:
    """The problem of a string in the JSON *value* that UTF-8 cannot encode.

    ``""`` when every string of *value*, keys included, can be encoded.
    """
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            match = _SURROGATE_RE.search(item)
            if match is not None:
                return (
                    f"a string holds the unpaired surrogate \\u{ord(match.group()):04x}, "
                    "which UTF-8 cannot encode"
                )
        elif isinstance(item, dict):
            stack.extend(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return ""


def _corpus_doc(obj: Mapping[str, Any]) -> Doc:
    doc_id = _required(obj, "doc_id")
    if not _is_id(doc_id):
        raise CorpusError("doc_id must be a non-empty string without whitespace")
    kind = _required(obj, "kind")
    if kind not in DOC_KINDS:
        raise CorpusError(f"kind must be one of {', '.join(DOC_KINDS)}, got {kind!r}")
    text = _required(obj, "text")
    if not isinstance(text, str) or not text:
        raise CorpusError("text must be a non-empty string")
    title = obj.get("title", "")
    if not isinstance(title, str):
        raise CorpusError("title must be a string")
    meta = obj.get("meta", {})
    if not isinstance(meta, dict):
        raise CorpusError("meta must be an object")
    if "task_id" in meta and not isinstance(meta["task_id"], str):
        raise CorpusError("meta.task_id must be a string")
    if "step" in meta and not (_is_int(meta["step"]) and meta["step"] >= 0):
        raise CorpusError("meta.step must be an integer >= 0")
    return Doc(doc_id=doc_id, kind=kind, text=text, title=title, meta=meta)


def _labelled_query(
    obj: Mapping[str, Any], corpus: Sequence[Doc], docs: Mapping[str, Doc]
) -> LabelledQuery:
    qid = _required(obj, "qid")
    if not isinstance(qid, str) or not qid:
        raise CorpusError("qid must be a non-empty string")
    text = _required(obj, "query")
    if not isinstance(text, str) or not text.strip():
        raise CorpusError("query must be a non-blank string")
    relevant_ids = _required(obj, "relevant")
    if not _is_id_list(relevant_ids) or not relevant_ids:
        raise CorpusError("relevant must be a non-empty list of ids")

    scope = obj.get("scope", {})
    if not isinstance(scope, dict):
        raise CorpusError("scope must be an object")
    unknown = [key for key in scope if key not in _SCOPE_KEYS]
    if unknown:
        raise CorpusError(
            f"scope has an unknown key {unknown[0]!r}; it takes "
            "task_id, before_step and exclude_ids"
        )
    task_id = scope.get("task_id", "")
    if not isinstance(task_id, str) or ("task_id" in scope and not task_id):
        raise CorpusError("scope.task_id must be a non-empty string")
    before_step = scope.get("before_step")
    if "before_step" in scope:
        if not (_is_int(before_step) and before_step >= 1):
            raise CorpusError("scope.before_step must be an integer >= 1")
        if not task_id:
            raise CorpusError("scope.before_step needs scope.task_id")
    exclude_ids = scope.get("exclude_ids", [])
    if not _is_id_list(exclude_ids):
        raise CorpusError("scope.exclude_ids must be a list of ids")
    grades = obj.get("grades", {})
    if not isinstance(grades, dict):
        raise CorpusError("grades must be an object")
    source = obj.get("source", "")
    if not isinstance(source, str):
        raise CorpusError("source must be a string")

    def full(doc_id: str, field: str) -> str:
        """*doc_id*, with the scope's task in front of a short id."""
        if ":" in doc_id:
            return doc_id
        if not task_id:
            raise CorpusError(
                f"{field} id {doc_id!r} has no task prefix, and the query "
                "has no scope.task_id"
            )
        return f"{task_id}:{doc_id}"

    relevant = list(dict.fromkeys(full(doc_id, "relevant") for doc_id in relevant_ids))
    excluded = frozenset(full(doc_id, "exclude_ids") for doc_id in exclude_ids)
    # An excluded doc can never be returned, so it cannot also be a target.
    both = [doc_id for doc_id in relevant if doc_id in excluded]
    if both:
        raise CorpusError(f"relevant id {both[0]!r} is also in scope.exclude_ids")
    graded: dict[str, int] = {}
    for key, value in grades.items():
        doc_id = full(key, "grades")
        if doc_id not in relevant:
            raise CorpusError(f"grades names {doc_id!r}, which is not a relevant id")
        if not (_is_int(value) and value >= 1):
            raise CorpusError(f"grades[{key!r}] must be an integer >= 1")
        graded[doc_id] = value

    query = LabelledQuery(
        qid=qid,
        query=text,
        relevant=tuple(relevant),
        grades=graded,
        task_id=task_id,
        before_step=before_step,
        exclude_ids=excluded,
        source=source,
    )
    candidates = {doc.doc_id for doc in query.candidates(corpus)}
    for doc_id in relevant:
        doc = docs.get(doc_id)
        if doc is None:
            raise CorpusError(f"relevant id {doc_id!r} is not in the corpus")
        if doc_id not in candidates:
            raise CorpusError(
                f"relevant id {doc_id!r} is not a candidate of the query: "
                + _outside_scope(doc, query)
            )
    return query


def _outside_scope(doc: Doc, query: LabelledQuery) -> str:
    """Why *doc* is not a candidate of *query*."""
    task_id = doc.meta.get("task_id")
    if task_id != query.task_id:
        return f"its task is {task_id!r}, not {query.task_id!r}"
    return f"its step {int_step(doc.meta) or 0} is not before {query.before_step}"


def _raw_call(obj: Mapping[str, Any]) -> RawCall:
    task_id = _required(obj, "task_id")
    if not _is_id(task_id):
        raise CorpusError("task_id must be a non-empty string without whitespace")
    seq = _required(obj, "seq")
    if not (_is_int(seq) and seq >= 0):
        raise CorpusError("seq must be an integer >= 0")
    kind = obj.get("kind", "plan")
    if kind not in CALL_KINDS:
        raise CorpusError(f"kind must be one of {', '.join(CALL_KINDS)}, got {kind!r}")
    step = _nullable_int(obj, "step", 0)
    attempt = _nullable_int(obj, "attempt", 1)
    content = obj.get("content", "")
    if not isinstance(content, str):
        raise CorpusError("content must be a string")
    thinking = obj.get("thinking", "")
    if not isinstance(thinking, str):
        raise CorpusError("thinking must be a string")

    raw_tool_calls = obj.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        raise CorpusError("tool_calls must be a list")
    tool_calls = tuple(
        _tool_call(item, index) for index, item in enumerate(raw_tool_calls)
    )
    raw_results = obj.get("tool_results", [])
    if not isinstance(raw_results, list):
        raise CorpusError("tool_results must be a list")
    if len(raw_results) > len(tool_calls):
        raise CorpusError(
            f"tool_results has {len(raw_results)} items, more than the "
            f"{len(tool_calls)} tool_calls"
        )
    tool_results = tuple(
        _tool_result(item, index, tool_calls[index])
        for index, item in enumerate(raw_results)
    )

    if "outcome" in obj:
        outcome = obj["outcome"]
        if outcome not in CALL_OUTCOMES:
            raise CorpusError(
                f"outcome must be one of {', '.join(CALL_OUTCOMES)}, got {outcome!r}"
            )
        if kind == "plan" and outcome == "aux":
            raise CorpusError(
                "a plan call's outcome is tools, answer or empty, not aux"
            )
    else:
        outcome = _derived_outcome(kind, content, tool_calls)
    timestamp = obj.get("timestamp", 0.0)
    if not _is_finite(timestamp):
        raise CorpusError("timestamp must be a finite number")
    return RawCall(
        task_id=task_id,
        seq=seq,
        kind=kind,
        step=step,
        attempt=attempt,
        content=content,
        thinking=thinking,
        tool_calls=tool_calls,
        tool_results=tool_results,
        outcome=outcome,
        timestamp=float(timestamp),
    )


def _tool_call(item: object, index: int) -> ToolCall:
    where = f"tool_calls[{index}]"
    if not isinstance(item, dict):
        raise CorpusError(f"{where} must be an object")
    name = _required(item, "tool_name", where)
    if not isinstance(name, str):
        raise CorpusError(f"{where}.tool_name must be a string")
    call_id = item.get("id", "")
    if not isinstance(call_id, str):
        raise CorpusError(f"{where}.id must be a string")
    return ToolCall(
        tool_name=name,
        arguments=_arguments(item.get("arguments", {}), where),
        id=call_id,
    )


def _arguments(raw: object, where: str) -> dict[str, Any]:
    """A call's arguments: an object, or a string that holds one.

    A string that does not decode to an object is kept as
    ``{RAW_ARGUMENTS_KEY: <text>}``, as ``parse_tool_arguments`` in
    ``support/providers/llm/_protocol.py`` keeps it (``melo/`` cannot
    import ``support/``).
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise CorpusError(f"{where}.arguments must be an object or a string")
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        return {RAW_ARGUMENTS_KEY: raw}
    if not isinstance(value, dict):
        return {RAW_ARGUMENTS_KEY: raw}
    # The outer line's check sees only the escape, not the decoded string.
    problem = _unpaired_surrogate(value)
    if problem:
        raise CorpusError(f"{where}.arguments: {problem}")
    return value


def _tool_result(item: object, index: int, call: ToolCall) -> ToolResult:
    where = f"tool_results[{index}]"
    if not isinstance(item, dict):
        raise CorpusError(f"{where} must be an object")
    texts: dict[str, str] = {}
    for key, default in (("tool_name", call.tool_name), ("output", ""), ("error", "")):
        value = item.get(key, default)
        if not isinstance(value, str):
            raise CorpusError(f"{where}.{key} must be a string")
        texts[key] = value
    duration_ms = item.get("duration_ms", 0.0)
    if not _is_finite(duration_ms):
        raise CorpusError(f"{where}.duration_ms must be a finite number")
    return ToolResult(
        tool_name=texts["tool_name"],
        output=texts["output"],
        error=texts["error"],
        duration_ms=float(duration_ms),
    )


def _required(obj: Mapping[str, Any], key: str, where: str = "") -> Any:
    if key not in obj:
        raise CorpusError(f"{where}.{key} is missing" if where else f"{key} is missing")
    return obj[key]


def _nullable_int(obj: Mapping[str, Any], key: str, default: int) -> int:
    """``obj[key]``, an integer >= 0; *default* when absent or ``null``."""
    value = obj.get(key)
    if value is None:
        return default
    if not (_is_int(value) and value >= 0):
        raise CorpusError(f"{key} must be an integer >= 0")
    return int(value)


def _is_int(value: object) -> TypeGuard[int]:
    """Whether *value* is an int (a bool is not one)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    """Whether *value* is an int or a float that is finite (a bool is neither)."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _is_id(value: object) -> bool:
    """Whether *value* is a non-empty string without whitespace."""
    return (
        isinstance(value, str)
        and bool(value)
        and not any(char.isspace() for char in value)
    )


def _is_id_list(value: object) -> bool:
    """Whether *value* is a list of ids (see :func:`_is_id`)."""
    return isinstance(value, list) and all(_is_id(item) for item in value)
