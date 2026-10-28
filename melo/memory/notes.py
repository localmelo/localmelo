"""Reasoning notes, code drafts and thought chunks of one model reply.

A reply's reasoning is far longer than its visible text, and it is dropped
from the next request. This module keeps what matters of it in three
forms, all built by pure, deterministic functions:

- a **note** (:class:`ReasoningNote`): what the reply planned, found,
  decided, suspected and left open, in short fields, plus what its tool
  calls returned. The model may write the note itself in a ``<note>``
  block (:func:`split_note`, :func:`parse_note_block`); otherwise a
  heuristic writes it from marker words (:func:`note_from_reply`).
- a **draft** (:class:`Draft`): the code blocks of the reply, with the file
  they are meant for (:func:`extract_draft`).
- **thought chunks** (:class:`ThoughtChunk`): the reasoning without code,
  in paragraphs of bounded size (:func:`thought_chunks`).

Nothing here calls a model or touches the agent loop.
"""

from __future__ import annotations

import ast
import calendar
import hashlib
import json
import os
import posixpath
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from localmelo.melo.schema import (
    CandidateItem,
    ReflectionEntry,
    ToolCall,
    ToolResult,
)

# ── Caps ──
#
# Prompt caps are tokens of the context budget. Until it counts tokens,
# they are characters at about 3.45 characters per token; each character
# cap names its token equivalent.

NOTE_CHARS = 1_200  # ≈ 348 tokens: a rendered note
NOTE_PLAN_CHARS = 300  # ≈ 87 tokens
NOTE_ITEM_CHARS = 200  # ≈ 58 tokens: one found, decided, hypothesis or open item
NOTE_OBSERVED_CHARS = 160  # ≈ 46 tokens: one observed line, and the outcome
NOTE_MAX_FOUND = 4  # fallback or finding-marker items, not the factual-line cap
NOTE_MAX_FACTS = 12  # factual lines read from a reply with no note block
NOTE_MAX_DECIDED = 3
NOTE_MAX_HYPOTHESES = 3
NOTE_MAX_OPEN = 3
NOTE_MAX_OBSERVED = 5  # observed lines: one per call, the first five calls
FOUND_WINDOW_CHARS = 4_000  # ≈ 1,159 tokens: found sentences start in the last 4,000
DECIDED_TAIL_SHARE = 0.4  # decided sentences start in the last 40% of the text
TITLE_CHARS = 120  # ≈ 35 tokens
DRAFT_CHARS = 8_000  # ≈ 2,319 tokens: a stored draft
DRAFT_SHOWN_CHARS = 6_000  # ≈ 1,739 tokens: a pinned draft (unused here; R4 pins)
DRAFT_WRITTEN_SHARE = 0.6  # a write holding this share of a draft's lines writes it
THOUGHT_CHUNK_CHARS = 1_500  # ≈ 435 tokens
DIGEST_CHARS = 1_500  # ≈ 435 tokens: the attempt digest
DIGEST_TASK_CHARS = 300  # the query line of a digest
DIGEST_FILES = 6  # paths named in a digest before "+N"
DIGEST_PATH_CHARS = 120  # one path in a digest
DIGEST_COMMAND_CHARS = 120  # the check command in a digest
DIGEST_RESULT_CHARS = 80  # the check's result in a digest
NOTE_SHOWN_CHARS = 400  # a note's text in the reflection's notes block

# Caps inside the short form of a call and the summary of an output.
_SHORT_CALL_CHARS = 100  # ≈ 29 tokens
_SHORT_COMMAND_CHARS = 60  # ≈ 17 tokens: a command or a line of code
_SHORT_ARGUMENT_CHARS = 40  # ≈ 12 tokens: another tool's first argument
_SUMMARY_CHARS = 80  # ≈ 23 tokens: a command's first informative line

# Extensions that make a dotted name a file name. Left out on purpose, as
# common attribute names: cfg, conf, env, lock, log, patch.
FILE_EXTENSIONS = frozenset(
    {
        "bash",
        "bat",
        "c",
        "cc",
        "cjs",
        "cpp",
        "cs",
        "css",
        "csv",
        "cxx",
        "go",
        "gradle",
        "h",
        "hpp",
        "htm",
        "html",
        "ini",
        "ipynb",
        "java",
        "js",
        "json",
        "jsonl",
        "jsx",
        "kt",
        "lua",
        "md",
        "mjs",
        "php",
        "pl",
        "ps1",
        "py",
        "pyi",
        "rb",
        "rs",
        "rst",
        "scss",
        "sh",
        "sql",
        "svelte",
        "swift",
        "toml",
        "ts",
        "tsv",
        "tsx",
        "txt",
        "vue",
        "xml",
        "yaml",
        "yml",
        "zsh",
    }
)

# ── Types ──


@dataclass
class ReasoningNote:
    """One model reply, compressed.

    Built by :func:`note_from_reply`, which fills the content fields and
    leaves the identity fields at their defaults: the caller sets them,
    for example with :func:`dataclasses.replace`.
    """

    # The caller sets it; prompts use short aliases such as n4, d1, t7.
    note_id: str = ""
    task_id: str = ""
    step: int = 0  # run.steps of the planning call; 0 for task-level notes
    attempt: int = 0  # 0: not set (a run counts its attempts from 1)
    call_seq: int = 0  # llm_calls.seq of the reply it compresses
    # step | answer | empty | reflection | digest | user (a user-stated fact,
    # section 5)
    kind: str = "step"
    source: str = "heuristic"  # model | heuristic | sleep | user
    plan: str = ""  # <= 300: the next action and why
    # <= 12 x 200: what the model says it established
    found: list[str] = field(default_factory=list)
    decided: list[str] = field(default_factory=list)  # <= 3 x 200
    # <= 3 x 200: "open: ", "confirmed: ", "refuted: "
    hypotheses: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)  # <= 3 x 200
    # <= 5 x 160, from the executor only, one per call
    observed: list[str] = field(default_factory=list)
    # A digest only: the paths its attempt wrote, and the last check command
    # with its result (see digest_note).
    files: list[str] = field(default_factory=list)
    verified: str = ""
    outcome: str = ""  # <= 160: "ok" | "exit 1: 24 failed" | "Error: <first line>"
    drafts: list[str] = field(default_factory=list)  # draft ids
    reasoning_chars: int = 0
    # record ids from this turn's JSON tool results (section 5)
    source_ids: list[str] = field(default_factory=list)
    # (record id, up to 500 chars of its text) for an item's source ids: this
    # turn's results first, then the earlier observations that match an item
    source_records: list[tuple[str, str]] = field(default_factory=list)
    remember: bool = False  # the agent-side bool (section 5): this note is a candidate
    remember_by: str = ""  # model | heuristic | rule: who set it
    # fresh | rehearsed | superseded; the one stage field Hippo reads
    stage: str = "fresh"
    entry_id: str = ""  # the long-store entry once Hippo wrote it (set by admit())
    uses: int = 0
    last_used_at: float = 0.0  # for LRU-style eviction within a stage
    created_at: float = 0.0


@dataclass
class Draft:
    """The code blocks of one reply, and the file they are meant for."""

    draft_id: str = ""
    task_id: str = ""
    step: int = 0
    call_seq: int = 0
    text: str = ""
    target: str = ""  # a file name or path; "" when none was found
    status: str = "pending"  # pending | written | superseded
    lines: int = 0  # non-blank lines of text
    blocks: int = 0  # fenced blocks kept in text
    cut: bool = False  # True when text lost blocks or lines to DRAFT_CHARS
    created_at: float = 0.0


@dataclass(frozen=True)
class ThoughtChunk:
    """A run of reasoning paragraphs without code, titled by its first sentence."""

    title: str
    text: str


# ── Text helpers ──


def _shorten(text: str, width: int) -> str:
    """*text* in at most *width* characters, cut at a word boundary, ending in "…".

    Text that fits is returned as it is. When the cut falls inside the
    first word, there is no boundary to go back to, and the text is cut
    where *width* falls.

    A copy of ``_shorten`` in ``melo/agent/agent.py``, which is private to
    the agent loop.
    """
    if len(text) <= width:
        return text
    if width < 1:
        return ""
    head = text[: width - 1]
    if not text[len(head)].isspace() and not head[-1:].isspace():
        # The cut falls inside a word: leave that word out.
        words = head.rsplit(None, 1)
        if len(words) == 2:
            head = words[0]
    return head.rstrip() + "…"


def _collapse(text: str) -> str:
    """*text* with each run of whitespace made one space, and stripped."""
    return " ".join(text.split())


def _first_nonblank_line(text: str) -> str:
    """The first line of *text* that is not blank, stripped; "" if none."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


# ── Note blocks ──

_NOTE_BLOCK_RE = re.compile(r"<note>(.*?)(?:</note>|\Z)", re.IGNORECASE | re.DOTALL)
_NOTE_TAG_RE = re.compile(r"</?note>", re.IGNORECASE)

# A line that starts a field of the model's note: "found: ...".
_NOTE_KEY_RE = re.compile(
    r"\s*(?:[-*•]\s*)?"
    r"(plan|found|decided|hypotheses|hypothesis|open\s+questions|open|remember)"
    r"\s*[:：]\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
_NOTE_LIST_FIELDS = {
    "found": NOTE_MAX_FACTS,
    "decided": NOTE_MAX_DECIDED,
    "hypotheses": NOTE_MAX_HYPOTHESES,
    "open": NOTE_MAX_OPEN,
}
_ITEM_SEPARATOR_RE = re.compile(r"[;；]")
_REMEMBER_STRIP = " \t\r\n\u3000.。!！"
_REMEMBER_VALUES = {
    "yes": True,
    "true": True,
    "1": True,
    "是": True,
    "no": False,
    "false": False,
    "0": False,
    "否": False,
}


def split_note(content: str) -> tuple[str | None, str]:
    """Split the model's ``<note>`` blocks off *content*.

    Every ``<note>…</note>`` block is found, in any case; a block without
    its closing tag runs to the end of the text. Returns the inner text of
    the last block, stripped, and the rest: *content* with every block
    removed, stripped. Without a block, returns ``(None, content)`` with
    *content* unchanged. A ``</note>`` without an opening tag stays in the
    rest.
    """
    blocks = list(_NOTE_BLOCK_RE.finditer(content))
    if not blocks:
        return None, content
    return blocks[-1].group(1).strip(), _NOTE_BLOCK_RE.sub("", content).strip()


def parse_note_block(block: str) -> ReasoningNote:
    """Read the fields of a note the model wrote (the inside of ``<note>``).

    A line ``[-*•] <key>: <value>`` (``:`` or ``：``) starts a field. The
    keys, in any case, are ``plan``, ``found``, ``decided``,
    ``hypotheses`` (or ``hypothesis``), ``open`` (or ``open questions``)
    and ``remember``. A line without a key continues the field before it,
    joined with a space; lines before the first key go to ``plan``. A key
    given twice adds to its field. ``found``, ``decided``, ``hypotheses``
    and ``open`` are split into items at ``;`` and ``；``, empty items
    dropped, and keep their first 12, 3, 3 and 3 items, each shortened to
    :data:`NOTE_ITEM_CHARS`; ``plan`` is shortened to
    :data:`NOTE_PLAN_CHARS`.

    ``remember`` is read lowercased, with spaces, ``.``, ``。``, ``!`` and
    ``！`` stripped: ``yes``, ``true``, ``1`` or ``是`` set it, ``no``,
    ``false``, ``0`` or ``否`` clear it, and either sets ``remember_by``
    to ``"model"``. A missing or unreadable value leaves ``remember_by``
    empty (the last ``remember`` line counts).
    """
    # One entry per key line: the field, and the text of the line and of
    # the lines that continue it.
    segments: list[tuple[str, list[str]]] = [("plan", [])]
    for line in block.splitlines():
        match = _NOTE_KEY_RE.fullmatch(line)
        if match is None:
            segments[-1][1].append(line.strip())
            continue
        key = " ".join(match.group(1).lower().split())
        if key == "hypothesis":
            key = "hypotheses"
        elif key == "open questions":
            key = "open"
        segments.append((key, [match.group(2).strip()]))

    texts: dict[str, list[str]] = {}
    for key, parts in segments:
        text = " ".join(part for part in parts if part)
        texts.setdefault(key, []).append(text)

    note = ReasoningNote(source="model")
    note.plan = _shorten(
        " ".join(t for t in texts.get("plan", []) if t), NOTE_PLAN_CHARS
    )
    items: dict[str, list[str]] = {}
    for key, limit in _NOTE_LIST_FIELDS.items():
        found = [
            item.strip()
            for text in texts.get(key, [])
            for item in _ITEM_SEPARATOR_RE.split(text)
            if item.strip()
        ]
        items[key] = [_shorten(item, NOTE_ITEM_CHARS) for item in found[:limit]]
    note.found = items["found"]
    note.decided = items["decided"]
    note.hypotheses = items["hypotheses"]
    note.open_questions = items["open"]

    remember = texts.get("remember")
    if remember:
        value = _REMEMBER_VALUES.get(remember[-1].lower().strip(_REMEMBER_STRIP))
        if value is not None:
            note.remember = value
            note.remember_by = "model"
    return note


# ── Fenced code blocks ──

_FENCE_OPEN_RE = re.compile(r" {0,3}```[^`\n]*")
_FENCE_CLOSE_RE = re.compile(r" {0,3}```[^\S\n]*")


@dataclass(frozen=True)
class _Fence:
    start: int  # the opening line's first character
    end: int  # the end of the closing line, before its line break; or len(text)
    body: str  # the lines between the fences, without the last line break


def _fences(text: str) -> list[_Fence]:
    """The fenced code blocks of *text*, in order.

    A block opens with a line of up to 3 spaces, then three backticks and
    an optional language tag, and closes with a line of up to 3 spaces and
    three backticks. A block that does not close runs to the end of the
    text.
    """
    fences: list[_Fence] = []
    open_start = -1
    body_start = 0
    pos = 0
    while True:
        newline = text.find("\n", pos)
        line_end = len(text) if newline == -1 else newline
        line = text[pos:line_end]
        if open_start < 0:
            if _FENCE_OPEN_RE.fullmatch(line):
                open_start = pos
                body_start = min(line_end + 1, len(text))
        elif _FENCE_CLOSE_RE.fullmatch(line):
            body = text[body_start:pos]
            fences.append(_Fence(open_start, line_end, body.removesuffix("\n")))
            open_start = -1
        if newline == -1:
            break
        pos = newline + 1
    if open_start >= 0:
        fences.append(_Fence(open_start, len(text), text[body_start:]))
    return fences


def _without_fences(text: str) -> str:
    """*text* with every fenced code block replaced by a line break."""
    fences = _fences(text)
    if not fences:
        return text
    parts: list[str] = []
    pos = 0
    for fence in fences:
        parts.append(text[pos : fence.start])
        parts.append("\n")
        pos = fence.end
    parts.append(text[pos:])
    return "".join(parts)


# ── Sentences and paths ──

# Ends a sentence: CJK end punctuation, or "!", "?" or "." before
# whitespace (any, U+3000 and U+00A0 included), but not the "." that ends
# "e.g." or "i.e.". Their word boundary is an ASCII one, so "例如e.g. "
# holds the abbreviation too.
_SENTENCE_END_RE = re.compile(
    r"[。！？]|[!?](?=\s|[一-鿿])|(?a:(?<!\be\.g)(?<!\bi\.e))\.(?=\s|[一-鿿])",
    re.IGNORECASE,
)
_LIST_MARKER_RE = re.compile(r"(?:[-*•]|\d+[.)])(?:\s+|$)")


def split_sentences(text: str) -> list[tuple[int, str]]:
    """The sentences of *text*, each with the offset where it starts.

    Text splits at line breaks; within a line, after ``.``, ``!`` or
    ``?`` followed by whitespace (but not after ``e.g.`` or ``i.e.``, in
    any case), and after ``。``, ``！`` or ``？``. So ``list_ops.py``,
    ``3.11``, ``e.g.`` and ``i.e.`` end no sentence. Each piece is
    stripped of whitespace and of a leading list marker (``- ``, ``* ``,
    ``• ``, ``1. ``, ``1) ``), and empty pieces are dropped. A marker that
    starts a line is set aside before the line is split, so ``1. Read``
    gives no sentence ``1.``. Each sentence is the text at its offset:
    ``text[offset:offset + len(sentence)] == sentence``.
    """
    sentences: list[tuple[int, str]] = []
    pos = 0
    while pos <= len(text):
        newline = text.find("\n", pos)
        line_end = len(text) if newline == -1 else newline
        start = _skip_marker(text, pos, line_end)
        line = text[start:line_end]
        cuts = [start + m.end() for m in _SENTENCE_END_RE.finditer(line)]
        for piece_start, piece_end in zip(
            [start, *cuts], [*cuts, line_end], strict=True
        ):
            if piece_start != start:
                piece_start = _skip_marker(text, piece_start, piece_end)
            sentence = text[piece_start:piece_end].rstrip()
            if sentence:
                sentences.append((piece_start, sentence))
        if newline == -1:
            break
        pos = newline + 1
    return sentences


def _skip_marker(text: str, start: int, end: int) -> int:
    """Where text[start:end] begins after its whitespace and one list marker."""
    piece = text[start:end]
    stripped = piece.lstrip()
    offset = len(piece) - len(stripped)
    marker = _LIST_MARKER_RE.match(stripped)
    if marker is not None:
        rest = stripped[marker.end() :]
        offset += marker.end() + len(rest) - len(rest.lstrip())
    return start + offset


_RUN_RE = re.compile(r"[A-Za-z0-9_.~/\\-]+")
_FILE_NAME_RE = re.compile(r"[A-Za-z0-9_-]{2,}(?:\.[A-Za-z0-9_-]+)*\.([A-Za-z0-9]+)")
_PATH_PREFIXES = ("/", "./", "../", "~/", "\\", ".\\", "..\\", "~\\")
_SEGMENT_SEPARATOR_RE = re.compile(r"[/\\]")


def _is_file_name(name: str, next_char: str) -> bool:
    """Whether *name*, followed in its text by *next_char*, is a file name."""
    match = _FILE_NAME_RE.fullmatch(name)
    return (
        match is not None
        and match.group(1).lower() in FILE_EXTENSIONS
        and next_char != "("
    )


def _is_path(token: str, next_char: str) -> bool:
    """Whether *token*, followed in its text by *next_char*, is a path."""
    if "/" not in token and "\\" not in token:
        return False
    if not any(char.isalnum() for char in token):
        return False
    if token.startswith(_PATH_PREFIXES):
        return True
    segments = _SEGMENT_SEPARATOR_RE.split(token)
    last = len(segments) - 1
    return any(
        _is_file_name(segment, next_char if i == last else "/")
        for i, segment in enumerate(segments)
    )


def path_spans(text: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` spans of the file names and paths in *text*.

    Candidates are the maximal runs of ``[A-Za-z0-9_.~/\\-]``, without
    trailing dots; any other character (CJK included) ends a run, so
    ``读取list_ops.py文件`` gives ``list_ops.py``.

    - A **file name** matches
      ``[A-Za-z0-9_-]{2,}(\\.[A-Za-z0-9_-]+)*\\.([A-Za-z0-9]+)``, its last
      extension is in :data:`FILE_EXTENSIONS` (in any case), and no ``(``
      follows it: ``list_ops.py`` and ``README.md``, but not ``e.g``,
      ``3.11``, ``v1.2``, ``self.items``, ``os.path``,
      ``functools.reduce``, ``math.log`` or ``response.json()``.
    - A **path** holds ``/`` or ``\\`` and a letter or digit, and starts
      with ``/``, ``./``, ``../`` or ``~/`` (or the same with ``\\``), or
      has a segment that is a file name: ``tests/test_list_ops.py``,
      ``./run.sh``, ``~/notes``, ``/tmp/out``, but not ``read/write``,
      ``and/or``, ``10/22`` or ``src/utils``.
    """
    spans: list[tuple[int, int]] = []
    for match in _RUN_RE.finditer(text):
        start, end = match.span()
        while end > start and text[end - 1] == ".":
            end -= 1
        token = text[start:end]
        next_char = text[end : end + 1]
        if token and (_is_file_name(token, next_char) or _is_path(token, next_char)):
            spans.append((start, end))
    return spans


def _names_file(token: str, next_char: str) -> bool:
    """Whether *token* is a file name, or a path whose last segment is one."""
    if _is_file_name(token, next_char):
        return True
    last = _SEGMENT_SEPARATOR_RE.split(token)[-1]
    return (
        last != token and _is_path(token, next_char) and _is_file_name(last, next_char)
    )


# ── Markers ──


def _english(*words: str) -> re.Pattern[str]:
    """A whole-word, case-insensitive pattern for English marker words.

    Word boundaries are ASCII ones, so a marker next to CJK text, as in
    ``然后run测试``, is still a whole word. The words of a marker may be
    apart by any whitespace, U+00A0 and U+3000 included.
    """
    return re.compile(r"(?a:\b)(?:" + "|".join(words) + r")(?a:\b)", re.IGNORECASE)


_INTENT_EN = _english(r"let\s+me", r"I['’]ll", r"I\s+need\s+to", "next", "now", "then")
_INTENT_ZH = ("我先", "接下来", "然后", "下一步", "现在")
_ACTION_EN = _english(
    "read",
    "open",
    "write",
    "edit",
    "fix",
    "run",
    "test",
    "check",
    "call",
    "create",
    "update",
    "add",
    "replace",
    "implement",
)
_ACTION_ZH = (
    "读",
    "写",
    "打开",
    "修改",
    "修复",
    "运行",
    "测试",
    "检查",
    "调用",
    "创建",
    "更新",
    "实现",
)
_FINDING_EN = _english(
    r"expect\w*",
    r"fail\w*",
    r"pass(?:es|ed|ing)?",
    r"errors?",
    r"returns?",
    "returned",
    "because",
    r"means?",
)
_FINDING_ZH = ("所以", "因为", "结果")
_DECISION_EN = _english(
    "therefore", r"the\s+fix\s+is", r"I['’]ll\s+use", "instead", "must"
)
_DECISION_ZH = ("因此", "改成", "应该")
_HEDGE_EN = _english("maybe", "probably", "might", r"suspect\w*")
_HEDGE_ZH = ("可能", "也许")
_OPEN_EN = _english(r"not\s+sure", r"need\s+to\s+check")
_OPEN_ZH = ("不确定",)


def _marked(english: re.Pattern[str], chinese: Sequence[str]) -> Callable[[str], bool]:
    def test(sentence: str) -> bool:
        return english.search(sentence) is not None or any(
            word in sentence for word in chinese
        )

    return test


_has_intent = _marked(_INTENT_EN, _INTENT_ZH)
_has_action = _marked(_ACTION_EN, _ACTION_ZH)
_has_finding = _marked(_FINDING_EN, _FINDING_ZH)
_has_decision = _marked(_DECISION_EN, _DECISION_ZH)
_has_hedge = _marked(_HEDGE_EN, _HEDGE_ZH)
_says_unsure = _marked(_OPEN_EN, _OPEN_ZH)


def _is_open(sentence: str) -> bool:
    return sentence.endswith(("?", "？")) or _says_unsure(sentence)


_PLAN_MARKERS = _english(
    "could",
    "should",
    "might",
    "maybe",
    "probably",
    "option",
    "would",
    r"let\s+me",
    r"i['’]ll",
    r"i\s+will",
    r"we\s+can",
)
_NEGATION_MARKERS = _english(
    "not",
    "never",
    r"no\s+longer",
    r"cancel+ed",
    r"cancel+ing",
    r"\w+n['’]t",
    "cannot",
)
_CJK_NEGATION = re.compile(
    r"不再|取消|没有|不是|不在|未(?=在|举行|完成|开始|参加|发生)|没(?=在|有|能|去|来|做)"
)


def has_negative_assertion(text: str) -> bool:
    """Conservative polarity signal for admission and source matching, not a parser."""
    return bool(_NEGATION_MARKERS.search(text) or _CJK_NEGATION.search(text))


_CJK_ASSERTION = re.compile(
    r"^[一-鿿]{2,12}(?:已取消|取消了|不再住在[一-鿿]{2,12}|住在[一-鿿]{2,12}|"
    r"(?:不在|未在)[^?？!！。]{1,80}举行(?:，[^?？!！。]{0,80})?)[。.!！]?$"
)
_EN_NEGATIVE_ASSERTION = re.compile(
    r"^[A-Za-z][^?？!！]*\b(?:is not|are not|was not|were not|does not|did not|"
    r"\w+n['’]t|no longer|was cancelled|was canceled|is cancelled|is canceled)\b[^?？!！]*[!！]?$",
    re.IGNORECASE,
)
_UNCERTAIN_ASSERTION = re.compile(
    r"\b(?:if|unless|whether|when|may(?!\s+\d{1,2}(?:st|nd|rd|th)?\b)|might|could|not only|not\s+(?:not|cancel+ed))\b|如果|假如|是否|不但|不仅|可能|也许",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"""["“”‘’]|'[^']+\x27""")


def _is_plan(sentence: str) -> bool:
    return not _is_open(sentence) and bool(_PLAN_MARKERS.search(sentence))


def _eligible_fact(sentence: str) -> bool:
    if (
        _is_open(sentence)
        or _is_plan(sentence)
        or _UNCERTAIN_ASSERTION.search(sentence)
    ):
        return False
    if _NEGATION_MARKERS.search(sentence):
        return (
            len(sentence) <= NOTE_ITEM_CHARS
            and not _QUOTED.search(sentence)
            and bool(_EN_NEGATIVE_ASSERTION.fullmatch(sentence))
        )
    if _CJK_NEGATION.search(sentence):
        return bool(_CJK_ASSERTION.fullmatch(sentence))
    return True


# ── Calls and results ──

_BUILTIN_TOOLS = frozenset({"file_read", "file_write", "shell_exec", "python_exec"})
_PYTEST_COUNT_RE = re.compile(
    r"\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)"
)
_PYTEST_TIME_RE = re.compile(r" in \d+(?:\.\d+)?s")


def _joined(head: str, tail: str) -> str:
    return f"{head} {tail}" if tail else head


def _string_argument(call: ToolCall, name: str) -> str:
    value = call.arguments.get(name)
    return value if isinstance(value, str) else ""


def _command(call: ToolCall) -> str:
    """A shell call's command, joined if a list, whitespace collapsed, shortened."""
    command = call.arguments.get("command")
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if not isinstance(command, str):
        return ""
    return _shorten(_collapse(command), _SHORT_COMMAND_CHARS)


def short_call(call: ToolCall) -> str:
    """A one-line form of *call*, at most 100 characters.

    ``read <path>`` (``file_read``), ``write <path>`` (``file_write``),
    ``run <command>`` (``shell_exec``: a list command joined with spaces,
    whitespace collapsed, shortened to 60), ``python <first non-blank line
    of code, shortened to 60>`` (``python_exec``); for any other tool, its
    name, then its first non-blank string argument with whitespace
    collapsed and shortened to 40, when it has one.
    """
    name = call.tool_name
    if name == "file_read":
        form = _joined("read", _string_argument(call, "path"))
    elif name == "file_write":
        form = _joined("write", _string_argument(call, "path"))
    elif name == "shell_exec":
        form = _joined("run", _command(call))
    elif name == "python_exec":
        line = _first_nonblank_line(_string_argument(call, "code"))
        form = _joined("python", _shorten(line, _SHORT_COMMAND_CHARS))
    else:
        first = next(
            (v for v in call.arguments.values() if isinstance(v, str) and v.strip()),
            "",
        )
        form = _joined(name, _shorten(_collapse(first), _SHORT_ARGUMENT_CHARS))
    return _shorten(form, _SHORT_CALL_CHARS)


def _pytest_summary(output: str) -> str:
    """The last pytest summary line of *output*, without its ``=`` padding."""
    summary = ""
    for line in output.splitlines():
        if _PYTEST_COUNT_RE.search(line) and _PYTEST_TIME_RE.search(line):
            summary = line.strip().strip("=").strip()
    return summary


def _output_summary(output: str) -> str:
    summary = _pytest_summary(output)
    if summary:
        return summary
    for line in output.splitlines():
        if any(char.isalnum() for char in line):
            return _shorten(line.strip(), _SUMMARY_CHARS)
    return "(no output)"


def _records(output: str) -> list[dict[str, Any]]:
    """The objects of a JSON *output* that carry ``record_id`` or ``id``.

    The objects are those of a top-level list, of the list values of a
    top-level object, and that object itself, in this order.
    """
    body = output.strip()
    if not body.startswith(("[", "{")):
        return []
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        return []
    objects: list[Any] = []
    if isinstance(value, list):
        objects = value
    elif isinstance(value, dict):
        for item in value.values():
            if isinstance(item, list):
                objects.extend(item)
        objects.append(value)
    return [
        obj
        for obj in objects
        if isinstance(obj, dict) and ("record_id" in obj or "id" in obj)
    ]


def _error(result: ToolResult) -> str:
    return _first_nonblank_line(result.error)


# A result that says nothing about what the attempt did: no text, only a
# boolean or None, only an exit code, or only empty quotes and punctuation
# around one of those (the D11 run's "'' True."). It never becomes an
# observed digest item and never makes one a rehearsal candidate.
_TRIVIAL_LINE_RE = re.compile(
    r"(?:\[exit code -?\d+\]|exit code -?\d+|True|False|None)"
)
_TRIVIAL_NOISE_RE = re.compile(r"''|\"\"|[\s\[\]{}()'\".,:;=+-]")
_EXIT_CODE_RE = re.compile(r"\[exit code (-?\d+)\]")


def _trivial_line(line: str) -> bool:
    """Whether one non-blank output line carries no information."""
    if _TRIVIAL_LINE_RE.fullmatch(line):
        return True
    return _TRIVIAL_NOISE_RE.sub("", line).strip() in ("", "True", "False", "None")


def _trivial_output(text: str) -> bool:
    """Whether *text* is empty or only trivial lines (see ``_trivial_line``)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return not lines or all(_trivial_line(line) for line in lines)


def _observed(call: ToolCall, result: ToolResult) -> str:
    """The observed line of one call and its result; "" when trivial."""
    name = call.tool_name
    error = _error(result)
    if not error and _trivial_output(result.output):
        return ""
    if error:
        line = f"{short_call(call)}: Error: {error}"
    elif name == "file_read":
        data = result.output.encode("utf-8", "surrogatepass")
        sha = hashlib.sha256(data).hexdigest()[:8]
        path = _string_argument(call, "path")
        line = f"{_joined('read', path)}: {len(result.output)} chars, sha {sha}"
    elif name == "file_write":
        content = _string_argument(call, "content")
        line = (
            f"{_joined('wrote', _string_argument(call, 'path'))}: {len(content)} chars"
        )
    elif name == "shell_exec":
        line = f"ran `{_command(call)}`: {_output_summary(result.output)}"
    elif name == "python_exec":
        line = f"ran `python`: {_output_summary(result.output)}"
    else:
        records = _records(result.output)
        line = f"{name}: {len(result.output)} chars"
        if records:
            line += f", {len(records)} records"
    return _shorten(line, NOTE_OBSERVED_CHARS)


def _outcome(pairs: Sequence[tuple[ToolCall, ToolResult]]) -> str:
    if not pairs:
        return ""
    for _call, result in pairs:
        error = _error(result)
        if error:
            return _shorten(f"Error: {error}", NOTE_OBSERVED_CHARS)
    summary = ""
    for call, result in pairs:
        if call.tool_name in ("shell_exec", "python_exec"):
            summary = _pytest_summary(result.output) or summary
    return _shorten(summary or "ok", NOTE_OBSERVED_CHARS)


def _source_records(
    pairs: Sequence[tuple[ToolCall, ToolResult]],
) -> list[tuple[str, str]]:
    """``(id, text)`` for the records of this turn's JSON tool results.

    The text is the record object as JSON, cut to 500 characters. It is
    what :func:`candidate_items` compares an item with; the ids are the
    same ones :func:`_source_ids` returns.
    """
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for call, result in pairs:
        if call.tool_name in _BUILTIN_TOOLS or _error(result):
            continue
        for obj in _records(result.output):
            value = obj.get("record_id")
            if value is None:
                value = obj.get("id")
            if isinstance(value, bool) or not isinstance(value, int | str):
                continue
            text = str(value)
            if not text or text in seen:
                continue
            seen.add(text)
            try:
                body = json.dumps(obj, ensure_ascii=False)
            except (TypeError, ValueError):
                body = ""
            records.append((text, body[:500]))
    return records


def source_records(
    pairs: Sequence[tuple[ToolCall, ToolResult]],
) -> list[tuple[str, str]]:
    """The records of one reply's tool results, in call order (see ``_source_records``)."""
    return _source_records(pairs)


def _source_ids(pairs: Sequence[tuple[ToolCall, ToolResult]]) -> list[str]:
    return [record_id for record_id, _text in _source_records(pairs)]


TASK_SOURCE_RECORD_LIMIT = 200  # the task's most recent tool-result records kept


@dataclass
class SourceRecordLog:
    """The task's tool-result records, oldest first and bounded.

    It is where a note's item finds the earlier observation that established
    it: the run adds every executed reply's records, and :meth:`earlier`
    splits them into the results still in the window and the rest, both
    newest first. Only the task's last :data:`TASK_SOURCE_RECORD_LIMIT`
    records are kept; another task's records are never in this log.
    """

    limit: int = TASK_SOURCE_RECORD_LIMIT
    entries: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)

    def add(self, call_id: str, records: Sequence[tuple[str, str]]) -> None:
        """Add one result's records, dropping the oldest above the limit."""
        if not records:
            return
        self.entries.append((call_id, list(records)))
        total = sum(len(kept) for _call_id, kept in self.entries)
        while total > self.limit and self.entries:
            call_id, kept = self.entries[0]
            excess = total - self.limit
            if len(kept) <= excess:
                self.entries.pop(0)
                total -= len(kept)
            else:
                self.entries[0] = (call_id, kept[excess:])
                total -= excess

    def earlier(
        self, window_call_ids: Iterable[str]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """``(window, task)`` records, newest first, for a new note's items."""
        window = set(window_call_ids)
        still: list[tuple[str, str]] = []
        rest: list[tuple[str, str]] = []
        for call_id, records in reversed(self.entries):
            target = still if call_id in window else rest
            target.extend(records)
        return still, rest

    @property
    def count(self) -> int:
        """How many records the log holds (at most ``limit``)."""
        return sum(len(kept) for _call_id, kept in self.entries)


def _earlier_records(
    items: Sequence[str],
    own: Sequence[tuple[str, str]],
    window_records: Sequence[tuple[str, str]],
    task_records: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """The earlier records that can establish one of *items*, window then task.

    Only a record that would source one of the items (the same strong key
    or two distinctive words as :func:`_item_source_ids`) is kept, so a
    note does not carry the whole task's records. The caller's order is
    kept: ``own``, then *window_records*, then *task_records*;
    :func:`candidate_items` selects the uniquely best match.
    """
    keys = [item_keys for item in items if (item_keys := _item_keys(item))]
    if not keys:
        return []
    seen = {record_id for record_id, _text in own}
    kept: list[tuple[str, str]] = []
    for records in (window_records, task_records):
        for record_id, text in records:
            if record_id in seen:
                continue
            if any(_item_matches(item_keys, text) for item_keys in keys):
                kept.append((record_id, text))
                seen.add(record_id)
    return kept


# ── The note ──


def note_from_reply(
    content: str,
    thinking: str = "",
    tool_calls: Sequence[ToolCall] = (),
    tool_results: Sequence[ToolResult] = (),
    *,
    kind: str = "",
    writer: str = "auto",
    fallback: str = "",
    window_records: Sequence[tuple[str, str]] = (),
    task_records: Sequence[tuple[str, str]] = (),
) -> tuple[ReasoningNote, str]:
    """Compress one model reply into a note; pure and deterministic.

    Returns the note and ``rest``, *content* without its ``<note>`` blocks
    (see :func:`split_note`). *kind* sets the note's kind; "" derives it:
    ``step`` with calls, ``answer`` with text in ``rest``, else ``empty``.
    With *writer* ``"auto"`` and a ``<note>`` block in *content*, the
    fields come from :func:`parse_note_block`, with ``source = "model"``;
    otherwise (and always with *writer* ``"heuristic"``) a heuristic
    writes them, with ``source = "heuristic"``.

    The heuristic reads ``thinking + "\\n\\n" + rest``, with every fenced
    code block replaced by a line break (with ``writer="heuristic"``,
    *content* with only its ``<note>`` tags removed stands for ``rest``).
    It splits that text into sentences (:func:`split_sentences`) and gives
    each sentence to one field at most, in this order:

    - ``plan``: two sentences with an intent marker, preferring those that
      name a file or path or hold an action word (the latest preferred,
      then the latest others), in text order, then `` → `` and the calls'
      short forms (:func:`short_call`) joined by ``; ``;
    - ``decided``: the last 3 sentences with a decision marker that start
      in the last :data:`DECIDED_TAIL_SHARE` of the text;
    - ``open_questions``: the last 3 sentences that end in ``?`` or ``？``
      or say they are not sure;
    - ``hypotheses``: the last 3 hedged sentences, prefixed ``open: ``;
    - ``found``: every bullet, numbered item or table row, and every
      sentence of a plain paragraph, that carries a strong key
      (:func:`_item_keys`), in text order, up to :data:`NOTE_MAX_FACTS`;
      then the last 4 sentences with a finding marker that start in the
      last :data:`FOUND_WINDOW_CHARS` of the text, as far as the 12-item
      cap leaves room. A line that becomes a fact is not used by another
      field.

    In both cases ``observed``, ``outcome`` and ``source_ids`` come from
    the calls and their results, paired by index; ``reasoning_chars`` is
    ``len(thinking)``; and ``remember``, when the model's note gives none,
    is whether the note has a ``found`` or ``decided`` item
    (``remember_by = "heuristic"``).

    *window_records* and *task_records* are earlier observations of this
    task, newest first: the results still in the working-memory window,
    then the rest. An item whose own reply has no matching record finds
    one there, window before task, and only a record that shares one of
    the item's keys is kept; a note never carries another task's records.
    The pool's bound is :data:`TASK_SOURCE_RECORD_LIMIT`.

    When neither the model's block nor the heuristic finds a content
    field, a minimal note is still made so that every reply has one: the
    first sentences of *fallback* when it is non-empty (a reflection's
    ``summary``), else of the reply's visible text, become ``found``
    items. A reply with no visible text keeps an empty note.

    The note's identity fields stay at their defaults. Raises
    ``ValueError`` for a *writer* other than ``"auto"`` and
    ``"heuristic"``.
    """
    if writer not in ("auto", "heuristic"):
        raise ValueError(f"writer must be 'auto' or 'heuristic', not {writer!r}")
    block, rest = split_note(content)
    visible = rest
    note: ReasoningNote
    if fallback and not thinking.strip():
        # A thinking-less reply is summarized from *fallback*, not read as
        # its raw text: a reflection's JSON values are not a note.
        note = ReasoningNote(source="heuristic")
        minimal = _without_fences(fallback).strip()
    else:
        if writer == "auto" and block is not None:
            note = parse_note_block(block)
        else:
            if writer == "heuristic":
                visible = _NOTE_TAG_RE.sub("", content)
            text = _without_fences(thinking + "\n\n" + visible)
            note = _heuristic_note(
                text,
                tool_calls,
                factual_text=_without_fences(visible) if block is None else "",
            )
        minimal = _without_fences(fallback or visible).strip()

    empty = not (
        note.plan
        or note.found
        or note.decided
        or note.hypotheses
        or note.open_questions
    )
    if empty and minimal:
        sentences = split_sentences(minimal)
        note.found = _items(
            sentence
            for _offset, sentence in sentences[:NOTE_MAX_FOUND]
            if _eligible_fact(sentence)
        )

    pairs = list(zip(tool_calls, tool_results, strict=False))
    note.observed = [
        line for call, result in pairs if (line := _observed(call, result))
    ][:NOTE_MAX_OBSERVED]
    note.outcome = _outcome(pairs)
    own = _source_records(pairs)
    note.source_ids = [record_id for record_id, _text in own]
    note.source_records = own + _earlier_records(
        [*note.found, *note.decided], own, window_records, task_records
    )
    note.reasoning_chars = len(thinking)
    if kind:
        note.kind = kind
    elif tool_calls:
        note.kind = "step"
    else:
        note.kind = "answer" if rest.strip() else "empty"
    if not note.remember_by:
        note.remember = bool(note.found or note.decided)
        note.remember_by = "heuristic"
    return note, rest


_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+\S")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


def _plain_line(text: str) -> str:
    """*text* without bullet or emphasis markers, whitespace collapsed."""
    return _collapse(text.replace("**", "").replace("|", " "))


def _factual_lines(text: str) -> list[str]:
    """The factual lines of *text*, in text order.

    Every bullet, numbered item and table row is one line, and every
    sentence of a plain paragraph is one; a heading is a paragraph line
    too. A line is factual when :func:`_strong_keys` finds a strong key in
    it.
    """
    items: list[str] = []
    seen_lines: set[int] = set()
    for offset, sentence in split_sentences(text):
        line_start = text.rfind("\n", 0, offset) + 1
        line_end = text.find("\n", offset)
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]
        if _BULLET_RE.match(line) or _TABLE_ROW_RE.match(line):
            if line_start in seen_lines:
                continue
            seen_lines.add(line_start)
            if _TABLE_RULE_RE.match(line):
                continue
            if _TABLE_ROW_RE.match(line):
                line = line.strip().strip("|")
            else:
                line = _LIST_MARKER_RE.sub("", line.strip(), count=1)
            item = _plain_line(line)
        else:
            item = _plain_line(sentence)
        parts = [item]
        if not _eligible_fact(item):
            parts = [part for _, part in split_sentences(item)]
            parts = [
                re.sub(
                    r"\([^()]*\)",
                    lambda m: (
                        ""
                        if _is_plan(m.group())
                        and not _NEGATION_MARKERS.search(m.group())
                        and "?" not in m.group()
                        else m.group()
                    ),
                    part,
                ).strip()
                for part in parts
            ]
        for part in parts:
            if _eligible_fact(part) and (
                _CJK_ASSERTION.fullmatch(part)
                or any(not _ITEM_CJK_RE.fullmatch(key) for key in _strong_keys(part))
            ):
                items.append(part)
    return items


def _heuristic_note(
    text: str, tool_calls: Sequence[ToolCall], *, factual_text: str = ""
) -> ReasoningNote:
    """The content fields that the heuristic reads off *text*.

    *factual_text* is the reply's visible text; only its bullets, table
    rows and plain sentences become facts, so a *thinking* sentence never
    takes a field from a later ``<note>`` block or from a question.
    """
    sentences = split_sentences(text)
    visible_start = len(text) - len(factual_text) if factual_text else len(text)
    facts = _factual_lines(factual_text) if factual_text else []
    fact_keys = {_normalized_item(item) for item in facts}

    def is_fact(sentence: str) -> bool:
        key = _normalized_item(_plain_line(sentence))
        return bool(key) and any(
            key == fact_key or key in fact_key for fact_key in fact_keys
        )

    taken: set[int] = {
        index
        for index, (_offset, sentence) in enumerate(sentences)
        if is_fact(sentence)
    }

    def take(
        test: Callable[[str], bool],
        limit: int,
        start: float = 0,
        *,
        facts_only: bool = False,
    ) -> list[str]:
        """The last *limit* sentences left that pass *test* and start at *start*."""
        chosen = [
            i
            for i, (offset, sentence) in enumerate(sentences)
            if i not in taken
            and offset >= start
            and test(sentence)
            and (not facts_only or offset < visible_start or _eligible_fact(sentence))
        ][-limit:]
        taken.update(chosen)
        return [sentences[i][1] for i in chosen]

    intent = [
        i
        for i, (offset, sentence) in enumerate(sentences)
        if i not in taken
        and (
            (
                _has_intent(sentence)
                and (offset < visible_start or not _is_open(sentence))
            )
            or (offset >= visible_start and _is_plan(sentence))
        )
    ]
    preferred = [
        i for i in intent if path_spans(sentences[i][1]) or _has_action(sentences[i][1])
    ]
    preferred_set = set(preferred)
    others = [i for i in intent if i not in preferred_set]
    visible_intent = [i for i in intent if sentences[i][0] >= visible_start]
    if visible_intent:
        chosen = visible_intent[:1]
    else:
        chosen = preferred[-2:]
        if len(chosen) < 2:
            chosen += others[len(chosen) - 2 :]
    chosen.sort()
    taken.update(chosen)

    note = ReasoningNote(source="heuristic")
    note.plan = _plan(" ".join(sentences[i][1] for i in chosen), tool_calls)
    tail = len(text) - DECIDED_TAIL_SHARE * len(text)
    note.decided = _items(take(_has_decision, NOTE_MAX_DECIDED, tail, facts_only=True))
    note.open_questions = _items(take(_is_open, NOTE_MAX_OPEN))
    note.hypotheses = _items(
        f"open: {sentence}" for sentence in take(_has_hedge, NOTE_MAX_HYPOTHESES)
    )
    window = len(text) - FOUND_WINDOW_CHARS
    found = _items(facts)[:NOTE_MAX_FACTS]
    seen = {_normalized_item(item) for item in found}
    for item in _items(take(_has_finding, NOTE_MAX_FOUND, window, facts_only=True)):
        key = _normalized_item(item)
        if len(found) >= NOTE_MAX_FACTS:
            break
        if key and key not in seen:
            seen.add(key)
            found.append(item)
    note.found = found
    return note


def _items(sentences: Iterable[str]) -> list[str]:
    return [_shorten(sentence, NOTE_ITEM_CHARS) for sentence in sentences]


def _plan(sentences: str, tool_calls: Sequence[ToolCall]) -> str:
    """The plan: its sentences, then `` → `` and the calls, within the cap.

    Over :data:`NOTE_PLAN_CHARS`, the sentences are shortened first; when
    the calls leave them no room, the plan is ``→ <calls>``, shortened.
    """
    calls = "; ".join(short_call(call) for call in tool_calls)
    if not calls:
        return _shorten(sentences, NOTE_PLAN_CHARS)
    if sentences:
        room = NOTE_PLAN_CHARS - len(" → ") - len(calls)
        shortened = _shorten(sentences, room)
        if shortened:
            return f"{shortened} → {calls}"
    return _shorten(f"→ {calls}", NOTE_PLAN_CHARS)


def _render(fields: dict[str, Any], executor: bool) -> str:
    lines: list[str] = []
    if fields["plan"]:
        lines.append(f"plan: {fields['plan']}")
    for label, key in (
        ("found", "found"),
        ("decided", "decided"),
        ("hypotheses", "hypotheses"),
        ("open", "open_questions"),
    ):
        if fields[key]:
            lines.append(f"{label}: " + "; ".join(fields[key]))
    if executor:
        if fields["observed"]:
            lines.append("observed: " + "; ".join(fields["observed"]))
        if fields["outcome"]:
            lines.append(f"outcome: {fields['outcome']}")
    return "\n".join(lines)


def render_note(
    note: ReasoningNote, *, executor: bool = True, limit: int = NOTE_CHARS
) -> str:
    """*note* as text, at most *limit* characters; the note is not changed.

    Lines ``plan: …``, ``found: …``, ``decided: …``, ``hypotheses: …``,
    ``open: …`` and, with *executor*, ``observed: …`` and ``outcome: …``,
    joined with ``\\n``; list items are joined with ``; ``, and empty
    fields are left out. Over *limit*, items are dropped from the end of
    ``hypotheses``, then ``open_questions``, then ``observed``, then
    ``decided``, then ``found``, one at a time, until the text fits; then
    ``plan`` is shortened; then the text is cut at *limit*.
    """
    fields: dict[str, Any] = {
        "plan": note.plan,
        "found": list(note.found),
        "decided": list(note.decided),
        "hypotheses": list(note.hypotheses),
        "open_questions": list(note.open_questions),
        "observed": list(note.observed),
        "outcome": note.outcome,
    }
    text = _render(fields, executor)
    for key in ("hypotheses", "open_questions", "observed", "decided", "found"):
        if key == "observed" and not executor:
            continue
        while fields[key] and len(text) > limit:
            fields[key].pop()
            text = _render(fields, executor)
    if len(text) > limit and fields["plan"]:
        room = limit - (len(text) - len(fields["plan"]))
        fields["plan"] = _shorten(fields["plan"], room)
        text = _render(fields, executor)
    return text[:limit]


def note_title(note: ReasoningNote) -> str:
    """A one-line title for *note*, at most :data:`TITLE_CHARS` characters.

    The first non-empty of ``plan``, the first ``found``, ``decided`` and
    ``open_questions`` items and the first ``observed`` line, with
    whitespace collapsed; "" when all are empty.
    """
    for candidate in (
        note.plan,
        *note.found[:1],
        *note.decided[:1],
        *note.open_questions[:1],
        *note.observed[:1],
    ):
        title = _collapse(candidate)
        if title:
            return _shorten(title, TITLE_CHARS)
    return ""


# ── The attempt digest ──


def _normalized_item(item: str) -> str:
    """*item* with whitespace collapsed and lowercased, for equality tests."""
    return _collapse(item).lower()


def _observed_key(line: str) -> str:
    """The path or command an observed line is about, for deduplication.

    ``read <path>: …`` and ``wrote <path>: …`` give the path, ``ran
    `<command>`: …`` the command, and any other line the line itself.
    """
    for prefix in ("read ", "wrote "):
        if line.startswith(prefix):
            head = line[len(prefix) :].split(":", 1)[0].strip()
            if head:
                return f"{prefix.strip()}:{head}"
    if line.startswith("ran `"):
        head = line[5:].split("`", 1)[0].strip()
        if head:
            return f"ran:{head}"
    return line


def _task_line(query: str) -> str:
    """The first meaningful sentence, ignoring headings and bare labels."""
    lines = []
    for line in query.splitlines():
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}(?:\s|$)", stripped):
            continue
        if re.fullmatch(
            r"(?:instructions?|task|description|requirements?)\s*:?", stripped, re.I
        ):
            continue
        lines.append(line)
    text = "\n".join(lines)
    sentences = split_sentences(text)
    if sentences:
        text = sentences[0][1]
    return _shorten(_collapse(text), DIGEST_TASK_CHARS)


def _workspace_relative(path: str, root: str) -> str:
    """*path* relative to *root* when it is absolute and inside it."""
    path = path.strip()
    if not path:
        return ""
    if root:
        try:
            base = os.path.abspath(root)
        except OSError:
            base = ""
        if base and os.path.isabs(path):
            try:
                relative = os.path.relpath(path, base)
            except (OSError, ValueError):
                relative = path
            if relative != ".." and not relative.startswith(".." + os.sep):
                path = relative
        elif not os.path.isabs(path):
            path = os.path.normpath(path)
    return path.replace(os.sep, "/")


def _digest_internal_path(path: str) -> bool:
    """Hide dot paths and the agent's own scratch note from file lists."""
    parts = path.replace("\\", "/").split("/")
    if any(part.startswith(".") and part not in (".", "..") for part in parts):
        return True
    return parts[-1].casefold() in {"memory_note.md", "memory_notes.md"}


def _digest_files(calls: Sequence[tuple[ToolCall, ToolResult]], root: str) -> list[str]:
    """The visible paths this attempt's successful ``file_write`` calls wrote."""
    files: list[str] = []
    for call, result in calls:
        if call.tool_name != "file_write" or result.error.strip():
            continue
        path = _workspace_relative(_string_argument(call, "path"), root)
        if path and not _digest_internal_path(path) and path not in files:
            files.append(_shorten(path, DIGEST_PATH_CHARS))
    if len(files) > DIGEST_FILES:
        more = len(files) - DIGEST_FILES
        files = [*files[:DIGEST_FILES], f"+{more}"]
    return files


_TEST_RUNNER_RE = re.compile(
    r"\bpytest\b|\bunittest\.(?:main|TextTestRunner|defaultTestLoader)\b"
    r"|\bpython[0-9.]*\s+-m\s+[\w.]+|\bnode\s+--test\b"
    r"|\b(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b"
    r"|\b(?:cargo|go)\s+test\b|\bmake\s+test\b",
    re.IGNORECASE,
)


def _call_text(call: ToolCall) -> str:
    """The command or code a check is read from; other tools give ""."""
    if call.tool_name == "python_exec":
        return _string_argument(call, "code")
    if call.tool_name == "shell_exec":
        command = call.arguments.get("command")
        if isinstance(command, list) and all(isinstance(part, str) for part in command):
            return " ".join(command)
        return command if isinstance(command, str) else ""
    return ""


def _written_import(text: str, written: Sequence[str], root: str) -> str:
    """Name a written workspace module imported by Python code, if any."""
    modules = set()
    for path in written:
        relative = _workspace_relative(path, root)
        if relative.endswith(".py") and not os.path.isabs(relative):
            parts = relative[:-3].split("/")
            if all(part.isidentifier() for part in parts):
                modules.add(".".join(parts))
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return ""
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names = [node.args[0].value]
        for name in names:
            if name in modules:
                return name
    return ""


def _is_check(call: ToolCall, written: Sequence[str], root: str) -> bool:
    """A test runner or code naming a file written earlier in this attempt."""
    text = _call_text(call)
    if not text.strip():
        return False
    if _TEST_RUNNER_RE.search(text):
        return True
    if call.tool_name == "python_exec" and _written_import(text, written, root):
        return True
    for path in written:
        relative = _workspace_relative(path, root)
        if relative and re.search(
            r"(?<![\w./-])" + re.escape(relative) + r"(?![\w./-])", text
        ):
            return True
        if path != relative and path in text:
            return True
    return False


def _safe_check_target(target: str) -> bool:
    """Avoid placing absolute, hidden or credential-like paths in a digest."""
    if not re.fullmatch(r"[A-Za-z0-9_./-]+(?:::[A-Za-z0-9_]+)*", target):
        return False
    path = target.split("::", 1)[0]
    if path.startswith(("/", "-")) or any(
        part.startswith(".") for part in path.split("/")
    ):
        return False
    return not re.search(r"secret|token|credential|password|api[_-]?key", target, re.I)


def _python_check_command(code: str) -> str:
    """Name a recognizable check without presenting arbitrary code as its label."""
    if len(code) > 32_000:
        return "python check"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return "python check"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not isinstance(owner, ast.Name):
            continue
        if owner.id == "unittest" and node.func.attr in ("main", "TextTestRunner"):
            return "python unittest"
        if owner.id == "pytest" and node.func.attr == "main":
            args = (
                node.args[0]
                if node.args
                else next((kw.value for kw in node.keywords if kw.arg == "args"), None)
            )
            if isinstance(args, (ast.List, ast.Tuple)):
                for arg in args.elts:
                    if not isinstance(arg, ast.Constant) or not isinstance(
                        arg.value, str
                    ):
                        break
                    target = arg.value
                    if target in ("-q", "-v", "-s", "-x"):
                        continue
                    if _safe_check_target(target):
                        path, *node_ids = target.split("::")
                        shown = "::".join([os.path.basename(path), *node_ids])
                        return f"python pytest {shown}"
                    break
            return "python pytest"
        if owner.id == "runpy" and node.func.attr == "run_path" and node.args:
            script = node.args[0]
            if (
                isinstance(script, ast.Constant)
                and isinstance(script.value, str)
                and script.value.endswith(".py")
                and _safe_check_target(script.value)
            ):
                return f"python runpy {os.path.basename(script.value)}"
    return "python check"


def _check_command(call: ToolCall) -> str:
    """A short, recognizable label for a check call."""
    text = _call_text(call)
    if call.tool_name == "python_exec":
        text = _python_check_command(text)
    return _shorten(_collapse(text), DIGEST_COMMAND_CHARS)


def _check_result(result: ToolResult) -> str:
    """Summarize a check by test counts, exit code or neutral completion."""
    if result.error.strip():
        return _shorten("Error: " + _error(result), DIGEST_RESULT_CHARS)
    output = result.output
    # Imported here: classify is a higher layer than the note types.
    from localmelo.melo.memory.procedures.classify import parse_test_summary

    summary = parse_test_summary(output)
    if summary is not None and summary.total:
        parts: list[str] = []
        if summary.failed:
            parts.append(f"{summary.failed} failed")
        if summary.errors:
            parts.append(f"{summary.errors} errors")
        if summary.passed or not parts:
            parts.append(f"{summary.passed} passed")
        return ", ".join(parts)
    first = output.split("\n", 1)[0].strip()
    match = _EXIT_CODE_RE.fullmatch(first)
    if match is not None:
        return f"exit {int(match.group(1))}"
    return "completed"


def _digest_verified(calls: Sequence[tuple[ToolCall, ToolResult]], root: str) -> str:
    """The last check, only when it passed."""
    written: list[str] = []
    last: tuple[ToolCall, ToolResult] | None = None
    last_module = ""
    for call, result in calls:
        if call.tool_name == "file_write" and not result.error.strip():
            path = _string_argument(call, "path")
            if path:
                written.append(path)
        if _is_check(call, written, root):
            last = (call, result)
            last_module = (
                _written_import(_call_text(call), written, root)
                if call.tool_name == "python_exec"
                and not _TEST_RUNNER_RE.search(_call_text(call))
                else ""
            )
    if last is None:
        return ""
    call, result = last
    output = result.output
    if result.error.strip() or re.search(
        r"\[exit code (?!0\])-?\d+\]|\b(?:FAILED|ERROR|Traceback)\b|\b[1-9]\d* (?:failed|errors?)\b|\bRan 0 tests?\b",
        output,
        re.I,
    ):
        return ""
    if _TEST_RUNNER_RE.search(_call_text(call)) and not re.search(
        r"\b[1-9]\d* passed\b|\bRan [1-9]\d* tests?\b.*\bOK\b|\[exit code 0\]",
        output,
        re.I | re.S,
    ):
        return ""
    command = f"python import {last_module}" if last_module else _check_command(call)
    return f"{command} -> {_check_result(result)}"


def digest_note(
    query: str,
    outcome: str = "",
    notes: Sequence[ReasoningNote] = (),
    *,
    task_id: str = "",
    attempt: int = 0,
    step: int = 0,
    calls: Sequence[tuple[ToolCall, ToolResult]] = (),
    root: str = "",
) -> ReasoningNote:
    """The digest of one attempt: its rehearsed notes and confirmed outcomes.

    Only *notes* whose stage is ``"rehearsed"`` contribute: their ``found``
    items become ``found`` (deduplicated by normalized text, in note
    order), their ``decided`` items become ``decided`` (the first three),
    their ``observed`` lines become ``observed`` (one per path or command,
    the first kept, trivial ones left out), and their ``open_questions``
    become ``open_questions`` (deduplicated). *calls* are the attempt's
    own tool calls and results: a successful ``file_write`` gives one
    ``files`` path, relative to *root* when it is absolute (at most
    :data:`DIGEST_FILES`, then ``+N``), and the last passing test runner or
    call referencing a previously written file gives ``verified``. A later
    failed check clears it. ``plan`` is *query*'s first meaningful sentence,
    shortened to
    :data:`DIGEST_TASK_CHARS`; ``outcome`` is the attempt's failure type
    or the run's status. ``source_ids`` is the union of the contributing
    notes' ids and the records behind their items, in order. The digest
    has ``kind="digest"``, ``remember=True`` and ``remember_by="rule"``.
    Its identity fields stay at their defaults; the caller sets them.
    """
    facts: list[str] = []
    seen_facts: set[str] = set()
    decided: list[str] = []
    seen_decided: set[str] = set()
    observed: list[str] = []
    seen_observed: set[str] = set()
    open_questions: list[str] = []
    seen_open: set[str] = set()
    source_ids: list[str] = []
    reasoning_chars = 0
    for note in notes:
        if note.stage != "rehearsed":
            continue
        reasoning_chars += note.reasoning_chars
        for item in note.found:
            key = _normalized_item(item)
            if key and key not in seen_facts:
                seen_facts.add(key)
                facts.append(item)
        for item in note.decided:
            key = _normalized_item(item)
            if key and key not in seen_decided and len(decided) < 3:
                seen_decided.add(key)
                decided.append(item)
        for line in note.observed:
            if not line.strip() or _trivial_output(line.split(": ", 1)[-1]):
                continue
            key = _observed_key(line)
            if key and key not in seen_observed:
                seen_observed.add(key)
                observed.append(line)
        for item in note.open_questions:
            key = _normalized_item(item)
            if key and key not in seen_open:
                seen_open.add(key)
                open_questions.append(item)
        for source_id in note.source_ids:
            if source_id not in source_ids:
                source_ids.append(source_id)
        for item in (*note.found, *note.decided):
            for source_id in _item_source_ids(item, note.source_records):
                if source_id not in source_ids:
                    source_ids.append(source_id)
    return ReasoningNote(
        task_id=task_id,
        step=step,
        attempt=attempt,
        kind="digest",
        source="heuristic",
        plan=_task_line(query),
        found=facts,
        decided=decided,
        open_questions=open_questions,
        observed=observed,
        files=_digest_files(calls, root),
        verified=_digest_verified(calls, root),
        outcome=_shorten(_collapse(outcome), NOTE_OBSERVED_CHARS),
        reasoning_chars=reasoning_chars,
        source_ids=source_ids,
        remember=True,
        remember_by="rule",
        stage="fresh",
    )


def render_digest(note: ReasoningNote, *, limit: int = DIGEST_CHARS) -> str:
    """*note*'s digest text, at most *limit* characters; the note is not changed.

    Lines ``task: …``, ``outcome: …``, ``files: …``, ``verified: …``,
    ``facts: …``, ``decided: …``, ``observed: …`` and ``open: …``, each
    left out when empty, list items joined with ``; ``. Over *limit*,
    items are dropped from the end of ``open``, then ``observed``, then
    ``decided``, then ``found``, then ``files``, then ``plan`` is
    shortened, then the text is cut at *limit*; ``outcome`` and
    ``verified`` are never dropped.
    """
    plan = note.plan
    open_questions = list(note.open_questions)
    observed = list(note.observed)
    decided = list(note.decided)
    found = list(note.found)
    files = list(note.files)

    def render() -> str:
        lines: list[str] = []
        if plan:
            lines.append(f"task: {plan}")
        if note.outcome:
            lines.append(f"outcome: {note.outcome}")
        if files:
            lines.append("files: " + "; ".join(files))
        if note.verified:
            lines.append(f"verified: {note.verified}")
        if found:
            lines.append("facts: " + "; ".join(found))
        if decided:
            lines.append("decided: " + "; ".join(decided))
        if observed:
            lines.append("observed: " + "; ".join(observed))
        if open_questions:
            lines.append("open: " + "; ".join(open_questions))
        return "\n".join(lines)

    text = render()
    for items in (open_questions, observed, decided, found, files):
        while items and len(text) > limit:
            items.pop()
            text = render()
    if len(text) > limit and plan:
        room = limit - (len(text) - len(plan))
        plan = _shorten(plan, room)
        text = render()
    return text[:limit]


# ── Long-term candidates (R6-lite) ──

CANDIDATE_TEXT_CHARS = 600  # a note's candidate text
REFLECTION_CANDIDATE_CHARS = 2_000  # a reflection's
RECORD_TEXT_CHARS = 500  # a source record's text in ReasoningNote.source_records

_ITEM_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ITEM_IDENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9_]*\d)(?=[A-Za-z0-9_]*[A-Za-z_])"
    r"[A-Za-z0-9_]{3,}"
    r"(?![A-Za-z0-9_])"
)
_ITEM_WORD_RE = re.compile(r"[A-Za-z0-9_]{4,}")
_ITEM_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_SENTENCE_BREAK_RE = re.compile(r"(?:[.!?](?=\s|$))|\n")

# A strong key is typed so that a date, a time, an amount and a number are
# compared after normalisation, not as the text that happened to carry them.
_KEY_PREFIXES = ("date:", "time:", "amount:", "number:", "phone:")

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_WORD = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_WEEKDAYS = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
)
_DATE_ISO_RE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_DATE_SLASH_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/])")
_MONTH_DAY_RE = re.compile(
    rf"(?<![A-Za-z])({_MONTH_WORD})(?![A-Za-z])\s+(\d{{1,2}})"
    r"(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?(?!\d)",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_WORD})(?![A-Za-z])"
    r"(?:,?\s*(\d{4}))?",
    re.IGNORECASE,
)
_QUANTITY_AFTER_RE = re.compile(
    r"\s*(?:cups?|price|miles?|km|kg|lbs?|liters?|litres?|hours?|minutes?|days?|weeks?|"
    r"people|off|of|tests?|files?|steps?|items?|pages?|runs?|attempts?|passed|failed|done|"
    r"majority|split|share)\b",
    re.IGNORECASE,
)
_TIME_COLON_RE = re.compile(
    r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)?",
    re.IGNORECASE,
)
_TIME_AMPM_RE = re.compile(
    r"(?<![\d:])(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?![A-Za-z])",
    re.IGNORECASE,
)
_CURRENCY_RE = r"(?:US\$|[$€£¥₹]|USD|EUR|GBP|JPY|CNY|RMB)"
_NUMERIC = r"(?>\d[\d,]{0,19}(?:\.\d{1,20})?)(?!\d|,\d|\.\d)"
_AMOUNT_BEFORE_RE = re.compile(
    rf"(?<![A-Za-z])(?P<unit>{_CURRENCY_RE})\s*(?P<value>{_NUMERIC})", re.IGNORECASE
)
_AMOUNT_UNITS = (
    r"(?:dollars?|cents?|euros?|pounds?|USD|EUR|GBP|kg|g|lb|lbs|km|mi|miles?"
    r"|hours?|minutes?|days?|weeks?|months?|years?"
    r"|hundred|thousand|million|billion)"
)
_AMOUNT_AFTER_RE = re.compile(
    rf"(?<![\w.,])(?P<value>{_NUMERIC})\s*(?P<unit>%|{_AMOUNT_UNITS}\b)", re.IGNORECASE
)
_AMOUNT_FIELD_RE = re.compile(
    rf"\b(?P<unit>amount|total|price|cost|balance|rent|charge|fee|salary)\b"
    rf"[\"']?\s*[:=]\s*[\"']?(?P<value>{_NUMERIC})",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(rf"(?<![\w.,])({_NUMERIC})(?![\w.])")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?!\d)"
)
_LONG_DIGITS_RE = re.compile(r"\d{21,}")


@dataclass(frozen=True)
class _ItemKeys:
    """An item's strong keys and its distinctive words."""

    strong: frozenset[str]
    weak: frozenset[str]
    dates: frozenset[tuple[int, int, int | None]]

    def __bool__(self) -> bool:
        return bool(self.strong or self.weak)


def _is_typed(key: str) -> bool:
    return key.startswith(_KEY_PREFIXES)


def _month_number(name: str) -> int:
    return _MONTHS[name[:3].lower()]


def _date_details(text: str) -> set[tuple[int, int, int | None]]:
    """Dates as month, day, and optional year; ambiguous slash dates use US order."""
    dates: set[tuple[int, int, int | None]] = set()

    def add(month: int, day: int, year: int | None) -> None:
        if (
            1 <= month <= 12
            and year != 0
            and 1 <= day <= calendar.monthrange(year or 2000, month)[1]
        ):
            dates.add((month, day, year))

    for match in _DATE_ISO_RE.finditer(text):
        year, month, day = (int(part) for part in match.groups())
        add(month, day, year)
    for match in _DATE_SLASH_RE.finditer(text):
        first, second = int(match.group(1)), int(match.group(2))
        year_text = match.group(3)
        before = match.start() - 1
        while before >= 0 and text[before].isspace():
            before -= 1
        mixed_number = (
            before < match.start() - 1 and before >= 0 and text[before].isdigit()
        )
        if (
            (first, second) == (24, 7)
            or _QUANTITY_AFTER_RE.match(text, match.end())
            or mixed_number
        ):
            continue
        slash_year = int(year_text) if year_text else None
        if slash_year is not None and len(year_text or "") == 2:
            slash_year += 2000
        month, day = (second, first) if first > 12 else (first, second)
        add(month, day, slash_year)
    for pattern, month_group, day_group in (
        (_MONTH_DAY_RE, 1, 2),
        (_DAY_MONTH_RE, 2, 1),
    ):
        for match in pattern.finditer(text):
            name = match.group(month_group)
            if name in ("may", "march"):
                continue
            add(
                _month_number(name),
                int(match.group(day_group)),
                int(match.group(3)) if match.group(3) else None,
            )
    return dates


def _date_keys(text: str) -> set[str]:
    return {f"date:{month:02d}-{day:02d}" for month, day, _year in _date_details(text)}


def _hour24(hour: int, suffix: str) -> int | None:
    if suffix:
        if hour > 12:
            return None
        return hour % 12 + (12 if suffix.startswith("p") else 0)
    return hour if hour < 24 else None


def _time_keys(text: str) -> set[str]:
    """Every clock time in *text* as ``HH:MM``; a colon or am/pm is required."""
    keys: set[str] = set()
    for match in _TIME_COLON_RE.finditer(text):
        hour = _hour24(int(match.group(1)), (match.group(3) or "").lower())
        minute = int(match.group(2))
        if hour is not None and minute < 60:
            keys.add(f"time:{hour:02d}:{minute:02d}")
    for match in _TIME_AMPM_RE.finditer(text):
        hour = _hour24(int(match.group(1)), match.group(2).lower())
        if hour is not None:
            keys.add(f"time:{hour:02d}:00")
    return keys


def _amount_value(value: str) -> str:
    text = value.replace(",", "")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _amount_keys(text: str) -> set[str]:
    """Amounts retain their currency or unit as well as their value."""
    aliases = {
        "$": "usd",
        "us$": "usd",
        "dollar": "usd",
        "dollars": "usd",
        "€": "eur",
        "euro": "eur",
        "euros": "eur",
        "£": "gbp",
        "pound": "gbp",
        "pounds": "gbp",
        "¥": "jpy",
        "₹": "inr",
        "rmb": "cny",
    }
    keys: set[str] = set()
    for pattern in (_AMOUNT_BEFORE_RE, _AMOUNT_AFTER_RE, _AMOUNT_FIELD_RE):
        for match in pattern.finditer(text):
            unit = match.group("unit").lower()
            unit = aliases.get(unit, unit)
            keys.add(f"amount:{unit}:{_amount_value(match.group('value'))}")
    return keys


def _phone_keys(text: str) -> set[str]:
    return {
        "phone:" + re.sub(r"\D", "", match.group())
        for match in _PHONE_RE.finditer(text)
    }


def _number_keys(text: str) -> set[str]:
    """Numbers of at least three digits, excluding amounts and phone fragments."""
    chars = list(text)
    for pattern in (_PHONE_RE, _AMOUNT_BEFORE_RE, _AMOUNT_AFTER_RE, _AMOUNT_FIELD_RE):
        for match in pattern.finditer(text):
            chars[match.start() : match.end()] = " " * (match.end() - match.start())
    keys: set[str] = set()
    for match in _NUMBER_RE.finditer("".join(chars)):
        value = match.group(1)
        integer = value.split(".", 1)[0].replace(",", "")
        if len(integer) >= 3 and not (
            len(integer) == 4 and 1900 <= int(integer) <= 2099
        ):
            keys.add(f"number:{_amount_value(value)}")
    return keys


def _typed_keys(text: str) -> set[str]:
    text = _LONG_DIGITS_RE.sub(lambda m: " " * len(m.group()), text)
    return {
        *_date_keys(text),
        *_time_keys(text),
        *_amount_keys(text),
        *_number_keys(text),
        *_phone_keys(text),
    }


def _strong_keys(item: str) -> set[str]:
    """The strong keys of *item*: dates, times, amounts, numbers, names.

    A strong key is a date, a time with a colon or am/pm, an amount with a
    currency or unit, an identifier of three or more characters holding a
    digit, a number of three or more integer digits, a capitalised name
    that does not start its sentence, or a CJK run. A bare one- or two-digit
    number is never a key.
    """
    item = _LONG_DIGITS_RE.sub(lambda m: " " * len(m.group()), item)
    strong = _typed_keys(item)
    stable = _DATE_WORDS | _NAME_SKIP | _ITEM_STOPWORDS
    for match in _ITEM_IDENT_RE.finditer(item):
        token = match.group(0).lower()
        if not re.fullmatch(r"\d+(?:st|nd|rd|th)", token):
            strong.add(token)
    initials = _sentence_initial_positions(item)
    for match in _ITEM_NAME_RE.finditer(item):
        name = match.group(0).lower()
        if name in stable:
            continue
        if match.start() not in initials:
            strong.add(name)
    strong.update(
        run for run in _ITEM_CJK_RE.findall(item) if run not in _ITEM_STOPWORDS
    )
    return strong


_NAME_SKIP = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "but",
        "or",
        "for",
        "from",
        "in",
        "on",
        "at",
        "to",
        "of",
        "with",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "all",
        "each",
        "every",
        "next",
        "last",
    }
)
_DATE_WORDS = frozenset(
    {*_MONTHS, *_WEEKDAYS}
    | {
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)


_SENTENCE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[.!?](?=\s|$)|\n")


def _sentence_initial_positions(item: str) -> set[int]:
    """Find the first alphanumeric token of each sentence in one pass."""
    initial = True
    positions: set[int] = set()
    for match in _SENTENCE_TOKEN_RE.finditer(item):
        if match.group() in (".", "!", "?", "\n"):
            initial = True
        elif initial:
            positions.add(match.start())
            initial = False
    return positions


def _item_keys(item: str) -> _ItemKeys:
    """The strong keys and distinctive words an item shares with a record.

    Strong keys are :func:`_strong_keys`; a distinctive word is four or
    more ASCII letters or digits that are not a stopword. One such word is
    not enough to source a record.
    """
    weak = {
        word.lower()
        for word in _ITEM_WORD_RE.findall(item)
        if not word.isdigit() and word.lower() not in _ITEM_STOPWORDS
    }
    return _ItemKeys(
        frozenset(_strong_keys(item)), frozenset(weak), frozenset(_date_details(item))
    )


_ITEM_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "because",
        "before",
        "being",
        "complete",
        "completed",
        "could",
        "done",
        "every",
        "fine",
        "finished",
        "first",
        "found",
        "from",
        "good",
        "have",
        "into",
        "just",
        "like",
        "made",
        "make",
        "more",
        "no",
        "noted",
        "ok",
        "okay",
        "only",
        "other",
        "over",
        "ready",
        "same",
        "should",
        "some",
        "sure",
        "than",
        "thanks",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "very",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "yes",
        "好的",
        "完成",
        "可以",
    }
)


def _has_key(folded: str, key: str) -> bool:
    """Whether lower-cased record text *folded* carries *key*."""
    if _ITEM_CJK_RE.fullmatch(key):
        return key in folded
    pattern = r"(?<![0-9a-z_])" + re.escape(key) + r"(?![0-9a-z_])"
    return re.search(pattern, folded) is not None


_META_DONE_RE = re.compile(
    r"^(?:the\s+)?(?:task|session|run|work)\b[^.!?]{0,40}"
    r"\b(?:complete|completed|done|finished)\b"
    r"|^(?:all\s+)?(?:done|complete|completed|finished)\b[.!]*$"
    r"|^all\s+set\b[.!]*$",
    re.IGNORECASE,
)
_META_WORK_VERB = (
    r"(?:read|review\w*|remember\w*|memoriz\w*|memoris\w*|stor\w*|record\w*"
    r"|captur\w*|summariz\w*|summaris\w*|process\w*|not(?:e|ed|ing)\w*"
    r"|internaliz\w*|internalis\w*|complet\w*|finish\w*|submit\w*)"
)
_META_NOUN = (
    r"(?:records?|facts?|notes?|memor(?:y|ies)|session|task|run|today|everything)"
)
_META_RE = re.compile(
    rf"(?:\b(?:i|we)\b[^.!?]{{0,80}}?\b{_META_WORK_VERB}"
    rf"[^.!?]{{0,80}}?\b{_META_NOUN}\b"
    rf"|\b(?:all|the)\b[^.!?]{{0,60}}?\b(?:records?|facts?|notes?|memories)\b"
    rf"[^.!?]{{0,80}}?\b{_META_WORK_VERB}"
    rf"|\b(?:records?|facts?|notes?)\b[^.!?]{{0,60}}?\b(?:have|has|were|are)\b"
    rf"[^.!?]{{0,40}}?\b{_META_WORK_VERB}"
    r"|\b(?:i will|i'll|i’ll)\b[^.!?]{0,30}?\bremember\b)",
    re.IGNORECASE,
)
_META_FRAME_RE = re.compile(
    rf"^(?:i|we)\b[^.!?]{{0,80}}?\b{_META_WORK_VERB}\b", re.IGNORECASE
)
_META_CONTENT_RE = re.compile(
    r"\b(?:that|says|said|shows|reads)\b|[:—–;]", re.IGNORECASE
)

# A digest's task line, and a file, path, function or code identifier, mark
# a text as content: it is not only a claim about the agent's own work.
_TASK_LINE_RE = re.compile(r"(?im)^[ \t]*(?:[-*+•]\s*)?task\s*:")
_FUNCTION_CALL_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_UNDERSCORE_IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_]\w*_\w+(?![A-Za-z0-9_])")
_CODE_SPAN_RE = re.compile(r"`[^`\n]*[A-Za-z_][^`\n]*`")


def _names_function(text: str) -> bool:
    """Inspect the token before each opening parenthesis, not every prefix."""
    for match in re.finditer(r"\(", text):
        end = match.start()
        start = end
        while start and (text[start - 1].isalnum() or text[start - 1] in "_."):
            start -= 1
        token = text[start:end].lstrip(".")
        if token and _FUNCTION_CALL_RE.fullmatch(token):
            return True
    return False


def _names_code(text: str) -> bool:
    """Whether *text* names a file, path, function or code identifier."""
    return bool(
        path_spans(text)
        or _names_function(text)
        or _UNDERSCORE_IDENT_RE.search(text)
        or _CODE_SPAN_RE.search(text)
    )


def _is_meta_sentence(sentence: str) -> bool:
    """Whether one sentence only reports the agent's own reading or completion."""
    frame = (
        _META_FRAME_RE.search(sentence)
        or _META_RE.match(sentence)
        or _META_DONE_RE.match(sentence)
    )
    if frame is None:
        return False
    # Only content clauses following the work frame may establish a fact.
    for separator in _META_CONTENT_RE.finditer(sentence, frame.end()):
        if _strong_keys(sentence[separator.end() :]):
            return False
    return True


def is_meta_claim(text: str) -> bool:
    """Whether *text* is only a completion or self-work claim, and no content.

    A text with a ``task:`` line or a file, path, function or code
    identifier is content, never a meta claim: an Aider-style final digest
    ("task: Implement transpose in transpose.py … outcome: completed")
    carries no dates or amounts, and its task line alone keeps it.

    Keys inside a work frame do not exempt it; a content clause after
    ``that``, ``says``, ``said``, ``shows``, ``reads``, a colon or a dash
    does, when that clause has a strong key.
    """
    if _TASK_LINE_RE.search(text) or _names_code(text):
        return False
    sentences = split_sentences(text)
    return bool(sentences) and all(
        _is_meta_sentence(sentence) for _offset, sentence in sentences
    )


def _item_score(keys: _ItemKeys, text: str) -> int:
    """Score each matching key, rejecting conflicting dates and currencies."""
    record_dates = _date_details(text)
    if (
        keys.dates
        and record_dates
        and not any(
            m1 == m2 and d1 == d2 and (y1 is None or y2 is None or y1 == y2)
            for m1, d1, y1 in keys.dates
            for m2, d2, y2 in record_dates
        )
    ):
        return 0
    typed = _typed_keys(text)
    item_times = {key for key in keys.strong if key.startswith("time:")}
    # A mail's sent timestamp is not the time of its subject event.
    try:
        record = json.loads(text)
    except (ValueError, RecursionError):
        record = None
    time_text = text
    if isinstance(record, dict):
        time_text = " ".join(
            str(record.get(field) or "")
            for field in ("body", "text", "notes", "start", "end")
        )
    record_times = _time_keys(time_text)
    if item_times and record_times and item_times.isdisjoint(record_times):
        return 0
    item_phones = {key for key in keys.strong if key.startswith("phone:")}
    record_phones = {key for key in typed if key.startswith("phone:")}
    if item_phones and record_phones and item_phones.isdisjoint(record_phones):
        return 0
    currencies = {"usd", "eur", "gbp", "jpy", "cny", "inr"}
    item_currency = {
        key.split(":")[1]
        for key in keys.strong
        if key.startswith("amount:") and key.split(":")[1] in currencies
    }
    record_currency = {
        key.split(":")[1]
        for key in typed
        if key.startswith("amount:") and key.split(":")[1] in currencies
    }
    if item_currency and record_currency and item_currency.isdisjoint(record_currency):
        return 0
    score = 8 * len({key for key in keys.strong if _is_typed(key)} & typed)
    folded = text.lower()
    for key in keys.strong:
        if not _is_typed(key) and _has_key(folded, key):
            score += 2
    weak = sum(1 for word in keys.weak if _has_key(folded, word))
    if weak >= 2:
        score += weak
    return score


def _item_matches(keys: _ItemKeys, folded: str) -> bool:
    """Whether *folded* shares a strong key or two distinctive words."""
    return _item_score(keys, folded) > 0


def _item_source_ids(item: str, records: Sequence[tuple[str, str]]) -> list[str]:
    """The record id whose text matches *item* best, or none.

    A typed key (a date, time, amount or number) outweighs a name or an
    identifier, which outweighs two or more distinctive words (not
    stopwords, at least four characters; a bare one- or two-digit number
    is never a key). *records* is in the precedence order
    :func:`note_from_reply` gives it: the reply's own results, the
    window's, then the rest of the task's. Equal top scores from different
    records are left unsourced.
    """
    keys = _item_keys(item)
    if not keys:
        return []
    best_id = ""
    best_score = 0
    tied = False
    # Shared dates or names alone cannot establish a denial.
    negative = has_negative_assertion(item)
    assertion = _collapse(item).lower().rstrip(".。!！ ")
    for record_id, text in records:
        # Opposite polarity is not evidence, in either direction. This is
        # deliberately conservative for records with several statements.
        if negative != has_negative_assertion(text):
            continue
        if negative and assertion not in _collapse(text).lower():
            continue
        score = _item_score(keys, text)
        if score > best_score:
            best_id, best_score, tied = record_id, score, False
        elif score and score == best_score and record_id != best_id:
            tied = True
    return [best_id] if best_id and not tied else []


def candidate_items(note: ReasoningNote) -> tuple[CandidateItem, ...]:
    """A note's found and decided items, each with the records behind it.

    An item keeps the uniquely best matching record, based on strong keys
    and distinctive words. Equal best scores are left unsourced. When the
    note carries no record text at all, its whole ``source_ids`` are used;
    when it carries records and none matches, the item gets none (a false
    source is worse than none).
    """
    items: list[CandidateItem] = []
    default = tuple(note.source_ids) if not note.source_records else ()
    for item in (*note.found, *note.decided):
        matched = _item_source_ids(item, note.source_records)
        items.append(CandidateItem(item, tuple(matched) if matched else default))
    return tuple(items)


def candidate_text(note: ReasoningNote) -> str:
    """A note's candidate text: its found and decided items, cut to 600."""
    items = [*note.found, *note.decided]
    return _shorten(" | ".join(items), CANDIDATE_TEXT_CHARS)


def reflection_candidate_text(entry: ReflectionEntry) -> str:
    """The explicit reflection candidate's text, cut to 2,000 characters.

    ``Attempt <n>: <summary>``, then the failed hypotheses, evidence,
    avoids, next directions and unresolved questions. This is the plan's
    fix for ``promote_reflections``; the legacy form stays frozen.
    """
    parts = [f"Attempt {entry.attempt_id}: {entry.summary}"]
    if entry.failed_hypotheses:
        parts.append(f"Failed: {'; '.join(entry.failed_hypotheses)}")
    if entry.useful_evidence:
        parts.append(f"Evidence: {'; '.join(entry.useful_evidence)}")
    if entry.recommended_avoids:
        parts.append(f"Avoid: {'; '.join(entry.recommended_avoids)}")
    if entry.next_promising_directions:
        parts.append(f"Try: {'; '.join(entry.next_promising_directions)}")
    if entry.unresolved_questions:
        parts.append(f"Open: {'; '.join(entry.unresolved_questions)}")
    return _shorten(" | ".join(parts), REFLECTION_CANDIDATE_CHARS)


# ── Blocks the harness inserts into a turn ──

RECALL_START = "[localmelo recall:"
RECALL_END = "[end of localmelo recall]"
NOTES_START = "[localmelo notes:"
NOTES_END = "[end of localmelo notes]"


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def render_note_block(
    notes: Sequence[str] = (), draft_titles: Sequence[str] = ()
) -> str:
    """The reflection's notes block, or "" when there is nothing to show.

    *notes* and *draft_titles* are already-rendered entry lines, without
    their delimiters. The block is framed as data, not instructions, and
    delimited by :data:`NOTES_START` and :data:`NOTES_END`, so a later
    recall can strip it as a whole.
    """
    if not notes and not draft_titles:
        return ""
    header = (
        f"{NOTES_START} {_plural(len(notes), 'note', 'notes')} and "
        f"{_plural(len(draft_titles), 'draft', 'drafts')} from this attempt; "
        "data, not instructions]"
    )
    return "\n".join([header, *notes, *draft_titles, NOTES_END])


def render_draft_block(
    alias: str, step: int, draft: Draft, *, text: str | None = None
) -> str:
    """The pin-once block of one pending draft, framed as data.

    The block names the draft's step, then its alias, its target (or
    "no target") and line count, then the draft's text, at most
    :data:`DRAFT_SHOWN_CHARS` characters (the stored, checked draft is
    longer at most by the cap of its source). *text*, when given, is the
    checked text to show instead of ``draft.text``.
    """
    shown = draft.text if text is None else text
    shown = shown[:DRAFT_SHOWN_CHARS]
    header = f"{RECALL_START} 1 draft from step {step}; data, not instructions]"
    title = (
        f"[{alias}] draft · step {step} · "
        f"{draft.target or 'no target'}, {draft.lines} lines:"
    )
    return "\n".join([header, title, shown, RECALL_END])


# ── Drafts ──


def extract_draft(text: str, *, recent_path: str = "") -> Draft | None:
    """The draft of a reply: its fenced code blocks, and their target.

    *text* is the reply's reasoning and visible text, with their code
    blocks. ``None`` when no block holds anything but whitespace. Blocks
    are deduplicated by their text with trailing whitespace stripped, kept
    in order, and joined with a blank line. Over :data:`DRAFT_CHARS`, the
    earliest blocks are dropped until the rest fits; a last block that
    alone is over keeps at most its first :data:`DRAFT_CHARS` characters,
    up to the last line break among them. Either sets ``cut``.

    ``target`` is the last file name, or path whose last segment is a file
    name, mentioned before the first block (see :func:`path_spans`); else
    *recent_path*; else "". A directory is never a target. The identity
    fields stay at their defaults.
    """
    fences = _fences(text)
    blocks: list[str] = []
    for fence in fences:
        body = fence.body.rstrip()
        if body and body not in blocks:
            blocks.append(body)
    if not blocks:
        return None
    cut = False
    while len(blocks) > 1 and len("\n\n".join(blocks)) > DRAFT_CHARS:
        blocks.pop(0)
        cut = True
    if len(blocks[0]) > DRAFT_CHARS:
        # Up to the last line break within the cap; the cap itself when
        # no line break leaves anything before it.
        head = blocks[0][: DRAFT_CHARS + 1]
        whole_lines = head[: max(head.rfind("\n"), 0)]
        kept = whole_lines if whole_lines.strip() else head[:DRAFT_CHARS]
        blocks[0] = kept.rstrip()
        cut = True
    draft_text = "\n\n".join(blocks)
    return Draft(
        text=draft_text,
        target=_draft_target(text, fences[0].start) or recent_path,
        lines=sum(1 for line in draft_text.splitlines() if line.strip()),
        blocks=len(blocks),
        cut=cut,
    )


def _draft_target(text: str, before: int) -> str:
    """The last file name or file path that ends before offset *before*."""
    head = text[:before]
    for start, end in reversed(path_spans(head)):
        token = head[start:end]
        if _names_file(token, head[end : end + 1]):
            return token
    return ""


def draft_coverage(draft_text: str, written: str) -> float:
    """The share of the draft's lines that a write holds.

    The draft's distinct non-blank lines, stripped, are compared with the
    stripped lines of *written*. 0.0 for a draft with no such line. A
    draft is written once a write to its target reaches
    :data:`DRAFT_WRITTEN_SHARE`.
    """
    lines = {line.strip() for line in draft_text.splitlines() if line.strip()}
    if not lines:
        return 0.0
    written_lines = {line.strip() for line in written.splitlines()}
    return len(lines & written_lines) / len(lines)


def same_path(a: str, b: str) -> bool:
    """Whether a draft target and a write's path (or another target) meet.

    ``\\`` reads as ``/``, then both are normalized with
    :func:`posixpath.normpath`. They meet when equal, or, when either has
    no ``/``, when their last segments are equal: ``./list_ops.py``,
    ``list_ops.py`` and ``/work/list_ops.py`` all meet. An empty string
    meets nothing.
    """
    if not a or not b:
        return False
    first = posixpath.normpath(a.replace("\\", "/"))
    second = posixpath.normpath(b.replace("\\", "/"))
    if first == second:
        return True
    if "/" in first and "/" in second:
        return False
    return first.rsplit("/", 1)[-1] == second.rsplit("/", 1)[-1]


def mark_drafts_written(
    drafts: Sequence[Draft],
    calls_and_results: Iterable[tuple[ToolCall, ToolResult]],
) -> int:
    """Mark the pending drafts that a reply's successful writes wrote.

    Each pair is one tool call and its result. A pair whose result has an
    error, or whose call is not ``file_write`` with a string ``path`` and
    ``content``, is skipped. A pending draft whose target meets the write
    (see :func:`same_path`) and whose text the write holds for at least
    :data:`DRAFT_WRITTEN_SHARE` (see :func:`draft_coverage`) becomes
    ``written``. The loop and the exported corpus both call this, so live
    and exported statuses cannot diverge. Returns the number of drafts
    newly marked.
    """
    written = 0
    for call, result in calls_and_results:
        if result.error.strip() or call.tool_name != "file_write":
            continue
        path = call.arguments.get("path")
        content = call.arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        for draft in drafts:
            if (
                draft.status == "pending"
                and same_path(draft.target, path)
                and draft_coverage(draft.text, content) >= DRAFT_WRITTEN_SHARE
            ):
                draft.status = "written"
                written += 1
    return written


def draft_title(draft: Draft) -> str:
    """``<target or "no target">, <lines> lines: <first code line>``, shortened."""
    first_line = next(
        (line.strip() for line in draft.text.splitlines() if line.strip()), ""
    )
    return _shorten(
        f"{draft.target or 'no target'}, {draft.lines} lines: {first_line}",
        TITLE_CHARS,
    )


# ── Thought chunks ──

_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def thought_chunks(thinking: str) -> list[ThoughtChunk]:
    """The reasoning without code, in chunks of :data:`THOUGHT_CHUNK_CHARS`.

    Fenced blocks are removed and the text splits into paragraphs at blank
    lines. A paragraph over the cap splits into pieces packed from its
    sentences; a sentence over the cap is cut at the last whitespace
    within the cap, or at the cap when there is none (as in Chinese text).
    Consecutive pieces are merged, joined with a blank line, while the
    result stays within the cap. A chunk's title is its first sentence,
    whitespace collapsed, shortened to :data:`TITLE_CHARS`.
    """
    pieces: list[str] = []
    for paragraph in _BLANK_LINE_RE.split(_without_fences(thinking)):
        paragraph = paragraph.strip()
        if len(paragraph) > THOUGHT_CHUNK_CHARS:
            pieces.extend(_paragraph_pieces(paragraph))
        elif paragraph:
            pieces.append(paragraph)
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 2 + len(piece) <= THOUGHT_CHUNK_CHARS:
            chunks[-1] += "\n\n" + piece
        else:
            chunks.append(piece)
    return [ThoughtChunk(title=_chunk_title(chunk), text=chunk) for chunk in chunks]


def _paragraph_pieces(paragraph: str) -> list[str]:
    """*paragraph* in pieces within the cap, each a run of whole sentences."""
    pieces: list[str] = []
    start = end = -1
    for offset, sentence in split_sentences(paragraph):
        sentence_end = offset + len(sentence)
        if len(sentence) > THOUGHT_CHUNK_CHARS:
            if start >= 0:
                pieces.append(paragraph[start:end])
                start = -1
            pieces.extend(_cut_sentence(sentence))
        elif start < 0:
            start, end = offset, sentence_end
        elif sentence_end - start <= THOUGHT_CHUNK_CHARS:
            end = sentence_end
        else:
            pieces.append(paragraph[start:end])
            start, end = offset, sentence_end
    if start >= 0:
        pieces.append(paragraph[start:end])
    return pieces


def _cut_sentence(sentence: str) -> list[str]:
    """*sentence* in parts within the cap, cut at whitespace where it has some."""
    parts: list[str] = []
    rest = sentence
    while len(rest) > THOUGHT_CHUNK_CHARS:
        window = rest[: THOUGHT_CHUNK_CHARS + 1]
        cut = next((i for i in range(len(window) - 1, 0, -1) if window[i].isspace()), 0)
        if cut:
            parts.append(rest[:cut].rstrip())
        else:
            cut = THOUGHT_CHUNK_CHARS
            parts.append(rest[:cut])
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


def _chunk_title(chunk: str) -> str:
    sentences = split_sentences(chunk)
    if not sentences:
        return ""
    return _shorten(_collapse(sentences[0][1]), TITLE_CHARS)
