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

import hashlib
import json
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
NOTE_MAX_FOUND = 4  # items per field, not characters
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
    # <= 4 x 200: what the model says it established
    found: list[str] = field(default_factory=list)
    decided: list[str] = field(default_factory=list)  # <= 3 x 200
    # <= 3 x 200: "open: ", "confirmed: ", "refuted: "
    hypotheses: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)  # <= 3 x 200
    # <= 5 x 160, from the executor only, one per call
    observed: list[str] = field(default_factory=list)
    outcome: str = ""  # <= 160: "ok" | "exit 1: 24 failed" | "Error: <first line>"
    drafts: list[str] = field(default_factory=list)  # draft ids
    reasoning_chars: int = 0
    # record ids from this turn's JSON tool results (section 5)
    source_ids: list[str] = field(default_factory=list)
    # (record id, up to 500 chars of its text), for an item's source ids
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
    "found": NOTE_MAX_FOUND,
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
    dropped, and keep their first 4, 3, 3 and 3 items, each shortened to
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
    r"[。！？]|[!?](?=\s)|(?a:(?<!\be\.g)(?<!\bi\.e))\.(?=\s)", re.IGNORECASE
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


def _observed(call: ToolCall, result: ToolResult) -> str:
    """The observed line of one call and its result."""
    name = call.tool_name
    error = _error(result)
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


def _source_ids(pairs: Sequence[tuple[ToolCall, ToolResult]]) -> list[str]:
    return [record_id for record_id, _text in _source_records(pairs)]


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
    - ``found``: the last 4 sentences with a finding marker that start in
      the last :data:`FOUND_WINDOW_CHARS` of the text.

    In both cases ``observed``, ``outcome`` and ``source_ids`` come from
    the calls and their results, paired by index; ``reasoning_chars`` is
    ``len(thinking)``; and ``remember``, when the model's note gives none,
    is whether the note has a ``found`` or ``decided`` item
    (``remember_by = "heuristic"``).

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
            note = _heuristic_note(text, tool_calls)
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
            sentence for _offset, sentence in sentences[:NOTE_MAX_FOUND]
        )

    pairs = list(zip(tool_calls, tool_results, strict=False))
    note.observed = [
        _observed(call, result) for call, result in pairs[:NOTE_MAX_OBSERVED]
    ]
    note.outcome = _outcome(pairs)
    note.source_ids = _source_ids(pairs)
    note.source_records = _source_records(pairs)
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


def _heuristic_note(text: str, tool_calls: Sequence[ToolCall]) -> ReasoningNote:
    """The content fields that the heuristic reads off *text*."""
    sentences = split_sentences(text)
    taken: set[int] = set()

    def take(test: Callable[[str], bool], limit: int, start: float = 0) -> list[str]:
        """The last *limit* sentences left that pass *test* and start at *start*."""
        chosen = [
            i
            for i, (offset, sentence) in enumerate(sentences)
            if i not in taken and offset >= start and test(sentence)
        ][-limit:]
        taken.update(chosen)
        return [sentences[i][1] for i in chosen]

    intent = [i for i, (_, sentence) in enumerate(sentences) if _has_intent(sentence)]
    preferred = [
        i for i in intent if path_spans(sentences[i][1]) or _has_action(sentences[i][1])
    ]
    preferred_set = set(preferred)
    others = [i for i in intent if i not in preferred_set]
    chosen = preferred[-2:]
    if len(chosen) < 2:
        chosen += others[len(chosen) - 2 :]
    chosen.sort()
    taken.update(chosen)

    note = ReasoningNote(source="heuristic")
    note.plan = _plan(" ".join(sentences[i][1] for i in chosen), tool_calls)
    tail = len(text) - DECIDED_TAIL_SHARE * len(text)
    note.decided = _items(take(_has_decision, NOTE_MAX_DECIDED, tail))
    note.open_questions = _items(take(_is_open, NOTE_MAX_OPEN))
    note.hypotheses = _items(
        f"open: {sentence}" for sentence in take(_has_hedge, NOTE_MAX_HYPOTHESES)
    )
    window = len(text) - FOUND_WINDOW_CHARS
    note.found = _items(take(_has_finding, NOTE_MAX_FOUND, window))
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


def digest_note(
    query: str,
    outcome: str = "",
    notes: Sequence[ReasoningNote] = (),
    *,
    task_id: str = "",
    attempt: int = 0,
    step: int = 0,
) -> ReasoningNote:
    """The digest of one attempt: its rehearsed notes and confirmed outcomes.

    Only *notes* whose stage is ``"rehearsed"`` contribute: their ``found``
    and ``decided`` items become ``found`` (deduplicated by normalized
    text, in note order), their ``observed`` lines become ``observed``
    (one per path or command, the first kept), and their
    ``open_questions`` become ``open_questions`` (deduplicated). ``plan``
    is *query*, shortened to :data:`DIGEST_TASK_CHARS`; ``outcome`` is the
    attempt's failure type or the run's status. ``source_ids`` is the
    union of the contributing notes', in order. The digest has
    ``kind="digest"``, ``remember=True`` and ``remember_by="rule"``. Its
    identity fields stay at their defaults; the caller sets them.
    """
    facts: list[str] = []
    seen_facts: set[str] = set()
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
        for item in (*note.found, *note.decided):
            key = _normalized_item(item)
            if key and key not in seen_facts:
                seen_facts.add(key)
                facts.append(item)
        for line in note.observed:
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
    return ReasoningNote(
        task_id=task_id,
        step=step,
        attempt=attempt,
        kind="digest",
        source="heuristic",
        plan=_shorten(_collapse(query), DIGEST_TASK_CHARS),
        found=facts,
        open_questions=open_questions,
        observed=observed,
        outcome=_shorten(_collapse(outcome), NOTE_OBSERVED_CHARS),
        reasoning_chars=reasoning_chars,
        source_ids=source_ids,
        remember=True,
        remember_by="rule",
        stage="fresh",
    )


def render_digest(note: ReasoningNote, *, limit: int = DIGEST_CHARS) -> str:
    """*note*'s digest text, at most *limit* characters; the note is not changed.

    Lines ``task: …``, ``outcome: …``, ``facts: …``, ``observed: …`` and
    ``open: …``, each left out when empty, list items joined with ``; ``.
    Over *limit*, items are dropped from the end of ``open``, then
    ``observed``, then ``found``, then ``plan`` is shortened, then the
    text is cut at *limit*.
    """
    plan = note.plan
    open_questions = list(note.open_questions)
    observed = list(note.observed)
    found = list(note.found)

    def render() -> str:
        lines: list[str] = []
        if plan:
            lines.append(f"task: {plan}")
        if note.outcome:
            lines.append(f"outcome: {note.outcome}")
        if found:
            lines.append("facts: " + "; ".join(found))
        if observed:
            lines.append("observed: " + "; ".join(observed))
        if open_questions:
            lines.append("open: " + "; ".join(open_questions))
        return "\n".join(lines)

    text = render()
    for items in (open_questions, observed, found):
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

_ITEM_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_ITEM_DATE_RE = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")
_ITEM_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ITEM_WORD_RE = re.compile(r"[A-Za-z0-9_]{4,}")


def _item_keys(item: str) -> set[str]:
    """The distinctive strings an item shares with a record's text."""
    keys: set[str] = set()
    keys.update(_ITEM_NUMBER_RE.findall(item))
    keys.update(_ITEM_DATE_RE.findall(item))
    keys.update(name.lower() for name in _ITEM_NAME_RE.findall(item))
    keys.update(
        word.lower()
        for word in _ITEM_WORD_RE.findall(item)
        if word.lower() not in _ITEM_STOPWORDS
    )
    keys.update(
        run for run in re.findall(r"[\u4e00-\u9fff]{2,}", item) if len(run) >= 2
    )
    return keys


_ITEM_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "because",
        "before",
        "being",
        "could",
        "every",
        "first",
        "found",
        "from",
        "have",
        "into",
        "just",
        "like",
        "made",
        "make",
        "more",
        "only",
        "other",
        "over",
        "same",
        "should",
        "some",
        "than",
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
    }
)


def _item_source_ids(item: str, records: Sequence[tuple[str, str]]) -> list[str]:
    """The record ids whose text shares a number, date, name or word with *item*."""
    keys = _item_keys(item)
    if not keys:
        return []
    ids: list[str] = []
    for record_id, text in records:
        folded = text.lower()
        if any(key in folded for key in keys) and record_id not in ids:
            ids.append(record_id)
    return ids


def candidate_items(note: ReasoningNote) -> tuple[CandidateItem, ...]:
    """A note's found and decided items, each with the records behind it.

    An item keeps the ids of the records whose text shares a number, a
    date, a capitalised name or a distinctive word with it; when none
    matches, the note's whole ``source_ids`` are used.
    """
    items: list[CandidateItem] = []
    default = tuple(note.source_ids)
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
