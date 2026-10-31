"""Where other coding assistants keep what they remember, and how those files are read.

Everything resolves from *home* and stays inside it: a path that resolves
outside (a symlink, an absolute ``@import``) is not read, and neither is a file
over :data:`MAX_FILE_BYTES` or one that looks like credentials, a history or a
transcript. A file is read once, by real path and by content, however many
profiles import it.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path

from localmelo.melo.memory.notes import split_sentences

MAX_FILE_BYTES = 256 * 1024
MAX_IMPORT_DEPTH = 4
MAX_STATEMENT_CHARS = 300
# Where a Claude config folder lives under HOME (globs); the folder named by
# CLAUDE_CONFIG_DIR is added when it is inside HOME.
CLAUDE_ROOTS = (".claude", ".cc-switch/claude-accounts/*/config")
# Single instruction files: (harness, path under HOME).
FILES = (("codex", ".codex/AGENTS.md"), ("gemini", ".gemini/GEMINI.md"))
_NEVER = re.compile(
    r"auth|credential|secret|token|password|keychain|\.env|history|transcript"
    r"|\.jsonl$|\.pem$|\.key$|\.ssh|id_(?:rsa|ed25519)",
    re.IGNORECASE,
)
_IMPORT = re.compile(r"(?<!\S)@((?:~|\.{1,2}|[\w.-]+)?/\S*|[\w.-]+\.\w+)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^(?:[-*+•]|\d+[.)])\s+")
_BLOCK = re.compile(r"[>|][+-]?")  # a YAML block scalar's marker, not a value
# A line that never continues the one above: a "Why:" note, or a link or URL alone.
_OWN_LINE = re.compile(
    r"(?:why|how to apply)\s*[:：]|(?:\[[^\]]*\]\([^)]*\)|https?://\S+)\s*$",
    re.IGNORECASE,
)
_ABBREVIATIONS = frozenset(
    (
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "cf.",
        "approx.",
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
    )
)
_NOISE = re.compile(
    r"^(?:\[[^\]]*\]\([^)]*\)|https?://\S+|(?:why|how to apply)\s*[:：])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Source:
    """One place to read: *kind* is ``md``, ``memory`` (a folder of notes),
    ``hermes`` (``§`` entries) or ``sqlite``."""

    id: str
    harness: str
    path: str
    scope: str = "user"
    workspace: str = ""
    kind: str = "md"


@dataclass(frozen=True)
class Statement:
    """One line of a source with its heading path; *kind* is a note file's ``type``."""

    text: str
    path: str
    line: int
    heading: str
    kind: str = ""


def readable(
    path: Path, home: Path, *, dir: bool = False, limit: int | None = None
) -> Path | None:
    """The real path of *path* when it may be read: inside *home*, a plain file
    (or a folder with *dir*) of at most *limit* bytes, not credentials."""
    try:
        real = path.resolve(strict=True)
        inside = real.is_relative_to(home)
        plain = real.is_dir() if dir else real.is_file()
        small = dir or limit is None or real.stat().st_size <= limit
    except (OSError, RuntimeError):
        return None
    if not (inside and plain and small) or _NEVER.search(str(real.relative_to(home))):
        return None
    return real


def _flat(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", name)


def decode_workspace(name: str, home: Path) -> str:
    """The folder under *home* that a Claude project folder name stands for, or ``""``.

    ``-Users-me-Dev-app`` is ``/Users/me/Dev/app``: every character that is not
    a letter or digit became ``-``, so each part is matched against the real
    folders, never split on ``-``.
    """
    prefix = _flat(str(home)) + "-"
    if not name.startswith(prefix):
        return ""
    rest, base = name[len(prefix) :], home
    while rest:
        try:
            kids = sorted(c for c in base.iterdir() if c.is_dir())
        except OSError:
            return ""
        match = next(
            (
                c
                for c in kids
                if rest == _flat(c.name) or rest.startswith(_flat(c.name) + "-")
            ),
            None,
        )
        if match is None:
            return ""
        base, rest = match, rest[len(_flat(match.name)) + 1 :]
    return str(base)


def discover(home: Path, env: Mapping[str, str] | None = None) -> list[Source]:
    """The sources under *home*, user-level ones first in a fixed order."""
    env = os.environ if env is None else env
    found: list[Source] = []

    def add(harness: str, path: Path, kind: str = "md", workspace: str = "") -> None:
        real = readable(
            path,
            home,
            dir=kind == "memory",
            limit=None if kind == "sqlite" else MAX_FILE_BYTES,
        )
        if real is not None:
            key = hashlib.sha1(str(real).encode()).hexdigest()[:8]
            found.append(
                Source(f"{harness}-{key}", harness, str(real), "user", workspace, kind)
            )

    roots = [Path(env["CLAUDE_CONFIG_DIR"])] if env.get("CLAUDE_CONFIG_DIR") else []
    for pattern in CLAUDE_ROOTS:
        roots += sorted(home.glob(pattern))
    for root in dict.fromkeys(roots):
        add("claude", root / "CLAUDE.md")
        for folder in sorted(root.glob("projects/*/memory")):
            add("claude", folder, "memory", decode_workspace(folder.parent.name, home))
    for harness, rel in FILES:
        add(harness, home / rel)
    for db in sorted((home / ".codex").glob("memories_*.sqlite")):
        add("codex", db, "sqlite")
    base = Path(env.get("HERMES_HOME") or home / ".hermes")
    for name in ("USER.md", "MEMORY.md"):
        add("hermes", base / "memories" / name, "hermes")
    return list({s.id: s for s in found}.values())


def _split(text: str) -> list[str]:
    """*text*, or its sentences when it is longer than a note can be."""
    if len(text) <= MAX_STATEMENT_CHARS:
        return [text] if text else []
    return [s for _, s in split_sentences(text)]


def _ended(text: str) -> bool:
    """Whether *text* ends a sentence, so the next line starts a new one."""
    last = text.rsplit(None, 1)[-1].lower()
    return (
        text.endswith((".", "!", "?", "。", "！", "？")) and last not in _ABBREVIATIONS
    )


def _emit(
    pending: tuple[int, str] | None, stack: list[tuple[int, str]], path: str, kind: str
) -> Iterator[Statement]:
    """The statements of one bullet or paragraph (*pending*: its first line's number and text)."""
    if pending is None:
        return
    number, body = pending
    if len(body) < 4 or body.endswith((":", "：")) or _NOISE.match(body):
        return
    heading = " › ".join(h for _, h in stack)
    for sentence in _split(body):
        yield Statement(sentence, path, number, heading, kind)


def markdown(
    text: str, path: str, kind: str = "", title: str = ""
) -> Iterator[Statement]:
    """The bullets and sentences of Markdown, each with its heading path.

    The lines of a bullet or paragraph that is wrapped over several lines are one
    statement, numbered by its first line (a line that ends a sentence ends it,
    so lines that each hold a sentence stay apart). Code, tables, comments, headings alone,
    lead-ins ending in a colon, links and URLs alone, and "Why:" notes are not
    statements.
    """
    stack: list[tuple[int, str]] = [(0, title)] if title else []
    fenced = False
    pending: tuple[int, str] | None = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        fence = line.startswith(("```", "~~~"))
        if fence or fenced or not line or line.startswith(("<!--", "|")):
            yield from _emit(pending, stack, path, kind)
            pending = None
            fenced = not fenced if fence else fenced
        elif head := _HEADING.match(line):
            yield from _emit(pending, stack, path, kind)
            pending = None
            level = len(head[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, head[2]))
        else:
            body = _BULLET.sub("", line.lstrip("> ")).replace("**", "").strip()
            if (
                pending is not None
                and not _ended(pending[1])
                and not _BULLET.match(line)
                and not line.startswith(">")
                and not _OWN_LINE.match(body)
            ):
                pending = (pending[0], f"{pending[1]} {body}")
            else:
                yield from _emit(pending, stack, path, kind)
                pending = (number, body)
    yield from _emit(pending, stack, path, kind)


def _unquote(value: str) -> str:
    """*value* without the quotes around it."""
    quoted = len(value) > 1 and value[0] == value[-1] and value[0] in "\"'"
    return value[1:-1].strip() if quoted else value


def _file(
    path: Path, src: Source, home: Path, seen: set[str], depth: int
) -> Iterator[Statement]:
    real = readable(path, home, limit=MAX_FILE_BYTES)
    if real is None:
        return
    text = real.read_text("utf-8", errors="replace")
    digest = hashlib.sha1(text.encode()).hexdigest()
    if str(real) in seen or digest in seen or "\0" in text:
        return
    seen.update((str(real), digest))
    if src.kind == "hermes":
        pos = 0
        for entry in text.split("§"):
            lead = len(entry) - len(entry.lstrip())
            line = text.count("\n", 0, pos + lead) + 1
            pos += len(entry) + 1
            for sentence in _split(" ".join(entry.split())):
                yield Statement(
                    sentence,
                    str(real),
                    line,
                    real.stem,
                    "user" if real.stem == "USER" else "",
                )
        return
    fields: dict[str, str] = {}
    rows: dict[str, int] = {}
    if front := re.match(r"---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.DOTALL):
        for number, row in enumerate(front[1].splitlines(), 2):
            key, _, value = row.partition(":")
            if value := _unquote(value.strip()):
                fields[key.strip()], rows[key.strip()] = value, number
        text = "\n" * front[0].count("\n") + text[front.end() :]
    name, kind = fields.get("name", ""), fields.get("type", "")
    gist = fields.get("description", "")
    # The one-line summary a note file was written around comes first.
    summary = (
        markdown("\n" * (rows["description"] - 1) + gist, str(real), kind, name)
        if src.kind == "memory" and gist and not _BLOCK.fullmatch(gist)
        else iter(())
    )
    for st in chain(summary, markdown(text, str(real), kind, name)):
        rest = st.text
        for token in (t.rstrip(".,;:)。") for t in _IMPORT.findall(st.text)):
            target = Path(
                token.replace("~", str(home), 1) if token.startswith("~") else token
            )
            target = target if target.is_absolute() else real.parent / target
            if readable(target, home, limit=MAX_FILE_BYTES) is not None:
                rest = rest.replace("@" + token, "")
                if depth < MAX_IMPORT_DEPTH:
                    yield from _file(target, src, home, seen, depth + 1)
        rest = rest.strip(" -*·:：")
        if len(rest) >= 4 and not re.fullmatch(r"(?:@\S+\s*)+", rest):
            yield replace(st, text=rest)


def _sqlite(path: Path, home: Path) -> Iterator[Statement]:
    """The ``raw_memory`` of a Codex memories file, opened read-only and immutable."""
    real = readable(path, home)
    if real is None:
        return
    try:
        db = sqlite3.connect(f"{real.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            rows = db.execute(
                "SELECT raw_memory FROM stage1_outputs LIMIT 200"
            ).fetchall()
        finally:
            db.close()
    except sqlite3.Error:
        return
    for number, (raw,) in enumerate(rows, 1):
        for st in markdown(str(raw or "")[:MAX_FILE_BYTES], str(real)):
            yield replace(
                st, heading=" › ".join(filter(None, (f"memory {number}", st.heading)))
            )


def read_source(src: Source, home: Path, seen: set[str]) -> Iterator[Statement]:
    """The statements of *src*, skipping files *seen* already (real path or content).

    A note file of a memory folder starts with its front matter ``description``.
    """
    path = Path(src.path)
    if src.kind == "sqlite":
        yield from _sqlite(path, home)
    else:
        for file in sorted(path.glob("*.md")) if src.kind == "memory" else [path]:
            yield from _file(file, src, home, seen, 0)
