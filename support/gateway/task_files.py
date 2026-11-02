"""The files a task wrote, and the files its request asked it to write (#440).

- :func:`written_files`: the files a task's file tools wrote or edited, in
  every run of it, that are there now, named as the person reads them (a
  path in the task's folder, else the full path). The main chat tells them
  with the task's answer; nothing here guesses what a command wrote.
- :func:`asked_files`: the file names the person's words ask a task to write
  or save ("write the advice to career-advice.md", "把结果保存到 notes.csv"): a
  name with a known file extension right after a writing verb and its
  preposition. Only names written as files count; nothing is guessed.
- :func:`missing_files`: of those, the ones not in the task's folder: the
  path as written, or a bare name anywhere under the folder.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Final

from localmelo.support.sessions.store import SessionStore

# The tools that write a file whose path is their ``path`` argument.
WRITE_TOOLS: Final = frozenset({"file_write", "file_edit"})
# The files the chat names at most.
MAX_FILES: Final = 20
# The tool events read to find them, newest first, at most.
MAX_EVENTS: Final = 2000
# Folders a walk for a bare name never enters, and how many entries it reads.
_SKIP: Final = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", ".cache", ".tox"}
)
MAX_WALK: Final = 5000

_EXTENSIONS: Final = (
    "md|markdown|txt|rst|csv|tsv|json|jsonl|ya?ml|toml|html?|xml|pdf|docx?|xlsx?|"
    "pptx?|odt|rtf|tex|ipynb|py|js|mjs|ts|sh|sql|ics|svg|png|jpe?g|log"
)
_NAME: Final = rf"(?:~/|\.{{0,2}}/)?[\w.-]*(?:/[\w.-]+)*\w\.(?:{_EXTENSIONS})(?![\w.])"
_QUOTE: Final = r"[\s`'\"“”‘’「」『』(（]*"
# After an English preposition: a space or a quote, so "an index.html" is no
# "in" before "dex.html".
_AFTER: Final = r"\b[\s`'\"“”‘’「」『』(（]+"
# A writing verb, a few words, then where to: "write the advice to x.md",
# "save it as x.csv", "create notes.md", "export the table into out/x.csv".
_ASKED_EN: Final = re.compile(
    r"\b(?:write|save|store|create|generate|produce|export|output|put|dump|draft|"
    r"record|keep)\b(?:\s+[^\s.;!?,]+){0,8}?\s+(?:to|into|in|as|at|named|called|file)"
    rf"{_AFTER}({_NAME})"
    rf"|\b(?:write|save|create|generate|export|draft)\s+(?:a\s+|an\s+|the\s+)?"
    rf"(?:new\s+)?(?:file\s+)?{_QUOTE}({_NAME})",
    re.IGNORECASE,
)
_ASKED_ZH: Final = re.compile(
    r"(?:写|保存|存(?!在)|生成|创建|输出|导出|整理|记录|放)[^，。；！？\n]{0,30}?"
    rf"(?:到|进|为|成|入|在){_QUOTE}({_NAME})"
    rf"|(?:保存为|存为|存成|另存为|写成|写入|新建|创建|生成|写){_QUOTE}({_NAME})"
)
# A Chinese word written against the name ("我的career-advice.md") is not part of it.
_CJK_LEAD: Final = re.compile(r"^[\u3400-\u9fff\uf900-\ufaff]+(?=[A-Za-z0-9_])")
# A host before a path ("arxiv.org/x.pdf") makes it a link, not a file to write.
_HOST: Final = re.compile(r"^[\w-]+(?:\.[\w-]+)+/")
_LINK: Final = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# What the chat reads of the files a task wrote: one line, each name at most
# NAME_CHARS, the whole at most LINE_CHARS.
NAME_CHARS: Final = 120
LINE_CHARS: Final = 400
# What is quoted after "the log says" is mentioned, not asked: left to the verbs.
_NOT_ASKED: Final = re.compile(
    r"\b(?:don['’]?t|do\s+not|never|no\s+need\s+to)\s+(?:\w+\s+){0,2}$"
    r"|(?:不要|别|不用|不必|无需)[^，。；！？\n]{0,6}$",
    re.IGNORECASE,
)


def asked_files(texts: Iterable[str]) -> list[str]:
    """The file names *texts* ask a task to write or save, in order, once each.

    "Go through my notes and write the summary to summary.md", "save it as
    jobs.csv", "把建议写到 career-advice.md", "保存为 report.md": a writing
    verb (not under "don't" or "不要"), then the name with a known extension.
    A name read or quoted after another verb ("summarize notes.md", "fix
    app.py") is not one.
    """
    found: list[str] = []
    for given in texts:
        text = _LINK.sub(" ", given or "")
        for pattern in (_ASKED_EN, _ASKED_ZH):
            for match in pattern.finditer(text):
                if _NOT_ASKED.search(text[: match.start()]):
                    continue
                name = next(group for group in match.groups() if group)
                name = _CJK_LEAD.sub("", name.rstrip("."))
                if _HOST.match(name) or name in found:
                    continue
                found.append(name)
    return found


def files_line(names: Sequence[str]) -> str:
    """*names* as one line the chat reads: control characters and line breaks
    gone, each name at most :data:`NAME_CHARS`, the whole at most
    :data:`LINE_CHARS`. A file's name is the task's text, never melo's words."""
    shown = []
    for name in names:
        flat = "".join(
            " " if unicodedata.category(ch)[0] in "CZ" and ch != " " else ch
            for ch in str(name)
        )
        flat = " ".join(flat.split())
        if len(flat) > NAME_CHARS:
            flat = flat[: NAME_CHARS - 1] + "…"
        if flat:
            shown.append(flat)
    line = ", ".join(shown)
    return line if len(line) <= LINE_CHARS else line[: LINE_CHARS - 1] + "…"


def _inside(folder: str, path: str) -> bool:
    try:
        return os.path.commonpath([folder, path]) == folder
    except ValueError:
        return False


def _resolve(folder: str | None, path: str) -> str | None:
    """*path* as the file tools read it: ``~`` expanded, relative to *folder*."""
    expanded = os.path.expanduser(path.strip())
    if not expanded:
        return None
    if not os.path.isabs(expanded):
        if not folder:
            return None
        expanded = os.path.join(folder, expanded)
    return os.path.realpath(expanded)


def shown_path(folder: str | None, path: str) -> str:
    """*path* as the person reads it: relative to the task's folder when it is in it."""
    if folder:
        root = os.path.realpath(folder)
        if _inside(root, path):
            return os.path.relpath(path, root)
    return path


async def written_files(
    store: SessionStore, task_id: str, folder: str | None
) -> list[str]:
    """The files *task_id*'s file tools wrote or edited, that are there now (newest last).

    Each ``tool.started`` of a writing tool is paired with its
    ``tool.finished`` by run and call id; only a call that succeeded counts.
    A file named twice is named once, where it was last written.
    """
    started = await store.events_of_type(task_id, "tool.started", limit=MAX_EVENTS)
    finished = await store.events_of_type(task_id, "tool.finished", limit=MAX_EVENTS)
    done = set()
    for stored in finished:
        data = stored.event.get("data") or {}
        if data.get("name") in WRITE_TOOLS and data.get("status") == "success":
            done.add((stored.event.get("run_id"), data.get("call_id")))
    paths: list[str] = []
    for stored in started:
        data = stored.event.get("data") or {}
        key = (stored.event.get("run_id"), data.get("call_id"))
        arguments = data.get("arguments")
        if data.get("name") not in WRITE_TOOLS or key not in done:
            continue
        given = arguments.get("path") if isinstance(arguments, dict) else None
        resolved = _resolve(folder, given) if isinstance(given, str) else None
        if resolved is None or not os.path.isfile(resolved):
            continue
        if resolved in paths:
            paths.remove(resolved)
        paths.append(resolved)
    return [shown_path(folder, path) for path in paths[-MAX_FILES:]]


def _found_under(folder: str, name: str) -> bool:
    """Whether a file called *name* is anywhere under *folder* (a bounded walk)."""
    seen = 0
    for _root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in _SKIP and not d.startswith(".")]
        if name in files:
            return True
        seen += len(dirs) + len(files)
        if seen > MAX_WALK:
            return False
    return False


def missing_files(folder: str | None, names: Sequence[str]) -> list[str]:
    """Of *names* (:func:`asked_files`), the ones not in *folder*.

    A path is looked for as written (relative to the folder, ``~`` expanded);
    a bare name is found anywhere under the folder too. Without a folder, a
    relative name cannot be found: it is missing.
    """
    missing: list[str] = []
    for name in names:
        resolved = _resolve(folder, name)
        if resolved is not None and os.path.isfile(resolved):
            continue
        bare = "/" not in name and not name.startswith("~")
        if bare and folder and os.path.isdir(folder) and _found_under(folder, name):
            continue
        missing.append(name)
    return missing


__all__ = [
    "MAX_FILES",
    "WRITE_TOOLS",
    "asked_files",
    "files_line",
    "missing_files",
    "shown_path",
    "written_files",
]
