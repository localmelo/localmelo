"""The files a task writes, as the core and the gateway both name them.

One derivation for both readers. :func:`written_paths`: the work checkpoint lists
the files a run wrote from the calls it made, and the gateway
(``support.gateway.task_files``) lists the files a task wrote from its stored
``tool.*`` events; each pairs a call with its outcome its own way. Nothing here
guesses what a command wrote. :func:`asked_files`: the file names the person's
words ask a task to write, which the gateway checks a task's ending against and a
run reads to know its result goes into a file. This module imports only the standard library and
:mod:`localmelo.melo.contracts.folders`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Final

from localmelo.melo.contracts.folders import WRITE_TOOLS

# The files a list names at most.
MAX_FILES: Final = 20


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


def _inside(folder: str, path: str) -> bool:
    try:
        return os.path.commonpath([folder, path]) == folder
    except ValueError:
        return False


def resolve(folder: str | None, path: str) -> str | None:
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


def written_paths(
    calls: Iterable[tuple[object, object, object]], folder: str | None
) -> list[str]:
    """The files that *calls* wrote or edited and that are there now (newest last).

    *calls* are ``(tool name, arguments, status)`` in the order they ran; only a
    file tool that writes (:data:`~localmelo.melo.contracts.folders.WRITE_TOOLS`)
    with the status ``success`` counts, at its ``path`` argument. A file named
    twice is named once, where it was last written; at most :data:`MAX_FILES`,
    each as :func:`shown_path` names it.
    """
    paths: list[str] = []
    for name, arguments, status in calls:
        if name not in WRITE_TOOLS or status != "success":
            continue
        given = arguments.get("path") if isinstance(arguments, dict) else None
        resolved = resolve(folder, given) if isinstance(given, str) else None
        if resolved is None or not os.path.isfile(resolved):
            continue
        if resolved in paths:
            paths.remove(resolved)
        paths.append(resolved)
    return [shown_path(folder, path) for path in paths[-MAX_FILES:]]


__all__ = ["MAX_FILES", "asked_files", "resolve", "shown_path", "written_paths"]
