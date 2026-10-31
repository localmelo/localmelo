"""A one-line summary of a tool call, for events and approval prompts."""

from __future__ import annotations

import json
import reprlib
from collections.abc import Mapping
from typing import Any

# Characters a preview keeps; a longer one ends in "...".
PREVIEW_CHARS = 200


def call_preview(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Summarize a call of *tool_name* with *arguments* in one line.

    - ``shell_exec``: the command, with runs of whitespace collapsed;
    - ``python_exec``: the first non-blank line of the code, followed by
      `` (+<n> more lines)`` when more lines follow it;
    - ``file_read``: the path;
    - ``list_files``: the path (``.`` without one), then `` (<n> levels)``
      for a depth above one;
    - ``file_write``: ``<path> (<n> chars)``, where ``n`` counts the content;
    - ``file_edit``: ``<path> (edit)``;
    - ``git_status``: ``git <command>``, then the revision when there is one;
    - any other tool, or an argument above that is missing or not a ``str``:
      the arguments as compact JSON, keys sorted.

    A line break inside the summary becomes a space, and a summary longer
    than :data:`PREVIEW_CHARS` keeps its first ``PREVIEW_CHARS - 3``
    characters followed by ``...``. It never raises.
    """
    text = _summary(tool_name, arguments)
    text = " ".join(text.splitlines())
    if len(text) > PREVIEW_CHARS:
        text = text[: PREVIEW_CHARS - 3] + "..."
    return text


def _summary(tool_name: str, arguments: Mapping[str, Any]) -> str:
    if tool_name == "shell_exec":
        command = arguments.get("command")
        if isinstance(command, str):
            return " ".join(command.split())
    elif tool_name == "python_exec":
        code = arguments.get("code")
        if isinstance(code, str):
            lines = code.rstrip().splitlines()
            for index, line in enumerate(lines):
                if line.strip():
                    more = len(lines) - index - 1
                    suffix = f" (+{more} more lines)" if more else ""
                    return line.strip() + suffix
    elif tool_name == "file_read":
        path = arguments.get("path")
        if isinstance(path, str):
            return path
    elif tool_name == "list_files":
        path, depth = arguments.get("path") or ".", arguments.get("depth")
        if isinstance(path, str) and (depth is None or isinstance(depth, int)):
            return f"{path} ({depth} levels)" if depth and depth > 1 else path
    elif tool_name == "file_write":
        path, content = arguments.get("path"), arguments.get("content")
        if isinstance(path, str) and isinstance(content, str):
            return f"{path} ({len(content)} chars)"
    elif tool_name == "file_edit":
        path = arguments.get("path")
        if isinstance(path, str):
            return f"{path} (edit)"
    elif tool_name == "git_status":
        command, rev = arguments.get("command"), arguments.get("rev") or ""
        if isinstance(command, str) and isinstance(rev, str):
            return " ".join(["git", command, rev]).strip()
    return _as_json(arguments)


def _as_json(arguments: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        # Keys that cannot be sorted together, nesting too deep, or a value
        # whose str() raises: reprlib's form is bounded and never raises.
        return reprlib.repr(arguments)
