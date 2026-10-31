"""A one-line summary of a tool call, for events and approval prompts."""

from __future__ import annotations

import ast
import json
import reprlib
import shlex
import warnings
from collections.abc import Mapping
from typing import Any

# Characters a preview keeps; a longer one ends in "...".
PREVIEW_CHARS = 200
# The calls of ``subprocess`` that run a command.
_SUBPROCESS_RUNS = frozenset({"run", "check_output", "check_call", "call", "Popen"})


def call_preview(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Summarize a call of *tool_name* with *arguments* in one line.

    - ``shell_exec``: the command, with runs of whitespace collapsed;
    - ``python_exec``: the command the code runs when it calls
      ``subprocess.run`` (or ``check_output`` and the like) with a literal
      command, or a helper of its own that does, with literal arguments
      (``run("git", "status")`` reads ``git status``); else its first line
      that is not blank, a comment or an import (the first line when there
      is none). Either is followed by `` (+<n> more lines)`` when more lines
      follow it. It is a hint; the approval shows the whole code;
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
            shown = _code_summary(code)
            if shown:
                return shown
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


def _code_summary(code: str) -> str:
    """What a piece of Python does, in one line, or ``""`` for blank code."""
    lines = code.rstrip().replace("\r\n", "\n").replace("\r", "\n").split("\n")
    command, last = "", 0
    tree: ast.Module | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
        command, last = _command_of(tree)
    except (SyntaxError, ValueError, RecursionError):
        pass
    if not command:
        last = _first_line(lines, tree)
        command = lines[last - 1].strip() if last else ""
    if not command:
        return ""
    more = len(lines) - last
    return f"{command} (+{more} more lines)" if more > 0 else command


def _first_line(lines: list[str], tree: ast.Module | None) -> int:
    """The 1-based number of the first line that is no blank, comment or import.

    The first non-blank line when the code is all imports and comments, and 0
    when it is blank.
    """
    if tree is not None:
        for stmt in tree.body:
            if not isinstance(stmt, ast.Import | ast.ImportFrom):
                return stmt.lineno
    else:
        for number, line in enumerate(lines, 1):
            text = line.strip()
            if text and not text.startswith(("#", "import ", "from ")):
                return number
    return next((n for n, line in enumerate(lines, 1) if line.strip()), 0)


def _command_of(tree: ast.Module) -> tuple[str, int]:
    """The first literal command the code runs, and the line it ends on.

    A call of ``subprocess.run`` and the like with a literal command, or a call
    of a function the code defines that runs one, with literal arguments.
    ``("", 0)`` when there is none.
    """
    nodes = list(ast.walk(tree))
    helpers = {
        node.name
        for node in nodes
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(isinstance(n, ast.Call) and _runs_command(n) for n in ast.walk(node))
    }
    calls = sorted(
        (node for node in nodes if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        if _runs_command(call):
            command = _literal_command(call.args, single=True)
        elif isinstance(call.func, ast.Name) and call.func.id in helpers:
            command = _literal_command(call.args, single=False)
        else:
            continue
        if command:
            return command, call.end_lineno or call.lineno
    return "", 0


def _runs_command(call: ast.Call) -> bool:
    """True for ``subprocess.run(...)`` and the like."""
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _SUBPROCESS_RUNS
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )


def _literal_command(args: list[ast.expr], *, single: bool) -> str:
    """A command from literal call arguments, or ``""``.

    The command is a list or tuple of strings, a string (a shell line, when
    *single*) or, for a helper, the strings the call passes one by one.
    """
    if not args:
        return ""
    first = args[0]
    if isinstance(first, ast.List | ast.Tuple):
        parts: list[ast.expr] = list(first.elts)
    elif single:
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return " ".join(first.value.split())
        return ""
    else:
        parts = list(args)
    words = [
        part.value
        for part in parts
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    ]
    return shlex.join(words) if parts and len(words) == len(parts) else ""


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
