"""Rules v0 of the action review (#279): routine, high-impact or undecided.

Pure standard library: ``shlex`` for shell, ``ast`` for Python. Nothing runs;
the only I/O is ``stat``/``realpath``, reading the local files a call runs or
edits (at most 10, 64 KB each, see :mod:`.files`), and, for a runner such as
``pytest``, a walk of the workspace for code changed in this chat.
"""

from __future__ import annotations

import os
import sys

from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.checker.review import files as file_rules
from localmelo.melo.checker.review.paths import Context, Findings, resolve, touch
from localmelo.melo.checker.review.python import review_python
from localmelo.melo.checker.review.shell import review_shell
from localmelo.melo.contracts.review import RULED_TOOLS, ActionReview

__all__ = ["RULED_TOOLS", "review_action"]


def review_action(payload: ActionReviewPayload) -> ActionReview:
    """Classify one call; a failure of the rules is undecided, never routine."""
    home = payload.home or os.path.expanduser("~")
    root = os.path.realpath(payload.workspace_root or os.getcwd())
    prefix = os.path.abspath(payload.prefix or sys.prefix)
    ctx = Context(
        root,
        os.path.realpath(home),
        payload.executable or sys.executable,
        prefix,
        since=payload.since,
    )
    f, args = Findings(), payload.arguments
    try:
        if payload.tool_name == "shell_exec":
            command = args.get("command")
            text = " ".join(map(str, command)) if isinstance(command, list) else command
            review_shell(ctx, f, str(text or ""))
        elif payload.tool_name == "python_exec":
            review_python(ctx, f, str(args.get("code") or ""))
        elif payload.tool_name in ("file_write", "file_edit"):
            # The workspace fence places it; what the path is counts, and the
            # code it writes.
            raw = str(args.get("path") or "")
            touch(ctx, f, raw, "write")
            f.high.pop("outside_write", None)
            path = resolve(ctx, raw)
            if path is not None and payload.tool_name == "file_write":
                file_rules.replaced(ctx, f, path, str(args.get("content") or ""))
            elif path is not None:
                old, new = str(args.get("old") or ""), str(args.get("new") or "")
                file_rules.edited(ctx, f, path, old, new)
        elif payload.destructive:
            name = payload.tool_name
            f.add("delete", f"Runs {name}, a tool that says it can destroy data.")
    except Exception as exc:  # the rules fail safe: the call is asked about
        f.doubt(f"melo could not check this call ({type(exc).__name__}).")
    return f.review()
