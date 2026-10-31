"""Rules v0 of the action review (#279): routine, high-impact or undecided.

Pure standard library: ``shlex`` for shell, ``ast`` for Python. Nothing runs;
the only I/O is ``stat``/``realpath`` and reading a local script the call runs
(at most 64 KB, one level deep).
"""

from __future__ import annotations

import os
import sys

from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.checker.review.paths import Context, Findings, touch
from localmelo.melo.checker.review.python import review_python
from localmelo.melo.checker.review.shell import review_shell
from localmelo.melo.contracts.review import ActionReview

__all__ = ["review_action"]


def review_action(payload: ActionReviewPayload) -> ActionReview:
    """Classify one call; a failure of the rules is undecided, never routine."""
    home = payload.home or os.path.expanduser("~")
    root = os.path.realpath(payload.workspace_root or os.getcwd())
    prefix = os.path.abspath(payload.prefix or sys.prefix)
    ctx = Context(
        root, os.path.realpath(home), payload.executable or sys.executable, prefix
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
            # The workspace fence places it; only what the path is counts.
            touch(ctx, f, str(args.get("path") or ""), "write")
            f.high.pop("outside_write", None)
    except Exception as exc:  # the rules fail safe: the call is asked about
        f.doubt(f"melo could not check this call ({type(exc).__name__}).")
    return f.review()
