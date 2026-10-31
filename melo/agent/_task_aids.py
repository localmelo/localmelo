"""What a task's worker gets that a direct run does not (``AgentOptions.task_aids``).

Two aids, both off unless the gateway's worker turns them on:

- :data:`FINISH_NOTE`, the user line the planning request carries when two
  steps are left in the run's last attempt, so a task keeps them for what it
  must finish (a commit, the report) instead of running out one step short;
- :func:`refusal_note`, the line a tool's text starts with when most of its
  error lines share one cause that the environment gives (a test suite that
  refuses the home folder, a sandbox that refuses a write), so a worker can
  tell the environment's refusal from failures of its own change.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Final

FINISH_NOTE: Final = (
    "Two steps are left. Use them to finish what the task asks: if it asks "
    "for a commit, commit your changes now (git add, then git commit), then "
    "give your final answer. Do not start new checks."
)

# Error lines at least, and the share of them that must have one cause.
MIN_ERROR_LINES: Final = 5
MIN_SHARE: Final = 0.5
CAUSE_CHARS: Final = 200
# One error line: pytest's summary (`ERROR path - Cause`, `FAILED path -
# Cause`), a traceback's `E   Cause`, or an exception line of its own.
_SUMMARY: Final = re.compile(r"^(?:ERROR|FAILED)\s+\S+\s+-\s+(.+)$")
_EXCEPTION: Final = re.compile(
    r"^(?:E\s+)?((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Exit)\b.*)$"
)
# A cause the environment gives, not the code under test.
_ENVIRONMENT: Final = re.compile(
    r"PermissionError|Permission denied|Operation not permitted|Read-only file "
    r"system|refus|real home|\bHOME\b|sandbox|EACCES|EPERM|ConnectionRefused"
    r"|Address already in use",
    re.I,
)
_QUOTED: Final = re.compile(r"(['\"]).*?\1")


def _cause(line: str) -> str | None:
    text = line.strip()
    summary = _SUMMARY.match(text)
    if summary:
        return summary.group(1).strip()
    found = _EXCEPTION.match(text)
    return found.group(1).strip() if found else None


def refusal_note(text: str) -> str:
    """The line naming the one environmental cause of most errors in *text*, or ``""``.

    ``The test environment refused: <cause> (<n> of <m> errors have this one
    cause; they are not failures of the change).`` Causes are compared with
    their quoted parts left out, so one refused path per test still counts
    as one cause.
    """
    causes = [c for c in map(_cause, text.splitlines()) if c]
    if len(causes) < MIN_ERROR_LINES:
        return ""
    keys = Counter(_QUOTED.sub("'…'", cause) for cause in causes)
    key, count = keys.most_common(1)[0]
    if count / len(causes) <= MIN_SHARE or not _ENVIRONMENT.search(key):
        return ""
    shown = next(c for c in causes if _QUOTED.sub("'…'", c) == key)
    if len(shown) > CAUSE_CHARS:
        shown = shown[: CAUSE_CHARS - 1] + "…"
    return (
        f"The test environment refused: {shown} ({count} of {len(causes)} "
        "errors have this one cause; they are not failures of the change)."
    )


__all__ = ["FINISH_NOTE", "refusal_note"]
