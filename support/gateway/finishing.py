"""Whether a task ended with only its commit left (#324).

A worker that runs out of steps may end with "Done except for the final
commit: git add … && git commit …". Its card then says so, with the
command, instead of reading as done: :func:`not_committed` finds the commit
command in the answer and checks that the folder still has changes.

A task that was continued but could not start its next run says so in plain
words: :func:`could_not_continue` makes the line its card shows.

A task whose run ran out of steps is *stopped*, not failed (#341): the store
keeps it ``failed`` with :data:`OUT_OF_STEPS` as its reason, and
:func:`is_out_of_steps` is the one test every surface (card, page, activity
row, terminal) reads, so each shows Stopped and offers Continue.
"""

from __future__ import annotations

import re
from typing import Final

from localmelo.melo.executor.git_status import run_git_status

# A git commit command in an answer: in backticks, else to the end of its
# line, with the `git add … &&` before it when there is one.
_INLINE: Final = re.compile(r"`([^`\n]*\bgit\s+commit\b[^`\n]*)`")
_LINE: Final = re.compile(r"((?:git\s+add\b[^\n`]*?&&\s*)?git\s+commit\b[^\n`]*)", re.M)
# Words that say the commit did not happen: an answer that names a commit it
# made ("committed with `git commit -m …`") is not one left over.
_LEFT: Final = re.compile(
    r"\b(not (yet )?committed|uncommitted|remaining|left to|still (need|has) to"
    r"|did not commit|didn't commit|could not commit|final commit|except for)\b"
    r"|未提交|还需要?提交|尚未提交",
    re.I,
)
COMMAND_CHARS: Final = 300
# The failure reason of a task whose run ended ``budget_exhausted``; it reads as
# plain words wherever a reason is shown as it is.
OUT_OF_STEPS: Final = "Out of steps"
# How a card starts when a continued run could not start, and how much of the
# error follows.
NOT_CONTINUED: Final = "Could not continue"
_REASON_CHARS: Final = 160


def is_out_of_steps(state: str, wait_reason: str) -> bool:
    """Whether a task with this *state* and failure reason stopped for want of steps."""
    return state == "failed" and wait_reason == OUT_OF_STEPS


def could_not_continue(error: BaseException) -> str:
    """``Could not continue: <why>`` for a run that failed before its turn began."""
    why = " ".join(str(error).split()) or type(error).__name__
    if len(why) > _REASON_CHARS:
        why = why[: _REASON_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return f"{NOT_CONTINUED}: {why}"


def commit_command(answer: str) -> str:
    """The last git commit command *answer* names, or ``""``."""
    found = [m.group(1) for m in _INLINE.finditer(answer or "")]
    found = found or [m.group(1) for m in _LINE.finditer(answer or "")]
    if not found:
        return ""
    command = " ".join(found[-1].split())
    return command[:COMMAND_CHARS]


async def has_changes(folder: str) -> bool:
    """Whether the git repository at *folder* has changes not committed."""
    try:
        status = await run_git_status(folder, "status")
    except Exception:  # not a repository, or git is missing: nothing to commit
        return False
    return any(line and not line.startswith("##") for line in status.splitlines())


async def not_committed(answer: str, folder: str | None) -> str:
    """``Not committed. Run: <command>`` when only the commit is left, else ``""``."""
    command = commit_command(answer)
    if not command or not _LEFT.search(answer) or not folder:
        return ""
    if not await has_changes(folder):
        return ""
    return f"Not committed. Run: {command}"


__all__ = [
    "OUT_OF_STEPS",
    "commit_command",
    "has_changes",
    "is_out_of_steps",
    "not_committed",
]
