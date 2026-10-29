"""The check gate of a ``--brief`` run.

A brief names its checks on lines that start with ``Check:``; the rest of
each line is one shell command. Before such a run ends ``completed`` the
agent runs them (``AgentOptions.brief_check``), because the model's own
report of its checks is not evidence.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from localmelo.melo.executor import Executor

# Times a failing check sends the run back to work.
MAX_CHECK_RETRIES = 2
# Lines of a failing check's output that the model is shown.
CHECK_TAIL_LINES = 60
CHECK_FAILED_LINE = (
    "The task's check failed. Fix what it shows, run it again, then finish."
)

_CHECK_LINE = re.compile(r"^Check:[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class BriefCheckResult:
    """One run of a brief's check.

    *exit_code* is ``None`` when no status was recorded (a timeout) or
    the check was refused, in which case *refused* holds the reason and
    nothing ran. *output* is the last :data:`CHECK_TAIL_LINES` lines,
    already redacted, as is *command*.
    """

    command: str
    exit_code: int | None
    duration_s: float
    output: str = ""
    refused: str = ""

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.refused


# Runs one check command with the run's executor, as a ``shell_exec`` call.
BriefCheck = Callable[["Executor", str], Awaitable[BriefCheckResult]]


def check_commands(brief: str) -> list[str]:
    """The commands on the ``Check:`` lines of *brief*, in order."""
    return _CHECK_LINE.findall(brief)


def failure_message(result: BriefCheckResult) -> str:
    """The user message that tells the model its check failed."""
    status = "none" if result.exit_code is None else str(result.exit_code)
    return (
        f"Command: {result.command}\n"
        f"Exit code: {status}\n"
        f"Last {CHECK_TAIL_LINES} lines of output:\n{result.output}\n\n"
        f"{CHECK_FAILED_LINE}"
    )
