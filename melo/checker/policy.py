"""The tool mode: which built-ins a surface offers, and whether it can ask.

A tier says what a tool can reach (:mod:`melo.checker.tiers`); the mode
says who is there to answer before a call runs. This module is the one
table that decides both, so a surface names its mode instead of choosing a
tool set next to the agent it builds. It imports only the schema and the
tiers, so the checker, the executor and the agent can read it without an
import cycle.

The surfaces set the mode where they are known (the CLI's two branches,
the gateway's sessions), and :class:`~melo.agent.options.AgentOptions`
defaults to ``python`` for the library. Direct mode keeps its old
behaviour in this release, including the six-pattern command blocklist; the
gateway and the console do not offer ``shell_exec`` at all.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal

from localmelo.melo.schema import ToolTier

ToolMode = Literal["direct", "interactive", "gateway", "python"]

# The built-ins a mode offers when the caller does not name a set.
# `shell_exec` stays in `direct` until P13b flips it, after the bench arm
# (`--builtins all` against the new default); P13a leaves direct mode
# byte-identical. The gateway has no approver, so it does not offer the
# tool at all; `python_exec` stays everywhere because it is a child process
# and the bench's Python tasks keep working.
DEFAULT_BUILTINS: Mapping[ToolMode, frozenset[str]] = {
    "direct": frozenset({"file_read", "file_write", "python_exec", "shell_exec"}),
    "interactive": frozenset({"file_read", "file_write", "python_exec", "shell_exec"}),
    "gateway": frozenset({"file_read", "file_write", "python_exec"}),
    "python": frozenset({"file_read", "file_write", "python_exec", "shell_exec"}),
}

# The modes in which a run can have an approver: the interactive UI always
# builds a policy, and a Python caller passes `approval=...` to run_task.
# Direct mode has no panel and the gateway no authenticated one (P12 is not
# on main), so neither can ask.
APPROVING_MODES: frozenset[ToolMode] = frozenset({"interactive", "python"})


def mode_requires_approval(mode: ToolMode) -> bool:
    """Whether *mode* can have a policy that answers before a call runs."""
    return mode in APPROVING_MODES


def builtin_set_for(mode: ToolMode, names: Collection[str] | None) -> frozenset[str]:
    """The built-ins to register: *names* when given, else *mode*'s default.

    An explicit set always wins, so ``--builtins all`` keeps a tool that
    the mode's default omits, and ``--builtins none`` registers nothing.
    """
    if names is not None:
        return frozenset(names)
    return DEFAULT_BUILTINS[mode]


def exec_refusal(tool_name: str, tier: ToolTier, mode: ToolMode) -> str | None:
    """Why *mode* refuses a call of *tool_name* of *tier*, or ``None``.

    This is the backstop for a tool that was registered explicitly even
    though the mode does not offer it: the default set keeps the tool out
    of the model's hands, and this refusal keeps it out of a run that named
    it anyway. An ``exec`` tool that the mode's default set omits is
    refused when the mode has no approver.

    `python_exec` stays allowed because it is in every default set and runs
    in a child process. This release applies this to the gateway only; direct
    mode still runs its six-pattern blocklist.
    """
    if tier != "exec" or mode_requires_approval(mode):
        return None
    if tool_name in DEFAULT_BUILTINS[mode]:
        return None
    return (
        f"{tool_name} is not available in this mode ({mode} mode has no "
        "approval policy). Use --builtins all to enable it."
    )


__all__ = [
    "APPROVING_MODES",
    "DEFAULT_BUILTINS",
    "ToolMode",
    "builtin_set_for",
    "exec_refusal",
    "mode_requires_approval",
]
