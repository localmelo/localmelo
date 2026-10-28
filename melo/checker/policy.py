"""The tool mode: which built-ins a surface offers, and whether it can ask.

A tier says what a tool can reach (:mod:`melo.checker.tiers`); the mode
says who is there to answer before a call runs. This module is the one
table that decides both, so a surface names its mode instead of choosing a
tool set next to the agent it builds. It imports only ``ToolTier`` from the
schema, so the checker, the executor and the agent can read it without an
import cycle.

The surfaces set the mode where they are known (the CLI's two branches,
the gateway's sessions), and :class:`~melo.agent.options.AgentOptions`
defaults to ``python`` for the library. A mode with an approver (the
interactive UI) keeps ``shell_exec``; the modes without one (direct, the
gateway and the library's ``python`` default) do not offer it. An explicit
``builtins`` set always wins, so a caller that names ``shell_exec`` gets it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal, get_args

from localmelo.melo.schema import ToolTier

ToolMode = Literal["direct", "interactive", "gateway", "python"]

# The built-ins a mode offers when the caller does not name a set.
# `shell_exec` is offered only by `interactive`, whose panel answers before
# an `exec` call runs. Direct mode has no panel; the gateway has no
# authenticated one (P12 is not on main); and the library's default
# `python` mode passes no approval policy, so a Python caller must name
# `shell_exec` in `builtins` to get it. `python_exec`, `file_read` and
# `file_write` stay everywhere: `python_exec` is a child process and the
# bench's Python tasks keep working.
DEFAULT_BUILTINS: Mapping[ToolMode, frozenset[str]] = {
    "direct": frozenset({"file_read", "file_write", "python_exec"}),
    "interactive": frozenset({"file_read", "file_write", "python_exec", "shell_exec"}),
    "gateway": frozenset({"file_read", "file_write", "python_exec"}),
    "python": frozenset({"file_read", "file_write", "python_exec"}),
}

# The modes in which every run has an approver: the interactive UI always
# builds a policy. A Python caller can pass `approval=...` to `run_task`,
# but the library's default has none, so `python`'s default set matches
# direct's; a caller that names `shell_exec` in `builtins` opts in.
APPROVING_MODES: frozenset[ToolMode] = frozenset({"interactive"})


def mode_requires_approval(mode: ToolMode) -> bool:
    """Whether every run of *mode* has a policy that answers first.

    The interactive UI does. A Python caller can pass ``approval=`` to
    ``run_task``, but the library's default run has none, which is why
    ``python``'s default set matches direct's.
    """
    return mode in APPROVING_MODES


def builtin_set_for(mode: ToolMode, names: Collection[str] | None) -> frozenset[str]:
    """The built-ins to register: *names* when given, else *mode*'s default.

    An explicit set always wins, so ``--builtins all`` keeps a tool that
    the mode's default omits, and ``--builtins none`` registers nothing.
    """
    if names is not None:
        return frozenset(names)
    return DEFAULT_BUILTINS[mode]


def exec_refusal(
    tool_name: str,
    tier: ToolTier,
    mode: ToolMode,
    named_builtins: Collection[str] | None = None,
) -> str | None:
    """Why *mode* refuses a call of *tool_name* of *tier*, or ``None``.

    This is the backstop for a tool that was registered even though the
    mode does not offer it: the default set keeps the tool out of the
    model's hands, and this refusal keeps it out of a run that registered
    it anyway. An ``exec`` tool that the mode's default set omits is
    refused when the mode has no approver, unless a direct or ``python``
    caller named it in *named_builtins*: an explicit set is the caller's
    opt-in there, so ``--builtins shell_exec`` and
    ``AgentOptions(builtins={"shell_exec"})`` both run. The gateway refuses
    it even when named: no local process can approve its own request (P12
    is not on main).

    `python_exec` stays allowed because it is in every default set and runs
    in a child process.
    """
    if tier != "exec" or mode_requires_approval(mode):
        return None
    if (
        mode in ("direct", "python")
        and named_builtins is not None
        and tool_name in named_builtins
    ):
        return None
    if tool_name in DEFAULT_BUILTINS[mode]:
        return None
    return (
        f"{tool_name} is not available in this mode ({mode} mode has no "
        "approval policy). Use --builtins all to enable it."
    )


def validated_mode(mode: object) -> ToolMode:
    """*mode* if it names a tool mode, else ``ValueError`` naming the field."""
    allowed = get_args(ToolMode)
    if mode not in allowed:
        names = ", ".join(repr(name) for name in allowed)
        raise ValueError(f"mode must be one of {names}, got {mode!r}")
    return mode  # type: ignore[return-value]


__all__ = [
    "APPROVING_MODES",
    "DEFAULT_BUILTINS",
    "ToolMode",
    "builtin_set_for",
    "exec_refusal",
    "mode_requires_approval",
    "validated_mode",
]
