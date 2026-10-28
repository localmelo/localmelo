"""The environment of a process a tool starts.

``shell_exec`` and ``python_exec`` start a child, and an MCP server runs as
one too. None of them inherits this process's environment: a child gets the
variables named in :data:`DEFAULT_ENV_VARS` and every ``LC_*`` locale
variable that are set here, then the caller's own values on top, and
nothing else. API keys and other secrets in the agent's environment
therefore reach no tool, nor anything a tool starts, unless the caller
passes them on purpose.

This module lives in ``melo/`` because both sides use it and ``melo/``
never imports ``support/``: the executor's built-ins and the MCP client
call :func:`child_env`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

# Variables a child inherits from this process: the official MCP SDK's
# POSIX defaults (``HOME``, ``LOGNAME``, ``PATH``, ``SHELL``, ``TERM``,
# ``USER``), the locale and ``TMPDIR``. Every ``LC_*`` variable passes too
# (see :func:`child_env`). Anything else must come from the caller.
DEFAULT_ENV_VARS: Final = (
    "HOME",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TERM",
    "USER",
    "LANG",
    "TMPDIR",
)

# A variable with this prefix is a locale setting, such as ``LC_ALL`` or
# ``LC_TIME``, and passes like the names above.
LOCALE_PREFIX: Final = "LC_"


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of a child process, as this process gives it.

    The variables of :data:`DEFAULT_ENV_VARS` and every ``LC_*`` variable
    that are set in this process, then *extra* on top as literal values.
    Nothing else is inherited.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if name in DEFAULT_ENV_VARS or name.startswith(LOCALE_PREFIX)
    }
    if extra:
        env.update(extra)
    return env
