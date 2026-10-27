"""Runs localmelo's CLI with run-end learning switched off, for an A/B.

Usage: ``python -I procedures_base_child.py MELO_ARGS...``

It calls ``localmelo.__main__.main()`` as ``melo MELO_ARGS...`` with
``Agent._learn`` replaced by a no-op: the CLI as it was before runs
learned step templates. Everything else, logging included, is the real
CLI's.
"""

from __future__ import annotations

import sys
from typing import Any

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent


async def _no_learning(self: Agent, *args: Any) -> None:
    return None


if __name__ == "__main__":
    Agent._learn = _no_learning  # type: ignore[method-assign,assignment]
    sys.argv = ["melo", *sys.argv[1:]]
    cli.main()
