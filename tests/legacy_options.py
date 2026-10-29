"""The three cache settings at their values before the A/B winners became defaults.

``fixed_prefix``, ``window_eviction`` and ``reflect_tool_guard`` now default to
``True``, ``"chunk"`` and ``True``. A test that freezes the requests of the
earlier defaults builds its options with these values, the same ones that
``--fixed-prefix off --window-eviction slide --reflect-tool-guard off`` set.
"""

from __future__ import annotations

from typing import Any

from localmelo.melo.agent import AgentOptions

LEGACY_CACHE_OPTIONS: dict[str, Any] = {
    "fixed_prefix": False,
    "window_eviction": "slide",
    "reflect_tool_guard": False,
}


def legacy_options(**options: Any) -> AgentOptions:
    """``AgentOptions(**options)`` with the earlier cache defaults, unless given."""
    return AgentOptions(**{**LEGACY_CACHE_OPTIONS, **options})
