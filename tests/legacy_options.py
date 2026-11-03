"""The cache settings at their values before the A/B winners became defaults.

``fixed_prefix``, ``window_eviction`` and ``reflect_tool_guard`` now default to
``True``, ``"chunk"`` and ``True``, ``cache_layout`` to ``"v3"`` and the window to
64 messages on a local server. A test that freezes the requests of the earlier
defaults builds its options with these values, the same ones that
``--fixed-prefix off --window-eviction slide --reflect-tool-guard off
--cache-layout v2 --window-messages 20`` set.
"""

from __future__ import annotations

from typing import Any

from localmelo.melo.agent import AgentOptions

LEGACY_CACHE_OPTIONS: dict[str, Any] = {
    "fixed_prefix": False,
    "window_eviction": "slide",
    "reflect_tool_guard": False,
    "cache_layout": "v2",
    "window_messages": 20,
}


# The gate and the answer request at their values before the usable-answers change:
# ``reflection_gate`` now defaults to ``"patient"`` and ``answer_on_stop`` to ``True``
# (the options ``--reflection-gate flags`` and ``-X agent.answer_on_stop=off`` set).
LEGACY_ANSWER_OPTIONS: dict[str, Any] = {
    "reflection_gate": "flags",
    "answer_on_stop": False,
}


# The run length before the long attempts: ``max_steps`` is now 100 and
# ``steps_per_attempt`` 50 (``-s 30 -X agent.steps_per_attempt=10`` set the earlier
# ones). Bench runs that pin ``--max-steps 30`` with ``--attempt-policy none`` are
# unchanged: ``steps_per_attempt`` does not matter without reflection.
LEGACY_STEP_OPTIONS: dict[str, Any] = {
    "max_steps": 30,
    "steps_per_attempt": 10,
}


def legacy_options(**options: Any) -> AgentOptions:
    """``AgentOptions(**options)`` with the earlier defaults, unless given.

    The earlier cache defaults (:data:`LEGACY_CACHE_OPTIONS`), the earlier gate
    and answer request (:data:`LEGACY_ANSWER_OPTIONS`) and the earlier run length
    (:data:`LEGACY_STEP_OPTIONS`).
    """
    return AgentOptions(
        **{
            **LEGACY_CACHE_OPTIONS,
            **LEGACY_ANSWER_OPTIONS,
            **LEGACY_STEP_OPTIONS,
            **options,
        }
    )
