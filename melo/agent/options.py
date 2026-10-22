"""Run options for :class:`~localmelo.melo.agent.agent.Agent`."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

from localmelo.melo.executor.builtins import builtin_names
from localmelo.melo.schema import MAX_AGENT_STEPS, MAX_ATTEMPTS, STEPS_PER_ATTEMPT

AttemptPolicy = Literal["reflect", "none"]
ToolExposure = Literal["retrieve", "all"]


def _quoted(names: Iterable[object]) -> str:
    return ", ".join(sorted(repr(name) for name in names))


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:  # an int too large for a float
        return False


@dataclass(frozen=True)
class AgentOptions:
    """How an :class:`Agent` runs its tasks.

    The defaults match the agent's behavior from before these options
    existed, except that a step now runs only the tools it offered (see
    ``tool_exposure``). An invalid value raises ``ValueError`` naming the
    field.

    - ``max_steps``: cap on planning calls for the whole run.
    - ``attempt_policy``: ``"reflect"`` runs attempts of at most
      ``steps_per_attempt`` steps, up to ``max_attempts`` of them, with a
      reflection after each attempt that runs out of steps or gets stuck;
      ``"none"`` runs one attempt with budget ``max_steps`` and makes no
      reflection calls.
    - ``tool_exposure``: ``"retrieve"`` offers the tools that hint
      extraction and BM25 resolve for each step; ``"all"`` offers every
      registered tool, sorted by name, on every step. In both modes the
      executor refuses a call to a tool that its step did not offer
      (``TOOL_NOT_OFFERED``), and that tool does not run.
    - ``builtins``: names of the built-in tools to register. ``None``
      registers all of them, an empty set none. Any iterable of names
      except a bare ``str`` is accepted and stored as a ``frozenset``.
    - ``memory``: ``False`` turns off long-term recall, long-term writes
      and reflection promotion; reflections are still cleared at the end
      of a run.
    - ``memory_dir``: directory for SQLite persistence. When it is
      ``None``, the ``LOCALMELO_PERSIST_MEMORY`` and
      ``LOCALMELO_MEMORY_DIR`` environment variables decide.
    - ``timeout_s``: wall-clock bound for one run, in seconds; ``None``
      means no bound. A run still going when it expires is cancelled and
      ends with status ``timeout``, except while ``python_exec`` code
      runs: that code runs on the agent's event loop and cannot be
      interrupted, so the run ends only once the code returns.
    """

    max_steps: int = MAX_AGENT_STEPS
    attempt_policy: AttemptPolicy = "reflect"
    steps_per_attempt: int = STEPS_PER_ATTEMPT
    max_attempts: int = MAX_ATTEMPTS
    tool_exposure: ToolExposure = "retrieve"
    builtins: frozenset[str] | None = None
    memory: bool = True
    memory_dir: str | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_steps", "steps_per_attempt", "max_attempts"):
            value = getattr(self, name)
            if not _is_positive_int(value):
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

        for name, allowed in (
            ("attempt_policy", get_args(AttemptPolicy)),
            ("tool_exposure", get_args(ToolExposure)),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"{name} must be one of {_quoted(allowed)}, got {value!r}"
                )

        if self.builtins is not None:
            object.__setattr__(self, "builtins", self._checked_builtins(self.builtins))

        if not isinstance(self.memory, bool):
            raise ValueError(f"memory must be a bool, got {self.memory!r}")

        if self.memory_dir is not None and (
            not isinstance(self.memory_dir, str) or not self.memory_dir
        ):
            raise ValueError(
                f"memory_dir must be None or a non-empty str, got {self.memory_dir!r}"
            )

        if self.timeout_s is not None and not _is_positive_finite(self.timeout_s):
            raise ValueError(
                f"timeout_s must be None or a finite number > 0, got {self.timeout_s!r}"
            )

    @staticmethod
    def _checked_builtins(value: object) -> frozenset[str]:
        if isinstance(value, str) or not isinstance(value, Iterable):
            raise ValueError(
                f"builtins must be None or an iterable of tool names, got {value!r}"
            )
        try:
            names = frozenset(value)
        except TypeError:
            raise ValueError(
                f"builtins must contain tool names (str), got {value!r}"
            ) from None
        known = builtin_names()
        unknown = names - known
        if unknown:
            raise ValueError(
                f"builtins has unknown names {_quoted(unknown)}; "
                f"known names: {_quoted(known)}"
            )
        return names
