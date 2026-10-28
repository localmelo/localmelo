"""Run options for :class:`~localmelo.melo.agent.agent.Agent`."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

from localmelo.melo.executor.builtins import builtin_names
from localmelo.melo.schema import MAX_AGENT_STEPS, MAX_ATTEMPTS, STEPS_PER_ATTEMPT

AttemptPolicy = Literal["reflect", "none"]
ReflectionGate = Literal["flags", "utility"]
ToolExposure = Literal["retrieve", "all"]
ReasoningMode = Literal["legacy", "drop", "replay"]

# The reasoning a replay request carries for one turn is capped at this many
# characters. At the plan's 3.45 characters per token that is about 11,600
# tokens; a longer turn is left out and counted (see Agent._as_sent).
REPLAY_TURN_CAP = 40_000


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
    existed, except that every step now offers all registered built-ins
    and runs only the tools it offered (see ``tool_exposure``), and that
    a reflection decides on its own checks whether the next attempt
    starts (see ``reflection_gate``). An invalid value raises
    ``ValueError`` naming the field.

    - ``max_steps``: cap on planning calls for the whole run.
    - ``attempt_policy``: ``"reflect"`` runs attempts of at most
      ``steps_per_attempt`` steps, up to ``max_attempts`` of them, with a
      reflection after each attempt that runs out of steps or gets stuck;
      ``"none"`` runs one attempt with budget ``max_steps`` and makes no
      reflection calls.
    - ``reflection_gate``: how a reflection decides whether the next
      attempt starts; it matters only with ``attempt_policy="reflect"``.
      ``"flags"`` starts it when the reflection chooses ``continue`` and
      its three checks hold (``task_still_feasible``,
      ``next_step_is_concrete`` and ``next_step_is_novel``). A reply that
      holds no reflection, also when asked again, starts it too when one
      can start, and is not kept; a reflection over
      ``MAX_REFLECTION_CHARS`` keeps its decision and is trimmed to fit.
      ``"utility"``, the behavior from before this option, also requires
      the utility score computed from the reflection's own estimates to
      reach 0.1; a reply without a reflection reads as ``stop``, and a
      reflection over the budget as ``decompose``, unless it chooses
      ``done``, which ends the run as it is.
    - ``tool_exposure``: ``"retrieve"`` offers, on every step, each
      registered built-in, sorted by name, then the other tools (provided
      tools, such as MCP tools) that hint extraction and BM25 resolve for
      the step, ranked among those tools only and also sorted by name;
      ``"all"`` offers every registered tool, sorted by name, on every
      step. In both modes the executor refuses a call to a tool that its
      step did not offer (``TOOL_NOT_OFFERED``), and that tool does not
      run.
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
      ends with status ``timeout``; a ``python_exec`` child process running
      then is killed with every process it started.
    - ``workspace_root``: the directory the built-in tools work in, or
      ``None`` (the default) for none. The agent resolves it to its real
      path (``~`` expanded) and raises ``ValueError`` when that is not a
      directory. With a root, ``file_read`` and ``file_write`` resolve a
      relative path against it and refuse a path whose real path is
      outside it, a symlink that replaces the checked file and anything
      but a regular file;
      ``shell_exec`` and ``python_exec`` run in a child process that
      starts in the root, but can still reach any path, so approval is
      their control; a non-zero exit of ``shell_exec`` starts its text
      with ``[exit code <n>]``. The planning prompt then names the root.
      Without a root, the built-ins behave as before this option, except
      that ``python_exec`` also runs in a child process, in this process's
      directory.
    - ``record_answers``: ``True`` ends every run by appending its answer
      to working memory as an assistant message, so the next run on the
      agent sees it: the result without a ``<think>`` block (see
      :func:`~localmelo.melo.schema.split_thinking`), or a label when
      there is no answer (see :func:`~localmelo.melo.schema.answer_text`),
      cut to ``SESSION_TEXT_CHARS`` characters. ``False``, the default,
      keeps working memory without answers, as before this option.
    - ``tool_timeout_s``: the time limit, in seconds, of the
      ``shell_exec`` and ``python_exec`` that a ``workspace_root`` gives;
      ``None`` keeps their defaults, 30 s for a shell command and the
      executor's 60 s for Python. It needs a ``workspace_root``. The file
      tools keep the executor's limit, and provided tools their own.
    - ``reasoning``: how the run sends a reply's reasoning back.
      ``"legacy"``, the default, keeps the behavior before this option: an
      OpenAI-compatible provider sends none, Ollama native sends its own
      field and inline copy. ``"drop"`` never sends it. ``"replay"`` sends
      it on the assistant turns in scope, under the provider's one key.
    - ``replay_turns``: with ``reasoning="replay"``, the newest K assistant
      turns that carry reasoning are replayed; ``None`` keeps every turn in
      scope. It is ignored in the other modes.
    - ``replay_scope``: ``"task"`` (default) replays only the current
      task's turns; ``"window"`` replays every turn in the window, earlier
      tasks included. It is API-only in R2.
    """

    max_steps: int = MAX_AGENT_STEPS
    attempt_policy: AttemptPolicy = "reflect"
    steps_per_attempt: int = STEPS_PER_ATTEMPT
    max_attempts: int = MAX_ATTEMPTS
    reflection_gate: ReflectionGate = "flags"
    tool_exposure: ToolExposure = "retrieve"
    builtins: frozenset[str] | None = None
    memory: bool = True
    memory_dir: str | None = None
    timeout_s: float | None = None
    workspace_root: str | None = None
    record_answers: bool = False
    tool_timeout_s: float | None = None
    reasoning: ReasoningMode = "legacy"
    replay_turns: int | None = None
    replay_scope: Literal["task", "window"] = "task"

    def __post_init__(self) -> None:
        for name in ("max_steps", "steps_per_attempt", "max_attempts"):
            value = getattr(self, name)
            if not _is_positive_int(value):
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

        for name, allowed in (
            ("attempt_policy", get_args(AttemptPolicy)),
            ("reflection_gate", get_args(ReflectionGate)),
            ("tool_exposure", get_args(ToolExposure)),
            ("reasoning", get_args(ReasoningMode)),
            ("replay_scope", ("task", "window")),
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

        if self.workspace_root is not None and (
            not isinstance(self.workspace_root, str) or not self.workspace_root
        ):
            raise ValueError(
                "workspace_root must be None or a non-empty str, "
                f"got {self.workspace_root!r}"
            )

        if not isinstance(self.record_answers, bool):
            raise ValueError(
                f"record_answers must be a bool, got {self.record_answers!r}"
            )

        if self.replay_turns is not None and not _is_positive_int(self.replay_turns):
            raise ValueError(
                "replay_turns must be None or an int >= 1, "
                f"got {self.replay_turns!r}"
            )

        if self.tool_timeout_s is not None:
            if not _is_positive_finite(self.tool_timeout_s):
                raise ValueError(
                    "tool_timeout_s must be None or a finite number > 0, "
                    f"got {self.tool_timeout_s!r}"
                )
            if self.workspace_root is None:
                raise ValueError("tool_timeout_s needs a workspace_root")

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


def reasoning_refusal(
    mode: str, reasoning_field: str | None, requires_replay: bool
) -> str | None:
    """Why *mode* cannot run on a provider with these capabilities, or ``None``.

    The arguments are the capability values, not a provider, so the CLI can
    check a flag before an Agent exists and :class:`Agent` can check after
    its provider is built. ``legacy`` is never refused, even on a provider
    that requires replay: it keeps the behavior before this option, which
    may draw an HTTP 400 there.
    """
    if mode == "drop" and requires_replay:
        return (
            "this backend requires reasoning replay, so drop is not possible: "
            "use replay"
        )
    if mode == "replay" and not reasoning_field:
        return (
            "replay needs a reasoning field: give --reasoning-field "
            "(auto, reasoning_content or reasoning) with --base-url, or set "
            "reasoning_field in the backend's config"
        )
    return None
