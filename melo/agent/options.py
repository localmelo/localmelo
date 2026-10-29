"""Run options for :class:`~localmelo.melo.agent.agent.Agent`."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from localmelo.melo.agent._brief_gate import BriefCheck
from localmelo.melo.checker.policy import ToolMode
from localmelo.melo.executor.builtins import builtin_names
from localmelo.melo.memory.recall import (
    DEFAULT_TRIGGERS,
    RECALL_K_DEFAULT,
    RECALL_K_MAX,
    RECALL_MIN_SCORE,
    RECALL_PLACEMENTS,
    RETRIEVER_MODES,
    RecallPlacement,
    RetrieverMode,
)
from localmelo.melo.schema import MAX_AGENT_STEPS, MAX_ATTEMPTS, STEPS_PER_ATTEMPT

AttemptPolicy = Literal["reflect", "none"]
ReflectionGate = Literal["flags", "utility"]
ToolExposure = Literal["retrieve", "all"]
ReasoningMode = Literal["legacy", "drop", "replay", "compress", "compress+replay"]
NoteWriter = Literal["auto", "heuristic"]
WindowEviction = Literal["slide", "chunk", "boundary"]

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
      registers the default set of ``mode``, an empty set none. Any
      iterable of names except a bare ``str`` is accepted and stored as a
      ``frozenset``.
    - ``mode``: the surface the run belongs to, one of ``"direct"``,
      ``"interactive"``, ``"gateway"`` or ``"python"`` (the default). It
      decides the built-ins that ``builtins=None`` registers: only the
      interactive UI offers ``shell_exec`` (its panel asks before an
      ``exec`` call runs); direct, the gateway and the library's ``python``
      default offer ``file_read``, ``file_write`` and ``python_exec``.
      The CLI sets its own surface's mode; a Python caller gets
      ``shell_exec`` by naming it in ``builtins``. An explicit ``builtins``
      wins.
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
    - ``tool_env``: extra variables for the ``shell_exec`` and
      ``python_exec`` children, with or without a ``workspace_root``. A
      child does not inherit this process's environment: it gets ``PATH``,
      ``HOME``, ``USER``, ``LOGNAME``, ``SHELL``, ``TERM``, ``LANG``, every
      ``LC_*`` variable and ``TMPDIR``, then this mapping on top. So an API
      key in the agent's environment reaches no tool; ``VIRTUAL_ENV``,
      ``PYTHONPATH``, ``SSH_AUTH_SOCK`` and the proxy variables pass only
      when this mapping names them.
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
    - ``compress`` (R4): sends no thinking, asks the model for a short
      note before each tool call, keeps the notes and drafts in the
      working-memory note store, pins a pending draft once after its
      turn's tool results, and puts the attempt's notes in the reflection
      request. ``compress+replay`` does the same and replays thinking as
      ``replay`` does. A provider that requires replay runs ``compress``
      as ``compress+replay``, and the effective mode is what the run
      reports. Neither mode recalls anything yet (R5).
    - ``captured_history``: opt in to a bounded, pattern-redacted history copy
      of tool results, separate from the checked model preview. The option is
      off by default; :meth:`Agent.run_tool` does not write history.
    - ``note_writer``: who writes a reply's note in the compress modes:
      ``"auto"`` (default) parses the model's ``<note>`` block and falls
      back to the heuristic, ``"heuristic"`` always writes it from the
      reply's text and thinking. It is accepted and inert in the other
      modes.
    - ``window_eviction``: how working memory drops old turns. ``"slide"``
      (default) drops the oldest unit at every step once the window is full.
      ``"chunk"`` drops whole units down to ``SHORT_TERM_MAX -
      WINDOW_EVICT_CHUNK`` messages at once, and the prompt-cap fit and the
      context fit then drop down to ``WINDOW_FIT_TARGET`` (70%) of their
      limit instead of just under it. ``"boundary"`` is ``"chunk"`` and also
      drops a finished attempt's turns when the next attempt starts after a
      reflection. Both keep the retained history stable for several steps,
      which a server's prompt cache can reuse. Experimental.
    - ``typed_assertions``: opt into conservative, evidence-checked typed
      memory updates and historical recall in explicit reasoning modes. The
      default is false; this experimental option is available through Python.
    - ``brief_probe``: set only by the ``--brief`` worker path. A function of
      the workspace root that returns a digest of the workspace's changes
      (``None`` outside a git tree). Setting it marks the run as a brief run:
      a reflection that says ``continue`` starts the next attempt while steps
      and attempts remain, whatever its flags say, and an attempt boundary
      that finds the digest unchanged since the run started adds a
      ``notice`` event (kind ``no_change``) and one user line that asks the
      next attempt to start by writing. ``None`` (the default) changes nothing.
    - ``brief_check``: set only by the ``--brief`` worker path. A coroutine
      function of the run's executor and one command that runs the command
      as a ``shell_exec`` call and returns a ``BriefCheckResult``. It gates
      a brief run: the commands on the query's ``Check:`` lines run before
      the run ends ``completed``, and a failing one sends the model back to
      work, at most twice (see ``melo/agent/_brief_gate.py``). ``None`` (the
      default) changes nothing.
    """

    context_window: int | None = None
    output_reserve: int | None = None
    safety_margin: int | None = None
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
    tool_env: Mapping[str, str] | None = None
    reasoning: ReasoningMode = "legacy"
    replay_turns: int | None = None
    replay_scope: Literal["task", "window"] = "task"
    note_writer: NoteWriter = "auto"
    # R5's recall: the retriever plan, its triggers, hits, kinds, placement
    # and the embedding/long score floor. Inert in ``legacy``.
    retriever: RetrieverMode = "auto"
    recall_at: frozenset[str] | None = None
    recall_k: int = RECALL_K_DEFAULT
    recall_kinds: frozenset[str] | None = None
    recall_placement: RecallPlacement = "tool"
    recall_min_score: float = RECALL_MIN_SCORE
    cross_task_ratio: float = 1.1
    cross_task_cap: int = 1
    cross_task_cold: int = 0
    # R6-lite's write path: admit the final digest at the run's end, and
    # keep a decision's text (an evaluation switch). Both are inert in
    # ``legacy``.
    admit_on_outcome: bool = False
    keep_decision_text: bool = False
    # The surface the run belongs to; it decides the default built-ins.
    mode: ToolMode = "python"
    fixed_prefix: bool = False
    captured_history: bool = False
    typed_assertions: bool = False
    window_eviction: WindowEviction = "slide"
    brief_probe: Callable[[str], str | None] | None = None
    brief_check: BriefCheck | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_prefix, bool):
            raise ValueError("fixed_prefix must be a bool")
        if self.brief_probe is not None and not callable(self.brief_probe):
            raise ValueError("brief_probe must be callable or None")
        if self.brief_check is not None and not callable(self.brief_check):
            raise ValueError("brief_check must be callable or None")
        budget = (self.context_window, self.output_reserve, self.safety_margin)
        if any(value is not None for value in budget):
            if not all(_is_positive_int(value) for value in budget):
                raise ValueError(
                    "context_window, output_reserve and safety_margin must all be positive integers"
                )
            assert self.context_window is not None
            assert self.output_reserve is not None
            assert self.safety_margin is not None
            if self.context_window <= self.output_reserve + self.safety_margin:
                raise ValueError(
                    "context_window must exceed output_reserve plus safety_margin"
                )
        for name in ("max_steps", "steps_per_attempt", "max_attempts"):
            value = getattr(self, name)
            if not _is_positive_int(value):
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

        for name, allowed in (
            ("attempt_policy", get_args(AttemptPolicy)),
            ("reflection_gate", get_args(ReflectionGate)),
            ("tool_exposure", get_args(ToolExposure)),
            ("mode", get_args(ToolMode)),
            ("reasoning", get_args(ReasoningMode)),
            ("replay_scope", ("task", "window")),
            ("note_writer", get_args(NoteWriter)),
            ("window_eviction", get_args(WindowEviction)),
            ("retriever", RETRIEVER_MODES),
            ("recall_placement", RECALL_PLACEMENTS),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"{name} must be one of {_quoted(allowed)}, got {value!r}"
                )

        if self.builtins is not None:
            object.__setattr__(self, "builtins", self._checked_builtins(self.builtins))

        if self.recall_at is not None:
            object.__setattr__(
                self, "recall_at", frozenset(_checked_triggers(self.recall_at))
            )
        if self.recall_kinds is not None:
            object.__setattr__(
                self, "recall_kinds", frozenset(_checked_kinds(self.recall_kinds))
            )
        if (
            isinstance(self.recall_k, bool)
            or not isinstance(self.recall_k, int)
            or not 1 <= self.recall_k <= RECALL_K_MAX
        ):
            raise ValueError(
                f"recall_k must be an int from 1 to {RECALL_K_MAX}, "
                f"got {self.recall_k!r}"
            )
        if (
            isinstance(self.recall_min_score, bool)
            or not isinstance(self.recall_min_score, int | float)
            or not 0 <= self.recall_min_score <= 1
        ):
            raise ValueError(
                "recall_min_score must be a number from 0 to 1, "
                f"got {self.recall_min_score!r}"
            )

        ratio = self.cross_task_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not 1 <= ratio <= 10
        ):
            raise ValueError("cross_task_ratio must be a number from 1 to 10")
        for name, maximum in (("cross_task_cap", 2), ("cross_task_cold", 1)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name} must be an int from 0 to {maximum}")

        if not isinstance(self.memory, bool):
            raise ValueError(f"memory must be a bool, got {self.memory!r}")

        for name in (
            "admit_on_outcome",
            "keep_decision_text",
            "captured_history",
            "typed_assertions",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")

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

        if self.tool_env is not None:
            object.__setattr__(self, "tool_env", self._checked_env(self.tool_env))

    @staticmethod
    def _checked_env(value: object) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError(
                f"tool_env must be None or a mapping of names to values, "
                f"got {value!r}"
            )
        env: dict[str, str] = {}
        for name, text in value.items():
            if not isinstance(name, str) or not name or "=" in name or "\0" in name:
                raise ValueError(f"tool_env has an invalid variable name {name!r}")
            if not isinstance(text, str):
                raise ValueError(f"tool_env[{name!r}] must be a str, got {text!r}")
            env[name] = text
        return env

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
        known = builtin_names() | {"recall"}
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
            "use replay or compress+replay"
        )
    if mode in ("replay", "compress+replay") and not reasoning_field:
        return (
            "replay and compress+replay need a reasoning field: give "
            "--reasoning-field (auto, reasoning_content or reasoning) with "
            "--base-url, or set reasoning_field in the backend's config"
        )
    return None


def _checked_triggers(values: Iterable[str]) -> tuple[str, ...]:
    """*values* deduplicated to the canonical trigger order."""
    names = {_checked_trigger(value) for value in values}
    return tuple(trigger for trigger in (*DEFAULT_TRIGGERS, "tool") if trigger in names)


def _checked_trigger(value: object) -> str:
    if not isinstance(value, str) or value not in (*DEFAULT_TRIGGERS, "tool"):
        raise ValueError(
            f"recall_at has an unknown trigger {value!r}; the triggers are "
            "start, attempt, step, tool"
        )
    return value


def _checked_kinds(values: Iterable[str]) -> tuple[str, ...]:
    """*values* deduplicated to the canonical kind order."""
    return tuple(
        kind
        for kind in _KIND_ORDER
        if kind in {_checked_kind(value) for value in values}
    )


def _checked_kind(value: object) -> str:
    if not isinstance(value, str) or value not in _KIND_ORDER:
        raise ValueError(
            f"recall_kinds has an unknown kind {value!r}; the kinds are "
            f"{', '.join(_KIND_ORDER)}"
        )
    return value


_KIND_ORDER = ("note", "draft", "thought", "digest", "reflection", "long")


def retrieval_refusal(
    mode: str,
    plan: object,
    has_embedding: bool,
    reasoning: str,
    *,
    memory: bool = True,
    placement: str = "tool",
    recall_at: Iterable[str] = (),
    mcp_only: bool = False,
    builtins: frozenset[str] | None = None,
) -> str | None:
    """Why this recall configuration cannot run, or ``None``.

    *plan* is a :class:`~localmelo.melo.memory.recall.RecallPlan` (or the
    options' retriever string); only its trigger map is read. The recall
    options are inert in ``legacy`` (decision 10), so nothing is refused
    there; the CLI refuses the flags themselves before this runs.
    """
    if reasoning == "legacy":
        return None
    if mode == "embedding" and not has_embedding:
        return "--retriever embedding needs an embedding provider"
    if mode == "embedding" and not memory:
        return "--retriever embedding needs --memory on"
    if placement == "message" and reasoning in ("replay", "compress+replay"):
        return "--recall-placement message is not available in replay modes"
    triggers: set[str] = set()
    enabled = getattr(plan, "enabled", None)
    if callable(enabled):
        triggers.update(enabled())
    if recall_at:
        triggers.update(recall_at)
    if not triggers and mode != "off":
        triggers.update(DEFAULT_TRIGGERS)
    if "tool" in triggers and mcp_only:
        return "--recall-at tool is not available with --mcp-only"
    if builtins is not None and "recall" in builtins and "tool" not in triggers:
        return "--builtins recall needs --recall-at tool"
    return None


__all__ = ["AgentOptions", "reasoning_refusal", "retrieval_refusal"]
