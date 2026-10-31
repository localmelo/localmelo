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
from localmelo.melo.schema import (
    LOCAL_WINDOW_MESSAGES,
    MAX_AGENT_STEPS,
    MAX_ATTEMPTS,
    STEPS_PER_ATTEMPT,
    WINDOW_MESSAGES_MAX,
    WINDOW_MESSAGES_MIN,
)

AttemptPolicy = Literal["reflect", "none"]
ReflectionGate = Literal["flags", "utility", "patient"]
ToolExposure = Literal["retrieve", "all"]
ReasoningMode = Literal["legacy", "drop", "replay", "compress", "compress+replay"]
NoteWriter = Literal["auto", "heuristic"]
WindowEviction = Literal["slide", "chunk", "boundary"]
CacheLayout = Literal["v2", "v3"]
ToolImages = Literal["auto", "on", "off"]

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
    and runs only the tools it offered (see ``tool_exposure``), that
    a reflection decides on its own checks whether the next attempt
    starts, patiently (see ``reflection_gate``), and that a run without
    a usable answer is asked for one (see ``answer_on_stop``). An
    invalid value raises ``ValueError`` naming the field.

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
      ``done``, which ends the run as it is. ``"patient"``, the default,
      is ``"flags"``, except that a ``continue`` that is feasible
      and concrete but not novel still starts the next attempt, once per
      run; a second one stops the run as ``"flags"`` does. With it, a
      reflection's ``useful_evidence``, ``failed_hypotheses`` and
      ``recommended_avoids`` keep at most 8 items of at most 300
      characters each (``"flags"`` does too when ``answer_on_stop`` is on).
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
      default offer ``file_read``, ``file_write`` and ``python_exec``, and
      the interactive UI and the gateway also ``list_files`` (read-only, so
      it never asks).
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
      but a regular file; ``list_files`` resolves and refuses a folder
      path the same way (no path lists the root);
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
    - ``fixed_prefix``: when true (the default), the run freezes its system
      prefix and local date and keeps each request an extension of the
      previous one, so a server's prompt cache can reuse it. ``False`` is the
      behavior before this default changed.
    - ``window_eviction``: how working memory drops old turns. ``"chunk"``
      (default) drops whole units down to ``window_messages -
      WINDOW_EVICT_CHUNK`` messages at once, and the prompt-cap fit and the
      context fit then drop down to ``WINDOW_FIT_TARGET`` (70%) of their
      limit instead of just under it. ``"boundary"`` is ``"chunk"`` and also
      drops a finished attempt's turns when the next attempt starts after a
      reflection. Both keep the retained history stable for several steps,
      which a server's prompt cache can reuse. ``"slide"`` drops the oldest
      unit at every step once the window is full, as before ``"chunk"`` became
      the default.
    - ``reflect_tool_guard``: when true, a reflection request offers the tools
      the step's planning request sent (the same list, order and schema, so a
      server's prompt prefix holds) and its last user message ends with
      ``REFLECT_TOOL_GUARD_NOTE``, which asks for the JSON object and no tool
      call. The retry request is unchanged. The default is true; ``False`` keeps
      the reflection request as it was before the guard (#183).
    - ``cache_layout``: ``"v3"`` (default), with ``fixed_prefix``, appends a
      reflection once, at the attempt boundary, as an ordinary window message
      (it moves with the history and leaves by eviction) instead of rebuilding
      it as the last message of every planning request, keeps a reflection
      request and its reply in the window so the next planning request
      continues it, and sends an unreadable reflection's retry on the same
      prefix (the reply as an assistant message, then the retry note) instead
      of as a text transcript. Older reflection blocks stay in the history
      until evicted. ``"v2"`` is the layout before this default changed.
      Without ``fixed_prefix`` neither value changes anything.
    - ``window_messages``: the working window's message cap (8 to 256, default
      ``LOCAL_WINDOW_MESSAGES``, 64). A larger cap leaves the prompt-size fit
      (see ``window_eviction``) to drop history. 20 is the cap before this
      default changed, and the one that the CLI, the console and the gateway
      use on a cloud vendor's backend when the config sets none (see
      ``support.config.default_window_messages``); a Python caller sets it
      here.
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
    - ``brief_items``: experimental, for a brief run only. When true, the
      brief's top-level numbered items that no edit has touched yet are named
      in each reflection request and in the deadline check (see
      ``melo/agent/_brief_aids.py``). Default false: nothing changes.
    - ``lean_output``: experimental, for direct runs: fewer output tokens. The
      system prompt asks for no text between tool calls, a final answer of at
      most three lines, relative paths without a ``sys.path`` setup, and for
      reading the files a task names before listing folders, in place of
      "Think step by step. Be concise." (the compress modes keep that line:
      their note before each tool call is such text). With
      ``reflection_gate="flags"`` the reflection prompt also leaves out the
      keys nothing reads and asks for ``best_effort_result`` only with ``done``
      or ``stop``. The CLI sets it for a task from ``[agent] lean_output`` or
      ``-X agent.lean_output=true``; the interactive UI and the gateway never
      do. Default false: requests are as they were before the option.
    - ``identity``: a paragraph for the system prompt of a chat surface: who
      the agent is and which model it runs on (the terminal chat and the
      gateway's sessions build it per Agent, from the backend the Agent
      talks to). The fixed system section carries it right after the base
      text, in planning and reflection requests. ``None`` (the default)
      adds nothing: direct runs and the benchmark send the base text alone
      (the gateway's task workers give their guide here).
    - ``task_aids``: set by the gateway's task workers. With two steps left
      in the run's last attempt, the next planning request ends with
      :data:`~localmelo.melo.agent._task_aids.FINISH_NOTE`, which asks the model to
      use them for what the task asks to finish (a commit, the report); and a
      tool's text whose error lines mostly share one cause (a test run that
      the environment refused) starts with a line that names that cause (see
      :func:`~localmelo.melo.agent._task_aids.refusal_note`). Default false:
      nothing changes.
    - ``answer_on_stop``: when true, a run that ends ``gate_stopped`` or
      ``budget_exhausted`` with no answer makes one more request, without
      tools, for the final answer from the evidence gathered (the working
      window and the latest reflection), in the format the task asks for,
      ``unknown`` allowed. The run keeps its status; its ``result`` is that
      answer and ``RunResult.answered_on_stop`` is true. The request counts
      in ``n_aux``. A failed request leaves the run as it was. It also
      answers a run that ends ``provider_error`` because the server refused a
      request for a budget or rate limit (HTTP 429, or an error that says the
      budget is used), as long as ``max_calls`` allows one more request: the
      status stays and the error is the server's. When no request is left,
      or the request fails, the refused run's ``result`` is the latest text
      the model wrote in the run, if any. It also keeps the answer format
      a task asks for: when the task text names a closing line (such as
      ``FINAL ANSWER:`` or ``Answer:``) and the run's answer lacks it, one more
      request, without tools, asks the model to restate its answer in that
      format from what it already has; the run keeps its status and its
      answer if the request fails, and ``RunResult.format_fixed`` is true when
      the restated answer is the result. A task that names no format is left
      as it is. Default true; false sends neither request.
    - ``max_calls``: a limit on the model requests one run sends (planning,
      reflection and auxiliary requests, a retried request once per
      attempt). Before the request that would pass it, the run stops and ends
      ``budget_exhausted`` with the error ``Call budget reached: ...``. With
      ``answer_on_stop`` the last call is kept for the answer: the run stops
      after ``max_calls - 1`` requests and the answer request is the last one;
      without it the run may send ``max_calls`` requests. A stopped run's
      ``result`` is that answer, else the latest text the model wrote in the
      run, else empty. ``None`` (the default) sets no limit: nothing changes.
    - ``confine_writes``: when true, the ``shell_exec`` and ``python_exec``
      children start, on macOS, under a ``sandbox-exec`` profile that
      refuses every file write outside the workspace (the
      ``workspace_root``, else this process's directory), except in the
      caches the project's tools write, the user's own temporary folder
      (where macOS's ``mktemp`` writes) and, for a linked git worktree, its
      own git directory and the object store (see
      :mod:`~localmelo.melo.executor.sandbox`); reads and the network stay
      open. ``TMPDIR``, ``TMP`` and ``TEMP`` point at ``.melo-tmp`` in the
      workspace, and a call's text that reports a refused write starts with
      a line that says so; ``RunResult.confinement`` counts them. When it
      cannot be applied (``sandbox-exec`` missing, or a sandbox that
      refuses another), the Agent raises ``ConfinementError`` before any
      tool runs. Elsewhere than macOS nothing is confined and
      ``RunResult.confinement`` says so. The CLI sets it for ``--brief``
      runs and the gateway for saved tasks, unless ``[agent]
      confine_writes`` is false. Default false: nothing changes.
    - ``tool_images``: whether the pictures that tools return reach the
      model. ``"off"`` names them in the tool message only, as before
      this option. ``"on"`` also sends the pixels of the newest
      ``tool_images_keep`` of them (up to ``tool_images_keep +
      tool_images_retire - 1``; the oldest ``tool_images_retire`` retire
      together) in each planning request, scaled to at most 1,280 pixels
      wide once, when stored, and each counts toward the prompt cap (see
      ``melo/agent/_tool_images.py``). ``"auto"``, the default, is
      ``"on"`` when the provider says the model reads images
      (``BaseLLMProvider.takes_images``, asked once, when a tool first
      returns an image) and ``"off"`` otherwise. A run whose tools return
      no image sends the same requests in every mode.
    """

    context_window: int | None = None
    output_reserve: int | None = None
    safety_margin: int | None = None
    max_steps: int = MAX_AGENT_STEPS
    attempt_policy: AttemptPolicy = "reflect"
    steps_per_attempt: int = STEPS_PER_ATTEMPT
    max_attempts: int = MAX_ATTEMPTS
    reflection_gate: ReflectionGate = "patient"
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
    fixed_prefix: bool = True
    captured_history: bool = False
    typed_assertions: bool = False
    window_eviction: WindowEviction = "chunk"
    brief_probe: Callable[[str], str | None] | None = None
    brief_check: BriefCheck | None = None
    reflect_tool_guard: bool = True
    cache_layout: CacheLayout = "v3"
    window_messages: int = LOCAL_WINDOW_MESSAGES
    brief_items: bool = False
    lean_output: bool = False
    confine_writes: bool = False
    identity: str | None = None
    tool_images: ToolImages = "auto"
    tool_images_keep: int = 3
    tool_images_retire: int = 2
    answer_on_stop: bool = True
    max_calls: int | None = None
    task_aids: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_prefix, bool):
            raise ValueError("fixed_prefix must be a bool")
        if (
            isinstance(self.window_messages, bool)
            or not isinstance(self.window_messages, int)
            or not WINDOW_MESSAGES_MIN <= self.window_messages <= WINDOW_MESSAGES_MAX
        ):
            raise ValueError(
                f"window_messages must be an int from {WINDOW_MESSAGES_MIN} "
                f"to {WINDOW_MESSAGES_MAX}, got {self.window_messages!r}"
            )
        if not isinstance(self.reflect_tool_guard, bool):
            raise ValueError("reflect_tool_guard must be a bool")
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
        if self.max_calls is not None and (
            isinstance(self.max_calls, bool)
            or not isinstance(self.max_calls, int)
            or self.max_calls < 2
        ):
            raise ValueError(
                f"max_calls must be None or an int >= 2, got {self.max_calls!r}"
            )
        for name in (
            "max_steps",
            "steps_per_attempt",
            "max_attempts",
            "tool_images_keep",
            "tool_images_retire",
        ):
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
            ("cache_layout", get_args(CacheLayout)),
            ("tool_images", get_args(ToolImages)),
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
            "brief_items",
            "lean_output",
            "answer_on_stop",
            "confine_writes",
            "task_aids",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")

        if self.identity is not None and (
            not isinstance(self.identity, str) or not self.identity.strip()
        ):
            raise ValueError(
                f"identity must be None or a non-empty str, got {self.identity!r}"
            )

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
