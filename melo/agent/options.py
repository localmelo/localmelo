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
    MAX_AGENT_STEPS,
    MAX_ATTEMPTS,
    MAX_EXTENSIONS,
    STEPS_PER_ATTEMPT,
    WINDOW_MESSAGES_MAX,
    WINDOW_MESSAGES_MIN,
)

AttemptPolicy = Literal["reflect", "none"]
# "compress+replay" is not a choice: a provider that requires replay runs "compress"
# as it (see Agent and the CLI).
ReasoningMode = Literal["legacy", "compress", "compress+replay"]
# The reasoning modes that compress each reply into a note: the reply's
# ``<note>`` block is cut off its answer.
NOTE_MODES: frozenset[ReasoningMode] = frozenset({"compress", "compress+replay"})
NoteWriter = Literal["auto", "heuristic"]
WindowEviction = Literal["chunk", "boundary"]
ToolImages = Literal["auto", "on", "off"]
EmptyReplyFallback = Literal["on", "off"]

# The reasoning a replay request carries for one turn is capped at this many
# characters. At the plan's 3.45 characters per token that is about 11,600
# tokens; a longer turn is left out and counted (see agent._replay_plan).
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
    existed, except that every step offers every registered tool, sorted by
    name, and runs only the tools it offered (the executor refuses a call to
    any other: ``TOOL_NOT_OFFERED``), that a reflection decides on its own
    checks whether the next attempt starts, patiently (see ``attempt_policy``),
    that a run without a usable answer is asked for one (see ``answer_on_stop``),
    and that an attempt is long: 50 steps, in a run of 100 (see
    ``steps_per_attempt``).
    An invalid value raises ``ValueError`` naming the field.

    - ``context_window``, ``output_reserve``, ``safety_margin``: the token
      budget planning and reflection requests are fitted to, all three or none.
      None (the default) takes the budget from the length the model server
      reports (``BaseLLMProvider.context_length``, at most
      ``DEFAULT_CONTEXT_CAP``), asked once before the first run; with no length
      the requests are fitted to the prompt cap's characters. A set budget
      always wins.
    - ``max_steps``: cap on planning calls for the whole run (100).
    - ``attempt_policy``: ``"reflect"`` runs attempts of at most
      ``steps_per_attempt`` steps (50), up to ``max_attempts`` of them, with a
      reflection after each attempt that runs out of steps or gets stuck;
      so a run reflects after 50 steps without an answer, then goes on to
      ``max_steps``. ``"none"`` runs one attempt with budget ``max_steps`` and
      makes no reflection calls (``[agent] reflection = false`` sets it for the
      chats, tasks and runs that read the config). A reflection starts the
      next attempt when it chooses ``continue`` and its checks hold:
      ``task_still_feasible`` and ``next_step_is_concrete``. The run is
      patient with the third, ``next_step_is_novel``: a ``continue`` that is
      feasible and concrete but not novel still starts the next attempt, once
      per run; a second one stops the run. A reply that holds no reflection,
      also when asked again, starts it too when one can start, and is not
      kept; a reflection over ``MAX_REFLECTION_CHARS`` keeps its decision and
      is trimmed to fit. A reflection's ``useful_evidence``,
      ``failed_hypotheses`` and ``recommended_avoids`` keep at most 8 items of
      at most 300 characters each.
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
    - ``memory``: ``False`` turns off long-term recall and long-term
      writes; reflections are still cleared at the end of a run.
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
      field and inline copy.
    - ``replay_scope``: ``"task"`` (default) replays only the current
      task's turns; ``"window"`` replays every turn in the window, earlier
      tasks included. It is API-only in R2, and applies where thinking is
      replayed (``compress`` on a provider that requires it).
    - ``compress`` (R4): sends no thinking, asks the model for a short
      note before each tool call, keeps the notes and drafts in the
      working-memory note store, pins a pending draft once after its
      turn's tool results, and puts the attempt's notes in the reflection
      request. A provider that requires replay cannot omit thinking, so it
      runs ``compress`` as ``compress+replay``: the same, with thinking
      replayed on the assistant turns in scope under the provider's one
      key. The effective mode is what the run reports.
    - ``note_writer``: who writes a reply's note in the compress modes:
      ``"auto"`` (default) parses the model's ``<note>`` block and falls
      back to the heuristic, ``"heuristic"`` always writes it from the
      reply's text and thinking. It is accepted and inert in the other
      modes.
    - ``window_eviction``: how working memory drops old turns. ``"chunk"``
      (default) drops whole units down to ``window_messages -
      WINDOW_EVICT_CHUNK`` messages at once, and a request fit (to the prompt
      cap, or to the context budget) cuts down to ``WINDOW_FIT_TARGET`` (70%)
      of its limit instead of just under it and holds that cut.
      ``"boundary"`` is ``"chunk"`` and also drops a finished attempt's turns
      when the next attempt starts after a reflection. Both keep the retained
      history stable for several steps, which a server's prompt cache can
      reuse.
    - ``window_messages``: the working window's message cap (8 to 256). None
      (the default) is ``LOCAL_WINDOW_MESSAGES`` (64), or
      ``WINDOW_MESSAGES_MAX`` (256) while a token budget is in force (see
      ``context_window``), so that the budget's fit is the limit that drops
      history. A larger cap leaves the prompt-size fit (see
      ``window_eviction``) to drop history. 20 is the cap that the CLI, the
      console and the gateway use on a cloud vendor's backend when the config
      sets none (see ``support.config.default_window_messages``); a Python
      caller sets it here. ``Agent.window_messages`` is the cap in force.
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
    - ``repeats``: the run is one of a task that repeats on a schedule (a saved
      chore). Set by code from the schedule, not read from any words: the posting
      review then asks before each post, as it does for a request that says
      "every day" (#570). Default false.
    - ``answer_on_stop``: when true, a run that ends ``gate_stopped``,
      ``budget_exhausted`` or ``stuck`` with no answer makes one more
      request, without tools, for the final answer from the evidence gathered
      (the working window and the latest reflection), in the format the task
      asks for, ``unknown`` allowed; a ``stuck`` run is asked to answer what
      it can in plain words instead. The run keeps its status; its ``result``
      is that answer and ``RunResult.answered_on_stop`` is true. The request
      counts in ``n_aux``. A failed request leaves the run as it was. It also
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
      this option. ``"on"`` also sends the pixels of the newest 3 of them
      (up to 4; the oldest 2 retire together) in each planning request, scaled to at most 1,280 pixels
      wide once, when stored, and each counts toward the prompt cap (see
      ``melo/agent/_tool_images.py``). ``"auto"``, the default, is
      ``"on"`` when the provider says the model reads images
      (``BaseLLMProvider.takes_images``, asked once, when a tool first
      returns an image) and ``"off"`` otherwise. A run whose tools return
      no image sends the same requests in every mode.
    - ``extend_steps``: how many steps a run still making progress is given
      by itself when it is :data:`~localmelo.melo.schema.EXTEND_WHEN` steps
      from the end of its last attempt (so the steps left finish the work and
      report), at most ``max_extensions`` times (#547). Progress is three
      different steps with no error among the last five, each with a call the
      run had not made before. Default 0: a run ends at ``max_steps`` as
      before; the sessions of the gateway, the console and the CLI chats turn
      it on
      (:func:`~localmelo.support.sessions.agents.configured_attempt_options`).
    - ``max_extensions``: how many times ``extend_steps`` may extend a run;
      work a person's chat owns has
      :data:`~localmelo.melo.schema.TASK_EXTENSIONS` (#536). The steps stay one
      count: nothing restarts, and a run that stalls ends at the steps it has.
    - ``refused_steps``: how many steps of a run may go uncounted toward
      ``max_steps`` because a policy check refused every call in them: the
      tool's own check (it returned ``refuse_call(...)``, see
      :mod:`~localmelo.melo.contracts.tools`), a question the tool asked that
      was refused, or the approval policy's denial. Such a step still sends
      its request and counts in ``RunResult.steps``; once the allowance is
      spent, refused steps count as any other. Default 0: nothing changes.
    - ``empty_reply_fallback``: what a run does when a model that did think
      (the provider reported reasoning) ends its turn with no answer and no tool
      call, after the one nudge every empty reply gets. ``"on"``, the default:
      where the service has a switch for thinking (``thinking_off_body``), the
      turn is asked once more with thinking off, and if it ends empty again the
      last lines of the thinking are the answer, ending with the note "(from
      melo's thinking)"; the run completes. ``RunResult.empty_reply`` says which
      of the two happened, and each comes with a ``notice`` event
      (``thinking_off_retry``, ``answer_from_thinking``). ``"off"``: neither;
      the run ends ``failed`` as it did before the fallback existed.
    - ``read_folders``: folders outside the workspace that the read tools
      (``file_read``, ``list_files``) may read without a question: the agent's
      own, such as a chat's work folder (:mod:`~localmelo.melo.contracts.work_folder`).
      A place never offered stays refused, and what ``file_read`` returns from
      there is marked as a web page's text. Default empty.
    """

    context_window: int | None = None
    output_reserve: int | None = None
    safety_margin: int | None = None
    max_steps: int = MAX_AGENT_STEPS
    attempt_policy: AttemptPolicy = "reflect"
    steps_per_attempt: int = STEPS_PER_ATTEMPT
    max_attempts: int = MAX_ATTEMPTS
    builtins: frozenset[str] | None = None
    memory: bool = True
    memory_dir: str | None = None
    timeout_s: float | None = None
    workspace_root: str | None = None
    record_answers: bool = False
    tool_timeout_s: float | None = None
    tool_env: Mapping[str, str] | None = None
    reasoning: ReasoningMode = "legacy"
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
    # R6-lite's write path: keep a decision's text (an evaluation switch);
    # inert in ``legacy``.
    keep_decision_text: bool = False
    # The surface the run belongs to; it decides the default built-ins.
    mode: ToolMode = "python"
    window_eviction: WindowEviction = "chunk"
    brief_probe: Callable[[str], str | None] | None = None
    brief_check: BriefCheck | None = None
    window_messages: int | None = None
    confine_writes: bool = False
    identity: str | None = None
    tool_images: ToolImages = "auto"
    answer_on_stop: bool = True
    max_calls: int | None = None
    task_aids: bool = False
    repeats: bool = False
    refused_steps: int = 0
    extend_steps: int = 0
    max_extensions: int = MAX_EXTENSIONS
    empty_reply_fallback: EmptyReplyFallback = "on"
    read_folders: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.window_messages is not None and (
            isinstance(self.window_messages, bool)
            or not isinstance(self.window_messages, int)
            or not WINDOW_MESSAGES_MIN <= self.window_messages <= WINDOW_MESSAGES_MAX
        ):
            raise ValueError(
                f"window_messages must be an int from {WINDOW_MESSAGES_MIN} "
                f"to {WINDOW_MESSAGES_MAX}, got {self.window_messages!r}"
            )
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
        if (
            isinstance(self.extend_steps, bool)
            or not isinstance(self.extend_steps, int)
            or self.extend_steps < 0
        ):
            raise ValueError(
                f"extend_steps must be an int >= 0, got {self.extend_steps!r}"
            )
        if (
            isinstance(self.max_extensions, bool)
            or not isinstance(self.max_extensions, int)
            or self.max_extensions < 0
        ):
            raise ValueError(
                f"max_extensions must be an int >= 0, got {self.max_extensions!r}"
            )
        if (
            isinstance(self.refused_steps, bool)
            or not isinstance(self.refused_steps, int)
            or self.refused_steps < 0
        ):
            raise ValueError(
                f"refused_steps must be an int >= 0, got {self.refused_steps!r}"
            )
        for name in (
            "max_steps",
            "steps_per_attempt",
            "max_attempts",
        ):
            value = getattr(self, name)
            if not _is_positive_int(value):
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")

        for name, allowed in (
            ("attempt_policy", get_args(AttemptPolicy)),
            ("mode", get_args(ToolMode)),
            ("reasoning", get_args(ReasoningMode)),
            ("replay_scope", ("task", "window")),
            ("note_writer", get_args(NoteWriter)),
            ("window_eviction", get_args(WindowEviction)),
            ("tool_images", get_args(ToolImages)),
            ("empty_reply_fallback", get_args(EmptyReplyFallback)),
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
            "keep_decision_text",
            "answer_on_stop",
            "confine_writes",
            "task_aids",
            "repeats",
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
    if mode == "compress+replay" and not reasoning_field:
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
    if placement == "message" and reasoning == "compress+replay":
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
