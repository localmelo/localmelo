"""Persistent configuration stored at ~/.cache/localmelo/config.toml."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import re
import stat
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from localmelo.melo.contracts.folders import (
    above_home,
    canonical_path,
    is_alias,
    never_offered,
)
from localmelo.melo.schema import (
    LOCAL_WINDOW_MESSAGES,
    SHORT_TERM_MAX,
    WINDOW_MESSAGES_MAX,
    WINDOW_MESSAGES_MIN,
)


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


# The cache directory and the config file, as ``~`` strings resolved when
# read or written. Resolving at import would fix them to the home the
# process started with, so a later HOME (a test's temporary one) could not
# redirect them.
CONFIG_DIR = "~/.cache/localmelo"
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.toml")


def config_dir() -> str:
    """The cache directory with ``~`` expanded for the current HOME."""
    return os.path.expanduser(CONFIG_DIR)


def config_path() -> str:
    """The config file with ``~`` expanded for the current HOME."""
    return os.path.expanduser(CONFIG_PATH)


VALID_CHAT_BACKENDS = (
    "mlc",
    "ollama",
    "vllm",
    "sglang",
    "openai",
    "gemini",
    "anthropic",
    "nvidia",
)
VALID_EMBEDDING_BACKENDS = ("mlc", "ollama", "vllm", "sglang", "none")
# Another spelling of "vllm" that a config file may use for either backend.
OMLX_ALIAS = "omlx"
REASONING_MODES = ("legacy", "drop", "replay", "compress", "compress+replay")
NOTE_WRITERS = ("auto", "heuristic")
WINDOW_EVICTIONS = ("slide", "chunk", "boundary")
CACHE_LAYOUTS = ("v2", "v3")
TOOL_IMAGES = ("auto", "on", "off")
REASONING_FIELDS = ("auto", "reasoning_content", "reasoning")
RETRIEVAL_MODES = ("auto", "off", "bm25", "llm", "embedding")
# How memory across chats finds notes: see MemoryConfig.retriever.
MEMORY_RETRIEVERS = ("auto", "bm25", "embedding", "llm")
# How notes are tidied while melo is idle: see MemoryConfig.sleep.
MEMORY_SLEEP = ("off", "rules", "model", "model+tags")
# The recall budget and its bounds (inclusive): see MemoryConfig.recall_records.
MEMORY_RECALL_RECORDS = (0, 20)
MEMORY_RECALL_DOCS = (0, 20)
MEMORY_RECALL_CHARS = (200, 8000)
# Whether recall carries notes, and what a turn that recalls nothing is told.
MEMORY_RECALL_NOTES = ("on", "off")
MEMORY_SEARCH_WHEN_EMPTY = ("off", "hint")
RECALL_KEYS = (
    "recall_records",
    "recall_docs",
    "recall_chars",
    "recall_notes",
    "search_when_empty",
)
RECALL_TRIGGERS = ("start", "attempt", "step", "tool")
RECALL_KINDS = ("note", "draft", "thought", "digest", "reflection", "long")
RECALL_PLACEMENTS = ("tool", "message")
REASONING_TAGS = ("explicit", "closing_only", "none")
UI_MODES = ("default", "developer")
# The console's and the app's language: "auto" follows the system.
CONSOLE_LANGUAGES = ("auto", "en", "zh-Hans")
# How a new chat works: melo asks before risky steps, or auto mode (#339).
APPROVAL_MODES = ("ask", "auto")
STATUS_LINES = ("auto", "off")
CLI_COLORS = ("auto", "16", "256", "truecolor")
CLI_BACKGROUNDS = ("auto", "dark", "light")
# What bare melo opens: the main chat through the gateway, or a chat of its own.
CLI_CHATS = ("main", "local")


@dataclass
class LocalBackendConfig:
    """Config shape shared by all local OpenAI-compatible backends."""

    chat_url: str = ""
    chat_model: str = ""
    embedding_url: str = ""
    embedding_model: str = ""
    # Extra top-level fields merged into every chat request body, e.g.
    # {"chat_template_kwargs": {"enable_thinking": False}}.
    extra_body: dict[str, Any] = field(default_factory=dict)
    # The request key the provider sends a reply's reasoning back under:
    # "" (unset), "auto", "reasoning_content" or "reasoning"; for the
    # [ollama] section, also "thinking".
    reasoning_field: str = ""
    # Whether the chat server requires a reply's reasoning in later requests.
    requires_reasoning_replay: bool = False


@dataclass
class CloudVendorConfig:
    """Config shape shared by all cloud vendor backends."""

    api_key_env: str = ""  # env var name, e.g. OPENAI_API_KEY
    chat_model: str = ""
    base_url: str = ""  # optional override; each vendor has a built-in default
    # Extra top-level fields merged into every chat request body.
    extra_body: dict[str, Any] = field(default_factory=dict)
    # See LocalBackendConfig: the reasoning field and whether the server
    # requires replay.
    reasoning_field: str = ""
    requires_reasoning_replay: bool = False


@dataclass
class GatewayConfig:
    port: int = 8401
    host: str = "127.0.0.1"
    workspace: str = ""


@dataclass
class SessionsConfig:
    """How an interactive surface restores a saved conversation.

    ``record_direct`` says whether a direct run (``melo "<task>"``) is recorded
    as a worker session; only ``false`` turns it off.
    """

    resume_mode: str = "pairs"
    record_direct: bool = True


@dataclass
class CliConfig:
    """The terminal UI's live streaming (on unless turned off), per model.

    ``reasoning_tags`` maps a model id to its trusted framing declaration
    (``explicit``, ``closing_only`` or ``none``). An undeclared model keeps the
    conservative rule: streamed content is held until the reply ends. Only the
    interactive UI reads ``stream``: direct runs and the gateway never stream.
    """

    stream: bool = True
    reasoning_tags: dict[str, str] = field(default_factory=dict)
    # "auto" adds cache, context, step and task segments to the status bar;
    # "off" keeps the bar as it was before #176.
    status_line: str = "auto"
    # The terminal's colour depth and background; "auto" detects them.
    colors: str = "auto"  # auto | 16 | 256 | truecolor
    background: str = "auto"  # auto | dark | light
    # "main": bare melo opens the main chat through the gateway (started when
    # none runs); "local": the terminal's own chat, without the gateway.
    chat: str = "main"


@dataclass
class UiConfig:
    """The ``[ui]`` table: ``mode`` decides which options ``melo --help`` shows."""

    mode: str = "default"  # default | developer


@dataclass
class ConsoleConfig:
    """The ``[console]`` table: the language of the web console and the macOS app.

    ``auto`` follows the browser's or the Mac's preferred languages; ``en`` and
    ``zh-Hans`` (Simplified Chinese) choose one. The terminal is always English.
    """

    language: str = "auto"  # auto | en | zh-Hans


@dataclass
class ApprovalsConfig:
    """The ``[approvals]`` table: the mode a new chat starts in (#339).

    ``ask``: melo asks before it runs commands or changes files. ``auto``:
    nobody is asked; routine steps run, risky steps and steps the person did
    not ask for are blocked and explained. Each chat keeps its own mode.
    """

    mode: str = "ask"  # ask | auto


@dataclass
class AgentConfig:
    """The ``[agent]`` table: how a direct run or a session reasons."""

    reasoning: str = "legacy"  # legacy | drop | replay | compress | compress+replay
    replay_turns: int | None = None  # None: every turn in scope
    note_writer: str = "auto"  # auto | heuristic (compress modes)
    tool_env: list[str] = field(default_factory=list)  # extra inherited variable names
    fixed_prefix: bool = True  # stable prompt prefix; false is the old request
    window_eviction: str = "chunk"  # slide | chunk | boundary
    reflect_tool_guard: bool = True  # reflection requests ask for JSON, not a tool
    cache_layout: str = "v3"  # v2 | v3 (the reflection layout; needs fixed_prefix)
    # Working window cap, 8-256; None: default_window_messages() of the backend.
    window_messages: int | None = None
    context_window: int | None = None
    output_reserve: int | None = None
    safety_margin: int | None = None
    captured_history: bool = False  # opt-in redacted captured tool results
    brief_items: bool = (
        False  # --brief runs name the items no edit touched (experimental)
    )
    # Experimental: direct runs ask for fewer output tokens (AgentOptions.lean_output).
    lean_output: bool = False
    # Worker runs (--brief) and saved tasks: shell_exec and python_exec write only
    # in the workspace on macOS (AgentOptions.confine_writes); false turns it off.
    confine_writes: bool = True
    # Developer setting: plain direct runs (melo "TASK", melo "") are confined too
    # (AgentOptions.confine_writes); worker runs and saved tasks follow confine_writes.
    confine_direct_runs: bool = False
    # Developer setting: a run that stops (gate_stopped, budget_exhausted) with no
    # answer asks the model once for one, and an answer that lacks the closing line
    # the task asks for gets it (AgentOptions.answer_on_stop). false turns both off.
    answer_on_stop: bool = True
    # Developer setting: most model requests one direct run sends (planning,
    # reflection and auxiliary); at the limit it stops and answers from what it
    # has (AgentOptions.max_calls). None: no limit.
    max_calls: int | None = None
    # Pictures tools return reach the model: auto (when the server says the model
    # reads images) | on | off; the newest keep, retired retire at a time.
    tool_images: str = "auto"
    tool_images_keep: int = 3
    tool_images_retire: int = 2

    def window_for(self, backend: str) -> int:
        """The working window's cap: ``window_messages``, else *backend*'s default."""
        if self.window_messages is not None:
            return self.window_messages
        return default_window_messages(backend)


IMAGE_KEYS = ("tool_images", "tool_images_keep", "tool_images_retire")


def tool_images_errors(agent: AgentConfig) -> list[str]:
    """What is wrong with ``[agent] tool_images`` and its two counts."""
    errors: list[str] = []
    if agent.tool_images not in TOOL_IMAGES:
        errors.append(
            f"[agent] tool_images {agent.tool_images!r} is not recognised. "
            f"Must be one of {TOOL_IMAGES}."
        )
    for name in IMAGE_KEYS[1:]:
        value = getattr(agent, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            errors.append(f"[agent] {name} must be an int >= 1, got {value!r}")
    return errors


def default_window_messages(backend: str) -> int:
    """The working window's message cap when nothing sets one, for a chat *backend*.

    A local server (ollama, mlc, vllm or oMLX, sglang) and an OpenAI-compatible
    URL reuse the prompt prefix from their cache, so a long window adds few new
    tokens per request and keeps more of the task in view: 64, the cap that two
    benchmark rounds measured. Without prompt caching a longer window costs input
    tokens on every request, so a cloud vendor's backend keeps 20 until a longer
    window is measured there. An explicit ``[agent] window_messages`` or
    ``--window-messages`` always wins.
    """
    return SHORT_TERM_MAX if backend in _CLOUD_SECTIONS else LOCAL_WINDOW_MESSAGES


def window_reason(backend: str) -> str:
    """Why the default window of *backend* is what it is: ``cloud`` or ``local server``."""
    return "cloud" if backend in _CLOUD_SECTIONS else "local server"


def window_default_label(backend: str) -> str:
    """The default window with its reason: ``64 (local server)`` or ``20 (cloud)``."""
    return f"{default_window_messages(backend)} ({window_reason(backend)})"


@dataclass
class RetrievalConfig:
    """The ``[retrieval]`` table: R5's recall defaults for direct runs."""

    retriever: str = "auto"  # auto | off | bm25 | llm | embedding
    recall_at: list[str] = field(default_factory=list)  # empty: the defaults
    k: int = 3
    kinds: list[str] = field(default_factory=list)  # empty: all but thought
    placement: str = "tool"  # tool | message
    min_score: float = 0.35
    cross_task_ratio: float = 1.1
    cross_task_cap: int = 1
    cross_task_cold: int = 0


@dataclass
class MemoryConfig:
    """The ``[memory]`` table: memory across chats, and R6-lite's default for direct runs.

    ``across_chats`` keeps notes of what the user says about themselves and an
    index of every chat, and recalls them in the chats (not in direct runs or
    workers); it is on unless the file turns it off. ``retriever`` is how a
    chat message finds its notes: ``auto`` (words, and meaning too when an
    embedding model is set up), ``bm25`` (words only), ``embedding`` (words and
    meaning; an embedding model is required) or ``llm`` (words pick candidates
    and the chat model chooses among them, with no embedding calls).
    ``keep_statements``, a developer setting for comparison and off by default,
    makes code keep after a chat turn the plain, lasting facts the user stated
    that the model did not keep with ``remember``. ``sleep``, another developer
    setting, tidies the notes while melo is idle: ``off`` (the default) or
    ``rules`` (no model call), ``model`` (rules then a bounded model pass), or
    ``model+tags`` (the same call limit, tagged turns first). ``sleep_lock`` is
    an optional path whose presence prevents model requests. Five more developer
    settings, each defaulting to today's behaviour, shape the recall block a chat
    message carries: ``recall_records`` notes and ``recall_docs`` chat turns at
    most (0-20 each) in a block of at most ``recall_chars`` characters
    (200-8000); ``recall_notes = "off"`` leaves notes out of the block and the
    pinned lines, so only turns are recalled; and ``search_when_empty = "hint"``
    adds one line to a block that recalls nothing, telling the model to call the
    memory search tool first.
    """

    admit_on_outcome: bool = False
    across_chats: bool = True
    retriever: str = "auto"  # auto | bm25 | embedding | llm
    keep_statements: bool = False
    sleep: str = "off"  # off | rules | model | model+tags
    sleep_lock: str = ""  # Optional lock path: no model calls while it exists.
    recall_records: int = 3  # Most notes a chat message's recall block holds.
    recall_docs: int = 2  # Most chat turns it holds.
    recall_chars: int = 900  # Its size bound, in characters.
    recall_notes: str = "on"  # on | off: no notes in the block or the pinned lines.
    search_when_empty: str = "off"  # off | hint: a line when nothing was recalled.


@dataclass
class SchedulerConfig:
    """The live gate's settings (S2): off, or one of the two enforcing arms.

    ``n_srv`` is the model server's slot count, ``r_f`` the reserve kept for
    callers the gate cannot see (the CLI and direct runs) and ``n_fg`` the
    background cap while a chat is focused. A chat is focused while a model
    call of its turn is running and for ``focus_s`` seconds after its last call
    or keystroke; an open chat with no turn running is not. The call log holds
    one JSONL file per gateway run; an empty ``call_log_dir`` means the default.
    """

    policy: str = "off"  # off | fcfs | class
    n_srv: int = 8
    r_f: int = 1
    n_fg: int = 1
    focus_s: float = 10.0
    call_log: bool = True
    call_log_dir: str = ""
    # The GPU lock and the battery (S2b). An empty path means no lock
    # reader; the gateway's battery reader is always on. ``record`` only
    # observes; ``wait`` holds every workspace call while a foreign lock is
    # present.
    gpu_lock_file: str = ""
    lock_self: str = ""
    lock_gate: str = "record"  # record | wait


BROWSER_DRIVERS = ("off", "safari-automation")


@dataclass
class BrowserConfig:
    """The ``[browser]`` table: whether melo may read web pages in a browser.

    ``driver`` is ``off`` (the default, no browser tools) or ``safari-automation``
    (Safari's automation windows). Only the terminal chat and the console offer it.
    """

    driver: str = "off"  # off | safari-automation


# The most folders ``[workspace] extra_folders`` keeps.
EXTRA_FOLDERS_MAX: Final = 100


@dataclass
class WorkspaceConfig:
    """The ``[workspace]`` table: folders outside the workspace melo may read (#405).

    ``extra_folders`` holds the folders the person chose "Always allow this
    folder" for, as real paths (``~/`` for the home folder). melo reads in
    them, and below them, without asking in a chat or task the person
    started; writing there still asks every time. An entry that is not
    valid (see :func:`extra_folder_problem`) is ignored, never used.
    """

    extra_folders: list[str] = field(default_factory=list)


@dataclass
class ToolsConfig:
    """The ``[tools]`` table: tools a chat may use, each on or off.

    ``web_fetch`` reads a public web page as text (on unless the file turns it
    off). The chats and the console's tasks offer it; direct runs never do.
    """

    web_fetch: bool = True


@dataclass
class Config:
    chat_backend: str = (
        ""  # mlc | ollama | vllm | sglang | openai | gemini | anthropic | nvidia
    )
    embedding_backend: str = ""  # mlc | ollama | vllm | sglang | none
    # The file spelled the backend "omlx", an alias of "vllm": the backend
    # field says "vllm" and this says how save() writes it back (see load()).
    chat_alias: str = field(default="", repr=False, compare=False)
    embedding_alias: str = field(default="", repr=False, compare=False)

    # A path to an MCP servers JSON file the CLI and the gateway both read.
    mcp_config: str = ""

    # HTTP timeout of each chat request, in seconds, for every chat backend.
    # None keeps each chat provider's own default (120 s; 300 s for the
    # ollama backend). Embedding requests keep their own timeout.
    request_timeout_s: float | None = None

    # Local backends
    mlc: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    ollama: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    vllm: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    sglang: LocalBackendConfig = field(default_factory=LocalBackendConfig)

    # Cloud vendors
    openai: CloudVendorConfig = field(default_factory=CloudVendorConfig)
    gemini: CloudVendorConfig = field(default_factory=CloudVendorConfig)
    anthropic: CloudVendorConfig = field(default_factory=CloudVendorConfig)
    nvidia: CloudVendorConfig = field(default_factory=CloudVendorConfig)

    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    cli: CliConfig = field(default_factory=CliConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    approvals: ApprovalsConfig = field(default_factory=ApprovalsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    # [pricing.<model>]: dollars per million tokens (input, output, and
    # cache_read for cached prompt tokens), for the terminal's session cost.
    # Configuration only: no price is bundled.
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    _allow_unconfigured: bool = field(default=False, repr=False, compare=False)
    _enable_playground: bool = field(default=False, repr=False, compare=False)
    # ``-X`` values applied by ``load()``: key -> (what the file had, what is set).
    _overridden: dict[str, tuple[Any, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )
    # ``melo --serve --port 0``: the gateway binds a free port and writes it
    # to ``gateway.port``. A saved ``[gateway] port = 0`` is still invalid.
    _ephemeral_port: bool = field(default=False, repr=False, compare=False)

    @property
    def gateway_port_ok(self) -> bool:
        """Whether ``gateway.port`` can be listened on: 1-65535, or 0 when asked."""
        port = self.gateway.port
        return 0 < port <= 65535 or (port == 0 and self._ephemeral_port)

    @property
    def is_configured(self) -> bool:
        return self.chat_backend != ""

    @property
    def has_embedding(self) -> bool:
        if self.embedding_backend in ("", "none"):
            return False
        from localmelo.support.backends.registry import get_backend

        try:
            backend = get_backend(self.embedding_backend)
            return backend.has_embedding(self)
        except KeyError:
            return False

    def validate(self) -> list[str]:
        """Return a list of validation errors. Empty list means valid."""
        errors: list[str] = []
        errors.extend(self._validate_agent())
        errors.extend(self._validate_retrieval())
        errors.extend(self._validate_memory())
        errors.extend(self._validate_scheduler())
        errors.extend(self._validate_browser())
        errors.extend(self._validate_tools())
        # [cli] is a display preference: like [sessions] resume_mode, a bad
        # value warns and falls back in the interactive UI (cli_errors()), so
        # direct mode and the gateway never fail on it.
        errors.extend(self._validate_gateway_workspace())

        # -- chat_backend validation --
        if not self.chat_backend:
            errors.append("chat_backend is not set (run 'melo setup')")
            return errors

        if self.chat_backend not in VALID_CHAT_BACKENDS:
            errors.append(
                f"chat_backend {self.chat_backend!r} is not recognised. "
                f"Must be one of {VALID_CHAT_BACKENDS}."
            )
            return errors

        # -- embedding_backend validation --
        if not self.embedding_backend:
            errors.append("embedding_backend is not set (run 'melo setup')")
            return errors

        if self.embedding_backend not in VALID_EMBEDDING_BACKENDS:
            errors.append(
                f"embedding_backend {self.embedding_backend!r} is not recognised. "
                f"Must be one of {VALID_EMBEDDING_BACKENDS}."
            )
            return errors

        # -- Delegate chat backend validation --
        chat_errors = self._validate_chat_backend()
        errors.extend(chat_errors)
        errors.extend(self._validate_extra_body())
        errors.extend(self._validate_reasoning_fields())

        # -- Delegate embedding backend validation (skip if "none") --
        if self.embedding_backend != "none":
            emb_errors = self._validate_embedding_backend()
            errors.extend(emb_errors)

        if self.memory.retriever == "embedding" and not self.has_embedding:
            errors.append(
                '[memory] retriever = "embedding" needs an embedding model, and '
                "none is set up. Run 'melo setup' to add one, or use "
                '"auto", "bm25" or "llm".'
            )

        # -- Chat request timeout --
        errors.extend(self._validate_request_timeout())

        # -- Gateway validation --
        if not self.gateway_port_ok:
            errors.append(
                f"[gateway] port={self.gateway.port} is out of range (1-65535)"
            )

        return errors

    def _validate_gateway_workspace(self) -> list[str]:
        """``[gateway] workspace`` is empty or an absolute/``~/`` single-line path."""
        problem = gateway_workspace_problem(self.gateway.workspace)
        return [] if problem is None else [f"[gateway] workspace: {problem}"]

    def _validate_memory(self) -> list[str]:
        """``[memory]`` holds bools and a known retriever."""
        errors = [
            f"[memory] {name} must be a bool, got {value!r}"
            for name in ("admit_on_outcome", "across_chats", "keep_statements")
            if not isinstance(value := getattr(self.memory, name), bool)
        ]
        if self.memory.retriever not in MEMORY_RETRIEVERS:
            errors.append(
                f"[memory] retriever {self.memory.retriever!r} is not "
                f"recognised. Must be one of {MEMORY_RETRIEVERS}."
            )
        if self.memory.sleep not in MEMORY_SLEEP:
            errors.append(
                f"[memory] sleep {self.memory.sleep!r} is not recognised. "
                f"Must be one of {MEMORY_SLEEP}."
            )
        if (
            not isinstance(self.memory.sleep_lock, str)
            or "\x00" in self.memory.sleep_lock
        ):
            errors.append("[memory] sleep_lock must be a path string without NUL")
        for name, (low, high) in (
            ("recall_records", MEMORY_RECALL_RECORDS),
            ("recall_docs", MEMORY_RECALL_DOCS),
            ("recall_chars", MEMORY_RECALL_CHARS),
        ):
            value = getattr(self.memory, name)
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"[memory] {name} must be a whole number, got {value!r}")
            elif not low <= value <= high:
                errors.append(
                    f"[memory] {name} must be from {low} to {high}, got {value}"
                )
        for name, allowed in (
            ("recall_notes", MEMORY_RECALL_NOTES),
            ("search_when_empty", MEMORY_SEARCH_WHEN_EMPTY),
        ):
            value = getattr(self.memory, name)
            if value not in allowed:
                errors.append(
                    f"[memory] {name} {value!r} is not recognised. "
                    f"Must be one of {allowed}."
                )
        return errors

    def _validate_browser(self) -> list[str]:
        """``[browser] driver`` names a driver."""
        if self.browser.driver in BROWSER_DRIVERS:
            return []
        return [
            f"[browser] driver={self.browser.driver!r} is not one of {BROWSER_DRIVERS}"
        ]

    def _validate_tools(self) -> list[str]:
        """``[tools]`` holds bools."""
        return [
            f"[tools] {name} must be a bool, got {value!r}"
            for name in ("web_fetch",)
            if not isinstance(value := getattr(self.tools, name), bool)
        ]

    def _validate_scheduler(self) -> list[str]:
        """``[scheduler]`` values are inside their documented ranges."""
        scheduler = self.scheduler
        errors: list[str] = []
        if scheduler.policy not in ("off", "fcfs", "class"):
            errors.append(
                f"[scheduler] policy={scheduler.policy!r} is not one of "
                "('off', 'fcfs', 'class')"
            )
        for name, minimum in (("n_srv", 1), ("r_f", 0), ("n_fg", 0)):
            value: object = getattr(scheduler, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                errors.append(
                    f"[scheduler] {name} must be an integer >= {minimum}, "
                    f"got {value!r}"
                )
        focus: object = scheduler.focus_s
        if (
            isinstance(focus, bool)
            or not isinstance(focus, int | float)
            or not 0 < focus < float("inf")
        ):
            errors.append(
                f"[scheduler] focus_s must be a number of seconds > 0, got {focus!r}"
            )
        if not isinstance(scheduler.call_log, bool):
            errors.append(
                f"[scheduler] call_log must be a boolean, got {scheduler.call_log!r}"
            )
        if not isinstance(scheduler.call_log_dir, str):
            errors.append(
                "[scheduler] call_log_dir must be a string, "
                f"got {scheduler.call_log_dir!r}"
            )
        if scheduler.lock_gate not in ("record", "wait"):
            errors.append(
                f"[scheduler] lock_gate={scheduler.lock_gate!r} is not one of "
                "('record', 'wait')"
            )
        for name in ("gpu_lock_file", "lock_self"):
            value = getattr(scheduler, name)
            if not isinstance(value, str):
                errors.append(f"[scheduler] {name} must be a string, got {value!r}")
            elif "\x00" in value:
                errors.append(f"[scheduler] {name} must not contain a NUL byte")
        return errors

    def _validate_cli(self) -> list[str]:
        """``[cli]`` holds a bool and a model-to-hint table of known values."""
        errors: list[str] = []
        cli = self.cli
        if not isinstance(cli.stream, bool):
            errors.append(f"[cli] stream must be a boolean, got {cli.stream!r}")
        if cli.status_line not in STATUS_LINES:
            errors.append(
                f"[cli] status_line {cli.status_line!r} is not one of {STATUS_LINES}"
            )
        for key, allowed in (
            ("colors", CLI_COLORS),
            ("background", CLI_BACKGROUNDS),
            ("chat", CLI_CHATS),
        ):
            if getattr(cli, key) not in allowed:
                errors.append(
                    f"[cli] {key} {getattr(cli, key)!r} is not one of {allowed}"
                )
        tags = cli.reasoning_tags
        if not isinstance(tags, dict):
            return [*errors, f"[cli] reasoning_tags must be a table, got {tags!r}"]
        for model, hint in tags.items():
            if (
                not isinstance(model, str)
                or not model.strip()
                or has_control_or_surrogate(model)
            ):
                errors.append(f"[cli] reasoning_tags has an invalid model id {model!r}")
            if not isinstance(hint, str) or hint not in REASONING_TAGS:
                errors.append(
                    f"[cli] reasoning_tags[{model!r}] {hint!r} is not recognised. "
                    f"Must be one of {REASONING_TAGS}."
                )
        return errors

    def _validate_ui(self) -> list[str]:
        """``[ui] mode`` is one of the two modes."""
        if self.ui.mode in UI_MODES:
            return []
        return [f"[ui] mode={self.ui.mode!r} is not one of {UI_MODES}"]

    def _validate_console(self) -> list[str]:
        """``[console] language`` is ``auto`` or a language the console has."""
        if self.console.language in CONSOLE_LANGUAGES:
            return []
        return [
            f"[console] language={self.console.language!r} "
            f"is not one of {CONSOLE_LANGUAGES}"
        ]

    def _validate_workspace(self) -> list[str]:
        """``[workspace] extra_folders`` is a list of folders melo may be given."""
        return extra_folders_errors(self.workspace.extra_folders)

    def _validate_approvals(self) -> list[str]:
        """``[approvals] mode`` is ``ask`` or ``auto``."""
        if self.approvals.mode in APPROVAL_MODES:
            return []
        return [
            f"[approvals] mode={self.approvals.mode!r} is not one of {APPROVAL_MODES}"
        ]

    def _validate_request_timeout(self) -> list[str]:
        """``request_timeout_s`` is unset or a finite number of seconds > 0."""
        value: object = self.request_timeout_s
        if value is None or _is_positive_seconds(value):
            return []
        return [
            "request_timeout_s must be a finite number of seconds > 0, "
            f"got {value!r}"
        ]

    def _validate_chat_backend(self) -> list[str]:
        from localmelo.support.backends.registry import get_backend

        try:
            backend = get_backend(self.chat_backend)
            return backend.validate(self)
        except KeyError:
            return [f"chat_backend {self.chat_backend!r}: no registered adapter found."]

    def _validate_agent(self) -> list[str]:
        """``[agent]`` holds known values for its mode and note writer."""
        errors: list[str] = []
        if not isinstance(self.agent.fixed_prefix, bool):
            errors.append("[agent] fixed_prefix must be a boolean")
        if not isinstance(self.agent.reflect_tool_guard, bool):
            errors.append("[agent] reflect_tool_guard must be a boolean")
        if not isinstance(self.agent.captured_history, bool):
            errors.append("[agent] captured_history must be a boolean")
        if not isinstance(self.agent.brief_items, bool):
            errors.append("[agent] brief_items must be a boolean")
        if not isinstance(self.agent.lean_output, bool):
            errors.append("[agent] lean_output must be a boolean")
        if not isinstance(self.agent.confine_writes, bool):
            errors.append("[agent] confine_writes must be a boolean")
        if not isinstance(self.agent.confine_direct_runs, bool):
            errors.append("[agent] confine_direct_runs must be a boolean")
        if not isinstance(self.agent.answer_on_stop, bool):
            errors.append("[agent] answer_on_stop must be a boolean")
        calls: object = self.agent.max_calls
        if calls is not None and (
            isinstance(calls, bool) or not isinstance(calls, int) or calls < 2
        ):
            errors.append(f"[agent] max_calls must be an int >= 2, got {calls!r}")
        errors.extend(tool_images_errors(self.agent))
        if self.agent.reasoning not in REASONING_MODES:
            errors.append(
                f"[agent] reasoning {self.agent.reasoning!r} is not recognised. "
                f"Must be one of {REASONING_MODES}."
            )
        if self.agent.note_writer not in NOTE_WRITERS:
            errors.append(
                f"[agent] note_writer {self.agent.note_writer!r} is not "
                f"recognised. Must be one of {NOTE_WRITERS}."
            )
        if self.agent.window_eviction not in WINDOW_EVICTIONS:
            errors.append(
                f"[agent] window_eviction {self.agent.window_eviction!r} is not "
                f"recognised. Must be one of {WINDOW_EVICTIONS}."
            )
        if self.agent.cache_layout not in CACHE_LAYOUTS:
            errors.append(
                f"[agent] cache_layout {self.agent.cache_layout!r} is not "
                f"recognised. Must be one of {CACHE_LAYOUTS}."
            )
        window: object = self.agent.window_messages
        if window is not None and (
            isinstance(window, bool)
            or not isinstance(window, int)
            or not WINDOW_MESSAGES_MIN <= window <= WINDOW_MESSAGES_MAX
        ):
            errors.append(
                f"[agent] window_messages must be an int from "
                f"{WINDOW_MESSAGES_MIN} to {WINDOW_MESSAGES_MAX}, got {window!r}"
            )
        turns: object = self.agent.replay_turns
        if turns is not None and (
            isinstance(turns, bool) or not isinstance(turns, int) or turns < 1
        ):
            errors.append(f"[agent] replay_turns must be an int >= 1, got {turns!r}")
        budget = (
            self.agent.context_window,
            self.agent.output_reserve,
            self.agent.safety_margin,
        )
        for name, value in zip(
            ("context_window", "output_reserve", "safety_margin"), budget, strict=True
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                errors.append(
                    f"[agent] {name} must be a positive integer, got {value!r}"
                )
        if any(value is not None for value in budget) and any(
            value is None for value in budget
        ):
            missing = [
                name
                for name, value in zip(
                    ("context_window", "output_reserve", "safety_margin"),
                    budget,
                    strict=True,
                )
                if value is None
            ]
            errors.append("[agent] context budget is missing " + ", ".join(missing))
        if all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in budget
        ):
            window, reserve, margin = budget
            assert window is not None and reserve is not None and margin is not None
            if window - reserve - margin <= 0:
                errors.append(
                    "[agent] context_window - output_reserve - safety_margin "
                    "must be positive"
                )
        names: object = self.agent.tool_env
        if not isinstance(names, list) or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            for name in names
        ):
            errors.append(
                "[agent] tool_env must be a list of valid environment variable names"
            )
        return errors

    def tool_environment(self) -> dict[str, str]:
        """Present variables explicitly named by the config; never persist their values."""
        if errors := self._validate_agent():
            raise ConfigError("; ".join(errors))
        return {
            name: os.environ[name] for name in self.agent.tool_env if name in os.environ
        }

    def _validate_retrieval(self) -> list[str]:
        """``[retrieval]`` holds known values inside their documented ranges."""
        retrieval = self.retrieval
        errors: list[str] = []
        if retrieval.retriever not in RETRIEVAL_MODES:
            errors.append(
                f"[retrieval] retriever {retrieval.retriever!r} is not "
                f"recognised. Must be one of {RETRIEVAL_MODES}."
            )
        if retrieval.placement not in RECALL_PLACEMENTS:
            errors.append(
                f"[retrieval] placement {retrieval.placement!r} is not "
                f"recognised. Must be one of {RECALL_PLACEMENTS}."
            )
        k: object = retrieval.k
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            errors.append(f"[retrieval] k must be an int from 1 to 10, got {k!r}")
        score: object = retrieval.min_score
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not 0 <= score <= 1
        ):
            errors.append(
                f"[retrieval] min_score must be a number from 0 to 1, got {score!r}"
            )
        ratio: object = retrieval.cross_task_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not 1 <= ratio <= 10
        ):
            errors.append("[retrieval] cross_task_ratio must be a number from 1 to 10")
        for name, maximum in (("cross_task_cap", 2), ("cross_task_cold", 1)):
            value = getattr(retrieval, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= maximum
            ):
                errors.append(f"[retrieval] {name} must be an int from 0 to {maximum}")
        for name, allowed in (
            ("recall_at", RECALL_TRIGGERS),
            ("kinds", RECALL_KINDS),
        ):
            values = getattr(retrieval, name)
            if not isinstance(values, list) or any(
                not isinstance(item, str) or item not in allowed for item in values
            ):
                errors.append(
                    f"[retrieval] {name} must be a list of "
                    f"{', '.join(allowed)}, got {values!r}"
                )
        return errors

    def _validate_reasoning_fields(self) -> list[str]:
        """Each backend's reasoning keys are known and its flags are bools."""
        errors: list[str] = []
        for section_name in _LOCAL_SECTIONS + _CLOUD_SECTIONS:
            section = getattr(self, section_name)
            value = section.reasoning_field
            allowed = (
                ("", "thinking", "auto")
                if section_name == "ollama"
                else ("",) + REASONING_FIELDS
            )
            if not isinstance(value, str) or value not in allowed:
                errors.append(
                    f"[{section_name}] reasoning_field {value!r} is not recognised. "
                    f"Must be one of {allowed}."
                )
            if not isinstance(section.requires_reasoning_replay, bool):
                errors.append(
                    f"[{section_name}] requires_reasoning_replay must be a bool, "
                    f"got {section.requires_reasoning_replay!r}"
                )
        return errors

    def _validate_extra_body(self) -> list[str]:
        from localmelo.support.providers.llm._protocol import extra_body_errors

        section = getattr(self, self.chat_backend, None)
        if section is None:
            return []
        return [
            f"[{self.chat_backend}] {err}"
            for err in extra_body_errors(getattr(section, "extra_body", {}))
        ]

    def _validate_embedding_backend(self) -> list[str]:
        from localmelo.support.backends.registry import get_backend

        try:
            backend = get_backend(self.embedding_backend)
            return backend.validate_embedding(self)
        except KeyError:
            return [
                f"embedding_backend {self.embedding_backend!r}: "
                f"no registered adapter found."
            ]

    def validate_or_raise(self) -> None:
        """Validate and raise ConfigError if any issues are found."""
        errors = self.validate()
        if errors:
            msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigError(msg)

    def setting_errors(self) -> list[str]:
        """The errors of the settings alone, which need no backend (``-X``, ``melo config``)."""
        return [
            *self._validate_agent(),
            *self._validate_retrieval(),
            *self._validate_memory(),
            *self._validate_scheduler(),
            *self._validate_browser(),
            *self._validate_tools(),
            *self._validate_gateway_workspace(),
            *self._validate_cli(),
            *self._validate_ui(),
            *self._validate_console(),
            *self._validate_approvals(),
            *self._validate_workspace(),
            *self._validate_request_timeout(),
        ]

    def cli_errors(self) -> list[str]:
        """The ``[cli]`` errors alone, for a path that does not read the rest."""
        return self._validate_cli()

    def ui_errors(self) -> list[str]:
        """The ``[ui]`` errors alone: the mode is display only, so a bad one warns."""
        return self._validate_ui()

    def console_errors(self) -> list[str]:
        """The ``[console]`` errors alone: the language is display only, so a bad one warns."""
        return self._validate_console()

    def pricing_errors(self) -> list[str]:
        """What is wrong in ``[pricing]``; display only, so a bad table warns.

        Each model's table needs ``input`` and ``output`` and may have
        ``cache_read``: numbers of at least 0, dollars per million tokens.
        """
        if not isinstance(self.pricing, dict):
            return [f"[pricing] must be a table, got {self.pricing!r}"]
        errors = []
        for model, price in self.pricing.items():
            if not isinstance(price, dict) or not {"input", "output"} <= set(price):
                errors.append(f"[pricing.{model!r}] needs input and output")
                continue
            for key, value in price.items():
                if key not in PRICE_KEYS:
                    errors.append(f"[pricing.{model!r}] has an unknown key {key!r}")
                elif isinstance(value, bool) or not isinstance(value, int | float):
                    errors.append(f"[pricing.{model!r}] {key} must be a number")
                elif not 0 <= value < float("inf"):
                    errors.append(f"[pricing.{model!r}] {key} must be 0 or more")
        return errors


PRICE_KEYS: Final = ("input", "output", "cache_read")


def has_control_or_surrogate(text: str) -> bool:
    """Whether *text* holds a control character, newline or lone surrogate."""
    return any(
        ord(ch) < 0x20
        or 0x7F <= ord(ch) <= 0x9F
        or ch in ("\\u2028", "\\u2029")
        or 0xD800 <= ord(ch) <= 0xDFFF
        for ch in text
    )


def gateway_workspace_problem(value: object, *, allow_empty: bool = True) -> str | None:
    """Why *value* is not a valid ``[gateway] workspace``, or ``None``."""
    if not isinstance(value, str):
        return f"must be a string, got {value!r}"
    if not value:
        return None if allow_empty else "Must be an absolute path or start with ~/."
    if has_control_or_surrogate(value):
        return "Must be a single line with no control characters."
    if not (os.path.isabs(value) or value.startswith("~/")):
        return "Must be an absolute path or start with ~/."
    resolved = os.path.realpath(os.path.expanduser(value))
    home = os.path.realpath(os.path.expanduser("~"))
    if resolved in ("/", home):
        return "Must not resolve to / or the home folder."
    try:
        if os.path.samefile(resolved, home):
            return "Must not resolve to / or the home folder."
    except OSError:
        pass  # A workspace may be created later.
    return None


def extra_folder_problem(value: object) -> str | None:
    """Why *value* cannot be an entry of ``[workspace] extra_folders``, or ``None``.

    An entry is an absolute path or one that starts with ``~/``, on one
    line, and no file-system alias (macOS ``/.vol``, ``/.resolve``,
    ``/.nofollow``). The folder the file system names for it must not be a
    place melo never opens (keys, its own data, system folders) nor the
    home folder or a folder above it.
    """
    if not isinstance(value, str):
        return f"must be a string, got {value!r}"
    if not value or has_control_or_surrogate(value):
        return "Must be a single line with no control characters."
    if not (os.path.isabs(value) or value.startswith("~/")):
        return "Must be an absolute path or start with ~/."
    if is_alias(os.path.realpath(os.path.expanduser(value))):
        return "Must not be a file-system alias (/.vol, /.resolve, /.nofollow)."
    real = canonical_path(os.path.expanduser(value))
    if above_home(real):
        return "Must not be /, the home folder or a folder above it."
    if never_offered(real) is not None:
        return "Must not be a folder melo never opens (keys, its own data, system folders)."
    return None


def extra_folders_errors(value: object) -> list[str]:
    """What is wrong with ``[workspace] extra_folders``."""
    if not isinstance(value, list):
        return [f"[workspace] extra_folders must be a list, got {value!r}"]
    if len(value) > EXTRA_FOLDERS_MAX:
        return [f"[workspace] extra_folders holds at most {EXTRA_FOLDERS_MAX} folders"]
    errors = []
    for item in value:
        problem = extra_folder_problem(item)
        if problem is not None:
            errors.append(f"[workspace] extra_folders entry {item!r}: {problem}")
    return errors


def folder_label(path: str) -> str:
    """*path* as ``extra_folders`` keeps it: ``~/`` for the home folder, else absolute."""
    home = canonical_path(os.path.expanduser("~"))
    if path.startswith(home + "/"):
        return "~/" + path[len(home) + 1 :]
    return path


def extra_folders(cfg: Config) -> list[str]:
    """The valid entries of ``[workspace] extra_folders``, as absolute paths.

    ``~`` is expanded, nothing else: a folder that became a symlink since
    it was kept does not cover what it points to. An invalid entry is left
    out, and so is a second one for the same folder.
    """
    value = cfg.workspace.extra_folders
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:EXTRA_FOLDERS_MAX]:
        if extra_folder_problem(item) is None:
            path = os.path.normpath(os.path.expanduser(item))
            if path not in out:
                out.append(path)
    return out


def add_extra_folder(folder: str) -> bool:
    """Keep the real folder *folder* in ``[workspace] extra_folders`` and save.

    Whether it was added: ``False`` when it is there already. ``ConfigError``
    when it may not be kept (see :func:`extra_folder_problem`).
    """
    real = canonical_path(folder)
    label = folder_label(real)
    problem = extra_folder_problem(label)
    if problem is not None:
        raise ConfigError(f"[workspace] extra_folders entry {label!r}: {problem}")
    cfg = load()
    kept = cfg.workspace.extra_folders
    if not isinstance(kept, list):
        kept = []
    if real in extra_folders(cfg) or label in kept:
        return False
    if len(kept) >= EXTRA_FOLDERS_MAX:
        raise ConfigError(
            f"[workspace] extra_folders holds at most {EXTRA_FOLDERS_MAX} folders"
        )
    cfg.workspace.extra_folders = [*kept, label]
    save(cfg)
    return True


class SavedFolders:
    """The extra folders in ``config.toml``, read on every use (#405).

    A folder removed in Settings asks again from the next call on; one
    added from a chat counts in every surface that reads the file.
    """

    def folders(self) -> list[str]:
        return extra_folders(load())

    def add(self, folder: str) -> bool:
        return add_extra_folder(folder)


def _is_positive_seconds(value: object) -> bool:
    """Whether *value* is a finite number > 0.

    An integer counts (TOML reads ``600`` as one); a bool does not, and
    neither does an integer too large to be a float.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        seconds = float(value)
    except OverflowError:
        return False
    return math.isfinite(seconds) and seconds > 0


# -- Compatibility aliases (temporary, reduces churn in files not yet migrated) --
MlcConfig = LocalBackendConfig
OllamaConfig = LocalBackendConfig
VllmConfig = LocalBackendConfig
SglangConfig = LocalBackendConfig

# -- Section names for iteration --

_LOCAL_SECTIONS = ("mlc", "ollama", "vllm", "sglang")
_CLOUD_SECTIONS = ("openai", "gemini", "anthropic", "nvidia")

# -- Migration helpers --

_CLOUD_API_PROVIDER_MIGRATION: dict[str, str] = {
    "openai": "openai",
    "gemini": "gemini",
    "anthropic": "anthropic",
    "nvidia": "nvidia",
}


# Settings ``-X KEY=VALUE`` gives this process: ``load()`` applies them, and ``save()``
# never writes them. A key is ``section.name``, or a top-level name without a dot.
_OVERRIDES: dict[str, Any] = {}


def set_overrides(values: Mapping[str, Any]) -> None:
    """Replace the settings this process runs with (empty: none)."""
    _OVERRIDES.clear()
    _OVERRIDES.update(values)


def overrides() -> dict[str, Any]:
    return dict(_OVERRIDES)


def _holder(cfg: Config, key: str) -> tuple[Any, str] | None:
    """The object that holds *key*, and its name there; None if *cfg* has no such field."""
    owner, _, name = key.rpartition(".")
    target = getattr(cfg, owner, None) if owner else cfg
    return (target, name) if target is not None and hasattr(target, name) else None


def get_key(cfg: Config, key: str) -> Any:
    holder = _holder(cfg, key)
    return getattr(*holder) if holder else None


def set_key(cfg: Config, key: str, value: Any) -> bool:
    """Set *key* on *cfg*; False when *cfg* has no such field."""
    holder = _holder(cfg, key)
    if holder:
        setattr(*holder, value)
    return holder is not None


def load() -> Config:
    cfg = _load_file()
    for key, value in _OVERRIDES.items():
        original = get_key(cfg, key)
        if set_key(cfg, key, value):
            cfg._overridden[key] = (original, value)
    return cfg


def _as_in_file(cfg: Config) -> Config:
    """*cfg* with each ``-X`` value that is still in place put back to what the file had.

    A value the caller changed since ``load()`` is theirs and is kept.
    """
    if not cfg._overridden:
        return cfg
    saved = copy.deepcopy(cfg)
    saved._overridden = {}
    for key, (original, applied) in cfg._overridden.items():
        if get_key(cfg, key) == applied:
            set_key(saved, key, original)
    return saved


def _load_file() -> Config:
    path = config_path()
    if not os.path.isfile(path):
        return Config()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    cfg = Config()

    # -- Migration: old single `backend` field --
    if "backend" in data and "chat_backend" not in data:
        cfg = _migrate_legacy(data)
        return cfg

    # -- Migration: old `mlc-llm` / `cloud_api` keys in chat_backend/embedding_backend --
    raw_chat = data.get("chat_backend", "")
    raw_emb = data.get("embedding_backend", "")
    cfg.chat_backend = _migrate_backend_key(raw_chat, data)
    cfg.embedding_backend = _migrate_embedding_key(raw_emb)
    # "omlx" is another spelling of "vllm" (one backend, one [vllm] table):
    # everything reads "vllm", and save() writes the spelling back.
    cfg.chat_alias = raw_chat if raw_chat == OMLX_ALIAS else ""
    cfg.embedding_alias = raw_emb if raw_emb == OMLX_ALIAS else ""

    # -- Load local backend sections --
    for section in _LOCAL_SECTIONS:
        if section in data:
            target = getattr(cfg, section)
            for k, v in data[section].items():
                if hasattr(target, k):
                    setattr(target, k, v)
            # Migration: flat chat_template_kwargs -> extra_body
            _migrate_extra_body(target, data[section])
        # Migration: old [mlc] section with chat_port -> chat_url
        if section == "mlc" and "mlc" in data:
            _migrate_mlc_section(cfg.mlc, data["mlc"])

    # -- Load cloud vendor sections --
    for section in _CLOUD_SECTIONS:
        if section in data:
            target = getattr(cfg, section)
            for k, v in data[section].items():
                if hasattr(target, k):
                    setattr(target, k, v)
            # Migration: flat chat_template_kwargs -> extra_body
            _migrate_extra_body(target, data[section])

    # Migration: old [cloud_api] section -> vendor-specific section
    if "cloud_api" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["cloud_api"])
    elif "online" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["online"])

    if "gateway" in data:
        for k, v in data["gateway"].items():
            if hasattr(cfg.gateway, k):
                setattr(cfg.gateway, k, v)

    if isinstance(data.get("agent"), dict):
        for k, v in data["agent"].items():
            if hasattr(cfg.agent, k):
                setattr(cfg.agent, k, v)

    # A [scheduler] that is not a table is ignored, like an unknown key; a
    # malformed value inside a real table is left for Config.validate.
    if isinstance(data.get("retrieval"), dict):
        for k, v in data["retrieval"].items():
            if hasattr(cfg.retrieval, k):
                setattr(cfg.retrieval, k, v)

    if isinstance(data.get("memory"), dict):
        for k, v in data["memory"].items():
            if hasattr(cfg.memory, k):
                setattr(cfg.memory, k, v)

    if isinstance(data.get("scheduler"), dict):
        for k, v in data["scheduler"].items():
            if hasattr(cfg.scheduler, k):
                setattr(cfg.scheduler, k, v)

    if isinstance(data.get("browser"), dict):
        for k, v in data["browser"].items():
            if hasattr(cfg.browser, k):
                setattr(cfg.browser, k, v)

    if isinstance(data.get("tools"), dict):
        for k, v in data["tools"].items():
            if hasattr(cfg.tools, k):
                setattr(cfg.tools, k, v)

    # A bad value is left for Config.validate to report.
    if isinstance(data.get("sessions"), dict):
        cfg.sessions.resume_mode = data["sessions"].get("resume_mode", "pairs")
        if data["sessions"].get("record_direct") is False:
            cfg.sessions.record_direct = False

    if isinstance(data.get("cli"), dict):
        section = data["cli"]
        if "stream" in section:
            cfg.cli.stream = section["stream"]
        if "status_line" in section:
            # A bad value is kept so Config.validate can report it.
            cfg.cli.status_line = section["status_line"]
        for key in ("colors", "background"):
            if key in section:  # a bad value is kept for Config.validate
                value = section[key]
                plain_int = isinstance(value, int) and not isinstance(value, bool)
                setattr(cfg.cli, key, str(value) if plain_int else value)
        if "reasoning_tags" in section:
            tags = section["reasoning_tags"]
            # Keep a malformed value so Config.validate can report it.
            cfg.cli.reasoning_tags = dict(tags) if isinstance(tags, dict) else tags

    if "pricing" in data:  # a bad table is kept for Config.pricing_errors
        pricing = data["pricing"]
        cfg.pricing = dict(pricing) if isinstance(pricing, dict) else pricing

    # A bad mode is kept so Config.ui_errors can report it.
    if isinstance(data.get("ui"), dict):
        cfg.ui.mode = data["ui"].get("mode", cfg.ui.mode)
    # A bad language is kept for Config.console_errors; the console reads it as auto.
    if isinstance(data.get("console"), dict):
        cfg.console.language = data["console"].get("language", cfg.console.language)
    if isinstance(data.get("approvals"), dict):
        cfg.approvals.mode = data["approvals"].get("mode", cfg.approvals.mode)
    # A bad value is kept for Config.setting_errors; extra_folders() ignores it.
    if isinstance(data.get("workspace"), dict):
        found = data["workspace"].get("extra_folders", [])
        cfg.workspace.extra_folders = list(found) if isinstance(found, list) else found

    cfg.request_timeout_s = data.get("request_timeout_s")
    cfg.mcp_config = str(data.get("mcp_config") or "")

    return cfg


def _migrate_backend_key(raw: str, data: dict) -> str:
    """Migrate old chat_backend keys to new ones."""
    if raw == "mlc-llm":
        return "mlc"
    if raw == OMLX_ALIAS:
        return "vllm"
    if raw == "cloud_api":
        provider = data.get("cloud_api", {}).get("provider", "")
        return _CLOUD_API_PROVIDER_MIGRATION.get(provider, raw)
    return raw


def _migrate_embedding_key(raw: str) -> str:
    """Migrate old embedding_backend keys to new ones."""
    if raw == "mlc-llm":
        return "mlc"
    if raw == OMLX_ALIAS:
        return "vllm"
    return raw


def _migrate_mlc_section(mlc: LocalBackendConfig, section: dict) -> None:
    """Migrate old [mlc] with chat_port/gpu_memory_gb to URL-based config."""
    if "chat_port" in section and not mlc.chat_url:
        port = section["chat_port"]
        mlc.chat_url = f"http://127.0.0.1:{port}/v1"
        if not mlc.embedding_url:
            mlc.embedding_url = mlc.chat_url


def _migrate_extra_body(
    target: LocalBackendConfig | CloudVendorConfig, section: Mapping[str, Any]
) -> None:
    """Fold a flat ``chat_template_kwargs`` table into ``extra_body``.

    ``extra_body`` is the canonical form; a section-level
    ``chat_template_kwargs = {...}`` is accepted as shorthand and is
    written back in canonical form by :func:`save`. An explicit
    ``extra_body.chat_template_kwargs`` wins over the flat key. A bad
    ``extra_body`` value is left for :meth:`Config.validate` to report.
    """
    flat = section.get("chat_template_kwargs")
    if not isinstance(flat, dict) or not isinstance(target.extra_body, dict):
        return
    if "chat_template_kwargs" not in target.extra_body:
        target.extra_body = {**target.extra_body, "chat_template_kwargs": flat}


def _migrate_cloud_api_section(cfg: Config, section: dict) -> None:
    """Migrate old [cloud_api] or [online] section to vendor-specific config."""
    vendor = cfg.chat_backend
    target = getattr(cfg, vendor, None)
    if target is None:
        return
    for k, v in section.items():
        if k in ("provider", "local_embedding"):
            continue
        if hasattr(target, k):
            setattr(target, k, v)
    _migrate_extra_body(target, section)


def _migrate_legacy(data: dict) -> Config:
    """Migrate the oldest format with a single `backend` field."""
    cfg = Config()
    old = data["backend"]

    if old == "mlc-llm":
        cfg.chat_backend = "mlc"
        cfg.embedding_backend = "mlc"
        if "mlc" in data:
            for k, v in data["mlc"].items():
                if hasattr(cfg.mlc, k):
                    setattr(cfg.mlc, k, v)
            _migrate_mlc_section(cfg.mlc, data["mlc"])
            _migrate_extra_body(cfg.mlc, data["mlc"])
    elif old == "online":
        online_sec = data.get("online", {})
        provider = online_sec.get("provider", "")
        cfg.chat_backend = _CLOUD_API_PROVIDER_MIGRATION.get(provider, "")
        if online_sec.get("local_embedding", False):
            cfg.embedding_backend = "mlc"
        else:
            cfg.embedding_backend = "none"
        if cfg.chat_backend in _CLOUD_SECTIONS:
            _migrate_cloud_api_section(cfg, online_sec)
    elif old == "ollama":
        cfg.chat_backend = "ollama"
        ollama_sec = data.get("ollama", {})
        cfg.embedding_backend = (
            "ollama" if ollama_sec.get("embedding_model", "") else "none"
        )
        if "ollama" in data:
            for k, v in data["ollama"].items():
                if hasattr(cfg.ollama, k):
                    setattr(cfg.ollama, k, v)
            _migrate_extra_body(cfg.ollama, data["ollama"])

    if "gateway" in data:
        for k, v in data["gateway"].items():
            if hasattr(cfg.gateway, k):
                setattr(cfg.gateway, k, v)

    if isinstance(data.get("agent"), dict):
        for k, v in data["agent"].items():
            if hasattr(cfg.agent, k):
                setattr(cfg.agent, k, v)

    if isinstance(data.get("sessions"), dict):
        cfg.sessions.resume_mode = data["sessions"].get("resume_mode", "pairs")
        if data["sessions"].get("record_direct") is False:
            cfg.sessions.record_direct = False

    cfg.request_timeout_s = data.get("request_timeout_s")

    return cfg


_BARE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_string(text: str) -> str:
    # JSON escapes the quotation mark, the backslash and the control
    # characters below U+0020 the way a TOML basic string does; TOML also
    # requires DEL to be escaped. Everything else is written as is: an
    # ASCII-only JSON literal would escape characters outside the BMP as
    # UTF-16 surrogate pairs, which TOML rejects.
    return json.dumps(text, ensure_ascii=False).replace("\x7f", "\\u007f")


def _toml_value(value: Any) -> str:
    """Render *value* as a single-line TOML value (inline tables for dicts)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            k = str(key)
            k = k if _BARE_TOML_KEY.match(k) else _toml_string(k)
            items.append(f"{k} = {_toml_value(item)}")
        return "{" + ", ".join(items) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ConfigError(f"Cannot write {type(value).__name__} value {value!r} to TOML")


def _spelling(backend: str, alias: str) -> str:
    """The backend key as the file spells it: the alias stands for ``vllm``."""
    return alias if backend == "vllm" and alias else backend


def save(cfg: Config) -> None:
    cfg = _as_in_file(cfg)
    os.makedirs(config_dir(), exist_ok=True)

    lines: list[str] = []
    chat = _spelling(cfg.chat_backend, cfg.chat_alias)
    embedding = _spelling(cfg.embedding_backend, cfg.embedding_alias)
    lines.append(f"chat_backend = {_toml_string(chat)}")
    lines.append(f"embedding_backend = {_toml_string(embedding)}")
    # Top-level keys go before the first table. Only a value that is set
    # is written, so a config without one is saved as it was before.
    if cfg.request_timeout_s is not None:
        lines.append(f"request_timeout_s = {_toml_value(cfg.request_timeout_s)}")
    if cfg.mcp_config:
        lines.append(f"mcp_config = {_toml_string(cfg.mcp_config)}")
    lines.append("")

    # Local backend sections
    for section in _LOCAL_SECTIONS:
        sec = getattr(cfg, section)
        lines.append(f"[{section}]")
        lines.append(f"chat_url = {_toml_string(sec.chat_url)}")
        lines.append(f"chat_model = {_toml_string(sec.chat_model)}")
        lines.append(f"embedding_url = {_toml_string(sec.embedding_url)}")
        lines.append(f"embedding_model = {_toml_string(sec.embedding_model)}")
        if sec.extra_body:
            lines.append(f"extra_body = {_toml_value(sec.extra_body)}")
        if sec.reasoning_field:
            lines.append(f"reasoning_field = {_toml_string(sec.reasoning_field)}")
        if sec.requires_reasoning_replay:
            lines.append("requires_reasoning_replay = true")
        lines.append("")

    # Cloud vendor sections
    for section in _CLOUD_SECTIONS:
        sec = getattr(cfg, section)
        lines.append(f"[{section}]")
        lines.append(f"api_key_env = {_toml_string(sec.api_key_env)}")
        lines.append(f"chat_model = {_toml_string(sec.chat_model)}")
        lines.append(f"base_url = {_toml_string(sec.base_url)}")
        if sec.extra_body:
            lines.append(f"extra_body = {_toml_value(sec.extra_body)}")
        if sec.reasoning_field:
            lines.append(f"reasoning_field = {_toml_string(sec.reasoning_field)}")
        if sec.requires_reasoning_replay:
            lines.append("requires_reasoning_replay = true")
        lines.append("")

    lines.append("[gateway]")
    lines.append(f"port = {cfg.gateway.port}")
    lines.append(f"host = {_toml_string(cfg.gateway.host)}")
    if cfg.gateway.workspace:
        lines.append(f"workspace = {_toml_string(cfg.gateway.workspace)}")
    lines.append("")

    defaults = AgentConfig()
    if (
        cfg.agent.reasoning != "legacy"
        or cfg.agent.replay_turns is not None
        or cfg.agent.note_writer != "auto"
        or cfg.agent.tool_env
        or cfg.agent.fixed_prefix != defaults.fixed_prefix
        or cfg.agent.window_eviction != defaults.window_eviction
        or cfg.agent.reflect_tool_guard != defaults.reflect_tool_guard
        or cfg.agent.cache_layout != defaults.cache_layout
        or cfg.agent.window_messages != defaults.window_messages
        or cfg.agent.captured_history
        or cfg.agent.brief_items
        or cfg.agent.lean_output
        or cfg.agent.confine_writes != defaults.confine_writes
        or cfg.agent.confine_direct_runs
        or cfg.agent.answer_on_stop != defaults.answer_on_stop
        or cfg.agent.max_calls is not None
        or any(
            getattr(cfg.agent, name) != getattr(defaults, name) for name in IMAGE_KEYS
        )
        or cfg.agent.context_window is not None
        or cfg.agent.output_reserve is not None
        or cfg.agent.safety_margin is not None
    ):
        lines.append("[agent]")
        lines.append(f"reasoning = {_toml_string(cfg.agent.reasoning)}")
        if cfg.agent.replay_turns is not None:
            lines.append(f"replay_turns = {cfg.agent.replay_turns}")
        if cfg.agent.note_writer != "auto":
            lines.append(f"note_writer = {_toml_string(cfg.agent.note_writer)}")
        if cfg.agent.fixed_prefix != defaults.fixed_prefix:
            lines.append(f"fixed_prefix = {_toml_value(cfg.agent.fixed_prefix)}")
        if cfg.agent.window_eviction != defaults.window_eviction:
            lines.append(f"window_eviction = {_toml_string(cfg.agent.window_eviction)}")
        if cfg.agent.reflect_tool_guard != defaults.reflect_tool_guard:
            lines.append(
                f"reflect_tool_guard = {_toml_value(cfg.agent.reflect_tool_guard)}"
            )
        if cfg.agent.cache_layout != defaults.cache_layout:
            lines.append(f"cache_layout = {_toml_string(cfg.agent.cache_layout)}")
        if cfg.agent.window_messages != defaults.window_messages:
            lines.append(f"window_messages = {_toml_value(cfg.agent.window_messages)}")
        if cfg.agent.captured_history:
            lines.append("captured_history = true")
        if cfg.agent.brief_items:
            lines.append("brief_items = true")
        if cfg.agent.lean_output:
            lines.append("lean_output = true")
        if cfg.agent.confine_writes != defaults.confine_writes:
            lines.append(f"confine_writes = {_toml_value(cfg.agent.confine_writes)}")
        if cfg.agent.confine_direct_runs:
            lines.append("confine_direct_runs = true")
        if cfg.agent.answer_on_stop != defaults.answer_on_stop:
            lines.append(f"answer_on_stop = {_toml_value(cfg.agent.answer_on_stop)}")
        if cfg.agent.max_calls is not None:
            lines.append(f"max_calls = {_toml_value(cfg.agent.max_calls)}")
        for name in IMAGE_KEYS:
            if getattr(cfg.agent, name) != getattr(defaults, name):
                lines.append(f"{name} = {_toml_value(getattr(cfg.agent, name))}")
        if cfg.agent.tool_env:
            lines.append(f"tool_env = {_toml_value(cfg.agent.tool_env)}")
        for name in ("context_window", "output_reserve", "safety_margin"):
            value = getattr(cfg.agent, name)
            if value is not None:
                lines.append(f"{name} = {_toml_value(value)}")
        lines.append("")

    if cfg.memory != MemoryConfig():
        lines.append("[memory]")
        if cfg.memory.admit_on_outcome:
            lines.append("admit_on_outcome = true")
        if not cfg.memory.across_chats:
            lines.append("across_chats = false")
        if cfg.memory.retriever != "auto":
            lines.append(f"retriever = {_toml_string(cfg.memory.retriever)}")
        if cfg.memory.keep_statements:
            lines.append("keep_statements = true")
        if cfg.memory.sleep != "off":
            lines.append(f"sleep = {_toml_string(cfg.memory.sleep)}")
        if cfg.memory.sleep_lock:
            lines.append(f"sleep_lock = {_toml_string(cfg.memory.sleep_lock)}")
        for name in RECALL_KEYS:
            if getattr(cfg.memory, name) != getattr(MemoryConfig(), name):
                lines.append(f"{name} = {_toml_value(getattr(cfg.memory, name))}")
        lines.append("")

    if cfg.browser != BrowserConfig():
        lines.extend(["[browser]", f"driver = {_toml_string(cfg.browser.driver)}", ""])

    if cfg.tools != ToolsConfig():
        lines.extend(["[tools]", f"web_fetch = {_toml_value(cfg.tools.web_fetch)}", ""])

    if cfg.sessions != SessionsConfig():
        lines.append("[sessions]")
        if cfg.sessions.resume_mode != "pairs":
            lines.append(f"resume_mode = {_toml_value(cfg.sessions.resume_mode)}")
        if not cfg.sessions.record_direct:
            lines.append("record_direct = false")
        lines.append("")
    if cfg.cli != CliConfig():
        lines.append("[cli]")
        if cfg.cli.stream != CliConfig().stream:  # the default is not pinned
            lines.append(f"stream = {'true' if cfg.cli.stream else 'false'}")
        if cfg.cli.status_line != CliConfig().status_line:
            lines.append(f"status_line = {_toml_value(cfg.cli.status_line)}")
        for key in ("colors", "background"):
            if getattr(cfg.cli, key) != "auto":
                lines.append(f"{key} = {_toml_value(getattr(cfg.cli, key))}")
        if cfg.cli.reasoning_tags:
            lines.append(f"reasoning_tags = {_toml_value(cfg.cli.reasoning_tags)}")
        lines.append("")
    if cfg.ui != UiConfig():
        lines.extend(["[ui]", f"mode = {_toml_value(cfg.ui.mode)}", ""])
    if cfg.console != ConsoleConfig():
        lines.extend(
            ["[console]", f"language = {_toml_value(cfg.console.language)}", ""]
        )
    if cfg.approvals != ApprovalsConfig():
        lines.extend(["[approvals]", f"mode = {_toml_value(cfg.approvals.mode)}", ""])
    # Only a list that holds a folder is written; a bad value is kept as it is.
    if cfg.workspace.extra_folders:
        folders = cfg.workspace.extra_folders
        lines.extend(["[workspace]", f"extra_folders = {_toml_value(folders)}", ""])
    retrieval = cfg.retrieval
    if retrieval != RetrievalConfig():
        lines.append("[retrieval]")
        lines.append(f"retriever = {_toml_string(retrieval.retriever)}")
        if retrieval.recall_at:
            lines.append(f"recall_at = {_toml_value(retrieval.recall_at)}")
        lines.append(f"k = {retrieval.k}")
        if retrieval.kinds:
            lines.append(f"kinds = {_toml_value(retrieval.kinds)}")
        lines.append(f"placement = {_toml_string(retrieval.placement)}")
        lines.append(f"min_score = {_toml_value(retrieval.min_score)}")
        lines.append(f"cross_task_ratio = {_toml_value(retrieval.cross_task_ratio)}")
        lines.append(f"cross_task_cap = {retrieval.cross_task_cap}")
        lines.append(f"cross_task_cold = {retrieval.cross_task_cold}")
        lines.append("")

    # Only a settings section that differs from the defaults is written, so
    # an existing config is saved exactly as before and a reconfigure keeps
    # a non-default scheduler (decision D7).
    scheduler = cfg.scheduler
    if scheduler != SchedulerConfig():
        lines.append("[scheduler]")
        lines.append(f"policy = {_toml_string(scheduler.policy)}")
        lines.append(f"n_srv = {scheduler.n_srv}")
        lines.append(f"r_f = {scheduler.r_f}")
        lines.append(f"n_fg = {scheduler.n_fg}")
        lines.append(f"focus_s = {_toml_value(scheduler.focus_s)}")
        lines.append(f"call_log = {'true' if scheduler.call_log else 'false'}")
        lines.append(f"call_log_dir = {_toml_string(scheduler.call_log_dir)}")
        lines.append(f"gpu_lock_file = {_toml_string(scheduler.gpu_lock_file)}")
        lines.append(f"lock_self = {_toml_string(scheduler.lock_self)}")
        lines.append(f"lock_gate = {_toml_string(scheduler.lock_gate)}")
        lines.append("")

    # Kept as written, even when invalid (it only warns), so a save loses nothing.
    for model, price in (cfg.pricing if isinstance(cfg.pricing, dict) else {}).items():
        if isinstance(price, dict):
            lines.append(f"[pricing.{_toml_string(str(model))}]")
            lines += [
                f"{_toml_string(str(k))} = {_toml_value(v)}" for k, v in price.items()
            ]
            lines.append("")

    # TOML files are UTF-8. Encode before opening the file, so text that
    # cannot be encoded (a str holding a lone surrogate) fails the save
    # without truncating the existing config. Write a temporary file in
    # CONFIG_DIR and move it into place with os.replace so a concurrent
    # load() never reads half a file and a failed write leaves the old file.
    text = "\n".join(lines)
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        bad = exc.object[exc.start : exc.end]
        raise ConfigError(f"Cannot write {bad!r} to TOML: {exc.reason}") from exc
    target_path = os.path.realpath(config_path())
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    old_mode: int | None = None
    with contextlib.suppress(OSError):
        old_mode = stat.S_IMODE(os.stat(target_path).st_mode)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".config.toml.", suffix=".tmp", dir=target_dir
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if old_mode is not None:
            with contextlib.suppress(OSError):
                os.chmod(tmp_path, old_mode)
        os.replace(tmp_path, target_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
