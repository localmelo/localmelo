"""Persistent configuration stored at ~/.cache/localmelo/config.toml."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import os
import re
import stat
import tempfile
import tomllib
import types
import typing
from collections.abc import Callable, Container, Mapping
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
EMPTY_REPLY_FALLBACKS = ("on", "off")
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
# The room one recalled chat turn takes in the block and in the search tool's
# list, and which part of a turn is shown: see MemoryConfig.recall_item_chars.
MEMORY_RECALL_ITEM_CHARS = (120, 2000)
MEMORY_RECALL_EXCERPT = ("head", "match")
# How recall shows earlier chat turns: as excerpts, or as a list of titles the model
# opens, and the list form's two sizes: see MemoryConfig.recall_form.
MEMORY_RECALL_FORM = ("excerpt", "list")
MEMORY_RECALL_LIST_MAX = (1, 50)
MEMORY_RECALL_OPEN_CHARS = (500, 16000)
# Whether a turn's excerpt that shows less than the whole turn carries its id, so the
# model can open the whole turn (on, the default; off is the developer comparison):
# see MemoryConfig.excerpt_open.
MEMORY_EXCERPT_OPEN = ("off", "on")
# Whether recall draws on the chat turns or on memory items encoded from them, and
# whether an opened item also shows its source turns; the size of the window a chat
# is cut into for that encoding: see MemoryConfig.recall_source.
MEMORY_RECALL_SOURCE = ("turns", "items")
MEMORY_OPEN_SOURCES = ("off", "on")
MEMORY_COMPRESS_AT_CHARS = (4000, 60000)
# How many windows of one ingest the encoder is asked for at once: see
# MemoryConfig.compress_parallel.
MEMORY_COMPRESS_PARALLEL = (1, 8)
# The bars an item must pass to be recalled, and their bounds (inclusive): see
# MemoryConfig.recall_min_cosine and recall_min_coverage.
MEMORY_RECALL_MIN_COSINE = (0.20, 0.95)
MEMORY_RECALL_MIN_COVERAGE = (0.10, 1.00)
# How recall looks wider for a question that counts, totals or compares, and the
# vector bar it uses then: see MemoryConfig.gather and gather_min_cosine.
MEMORY_GATHER = ("off", "wide", "guide")
MEMORY_GATHER_DEFAULT = "guide"
MEMORY_GATHER_MIN_COSINE = (0.20, 0.55)
# How far back a new chat's first turn looks into the other chats: see
# ChatConfig.recent_digest_minutes (0 turns it off).
CHAT_RECENT_DIGEST_MINUTES = (0, 1440)
# Whether recall carries notes, and what a turn that recalls nothing is told.
MEMORY_RECALL_NOTES = ("on", "off")
MEMORY_SEARCH_WHEN_EMPTY = ("off", "hint")
# Where a note the chat's model keeps may come from, in four cumulative levels,
# and the bar a note's meaning must clear against its sources: see
# MemoryConfig.note_gate.
MEMORY_NOTE_GATE = ("typed", "pointed", "meaning", "model")
MEMORY_NOTE_MIN_COSINE = (0.20, 0.95)
# What a message with no content of its own ("go look it up") is searched with:
# the user's words of the chat's last turns, or its own words: see
# MemoryConfig.followup_query.
MEMORY_FOLLOWUP_QUERY = ("recent", "message")
# Whether a task reads its project's own instruction file: see MemoryConfig.
MEMORY_PROJECT_INSTRUCTIONS = ("on", "off")
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

    api_key_env: str = ""  # env var name, e.g. OPENAI_API_KEY; wins when it is set
    chat_model: str = ""
    base_url: str = ""  # optional override; each vendor has a built-in default
    # A reference to the key in the macOS keychain, "keychain:localmelo.<name>"
    # (see support.keychain); the key itself is never in the config.
    api_key: str = ""
    # Whether the model thinks before it answers, for a service with such a
    # switch (see support.hosted): "" follows the preset's default for the
    # model, "on" and "off" decide.
    thinking: str = ""
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
    routine steps run; a step the person did not ask for asks on the chat's
    card while someone sees the chat, and the riskiest are blocked and
    explained. Each chat keeps its own mode.

    ``auto_asks`` (developer setting, #465): ``false`` never asks in Auto,
    so every step it stops is blocked, as before #465 (for benchmarks).
    """

    mode: str = "ask"  # ask | auto
    auto_asks: bool = True


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
    # Developer setting: a turn that ends with no answer from a model that did
    # think (an empty reply, or a reply that ends inside its thinking) is asked
    # once more with thinking off where the service has a switch for it, and
    # then answered with the end of its thinking, marked as such
    # (AgentOptions.empty_reply_fallback). "off" ends such a run as failed.
    empty_reply_fallback: str = "on"  # on | off
    # Reflect when stuck after many steps (AgentOptions.attempt_policy): true runs
    # attempts of steps_per_attempt steps with a reflection between them; false
    # never reflects and runs straight to the run's total.
    reflection: bool = True
    # Developer setting: steps in one attempt before a reflection
    # (AgentOptions.steps_per_attempt). None: the built-in (STEPS_PER_ATTEMPT, 50).
    steps_per_attempt: int | None = None

    def window_for(self, backend: str) -> int:
        """The working window's cap: ``window_messages``, else *backend*'s default."""
        if self.window_messages is not None:
            return self.window_messages
        return default_window_messages(backend)


def tool_images_errors(agent: AgentConfig) -> list[str]:
    """What is wrong with ``[agent] tool_images`` and its two counts."""
    errors: list[str] = []
    if agent.tool_images not in TOOL_IMAGES:
        errors.append(
            f"[agent] tool_images {agent.tool_images!r} is not recognised. "
            f"Must be one of {TOOL_IMAGES}."
        )
    for name in ("tool_images_keep", "tool_images_retire"):
        value = getattr(agent, name)
        if value < 1:
            errors.append(f"[agent] {name} must be an int >= 1, got {value!r}")
    return errors


def empty_reply_fallback_errors(agent: AgentConfig) -> list[str]:
    """What is wrong with ``[agent] empty_reply_fallback``."""
    if agent.empty_reply_fallback in EMPTY_REPLY_FALLBACKS:
        return []
    return [
        f"[agent] empty_reply_fallback {agent.empty_reply_fallback!r} is not "
        f"recognised. Must be one of {EMPTY_REPLY_FALLBACKS}."
    ]


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
    settings shape the recall block a chat message carries: ``recall_records``
    notes (3 by default) and ``recall_docs`` chat turns (2) at most (0-20 each) in a
    block of at most ``recall_chars`` characters (200-8000, 4000 by default);
    ``recall_notes = "off"`` leaves notes out of the block and the
    pinned lines, so only turns are recalled; and ``search_when_empty = "hint"``
    adds one line to a block that recalls nothing, telling the model to call the
    memory search tool first. Two more set the bars an item must pass to be
    recalled at all, in the block and in the memory search tool:
    ``recall_min_cosine`` (0.20-0.95), the least similarity of meaning, and
    ``recall_min_coverage`` (0.10-1.00), the least share of the question's
    content words that a note or turn must hold. Two more set how much of a
    recalled chat turn is shown, in the block and in the search tool:
    ``recall_item_chars`` (120-2000, 900 by default), the room one turn takes, and
    ``recall_excerpt``, ``head`` (the start of each part, the default) or
    ``match`` (the sentences of each part that hold most of the question's
    content words). Three more set how recalled chat turns are shown at all:
    ``recall_form``, ``excerpt`` (an excerpt of each turn, the default) or
    ``list`` (one line per turn, with its id, date, chat, a title and its
    similarity, and a tool, ``open_memory``, with which the model opens the turns
    it needs in full). Notes are shown as before in both forms. The list form
    lists every turn that passes the bars, ``recall_list_max`` (1-50) at most, in
    the block and in the search tool alike, and ``recall_open_chars`` (500-16000)
    is the room one opened turn takes; ``recall_docs`` and ``recall_chars`` do not
    bound the list, and ``recall_item_chars`` and ``recall_excerpt`` apply to the
    excerpt form only. ``excerpt_open``, ``on`` (the default) or ``off``, is for the
    excerpt form of chat turns: with ``on`` a line that shows less than the whole
    turn (an excerpt that was cut, or a piece of a long turn) starts with the
    turn's id, such as ``[t123]``, a line under the items says so, and the model
    has ``open_memory`` to open the turn whole within ``recall_open_chars``; a line
    that shows the whole turn reads as before; ``off`` is the developer comparison
    (the block, the search result and the tools as before). It has no effect with
    ``recall_form = "list"`` (which always offers ``open_memory``) or
    ``recall_source = "items"``. Four more are about memory items (#428). ``recall_source``,
    ``turns`` (recall draws on the chat turns, the default) or ``items`` (it draws
    on the memory items a chat model encoded from the chats, in place of the turns;
    ``melo memory ingest FILE -X ingest=compress`` writes them); ``open_sources``,
    ``off`` (the default: an opened item shows its title and text only) or ``on``
    (it also shows its source turns whole, within ``recall_open_chars``);
    ``compress_at_chars`` (4000-60000), the size of the window a chat is cut into
    for the chat model to encode, one model call each; ``compress_parallel`` (1-8,
    default 1), how many of those windows ``melo memory ingest`` has the chat
    model read at once (the items are the same, only the time changes); and ``compress_cache``, the
    path of the file that keeps each window's reply (empty: ``compress_cache.jsonl``
    in the memory folder). Two more are about where a note the chat's model keeps
    with ``remember`` may come from (#442). ``note_gate``, ``typed`` (the words the
    user typed in the chat and a finished task's result shown in it, the default),
    ``pointed`` (also a page or file the user pointed to in the chat), ``meaning``
    (also a note that says the same in other words or another language: its key
    tokens must be found and its meaning must be close to a source) or ``model``
    (also a note the model infers from what the user typed, kept as inferred); each
    level includes the ones before it. ``note_min_cosine`` (0.20-0.95) is the least
    similarity of meaning that counts as support in the last two levels (the
    inferred note's bar is 0.15 lower). ``followup_query`` (#452) is what a message
    with no content of its own ("go look it up", "continue", "继续") is searched
    with: ``recent`` (the default) the message and the user's words of the chat's
    last two turns that the model still shows, or ``message`` the message's own
    words, as recall did before; with ``recent`` the first message of a new chat
    that has no content recalls nothing. Two more are for questions about the person's
    own life that count, total or compare across things (#428), and apply to
    ``recall_form = "list"``.
    ``gather``, ``guide`` (the default: for such a question the block's vector bar is
    the lower of ``recall_min_cosine`` and ``gather_min_cosine``, every list-form
    ``search_memory`` uses that lower bar, and one more sentence under the list says to
    open every item that may belong and to search again for each kind not listed yet),
    ``wide`` (the lower bar without the sentence) or ``off`` (recall as before; a
    message that does not ask to count, total or compare is recalled for the same way
    under every value); ``gather_min_cosine`` (0.20-0.55, 0.40 by default) is that
    lower bar.
    ``project_instructions = "off"`` stops a
    task that the person made from reading the instruction file of its folder (``AGENTS.md``,
    ``CLAUDE.md`` or ``GEMINI.md``, the nearest first) when it starts.
    """

    admit_on_outcome: bool = False
    across_chats: bool = True
    retriever: str = "auto"  # auto | bm25 | embedding | llm
    keep_statements: bool = False
    sleep: str = "off"  # off | rules | model | model+tags
    sleep_lock: str = ""  # Optional lock path: no model calls while it exists.
    recall_records: int = 3  # Most notes a chat message's recall block holds.
    recall_docs: int = 2  # Most chat turns it holds.
    recall_chars: int = 4000  # Its size bound, in characters.
    recall_notes: str = "on"  # on | off: no notes in the block or the pinned lines.
    search_when_empty: str = "off"  # off | hint: a line when nothing was recalled.
    recall_min_cosine: float = 0.55  # Least similarity of meaning to be recalled.
    recall_min_coverage: float = 0.6  # Least share of the question's words held.
    recall_item_chars: int = 900  # The room one recalled chat turn takes.
    recall_excerpt: str = "head"  # head | match: which part of a turn is shown.
    recall_form: str = "excerpt"  # excerpt | list: turns as excerpts, or as a list.
    recall_list_max: int = 20  # The most turns the list form lists.
    recall_open_chars: int = 4000  # The room one turn the model opens takes.
    excerpt_open: str = "on"  # on | off: a cut excerpt carries its id to open.
    recall_source: str = "turns"  # turns | items: recall draws on turns, or on items.
    open_sources: str = "off"  # off | on: an opened item also shows its source turns.
    compress_at_chars: int = 12000  # The size of a window a chat is encoded in.
    compress_parallel: int = 1  # Windows an ingest asks the encoder for at once.
    compress_cache: str = (
        ""  # Where encoder replies are kept; empty: in the memory dir.
    )
    note_gate: str = "typed"  # typed | pointed | meaning | model: note sources.
    note_min_cosine: float = 0.55  # Least similarity of a note to its source.
    followup_query: str = "recent"  # recent | message: what "continue" searches with.
    gather: str = MEMORY_GATHER_DEFAULT  # guide | wide | off: look wider to count.
    gather_min_cosine: float = 0.40  # The vector bar for such a question.
    project_instructions: str = "on"  # on | off: a task reads its folder's AGENTS.md.


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


BROWSER_DRIVERS = ("off", "safari-automation", "chrome")


@dataclass
class BrowserConfig:
    """The ``[browser]`` table: whether melo may read web pages in a browser.

    ``driver`` is ``off`` (the default, no browser tools), ``safari-automation``
    (Safari's automation windows, which come to the front) or ``chrome`` (Google
    Chrome in the background, with no window and a profile of melo's own). Only
    the terminal chat and the console offer it.
    ``keep_tabs`` leaves the tabs melo opened in the window when a chat ends or
    sits idle; by default melo closes them. ``app_windows`` (on by default) lets
    melo see a window another app opens after its own act in the browser, such
    as a chat app's sign-in window, and answer it with the person's say, when the
    desktop control helper is installed; without the helper melo still says which
    app came to the front (:mod:`localmelo.support.browser.desktop`).
    ``live`` (on by default) lets the console and the app show the Chrome browser
    to the person while melo works (a developer option; off refuses the view),
    and ``live_fps`` (1 to 60, default 15) holds the pictures of that view to that
    many a second.
    """

    driver: str = "off"  # off | safari-automation | chrome
    keep_tabs: bool = False
    app_windows: bool = True
    live: bool = True
    live_fps: int = 15


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

    ``web_fetch_allow_local`` is a developer setting, empty by default: the
    test servers on this computer ``web_fetch`` may also read, as exact
    ``127.0.0.1:PORT`` or ``[::1]:PORT`` entries. Any other entry is an error,
    and melo's own ports stay refused when listed (see
    :mod:`localmelo.support.web_local`). Config and ``-X`` only.
    """

    web_fetch: bool = True
    web_fetch_allow_local: list[str] = field(default_factory=list)


# The most results one search returns, and the range of ``[web] search_count``.
WEB_SEARCH_COUNT: Final = (1, 8)


@dataclass
class WebConfig:
    """The ``[web]`` table: searching the public web.

    ``search`` (off by default) gives every chat and every task's worker a
    ``web_search`` tool, and tells them when to use it. The words of a search
    leave the computer, which is why it is a switch and not always on (see
    :mod:`localmelo.support.web.search`).

    ``search_order`` and ``search_count`` are developer settings. The order is
    the ways melo tries one after another (the names in
    :data:`localmelo.support.web.search.WAYS`, then ``browser``); empty means the
    built-in order. The count is how many results one search shows by default,
    from 1 to 8 (a call may ask for fewer).
    """

    search: bool = False
    search_order: list[str] = field(default_factory=list)
    search_count: int = 6


@dataclass
class TasksConfig:
    """The ``[tasks]`` table: what a chat does about the tasks it started.

    ``follow_up``: when a task a chat started fails or stops, that chat gets
    one turn of its own to continue it once, work around a blocked step, or
    ask the person one question, then says what got done (#411). On unless
    the file turns it off; a developer setting.
    """

    follow_up: bool = True


@dataclass
class ChatConfig:
    """The ``[chat]`` table: what a new chat knows when it starts.

    ``recent_digest_minutes``: a chat's first turn (not the main chat's) gets a
    short digest of the user's other chats in the last that many minutes: up to
    five lines, one per recent turn, saying what it was about and where (see
    :mod:`localmelo.support.memory.digest`). It reads memory's turn index, so
    ``[memory] across_chats = false`` turns it off too. ``0`` turns it off;
    30 by default, at most 1440.
    """

    recent_digest_minutes: int = 30


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
    web: WebConfig = field(default_factory=WebConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
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
        """Return a list of validation errors. Empty list means valid.

        A value of the wrong kind is reported alone, first: the checks below
        compare values and would otherwise stop on it.
        """
        # [cli] is a display preference: like [sessions] resume_mode, a bad
        # value warns and falls back in the interactive UI (cli_errors()), so
        # direct mode and the gateway never fail on it.
        if errors := self._type_errors(skip=_FALLBACK_SECTIONS):
            return errors
        errors = []
        errors.extend(self._validate_agent())
        errors.extend(self._validate_retrieval())
        errors.extend(self._validate_memory())
        errors.extend(self._validate_scheduler())
        errors.extend(self._validate_browser())
        errors.extend(self._validate_tools())
        errors.extend(self._validate_web())
        errors.extend(self._validate_chat())
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
        errors.extend(self._validate_hosted_fields())

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
        """``[memory]`` holds known values inside their documented ranges."""
        errors: list[str] = []
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
        for name in ("sleep_lock", "compress_cache"):
            if "\x00" in getattr(self.memory, name):
                errors.append(f"[memory] {name} must be a path string without NUL")
        for name, (low, high) in (
            ("recall_records", MEMORY_RECALL_RECORDS),
            ("recall_docs", MEMORY_RECALL_DOCS),
            ("recall_chars", MEMORY_RECALL_CHARS),
            ("recall_item_chars", MEMORY_RECALL_ITEM_CHARS),
            ("recall_list_max", MEMORY_RECALL_LIST_MAX),
            ("recall_open_chars", MEMORY_RECALL_OPEN_CHARS),
            ("compress_at_chars", MEMORY_COMPRESS_AT_CHARS),
            ("compress_parallel", MEMORY_COMPRESS_PARALLEL),
        ):
            value = getattr(self.memory, name)
            if not low <= value <= high:
                errors.append(
                    f"[memory] {name} must be from {low} to {high}, got {value}"
                )
        for name, (floor, ceiling) in (
            ("recall_min_cosine", MEMORY_RECALL_MIN_COSINE),
            ("recall_min_coverage", MEMORY_RECALL_MIN_COVERAGE),
            ("note_min_cosine", MEMORY_NOTE_MIN_COSINE),
            ("gather_min_cosine", MEMORY_GATHER_MIN_COSINE),
        ):
            value = getattr(self.memory, name)
            if not floor <= value <= ceiling:
                errors.append(
                    f"[memory] {name} must be from {floor} to {ceiling}, got {value}"
                )
        for name, allowed in (
            ("recall_notes", MEMORY_RECALL_NOTES),
            ("search_when_empty", MEMORY_SEARCH_WHEN_EMPTY),
            ("recall_excerpt", MEMORY_RECALL_EXCERPT),
            ("recall_form", MEMORY_RECALL_FORM),
            ("excerpt_open", MEMORY_EXCERPT_OPEN),
            ("note_gate", MEMORY_NOTE_GATE),
            ("followup_query", MEMORY_FOLLOWUP_QUERY),
            ("gather", MEMORY_GATHER),
            ("recall_source", MEMORY_RECALL_SOURCE),
            ("open_sources", MEMORY_OPEN_SOURCES),
            ("project_instructions", MEMORY_PROJECT_INSTRUCTIONS),
        ):
            value = getattr(self.memory, name)
            if value not in allowed:
                errors.append(
                    f"[memory] {name} {value!r} is not recognised. "
                    f"Must be one of {allowed}."
                )
        return errors

    def _validate_browser(self) -> list[str]:
        """``[browser] driver`` names a driver; ``live_fps`` is from 1 to 60."""
        errors = []
        if self.browser.driver not in BROWSER_DRIVERS:
            errors.append(
                f"[browser] driver={self.browser.driver!r} is not one of {BROWSER_DRIVERS}"
            )
        fps = self.browser.live_fps
        if not 1 <= fps <= 60:
            errors.append(
                f"[browser] live_fps={fps!r} must be a whole number from 1 to 60"
            )
        return errors

    def _validate_tools(self) -> list[str]:
        """``[tools] web_fetch_allow_local`` is a list of exact local test server entries."""
        from localmelo.support import web_local

        return web_local.problems(self)

    def _validate_web(self) -> list[str]:
        """``[web]``: the order names known ways, the count is in range."""
        from localmelo.support.web.search import way_names

        errors: list[str] = []
        order = self.web.search_order
        if not all(isinstance(x, str) for x in order):
            errors.append(f"[web] search_order must be a list of names, got {order!r}")
        else:
            known = way_names()
            errors.extend(
                f"[web] search_order: {name!r} is not one of {known}"
                for name in order
                if name not in known
            )
            if len(set(order)) != len(order):
                errors.append("[web] search_order names a way twice")
        count, (low, high) = self.web.search_count, WEB_SEARCH_COUNT
        if not low <= count <= high:
            errors.append(
                f"[web] search_count must be a whole number from {low} to {high}, "
                f"got {count!r}"
            )
        return errors

    def _validate_chat(self) -> list[str]:
        """``[chat] recent_digest_minutes`` is a whole number of minutes, 0-1440."""
        value = self.chat.recent_digest_minutes
        low, high = CHAT_RECENT_DIGEST_MINUTES
        if low <= value <= high:
            return []
        return [
            f"[chat] recent_digest_minutes must be an integer from {low} to {high}, "
            f"got {value!r}"
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
            value = getattr(scheduler, name)
            if value < minimum:
                errors.append(
                    f"[scheduler] {name} must be an integer >= {minimum}, "
                    f"got {value!r}"
                )
        focus = scheduler.focus_s
        if not 0 < focus < float("inf"):
            errors.append(
                f"[scheduler] focus_s must be a number of seconds > 0, got {focus!r}"
            )
        if scheduler.lock_gate not in ("record", "wait"):
            errors.append(
                f"[scheduler] lock_gate={scheduler.lock_gate!r} is not one of "
                "('record', 'wait')"
            )
        for name in ("gpu_lock_file", "lock_self"):
            if "\x00" in getattr(scheduler, name):
                errors.append(f"[scheduler] {name} must not contain a NUL byte")
        return errors

    def _validate_cli(self) -> list[str]:
        """``[cli]`` holds known values and a model-to-hint table of known hints."""
        errors: list[str] = []
        cli = self.cli
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
        for model, hint in cli.reasoning_tags.items():
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
        """``[agent]`` holds known values inside their documented ranges."""
        agent = self.agent
        errors: list[str] = []
        attempt = agent.steps_per_attempt
        if attempt is not None and attempt < 1:
            errors.append(
                f"[agent] steps_per_attempt must be an int >= 1, got {attempt!r}"
            )
        calls = agent.max_calls
        if calls is not None and calls < 2:
            errors.append(f"[agent] max_calls must be an int >= 2, got {calls!r}")
        errors.extend(tool_images_errors(agent))
        errors.extend(empty_reply_fallback_errors(agent))
        for name, allowed in (
            ("reasoning", REASONING_MODES),
            ("note_writer", NOTE_WRITERS),
            ("window_eviction", WINDOW_EVICTIONS),
            ("cache_layout", CACHE_LAYOUTS),
        ):
            value = getattr(agent, name)
            if value not in allowed:
                errors.append(
                    f"[agent] {name} {value!r} is not recognised. "
                    f"Must be one of {allowed}."
                )
        window = agent.window_messages
        if (
            window is not None
            and not WINDOW_MESSAGES_MIN <= window <= WINDOW_MESSAGES_MAX
        ):
            errors.append(
                f"[agent] window_messages must be an int from "
                f"{WINDOW_MESSAGES_MIN} to {WINDOW_MESSAGES_MAX}, got {window!r}"
            )
        turns = agent.replay_turns
        if turns is not None and turns < 1:
            errors.append(f"[agent] replay_turns must be an int >= 1, got {turns!r}")
        names = ("context_window", "output_reserve", "safety_margin")
        budget = [getattr(agent, name) for name in names]
        errors.extend(
            f"[agent] {name} must be a positive integer, got {value!r}"
            for name, value in zip(names, budget, strict=True)
            if value is not None and value < 1
        )
        if any(value is not None for value in budget) and None in budget:
            missing = [n for n, v in zip(names, budget, strict=True) if v is None]
            errors.append("[agent] context budget is missing " + ", ".join(missing))
        elif None not in budget and all(value > 0 for value in budget):
            size, reserve, margin = budget
            if size - reserve - margin <= 0:
                errors.append(
                    "[agent] context_window - output_reserve - safety_margin "
                    "must be positive"
                )
        errors.extend(self._tool_env_errors())
        return errors

    def _tool_env_errors(self) -> list[str]:
        """``[agent] tool_env`` is a list of valid environment variable names."""
        names = self.agent.tool_env
        if isinstance(names, list) and all(
            isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            for name in names
        ):
            return []
        return ["[agent] tool_env must be a list of valid environment variable names"]

    def tool_environment(self) -> dict[str, str]:
        """Present variables explicitly named by the config; never persist their values."""
        if errors := self._tool_env_errors():
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
        k = retrieval.k
        if not 1 <= k <= 10:
            errors.append(f"[retrieval] k must be an int from 1 to 10, got {k!r}")
        score = retrieval.min_score
        if not 0 <= score <= 1:
            errors.append(
                f"[retrieval] min_score must be a number from 0 to 1, got {score!r}"
            )
        if not 1 <= retrieval.cross_task_ratio <= 10:
            errors.append("[retrieval] cross_task_ratio must be a number from 1 to 10")
        for name, maximum in (("cross_task_cap", 2), ("cross_task_cold", 1)):
            if not 0 <= getattr(retrieval, name) <= maximum:
                errors.append(f"[retrieval] {name} must be an int from 0 to {maximum}")
        for name, allowed in (
            ("recall_at", RECALL_TRIGGERS),
            ("kinds", RECALL_KINDS),
        ):
            values = getattr(retrieval, name)
            if any(item not in allowed for item in values):
                errors.append(
                    f"[retrieval] {name} must be a list of "
                    f"{', '.join(allowed)}, got {values!r}"
                )
        return errors

    def _validate_reasoning_fields(self) -> list[str]:
        """Each backend's ``reasoning_field`` is a known key."""
        errors: list[str] = []
        for section_name in _BACKEND_SECTIONS:
            value = getattr(self, section_name).reasoning_field
            allowed = (
                ("", "thinking", "auto")
                if section_name == "ollama"
                else ("",) + REASONING_FIELDS
            )
            if value not in allowed:
                errors.append(
                    f"[{section_name}] reasoning_field {value!r} is not recognised. "
                    f"Must be one of {allowed}."
                )
        return errors

    def _validate_hosted_fields(self) -> list[str]:
        """Each hosted section's ``api_key`` reference and ``thinking`` value."""
        from localmelo.support import hosted

        errors: list[str] = []
        for section_name in _CLOUD_SECTIONS:
            errors.extend(
                hosted.hosted_fields_errors(section_name, getattr(self, section_name))
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

    def _type_errors(
        self, *, only: Container[str] | None = None, skip: Container[str] = ()
    ) -> list[str]:
        """Each table field that holds the wrong kind of value (of the tables in *only*, else all but *skip*)."""
        errors = []
        for key, value in flat(self).items():
            section, dot, name = key.rpartition(".")
            if dot and (only is None or section in only) and section not in skip:
                hint = hints(type(getattr(self, section)))[name]
                if problem := type_problem(value, hint):
                    errors.append(f"[{section}] {name} {problem}")
        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise ConfigError if any issues are found."""
        errors = self.validate()
        if errors:
            msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigError(msg)

    def setting_errors(self) -> list[str]:
        """The errors of the settings alone, which need no backend (``-X``, ``melo config``)."""
        if errors := self._type_errors(skip=_BACKEND_SECTIONS):
            return errors
        return [
            *self._validate_agent(),
            *self._validate_retrieval(),
            *self._validate_memory(),
            *self._validate_scheduler(),
            *self._validate_browser(),
            *self._validate_tools(),
            *self._validate_web(),
            *self._validate_chat(),
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
        return self._type_errors(only=("cli",)) or self._validate_cli()

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

# What a field of each kind must hold, as the sentence that reports it says.
KIND_WORDS: Final = {
    bool: "true or false",
    int: "a whole number",
    float: "a number",
    str: "text",
    list: "a list",
    dict: "a table",
}


def flat(obj: object, prefix: str = "") -> dict[str, Any]:
    """Every field of a config as ``section.name`` -> value (top-level names have no dot).

    The one walk over the config's fields: ``save``, the type check, ``melo config``
    and the ``-X`` catalog all read the settings through it.
    """
    out: dict[str, Any] = {}
    for item in dataclasses.fields(obj):  # type: ignore[arg-type]
        if not item.compare:
            continue  # bookkeeping (a private flag, the spelling of a backend), not a setting
        value = getattr(obj, item.name)
        if dataclasses.is_dataclass(value):
            out.update(flat(value, f"{prefix}{item.name}."))
        else:
            out[prefix + item.name] = value
    return out


def changed(cfg: Config) -> dict[str, Any]:
    """The settings of *cfg* that differ from the defaults, as ``section.name`` -> value."""
    defaults = flat(Config())
    return {key: value for key, value in flat(cfg).items() if value != defaults[key]}


_HINTS: dict[type, dict[str, Any]] = {}


def hints(table: type) -> dict[str, Any]:
    """The type hints of the dataclass *table*, read once."""
    if table not in _HINTS:
        _HINTS[table] = typing.get_type_hints(table)
    return _HINTS[table]


def hint_base(hint: Any) -> tuple[Any, bool]:
    """The plain type of a field's type hint, and whether ``None`` (unset) is allowed.

    ``int | None`` is ``(int, True)`` and ``list[str]`` is ``(list, False)``.
    """
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = typing.get_args(hint)
        inner = [arg for arg in args if arg is not type(None)]
        return hint_base(inner[0])[0], len(inner) != len(args)
    return typing.get_origin(hint) or hint, False


def type_problem(value: Any, hint: Any) -> str | None:
    """Why *value* is not the kind *hint* names (``must be a whole number, got '8401'``), or None.

    A bool is not a number here, and a whole number is a number. A hint of a kind
    not in :data:`KIND_WORDS` is not checked.
    """
    base, optional = hint_base(hint)
    if base not in KIND_WORDS or (value is None and optional):
        return None
    accepted: Any = int | float if base is float else base
    if isinstance(value, accepted) and (base is bool or not isinstance(value, bool)):
        return None
    return f"must be {KIND_WORDS[base]}, got {value!r}"


def trim_text(section: LocalBackendConfig | CloudVendorConfig) -> None:
    """Strip the blanks around each text field of a backend table, but never the key.

    A model name or an address with a space after it reads the same to a person and
    is another name to the server.
    """
    for item in dataclasses.fields(section):
        value = getattr(section, item.name)
        if isinstance(value, str) and item.name != "api_key":
            setattr(section, item.name, value.strip())


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
_BACKEND_SECTIONS = (*_LOCAL_SECTIONS, *_CLOUD_SECTIONS)
# The tables of display preferences: a bad value warns and falls back (see
# ``cli_errors()`` and the like) instead of stopping a run, so ``validate()`` leaves
# them alone. ``[approvals]`` is not one: it changes what runs without asking.
_FALLBACK_SECTIONS = ("sessions", "cli", "ui", "console", "workspace")

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
    for name in _BACKEND_SECTIONS:
        trim_text(getattr(cfg, name))
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


def _digits_as_text(value: Any) -> Any:
    """A bare number as its text (``colors = 256`` is ``"256"``); anything else as it is."""
    return (
        str(value) if isinstance(value, int) and not isinstance(value, bool) else value
    )


# The few fields the file spells loosely: key -> what the file's value stands for.
# Every other field is read as the file has it.
_SPELLINGS: dict[str, Callable[[Any], Any]] = {
    "cli.colors": _digits_as_text,
    "cli.background": _digits_as_text,
    "sessions.record_direct": lambda value: value
    is not False,  # only false turns it off
}


def _read(
    cfg: Config, data: Mapping[str, Any], tables: Container[str] | None = None
) -> None:
    """Set each field that a table of *data* (those in *tables*, else all) holds.

    A value is not checked or changed here: a wrong one is kept for ``validate()``
    to report. A key that is no field, and a table that is not one, are ignored.
    """
    fields = flat(cfg)
    for table, values in data.items():
        if isinstance(values, dict) and (tables is None or table in tables):
            for name, value in values.items():
                key = f"{table}.{name}"
                if key in fields:
                    set_key(cfg, key, _SPELLINGS.get(key, lambda v: v)(value))


def _load_file() -> Config:
    path = config_path()
    if not os.path.isfile(path):
        return Config()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # -- Migration: old single `backend` field --
    if "backend" in data and "chat_backend" not in data:
        return _migrate_legacy(data)

    cfg = Config()

    # -- Migration: old `mlc-llm` / `cloud_api` keys in chat_backend/embedding_backend --
    raw_chat = data.get("chat_backend", "")
    raw_emb = data.get("embedding_backend", "")
    cfg.chat_backend = _migrate_backend_key(raw_chat, data)
    cfg.embedding_backend = _migrate_embedding_key(raw_emb)
    # "omlx" is another spelling of "vllm" (one backend, one [vllm] table):
    # everything reads "vllm", and save() writes the spelling back.
    cfg.chat_alias = raw_chat if raw_chat == OMLX_ALIAS else ""
    cfg.embedding_alias = raw_emb if raw_emb == OMLX_ALIAS else ""

    _read(cfg, data)
    for section in _BACKEND_SECTIONS:
        # Migration: flat chat_template_kwargs -> extra_body
        if isinstance(data.get(section), dict):
            _migrate_extra_body(getattr(cfg, section), data[section])
    # Migration: old [mlc] section with chat_port -> chat_url
    if "mlc" in data:
        _migrate_mlc_section(cfg.mlc, data["mlc"])

    # Migration: old [cloud_api] section -> vendor-specific section
    if "cloud_api" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["cloud_api"])
    elif "online" in data and cfg.chat_backend in _CLOUD_SECTIONS:
        _migrate_cloud_api_section(cfg, data["online"])

    cfg.request_timeout_s = data.get("request_timeout_s")
    cfg.mcp_config = str(data.get("mcp_config") or "")
    if "pricing" in data:  # a bad table is kept for Config.pricing_errors
        cfg.pricing = data["pricing"]

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
    _read(cfg, {vendor: section})
    _migrate_extra_body(target, section)


def _migrate_legacy(data: dict) -> Config:
    """Migrate the oldest format with a single `backend` field."""
    cfg = Config()
    old = data["backend"]

    if old == "mlc-llm":
        cfg.chat_backend = "mlc"
        cfg.embedding_backend = "mlc"
        if "mlc" in data:
            _read(cfg, data, ("mlc",))
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
            _read(cfg, data, ("ollama",))
            _migrate_extra_body(cfg.ollama, data["ollama"])

    _read(cfg, data, ("gateway", "agent", "sessions"))
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

    # The two backends are always written, the other top-level keys before the
    # first table, and a setting only when it is not at its default.
    chat = _spelling(cfg.chat_backend, cfg.chat_alias)
    embedding = _spelling(cfg.embedding_backend, cfg.embedding_alias)
    top: list[str] = []
    tables: dict[str, list[str]] = {}
    for key, value in changed(cfg).items():
        section, _, name = key.rpartition(".")
        if key not in ("chat_backend", "embedding_backend", "pricing"):
            line = f"{name} = {_toml_value(value)}"
            (tables.setdefault(section, []) if section else top).append(line)
    lines = [
        f"chat_backend = {_toml_string(chat)}",
        f"embedding_backend = {_toml_string(embedding)}",
        *top,
        "",
    ]
    for section, body in tables.items():
        lines.extend([f"[{section}]", *body, ""])

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
