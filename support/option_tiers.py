"""The tier of every ``melo`` option, the table of settings, and the help built from them.

``melo -h`` shows the everyday options and three everyday commands in 23 lines;
``melo -h all`` (also plain ``-h`` in developer mode, ``[ui] mode = "developer"`` or
``--developer``) shows every option, command and setting. The tier and the mode decide
only what help shows: every option parses and works in both modes.

A setting that tunes a run or compares designs is a ``config.toml`` key, not a new
flag: add a ``Setting`` row, and for a comparison the issue that decides it to
``EXPERIMENTAL_ISSUES``. After the A/B the winner becomes the default and the flag goes
one release later. ``docs/reference/cli.md`` carries the same tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

USER = "user"
ADVANCED = "advanced"
EXPERIMENTAL = "experimental"

# The user tier is what ``melo -h`` shows (minus -h itself).
_USER = (
    "--help",
    "--version",
    "--model",
    "--url",
    "--workspace",
    "--steps",
    "--timeout",
    "--output",
    "--continue",
    "--resume",
    "--yolo",
)

# Advanced: for scripts, gateway operators and tuning; stable, not a comparison.
# The first group is shown by ``melo -h all``; the rest are old spellings that stay
# for scripts (OLD_SPELLINGS says what replaces each).
_ADVANCED = (
    "--developer",
    "--memory",
    "--builtins",
    "--mcp",
    "--routine",
    "--progress",
    "--title",
    "--mcp-only",
    "--chat-model",
    "--base-url",
    "--max-steps",
    "--output-format",
    "--mcp-config",
    "--brief",
    "--serve",
    "--gateway-login",
    "--reconfigure",
    "--gateway-rotate-token",
    "--enable-playground",
    "--host",
    "--port",
    "--allow-unconfigured",
    "--daemon",
    "--import-session",
    "--save-session",
    "--no-save-session",
    "--parent-run",
    "--inbox",
    "--inbox-fd",
    "--context-window",
    "--output-reserve",
    "--safety-margin",
    "--cache-layout",
    "--window-messages",
    "--memory-dir",
    "--request-timeout",
    "--embedding-base-url",
    "--embedding-model",
)

# The old spellings of the everyday options and of the modes: still accepted, and in no
# list of options to use (the console's list of advanced options leaves them out).
EVERYDAY_OLD = (
    "--chat-model",
    "--base-url",
    "--max-steps",
    "--output-format",
    "--brief",
    "--serve",
    "--gateway-login",
    "--reconfigure",
)

# Experimental: each option compares designs; the value names the issue that
# decides it.
EXPERIMENTAL_ISSUES: dict[str, str] = {
    "--attempt-policy": "#185",
    "--reflection-gate": "#185",
    "--brief-items": "#185",
    "--fixed-prefix": "#183",
    "--window-eviction": "#183",
    "--reflect-tool-guard": "#183",
    "--reasoning": "#183",
    "--reasoning-field": "#183",
    "--replay-turns": "#183",
    "--note-writer": "#183",
    "--tool-exposure": "#3",
    "--captured-history": "#4",
    "--retriever": "#4",
    "--recall-at": "#4",
    "--recall-k": "#4",
    "--recall-kinds": "#4",
    "--recall-placement": "#4",
    "--recall-min-score": "#4",
    "--admit-on-outcome": "#4",
    "--keep-decision-text": "#4",
    "--retrieval-extra-body": "#4",
}

# Experimental settings that have no option: the issue that decides each.
EXPERIMENTAL_KEYS: dict[str, str] = {
    "agent.lean_output": "#183",
    "agent.confine_direct_runs": "#183",
    "agent.answer_on_stop": "#185",
    "agent.max_calls": "#185",
    "agent.tool_images": "#280",
    "agent.tool_images_keep": "#280",
    "agent.tool_images_retire": "#280",
    "memory.retriever": "#299",
    "memory.keep_statements": "#299",
    "memory.sleep": "#363",
    "memory.sleep_lock": "#363",
    "memory.recall_records": "#349",
    "memory.recall_docs": "#349",
    "memory.recall_chars": "#349",
    "memory.recall_notes": "#349",
    "memory.search_when_empty": "#349",
    "memory.recall_min_cosine": "#349",
    "memory.recall_min_coverage": "#349",
    "memory.recall_item_chars": "#349",
    "memory.recall_excerpt": "#349",
}

# Long option name -> tier, for every option the parser has.
OPTION_TIERS: dict[str, str] = {
    **dict.fromkeys(_USER, USER),
    **dict.fromkeys(_ADVANCED, ADVANCED),
    **dict.fromkeys(EXPERIMENTAL_ISSUES, EXPERIMENTAL),
}


class Setting(NamedTuple):
    """A ``config.toml`` key and the old option that sets it for one run."""

    key: str  # "section.name"; a top-level key has no dot
    flag: str = ""
    file: bool = True  # False: for one run only, the file does not hold it
    default: str = ""  # shown when the key is not a field of ``Config``


SETTINGS = (
    Setting("agent.attempt_policy", "--attempt-policy", False, "reflect"),
    Setting("agent.reflection_gate", "--reflection-gate", False, "patient"),
    Setting("agent.fixed_prefix", "--fixed-prefix"),
    Setting("agent.window_eviction", "--window-eviction"),
    Setting("agent.reflect_tool_guard", "--reflect-tool-guard"),
    Setting("agent.cache_layout", "--cache-layout"),
    Setting("agent.window_messages", "--window-messages"),
    Setting("agent.brief_items", "--brief-items"),
    Setting("agent.reasoning", "--reasoning"),
    Setting("agent.replay_turns", "--replay-turns"),
    Setting("agent.note_writer", "--note-writer"),
    Setting("agent.captured_history", "--captured-history"),
    Setting("agent.context_window", "--context-window"),
    Setting("agent.output_reserve", "--output-reserve"),
    Setting("agent.safety_margin", "--safety-margin"),
    Setting("agent.tool_exposure", "--tool-exposure", False, "retrieve"),
    Setting("retrieval.retriever", "--retriever"),
    Setting("retrieval.recall_at", "--recall-at"),
    Setting("retrieval.k", "--recall-k"),
    Setting("retrieval.kinds", "--recall-kinds"),
    Setting("retrieval.placement", "--recall-placement"),
    Setting("retrieval.min_score", "--recall-min-score"),
    Setting("retrieval.extra_body", "--retrieval-extra-body", False, "none"),
    Setting("memory.admit_on_outcome", "--admit-on-outcome"),
    Setting("memory.keep_decision_text", "--keep-decision-text", False, "false"),
    Setting("memory.dir", "--memory-dir", False, "built in"),
    Setting("request_timeout_s", "--request-timeout"),
    Setting("sessions.record_direct", "--no-save-session"),
    Setting("endpoint.embedding_url", "--embedding-base-url", False, "none"),
    Setting("endpoint.embedding_model", "--embedding-model", False, "none"),
    Setting("endpoint.reasoning_field", "--reasoning-field", False, "auto"),
)

# The two settings whose option takes no value: the option means the key has this one.
_SWITCH_VALUES = {
    "memory.keep_decision_text": "true",
    "sessions.record_direct": "false",
}

# Old option -> what replaces it. Each one still works and parses to the same result;
# ``melo -h`` leaves them out, and docs/reference/cli.md has this table.
OLD_SPELLINGS: dict[str, str] = {
    "--chat-model": "-m, --model",
    "--base-url": "-u, --url",
    "--max-steps": "-s, --steps",
    "--output-format": "-o, --output",
    "--mcp-config": "--mcp",
    "--save-session": "--title",
    "--reconfigure": "melo setup",
    "--serve": "melo serve",
    "--daemon": "melo serve install|uninstall|status|stop",
    "--host": "melo serve --host",
    "--port": "melo serve -p",
    "--enable-playground": "melo serve --playground",
    "--allow-unconfigured": "melo serve --force",
    "--gateway-login": "melo console",
    "--gateway-rotate-token": "melo console --reset",
    "--brief": "melo worker BRIEF",
    "--inbox": "melo worker BRIEF --inbox",
    "--inbox-fd": "melo worker BRIEF --inbox-fd",
    "--parent-run": "melo worker BRIEF --parent",
    "--import-session": "melo import FORMAT FILE",
    # A setting is a config key; -X KEY=VALUE sets it for one run.
    **{
        item.flag: f"-X {item.key}={_SWITCH_VALUES.get(item.key, 'VALUE')}"
        for item in SETTINGS
    },
}

# (options, what it takes, what it does): the rows of ``melo -h all``; the first
# ``SHORT_ROWS`` are the rows of ``melo -h``.
OPTION_ROWS = (
    ("-m, --model", "NAME", "chat model for this run"),
    ("-u, --url", "URL", "model server for this run (http://127.0.0.1:8000/v1)"),
    ("-w, --workspace", "DIR", "folder to work in (default: this one)"),
    ("-c, --continue", "", "continue the newest chat here"),
    ("-r, --resume", "[ID]", "resume a chat, or pick one from a list"),
    ("-s, --steps", "N", "most model calls per task (default 30)"),
    ("-t, --timeout", "SEC", "stop the run after SEC seconds"),
    ("-o, --output", "FORMAT", "text (default) or json"),
    ("-y, --yolo", "", "run tools without asking"),
    (
        "-X, --set",
        "KEY=VALUE",
        "change a setting for this run (list: melo config keys)",
    ),
    ("-V, --version", "", "show the version"),
    ("--memory", "on|off", "long-term memory (default on)"),
    ("--builtins", "SET", "built-in tools: all, none or NAME,... (default: per mode)"),
    ("--mcp", "PATH", "MCP servers to use (mcpServers JSON file)"),
    ("--mcp-only", "", "only the MCP tools (--mcp PATH --builtins none)"),
    ("--routine", "NAME", "learn and recall the task's steps as NAME"),
    ("--progress", "auto|off", "progress line on stderr (default auto)"),
    ("--title", "T", "title of this run's worker session"),
    ("--developer", "", "with -h: every option and setting, as -h all does"),
)
SHORT_ROWS = 11

USAGE = 'usage: melo [options] ["task"]\n       melo COMMAND [options]'
DESCRIPTION = 'melo opens a chat here. melo "task" runs one task and prints the answer.'


def _table(rows: Sequence[tuple[str, str]], indent: int = 2) -> list[str]:
    width = max(len(left) for left, _ in rows) + 2
    return [f"{' ' * indent}{left:<{width}}{right}" for left, right in rows]


def _options(rows: Sequence[tuple[str, str, str]]) -> list[str]:
    return _table([(f"{names} {meta}".strip(), text) for names, meta, text in rows])


def help_short() -> str:
    """``melo -h``: the everyday options, three commands with what they do, and more."""
    from localmelo.support.command_words import EVERYDAY

    lines = [
        USAGE,
        "",
        DESCRIPTION,
        "",
        "Options:",
        *_options(OPTION_ROWS[:SHORT_ROWS]),
        "",
        "Commands:",
        *(f"  {word:<10}{text}" for word, text in EVERYDAY),
        "More commands and every setting: melo -h all",
    ]
    return "\n".join(lines) + "\n"


def help_all() -> str:
    """``melo -h all``: every option, every command, every setting with its default."""
    from localmelo.support import settings
    from localmelo.support.command_words import SUMMARIES

    rows = []
    for key in settings.catalog().values():
        default = key.default
        shown = (
            default if isinstance(default, str) and default else settings.show(default)
        )
        issue = EXPERIMENTAL_ISSUES.get(key.flag) or EXPERIMENTAL_KEYS.get(key.name)
        row = f"  {key.name:<29}{shown:<12}{key.flag}" + (
            f" ({issue})" if issue else ""
        )
        rows.append(row.rstrip())
    lines = [
        USAGE,
        "",
        DESCRIPTION,
        "",
        "Options:",
        *_options(OPTION_ROWS),
        "",
        "Commands (melo COMMAND -h):",
        *_table(SUMMARIES),
        "",
        "Developer tools, own -h: melo-retrieve · melo-procedures · melo-task",
        'Old spellings of the options: docs/reference/cli.md, "Old spelling to new".',
        "",
        "Settings, by config.toml key (melo config keys shows the values in use):",
        "-X KEY=VALUE changes one for this run and melo config set KEY VALUE saves",
        "it; the option shown still works. A comparison names the issue that decides it.",
        *rows,
    ]
    return "\n".join(lines) + "\n"
