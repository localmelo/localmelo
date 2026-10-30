"""The tier of every ``melo`` option, and the ``--help`` text built from them (#186).

Default mode shows the ``user`` options; developer mode (``[ui] mode =
"developer"`` or ``--developer``) shows all three tiers under headings. The tier
only decides what help shows: every option parses and works in both modes.

A new option that compares designs starts as ``experimental`` with the issue that
will remove it in ``EXPERIMENTAL_ISSUES``; after the A/B the winner becomes the
default, and the flag goes one release later. ``docs/reference/cli.md`` carries the
same table.
"""

from __future__ import annotations

import argparse
import copy

USER = "user"
ADVANCED = "advanced"
EXPERIMENTAL = "experimental"

# The user tier is the only one default-mode help shows.
_USER = (
    "--help",
    "--version",
    "--base-url",
    "--chat-model",
    "--workspace",
    "--max-steps",
    "--timeout",
    "--output-format",
    "--memory",
    "--brief",
    "--continue",
    "--resume",
    "--yolo",
    "--serve",
    "--gateway-login",
    "--reconfigure",
)

# Advanced: for scripts, gateway operators and tuning; stable, not a comparison.
_ADVANCED = (
    "--developer",
    "--gateway-rotate-token",
    "--enable-playground",
    "--host",
    "--port",
    "--allow-unconfigured",
    "--daemon",
    "--import-session",
    "--title",
    "--save-session",
    "--no-save-session",
    "--parent-run",
    "--inbox",
    "--inbox-fd",
    "--context-window",
    "--output-reserve",
    "--safety-margin",
    "--builtins",
    "--mcp-config",
    "--mcp-only",
    "--memory-dir",
    "--request-timeout",
    "--embedding-base-url",
    "--embedding-model",
    "--progress",
    "--routine",
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
    "--cache-layout": "#183",
    "--window-messages": "#183",
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

# Long option name -> tier. The one table: help, tests and the CLI reference read it.
OPTION_TIERS: dict[str, str] = {
    **dict.fromkeys(_USER, USER),
    **dict.fromkeys(_ADVANCED, ADVANCED),
    **dict.fromkeys(EXPERIMENTAL_ISSUES, EXPERIMENTAL),
}

_HEADINGS = {
    USER: "Options",
    ADVANCED: "Advanced",
    EXPERIMENTAL: "Experimental (compared on the benchmark, may change or go away)",
}
DEFAULT_FOOTER = (
    'More options: melo --developer --help (or set [ui] mode = "developer").'
)
DEVELOPER_FOOTER = (
    "The mode changes only what --help shows; every option works in both modes. "
    "Which options bare melo takes: docs/reference/cli.md#interactive-ui."
)


def long_name(action: argparse.Action) -> str:
    return next(s for s in action.option_strings if s.startswith("--"))


def format_help(parser: argparse.ArgumentParser, *, developer: bool) -> str:
    """The parser's help: user options only, or every option grouped by tier.

    Each section lists its options in table order; an option missing from the
    table counts as advanced and comes last.
    """
    order = list(OPTION_TIERS)
    options = sorted(
        (a for a in parser._actions if a.option_strings),
        key=lambda a: (
            order.index(long_name(a)) if long_name(a) in order else len(order)
        ),
    )
    formatter = parser._get_formatter()
    formatter.add_usage(f"{parser.prog} [options] [--] [task ...]", [], [])
    formatter.add_text(parser.description)
    formatter.start_section("Arguments")
    formatter.add_arguments([a for a in parser._actions if not a.option_strings])
    formatter.end_section()
    for tier in (USER, ADVANCED, EXPERIMENTAL) if developer else (USER,):
        members = [
            a for a in options if OPTION_TIERS.get(long_name(a), ADVANCED) == tier
        ]
        if tier == EXPERIMENTAL:
            members = [_with_issue(a) for a in members]
        formatter.start_section(_HEADINGS[tier])
        formatter.add_arguments(members)
        formatter.end_section()
    formatter.add_text(DEVELOPER_FOOTER if developer else DEFAULT_FOOTER)
    return formatter.format_help()


def _with_issue(action: argparse.Action) -> argparse.Action:
    """A copy of ``action`` whose help names the issue that decides it."""
    labelled = copy.copy(action)
    labelled.help = (
        f"{action.help} (decided in {EXPERIMENTAL_ISSUES[long_name(action)]})"
    )
    return labelled
