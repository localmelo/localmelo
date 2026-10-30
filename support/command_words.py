"""Command words: ``melo serve`` and its kin become the options they replace.

A mode is a word, not a flag. Each word except ``task`` and ``config`` is rewritten to
the options the parser has always taken, before either parser runs, so it behaves
exactly as they do; ``task`` runs the ``melo-task`` client and ``config`` is new.

A word counts only as the first argument and only as an exact match: ``melo "serve
the docs"`` (one quoted argument) and ``melo -- serve the docs`` are tasks. Unquoted
extra words (``melo setup a venv``) end in a hint and exit 2, so a typo never starts
a model run.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from typing import NoReturn

COMMANDS = ("setup", "config", "serve", "console", "task", "worker", "import")
DAEMON_ACTIONS = ("install", "uninstall", "status")
TASK_VERBS = ("new", "list", "show", "pause", "resume", "stop", "follow", "level")

# (usage, description, options, example) for ``melo COMMAND -h``.
_HELP = {
    "setup": (
        "melo setup",
        "Choose the model server again and save it, then open the chat.",
        "",
        "melo setup",
    ),
    "config": (
        "melo config",
        "Show where the config file is and which settings differ from the defaults.",
        "",
        "melo config",
    ),
    "serve": (
        "melo serve [install|uninstall|status] [options]",
        "Run the gateway (web console and HTTP API). install, uninstall and status\n"
        "manage it as a login service on macOS.",
        "  --host H        listen address (default 127.0.0.1)\n"
        "  -p, --port N    listen port (default 8401; 0 picks a free one)\n"
        "  --playground    enable the development playground routes\n"
        "  --force         start even when no model server is set up yet",
        "melo serve -p 9000",
    ),
    "console": (
        "melo console [--reset] [options]",
        "Print a one-use link that opens the web console (valid for 60 seconds).\n"
        "--reset signs every browser out and makes a new access token instead.",
        "  --reset         rotate the access token\n"
        "  --host H        the gateway's address (default: the configured one)\n"
        "  -p, --port N    the gateway's port (default: the configured one)",
        "melo console",
    ),
    "task": (
        "melo task [--gateway URL] [--json] VERB ...",
        "Manage the background tasks of a running gateway. VERB is new, list, show,\n"
        "pause, resume, stop, follow or level; melo task VERB -h explains one.",
        "",
        "melo task list",
    ),
    "worker": (
        "melo worker BRIEF [options]",
        "Run a worker brief and print one JSON result. Its Check: lines run when the\n"
        "model is done, and a failed check exits 4. Every task option works too.",
        "  --inbox FILE    read instructions added while it runs from FILE\n"
        "  --inbox-fd N    read them from the inherited pipe N\n"
        "  --parent ID     link the run to a launcher run\n"
        "  --title T       title of the worker session\n"
        "  -o FORMAT       json (default) or stream-json",
        "melo worker brief.md -w ~/project -o stream-json",
    ),
    "import": (
        "melo import FORMAT FILE [--title T]",
        "Import a worker history as a worker session. FORMAT is dsh-jsonl. Importing\n"
        "the same file again changes nothing.",
        "  --title T       title of the imported session",
        "melo import dsh-jsonl run.jsonl",
    ),
}

# melo -h all: (what to type, what it does).
SUMMARIES = (
    ("setup", "choose the model server again"),
    ("config", "show or change the settings"),
    ("serve [ACTION]", "run the gateway; install | uninstall | status"),
    ("console", "print a one-use link to the web console"),
    ("task VERB", "manage background tasks (melo task -h)"),
    ("worker BRIEF", "run a worker brief and print its result"),
    ("import FORMAT FILE", "import a worker history"),
)


def help_text(word: str) -> str:
    usage, description, options, example = _HELP[word]
    parts = [f"usage: {usage}", description]
    if options:
        parts.append("Options:\n" + options)
    parts.append(f"Example: {example}")
    return "\n\n".join(parts) + "\n"


def refuse(word: str, rest: Sequence[str]) -> NoReturn:
    """A task that starts with a command word, typed without quotes."""
    task = " ".join([word, *rest])
    quoted = '"' + re.sub(r'(["\\$`])', r"\\\1", task) + '"'
    sys.stderr.write(
        f'melo: "{word}" is a melo command. '
        f"To run this as a task, quote it: melo {quoted}\n"
    )
    raise SystemExit(2)


def _strict(
    word: str, rest: Sequence[str], build: Callable[[argparse.ArgumentParser], None]
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse the options of *word*; a bare extra word is a task, anything else an error."""
    parser = argparse.ArgumentParser(
        prog=f"melo {word}", add_help=False, allow_abbrev=False
    )
    build(parser)
    args, extras = parser.parse_known_args(list(rest))
    if any(not extra.startswith("-") for extra in extras):
        refuse(word, rest)
    if extras:
        parser.error("unrecognized arguments: " + " ".join(extras))
    return parser, args


def _gateway_address(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host")
    parser.add_argument("-p", "--port", type=int)


def _address(args: argparse.Namespace) -> list[str]:
    return [
        *(["--host", args.host] if args.host is not None else []),
        *(["--port", str(args.port)] if args.port is not None else []),
    ]


def _setup(rest: list[str]) -> list[str]:
    _strict("setup", rest, lambda parser: None)
    return ["--reconfigure"]


def _serve(rest: list[str]) -> list[str]:
    def build(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("action", nargs="?")
        _gateway_address(parser)
        parser.add_argument("--playground", action="store_true")
        parser.add_argument("--force", action="store_true")

    parser, args = _strict("serve", rest, build)
    if args.action not in (None, *DAEMON_ACTIONS):
        refuse("serve", rest)
    if args.action and (args.playground or args.force):
        parser.error("--playground and --force are for running the gateway")
    return [
        *(["--daemon", args.action] if args.action else ["--serve"]),
        *_address(args),
        *(["--enable-playground"] if args.playground else []),
        *(["--allow-unconfigured"] if args.force else []),
    ]


def _console(rest: list[str]) -> list[str]:
    def build(parser: argparse.ArgumentParser) -> None:
        _gateway_address(parser)
        parser.add_argument("--reset", action="store_true")

    _, args = _strict("console", rest, build)
    flag = "--gateway-rotate-token" if args.reset else "--gateway-login"
    return [flag, *_address(args)]


def _import(rest: list[str]) -> list[str]:
    def build(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("format")
        parser.add_argument("file")
        parser.add_argument("--title")

    _, args = _strict("import", rest, build)
    return [
        "--import-session",
        args.format,
        args.file,
        *(["--title", args.title] if args.title is not None else []),
    ]


def _worker(rest: list[str]) -> list[str]:
    """``melo worker BRIEF`` is ``--brief BRIEF``; every other option is passed on."""
    if not rest or rest[0].startswith("-"):
        parser = argparse.ArgumentParser(
            prog="melo worker", usage="melo worker BRIEF [options]"
        )
        parser.error("the brief file comes first")
    if rest[1:2] and not rest[1].startswith("-"):
        refuse("worker", rest)
    tail = []
    for token in rest[1:]:
        name, equals, value = token.partition("=")
        tail.append("--parent-run" + equals + value if name == "--parent" else token)
    return ["--brief", rest[0], *tail]


def _config(rest: list[str]) -> NoReturn:
    from localmelo.support import config_cli

    raise SystemExit(config_cli.main(rest))


def _task(rest: list[str]) -> NoReturn:
    if rest and not rest[0].startswith("-") and rest[0] not in TASK_VERBS:
        refuse("task", rest)
    from localmelo.support import task_cli

    raise SystemExit(task_cli.main(rest, prog="melo task"))


_EXPANDERS: dict[str, Callable[[list[str]], list[str]]] = {
    "setup": _setup,
    "config": _config,
    "serve": _serve,
    "console": _console,
    "task": _task,
    "worker": _worker,
    "import": _import,
}


def expand(argv: Sequence[str]) -> list[str]:
    """The options a command line stands for; a line with no command word is unchanged.

    ``melo task`` and ``melo config`` do their work here and exit.
    """
    if not argv or argv[0] not in COMMANDS:
        return list(argv)
    word, rest = argv[0], list(argv[1:])
    if word != "task" and ("-h" in rest or "--help" in rest):
        sys.stdout.write(help_text(word))
        raise SystemExit(0)
    return _EXPANDERS[word](rest)
