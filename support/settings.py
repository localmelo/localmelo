"""The keys of ``config.toml``, ``-X KEY=VALUE``, and what ``melo config`` lists.

``-X KEY=VALUE`` (long form ``--set``) changes one setting for one run. KEY is the
config key with its section and a dot (``retrieval.k``; a top-level key has no dot).
The value is read as TOML (``true``, ``5``, ``"v3"``), else taken as text; ``on`` and
``off`` also mean a boolean. A key that has an old option becomes that option before
the parser runs, so the run is the one the option gives. Any other key is applied to
the config this process loads (``config.set_overrides``), and ``save()`` never writes it.
"""

from __future__ import annotations

import difflib
import functools
import json
import sys
import tomllib
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, NoReturn

from localmelo.support import config
from localmelo.support.option_tiers import (
    OLD_SPELLINGS,
    OPTION_ROWS,
    REMOVED_SETTINGS,
    SETTINGS,
)

# The order of sections in listings; a key with no section comes first.
SECTIONS = (
    "",
    "agent",
    "retrieval",
    "memory",
    "endpoint",
    "sessions",
    "cli",
    "ui",
    "console",
    "approvals",
    "workspace",
    "scheduler",
    "browser",
    "tools",
    "web",
    "tasks",
    "chat",
    "gateway",
)


# Keys that loosen a safety rule, with the plain warning ``melo config`` prints
# beside them (on stderr, after the listing). A developer setting each.
WARNINGS = {
    "tools.web_fetch_allow_local": (
        "developer only. It lets web_fetch read the test servers listed in it "
        "on this computer (127.0.0.1:PORT), which web_fetch never reads otherwise. "
        "Leave it empty unless you run a test server of your own."
    ),
}


class Key(NamedTuple):
    name: str
    flag: str  # the old option that sets it for one run ("" if none)
    file: bool  # False: for one run only, config.toml does not hold it
    default: Any
    hint: Any  # the type of its value
    section: str


def warn(names: Sequence[str]) -> None:
    """Print the warning of each of *names* that has one (on stderr)."""
    for name in names:
        if name in WARNINGS:
            print(f"melo: warning: {name}: {WARNINGS[name]}", file=sys.stderr)


def fail(message: str) -> NoReturn:
    print(f"melo: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def show(value: Any) -> str:
    """A value as ``melo config`` prints it."""
    return "unset" if value is None else json.dumps(value, ensure_ascii=False)


@functools.cache
def catalog() -> dict[str, Key]:
    """Every key ``-X`` and ``melo config`` know, grouped by section."""
    cfg = config.Config()
    backends = set(config.VALID_CHAT_BACKENDS)
    flags = {item.key: item.flag for item in SETTINGS}
    out: dict[str, Key] = {}
    for name, default in config.flat(cfg).items():
        section, dot, field = name.rpartition(".")
        if name in ("chat_backend", "embedding_backend") or section in backends:
            continue  # the backend is `melo setup`'s
        if isinstance(default, dict):
            continue  # a table, not a value
        owner = getattr(cfg, section) if dot else cfg
        hint = config.hints(type(owner))[field]
        out[name] = Key(name, flags.get(name, ""), True, default, hint, section)
    for item in SETTINGS:
        if not item.file:
            hint = bool if item.key == "memory.keep_decision_text" else str
            section = item.key.partition(".")[0]
            out[item.key] = Key(item.key, item.flag, False, item.default, hint, section)
    rank = {name: n for n, name in enumerate(SECTIONS)}
    return dict(sorted(out.items(), key=lambda pair: rank.get(pair[1].section, 99)))


def spelled(name: str) -> str:
    """The option that the last part of the setting *name* spells (``--max-steps``)."""
    return "--" + name.rpartition(".")[2].replace("_", "-")


def option_for(name: str) -> str | None:
    """The option that the last part of *name* spells, as ``melo -h`` writes it.

    ``agent.max_steps`` spells ``--max-steps``, which ``-s N (--steps)`` replaces; None
    when no option has that name.
    """
    option = spelled(name)
    for names, meta, _ in OPTION_ROWS:
        spellings = names.split(", ")
        if option in spellings or OLD_SPELLINGS.get(option) == names:
            first, *others = spellings
            use = f"{first} {meta}".strip()
            return f"{use} ({', '.join(others)})" if others else use
    return None


def refuse_removed(words: Sequence[str]) -> None:
    """Exit 2 on the option of a removed setting in *words* (those before ``--``)."""
    gone = {spelled(name): name for name in REMOVED_SETTINGS}
    for word in words:
        if word == "--":
            return
        option = word.partition("=")[0]
        if option in gone:
            fail(f"{option} was removed: {REMOVED_SETTINGS[gone[option]]}.")


def lookup(name: str) -> Key:
    keys = catalog()
    if name in REMOVED_SETTINGS:
        fail(f"{name} was removed: {REMOVED_SETTINGS[name]}.")
    if name not in keys:
        # A name that is a key by its last part (`agent.mcp_config`) gets the nearest
        # keys; one that is only an option (`agent.max_steps`) gets the option.
        use = None if name.rpartition(".")[2] in keys else option_for(name)
        if use:
            fail(f"{name} is not a setting; use {use} for this run.")
        near = difflib.get_close_matches(name, keys, n=3, cutoff=0.6)
        hint = f"; nearest: {', '.join(near)}" if near else ""
        fail(f"unknown setting {name!r}{hint}. Every key: melo config keys")
    return keys[name]


def parse_value(text: str) -> Any:
    """TOML if *text* is a TOML value, else the text itself."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


_EXPECTED = {**config.KIND_WORDS, list: 'a list such as ["a", "b"] or a,b'}


def coerce(key: Key, value: Any, text: str) -> Any:
    """*value* (what TOML or the text gave) as the type of *key*; exit 2 if it is not."""
    base, optional = config.hint_base(key.hint)
    if optional and text.strip().lower() in ("", "none", "null", "unset"):
        return None
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if base is bool:
        word = str(value).lower()
        if word in ("true", "on", "false", "off"):
            return word in ("true", "on")
    elif base is int and number and isinstance(value, int):
        return value
    elif base is float and number:
        return float(value)
    elif base is str:
        return value if isinstance(value, str) else text
    elif base is list and isinstance(value, list | str):
        items = value.split(",") if isinstance(value, str) else value
        if all(isinstance(item, str) for item in items):
            return [item for item in items if item]
    fail(f"{key.name}: expected {_EXPECTED.get(base, 'a value')}, got {text!r}")


def check(key: Key, value: Any) -> Any:
    """*value* if the config accepts it for *key*; else exit 2 with why."""
    if key.file:
        cfg = config.Config()
        before = cfg.setting_errors()
        config.set_key(cfg, key.name, value)
        new = [error for error in cfg.setting_errors() if error not in before]
        if new:
            fail(f"{key.name}: {new[0]}")
    return value


def _flag_value(key: Key, value: Any, text: str) -> str:
    base, _ = config.hint_base(key.hint)
    if base is bool:
        return "on" if coerce(key, value, text) else "off"
    if base is list and isinstance(value, list):
        return ",".join(map(str, value))
    return value if isinstance(value, str) else text


def scan(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """argv without its settings, and the ``KEY=VALUE`` of each.

    ``-X KEY=VALUE``, ``-XKEY=VALUE``, ``--set KEY=VALUE`` and ``--set=KEY=VALUE`` count
    before a ``--``; after it they are task text. A word with a space in it (a task
    such as ``-Xmx is a java flag``) is left alone.
    """
    rest: list[str] = []
    pairs: list[str] = []
    args = list(argv)
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            rest.extend(args[i:])
            break
        if token in ("-X", "--set"):
            if i + 1 >= len(args):
                fail(f"{token} needs KEY=VALUE")
            pairs.append(args[i + 1])
            i += 2
        elif token.startswith("--set="):
            pairs.append(token[6:])
            i += 1
        elif token.startswith("-X") and "=" in token and " " not in token.split("=")[0]:
            pairs.append(token[2:])
            i += 1
        else:
            rest.append(token)
            i += 1
    return rest, pairs


# An option that means "this key is this value", with no value of its own.
_SWITCHES = {"memory.keep_decision_text": True, "sessions.record_direct": False}


def expand(argv: Sequence[str], *, options: bool = True) -> list[str]:
    """Apply the ``-X`` settings in *argv*; return argv without them.

    A key with an old option becomes that option (before any ``--``), so it is parsed,
    checked and refused as the option is. Every other key, and every key when *options*
    is False, becomes an override of the config this process loads.
    """
    rest, pairs = scan(argv)
    refuse_removed(rest)
    found: list[str] = []
    values: dict[str, Any] = {}
    for pair in pairs:
        name, equals, text = pair.partition("=")
        if not equals:
            fail(f"-X needs KEY=VALUE, got {pair!r}")
        key = lookup(name.strip())
        value = parse_value(text)
        if not (options and key.flag):
            values[key.name] = check(key, coerce(key, value, text))
        elif key.name not in _SWITCHES:
            found.append(f"{key.flag}={_flag_value(key, value, text)}")
        elif coerce(key, value, text) is _SWITCHES[key.name]:
            found.append(key.flag)
        elif key.file:  # record_direct = true beats a file that says false
            values[key.name] = True
    config.set_overrides(values)
    cut = rest.index("--") if "--" in rest else len(rest)
    return [*rest[:cut], *found, *rest[cut:]]


def removed_in(data: Mapping[str, Any]) -> list[str]:
    """One note for each removed setting that a parsed ``config.toml`` still holds."""
    held = file_keys(data)
    notes = [
        f"{name} was removed ({why}); the line in the file is ignored"
        for name, why in REMOVED_SETTINGS.items()
        if name in held
    ]
    agent = data.get("agent")
    if isinstance(agent, dict) and agent.get("window_eviction") == "slide":
        # config._SPELLINGS reads it as "chunk".
        notes.append(
            'agent.window_eviction = "slide" was removed (a full window drops in '
            'chunks); it reads as "chunk"'
        )
    return notes


def file_keys(data: Mapping[str, Any]) -> set[str]:
    """The keys a parsed ``config.toml`` holds."""
    keys: set[str] = set()
    for name, value in data.items():
        if isinstance(value, dict):
            keys.update(f"{name}.{sub}" for sub in value)
        else:
            keys.add(name)
    return keys
