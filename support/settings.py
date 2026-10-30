"""The keys of ``config.toml``, ``-X KEY=VALUE``, and what ``melo config`` lists.

``-X KEY=VALUE`` (long form ``--set``) changes one setting for one run. KEY is the
config key with its section and a dot (``retrieval.k``; a top-level key has no dot).
The value is read as TOML (``true``, ``5``, ``"v3"``), else taken as text; ``on`` and
``off`` also mean a boolean. A key that has an old option becomes that option before
the parser runs, so the run is the one the option gives. Any other key is applied to
the config this process loads (``config.set_overrides``), and ``save()`` never writes it.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import json
import sys
import tomllib
import typing
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, NoReturn

from localmelo.support import config
from localmelo.support.option_tiers import SETTINGS

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
    "scheduler",
    "gateway",
)


class Key(NamedTuple):
    name: str
    flag: str  # the old option that sets it for one run ("" if none)
    file: bool  # False: for one run only, config.toml does not hold it
    default: Any
    hint: Any  # the type of its value
    section: str


def fail(message: str) -> NoReturn:
    print(f"melo: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def flat(obj: object, prefix: str = "") -> dict[str, Any]:
    """Every field of a config as ``section.name`` -> value (top-level names have no dot)."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        if not field.compare:
            continue  # bookkeeping (a private flag, the spelling of a backend), not a setting
        value = getattr(obj, field.name)
        if dataclasses.is_dataclass(value):
            out.update(flat(value, f"{prefix}{field.name}."))
        else:
            out[prefix + field.name] = value
    return out


def show(value: Any) -> str:
    """A value as ``melo config`` prints it."""
    return "unset" if value is None else json.dumps(value, ensure_ascii=False)


@functools.cache
def catalog() -> dict[str, Key]:
    """Every key ``-X`` and ``melo config`` know, grouped by section."""
    cfg = config.Config()
    backends = {*config._LOCAL_SECTIONS, *config._CLOUD_SECTIONS}
    flags = {item.key: item.flag for item in SETTINGS}
    out: dict[str, Key] = {}
    for name, default in flat(cfg).items():
        section, dot, field = name.rpartition(".")
        if name in ("chat_backend", "embedding_backend") or section in backends:
            continue  # the backend is `melo setup`'s
        if isinstance(default, dict):
            continue  # a table, not a value
        owner = getattr(cfg, section) if dot else cfg
        hint = typing.get_type_hints(type(owner))[field]
        out[name] = Key(name, flags.get(name, ""), True, default, hint, section)
    for item in SETTINGS:
        if not item.file:
            hint = bool if item.key == "memory.keep_decision_text" else str
            section = item.key.partition(".")[0]
            out[item.key] = Key(item.key, item.flag, False, item.default, hint, section)
    rank = {name: n for n, name in enumerate(SECTIONS)}
    return dict(sorted(out.items(), key=lambda pair: rank.get(pair[1].section, 99)))


def lookup(name: str) -> Key:
    keys = catalog()
    if name not in keys:
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


_EXPECTED = {
    bool: "true or false",
    int: "a whole number",
    float: "a number",
    str: "text",
    list: 'a list such as ["a", "b"] or a,b',
}


def _base(hint: Any) -> tuple[Any, bool]:
    """The type of a value, and whether ``None`` (unset) is allowed."""
    args = typing.get_args(hint)
    if typing.get_origin(hint) is list:
        return list, False
    inner = [arg for arg in args if arg is not type(None)]
    return (inner[0] if inner else hint), len(inner) != len(args)


def coerce(key: Key, value: Any, text: str) -> Any:
    """*value* (what TOML or the text gave) as the type of *key*; exit 2 if it is not."""
    base, optional = _base(key.hint)
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
    base, _ = _base(key.hint)
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


def file_keys(data: Mapping[str, Any]) -> set[str]:
    """The keys a parsed ``config.toml`` holds."""
    keys: set[str] = set()
    for name, value in data.items():
        if isinstance(value, dict):
            keys.update(f"{name}.{sub}" for sub in value)
        else:
            keys.add(name)
    return keys
