"""``melo config``: where the config file is, what differs from the defaults, and the keys.

``melo config`` alone shows the path and the settings that differ from the defaults;
``keys [FILTER]`` lists every setting with its value, default and source; ``get KEY``
prints one; ``set KEY VALUE`` checks one and saves it in the file.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Sequence
from typing import Any

from localmelo.support import command_words, config, settings

VERBS = ("keys", "get", "set")


def _home_relative(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home) :] if path.startswith(home + os.sep) else path


def _read() -> tuple[config.Config, dict[str, Any]] | None:
    """The config as this process sees it, and the file's own tables; None if unreadable."""
    path = config.config_path()
    try:
        cfg = config.load()
        data: dict[str, Any] = {}
        if os.path.isfile(path):
            with open(path, "rb") as source:
                data = tomllib.load(source)
    except (OSError, ValueError) as exc:  # a file that is not valid TOML
        print(f"melo: cannot read {_home_relative(path)}: {exc}", file=sys.stderr)
        return None
    return cfg, data


def _show_changes() -> int:
    path = config.config_path()
    read = _read()
    if read is None:
        return 1
    defaults = settings.flat(config.Config())
    changed = {
        key: value
        for key, value in settings.flat(read[0]).items()
        if value != defaults[key]
    }
    print(_home_relative(path) + ("" if os.path.isfile(path) else " (no file yet)"))
    for key, value in changed.items():
        print(f"{key} = {settings.show(value)}")
    if not changed:
        print("every setting is at its default")
    return 0


def _value(key: settings.Key, cfg: config.Config) -> Any:
    """What *key* is for this run: a config field, else its -X value, else its default."""
    if key.file:
        return config.get_key(cfg, key.name)
    return config.overrides().get(key.name, key.default)


def _keys(pattern: str) -> int:
    read = _read()
    if read is None:
        return 1
    cfg, data = read
    in_file = settings.file_keys(data)
    given = config.overrides()
    rows = [("key", "value", "default", "source")]
    for key in settings.catalog().values():
        if pattern.lower() not in f"{key.name} {key.flag}".lower():
            continue
        source = (
            "-X" if key.name in given else "file" if key.name in in_file else "default"
        )
        note = "" if key.file else " (one run only)"
        rows.append(
            (
                key.name,
                settings.show(_value(key, cfg)),
                settings.show(key.default),
                source + note,
            )
        )
    if len(rows) == 1:
        print(f"melo: no setting matches {pattern!r}", file=sys.stderr)
        return 1
    widths = [max(len(row[col]) for row in rows) for col in range(3)]
    for row in rows:
        print("  ".join([*(row[col].ljust(widths[col]) for col in range(3)), row[3]]))
    return 0


def _get(name: str) -> int:
    key = settings.lookup(name)
    read = _read()
    if read is None:
        return 1
    print(settings.show(_value(key, read[0])))
    return 0


def _set(name: str, text: str) -> int:
    key = settings.lookup(name)
    if not key.file:
        settings.fail(f"{key.name} is for one run only; use melo -X {key.name}=VALUE")
    value = settings.check(key, settings.coerce(key, settings.parse_value(text), text))
    try:
        cfg = config.load()
        config.set_key(cfg, key.name, value)
        config.save(cfg)
    except (OSError, ValueError, config.ConfigError) as exc:
        print(f"melo: cannot save {key.name}: {exc}", file=sys.stderr)
        return 1
    print(
        f"{key.name} = {settings.show(value)}  ({_home_relative(config.config_path())})"
    )
    return 0


def main(argv: Sequence[str]) -> int:
    rest = settings.expand(argv, options=False)  # -X shows as a source in `keys`
    if not rest:
        return _show_changes()
    verb, args = rest[0], rest[1:]
    if verb == "set" and config.overrides():
        settings.fail("-X does not apply to melo config set, which writes the file")
    if verb == "keys" and len(args) <= 1:
        return _keys(args[0] if args else "")
    if verb == "get" and len(args) == 1:
        return _get(args[0])
    if verb == "set" and len(args) == 2:
        return _set(*args)
    if verb in VERBS:
        settings.fail(f"usage: melo config {_USAGE[verb]}")
    if verb.startswith("-"):
        settings.fail(f"unrecognized arguments: {' '.join(rest)}")
    command_words.refuse("config", rest)


_USAGE = {"keys": "keys [FILTER]", "get": "get KEY", "set": "set KEY VALUE"}
