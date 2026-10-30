"""``melo config``: where the config file is, and what in it differs from the defaults."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from localmelo.support import command_words, config, settings


def _home_relative(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home) :] if path.startswith(home + os.sep) else path


def main(argv: Sequence[str]) -> int:
    command_words._strict("config", argv, lambda parser: None)
    path = config.config_path()
    try:
        cfg = config.load()
    except (OSError, ValueError) as exc:  # a file that is not valid TOML
        print(f"melo: cannot read {_home_relative(path)}: {exc}", file=sys.stderr)
        return 1
    defaults = settings.flat(config.Config())
    changed = {
        key: value
        for key, value in settings.flat(cfg).items()
        if value != defaults[key]
    }
    print(_home_relative(path) + ("" if os.path.isfile(path) else " (no file yet)"))
    for key, value in changed.items():
        print(f"{key} = {settings.show(value)}")
    if not changed:
        print("every setting is at its default")
    return 0
