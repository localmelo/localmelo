"""The command-line flags that set one AgentOptions field.

A new flag of that kind is one row in :data:`FLAG_OPTIONS` and its parser
entry. The direct-mode options, the refusal that names the direct-mode flags
and the settings a flag keeps from the config file all read this table, so
nothing else lists them. The module imports nothing heavy: ``melo --version``
loads it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, Final, NamedTuple


class FlagOption(NamedTuple):
    """A flag and the AgentOptions field it sets.

    *dest* is the flag's name in the parsed arguments and *convert* turns its
    value into the field's (``on`` or ``off`` into a boolean, say).
    """

    dest: str
    field: str
    convert: Callable[[Any], Any]


def _same(value: Any) -> Any:
    return value


def _on(value: str) -> bool:
    return value == "on"


def _flag(
    dest: str, field: str | None = None, convert: Callable[[Any], Any] = _same
) -> FlagOption:
    return FlagOption(dest, field or dest, convert)


# The order is the order a refusal names the flags in.
FLAG_OPTIONS: Final = (
    _flag("window_eviction"),
    _flag("window_messages"),
    _flag("max_steps"),
    _flag("attempt_policy"),
    _flag("tool_exposure"),
    _flag("memory", convert=_on),
    _flag("memory_dir"),
    _flag("timeout", "timeout_s"),
    _flag("reasoning"),
    _flag("note_writer"),
    _flag("retriever"),
    _flag("recall_at"),
    _flag("recall_k"),
    _flag("recall_kinds"),
    _flag("recall_placement"),
    _flag("recall_min_score"),
    _flag("keep_decision_text"),
)


def flag_options(args: argparse.Namespace) -> dict[str, Any]:
    """The AgentOptions fields the flags in *args* set, by field name.

    A flag that was not given (None, or a switch left False) sets nothing. The
    names are the settings the config file must leave alone.
    """
    fields: dict[str, Any] = {}
    for row in FLAG_OPTIONS:
        value = getattr(args, row.dest, None)
        if value is not None and value is not False:
            fields[row.field] = row.convert(value)
    return fields
