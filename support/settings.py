"""The keys of ``config.toml``: what ``melo config`` lists, and how a value is shown."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


def flat(obj: object, prefix: str = "") -> dict[str, Any]:
    """Every field of a config as ``section.name`` -> value (top-level names have no dot)."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        if field.name.startswith("_"):
            continue
        value = getattr(obj, field.name)
        if dataclasses.is_dataclass(value):
            out.update(flat(value, f"{prefix}{field.name}."))
        else:
            out[prefix + field.name] = value
    return out


def show(value: Any) -> str:
    """A value as ``melo config`` prints it."""
    return "unset" if value is None else json.dumps(value, ensure_ascii=False)
