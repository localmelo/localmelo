"""The request bodies API v2 accepts: unknown keys refused, strings UTF-8.

SQLite stores text, and a lone surrogate (``"\\ud800"``, which a JSON body
can carry and Python accepts) encodes to nothing: written, it fails the
turn after the route answered. The walk runs on the raw input, before any
field validates and before any write.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .errors import ApiError


def _utf8_problem(value: Any, path: str) -> str | None:
    """``field: problem`` for the first string that is not UTF-8, else ``None``.

    Walks the raw body, so a nested or listed string names its own path
    (``turn.text``, ``attachments.0.text``).
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return f"{path or 'request'}: the text must be valid UTF-8"
        return None
    if isinstance(value, dict):
        for key, inner in value.items():
            found = _utf8_problem(inner, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            found = _utf8_problem(inner, f"{path}.{index}" if path else str(index))
            if found is not None:
                return found
    return None


class Body(BaseModel):
    """The base of every write route's request body: see the module's rules."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _utf8(cls, data: Any) -> Any:
        problem = _utf8_problem(data, "")
        if problem is not None:
            raise ApiError(400, "invalid_request", problem)
        return data
