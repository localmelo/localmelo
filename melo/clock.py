"""The instant "now" means for the model's date line and for memory recall.

``LOCALMELO_NOW`` is a developer setting for tests and memory benchmarks that
ask a question "on" a given day: ISO 8601 text (no zone means local time) or
epoch seconds. While it holds a valid value, :func:`now` returns that instant
and the model's ``Date:`` line (:func:`localmelo.melo.agent._chat.local_date_line`)
and recall (age decay of old chat turns) read it. Unset, empty or invalid, they
read the wall clock; an invalid value logs one warning.

Only those reads follow it. The times a chat session and its turns are stamped
with stay on the wall clock, so the variable never changes what is stored.
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

ENV = "LOCALMELO_NOW"
_warned: set[str] = set()


def parse(value: object) -> float | None:
    """Epoch seconds of *value*, or ``None`` when it is not a time.

    A number, or text of one (digits only: ``1684590000``), is epoch seconds.
    Other text is ISO 8601 (``2023-05-20``, ``2023-05-20T13:40:00``,
    ``2023-05-20T13:40:00Z``, ``2023-05-20T13:40:00+02:00``); without a zone
    it is local time.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        return seconds if math.isfinite(seconds) else None
    try:
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def pinned() -> float | None:
    """The instant ``LOCALMELO_NOW`` pins, or ``None`` when it pins none."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return None
    seconds = parse(raw)
    if seconds is None and raw not in _warned:
        _warned.add(raw)
        logger.warning(
            "%s=%r is not an ISO 8601 time or epoch seconds; using the real time",
            ENV,
            raw,
        )
    return seconds


def now() -> float:
    """Epoch seconds now: the pinned instant, else the wall clock."""
    seconds = pinned()
    return time.time() if seconds is None else seconds
