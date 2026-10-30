"""The terminal's name for the file-edit helpers in ``support/sessions/edits.py``.

The approval panel and the feed block read a ``file_write`` target and its
diff from there, as the web console's approval card does.
"""

from __future__ import annotations

from localmelo.support.sessions.edits import (
    DIFF_MAX_BYTES,
    DiffLine,
    Target,
    diff,
    read_target,
)

__all__ = ["DIFF_MAX_BYTES", "DiffLine", "Target", "diff", "read_target"]
