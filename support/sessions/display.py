"""How the surfaces that hold sessions show times, tokens, paths and run ends.

The terminal UI and the web console format these the same way, so a turn
reads alike on both: :func:`format_duration` and :func:`format_tokens`,
:func:`home_label` and :func:`display_path`, :func:`status_line` (the
direct mode's line for a run that did not complete), :func:`clean_output`
for text a tool or a model printed, and :func:`localmelo_version`.
Nothing here imports the UI's libraries or the gateway.
"""

from __future__ import annotations

import functools
import importlib.metadata
import math
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from localmelo.support.progress_line import run_elapsed

# Characters of the error that a status line shows at most (as direct mode).
STATUS_LINE_ERROR_CHARS: Final = 500

# Escape sequences: CSI; OSC, DCS, SOS, PM and APC up to BEL or ST; nF; the
# two-character ones; and their 8-bit forms.
_ESCAPES: Final = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b[\]PX^_].*?(?:\x07|\x1b\\|\x9c)"
    r"|\x1b[ -/]+[0-~]"
    r"|\x1b[0-~]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|[\x90\x98\x9d\x9e\x9f].*?(?:\x07|\x1b\\|\x9c)",
    re.DOTALL,
)

# Control characters other than \n and \t, and Unicode's tag characters.
_CONTROLS: Final = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\U000e0000-\U000e007f]",
)


def format_duration(seconds: float) -> str:
    """``0.4s`` and ``6.4s`` below 10 s (tenths, cut); from 10 s ``12s``, ``2m03s``.

    From 10 s it is the progress line's run time, so a finished turn and a
    running one read alike.
    """
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{math.floor(seconds * 10) / 10:.1f}s"
    return run_elapsed(seconds)


def format_tokens(n: int) -> str:
    """``412``, ``3.4k``, ``61k``, ``1.2M``: cut, never rounded up."""
    n = max(0, n)
    if n < 1_000:
        return str(n)
    if n < 10_000:
        return f"{n // 100 / 10:.1f}k"
    if n < 1_000_000:
        return f"{n // 1_000}k"
    return f"{n // 100_000 / 10:.1f}M"


def format_bytes(n: int) -> str:
    """``412 bytes``, ``1.2 KB``, ``312 KB``, ``12 MB``: decimal units, cut.

    1 KB is 1,000 bytes, as macOS shows sizes: below 1,000 bytes the count
    with ``bytes``; then one decimal below 10 of a unit, and whole units
    from 10. A value is cut, never rounded up.
    """
    n = max(0, int(n))
    if n < 1_000:
        return f"{n} bytes"
    value = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1_000
        if value < 1_000 or unit == "TB":
            if value < 10:
                return f"{math.floor(value * 10) / 10:.1f} {unit}"
            return f"{math.floor(value)} {unit}"
    raise AssertionError("unreachable")


def _homes() -> tuple[str, ...]:
    home = os.path.expanduser("~").rstrip(os.sep) or os.sep
    real = os.path.realpath(home)
    return (home,) if real == home else (home, real)


def home_label(path: str) -> str:
    """*path* with the home folder written ``~``."""
    for home in _homes():
        if path == home:
            return "~"
        if home != os.sep and path.startswith(home + os.sep):
            return "~" + path[len(home) :]
    return path


def display_path(path: str, root: str | None) -> str:
    """*path* relative to *root* when it is inside it, else :func:`home_label`."""
    if root:
        root = root.rstrip(os.sep) or os.sep
        if path == root:
            return "."
        prefix = root if root == os.sep else root + os.sep
        if path.startswith(prefix):
            return path[len(prefix) :]
    return home_label(path)


def approval_target(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """A ``file_write`` request's target, or None for any other tool.

    ``{"path_label": display_path(...), "exists": bool, "size_bytes": int}``.
    The facts come from ``os.open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)``
    and ``fstat``, as the panel's diff reads them: a final symlink is not
    followed and a FIFO never blocks. ``exists`` is false only for a missing
    path; any other failure to open counts as existing, with size 0.
    """
    if request.get("tool_name") != "file_write":
        return None
    arguments = request.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    path = str(arguments.get("path") or "")
    root = request.get("workspace_root")
    label = display_path(path, root if isinstance(root, str) else None)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {"path_label": label, "exists": False, "size_bytes": 0}
    except OSError:
        return {"path_label": label, "exists": True, "size_bytes": 0}
    try:
        info = os.fstat(fd)
    finally:
        os.close(fd)
    return {"path_label": label, "exists": True, "size_bytes": info.st_size}


def status_line(status: str, error: str) -> str:
    """``melo: run ended with status <status>: <reason>``, as direct mode prints it.

    The reason is *error* with its whitespace collapsed, cut at
    :data:`STATUS_LINE_ERROR_CHARS` characters with ``...``, or ``no error
    message``.
    """
    reason = " ".join(error.split())
    if len(reason) > STATUS_LINE_ERROR_CHARS:
        reason = reason[: STATUS_LINE_ERROR_CHARS - 3] + "..."
    return f"melo: run ended with status {status}: {reason or 'no error message'}"


def clean_output(text: str) -> str:
    """*text* as a terminal would leave it on screen, with nothing it could act on.

    Escape sequences go, so output cannot move the cursor, set a title or
    write the clipboard; ``\\r\\n`` reads as ``\\n``, and of each line only
    the text after its last ``\\r`` stays, as a progress bar ends; other
    control characters but ``\\n`` and ``\\t`` go, and so do Unicode's tag
    characters. Tabs stay: expanding them is the terminal's job.
    """
    text = _ESCAPES.sub("", text.replace("\r\n", "\n"))
    lines = [line.rpartition("\r")[2] for line in text.split("\n")]
    return _CONTROLS.sub("", "\n".join(lines))


@functools.cache
def localmelo_version() -> str:
    """``git describe --tags --always --dirty`` of the package's checkout.

    Only when the package directory is the top of a git checkout; else the
    installed package's version, or ``unknown``. Computed once a process,
    as direct mode computes it per run.
    """
    package_dir = Path(__file__).resolve().parents[2]
    # Read-only: never take the index lock another git command may hold.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args],
            cwd=package_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return done.stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() == package_dir:
            described = git("describe", "--tags", "--always", "--dirty")
            if described:
                return described
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return importlib.metadata.version("localmelo")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
