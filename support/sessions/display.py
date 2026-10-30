"""How the surfaces that hold sessions show times, tokens, paths and run ends.

The terminal UI and the web console format these the same way, so a turn
reads alike on both: :func:`format_duration` and :func:`format_tokens`,
:func:`home_label` and :func:`display_path`, :func:`status_line` (the
direct mode's line for a run that did not complete), :func:`clean_output`
for text a tool or a model printed, :func:`localmelo_version`, :func:`display_version`,
:func:`split_thinking` (re-exported from the core), and the words of a run's
end: :data:`STATUS_TEXT`, looked up by status and cause through
:func:`status_text`, and :func:`approval_tag` for how a call's approval
ended, and :func:`activity_text` for what a running session is doing, in
plain words (:func:`event_activity` reads it from the session's newest
events). Nothing here imports the UI's libraries or the gateway.
"""

from __future__ import annotations

import functools
import importlib.metadata
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

from localmelo.melo.schema import split_thinking as split_thinking
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


def session_age(updated_at: float, now: float) -> str:
    """A session's age, in whole minutes, hours or days."""
    seconds = max(0, int(now - updated_at))
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        count = seconds // size
        if count:
            return f"{count} {unit}{'' if count == 1 else 's'} ago"
    return "just now"


def resume_note(
    mode: str, turns: int, checkpoint_turns: int, *, detail: bool = True
) -> str:
    """Use the runner's post-eviction counts, not the number of stored turns.

    Without *detail* it only says how many turns the model remembers; what became
    of their tool calls and steps is for developer mode.
    """

    def count(n: int) -> str:
        return f"{n} turn{' is' if n == 1 else 's are'} back in the model's memory"

    if not detail:
        return count(turns) + "."
    if mode != "window":
        return count(turns) + (
            "; its tool calls are not." if turns == 1 else "; their tool calls are not."
        )
    note = count(checkpoint_turns) + ", with their tool steps"
    tail = turns - checkpoint_turns
    return note + (
        f"; {tail} later turn{'s' if tail != 1 else ''} without them." if tail else "."
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


def folder_name(root: str | None) -> str:
    """The last part of the workspace *root*, for a question that names the folder."""
    name = (root or "").rstrip(os.sep).rsplit(os.sep, 1)[-1]
    return name or "the workspace"


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
    and ``fstat``, as the CLI panel's diff reads them: a final symlink is not
    followed and a FIFO never blocks. ``exists`` is false only for a missing
    path; any other failure to open counts as existing, with size 0.

    A write of text over an existing file adds ``"diff"``, what it replaces,
    from :func:`~localmelo.support.sessions.edits.approval_diff`.
    """
    if request.get("tool_name") != "file_write":
        return None
    arguments = request.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    path = str(arguments.get("path") or "")
    root = request.get("workspace_root")
    label = display_path(path, root if isinstance(root, str) else None)
    exists, size = file_facts(path)
    target: dict[str, Any] = {
        "path_label": label,
        "exists": exists,
        "size_bytes": size,
    }
    content = arguments.get("content")
    if exists and isinstance(content, str):
        # Imported here: edits reads file_facts from this module.
        from localmelo.support.sessions import edits

        target["diff"] = edits.approval_diff(path, content)
    return target


def file_facts(path: str) -> tuple[bool, int]:
    """(exists, size) for *path*, without following a final symlink or blocking.

    The one rule for a ``file_write`` target's facts, shared by
    :func:`approval_target` and the CLI panel's diff.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False, 0
    except OSError:
        return True, 0
    try:
        info = os.fstat(fd)
    finally:
        os.close(fd)
    return True, info.st_size


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


# How much a run's end asks of a person: "error" marks what they must fix
# outside the conversation (the model server, the tools, a bug), "warn" a
# run the model did not finish, "stopped" is neutral, and "ok" completed.
StatusSeverity = Literal["ok", "stopped", "warn", "error"]


@dataclass(frozen=True)
class StatusText:
    """A run status's word, severity and title, for a status line and a chip."""

    word: str
    severity: StatusSeverity
    title: str


# Keyed by status, and by ``(status, cause)`` for a cause with a word of its
# own. :func:`status_text` falls back from the cause row to the status row,
# so a cause this table does not name reads as its status does.
STATUS_TEXT: Final[Mapping[str | tuple[str, str], StatusText]] = {
    "completed": StatusText("Done", "ok", "Done"),
    "cancelled": StatusText("Stopped", "stopped", "Stopped"),
    "failed": StatusText("Failed", "warn", "A check stopped the run"),
    "stuck": StatusText("Stuck", "warn", "Stopped: the same calls kept repeating"),
    "gate_stopped": StatusText(
        "Stopped after reflection", "warn", "Stopped after reflection"
    ),
    "budget_exhausted": StatusText("Out of steps", "warn", "Ran out of steps"),
    "context_budget_exceeded": StatusText(
        "Context too large", "warn", "Context budget exceeded"
    ),
    "timeout": StatusText("Timed out", "warn", "Timed out"),
    "provider_error": StatusText(
        "Model server error", "error", "Could not get an answer from the model server"
    ),
    "setup_error": StatusText(
        "Tools failed to start", "error", "Tools could not start"
    ),
    # A run crashes as it runs (cause "run"); recovery's crash is the
    # cause "crash" row below, which is not a bug in localmelo.
    "crashed": StatusText("Crashed", "error", "Something went wrong inside localmelo"),
    ("crashed", "run"): StatusText(
        "Crashed", "error", "Something went wrong inside localmelo"
    ),
    ("crashed", "crash"): StatusText(
        "Interrupted", "warn", "Interrupted: melo stopped unexpectedly"
    ),
    ("cancelled", "user_stop"): StatusText("Stopped", "stopped", "Stopped by you"),
    ("cancelled", "crash"): StatusText(
        "Interrupted", "warn", "Interrupted: melo stopped unexpectedly"
    ),
    ("cancelled", "shutdown"): StatusText(
        "Interrupted", "warn", "Interrupted: melo stopped before this turn finished"
    ),
    ("cancelled", "closed"): StatusText(
        "Stopped", "stopped", "Stopped: the session was closed"
    ),
    ("cancelled", "takeover"): StatusText(
        "Stopped",
        "stopped",
        "Stopped: another melo window took over this session",
    ),
    ("cancelled", "park"): StatusText("Parked", "stopped", "Parked"),
}


def status_text(status: str, cause: str = "") -> StatusText:
    """The words of *status*, ended by *cause*: the cause row first, then the status.

    A status this version does not know reads as its own name, in the
    ``warn`` style, so a newer writer's status still shows.
    """
    text = STATUS_TEXT.get((status, cause))
    if text is None:
        text = STATUS_TEXT.get(status)
    if text is None:
        return StatusText(status or "Unknown", "warn", status or "Unknown")
    return text


@dataclass(frozen=True)
class ApprovalText:
    """How a call's approval ended: its short tag and its long text."""

    short: str
    long: str


def approval_tag(
    status: str, error_category: str, approval: Mapping[str, Any] | None
) -> ApprovalText | None:
    """The short and long text of how the approval of a call ended, or None.

    *approval* is ``tool.finished``'s ``data.approval`` (``allowed``,
    ``origin``, ``remember`` and ``reason``), or None when the call was not
    asked about. The texts follow the call's *status* and *error_category*
    and the approval: a person's allow or denial, a session grant, a policy,
    a timeout, a policy that failed, and a request the turn's end withdrew.
    A denial's reason is collapsed to one line.
    """
    if not isinstance(approval, Mapping):
        return None
    origin = approval.get("origin")
    reason = " ".join(str(approval.get("reason") or "").split())
    if origin == "withdrawn":
        if status != "cancelled":
            return None
        return ApprovalText("not run", "Not run: the turn was stopped")
    if origin == "error":
        return ApprovalText("approval failed", "Approval failed; the call did not run")
    if origin == "timeout":
        return ApprovalText("not answered in time", "Not approved in time")
    if approval.get("allowed"):
        if origin == "session":
            return ApprovalText(
                "allowed by a session grant", "Allowed by a grant for this session"
            )
        if origin == "policy":
            return ApprovalText("auto-allowed", "Allowed: approvals are set to auto")
        if approval.get("remember"):
            return ApprovalText(
                "allowed for the session", "Allowed by you for this session"
            )
        return ApprovalText("allowed", "Allowed once by you")
    if origin == "policy":
        return ApprovalText(
            f"denied by policy: {reason}" if reason else "denied by policy",
            (
                f"Denied by the approval policy: {reason}"
                if reason
                else "Denied by the approval policy"
            ),
        )
    if origin == "user" and (
        status == "blocked" or error_category == "approval_denied"
    ):
        return ApprovalText(
            f"denied: {reason}" if reason else "denied",
            f"Denied by you: {reason}" if reason else "Denied by you",
        )
    return None


@dataclass(frozen=True)
class ActivityText:
    """What a running chat, task or run is doing: plain words and a kind icon.

    ``icon`` names the kind (``wait``, ``held``, ``paused``, ``read``,
    ``write``, ``run``, ``web``, ``memory``, ``think``, ``tool``); each surface
    draws it with its own glyph and hue, so a row never reads by colour alone.
    """

    text: str
    icon: str


# A command that runs a test suite reads as "Running the tests".
_TEST_COMMAND: Final = re.compile(
    r"(?:^|[\s;&|(])(?:pytest|py\.test|tox|nox|jest|vitest|"
    r"(?:npm|pnpm|yarn|bun)(?: run)? test|(?:cargo|go|swift|deno|mix) test|"
    r"python3? -m (?:pytest|unittest)|make (?:test|check))(?:$|[\s;&|)])"
)
_STATE_WORDS: Final[Mapping[str, ActivityText]] = {
    "waiting": ActivityText("Waiting for your approval", "wait"),
    "held": ActivityText("Waiting for the model", "held"),
    "queued": ActivityText("Waiting to start", "held"),
    "paused": ActivityText("Paused", "paused"),
    "reflecting": ActivityText("Checking its work", "think"),
    "remembering": ActivityText("Remembering", "memory"),
    "thinking": ActivityText("Thinking", "think"),
}
_BROWSER_READS: Final = frozenset(
    {"browser_read", "browser_find", "browser_back", "browser_tabs", "browser_wait"}
)


def _file_name(arguments: Mapping[str, Any]) -> str:
    path = str(arguments.get("path") or "").rstrip("/")
    return path.rsplit("/", 1)[-1]


def activity_text(
    tool: str = "",
    arguments: object = None,
    *,
    state: str = "",
    root: str | None = None,
) -> ActivityText:
    """The words for a *state* (``waiting``, ``held``, ``queued``, ``paused``,
    ``reflecting``, ``remembering``, ``thinking``), else for a running *tool*.

    A tool reads by what it does: ``Reading notes.md``, ``Looking through
    ~/talk``, ``Writing slides.md``, ``Running the tests``, ``Running a
    command``, ``Running Python``, ``Reading example.com``. Only a file's
    name, a folder relative to *root* (or with ``~``) and a page's host are
    shown, never a command line, code or text. A tool this table does not
    know reads ``Using a tool``; no state and no tool reads ``Thinking``.
    """
    if state in _STATE_WORDS:
        return _STATE_WORDS[state]
    args: Mapping[str, Any] = arguments if isinstance(arguments, Mapping) else {}
    name = tool.rsplit("__", 1)[-1] if tool.startswith("mcp__") else tool
    if name == "file_read":
        return ActivityText(f"Reading {_file_name(args) or 'a file'}", "read")
    if name == "file_write":
        return ActivityText(f"Writing {_file_name(args) or 'a file'}", "write")
    if name == "list_files":
        path = str(args.get("path") or "")
        label = display_path(path, root) if path else "."
        return ActivityText(
            (
                "Looking through the folder"
                if label == "."
                else f"Looking through {label}"
            ),
            "read",
        )
    if name == "shell_exec":
        command = " ".join(str(args.get("command") or "").split())
        tests = _TEST_COMMAND.search(command) is not None
        return ActivityText(
            "Running the tests" if tests else "Running a command", "run"
        )
    if name == "python_exec":
        return ActivityText("Running Python", "run")
    if name in ("recall", "search_memory"):
        return _STATE_WORDS["remembering"]
    if name == "browser_open":
        host = (urlsplit(str(args.get("url") or "")).hostname or "").removeprefix(
            "www."
        )
        return ActivityText(f"Reading {host}" if host else "Opening a page", "web")
    if name == "browser_follow":
        return ActivityText("Following a link", "web")
    if name == "browser_screenshot":
        return ActivityText("Looking at the page", "web")
    if name in _BROWSER_READS:
        return ActivityText("Reading the page", "web")
    if not name:
        return _STATE_WORDS["thinking"]
    return ActivityText("Using a tool", "tool")


def event_activity(
    newest_first: Iterable[Mapping[str, Any]], root: str | None = None
) -> ActivityText | None:
    """What a session's turn is doing, from its stored events, newest first.

    An approval that waits wins, then a call still running, then a
    reflection; any other event of a running turn reads ``Thinking``. A turn
    whose end is among them is doing nothing: ``None``.
    """
    finished: set[str] = set()
    resolved: set[str] = set()
    for event in newest_first:
        kind = event.get("type")
        data = event.get("data")
        data = data if isinstance(data, Mapping) else {}
        if kind in ("turn.finished", "run.finished"):
            return None
        if kind == "tool.finished":
            finished.add(str(data.get("call_id", "")))
        elif kind == "approval.resolved":
            resolved.add(str(data.get("request_id", "")))
        elif kind == "approval.requested":
            if str(data.get("request_id", "")) not in resolved:
                return _STATE_WORDS["waiting"]
        elif kind == "tool.started":
            if str(data.get("call_id", "")) not in finished:
                return activity_text(
                    str(data.get("name") or ""), data.get("arguments"), root=root
                )
        elif kind == "reflection.started":
            return _STATE_WORDS["reflecting"]
        elif kind in ("recall", "memory"):
            return _STATE_WORDS["remembering"]
        elif kind in (
            "reflection",
            "step.started",
            "assistant.message",
            "turn.started",
        ):
            return _STATE_WORDS["thinking"]
    return _STATE_WORDS["thinking"]


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


def display_version() -> str:
    """The version a person reads: ``0.1.0 (baab21e)`` on a git checkout.

    The package version, then the checkout's :func:`localmelo_version`; plain
    package version elsewhere. ``localmelo_version`` itself stays what JSON
    records and the gateway report.
    """
    described = localmelo_version()
    try:
        package = importlib.metadata.version("localmelo")
    except importlib.metadata.PackageNotFoundError:
        return described
    return package if described == package else f"{package} ({described})"
