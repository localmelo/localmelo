"""Approvals in the terminal: the panel that asks, and the session's policy.

:class:`TerminalApprover` holds the request the panel shows and resolves
it from the panel's keys: allow once (1), allow the tool for the session
(2), or deny with a reason (3). :class:`Approvals` is the session's policy:
in ``ask`` mode a :class:`RememberingPolicy` that asks through the panel
and keeps session grants; with ``--yolo`` no policy at all.

The panel shows what will run escaped, never cleaned: approval is the only
control before a shell command or Python code runs, so every byte shows.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from localmelo.melo.contracts.approvals import (
    AllowAllPolicy,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    DenyAllPolicy,
)
from localmelo.support.cli import edits
from localmelo.support.cli.text import escape_arguments, wrap_cells
from localmelo.support.sessions.approvals import AskFunction, RememberingPolicy
from localmelo.support.sessions.display import display_path, home_label

# Keys other than Esc and Ctrl+C do nothing this long after the panel opens,
# so a draft being typed cannot answer it.
APPROVAL_KEY_GUARD_S: Final = 0.4
# The reason the model receives keeps this many characters.
REASON_MAX_CHARS: Final = 500
# Lines of the panel's body before it is cut.
BODY_LINES: Final = 8
# The cells a line number takes in front of code: ``"   1 "``.
NUMBER_CELLS: Final = 5

ESCAPED_WARNING: Final = "The arguments contain control characters, shown escaped."

Mode = Literal["ask", "auto", "deny", "yolo"]
# The reason a denial in deny mode gives the model.
DENY_REASON: Final = "approvals are set to deny"
_MODE_WORDS: Final = {
    "ask": "asks before running commands or changing files",
    "auto": "runs commands and changes files without asking",
    "deny": "never runs commands or changes files",
    "yolo": "runs everything without asking (--yolo)",
}
# What a session grant covers, in the words of the panel that asked.
_GRANT_WORDS: Final = {
    "shell_exec": "shell commands",
    "python_exec": "Python code",
    "file_write": "file writes",
}
Choice = Literal["once", "session", "deny"]


@dataclass
class PendingApproval:
    """What the panel shows; the approver holds one while a request waits."""

    request: ApprovalRequest
    offer_session: bool
    opened_at: float
    future: asyncio.Future[ApprovalDecision]
    # For file_write: what the path holds now, for the diff.
    target: edits.Target | None = None
    selected: int = 0
    reason: bool = False  # the reason field is open


@dataclass(frozen=True)
class PanelText:
    """The parts of the panel for one request: title, header, body lines."""

    title: str
    header: str
    body: list[tuple[str, str]]  # style and text per row
    escaped: bool


def _command_text(command: object) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command if command is not None else "")


def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _numbered(lines: list[str], width: int) -> list[str]:
    """*lines* numbered, each wrapped to *width* with its rest indented under it."""
    room = width - NUMBER_CELLS
    rows: list[str] = []
    for number, line in enumerate(lines, 1):
        first, *rest = wrap_cells(line, room)
        rows.append(f"{number:4} {first}")
        rows += [" " * NUMBER_CELLS + part for part in rest]
    return rows


def _wrapped(lines: list[str], width: int) -> list[str]:
    return [row for line in lines for row in wrap_cells(line, width)]


def _diff_body(
    target: edits.Target,
    content: str,
    width: int,
    shown: Callable[[str], str],
    minus: str,
) -> tuple[list[tuple[str, str]], int, int]:
    """The diff's rows and counts: old line numbers on removals, new otherwise."""
    rows, added, removed = edits.diff(target.text or "", content)
    body: list[tuple[str, str]] = []
    for row in rows:
        if row.kind == "…":
            body.append(("muted", "      " + "…"))
            continue
        number = row.old if row.kind == "-" else row.new
        sign = {"+": "+", "-": minus, " ": " "}[row.kind]
        style = {"+": "ok", "-": "error", " ": "muted"}[row.kind]
        text = shown(row.text)
        for index, part in enumerate(wrap_cells(f"{number:4} {sign} {text}", width)):
            body.append((style, part if index == 0 else "      " + part))
    return body, added, removed


def panel_text(
    request: ApprovalRequest,
    *,
    target: edits.Target | None = None,
    width: int,
    minus: str = "−",
    show_tier: bool = True,
) -> PanelText:
    """The panel's title, header and whole body for *request*, escaped.

    *target* is what a ``file_write`` path holds now; *width* is the width
    the body is wrapped to; *minus* is the UI's minus sign, for a diff;
    *show_tier* puts the tool's permission tier in the header (developer mode).
    """
    arguments: Mapping[str, Any] = request.arguments
    root = request.workspace_root
    workspace = home_label(root) if root else "no workspace"
    escaped_any = False

    def shown(text: str) -> str:
        nonlocal escaped_any
        out, changed = escape_arguments(text)
        escaped_any = escaped_any or changed
        return out

    tool = request.tool_name
    # A caller that built the request itself can leave the tier out; an
    # agent always fills it, and the header then shows one more word.
    mark = "" if request.tier is None or not show_tier else f" · {shown(request.tier)}"
    if tool == "shell_exec":
        command = shown(_command_text(arguments.get("command")))
        body = [("", row) for row in _wrapped(command.split("\n"), width)]
        return PanelText(
            "Run a shell command?",
            f"shell_exec{mark} · in {workspace}",
            body,
            escaped_any,
        )
    if tool == "python_exec":
        code = shown(str(arguments.get("code") or ""))
        return PanelText(
            "Run Python code?",
            f"python_exec{mark} · in {workspace}",
            [("", row) for row in _numbered(_lines(code), width)],
            escaped_any,
        )
    if tool == "file_write":
        path = str(arguments.get("path") or "")
        content = str(arguments.get("content") or "")
        where = shown(display_path(path, root))
        if target is None or not target.exists:
            lines = _lines(shown(content))
            count = len(lines)
            header = f"file_write  {where}{mark} · new file, {count} line" + (
                "" if count == 1 else "s"
            )
            body = [("", row) for row in _numbered(lines, width)]
            return PanelText("Write a new file?", header, body, escaped_any)
        if target.diffable:
            body, added, removed = _diff_body(target, content, width, shown, minus)
            header = f"file_write  {where}{mark} · +{added} {minus}{removed} lines"
            return PanelText("Change a file?", header, body, escaped_any)
        header = f"file_write  {where}{mark} · {shown(target.problem)}; no diff shown"
        lines = _lines(shown(content))
        body = [("", row) for row in _numbered(lines, width)]
        return PanelText("Change a file?", header, body, escaped_any)
    if tool.startswith("browser:"):
        site = shown(str(arguments.get("site") or tool[len("browser:") :]))
        said = f"melo wants to open {site} in Safari and read it."
        note = (
            "Safari opens it in its own automation window; your sign-ins are not used."
        )
        body = [("", row) for row in _wrapped([said], width)]
        body += [("muted", row) for row in _wrapped([note], width)]
        return PanelText(
            "Open a website?", f"{site}{mark} · read only", body, escaped_any
        )
    text = json.dumps(dict(arguments), indent=2, ensure_ascii=False, default=str)
    body = [("", row) for row in _wrapped(shown(text).split("\n"), width)]
    return PanelText(f"Call {shown(tool)}?", f"{shown(tool)}{mark}", body, escaped_any)


def session_option(request: ApprovalRequest) -> str:
    """The label of the answer that allows *request* for the rest of the session."""
    if request.tool_name.startswith("browser:"):
        site = str(request.arguments.get("site") or "this site")
        return f"Yes, and allow {site} for the rest of this chat"
    return f"Yes, and allow every {request.tool_name} call in this session"


def session_warning(tool: str, workspace: str) -> list[str]:
    """The lines under option 2 for a tool whose grant reaches past the workspace."""
    if tool == "shell_exec":
        what, who = "any shell command", "Commands"
    elif tool == "python_exec":
        what, who = "any Python code", "Python"
    else:
        return []
    return [
        f'"Allow every" lets {what} run without asking for the rest of this session.',
        f"{who} can reach files outside {workspace}.",
    ]


class TerminalApprover:
    """Asks the person at the terminal about one request at a time."""

    def __init__(
        self,
        on_change: Callable[[], None] = lambda: None,
        *,
        snapshots: EditSnapshots | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.on_change = on_change
        self.snapshots = snapshots if snapshots is not None else EditSnapshots()
        self._clock = clock
        self.pending: PendingApproval | None = None

    async def ask(
        self, request: ApprovalRequest, *, offer_session: bool = True
    ) -> ApprovalDecision:
        """Show the panel for *request* and wait for the answer.

        A cancellation (a stop, a shutdown) withdraws the panel and
        propagates: a withdrawn prompt is not a denial.
        """
        target: edits.Target | None = None
        if request.tool_name == "file_write":
            target = self.snapshots.target_for(request.request_id)
            if target is None:
                path = str(request.arguments.get("path") or "")
                target = await asyncio.to_thread(edits.read_target, path)
        future: asyncio.Future[ApprovalDecision] = (
            asyncio.get_running_loop().create_future()
        )
        self.pending = PendingApproval(
            request=request,
            offer_session=offer_session,
            opened_at=self._clock(),
            future=future,
            target=target,
        )
        self.on_change()
        try:
            return await future
        finally:
            self.pending = None
            self.on_change()

    def guarded(self) -> bool:
        """Whether the panel opened too recently for a key to answer it."""
        pending = self.pending
        return pending is not None and (
            self._clock() - pending.opened_at < APPROVAL_KEY_GUARD_S
        )

    def answer(self, choice: Choice, reason: str = "") -> None:
        """Resolve the pending request with *choice*."""
        pending = self.pending
        if pending is None or pending.future.done():
            return
        if choice == "session" and not pending.offer_session:
            return
        if choice == "once":
            decision = ApprovalDecision(True)
        elif choice == "session":
            decision = ApprovalDecision(True, remember=True)
        else:
            decision = ApprovalDecision(False, reason[:REASON_MAX_CHARS])
        pending.future.set_result(decision)


def _stat_key(path: str) -> tuple[int, int, int, int] | None:
    """A file's identity and size/time, or None when it cannot be stat'ed."""
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True)
class Snapshot:
    """What a ``file_write`` call is about to change, by its request id."""

    request_id: str
    target: edits.Target
    new: str
    added: int
    removed: int
    # The target's identity and size/time when it was read (F6).
    stat: tuple[int, int, int, int] | None = None


class EditSnapshots:
    """Reads a ``file_write`` target before the call is answered and keeps it.

    Wraps another policy; the renderer then shows the change by the
    request's id when the call ends. Every marked call is snapshotted, a
    granted or auto-allowed one and a denied one too (only an allowed
    write has a change to show).
    """

    def __init__(self, inner: ApprovalPolicy | None = None) -> None:
        self.inner = inner
        self._snapshots: dict[str, Snapshot] = {}

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        path = ""
        stat: tuple[int, int, int, int] | None = None
        if request.tool_name == "file_write" and isinstance(request.arguments, Mapping):
            path = str(request.arguments.get("path") or "")
            target = await asyncio.to_thread(edits.read_target, path)
            stat = await asyncio.to_thread(_stat_key, path)
            new = str(request.arguments.get("content") or "")
            if target.diffable:
                _, added, removed = edits.diff(target.text or "", new)
            else:
                added = removed = 0
            self._snapshots[request.request_id] = Snapshot(
                request_id=request.request_id,
                target=target,
                new=new,
                added=added,
                removed=removed,
                stat=stat,
            )
        decision = (
            ApprovalDecision(True)
            if self.inner is None
            else await self.inner.decide(request)
        )
        # Q1 a: the person approved what they saw. If the file changed while
        # the panel was open, deny rather than write over the newer file.
        # A file that did not exist then (stat None) and exists now counts too.
        if decision.allow and path and await asyncio.to_thread(_stat_key, path) != stat:
            return ApprovalDecision(
                False,
                "The file changed while this was waiting; look again.",
                origin="policy",
            )
        return decision

    def target_for(self, request_id: str) -> edits.Target | None:
        """The target read for *request_id*, kept for the panel."""
        snapshot = self._snapshots.get(request_id)
        return snapshot.target if snapshot is not None else None

    def take(self, request_id: str) -> Snapshot | None:
        """The snapshot of *request_id*, removed: a finished call shows it once."""
        return self._snapshots.pop(request_id, None)

    def clear(self) -> None:
        """Drop every snapshot, as ``/new`` does."""
        self._snapshots.clear()


def approval_words(mode: Mode, grants: Iterable[str]) -> str:
    """What the approval *mode* does, and in ask mode what the session allowed already."""
    allowed = ", ".join(_GRANT_WORDS.get(tool, tool) for tool in sorted(grants))
    if mode == "ask" and allowed:
        return f"{_MODE_WORDS[mode]} · allowed this session: {allowed}"
    return _MODE_WORDS[mode]


class _AutoPolicy(AllowAllPolicy):
    """Auto mode: allows every call, but a website still asks (once per site per chat)."""

    def __init__(self, asking: ApprovalPolicy) -> None:
        self._asking = asking

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.tool_name.startswith("browser:"):
            return await self._asking.decide(request)
        return ApprovalDecision(True, origin="policy")


class Approvals:
    """The session's approval mode and policy, which keeps the grants."""

    def __init__(
        self,
        ask: AskFunction,
        mode: Mode,
        *,
        snapshots: EditSnapshots | None = None,
    ) -> None:
        self.mode: Mode = mode
        self._ask = ask
        self.snapshots = snapshots if snapshots is not None else EditSnapshots()
        self._policy = RememberingPolicy(ask)

    def policy(self) -> ApprovalPolicy | None:
        """The policy for a turn: None with ``--yolo``, so nothing asks."""
        if self.mode == "yolo":
            return None
        if self.mode == "deny":
            return DenyAllPolicy(DENY_REASON)
        self.snapshots.inner = (
            _AutoPolicy(self._policy) if self.mode == "auto" else self._policy
        )
        return self.snapshots

    def set_mode(self, mode: Literal["ask", "auto", "deny"]) -> None:
        """Switch the mode; a turn under way keeps the policy it started with."""
        self.mode = mode

    def forget(self, tool_name: str | None = None) -> None:
        """Take back one tool's grant, or every grant."""
        self._policy.forget(tool_name)

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self._policy.allowed_tools

    def reset(self) -> None:
        """Forget every grant: a new session starts with none."""
        self._policy = RememberingPolicy(self._ask)
        self.snapshots.clear()
