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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
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

ESCAPED_WARNING: Final = "The arguments contain control characters, shown escaped."

Mode = Literal["ask", "yolo"]
Choice = Literal["once", "session", "deny"]


@dataclass
class PendingApproval:
    """What the panel shows; the approver holds one while a request waits."""

    request: ApprovalRequest
    offer_session: bool
    opened_at: float
    future: asyncio.Future[ApprovalDecision]
    exists: bool | None = None  # for file_write: does the file exist
    selected: int = 0
    reason: bool = False  # the reason field is open


@dataclass(frozen=True)
class PanelText:
    """The parts of the panel for one request: title, header, body lines."""

    title: str
    header: str
    body: list[str]
    escaped: bool


def _command_text(command: object) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command if command is not None else "")


def _numbered(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [f"{number:4} {line}" for number, line in enumerate(lines, 1)]


def panel_text(
    request: ApprovalRequest, *, exists: bool | None, width: int
) -> PanelText:
    """The panel's title, header and whole body for *request*, escaped.

    *exists* says whether a ``file_write`` target exists; *width* is the
    width the body is wrapped to.
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
    if tool == "shell_exec":
        command = shown(_command_text(arguments.get("command")))
        body = [row for line in command.split("\n") for row in wrap_cells(line, width)]
        return PanelText(
            "Run a shell command?", f"shell_exec · in {workspace}", body, escaped_any
        )
    if tool == "python_exec":
        code = shown(str(arguments.get("code") or ""))
        return PanelText(
            "Run Python code?",
            f"python_exec · in {workspace}",
            _numbered(code),
            escaped_any,
        )
    if tool == "file_write":
        path = str(arguments.get("path") or "")
        content = str(arguments.get("content") or "")
        lines = _numbered(shown(content))
        where = shown(display_path(path, root))
        if exists:
            header = f"file_write  {where} · overwrites the file; no diff shown"
            return PanelText("Change a file?", header, lines, escaped_any)
        count = len(lines)
        header = f"file_write  {where} · new file, {count} line" + (
            "" if count == 1 else "s"
        )
        return PanelText("Write a new file?", header, lines, escaped_any)
    text = json.dumps(dict(arguments), indent=2, ensure_ascii=False, default=str)
    return PanelText(
        f"Call {shown(tool)}?", shown(tool), shown(text).split("\n"), escaped_any
    )


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.on_change = on_change
        self._clock = clock
        self.pending: PendingApproval | None = None

    async def ask(
        self, request: ApprovalRequest, *, offer_session: bool = True
    ) -> ApprovalDecision:
        """Show the panel for *request* and wait for the answer.

        A cancellation (a stop, a shutdown) withdraws the panel and
        propagates: a withdrawn prompt is not a denial.
        """
        exists: bool | None = None
        if request.tool_name == "file_write":
            path = str(request.arguments.get("path") or "")
            exists = await asyncio.to_thread(os.path.lexists, path)
        future: asyncio.Future[ApprovalDecision] = (
            asyncio.get_running_loop().create_future()
        )
        self.pending = PendingApproval(
            request=request,
            offer_session=offer_session,
            opened_at=self._clock(),
            future=future,
            exists=exists,
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


class Approvals:
    """The session's approval mode and policy, which keeps the grants."""

    def __init__(self, ask: AskFunction, mode: Mode) -> None:
        self.mode: Mode = mode
        self._ask = ask
        self._policy = RememberingPolicy(ask)

    def policy(self) -> ApprovalPolicy | None:
        """The policy for a turn: None with ``--yolo``, so nothing asks."""
        return None if self.mode == "yolo" else self._policy

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self._policy.allowed_tools

    def reset(self) -> None:
        """Forget every grant: a new session starts with none."""
        self._policy = RememberingPolicy(self._ask)
