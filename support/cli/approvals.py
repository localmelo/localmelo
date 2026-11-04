"""Approvals in the terminal: the panel that asks, and the session's policy.

:class:`TerminalApprover` holds the request the panel shows and resolves
it from the panel's keys: allow once (1), allow the tool for the session
(2), or deny with a reason (3). A high-impact call (#279) offers no session
choice: its options are allow once (1) and deny (2), and the panel starts
on deny. A read in a folder outside the workspace (#405) offers allow once,
allow the folder for this chat, always allow the folder (kept in the
settings) and deny; a write there offers allow once and deny, and starts on
deny. :class:`Approvals` is the session's policy:
in ``ask`` mode a :class:`RememberingPolicy` that asks through the panel
and keeps session grants, in ``auto`` mode an
:class:`~localmelo.melo.contracts.approvals.AutoPolicy`: a routine call
runs, a stopped one the person may allow asks on the same panel (#465), and
any other is blocked and explained (#339); with ``--yolo`` no policy at all.

The panel shows what will run escaped, never cleaned: approval is the only
control before a shell command or Python code runs, so every byte shows.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from localmelo.melo.checker.review.gist import step_gist
from localmelo.melo.contracts.approvals import (
    DESKTOP_ACT,
    WINDOW_READ,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    AskWatch,
    AutoPolicy,
    DenyAllPolicy,
    grant_offered,
    never_granted,
)
from localmelo.melo.contracts.review import because, review_title, url_site
from localmelo.support.cli import edits
from localmelo.support.cli.text import escape_arguments, wrap_cells
from localmelo.support.config import SavedFolders
from localmelo.support.sessions.approvals import (
    AskFunction,
    CardWatch,
    RememberingPolicy,
)
from localmelo.support.sessions.display import display_path, folder_name, home_label
from localmelo.support.sessions.grants import SessionGrants
from localmelo.support.words import gist_note, gist_text

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
    "auto": "works on its own; asks before a step you did not ask for, blocks the "
    "riskiest steps and explains them",
    "deny": "never runs commands or changes files",
    "yolo": "runs everything without asking (--yolo)",
}
# The quiet line a switch leaves in the transcript (the key and /approvals say the same).
MODE_SWITCHED: Final = {
    "ask": "Ask: melo asks before running commands or changing files.",
    "auto": "Auto: routine steps run; melo asks about the rest, the riskiest are blocked.",
}
# What a session grant covers, in the words of the panel that asked.
_GRANT_WORDS: Final = {
    "shell_exec": "shell commands",
    "python_exec": "Python code",
    "file_write": "file writes",
}
Choice = Literal["once", "session", "always", "deny"]


@dataclass
class PendingApproval:
    """What the panel shows; the approver holds one while a request waits."""

    request: ApprovalRequest
    offer_session: bool
    opened_at: float
    future: asyncio.Future[ApprovalDecision]
    # A folder outside the workspace that may be kept in the settings (#405).
    offer_always: bool = False
    # For file_write: what the path holds now, for the diff.
    target: edits.Target | None = None
    selected: int = 0
    reason: bool = False  # the reason field is open
    # The main chat's own request: what it allows for the chat counts in every chat.
    everywhere: bool = False


@dataclass(frozen=True)
class PanelText:
    """The parts of the panel for one request: title, header, body lines."""

    title: str
    header: str
    body: list[tuple[str, str]]  # style and text per row
    escaped: bool
    # Why melo asks, for a high-impact call: one sentence per entry.
    notes: tuple[str, ...] = ()
    # What the step will do in one plain sentence (#454), first on the panel;
    # the warning is set when the model's words do not match the code.
    says: str = ""
    warning: str = ""
    # The model's own words, labelled ("melo says: ..."), below the sentence
    # of a step the rules found risky.
    said: str = ""


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
    A high-impact call (#279) gets the review's title and a note that says
    why melo asks.
    """
    if request.outside is not None:
        return _outside_panel(request, target, width, minus, show_tier)
    text = _says(request, _tool_panel(request, target, width, minus, show_tier))
    review = request.review
    title = review_title(review, folder_name(request.workspace_root))
    if (
        review is None
        or not title
        or request.tool_name.startswith((WINDOW_READ, DESKTOP_ACT, "browser-point:"))
    ):
        # Another app's window, a press in one and a click melo cannot check are asked the
        # same way in Ask and in Auto: their own words, never a review's title.
        return text
    note, changed = escape_arguments(f"melo asks because this call {because(review)}.")
    return dataclasses.replace(
        text, title=title, notes=(note,), escaped=text.escaped or changed
    )


def _says(request: ApprovalRequest, text: PanelText) -> PanelText:
    """*text* with the sentence the panel starts with, escaped like the code below it.

    A tool whose question says as much itself (a file write, a tool's name, a
    site) has none.
    """
    gist = step_gist(request.tool_name, request.arguments, request.review)
    if gist is None:
        return text
    shown = gist.to_json()
    says, warning = gist_text(shown, "en")
    said = gist_note(shown, "en")
    if gist.plain:
        says = ""
    if not (says or warning or said):
        return text
    says, changed = escape_arguments(says)
    warning, warned = escape_arguments(warning)
    said, noted = escape_arguments(said)
    return dataclasses.replace(
        text,
        says=says,
        warning=warning,
        said=said,
        escaped=text.escaped or changed or warned or noted,
    )


def _tool_panel(
    request: ApprovalRequest,
    target: edits.Target | None,
    width: int,
    minus: str,
    show_tier: bool,
) -> PanelText:
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
    if tool.startswith("browser:") and arguments.get("via") == "web_fetch":
        site = shown(str(arguments.get("site") or tool[len("browser:") :]))
        said = (
            f"melo wants to read {site} as text: one request, no sign-ins, no cookies."
        )
        note = "The address it asks for is sent to that site."
        body = [("", row) for row in _wrapped([said], width)]
        if url := arguments.get("url"):
            body += [("", row) for row in _wrapped([shown(str(url))], width)]
        body += [("muted", row) for row in _wrapped([note], width)]
        return PanelText(
            "Read a website?",
            f"{site}{mark} · web_fetch · read only",
            body,
            escaped_any,
        )
    background = arguments.get("browser") == "chrome"
    window = (
        "in melo's browser, which runs in the background"
        if background
        else "in Safari's automation window"
    )
    if tool.startswith(
        ("browser-act:", "browser-type:", "browser-submit:", "browser-point:")
    ):
        sending = tool.startswith("browser-submit:")
        typing = tool.startswith("browser-type:")
        pointing = tool.startswith("browser-point:")
        site = shown(str(arguments.get("site") or tool.split(":", 1)[1]))
        goes = str(arguments.get("to") or "")
        if sending:
            said = f"melo wants to send a form from {site}."
            note = (
                "Whatever the form holds goes to the address below."
                if goes
                else "The page decides where this goes: melo cannot say."
            )
        elif pointing:
            on = shown(str(arguments.get("on") or "a point of the page"))
            how = shown(str(arguments.get("how") or "click"))
            said = f"melo wants to {how} {on} on {site} {window}."
            note = "melo cannot check this one itself, so it asks every time."
        elif typing:
            said = f"melo wants to type this text on {site} {window}:"
            note = (
                "It never types passwords, card numbers or one-time codes. Once the "
                "text is in the page, the page may send it on its own; melo asks "
                "before it presses send, but cannot stop every send the page makes."
            )
        else:
            said = f"melo wants to click and choose on {site} {window}, as you would."
            note = "Each text it types is asked about apart, every time."
        body = [("", row) for row in _wrapped([said], width)]
        if sending and goes:
            body += [("", row) for row in _wrapped([shown(goes)], width)]
        if sending and (rows := arguments.get("fields")):
            body += [
                ("muted", row)
                for row in _wrapped(
                    [f"{shown(str(n))}: {shown(str(v))}" for n, v in rows], width
                )
            ]
        if typing:
            typed = shown(str(arguments.get("text") or ""))
            body += [("", row) for row in _wrapped(typed.split("\n"), width)]
        body += [("muted", row) for row in _wrapped([note], width)]
        return PanelText(
            (
                "Send a form?"
                if sending
                else "Type on a website?" if typing else "Click on a website?"
            ),
            f"{site}{mark} · "
            + (
                "sends a form"
                if sending
                else "types text" if typing else "acts on the page"
            ),
            body,
            escaped_any,
        )
    if tool.startswith("desktop-act:"):
        # A button in another app's window that appeared after melo's own step
        # in the browser: asked every time.
        app = shown(str(arguments.get("app") or tool[len("desktop-act:") :]))
        on = shown(str(arguments.get("on") or "a button"))
        window = shown(str(arguments.get("window") or ""))
        said = f"melo wants to press {on} in {app}'s window" + (
            f" “{window}”." if window else "."
        )
        note = (
            "The window appeared after melo's own step in the browser. melo never "
            "types into other apps, and asks every time it presses."
        )
        body = [("", row) for row in _wrapped([said], width)]
        if arguments.get("delivery") == "foreground_retry":
            # A click at a point is asked once for both tries: the second one moves the
            # pointer and brings the app to the front.
            front = (
                "If a click in the background does not reach it, melo clicks once "
                f"more in the foreground: the pointer moves there and {app} comes to "
                "the front for a moment."
            )
            body += [("", row) for row in _wrapped([front], width)]
        body += [("muted", row) for row in _wrapped([note], width)]
        return PanelText(
            "Press a button in another app?",
            f"{app}{mark} · another app's window",
            body,
            escaped_any,
        )
    if tool.startswith(("handoff:", "browser-dialog:")):
        site = shown(str(arguments.get("site") or "a page"))
        if tool.startswith("handoff:"):
            # A page asked to open an address of another app's kind, which the
            # browser in the background does not open by itself.
            app = shown(str(arguments.get("app") or "an app"))
            said = f"A page on {site} wants to open {app}. melo would open this address with it:"
            note = (
                # The kept answer below says where it holds (this chat, every chat).
                f"{app} comes to the front. A kept answer lets pages open such addresses again."
            )
            body = [("", row) for row in _wrapped([said], width)]
            body += [
                ("", row)
                for row in _wrapped([shown(str(arguments.get("url") or ""))], width)
            ]
            body += [("muted", row) for row in _wrapped([note], width)]
            return PanelText(
                f"Open {app}?",
                f"{site}{mark} · a page asks for an app",
                body,
                escaped_any,
            )
        said = f"The page on {site} asks this question. Answer OK?"
        note = "Cancel is melo's answer when you say no. Asked every time."
        body = [("", row) for row in _wrapped([said], width)]
        body += [
            ("", row)
            for row in _wrapped([shown(str(arguments.get("text") or ""))], width)
        ]
        body += [("muted", row) for row in _wrapped([note], width)]
        return PanelText(
            "Answer OK on a website?",
            f"{site}{mark} · the page asks",
            body,
            escaped_any,
        )
    if tool.startswith("window-read:"):
        # A window another app put up right after melo's own step in the browser: nothing
        # about it goes to the model unless the person allows it here. The title is for the
        # person only.
        app = shown(str(arguments.get("app") or tool[len("window-read:") :]))
        window = shown(str(arguments.get("window") or ""))
        said = f"{app} opened a window right after melo's click: let melo see it?"
        body = [("", row) for row in _wrapped([said], width)]
        if window:
            body += [("", row) for row in _wrapped([f"The window: “{window}”."], width)]
        note = (
            "Only you see its name. If you allow it, melo reads the window's text and "
            "buttons and can press one only with your say. melo never types into other apps."
        )
        body += [("muted", row) for row in _wrapped([note], width)]
        return PanelText(
            "Let melo see another app's window?",
            f"{app}{mark} · another app's window",
            body,
            escaped_any,
        )
    if tool.startswith("browser:"):
        site = shown(str(arguments.get("site") or tool[len("browser:") :]))
        if background:
            said = f"melo wants to open {site} in Chrome and read it."
            note = (
                "Chrome runs in the background with melo's own profile: no window "
                "comes to the front, and your own sign-ins are not used."
            )
        else:
            said = f"melo wants to open {site} in Safari and read it."
            note = "Safari opens it in its own automation window; your sign-ins are not used."
        body = [("", row) for row in _wrapped([said], width)]
        body += [("muted", row) for row in _wrapped([note], width)]
        return PanelText(
            "Open a website?", f"{site}{mark} · read only", body, escaped_any
        )
    text = json.dumps(dict(arguments), indent=2, ensure_ascii=False, default=str)
    body = [("", row) for row in _wrapped(shown(text).split("\n"), width)]
    return PanelText(f"Call {shown(tool)}?", f"{shown(tool)}{mark}", body, escaped_any)


def _outside_panel(
    request: ApprovalRequest,
    target: edits.Target | None,
    width: int,
    minus: str,
    show_tier: bool,
) -> PanelText:
    """The panel of a call on a path outside the workspace (#405).

    The title names the real folder; the header says outside which
    workspace, and whether the call reads or writes. A read's body says
    what melo opens; a write's is the file write's body.
    """
    outside = request.outside
    assert outside is not None
    root = request.workspace_root
    workspace = home_label(root) if root else "the workspace"
    folder, escaped = escape_arguments(home_label(outside.folder))
    path, changed = escape_arguments(home_label(outside.path))
    escaped = escaped or changed
    tool = request.tool_name
    mark = "" if request.tier is None or not show_tier else f" · {request.tier}"
    if outside.access == "write":
        text = _tool_panel(request, target, width, minus, show_tier)
        note, changed = escape_arguments(
            f"{path} is outside {workspace}. A write there asks every time."
        )
        return dataclasses.replace(
            text,
            title=f"Write outside the workspace, in {folder}?",
            notes=(note,),
            escaped=text.escaped or escaped or changed,
        )
    if tool == "list_files":
        title = f"Let melo look in {folder}?"
        said = f"melo wants to list {path}."
    else:
        title = f"Let melo read in {folder}?"
        said = f"melo wants to read {path}."
    header = f"{tool}{mark} · outside {workspace} · read only"
    body = [("", row) for row in _wrapped([said], width)]
    if not outside.grantable:
        rule = "melo never keeps your whole home folder open: this asks every time."
        body += [("muted", row) for row in _wrapped([rule], width)]
    return PanelText(title, header, body, escaped)


def folder_options(
    request: ApprovalRequest, *, everywhere: bool = False
) -> tuple[str, str]:
    """The labels of "this folder for this chat" and "always this folder" (#405).

    In the main chat (*everywhere*) the first says it counts in every chat.
    """
    outside = request.outside
    folder = home_label(outside.folder) if outside is not None else "this folder"
    shown, _ = escape_arguments(folder)
    return (
        f"Yes, and allow {shown} {'in every chat' if everywhere else 'for this chat'}",
        f"Yes, and always allow {shown} (kept in Settings)",
    )


# Under the options of the main chat's own question that may be kept: its
# "allow" counts in every chat.
EVERYWHERE_RULE: Final = (
    "In the main chat this counts in every chat, the tasks you start and the "
    "terminal, until you take it back with /grants."
)


# Under the options of a read outside the workspace that may be granted.
FOLDER_RULE: Final = (
    "Always keeps the folder in Settings, where you can remove it. "
    "Writing outside the workspace still asks every time."
)


def session_option(request: ApprovalRequest, *, everywhere: bool = False) -> str:
    """The label of the answer that allows *request* for the rest of the session.

    In the main chat (*everywhere*) it says the answer counts in every chat.
    """
    if request.outside is not None:
        return folder_options(request, everywhere=everywhere)[0]
    rest = "in every chat" if everywhere else "for the rest of this chat"
    if request.tool_name.startswith("browser:"):
        site = str(request.arguments.get("site") or "this site")
        return f"Yes, and allow {site} {rest}"
    if request.tool_name.startswith("browser-act:"):
        site = str(request.arguments.get("site") or "this site")
        return f"Yes, and allow clicking and choosing on {site} {rest}"
    if request.tool_name.startswith("handoff:"):
        app = str(request.arguments.get("app") or "that app")
        return f"Yes, and let pages open {app} {rest}"
    if request.tool_name.startswith("window-read:"):
        app, _ = escape_arguments(str(request.arguments.get("app") or "this app"))
        return f"Yes, and allow {app} {'everywhere' if everywhere else 'in this chat'}"
    where = "in every chat" if everywhere else "in this session"
    return f"Yes, and allow every {request.tool_name} call {where}"


def session_warning(tool: str, workspace: str) -> list[str]:
    """The lines under option 2 for a tool whose grant reaches past the workspace."""
    if tool == "shell_exec":
        what = "shell commands"
    elif tool == "python_exec":
        what = "Python code"
    else:
        return []
    return [
        f'"Allow every" lets routine {what} run without asking for the rest of '
        "this session.",
        f"Installs, downloads and changes outside {workspace} still ask.",
    ]


# Under the options of a high-impact call, which has no session choice.
NO_GRANT_RULE: Final = (
    "Session grants never cover installs, downloads or changes outside the folder."
)


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
        self,
        request: ApprovalRequest,
        *,
        offer_session: bool | None = None,
        everywhere: bool = False,
    ) -> ApprovalDecision:
        """Show the panel for *request* and wait for the answer.

        The session choice is offered unless *offer_session* says no or the
        request is a high-impact call, which a grant never covers; that
        panel starts on No. *everywhere* is the main chat's own request,
        whose session choice says it counts in every chat. A cancellation (a
        stop, a shutdown) withdraws the panel and propagates: a withdrawn
        prompt is not a denial.
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
        # A grant never covers a high-impact call, whatever the caller says;
        # a folder outside the workspace is offered only when it may be kept.
        outside = request.outside
        offered = grant_offered(request) if outside is None else outside.grantable
        # An answer to typing or to a form's send is never kept for the chat: no session choice.
        grantable = offered and not never_granted(request.tool_name)
        self.pending = PendingApproval(
            request=request,
            offer_session=grantable and offer_session is not False,
            opened_at=self._clock(),
            future=future,
            target=target,
            selected=2 if outside is not None and outside.access == "write" else 0,
            offer_always=outside is not None and offered,
            everywhere=everywhere,
        )
        if outside is None and not offered:
            self.pending.selected = 2
        self.on_change()
        try:
            return await future
        finally:
            self.pending = None
            self.on_change()

    def ends(self, request_id: str, decision: ApprovalDecision) -> bool:
        """End the panel of *request_id* with *decision*, for a card nobody answered (#465).

        Whether that panel was up; one already answered is gone and left as it is.
        """
        pending = self.pending
        if pending is None or pending.request.request_id != request_id:
            return False
        if not pending.future.done():
            pending.future.set_result(decision)
        return True

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
        if choice == "always" and not pending.offer_always:
            return
        if choice == "once":
            decision = ApprovalDecision(True)
        elif choice == "session":
            decision = ApprovalDecision(True, remember=True)
        elif choice == "always":
            decision = ApprovalDecision(True, remember=True, always=True)
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
        # The policy that asks about a folder outside the workspace (#405), in
        # Ask and in Auto alike; None asks nobody about one.
        self.folders: RememberingPolicy | None = None
        self._snapshots: dict[str, Snapshot] = {}

    @property
    def asks_folders(self) -> bool:
        return self.folders is not None and self.folders.asks_folders

    async def decide_folder(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask :attr:`folders` about a path outside the workspace, a write snapshotted."""
        folders = self.folders
        if folders is None:
            raise ValueError("no policy asks about folders")
        return await self._snapshotted(request, folders.decide_folder)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        inner = self.inner
        if inner is None:
            return await self._snapshotted(request, None)
        return await self._snapshotted(request, inner.decide)

    async def _snapshotted(
        self,
        request: ApprovalRequest,
        decide: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None,
    ) -> ApprovalDecision:
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
        decision = ApprovalDecision(True) if decide is None else await decide(request)
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


def next_mode(mode: Mode) -> Literal["ask", "auto"] | None:
    """What Shift+Tab switches *mode* to: Ask to Auto, Auto and Deny to Ask.

    ``None`` under ``--yolo``: the key never leaves it, and never reaches Deny.
    """
    if mode == "yolo":
        return None
    return "auto" if mode == "ask" else "ask"


# A site question's grant in words (#465): its prefix, and what it lets melo do.
_SITE_GRANT_WORDS: Final = (
    ("browser-act:", "clicking and choosing on {site}"),
    ("browser:", "opening {site}"),
)


def grant_site(name: str) -> str:
    """The host a site question's grant *name* is about, or ``""`` for any other grant."""
    for prefix, _ in _SITE_GRANT_WORDS:
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            return url_site(rest) or rest
    return ""


def grant_words(name: str) -> str:
    """A grant in words: ``shell commands``, ``clicking and choosing on example.com``."""
    if name in _GRANT_WORDS:
        return _GRANT_WORDS[name]
    if name.startswith("handoff:"):
        return f"opening {name[len('handoff:') :]}: addresses in their app"
    for prefix, words in _SITE_GRANT_WORDS:
        if name.startswith(prefix):
            return words.format(site=grant_site(name))
    return name


def grant_listed(name: str) -> str:
    """A grant as ``/approvals`` lists it: a tool by its name, a site's by its site.

    ``shell_exec``; ``example.com (clicking and choosing)``: the name or the
    site is what ``/approvals forget`` takes.
    """
    site = grant_site(name)
    if not site:
        return name
    what = "clicking and choosing" if name.startswith("browser-act:") else "opening"
    return f"{site} ({what})"


def approval_words(mode: Mode, grants: Iterable[str]) -> str:
    """What the approval *mode* does, and what the session allowed already.

    In Ask every grant; in Auto the sites allowed for the chat (#465), the
    only grants it uses.
    """
    names = sorted(grants)
    if mode == "auto":
        names = [name for name in names if grant_site(name)]
    allowed = ", ".join(grant_words(name) for name in names)
    if mode in ("ask", "auto") and allowed:
        return f"{_MODE_WORDS[mode]} · allowed this session: {allowed}"
    return _MODE_WORDS[mode]


def at_terminal() -> bool:
    """Whether a person can answer the panel: stdin and stdout are a terminal (#465)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def terminal_watch(
    approver: TerminalApprover,
    *,
    auto_asks: bool = True,
    present: Callable[[], bool] = at_terminal,
) -> CardWatch:
    """Who sees the panel Auto shows in the terminal, and how long it waits (#465).

    The person at the terminal, while stdin and stdout are one (*present*):
    with stdin piped or no terminal, or with ``[approvals] auto_asks`` off,
    Auto blocks as before and never waits. A panel nobody answers ends
    through *approver*.
    """

    async def viewers() -> int:
        return int(auto_asks and present())

    return CardWatch(viewers, approver.ends)


class Approvals:
    """The session's approval mode and policy, which keeps the grants.

    With *grants* (the terminal's own chat), the grants live in the table
    of the terminal's sessions file and hold in the session that gave them,
    beside what the person allowed in the main chat; without, in this
    object only.
    """

    def __init__(
        self,
        ask: AskFunction,
        mode: Mode,
        *,
        snapshots: EditSnapshots | None = None,
        watch: AskWatch | None = None,
        grants: SessionGrants | None = None,
    ) -> None:
        self.mode: Mode = mode
        self._ask = ask
        self.grants = grants
        self.snapshots = snapshots if snapshots is not None else EditSnapshots()
        self._policy = self._asking()
        # Auto mode's policy, kept across turns for its allowances; a step it
        # stops that the person may allow asks on the panel while *watch*
        # says someone is at the terminal (#465).
        self._watch = watch
        self._auto = AutoPolicy(asking=self._policy, watch=watch)

    def _asking(self) -> RememberingPolicy:
        # It also asks about a folder outside the workspace (#405).
        return RememberingPolicy(
            self._ask, folders=True, saved=SavedFolders(), grants=self.grants
        )

    def policy(self) -> ApprovalPolicy | None:
        """The policy for a turn: None with ``--yolo``, so nothing asks.

        In Ask and in Auto the panel asks about a folder outside the
        workspace; deny mode and ``--yolo`` refuse such a path.
        """
        if self.mode == "yolo":
            return None
        if self.mode == "deny":
            return DenyAllPolicy(DENY_REASON)
        auto = self.mode == "auto"
        self.snapshots.inner = self._auto if auto else self._policy
        self.snapshots.folders = self._policy
        return self.snapshots

    def set_mode(self, mode: Literal["ask", "auto", "deny"]) -> None:
        """Switch the mode; a turn under way keeps the policy it started with."""
        self.mode = mode

    def forget(self, tool_name: str | None = None) -> None:
        """Take back one grant, or every grant.

        *tool_name* is a grant's name, or a site: every grant of that site
        goes (#465).
        """
        if tool_name is None or tool_name in self._policy.allowed_tools:
            self._policy.forget(tool_name)
            return
        site = tool_name.lower().removeprefix("www.")
        for name in self._policy.allowed_tools:
            if grant_site(name).removeprefix("www.") == site:
                self._policy.forget(name)

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self._policy.allowed_tools

    @property
    def allowed_folders(self) -> tuple[str, ...]:
        return self._policy.allowed_folders

    def reset(self) -> None:
        """Start the next session without this one's grants.

        Kept grants stay with the session that gave them (a resumed session
        has its own again); grants kept in this object only are forgotten.
        """
        self._policy = self._asking()
        self._auto = AutoPolicy(asking=self._policy, watch=self._watch)
        self.snapshots.clear()
