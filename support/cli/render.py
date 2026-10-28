"""A turn's events, as blocks in scrollback and as the live area's rows.

:class:`TurnRenderer` gets each stored event of the turn (the runner's
``on_event``), keeps :class:`LiveState` for the live area and prints what
has finished: the echo of the user's message, thinking, intermediate text,
one block per tool call, notices and reflections, the answer as Markdown,
the status line of a run that did not complete, with its hint, and the
footer.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from rich.console import (
    Console,
    ConsoleOptions,
    Group,
    RenderableType,
    RenderResult,
)
from rich.markdown import Heading, Markdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from localmelo.melo.schema import RunResult, split_thinking
from localmelo.support.cli.output import Out
from localmelo.support.cli.text import (
    cells,
    cut_cells,
    escape_arguments,
    expand_tabs,
    wrap_words,
)
from localmelo.support.cli.theme import Symbols
from localmelo.support.cli.views import BackendInfo
from localmelo.support.progress_line import format_line, format_past, tool_elapsed
from localmelo.support.sessions.agents import SESSION_TOOL_TIMEOUT_S
from localmelo.support.sessions.display import (
    clean_output,
    format_bytes,
    format_duration,
    format_tokens,
    status_line,
)
from localmelo.support.sessions.store import StoredEvent

# Output lines a tool block shows under its header (20 with /verbose).
BLOCK_LINES: Final = 3
VERBOSE_BLOCK_LINES: Final = 20
# Lines of the thinking excerpt (40 with /verbose).
THINKING_LINES: Final = 3
VERBOSE_THINKING_LINES: Final = 40
# Lines of a call's arguments under its header, with /verbose.
ARGUMENT_LINES: Final = 20
# Lines /expand prints, then the rest's count.
EXPAND_LINES: Final = 200
# A block header's last part that carries free text keeps this much room.
FREE_TEXT_CELLS: Final = 24
_FREE_TEXT_TAGS: Final = ("denied: ", "denied by policy: ", "blocked: ")
# Lines of the user's message echoed.
ECHO_LINES: Final = 6
# Tools whose running line shows the session's time limit.
LIMITED_TOOLS: Final = frozenset({"shell_exec", "python_exec"})

LOG_PATH: Final = "~/.cache/localmelo/logs/cli.log"

_HTTP_STATUS: Final = re.compile(r"failed with HTTP (\d{3})")
# Markdown that starts a block: a fence, a heading, a list, a quote, a table.
_BLOCK_START: Final = re.compile(r"\s*(```|~~~|#|[-*+] |\d+[.)] |>|\|)")
_FAILED_WITH: Final = re.compile(r"failed: (\w+)")

# The severity of each status that ended a run, for its status line.
_SEVERE: Final = frozenset({"provider_error", "setup_error", "crashed"})


class _LeftHeading(Heading):
    """A Markdown heading, bold and left-aligned, without rich's box."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        text = self.text
        text.justify = "left"
        text.stylize("bold")
        if self.tag == "h1":
            text.stylize("underline")
        yield text


class AnswerMarkdown(Markdown):
    """rich's Markdown with left-aligned headings."""

    elements = {**Markdown.elements, "heading_open": _LeftHeading}


def markdown(text: str) -> Markdown:
    return AnswerMarkdown(text, code_theme="ansi_dark", hyperlinks=False)


@dataclass
class LiveState:
    """What the live area shows while a turn runs."""

    activity: str
    started: float
    step: int = 0
    attempt: int = 1
    # The latest run.progress data; None before one (facts from step.started).
    progress: Mapping[str, Any] | None = None
    # The session's first turn: only then may the past-runs row show.
    first_turn: bool = False
    output_tokens: int = 0
    tool: str = ""
    preview: str = ""
    tool_started: float = 0.0
    tool_limit_s: float = 0.0
    # When a retry notice's wait ends; the activity counts it down.
    retry_until: float = 0.0
    finished: bool = False
    waiting: bool = False


@dataclass(frozen=True)
class ToolOutput:
    """One finished call, for /expand."""

    ref: str  # "<turn>.<call>", or "!<n>" for a shell command
    name: str
    preview: str
    status: str
    text: str  # tool.finished output, or error when the call failed


@dataclass
class _Call:
    name: str
    preview: str


def _marked(symbol: str, style: str, body: RenderableType) -> Table:
    """*body* after *symbol*, its later lines indented under the first."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=cells(symbol), no_wrap=True)
    # Fold, not cut: a long word, a URL or a line of Chinese keeps every
    # character.
    grid.add_column(overflow="fold")
    grid.add_row(Text(symbol, style=style), body)
    return grid


def plain(text: str) -> str:
    """*text* from the model or a server as one line, with nothing it could act on."""
    return " ".join(clean_output(text).split())


def approval_tag(status: str, approval: Mapping[str, Any] | None) -> str:
    """The short tag of how the approval of a call ended; ``""`` when not asked."""
    if not isinstance(approval, Mapping):
        return ""
    origin = approval.get("origin")
    reason = " ".join(str(approval.get("reason") or "").split())
    if origin == "error":
        return "approval failed"
    if origin == "timeout":
        return "not answered in time"
    if approval.get("allowed"):
        if origin == "session":
            return "allowed by a session grant"
        if origin == "policy":
            return "auto-allowed"
        return "allowed for the session" if approval.get("remember") else "allowed"
    if origin == "policy":
        return f"denied by policy: {reason}" if reason else "denied by policy"
    if origin == "user":
        return f"denied: {reason}" if reason else "denied"
    return ""


def status_hint(
    status: str, error: str, backend: BackendInfo, cause: str = ""
) -> str | None:
    """The hint under the status line of *status*, or None (U2a.1's commands)."""
    url = backend.base_url or backend.label
    if status == "provider_error":
        http = _HTTP_STATUS.search(error)
        code = int(http.group(1)) if http else 0
        failed = _FAILED_WITH.search(error)
        name = failed.group(1) if failed and not http else ""
        if name in ("ConnectError", "ConnectTimeout"):
            return f"Is the model server running at {url}?"
        if name in ("ReadTimeout", "WriteTimeout", "PoolTimeout"):
            return (
                "The model server did not answer in time; it may still be "
                "loading the model."
            )
        if code == 404:
            hint = (
                f"The server does not know the model '{backend.model}'; check the "
                "name (/model lists the models it has)."
            )
            if backend.key == "ollama":
                hint += f" ollama pull {backend.model} fetches it."
            return hint
        if code == 400:
            return (
                "The server refused the request; if it says the model does not "
                "support tools, choose one that does."
            )
        if code in (401, 403):
            if backend.local:
                return (
                    "The server requires an API key; localmelo sends none to local "
                    'servers. Turn the key off in the server (oMLX: "Allow '
                    'inference without API key").'
                )
            return (
                f"The server refused the key in {backend.key_env or 'its key variable'}; "
                "melo --reconfigure changes the backend."
            )
        return None
    if status == "setup_error":
        return "A tool server failed to start."
    if status == "failed":
        if error == "empty replies":
            return "The model returned only empty replies; try another model."
        return "A check stopped the run (the reason is above); try another model."
    if status == "gate_stopped":
        return "melo judged that another attempt would not help. Rephrase the request."
    if status == "stuck":
        return "The model repeated itself; rephrase the request."
    if status == "crashed":
        if cause == "crash":
            return (
                "melo stopped before this turn finished (the process was killed "
                "or the computer lost power)."
            )
        return f"This is a bug in localmelo; the log is {LOG_PATH}."
    return None


def stopped_line(cause: str, symbols: Symbols) -> Text:
    if cause == "takeover":
        text = "Stopped: another melo window took over this session"
    elif cause in ("shutdown", "crash"):
        text = "Interrupted"
    else:
        text = "Stopped"
    return Text(f"{symbols.stopped} {text}", style="warn")


class TurnRenderer:
    """Turns a turn's events into scrollback blocks and the live area's state."""

    def __init__(
        self,
        out: Out,
        *,
        symbols: Symbols,
        backend: Callable[[], BackendInfo],
        verbose: bool = False,
        thinking: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.out = out
        self.symbols = symbols
        # Per session, not saved: /verbose and /thinking flip them.
        self.verbose = verbose
        self.thinking = thinking
        self._backend = backend
        self._clock = clock
        self.live: LiveState | None = None
        # The latest run.progress of this session, for /progress.
        self.progress: Mapping[str, Any] | None = None
        # A cancelled run.finished whose turn.finished has not come yet.
        self._cancelled_pending = False
        # The latest request's prompt tokens, for the status bar.
        self.context_tokens = 0
        # Calls a reply listed that have not started, and the previews of
        # those that did, by call id.
        self._calls: dict[str, _Call] = {}
        self._previews: dict[str, str] = {}
        # The arguments each started call was given, for /verbose blocks.
        self._arguments: dict[str, Any] = {}
        self._finished = False
        self._tool_count = 0
        self._stop_cause = ""
        # An assistant message without tool calls, until the next step or
        # the end of the run decides whether it prints (section 3).
        self._held = ""
        # The finished calls of this session, for /expand: turn number and
        # call number by turn.
        self.tool_outputs: list[ToolOutput] = []
        self._turn = 0
        self._call_number = 0

    # ── A turn ──

    def echo(self, text: str) -> None:
        """Print *text* after the prompt symbol, as it was sent (a command's echo)."""
        lines = text.splitlines() or [""]
        shown = lines[:ECHO_LINES]
        echo = Text()
        for number, line in enumerate(shown):
            if number:
                echo.append("\n")
            echo.append(expand_tabs(clean_output(line)))
        if len(lines) > len(shown):
            echo.append(
                f"\n{self.symbols.ellipsis} +{len(lines) - len(shown)} lines",
                style="muted",
            )
        self.out.print(_marked(self.symbols.prompt, "accent", echo))

    def begin_turn(self, text: str, *, first_turn: bool = False) -> None:
        """Echo *text* and start the live area."""
        self.echo(text)
        self.out.print(Text(""))
        self.live = LiveState(
            activity="Starting", started=self._clock(), first_turn=first_turn
        )
        self.progress = None
        self._calls = {}
        self._previews = {}
        self._finished = False
        self._cancelled_pending = False
        self._tool_count = 0
        self._stop_cause = ""
        self._held = ""
        self._turn += 1
        self._call_number = 0

    @property
    def finished(self) -> bool:
        """Whether this turn's ``run.finished`` has come."""
        return self._finished

    def on_event(self, stored: StoredEvent) -> None:
        """One stored event of the turn (the runner's ``on_event``)."""
        event = stored.event
        kind = event.get("type")
        data = event.get("data")
        if not isinstance(kind, str) or not isinstance(data, Mapping):
            return
        handler = getattr(self, "_on_" + kind.replace(".", "_"), None)
        if handler is not None:
            handler(data)

    def end_turn(
        self,
        result: RunResult | None,
        error: BaseException | None,
        *,
        cause: str = "",
    ) -> None:
        """Close the live area and print what no event covered."""
        if not self._finished:
            if error is not None:
                message = str(error)
                text = (
                    f"{type(error).__name__}: {message}"
                    if message
                    else (type(error).__name__)
                )
                self.out.print(
                    Text(
                        f"{self.symbols.failed} The turn could not run: {text}",
                        style="error",
                    )
                )
            elif result is not None and result.status == "cancelled":
                self._print_unstarted()
                self._cancelled_pending = False
                self.out.print(
                    stopped_line(cause or self._stop_cause or "user_stop", self.symbols)
                )
            elif result is not None:
                self._print_result(result, footer=False)
        elif self._cancelled_pending:
            # turn.finished never reached the sink: use what the CLI knows.
            self._cancelled_pending = False
            self.out.print(
                stopped_line(cause or self._stop_cause or "user_stop", self.symbols)
            )
        self.out.print(Text(""))
        self.live = None

    # ── Events ──

    def _on_run_started(self, data: Mapping[str, Any]) -> None:
        if data.get("approval") is False:
            self.out.print(
                Text("  No approvals for this turn (--yolo).", style="muted")
            )

    def _clear_retry(self) -> None:
        """A non-retry event ends the retry state (section 3's activity table)."""
        live = self.live
        if live is None:
            return
        if live.activity == "Retrying":
            live.activity = "Thinking"
        live.retry_until = 0.0

    def _on_step_started(self, data: Mapping[str, Any]) -> None:
        self._print_held()
        self._print_unstarted()
        if self.live is not None:
            self.live.step = _int(data.get("step"), self.live.step)
            self.live.attempt = _int(data.get("attempt"), self.live.attempt)
            self.live.activity = "Thinking"
            self.live.retry_until = 0.0

    def _on_assistant_message(self, data: Mapping[str, Any]) -> None:
        self._count_usage(data.get("usage"))
        calls = data.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, Mapping) and isinstance(call.get("id"), str):
                    self._calls[call["id"]] = _Call(
                        name=str(call.get("name") or "?"),
                        preview=str(call.get("preview") or ""),
                    )
        self._clear_retry()
        content = str(data.get("content") or "")
        block, text = split_thinking(content)
        thinking = str(data.get("thinking") or "")
        if not thinking.strip():
            thinking = block
        if thinking.strip():
            self._print_thinking(thinking, _float(data.get("duration_s")))
        text = clean_output(text).strip()
        if not isinstance(calls, list) or not calls:
            # The last message may be the answer: hold it until the end
            # says whether to print it (section 3).
            self._held = text
            return
        self._held = ""
        if text:
            self.out.print(_marked(self.symbols.assistant, "", markdown(text)))

    def _print_thinking(self, text: str, duration_s: float) -> None:
        """``✻ Thought for 1.2s`` and the thinking's first rows (section 3)."""
        self.out.print(
            Text(f"{self.symbols.thought} Thought for {format_duration(duration_s)}")
        )
        if not self.thinking:
            return
        limit = VERBOSE_THINKING_LINES if self.verbose else THINKING_LINES
        rows = _wrapped_paragraphs(text, max(1, self.out.width - 2))
        shown = rows[:limit]
        body = Text(no_wrap=True, overflow="ellipsis")
        for number, row in enumerate(shown):
            if number:
                body.append("\n")
            body.append("  " + row)
        if len(rows) > len(shown):
            marker = f"\n  {self.symbols.ellipsis} +{len(rows) - len(shown)} lines"
            body.append(marker + ("" if self.verbose else " (/verbose)"))
        self.out.print(Padding(body, (0, 0, 0, 0), style="muted"))

    def _print_held(self) -> None:
        """Print a held assistant message as intermediate text."""
        text, self._held = self._held, ""
        if text:
            self.out.print(_marked(self.symbols.assistant, "", markdown(text)))

    def _on_tool_started(self, data: Mapping[str, Any]) -> None:
        preview = str(data.get("preview") or "")
        call_id = data.get("call_id")
        if isinstance(call_id, str):
            self._calls.pop(call_id, None)
            self._previews[call_id] = preview
            arguments = data.get("arguments")
            if isinstance(arguments, Mapping):
                self._arguments[call_id] = arguments
        if self.live is None:
            return
        name = str(data.get("name") or "?")
        self._clear_retry()
        self.live.activity = "Running"
        self.live.tool = name
        self.live.preview = preview
        self.live.tool_started = self._clock()
        self.live.tool_limit_s = SESSION_TOOL_TIMEOUT_S if name in LIMITED_TOOLS else 0

    def _on_tool_finished(self, data: Mapping[str, Any]) -> None:
        self._tool_count += 1
        self._call_number += 1
        self.out.print(self.tool_block(data, ref=f"{self._turn}.{self._call_number}"))
        self._remember_tool_output(data)
        call_id = data.get("call_id")
        if isinstance(call_id, str):
            self._arguments.pop(call_id, None)
        if self.live is not None:
            self.live.tool = ""
            self._clear_retry()
            self.live.activity = "Thinking"

    def _remember_tool_output(self, data: Mapping[str, Any]) -> None:
        """Keep a finished call's text for /expand."""
        status = str(data.get("status") or "")
        text = str(data.get("output") or "")
        error = str(data.get("error") or "")
        self.tool_outputs.append(
            ToolOutput(
                ref=f"{self._turn}.{self._call_number}",
                name=str(data.get("name") or "?"),
                preview=self._preview_of(data),
                status=status,
                text=error if error and status != "success" else text,
            )
        )

    def _on_notice(self, data: Mapping[str, Any]) -> None:
        kind = str(data.get("kind") or "")
        message = plain(str(data.get("message") or ""))
        if kind == "retry":
            line = Text(f"{self.symbols.retry} {message}", style="muted")
            line.truncate(self.out.width, overflow="ellipsis")
            self.out.print(line)
            if self.live is not None:
                detail = data.get("detail")
                delay = (
                    _float(detail.get("delay_s"))
                    if isinstance(detail, Mapping)
                    else 0.0
                )
                self.live.activity = "Retrying"
                self.live.retry_until = self._clock() + delay if delay > 0 else 0.0
            return
        if kind == "empty_reply":
            line = Text(f"{self.symbols.dot} {message}", style="muted")
        else:
            line = Text(f"{self.symbols.warn} {message}", style="warn")
        line.truncate(self.out.width, overflow="ellipsis")
        self.out.print(line)

    def _on_reflection_started(self, data: Mapping[str, Any]) -> None:
        if self.live is not None:
            self._clear_retry()
            self.live.activity = "Reflecting"

    def _on_reflection(self, data: Mapping[str, Any]) -> None:
        self._count_usage(data.get("usage"))
        if self.live is not None:
            self._clear_retry()
            self.live.activity = "Thinking"
        attempt = _int(data.get("attempt"), 0)
        duration = format_duration(_float(data.get("duration_s")))
        if data.get("readable") is False:
            tail = (
                "the run continues"
                if data.get("continue_run")
                else ("the budget is used up")
            )
            line = (
                f"{self.symbols.thought} Could not read the reflection after "
                f"attempt {attempt}; {tail} ({duration})"
            )
            self.out.print(_one_line(line, self.out.width, self.symbols))
            return
        action = plain(str(data.get("action") or ""))
        first = _first_line(str(data.get("rationale") or ""))
        line = f"{self.symbols.thought} Reflected after attempt {attempt}: {action}"
        if self.verbose and _int(data.get("requests"), 0) == 2:
            line += f" {self.symbols.dot} asked twice"
        if first:
            line += f" {self.symbols.dot} {first}"
        line += f" ({duration})"
        self.out.print(_one_line(line, self.out.width, self.symbols))
        if self.verbose:
            summary = plain(str(data.get("summary") or ""))
            rationale = plain(str(data.get("rationale") or ""))
            detail = Text(no_wrap=True, overflow="ellipsis")
            if summary:
                detail.append("  " + summary)
            if rationale and rationale != summary:
                if summary:
                    detail.append("\n")
                detail.append("  " + rationale)
            if detail.plain:
                self.out.print(Padding(detail, (0, 0, 0, 0), style="muted"))

    def _on_run_progress(self, data: Mapping[str, Any]) -> None:
        self.progress = dict(data)
        if self.live is not None:
            self.live.progress = self.progress
            self.live.step = _int(data.get("step"), self.live.step)
            self.live.attempt = _int(data.get("attempt"), self.live.attempt)

    def _on_approval_requested(self, data: Mapping[str, Any]) -> None:
        if self.live is not None:
            self.live.waiting = True
            self.live.activity = "Waiting for your approval"

    def _on_approval_resolved(self, data: Mapping[str, Any]) -> None:
        if self.live is not None:
            self.live.waiting = False
            self.live.activity = "Thinking"

    def _on_run_finished(self, data: Mapping[str, Any]) -> None:
        raw = data.get("result")
        if not isinstance(raw, Mapping):
            return
        result = RunResult(
            task_id=str(raw.get("task_id") or ""),
            status=str(raw.get("status") or ""),
            result=str(raw.get("result") or ""),
            error=str(raw.get("error") or ""),
            n_tool=_int(raw.get("n_tool"), 0),
            duration_s=float(raw.get("duration_s") or 0.0),
        )
        self._finished = True
        self._clear_retry()
        self._print_unstarted()
        # The last message prints when the answer differs from it (section 3).
        held, self._held = self._held, ""
        _, answer = split_thinking(result.result)
        if held and held.strip() != answer.strip():
            self.out.print(_marked(self.symbols.assistant, "", markdown(held)))
        if result.status == "cancelled":
            # The cause is turn.finished's, which comes after this event.
            self._cancelled_pending = True
        else:
            self._print_result(result, footer=True)
        if self.live is not None:
            self.live.finished = True
            self.live.waiting = False
            self.live.tool = ""
            self.live.activity = "Wrapping up"
            self.live.retry_until = 0.0

    def _on_turn_finished(self, data: Mapping[str, Any]) -> None:
        """The turn's end: a cancelled run's cause, or a stale turn's status."""
        cause = str(data.get("cause") or "")
        if self._cancelled_pending:
            self._cancelled_pending = False
            self.out.print(
                stopped_line(cause or self._stop_cause or "user_stop", self.symbols)
            )
            return
        if self._finished:
            return
        self._finished = True
        status = str(data.get("status") or "")
        error = str(data.get("error") or "")
        if status == "cancelled":
            self.out.print(
                stopped_line(cause or self._stop_cause or "user_stop", self.symbols)
            )
        elif status:
            # A crash is a bug only when the run raised it; a crash cause
            # means the process stopped (section 3's severity table).
            severe = status in {"provider_error", "setup_error"} or (
                status == "crashed" and cause != "crash"
            )
            mark = self.symbols.failed if severe else self.symbols.warn
            line = Text(
                f"{mark} {status_line(status, plain(error))}",
                style="error" if severe else "warn",
                no_wrap=True,
                overflow="ellipsis",
            )
            line.truncate(self.out.width, overflow="ellipsis")
            self.out.print(line)
            hint = status_hint(status, error, self._backend(), cause)
            hint = hint if hint is None else clean_output(hint)
            if hint:
                self.out.print(Padding(Text(hint, style="muted"), (0, 0, 0, 2)))

    def new_session(self) -> None:
        """Forget this process's calls, progress facts and turn numbers (/new)."""
        self.tool_outputs = []
        self.progress = None
        self._turn = 0
        self._call_number = 0
        self._held = ""
        self._calls = {}
        self._previews = {}
        self._arguments = {}
        self._finished = False
        self._cancelled_pending = False
        self._tool_count = 0
        self._stop_cause = ""
        self.verbose = False
        self.thinking = True

    def stopped_by(self, cause: str) -> None:
        """Note that the turn is being stopped for *cause* (its stopped line)."""
        self._stop_cause = cause
        if self.live is not None and not self.live.finished:
            self.live.activity = "Stopping"

    # ── Blocks ──

    def _count_usage(self, usage: object) -> None:
        if not isinstance(usage, Mapping):
            return
        prompt = usage.get("prompt_tokens")
        if isinstance(prompt, int) and prompt > 0:
            self.context_tokens = prompt
        completion = usage.get("completion_tokens")
        if self.live is not None and isinstance(completion, int):
            self.live.output_tokens += max(0, completion)

    def _print_unstarted(self) -> None:
        """Print the calls a reply listed that never started (not run)."""
        calls, self._calls = self._calls, {}
        for call in calls.values():
            self.out.print(
                self._header(
                    self.symbols.not_run, "muted", call.name, call.preview, ["not run"]
                )
            )

    def _header(
        self, symbol: str, style: str, name: str, preview: str, meta: list[str]
    ) -> Text:
        """``<symbol> <name>  <preview> · <meta> · ...``, cut to the width.

        The preview gives way to the meta, except to a long last part that
        carries free text (a denial's reason, a block's error), which is cut
        at the line's end; a fixed tag such as ``allowed by a session
        grant`` always shows whole.
        """
        width = self.out.width
        name = plain(name)
        dot = f" {self.symbols.dot} "
        meta = [part for part in meta if part]
        start = f"{symbol} {name}  "
        shown, _ = escape_arguments(" ".join(preview.split("\n")))
        fixed = sum(cells(dot + part) for part in meta[:-1])
        last = cells(dot + meta[-1]) if meta else 0
        if meta and meta[-1].startswith(_FREE_TEXT_TAGS):
            last = min(last, FREE_TEXT_CELLS)
        room = width - cells(start) - fixed - last
        head = Text(f"{symbol} ", style=style)
        head.append(name, style="tool")
        if shown and room >= 4:
            head.append("  " + cut_cells(shown, room, self.symbols.ellipsis))
        for part in meta:
            head.append(dot, style="muted")
            head.append(part, style="warn" if part.startswith("exit ") else "muted")
        head.no_wrap = True
        head.truncate(width, overflow="ellipsis")
        return head

    def tool_block(self, data: Mapping[str, Any], *, ref: str = "") -> RenderableType:
        """The block of one finished tool call: its header, arguments, output, images."""
        status = str(data.get("status") or "")
        name = str(data.get("name") or "?")
        preview = self._preview_of(data)
        approval = data.get("approval")
        approval = approval if isinstance(approval, Mapping) else None
        duration = format_duration(_float(data.get("duration_s")))
        output = str(data.get("output") or "")
        error = str(data.get("error") or "")
        symbols = self.symbols
        tag = approval_tag(status, approval)
        if status == "cancelled":
            withdrawn = approval is not None and approval.get("origin") == "withdrawn"
            meta = ["not run" if withdrawn else f"stopped after {duration}"]
            return self._header(symbols.not_run, "muted", name, preview, meta)
        if status == "blocked":
            if not tag:
                first = clean_output(error).strip().split("\n")[0]
                tag = f"blocked: {first}" if first else "blocked"
            return self._header(symbols.blocked, "blocked", name, preview, [tag])
        meta = [duration]
        exit_code = data.get("exit_code")
        if status == "success":
            symbol, style, body = symbols.ok, "ok", output
            if isinstance(exit_code, int) and exit_code != 0:
                meta.append(f"exit {exit_code}")
        else:
            symbol, style = symbols.failed, "error"
            body = error + ("\n" + output if output.strip() else "")
        meta.append(tag)
        parts: list[RenderableType] = [self._header(symbol, style, name, preview, meta)]
        if self.verbose:
            arguments = _argument_lines(self._arguments_of(data), self.out.width)
            if arguments is not None:
                parts.append(arguments)
        parts.append(self._body(body, ref=ref))
        images = _image_lines(data.get("images"), self.symbols)
        if images is not None:
            parts.append(images)
        return Group(*parts)

    def _arguments_of(self, data: Mapping[str, Any]) -> object:
        """The arguments a finished call was started with (tool.started's)."""
        arguments = data.get("arguments")
        if isinstance(arguments, Mapping):
            return arguments
        call_id = data.get("call_id")
        return self._arguments.get(call_id) if isinstance(call_id, str) else None

    def tool_output(self, ref: str | None) -> ToolOutput | None:
        """The finished call *ref* names, or the latest when *ref* is None."""
        if not self.tool_outputs:
            return None
        if ref is None:
            return self.tool_outputs[-1]
        return next((call for call in self.tool_outputs if call.ref == ref), None)

    def _preview_of(self, data: Mapping[str, Any]) -> str:
        call_id = data.get("call_id")
        if not isinstance(call_id, str):
            return ""
        call = self._calls.get(call_id)
        return call.preview if call is not None else self._previews.get(call_id, "")

    def _body(self, text: str, *, ref: str = "") -> Text:
        lines = [
            expand_tabs(line) for line in clean_output(text).rstrip("\n").split("\n")
        ]
        if not "".join(lines).strip():
            lines = ["(no output)"]
        limit = VERBOSE_BLOCK_LINES if self.verbose else BLOCK_LINES
        # A single line left prints instead of the "+1 lines" marker.
        shown = lines if len(lines) <= limit + 1 else lines[:limit]
        width = max(1, self.out.width - 4)
        body = Text(no_wrap=True, overflow="ellipsis")
        for number, line in enumerate(shown):
            prefix = f"  {self.symbols.connector} " if number == 0 else "    "
            if number:
                body.append("\n")
            body.append(prefix, style="muted")
            body.append(cut_cells(line, width, self.symbols.ellipsis))
        if len(shown) < len(lines):
            marker = f"\n    {self.symbols.ellipsis} +{len(lines) - len(shown)} lines"
            if ref:
                marker += f" (/expand {ref})"
            body.append(marker, style="muted")
        return body

    def _print_result(self, result: RunResult, *, footer: bool) -> None:
        symbols = self.symbols
        _, answer = split_thinking(result.result)
        answer = clean_output(answer)
        if result.status == "completed":
            body: RenderableType = (
                markdown(answer) if answer else Text("(no answer text)", style="muted")
            )
            self.out.print(_marked(symbols.assistant, "", body))
        else:
            if answer:
                # Inline before a paragraph; on a line of its own before a
                # block, which "Best effort: " in front would break.
                effort: RenderableType = (
                    Group(Text("Best effort:"), markdown(answer))
                    if _BLOCK_START.match(answer)
                    else markdown("Best effort: " + answer)
                )
                self.out.print(_marked(symbols.assistant, "", effort))
            # A run that crashes as it runs is a bug (cause run): an error.
            severe = result.status in _SEVERE
            mark = symbols.failed if severe else symbols.warn
            line = Text(
                f"{mark} {status_line(result.status, plain(result.error))}",
                style="error" if severe else "warn",
                no_wrap=True,
                overflow="ellipsis",
            )
            line.truncate(self.out.width, overflow="ellipsis")
            self.out.print(line)
            hint = status_hint(result.status, result.error, self._backend())
            hint = hint if hint is None else clean_output(hint)
            if hint:
                self.out.print(Padding(Text(hint, style="muted"), (0, 0, 0, 2)))
        if footer:
            self.out.print(Text(self.footer(result), style="muted", no_wrap=True))

    def footer(self, result: RunResult) -> str:
        """``✻ Worked for 14s · 1 tool call · ↓ 412 tokens``, on one line."""
        calls = result.n_tool or self._tool_count
        parts = [
            f"{self.symbols.thought} Worked for {format_duration(result.duration_s)}",
            f"{calls} tool call" + ("" if calls == 1 else "s"),
        ]
        tokens = self.live.output_tokens if self.live is not None else 0
        token_part = (
            self.symbols.output_tokens.format(format_tokens(tokens)) if tokens else ""
        )
        text = f" {self.symbols.dot} ".join(
            parts + ([token_part] if token_part else [])
        )
        if cells(text) > self.out.width and token_part:
            text = f" {self.symbols.dot} ".join(parts)  # the tokens go first
        return cut_cells(text, self.out.width, self.symbols.ellipsis)


def _one_line(line: str, width: int, symbols: Symbols) -> Text:
    """*line* cut to *width* cells with the ellipsis: never wrapped."""
    return Text(
        cut_cells(line, width, symbols.ellipsis), no_wrap=True, overflow="ellipsis"
    )


def _first_line(text: str) -> str:
    """The first non-blank line of *text*, cleaned and collapsed."""
    for line in clean_output(text).splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


def _int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _wrapped_paragraphs(text: str, width: int) -> list[str]:
    """*text*'s paragraphs with their inner line breaks joined, wrapped by cells."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in clean_output(text).splitlines():
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    rows: list[str] = []
    for paragraph in paragraphs:
        # A wrap can land on a space, which would then lead the next row.
        rows.extend(
            row.strip() for row in wrap_words(paragraph, max(2, width)) if row.strip()
        )
    return rows


def _argument_lines(arguments: object, width: int) -> Text | None:
    """A call's arguments as indented JSON under its header (/verbose)."""
    if not isinstance(arguments, Mapping) or not arguments:
        return None
    text = json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
    body = Text(style="muted", no_wrap=True, overflow="ellipsis")
    for number, line in enumerate(expand_tabs(text).split("\n")[:ARGUMENT_LINES]):
        if number:
            body.append("\n")
        shown, _ = escape_arguments(line)
        body.append("    " + cut_cells(shown, max(1, width - 4), "…"))
    return body


def _image_lines(images: object, symbols: Symbols) -> Text | None:
    """One dim line per image a call produced: ``▪ image/png · 312 KB · not shown``."""
    if not isinstance(images, list) or not images:
        return None
    body = Text(no_wrap=True, overflow="ellipsis")
    first = True
    for image in images:
        if not isinstance(image, Mapping):
            continue
        if not first:
            body.append("\n")
        first = False
        mime = clean_output(str(image.get("mime_type") or "image"))
        size = format_bytes(_int(image.get("size_bytes"), 0))
        body.append(
            "    "
            + f"{symbols.image} {mime} {symbols.dot} {size} {symbols.dot} not shown",
            style="muted",
        )
    return None if first else body


# ── The live area ──

SPINNER_FRAME_S: Final = 0.12


def live_rows(
    live: LiveState,
    now: float,
    width: int,
    symbols: Symbols,
    *,
    frame: int | None = None,
) -> list[list[tuple[str, str]]]:
    """The live area's rows as prompt_toolkit fragments, each within *width* cells.

    Row 1 is the spinner line: the frame, the activity, the run's facts
    from P3's :func:`format_line`, the turn's output tokens and the stop
    hint. When the width is short, whole parts go: the hint, then the
    tokens, then ``format_line`` leaves out its own parts and cuts the rest
    with ``…`` (section 3). Row 2, while a tool runs, is its name, preview,
    clock and limit. Row 3, on a session's first turn with past runs, is
    the progress line's past facts and items (section 3). *frame* fixes the
    spinner's frame (``LOCALMELO_SPINNER_FRAME``).
    """
    dot = f" {symbols.dot} "
    if live.finished:
        return [[("class:muted", cut_cells(f"Wrapping up{symbols.ellipsis}", width))]]
    if live.waiting:
        # The panel below names its keys.
        head, hint = f"{symbols.warn} Waiting for your approval", ""
    else:
        index = (
            frame if frame is not None else int((now - live.started) / SPINNER_FRAME_S)
        )
        mark = symbols.spinner[index % len(symbols.spinner)]
        if live.retry_until > now:
            left = max(0, math.ceil(live.retry_until - now))
            activity = f"Retrying in {left}s"
        else:
            activity = live.activity
        head = f"{mark} {activity}{symbols.ellipsis}"
        hint = "" if live.activity == "Stopping" else "esc to stop"
    tokens = (
        symbols.output_tokens.format(format_tokens(live.output_tokens))
        if live.output_tokens
        else ""
    )
    payload: dict[str, Any] = (
        dict(live.progress)
        if isinstance(live.progress, Mapping)
        else {"step": live.step, "attempt": live.attempt}
    )
    # The third row carries these; the first one holds the facts alone.
    payload.pop("past", None)
    payload.pop("items", None)

    def compose(extras: Sequence[str]) -> str:
        suffix = dot.join(extras)
        used = cells(head) + (cells(dot + suffix) if suffix else 0)
        room = max(1, width - used - cells(dot))
        facts = format_line(payload, None, now, room, run_started=live.started)
        # P3's line joins with its own dot; the UI's set decides the glyph.
        facts = facts.replace(" · ", dot)
        return dot.join([head, facts, suffix] if suffix else [head, facts])

    # Whole parts go when the row is short, never half a number: the hint,
    # then the tokens; then format_line drops its own parts and cuts the
    # rest with an ellipsis (section 3).
    full_facts = format_line(payload, None, now, 10**6, run_started=live.started)

    def total(extras: Sequence[str]) -> int:
        suffix = dot.join(extras)
        used = cells(head) + cells(dot + full_facts)
        return used + (cells(dot + suffix) if suffix else 0)

    extras = [part for part in (tokens, hint) if part]
    for drop in (hint, tokens):
        if drop and drop in extras and total(extras) > width:
            extras = [part for part in extras if part != drop]
    first = cut_cells(compose(extras), width, symbols.ellipsis)
    rows = [[("class:accent", first)]]
    if live.tool and not live.waiting:
        clock = tool_elapsed(now - live.tool_started)
        if live.tool_limit_s:
            clock += f" of {tool_elapsed(live.tool_limit_s)}"
        preview, _ = escape_arguments(" ".join(live.preview.split("\n")))
        start = f"  {symbols.connector} {plain(live.tool)}  "
        room = width - cells(start) - cells(dot + clock)
        shown = cut_cells(preview, room, symbols.ellipsis) if room >= 4 else ""
        line = cut_cells(start + shown + dot + clock, width, symbols.ellipsis)
        rows.append([("class:muted", line)])
    if live.first_turn and isinstance(live.progress, Mapping):
        basis = live.progress.get("basis")
        match = basis.get("match") if isinstance(basis, Mapping) else None
        if live.progress.get("past") is not None and match != "new":
            past = format_past(live.progress, width).replace(" · ", dot)
            if past:
                rows.append([("class:muted", cut_cells(past, width, symbols.ellipsis))])
    return rows
