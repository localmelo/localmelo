"""A turn's events, as blocks in scrollback and as the live area's rows.

:class:`TurnRenderer` gets each stored event of the turn (the runner's
``on_event``), keeps :class:`LiveState` for the live area and prints what
has finished: the echo of the user's message, one block per tool call, the
answer as Markdown, the status line of a run that did not complete, with
its hint, and the footer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
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
)
from localmelo.support.cli.theme import Symbols
from localmelo.support.cli.views import BackendInfo
from localmelo.support.progress_line import format_line, tool_elapsed
from localmelo.support.sessions.agents import SESSION_TOOL_TIMEOUT_S
from localmelo.support.sessions.display import (
    clean_output,
    format_duration,
    format_tokens,
    status_line,
)
from localmelo.support.sessions.store import StoredEvent

# Output lines a tool block shows under its header.
BLOCK_LINES: Final = 3
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
    output_tokens: int = 0
    tool: str = ""
    preview: str = ""
    tool_started: float = 0.0
    tool_limit_s: float = 0.0
    finished: bool = False
    waiting: bool = False


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


def status_hint(status: str, error: str, backend: BackendInfo) -> str | None:
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.out = out
        self.symbols = symbols
        self._backend = backend
        self._clock = clock
        self.live: LiveState | None = None
        # The latest request's prompt tokens, for the status bar.
        self.context_tokens = 0
        # Calls a reply listed that have not started, and the previews of
        # those that did, by call id.
        self._calls: dict[str, _Call] = {}
        self._previews: dict[str, str] = {}
        self._finished = False
        self._tool_count = 0
        self._stop_cause = ""

    # ── A turn ──

    def begin_turn(self, text: str) -> None:
        """Echo *text* and start the live area."""
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
        self.out.print(_marked(self.symbols.prompt, "accent", echo), Text(""))
        self.live = LiveState(activity="Starting", started=self._clock())
        self._calls = {}
        self._previews = {}
        self._finished = False
        self._tool_count = 0
        self._stop_cause = ""

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
                self.out.print(
                    stopped_line(cause or self._stop_cause or "user_stop", self.symbols)
                )
            elif result is not None:
                self._print_result(result, footer=False)
        self.out.print(Text(""))
        self.live = None

    # ── Events ──

    def _on_run_started(self, data: Mapping[str, Any]) -> None:
        if data.get("approval") is False:
            self.out.print(
                Text("  No approvals for this turn (--yolo).", style="muted")
            )

    def _on_step_started(self, data: Mapping[str, Any]) -> None:
        self._print_unstarted()
        if self.live is not None:
            self.live.step = _int(data.get("step"), self.live.step)
            self.live.attempt = _int(data.get("attempt"), self.live.attempt)
            self.live.activity = "Thinking"

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

    def _on_tool_started(self, data: Mapping[str, Any]) -> None:
        preview = str(data.get("preview") or "")
        call_id = data.get("call_id")
        if isinstance(call_id, str):
            self._calls.pop(call_id, None)
            self._previews[call_id] = preview
        if self.live is None:
            return
        name = str(data.get("name") or "?")
        self.live.activity = "Running"
        self.live.tool = name
        self.live.preview = preview
        self.live.tool_started = self._clock()
        self.live.tool_limit_s = SESSION_TOOL_TIMEOUT_S if name in LIMITED_TOOLS else 0

    def _on_tool_finished(self, data: Mapping[str, Any]) -> None:
        self._tool_count += 1
        self.out.print(self.tool_block(data))
        if self.live is not None:
            self.live.tool = ""
            self.live.activity = "Thinking"

    def _on_reflection_started(self, data: Mapping[str, Any]) -> None:
        if self.live is not None:
            self.live.activity = "Reflecting"

    def _on_reflection(self, data: Mapping[str, Any]) -> None:
        self._count_usage(data.get("usage"))
        if self.live is not None:
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
        self._print_unstarted()
        if result.status == "cancelled":
            self.out.print(stopped_line(self._stop_cause or "user_stop", self.symbols))
        else:
            self._print_result(result, footer=True)
        if self.live is not None:
            self.live.finished = True
            self.live.tool = ""
            self.live.activity = "Wrapping up"

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

        The preview gives way to the meta, except to a long last part (an
        approval's reason, a block's), which is cut at the line's end.
        """
        width = self.out.width
        name = plain(name)
        dot = f" {self.symbols.dot} "
        meta = [part for part in meta if part]
        start = f"{symbol} {name}  "
        shown, _ = escape_arguments(" ".join(preview.split("\n")))
        fixed = sum(cells(dot + part) for part in meta[:-1])
        last = min(cells(dot + meta[-1]), 24) if meta else 0
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

    def tool_block(self, data: Mapping[str, Any]) -> RenderableType:
        """The block of one finished tool call: its header and its first lines."""
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
        header = self._header(symbol, style, name, preview, meta)
        return _Block(header, self._body(body))

    def _preview_of(self, data: Mapping[str, Any]) -> str:
        call_id = data.get("call_id")
        if not isinstance(call_id, str):
            return ""
        call = self._calls.get(call_id)
        return call.preview if call is not None else self._previews.get(call_id, "")

    def _body(self, text: str) -> Text:
        lines = [
            expand_tabs(line) for line in clean_output(text).rstrip("\n").split("\n")
        ]
        if not "".join(lines).strip():
            lines = ["(no output)"]
        shown = lines[:BLOCK_LINES]
        width = max(1, self.out.width - 4)
        body = Text(no_wrap=True, overflow="ellipsis")
        for number, line in enumerate(shown):
            prefix = f"  {self.symbols.connector} " if number == 0 else "    "
            if number:
                body.append("\n")
            body.append(prefix, style="muted")
            body.append(cut_cells(line, width, self.symbols.ellipsis))
        if len(shown) < len(lines):
            body.append(
                f"\n    {self.symbols.ellipsis} +{len(lines) - len(shown)} lines",
                style="muted",
            )
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
            self.out.print(Text(self.footer(result), style="muted"))

    def footer(self, result: RunResult) -> str:
        """``✻ Worked for 14s · 1 tool call · ↓ 412 tokens``."""
        calls = result.n_tool or self._tool_count
        parts = [
            f"{self.symbols.thought} Worked for {format_duration(result.duration_s)}",
            f"{calls} tool call" + ("" if calls == 1 else "s"),
        ]
        tokens = self.live.output_tokens if self.live is not None else 0
        if tokens:
            parts.append(self.symbols.output_tokens.format(format_tokens(tokens)))
        return f" {self.symbols.dot} ".join(parts)


class _Block:
    """A tool block: its header line and its body, as one renderable."""

    def __init__(self, header: Text, body: Text) -> None:
        self.header = header
        self.body = body

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield self.header
        yield self.body


def _int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


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

    Row 1 is the spinner line: the frame, the activity, the step, the
    attempt from the second, the elapsed time, the output tokens and the
    stop hint; when the width is short, whole parts go: the hint, then the
    tokens, then the elapsed time, the attempt and the step. Row 2, while a tool runs, is its name, preview, clock and limit.
    *frame* fixes the spinner's frame (``LOCALMELO_SPINNER_FRAME``).
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
        head = f"{mark} {live.activity}{symbols.ellipsis}"
        hint = "" if live.activity == "Stopping" else "esc to stop"
    tokens = (
        symbols.output_tokens.format(format_tokens(live.output_tokens))
        if live.output_tokens
        else ""
    )
    extras = [part for part in (tokens, hint) if part]
    payload = {"step": live.step, "attempt": live.attempt}
    line = format_line(payload, None, now, 10**6, run_started=live.started)
    facts = [part for part in line.split(" · ") if part]
    # Whole parts go when the row is short, never half a number: the hint,
    # the tokens, then the facts from the end (the elapsed time, then the
    # attempt, then the step).
    while True:
        first = dot.join([head, *facts, *extras])
        if cells(first) <= width or not (extras or facts):
            break
        (extras or facts).pop()
    first = cut_cells(first, width, symbols.ellipsis)
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
    return rows
