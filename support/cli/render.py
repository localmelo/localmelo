"""A turn's events, as blocks in scrollback and as the live area's rows.

:class:`TurnRenderer` gets each stored event of the turn (the runner's
``on_event``), keeps :class:`LiveState` for the live area and prints what
has finished: the echo of the user's message, thinking, intermediate text,
one block per tool call, notices and reflections, the answer as Markdown,
the status line of a run that did not complete, with its hint, the
footer and the line that names the files the turn wrote.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple, cast

from markdown_it.token import Token as MarkdownToken
from pygments.token import (  # type: ignore[import-untyped]
    Comment,
    Keyword,
    Name,
    String,
    Token,
)
from rich.console import (
    Console,
    ConsoleOptions,
    Group,
    RenderableType,
    RenderResult,
)
from rich.markdown import CodeBlock, Heading, Markdown
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.syntax import ANSISyntaxTheme, Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from localmelo.melo.checker.tiers import BUILTIN_TIERS
from localmelo.melo.schema import (
    UNFINISHED_ANSWER_ERROR,
    ReasoningTags,
    RunResult,
    split_thinking,
)
from localmelo.support.cli import edits
from localmelo.support.cli.approvals import EditSnapshots, Snapshot
from localmelo.support.cli.hanging import Hung
from localmelo.support.cli.hanging import prefix as hung_prefix
from localmelo.support.cli.output import Out
from localmelo.support.cli.steps import StepRow, call_status
from localmelo.support.cli.text import (
    cells,
    cut_cells,
    cut_middle,
    escape_arguments,
    expand_tabs,
    wrap_words,
)
from localmelo.support.cli.theme import Symbols
from localmelo.support.cli.views import BackendInfo
from localmelo.support.progress_line import (
    format_line,
    format_past,
    run_elapsed,
    tool_elapsed,
)
from localmelo.support.providers._http import UNREACHABLE_PREFIX
from localmelo.support.sessions.agents import SESSION_TOOL_TIMEOUT_S
from localmelo.support.sessions.display import (
    clean_output,
    display_path,
    format_bytes,
    format_duration,
    format_tokens,
    status_text,
)
from localmelo.support.sessions.store import (
    SessionBusyError,
    StoredEvent,
    UnknownSessionError,
)
from localmelo.support.usage_snapshot import UsageSnapshot

# Output lines a tool block shows under its header (20 with /verbose). A call
# that succeeded shows QUIET_BLOCK_LINES outside developer mode.
BLOCK_LINES: Final = 3
QUIET_BLOCK_LINES: Final = 1
VERBOSE_BLOCK_LINES: Final = 20
# The least room an output line keeps beside the "+N lines" marker of its row.
MARKER_MIN_LINE_CELLS: Final = 8
# Lines of the thinking excerpt (40 with /verbose).
THINKING_LINES: Final = 3
VERBOSE_THINKING_LINES: Final = 40
# Lines of a call's arguments under its header, with /verbose.
ARGUMENT_LINES: Final = 20
# A slash command named in what melo says: /retry, /model, /status.
COMMAND_PATTERN: Final = r"(?<![\w/.])/[a-z]+(?![\w/])"
# Lines /expand prints, then the rest's count.
EXPAND_LINES: Final = 200
# A block header's last part that carries free text keeps this much room.
FREE_TEXT_CELLS: Final = 24
_FREE_TEXT_TAGS: Final = ("denied: ", "denied by policy: ", "blocked: ")
# Lines of the user's message echoed.
ECHO_LINES: Final = 6
# Files the line under a footer names before "and N more".
CHANGED_NAMES: Final = 3
# Tools whose running line shows the session's time limit.
LIMITED_TOOLS: Final = frozenset({"shell_exec", "python_exec"})
# A tool's kind by its permission tier; a tool the session does not know is
# drawn as the cautious default, write.
KINDS: Final = {"read": "read", "write": "write", "exec": "run", "network": "network"}

LOG_PATH: Final = "~/.cache/localmelo/logs/cli.log"

# What a model with no declared reasoning format means for a streamed reply.
STREAM_BUFFERED: Final = (
    "This model's replies appear when complete until its reasoning format is set "
    "(see Streamed replies)."
)

# What a server that does not stream means for the replies of this session.
SERVER_WHOLE: Final = (
    "This server sends replies whole; melo shows them when they are complete."
)

_HTTP_STATUS: Final = re.compile(r"failed with HTTP (\d{3})")
_POST_FAILED: Final = re.compile(r"^POST \S+ failed (?:with )?")
# The longest reason a run's end shows after its title.
REASON_CHARS: Final = 240
# Markdown that starts a block: a fence, a heading, a list, a quote, a table.
_BLOCK_START: Final = re.compile(r"\s*(```|~~~|#|[-*+] |\d+[.)] |>|\|)")
_FAILED_WITH: Final = re.compile(r"failed: (\w+)")


class _LeftHeading(Heading):
    """A Markdown heading, left-aligned, without rich's box: bold, or the answer's role."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        text = self.text
        text.justify = "left"
        text.stylize(console.get_style("markdown.heading", default="bold"))
        if self.tag == "h1":
            text.stylize("underline")
        yield text


class _Code(CodeBlock):
    """A code block: on the raised panel when there is a fill, else beside a gutter.

    Its tokens take the roles' colours (``syntax.*`` of the theme), so a code
    block follows the terminal's background like everything else.
    """

    gutter = "│"

    @classmethod
    def create(cls, markdown: Markdown, token: MarkdownToken) -> _Code:
        block = cls((token.info or "").partition(" ")[0] or "text", markdown.code_theme)
        block.gutter = getattr(markdown, "gutter", cls.gutter)
        return block

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        def style(name: str) -> Style:
            return console.get_style(f"syntax.{name}", default="none")

        theme = ANSISyntaxTheme(
            {
                Token: Style(),
                Keyword: style("keyword"),
                Name.Function: style("name"),
                Name.Class: style("name"),
                Name.Builtin: style("name"),
                String: style("string"),
                Comment: style("comment"),
            }
        )
        fill = console.get_style("raised", default="none").bgcolor
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=theme,
            word_wrap=True,
            background_color=fill.name if fill is not None else None,
            padding=(1, 2) if fill is not None else 0,
        )
        if fill is not None:
            yield syntax
            return
        bar = Segment(self.gutter + " ", console.get_style("muted", default="none"))
        width = max(1, options.max_width - cells(bar.text))
        for line in console.render_lines(syntax, options.update(width=width)):
            yield bar
            yield from line
            yield Segment.line()


class AnswerMarkdown(Markdown):
    """rich's Markdown with left-aligned headings and melo's code blocks."""

    elements = {
        **Markdown.elements,
        "heading_open": _LeftHeading,
        "fence": _Code,
        "code_block": _Code,
    }


class _Answer:
    """The final answer's Markdown: headings and bullets in the answer's roles."""

    def __init__(self, body: RenderableType) -> None:
        self.body = body

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        console.push_theme(
            Theme(
                {
                    "markdown.heading": console.get_style("markdown.answer.heading"),
                    "markdown.item.bullet": console.get_style("markdown.answer.bullet"),
                    "markdown.item.number": console.get_style("markdown.answer.bullet"),
                },
                inherit=False,  # rich's defaults would undo the palette's
            )
        )
        try:
            lines = console.render_lines(self.body, options, pad=False, new_lines=True)
        finally:
            console.pop_theme()
        for line in lines:
            yield from line


class Band:
    """*body* on the raised fill across the whole row; *body* alone without a fill."""

    def __init__(self, body: RenderableType) -> None:
        self.body = body

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        fill = console.get_style("raised", default="none")
        if fill.bgcolor is None:
            yield self.body
            return
        for line in console.render_lines(self.body, options, style=fill, pad=True):
            yield from line
            yield Segment.line()


def markdown(text: str, symbols: Symbols, *, answer: bool = False) -> RenderableType:
    """*text* as Markdown; the final *answer* has its heading and bullets in Melo's fur."""
    body = AnswerMarkdown(text, hyperlinks=False)
    body.gutter = symbols.gutter  # type: ignore[attr-defined]
    return _Answer(body) if answer else body


@dataclass
class LiveState:
    """What the live area shows while a turn runs."""

    activity: str
    started: float
    step: int = 0
    attempt: int = 1
    # The latest run.progress data; None before one (facts from step.started).
    progress: Mapping[str, Any] | None = None
    # The session's first turn: outside developer mode only then may the
    # past-runs row show.
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
    # Released provisional text of the planning reply being streamed.
    stream_text: str = ""


@dataclass(frozen=True)
class ToolOutput:
    """One finished call, for /expand."""

    ref: str  # "<turn>.<call>", or "!<n>" for a shell command
    name: str
    preview: str
    status: str
    text: str  # tool.finished output, or error when the call failed
    duration_s: float | None = None
    exit_code: int | None = None


@dataclass
class _Call:
    name: str
    preview: str


@dataclass
class _Written:
    """A file the turn wrote with ``file_write``, as the turn found it and as it left it.

    *before* is the file's text when the turn first wrote it: ``""`` for a new file,
    ``None`` when it is not known (no approval read it first, or it is too large, or
    not text). *after* is the text of the last write, and *created* tells a new empty
    file from an old one that was written with the text it had.
    """

    before: str | None
    after: str
    created: bool


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
        if origin in ("policy", "auto"):
            review = approval.get("review")
            if (
                origin == "auto"
                and isinstance(review, Mapping)
                and review.get("categories")
            ):
                return "allowed: you asked for it"
            return "auto-allowed"
        if approval.get("allowance") == "once":
            return "allowed once"
        return "allowed for the session" if approval.get("remember") else "allowed"
    if origin == "auto":
        return f"blocked: {reason}" if reason else "blocked"
    if origin == "policy":
        return f"denied by policy: {reason}" if reason else "denied by policy"
    if origin == "user":
        return f"denied: {reason}" if reason else "denied"
    return ""


# What a person does next, by status; a status that is not here has no action.
_ACTIONS: Final = {
    "setup_error": "/mcp shows why.",
    "failed": "/retry, or try another model.",
    "gate_stopped": (
        "melo judged that another attempt would not help. "
        "Rephrase the request, or /retry."
    ),
    "context_budget_exceeded": (
        "Reduce the task or increase the configured context window; then /retry."
    ),
    "budget_exhausted": (
        "/retry tries again; for more steps, quit and run melo -c --steps 60."
    ),
    "stuck": "Rephrase the request, or /retry.",
}
# The statuses whose error is a sentence worth showing after the title.
_WITH_REASON: Final = frozenset(
    {"failed", "gate_stopped", "setup_error", "context_budget_exceeded", "crashed"}
)


def _reason(error: str) -> str:
    """*error* as one cleaned line without the request it came from, cut short."""
    text = _POST_FAILED.sub("", plain(error))
    if len(text) > REASON_CHARS:
        text = text[: REASON_CHARS - 1] + "…"
    return text


def _server_end(error: str, backend: BackendInfo) -> tuple[str, str]:
    """The end of a run whose model server failed: what happened, what to do."""
    url = backend.base_url or backend.label
    where = f"the model server at {url}" if backend.base_url else backend.label
    http = _HTTP_STATUS.search(error)
    code = int(http.group(1)) if http else 0
    failed = _FAILED_WITH.search(error)
    name = failed.group(1) if failed and not http else ""
    # ConnectError is the pre-"cannot reach" text of older saved runs.
    if error.startswith(UNREACHABLE_PREFIX) or name in (
        "ConnectError",
        "ConnectTimeout",
    ):
        fix = "Start it" if backend.local else "Check the network"
        return f"Could not reach {where}.", f"{fix}, then /retry."
    if name in ("ReadTimeout", "WriteTimeout", "PoolTimeout"):
        return (
            "The model server did not answer in time.",
            "It may still be loading the model. /retry tries again.",
        )
    if code == 404:
        action = "Check the name (/model lists the models it has)."
        if backend.key == "ollama":
            action += f" ollama pull {backend.model} fetches it."
        return f"The model server does not know the model '{backend.model}'.", action
    if code == 400:
        return (
            f"The model server refused the request: {_reason(error)}",
            "If it says the model does not support tools, choose one that does.",
        )
    if code in (401, 403):
        if backend.local:
            return (
                "The model server requires an API key.",
                "localmelo sends none to local servers. Turn the key off in the "
                'server (oMLX: "Allow inference without API key").',
            )
        key = backend.key_env or "its key variable"
        return (
            f"The server refused the key in {key}.",
            "melo setup changes the backend.",
        )
    return (
        f"Could not get an answer from the model server: {_reason(error)}",
        f"/status checks {where}; /retry tries again.",
    )


def run_end(
    status: str, error: str, backend: BackendInfo, cause: str = "", steps: int = 0
) -> tuple[str, str]:
    """What a run that did not complete says: what happened, then what to do.

    The words are the title of ``STATUS_TEXT`` with the reason that helps (the
    server's URL, the model, the step limit, the error of a few statuses); the
    action may be empty. Direct mode keeps its own machine line.
    """
    if status == "provider_error":
        headline, action = _server_end(error, backend)
    elif status == "failed" and error == "empty replies":
        headline, action = "The model returned only empty replies", _ACTIONS["failed"]
    elif status == "failed" and error == UNFINISHED_ANSWER_ERROR:
        headline, action = UNFINISHED_ANSWER_ERROR, _ACTIONS["failed"]
    else:
        headline, action = status_text(status, cause).title, _ACTIONS.get(status, "")
        if status == "budget_exhausted" and steps:
            headline += f" ({steps})"
        if status in _WITH_REASON and cause != "crash" and (reason := _reason(error)):
            headline += f": {reason}"
        if status == "crashed":
            action = (
                "The process was killed or the computer lost power. /retry runs it again."
                if cause == "crash"
                else f"This is a bug; the log is {LOG_PATH}."
            )
    return (headline if headline.endswith("…") else headline.rstrip(".") + "."), action


def stopped_line(cause: str, symbols: Symbols) -> Text:
    if cause == "takeover":
        text = "Stopped: another melo window took over this session"
    elif cause in ("shutdown", "crash"):
        text = "Interrupted"
    else:
        text = f"Stopped {symbols.dot} /retry runs it again"
    return Text(f"{symbols.stopped} {text}", style="warn")


def _same_display_text(left: str, right: str) -> bool:
    """Whether two texts would read the same after cleaning and whitespace."""
    return " ".join(clean_output(left).split()) == " ".join(clean_output(right).split())


class StreamFeed:
    """One turn's stream display: the agent's seam onto :class:`TurnRenderer`.

    Deltas only update the live region, which the app redraws on its tick,
    so a stream costs no direct terminal writes.
    """

    def __init__(self, renderer: TurnRenderer) -> None:
        self._renderer = renderer
        self.reasoning_tags = renderer.stream_hint()

    def step(self, number: int) -> None:
        self._renderer.stream_step(number)

    def opened(self) -> None:
        self._renderer.stream_opened()

    def held(self) -> None:
        self._renderer.stream_held()

    def content(self, delta: str) -> None:
        self._renderer.stream_content(delta)

    def whole(self) -> None:
        self._renderer.stream_whole()

    def closed(self) -> None:
        self._renderer.stream_closed()


class TurnRenderer:
    """Turns a turn's events into scrollback blocks and the live area's state."""

    def __init__(
        self,
        out: Out,
        *,
        symbols: Symbols,
        backend: Callable[[], BackendInfo],
        verbose: bool = False,
        thinking: bool | None = None,
        developer: bool = False,
        reasoning_tags: Mapping[str, str] | None = None,
        snapshots: EditSnapshots | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.out = out
        self.symbols = symbols
        # The session's file_write snapshots, for a block's change.
        self._snapshots = snapshots
        # Developer mode keeps the record's full detail; the default mode
        # shows a quiet one (thinking and a call's output in one line).
        self.developer = developer
        # Per session, not saved: /verbose and /thinking flip them. Thinking
        # excerpts are on by default in developer mode only.
        self.verbose = verbose
        self.thinking = developer if thinking is None else thinking
        self._default_thinking = self.thinking
        # Per model, from [cli.reasoning_tags]: trusted framing declarations
        # for the stream and for the whole-reply display.
        self.reasoning_tags = dict(reasoning_tags or {})
        # Provisional streamed text per planning step (see stream_step).
        self._stream_texts: dict[int, str] = {}
        self._stream_step = 0
        self._stream_open = False
        # How many parts the open stream released, whether this turn's reply
        # arrived whole, and the models already told why.
        self._stream_chunks = 0
        self._buffered = False
        self._noted: set[str] = set()
        # Whether this turn's server sent a reply whole after a stream was
        # asked for, and whether the session was told so (once).
        self._whole = False
        self._whole_told = False
        self._backend = backend
        self._clock = clock
        # Each tool's permission tier by name (the session sets the lookup);
        # read again when a name is not known yet.
        self.tool_tiers: Callable[[], Mapping[str, str]] | None = None
        self._tiers: dict[str, str] = {}
        self.live: LiveState | None = None
        # The latest run.progress of this session, for /progress.
        self.progress: Mapping[str, Any] | None = None
        # What the session's events say about cache, context and steps, for /status.
        self.usage = UsageSnapshot()
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
        # The files this turn's file_write calls wrote, by path, in the order they
        # were first written, and the workspace the paths are shown against.
        self._written: dict[str, _Written] = {}
        self._root = ""
        # The turn's step limit, and whether its retry line has printed.
        self._max_steps = 0
        self._retry_noted = False
        # An assistant message without tool calls, until the next step or
        # the end of the run decides whether it prints (section 3).
        self._held = ""
        # The finished calls of this session, for /expand: turn number and
        # call number by turn.
        self.tool_outputs: list[ToolOutput] = []
        self._turn = 0
        self._call_number = 0
        # Where the running or last turn's calls start in tool_outputs.
        self._turn_outputs = 0

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
        # The raised band marks where a turn starts (with a fill).
        self.out.print(Band(_marked(self.symbols.prompt, "user", echo)))

    def begin_turn(
        self, text: str, *, first_turn: bool = False, echo: bool = True
    ) -> None:
        """Echo *text* (unless not *echo*) and start the live area."""
        if echo:
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
        self._written = {}
        self._stop_cause = ""
        self._max_steps = 0
        self._retry_noted = False
        self._held = ""
        self._stream_texts = {}
        self._stream_step = 0
        self._stream_open = False
        self._buffered = False
        self._whole = False
        self._turn += 1
        self._call_number = 0
        self._turn_outputs = len(self.tool_outputs)

    # ── Streamed provisional text ──

    def stream_hint(self) -> ReasoningTags | None:
        """The model's declared framing, or ``None`` for the conservative rule."""
        hint = self.reasoning_tags.get(self._backend().model)
        if hint in ("explicit", "closing_only", "none"):
            return cast(ReasoningTags, hint)
        return None

    def stream_feed(self) -> StreamFeed:
        """A display for one turn's planning streams (set around the run)."""
        return StreamFeed(self)

    def stream_step(self, number: int) -> None:
        """The agent is about to plan step *number* (in-band, not an event).

        Events reach the renderer after they are stored, so a step's first
        delta can come before its own ``step.started``, and a fast tool lets
        the next step stream before the last one's reply event. The text is
        kept per step so each event finds its own.
        """
        self._stream_step = number

    def stream_opened(self) -> None:
        """A provider stream or its complete-reply fallback started."""
        # An earlier attempt of this step that no reply claimed ended in an
        # error: close it here, as its retry event may still be on its way.
        self._flush_stream(
            "retrying", since=self._stream_step, upto=self._stream_step + 1
        )
        self._stream_open = True
        self._stream_chunks = 0
        if self.live is not None:
            self.live.stream_text = ""
            self.live.activity = "Thinking"

    def stream_held(self) -> None:
        """Text arrived but the release filter holds all of it for now.

        Nothing is drawn for it: the live row already says Thinking.
        """

    def stream_content(self, delta: str) -> None:
        """Keep and show a released, already-redacted content delta."""
        step = self._stream_step
        self._stream_chunks += 1
        self._stream_texts[step] = self._stream_texts.get(step, "") + delta
        if self.live is not None:
            self.live.stream_text = (self.live.stream_text + delta)[-STREAM_LIVE_CHARS:]
            self.live.activity = "Replying"

    def stream_whole(self) -> None:
        """The server sent the reply whole after a stream was asked for."""
        self._whole = True

    def stream_closed(self) -> None:
        self._stream_open = False
        # No hint and one released part: the filter held the reply to its end.
        if self._stream_chunks == 1 and self.stream_hint() is None:
            self._buffered = True

    def _claim_stream(self, step: object) -> bool:
        """Take the text of *step*'s reply as claimed; whether it had any.

        The completed reply is about to print (or be held), so the provisional
        copy goes. The live region is cleared only when *step* is the one now
        streaming.
        """
        key = step if isinstance(step, int) else self._stream_step
        text = self._stream_texts.pop(key, "")
        if key == self._stream_step and self.live is not None:
            self.live.stream_text = ""
        return bool(text.strip())

    def _clear_stream(self) -> None:
        """Drop all provisional text without printing it."""
        self._stream_texts.clear()
        if self.live is not None:
            self.live.stream_text = ""

    def _flush_stream(
        self,
        label: str,
        *,
        answer: str = "",
        since: int | None = None,
        upto: int | None = None,
    ) -> str:
        """Print provisional text once, its label saying how it ended.

        Prints the text of the steps from *since* up to below *upto* (all when
        ``None``) in step order and returns the last text printed, empty when there was none or
        it equals *answer*. A held assistant message with the same text is
        dropped: it is the completed copy of the same reply, not a second one.
        """
        keys = sorted(
            k
            for k in self._stream_texts
            if (since is None or k >= since) and (upto is None or k < upto)
        )
        if not keys:
            return ""
        printed = ""
        for key in keys:
            shown = clean_output(self._stream_texts.pop(key)).strip()
            if not shown:
                continue
            if self._held and _same_display_text(self._held, shown):
                self._held = ""
            if answer and _same_display_text(answer, shown):
                continue
            if self.developer:  # the text says enough outside developer mode
                self.out.print(
                    Text(
                        f"{self.symbols.queued} partial reply ({label})",
                        style="muted",
                    )
                )
            self.out.print(self._intermediate(shown))
            printed = shown
        if self.live is not None:
            self.live.stream_text = ""
        return printed

    @property
    def finished(self) -> bool:
        """Whether this turn's ``run.finished`` has come."""
        return self._finished

    def on_event(self, stored: StoredEvent) -> None:
        """One stored event of the turn (the runner's ``on_event``)."""
        event = stored.event
        self.usage.observe(event)
        if stored.turn is not None:
            self._turn = stored.turn
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
        if result is not None and result.status == "completed":
            self._clear_stream()  # the final answer prints below, once
        else:
            self._flush_stream("stopped")
        if not self._finished:
            if error is not None:
                message = str(error)
                text = (
                    f"{type(error).__name__}: {message}"
                    if message
                    else (type(error).__name__)
                )
                if isinstance(error, SessionBusyError | UnknownSessionError):
                    text = message
                else:
                    text = f"The turn could not run: {text}"
                self.out.print(
                    Text(f"{self.symbols.failed} {clean_output(text)}", style="error")
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
        self._max_steps = _int(data.get("max_steps"), 0)
        root = data.get("workspace_root")
        self._root = root if isinstance(root, str) else ""
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
        # Text of an earlier step that its reply did not claim. The event
        # follows the agent's own stream_step, so this step's text stays.
        step = data.get("step")
        self._flush_stream(
            "not the final answer", upto=step if isinstance(step, int) else None
        )
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
        block, text = split_thinking(content, reasoning_tags=self.stream_hint())
        thinking = str(data.get("thinking") or "")
        if not thinking.strip():
            thinking = block
        if thinking.strip():
            # A turn without tool calls ends in this reply: the time the
            # footer gives is this reply's time, so the line does not repeat it.
            timed = bool(calls) or self._tool_count > 0
            self._print_thinking(thinking, _float(data.get("duration_s")), timed=timed)
        text = clean_output(text).strip()
        if not isinstance(calls, list) or not calls:
            # The last message may be the answer: hold it until the end
            # says whether to print it (section 3). One with no text (only
            # thinking, #278) leaves no provisional copy behind either.
            self._held = text
            if not text:
                self._claim_stream(data.get("step"))
            return
        self._held = ""
        streamed = self._claim_stream(data.get("step"))
        if text:
            if streamed and self.developer:
                self.out.print(
                    Text(
                        f"{self.symbols.queued} partial reply (tool call)",
                        style="muted",
                    )
                )
            self.out.print(self._intermediate(text))

    def _print_thinking(
        self, text: str, duration_s: float, *, timed: bool = True
    ) -> None:
        """``✻ Thought for 1.2s`` and the thinking's first rows (section 3).

        Without *timed* the line is ``✻ Thought``.
        """
        label = f"Thought for {format_duration(duration_s)}" if timed else "Thought"
        self.out.print(Text(f"{self.symbols.thought} {label}", style="thinking"))
        if not (self.thinking or self.verbose):
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

    def _intermediate(self, text: str) -> RenderableType:
        """Text that is not the answer: a muted mark, the text in the body's ink."""
        return _marked(
            self.symbols.assistant, "intermediate", markdown(text, self.symbols)
        )

    def _print_held(self) -> None:
        """Print a held assistant message as intermediate text."""
        text, self._held = self._held, ""
        if text:
            self.out.print(self._intermediate(text))

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
        snapshot = self._take_snapshot(data)
        self._note_write(data, snapshot)
        self.out.print(
            self.tool_block(
                data, ref=f"{self._turn}.{self._call_number}", snapshot=snapshot
            )
        )
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
        exit_code = data.get("exit_code")
        self.tool_outputs.append(
            ToolOutput(
                ref=f"{self._turn}.{self._call_number}",
                name=str(data.get("name") or "?"),
                preview=self._preview_of(data),
                status=status,
                text=error if error and status != "success" else text,
                duration_s=_float(data.get("duration_s")),
                exit_code=exit_code if isinstance(exit_code, int) else None,
            )
        )

    def steps(self) -> list[StepRow]:
        """The running or last turn's calls, then the steps still to come.

        A step to come is one of the steps the task took last time that this
        run has not reached (the progress's pending items), while it runs.
        """
        rows = [
            StepRow(
                call_status(output.status, output.exit_code),
                output.name,
                output.preview,
                output.duration_s,
                self.kind(output.name),
            )
            for output in self.tool_outputs[self._turn_outputs :]
            if not output.ref.startswith("!")
        ]
        live = self.live
        if live is not None and live.tool and not live.finished:
            seconds = self._clock() - live.tool_started
            rows.append(
                StepRow(
                    "running", live.tool, live.preview, seconds, self.kind(live.tool)
                )
            )
        # Steps to come only while the turn runs: after it, none is coming.
        items = (self.progress or {}).get("items") if live is not None else None
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, Mapping) or item.get("state") != "pending":
                continue
            label = " ".join(str(item.get("label") or "").split())
            if label:
                rows.append(StepRow("next", label))
        return rows

    def _on_notice(self, data: Mapping[str, Any]) -> None:
        kind = str(data.get("kind") or "")
        message = plain(str(data.get("message") or ""))
        if kind == "retry":
            if not self._stream_open:  # else the next attempt already closed it
                self._flush_stream("retrying")
            detail = data.get("detail")
            delay = (
                _float(detail.get("delay_s")) if isinstance(detail, Mapping) else 0.0
            )
            if not self._retry_noted:  # later retries show in the live row
                self._retry_noted = True
                line = Text(
                    f"{self.symbols.retry} {self._retry_text(message, delay)}",
                    style="muted",
                )
                line.truncate(self.out.width, overflow="ellipsis")
                self.out.print(line)
            if self.live is not None:
                self.live.activity = "Retrying"
                self.live.retry_until = self._clock() + delay if delay > 0 else 0.0
            return
        if kind in ("empty_reply", "unfinished_answer"):
            line = Text(f"{self.symbols.dot} {message}", style="muted")
        else:
            line = Text(f"{self.symbols.warn} {message}", style="warn")
        line.truncate(self.out.width, overflow="ellipsis")
        self.out.print(line)

    def _retry_text(self, message: str, delay: float) -> str:
        """The retry notice's words; a server that cannot be reached keeps its URL.

        The notice's own text is ``Model request failed; retrying in 1 s: cannot
        reach the model server at URL. Start it, … (ConnectError: …)``, which a
        row of 80 cells cuts before the URL ends. The row gives up its wait
        before it gives up the URL.
        """
        url = self._backend().base_url
        if UNREACHABLE_PREFIX not in message or not url:
            return message
        wait = f"; trying again in {delay:g} s" if delay > 0 else ""
        text = f"Could not reach {url}{wait}"
        if cells(text) > self.out.width - cells(self.symbols.retry) - 1:
            return f"Could not reach {url}"
        return text

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
        _, answer = split_thinking(result.result, reasoning_tags=self.stream_hint())
        shown = ""
        if result.status == "completed":
            # The final answer prints from the result, once; streamed text was
            # only ever provisional in the live region.
            self._clear_stream()
        else:
            shown = self._flush_stream(
                "stopped" if result.status == "cancelled" else "not the final answer",
                answer=answer,
            )
        if (
            held
            and held.strip() != answer.strip()
            and not (shown and _same_display_text(held, shown))
        ):
            self.out.print(self._intermediate(held))
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
            self._print_run_end(status, error, cause)

    def clear_session(self) -> None:
        """Forget calls, progress and numbering, retaining display preferences on resume."""
        self.tool_outputs = []
        self.progress = None
        self.usage = UsageSnapshot()
        self._turn = 0
        self._call_number = 0
        self._held = ""
        self._calls = {}
        self._previews = {}
        self._arguments = {}
        self._finished = False
        self._cancelled_pending = False
        self._tool_count = 0
        self._written = {}
        self._stop_cause = ""
        self._max_steps = 0
        self._retry_noted = False

    def new_session(self) -> None:
        """Start a new conversation with the default display preferences."""
        self.clear_session()
        self.verbose = False
        self.thinking = self._default_thinking

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

    def kind(self, name: str) -> str:
        """The kind of the tool *name*: read, write, run or network, from its tier."""
        if name not in self._tiers and self.tool_tiers is not None:
            self._tiers = dict(self.tool_tiers())
        tier = self._tiers.get(name) or BUILTIN_TIERS.get(name, "write")
        return KINDS.get(tier, "write")

    def _header(
        self, symbol: str, style: str, name: str, preview: str, meta: list[str]
    ) -> Text:
        """``<symbol> <kind> <name>  <preview> · <meta> · ...``, cut to the width.

        The kind's glyph and the name take the kind's color (a read fades).
        The preview gives way to the meta, except to a long last part that
        carries free text (a denial's reason, a block's error), which is cut
        at the line's end; a fixed tag such as ``allowed by a session
        grant`` always shows whole.
        """
        width = self.out.width
        kind = self.kind(name)
        glyph = getattr(self.symbols, f"kind_{kind}")
        name = plain(name)
        dot = f" {self.symbols.dot} "
        meta = [part for part in meta if part]
        start = f"{symbol} {glyph} {name}  "
        shown, _ = escape_arguments(" ".join(preview.split("\n")))
        fixed = sum(cells(dot + part) for part in meta[:-1])
        last = cells(dot + meta[-1]) if meta else 0
        if meta and meta[-1].startswith(_FREE_TEXT_TAGS):
            last = min(last, FREE_TEXT_CELLS)
        room = width - cells(start) - fixed - last
        # The mark carries the state's color and the row stays the default
        # color; only a call that did not run is dim as a whole.
        head = Text(style="muted" if style == "muted" else "")
        head.append(symbol, style=style)
        head.append(" ")
        head.append(f"{glyph} {name}", style=f"kind_{kind}")
        if shown and room >= 4:
            head.append("  " + cut_cells(shown, room, self.symbols.ellipsis))
        for part in meta:
            head.append(dot, style="muted")
            head.append(part, style="error" if part.startswith("exit ") else "muted")
        head.no_wrap = True
        head.truncate(width, overflow="ellipsis")
        return head

    def tool_block(
        self,
        data: Mapping[str, Any],
        *,
        ref: str = "",
        snapshot: Snapshot | None = None,
    ) -> RenderableType:
        """The block of one finished tool call: its header, arguments, output, images.

        A ``file_write`` shows its change when it has the *snapshot* its approval took.
        """
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
        nonzero = isinstance(exit_code, int) and exit_code != 0
        if status == "success":
            # A command that exits nonzero ran (the event says success), but
            # its row reads as a failure: the thing to notice in a turn.
            symbol, style = (symbols.failed, "error") if nonzero else (symbols.ok, "ok")
            body = output
            if nonzero:
                meta.append(f"exit {exit_code}")
        else:
            symbol, style = symbols.failed, "error"
            body = error + ("\n" + output if output.strip() else "")
        meta.append(tag)
        parts: list[RenderableType] = [self._header(symbol, style, name, preview, meta)]
        if status == "success" and snapshot is not None:
            parts.append(self._change_body(snapshot))
            images = _image_lines(data.get("images"), self.symbols)
            if images is not None:
                parts.append(images)
            return Group(*parts)
        if self.verbose:
            arguments = _argument_lines(self._arguments_of(data), self.out.width)
            if arguments is not None:
                parts.append(arguments)
        parts.append(
            self._body(
                body,
                ref=ref,
                limit=self._output_limit(failed=nonzero or status != "success"),
                first_style="error" if nonzero or status != "success" else "",
            )
        )
        images = _image_lines(data.get("images"), self.symbols)
        if images is not None:
            parts.append(images)
        return Group(*parts)

    def _output_limit(self, *, failed: bool) -> int:
        """The output lines a call shows: a success shows one outside developer mode."""
        if self.verbose:
            return VERBOSE_BLOCK_LINES
        return BLOCK_LINES if failed or self.developer else QUIET_BLOCK_LINES

    def _take_snapshot(self, data: Mapping[str, Any]) -> Snapshot | None:
        """The file_write snapshot of this call, removed (only a write has one)."""
        if self._snapshots is None:
            return None
        approval = data.get("approval")
        if not isinstance(approval, Mapping):
            return None
        request_id = approval.get("request_id")
        if not isinstance(request_id, str):
            return None
        return self._snapshots.take(request_id)

    def _note_write(self, data: Mapping[str, Any], snapshot: Snapshot | None) -> None:
        """Remember the file a finished ``file_write`` wrote, for the line under the footer.

        Only a call whose result says it succeeded wrote one: a path the executor
        refused, a denial and a failure left the file alone. The text is what the
        call wrote, never what the folder holds now.
        """
        if data.get("name") != "file_write" or data.get("status") != "success":
            return
        arguments = self._arguments_of(data)
        given = arguments.get("path") if isinstance(arguments, Mapping) else None
        if snapshot is not None:  # its path is the one the executor checked
            path = snapshot.target.path
        elif isinstance(given, str) and given:
            path = os.path.normpath(os.path.join(self._root, given))
        else:
            return
        if snapshot is None:  # no approval read the file first: no counts
            before, after, created = None, "", False
        else:
            created = not snapshot.target.exists
            before = "" if created else snapshot.target.text
            after = snapshot.new
        known = self._written.get(path)
        if known is None:
            self._written[path] = _Written(before, after, created)
        else:  # written again: the first write's "before" stays
            known.after = after

    def _changes(self) -> list[Change]:
        """The files the turn changed, each new or modified, with its lines added and removed.

        Net of the turn: the file as the turn found it against as it left it. A file
        written back as it was is not listed. A file no approval read first has no
        known state and no counts.
        """
        changes: list[Change] = []
        for path, written in self._written.items():
            counts, state = None, ""
            if written.before is not None:
                _, added, removed = edits.diff(written.before, written.after)
                if not (added or removed or written.created):
                    continue
                counts = (added, removed) if added or removed else None
                state = "new" if written.created else "modified"
            changes.append(Change(display_path(path, self._root), counts, state))
        return changes

    def changed_line(self) -> Text | None:
        """``  Changed + hello.py (+1), ~ README.md (+3 −1)``, or ``None`` when none changed."""
        return changed_text(self._changes(), self.out.width, self.symbols)

    def _change_body(self, snapshot: Snapshot) -> Hung:
        """A file write's change: ``+1 −1 lines``, or ``new file, 12 lines``.

        With /verbose the diff's hunks follow, dim, removed lines in the
        error style and added lines in the ok style.
        """
        target = snapshot.target
        if not target.exists:
            count = len(snapshot.new.splitlines())
            body = Text(no_wrap=True, overflow="ellipsis")
            body.append("new file", style="file_new")
            body.append(f", {count} line" + ("" if count == 1 else "s"), style="muted")
            return Hung(body, self.symbols)
        if not target.diffable:
            # No text was read (too large, binary, not regular): no counts
            # are known, and the panel said why.
            return Hung(
                Text(
                    "changed; no diff shown",
                    style="muted",
                    no_wrap=True,
                    overflow="ellipsis",
                ),
                self.symbols,
            )
        rows, _, _ = edits.diff(target.text or "", snapshot.new)
        body = Text(no_wrap=True, overflow="ellipsis")
        body.append(f"+{snapshot.added}", style="ok_text")
        body.append(" ")
        body.append(f"{self.symbols.minus}{snapshot.removed}", style="error")
        body.append(" lines", style="muted")
        if self.verbose:
            width = max(1, self.out.width - 6)
            for row in rows:
                if row.kind == "…":
                    body.append("\n  " + self.symbols.ellipsis, style="muted")
                    continue
                sign = {"+": "+", "-": self.symbols.minus, " ": " "}[row.kind]
                style = {"+": "ok", "-": "error", " ": "muted"}[row.kind]
                number = row.old if row.kind == "-" else row.new
                text = clean_output(row.text)
                line = cut_cells(f"{number:4} {sign} {text}", width)
                body.append("\n  " + line, style=style)
        return Hung(body, self.symbols)

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

    def _body(
        self,
        text: str,
        *,
        ref: str = "",
        limit: int = BLOCK_LINES,
        first_style: str = "",
    ) -> Hung:
        """A call's output under its header, *limit* lines, then the rest's count.

        With one line shown, the count shares its row when the row has room:
        ``└ app.py · +4 lines (/expand 1.1)``.
        """
        lines = [
            expand_tabs(line) for line in clean_output(text).rstrip("\n").split("\n")
        ]
        if not "".join(lines).strip():
            lines = ["(no output)"]
        # A single line left prints instead of the "+1 lines" marker.
        shown = lines if len(lines) <= limit + 1 else lines[:limit]
        width = max(1, self.out.width - 4)
        marker = ""
        if len(shown) < len(lines):
            marker = f"+{len(lines) - len(shown)} lines"
            if ref:
                marker += f" (/expand {ref})"
        beside = f" {self.symbols.dot} {marker}" if marker and len(shown) == 1 else ""
        if beside and width - cells(beside) < MARKER_MIN_LINE_CELLS:
            beside = ""
        body = Text(no_wrap=True, overflow="ellipsis")
        for number, line in enumerate(shown):
            if number:
                body.append("\n")
            room = width - cells(beside)
            body.append(
                cut_cells(line, room, self.symbols.ellipsis),
                style=first_style if number == 0 else "",
            )
        if beside:
            body.append(beside, style="muted")
        elif marker:
            body.append(f"\n{self.symbols.ellipsis} {marker}", style="muted")
        return Hung(body, self.symbols)

    def _print_result(self, result: RunResult, *, footer: bool) -> None:
        symbols = self.symbols
        _, answer = split_thinking(result.result, reasoning_tags=self.stream_hint())
        answer = clean_output(answer)
        if result.status == "completed":
            body: RenderableType = (
                markdown(answer, symbols, answer=True)
                if answer
                else Text("(no answer text)", style="muted")
            )
            self.out.print(_marked(symbols.assistant, "answer", body))
        else:
            if answer:
                # Inline before a paragraph; on a line of its own before a
                # block, which "Best effort: " in front would break.
                effort: RenderableType = (
                    Group(Text("Best effort:"), markdown(answer, symbols, answer=True))
                    if _BLOCK_START.match(answer)
                    else markdown("Best effort: " + answer, symbols, answer=True)
                )
                self.out.print(_marked(symbols.assistant, "answer", effort))
            self._print_run_end(result.status, result.error)
        if footer:
            self.out.print(Text(self.footer(result), style="muted", no_wrap=True))
            changed = self.changed_line()
            if changed is not None:
                self.out.print(changed)
            model = self._backend().model
            if (
                self._buffered
                and self.developer  # a new user has no use for the reasoning format
                and result.status == "completed"
                and model not in self._noted
            ):
                self._noted.add(model)  # once per model, after its first whole reply
                self.out.print(_marked(symbols.dot, "muted", Text(STREAM_BUFFERED)))
            if self._whole and result.status == "completed" and not self._whole_told:
                self._whole_told = True  # once per session, after the first one
                self.out.print(
                    _marked(symbols.dot, "muted", Text(SERVER_WHOLE, style="muted"))
                )

    def _print_run_end(self, status: str, error: str, cause: str = "") -> None:
        """What happened to a run that did not complete, then what to do."""
        # A crash is a bug only when the run raised it; a crash cause means
        # the process stopped (section 3's severity table).
        severe = status_text(status, cause).severity == "error"
        style = "error" if severe else "warn"
        headline, action = run_end(
            status, error, self._backend(), cause, self._max_steps
        )
        mark = self.symbols.failed if severe else self.symbols.warn
        self.out.print(_marked(mark, style, Text(clean_output(headline), style=style)))
        if action:
            line = Text(clean_output(action))
            line.highlight_regex(COMMAND_PATTERN, "command")
            self.out.print(Padding(line, (0, 0, 0, 2)))

    def footer(self, result: RunResult) -> str:
        """``✻ Worked for 14s · 1 tool call · ↓ 412 tokens``, on one line.

        A run that did not complete reads ``Stopped after 14s``, and a count of
        zero tool calls is left out.
        """
        calls = result.n_tool or self._tool_count
        done = result.status == "completed"
        parts = [
            f"{self.symbols.thought} {'Worked for' if done else 'Stopped after'} "
            f"{format_duration(result.duration_s)}",
        ]
        if done or calls:
            parts.append(f"{calls} tool call" + ("" if calls == 1 else "s"))
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


class Change(NamedTuple):
    """A file the turn changed: its path, lines added and removed, and its state.

    *state* is ``new``, ``modified``, or ``""`` when it is not known.
    """

    path: str
    counts: tuple[int, int] | None = None
    state: str = ""


def _counts(counts: tuple[int, int] | None, symbols: Symbols) -> Text:
    """`` (+3 −1)``: the lines a file gained and lost; empty when they are not known."""
    added, removed = counts or (0, 0)
    parts = [(f"+{added}", "ok_text")] if added else []
    parts += [(f"{symbols.minus}{removed}", "error")] if removed else []
    text = Text()
    if parts:
        text.append(" (", style="muted")
        for number, (part, style) in enumerate(parts):
            text.append(" " if number else "", style="muted")
            text.append(part, style=style)
        text.append(")", style="muted")
    return text


# A changed file as the line draws it: its name, counts, and state mark and style.
_Entry = tuple[str, Text, tuple[str, str] | None]


def changed_text(
    changes: Sequence[Change], width: int, symbols: Symbols
) -> Text | None:
    """The line under a footer that names a turn's changed files, within *width* cells.

    ``  Changed + hello.py (+1), ~ README.md (+3 −1)``: each file's state (``+``
    new, ``~`` modified; none when not known) and its name in the state's color,
    at most :data:`CHANGED_NAMES` of them, then ``and N more``. The words around
    them are dim. A line that is too wide gives up names before it cuts one; the
    first path that still does not fit is cut in the middle, so its file name
    stays, after its counts have gone if it must. ``None`` when nothing changed.
    """
    if not changes:
        return None
    marks = {
        "new": (symbols.file_new, "file_new"),
        "modified": (symbols.file_modified, "file_mod"),
    }
    entries = [
        (plain(change.path), _counts(change.counts, symbols), marks.get(change.state))
        for change in changes
    ]

    def line(shown: Sequence[_Entry], more: int) -> Text:
        text = Text("  ", no_wrap=True, overflow="ellipsis")
        text.append("Changed ", style="muted")
        for number, (name, counts, mark) in enumerate(shown):
            if number:
                text.append(", ", style="muted")
            if mark is not None:
                text.append(f"{mark[0]} ", style=mark[1])
            text.append(name, style=mark[1] if mark else "")
            text.append_text(counts)
        if more:
            text.append(f" and {more} more", style="muted")
        return text

    for count in range(min(len(entries), CHANGED_NAMES), 0, -1):
        text = line(entries[:count], len(entries) - count)
        if cells(text.plain) <= width:
            return text
    name, counts, mark = entries[0]
    more = len(entries) - 1
    file_name = name.rpartition("/")[2]
    for kept in (counts, Text()):  # the counts go before the file name is cut
        room = width - cells(line([("", kept, mark)], more).plain)
        cut = cut_middle(name, max(1, room), symbols.ellipsis)
        if cut.endswith(file_name):
            break
    text = line([(cut, kept, mark)], more)
    text.truncate(width, overflow="ellipsis")
    return text


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
# Released provisional text keeps at most this many live rows, so the draft
# and the approval panel always fit. The live copy keeps a bounded tail; the
# full text is kept separately for the one flush.
STREAM_LIVE_LINES: Final = 8
STREAM_LIVE_CHARS: Final = 16_384


def _stream_rows(
    text: str, width: int, symbols: Symbols, limit: int = STREAM_LIVE_LINES
) -> list[list[tuple[str, str]]]:
    """The tail of released provisional text, wrapped and cut to *width*."""
    if limit <= 0:
        return []
    lines: list[str] = []
    for line in expand_tabs(clean_output(text)).split("\n"):
        if line:
            lines.extend(wrap_words(line, max(1, width)) or [""])
        else:
            lines.append("")
    while lines and not lines[0]:
        lines.pop(0)
    shown = lines[-limit:]
    rows: list[list[tuple[str, str]]] = []
    if len(lines) > len(shown):
        rows.append(
            [
                (
                    "class:muted",
                    f"{symbols.ellipsis} +{len(lines) - len(shown)} lines",
                )
            ]
        )
    for line in shown:
        rows.append([("class:muted", cut_cells(line, width, symbols.ellipsis))])
    return rows


def live_rows(
    live: LiveState,
    now: float,
    width: int,
    symbols: Symbols,
    *,
    frame: int | None = None,
    stream_lines: int = STREAM_LIVE_LINES,
    developer: bool = False,
) -> list[list[tuple[str, str]]]:
    """The live area's rows as prompt_toolkit fragments, each within *width* cells.

    *stream_lines* caps the rows of released streamed text; the app lowers it
    on a short terminal so the draft and the panels keep their rows.

    Row 1 is the spinner line: the frame, the activity, the run's facts
    from P3's :func:`format_line`, the turn's output tokens and the stop
    hint. Outside developer mode the facts are the time alone (no step, attempt,
    phase or tests). When the width is short, whole parts go: the hint, then
    the tokens, then ``format_line`` leaves out its own parts and cuts the rest
    with ``…`` (section 3). Row 2, while a tool runs, is its name, preview,
    clock and limit. Row 3, with past runs, is the progress line's past facts
    and items (section 3): on a session's first turn, and in developer mode on
    every turn. *frame* fixes the spinner's frame
    (``LOCALMELO_SPINNER_FRAME``).
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

    def facts_in(room: int) -> str:
        if not developer:
            return run_elapsed(now - live.started)
        facts = format_line(payload, None, now, room, run_started=live.started)
        # P3's line joins with its own dot; the UI's set decides the glyph.
        return facts.replace(" · ", dot)

    def compose(extras: Sequence[str]) -> str:
        suffix = dot.join(extras)
        used = cells(head) + (cells(dot + suffix) if suffix else 0)
        room = max(1, width - used - cells(dot))
        facts = facts_in(room)
        return dot.join([head, facts, suffix] if suffix else [head, facts])

    # Whole parts go when the row is short, never half a number: the hint,
    # then the tokens; then format_line drops its own parts and cuts the
    # rest with an ellipsis (section 3).
    full_facts = facts_in(10**6)

    def total(extras: Sequence[str]) -> int:
        suffix = dot.join(extras)
        used = cells(head) + cells(dot + full_facts)
        return used + (cells(dot + suffix) if suffix else 0)

    extras = [part for part in (tokens, hint) if part]
    for drop in (hint, tokens):
        if drop and drop in extras and total(extras) > width:
            extras = [part for part in extras if part != drop]
    first = cut_cells(compose(extras), width, symbols.ellipsis)
    # The spinner (or the approval's mark) in its colour, the activity in the
    # body's ink (bold while waiting), the facts muted.
    mark_cells = len(head.split(" ", 1)[0])
    rows = [
        [
            ("class:warn" if live.waiting else "class:accent", first[:mark_cells]),
            ("class:title" if live.waiting else "", first[mark_cells : len(head)]),
            ("class:muted", first[len(head) :]),
        ]
    ]
    rows[0] = [fragment for fragment in rows[0] if fragment[1]]
    if live.stream_text:
        rows.extend(_stream_rows(live.stream_text, width, symbols, stream_lines))
    if live.tool and not live.waiting:
        clock = tool_elapsed(now - live.tool_started)
        if live.tool_limit_s:
            clock += f" of {tool_elapsed(live.tool_limit_s)}"
        preview, _ = escape_arguments(" ".join(live.preview.split("\n")))
        start = f"{hung_prefix(symbols)}{plain(live.tool)}  "
        room = width - cells(start) - cells(dot + clock)
        shown = cut_cells(preview, room, symbols.ellipsis) if room >= 4 else ""
        line = cut_cells(start + shown + dot + clock, width, symbols.ellipsis)
        rows.append([("class:muted", line)])
    # The row of past runs and their steps: a session's first turn, and in
    # developer mode every turn.
    if (live.first_turn or developer) and isinstance(live.progress, Mapping):
        basis = live.progress.get("basis")
        match = basis.get("match") if isinstance(basis, Mapping) else None
        if live.progress.get("past") is not None and match != "new":
            past = format_past(live.progress, width).replace(" · ", dot)
            if past:
                rows.append([("class:muted", cut_cells(past, width, symbols.ellipsis))])
    return rows
