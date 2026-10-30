"""The terminal UI's prompt_toolkit application.

One inline ``Application`` runs for the whole session, so stdin stays in
raw mode: Esc and Ctrl+C always arrive as keys, and what is typed during a
turn becomes a draft. From the top, the app draws the live area (while a
turn runs), the queued message, the input box between two rules, the
completion menu, a hint and the status bar. What a turn prints goes into
scrollback above it (see :mod:`~localmelo.support.cli.output`).

The app acts through a :class:`Controller` (the session): Enter hands it
the draft, Esc and Ctrl+C ask it to stop. Key handlers never await.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import select
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, TextIO, TypeVar

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import CompleteEvent, Completer, ThreadedCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_focus, is_searching
from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.history import FileHistory, History
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Dimension,
    DynamicContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import (
    BufferControl,
    GetLinePrefixCallable,
    UIContent,
)
from prompt_toolkit.layout.processors import (
    Processor,
    Transformation,
    TransformationInput,
)
from prompt_toolkit.output import Output
from prompt_toolkit.search import SearchState
from prompt_toolkit.widgets import SearchToolbar
from rich.text import Text

from localmelo.support.cli.approvals import (
    BODY_LINES,
    ESCAPED_WARNING,
    REASON_MAX_CHARS,
    PendingApproval,
    TerminalApprover,
    panel_text,
    session_option,
    session_warning,
)
from localmelo.support.cli.completion import (
    AttachmentCompletion,
    CompletionList,
    SlashCompleter,
)
from localmelo.support.cli.picker import (
    ChoicePicker,
    ListPicker,
    ModelPicker,
    ModelRow,
    SessionPicker,
    fit_row,
    key_line,
)
from localmelo.support.cli.render import STREAM_LIVE_LINES, TurnRenderer, live_rows
from localmelo.support.cli.steps import STEPS_BINDING, StepsPicker
from localmelo.support.cli.text import (
    cells,
    cut_cells,
    cut_middle,
    drawn_cells,
    input_rows,
    shown,
    wrap_words,
)
from localmelo.support.cli.theme import PT_STYLE, Symbols, in_symbols, pt_color_depth
from localmelo.support.cli.usershell import parse as parse_shell
from localmelo.support.sessions.display import home_label
from localmelo.support.sessions.store import SessionInfo, SessionStore

# Seconds between redraws of the live area while the session is busy.
TICK_S: Final = 0.12
# A second Ctrl+C within this many seconds quits.
DOUBLE_PRESS_S: Final = 2.0
# How long a hint stays.
HINT_S: Final = 2.0
COMPLETION_ROWS: Final = 8
# The fewest cells a cut workspace keeps in the status bar.
MIN_PART_CELLS: Final = 4
PLACEHOLDER: Final = "Ask melo · / for commands"
# Continuation rows of the input box start this many cells in.
PREFIX_CELLS: Final = 2

HISTORY_MAX_BYTES: Final = 1_000_000
HISTORY_KEPT: Final = 1_000

SPINNER_FRAME_ENV: Final = "LOCALMELO_SPINNER_FRAME"

T = TypeVar("T")


@dataclass(frozen=True)
class StatusBar:
    """The status bar's parts: its zones' parts in order, and the approval mode."""

    model: str
    workspace: str
    context: str
    title: str
    mode: str
    mode_style: str = "class:muted"
    # The usage segments (cache, ctx, tasks): the resources zone with the
    # context; empty with [cli] status_line = "off".
    segments: tuple[str, ...] = ()
    # While a turn runs: its brief steps and the key that lists them
    # (``2/~8 · ctrl+o steps``); the run zone shows the title otherwise.
    run: str = ""


class Controller(Protocol):
    """What the app drives: the session."""

    @property
    def state(self) -> str: ...

    @property
    def queued(self) -> str | None: ...

    @property
    def activity(self) -> str | None:
        """What a busy session does when no turn runs (a switch), or None."""
        ...

    def enter(self, text: str) -> str | None:
        """Act on Enter with *text*: None when the draft was used.

        Otherwise the draft stays, with the hint returned shown (none for "").
        """
        ...

    def stop(self) -> bool: ...

    def take_queued(self) -> str | None: ...

    def clear_queued(self) -> bool: ...

    def tick(self) -> None: ...

    def status_bar(self) -> StatusBar: ...


def status_fragments(
    bar: StatusBar,
    width: int,
    ellipsis: str,
    dot: str = " · ",
    sep: str = " │ ",
) -> StyleAndTextTuples:
    """The status bar at *width*, in zones apart: who and where, the run, resources.

    The zones keep their place: the model and the folder; the run (while a
    turn runs its brief steps and the key that lists them, else the chat's
    title); the prompt cache and the context. The mode always shows, at the
    right edge. When the width is short, the title is cut first, then the
    resources go right to left, then the folder is cut (from the middle),
    then it goes, then the model is cut; a running turn's figure stays. Every
    cut counts cells, and no part is left as a bare ellipsis.
    """
    # Control characters (a pasted ^Z) count the cells they are drawn in.
    mode = cut_cells(shown(bar.mode), max(1, width - 3), ellipsis)
    # No mode (ask, which everyone expects): the room of its cells and gap is the rest's.
    room = width - 2 - (cells(mode) + 1 if mode else 0)
    model, folder = shown(bar.model), shown(bar.workspace)
    running = bool(bar.run)
    run = shown(bar.run or bar.title)
    resources = [shown(part) for part in (*bar.segments, bar.context) if part]

    def zones(workspace: str) -> list[str]:
        who = dot.join(part for part in (model, workspace) if part)
        return [zone for zone in (who, run, dot.join(resources)) if zone]

    def used(workspace: str) -> int:
        return cells(sep.join(zones(workspace)))

    workspace = folder
    if used(workspace) > room and run and not running:
        run = cut_cells(run, max(0, room - used(workspace) + cells(run)), ellipsis)
        if cells(run) < 6:
            run = ""
    while resources and used(workspace) > room:
        resources.pop()
    if used(workspace) > room:
        keep = room - used(folder) + cells(folder)
        workspace = cut_middle(folder, max(MIN_PART_CELLS, keep), ellipsis)
    if used(workspace) > room:
        workspace = ""
    # A running turn's key loses its word, then the model is cut, then the
    # key goes: the figure stays.
    shorter = [run.removesuffix(" steps"), run.partition(dot)[0]] if running else []
    if used(workspace) > room and shorter:
        run = shorter.pop(0)
    if used(workspace) > room and model:
        model = cut_cells(model, max(0, room - used("") + cells(model)), ellipsis)
        if cells(model) <= cells(ellipsis) + 1 and ellipsis in model:
            model = ""  # no room for even a cut model
    if used(workspace) > room and shorter:
        run, model = shorter.pop(0), shown(bar.model)
        model = cut_cells(model, max(0, room - used("") + cells(model)), ellipsis)
        if cells(model) <= cells(ellipsis) + 1 and ellipsis in model:
            model = ""
    if used(workspace) > room:
        run = cut_cells(run, room, ellipsis) if running else ""
        model = ""
        if cells(run) <= cells(ellipsis) + 1 and ellipsis in run:
            run = ""
    left: StyleAndTextTuples = []
    for zone, parts in (
        ("who", [("class:status_model", model), ("class:status_path", workspace)]),
        ("run", [("class:status_run" if running else "class:status", run)]),
        ("resources", [("class:status", part) for part in resources]),
    ):
        parts = [part for part in parts if part[1]]
        if not parts:
            continue
        if left:
            left.append(("class:status_sep", sep))
        for number, (style, text) in enumerate(parts):
            if number:
                left.append(("class:status", dot))
            if zone == "run" and running and dot in text:
                figure, _, rest = text.partition(dot)
                left += [(style, figure), *key_line(dot + rest, dot)]
            else:
                left.append((style, text))
    drawn = sum(cells(text) for _, text, *_ in left)
    gap = max(1 if mode else 0, width - 2 - drawn - cells(mode))
    return [
        ("class:status", "  "),
        *left,
        ("class:status", " " * gap),
        (bar.mode_style, mode),
    ]


@dataclass
class _Question:
    prompt: str
    default: str
    answer: asyncio.Future[str | None]
    draft: Document


class _Placeholder(Processor):
    """Shows *text* dim in an empty buffer."""

    def __init__(self, text: Callable[[], str]) -> None:
        self._text = text

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        if ti.lineno == 0 and not ti.document.text:
            text = self._text()
            if text:
                return Transformation([*ti.fragments, ("class:placeholder", text)])
        return Transformation(ti.fragments)


class _InputBuffer(Buffer):
    """The draft: the menu is asked again after characters are deleted.

    prompt_toolkit completes while typing only after an insert, so the menu
    of ``/ex`` was gone once the ``i`` of ``/exi`` was deleted (#277). A
    deletion that removed text starts the completer again on the text
    before the cursor; the completer offers nothing once that is no longer
    a slash command or an ``@`` path, which closes the menu. Moving the
    cursor alone opens nothing, and Esc closes the menu until the next
    change.
    """

    def delete_before_cursor(self, count: int = 1) -> str:
        return self._complete_again(super().delete_before_cursor(count))

    def delete(self, count: int = 1) -> str:
        return self._complete_again(super().delete(count))

    def _complete_again(self, deleted: str) -> str:
        if deleted and self.completer is not None and self.complete_while_typing():
            self.start_completion(complete_event=CompleteEvent())
        return deleted


class _InputControl(BufferControl):
    """The input box's control: a line's height is counted as it is drawn.

    prompt_toolkit's own count divides a line's width by the columns,
    which misses the cell a wide character leaves empty at a row's end; a
    tall draft then scrolls its cursor's rows out of the box.
    """

    def create_content(
        self, width: int, height: int, preview_search: bool = False
    ) -> UIContent:
        content = super().create_content(width, height, preview_search)
        cache: dict[tuple[int, int, int | None], int] = {}

        def height_for_line(
            lineno: int,
            width: int,
            get_line_prefix: GetLinePrefixCallable | None,
            slice_stop: int | None = None,
        ) -> int:
            key = (lineno, width, slice_stop)
            if key not in cache:
                text = fragment_list_to_text(content.get_line(lineno))[:slice_stop]
                cache[key] = line_height(text, lineno, width, get_line_prefix)
            return cache[key]

        content.get_height_for_line = height_for_line  # type: ignore[method-assign]
        return content


def line_height(
    text: str,
    lineno: int,
    width: int,
    get_line_prefix: GetLinePrefixCallable | None,
) -> int:
    """The rows *text*, line *lineno* of the input box, takes at *width*."""
    if width <= 0:
        return 10**8

    def prefix(wrap: int) -> int:
        if get_line_prefix is None:
            return 0
        return cells(
            fragment_list_to_text(to_formatted_text(get_line_prefix(lineno, wrap)))
        )

    x = prefix(0)
    rows = 1 + (max(x, 1) - 1) // width  # a prompt wider than the box wraps
    x -= (rows - 1) * width
    for char in text:
        size = drawn_cells(char)
        if x + size > width:
            x = prefix(rows)
            rows += 1
            if x >= width:
                return 10**8
        x += size
    return rows


def terminal_input(stdin: TextIO, on_close: Callable[[], None] | None = None) -> Input:
    """The app's input on *stdin* (POSIX); *on_close* runs once when it closes.

    prompt_toolkit reads EIO as an interrupted read (a resize), so after a
    hangup the input would never close; and an app whose terminal is gone
    may never end on its own. The caller learns of the close here.
    """
    from prompt_toolkit.input.posix_utils import PosixStdinReader
    from prompt_toolkit.input.vt100 import Vt100Input

    class ClosingReader(PosixStdinReader):
        def read(self, count: int = 1024) -> str:
            if self.closed:
                return ""
            try:
                if not select.select([self.stdin_fd], [], [], 0)[0]:
                    return ""
                data = os.read(self.stdin_fd, count)
            except OSError as exc:
                if exc.errno in (errno.EIO, errno.EBADF, errno.ENXIO):
                    self._close()  # the terminal is gone
                return ""
            if not data:
                self._close()
                return ""
            return self._stdin_decoder.decode(data)

        def _close(self) -> None:
            self.closed = True
            if on_close is not None:
                on_close()

    reading = Vt100Input(stdin)
    reading.stdin_reader = ClosingReader(
        reading.stdin_reader.stdin_fd, encoding=stdin.encoding or "utf-8"
    )
    return reading


def open_history(path: str) -> FileHistory:
    """The input history at *path*: private (0600), and kept under a size.

    The file is created with mode 0600 before prompt_toolkit opens it,
    because drafts can hold secrets. A file over :data:`HISTORY_MAX_BYTES`
    is rewritten first with its newest :data:`HISTORY_KEPT` entries, into a
    private file in the same folder that then replaces it.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    os.close(os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600))
    if os.path.getsize(path) > HISTORY_MAX_BYTES:
        newest = list(FileHistory(path).load_history_strings())[:HISTORY_KEPT]
        temporary = f"{path}.{os.getpid()}.tmp"
        fd = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for entry in reversed(newest):
                file.write("\n# kept\n")
                for line in entry.split("\n"):
                    file.write(f"+{line}\n")
        os.replace(temporary, path)
    return FileHistory(path)


class InteractiveApp:
    """The prompt_toolkit application of a session (see the module docstring)."""

    def __init__(
        self,
        controller: Controller,
        renderer: TurnRenderer,
        *,
        symbols: Symbols,
        history: History | None = None,
        completer: Completer | None = None,
        input: Input | None = None,
        output: Output | None = None,
        approver: TerminalApprover | None = None,
        no_color: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.controller = controller
        self._completer = completer
        if isinstance(completer, SlashCompleter):
            completer.workspace = lambda: getattr(
                getattr(controller, "agent", None), "workspace_root", None
            )
        self.renderer = renderer
        self.symbols = symbols
        self.approver = approver
        self._clock = clock
        self._hint = ""
        self._hint_until = 0.0
        self._last_ctrl_c = -DOUBLE_PRESS_S
        self._question: _Question | None = None
        self.picker: ListPicker[Any] | None = None
        self._last_rows = 0
        frame = os.environ.get(SPINNER_FRAME_ENV, "")
        self._frame = int(frame) if frame.isdigit() else None

        self.buffer = _InputBuffer(
            multiline=True,
            history=history,
            completer=None if completer is None else ThreadedCompleter(completer),
            complete_while_typing=True,
        )
        self.search = SearchToolbar()
        self.input_window = Window(
            _InputControl(
                self.buffer,
                search_buffer_control=self.search.control,
                input_processors=[_Placeholder(self._placeholder)],
                preview_search=True,
            ),
            height=self._input_height,
            wrap_lines=True,
            get_line_prefix=self._line_prefix,
            dont_extend_height=True,
        )
        rule = Window(char=symbols.rule, height=1, style="class:rule")
        self.input_box = HSplit([rule, self.input_window, self.search, rule])
        self.panel_window = Window(
            FormattedTextControl(
                self._panel_fragments, focusable=True, show_cursor=False
            ),
            dont_extend_height=True,
        )
        self.reason = Buffer(multiline=False, on_text_changed=self._cap_reason)
        self.reason_window = Window(
            BufferControl(self.reason),
            height=1,
            get_line_prefix=lambda lineno, wrap: [
                ("class:muted", self._reason_label())
            ],
        )
        self.reason_row = VSplit(
            [
                self.reason_window,
                Window(
                    FormattedTextControl(self._reason_count),
                    width=lambda: Dimension.exact(len(f"{REASON_MAX_CHARS}") * 2 + 2),
                ),
            ]
        )
        if approver is not None:
            approver.on_change = self._approval_changed
        body = HSplit(
            [
                ConditionalContainer(
                    Window(
                        FormattedTextControl(self._live_fragments),
                        dont_extend_height=True,
                    ),
                    filter=Condition(self._live_visible),
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(self._queued_fragments), height=1),
                    filter=Condition(lambda: self.controller.queued is not None),
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(self._attachments_fragments), height=1),
                    filter=Condition(
                        lambda: bool(getattr(self.controller, "attachment_line", ""))
                    ),
                ),
                *self._panels(),
                ConditionalContainer(
                    DynamicContainer(
                        lambda: (
                            self.picker.container if self.picker else Window(height=0)
                        )
                    ),
                    filter=Condition(lambda: self.picker is not None),
                ),
                ConditionalContainer(
                    self.input_box,
                    filter=Condition(
                        lambda: not self._panel_open() and self.picker is None
                    ),
                ),
                CompletionList(symbols, max_rows=COMPLETION_ROWS),
                ConditionalContainer(
                    Window(FormattedTextControl(self._hint_fragments), height=1),
                    filter=Condition(self._hint_visible),
                ),
                Window(FormattedTextControl(self._status_fragments), height=1),
            ]
        )
        self.app: Application[None] = Application(
            layout=Layout(body, focused_element=self.input_window),
            key_bindings=merge_key_bindings(
                [load_key_bindings(), self._bindings(), *self._panel_bindings()]
            ),
            style=PT_STYLE,
            input=input,
            output=output,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
            color_depth=pt_color_depth(no_color),
        )
        # Esc waits this long for a key that would make it Alt+key.
        self.app.ttimeoutlen = 0.1
        self.app.timeoutlen = 0.3
        self.app.before_render += self._before_render

    # ── Running ──

    async def run(self) -> None:
        """Run until :meth:`exit`."""
        self.app.create_background_task(self._ticker())
        await self.app.run_async(handle_sigint=True, set_exception_handler=False)

    def exit(self) -> None:
        """End the app, when it runs and has not ended."""
        if self.app.is_running and not self.app.is_done:
            self.app.exit()

    def invalidate(self) -> None:
        self.app.invalidate()

    def hint(self, text: str, seconds: float = HINT_S) -> None:
        """Show *text* under the input box for *seconds*."""
        self._hint = in_symbols(text, self.symbols)
        self._hint_until = self._clock() + seconds
        self.app.invalidate()
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().call_later(seconds + 0.05, self.app.invalidate)

    def bell(self) -> None:
        if os.environ.get("LOCALMELO_BELL") != "0":
            self.app.output.bell()

    def _width(self) -> int:
        return max(1, self.app.output.get_size().columns)

    async def choose(self, picker: ListPicker[T]) -> T | None:
        """Open *picker* over the input box; its choice, or None when it is closed."""
        self.picker = picker
        self.buffer.cancel_completion()
        self.app.layout.focus(picker.control)
        self.invalidate()
        try:
            return await picker.run()
        finally:
            self.picker = None
            # An approval that opened meanwhile keeps the keys.
            self.app.layout.focus(
                self.panel_window if self._panel_open() else self.input_window
            )
            self.invalidate()

    def open_steps(self) -> None:
        """ctrl+o: the running turn's steps, or the last turn's, as a list."""
        if self.picker is not None:
            return
        if not self.renderer.steps():
            self.hint("No steps yet: a turn's tool calls show here")
            return
        title = "Steps of the last turn"
        if self.renderer.live is not None:
            title = "Steps of this turn"
        picker = StepsPicker(
            self.renderer.steps,
            title=title,
            width=self._width,
            invalidate=self.invalidate,
            symbols=self.symbols,
        )

        async def show() -> None:
            await self.choose(picker)  # enter or esc closes it; nothing to do

        self.app.create_background_task(show())

    async def choose_session(
        self,
        store: SessionStore,
        workspace: str,
        *,
        query: str = "",
        choices: list[SessionInfo] | None = None,
    ) -> SessionInfo | None:
        return await self.choose(
            SessionPicker(
                store,
                workspace,
                query=query,
                choices=choices,
                width=self._width,
                invalidate=self.invalidate,
                symbols=self.symbols,
            )
        )

    async def choose_model(
        self,
        models: Sequence[ModelRow],
        *,
        where: str,
        current: str,
        start: str | None = "",
        title: str = "Switch model",
    ) -> str | None:
        """The model picked from *models*, with *current* marked; None when closed.

        The highlight starts on *start* (None: on no row); by default on *current*.
        """
        row = await self.choose(
            ModelPicker(
                models,
                where=where,
                current=current,
                start=current if start == "" else start,
                title=title,
                width=self._width,
                invalidate=self.invalidate,
                symbols=self.symbols,
            )
        )
        return None if row is None else row.id

    async def pick(
        self, title: str, options: Sequence[str], *, default: int, escape: int
    ) -> int:
        """Ask *title* as a list of *options*; the chosen index, *escape* on Esc."""
        row = await self.choose(
            ChoicePicker(
                title,
                options,
                default=default,
                escape=escape,
                width=self._width,
                invalidate=self.invalidate,
                symbols=self.symbols,
            )
        )
        return escape if row is None else row

    async def ask(self, prompt: str, default: str = "") -> str | None:
        """Ask *prompt* in the input box; the answer, *default* on Enter, None on Esc."""
        answer: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        question = _Question(prompt, default, answer, self.buffer.document)
        self._question = question
        self.buffer.reset()
        self.app.invalidate()
        try:
            return await answer
        finally:
            self._question = None
            self.buffer.reset(document=question.draft)
            self.app.invalidate()

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            busy = self.renderer.live is not None or self.controller.state != "idle"
            if busy:
                self.controller.tick()
            if busy or self._hint_visible():
                self.app.invalidate()

    def _before_render(self, app: Application[None]) -> None:
        """Erase the app before drawing it smaller.

        prompt_toolkit's inline renderer never draws fewer rows than it did
        while it runs, so deleting a wrapped draft, or a closed menu, would
        leave blank rows in the box.
        """
        size = app.output.get_size()
        rows = app.layout.container.preferred_height(size.columns, size.rows).preferred
        if rows < self._last_rows:
            app.renderer.erase(leave_alternate_screen=False)
        self._last_rows = rows

    # ── Parts of the layout ──

    def _prompt(self) -> str:
        if self._question is not None:
            question = self._question
            suffix = f" [{question.default}]" if question.default else ""
            return f"{question.prompt}{suffix}: "
        return f"{self.symbols.prompt} "

    def _line_prefix(self, lineno: int, wrap_count: int) -> StyleAndTextTuples:
        if lineno == 0 and wrap_count == 0:
            style = "class:question" if self._question is not None else "class:prompt"
            return [(style, self._prompt())]
        return [("", " " * PREFIX_CELLS)]

    def _placeholder(self) -> str:
        if self._question is not None:
            return ""
        # One row: cut, with an ellipsis, on a narrow terminal.
        room = self.app.output.get_size().columns - cells(self._prompt()) - 1
        return cut_cells(
            in_symbols(PLACEHOLDER, self.symbols), room, self.symbols.ellipsis
        )

    def _input_height(self) -> Dimension:
        """The draft's rows, at most what the screen has left (at least 1).

        The status bar, the rules, the live area (a turn's spinner and its
        running tool), a queued line, a hint and the search field keep
        their rows; a taller draft scrolls to the cursor.
        """
        size = self.app.output.get_size()
        rows = input_rows(
            self._shown_text(), cells(self._prompt()), PREFIX_CELLS, size.columns
        )
        others = 3 + sum(
            (
                self._live_rows(),
                self.controller.queued is not None,
                bool(getattr(self.controller, "attachment_line", "")),
                self._hint_visible(),
                self.app.layout.is_searching,
            )
        )
        rows = max(1, min(rows, size.rows - others))
        return Dimension(min=1, preferred=rows, max=rows)

    def _live_rows(self) -> int:
        if not self._live_visible():
            return 0
        text = "".join(fragment[1] for fragment in self._live_fragments())
        return text.count("\n") + 1

    def _shown_text(self) -> str:
        """The draft, or during Ctrl+R the history entry the box previews."""
        control = self.input_window.content
        assert isinstance(control, BufferControl)
        search = control.search_buffer_control
        if (
            search is None
            or not search.buffer.text
            or self.app.layout.search_target_buffer_control is not control
        ):
            return self.buffer.text
        state = control.search_state
        return self.buffer.document_for_search(
            SearchState(
                text=search.buffer.text,
                direction=state.direction,
                ignore_case=state.ignore_case,
            )
        ).text

    def _live_visible(self) -> bool:
        return self.renderer.live is not None or self.controller.activity is not None

    def _live_fragments(self) -> StyleAndTextTuples:
        width = self.app.output.get_size().columns
        now = self._clock()
        live = self.renderer.live
        if live is not None:
            if self.controller.state == "stopping" and not live.finished:
                live.activity = "Stopping"
            rows = live_rows(
                live,
                now,
                width,
                self.symbols,
                frame=self._frame,
                # Leave a short terminal its draft, rules and status bar.
                stream_lines=min(
                    STREAM_LIVE_LINES, max(0, self.app.output.get_size().rows - 14)
                ),
                developer=self.renderer.developer,
            )
        else:
            activity = self.controller.activity or ""
            index = self._frame if self._frame is not None else int(now / TICK_S)
            mark = self.symbols.spinner[index % len(self.symbols.spinner)]
            text = cut_cells(
                f"{mark} {activity}{self.symbols.ellipsis}",
                width,
                self.symbols.ellipsis,
            )
            rows = [[("class:accent", text[: len(mark)]), ("", text[len(mark) :])]]
        fragments: StyleAndTextTuples = []
        for number, row in enumerate(rows):
            if number:
                fragments.append(("", "\n"))
            fragments.extend(row)
        return fragments

    def _queued_fragments(self) -> StyleAndTextTuples:
        queued = self.controller.queued or ""
        first = queued.split("\n", 1)[0]
        width = self.app.output.get_size().columns
        text = f"{self.symbols.queued} Queued for the next turn: {shown(first)}"
        return [("class:muted", cut_cells(text, width, self.symbols.ellipsis))]

    def _attachments_fragments(self) -> StyleAndTextTuples:
        line = cut_cells(
            "  " + shown(str(getattr(self.controller, "attachment_line", ""))),
            self.app.output.get_size().columns,
            self.symbols.ellipsis,
        )
        label, separator, names = line.partition(": ")
        return [("class:muted", label + separator), ("", names)]

    def _hint_visible(self) -> bool:
        return bool(self._hint) and self._clock() < self._hint_until

    def _hint_fragments(self) -> StyleAndTextTuples:
        width = self.app.output.get_size().columns
        return [
            (
                "class:hint",
                cut_cells("  " + shown(self._hint), width, self.symbols.ellipsis),
            )
        ]

    def _status_fragments(self) -> StyleAndTextTuples:
        width = self.app.output.get_size().columns
        return status_fragments(
            self.controller.status_bar(),
            width,
            self.symbols.ellipsis,
            f" {self.symbols.dot} ",
            f" {self.symbols.gutter} ",
        )

    # ── The approval panel ──

    def _pending(self) -> PendingApproval | None:
        return None if self.approver is None else self.approver.pending

    def _panel_open(self) -> bool:
        """Whether the approval panel replaces the input box."""
        return self._pending() is not None

    def _panels(self) -> list[ConditionalContainer]:
        reason_open = Condition(
            lambda: (pending := self._pending()) is not None and pending.reason
        )
        return [
            ConditionalContainer(
                HSplit(
                    [
                        self.panel_window,
                        ConditionalContainer(self.reason_row, filter=reason_open),
                    ]
                ),
                filter=Condition(self._panel_open),
            )
        ]

    def _approval_changed(self) -> None:
        """The panel opened or closed: move the focus, freeze the spinner."""
        pending = self._pending()
        live = self.renderer.live
        if live is not None:
            live.waiting = pending is not None
        if pending is not None:
            self.reason.reset()
            if isinstance(self.picker, StepsPicker):
                self.picker.close()  # the question comes first
            with contextlib.suppress(ValueError):
                self.app.layout.focus(self.panel_window)
            self.bell()
        else:
            with contextlib.suppress(ValueError):
                self.app.layout.focus(self.input_window)
        self.app.invalidate()

    def _options(self, pending: PendingApproval) -> list[tuple[int, str]]:
        options = [(0, "Yes")]
        if pending.offer_session:
            options.append((1, session_option(pending.request)))
        options.append((2, "No, and tell melo why"))
        return options

    def _panel_fragments(self) -> StyleAndTextTuples:
        """The approval panel: every row wrapped, never cut, and fit to the screen.

        What will run shows whole. On a terminal too short for the whole
        panel, fewer body rows show, with ``… +n lines (v views all)``; then
        the box goes, then the key line, so the options always show.
        """
        pending = self._pending()
        if pending is None:
            return []
        size = self.app.output.get_size()
        width = max(10, size.columns)
        symbols = self.symbols

        def wrapped(
            style: str, line: str, room: int, indent: str = ""
        ) -> list[tuple[str, str]]:
            # Prose: a row that wraps does not start with a space.
            first, *rest = wrap_words(line, room)
            rest = wrap_words("".join(rest).lstrip(" "), room - cells(indent))
            return [(style, first)] + [(style, indent + part) for part in rest if part]

        options: list[tuple[str, str]] = []
        for index, label in self._options(pending):
            chosen = index == pending.selected and not pending.reason
            mark = symbols.selection if chosen else " "
            # The chosen answer is a band; while a reason is typed, its answer
            # keeps the accent.
            style = "class:accent" if index == pending.selected else ""
            style = "class:chosen" if chosen else style
            options += wrapped(style, f"{mark} {index + 1}. {label}", width, "     ")
        keys: list[tuple[str, str]] = []
        if not pending.reason:
            if pending.offer_session:
                where = home_label(pending.request.workspace_root or "")
                for line in session_warning(pending.request.tool_name, where):
                    options += wrapped("", "  " + line, width, "  ")
            dot = f" {symbols.dot} "
            line = dot.join(
                [
                    f"{symbols.arrows} select",
                    "enter confirm",
                    "1-3",
                    "v view all",
                    "esc stop the turn",
                ]
            )
            keys = wrapped("keys", "  " + line, width, "  ")
        # The rows the rest of the screen keeps: the status bar, the live
        # area, a queued line, a hint and the reason field.
        rows = size.rows - 1
        rows -= sum(
            (
                self._live_rows(),
                self.controller.queued is not None,
                bool(getattr(self.controller, "attachment_line", "")),
                self._hint_visible(),
                pending.reason,
            )
        )
        boxed = width >= 40 and symbols.rule == "─"
        while True:
            inner = width - 4 if boxed else width
            text = panel_text(
                pending.request,
                target=pending.target,
                width=inner - 2,
                minus=symbols.minus,
                show_tier=self.renderer.developer,
            )
            # The header holds a path: its rows keep every character.
            header = text.header.replace(" · ", f" {symbols.dot} ")
            head = [("class:tool", part) for part in wrap_words(header, inner)]
            tail = (
                wrapped("class:warn", f"{symbols.warn} {ESCAPED_WARNING}", inner)
                if text.escaped
                else []
            )
            fixed = (2 if boxed else 1) + len(head) + len(tail) + len(options)
            room = rows - fixed - len(keys)
            if room >= min(2, len(text.body)) or not (boxed or keys):
                break
            if boxed:
                boxed = False
            else:
                keys = []
        limit = max(1, min(BODY_LINES, room))
        body = [
            (f"class:{style}" if style else "", "  " + line)
            for style, line in text.body
        ]
        if len(body) > limit:
            hidden = len(body) - (limit - 1)
            body = body[: limit - 1] + [
                ("class:muted", f"{symbols.ellipsis} +{hidden} lines (v views all)")
            ]
        out: StyleAndTextTuples = []
        # A diff's added and removed rows are bands across the panel.
        bands = {"class:ok": "class:diff_add", "class:error": "class:diff_remove"}

        def row(style: str, line: str, room: int = width) -> None:
            if style in bands:
                out.extend(fit_row([("", line)], room, bands[style]))
            elif style == "class:chosen":
                mark = (
                    len(symbols.selection) if line.startswith(symbols.selection) else 0
                )
                parts = [("class:accent", line[:mark]), ("", line[mark:])]
                out.extend(fit_row(parts, room, style))
            elif style == "keys":
                out.extend(key_line(line, f" {symbols.dot} "))
            else:
                out.append((style, line))
            if room == width:
                out.append(("", "\n"))

        if boxed:
            # The frame in the warn colour, its title bold.
            label = cut_cells(text.title, width - 5)
            out.append(("class:frame", "╭─ "))
            out.append(("class:title", label))
            dashes = "─" * max(0, width - 5 - cells(label))
            row("class:frame", f" {dashes}╮")
            for style, line in head + body + tail:
                shown = cut_cells(line, inner, symbols.ellipsis)
                out.append(("class:frame", "│ "))
                row(style, shown, inner)
                if style not in bands:
                    out.append(("", " " * (inner - cells(shown))))
                row("class:frame", " │")
            row("class:frame", "╰" + "─" * (width - 2) + "╯")
        else:
            row("class:question", cut_cells(text.title, width, symbols.ellipsis))
            for style, line in head + body + tail:
                row(style, line)
        for style, line in options + keys:
            row(style, line)
        if out:
            out.pop()  # no newline after the last row
        return out

    def _reason_label(self) -> str:
        # The long label leaves room for the reason from 60 columns on.
        if self.app.output.get_size().columns >= 60:
            return "  Reason (optional, enter sends): "
        return "  Reason: "

    def _reason_count(self) -> StyleAndTextTuples:
        return [("class:muted", f" {len(self.reason.text)}/{REASON_MAX_CHARS}")]

    def _cap_reason(self, buffer: Buffer) -> None:
        if len(buffer.text) > REASON_MAX_CHARS:
            buffer.document = Document(buffer.text[:REASON_MAX_CHARS])

    def _move(self, step: int) -> None:
        pending = self._pending()
        if pending is None or self.approver is None or self.approver.guarded():
            return
        indexes = [index for index, _ in self._options(pending)]
        position = indexes.index(pending.selected) if pending.selected in indexes else 0
        pending.selected = indexes[(position + step) % len(indexes)]
        self.app.invalidate()

    def _choose(self, index: int) -> None:
        pending = self._pending()
        if pending is None or self.approver is None or self.approver.guarded():
            return
        if index == 0:
            self.approver.answer("once")
        elif index == 1:
            self.approver.answer("session")
        elif index == 2:
            pending.selected = 2
            pending.reason = True
            self.app.layout.focus(self.reason_window)
            self.app.invalidate()

    def _back_to_choices(self) -> None:
        pending = self._pending()
        if pending is not None:
            pending.reason = False
            self.reason.reset()
            self.app.layout.focus(self.panel_window)
            self.app.invalidate()

    def _view_all(self) -> None:
        pending = self._pending()
        if pending is None or self.approver is None or self.approver.guarded():
            return
        text = panel_text(
            pending.request,
            target=pending.target,
            width=self.renderer.out.width - 2,
            minus=self.symbols.minus,
            show_tier=self.renderer.developer,
        )
        header = text.header.replace(" · ", f" {self.symbols.dot} ")
        self.renderer.out.print(
            Text("\n".join([header, *(line for _, line in text.body)]))
        )

    def _panel_bindings(self) -> list[KeyBindings]:
        kb = KeyBindings()
        choices = has_focus(self.panel_window)
        reason = has_focus(self.reason)

        for number in (1, 2, 3):

            @kb.add(str(number), filter=choices)
            def _(event: KeyPressEvent, index: int = number - 1) -> None:
                self._choose(index)

        @kb.add("up", filter=choices)
        def _(event: KeyPressEvent) -> None:
            self._move(-1)

        @kb.add("down", filter=choices)
        def _(event: KeyPressEvent) -> None:
            self._move(1)

        @kb.add("enter", filter=choices)
        def _(event: KeyPressEvent) -> None:
            pending = self._pending()
            if pending is not None:
                self._choose(pending.selected)

        @kb.add("v", filter=choices)
        def _(event: KeyPressEvent) -> None:
            self._view_all()

        @kb.add("escape", filter=choices, eager=True)
        def _(event: KeyPressEvent) -> None:
            self.controller.stop()
            self.app.invalidate()

        @kb.add("c-c", filter=choices | reason)
        def _(event: KeyPressEvent) -> None:
            self.ctrl_c()

        @kb.add("enter", filter=reason)
        def _(event: KeyPressEvent) -> None:
            if self.approver is not None:
                self.approver.answer("deny", self.reason.text)

        @kb.add("escape", filter=reason, eager=True)
        def _(event: KeyPressEvent) -> None:
            self._back_to_choices()

        return [kb]

    # ── Keys ──

    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()
        in_input = has_focus(self.buffer) & ~is_searching
        empty = Condition(lambda: not self.buffer.text)
        queued = Condition(lambda: self.controller.queued is not None)

        @kb.add("enter", filter=in_input)
        def _(event: KeyPressEvent) -> None:
            self._enter()

        @kb.add("escape", "enter", filter=in_input)
        @kb.add("c-j", filter=in_input)
        def _(event: KeyPressEvent) -> None:
            self.buffer.insert_text("\n")

        @kb.add("escape", filter=in_input)
        def _(event: KeyPressEvent) -> None:
            self._escape()

        @kb.add("c-c", filter=in_input)
        @kb.add("<sigint>")
        def _(event: KeyPressEvent) -> None:
            self.ctrl_c()

        @kb.add("c-d", filter=in_input & empty)
        def _(event: KeyPressEvent) -> None:
            self._ctrl_d()

        @kb.add("tab", filter=in_input)
        def _(event: KeyPressEvent) -> None:
            self._tab()

        # Over prompt_toolkit's emacs "operate and get next", which melo never used.
        @kb.add(
            STEPS_BINDING, filter=in_input & ~Condition(lambda: bool(self._question))
        )
        def _(event: KeyPressEvent) -> None:
            self.open_steps()

        # Not bound in this version (no suspend): Ctrl+Z inserts nothing.
        @kb.add("c-z", filter=in_input)
        def _(event: KeyPressEvent) -> None:
            pass

        @kb.add("up", filter=in_input & empty & queued)
        def _(event: KeyPressEvent) -> None:
            text = self.controller.take_queued()
            if text is not None:
                self.buffer.document = Document(text, len(text))

        # A question's answer is one line: the arrows do not bring the
        # message history into it.
        asking = Condition(lambda: self._question is not None)

        @kb.add("up", filter=in_input & asking)
        @kb.add("down", filter=in_input & asking)
        @kb.add("c-p", filter=in_input & asking)
        @kb.add("c-n", filter=in_input & asking)
        def _(event: KeyPressEvent) -> None:
            pass

        return kb

    def _tab(self) -> None:
        """Tab: the next completion; before the menu shows, complete at once.

        The menu's completions come from a thread while typing, and a Tab
        pressed before they arrive was lost: the completer is asked here.
        """
        buffer = self.buffer
        state = buffer.complete_state
        if state is not None and state.completions:
            buffer.complete_next()
            return
        if self._completer is None:
            return
        found = list(
            self._completer.get_completions(
                buffer.document, CompleteEvent(completion_requested=True)
            )
        )
        if len(found) == 1 and not isinstance(found[0], AttachmentCompletion):
            buffer.apply_completion(found[0])
        elif found:
            buffer.complete_state = CompletionState(buffer.document, found)
            buffer.go_to_completion(0)

    def _enter(self) -> None:
        buffer = self.buffer
        before = buffer.document.text_before_cursor
        if before.endswith("\\") and self._question is None:
            buffer.delete_before_cursor(1)
            buffer.insert_text("\n")
            return
        state = buffer.complete_state
        if state is not None and state.current_completion is not None:
            selected = state.current_completion
            if isinstance(selected, AttachmentCompletion) and self._question is None:
                buffer.complete_state = None
                attach = getattr(self.controller, "attach_path", None)
                if attach is not None:
                    hint = attach(selected.path)
                    if hint:
                        self.hint(hint)
                return
            # Tab or the arrows picked one: the draft holds it. (Cancelling
            # the completion would put back what was typed.)
            buffer.complete_state = None
        elif (
            state is not None
            and state.completions
            and buffer.text.startswith("/")
            and " " not in buffer.text
        ):
            buffer.apply_completion(state.completions[0])
        buffer.cancel_completion()
        if self._question is not None:
            question = self._question
            answer = buffer.text.strip() or question.default
            if not question.answer.done():
                question.answer.set_result(answer)
            return
        text = buffer.text
        if not text.strip():
            return
        hint = self.controller.enter(text)
        if hint is not None:  # the draft stays; "" shows no hint
            if hint:
                self.hint(hint)
            return
        # A shell command can hold a secret: it is not saved in the history.
        shell = parse_shell(text)
        buffer.reset(append_to_history=shell is None or shell.kind == "literal")

    def _escape(self) -> None:
        """Esc: close the menu, leave a question, else stop what runs."""
        if self.buffer.complete_state is not None:
            self.buffer.cancel_completion()
            return
        if self._question is not None:
            if not self._question.answer.done():
                self._question.answer.set_result(None)
            return
        if self.controller.state in ("running", "switching"):
            self.controller.stop()
            self.app.invalidate()

    def ctrl_c(self) -> None:
        """Ctrl+C: close what is open, clear the draft or the queue, stop, quit.

        Only a press that stopped something or said "press ctrl+c again"
        arms the second press, so clearing a draft never quits.
        """
        if self.picker is not None:
            self.picker.close()
            return
        now = self._clock()
        double = now - self._last_ctrl_c < DOUBLE_PRESS_S
        pending = self._pending()
        if pending is not None:
            if pending.reason and self.reason.text:
                self.reason.reset()
            elif pending.reason:
                self._back_to_choices()
            elif double:
                self.exit()
            elif self.controller.stop():
                self._last_ctrl_c = now
                self.hint("Stopping… press ctrl+c again to quit")
            return
        self._last_ctrl_c = -DOUBLE_PRESS_S
        buffer = self.buffer
        if buffer.complete_state is not None:
            buffer.cancel_completion()
            return
        if self._question is not None:
            if not self._question.answer.done():
                self._question.answer.set_result(None)
            return
        if buffer.text:
            buffer.reset()
            self.hint("Draft cleared")
            return
        if self.controller.clear_queued():
            self.hint("Queued message cleared")
            return
        state = self.controller.state
        if state in ("running", "switching"):
            # Not while the turn wraps up: the session says so itself.
            if self.controller.stop():
                self._last_ctrl_c = now
                self.hint("Stopping… press ctrl+c again to quit")
            return
        if double:
            self.exit()
            return
        self._last_ctrl_c = now
        self.hint(
            "Stopping… press ctrl+c again to quit"
            if state == "stopping"
            else "Press ctrl+c again to quit"
        )

    def _ctrl_d(self) -> None:
        if self._question is not None:
            return
        if self.controller.state == "idle":
            self.exit()
        else:
            self.hint("A turn is running: esc stops it, /exit quits")
