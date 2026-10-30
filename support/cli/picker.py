"""Inline pickers: a filter field, a few rows, ↑↓, enter and esc.

:class:`ListPicker` is the one list. :class:`SessionPicker` fills it with chat
sessions (``/sessions``, ``-r``) and :class:`ModelPicker` with the models of a
server (``/model``, and the first run's question, which :func:`run_inline` asks
before melo's own UI exists).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import (
    BufferControl,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.output import ColorDepth

from localmelo.support.cli.text import cells, cut_cells
from localmelo.support.cli.theme import PT_STYLE, Symbols, in_symbols
from localmelo.support.sessions.display import clean_output, home_label, session_age
from localmelo.support.sessions.store import SessionInfo, SessionStore

T = TypeVar("T")

# Rows shown at once; the list scrolls to keep the highlighted one in view.
VISIBLE_ROWS: Final = 8


class ListPicker(Generic[T]):
    """A list to choose from: type to filter, ↑↓ to move, enter to choose, esc to close.

    A subclass says where the rows come from (:meth:`load`) and how the header,
    a row and the footer read. :meth:`run` answers the chosen row, or None
    when the picker is closed. ``selected`` is the highlighted row; -1 means
    none yet, and enter does nothing until a key moves it.
    """

    # How long typing waits before the rows are loaded again.
    debounce_s = 0.15

    def __init__(
        self,
        *,
        width: Callable[[], int],
        invalidate: Callable[[], None],
        symbols: Symbols,
        query: str = "",
    ) -> None:
        self.width, self.invalidate, self.symbols = width, invalidate, symbols
        self.rows: list[T] = []
        self.selected = 0
        self.loading = True
        self.answer: asyncio.Future[T | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._load: asyncio.Task[None] | None = None
        self._first = True
        self.buffer = Buffer(
            document=Document(query, len(query)),
            multiline=False,
            on_text_changed=lambda _: self.refresh(),
        )
        keys = KeyBindings()
        for key in ("escape", "c-c", "c-d"):
            keys.add(key)(lambda event: self.close())
        keys.add("enter")(self._enter)
        keys.add("tab")(lambda event: self.on_tab())
        keys.add("up")(lambda event: self.move(-1))
        keys.add("down")(lambda event: self.move(1))
        self.control = BufferControl(self.buffer, key_bindings=keys)
        self.container = HSplit(
            [
                Window(FormattedTextControl(self._header), height=1),
                Window(
                    self.control,
                    height=1,
                    get_line_prefix=lambda line, wrap: f" Filter {symbols.prompt} ",
                ),
                Window(FormattedTextControl(self._fragments), dont_extend_height=True),
            ]
        )

    # ── What a subclass gives ──

    async def load(self, query: str) -> list[T]:
        """The rows for the filter text *query*."""
        raise NotImplementedError

    def _header(self) -> str:
        raise NotImplementedError

    def _row(self, item: T) -> str:
        """One row, after its mark."""
        raise NotImplementedError

    def _footer(self) -> str:
        raise NotImplementedError

    def _start(self, rows: list[T]) -> int:
        """The row highlighted when the first rows arrive."""
        return 0

    def on_tab(self) -> None:
        """Tab: nothing, unless the list has a scope to switch."""

    # ── Running ──

    async def run(self) -> T | None:
        self.refresh(delay=0)
        try:
            return await self.answer
        finally:
            if self._load is not None:
                self._load.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._load

    def close(self) -> None:
        if not self.answer.done():
            self.answer.set_result(None)

    def move(self, step: int) -> None:
        self.selected = max(0, min(len(self.rows) - 1, self.selected + step))
        self.invalidate()

    def _enter(self, event: KeyPressEvent) -> None:
        if (
            not self.loading
            and 0 <= self.selected < len(self.rows)
            and not self.answer.done()
        ):
            self.answer.set_result(self.rows[self.selected])

    def refresh(self, delay: float | None = None) -> None:
        if self._load is not None:
            self._load.cancel()
        self.loading = True
        wait = self.debounce_s if delay is None else delay
        self._load = asyncio.create_task(self._query(self.buffer.text, wait))
        self.invalidate()

    async def _query(self, query: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            rows = await self.load(query)
            # A provider that finishes despite cancellation cannot replace a newer query.
            if asyncio.current_task() is not self._load:
                return
            self.rows = rows
            self.selected = self._start(rows) if self._first else 0
            self._first, self.loading = False, False
            self.invalidate()
        except Exception as exc:
            if asyncio.current_task() is self._load and not self.answer.done():
                self.answer.set_exception(exc)

    # ── Drawing ──

    def _lines(self) -> list[tuple[str, str]]:
        """The visible rows and the footer as (style, text), cut to the width."""
        first = max(0, self.selected - (VISIBLE_ROWS - 1))
        lines: list[tuple[str, str]] = []
        for index, item in enumerate(self.rows[first : first + VISIBLE_ROWS], first):
            chosen = index == self.selected
            mark = "❯" if chosen else " "
            lines.append(
                ("class:accent" if chosen else "", f" {mark} {self._row(item)}")
            )
        lines.append(("", " " + self._footer()))
        return [
            (
                style,
                cut_cells(clean_output(in_symbols(text, self.symbols)), self.width()),
            )
            for style, text in lines
        ]

    def _rows(self) -> str:
        return "\n".join(text for _, text in self._lines())

    def _fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for number, (style, text) in enumerate(self._lines()):
            fragments.append((style, ("\n" if number else "") + text))
        return fragments


class SessionPicker(ListPicker[SessionInfo]):
    """Query at most 50 eligible chat sessions, debounced by 150 ms."""

    def __init__(
        self,
        store: SessionStore,
        workspace: str,
        *,
        width: Callable[[], int],
        invalidate: Callable[[], None],
        symbols: Symbols,
        query: str = "",
        choices: Sequence[SessionInfo] | None = None,
    ) -> None:
        self.store, self.workspace = store, workspace
        self.choices = choices
        self.all_folders = choices is not None
        super().__init__(
            width=width, invalidate=invalidate, symbols=symbols, query=query
        )

    def on_tab(self) -> None:
        self.all_folders = not self.all_folders
        self.choices = None
        self.refresh(delay=0)

    async def load(self, query: str) -> list[SessionInfo]:
        if self.choices is None:
            rows = await self.store.list_sessions(
                limit=50,
                query=query,
                min_turns=1,
                workspace=None if self.all_folders else self.workspace,
            )
        else:
            rows = [
                s
                for s in self.choices
                if query.casefold() in f"{s.title} {s.id}".casefold()
            ][:50]
        return [s for s in rows if s.kind == "chat" and s.turns >= 1]

    def _header(self) -> str:
        scope = (
            "matching sessions"
            if self.choices is not None
            else "all folders" if self.all_folders else home_label(self.workspace)
        )
        return cut_cells(
            clean_output(
                in_symbols(
                    f" Resume a session · {scope} · tab scope · esc close", self.symbols
                )
            ),
            self.width(),
        )

    def _row(self, info: SessionInfo) -> str:
        where = (
            " · "
            + cut_cells(
                " ".join(clean_output(home_label(info.workspace or "")).split()),
                max(6, self.width() // 5),
            )
            if self.all_folders
            else ""
        )
        ref = "running elsewhere" if info.busy else info.id[:12]
        suffix = f"{where} · {info.turns} turn{'s' if info.turns != 1 else ''} · {session_age(info.updated_at, time.time())} · {ref}"
        title = " ".join(clean_output(info.title).split())
        title = cut_cells(title, max(1, self.width() - cells(suffix) - 3))
        return f"{title}{suffix}"

    def _footer(self) -> str:
        if self.loading:
            return "Loading…"
        if not self.rows:
            return "No sessions yet."
        return f"{len(self.rows)} session{'s' if len(self.rows) != 1 else ''}"


@dataclass(frozen=True)
class ModelRow:
    """A model a server lists: its id, whether it is in memory, and what it is for."""

    id: str
    loaded: bool = False
    kind: str = "chat"  # chat, embedding or reranking


class ModelPicker(ListPicker[ModelRow]):
    """The models of a server, filtered by what is typed.

    *current* is marked ``✓`` and is the model that esc keeps. *start* is the
    id the highlight starts on (the first row if it is not listed); None
    highlights nothing, so enter chooses nothing until ↑↓ moves to a row.
    """

    debounce_s = 0.0

    def __init__(
        self,
        models: Sequence[ModelRow],
        *,
        where: str,
        current: str = "",
        start: str | None = "",
        title: str = "Switch model",
        width: Callable[[], int],
        invalidate: Callable[[], None],
        symbols: Symbols,
    ) -> None:
        self.models = list(models)
        self.where, self.current, self.start, self.title = where, current, start, title
        super().__init__(width=width, invalidate=invalidate, symbols=symbols)

    async def load(self, query: str) -> list[ModelRow]:
        needle = query.strip().casefold()
        return [m for m in self.models if needle in m.id.casefold()]

    def _start(self, rows: list[ModelRow]) -> int:
        if self.start is None:
            return -1
        return next((i for i, m in enumerate(rows) if m.id == self.start), 0)

    def _header(self) -> str:
        text = clean_output(in_symbols(f" {self.title} · {self.where}", self.symbols))
        return cut_cells(text, self.width(), self.symbols.ellipsis)

    def _row(self, model: ModelRow) -> str:
        tags = [] if model.kind == "chat" else [model.kind]
        tags += ["loaded"] if model.loaded else []
        suffix = f"  ({', '.join(tags)})" if tags else ""
        mark = "✓" if model.id == self.current else " "
        name = " ".join(clean_output(model.id).split())
        return (
            f"{mark} "
            + cut_cells(name, max(1, self.width() - cells(suffix) - 5))
            + suffix
        )

    def _footer(self) -> str:
        if self.loading:
            return "Loading…"
        if not self.rows:
            return f"No model matches '{self.buffer.text.strip()}'."
        count = f"{len(self.rows)} model{'s' if len(self.rows) != 1 else ''}"
        keeps = f"esc keeps {self.current}" if self.current else "esc cancels"
        arrows = self.symbols.arrows
        if self.selected < 0:
            options = [
                f"{count} · {arrows} choose one, then enter · {keeps}",
                f"{count} · {arrows} then enter",
            ]
        elif self.current:
            options = [
                f"{count} · enter switches and saves · {keeps}",
                f"{count} · enter switches · esc cancels",
            ]
        else:
            options = [f"{count} · enter chooses · esc cancels"]
        room = self.width() - 1
        return next((o for o in options if cells(o) <= room), options[-1])


def run_inline(
    build: Callable[[Callable[[], int], Callable[[], None]], ListPicker[T]],
    *,
    no_color: bool = False,
) -> T | None:
    """Ask with a picker of its own, drawn below the cursor and erased when answered.

    For a question that comes before melo's UI exists (the first run).
    *build* is given the picker's width and invalidate callbacks and makes
    the picker; the answer is the chosen row, or None when it was closed.
    """
    from prompt_toolkit.output.vt100 import Vt100_Output

    output = Vt100_Output.from_pty(sys.stdout, term=os.environ.get("TERM"))
    output.enable_cpr = os.environ.get("LOCALMELO_CPR") == "1"
    holder: list[Application[T | None]] = []

    def invalidate() -> None:
        if holder:
            holder[0].invalidate()

    async def ask() -> T | None:
        picker = build(lambda: max(1, output.get_size().columns), invalidate)
        app: Application[T | None] = Application(
            layout=Layout(picker.container, focused_element=picker.control),
            style=PT_STYLE,
            output=output,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
            color_depth=ColorDepth.DEPTH_1_BIT if no_color else None,
        )
        holder.append(app)

        async def answer() -> None:
            try:
                result = await picker.run()
            except Exception as exc:
                app.exit(exception=exc)
            else:
                app.exit(result=result)

        app.create_background_task(answer())
        return await app.run_async(handle_sigint=False, set_exception_handler=False)

    return asyncio.run(ask())
