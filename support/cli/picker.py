"""The inline session picker: a separate filter buffer, scope and bounded rows."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Sequence

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import BufferControl, FormattedTextControl, HSplit, Window

from localmelo.support.cli.text import cells, cut_cells
from localmelo.support.cli.theme import Symbols, in_symbols
from localmelo.support.sessions.display import clean_output, home_label, session_age
from localmelo.support.sessions.store import SessionInfo, SessionStore


class SessionPicker:
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
        self.width, self.invalidate, self.symbols = width, invalidate, symbols
        self.choices = choices
        self.all_folders = choices is not None
        self.rows: list[SessionInfo] = []
        self.selected = 0
        self.loading = True
        self.answer: asyncio.Future[SessionInfo | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._load: asyncio.Task[None] | None = None
        self.buffer = Buffer(
            document=Document(query, len(query)),
            multiline=False,
            on_text_changed=lambda _: self.refresh(),
        )
        keys = KeyBindings()
        for key in ("escape", "c-c", "c-d"):
            keys.add(key)(lambda event: self.close())
        keys.add("enter")(self._enter)
        keys.add("tab")(self._scope)
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
                Window(FormattedTextControl(self._rows), dont_extend_height=True),
            ]
        )

    async def run(self) -> SessionInfo | None:
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
        if not self.loading and self.rows and not self.answer.done():
            self.answer.set_result(self.rows[self.selected])

    def _scope(self, event: KeyPressEvent) -> None:
        self.all_folders = not self.all_folders
        self.choices = None
        self.refresh(delay=0)

    def refresh(self, delay: float = 0.15) -> None:
        if self._load is not None:
            self._load.cancel()
        self.loading = True
        self._load = asyncio.create_task(self._query(self.buffer.text, delay))
        self.invalidate()

    async def _query(self, query: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
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
            # A provider that finishes despite cancellation cannot replace a newer query.
            if asyncio.current_task() is not self._load:
                return
            self.rows = [s for s in rows if s.kind == "chat" and s.turns >= 1]
            self.selected, self.loading = 0, False
            self.invalidate()
        except Exception as exc:
            if asyncio.current_task() is self._load and not self.answer.done():
                self.answer.set_exception(exc)

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

    def _rows(self) -> str:
        lines: list[str] = []
        first = max(0, self.selected - 7)
        for index, info in enumerate(self.rows[first : first + 8], first):
            mark = "❯" if index == self.selected else " "
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
            lines.append(f" {mark} {title}{suffix}")
        lines.append(
            " Loading…"
            if self.loading
            else (
                f" {len(self.rows)} session{'s' if len(self.rows) != 1 else ''}"
                if self.rows
                else " No sessions yet."
            )
        )
        return "\n".join(
            cut_cells(clean_output(in_symbols(line, self.symbols)), self.width())
            for line in lines
        )
