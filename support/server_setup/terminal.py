"""The first-run setup on a terminal: the list a person says yes to, and the progress.

Text only; nothing here asks a question or runs anything. :func:`consent_lines`
is the list (:mod:`~localmelo.support.server_setup.listing` holds it as data), and
:class:`Progress` the setup job's ``on_event`` that shows each row as it goes.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Collection
from typing import TextIO

from localmelo.support import progress_line
from localmelo.support.cli.theme import ASCII, Symbols, in_symbols, symbols_for
from localmelo.support.server_setup import listing, run
from localmelo.support.server_setup.listing import bare, human, left


def _symbols() -> Symbols:
    return symbols_for(sys.stdout.encoding)


def _of(part: int, whole: int, word: str = "of") -> str:
    """``14.1 of 21.2 GB`` (or MB): *part* and *whole* in one unit."""
    shown, total = listing.amounts(part, whole)
    return f"{shown} {word} {total}"


def setup_row(plan: run.Plan, skip: Collection[str]) -> str:
    """The row that says yes: ``Set up Qwen3 8B (5.8 GB)``."""
    return listing.setup_row(plan, skip, listing.english)


def skippable(plan: run.Plan) -> list[tuple[str, str]]:
    """The ids a person may leave out, each with its question: what is not here yet."""
    items = {item.id: item for item in plan.items}
    login = {
        run.SERVICE_MELO_SERVE: "Start melo_serve at login?",
        run.SERVICE_GATEWAY: "Start the gateway at login?",
    }
    return [
        (
            id_,
            (
                login[id_]
                if id_ in login
                else f"Install {bare(items[id_].name)} ({human(left(items[id_]))})?"
            ),
        )
        for id_ in listing.skippable_ids(plan)
    ]


def consent_lines(
    plan: run.Plan,
    skip: Collection[str],
    melo_serve_there: str,
    *,
    chosen: bool = False,
    width: int | None = None,
    symbols: Symbols | None = None,
) -> list[str]:
    """The list a person says yes to; *plan* must be available.

    *skip* is what they left out; *melo_serve_there* the version of the
    melo_serve already on this Mac, or ``""``; *chosen* whether the pick is
    their choice, not the recommendation. A row too wide for *width* puts where
    it goes on a line of its own. Symbols are plain ASCII where the terminal
    is not UTF-8.
    """
    found = listing.build(
        plan, skip, melo_serve_there, listing.english, chosen=chosen, terminal=True
    )
    rows = [(row.name, row.kind, row.size, row.where) for row in found.rows]
    name_w, kind_w, size_w = (max(len(row[i]) for row in rows) for i in range(3))
    lines = [found.this_mac, found.recommended, ""]
    heads = [
        f"  {name:<{name_w}}  {kind:<{kind_w}}    {size:>{size_w}}"
        for name, kind, size, _ in rows
    ]
    # Where it goes drops to a line of its own, for every row, when one is too wide.
    narrow = width is not None and any(
        len(head) + 3 + len(row[3]) > width
        for head, row in zip(heads, rows, strict=True)
    )
    for head, row in zip(heads, rows, strict=True):
        if narrow:
            lines += [head.rstrip(), " " * (name_w + 4) + row[3]]
        else:
            lines.append(f"{head}   {row[3]}".rstrip())
    lines.append(f"{found.download} {found.free}".rstrip())
    lines += found.notes
    symbols = symbols or _symbols()
    return [in_symbols(line, symbols) for line in lines]


class Progress:
    """The setup job's ``on_event``: one live line for a row with bytes, one closing line per row.

    On a terminal a row that is running shows as a line redrawn in place, with
    a bar when it has bytes, clipped to the width. When the row ends the line
    is replaced by its closing line (``✓ oMLX  installed 0.7.3 into …``,
    ``✗ …``, ``· skipped …``). Without a terminal only the closing lines are
    written. Events come from two threads; a lock keeps the lines whole, and
    nothing is raised.
    """

    def __init__(
        self,
        plan: run.Plan,
        stream: TextIO | None = None,
        *,
        live: bool | None = None,
    ) -> None:
        self._stream = stream or sys.stdout
        self._live = (
            self._terminal(self._stream) if live is None else live
        )  # a line redrawn in place
        self._names = self._name_rows(plan)
        self._lock = threading.Lock()
        self._drawn = False  # a live line is on the screen
        self._failed: set[str] = set()
        self._symbols = _symbols()

    @staticmethod
    def _terminal(stream: TextIO) -> bool:
        try:
            return stream.isatty() and os.environ.get("TERM") != "dumb"
        except (AttributeError, ValueError):
            return False

    @staticmethod
    def _name_rows(plan: run.Plan) -> dict[str, str]:
        """The plain name of each row the job can report."""
        return listing.row_names(plan, listing.english)

    def __call__(self, event: run.Event) -> None:
        try:
            with self._lock:
                self._show(event)
        except Exception:  # a screen that fails must not stop a download
            pass

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()

    def _erase(self) -> str:
        if not self._drawn:
            return ""
        self._drawn = False
        return progress_line.ERASE

    def _show(self, event: run.Event) -> None:
        if isinstance(event, run.Finished):
            text = self._erase()
            if event.state == "failed" and event.error not in self._failed:
                text += f"{self._symbols.failed} {event.error}\n"
            if text:
                self._write(text)
            return
        name = self._names.get(event.item, event.item.split(":", 1)[-1])
        if event.state == "running":
            if self._live:
                self._write(self._erase() + self._live_line(name, event))
                self._drawn = True
            return
        mark = {
            "done": self._symbols.ok,
            "failed": self._symbols.failed,
        }.get(event.state, self._symbols.dot)
        if event.state == "failed":
            self._failed.add(event.line)
        line = f"{mark} {name}" + (f"  {event.line}" if event.line else "")
        self._write(f"{self._erase()}{line}\n")

    def _live_line(self, name: str, event: run.ItemEvent) -> str:
        """The line of a running row, cut to the terminal's width."""
        text = f"  {name}"
        if event.total_bytes:
            bar = progress_line.progress_bar(
                event.done_bytes / event.total_bytes, False
            )
            if self._symbols is ASCII:
                bar = bar.replace("▓", "#").replace("░", "-")
            text += f"  {bar}  {_of(event.done_bytes, event.total_bytes, '/')}"
        else:
            text += f"  {event.line}" if event.line else self._symbols.ellipsis
        wide = progress_line.wants_wide_ambiguous()
        columns = progress_line.terminal_columns(self._stream)
        return progress_line.cut_to_width(text, columns - 1, wide_ambiguous=wide)
