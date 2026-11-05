"""The first-run setup on a terminal: the list a person says yes to, and the progress.

Text only; nothing here asks a question or runs anything. :func:`consent_lines`
is the one list, :func:`size_left` its total, and :class:`Progress` the
setup job's ``on_event`` that shows each row as it goes.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from collections.abc import Collection
from typing import Final, TextIO

from localmelo.support import progress_line
from localmelo.support.cli.theme import ASCII, Symbols, in_symbols, symbols_for
from localmelo.support.server_setup import run

MELO_SERVE_SIZE: Final = "about 30 MB"
LOGIN_NOTE: Final = 'macOS shows a "Login Items" notice'
HERE: Final = "✓ already here"
# The ways an item reaches this Mac, in the words of the list.
VIA: Final = {
    "dmg": "pinned download",
    "homebrew": "Homebrew",
    "huggingface": "Hugging Face",
}
# A backend's name when the plan has no row for it.
BACKENDS: Final = {"omlx": "oMLX", "splash": "Splash"}
# The rows of the job that are not a plan item, and melo_serve's own.
ROWS: Final = {
    run.MELO_SERVE: "melo_serve",
    run.SERVICE_MELO_SERVE: "Start at login",
    run.SERVICE_GATEWAY: "Gateway at login",
    run.DOWNLOAD: "Download",
    run.SAVE: "Saving the settings",
    "config:hf_home": "Model folder",
    "cache:hf_home": "Model folder",
    "disk": "Free space",
    "get": "Download",
}
_VERSION_TAIL: Final = re.compile(r"\s+\d[\d.]*$")


def _symbols() -> Symbols:
    return symbols_for(sys.stdout.encoding)


def _bare(name: str) -> str:
    """*name* without its version: ``oMLX 0.7.3`` is ``oMLX``."""
    return _VERSION_TAIL.sub("", name)


def backend_name(plan: run.Plan, backend: str) -> str:
    """The name of *backend* (an id such as ``omlx``) as a person says it."""
    item = next((i for i in plan.items if i.id == f"backend:{backend}"), None)
    return _bare(item.name) if item and item.name else BACKENDS.get(backend, backend)


def _gb(count: float) -> str:
    return f"{count / 1e9:.1f}"


def human(count: int) -> str:
    """A size as a person reads it: ``881 MB``, ``4.6 GB``."""
    if count >= 1_000_000_000:
        return f"{_gb(count)} GB"
    return f"{max(1, round(count / 1e6))} MB"


def _of(part: int, whole: int, word: str = "of") -> str:
    """``14.1 of 21.2 GB`` (or MB): *part* and *whole* in one unit."""
    shown = _gb(part) if whole >= 1_000_000_000 else str(round(part / 1e6))
    return f"{shown} {word} {human(whole)}"


def _left(item: run.Item) -> int:
    """What is still to download for *item*."""
    if item.state == "done":
        return 0
    return max(item.size_bytes - item.cached_bytes, 0)


def size_left(plan: run.Plan, skip: Collection[str]) -> int:
    """Bytes still to download for *plan* when the items in *skip* are left out."""
    return max(
        plan.download_bytes
        - sum(_left(item) for item in plan.items if item.id in skip),
        0,
    )


def setup_row(plan: run.Plan, skip: Collection[str]) -> str:
    """The row that says yes: ``Set up Qwen3 8B (5.8 GB)``."""
    assert plan.pick is not None
    left = size_left(plan, skip)
    return f"Set up {plan.pick.name} ({human(left) if left else 'nothing to download'})"


def skippable(plan: run.Plan) -> list[tuple[str, str]]:
    """The ids a person may leave out, each with its question: what is not here yet."""
    asks = [
        (item.id, f"Install {_bare(item.name)} ({human(_left(item))})?")
        for item in plan.items
        if item.state != "done" and item.kind in ("backend", "model")
    ]
    return [*asks, (run.SERVICE_MELO_SERVE, "Start melo_serve at login?")]


def _free(plan: run.Plan) -> str:
    free = plan.machine.free_gb
    return f"{free:.1f} GB free" if 0 < free < 10 else f"{free:.0f} GB free"


def _this_mac(plan: run.Plan) -> str:
    mac = plan.machine
    parts = [
        mac.chip,
        f"{mac.ram_gb} GB" if mac.ram_gb else "",
        f"macOS {mac.macos}" if mac.macos else "",
        _free(plan) if mac.free_gb else "",
    ]
    return "This Mac: " + " · ".join(part for part in parts if part)


def _row(
    item: run.Item,
    skip: Collection[str],
    chat: run.Pick,
    seen: set[tuple[str, str]],
) -> tuple[str, str, str, str]:
    """The name, kind, size and where of one plan item."""
    if item.kind == "backend":
        chat_side = item.id == f"backend:{chat.backend}"
        kind = "chat backend" if chat_side else "memory backend"
    else:
        kind = "chat model" if item.role == "chat" else "memory model"
    left = _left(item)
    if item.state == "done":
        return item.name, kind, "", HERE
    if item.id in skip:
        return item.name, kind, "", "skipped"
    if item.state == "partial":
        return (
            item.name,
            kind,
            human(left),
            f"{_of(item.cached_bytes, item.size_bytes)} already here",
        )
    place = (item.via, item.dest)  # a place that is shown once is "same place" after
    where = VIA.get(item.via, item.via) + (f" → {item.dest}" if item.dest else "")
    if item.kind == "model" and place in seen:
        where = "same place"
    seen.add(place)
    return item.name, kind, human(left), where


def _names(plan: run.Plan, ids: Collection[str], wanted: Collection[str]) -> str:
    """The names of the items in *ids* that are in *wanted*, such as ``oMLX and Qwen3 8B``."""
    names = [_bare(i.name) for i in plan.items if i.id in ids and i.id in wanted]
    return " and ".join(names)


def skip_effects(plan: run.Plan, skip: Collection[str]) -> list[str]:
    """One line for each thing that leaving out *skip* leads to."""
    if plan.pick is None:
        return []
    chat, embedding = run.role_items(plan, plan.pick)
    lines = []
    if left := _names(plan, skip, chat):
        lines.append(
            f"Without {left}, melo_serve cannot chat yet; "
            f"melo-serve get {plan.pick.id} finishes it later."
        )
    if left := _names(plan, skip, embedding):
        lines.append(f"Without {left}, memory goes by words only.")
    if run.SERVICE_MELO_SERVE in skip:
        lines.append(
            "Without the login item, melo serve starts melo_serve when needed."
        )
    return lines


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
    assert plan.pick is not None
    pick = plan.pick
    if melo_serve_there:
        service = ("melo_serve", "model service", "", HERE)
    elif run.MELO_SERVE in skip:
        service = ("melo_serve", "model service", "", "skipped")
    else:
        service = ("melo_serve", "model service", MELO_SERVE_SIZE, "uv tool")
    seen: set[tuple[str, str]] = set()
    rows = [
        service,
        *(_row(item, skip, pick, seen) for item in plan.items),
        (
            "Start at login",
            "melo_serve",
            "",
            "skipped" if run.SERVICE_MELO_SERVE in skip else LOGIN_NOTE,
        ),
    ]
    name_w, kind_w, size_w = (max(len(row[i]) for row in rows) for i in range(3))
    left = size_left(plan, skip)
    lines = [
        _this_mac(plan),
        f"{'Chosen' if chosen else 'Recommended'}: {pick.name} on "
        f"{backend_name(plan, pick.backend)}. Nothing is installed until you say yes.",
        "",
    ]
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
    free = f" {_free(plan)}." if plan.machine.free_gb else ""
    lines.append(
        f"Download about {human(left)}.{free}"
        if left
        else f"Nothing to download.{free}"
    )
    if not plan.enough_disk and not skip:
        lines.append("There is not enough free space for this. Another model may fit.")
    lines += skip_effects(plan, skip)
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
        names = {
            **ROWS,
            **{i.id: _bare(i.name) for i in plan.items if i.name},
        }
        for role, row in (("chat", run.LOAD_CHAT), ("embedding", run.LOAD_EMBEDDING)):
            item = next((i for i in plan.items if i.role == role), None)
            names[row] = (
                f"Loading {_bare(item.name) if item else 'the ' + role + ' model'}"
            )
        return names

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
