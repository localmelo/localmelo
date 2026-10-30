"""A turn's steps: the status bar's brief figure, and the list ctrl+o opens.

While a turn runs the bar shows where it is in a few cells: ``2/8`` with a
plan's total, ``2/~8`` with the number of steps the task took before (an
estimate, so marked), or ``step 2``. The steps themselves, one row per tool
call with its status and time, open on request in :class:`StepsPicker`,
the terminal's own list.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from prompt_toolkit.formatted_text import StyleAndTextTuples

from localmelo.support.cli.picker import ListPicker
from localmelo.support.cli.text import cells, cut_cells, escape_arguments
from localmelo.support.progress_line import tool_elapsed

# The key that opens the list, as the bar names it and prompt_toolkit binds it.
STEPS_KEY: Final = "ctrl+o"
STEPS_BINDING: Final = "c-o"


@dataclass(frozen=True)
class StepRow:
    """One row of the list: a tool call, or a step still to come.

    *status* is ``ok``, ``failed``, ``refused``, ``stopped``, ``running`` or
    ``next``; *kind* is the tool's kind (read, write, run, network), empty
    for a step still to come.
    """

    status: str
    tool: str
    target: str = ""
    seconds: float | None = None
    kind: str = ""


def step_figure(
    step: int, *, planned: int | None = None, learned: int | None = None
) -> str:
    """``2/8`` with a plan's total, ``2/~8`` with a learned one, else ``step 2``.

    A total the run has passed says nothing any more: past it, the figure is
    the step alone. No step yet gives nothing.
    """
    if step <= 0:
        return ""
    if planned and step <= planned:
        return f"{step}/{planned}"
    if learned and step <= learned:
        return f"{step}/~{learned}"
    return f"step {step}"


def learned_total(progress: Mapping[str, Any] | None) -> int | None:
    """The steps the task took before, from a ``run.progress`` payload; None when new.

    The bar's expected count when it has one, else the last run's steps,
    else the most any past run took.
    """
    if not isinstance(progress, Mapping):
        return None
    basis = progress.get("basis")
    if isinstance(basis, Mapping) and basis.get("match") == "new":
        return None
    expected = progress.get("expected_steps")
    if isinstance(expected, int) and expected > 0:
        return expected
    past = progress.get("past")
    if isinstance(past, Mapping):
        for key in ("last_steps", "steps_hi"):
            value = past.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


def call_status(status: str, exit_code: object = None) -> str:
    """A finished call's row status from its event's status and exit code."""
    if status == "success":
        nonzero = isinstance(exit_code, int) and exit_code != 0
        return "failed" if nonzero else "ok"
    return {"blocked": "refused", "cancelled": "stopped"}.get(status, "failed")


# Each status's mark (a field of Symbols) and style.
_MARKS: Final = {
    "ok": ("ok", "class:ok"),
    "failed": ("failed", "class:error"),
    "refused": ("blocked", "class:warn"),
    "stopped": ("not_run", "class:muted"),
    "next": ("not_run", "class:muted"),
}


class StepsPicker(ListPicker[StepRow]):
    """The steps of the running turn, or of the last one: ↑↓ move, esc closes.

    *source* gives the rows each time the list is drawn, so a running turn's
    new calls appear while it is open.
    """

    debounce_s = 0.0
    filterable = False

    def __init__(
        self,
        source: Callable[[], list[StepRow]],
        *,
        title: str,
        width: Callable[[], int],
        invalidate: Callable[[], None],
        symbols: Any,
    ) -> None:
        self.source, self.title = source, title
        super().__init__(width=width, invalidate=invalidate, symbols=symbols)
        self.rows, self.loading = source(), False
        done = [row for row in self.rows if row.status != "next"]
        self.selected = max(0, len(done) - 1)

    def refresh(self, delay: float | None = None) -> None:
        """The rows come from the source on each draw: nothing to load."""

    def _drawn(self) -> list[StyleAndTextTuples]:
        self.rows = self.source()
        self.selected = max(0, min(self.selected, len(self.rows) - 1))
        return super()._drawn()

    def _header(self) -> str:
        return f" {self.title}"

    def _row(self, row: StepRow) -> str:
        return "".join(part[1] for part in self._row_fragments(row))

    def _row_fragments(self, row: StepRow) -> StyleAndTextTuples:
        """``✓ → file_read  app.py · 0.4s``; a step to come is muted."""
        symbols = self.symbols
        dot = f" {symbols.dot} "
        if row.status == "running":
            mark, mark_style = symbols.spinner[1 % len(symbols.spinner)], "class:accent"
        else:
            field, mark_style = _MARKS.get(row.status, _MARKS["failed"])
            mark = getattr(symbols, field)
        if row.status == "next":
            return [("class:muted", f"{mark} {row.tool}{dot}to come")]
        kind = row.kind or "write"
        name = f"{getattr(symbols, f'kind_{kind}')} {row.tool}"
        if row.seconds is None:
            when = ""
        elif row.status == "running":
            when = f"running {tool_elapsed(row.seconds)}"
        else:
            when = f"{row.seconds:.1f}s"
        meta = dot + when if when else ""
        target, _ = escape_arguments(" ".join(row.target.split()))
        # The row's mark column (3 cells), its status, the gaps: 7 cells.
        room = self.width() - 7 - cells(name) - cells(meta)
        shown = "  " + cut_cells(target, room, symbols.ellipsis) if room >= 4 else ""
        return [
            (mark_style, mark),
            ("", " "),
            (f"class:kind_{kind}", name),
            ("", shown if target else ""),
            ("class:muted", meta),
        ]

    def _footer(self) -> str:
        done = sum(row.status != "next" for row in self.rows)
        planned = len(self.rows) - done
        count = f"{done} step{'' if done == 1 else 's'}"
        if planned:
            count += f", {planned} to come"
        return f"{count} · {self.symbols.arrows} move · esc close"
