"""A one-line progress display for a direct run on a terminal.

``melo "<task>"`` draws it on stderr when stderr is a terminal and the
output is text: the CLI runs the task with ``observe=`` and
``progress=True``, and :class:`ProgressLine` turns the run's events into
one line, redrawn on each event and once a second. :func:`format_line`
builds that line from the latest ``run.progress`` payload; it is pure, so
every level and state can be tested on synthetic payloads.
:func:`format_past` gives just that line's past facts and items, for the
interactive UI's own row (U2).

What the line shows follows the honesty rules of the progress spec
(sections 6 and 7):

- level 0, facts only: ``step 4 · attempt 2 · Verify (cycle 1) · pytest
  0:41 · tests 3/8 · 1m03s``;
- level 1, also past facts and the steps of past runs, without a percent:
  ``last time 6 steps, 2m10s: Read test_parser.py ✓ · Edit parser.py ◐ ·
  pytest``;
- level 2, a bar: ``▓▓▓▓▓▓░░░░ 57% · step 4 of ~7 · ...``; off its
  template the bar freezes and the line says why.

Never a percent at level 0 or 1, never ``step n/max``, and 100% only on
the end of a completed run.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
import os
import queue
import threading
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TextIO

from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.procedures.classify import label_for

# Moves to the start of the row and clears it; every draw starts with it.
ERASE: Final = "\r\x1b[2K"
DIM: Final = "\x1b[2m"
RESET: Final = "\x1b[0m"

SEPARATOR: Final = " · "
# Seconds between redraws that no event asks for, so the timers advance.
REDRAW_INTERVAL_S: Final = 1.0
# Columns assumed when the terminal does not report its width.
FALLBACK_COLUMNS: Final = 80
# Cells of the level-2 bar.
BAR_CELLS: Final = 10
# The highest percent a running bar shows (the tracker's cap).
RUNNING_CAP_PERCENT: Final = 95
# Seconds an event or a log record waits for the line to be drawn. After
# Ctrl+S, a terminal takes no output once its queue is full, and a write
# waits until Ctrl+Q: the run then waits here once, not for the terminal.
DRAW_WAIT_S: Final = 1.0
# Seconds close() waits for the line to be erased.
JOIN_TIMEOUT_S: Final = 1.0

# Marks of a step at level 1 and 2; a pending step at level 1 has none.
ITEM_MARKS: Final[Mapping[str, str]] = {
    "done": "✓",
    "current": "◐",
    "skipped": "–",
    "pending": "○",
}

# What is left out, in this order, while the line is too wide (section 7).
DROP_ORDER: Final = ("past_times", "tests", "items", "attempt")

# What a similar match's past facts and steps follow when the event names
# no routine: they are another routine's, a hint (spec section 5.4), so the
# line never says "last time" for a task that has not run.
SIMILAR_HINT: Final = "like a similar task:"


@dataclass(frozen=True)
class RunningTool:
    """The tool call under way: its label and when it started."""

    label: str
    started: float
    call_id: str = ""


# ── Text helpers ──


def run_elapsed(seconds: float) -> str:
    """A run's time: ``9s``, ``1m03s``, ``1h02m``."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def tool_elapsed(seconds: float) -> str:
    """A tool call's time, as a clock: ``0:41``, ``12:03``, ``1:02:03``."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}:{secs:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def wall_range(lo: float, hi: float) -> str:
    """Past wall times from *lo* to *hi* seconds, in a range that holds both.

    ``40-55s`` under a minute, ``2-3 min`` above (whole minutes, *lo*
    rounded down and *hi* up), ``45s-2 min`` across; one value when the
    ends meet: ``45s``, ``2 min``.
    """
    lo, hi = max(0.0, min(lo, hi)), max(0.0, lo, hi)
    if hi < 60:
        low, high = math.floor(lo), math.ceil(hi)
        return f"{low}s" if low == high else f"{low}-{high}s"
    high_min = math.ceil(hi / 60)
    if lo < 60:
        return f"{math.floor(lo)}s-{high_min} min"
    low_min = math.floor(lo / 60)
    return f"{low_min} min" if low_min == high_min else f"{low_min}-{high_min} min"


def char_width(char: str, *, wide_ambiguous: bool = False) -> int:
    """Terminal cells of *char*: 2 for wide and full-width, 0 for combining.

    With *wide_ambiguous*, East Asian Ambiguous characters, such as the
    line's own ``·`` and ``…``, count 2 as well (see
    :func:`wants_wide_ambiguous`).
    """
    if unicodedata.combining(char):
        return 0
    wide = ("W", "F", "A") if wide_ambiguous else ("W", "F")
    return 2 if unicodedata.east_asian_width(char) in wide else 1


def display_width(text: str, *, wide_ambiguous: bool = False) -> int:
    """Terminal cells of *text* (see :func:`char_width`)."""
    return sum(char_width(char, wide_ambiguous=wide_ambiguous) for char in text)


def cut_to_width(text: str, width: int, *, wide_ambiguous: bool = False) -> str:
    """*text* in at most *width* cells, ending with ``…`` when it was cut."""
    if display_width(text, wide_ambiguous=wide_ambiguous) <= width:
        return text
    mark = char_width("…", wide_ambiguous=wide_ambiguous)
    if width < mark:
        return ""
    kept: list[str] = []
    used = 0
    for char in text:
        cells = char_width(char, wide_ambiguous=wide_ambiguous)
        if used + cells > width - mark:
            break
        kept.append(char)
        used += cells
    return "".join(kept) + "…"


# Format characters that join or part the glyphs around them, as in a
# family emoji (man, ZWJ, woman, ZWJ, girl): kept by printable(). Each
# counts as one cell, so a terminal that joins them shows a line narrower
# than counted, never wider.
JOINERS: Final = frozenset("\u200c\u200d")


def _unprintable(char: str) -> bool:
    return char not in JOINERS and unicodedata.category(char).startswith("C")


def printable(text: str) -> str:
    """*text* on one line, with control and format characters as ``?``.

    Labels name files and programs, and a file name may hold escape
    sequences or bidirectional controls that would rewrite the terminal.
    The zero-width joiner and non-joiner (:data:`JOINERS`) are kept.
    """
    return "".join(
        "?" if _unprintable(char) else char for char in " ".join(text.split())
    )


def _int(value: Any) -> int | None:
    """*value* as an int, or None when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return int(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _steps(n: int) -> str:
    return f"{n} step" if n == 1 else f"{n} steps"


def _step_range(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


# ── The line ──


@dataclass(frozen=True)
class _Parts:
    """The parts of one line, before any is left out for width."""

    head: list[str]  # bar and step: never left out
    attempt: str | None
    facts: list[str]  # phase and the running tool
    tests: str | None
    elapsed: str
    past_steps: str | None  # "last time 6 steps", "past 5-8 steps"
    past_times: str | None  # "2m10s", "2-3 min", "past runs 2-3 min"
    items: str | None
    items_title: str | None  # "last time:" (or a similar match's) at level 1
    merge_past: bool  # one past run: the items follow its facts

    def compose(self, dropped: Sequence[str]) -> str:
        parts = list(self.head)
        if self.attempt and "attempt" not in dropped:
            parts.append(self.attempt)
        parts += self.facts
        if self.tests and "tests" not in dropped:
            parts.append(self.tests)
        parts.append(self.elapsed)
        past = self.past_steps or ""
        if self.past_times and "past_times" not in dropped:
            past = f"{past}, {self.past_times}" if past else self.past_times
        items = self.items if "items" not in dropped else None
        if items and self.merge_past and past:
            parts.append(f"{past}: {items}")
            return SEPARATOR.join(parts)
        if past:
            parts.append(past)
        if items:
            parts.append(f"{self.items_title} {items}" if self.items_title else items)
        return SEPARATOR.join(parts)


def items_text(items: Any, level: int) -> str | None:
    """The steps of past runs, each with its mark; None when there are none."""
    if not isinstance(items, list) or not items:
        return None
    shown: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        label = printable(str(item.get("label") or ""))
        if not label:
            continue
        repeat = _int(item.get("repeat"))
        if repeat is not None and repeat > 1:
            label = f"{label} (×{repeat})"
        state = item.get("state")
        # Level 1 has no unticked to-do list: a step not matched yet is
        # drawn plain (section 5.7).
        mark = None if level < 2 and state == "pending" else ITEM_MARKS.get(str(state))
        shown.append(f"{label} {mark}" if mark else label)
    return SEPARATOR.join(shown) or None


def _similar_hint(title: Any) -> str:
    """``like <routine title>:``, or :data:`SIMILAR_HINT` without a title.

    The title names the routine a similar match's steps are from (spec
    section 5.4); it is printed as a label is (:func:`printable`), since a
    query may hold escape sequences.
    """
    text = printable(title).strip() if isinstance(title, str) else ""
    return f"like {text}:" if text else SIMILAR_HINT


def past_facts(
    past: Any, level: int, *, hint: str | None = None
) -> tuple[str | None, str | None, bool]:
    """(steps text, times text, a single past run) from the payload's ``past``.

    *hint*: the facts are a similar task's, shown after it (see
    :func:`_similar_hint`) instead of ``last time`` or ``past``.
    """
    if not isinstance(past, Mapping):
        return None, None, False
    runs = _int(past.get("runs")) or 0
    if level >= 2:
        lo, hi = _number(past.get("wall_lo_s")), _number(past.get("wall_hi_s"))
        times = None if lo is None or hi is None else f"past runs {wall_range(lo, hi)}"
        return None, times, False
    if runs <= 1:
        steps = _int(past.get("last_steps"))
        if steps is None:
            steps = _int(past.get("steps_hi"))
        wall = _number(past.get("last_wall_s"))
        if wall is None:
            wall = _number(past.get("wall_hi_s"))
        if steps is None:
            return None, None, False
        return (
            f"{hint or 'last time'} {_steps(steps)}",
            None if wall is None else run_elapsed(wall),
            True,
        )
    lo_steps, hi_steps = _int(past.get("steps_lo")), _int(past.get("steps_hi"))
    if lo_steps is None or hi_steps is None:
        return None, None, False
    lo, hi = _number(past.get("wall_lo_s")), _number(past.get("wall_hi_s"))
    steps_text = f"{hint or 'past'} {_step_range(lo_steps, hi_steps)} steps"
    times = None if lo is None or hi is None else wall_range(lo, hi)
    return steps_text, times, False


def bar_percent(fraction: float) -> int:
    """The percent a bar shows before the end: rounded down, at most 95."""
    return min(RUNNING_CAP_PERCENT, max(0, math.floor(fraction * 100 + 1e-9)))


def _bar(fraction: float, complete: bool) -> str:
    """:data:`BAR_CELLS` cells, all full only for a *complete* run."""
    if complete:
        cells = BAR_CELLS
    else:
        cells = min(BAR_CELLS - 1, round(bar_percent(fraction) / 100 * BAR_CELLS))
    return "▓" * cells + "░" * (BAR_CELLS - cells)


def _head(payload: Mapping[str, Any], level: int, step: int) -> list[str]:
    """The bar (level 2) and the step, with the reason when off the template."""
    step_text = f"step {step}" if step > 0 else "starting"
    fraction = _number(payload.get("fraction"))
    if level < 2 or fraction is None:
        return [step_text]
    ended = payload.get("state") == "ended"
    # 100% only on the end of a completed run: the tracker gives 1.0 then
    # and only then (section 6.3).
    complete = ended and fraction >= 1.0
    bar = _bar(fraction, complete)
    percent = "100%" if complete else f"{bar_percent(fraction)}%"
    reason = payload.get("off_reason")
    if reason and not ended:
        past = payload.get("past")
        why = {"longer": "longer than usual", "retrying": "retrying", "stuck": "stuck"}
        text = why.get(str(reason), str(reason))
        if reason == "longer" and isinstance(past, Mapping):
            lo, hi = _int(past.get("steps_lo")), _int(past.get("steps_hi"))
            if lo is not None and hi is not None:
                text += f" (past {_step_range(lo, hi)})"
        return [f"{bar} paused at {percent}", f"{step_text}, {printable(text)}"]
    if ended and not complete:
        return [f"{bar} stopped at {percent}", step_text]
    expected = _int(payload.get("expected_steps"))
    if not ended and expected is not None and step > 0:
        step_text += f" of ~{expected}"
    return [f"{bar} {percent}", step_text]


def _past_parts(
    payload: Mapping[str, Any], level: int
) -> tuple[str | None, str | None, str | None, str | None, bool]:
    """(steps, times, items, items title, merge) of a payload's past runs.

    The items title names what their steps are (spec section 5.4); the
    merge says a single past run's items follow its facts after a colon.
    """
    basis = payload.get("basis")
    title = basis.get("title") if isinstance(basis, Mapping) else None
    similar = isinstance(basis, Mapping) and basis.get("match") == "similar"
    hint = _similar_hint(title) if similar else None
    past_steps, past_times, single = (
        past_facts(payload.get("past"), level, hint=hint)
        if level >= 1
        else (None, None, False)
    )
    items = items_text(payload.get("items"), level) if level >= 1 else None
    items_title = None
    if level == 1:
        # A similar task's steps follow its facts, or carry the hint.
        if not similar:
            items_title = "last time:"
        else:
            items_title = "its steps:" if past_steps else hint
    return past_steps, past_times, items, items_title, single and level == 1


def _parts(
    payload: Mapping[str, Any] | None,
    tool: RunningTool | None,
    now: float,
    run_started: float,
) -> _Parts:
    payload = payload if isinstance(payload, Mapping) else {}
    level = _int(payload.get("level")) or 0
    step = _int(payload.get("step")) or 0
    attempt = _int(payload.get("attempt")) or 0
    head = _head(payload, level, step)

    facts: list[str] = []
    phase = payload.get("phase")
    if isinstance(phase, str) and phase:
        cycle = _int(payload.get("cycle")) or 0
        text = printable(phase)
        facts.append(f"{text} (cycle {cycle})" if cycle >= 1 else text)
    if tool is not None:
        facts.append(f"{printable(tool.label)} {tool_elapsed(now - tool.started)}")

    tests_text = None
    tests = payload.get("tests")
    if isinstance(tests, Mapping):
        passed, total = _int(tests.get("passed")), _int(tests.get("total"))
        if passed is not None and total is not None:
            tests_text = f"tests {passed}/{total}"

    past_steps, past_times, items, items_title, merge_past = _past_parts(payload, level)
    return _Parts(
        head=head,
        attempt=f"attempt {attempt}" if attempt >= 2 else None,
        facts=facts,
        tests=tests_text,
        elapsed=run_elapsed(now - run_started),
        past_steps=past_steps,
        past_times=past_times,
        items=items,
        items_title=items_title,
        merge_past=merge_past,
    )


def format_line(
    payload: Mapping[str, Any] | None,
    current_tool: RunningTool | None,
    now: float,
    width: int,
    *,
    run_started: float,
    color: bool = False,
    wide_ambiguous: bool = False,
) -> str:
    """The progress line for the latest ``run.progress`` *payload*.

    *payload* is that event's data, or None before the first one;
    *current_tool* the call under way, if any; *now* and *run_started*
    are clock readings in seconds. The line fits *width* cells, counted
    as :func:`char_width` says with *wide_ambiguous*: past times, tests,
    items and the attempt are left out in that order (section 7), and
    what still does not fit is cut with ``…``. With *color*, the line is
    dim. It holds no newline and no erase.
    """
    parts = _parts(payload, current_tool, now, run_started)
    budget = max(1, width)
    dropped: list[str] = []
    line = parts.compose(dropped)
    for part in DROP_ORDER:
        if display_width(line, wide_ambiguous=wide_ambiguous) <= budget:
            break
        dropped.append(part)
        line = parts.compose(dropped)
    line = cut_to_width(line, budget, wide_ambiguous=wide_ambiguous)
    return f"{DIM}{line}{RESET}" if color else line


def format_past(
    payload: Mapping[str, Any] | None,
    width: int,
    *,
    wide_ambiguous: bool = False,
) -> str:
    """The past facts and past runs' items of *payload*, as level 1 draws them.

    This is the segment :func:`format_line` composes at level 1 after its
    facts and elapsed time: ``last time 6 steps, 2m10s: Read …``, ``past
    5-8 steps, 2-3 min · last time: Read …``, or, for a similar match,
    ``like <routine title>: …`` and ``its steps: …`` (spec section 5.4).
    A single past run's items follow its facts after a colon. It is empty
    below level 1, or when the payload has neither past facts nor items.
    It fits *width* cells, counted as :func:`char_width` says with
    *wide_ambiguous*: past times go first, then the items, and what still
    does not fit is cut with ``…``.
    """
    payload = payload if isinstance(payload, Mapping) else {}
    level = _int(payload.get("level")) or 0
    if level < 1:
        return ""
    past_steps, past_times, items, items_title, merge_past = _past_parts(payload, level)

    def compose(dropped: Sequence[str]) -> str:
        parts: list[str] = []
        past = past_steps or ""
        if past_times and "past_times" not in dropped:
            past = f"{past}, {past_times}" if past else past_times
        shown = items if "items" not in dropped else None
        if shown and merge_past and past:
            return f"{past}: {shown}"
        if past:
            parts.append(past)
        if shown:
            parts.append(f"{items_title} {shown}" if items_title else shown)
        return SEPARATOR.join(parts)

    budget = max(1, width)
    dropped: list[str] = []
    text = compose(dropped)
    for part in ("past_times", "items"):
        if display_width(text, wide_ambiguous=wide_ambiguous) <= budget:
            break
        dropped.append(part)
        text = compose(dropped)
    return cut_to_width(text, budget, wide_ambiguous=wide_ambiguous)


# ── When to draw ──


def wants_line(
    mode: str | None,
    *,
    query: str,
    output_format: str,
    stream: TextIO,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Whether a direct run draws the line: ``--progress auto`` (the default).

    Only for a task (not the ``melo ""`` loop) with text output, on a
    *stream* that is a terminal other than ``TERM=dumb``.
    """
    environ = os.environ if environ is None else environ
    if mode == "off" or not query or output_format != "text":
        return False
    if environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def wants_color(environ: Mapping[str, str] | None = None) -> bool:
    """Dim text unless ``NO_COLOR`` is set to a non-empty value (no-color.org)."""
    environ = os.environ if environ is None else environ
    return not environ.get("NO_COLOR")


# Locales (language codes) whose terminals often draw East Asian Ambiguous
# characters in two cells.
WIDE_AMBIGUOUS_LANGUAGES: Final = ("zh", "ja", "ko")


def wants_wide_ambiguous(environ: Mapping[str, str] | None = None) -> bool:
    """Whether to count East Asian Ambiguous characters two cells wide.

    A terminal can draw them, among them the line's own ``·``, ``…``,
    ``◐`` and ``▓``, in two cells: an option in iTerm2 and Terminal.app,
    common with Chinese, Japanese and Korean. In such a locale (the first
    of ``LC_ALL``, ``LC_CTYPE`` and ``LANG`` that is set), they count two:
    where the terminal draws them in one, a cut line ends a few cells
    short, rather than wrapping where it draws them in two.
    """
    environ = os.environ if environ is None else environ
    for name in ("LC_ALL", "LC_CTYPE", "LANG"):
        locale = environ.get(name)
        if locale:
            return locale.lower().startswith(WIDE_AMBIGUOUS_LANGUAGES)
    return False


def terminal_columns(stream: TextIO) -> int:
    """The width of *stream*'s terminal, or :data:`FALLBACK_COLUMNS`."""
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, OSError, ValueError):
        return FALLBACK_COLUMNS
    return columns if columns > 0 else FALLBACK_COLUMNS


def in_foreground(stream: TextIO) -> bool:
    """Whether melo may draw on *stream*: it is in the foreground there.

    A job in the background (``melo "<task>" &``, or Ctrl+Z then ``bg``)
    draws nothing, as git's progress does: its draws would wipe what the
    user types at the shell's prompt, and with ``stty tostop`` the first
    one would stop the job. A stream that is not melo's controlling
    terminal, or no terminal, counts as the foreground.
    """
    try:
        return os.tcgetpgrp(stream.fileno()) == os.getpgrp()
    except (AttributeError, OSError, ValueError):
        return True


# ── The display ──


# A request to the writer: its number, a record to write above the line
# (None to draw the line only), and whether to erase the line and stop.
_Request = tuple[int, str | None, bool]


class ProgressLine:
    """Draws a run's progress line on *stream* until the run ends.

    :meth:`observe` is the run's ``observe=`` sink. One daemon thread, the
    writer, does every write: the line, its erase and the log records
    written above it (see :meth:`start`), in the order they are asked for,
    so they never mix. It draws the line again on each event and once a
    second with the latest state, so the timers advance also while
    ``python_exec`` code holds the event loop. An event or a record waits
    for its draw, but at most :data:`DRAW_WAIT_S`: a terminal whose output
    was stopped (Ctrl+S) holds the writer, never the run, and SIGTERM can
    still end it. A draw is :data:`ERASE` and the line, never a newline.
    ``run.finished`` or :meth:`close` erases the line for good.
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        color: bool = False,
        interval_s: float = REDRAW_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        columns: Callable[[], int] | None = None,
        wide_ambiguous: bool = False,
    ) -> None:
        self._stream = stream
        self._color = color
        self._wide_ambiguous = wide_ambiguous
        self._interval_s = interval_s
        self._clock = clock
        self._columns = columns or (lambda: terminal_columns(stream))
        # Held to read or change the flags below, never while writing.
        # Re-entrant: a record can reach the log handler on a thread that
        # holds it, from a finalizer that garbage collection runs there.
        self._lock = threading.RLock()
        self._carried = threading.Condition(self._lock)
        # What the writer is asked to do, in order (see _Request); put()
        # is safe also from a finalizer.
        self._requests: queue.SimpleQueue[_Request] = queue.SimpleQueue()
        self._numbers = itertools.count(1)
        self._done = 0  # the last request the writer carried out
        self._stalled = False  # a write took longer than DRAW_WAIT_S
        self._thread: threading.Thread | None = None
        self._handler: _LineLogHandler | None = None
        self._run_started = clock()
        # Set by the run's events, read by the writer.
        self._payload: Mapping[str, Any] | None = None
        self._tool: RunningTool | None = None
        self._started = False
        self._closed = False
        # The writer's own: whether the line is on the screen, and its text.
        self._drawn = False
        self._last = ""

    @property
    def active(self) -> bool:
        """Whether the line is started and not yet closed."""
        return self._started and not self._closed

    def start(self) -> None:
        """Draw the line, start the redraws and catch log records above it.

        While the line is up, a handler on the root logger writes each
        WARNING or higher record as ``%(message)s`` on a line of its own
        above it, as logging's last resort would, and redraws the line.
        """
        with self._lock:
            if self._started or self._closed:
                return
            self._started = True
            self._run_started = self._clock()
            self._thread = threading.Thread(
                target=self._write_requests, name="melo-progress", daemon=True
            )
        self._thread.start()
        self._ask(None)
        self._handler = _LineLogHandler(self)
        logging.getLogger().addHandler(self._handler)

    def observe(self, event: AgentEvent) -> None:
        """The run's sink: follow *event* and redraw. Never raises."""
        # A display never fails a run, and logs nothing: a record about a
        # failing sink would go through the line's own log handler.
        with contextlib.suppress(Exception):
            self._follow(event)

    def close(self) -> None:
        """Erase the line and stop drawing it, for good. Idempotent.

        Waits for the erase at most :data:`JOIN_TIMEOUT_S`, and not at all
        once a write is known to wait for the terminal: the line is then
        erased when the terminal takes output again, if melo still runs.
        """
        handler, self._handler = self._handler, None
        if handler is not None:
            logging.getLogger().removeHandler(handler)
        with self._lock:
            if not self._closed:
                self._closed = True
                self._requests.put((next(self._numbers), None, True))
            thread, stalled = self._thread, self._stalled
        if thread is None or thread is threading.current_thread() or stalled:
            return
        thread.join(JOIN_TIMEOUT_S)
        if thread.is_alive():
            with self._lock:
                self._stalled = True  # no later close() waits again

    def write_above(self, text: str) -> None:
        """Write *text* and a newline on the stream, above the line if drawn."""
        if not self._ask(text):  # no line: write it as it is
            self._write(text + "\n")

    # ── Internals ──

    def _follow(self, event: AgentEvent) -> None:
        if event.type == "run.finished":
            self.close()
            return
        # Only the run's events change these, and the writer reads them
        # whole, so they are set without the lock.
        data = event.data
        if event.type == "run.progress":
            self._payload = dict(data)
        elif event.type == "tool.started":
            self._tool = RunningTool(
                label=label_for(str(data.get("name") or ""), data.get("arguments")),
                started=self._clock(),
                call_id=str(data.get("call_id") or ""),
            )
        elif event.type == "tool.finished":
            tool = self._tool
            if tool is None or tool.call_id == str(data.get("call_id") or ""):
                self._tool = None
        elif event.type == "step.started":
            # The run.progress that follows says the same; this keeps the
            # step right should the run's progress events stop.
            self._payload = {
                **(self._payload or {}),
                "step": data.get("step"),
                "attempt": data.get("attempt"),
            }
            self._tool = None
        elif event.type != "run.started":
            return
        self._ask(None)

    def _ask(self, record: str | None) -> bool:
        """Have the writer draw the line, with *record* above it if given.

        Waits until it did, at most :data:`DRAW_WAIT_S`, except on the
        writer itself and once a write is known to take longer. False
        when there is no writer to ask: before :meth:`start` or once closed.
        """
        with self._carried:
            if not self._started or self._closed:
                return False
            number = next(self._numbers)
            self._requests.put((number, record, False))
            if self._stalled or threading.current_thread() is self._thread:
                return True
            deadline = time.monotonic() + DRAW_WAIT_S
            while self._done < number and not self._stalled:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._stalled = True  # the terminal takes no output
                    break
                self._carried.wait(left)
        return True

    def _write_requests(self) -> None:
        """The writer: carry out the requests, and redraw every interval."""
        while True:
            try:
                batch = [self._requests.get(timeout=self._interval_s)]
            except queue.Empty:
                batch = []
            with contextlib.suppress(queue.Empty):
                while True:
                    batch.append(self._requests.get_nowait())
            with self._lock:
                closing = self._closed or any(stop for _, _, stop in batch)
            try:
                going_on = self._carry_out(
                    [record for _, record, _ in batch if record is not None],
                    closing=closing,
                )
            except Exception:  # a display never fails a run, and logs nothing
                going_on = False
                with self._lock:
                    self._closed = True
            with self._carried:
                self._done = max([self._done, *(number for number, _, _ in batch)])
                self._stalled = False
                self._carried.notify_all()
            if not going_on:
                return

    def _carry_out(self, records: list[str], *, closing: bool) -> bool:
        """Write *records* above the line, then draw it, or erase it if
        *closing*, in one write. False once the writer is done for good.

        In the background (see :func:`in_foreground`) the records are
        written as they are, and the line is neither drawn nor erased: the
        shell wrote below it, so it is no longer where the cursor is.
        """
        foreground = in_foreground(self._stream)
        text, drawn, line = "", self._drawn and foreground, None
        if records:
            if drawn:
                text += ERASE
                drawn = False
            text += "".join(record + "\n" for record in records)
        if closing:
            if drawn:
                text += ERASE
                drawn = False
        elif foreground:
            # One cell short of the width, so the cursor never wraps a row.
            line = format_line(
                self._payload,
                self._tool,
                self._clock(),
                self._columns() - 1,
                run_started=self._run_started,
                color=self._color,
                wide_ambiguous=self._wide_ambiguous,
            )
            if not drawn or line != self._last:  # else the terminal shows it
                text += ERASE + line
            drawn = True
        try:
            if text:
                self._write(text)
        except (OSError, ValueError):  # the terminal is gone: stop drawing
            self._drawn = False
            with self._lock:
                self._closed = True
            return False
        self._drawn = drawn
        if line is not None:
            self._last = line
        return not closing

    def _write(self, text: str) -> None:
        """Write *text* on the stream now; raises OSError or ValueError.

        On a stream with a file, the stream is flushed and the bytes go to
        the file itself: a write that fails or waits leaves nothing in the
        stream's buffer and holds none of its locks, so flushing sys.stderr
        at exit neither fails nor waits because of the line.
        """
        stream = self._stream
        try:
            fd = stream.fileno()
        except (AttributeError, OSError, ValueError):  # no file: StringIO
            stream.write(text)
            stream.flush()
            return
        data = memoryview(
            text.encode(stream.encoding or "utf-8", stream.errors or "strict")
        )
        stream.flush()
        while data:
            written = os.write(fd, data)
            if written <= 0:
                raise OSError("the terminal took no output")
            data = data[written:]


class _LineLogHandler(logging.Handler):
    """Writes WARNING and higher records above the line (see ProgressLine.start)."""

    def __init__(self, line: ProgressLine) -> None:
        super().__init__(logging.WARNING)
        self.setFormatter(logging.Formatter("%(message)s"))
        self._line = line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._line.write_above(self.format(record))
        except Exception:
            self.handleError(record)
