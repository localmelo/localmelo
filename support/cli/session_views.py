"""Read-only views of a session's recorded work and the model's working window."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.schema import (
    EMPTY_REPLY_NUDGE,
    SHORT_TERM_MAX,
    Message,
    split_thinking,
)
from localmelo.support.cli.text import cut_cells, expand_tabs
from localmelo.support.cli.theme import Symbols, in_symbols
from localmelo.support.cli.views import BackendInfo
from localmelo.support.sessions.display import (
    clean_output,
    format_duration,
    format_tokens,
    status_text,
)
from localmelo.support.sessions.runner import ResumeInfo
from localmelo.support.sessions.store import TurnInfo


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        if not math.isfinite(value) or value < 0:
            return None
    except OverflowError:
        return None
    if integer:
        return int(value) if isinstance(value, int) or value.is_integer() else None
    return value


def usage_view(turns: Sequence[TurnInfo]) -> RenderableType:
    """Include undone work; a missing counter is unknown, never a fabricated zero."""
    if not turns:
        return Text("No turns yet.")
    table = Table(box=None, padding=(0, 1), show_edge=False)
    for name in ("Turn", "Started", "Status", "Steps", "Tools", "In", "Out", "Time"):
        table.add_column(
            name, justify="left" if name in ("Started", "Status") else "right"
        )
    totals: list[int | float] = [0] * 5
    unknown = [False] * 5
    for turn in turns:
        result = turn.result if turn.status != "running" else None
        result = result or {}
        usage = result.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        values = [
            _number(result.get("steps"), integer=True),
            _number(result.get("n_tool"), integer=True),
            _number(
                usage.get("prompt_tokens", usage.get("input_tokens")), integer=True
            ),
            _number(
                usage.get("completion_tokens", usage.get("output_tokens")), integer=True
            ),
            _number(result.get("duration_s")),
        ]
        cells: list[str] = []
        for index, value in enumerate(values):
            unknown[index] |= value is None
            totals[index] += value or 0
            cells.append(
                "?"
                if value is None
                else format_duration(value) if index == 4 else str(int(value))
            )
        status = status_text(turn.status, turn.cause).word
        status += " (undone)" if turn.undone else ""
        table.add_row(
            str(turn.turn),
            time.strftime("%m-%d %H:%M", time.localtime(turn.started_at)),
            status,
            *cells,
        )
    total_cells = [
        (
            "?"
            if _number(value) is None
            else (format_duration(value) if index == 4 else format_tokens(int(value)))
            + ("+?" if unknown[index] else "")
        )
        for index, value in enumerate(totals)
    ]
    table.add_row("Totals", "", "", *total_cells, style="bold")
    return Group(
        table, Text("Totals include undone turns; ? is not recorded.", style="muted")
    )


def memory_view(
    messages: Sequence[Message],
    backend: BackendInfo,
    *,
    memory: bool,
    loaded: ResumeInfo | None,
    width: int,
    symbols: Symbols,
) -> RenderableType:
    """Inspect a copied window, without asking the model or modifying its memory."""
    turns = 0
    with_tools: set[int] = set()
    names: dict[str, str] = {}
    rows: list[str] = []
    for message in messages:
        if message.role == "user" and message.content != EMPTY_REPLY_NUDGE:
            turns += 1
        if message.role == "tool" and turns:
            with_tools.add(turns)
        for call in message.tool_calls:
            names[call.id] = call.tool_name
        content = split_thinking(message.content)[1]
        content = " ".join(clean_output(redact_secrets(content)[0]).split())
        if message.role == "tool":
            prefix = f"└ {names.get(message.tool_call_id, 'tool')} "
        elif message.role == "user":
            prefix = "› "
        elif message.role == "assistant":
            prefix = "● "
            content = content or ("(tool calls)" if message.tool_calls else "(no text)")
        else:
            prefix = f"[{message.role}] "
        rows.append(
            cut_cells(
                expand_tabs(clean_output(in_symbols("  " + prefix + content, symbols))),
                max(1, width),
            )
        )
    chars = sum(len(message.content) for message in messages)
    title = f"Working memory: {len(messages)} of {SHORT_TERM_MAX} messages · {turns} turn{'s' if turns != 1 else ''}, {len(with_tools)} with tool steps · {chars:,} chars"
    long_term = (
        "off (--memory off)"
        if not memory
        else backend.embedding or "off (no embedding backend)"
    )
    notes: list[str] = []
    if loaded is not None:
        tail = loaded.turns - loaded.checkpoint_turns
        if loaded.mode != "window":
            notes.append("Earlier tool steps are not reloaded.")
        elif tail:
            which = (
                "the newest resumed turn"
                if tail == 1
                else f"the {tail} newest resumed turns"
            )
            notes.append(f"Tool steps of {which} are not reloaded.")
    return Group(
        Text(in_symbols(title, symbols)),
        *(Text(row) for row in rows),
        Text("Long-term memory: " + clean_output(long_term)),
        *(Text(note, style="muted") for note in notes),
    )
