"""Print the prompt-cache use of each model request in a stream-json log.

usage: python scripts/cache_report.py LOG [LOG ...]

Reads the NDJSON that ``melo --brief --output-format stream-json`` writes
(``{"v": 1, "type": "event", "event": ...}`` lines), or a file of bare event
objects, and prints one table per run: for each planning request its
prompt tokens, the ``cached`` tokens the server reported, the ``new`` tokens
(prompt minus cached), the leading messages and characters the request
shared with the run's previous one (``reused``), and the role of the first
message that changed. Then the totals and the cached share. ``-`` marks a
value the log does not hold: a server that sends no ``cached_tokens``, the
run's first request, or a log from before the ``prefix`` object existed.
Standard library only; it changes nothing.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from typing import Any

HEADER = ("step", "prompt", "cached", "new", "reused", "first changed")


def read_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    """The ``assistant.message`` events of *lines*, in order; others are skipped."""
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        event = obj.get("event") if isinstance(obj, dict) else None
        event = event if isinstance(event, dict) else obj
        if isinstance(event, dict) and event.get("type") == "assistant.message":
            events.append(event)
    return events


def _row(event: dict[str, Any]) -> tuple[list[str], int, int | None]:
    data = event.get("data") or {}
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    cached = usage.get("cached_tokens")
    prefix = data.get("prefix")
    new = prompt - cached if prompt is not None and cached is not None else None
    if prefix is None:
        reused, changed = "-", "-"
    else:
        reused = f"{prefix['messages']} msgs, {prefix['chars']} chars"
        changed = prefix["first_changed"] or "none"
        if prefix.get("tools_changed"):
            changed += " +tools"
    cells = [data.get("step", "?"), prompt, cached, new, reused, changed]
    return ["-" if c is None else str(c) for c in cells], prompt or 0, cached


def report(events: list[dict[str, Any]]) -> str:
    """The tables of *events*, one per run, and a total when there are several."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        runs.setdefault(str(event.get("run_id", "?")), []).append(event)
    out: list[str] = []
    total_prompt = total_cached = total_reported = 0
    for run_id, run_events in runs.items():
        rows = [_row(event) for event in run_events]
        table = [list(HEADER)] + [cells for cells, _, _ in rows]
        widths = [max(len(r[i]) for r in table) for i in range(len(HEADER))]
        out.append(f"run {run_id}")
        for r in table:
            out.append(
                "  " + "  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True))
            )
        prompt = sum(p for _, p, _ in rows)
        cached = sum(c for _, _, c in rows if c is not None)
        reported = sum(p for _, p, c in rows if c is not None)
        out.append("  " + _totals(len(rows), prompt, cached, reported))
        total_prompt += prompt
        total_cached += cached
        total_reported += reported
    if len(runs) > 1:
        out.append(f"all {len(runs)} runs")
        out.append(
            "  " + _totals(len(events), total_prompt, total_cached, total_reported)
        )
    return "\n".join(out)


def _totals(requests: int, prompt: int, cached: int, reported: int) -> str:
    line = f"total: {requests} requests, {prompt} prompt tokens"
    if not reported:
        return line + ", no cached_tokens reported, cached share n/a"
    return (
        f"{line}, {cached} cached, {reported - cached} new "
        f"(of the requests that reported cached_tokens), "
        f"cached share {100 * cached / reported:.1f}%"
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    events: list[dict[str, Any]] = []
    for path in argv:
        with open(path, encoding="utf-8") as handle:
            events.extend(read_events(handle))
    if not events:
        print("no assistant.message events found", file=sys.stderr)
        return 1
    print(report(events))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
