"""Print the prompt-cache use of each model request in a stream-json log.

usage: python scripts/cache_report.py LOG [LOG ...]

Reads the NDJSON that ``melo --brief --output-format stream-json`` writes
(``{"v": 1, "type": "event", "event": ...}`` lines), or a file of bare event
objects, and prints one table per run with a row for each model request that
the log reports: a planning request (``assistant.message``, kind ``plan``,
the step in ``at``) or a reflection (``reflection``, kind ``reflect``, the
attempt in ``at`` as ``a<n>``). A row holds its prompt tokens, the ``cached``
tokens the server reported, the ``new`` tokens (prompt minus cached), the
leading messages and characters it shared with the best of the run's earlier
requests (``reused``), that request (``against``: kind and number ``#n``) and
the role of the first message that changed. Numbers count the run's
requests; a reflection that had to be asked twice takes two (its row shows
the first request's prefix and the usage of both). Then the totals by kind
and overall, with the cached share (of all prompt tokens), and a ``reusable``
line: the requests after the run's first, with ``fresh`` tokens (content no
earlier request of the run held, which no prompt cache could have) estimated
from each request's own tokens-per-character ratio, and the reusable share
``cached / (prompt - fresh)`` beside the all-token share of the same
requests. The fresh tokens are an estimate. ``-`` marks a value the log does
not hold: a server that sends no ``cached_tokens``, the run's first request, or
a log from before the ``prefix`` object (or its ``fresh_chars``) existed.
Standard library only; it changes nothing.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from typing import Any

HEADER = (
    "#",
    "kind",
    "at",
    "prompt",
    "cached",
    "new",
    "reused",
    "against",
    "first changed",
)
KINDS = {"assistant.message": "plan", "reflection": "reflect"}


def read_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    """The ``assistant.message`` and ``reflection`` events of *lines*, in order."""
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
        if isinstance(event, dict) and event.get("type") in KINDS:
            events.append(event)
    return events


Stat = tuple[str, int, "int | None", "float | None"]


def _fresh_tokens(prefix: dict[str, Any] | None, prompt: int | None) -> float | None:
    """The tokens of a request's first-seen content, from its characters."""
    if prefix is None or prompt is None:
        return None
    fresh, size = prefix.get("fresh_chars"), prefix.get("request_chars")
    if not isinstance(fresh, int) or not isinstance(size, int) or size <= 0:
        return None
    return fresh * prompt / size


def _row(event: dict[str, Any], number: int) -> tuple[list[str], Stat]:
    data = event.get("data") or {}
    kind = KINDS[event["type"]]
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    cached = usage.get("cached_tokens")
    prefix = data.get("prefix")
    new = prompt - cached if prompt is not None and cached is not None else None
    reused = against = changed = "-"
    if prefix is not None:
        reused = f"{prefix['messages']} msgs, {prefix['chars']} chars"
        changed = prefix["first_changed"] or "none"
        if prefix.get("tools_changed"):
            changed += " +tools"
        match = prefix.get("against")
        if match:
            against = f"{match['kind']} #{match['request']}"
    at = data.get("step", "?") if kind == "plan" else f"a{data.get('attempt', '?')}"
    cells = [number, kind, at, prompt, cached, new, reused, against, changed]
    return (
        ["-" if c is None else str(c) for c in cells],
        (kind, prompt or 0, cached, _fresh_tokens(prefix, prompt)),
    )


def report(events: list[dict[str, Any]]) -> str:
    """The tables of *events*, one per run, and a total when there are several."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        runs.setdefault(str(event.get("run_id", "?")), []).append(event)
    out: list[str] = []
    overall: list[Stat] = []
    for run_id, run_events in runs.items():
        rows = []
        number = 1
        for event in run_events:
            rows.append(_row(event, number))
            requests = (event.get("data") or {}).get("requests")
            number += requests if isinstance(requests, int) and requests > 1 else 1
        table = [list(HEADER)] + [row[0] for row in rows]
        widths = [max(len(r[i]) for r in table) for i in range(len(HEADER))]
        out.append(f"run {run_id}")
        for r in table:
            if len(r) != len(widths):
                raise ValueError(f"row has {len(r)} cells, the header {len(widths)}")
            out.append("  " + "  ".join(map(str.rjust, r, widths)))
        stats = [stat for _, stat in rows]
        out.extend(_totals(stats, "  "))
        overall.extend(stats)
    if len(runs) > 1:
        out.append(f"all {len(runs)} runs")
        out.extend(_totals(overall, "  "))
    return "\n".join(out)


def _totals(stats: list[Stat], indent: str) -> list[str]:
    """The total line of each kind present (when more than one), overall, and the
    reusable share."""
    kinds = sorted({stat[0] for stat in stats})
    groups = [(k, [s for s in stats if s[0] == k]) for k in kinds if len(kinds) > 1]
    lines = [
        f"{indent}{label}: {_total(group)}"
        for label, group in [*groups, ("total", stats)]
    ]
    return [*lines, f"{indent}reusable: {_reusable(stats)}"]


def _reusable(stats: list[Stat]) -> str:
    """The share of the reusable prompt tokens (prompt minus fresh) that hit."""
    rows = [
        (p, c, f) for _, p, c, f in stats if c is not None and f is not None and p > 0
    ]
    prompt = sum(p for p, _, _ in rows)
    fresh = sum(f for _, _, f in rows)
    cached = sum(c for _, c, _ in rows)
    if not rows or prompt - fresh <= 0:
        return "n/a (needs cached_tokens and fresh_chars, after a run's first request)"
    return (
        f"{len(rows)} requests after the first, {prompt} prompt tokens, "
        f"~{round(fresh)} fresh (estimated), {cached} cached, "
        f"reusable share {100 * cached / (prompt - fresh):.1f}% "
        f"(all-token share of these requests {100 * cached / prompt:.1f}%)"
    )


def _total(stats: list[Stat]) -> str:
    prompt = sum(p for _, p, _, _ in stats)
    cached = sum(c for _, _, c, _ in stats if c is not None)
    reported = sum(p for _, p, c, _ in stats if c is not None)
    line = f"{len(stats)} requests, {prompt} prompt tokens"
    if not reported:
        return line + ", no cached_tokens reported, cached share n/a"
    return (
        f"{line}, {cached} cached, {reported - cached} new "
        f"(of the requests that reported cached_tokens), "
        f"cached share {100 * cached / reported:.1f}%"
    )


def main(argv: list[str]) -> int:
    if any(arg in ("-h", "--help") for arg in argv):
        print(__doc__)
        return 0
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    events: list[dict[str, Any]] = []
    for path in argv:
        with open(path, encoding="utf-8") as handle:
            events.extend(read_events(handle))
    if not events:
        print("no assistant.message or reflection events found", file=sys.stderr)
        return 1
    print(report(events))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
