"""``melo-procedures``: the procedural store's offline tool.

Four commands::

    melo-procedures episodes --history DB [--model M] [--key K] [--place P]
                             [--result-json FILE]
    melo-procedures replay FILE... [--protocol loo|first-k|online] [--arms]
                           [--denominator D] [--spread S] [--drop D]
                           [--stats S] [--key-build]
    melo-procedures export [--db FILE | --memory-dir DIR]
    melo-procedures forget (--routine ID | --all) [--db FILE | --memory-dir DIR]

``episodes`` turns a run's ``history.db`` into rows of the export format;
``replay`` measures the progress rules on such rows, offline; ``export``
prints every row of a ``procedures.db``; ``forget`` deletes rows from it.
``docs/procedures.md`` documents the formats, the output and the exit
codes: 0 on success, 2 for usage and input errors, 130 on Ctrl-C.

The command calls no model and reads no config file; only ``forget``
writes, and only to the store it names.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from localmelo.melo.memory.procedures.classify import PhaseState, phase_agrees
from localmelo.melo.memory.procedures.normalize import (
    RoutineKey,
    key_for_query,
    length_options,
    normalize_query,
    stored_query_text,
    template_id,
)
from localmelo.melo.memory.procedures.store import (
    TABLES,
    MemoryProcedures,
    Routine,
    RunFacts,
    SqliteProcedures,
    episode_from_row,
    episode_row,
    routine_from_row,
    routine_row,
)
from localmelo.melo.memory.procedures.template import (
    DENOMINATORS,
    DROPS,
    STATISTICS,
    Episode,
    Params,
    TemplateView,
    compute_stats,
)
from localmelo.melo.memory.procedures.trace import Trace, trace_from_segments
from localmelo.melo.memory.procedures.tracker import LONGER, bar_series

PROG = "melo-procedures"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

DB_FILE = "procedures.db"
DEFAULT_MEMORY_DIR = "~/.cache/localmelo/memory"  # as the agent's default
PROTOCOLS: Final = ("loo", "first-k", "online")

# Section 10's thresholds, checked on the online protocol's level-2 runs.
MAX_BAR_ERROR: Final = 0.15
MIN_SEPARATION: Final = 0.05
MAX_CAP_SHARE: Final = 0.05
MAX_END_JUMP: Final = 0.30
LONGER_MARGIN: Final = 0.05
MIN_PHASE_AGREEMENT: Final = 0.9
# first-k judges each later run of a group on its first runs only.
FIRST_K: Final = 2
# loo needs a group of at least this many runs.
LOO_MIN_GROUP: Final = 3


class _InputError(Exception):
    """An input the command cannot use, reported in one line with exit code 2."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``melo-procedures`` with *argv* (default: ``sys.argv[1:]``).

    Returns the exit code.
    """
    try:
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:  # --help, or a usage error argparse printed
            return exc.code if isinstance(exc.code, int) else EXIT_USAGE
        try:
            if args.command == "episodes":
                return _episodes(args)
            if args.command == "replay":
                return _replay(args)
            if args.command == "export":
                return asyncio.run(_export(args))
            return asyncio.run(_forget(args))
        except _InputError as exc:
            print(f"{PROG}: error: {exc}", file=sys.stderr)
            return EXIT_USAGE
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


# ── Flags ──


def _spread(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a number > 0, got {text}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Build, replay, export and forget procedural memory.",
        epilog="Formats, output lines and exit codes: docs/procedures.md",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    episodes = commands.add_parser(
        "episodes", help="print a run's history.db as episode rows"
    )
    episodes.add_argument(
        "--history", required=True, metavar="DB", help="a history.db, or its directory"
    )
    episodes.add_argument("--model", default="", help="the run's model id")
    episodes.add_argument(
        "--key", metavar="K", help="the routine key (default: from the query)"
    )
    episodes.add_argument(
        "--place", default="", metavar="P", help="the run's place (default: none)"
    )
    episodes.add_argument(
        "--result-json",
        metavar="FILE",
        help="the run's --output-format json line: steps, status, config",
    )

    replay = commands.add_parser("replay", help="measure the progress rules")
    replay.add_argument("files", nargs="+", metavar="FILE", help="export-format JSONL")
    replay.add_argument("--protocol", choices=PROTOCOLS, default="online")
    replay.add_argument(
        "--arms",
        action="store_true",
        help="online: also each arm of section 10 that changes one default",
    )
    replay.add_argument(
        "--denominator",
        choices=DENOMINATORS,
        help="the bar's rule (online default: dynamic; loo, first-k: all)",
    )
    replay.add_argument("--spread", type=_spread, help="the gate's spread (0.5)")
    replay.add_argument("--drop", choices=DROPS, help="the exploration drop")
    replay.add_argument("--stats", choices=STATISTICS, help="the statistics' runs")
    replay.add_argument(
        "--key-build",
        action="store_true",
        help="put each episode's harness_version in its template's key",
    )

    for name, text in (
        ("export", "print every row of a procedures.db"),
        ("forget", "delete a routine, or everything, from a procedures.db"),
    ):
        command = commands.add_parser(name, help=text)
        where = command.add_mutually_exclusive_group()
        where.add_argument("--db", metavar="FILE", help="the procedures.db")
        where.add_argument(
            "--memory-dir",
            metavar="DIR",
            help="the memory directory (default: $LOCALMELO_MEMORY_DIR, else "
            f"{DEFAULT_MEMORY_DIR})",
        )
        if name == "forget":
            which = command.add_mutually_exclusive_group(required=True)
            which.add_argument("--routine", metavar="ID", help="a routine_id")
            which.add_argument("--all", action="store_true", help="every row")
    return parser


# ── export, forget ──


def _db_path(args: argparse.Namespace) -> str:
    if args.db is not None:
        return str(args.db)
    directory = args.memory_dir or os.environ.get(
        "LOCALMELO_MEMORY_DIR", os.path.expanduser(DEFAULT_MEMORY_DIR)
    )
    return os.path.join(directory, DB_FILE)


async def _export(args: argparse.Namespace) -> int:
    path = _db_path(args)
    if not os.path.isfile(path):
        raise _InputError(f"{path} does not exist")
    store = SqliteProcedures(path, create=False)
    try:
        rows = await store.export_rows()
    except (sqlite3.Error, ValueError, RuntimeError) as exc:
        raise _InputError(f"cannot read {path}: {exc}") from None
    finally:
        await store.aclose()
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return EXIT_OK


async def _forget(args: argparse.Namespace) -> int:
    path = _db_path(args)
    if not os.path.isfile(path):
        raise _InputError(f"{path} does not exist")
    store = SqliteProcedures(path, create=False)
    try:
        deleted = await store.forget(None if args.all else args.routine)
    except (sqlite3.Error, ValueError, RuntimeError) as exc:
        raise _InputError(f"cannot change {path}: {exc}") from None
    finally:
        await store.aclose()
    print(json.dumps({"deleted": deleted}))
    return EXIT_OK


# ── episodes ──


def _read_result(path: str) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise _InputError(f"cannot read {path}: {exc.strerror or exc}") from None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            break
        if isinstance(value, dict):
            return value
        break
    raise _InputError(f"{path} does not end with a JSON object")


def _history_path(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "history.db")
    if not os.path.isfile(path):
        raise _InputError(f"{path} does not exist")
    return path


def _step_map(
    conn: sqlite3.Connection, task_id: str
) -> tuple[dict[str, tuple[int, int]], list[int], int, int]:
    """From ``llm_calls``: each step row's (step, attempt), the reflections'
    steps, the last planning step and the highest attempt."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    if "llm_calls" not in tables:
        return {}, [], 0, 0
    rows = conn.execute(
        "SELECT kind, step, attempt, step_ids_json FROM llm_calls "
        "WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    mapping: dict[str, tuple[int, int]] = {}
    reflections: list[int] = []
    last_step = attempts = 0
    for kind, step, attempt, step_ids_json in rows:
        step, attempt = int(step or 0), int(attempt or 0)
        attempts = max(attempts, attempt)
        if kind == "reflect":
            reflections.append(step)
        elif kind == "plan":
            last_step = max(last_step, step)
            step_ids = json.loads(step_ids_json or "[]")
            if not isinstance(step_ids, list):
                raise ValueError(f"step_ids_json of task {task_id} is not a list")
            for step_id in step_ids:
                mapping[str(step_id)] = (step, attempt)
    return mapping, reflections, last_step, attempts


def _episode_lines(args: argparse.Namespace) -> list[dict[str, Any]]:
    result = _read_result(args.result_json) if args.result_json else {}
    path = _history_path(args.history)
    lines: list[dict[str, Any]] = []
    try:
        uri = f"{Path(path).resolve().as_uri()}?mode=ro"
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as conn:
            tasks = conn.execute(
                "SELECT task_id, query, status FROM tasks ORDER BY rowid"
            ).fetchall()
            if result.get("task_id"):
                tasks = [task for task in tasks if task[0] == result["task_id"]]
                if not tasks:
                    raise _InputError(f"task {result['task_id']} is not in the history")
            elif result and len(tasks) > 1:
                # The result describes one run: which one is unknown.
                raise _InputError(
                    f"{args.result_json} names no task_id, and the history "
                    f"holds {len(tasks)} tasks"
                )
            for task_id, query, status in tasks:
                lines.extend(_task_lines(conn, args, result, task_id, query, status))
    except (sqlite3.Error, ValueError, TypeError) as exc:
        # A file that is not a history, or one whose rows are damaged.
        raise _InputError(f"cannot read {path}: {exc}") from None
    return lines


def _task_lines(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    result: Mapping[str, Any],
    task_id: str,
    query: str,
    status: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT step_id, thought, tool_call_json, tool_result_json, timestamp "
        "FROM steps WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    mapping, reflections, last_plan_step, top_attempt = _step_map(conn, task_id)
    trace = Trace(clock=lambda: 0.0)
    reflect_after = sorted(reflections)
    step = 0
    attempt = 1
    previous_thought: str | None = None

    def start_steps(to_step: int, to_attempt: int) -> None:
        nonlocal step, attempt
        for k in range(step + 1, to_step + 1):
            # A reflection that ran after step j comes before step j + 1.
            while reflect_after and reflect_after[0] < k:
                reflect_after.pop(0)
                trace.reflection()
            trace.step(k, to_attempt)
        step, attempt = max(step, to_step), max(attempt, to_attempt)

    for step_id, thought, call_json, result_json, _ in rows:
        if mapping:
            row_step, row_attempt = mapping.get(step_id, (max(step, 1), attempt))
        else:
            # Without llm_calls, consecutive rows with one thought are one step.
            new = step == 0 or thought != previous_thought
            row_step, row_attempt = step + new, attempt
        previous_thought = thought
        start_steps(row_step, row_attempt)
        if not call_json:
            continue
        call = json.loads(call_json)
        outcome = json.loads(result_json) if result_json else {}
        if not isinstance(call, dict) or not isinstance(outcome, dict):
            raise ValueError(f"a step of task {task_id} is not a JSON object")
        error = str(outcome.get("error") or "")
        trace.tool(
            str(call.get("tool_name", "")),
            call.get("arguments") or {},
            "error" if error else "success",
            str(outcome.get("output") or ""),
        )
    total = int(result.get("steps") or max(step, last_plan_step))
    start_steps(total, attempt)
    timestamps = [row[4] for row in rows if row[4] is not None]
    duration = result.get("duration_s")
    if duration is None:
        duration = (max(timestamps) - min(timestamps)) if timestamps else 0.0
    if args.key:
        key = RoutineKey(
            key=args.key, place=args.place, name=None, query_text=_text(query)
        )
    else:
        key = key_for_query(query or "", args.place)
    routine_id = hashlib.sha256(key.key.encode()).hexdigest()[:32]
    options = length_options(result.get("config") or {})
    facts = RunFacts.from_trace(
        trace,
        task_id=task_id,
        key=key,
        model=args.model,
        options=options,
        status=str(result.get("status") or status),
        steps=total,
        attempts=int(result.get("attempts") or max(top_attempt, attempt)),
        duration_s=float(duration),
        harness_version=str(result.get("localmelo_version") or "unknown"),
    )
    routine = Routine(
        routine_id=routine_id,
        key=key.key,
        name=None,
        query_text=key.query_text,
        place=key.place,
        created_at=min(timestamps) if timestamps else 0.0,
    )
    episode = Episode(
        task_id=task_id,
        routine_id=routine_id,
        template_id=template_id(routine_id, args.model, options),
        model=args.model,
        harness_version=facts.harness_version,
        status=facts.status,
        steps=facts.steps,
        attempts=facts.attempts,
        n_tool=int(result.get("n_tool") or facts.n_tool),
        duration_s=facts.duration_s,
        first_pass_step=facts.first_pass_step,
        cycles=facts.cycles,
        segments=facts.segments,
        created_at=routine.created_at,
    )
    return [
        {"table": "routines", **routine_row(routine)},
        {"table": "episodes", **episode_row(episode)},
    ]


def _text(query: str | None) -> str:
    return stored_query_text(normalize_query(query or ""))


def _episodes(args: argparse.Namespace) -> int:
    for line in _episode_lines(args):
        print(json.dumps(line, ensure_ascii=False))
    return EXIT_OK


# ── replay ──


@dataclass
class _Data:
    routines: dict[str, Routine] = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)


def load_rows(paths: Iterable[str]) -> _Data:
    """The routines and episodes in export-format files; templates are skipped.

    Raises ``_InputError`` naming the file and line of a bad row.
    """
    data = _Data()
    seen: set[str] = set()
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise _InputError(f"cannot read {path}: {exc.strerror or exc}") from None
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not a JSON object")
                table = row.get("table")
                if table not in TABLES:
                    raise ValueError(f"unknown table {table!r}")
                if table == "routines":
                    routine = routine_from_row(row)
                    data.routines.setdefault(routine.routine_id, routine)
                elif table == "episodes":
                    episode = episode_from_row(row, strict=True)
                    if episode.task_id in seen:
                        raise ValueError(f"task {episode.task_id} appears twice")
                    seen.add(episode.task_id)
                    data.episodes.append(episode)
            except (KeyError, TypeError, ValueError) as exc:
                raise _InputError(f"{where}: {exc}") from None
    order = {id(e): i for i, e in enumerate(data.episodes)}
    data.episodes.sort(key=lambda e: (e.created_at, order[id(e)]))
    return data


def _group_key(episode: Episode, key_build: bool) -> str:
    return (
        f"{episode.template_id}\0{episode.harness_version}"
        if key_build
        else episode.template_id
    )


def _trace_of(episode: Episode) -> Trace:
    return trace_from_segments(
        episode.segments, episode.steps, first_pass_step=episode.first_pass_step
    )


def _view(stats_from: Sequence[Episode], params: Params) -> TemplateView:
    """A level-2 view whose statistics are *stats_from*, with no drop and no gate."""
    open_params = replace(params, drop="off", stats="normal", min_episodes=1)
    stats = compute_stats(stats_from, None, open_params)
    return TemplateView(
        level=2,
        reason="",
        match="exact",
        routine_id="",
        template_id="",
        version=0,
        model="",
        stats=stats,
        skeleton=(),
        params=params,
    )


@dataclass
class _Bars:
    """The bar errors, cap shares and end jumps of a set of runs."""

    errors: list[float] = field(default_factory=list)
    caps: list[float] = field(default_factory=list)
    jumps: list[float] = field(default_factory=list)
    backward: int = 0
    longer: int = 0
    longer_expected: list[float] = field(default_factory=list)

    def add(self, view: TemplateView, episode: Episode) -> tuple[float, str | None]:
        series, off = bar_series(view, _trace_of(episode))
        n = episode.steps
        series = (series + [series[-1] if series else 0.0] * n)[:n]
        error = statistics.fmean(abs(s - k / n) for k, s in enumerate(series, start=1))
        cap = view.params.cap
        self.errors.append(error)
        self.caps.append(sum(s >= cap - 1e-9 for s in series) / n)
        self.jumps.append(1 - series[-1])
        self.backward += sum(
            b < a - 1e-12 for a, b in zip(series, series[1:], strict=False)
        )
        self.longer += off == LONGER
        self.longer_expected.append(1 / (len(view.stats.steps) + 1))
        return error, off

    def summary(self) -> dict[str, Any]:
        if not self.errors:
            return {"runs": 0}
        return {
            "runs": len(self.errors),
            "bar_error": _round(statistics.fmean(self.errors)),
            "cap_share": _round(statistics.fmean(self.caps)),
            "end_jump": _round(statistics.fmean(self.jumps)),
            "backward_moves": self.backward,
            "longer_rate": _round(self.longer / len(self.errors)),
            "longer_expected": _round(statistics.fmean(self.longer_expected)),
        }


def _round(value: float) -> float:
    return round(value, 4)


def _params(args: argparse.Namespace, **changes: Any) -> Params:
    base: dict[str, Any] = {}
    for name in ("denominator", "spread", "drop", "stats"):
        value = getattr(args, name, None)
        if value is not None:
            base[name] = value
    base.update(changes)
    return Params(**base)


def _groups(data: _Data, key_build: bool) -> dict[str, list[Episode]]:
    groups: dict[str, list[Episode]] = defaultdict(list)
    for episode in data.episodes:
        if episode.steps > 0:
            groups[_group_key(episode, key_build)].append(episode)
    return groups


def replay_loo(data: _Data, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Each run of a group of 3 or more, predicted from the group's others."""
    lines = []
    denominators = [args.denominator] if args.denominator else list(DENOMINATORS)
    groups = _groups(data, args.key_build)
    for denominator in denominators:
        params = _params(args, denominator=denominator)
        bars = _Bars()
        for runs in groups.values():
            if len(runs) < LOO_MIN_GROUP:
                continue
            for i, run in enumerate(runs):
                bars.add(_view(runs[:i] + runs[i + 1 :], params), run)
        lines.append({"protocol": "loo", "denominator": denominator, **bars.summary()})
    return lines


def replay_first_k(data: _Data, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every later run of a group, on the group's first two runs, held fixed."""
    lines = []
    denominators = [args.denominator] if args.denominator else list(DENOMINATORS)
    groups = _groups(data, args.key_build)
    for denominator in denominators:
        params = _params(args, denominator=denominator)
        shown, held = _Bars(), _Bars()
        for runs in groups.values():
            if len(runs) <= FIRST_K:
                continue
            prior = runs[:FIRST_K]
            view = _view(prior, params)
            gate = view.stats.spread <= params.spread
            for run in runs[FIRST_K:]:
                (shown if gate else held).add(view, run)
        lines.append(
            {
                "protocol": "first-k",
                "denominator": denominator,
                "spread": params.spread,
                "shown": shown.summary(),
                "held": held.summary(),
                **_separation(shown, held),
            }
        )
    return lines


def _separation(shown: _Bars, held: _Bars) -> dict[str, Any]:
    if not shown.errors or not held.errors:
        return {"separation": None}
    return {
        "separation": _round(
            statistics.fmean(held.errors) - statistics.fmean(shown.errors)
        )
    }


def _phase_agreement(episodes: Sequence[Episode]) -> float | None:
    agree = total = 0
    for episode in episodes:
        state = PhaseState()
        by_step: dict[int, set[str]] = defaultdict(set)
        phase_at: dict[int, str | None] = {}
        for call in _trace_of(episode).calls:
            state.add(call.cls, call.passed)
            by_step[call.step].add(call.cls)
            phase_at[call.step] = state.phase
        for step, classes in by_step.items():
            total += 1
            agree += phase_agrees(phase_at[step], classes)
    return None if not total else agree / total


async def replay_online(
    data: _Data, args: argparse.Namespace, params: Params, arm: str
) -> dict[str, Any]:
    """Every episode in time order through the learner and the tracker."""
    store = MemoryProcedures(params=params)
    shown, held = _Bars(), _Bars()
    by_prior: dict[int, list[float]] = defaultdict(list)
    prior_stats: dict[str, list[str]] = defaultdict(list)
    levels = {0: 0, 1: 0, 2: 0}
    demoted = 0
    statuses = params.statistics_statuses
    for episode in data.episodes:
        routine = data.routines.get(episode.routine_id)
        key = (
            routine.routine_key()
            if routine is not None
            else RoutineKey(key=episode.routine_id, place="", name=None, query_text="")
        )
        options: dict[str, Any] = {"template": episode.template_id}
        if args.key_build:
            options["build"] = episode.harness_version
        group = template_id(key.key, episode.model, options)
        view = await store.lookup(key, model=episode.model, options=options)
        level = 0 if view is None else view.level
        levels[level] += 1
        prior = prior_stats[group]
        in_population = (
            episode.steps > 0
            and len(prior) >= params.min_episodes
            and "completed" in prior
        )
        if view is not None and view.reason == "model demoted":
            demoted += 1
        if in_population and view is not None:
            if level == 2:
                error, _ = shown.add(view, episode)
                by_prior[len(view.stats.steps)].append(error)
            else:
                held.add(view, episode)
        await store.learn(
            RunFacts(
                task_id=episode.task_id,
                key=key,
                model=episode.model,
                options=options,
                status=episode.status,
                steps=episode.steps,
                attempts=episode.attempts,
                duration_s=episode.duration_s,
                segments=episode.segments,
                first_pass_step=episode.first_pass_step,
                cycles=episode.cycles,
                n_tool=episode.n_tool,
                harness_version=episode.harness_version,
                created_at=episode.created_at,
            ),
            view=view,
        )
        if episode.status in statuses and episode.steps > 0:
            prior.append(episode.status)
    await store.aclose()
    summary = shown.summary()
    held_summary = held.summary()
    separation = _separation(shown, held)["separation"]
    phase = _phase_agreement(data.episodes)
    checks: dict[str, bool | None] = {}
    if shown.errors:
        expected = statistics.fmean(shown.longer_expected) + LONGER_MARGIN
        checks = {
            "bar_error": summary["bar_error"] <= MAX_BAR_ERROR,
            # null when no run was held back: nothing to compare.
            "separation": (
                None if separation is None else separation >= MIN_SEPARATION
            ),
            "backward_moves": summary["backward_moves"] == 0,
            "cap_share": summary["cap_share"] <= MAX_CAP_SHARE,
            "end_jump": summary["end_jump"] <= MAX_END_JUMP,
            "longer_rate": summary["longer_rate"] <= expected,
        }
    if phase is not None:
        checks["phase_agreement"] = phase >= MIN_PHASE_AGREEMENT
    return {
        "protocol": "online",
        "arm": arm,
        "params": params.as_dict(),
        "key_build": bool(args.key_build),
        "runs": len(data.episodes),
        "levels": {str(k): v for k, v in levels.items()},
        "model_demoted": demoted,
        "level2": summary,
        "held": held_summary,
        "separation": separation,
        "level2_by_prior_runs": {
            str(n): {"runs": len(errors), "bar_error": _round(statistics.fmean(errors))}
            for n, errors in sorted(by_prior.items())
        },
        "phase_agreement": None if phase is None else _round(phase),
        "thresholds": checks,
    }


def _arms(base: Params) -> list[tuple[str, Params]]:
    """The defaults, then each arm of section 10 that changes one of them."""
    arms = [("default", base)]
    for denominator in DENOMINATORS:
        if denominator != base.denominator:
            arms.append(
                (f"denominator={denominator}", replace(base, denominator=denominator))
            )
    for spread in (0.4, 0.5, 0.6):
        if spread != base.spread:
            arms.append((f"spread={spread}", replace(base, spread=spread)))
    for drop in DROPS:
        if drop != base.drop:
            arms.append((f"drop={drop}", replace(base, drop=drop)))
    for stats in STATISTICS:
        if stats != base.stats:
            arms.append((f"stats={stats}", replace(base, stats=stats)))
    return arms


def _replay(args: argparse.Namespace) -> int:
    data = load_rows(args.files)
    if not data.episodes:
        raise _InputError("no episodes in the input")
    if args.protocol == "loo":
        lines = replay_loo(data, args)
    elif args.protocol == "first-k":
        lines = replay_first_k(data, args)
    else:
        base = _params(args)
        arms = _arms(base) if args.arms else [("default", base)]
        lines = [
            asyncio.run(replay_online(data, args, params, arm)) for arm, params in arms
        ]
    for line in lines:
        print(json.dumps(line, ensure_ascii=False))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
