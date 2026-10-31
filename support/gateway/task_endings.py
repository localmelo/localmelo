"""How a task ended, decided by code from what its own run did.

A worker ends its report with a proposal (``Ending: ready``, ``Ending:
blocked: <what it needs>``, ``Ending: failed``, ``Ending: dropped``). It is a
claim, never evidence: :func:`decide` reads the run's own tool calls (each
``tool.started`` paired with its ``tool.finished`` by call id, in order) and
the folder's git state. In a git repository, ``ready`` becomes ``draft``
(done, not finished) when the run changed code and the last test after its
latest change failed, has a result nobody checked, or never ran, or when a
commit was asked and the run's changes are not committed; a question makes it
``blocked``. A test is a runner the run started in ``python_exec``; a wrapper
that exits 0 proves nothing unless it passes the runner's exit code on
(``sys.exit(...returncode)``, ``check=True``): its result is ``unknown``.
:func:`shown` gives a finished task's ending, or the one its state implies.
"""

from __future__ import annotations

import ast
import asyncio
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from localmelo.melo.agent._brief_aids import FileStamp, edited_files, workspace_stamps
from localmelo.support.sessions.store import SessionStore

from .finishing import OUT_OF_STEPS
from .i18n import t
from .task_store import TaskRecord

STATES: Final = ("ready", "draft", "blocked", "failed", "dropped")
_KEPT: Final = 5
_CHARS: Final = 200
_LINE: Final = re.compile(
    r"^[\s>*_`-]*ending[*_`\s]*[:：][*_`\s]*(ready|blocked|failed|dropped)\b[*_`]*"
    r"\s*(?:[:：—–-]\s*(.*?))?\s*$",
    re.I | re.M,
)
_SEP: Final = r"['\",\s]+"  # between a command's words, in a string or a list
_RUNNER: Final = re.compile(
    rf"\b(?:pytest|py\.test|unittest|tox|nox|jest|vitest)\b|\bnode{_SEP}--test\b"
    rf"|\b(?:npm|pnpm|yarn|bun|cargo|go|swift|deno){_SEP}(?:run{_SEP})?test\b"
    rf"|\bmake{_SEP}(?:test|check)\b",
    re.I,
)
_RUNS: Final = re.compile(
    r"^(?:\w+\.)?(?:run|call|check_call|check_output|Popen|main)$"
)
_EXITS: Final = frozenset({"sys.exit", "exit", "quit", "os._exit", "SystemExit"})
# Code that writes files: a call that has it may have changed code.
_WRITES: Final = re.compile(
    r"\.write_(?:text|bytes)\(|\bopen\([^)]*,\s*(?:mode\s*=\s*)?['\"][^'\"]*[wxa+]"
    r"|\bshutil\.(?:copy|move|rmtree)|\bos\.(?:remove|unlink|rename|replace)\("
    r"|\.(?:unlink|rename)\(|\b(?:black|isort)\b(?![^\n]*--(?:check|diff))"
    r"|\b(?:prettier\b[^\n]*--write|ruff\b[^\n]*(?:format|--fix)|sed\b[^\n]*[\s'\"]-i)\b"
    rf"|\bpatch\s+-p\d|\bgit{_SEP}(?:apply|checkout|restore|stash|reset|merge|rebase"
    r"|cherry-pick|am|pull|mv|rm)\b"
)
_DOC: Final = re.compile(r"\.(?:md|markdown|rst|txt|adoc)$", re.I)
# What running code leaves behind is not work: never owed a commit.
_CACHE: Final = re.compile(
    r"(?:^|/)(?:__pycache__|\.\w+_cache|node_modules|\.tox|\.nox)/|\.py[co]$"
)
_DEFAULTS: Final = {
    "failed": ("failed", "run_failed"),
    "cancelled": ("dropped", "stopped"),
    "expired": ("dropped", "stopped"),
}


def proposal(answer: str) -> tuple[str | None, str]:
    """The ending the report's last ``Ending:`` line proposes, and its question.

    The question is a blocked line's words, else the report's last line when
    it asks something.
    """
    found = list(_LINE.finditer(answer or ""))
    said = found[-1].group(1).lower() if found else None
    if said == "blocked":
        return said, (found[-1].group(2) or "")[:_CHARS]
    lines = [line.strip() for line in report_text(answer).splitlines() if line.strip()]
    asks = bool(lines) and lines[-1].endswith(("?", "？"))
    return said, lines[-1][:_CHARS] if asks else ""


def report_text(answer: str) -> str:
    """*answer* without its ``Ending:`` line, as the person reads it."""
    return _LINE.sub("", answer or "").rstrip()


def _src(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (ValueError, TypeError, AttributeError):
        return ""


def python_test(code: str) -> tuple[str, bool]:
    """``(command, passed on)``: the test runner python_exec *code* starts, or ``""``.

    *passed on*: the runner's exit code becomes the call's, so it is the result.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "", False
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    runs = [c for c in calls if _RUNS.match(_src(c.func)) and _RUNNER.search(_src(c))]
    if not runs:
        return "", False
    command = " ".join(_src(runs[0]).split())[:120]
    for node in calls:
        name, given = _src(node.func), {k.arg: _src(k.value) for k in node.keywords}
        if (
            name.endswith(("check_call", "check_output", "check_returncode"))
            or (node in runs and given.get("check") == "True")
            or (name == "unittest.main" and "exit" not in given)
            or (
                name in _EXITS
                and node.args
                and not isinstance(node.args[0], ast.Constant)
            )
        ):
            return command, True
    for item in ast.walk(tree):
        if not isinstance(item, ast.Assert | ast.If):
            continue
        if "returncode" not in _src(item.test):
            continue
        if isinstance(item, ast.Assert) or any(
            isinstance(n, ast.Raise)
            or (isinstance(n, ast.Call) and _src(n.func) in _EXITS)
            for stmt in item.body
            for n in ast.walk(stmt)
        ):
            return command, True
    return command, False


@dataclass
class _Call:
    start: int
    name: str
    arguments: Mapping[str, Any]
    finish: int | None = None
    status: str = ""
    exit_code: int | None = None

    def test(self) -> tuple[str, str] | None:
        """``(command, outcome)`` of a test run: ``passed``, ``failed`` or ``unknown``."""
        code = self.arguments.get("code")
        if self.name != "python_exec" or not isinstance(code, str):
            return None
        command, passes = python_test(code)
        if not command or self.status == "blocked":
            return None
        if self.finish is None:
            return command, "unknown"
        if self.status != "success" or self.exit_code not in (0, None):
            return command, "failed"
        return command, "passed" if passes else "unknown"

    def writes_code(self) -> bool:
        if self.name in ("file_write", "file_edit"):
            path = str(self.arguments.get("path") or "")
            return self.status not in ("error", "blocked") and not _DOC.search(path)
        code = self.arguments.get("code")
        return self.status != "blocked" and bool(_WRITES.search(str(code or "")))


@dataclass(frozen=True)
class Baseline:
    """The task folder before a run, and what the run before it left owed."""

    stamps: Mapping[str, FileStamp] | None
    head: str = ""
    owed: Mapping[str, Any] = field(default_factory=dict)


def _git(folder: str, *args: str) -> str:
    """A read-only git command's output in *folder*, or ``""``."""
    argv = ["git", "-C", folder, "-c", "core.fsmonitor=false", *args]
    try:
        done = subprocess.run(argv, capture_output=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.decode("utf-8", "replace")


def _head(folder: str) -> str:
    return _git(folder, "rev-parse", "-q", "--verify", "HEAD").strip()


async def baseline(folder: str | None, before: Mapping[str, Any] | None) -> Baseline:
    """The folder's git state now; *before* is the task's last ending, owed unless ready."""
    stamps = await asyncio.to_thread(workspace_stamps, folder) if folder else None
    if stamps is None or folder is None:
        return Baseline(None)
    owed = before if before and before.get("state") in ("draft", "blocked") else {}
    return Baseline(stamps, await asyncio.to_thread(_head, folder), dict(owed))


async def _calls(store: SessionStore, task_id: str, turn: Any) -> list[_Call]:
    """The tool calls of *turn*'s own run, each start paired with its finish."""
    calls: list[_Call] = []
    waiting: dict[str, list[_Call]] = {}
    after = 0
    while True:
        page = await store.list_events(
            task_id, after_id=after, limit=1000, turns=(turn.turn, turn.turn)
        )
        for stored in page:
            kind, data = stored.event.get("type"), stored.event.get("data") or {}
            key = str(data.get("call_id") or "")
            if stored.event.get("run_id") != turn.run_id or not stored.id or not key:
                continue
            if kind == "tool.started":
                given = data.get("arguments")
                given = given if isinstance(given, dict) else {}
                call = _Call(stored.id, str(data.get("name") or ""), given)
                calls.append(call)
                waiting.setdefault(key, []).append(call)
            elif kind == "tool.finished" and waiting.get(key):
                call = waiting[key].pop(0)
                code = data.get("exit_code")
                call.finish, call.status = stored.id, str(data.get("status") or "")
                call.exit_code = code if type(code) is int else None
        if len(page) < 1000:
            return calls
        after = page[-1].id or after


async def _evidence(
    store: SessionStore,
    task_id: str,
    turn: Any,
    folder: str,
    base: Baseline,
    commit: bool,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Why the run's work is not finished, the tests it ran, its uncommitted files."""
    assert base.stamps is not None
    calls = await _calls(store, task_id, turn)
    now = await asyncio.to_thread(workspace_stamps, folder) or {}
    edited = {p for p in edited_files(base.stamps, now) if not _CACHE.search(p)}
    own = edited | {p for p in base.owed.get("uncommitted", ()) if p in now}
    head = await asyncio.to_thread(_head, folder)
    committed: list[str] = []
    if head and head != base.head:  # committed during the run: its changes count too
        since = base.head or "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # empty tree
        names = await asyncio.to_thread(
            _git, folder, "diff", "--name-only", "-z", since, head
        )
        committed = [name for name in names.split("\0") if name]
    tests = [(call, found) for call in calls if (found := call.test())]
    writes = [call.finish for call in calls if call.writes_code()]
    last: float | None = None
    if writes and any(not _DOC.search(p) for p in [*edited, *committed]):
        last = max(float("inf") if at is None else at for at in writes)
    elif any(not _DOC.search(p) for p in edited):
        # Code changed and no call says how: the last call that may have done it.
        maybe = [c for c in calls if c.name == "python_exec" and not c.test()]
        last = float(maybe[-1].finish or maybe[-1].start) if maybe else None
    if any(str(r).startswith("tests_") for r in base.owed.get("reasons", ())):
        last = max(last or 0.0, 0.0)  # the run before changed code it did not test
    reasons = []
    if last is not None:
        fresh = [found[1] for call, found in tests if call.start > last]
        failed = {"failed": "tests_failed", "unknown": "tests_unverified"}
        reasons += [failed[fresh[-1]]] if fresh and fresh[-1] in failed else []
        reasons += [] if fresh else ["tests_missing"]
    if commit and own:
        reasons.append("commit_missing")
    kept = [
        {"command": command, "exit_code": call.exit_code, "outcome": outcome}
        for call, (command, outcome) in tests[-_KEPT:]
    ]
    return reasons, kept, sorted(own)[:20]


async def decide(
    store: SessionStore,
    task_id: str,
    *,
    phase: str,
    failure_reason: str,
    folder: str | None,
    base: Baseline,
    asks_commit: bool,
) -> dict[str, Any] | None:
    """The ending of a run that ended in *phase*, or ``None`` for its state's default."""
    out_of_steps = phase == "failed" and failure_reason == OUT_OF_STEPS
    if phase != "done" and not out_of_steps:
        return None
    turns = await store.list_turns(task_id, limit=1)
    turn = turns[-1] if turns else None
    proposed, question = proposal(turn.answer if turn is not None else "")
    reasons: list[str] = []
    tests: list[dict[str, Any]] = []
    uncommitted: list[str] = []
    if turn is not None and folder and base.stamps is not None:
        found = await _evidence(store, task_id, turn, folder, base, asks_commit)
        reasons, tests, uncommitted = found
    if out_of_steps:
        state, reasons = "draft", ["out_of_steps", *reasons]
    elif proposed in ("failed", "dropped"):
        state = str(proposed)  # a worker may give up; code never raises a claim
    elif proposed == "blocked" or question:
        state, reasons = "blocked", ["question", *reasons]
    else:
        state = "draft" if reasons else "ready"
    return {
        "state": state,
        "proposed": proposed,
        "reasons": reasons,
        "question": question,
        "tests": tests,
        "uncommitted": uncommitted,
        "turn": turn.turn if turn is not None else None,
        "run_id": turn.run_id if turn is not None else "",
    }


def finish_message(ending: Mapping[str, Any]) -> str:
    """A draft's one-tap instruction: run the tests, and commit when that is missing."""
    if "commit_missing" in ending.get("reasons", ()):
        return t("ending.finishCommitMessage")
    return t("ending.finishMessage")


def shown(record: TaskRecord) -> dict[str, Any] | None:
    """A finished task's ending: decided at its last transition, else its state's."""
    ending = getattr(record, "ending", None)  # a stand-in record may have none
    ended = record.state == "done" or record.state in _DEFAULTS
    if ended and isinstance(ending, Mapping) and ending.get("at") == record.finished_at:
        return dict(ending)
    if record.state == "failed" and record.wait_reason == OUT_OF_STEPS:
        return {"state": "draft", "reasons": ["out_of_steps"]}
    found = _DEFAULTS.get(record.state)
    return None if found is None else {"state": found[0], "reasons": [found[1]]}
