"""How a task ended, decided by code from what its own run did; fail closed.

A worker ends its report with a proposal (``Ending: ready``, ``Ending:
blocked: <what it needs>``, ``Ending: failed``, ``Ending: dropped``). It is a
claim, never evidence: :func:`decide` reads the run's own tool calls (each
``tool.started`` paired with its ``tool.finished`` by call id, in order) and
the folder's git state against the :class:`Baseline` of the first run since
the task last ended ready. Ready needs proof; any doubt leaves ``draft`` with
its reason:

- ``tests_failed`` / ``tests_unverified``: the run's last test failed, or its
  result is unknown (in git or not);
- ``tests_missing``: code changed since the baseline and no test started after
  the latest call that could have changed it;
- ``commit_missing``: a commit was asked and changes since the baseline are not
  committed;
- ``unchecked``: git could not be read, or the run grew ``git stash list``.

A test is one shape only: a plain test-runner command (no shell operators) or a
``python_exec`` that is imports plus one ``sys.exit(subprocess.run([...]).
returncode)``, ``subprocess.run([...], check=True)`` or
``subprocess.check_call([...])``. Its own exit code is the result; any other
call that starts a runner is a test with an ``unknown`` result. A question
makes it ``blocked``. :func:`shown` gives a finished task's ending.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from localmelo.melo.agent._brief_aids import FileStamp, edited_files, workspace_stamps
from localmelo.melo.checker.redaction import redact_secrets
from localmelo.support.sessions.store import SessionStore

from .finishing import OUT_OF_STEPS
from .i18n import t
from .task_store import TaskRecord

STATES: Final = ("ready", "draft", "blocked", "failed", "dropped")
# The reasons a Finish it can settle: work left untested, uncommitted or unread.
OWED: Final = frozenset(
    {"tests_failed", "tests_missing", "tests_unverified", "commit_missing", "unchecked"}
)
_KEPT: Final = 5
_CHARS: Final = 200
_LINE: Final = re.compile(
    r"^[\s>*_`-]*ending[*_`\s]*[:：][*_`\s]*(ready|blocked|failed|dropped)\b[*_`]*"
    r"\s*(?:[:：—–-]\s*(.*?))?\s*$",
    re.I | re.M,
)
_OPERATORS: Final = re.compile(r"[|&;<>`]|\$\(")
_NOT_A_RUN: Final = frozenset({"--version", "--collect-only", "--co", "-h", "--help"})
_STARTS: Final = re.compile(
    r"^(?:subprocess\.(?:run|call|check_call|check_output|Popen|getoutput|getstatusoutput)"
    r"|os\.(?:system|popen|exec\w*|spawn\w*))$"
)
_KEYWORDS: Final = frozenset(
    {
        "check",
        "capture_output",
        "text",
        "timeout",
        "encoding",
        "errors",
        "stdout",
        "stderr",
    }
)
_READ_TOOLS: Final = frozenset(
    {"file_read", "list_files", "git_status", "search_files", "ask_owner"}
)
# git subcommands that never change a file's content (add and commit change refs only).
_GIT_READS: Final = frozenset(
    {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "branch"}
    | {"remote", "describe", "shortlog", "add", "commit"}
)
_DOC: Final = re.compile(r"\.(?:md|markdown|rst|adoc)$|(?:^|/)docs/", re.I)
# What running code leaves behind is not work: never owed a commit.
_CACHE: Final = re.compile(
    r"(?:^|/)(?:__pycache__|\.\w+_cache|node_modules|\.tox|\.nox)/|\.py[co]$"
)
_DEFAULTS: Final = {
    "failed": ("failed", "run_failed"),
    "cancelled": ("dropped", "stopped"),
    "expired": ("dropped", "stopped"),
}
_ENV: Final = {"LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}


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


def _runner(argv: list[str]) -> tuple[bool, bool]:
    """``(starts a test runner, on this folder only)`` for a command's words."""
    words = argv[2:] if argv[:2] == ["uv", "run"] else argv
    first = os.path.basename(words[0]) if words else ""
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", first) and words[1:2] == ["-m"]:
        runs, rest = words[2:3] in (["pytest"], ["unittest"]), words[3:]
    elif first in ("pytest", "py.test"):
        runs, rest = True, words[1:]
    elif words[1:2] == ["test"] and first in ("npm", "pnpm", "yarn", "cargo", "go"):
        runs, rest = True, words[2:]
    elif first in ("npm", "pnpm", "yarn") and words[1:3] == ["run", "test"]:
        runs, rest = True, words[3:]
    else:
        pair = {("node", "--test"), ("swift", "test")}
        runs, rest = (first, words[1] if len(words) > 1 else "") in pair, words[2:]
    if not runs or _NOT_A_RUN & set(rest):
        return False, False
    here = all(
        not word.startswith(("/", "~")) and ".." not in word.split("/") for word in rest
    )
    return True, here


def _words(node: ast.AST) -> list[str] | None:
    """A literal command's words (``sys.executable`` reads as python), or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return shlex.split(node.value)
        except ValueError:
            return node.value.split()
    if not isinstance(node, ast.List | ast.Tuple):
        return None
    words = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            words.append(item.value)
        elif ast.unparse(item) == "sys.executable":
            words.append("python")
        else:
            return None
    return words


def _plain_call(call: ast.Call, check: bool) -> list[str] | None:
    """The words of ``subprocess.<fn>([literal argv], <plain keywords>)``, or None."""
    given = {k.arg: k.value for k in call.keywords}
    plain = all(
        name in _KEYWORDS
        and (
            isinstance(value, ast.Constant)
            or ast.unparse(value).startswith("subprocess.")
        )
        for name, value in given.items()
    )
    if not plain or len(call.args) != 1 or not isinstance(call.args[0], ast.List):
        return None
    if check and ("check" not in given or ast.unparse(given["check"]) != "True"):
        return None
    return _words(call.args[0])


def _shaped(tree: ast.Module) -> list[str] | None:
    """The command of the one statement a test's python_exec may be, or None."""
    imports = [s for s in tree.body if isinstance(s, ast.Import)]
    body = [s for s in tree.body if s not in imports]
    names = {alias.name for s in imports for alias in s.names}
    if names - {"subprocess", "sys"} or len(body) != 1:
        return None
    statement = body[0]
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call, name = statement.value, ast.unparse(statement.value.func)
    if name == "sys.exit" and len(call.args) == 1 and not call.keywords:
        inner = call.args[0]
        if not (isinstance(inner, ast.Attribute) and inner.attr == "returncode"):
            return None
        run = inner.value
        if isinstance(run, ast.Call) and ast.unparse(run.func) == "subprocess.run":
            return _plain_call(run, check=False)
        return None
    if name in ("subprocess.run", "subprocess.check_call"):
        return _plain_call(call, check=name == "subprocess.run")
    return None


def python_test(code: str) -> tuple[str, bool]:
    """``(command, passed on)``: the test runner python_exec *code* starts, or ``""``.

    *passed on*: the code is the one shape whose exit code is the runner's own.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "", False
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name in ("pytest.main", "unittest.main"):
            found.append(ast.unparse(node))
        elif _STARTS.match(name) and node.args:
            words = _words(node.args[0])
            if words and _runner(words)[0]:
                found.append(" ".join(words))
    if not found:
        return "", False
    shaped = _shaped(tree)
    passes = shaped is not None and _runner(shaped) == (True, True)
    return " ".join(found[0].split())[:120], passes


def _shell_test(command: str) -> tuple[str, bool]:
    """``(command, passed on)`` of a shell command: one plain runner command passes on."""
    first = _OPERATORS.split(command, maxsplit=1)[0]
    try:
        words = shlex.split(first)
    except ValueError:
        words = first.split()
    runs, here = _runner(words)
    if not runs:
        return "", False
    return " ".join(command.split())[:120], here and not _OPERATORS.search(command)


def _git_call(node: ast.AST) -> bool:
    """A plain subprocess call of a git subcommand that changes no file."""
    if not isinstance(node, ast.Call) or not _STARTS.match(ast.unparse(node.func)):
        return False
    words = _plain_call(node, check=False)
    words = words[1:] if words and words[0] == "git" else None
    while words and words[0] in ("-C", "-c"):
        words = words[2:]
    return words is not None and bool(words) and words[0] in _GIT_READS


def _reads_only(code: str) -> bool:
    """Code that can change no file: git reads, commits and prints, nothing else."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            if {a.name for a in stmt.names} - {"subprocess", "sys", "json"}:
                return False
        elif isinstance(stmt, ast.Assign | ast.Expr) and _git_call(stmt.value):
            continue
        elif not isinstance(stmt, ast.Expr) or not _printed(stmt.value):
            return False
    return True


def _printed(node: ast.AST) -> bool:
    """``print(...)`` of values and git reads only."""
    if not isinstance(node, ast.Call) or ast.unparse(node.func) != "print":
        return False
    inner = [n for arg in node.args for n in ast.walk(arg) if isinstance(n, ast.Call)]
    return all(_git_call(n) for n in inner)


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
        given = self.arguments.get("code" if self.name == "python_exec" else "command")
        if self.status == "blocked" or not isinstance(given, str):
            return None
        if self.name == "python_exec":
            command, passes = python_test(given)
        elif self.name == "shell_exec":
            command, passes = _shell_test(given)
        else:
            return None
        if not command:
            return None
        if self.finish is None:
            return command, "unknown"
        if self.status != "success" or self.exit_code not in (0, None):
            return command, "failed"
        return command, "passed" if passes and self.exit_code == 0 else "unknown"

    def writes(self) -> bool:
        """Whether this call could have changed code (fail closed: unknown tools could)."""
        if self.status == "blocked" or self.name in _READ_TOOLS or self.test():
            return False
        if self.name in ("file_write", "file_edit"):
            path = str(self.arguments.get("path") or "")
            return self.status != "error" and not _DOC.search(path)
        code = self.arguments.get("code" if self.name == "python_exec" else "command")
        if self.name == "python_exec" and isinstance(code, str):
            return not _reads_only(code)
        return True


@dataclass(frozen=True)
class Baseline:
    """A task folder's git state before the first run since the task last ended ready."""

    stamps: Mapping[str, FileStamp] | None
    head: str = ""
    stash: int = 0
    git: bool = False
    error: bool = False

    def to_json(self) -> dict[str, Any]:
        stamps = {k: list(v) for k, v in (self.stamps or {}).items()}
        return {
            "git": self.git,
            "error": self.error,
            "head": self.head,
            "stash": self.stash,
            "stamps": stamps,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Baseline:
        stamps = data.get("stamps")
        files = {k: (int(v[0]), int(v[1])) for k, v in (stamps or {}).items()}
        return cls(
            files if data.get("git") else None,
            str(data.get("head") or ""),
            int(data.get("stash") or 0),
            bool(data.get("git")),
            bool(data.get("error")),
        )


def _git(folder: str, *args: str) -> tuple[int, str, str] | None:
    """A read-only git command's exit code, output and error, or None if it never ran."""
    argv = ["git", "-C", folder, "-c", "core.fsmonitor=false", *args]
    try:
        done = subprocess.run(
            argv, capture_output=True, timeout=10, env={**os.environ, **_ENV}
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out, err = (s.decode("utf-8", "replace") for s in (done.stdout, done.stderr))
    return done.returncode, out, err


def _look(folder: str) -> Baseline:
    """The folder's git state now; ``error`` when git could not say."""
    inside = _git(folder, "rev-parse", "--is-inside-work-tree")
    if inside is not None and inside[0] and "not a git repository" in inside[2]:
        return Baseline(None)
    stamps = workspace_stamps(folder) if inside and inside[0] == 0 else None
    head = _git(folder, "rev-parse", "-q", "--verify", "HEAD")
    stash = _git(folder, "stash", "list")
    if stamps is None or head is None or stash is None or stash[0]:
        return Baseline(None, git=True, error=True)
    lines = stash[1].splitlines()
    return Baseline(stamps, head[1].strip(), len(lines), git=True)


async def baseline(folder: str | None) -> Baseline:
    """The folder's git state now, to judge the runs after it by."""
    return await asyncio.to_thread(_look, folder) if folder else Baseline(None)


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
    store: SessionStore, task_id: str, turn: Any, folder: str | None, base: Baseline
) -> tuple[list[str], list[tuple[_Call, tuple[str, str]]], list[str], list[str]]:
    """Reasons about the run's tests and folder; its tests; changed code; own changes."""
    calls = await _calls(store, task_id, turn) if turn is not None else []
    tests = [(call, found) for call in calls if (found := call.test())]
    last = tests[-1][1][1] if tests else "passed"
    reasons = {"failed": ["tests_failed"], "unknown": ["tests_unverified"]}.get(
        last, []
    )
    now = await baseline(folder)
    if base.error or now.error or (base.git and not now.git):
        return [*reasons, "unchecked"], tests, [], []
    if not now.git:
        return reasons, tests, [], []
    before = base.stamps if base.git and base.stamps is not None else {}
    edited = sorted(
        p for p in edited_files(before, now.stamps or {}) if not _CACHE.search(p)
    )
    committed: list[str] = []
    if now.head and now.head != base.head:
        since = base.head or "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # empty tree
        names = await asyncio.to_thread(
            _git, folder or "", "diff", "--name-only", "-z", since, now.head
        )
        if names is None or names[0]:
            return [*reasons, "unchecked"], tests, [], edited
        committed = [name for name in names[1].split("\0") if name]
    if now.stash > base.stash:
        return [*reasons, "unchecked"], tests, [], edited
    code = [p for p in [*edited, *committed] if not _DOC.search(p)]
    if code and not reasons:
        # Code changed: a test must start after every call that could have changed it.
        writers = [call.finish for call in calls if call.writes()]
        latest = max((float("inf") if w is None else w for w in writers), default=0.0)
        if not any(call.start > latest for call, _ in tests):
            reasons.append("tests_missing")
    return reasons, tests, code, edited


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
    reasons, tests, _, own = await _evidence(store, task_id, turn, folder, base)
    if asks_commit and own:
        reasons.append("commit_missing")
    if out_of_steps:
        state, reasons = "draft", ["out_of_steps", *reasons]
    elif proposed in ("failed", "dropped"):
        state = str(proposed)  # a worker may give up; code never raises a claim
    elif proposed == "blocked" or question:
        state, reasons = "blocked", ["question", *reasons]
    else:
        state = "draft" if reasons else "ready"
    kept = [
        {
            "command": redact_secrets(command)[0],
            "exit_code": call.exit_code,
            "outcome": outcome,
        }
        for call, (command, outcome) in tests[-_KEPT:]
    ]
    return {
        "state": state,
        "proposed": proposed,
        "reasons": reasons,
        "question": question,
        "tests": kept,
        "uncommitted": own,
        "turn": turn.turn if turn is not None else None,
        "run_id": turn.run_id if turn is not None else "",
    }


def can_finish(ending: Mapping[str, Any] | None) -> bool:
    """Whether Finish it settles this ending: work owed, and not a run out of steps."""
    reasons = set((ending or {}).get("reasons") or ())
    state = (ending or {}).get("state")
    owed = bool(reasons & OWED) and "out_of_steps" not in reasons
    return owed and state in ("draft", "failed", "dropped")


def finish_message(ending: Mapping[str, Any]) -> str:
    """A draft's one-tap instruction: run the tests, and commit when that is missing."""
    if "commit_missing" in ending.get("reasons", ()):
        return t("ending.finishCommitMessage")
    return t("ending.finishMessage")


def default_ending(state: str, wait_reason: str) -> dict[str, Any] | None:
    """The ending a task's state implies when no run decided one."""
    if state == "failed" and wait_reason == OUT_OF_STEPS:
        return {"state": "draft", "reasons": ["out_of_steps"]}
    found = _DEFAULTS.get(state)
    return None if found is None else {"state": found[0], "reasons": [found[1]]}


def shown(record: TaskRecord) -> dict[str, Any] | None:
    """A finished task's ending: decided at its last transition, else its state's."""
    ending = getattr(record, "ending", None)  # a stand-in record may have none
    ended = record.state == "done" or record.state in _DEFAULTS
    if ended and isinstance(ending, Mapping) and ending.get("at") == record.finished_at:
        return dict(ending)
    return default_ending(record.state, record.wait_reason)
