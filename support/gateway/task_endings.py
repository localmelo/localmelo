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
- ``tests_still_fail``: the last test failed and nothing was changed, so there
  is nothing to finish;
- ``tests_missing``: code changed (in git since the baseline, outside git by
  the run's own edits) and no test started after the latest call that could
  have changed it;
- ``commit_missing``: a commit was asked and changes since the baseline are not
  committed;
- ``unchecked``: git could not be read, the folder lists too many changes, or
  the run grew ``git stash list``.

A test is one shape only: a plain test-runner command (no shell operators) or a
``python_exec`` that is imports plus one ``sys.exit(subprocess.run([...]).
returncode)``, ``subprocess.run([...], check=True)`` or
``subprocess.check_call([...])``. Its own exit code is the result; any other
call that starts a runner is a test with an ``unknown`` result. A call could
change code unless every call it makes is known to read (git reads, ``git
add``/``commit``, prints, string methods). A question makes it ``blocked``.
:func:`shown` gives a finished task's ending.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import shlex
import subprocess
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from localmelo.melo.agent._brief_aids import FileStamp, edited_files
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
# Changed files a baseline keeps at most; a folder with more is not checked.
MAX_STAMPS: Final = 2000
_KEPT: Final = 5
_CHARS: Final = 200
_LINE: Final = re.compile(
    r"^[\s>*_`-]*ending[*_`\s]*[:：][*_`\s]*(ready|blocked|failed|dropped)\b[*_`]*"
    r"\s*(?:[:：—–-]\s*(.*?))?\s*$",
    re.I | re.M,
)
_OPERATORS: Final = re.compile(r"[|&;<>`\n\r]|\$\(")
_STARTS: Final = re.compile(
    r"^(?:subprocess\.(?:run|call|check_call|check_output|Popen|getoutput|getstatusoutput)"
    r"|os\.(?:system|popen|exec\w*|spawn\w*))$"
)
_KEYWORDS: Final = frozenset(
    {"check", "capture_output", "text", "timeout", "encoding", "errors"}
    | {"stdout", "stderr"}
)
_READ_TOOLS: Final = frozenset(
    {"file_read", "list_files", "git_status", "search_files", "ask_owner"}
)

# ── Test runners: their words, and the flags that make them run no test ──

_NO_RUN: Final = frozenset({"--version", "-h", "--help"})
_PYTEST_NO_RUN: Final = _NO_RUN | {
    "-V",
    "--setup-plan",
    "--setup-only",
    "--fixtures",
    "--fixtures-per-test",
    "--collect-only",
    "--co",
    "--pyargs",
    "--markers",
}
_RUNNERS: Final[dict[tuple[str, ...], frozenset[str]]] = {
    ("pytest",): _PYTEST_NO_RUN,
    ("py.test",): _PYTEST_NO_RUN,
    ("python", "-m", "pytest"): _PYTEST_NO_RUN,
    ("python", "-m", "unittest"): _NO_RUN,
    **{(npm, "test"): _NO_RUN for npm in ("npm", "pnpm", "yarn", "bun")},
    **{(npm, "run", "test"): _NO_RUN for npm in ("npm", "pnpm", "yarn")},
    ("cargo", "test"): _NO_RUN | {"--no-run", "--list"},
    ("go", "test"): _NO_RUN | {"-list", "-c", "-n"},
    ("swift", "test"): _NO_RUN | {"--list-tests", "-l", "list"},
    ("node", "--test"): _NO_RUN,
    ("make", "test"): _NO_RUN | {"-n", "--dry-run", "--just-print", "-q"},
    ("tox",): _NO_RUN
    | {"-l", "--listenvs", "-a", "--listenvs-all", "--showconfig", "--notest"}
    | {"l", "list", "c", "config", "de", "depends", "devenv", "q", "quickstart"},
    ("nox",): _NO_RUN | {"-l", "--list", "--list-sessions", "--install-only"},
    ("npx", "jest"): _NO_RUN | {"--listTests", "--showConfig"},
    ("npx", "vitest"): _NO_RUN | {"-v", "list"},
    ("xcodebuild",): _NO_RUN
    | {"-list", "-showBuildSettings", "-version", "-showdestinations"}
    | {"-showTestPlans", "-showsdks", "-help", "-usage"},
}
# Only these runners may follow ``uv run`` or ``poetry run``, after its options.
_PYTHON_RUNNERS: Final = frozenset(
    {
        ("pytest",),
        ("py.test",),
        ("python", "-m", "pytest"),
        ("python", "-m", "unittest"),
    }
)
_RUN_VALUES: Final = frozenset(
    {"--with", "--with-editable", "--with-requirements", "--extra", "--group"}
    | {"--only-group", "--no-group", "--no-extra", "--python", "-p", "--package"}
    | {"--project", "--directory", "--env-file", "--index", "--default-index"}
    | {"--index-url", "--extra-index-url", "--find-links", "-f", "--config-file"}
    | {"--cache-dir", "-C", "-P"}
)

# ── Calls that read only: anything else could change a file ──

_SAFE_MODULES: Final = frozenset({"subprocess", "sys", "os", "json", "shlex", "re"}) | {
    "pathlib",
    "textwrap",
}
_SAFE_CALLS: Final = frozenset(
    {"print", "len", "str", "int", "float", "bool", "repr", "sorted", "reversed"}
    | {"list", "dict", "set", "tuple", "min", "max", "sum", "any", "all", "abs"}
    | {"enumerate", "range", "zip", "isinstance", "round", "format"}
    | {"sys.exit", "sys.stdout.write", "sys.stderr.write", "sys.stdout.flush"}
    | {"json.dumps", "json.loads", "shlex.split", "shlex.quote", "shlex.join"}
    | {"re.search", "re.match", "re.fullmatch", "re.findall", "re.sub", "re.split"}
    | {"os.getcwd", "os.listdir", "os.getenv", "os.environ.get", "textwrap.dedent"}
)
_SAFE_METHODS: Final = frozenset(
    {"strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "decode"}
    | {"encode", "join", "format", "startswith", "endswith", "lower", "upper"}
    | {"replace", "count", "find", "partition", "get", "items", "keys", "values"}
    | {"append", "extend", "read", "readline", "readlines", "check_returncode"}
    | {"group", "groups"}
)
# git subcommands that change no file in the folder (add and commit change refs only).
_GIT_READS: Final = frozenset(
    {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "branch"}
    | {"remote", "describe", "shortlog", "add", "commit"}
)
# git -c settings that run nothing.
_GIT_CONFIG: Final = re.compile(
    r"(?:user\.(?:name|email|signingkey)|(?:commit|tag)\.gpgsign|color\.[\w.]+"
    r"|core\.quotepath|i18n\.[\w.]+|advice\.[\w.]+|init\.defaultbranch)=",
    re.I,
)
# Module values a reading python_exec may use other than by calling them.
_VALUES: Final = frozenset(
    {"sys.executable", "sys.argv", "sys.platform", "sys.stdout", "sys.stderr"}
    | {"os.environ", "os.sep", "os.linesep", "os.name", "os.path"}
    | {"subprocess.PIPE", "subprocess.DEVNULL", "subprocess.STDOUT"}
    | {"subprocess.CalledProcessError", "subprocess.TimeoutExpired"}
    | {"subprocess.SubprocessError"}
)
_SHELL_READS: Final = frozenset(
    {"ls", "pwd", "cat", "head", "tail", "wc", "grep", "echo", "which", "true"}
    | {"stat", "file", "du", "df", "date", "whoami", "uname"}
)
_DOC: Final = re.compile(r"\.(?:md|markdown|rst|adoc)$", re.I)
# What running code leaves behind is not work: never owed a commit.
_CACHE: Final = re.compile(
    r"(?:^|/)(?:__pycache__|\.\w+_cache|node_modules|\.tox|\.nox)/|\.py[co]$"
)
# git that cannot run here: no binary, or the system's stand-in without tools.
_NO_GIT: Final = re.compile(
    r"xcrun: error|invalid active developer path|no developer tools", re.I
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


def _here(words: list[str]) -> bool:
    """Whether no word (nor an ``--opt=value``'s value) points outside the folder."""
    for word in words:
        value = word.split("=", 1)[1] if word.startswith("-") and "=" in word else word
        if value.startswith(("/", "~")) or ".." in value.split("/"):
            return False
    return True


def _empty_match(words: list[str]) -> bool:
    """A ``go test -run`` pattern that matches no test (``^$``)."""
    for at, word in enumerate(words):
        if word in ("-run", "--run", "-test.run") and at + 1 < len(words):
            pattern = words[at + 1]
        elif word.startswith(("-run=", "--run=", "-test.run=")):
            pattern = word.split("=", 1)[1]
        else:
            continue
        if not pattern.strip("^$"):
            return True
    return False


def _runner(argv: list[str]) -> tuple[bool, bool]:
    """``(starts a test runner, on this folder only)`` for a command's words."""
    words = [os.path.basename(argv[0]), *argv[1:]] if argv else []
    if words and re.fullmatch(r"python(?:3(?:\.\d+)?)?", words[0]):
        words[0] = "python"
    lead: list[str] = []
    allowed: Collection[tuple[str, ...]] = _RUNNERS.keys()
    if words[:1] in (["uv"], ["poetry"]) and words[1:2] == ["run"]:
        allowed, words = _PYTHON_RUNNERS, words[2:]
        while words and words[0].startswith("-"):
            lead.append(words.pop(0))
            if "=" not in lead[-1] and lead[-1] in _RUN_VALUES and words:
                lead.append(words.pop(0))
        if words and re.fullmatch(r"python(?:3(?:\.\d+)?)?", words[0]):
            words[0] = "python"
    found = max(
        (k for k in allowed if tuple(words[: len(k)]) == k), key=len, default=None
    )
    if found is None:
        return False, False
    rest = words[len(found) :]
    if found == ("xcodebuild",):
        if "test" not in rest:
            return False, False
        rest = [word for word in rest if word != "test"]
    if _RUNNERS[found] & set(rest) or (found == ("go", "test") and _empty_match(rest)):
        return False, False
    return True, _here(lead + rest)


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
    renamed = any(alias.asname for s in imports for alias in s.names)
    if names - {"subprocess", "sys"} or renamed or len(body) != 1:
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


def _git_reads(words: Sequence[str | None]) -> bool:
    """A git command that changes no file: a read, ``add`` or ``commit``, no ``--output``."""
    if not words or words[0] is None or os.path.basename(words[0]) != "git":
        return False
    at = 1
    while at < len(words) and words[at] in ("-C", "-c", "--no-pager"):
        value = words[at + 1] if at + 1 < len(words) else None
        if words[at] == "-c" and (value is None or not _GIT_CONFIG.match(value)):
            return False
        at += 1 if words[at] == "--no-pager" else 2
    sub = words[at] if at < len(words) else None
    if sub not in _GIT_READS:
        return False
    rest = words[at + 1 :]
    if sub not in ("add", "commit") and None in rest:
        return False  # a computed word could be --output
    return not any(
        word and (word.startswith("--output") or (word == "-o" and sub != "commit"))
        for word in rest
    )


def _shell_writes(command: str) -> bool:
    """Whether a shell command could change a file: anything but a plain read."""
    if _OPERATORS.search(command):
        return True
    try:
        words = shlex.split(command)
    except ValueError:
        return True
    if not words or os.path.basename(words[0]) in _SHELL_READS:
        return False
    return not _git_reads(words)


def _aliases(tree: ast.Module) -> dict[str, str] | None:
    """The names imports bind, as dotted names; None for a module that may write."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _SAFE_MODULES:
                    return None
                names[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module.split(".")[0] not in _SAFE_MODULES:
                return None
            for alias in node.names:
                if alias.name == "*":
                    return None
                names[alias.asname or alias.name] = f"{module}.{alias.name}"
    return names


def _argv(call: ast.Call) -> list[str | None] | None:
    """A started command's words (computed ones None), or None when unknown."""
    if len(call.args) != 1 or any(
        k.arg in (None, "executable", "env") for k in call.keywords
    ):
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        if _OPERATORS.search(first.value):
            return None
        try:
            return list(shlex.split(first.value))
        except ValueError:
            return None
    if not isinstance(first, ast.List | ast.Tuple):
        return None
    words: list[str | None] = []
    for item in first.elts:
        items = [item]
        if isinstance(item, ast.Starred):
            if not isinstance(item.value, ast.List | ast.Tuple):
                return None
            items = list(item.value.elts)
        words += [
            (
                e.value
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
                else None
            )
            for e in items
        ]
    return words


def _reads(call: ast.Call, names: Mapping[str, str]) -> bool:
    """Whether *call* is known to change no file."""
    dotted = ast.unparse(call.func)
    head, _, tail = dotted.partition(".")
    name = names.get(head, head) + (f".{tail}" if tail else "")
    if name in _SAFE_CALLS or name.startswith("os.path."):
        return True
    if name == "open":
        modes = [*call.args[1:2], *(k.value for k in call.keywords if k.arg == "mode")]
        starred = any(isinstance(a, ast.Starred) for a in call.args)
        if starred or any(k.arg is None for k in call.keywords):
            return False
        mode = modes[0] if modes else ast.Constant("r")
        return (
            isinstance(mode, ast.Constant)
            and isinstance(mode.value, str)
            and not set(mode.value) & set("wax+")
        )
    if _STARTS.match(name):
        words = _argv(call)
        return words is not None and _git_reads(words)
    if isinstance(call.func, ast.Attribute) and call.func.attr in _SAFE_METHODS:
        root = call.func.value
        while isinstance(root, ast.Attribute | ast.Subscript):
            root = root.value
        return not (
            isinstance(root, ast.Name)
            and (root.id in names or root.id in _SAFE_MODULES)
        )
    return False


def _python_writes(code: str) -> bool:
    """Whether python_exec *code* could change a file: any call not known to read.

    A module or its attribute used other than by a call (``x = os``) or a reader
    rebound (``print = ...``) could write too.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return True
    names = _aliases(tree)
    if names is None:
        return True
    parents = {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    readers = {n for n in _SAFE_CALLS if "." not in n} | {"open"} | _SAFE_MODULES
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and not _reads(node, names):
            return True
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return True
        defined = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        if isinstance(node, defined) and node.decorator_list:
            return True
        if not isinstance(node, ast.Name | ast.Attribute):
            continue
        if isinstance(node.ctx, ast.Store | ast.Del):
            rebound = isinstance(node, ast.Name) and node.id in names.keys() | readers
            if isinstance(node, ast.Attribute) or rebound:
                return True  # an attribute set, or a reader rebound
        elif isinstance(node, ast.Name) and node.id in names:
            top: ast.AST = node
            while isinstance(up := parents.get(id(top)), ast.Attribute):
                top = up
            called = isinstance(up, ast.Call) and up.func is top
            head, _, tail = ast.unparse(top).partition(".")
            dotted = names[head] + (f".{tail}" if tail else "")
            if not called and dotted not in _VALUES:
                return True  # a module passed around: what reads it is unknown
    return False


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
            return self.edits_code()
        given = self.arguments.get("code" if self.name == "python_exec" else "command")
        if self.name == "python_exec":
            return not isinstance(given, str) or _python_writes(given)
        if self.name == "shell_exec":
            return not isinstance(given, str) or _shell_writes(given)
        return True

    def edits_code(self) -> bool:
        """A call known to change code: a code file edited, or a writing python_exec."""
        if self.name in ("file_write", "file_edit"):
            path = str(self.arguments.get("path") or "")
            return self.status not in ("blocked", "error") and not _DOC.search(path)
        return self.name == "python_exec" and self.writes()


@dataclass(frozen=True)
class Baseline:
    """A task folder's git state: changed files under it, HEAD and its stash count."""

    stamps: Mapping[str, FileStamp] | None
    head: str = ""
    stash: int = 0
    git: bool = False
    error: bool = False

    def to_json(self) -> dict[str, Any]:
        """What is kept until the task ends ready: code paths and HEAD, no stash."""
        code = {
            k: list(v) for k, v in (self.stamps or {}).items() if not _DOC.search(k)
        }
        return {"git": self.git, "head": self.head, "stamps": code}

    def since(self, kept: Mapping[str, Any]) -> Baseline:
        """This run's baseline: *kept* code and HEAD, and this run's docs and stash."""
        git = bool(kept.get("git"))
        files = {
            k: (int(v[0]), int(v[1])) for k, v in (kept.get("stamps") or {}).items()
        }
        docs = {k: v for k, v in (self.stamps or {}).items() if _DOC.search(k)}
        stamps = {**files, **docs} if git else None
        return Baseline(stamps, str(kept.get("head") or ""), self.stash, git)


_ERROR: Final = Baseline(None, git=True, error=True)


def _git(folder: str, *args: str) -> tuple[int, str, str] | None:
    """A read-only git command's exit code, output and error; None if git cannot run."""
    argv = ["git", "-C", folder, "-c", "core.fsmonitor=false", *args]
    try:
        done = subprocess.run(
            argv, capture_output=True, timeout=10, env={**os.environ, **_ENV}
        )
    except (FileNotFoundError, PermissionError):
        return None
    except (OSError, subprocess.SubprocessError) as error:
        return -1, "", str(error)
    out, err = (s.decode("utf-8", "replace") for s in (done.stdout, done.stderr))
    if done.returncode and _NO_GIT.search(err):
        return None
    return done.returncode, out, err


def _look(folder: str) -> Baseline:
    """The folder's git state now; ``error`` when git could not say."""
    where = _git(folder, "rev-parse", "--show-toplevel", "--show-prefix")
    if where is None or (where[0] and "not a git repository" in where[2]):
        return Baseline(None)  # not a git folder, or git cannot run here
    status = _git(folder, "status", "--porcelain", "-z", "-uall", "--", ".")
    head = _git(folder, "rev-parse", "-q", "--verify", "HEAD")
    stash = _git(folder, "stash", "list")
    if where[0] or status is None or status[0] or head is None or stash is None:
        return _ERROR
    if stash[0]:
        return _ERROR
    top, prefix = ([*where[1].split("\n"), ""])[:2]
    stamps: dict[str, FileStamp] = {}
    entries = iter(status[1].split("\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        name = entry[3:]
        if entry[0] in "RC":  # a rename lists its old name next
            next(entries, None)
        if not name.startswith(prefix) or _CACHE.search(name):
            continue
        try:
            info = os.stat(os.path.join(top, name))
            stamps[name[len(prefix) :]] = (info.st_size, info.st_mtime_ns)
        except OSError:
            stamps[name[len(prefix) :]] = (-1, -1)
        if len(stamps) > MAX_STAMPS:
            return _ERROR
    return Baseline(stamps, head[1].strip(), len(stash[1].splitlines()), git=True)


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


async def _changes(
    folder: str | None, base: Baseline
) -> tuple[bool, list[str], list[str]]:
    """``(unchecked, changed code, own uncommitted changes)`` since *base*."""
    now = await baseline(folder)
    if base.error or now.error or (base.git and not now.git):
        return True, [], []
    if not now.git:
        return False, [], []
    before = base.stamps if base.git and base.stamps is not None else {}
    own = sorted(edited_files(before, now.stamps or {}))
    committed: list[str] = []
    if now.head and now.head != base.head:
        since = base.head or "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # empty tree
        names = await asyncio.to_thread(
            _git,
            folder or "",
            "diff",
            "--name-only",
            "-z",
            "--relative",
            since,
            now.head,
        )
        if names is None or names[0]:
            return True, [], own
        committed = [name for name in names[1].split("\0") if name]
    code = sorted(p for p in {*own, *committed} if not _DOC.search(p))
    return now.stash > base.stash, code, own


async def _evidence(
    store: SessionStore, task_id: str, turn: Any, folder: str | None, base: Baseline
) -> tuple[list[str], list[tuple[_Call, tuple[str, str]]], list[str]]:
    """Reasons about the run's tests and folder; its tests; its own changes."""
    calls = await _calls(store, task_id, turn) if turn is not None else []
    tests = [(call, found) for call in calls if (found := call.test())]
    unchecked, code, own = await _changes(folder, base)
    # Outside git only the run's own calls say whether code changed.
    changed = bool(code) or (not base.git and any(c.edits_code() for c in calls))
    last = tests[-1][1][1] if tests else "passed"
    reasons: list[str] = []
    if last == "failed":
        owed = changed or unchecked
        reasons.append("tests_failed" if owed else "tests_still_fail")
    elif last == "unknown":
        reasons.append("tests_unverified")
    elif changed:
        # A test must start after every call that could have changed code.
        writers = [call.finish for call in calls if call.writes()]
        latest = max((float("inf") if w is None else w for w in writers), default=0.0)
        if not any(call.start > latest for call, _ in tests):
            reasons.append("tests_missing")
    return [*reasons, *(["unchecked"] if unchecked else [])], tests, own


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
    reasons, tests, own = await _evidence(store, task_id, turn, folder, base)
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
