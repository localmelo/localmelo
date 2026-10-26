"""Classes, passes, labels and phases of tool calls.

Each tool call gets one class, by the rules the bench measurement used:
**R** reads, **W** writes, **T** tests, **X** is anything else
(:func:`classify_call`). A **T** call whose output holds a test summary
with a pass and no failure or error is a *pass* (:func:`is_pass`); exit
status alone never makes one. A label names a call without its
arguments' content (:func:`label_for`). Phases (Explore, Write, Verify,
Finishing) and cycles are read from the classes (:class:`PhaseState`).
Everything here is pure and never raises on odd arguments.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from localmelo.melo.checker.redaction import redact_secrets

READ: Final = "R"
WRITE: Final = "W"
TEST: Final = "T"
OTHER: Final = "X"
CLASSES: Final = (READ, WRITE, TEST, OTHER)
# The milestone of the first pass, beside the classes W and T.
PASS_ANCHOR: Final = "P"

EXPLORE: Final = "Explore"
WRITING: Final = "Write"
VERIFY: Final = "Verify"
FINISHING: Final = "Finishing"

# The output window a pass is read from: the last characters of a call's
# checked output.
OUTPUT_TAIL: Final = 400
MAX_LABEL: Final = 40

_READ_RE: Final = re.compile(
    r"^\s*(cd [^&;]+(&&|;)\s*)?(ls|cat|head|tail|find|grep|rg|wc|tree|stat|du"
    r"|file|echo|pwd|sort|uniq|awk|sed -n|diff|md5|shasum|xxd|od)\b"
)
_TEST_RE: Final = re.compile(
    r"pytest|unittest|_test\.py|\bverify\b|\bcheck\b|assert ", re.IGNORECASE
)
_WRITE_SHELL_RE: Final = re.compile(
    r"(?<![0-9&])>(?!&)|\bmv\b|\bcp\b|\bmkdir\b|\brm\b|sed -i|\btee\b|<<|\btouch\b"
    r"|\bln\b"
)
_WRITE_PY_RE: Final = re.compile(
    r"""open\([^)]*['"][wa]b?\+?['"]|\.write\(|os\.rename|shutil\.|os\.makedirs"""
    r"""|os\.mkdir|os\.remove|\.write_text\(|\.rename\(|os\.replace"""
)
_READ_PY_RE: Final = re.compile(r"open\(|listdir|os\.walk|glob|read\(|print\(")

# Test summaries. pytest: "3 passed, 1 failed, 2 errors in 0.1s".
_PYTEST_COUNT: Final = re.compile(r"(\d+) (passed|failed|errors?)\b")
# unittest: "Ran 3 tests in 0.001s" then "OK" or "FAILED (failures=1, errors=2)".
_UNITTEST_RAN: Final = re.compile(r"\bRan (\d+) tests?\b")
_UNITTEST_FAILED: Final = re.compile(r"\bFAILED \(([^)]*)\)")
_UNITTEST_OK: Final = re.compile(r"^OK\b", re.MULTILINE)
_UNITTEST_FIELD: Final = re.compile(r"(failures|errors|skipped)=(\d+)")
_EXIT_CODE: Final = re.compile(r"\[exit code (-?\d+)\]")

# Programs whose first word after the program names what they do.
_SUBCOMMAND_PROGRAMS: Final = frozenset(
    {
        "apt",
        "apt-get",
        "brew",
        "bundle",
        "cargo",
        "conda",
        "docker",
        "dotnet",
        "gh",
        "git",
        "go",
        "gradle",
        "kubectl",
        "make",
        "mvn",
        "npm",
        "pip",
        "pip3",
        "pnpm",
        "poetry",
        "uv",
        "yarn",
    }
)
_PYTHONS: Final = frozenset({"python", "python3"})
_WORD: Final = re.compile(r"^[a-z][a-z0-9_-]*$")
_ASSIGNMENT: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATORS: Final = ("&&", "||", ";", "|")


def call_text(tool_name: str, arguments: Any) -> str:
    """The text a call is classified by.

    ``command`` for ``shell_exec``, ``code`` for ``python_exec``, the JSON
    of the arguments for any other tool (or when that field is missing).
    """
    if isinstance(arguments, Mapping):
        field = {"shell_exec": "command", "python_exec": "code"}.get(tool_name)
        value = arguments.get(field) if field else None
        if isinstance(value, str):
            return value
    try:
        return json.dumps(arguments, default=str)
    except (TypeError, ValueError):
        return str(arguments)


def classify_call(tool_name: str, arguments: Any) -> str:
    """The class of a call: ``R``, ``W``, ``T`` or ``X``.

    ``file_read`` is R and ``file_write`` W. For any other tool, a text
    (:func:`call_text`) that matches the test pattern is T; then
    ``shell_exec`` is W when it writes (``>``, ``mv``, ``cp``, ``mkdir``,
    ``rm``, ``sed -i``, ``tee``, ``<<``, ``touch``, ``ln``, searched after
    removing ``2>&1`` and ``2>/dev/null``), R when it starts with a reading
    program, else X; ``python_exec`` is W when it writes a file, R when it
    reads or prints, else X; every other tool is X.
    """
    if tool_name == "file_read":
        return READ
    if tool_name == "file_write":
        return WRITE
    text = call_text(tool_name, arguments)
    if _TEST_RE.search(text):
        return TEST
    if tool_name == "shell_exec":
        if _WRITE_SHELL_RE.search(text.replace("2>&1", "").replace("2>/dev/null", "")):
            return WRITE
        return READ if _READ_RE.match(text) else OTHER
    if tool_name == "python_exec":
        if _WRITE_PY_RE.search(text):
            return WRITE
        return READ if _READ_PY_RE.search(text) else OTHER
    return OTHER


@dataclass(frozen=True)
class TestSummary:
    """The counts of the last test summary in a call's output."""

    __test__ = False  # not a pytest test class

    passed: int
    failed: int
    errors: int

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def is_pass(self) -> bool:
        """At least one pass and no failure or error."""
        return self.passed > 0 and not self.failed and not self.errors

    def as_dict(self) -> dict[str, int]:
        """``{passed, failed, total}``, errors counted as failures."""
        return {
            "passed": self.passed,
            "failed": self.failed + self.errors,
            "total": self.total,
        }


def parse_test_summary(output: str) -> TestSummary | None:
    """The last pytest or unittest summary in *output*, or ``None``.

    pytest: the last line with ``N passed``, ``M failed`` or ``K
    error(s)``. unittest: the last ``Ran N tests``, followed by ``OK``
    (``OK (skipped=S)`` leaves N − S passed) or ``FAILED (failures=M,
    errors=K)``; a ``Ran`` line with neither after it is no summary.
    When both kinds are present, the later one counts.
    """
    pytest_at, pytest_summary = _pytest_summary(output)
    unittest_at, unittest_summary = _unittest_summary(output)
    if pytest_summary is None:
        return unittest_summary
    if unittest_summary is None:
        return pytest_summary
    return pytest_summary if pytest_at > unittest_at else unittest_summary


def _pytest_summary(output: str) -> tuple[int, TestSummary | None]:
    matches = list(_PYTEST_COUNT.finditer(output))
    if not matches:
        return -1, None
    last = matches[-1].start()
    start = output.rfind("\n", 0, last) + 1
    end = output.find("\n", last)
    line = output[start : end if end >= 0 else len(output)]
    counts = {"passed": 0, "failed": 0, "errors": 0}
    for match in _PYTEST_COUNT.finditer(line):
        word = match.group(2)
        counts["errors" if word.startswith("error") else word] = int(match.group(1))
    return start, TestSummary(counts["passed"], counts["failed"], counts["errors"])


def _unittest_summary(output: str) -> tuple[int, TestSummary | None]:
    ran = list(_UNITTEST_RAN.finditer(output))
    if not ran:
        return -1, None
    last = ran[-1]
    tests = int(last.group(1))
    rest = output[last.end() :]
    failed = _UNITTEST_FAILED.search(rest)
    ok = _UNITTEST_OK.search(rest)
    if failed is not None and (ok is None or failed.start() < ok.start()):
        fields = dict(_UNITTEST_FIELD.findall(failed.group(1)))
        failures = int(fields.get("failures", 0))
        errors = int(fields.get("errors", 0))
        skipped = int(fields.get("skipped", 0))
        passed = max(0, tests - failures - errors - skipped)
        return last.start(), TestSummary(passed, failures, errors)
    if ok is not None:
        fields = dict(_UNITTEST_FIELD.findall(rest[ok.start() : ok.start() + 80]))
        passed = max(0, tests - int(fields.get("skipped", 0)))
        return last.start(), TestSummary(passed, 0, 0)
    return -1, None


def output_tail(output: str) -> str:
    """The last :data:`OUTPUT_TAIL` characters of *output*."""
    return output[-OUTPUT_TAIL:]


def call_failed(status: str, output: str) -> bool:
    """Whether a call failed: a status other than ``success``, or an exit.

    A first line ``[exit code <n>]`` (``shell_exec`` with a workspace
    root) with n ≠ 0 marks the call failed whatever its output says.
    """
    if status != "success":
        return True
    first_line = output.split("\n", 1)[0].strip()
    match = _EXIT_CODE.fullmatch(first_line)
    return match is not None and int(match.group(1)) != 0


def is_pass(cls: str, summary: TestSummary | None, failed: bool) -> bool:
    """Whether a call is a pass: a T call, not failed, with a passing summary."""
    return cls == TEST and not failed and summary is not None and summary.is_pass


def _basename(arguments: Any) -> str:
    path = arguments.get("path") if isinstance(arguments, Mapping) else None
    if not isinstance(path, str) or not path.strip():
        return ""
    return os.path.basename(path.strip().rstrip("/\\")) or path.strip()


def _command_words(command: str) -> list[str]:
    """The words of *command*'s first simple command after any ``cd``."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        words = command.split()
    commands: list[list[str]] = [[]]
    for word in words:
        if word in _SEPARATORS or (word and set(word) <= set("&|;")):
            commands.append([])
        else:
            commands[-1].append(word)
    for simple in commands:
        while simple and _ASSIGNMENT.match(simple[0]):
            simple = simple[1:]
        if simple and simple[0] != "cd":
            return simple
    return []


def shell_label(command: str) -> str:
    """The program of *command* and its first subcommand: ``git status``.

    The first simple command that is not a ``cd`` counts, without its
    variable assignments. The program is its first word's base name; a
    subcommand follows only for programs that take one (``git``, ``npm``,
    ``cargo`` and the like) and only right after the program, so that an
    option's value (``git -C <dir>``) never shows; ``python -m <module>``
    keeps its module.
    """
    words = _command_words(command)
    if not words:
        return "shell"
    program = os.path.basename(words[0]) or words[0]
    rest = words[1:]
    if program in _PYTHONS and len(rest) >= 2 and rest[0] == "-m":
        return f"{program} -m {rest[1]}"
    if program in _SUBCOMMAND_PROGRAMS and rest and _WORD.match(rest[0]):
        return f"{program} {rest[0]}"
    return program


def label_for(tool_name: str, arguments: Any) -> str:
    """A short name for a call that holds no argument content.

    ``Read <basename>`` and ``Edit <basename>`` for ``file_read`` and
    ``file_write``; the program and its first subcommand for
    ``shell_exec``; ``Python`` for ``python_exec``; the tool's name for
    any other tool. The label passes ``redact_secrets`` and is cut to
    :data:`MAX_LABEL` characters.
    """
    if tool_name in ("file_read", "file_write"):
        verb = "Read" if tool_name == "file_read" else "Edit"
        base = _basename(arguments)
        label = f"{verb} {base}" if base else verb
    elif tool_name == "shell_exec":
        label = shell_label(call_text(tool_name, arguments))
    elif tool_name == "python_exec":
        label = "Python"
    else:
        label = tool_name or "tool"
    return clean_label(label)


def clean_label(label: str) -> str:
    """*label* redacted, on one line, cut to :data:`MAX_LABEL` characters."""
    redacted, _ = redact_secrets(" ".join(label.split()))
    return redacted[:MAX_LABEL]


def target_for(tool_name: str, arguments: Any, label: str) -> str | None:
    """What a call works on, for matching steps: a base name, else the label.

    ``file_read`` and ``file_write`` give the file's base name (redacted,
    cut); ``python_exec`` gives none; any other tool gives its label.
    """
    if tool_name in ("file_read", "file_write"):
        base = _basename(arguments)
        return clean_label(base) if base else None
    if tool_name == "python_exec":
        return None
    return label


@dataclass
class PhaseState:
    """The phase and cycle of a run, read from its calls' classes in order.

    R or X before the first W is Explore; W is Write; T is Verify; after
    the first pass, every call is Finishing. An R or X call after the
    first W keeps the phase it found. A cycle is one W → T transition: a
    T call with a W since the last T.
    """

    phase: str | None = None
    cycle: int = 0
    seen_write: bool = False
    write_since_test: bool = False
    passed: bool = False

    def add(self, cls: str, passed: bool) -> None:
        """Account for one call of class *cls*; *passed* if it is a pass."""
        if cls == WRITE:
            self.seen_write = True
            self.write_since_test = True
        elif cls == TEST:
            if self.write_since_test:
                self.cycle += 1
            self.write_since_test = False
        self.passed = self.passed or passed
        if self.passed:
            self.phase = FINISHING
        elif cls == WRITE:
            self.phase = WRITING
        elif cls == TEST:
            self.phase = VERIFY
        elif not self.seen_write:
            self.phase = EXPLORE


def phase_agrees(phase: str | None, classes: set[str]) -> bool:
    """Whether *phase* describes a step whose calls have *classes*."""
    if phase == FINISHING:
        return True
    expected = {EXPLORE: {READ, OTHER}, WRITING: {WRITE}, VERIFY: {TEST}}.get(
        phase or "", set()
    )
    return bool(expected & classes)
