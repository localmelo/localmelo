"""Classes, passes, test summaries, labels and phases of tool calls."""

from __future__ import annotations

import json
from typing import Any

import pytest

from localmelo.melo.memory.procedures.classify import (
    EXPLORE,
    FINISHING,
    MAX_LABEL,
    OUTPUT_TAIL,
    VERIFY,
    WRITING,
    PhaseState,
    TestSummary,
    call_failed,
    call_text,
    classify_call,
    is_pass,
    label_for,
    output_tail,
    parse_test_summary,
    phase_agrees,
    shell_label,
    target_for,
)


def shell(command: str) -> str:
    return classify_call("shell_exec", {"command": command})


def python(code: str) -> str:
    return classify_call("python_exec", {"code": code})


# ── classify_call ───────────────────────────────────────────────────────────


def test_file_tools_are_read_and_write_whatever_their_arguments() -> None:
    assert classify_call("file_read", {"path": "check_test.py"}) == "R"
    assert (
        classify_call("file_write", {"path": "x_test.py", "content": "assert x"}) == "W"
    )


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "python -m unittest",
        "python parser_test.py",
        "./verify.sh",
        "make check",
        "python -c 'assert f() == 1'",
        "PYTEST_ADDOPTS=-x PyTest",
    ],
)
def test_shell_tests(command: str) -> None:
    assert shell(command) == "T"


def test_the_test_pattern_wins_over_writing() -> None:
    assert shell("pytest > out.txt") == "T"
    assert python("open('check.txt', 'w').write('x')") == "T"


def test_assert_needs_a_following_space() -> None:
    assert shell("echo asserted") == "R"
    assert shell("grep assertEqual x.py") == "R"


@pytest.mark.parametrize(
    "command",
    [
        "echo hi > out.txt",
        "cat a >> b",
        "mv a b",
        "cp a b",
        "mkdir -p out",
        "rm -f x",
        "sed -i 's/a/b/' f",
        "ls | tee list.txt",
        "cat <<EOF > f\nx\nEOF",
        "touch done",
        "ln -s a b",
    ],
)
def test_shell_writes(command: str) -> None:
    assert shell(command) == "W"


def test_redirecting_errors_alone_is_not_a_write() -> None:
    assert shell("ls 2>&1") == "R"
    assert shell("cat x 2>/dev/null") == "R"
    assert shell("wc -l f 2>&1 | head") == "R"
    # A stream duplication is no file write either.
    assert shell("ls 1>&2") == "R"


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat file.txt",
        "  head -n 5 f",
        "cd /work && grep -rn foo .",
        "cd /work; find . -name '*.py'",
        "sed -n 1,5p f",
        "wc -l f",
        "diff a b",
    ],
)
def test_shell_reads(command: str) -> None:
    assert shell(command) == "R"


@pytest.mark.parametrize(
    "command", ["python run.py", "git status", "npm install", "python run.py 2>err.log"]
)
def test_other_shell_commands(command: str) -> None:
    assert shell(command) == "X"


@pytest.mark.parametrize(
    ("code", "cls"),
    [
        ("open('out.txt', 'w').write(data)", "W"),
        ("open(p, 'ab+')", "W"),
        ("Path('x').write_text('y')", "W"),
        ("import shutil; shutil.copy(a, b)", "W"),
        ("os.makedirs('d')", "W"),
        ("os.replace(a, b)", "W"),
        ("print(open('f').read())", "R"),
        ("print(x)", "R"),
        ("import os; os.listdir('.')", "R"),
        ("for p in glob.glob('*.txt'): pass", "R"),
        ("x = 1 + 1", "X"),
    ],
)
def test_python_classes(code: str, cls: str) -> None:
    assert python(code) == cls


def test_other_tools_use_the_json_of_their_arguments() -> None:
    assert classify_call("file_edit", {"path": "a.py"}) == "X"
    assert classify_call("read_file", {"path": "a_test.py"}) == "T"
    assert classify_call("mcp_notes", {"query": "check the notes"}) == "T"
    assert classify_call("mcp_notes", {"query": "list"}) == "X"


def test_odd_arguments_never_raise() -> None:
    assert classify_call("shell_exec", None) == "X"
    assert classify_call("shell_exec", {"command": 42}) == "X"
    assert classify_call("python_exec", ["code"]) == "X"
    assert call_text("x", {"a": object()}).startswith('{"a": "<object')
    assert classify_call("", {}) == "X"


# ── a command sent as an argv array ─────────────────────────────────────────


def test_an_argv_array_command_is_joined_with_shlex() -> None:
    """The tool refuses an argv array, but the line and the store still name
    it: the words joined as a shell command (P3 follow-up)."""
    arguments = {"command": ["git", "commit", "-m", "fix x"]}
    assert call_text("shell_exec", arguments) == "git commit -m 'fix x'"
    assert label_for("shell_exec", arguments) == "git commit"
    assert target_for("shell_exec", arguments, "git commit") == "git commit"
    assert shell_label("git commit -m 'fix x'") == "git commit"


def test_an_argv_array_read_command_is_a_read() -> None:
    # The JSON began with "{" so READ_RE (anchored) never matched it.
    assert classify_call("shell_exec", {"command": ["ls", "-la"]}) == "R"
    assert classify_call("shell_exec", {"command": ["pytest", "-q"]}) == "T"
    assert classify_call("shell_exec", {"command": ["rm", "-rf", "build"]}) == "W"


def test_an_empty_argv_array_is_an_empty_command() -> None:
    assert call_text("shell_exec", {"command": []}) == ""
    assert label_for("shell_exec", {"command": []}) == "shell"


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": ["echo", 3]},  # not all strings
        {"command": {"argv": ["ls"]}},
        {},
        ["ls"],  # not a mapping
        "ls",
    ],
)
def test_other_argument_shapes_keep_the_json(arguments: Any) -> None:
    assert call_text("shell_exec", arguments) == json.dumps(arguments, default=str)


def test_python_code_is_never_joined_as_an_argv() -> None:
    assert call_text("python_exec", {"code": ["print(1)"]}) == (
        '{"code": ["print(1)"]}'
    )


# ── test summaries and passes ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("===== 5 passed in 0.01s =====", (5, 0, 0)),
        ("1 failed, 3 passed in 0.2s", (3, 1, 0)),
        ("2 passed, 1 error in 0.1s", (2, 0, 1)),
        ("3 errors in 0.1s", (0, 0, 3)),
        ("4 passed, 2 warnings in 1s", (4, 0, 0)),
        ("Ran 3 tests in 0.001s\n\nOK", (3, 0, 0)),
        ("Ran 3 tests in 0.001s\n\nOK (skipped=1)", (2, 0, 0)),
        ("Ran 4 tests in 0.1s\n\nFAILED (failures=1, errors=1)", (2, 1, 1)),
        ("Ran 1 test in 0.1s\n\nFAILED (errors=1)", (0, 0, 1)),
        ("Ran 5 tests in 0.1s\n\nFAILED (failures=1, skipped=2)", (2, 1, 0)),
    ],
)
def test_parse_test_summary(output: str, expected: tuple[int, int, int]) -> None:
    summary = parse_test_summary(output)
    assert summary is not None
    assert (summary.passed, summary.failed, summary.errors) == expected


def test_the_last_summary_counts() -> None:
    output = "1 failed, 2 passed in 0.1s\n...\n3 passed in 0.1s"
    assert parse_test_summary(output) == TestSummary(3, 0, 0)
    mixed = "Ran 2 tests in 0.1s\n\nOK\n\n1 failed in 0.1s"
    assert parse_test_summary(mixed) == TestSummary(0, 1, 0)
    reverse = "1 failed in 0.1s\nRan 2 tests in 0.1s\n\nOK"
    assert parse_test_summary(reverse) == TestSummary(2, 0, 0)


@pytest.mark.parametrize(
    "output", ["", "no tests ran in 0.01s", "Ran 3 tests in 0.1s", "all good"]
)
def test_no_summary(output: str) -> None:
    assert parse_test_summary(output) is None


def test_summary_counts_as_progress_tests() -> None:
    assert TestSummary(3, 1, 1).as_dict() == {"passed": 3, "failed": 2, "total": 5}
    assert TestSummary(3, 0, 0).is_pass
    assert not TestSummary(0, 0, 0).is_pass
    assert not TestSummary(3, 1, 0).is_pass
    assert not TestSummary(3, 0, 1).is_pass


def test_a_pass_needs_a_t_call_a_passing_summary_and_no_failure() -> None:
    good = TestSummary(3, 0, 0)
    assert is_pass("T", good, failed=False)
    assert not is_pass("R", good, failed=False)
    assert not is_pass("T", None, failed=False)
    assert not is_pass("T", good, failed=True)
    # Exit status alone never makes one: "cat ... # check" exits 0.
    assert not is_pass("T", parse_test_summary("hello"), failed=False)


def test_call_failed() -> None:
    assert not call_failed("success", "3 passed")
    assert call_failed("error", "3 passed")
    assert call_failed("timeout", "")
    assert call_failed("success", "[exit code 1]\n3 passed")
    assert call_failed("success", "[exit code -9]")
    assert not call_failed("success", "[exit code 0]\nok")
    assert not call_failed("success", "text\n[exit code 1]")


def test_output_tail() -> None:
    assert OUTPUT_TAIL == 400
    assert output_tail("x" * 1000 + "5 passed") == ("x" * 1000 + "5 passed")[-400:]


# ── labels and targets ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "label"),
    [
        ("pytest -q tests/", "pytest"),
        ("git status", "git status"),
        # An option's value is never taken for the subcommand.
        ("git -C repo log --oneline", "git"),
        ("npm test", "npm test"),
        ("cd /work && pytest -x", "pytest"),
        ("cd /w; cd sub && make build", "make build"),
        ("FOO=1 BAR=2 python3 -m pytest a_test.py", "python3 -m pytest"),
        ("/usr/bin/python3 script.py", "python3"),
        ("ls -la /secret/place", "ls"),
        ("cat notes.txt | grep todo", "cat"),
        ("", "shell"),
        ("cd /only", "shell"),
        ("echo 'unbalanced", "echo"),
    ],
)
def test_shell_label(command: str, label: str) -> None:
    assert shell_label(command) == label


@pytest.mark.parametrize(
    ("tool", "arguments", "label"),
    [
        ("file_read", {"path": "/home/u/proj/parser.py"}, "Read parser.py"),
        ("file_write", {"path": "src/out/", "content": "x"}, "Edit out"),
        ("file_read", {}, "Read"),
        ("shell_exec", {"command": "git status --short"}, "git status"),
        ("python_exec", {"code": "print(secret)"}, "Python"),
        ("mcp_mail_search", {"q": "from:boss"}, "mcp_mail_search"),
    ],
)
def test_labels_hold_no_argument_content(
    tool: str, arguments: dict[str, Any], label: str
) -> None:
    assert label_for(tool, arguments) == label


def test_labels_are_redacted_and_cut() -> None:
    key = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    assert label_for("file_read", {"path": f"/tmp/{key}"}) == "Read [redacted]"
    long_name = "a" * 100 + ".py"
    assert MAX_LABEL == 40
    assert len(label_for("file_write", {"path": long_name})) == 40


def test_targets() -> None:
    args = {"path": "/x/parser.py"}
    assert target_for("file_read", args, label_for("file_read", args)) == "parser.py"
    assert target_for("python_exec", {"code": "1"}, "Python") is None
    assert target_for("shell_exec", {"command": "pytest"}, "pytest") == "pytest"
    assert target_for("file_write", {}, "Edit") is None


# ── phases ──────────────────────────────────────────────────────────────────


def phases(sequence: str) -> list[tuple[str | None, int]]:
    """The phase and cycle after each call; a ``P`` is a passing T call."""
    state = PhaseState()
    out = []
    for symbol in sequence:
        state.add("T" if symbol == "P" else symbol, symbol == "P")
        out.append((state.phase, state.cycle))
    return out


def test_phases_and_cycles() -> None:
    assert phases("RRWTWP") == [
        (EXPLORE, 0),
        (EXPLORE, 0),
        (WRITING, 0),
        (VERIFY, 1),
        (WRITING, 1),
        (FINISHING, 2),
    ]


def test_a_read_after_the_first_write_keeps_the_phase() -> None:
    assert phases("RWRXT")[2:] == [(WRITING, 0), (WRITING, 0), (VERIFY, 1)]


def test_every_call_after_the_first_pass_is_finishing() -> None:
    assert [phase for phase, _ in phases("TPRWT")] == [
        VERIFY,
        FINISHING,
        FINISHING,
        FINISHING,
        FINISHING,
    ]


def test_a_test_without_a_write_before_it_is_no_cycle() -> None:
    assert phases("TTWWTT")[-1] == (VERIFY, 1)
    assert phases("WRT")[-1] == (VERIFY, 1)


def test_phase_agrees() -> None:
    assert phase_agrees(EXPLORE, {"R"})
    assert phase_agrees(EXPLORE, {"X", "W"})
    assert not phase_agrees(EXPLORE, {"W"})
    assert phase_agrees(WRITING, {"W"})
    assert not phase_agrees(WRITING, {"R"})
    assert phase_agrees(VERIFY, {"T"})
    assert phase_agrees(FINISHING, {"R"})
    assert not phase_agrees(None, {"R"})


def test_a_unittest_failure_after_ok_is_no_pass() -> None:
    output = "Ran 3 tests in 0.1s\n\nOK\n\nRan 3 tests\nFAILED (errors=1)"
    summary = parse_test_summary(output)
    assert summary is not None and not summary.is_pass
    once = "Ran 3 tests in 0.1s\n\nOK\nthen\nFAILED (failures=2)"
    assert parse_test_summary(once) == TestSummary(1, 2, 0)


def test_odd_outputs_and_names_never_raise() -> None:
    assert call_failed("success", None) is False  # type: ignore[arg-type]
    assert call_failed(None, "") is True  # type: ignore[arg-type]
    assert parse_test_summary(None) is None  # type: ignore[arg-type]
    assert parse_test_summary(b"3 passed") is None  # type: ignore[arg-type]
    assert label_for(5, {}) == "5"  # type: ignore[arg-type]
    assert label_for(None, {}) == "tool"  # type: ignore[arg-type]
    assert classify_call(None, {}) == "X"  # type: ignore[arg-type]


def test_targets_are_redacted() -> None:
    key = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    args = {"path": f"/tmp/{key}"}
    assert target_for("file_read", args, label_for("file_read", args)) == "[redacted]"
