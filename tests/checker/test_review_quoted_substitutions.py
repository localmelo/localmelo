"""A command substitution with quotes in it is read to its own end and reviewed on its own.

The lexer that cuts a command into steps does not know ``$(``. A double quote
inside a substitution that is itself inside double quotes ends the string for
it, so a command in the substitution can look like quoted text, and a quote
the shell pairs one way can be paired another way by the lexer. Each
substitution that has a quote, a backslash or a ``case`` in it is therefore
read from the text itself and reviewed as a command of its own; where the end
cannot be found, or the lexer would read the text after it in another quote,
the step is undecided.

A name that held a started process also loses that standing when it may be
written inside brackets of any length: a bracket expression too long to be
scanned is not skipped, it takes the standing from every name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.checker.payloads import ActionReviewPayload
from localmelo.melo.contracts.review import ActionReview


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    folder = tmp_path / "home" / "ws"
    folder.mkdir(parents=True)
    return folder


def review(ws: Path, tool: str, arguments: dict[str, Any]) -> ActionReview:
    return Checker().review_action(
        ActionReviewPayload(
            tool,
            arguments,
            str(ws),
            home=str(ws.parent),
            executable="/opt/melo/bin/python3",
            prefix="/opt/melo",
        )
    )


def sh(ws: Path, command: str) -> ActionReview:
    return review(ws, "shell_exec", {"command": command})


def is_system(found: ActionReview) -> bool:
    return found.verdict == "high" and "system" in found.categories


# ── An operation that reaches the system is found wherever it is written ──

OPERATIONS = [
    "pkill foo",
    "sudo ls",
    "killall Dock",
    "defaults write com.example key 1",
    "shutdown now",
]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_each_operation_is_system_when_written_on_its_own(
    ws: Path, operation: str
) -> None:
    assert is_system(sh(ws, operation))


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "shape",
    [
        # a quoted parenthesis in the substitution ends the string for the lexer
        'echo "$(echo ")"; {op})"',
        'echo "$(echo "(" ; {op})"',
        "echo \"$(printf '%s' ')'; {op})\"",
        # the same one level down, and next to another substitution
        'echo "a $(echo "b $(echo ")"; {op})")"',
        'echo "$(echo "x") $(echo ")"; {op})"',
        'echo "$(echo "a" "$(echo ")"; {op})")"',
        # assigned, and after a command that opens quotes of its own
        'x="$(echo ")"; {op})"',
        'd="$(cd "$(dirname "$0")" && echo ")"; {op})"',
        'a="$(echo ")"; {op})" && b="$(echo "(" ; true)"',
        # inside a here-document that expands it
        'cat <<EOF\n"$(echo ")"; {op})"\nEOF',
        'cat <<EOF\n$(echo "$(echo ")")" ; {op})\nEOF',
    ],
)
def test_an_operation_in_a_substitution_with_quotes_is_reviewed(
    ws: Path, shape: str, operation: str
) -> None:
    assert is_system(sh(ws, shape.format(op=operation)))


def test_an_operation_in_python_text_inside_such_a_substitution_is_reviewed(
    ws: Path,
) -> None:
    found = sh(ws, "echo \"$(echo ')'; python3 -c 'import os; os.kill(1, 9)')\"")
    assert is_system(found)


# ── What the lexer would read in another quote makes the step undecided ──


@pytest.mark.parametrize(
    "command",
    [
        # the lexer takes the apostrophe in the substitution for an opening quote
        'echo "$(echo "\'")"; pkill foo; echo "\'"',
        # a hash after a space starts a comment for the lexer, not for the shell
        'echo "$(echo "a" "b #c")"; pkill foo',
        # an escaped quote
        'echo "$(echo \\")"; pkill foo)"',
        # a case, whose patterns end in a parenthesis
        'echo "$(case x in a) echo ")"; pkill foo;; esac)"',
        'echo "$(case x in a) echo 1; pkill foo;; esac)"',
        # no end
        'echo "$(echo "x"',
        'echo "$(echo \'x)"',
    ],
)
def test_a_substitution_the_lexer_would_read_wrongly_is_not_routine(
    ws: Path, command: str
) -> None:
    found = sh(ws, command)
    assert found.verdict != "routine"
    assert found.verdict == "high" or any(
        "melo cannot read exactly" in reason or "does not close" in reason
        for reason in found.reasons
    )


# ── Ordinary scripts with quotes in a substitution stay as they were ──


@pytest.mark.parametrize(
    "command",
    [
        'echo "$(dirname "$0")"',
        'echo "$(basename "$f")"',
        'cd "$(dirname "$0")" && ls',
        'd="$(cd "$(dirname "$0")" && pwd)"; echo "$d"',
        'echo "Today: $(date +"%Y-%m-%d")"',
        'echo "$(printf \'%s\\n\' "$x")"',
        'echo "$(echo "a b")"',
        'echo "$(ls "$HOME" | wc -l)"',
        'files="$(find . -name "*.py" | head -3)"',
        'echo "$(grep -c "x" f.txt)"',
        'echo "version: $(python3 -c "print(1)")"',
        'echo "$(echo `date`)"',
        "echo \"$(cat <<'EOF'\nhello\nEOF\n)\"",
        'git commit -m "$(cat <<\'EOF\'\nDon\'t break "it"\n\nMore (text).\nEOF\n)"',
        'git commit -m "$(cat <<EOF\nFix "the" parser\nEOF\n)"',
        "echo '$(echo \")\")'",
        'echo "one" "$(echo "two")" "$(echo "three")"',
        'for f in "$(ls "$d")"; do echo "$f"; done',
    ],
)
def test_an_ordinary_substitution_with_quotes_stays_routine(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


# ── A subscript too long to scan takes the standing from every name ──

STARTED = "sleep 9 & pid=$!; "
BOUND = 200  # the longest bracket expression that is scanned


def padded(head: str, tail: str, size: int) -> str:
    """*head* + spaces + *tail* with exactly *size* characters between ``[`` and ``]``."""
    inside = head + " " * (size - len(head) - len(tail)) + tail
    assert len(inside) == size
    return inside


@pytest.mark.parametrize("size", [BOUND - 1, BOUND, BOUND + 1, 2 * BOUND, 5000])
@pytest.mark.parametrize(
    "shape",
    [
        "arr[{inside}]=1; kill $pid",
        "echo ${{arr[{inside}]}}; kill $pid",
        "echo $[{inside}]; kill $pid",
        "[[{inside}]]; kill $pid",
    ],
)
def test_a_write_in_a_subscript_at_or_over_the_bound_drops_the_name(
    ws: Path, shape: str, size: int
) -> None:
    inside = padded("pid=5", " ", size)
    assert is_system(sh(ws, STARTED + shape.format(inside=inside)))


@pytest.mark.parametrize("size", [BOUND + 1, 3 * BOUND])
def test_a_subscript_over_the_bound_takes_the_standing_from_every_name(
    ws: Path, size: int
) -> None:
    """Melo does not trust what it cannot scan: it is not skipped."""
    inside = padded("other=5", " ", size)
    assert is_system(sh(ws, STARTED + f"arr[{inside}]=1; kill $pid"))


def test_a_subscript_at_the_bound_is_scanned_and_writes_nothing_to_the_name(
    ws: Path,
) -> None:
    inside = padded("other=5", " ", BOUND)
    assert sh(ws, STARTED + f"arr[{inside}]=1; kill $pid").verdict == "routine"


@pytest.mark.parametrize(
    "command",
    [
        "[[ pid=5 -eq\n 5 ]]; kill $pid",
        "arr[\npid=5]=1; kill $pid",
        "echo ${arr[ 1 +\n (pid=5) ]}; kill $pid",
    ],
)
def test_a_subscript_over_several_lines_drops_the_name(ws: Path, command: str) -> None:
    assert is_system(sh(ws, STARTED + command))


@pytest.mark.parametrize(
    "command",
    [
        "arr[1]=x; kill $pid",
        "echo [a-z]*; kill $pid",
        '[ -n "$pid" ] && kill $pid',
        "[[ $pid -gt 1 ]]; kill $pid",
        "echo ']'; kill $pid",
        "echo '[' ; kill $pid",
    ],
)
def test_short_brackets_that_write_nothing_keep_the_standing(
    ws: Path, command: str
) -> None:
    assert sh(ws, STARTED + command).verdict == "routine"


def test_a_long_bracket_text_without_a_started_process_is_routine(ws: Path) -> None:
    inside = "x" * 5000
    assert sh(ws, f"echo [{inside}]").verdict == "routine"
