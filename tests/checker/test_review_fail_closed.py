"""Shell text that is not read exactly makes the step undecided.

The review reads here-documents and the commands inside substitutions with a
small reader. Where that reader cannot be sure it has the text right, it does
not guess: the step is undecided, and nothing after a here-document that was
not read exactly is taken for its body. Three kinds of shape:

- a here-document whose operator or delimiter is not in a plain form (``<<-``,
  a partly quoted, escaped or longer delimiter, two on a line, a delimiter
  line that never comes, a marker that is only text in a quote or a comment);
- a command substitution in a here-document with nested quotes, nested
  substitutions or groups that cannot be balanced exactly;
- a name that held a started process and is written in some other place (an
  option with the name attached, an element, arithmetic in brackets, an
  integer variable, a name built by an expansion, a descriptor).

Ordinary here-documents and substitutions keep the verdicts they had.
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


def is_delete(found: ActionReview) -> bool:
    return found.verdict == "high" and "delete" in found.categories


# ── A here-document that is not in a plain form is not read, and hides nothing ──

RM = "rm -rf ~/Documents"


@pytest.mark.parametrize(
    "command",
    [
        # a partly quoted delimiter
        f'cat <<E"O"F\nx\nEOF\n{RM}\nE"O"F',
        f"cat <<E'O'F\nx\nEOF\n{RM}\nEOF",
        # a delimiter with more word after it, or built by an expansion
        f"cat <<EOF.txt\nx\nEOF.txt\n{RM}",
        f'cat <<"EOF"x\nx\nEOFx\n{RM}',
        f"cat <<$X\nx\n{RM}",
        # an escaped delimiter
        f"cat <<\\EOF\nx\nEOF\n{RM}",
        # <<-, which strips tabs from the delimiter line
        f"cat <<-EOF\n\tx\n\tEOF\n{RM}",
        # two here-documents on a line
        f"cat <<A <<B\nx\nA\ny\nB\n{RM}",
        # a delimiter line that never comes, or not exactly
        f"cat <<EOF\nx\n{RM}",
        f"cat <<EOF\nx\nEOF \n{RM}",
        f"cat <<EOF\nx\n EOF\n{RM}",
        # the operator's line goes on after a backslash
        f"cat <<EOF \\\n> out\nx\nEOF\n{RM}",
        # the operator is the last thing in the text
        f"{RM}; cat <<EOF",
    ],
)
def test_a_here_document_not_read_exactly_swallows_nothing(
    ws: Path, command: str
) -> None:
    assert is_delete(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        'cat <<E"O"F\nx\nEOF\nE"O"F',
        "cat <<EOF.txt\nx\nEOF.txt",
        "cat <<-EOF\n\tx\n\tEOF",
        "cat <<A <<B\nx\nA\ny\nB",
        "cat <<EOF\nx",
        "cat <<EOF\nx\nEOF ",
        "cat <<",
        "cat <<EOF",
    ],
)
def test_a_here_document_not_read_exactly_is_undecided(ws: Path, command: str) -> None:
    found = sh(ws, command)
    assert found.verdict == "undecided"
    assert any("here-document" in reason for reason in found.reasons)


@pytest.mark.parametrize(
    "command",
    [
        f'echo "<<EOF"\n{RM}\nEOF',
        f"echo '<<EOF'\n{RM}\nEOF",
        f"# cat <<EOF\n{RM}\nEOF",
        f"echo hi # cat <<EOF\n{RM}\nEOF",
        f"echo a <<< EOF\n{RM}\nEOF",
    ],
)
def test_a_marker_that_is_only_text_starts_no_here_document(
    ws: Path, command: str
) -> None:
    assert is_delete(sh(ws, command))


# ── A command substitution in a here-document is read exactly or not at all ──


@pytest.mark.parametrize(
    "command",
    [
        f'cat <<EOF\n$(echo "$(echo ")")" ; {RM})\nEOF',
        f'cat <<EOF\n$(echo ")"; {RM})\nEOF',
        f"cat <<EOF\n$(echo '(' ; {RM})\nEOF",
        f'cat <<EOF\n$(echo "a $(echo b) c" ; {RM})\nEOF',
        f"cat <<EOF\n$(echo $((1 + 2)); {RM})\nEOF",
        f"cat <<EOF\n$(echo ${{x:-y}}; {RM})\nEOF",
        f"cat <<EOF\n$(echo a; (echo b); {RM})\nEOF",
    ],
)
def test_a_nested_substitution_read_exactly_is_reviewed_to_its_end(
    ws: Path, command: str
) -> None:
    assert is_delete(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        f"cat <<EOF\n$(case x in a) {RM};; esac)\nEOF",
        "cat <<EOF\n$(echo `date`)\nEOF",
        'cat <<EOF\n$(echo "unclosed)\nEOF',
        "cat <<EOF\n$(echo 'unclosed)\nEOF",
        'cat <<EOF\n$(echo ${x:-")"})\nEOF',
        f"cat <<EOF\n$(echo a # ) b\n{RM})\nEOF",
        "cat <<EOF\n$(cat <<'IN'\ninner\nIN\n)\nEOF",
        "cat <<EOF\n" + "$(echo " * 8 + "x" + ")" * 8 + "\nEOF",
        "cat <<EOF\n$(echo a\nEOF",
        "cat <<EOF\n`echo \\`date\\``\nEOF",
        "cat <<EOF\n`echo a\nEOF",
    ],
)
def test_a_substitution_that_cannot_be_balanced_exactly_is_not_routine(
    ws: Path, command: str
) -> None:
    found = sh(ws, command)
    assert found.verdict != "routine"
    assert found.verdict == "high" or any(
        "here-document" in reason for reason in found.reasons
    )


@pytest.mark.parametrize(
    "command",
    ["echo $((1 << 3))", "x=$((1<<4)); echo $x", "(( y = 1 << 2 )); echo $y"],
)
def test_a_shift_in_arithmetic_is_not_told_from_a_here_document(
    ws: Path, command: str
) -> None:
    """Melo does not guess which ``<<`` starts a document: such a step is undecided."""
    found = sh(ws, command)
    assert found.verdict == "undecided"
    assert any("here-document" in reason for reason in found.reasons)


# ── A name that held a started process is dropped when it is written elsewhere ──

STARTED = "sleep 9 & pid=$!; "


@pytest.mark.parametrize(
    "command",
    [
        # an output option with the name attached, and elements of the name
        STARTED + "printf -vpid %s 1234; kill $pid",
        STARTED + "printf -v pid %s 1234; kill $pid",
        STARTED + "printf -v 'pid[0]' %s 1234; kill $pid",
        STARTED + "read -apid < /tmp/p; kill $pid",
        STARTED + "read -a pid < /tmp/p; kill $pid",
        STARTED + "read -rapid < /tmp/p; kill $pid",
        STARTED + "mapfile -t pid < /tmp/p; kill $pid",
        STARTED + "readarray pid < /tmp/p; kill $pid",
        STARTED + "wait -n -ppid; kill $pid",
        STARTED + "wait -n -p pid; kill $pid",
        STARTED + "getopts ab pid; kill $pid",
        STARTED + "read 'pid[0]' < /tmp/p; kill $pid",
        # arithmetic where it is not in (( ))
        STARTED + "arr[pid=5]=1; kill $pid",
        STARTED + "echo ${arr[pid=5]}; kill $pid",
        STARTED + "echo $[ pid=5 ]; kill $pid",
        STARTED + "[[ pid=5 -eq 5 ]]; kill $pid",
        STARTED + "[[ arr[pid=7] -eq 1 ]]; kill $pid",
        STARTED + "declare -a list=([pid=5]=x); kill $pid",
        # a variable that evaluates what it is given as arithmetic
        STARTED + "declare -i n; n='pid=7'; kill $pid",
        STARTED + "typeset -i n; n='pid=7'; kill $pid",
        STARTED + "local -i n; n='pid=7'; kill $pid",
        STARTED + "declare -ai n; n='pid=7'; kill $pid",
        # a name built by an expansion, a descriptor, a coprocess
        STARTED + 'ref=pid; printf -v "$ref" %s 1; kill $pid',
        STARTED + "ref=pid; printf -v$ref %s 1; kill $pid",
        STARTED + "ref=pid; read $ref < /tmp/p; kill $pid",
        STARTED + 'ref=pid; unset "$ref"; kill $pid',
        STARTED + "ref=pid; wait -p $ref; kill $pid",
        STARTED + "exec {pid}< /tmp/p; kill $pid",
        STARTED + "coproc pid { sleep 1; }; kill $pid",
    ],
)
def test_any_other_write_of_a_started_name_drops_it(ws: Path, command: str) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        STARTED + "wait $pid; kill $pid",
        STARTED + 'wait "$pid"; kill $pid',
        STARTED + "printf '%s\\n' $pid; kill $pid",
        STARTED + "printf -v line %s 1; kill $pid",
        STARTED + "read -r line < /tmp/p; kill $pid",
        STARTED + "read -ra words < /tmp/p; kill $pid",
        STARTED + "mapfile -t lines < /tmp/p; kill $pid",
        STARTED + "arr[1]=x; kill $pid",
        STARTED + "echo ${arr[1]}; kill $pid",
        STARTED + "[[ $pid -eq 5 ]]; kill $pid",
        STARTED + '[ -n "$pid" ]; kill $pid',
        STARTED + "[[ -n $pid ]]; kill $pid",
        STARTED + "declare -a list; kill $pid",
        STARTED + "declare -x flag=1; kill $pid",
        STARTED + "tail -f log --pid=$pid; kill $pid",
        STARTED + "exec 3< /tmp/p; kill $pid",
        STARTED + "coproc other { sleep 1; }; kill $pid",
        STARTED + "echo $(( 1 + 2 )); kill $pid",
    ],
)
def test_a_started_name_stays_when_only_other_names_are_written(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


# ── Ordinary here-documents and substitutions keep their verdicts ──

# Recorded from the rules before this change (command, verdict, categories).
GUARD: list[tuple[str, str, tuple[str, ...]]] = [
    ("cat <<EOF\nhello\nEOF", "routine", ()),
    ("cat > notes.txt <<EOF\nhello $USER\nEOF", "routine", ()),
    ("cat <<'EOF' > script.sh\n#!/bin/bash\necho \"$1\"\nEOF", "routine", ()),
    ('cat <<"EOF"\nliteral $(rm -rf ~/Documents)\nEOF', "routine", ()),
    ("python3 <<'PY'\nprint('hi')\nPY", "routine", ()),
    ("python3 - <<PY\nimport os\nprint(os.getcwd())\nPY", "routine", ()),
    ("bash <<'EOF'\necho hi\nEOF", "routine", ()),
    (
        "git commit -m \"$(cat <<'EOF'\nFix the parser\n\nLonger text here.\nEOF\n)\"",
        "routine",
        (),
    ),
    ('git commit -m "$(cat <<EOF\nFix the parser\nEOF\n)"', "routine", ()),
    ("cat <<EOF\ntoday is $(date)\nEOF", "routine", ()),
    ("cat <<EOF\nuser: $(whoami) in $(pwd)\nEOF", "routine", ()),
    ("cat <<EOF\nversion: $(git describe --tags)\nEOF", "routine", ()),
    ('cat <<EOF\n$(basename "$PWD")\nEOF', "routine", ()),
    ('cat <<EOF\n$(dirname "$(pwd)")\nEOF', "routine", ()),
    ("cat <<EOF\n$(printf '%s\\n' \"$HOME\")\nEOF", "routine", ()),
    ("cat <<EOF\n${HOME}/bin and $((1 + 2))\nEOF", "routine", ()),
    ("cat <<EOF\n`date`\nEOF", "routine", ()),
    ("cat <<A > a.txt\nx\nA\ncat <<B > b.txt\ny\nB", "routine", ()),
    ("cat <<EOF | sort\nb\na\nEOF", "routine", ()),
    ("tee out.txt <<EOF\ndata\nEOF", "routine", ()),
    ('while read line; do echo "$line"; done <<EOF\na\nb\nEOF', "routine", ()),
    ("cat <<EOF\ndon't stop\nEOF", "routine", ()),
    ("cat <<EOF\nline1\n\nline3\nEOF\nls", "routine", ()),
    ("cat <<EOF\n# not a comment\n#!/bin/sh\nEOF", "routine", ()),
    ('echo "$(date)"', "routine", ()),
    ("echo $(date +%s)", "routine", ()),
    ("x=$(ls | wc -l); echo $x", "routine", ()),
    ('echo "$(basename "$0")"', "routine", ()),
    ('echo "files: $(ls "$HOME")"', "routine", ()),
    ("latest=$(ls -t | head -1)", "routine", ()),
    ('out=$(git rev-parse --short HEAD) && echo "$out"', "routine", ()),
    ('echo "a<<b"', "routine", ()),
    ("echo '<<EOF'; ls", "routine", ()),
    ("awk 'BEGIN{print 1<<2}'", "routine", ()),
    ("python3 -c 'print(1 << 3)'", "routine", ()),
    ("echo $((1 + 2))", "routine", ()),
    ("cat <<EOF >> ~/notes.txt\nentry\nEOF", "high", ("outside_write",)),
    ("ssh host <<'EOF'\nls\nEOF", "undecided", ()),
    ('result=$(cat <<EOF\ntext\nEOF\n)\necho "$result"', "routine", ()),
]


@pytest.mark.parametrize(("command", "verdict", "categories"), GUARD)
def test_an_ordinary_script_keeps_its_verdict(
    ws: Path, command: str, verdict: str, categories: tuple[str, ...]
) -> None:
    found = sh(ws, command)
    assert found.verdict == verdict
    for category in categories:
        assert category in found.categories
