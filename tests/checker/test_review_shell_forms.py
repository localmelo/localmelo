"""Shell spellings that the review read differently from the shell.

Each case is a shape that an independent review found passing as ``routine``
(or being read as another command) after the rules for steps that reach macOS
or destroy work were hardened:

- a program table keyed by a name the lookup never produces (``Rscript``);
- options that take a value inside a cluster (``bash -c -ol pipefail '…'``);
- names that held a started process and are set again by a loop, a ``read``
  behind a keyword or a prefix, an arithmetic assignment, a sourced file, a
  name reference or a pipeline;
- substitutions in the body of a here-document that expands them;
- ANSI-C quotes (``$'…'``), which the lexer and the shell end differently;
- text so long that reading it would cost more than it is worth.

The read-only forms next to them stay routine.
"""

from __future__ import annotations

import time
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


def py(ws: Path, code: str) -> ActionReview:
    return review(ws, "python_exec", {"code": code})


def is_system(found: ActionReview) -> bool:
    return found.verdict == "high" and "system" in found.categories


def is_delete(found: ActionReview) -> bool:
    return found.verdict == "high" and "delete" in found.categories


# ── Program tables are looked up by the lower-case name ──


def test_every_program_table_is_keyed_in_lower_case() -> None:
    """The program is lower-cased before the lookup, so a capital never matches."""
    from localmelo.melo.checker.review import shell

    tables = [
        "_AWKS",
        "_BARE",
        "_BUILDS",
        "_CONTAINERS",
        "_DECLARERS",
        "_EDITORS",
        "_ELSEWHERE",
        "_FILE_RUNNERS",
        "_GNU_NAMES",
        "_HOOKED",
        "_HTTPIE",
        "_INTERPRETERS",
        "_LAUNCHABLE",
        "_MELO",
        "_NETWORK",
        "_PACKAGE_RUNNERS",
        "_PREFIXES",
        "_READERS",
        "_RECURSIVE",
        "_RUNNERS",
        "_RUN_WRAPPERS",
        "_SHELLS",
        "_VMS",
        "_WRAPPERS",
        "_WRAPPER_VALUED",
        "_WRITERS",
    ]
    for name in tables:
        names = list(getattr(shell, name))
        assert names, f"{name} is empty or gone"
        assert [n for n in names if n != n.lower()] == [], name


@pytest.mark.parametrize(
    "command",
    [
        "Rscript -e 'system(\"id\")'",
        "R -e 'print(1)'",
        "R --vanilla -e 'print(1)'",
        "echo 'print(1)' | Rscript -",
        "rscript -e 'print(1)'",
        "/usr/local/bin/Rscript -e 'print(1)'",
    ],
)
def test_r_code_the_review_does_not_read_is_a_doubt(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "undecided"


def test_running_an_r_script_file_stays_routine(ws: Path) -> None:
    assert sh(ws, "Rscript analysis.R").verdict == "routine"


# ── Options that take a value, inside a cluster ──


@pytest.mark.parametrize(
    "command",
    [
        "bash -c -ol pipefail 'rm -rf ~/Documents'",
        "bash -c -Ol extglob 'rm -rf ~/Documents'",
        "bash -c -oo pipefail errexit 'rm -rf ~/Documents'",
        "bash -c -o pipefail 'rm -rf ~/Documents'",
        "bash -c -O extglob 'rm -rf ~/Documents'",
        "bash -ec -ol pipefail 'rm -rf ~/Documents'",
    ],
)
def test_an_o_option_inside_a_cluster_takes_its_value(ws: Path, command: str) -> None:
    assert is_delete(sh(ws, command))


def test_a_shell_option_with_its_value_before_a_harmless_command_stays_routine(
    ws: Path,
) -> None:
    assert sh(ws, "bash -c -ol pipefail 'echo hi'").verdict == "routine"


# ── The value of an option is not an option ──


@pytest.mark.parametrize(
    "command",
    [
        "pkill -u -0 Safari",
        "pkill -U -0 Safari",
        "pkill -g -0 Safari",
        "pkill -P -0 Safari",
        "pkill -t -0 ttys001",
        "pkill -F -0 Safari",
        "killall -u -0 Safari",
        "pkill -f -u -0 Safari",
    ],
)
def test_a_minus_zero_that_is_an_options_value_is_not_signal_zero(
    ws: Path, command: str
) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "pkill -0 Safari",
        "pkill -u bob -0 Safari",
        "pkill -f -0 Safari",
        "killall -0 Safari",
        "kill -0 1234",
        "kill -s 0 1234",
    ],
)
def test_signal_zero_stays_a_check(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


# ── The body of a function or a coprocess is a command ──


@pytest.mark.parametrize(
    "command",
    [
        "function f { rm -rf ~/Documents; }; f",
        "function f() { rm -rf ~/Documents; }; f",
        "f() { rm -rf ~/Documents; }; f",
        "coproc rm -rf ~/Documents",
        "coproc X { rm -rf ~/Documents; }",
        "coproc { rm -rf ~/Documents; }",
    ],
)
def test_the_body_of_a_function_or_coprocess_is_reviewed(
    ws: Path, command: str
) -> None:
    assert is_delete(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    ["function f { echo hi; }; f", "coproc cat", "coproc X { cat; }", "f() { :; }; f"],
)
def test_a_harmless_function_or_coprocess_stays_routine(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


# ── A name that held $! and is given another value ──


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & pid=$!; sleep 1 | while read pid; do :; done; kill $pid",
        "sleep 9 & pid=$!; echo 5 | while read pid; do kill $pid; done",
        "sleep 9 & pid=$!; IFS= read -r pid < /tmp/p; kill $pid",
        "sleep 9 & pid=$!; IFS=: read pid rest < /tmp/p; kill $pid",
        "sleep 9 & pid=$!; if true; then for pid in $(pgrep Safari); do kill $pid; done; fi",
        "sleep 9 & pid=$!; while true; do read pid < /tmp/p; kill $pid; done",
        "sleep 9 & pid=$!; if true; then read pid < /tmp/p; fi; kill $pid",
        "sleep 9 & pid=$!; while read pid; do kill $pid; done < /tmp/p",
        "sleep 9 & pid=$!; until read pid; do :; done < /tmp/p; kill $pid",
        "sleep 9 & pid=$!; if true; then select pid in 1 2; do kill $pid; done; fi",
    ],
)
def test_a_while_read_of_a_started_name_is_not_own(ws: Path, command: str) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & pid=$!; IFS= read -r pid; kill $pid",
        "sleep 9 & pid=$!; LC_ALL=C read pid; kill $pid",
        "sleep 9 & pid=$!; IFS= mapfile pid < /tmp/p; kill $pid",
        "sleep 9 & pid=$!; do read pid; kill $pid",
    ],
)
def test_an_assignment_prefix_before_read_still_reassigns(
    ws: Path, command: str
) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & pid=$!; (( pid = 1234 )); kill $pid",
        "sleep 9 & pid=$!; ((pid=1234)); kill $pid",
        "sleep 9 & pid=$!; (( pid += 1 )); kill $pid",
        "sleep 9 & pid=$!; (( pid++ )); kill $pid",
        "sleep 9 & pid=$!; (( ++pid )); kill $pid",
        "sleep 9 & pid=$!; for (( pid = 0; pid < 3; pid++ )); do kill $pid; done",
        "sleep 9 & pid=$!; echo $(( pid = 1234 )); kill $pid",
        "sleep 9 & pid=$!; let pid=1234; kill $pid",
        "sleep 9 & pid=$!; let 'pid = 1234'; kill $pid",
        'sleep 9 & pid=$!; let "pid += 1"; kill $pid',
        "sleep 9 & pid=$!; let n=1 'pid = 5'; kill $pid",
        "sleep 9 & pid=$!; unset pid; kill $pid",
        "sleep 9 & pid=$!; printf -v pid %s 1234; kill $pid",
    ],
)
def test_an_arithmetic_assignment_to_a_started_name_is_not_own(
    ws: Path, command: str
) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & pid=$!; source /tmp/x.sh; kill $pid",
        "sleep 9 & pid=$!; . /tmp/x.sh; kill $pid",
        "sleep 9 & pid=$!; builtin source /tmp/x.sh; kill $pid",
        "sleep 9 & pid=$!; declare -n ref=pid; ref=1234; kill $pid",
        "sleep 9 & pid=$!; typeset -n ref=pid; ref=1234; kill $pid",
        "sleep 9 & pid=$!; local -n ref=pid; ref=1234; kill $pid",
        "sleep 9 & pid=$!; declare -gn ref=pid; ref=1234; kill $pid",
        "sleep 9 & pid=$!; command eval 'pid=1234'; kill $pid",
    ],
)
def test_a_sourced_file_or_a_name_reference_could_set_any_name(
    ws: Path, command: str
) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & echo x | pid=$!; kill $pid",
        "sleep 9 & pid=$! | cat; kill $pid",
        "sleep 9 & echo x | { pid=$!; }; kill $pid",
        "sleep 9 & { pid=$!; } | cat; kill $pid",
        "sleep 9 & pid=$! |& cat; kill $pid",
    ],
)
def test_dollar_bang_set_inside_a_pipeline_is_not_own(ws: Path, command: str) -> None:
    assert is_system(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "sleep 9 & pid=$!; while read line; do echo $line; done < f; kill $pid",
        "sleep 9 & pid=$!; for i in 1 2; do echo $i; done; kill $pid",
        "sleep 9 & pid=$!; (( n = 1 )); kill $pid",
        "sleep 9 & pid=$!; (( n++ )); kill $pid",
        "sleep 9 & pid=$!; (( pid == 1 )); kill $pid",
        "sleep 9 & pid=$!; (( pid <= 1 )); kill $pid",
        "sleep 9 & pid=$!; let n=1; kill $pid",
        "sleep 9 & pid=$!; IFS= read -r line < f; kill $pid",
        "sleep 9 & pid=$!; sleep 1 | cat; kill $pid",
        "sleep 9 & pid=$!; if true; then wait $pid; fi; kill $pid",
        "sleep 9 & pid=$!; do_it=1; kill $pid",
        "sleep 9 & pid=$!; declare -a list; kill $pid",
        "sleep 9 & pid=$!; kill $pid; sleep 1 | cat",
        "{ sleep 9 & pid=$!; }; kill $pid",
    ],
)
def test_other_names_and_reads_leave_a_started_pid_alone(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


# ── A here-document that expands its body runs the substitutions in it ──


@pytest.mark.parametrize(
    "command",
    [
        "cat <<EOF\n$(rm -rf ~/Documents)\nEOF",
        "cat <<EOF\n`rm -rf ~/Documents`\nEOF",
        "cat <<-EOF\n\t$(rm -rf ~/Documents)\nEOF",
        "cat <<EOF > notes.txt\nvalue: ${x:-$(rm -rf ~/Documents)}\nEOF",
        "cat <<EOF\nfirst\n # $(rm -rf ~/Documents)\nEOF",
        "cat <<EOF\n$(\nrm -rf ~/Documents\n)\nEOF",
        "cat <<EOF\n$(echo $(rm -rf ~/Documents))\nEOF",
        "cat <<EOF\nok\nEOF\ncat <<EOF\n$(rm -rf ~/Documents)\nEOF",
        "python3 <<EOF\nprint('$(rm -rf ~/Documents)')\nEOF",
        "cat <<EOF\n$(rm -rf ~/Documents",
    ],
)
def test_a_substitution_in_an_unquoted_heredoc_is_reviewed(
    ws: Path, command: str
) -> None:
    assert is_delete(sh(ws, command))


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'EOF'\n$(rm -rf ~/Documents)\nEOF",
        'cat <<"EOF"\n$(rm -rf ~/Documents)\nEOF',
        "cat <<'EOF'\n`rm -rf ~/Documents`\nEOF",
        "cat <<EOF\n\\$(rm -rf ~/Documents)\nEOF",
        "cat <<EOF\n\\`rm -rf ~/Documents\\`\nEOF",
        "cat <<EOF\ntoday is $(date)\nEOF",
        "cat <<EOF\nsum: $((1 + 2))\nEOF",
        "cat <<EOF\nplain text\nEOF",
    ],
)
def test_a_quoted_or_harmless_heredoc_stays_routine(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "routine"


# ── ANSI-C quotes end where the shell ends them, not where the lexer does ──


def test_ansi_c_quoting_does_not_hide_the_next_command(ws: Path) -> None:
    found = sh(ws, "echo $'\\'' ; rm -rf ~/Documents #'")
    assert found.verdict != "routine"
    assert any("quote" in reason for reason in found.reasons)


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m $'a\\nb'",
        "grep $'\\t' notes.txt",
        "echo $'\\x2f'",
        "echo ok; echo $'x'",
        "bash -c \"echo $'x'\"",
    ],
)
def test_any_ansi_c_quote_is_a_doubt(ws: Path, command: str) -> None:
    assert sh(ws, command).verdict == "undecided"


@pytest.mark.parametrize(
    "command",
    [
        "grep -c '$' notes.txt",
        'echo "cost: $\' end"',
        "echo \\$'x'",
        "echo '$'",
        "sed -n '$p' notes.txt",
        "awk '{ print \"$\" }' notes.txt",
    ],
)
def test_a_dollar_before_a_quote_that_is_not_ansi_c_stays_routine(
    ws: Path, command: str
) -> None:
    assert sh(ws, command).verdict == "routine"


# ── Text too long to read ──

LIMIT = 256 * 1024


def test_a_command_over_the_size_limit_is_undecided(ws: Path) -> None:
    found = sh(ws, "echo " + "a" * LIMIT)
    assert found.verdict == "undecided"
    assert any("too long" in reason for reason in found.reasons)


def test_python_code_over_the_size_limit_is_undecided(ws: Path) -> None:
    found = py(ws, "x = 1\n" * (LIMIT // 6 + 1))
    assert found.verdict == "undecided"
    assert any("too long" in reason for reason in found.reasons)


def test_the_size_limit_is_a_quarter_of_a_megabyte() -> None:
    from localmelo.melo.checker import review

    assert review.MAX_TEXT == LIMIT


def test_a_text_at_the_size_limit_is_still_read(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit is on the length: a text of exactly that length is reviewed."""
    from localmelo.melo.checker import review

    monkeypatch.setattr(review, "MAX_TEXT", 400)
    assert sh(ws, "echo " + "a" * 395).verdict == "routine"
    assert is_delete(sh(ws, "rm -rf ~/Documents; echo " + "a" * 375))
    assert sh(ws, "echo " + "a" * 396).verdict == "undecided"
    assert py(ws, "x = 1\n" * 66 + "pass").verdict == "routine"
    assert py(ws, "x = 1\n" * 67).verdict == "undecided"


def test_a_long_command_listed_as_a_list_counts_by_its_whole_text(ws: Path) -> None:
    words = ["echo", "a" * 1000] * 270
    found = review(ws, "shell_exec", {"command": words})
    assert found.verdict == "undecided"


# ── The reading of a long text is bounded ──

BIG = 100_000


def timed(ws: Path, command: str) -> float:
    from localmelo.melo.checker.review.gist import step_gist

    arguments = {"command": command, "why": "Delete a.txt"}
    # CPU time of the fastest of three runs, so that a busy machine (a parallel
    # test run) cannot fail a bound meant to catch a rule that backtracks.
    times = []
    for _ in range(3):
        started = time.process_time()
        step_gist("shell_exec", arguments, review(ws, "shell_exec", arguments))
        times.append(time.process_time() - started)
    return min(times)


@pytest.mark.parametrize(
    ("tool", "key", "text"),
    [
        ("shell_exec", "command", "echo " + "a" * (LIMIT + 1)),
        ("shell_exec", "command", "echo " + "a " * (LIMIT // 2)),
        ("python_exec", "code", "x = 1\n" * (LIMIT // 6 + 1)),
    ],
)
def test_neither_the_review_nor_the_card_reads_a_text_over_the_limit(
    ws: Path, tool: str, key: str, text: str
) -> None:
    """The sentence of the approval card is not built from such a text either."""
    from localmelo.melo.checker.review.gist import step_gist

    arguments = {key: text, "why": "Run a long step"}
    started = time.process_time()  # CPU time: a busy machine cannot fail it
    found = review(ws, tool, arguments)
    gist = step_gist(tool, arguments, found)
    assert time.process_time() - started < 1.0
    assert found.verdict == "undecided"
    assert gist is not None


@pytest.mark.parametrize(
    "command",
    [
        "echo " + "`true` " * (BIG // 7),
        "echo `" + "a " * (BIG // 2),
        "echo " + "$(true) " * (BIG // 8),
        "cat <<EOF\n" + "$(true) `true`\n" * (BIG // 15) + "EOF",
    ],
)
def test_review_of_a_long_substitution_text_is_fast(ws: Path, command: str) -> None:
    """Under 2 seconds, or three times what a plain command of the same size costs."""
    plain = timed(ws, "echo " + "a" * BIG)
    assert timed(ws, command) < max(2.0, 3 * plain)
