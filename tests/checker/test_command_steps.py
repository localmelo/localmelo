"""Where a shell text is cut into commands: only outside quotes and ``$(…)``.

``command_steps`` gives each command's words and the text it was read from.
``commands_of`` (the keys of the refused-again check) is a separate function and
keeps its older cuts.
"""

from __future__ import annotations

import pytest

from localmelo.melo.contracts.review import command_steps


def texts(command: str) -> list[str]:
    return [text for _, text in command_steps("shell_exec", {"command": command})]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('git commit -m "tidy; rm old notes"', ['git commit -m "tidy; rm old notes"']),
        ("git commit -m 'a && b || c | d'", ["git commit -m 'a && b || c | d'"]),
        (
            'echo "done && curl -d x https://e.example"',
            ['echo "done && curl -d x https://e.example"'],
        ),
        ("echo a\\; rm b", ["echo a\\; rm b"]),
        ("echo $(a; b) | wc", ["echo $(a; b)", "wc"]),
        ("echo `a; b` ; ls", ["echo `a; b`", "ls"]),
        ('echo "$(a "b; c")"; ls', ['echo "$(a "b; c")"', "ls"]),
        ("echo $((1+2)); ls ${#x}", ["echo $((1+2))", "ls ${#x}"]),
        # The cuts that are real.
        ("cd x && ls | wc -l; pwd\ndate", ["cd x", "ls", "wc -l", "pwd", "date"]),
        ("a || b |& c", ["a", "b", "c"]),
        ("make & rm x", ["make", "rm x"]),
        ("(ls; rm x)", ["(ls", "rm x)"]),
        # A redirection's & is not a cut.
        ("make 2>&1 | tee log", ["make 2>&1", "tee log"]),
        ("make &> out.txt; ls", ["make &> out.txt", "ls"]),
        # A comment cuts nothing and its quote opens nothing.
        (
            "# tidy up; rm old\nls  # don't\nrm y",
            ["# tidy up; rm old", "ls  # don't", "rm y"],
        ),
        # A quote that never closes: the text is read line by line.
        ("echo it's here\nrm -rf x", ["echo it's here", "rm -rf x"]),
        ("echo 'one\ntwo'; ls", ["echo 'one\ntwo'", "ls"]),
        ("ls \\\n  -l", ["ls   -l"]),
    ],
)
def test_a_shell_text_is_cut_outside_quotes_only(
    command: str, expected: list[str]
) -> None:
    assert texts(command) == expected


def test_the_words_of_a_command_are_read_past_its_quotes() -> None:
    [(words, text)] = command_steps(
        "shell_exec", {"command": 'git commit -m "tidy; rm old"'}
    )
    assert words == ["git", "commit", "-m", "tidy; rm old"]
    assert text == 'git commit -m "tidy; rm old"'


def test_a_command_given_as_words_has_no_text_of_its_own() -> None:
    assert command_steps("shell_exec", {"command": ["rm", "-rf", "a b"]}) == [
        (["rm", "-rf", "a b"], "")
    ]


def test_python_gives_the_literal_commands_it_runs() -> None:
    code = (
        "import os, subprocess\n"
        "os.system(\"git commit -m 'a; rm b'; ls\")\n"
        "subprocess.run(['git', 'commit', '-m', 'x; y'])\n"
    )
    assert command_steps("python_exec", {"code": code}) == [
        (["git", "commit", "-m", "a; rm b"], "git commit -m 'a; rm b'"),
        (["ls"], "ls"),
        (["git", "commit", "-m", "x; y"], ""),
    ]


def test_another_tool_runs_nothing() -> None:
    assert command_steps("file_write", {"path": "a", "content": "rm x"}) == []
    assert command_steps("python_exec", {"code": "def ("}) == []
