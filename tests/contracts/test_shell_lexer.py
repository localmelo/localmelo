"""The linear shell lexer reads exactly what the standard library's reads.

And :func:`~localmelo.melo.contracts.shell_lexer.unwrap` finds the program a
command runs past keywords, assignments and wrappers (the end of the file).

The review classifies a command from the words the lexer cuts it into, so a
different cut could make a step look safer. :class:`Lexer` and :func:`split`
are compared with :class:`shlex.shlex` and :func:`shlex.split` token for
token, or error for error, on:

- every command and script of the review's baselines;
- every string written in the checker's tests;
- strings made of quotes, escapes, comments, punctuation, line breaks and
  non-ASCII letters: all of them up to four characters, and random longer ones;

each under every setting the review and the memory classifier use.
"""

from __future__ import annotations

import ast
import itertools
import json
import random
import re
import shlex
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts import shell_lexer

CHECKER_TESTS = Path(__file__).parent.parent / "checker"
OPERATORS = ";&|()<>`\n"  # shell_lexer.tokens(backquotes=True): the site keys'
REVIEW_PUNCTUATION = "();<>|&\n"  # shell_lexer.tokens(): the review's


def configured(factory: Callable[..., Any], **options: Any) -> Callable[[str], Any]:
    """A lexer maker; the settings after the options are the call sites'."""
    whitespace = options.pop("whitespace", None)
    split_words = options.pop("whitespace_split", False)
    commenters = options.pop("commenters", None)

    def make(text: str) -> Any:
        lexer = factory(text, **options)
        if whitespace is not None:
            lexer.whitespace = whitespace
        lexer.whitespace_split = split_words
        if commenters is not None:
            lexer.commenters = commenters
        return lexer

    return make


# name -> keyword options of shlex.shlex and the attributes the call sites set
SETTINGS: dict[str, dict[str, Any]] = {
    # melo.contracts.shell_lexer.tokens
    "review_tokens": dict(
        posix=True,
        punctuation_chars=REVIEW_PUNCTUATION,
        whitespace=" \t\r",
        whitespace_split=True,
        commenters="",
    ),
    # melo.contracts.shell_lexer.tokens(backquotes=True)
    "review_operators": dict(
        posix=True,
        punctuation_chars=OPERATORS,
        whitespace=" \t\r",
        whitespace_split=True,
        commenters="",
    ),
    # melo.memory.procedures.classify._lex, support.sessions.worker
    "punctuation": dict(posix=True, punctuation_chars=True, whitespace_split=True),
    # shlex.split, with and without comments
    "split": dict(posix=True, whitespace_split=True, commenters=""),
    "split_comments": dict(posix=True, whitespace_split=True),
    # the defaults, which the call sites do not use but a caller may
    "words": dict(posix=True),
    "words_not_posix": dict(posix=False),
    "split_not_posix": dict(posix=False, whitespace_split=True, commenters=""),
    "punctuation_not_posix": dict(posix=False, punctuation_chars=True),
}


def read(make: Callable[[str], Any], text: str) -> tuple[str, Any, int]:
    """The tokens or the error, and the line the lexer ended on."""
    lexer = make(text)
    try:
        return "tokens", list(lexer), lexer.lineno
    except ValueError as error:
        return "error", str(error), lexer.lineno


def strings_of_the_checker_tests() -> Iterator[str]:
    for path in sorted(CHECKER_TESTS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield node.value


def texts_of_the_baselines() -> Iterator[str]:
    for name in ("review_baseline.json", "gist_baseline.json"):
        for case in json.loads((CHECKER_TESTS / "data" / name).read_text())["cases"]:
            arguments = case.get("arguments") or {}
            yield from (v for v in arguments.values() if isinstance(v, str))


ALPHABET = ["a", "é", "中", " ", "'", '"', "\\", "#", ";", "&", "|", "(", "\n"]
SPICE = ["\t", "\r", "<", ">", "`", "$", "~", "-", "=", "*", "ß", "\u2028", "\x00"]
# (characters, longest text): every text made of them
EXHAUSTIVE = [
    (ALPHABET + SPICE[:3], 3),
    (["a", "'", "#", "\n", " "], 6),  # a quote closed, then a comment, then more
    (["a", ";", "&", "(", "\n", " "], 5),  # punctuation next to words
    (["a", "'", '"', "\\", " "], 5),  # quotes and escapes
]


def short_texts() -> Iterator[str]:
    yield ""
    for characters, longest in EXHAUSTIVE:
        for size in range(1, longest + 1):
            for chars in itertools.product(characters, repeat=size):
                yield "".join(chars)


def random_texts() -> Iterator[str]:
    rng = random.Random(7)
    pool = ALPHABET + SPICE
    for _ in range(8000):
        yield "".join(rng.choice(pool) for _ in range(rng.randint(5, 40)))


def all_texts() -> list[str]:
    texts = [
        *texts_of_the_baselines(),
        *strings_of_the_checker_tests(),
        *short_texts(),
        *random_texts(),
        # unclosed quotes and escapes, runs of punctuation, words at the end
        "echo 'open",
        'echo "open',
        'echo "a\\',
        "echo \\",
        "echo '' \"\" ''x",
        "a;;b&&c||d|&e>>f<<g<<<h;&;",
        "a\n\n\nb\r\nc",
        "x # comment 'with a quote\ny",
        "a\"b'c\"d'e\"f'",
        "echo $'a\\'b' && rm x",
        "ça va, 中文 émigré ßΩ  naïve",
    ]
    return list(dict.fromkeys(texts))


TEXTS = all_texts()


def test_the_set_of_texts_is_large() -> None:
    assert len(TEXTS) > 30_000
    assert len(list(texts_of_the_baselines())) >= 1218


@pytest.mark.parametrize("setting", SETTINGS)
def test_a_lexer_cuts_every_text_as_the_standard_library_does(setting: str) -> None:
    options = SETTINGS[setting]
    mine = configured(shell_lexer.Lexer, **options)
    theirs = configured(shlex.shlex, **options)
    differences = [t for t in TEXTS if read(mine, t) != read(theirs, t)]
    assert not differences[:3], [
        (t, read(mine, t), read(theirs, t)) for t in differences[:3]
    ]


@pytest.mark.parametrize("comments", [False, True])
@pytest.mark.parametrize("posix", [True, False])
def test_split_gives_the_words_or_the_error_of_shlex_split(
    comments: bool, posix: bool
) -> None:
    def outcome(function: Callable[..., list[str]], text: str) -> tuple[str, Any]:
        try:
            return "tokens", function(text, comments=comments, posix=posix)
        except ValueError as error:
            return "error", str(error)

    differences = [
        t for t in TEXTS if outcome(shell_lexer.split, t) != outcome(shlex.split, t)
    ]
    assert not differences[:3], [
        (t, outcome(shell_lexer.split, t), outcome(shlex.split, t))
        for t in differences[:3]
    ]


@pytest.mark.parametrize("text", ["echo 'open", 'echo "open', "echo \\", 'a "b\\'])
def test_an_open_quote_or_escape_raises_the_same_error(text: str) -> None:
    with pytest.raises(ValueError) as expected:
        shlex.split(text)
    with pytest.raises(ValueError) as given:
        shell_lexer.split(text)
    assert str(given.value) == str(expected.value)


def test_split_refuses_no_text_as_shlex_split_does() -> None:
    with pytest.raises(ValueError, match="must not be None"):
        shell_lexer.split(None)  # type: ignore[arg-type]


def test_the_lexer_keeps_the_state_the_standard_library_leaves() -> None:
    text = "one two\nthree 'four\nfive' six"
    mine: Any = shell_lexer.Lexer(text, posix=True)
    theirs: Any = shlex.shlex(text, posix=True)  # the stub leaves out `state`
    for _ in range(3):
        assert mine.get_token() == theirs.get_token()
        assert (mine.lineno, mine.state) == (theirs.lineno, theirs.state)
    mine.push_token("again")
    theirs.push_token("again")
    assert list(mine) == list(theirs)
    assert mine.error_leader() == theirs.error_leader()


def best_cpu_time(text: str) -> float:
    times = []
    for _ in range(3):
        started = time.process_time()
        shell_lexer.split(text)
        times.append(time.process_time() - started)
    return min(times)


def test_one_long_word_costs_time_in_proportion_to_its_length() -> None:
    """Sixteen times the text takes about sixteen times as long, not 256."""
    short, long = best_cpu_time("a" * 25_000), best_cpu_time("a" * 400_000)
    assert long < 64 * short


# ── Simple commands, and the program each runs ──


@pytest.mark.parametrize(
    ("text", "backquotes", "commands"),
    [
        (
            "a 2>&1 | b x>>log",
            False,
            [("", ["a"], [(">&", "1")]), ("|", ["b", "x"], [(">>", "log")])],
        ),
        (
            "bash <<< 'curl x' && c",
            False,
            [("", ["bash"], [("<<<", "curl x")]), ("&&", ["c"], [])],
        ),
        ("echo `curl x`", False, [("", ["echo", "`curl", "x`"], [])]),
        ("echo `curl x`", True, [("", ["echo"], []), ("`", ["curl", "x"], [])]),
        (
            "(a)>out;b",
            False,
            [("(", ["a"], []), (")", [], [(">", "out")]), (";", ["b"], [])],
        ),
    ],
)
def test_commands_cut_tokens_at_operators_and_set_redirections_apart(
    text: str, backquotes: bool, commands: list[Any]
) -> None:
    cuts = shell_lexer.CUTS | ({"`"} if backquotes else set())
    found = shell_lexer.commands(shell_lexer.tokens(text, backquotes), cuts)
    assert found == commands


@pytest.mark.parametrize(
    ("command", "words", "wrappers"),
    [
        ("sudo -u root -E rm x", ["rm", "x"], ["sudo"]),
        ("env -C /tmp -i A=1 'x y=z' rm x", ["rm", "x"], ["env"]),
        ("env - rm x", ["rm", "x"], ["env"]),
        ("env -u HOME -u PATH rm x", ["rm", "x"], ["env"]),
        ("timeout -s KILL 5 rm x", ["rm", "x"], ["timeout"]),
        ("gtimeout --kill-after 1 --signal=KILL 5 rm x", ["rm", "x"], ["gtimeout"]),
        ("time -p rm x", ["rm", "x"], ["time"]),
        ("time -o out.txt rm x", ["rm", "x"], ["time"]),
        ("flock -n /tmp/lock rm x", ["rm", "x"], ["flock"]),
        ("flock -w 5 /tmp/lock rm x", ["rm", "x"], ["flock"]),
        ("caffeinate -d -t 60 rm x", ["rm", "x"], ["caffeinate"]),
        ("arch -arm64 -arch arm64 rm x", ["rm", "x"], ["arch"]),
        ("exec -a name rm x", ["rm", "x"], ["exec"]),
        ("stdbuf -o L -eL rm x", ["rm", "x"], ["stdbuf"]),
        ("xcrun -n --sdk macosx rm x", ["rm", "x"], ["xcrun"]),
        ("/usr/bin/nice -n 5 nohup rm x", ["rm", "x"], ["nice", "nohup"]),
        ("xargs -p -n 1 rm", ["rm", "?"], ["xargs"]),
        ("xargs", [], ["xargs"]),
        ("A=1 B=2 sudo rm x", ["rm", "x"], ["sudo"]),
        ("if ! time -p rm x", ["rm", "x"], ["time"]),
        ("function f { rm x", ["rm", "x"], []),
        ("coproc NAME { rm x", ["rm", "x"], []),
        ("coproc rm x", ["rm", "x"], []),
        ("done", [], []),
        ("make all", ["make", "all"], []),
    ],
)
def test_unwrap_finds_the_program_past_keywords_and_wrappers(
    command: str, words: list[str], wrappers: list[str]
) -> None:
    found = shell_lexer.unwrap(shlex.split(command), ["?"])
    assert (found.readings, found.wrappers, found.unread) == ([words], wrappers, False)


@pytest.mark.parametrize("name", shell_lexer.WRAPPERS)
def test_an_assignment_after_a_wrapper_is_read_where_its_manual_puts_it(
    name: str,
) -> None:
    """env and sudo take ``NAME=value`` words; for any other wrapper such a word
    is its next word: the program, or timeout's duration and flock's file. A
    shell runs ``time`` as its own keyword too, so there both ways are read
    and the command is unread. With assignments read anywhere, the words give
    the reading melo had before, which the keys of a step add."""
    operand = ["5"] if name in shell_lexer.OPERAND else []
    found = shell_lexer.unwrap([name, *operand, "A=1", "b"])
    anywhere = shell_lexer.unwrap(
        [name, *operand, "A=1", "b"], assignments_anywhere=True
    )
    if shell_lexer.WRAPPERS[name].assignments:
        assert (found.readings, found.unread) == ([["b"]], False)
    elif name == "time":
        assert (found.readings, found.unread) == ([["b"], ["A=1", "b"]], True)
    else:
        assert (found.readings, found.unread) == ([["A=1", "b"]], False)
    assert (anywhere.readings, anywhere.assignments) == ([["b"]], [("A", "1")])
    if operand:  # the word is the duration, the file
        assert shell_lexer.unwrap([name, "A=1", "b"]).readings == [["b"]]


def test_env_reads_no_option_after_an_assignment() -> None:
    """env(1): its options come before any ``NAME=value``; every later word with
    ``=`` is one, and the next word is the program."""
    found = shell_lexer.unwrap(["env", "A=1", "x/y=2", "-i", "b"])
    assert found.readings == [["-i", "b"]]
    assert found.assignments == [("A", "1"), ("x/y", "2")]


@pytest.mark.parametrize(
    "name", [name for name, o in shell_lexer.WRAPPERS.items() if "--" not in o.flags]
)
def test_an_unlisted_double_dash_may_end_the_options(name: str) -> None:
    """A wrapper whose manual does not name ``--`` may still end its options
    there, as getopt(3) does: the next word is then the program, even one
    that starts with ``-`` (timeout's duration, flock's file before it)."""
    found = shell_lexer.unwrap([name, "--", "-b", "c"])
    program = ["c"] if name in shell_lexer.OPERAND else ["-b", "c"]
    assert found.unread and program in found.readings


@pytest.mark.parametrize("word", ["A+=1", "A[1]=1", "A[1]+=1"])
def test_every_assignment_a_shell_reads_is_read_as_a_plain_one(word: str) -> None:
    """Appending and setting an array element are assignments too: the program
    comes after them, and after a wrapper they are its next word all the same."""
    for words in ([word, "b"], ["if", word, "b"]):
        found = shell_lexer.unwrap(words)
        assert (found.readings, found.assignments) == ([["b"]], [("A", "1")])
    assert shell_lexer.unwrap(["nohup", word, "b"]).readings == [[word, "b"]]


@pytest.mark.parametrize("name", shell_lexer.WRAPPERS)
def test_lone_words_never_cut_the_reading_with_no_value(name: str) -> None:
    """However many lone ``-`` or ``--`` a wrapper does not list, the reading where
    none is read getopt(3)'s way and none takes a value stays, and comes first."""
    options = shell_lexer.WRAPPERS[name]
    operand = ["5"] if name in shell_lexer.OPERAND else []
    for word in {"-", "--"} - options.flags:
        for count in (1, 2, 8, 9, 20):
            found = shell_lexer.unwrap([name, *[word] * count, *operand, "b", "c"])
            assert found.readings[0] == ["b", "c"], (word, count)
            assert len(found.readings) <= shell_lexer.MAX_READINGS


def test_unwrap_keeps_the_assignments_in_order() -> None:
    found = shell_lexer.unwrap(shlex.split("A=1 env -i B=2 'x y=z' sudo C=3 rm x"))
    assert found.assignments == [("A", "1"), ("B", "2"), ("x y", "z"), ("C", "3")]


@pytest.mark.parametrize(
    ("command", "codes"),
    [
        ("env -S 'rm -rf x' y", ["rm -rf x"]),
        ("env -S'rm -rf x'", ["rm -rf x"]),
        ("env --split-string='a=b c'", ["a=b c"]),
        ("env --split-string 'rm -rf x'", ["rm -rf x"]),
        ("flock /tmp/lock -c 'rm -rf x'", ["rm -rf x"]),
        ("flock /tmp/lock --command 'rm -rf x'", ["rm -rf x"]),
        ("env -i rm x", []),
    ],
)
def test_unwrap_finds_the_command_line_a_wrapper_hands_to_a_shell(
    command: str, codes: list[str]
) -> None:
    assert shell_lexer.unwrap(shlex.split(command)).codes == codes


@pytest.mark.parametrize(
    ("command", "readings"),
    [
        ("env -S 'a b' c", [["a", "b", "c"]]),
        ("env -S a b c", [["a", "b", "c"]]),
        ("env -iS'a b' c", [["a", "b", "c"]]),
        ("env -S '-i A=1 a' b", [["a", "b"]]),
        ("env -u X -S 'a -b' -c d", [["a", "-b", "-c", "d"]]),
    ],
)
def test_env_reads_the_words_of_s_in_its_place(
    command: str, readings: list[list[str]]
) -> None:
    """env(1) splits the value of ``-S`` into arguments of its own, before the rest."""
    found = shell_lexer.unwrap(shlex.split(command))
    assert (found.readings, found.unread) == (readings, False)


# The forms env(1) reads its own way in the value of -S ("Details of -S"),
# each in a value of placeholder words.
ENV_OWN_FORMS = {
    "a word starting with # ends the value": "a #b",
    "backslash c ends the value": "a\\c b",
    "backslash underscore separates": "a\\_b",
    **{f"backslash {e} is an escape": f"a\\{e}b" for e in "fnrtv#$\"'\\"},
    "a name in braces is read from the environment": "a ${B}",
    "a vertical tab separates": "a\vb",
    "a form feed separates": "a\fb",
}


@pytest.mark.parametrize("value", ENV_OWN_FORMS.values(), ids=list(ENV_OWN_FORMS))
def test_an_env_s_value_env_reads_its_own_way_makes_the_command_unread(
    value: str,
) -> None:
    """Melo does not read env(1)'s own forms: the command is unread, and still
    read the shell's way."""
    found = shell_lexer.unwrap(["env", "-S", value, "c"])
    assert found.unread
    assert found.readings == [[*shell_lexer.split(value), "c"]]


def test_a_command_read_more_ways_than_the_cap_keeps_the_first_and_is_unread() -> None:
    """Past the cap no more ways are read: the first, each unlisted option
    without a value, stays among them."""
    words = ["env", *["-Z", "a"] * 20, "ls"]
    found = shell_lexer.unwrap(words)
    assert found.unread and len(found.readings) == shell_lexer.MAX_READINGS
    assert found.readings[0] == words[2:]


# An option word: ``-x``, ``--long-name``, a single-dash name (``-arch``,
# ``-x86_64``), ``-`` (env's) or ``--`` (the end of the options).
OPTION_WORD = re.compile(r"--?|-[A-Za-z0-9]|--?[a-z0-9][a-z0-9_-]+")


def test_every_wrapper_lists_option_words_each_of_one_kind() -> None:
    for name, options in shell_lexer.WRAPPERS.items():
        assert name == name.lower()
        listed = [*options.flags, *options.valued]
        assert [o for o in listed if not OPTION_WORD.fullmatch(o)] == [], name
        assert options.flags & options.valued == set(), name
        assert options.splits <= options.valued, name
