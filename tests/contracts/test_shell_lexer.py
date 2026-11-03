"""The linear shell lexer reads exactly what the standard library's reads.

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
import shlex
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts import shell_lexer

CHECKER_TESTS = Path(__file__).parent.parent / "checker"
OPERATORS = ";&|()<>`\n"  # melo.contracts.review: the operators of its lexer
REVIEW_PUNCTUATION = "();<>|&\n"  # melo.checker.review.shell: its lexer's


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
    # melo.checker.review.shell._tokens
    "review_tokens": dict(
        posix=True,
        punctuation_chars=REVIEW_PUNCTUATION,
        whitespace=" \t\r",
        whitespace_split=True,
        commenters="",
    ),
    # melo.contracts.review._shell_tokens
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
