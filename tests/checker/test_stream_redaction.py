from __future__ import annotations

import ast
import random
import re
from collections.abc import Sequence
from pathlib import Path
from re import _parser  # type: ignore[attr-defined]
from typing import Any

import pytest

from localmelo.melo.checker import redaction
from localmelo.melo.checker._stream_redaction import LineRedactor
from localmelo.tests.checker.test_pem_redaction import block, stream
from localmelo.tests.checker.test_redaction import EXAMPLES


@pytest.mark.parametrize(
    "text",
    [
        "plain line\nsecond line\nlast",
        "start sk-" + "a" * 30 + "\nend",
        "ghp_" + "b" * 40 + "\nAKIA" + "A" * 16 + "\n",
        "Bearer\n  " + "x" * 40 + "\nnext\n",
        "123456 -- OTP\nverification code: 765432\n",
        "中文 验证码 123456\nhello\n",
    ],
)
def test_every_utf8_byte_split_matches_whole_redaction(text: str) -> None:
    import codecs

    raw = text.encode("utf-8")
    expected = redaction.redact_secrets(text)[0]
    for offset in range(len(raw) + 1):
        decoder = codecs.getincrementaldecoder("utf-8")()
        release = LineRedactor()
        out = release.feed(decoder.decode(raw[:offset]))
        out += release.feed(decoder.decode(raw[offset:], final=True))
        assert out + release.finish() == expected


def test_completed_line_releases_and_open_bearer_waits() -> None:
    release = LineRedactor()
    assert release.feed("safe\nBearer\n ") == "safe\n"
    assert release.feed("x" * 20 + "\nnext\n") == "Bearer\n [redacted]\nnext\n"
    assert release.finish() == ""


def test_a_comment_added_to_the_rules_file_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stream once held every reply until its end when the rules' source
    # differed from a pinned copy, a comment included.
    assert redaction.__file__ is not None
    edited = tmp_path / "redaction.py"
    edited.write_text(Path(redaction.__file__).read_text() + "\n# a new comment\n")
    monkeypatch.setattr(redaction, "__file__", str(edited))
    release = LineRedactor()
    assert release.feed("first\nsecond\n") == "first\nsecond\n"
    assert release.finish() == ""


# The streamed answer is redacted one complete line at a time, which is the
# same as redacting it whole only if no rule matches across a line break
# (or looks at text past one). Two rules do by design, and LineRedactor
# holds what they need: a PEM block from its BEGIN line to its END line,
# and the bearer word and its white space until the token arrives. Three
# tests keep that true: the patterns are walked for a newline, redaction.py
# is read for any regex use that walk would not see, and the Python between
# the matches is compared with whole-text redaction on streamed splits.
RULES = {"_PEM", "_KEY", "_BEARER", "_CODE_KEYWORD", "_CODE"}
SPANNING_BY_DESIGN = {"_PEM", "_BEARER"}


def _can_see_a_newline(nodes: Sequence[Any], dotall: bool) -> bool:
    """Whether a part of a parsed pattern can match, or look at, a newline.

    Anything it does not know counts as yes, so a new construct is reviewed.
    """
    for op, arg in nodes:
        kind = str(op)
        if kind == "LITERAL":
            seen = arg == ord("\n")
        elif kind == "NOT_LITERAL":
            seen = arg != ord("\n")
        elif kind == "ANY":
            seen = dotall
        elif kind == "IN":
            seen = _set_has_a_newline(arg)
        elif kind in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
            seen = _can_see_a_newline(arg[2], dotall)
        elif kind == "SUBPATTERN":
            seen = _can_see_a_newline(arg[3], dotall or bool(arg[1] & re.DOTALL))
        elif kind == "BRANCH":
            seen = any(_can_see_a_newline(branch, dotall) for branch in arg[1])
        elif kind in ("ASSERT", "ASSERT_NOT"):
            seen = _can_see_a_newline(arg[1], dotall)
        elif kind == "AT":  # \b, \Z: they look at no character
            seen = False
        else:
            seen = True
        if seen:
            return True
    return False


def _set_has_a_newline(items: Sequence[Any]) -> bool:
    """Whether a character set of a parsed pattern contains a newline."""
    negated = bool(items) and str(items[0][0]) == "NEGATE"
    inside = False
    for op, arg in items[1:] if negated else items:
        kind = str(op)
        if kind == "LITERAL":
            inside |= arg == ord("\n")
        elif kind == "RANGE":
            inside |= arg[0] <= ord("\n") <= arg[1]
        elif kind == "CATEGORY":
            inside |= str(arg) not in (
                "CATEGORY_DIGIT",
                "CATEGORY_WORD",
                "CATEGORY_NOT_SPACE",
            )
        else:
            return True
    return not inside if negated else inside


def test_only_the_two_rules_the_stream_holds_can_match_across_lines() -> None:
    rules = {
        name: value
        for name, value in vars(redaction).items()
        if isinstance(value, re.Pattern)
    }
    # A new rule is a new pattern here: decide whether it may span lines
    # (then LineRedactor must hold what it needs) before it is released.
    assert set(rules) == RULES
    spanning = {
        name
        for name, rule in rules.items()
        if _can_see_a_newline(
            _parser.parse(rule.pattern, rule.flags), rule.flags & re.DOTALL > 0
        )
    }
    assert spanning == SPANNING_BY_DESIGN


# Where redaction.py matches: each rule at one call, and nowhere else.
RULE_SITES = [
    ("_BEARER", "subn"),
    ("_CODE", "finditer"),
    ("_CODE_KEYWORD", "finditer"),
    ("_KEY", "subn"),
    ("_PEM", "sub"),
]
MATCH_METHODS = {
    "match",
    "fullmatch",
    "search",
    "findall",
    "finditer",
    "sub",
    "subn",
    "split",
}


def _is_re_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
    )


def test_redaction_matches_with_no_regex_the_walk_above_cannot_see() -> None:
    # The walk above reads the patterns bound at the top of redaction.py. A
    # pattern written inline, kept in a list, copied to a local name or
    # imported from another module would ship without that walk.
    assert redaction.__file__ is not None
    tree = ast.parse(Path(redaction.__file__).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"__future__", "bisect", "re", "typing"}
    top = {
        target.id: stmt.value
        for stmt in tree.body
        if isinstance(stmt, ast.Assign) and _is_re_call(stmt.value)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }
    assert set(top) == RULES
    # Every use of the re module is one of those five compiles.
    every = {id(call) for call in ast.walk(tree) if _is_re_call(call)}
    assert every == {id(call) for call in top.values()}
    assert all(call.func.attr == "compile" for call in top.values())  # type: ignore[attr-defined]
    # Each rule is used once, as the receiver of its one match call, and no
    # other match call is made on anything (a call on a non-name included).
    uses = sorted(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id in RULES
        and isinstance(node.ctx, ast.Load)
    )
    assert uses == [name for name, _ in RULE_SITES]
    calls = sorted(
        (
            node.func.value.id if isinstance(node.func.value, ast.Name) else "?",
            node.func.attr,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in MATCH_METHODS
    )
    assert calls == RULE_SITES


def test_any_split_of_the_secret_fixtures_streams_to_the_whole_redaction() -> None:
    # What the line-by-line release rests on, whatever the rules are written
    # as: redacting a text in pieces gives what redacting it whole gives. The
    # fixtures are the redaction examples (a keyword and its code on adjacent
    # lines, keys, bearer tokens) and PEM blocks, in random order, cut at
    # random places, inside lines and inside secrets, and at every line end.
    fixtures = [text + "\n" for text, _, _ in EXAMPLES] + [
        "Bearer\n  " + "x" * 30 + "\n",
        block("RSA PRIVATE KEY") + "\n",
        block("PRIVATE KEY") + "\n",
    ]
    rng = random.Random(11)
    for _ in range(300):
        text = "".join(rng.sample(fixtures, len(fixtures)))
        whole = redaction.redact_secrets(text)[0]
        cuts = tuple(
            sorted(rng.randrange(len(text) + 1) for _ in range(rng.randint(1, 40)))
        )
        assert stream(text, cuts) == whole, cuts
        by_line = tuple(i + 1 for i, char in enumerate(text) if char == "\n")
        assert stream(text, by_line) == whole
