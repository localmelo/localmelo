from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from re import _parser  # type: ignore[attr-defined]
from typing import Any

import pytest

from localmelo.melo.checker import redaction
from localmelo.melo.checker._stream_redaction import LineRedactor


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
# and the bearer word and its white space until the token arrives.
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
    assert set(rules) == {"_PEM", "_KEY", "_BEARER", "_CODE_KEYWORD", "_CODE"}
    spanning = {
        name
        for name, rule in rules.items()
        if _can_see_a_newline(
            _parser.parse(rule.pattern, rule.flags), rule.flags & re.DOTALL > 0
        )
    }
    assert spanning == SPANNING_BY_DESIGN
