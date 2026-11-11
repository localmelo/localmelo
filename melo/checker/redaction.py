"""Minimal redaction of secrets in a model's text.

:func:`redact_secrets` replaces four kinds of secret with
:data:`REDACTED`: PEM private key blocks, well-known key shapes, bearer
tokens, and one-time codes next to a code keyword. It is a pure function, for
text that leaves the run's own memory, such as the ``thinking`` and ``content``
of an ``assistant.message`` event.
History's ``llm_calls`` table keeps the ``thinking`` and ``content`` of
every model response redacted by it too.
"""

from __future__ import annotations

import bisect
import re
from typing import Any, Final

REDACTED: Final = "[redacted]"

# Pass 1, PEM private key blocks: the BEGIN and END lines stay, the body
# between them goes. Which BEGIN and END lines open and close a block is read in
# code, in _pem_blocks. Only labels that end in PRIVATE KEY match (RSA, EC, DSA,
# OPENSSH, ENCRYPTED, plain PKCS#8), and the PGP armor's PRIVATE KEY BLOCK, the
# same for both lines; a label has at most 3 words of at most 16 characters
# before PRIVATE KEY, so a scan for a line stays linear. The stream release
# reads the same two line patterns.
_PEM_LABEL = r" (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY(?: BLOCK)?-----"
PEM_BEGIN = re.compile("-----BEGIN" + _PEM_LABEL)
PEM_END = re.compile("-----END" + _PEM_LABEL)

# Pass 2, key shapes, each match whole: an "sk-" key, a GitHub token, an
# AWS access key id. Each takes the whole run of its characters and starts
# after none of them, so a replaced key has no ASCII letter or digit next
# to it.
_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"
    r"|(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"
    r"|(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"
)

# Pass 3, bearer tokens: only the token after the word is replaced. The word
# has no ASCII letter, digit or underscore right before it; any other character
# (a Chinese character, an accented letter, punctuation) lets it count.
# ``(?-i:`` keeps IGNORECASE from reading the Kelvin sign, the long s and the
# dotless i as ASCII letters. The stream release reads the same word.
BEARER_WORD: Final = r"(?-i:(?<![A-Za-z0-9_]))bearer"
_BEARER = re.compile(r"(" + BEARER_WORD + r"\s+)[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)

# Pass 4, one-time codes: 4 to 8 ASCII digits, with no ASCII letter or digit
# on either side, within _CODE_REACH characters of a code keyword and on its
# line.
_CODE_KEYWORD = re.compile(
    r"verification code|passcode|(?<![A-Za-z0-9_])otp(?![A-Za-z0-9_])|验证码",
    re.IGNORECASE,
)
_CODE = re.compile(r"(?<![A-Za-z0-9])[0-9]{4,8}(?![A-Za-z0-9])")
_CODE_REACH = 20


def redact_secrets(text: str) -> tuple[str, int]:
    """Return *text* with its secrets replaced by :data:`REDACTED`, and their count.

    Four passes run in this order, each on the previous pass's output:

    1. PEM private key blocks. The BEGIN lines ``-----BEGIN <label>PRIVATE
       KEY-----`` and END lines ``-----END <label>PRIVATE KEY-----`` of the
       text are read in order (the label is up to 3 upper-case words, so RSA,
       EC, DSA, OPENSSH, ENCRYPTED and plain PKCS#8 all match; ``PRIVATE KEY
       BLOCK``, the PGP armor, too; the label of an END line need not be the
       BEGIN line's). A block opens at a BEGIN line when none is open; while
       one is open, a further BEGIN line is part of its body. It closes at an
       END line after which the next BEGIN or END line is a BEGIN line, or
       after which there is none: the last of a run of END lines, so the text
       between two END lines is body too. A block with no END line runs to the
       end of the text, and an END line with no block open is nothing. The
       BEGIN line and the closing END line stay and the body becomes one line,
       :data:`REDACTED`: ``BEGIN``, newline, ``[redacted]``, newline, ``END``.
       ``PUBLIC KEY`` and ``CERTIFICATE`` blocks, and ``ssh-ed25519`` public
       key lines, do not match. A block already in that shape is left alone
       and not counted. The count goes up by one per block.
    2. Key shapes, the whole match: ``sk-`` followed by at least 20 of
       ``[A-Za-z0-9_-]``, not preceded by one of them (so ``task-…`` stays);
       ``ghp_``, ``gho_``, ``ghu_``, ``ghs_`` or ``ghr_`` followed by at least
       36 ASCII letters and digits, or ``github_pat_`` followed by at least
       22 of ``[A-Za-z0-9_]``, neither preceded by ``[A-Za-z0-9_]``; and
       ``AKIA`` or ``ASIA`` followed by exactly 16 of ``[A-Z0-9]``, with no
       ASCII letter or digit on either side.
    3. Bearer tokens: the word ``bearer`` in any case with no ASCII letter,
       digit or underscore right before it (a Chinese character, an accented
       letter, punctuation or the start of the text before it does not
       prevent it; ``torchbearer`` gives nothing), whitespace, then at least
       16 of ``[A-Za-z0-9._~+/=-]``. Only the token is replaced:
       ``Bearer [redacted]``.
    4. One-time codes: a run of 4 to 8 ASCII digits, with no ASCII letter or
       digit right before or after it, that starts at most 20 characters
       after a code keyword ends, or ends at most 20 characters before one
       starts, with no newline between them. The keywords, in any case, are
       ``verification code``, ``passcode``, ``OTP`` as a whole word and
       ``验证码``. Only the code is replaced.

    A replaced span never has an ASCII letter or digit next to it, and
    :data:`REDACTED` matches no pattern, so a second call replaces nothing,
    with one exception: an ``sk-`` key or a bearer token that stands directly
    against a BEGIN line, with no white space between (their characters include
    ``-`` and letters, so they take the line's dashes and the word ``BEGIN``),
    breaks that line. When a block closed earlier in the text, a second call
    then reads that block's END line and the next END line as one run and
    replaces the text between them. A second call only ever hides more.
    No pattern repeats an unbounded repetition (the PEM label's are capped at
    3 words of 16 characters) and the key block lines are read in one pass in
    code, so the passes take linear time.
    """
    if not text:
        return text, 0
    text, blocks = _redact_pem(text)
    text, keys = _KEY.subn(REDACTED, text)
    text, tokens = _BEARER.subn(lambda match: match.group(1) + REDACTED, text)
    text, codes = _redact_codes(text)
    return text, blocks + keys + tokens + codes


def secret_spans(text: str) -> list[tuple[int, int]]:
    """Where *text* holds a secret by the rules of :func:`redact_secrets`: the
    ``(start, end)`` of each key block's body, key, bearer token and code, sorted.

    Each kind is read in *text* as given, not in an earlier pass's output, so a
    span is a place in *text*. It is for a reader that takes text out by its
    place: a secret can need words outside its own sentence to be seen (a code's
    keyword, the word before a token on the next line), and once those are taken
    out it reads as a number or a word. There is a span exactly when
    :func:`redact_secrets` counts a secret. The spans are not always what the
    passes replace one after the other: a replacement can bring a keyword within
    reach of a code, or take a keyword away, and only the passes see that.
    """
    spans = [(start, end) for start, end, _ in _pem_blocks(text)]
    spans += [match.span() for match in _KEY.finditer(text)]
    spans += [(match.end(1), match.end()) for match in _BEARER.finditer(text)]
    spans += _code_spans(text)
    return sorted(spans)


def redacted(value: Any) -> Any:
    """*value* with every string in it (and in its lists) redacted."""
    if isinstance(value, str):
        return redact_secrets(value)[0]
    if isinstance(value, list):
        return [redacted(item) for item in value]
    return value


def _redact_pem(text: str) -> tuple[str, int]:
    """Pass 1 of :func:`redact_secrets`: the PEM private key blocks."""
    blocks = _pem_blocks(text)
    pieces = [(start, end, _pem_body(closed)) for start, end, closed in blocks]
    return _splice(text, pieces), len(blocks)


def _pem_body(closed: bool) -> str:
    """What stands for a key block's body: :data:`REDACTED` on a line of its own."""
    return "\n" + REDACTED + ("\n" if closed else "")


def _pem_blocks(text: str) -> list[tuple[int, int, bool]]:
    """The ``(start, end, closed)`` of each key block's body in *text*, in order.

    The one place that reads where a block is (rule 1 of :func:`redact_secrets`):
    the BEGIN and END lines are taken in order, in one pass. *closed* is False
    for a block with no END line, which runs to the end of the text. A block
    already in the shape of :func:`_pem_body` is left out.
    """
    if "PRIVATE KEY" not in text:
        return []
    markers = sorted(
        [(m.start(), m.end(), True) for m in PEM_BEGIN.finditer(text)]
        + [(m.start(), m.end(), False) for m in PEM_END.finditer(text)]
    )
    found: list[tuple[int, int, bool]] = []
    body = -1  # where the open block's body starts; -1 when none is open
    stop = -1  # where the last END line of the open block starts; -1 when none yet
    reach = 0  # a line that starts before this overlaps one already read
    for start, end, is_begin in markers:
        if start < reach:
            continue
        if is_begin and (body < 0 or stop >= 0):
            if body >= 0:  # a BEGIN line after the run's last END line closes it
                found.append((body, stop, True))
            body, stop, reach = end, -1, end
        elif not is_begin and body >= 0:
            stop, reach = start, end
    if body >= 0:
        found.append((body, stop, True) if stop >= 0 else (body, len(text), False))
    return [
        (start, end, closed)
        for start, end, closed in found
        if text[start:end] != _pem_body(closed)
    ]


def _splice(text: str, pieces: list[tuple[int, int, str]]) -> str:
    """*text* with each ``(start, end, replacement)`` of *pieces*, in order, put in."""
    parts: list[str] = []
    kept_from = 0
    for start, end, replacement in pieces:
        parts += [text[kept_from:start], replacement]
        kept_from = end
    parts.append(text[kept_from:])
    return "".join(parts)


def _redact_codes(text: str) -> tuple[str, int]:
    """Pass 4 of :func:`redact_secrets`: the codes next to a keyword."""
    spans = _code_spans(text)
    return _splice(text, [(start, end, REDACTED) for start, end in spans]), len(spans)


def _code_spans(text: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` of each code of *text* next to a keyword, in order."""
    keywords = [match.span() for match in _CODE_KEYWORD.finditer(text)]
    if not keywords:
        return []
    starts = [start for start, _ in keywords]
    ends = sorted(end for _, end in keywords)
    return [
        match.span()
        for match in _CODE.finditer(text)
        if _near_a_keyword(text, *match.span(), starts, ends)
    ]


def _near_a_keyword(
    text: str, start: int, end: int, starts: list[int], ends: list[int]
) -> bool:
    """Whether the code at *start*:*end* is close enough to a keyword.

    *starts* and *ends* are the keywords' start and end offsets, sorted.
    Only the nearest keyword on each side counts: a farther one could only
    be farther away.
    """
    before = bisect.bisect_right(ends, start) - 1  # the last keyword ending by start
    if before >= 0:
        gap = text[ends[before] : start]
        if len(gap) <= _CODE_REACH and "\n" not in gap:
            return True
    after = bisect.bisect_left(starts, end)  # the first keyword starting from end
    if after < len(starts):
        gap = text[end : starts[after]]
        if len(gap) <= _CODE_REACH and "\n" not in gap:
            return True
    return False
