"""Minimal redaction of secrets in a model's text.

:func:`redact_secrets` replaces two kinds of secret with
:data:`REDACTED`: well-known key shapes, and one-time codes next to a code
keyword. It is a pure function, for text that leaves the run's own memory,
such as the ``thinking`` and ``content`` of an ``assistant.message`` event.
"""

from __future__ import annotations

import bisect
import re
from typing import Final

REDACTED: Final = "[redacted]"

# Pass 1, key shapes, each match whole: an "sk-" key, a GitHub token, an
# AWS access key id. Each takes the whole run of its characters and starts
# after none of them, so a replaced key has no ASCII letter or digit next
# to it.
_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"
    r"|(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"
    r"|(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"
)

# Pass 2, bearer tokens: only the token after the word is replaced.
_BEARER = re.compile(r"(\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)

# Pass 3, one-time codes: 4 to 8 ASCII digits, with no ASCII letter or digit
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

    Three passes run in this order, each on the previous pass's output:

    1. Key shapes, the whole match: ``sk-`` followed by at least 20 of
       ``[A-Za-z0-9_-]``, not preceded by one of them (so ``task-…`` stays);
       ``ghp_``, ``gho_``, ``ghu_``, ``ghs_`` or ``ghr_`` followed by at least
       36 ASCII letters and digits, or ``github_pat_`` followed by at least
       22 of ``[A-Za-z0-9_]``, neither preceded by ``[A-Za-z0-9_]``; and
       ``AKIA`` or ``ASIA`` followed by exactly 16 of ``[A-Z0-9]``, with no
       ASCII letter or digit on either side.
    2. Bearer tokens: the word ``bearer`` in any case, whitespace, then at
       least 16 of ``[A-Za-z0-9._~+/=-]``. Only the token is replaced:
       ``Bearer [redacted]``.
    3. One-time codes: a run of 4 to 8 ASCII digits, with no ASCII letter or
       digit right before or after it, that starts at most 20 characters
       after a code keyword ends, or ends at most 20 characters before one
       starts, with no newline between them. The keywords, in any case, are
       ``verification code``, ``passcode``, ``OTP`` as a whole word and
       ``验证码``. Only the code is replaced.

    A replaced span never has an ASCII letter or digit next to it, and
    :data:`REDACTED` matches no pattern, so a second call replaces nothing.
    No pattern repeats a repetition, so the passes take linear time.
    """
    if not text:
        return text, 0
    text, keys = _KEY.subn(REDACTED, text)
    text, tokens = _BEARER.subn(lambda match: match.group(1) + REDACTED, text)
    text, codes = _redact_codes(text)
    return text, keys + tokens + codes


def _redact_codes(text: str) -> tuple[str, int]:
    """Pass 3 of :func:`redact_secrets`: the codes next to a keyword."""
    keywords = [match.span() for match in _CODE_KEYWORD.finditer(text)]
    if not keywords:
        return text, 0
    starts = [start for start, _ in keywords]
    ends = sorted(end for _, end in keywords)
    parts: list[str] = []
    kept_from = 0
    count = 0
    for match in _CODE.finditer(text):
        start, end = match.span()
        if not _near_a_keyword(text, start, end, starts, ends):
            continue
        parts += [text[kept_from:start], REDACTED]
        kept_from = end
        count += 1
    if not count:
        return text, 0
    parts.append(text[kept_from:])
    return "".join(parts), count


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
