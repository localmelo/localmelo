"""Minimal redaction of secrets in a model's text.

:func:`redact_secrets` replaces three kinds of secret with
:data:`REDACTED`: PEM private key blocks, well-known key shapes, and
one-time codes next to a code keyword. It is a pure function, for text that
leaves the run's own memory, such as the ``thinking`` and ``content`` of an ``assistant.message`` event.
History's ``llm_calls`` table keeps the ``thinking`` and ``content`` of
every model response redacted by it too.
"""

from __future__ import annotations

import bisect
import re
from typing import Final

REDACTED: Final = "[redacted]"

# Pass 1, PEM private key blocks: the BEGIN and END lines stay, the body
# between them goes. A block with no END line runs to the end of the text.
# Only labels that end in PRIVATE KEY match (RSA, EC, DSA, OPENSSH, ENCRYPTED,
# plain PKCS#8), and the PGP armor's PRIVATE KEY BLOCK; a label has at most 3
# words of at most 16 characters before PRIVATE KEY, so a scan for the END line
# stays linear.
_PEM = re.compile(
    r"(-----BEGIN (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY(?: BLOCK)?-----)"
    r"(.*?)"
    r"(-----END (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY(?: BLOCK)?-----|\Z)",
    re.DOTALL,
)

# Pass 2, key shapes, each match whole: an "sk-" key, a GitHub token, an
# AWS access key id. Each takes the whole run of its characters and starts
# after none of them, so a replaced key has no ASCII letter or digit next
# to it.
_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"
    r"|(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"
    r"|(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"
)

# Pass 3, bearer tokens: only the token after the word is replaced.
_BEARER = re.compile(r"(\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)

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

    1. PEM private key blocks: from ``-----BEGIN <label>PRIVATE KEY-----``
       (the label is up to 3 upper-case words, so RSA, EC, DSA, OPENSSH,
       ENCRYPTED and plain PKCS#8 all match; ``PRIVATE KEY BLOCK``, the PGP
       armor, too) to the next
       ``-----END <label>PRIVATE KEY-----``, or to the end of the text when
       there is none. The BEGIN and END lines stay and the body becomes one
       line, :data:`REDACTED`: ``BEGIN``, newline, ``[redacted]``, newline,
       ``END``. ``PUBLIC KEY`` and ``CERTIFICATE`` blocks, and
       ``ssh-ed25519`` public key lines, do not match. A block already in
       that shape is left alone and not counted. The count goes up by one
       per block.
    2. Key shapes, the whole match: ``sk-`` followed by at least 20 of
       ``[A-Za-z0-9_-]``, not preceded by one of them (so ``task-…`` stays);
       ``ghp_``, ``gho_``, ``ghu_``, ``ghs_`` or ``ghr_`` followed by at least
       36 ASCII letters and digits, or ``github_pat_`` followed by at least
       22 of ``[A-Za-z0-9_]``, neither preceded by ``[A-Za-z0-9_]``; and
       ``AKIA`` or ``ASIA`` followed by exactly 16 of ``[A-Z0-9]``, with no
       ASCII letter or digit on either side.
    3. Bearer tokens: the word ``bearer`` in any case, whitespace, then at
       least 16 of ``[A-Za-z0-9._~+/=-]``. Only the token is replaced:
       ``Bearer [redacted]``.
    4. One-time codes: a run of 4 to 8 ASCII digits, with no ASCII letter or
       digit right before or after it, that starts at most 20 characters
       after a code keyword ends, or ends at most 20 characters before one
       starts, with no newline between them. The keywords, in any case, are
       ``verification code``, ``passcode``, ``OTP`` as a whole word and
       ``验证码``. Only the code is replaced.

    A replaced span never has an ASCII letter or digit next to it, and
    :data:`REDACTED` matches no pattern, so a second call replaces nothing.
    No pattern repeats an unbounded repetition (the PEM label's are capped at
    3 words of 16 characters), so the passes take linear time.
    """
    if not text:
        return text, 0
    text, blocks = _redact_pem(text)
    text, keys = _KEY.subn(REDACTED, text)
    text, tokens = _BEARER.subn(lambda match: match.group(1) + REDACTED, text)
    text, codes = _redact_codes(text)
    return text, blocks + keys + tokens + codes


def _redact_pem(text: str) -> tuple[str, int]:
    """Pass 1 of :func:`redact_secrets`: the PEM private key blocks."""
    if "PRIVATE KEY" not in text:
        return text, 0
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        begin, body, end = match.groups()
        redacted = "\n" + REDACTED + ("\n" if end else "")
        if body == redacted:
            return match.group(0)
        count += 1
        return begin + redacted + end

    return _PEM.sub(replace, text), count


def _redact_codes(text: str) -> tuple[str, int]:
    """Pass 4 of :func:`redact_secrets`: the codes next to a keyword."""
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
