"""A minimal mask for credentials in what procedural memory keeps.

``procedures.db`` keeps each routine's query text and each tool call's
label. Both pass :func:`~localmelo.melo.checker.redaction.redact_secrets`
(key shapes, bearer tokens, one-time codes) and then
:func:`mask_credentials`, which also replaces with :data:`MASK`:

- a URL's password: ``postgres://app:pw@host`` becomes
  ``postgres://app:<redacted>@host``; the user name stays;
- the value of ``NAME=value`` or ``NAME: value`` when NAME contains
  ``key``, ``token``, ``secret``, ``password``, ``passwd`` or ``pwd``, in
  any case: ``API_KEY=<redacted>``. A quoted value goes whole; an
  unquoted one up to whitespace or one of ``"',;&``.

Everything else stays: host names, paths, other ``NAME=value`` pairs.
Only procedural memory applies this mask; history and events keep the
redaction of :func:`redact_secrets` alone. Each pattern starts only where
a run of its characters starts, so the mask takes linear time.
"""

from __future__ import annotations

import re
from typing import Final

MASK: Final = "<redacted>"

# scheme://user:password@ — the password is replaced.
_URL_PASSWORD: Final = re.compile(
    r"(?<![A-Za-z0-9+.-])([A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]+:)[^\s/@]+@"
)

# NAME=value or NAME: value, NAME holding a secret's word: the value is
# replaced. The lookahead finds the word within NAME's own run.
_NAMED_SECRET: Final = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?=[A-Za-z0-9_.-]*?(?:key|token|secret|passw(?:or)?d|pwd))"
    r"([A-Za-z0-9_.-]+[ \t]*[=:][ \t]*)"
    r"(\"[^\"\n]*\"|'[^'\n]*'|[^\s\"',;&]+)",
    re.IGNORECASE,
)


def mask_credentials(text: str) -> str:
    """*text* with URL passwords and named secrets' values replaced by :data:`MASK`."""
    if not text:
        return text
    text = _URL_PASSWORD.sub(lambda match: f"{match.group(1)}{MASK}@", text)
    return _NAMED_SECRET.sub(lambda match: f"{match.group(1)}{MASK}", text)
