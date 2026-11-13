"""The redaction of secrets, under its older path.

The code is in :mod:`localmelo.melo.contracts.redaction`, below the checker, where
the password door reads it, and the package's own modules import it from there.
This module keeps the names that ``docs/reference/python-api.md`` gives under
this path.
"""

from __future__ import annotations

from localmelo.melo.contracts.redaction import (
    BEARER_WORD,
    PEM_BEGIN,
    PEM_END,
    REDACTED,
    redact_secrets,
    redacted,
    secret_spans,
)

__all__ = [
    "BEARER_WORD",
    "PEM_BEGIN",
    "PEM_END",
    "REDACTED",
    "redact_secrets",
    "redacted",
    "secret_spans",
]
