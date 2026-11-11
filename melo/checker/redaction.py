"""The redaction of secrets, under the name the checker's importers use.

The code is in :mod:`localmelo.melo.contracts.redaction`, below the checker, where
the password door reads it. This module gives the names that are imported from here.
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
