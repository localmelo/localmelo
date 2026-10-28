"""R6-lite's admission layer: the typed store's policy, staging and signals.

The package is pure policy and RAM state: it imports the long store, the
reasoning-note helpers and :mod:`localmelo.melo.schema`, never the agent or
``support/``. :class:`~localmelo.melo.memory.coordinator.Hippo` owns the
write path and calls into it.
"""

from __future__ import annotations

from dataclasses import dataclass

from localmelo.melo.memory.hippo.policy import (
    ACTIONS,
    CHANGE_MARKERS,
    EXACT_DUPLICATE_REASON,
    FILE_ONLY_WORDS,
    PROFILE_PREDICATES,
    SEMANTIC_ATTRIBUTES,
    HippoPolicy,
    PolicyDecision,
    classify_kind,
    decide,
    normalise,
    overlap,
    subjects,
    value_tokens,
)
from localmelo.melo.memory.hippo.signals import (
    REMEMBER_MARKERS,
    USER_STATEMENT_MARKERS,
    UserSignals,
    classify_user_text,
    request_candidates,
    user_candidates,
)
from localmelo.melo.memory.hippo.stage import STAGE_CAP, StagedCandidates
from localmelo.melo.memory.long import (
    KINDS,
    SIGNALS,
    STATUSES,
)
from localmelo.melo.memory.long import SOURCES as STORE_SOURCES
from localmelo.melo.schema import CandidateItem, MemoryCandidate

# The candidate sources a MemoryCandidate may name (a step summary is a
# legacy store row, not a candidate).
SOURCES = ("reasoning_note", "digest", "reflection", "user", "sleep")


@dataclass
class MemoryDecision:
    """One item's outcome: a stored decision, or a boundary refusal.

    A refusal is not one of the five actions: ``refused`` is true, no row
    was written, and the agent counts it in ``candidates_refused``.
    """

    candidate_id: str
    action: str
    kind: str = ""
    admitted_by: str = ""
    entry_id: int | None = None
    target_id: int | None = None
    reason: str = ""
    source_ids: tuple[str, ...] = ()
    text_sha256: str = ""
    text: str | None = None
    task_id: str = ""
    refused: bool = False
    redactions: int = 0


__all__ = [
    "ACTIONS",
    "CHANGE_MARKERS",
    "EXACT_DUPLICATE_REASON",
    "FILE_ONLY_WORDS",
    "KINDS",
    "PROFILE_PREDICATES",
    "REMEMBER_MARKERS",
    "SEMANTIC_ATTRIBUTES",
    "SIGNALS",
    "SOURCES",
    "STAGE_CAP",
    "STATUSES",
    "STORE_SOURCES",
    "CandidateItem",
    "HippoPolicy",
    "MemoryCandidate",
    "MemoryDecision",
    "PolicyDecision",
    "StagedCandidates",
    "USER_STATEMENT_MARKERS",
    "UserSignals",
    "classify_kind",
    "classify_user_text",
    "decide",
    "normalise",
    "overlap",
    "request_candidates",
    "subjects",
    "user_candidates",
    "value_tokens",
]
