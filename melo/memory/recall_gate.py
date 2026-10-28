"""Task identity and relative gating for long-term recall."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory.notes import _task_line
from localmelo.melo.memory.retrieval.base import Doc


def digest_task_key(query: str, workspace: str = "") -> str:
    """Hash the normalized, redacted task line and the workspace's realpath.

    A digest and the recall that looks for it must hash the same workspace:
    two tasks with the same text in different folders are different tasks.
    An empty *workspace* keeps a text-only key for callers without one.
    """
    line = _task_line(redact_secrets(query)[0])
    normalized = " ".join(line.casefold().split())
    if not normalized:
        return ""
    place = os.path.realpath(workspace) if workspace else ""
    identity = f"{place}\0{normalized}" if place else normalized
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def digest_has_evidence(text: str) -> bool:
    """Require an actual file or verification line after candidate sanitization."""
    return any(
        line.startswith(("files: ", "verified: ")) and line.split(":", 1)[1].strip()
        for line in text.splitlines()
    )


def is_task_digest(doc: Doc) -> bool:
    """Only episodic attempt digests are task-scoped; facts remain general."""
    return (
        doc.meta.get("source") == "digest"
        and doc.meta.get("store_kind", doc.meta.get("kind")) == "episodic"
    )


def same_task(doc: Doc, key: str) -> bool:
    provenance = doc.meta.get("provenance")
    return bool(
        is_task_digest(doc)
        and key
        and isinstance(provenance, dict)
        and provenance.get("task_key") == key
    )


def gate_long_docs(
    docs: Sequence[Doc], key: str, *, ratio: float, cap: int, cold: int
) -> list[Doc]:
    """Rank same-task first and bound cross-task hits on comparable score scales."""
    # Preserve the existing long-hit score order for unscoped memories.
    # The cross-task ratio below is applied only within one backend.
    ranked = sorted(docs, key=lambda d: float(d.meta.get("score") or 0.0), reverse=True)
    same = [doc for doc in ranked if same_task(doc, key)]
    cross = [doc for doc in ranked if is_task_digest(doc) and not same_task(doc, key)]
    general = [doc for doc in ranked if not is_task_digest(doc)]
    if not same:
        eligible = cross[: min(cap, cold)]
    else:
        best: dict[str, float] = {}
        for doc in same:
            backend = str(doc.meta.get("retrieval") or "")
            score = float(doc.meta.get("score") or 0.0)
            if math.isfinite(score):
                best[backend] = max(best.get(backend, -math.inf), score)
        eligible = []
        for doc in cross:
            backend = str(doc.meta.get("retrieval") or "")
            score = float(doc.meta.get("score") or 0.0)
            if backend not in best or not math.isfinite(score):
                continue
            threshold = ratio * best[backend]
            # A float product can miss an exact boundary (1.1 * 0.8 is
            # 0.8800000000000001); compare with a relative tolerance.
            if score >= threshold or math.isclose(score, threshold, rel_tol=1e-9):
                eligible.append(doc)
        eligible = eligible[:cap]
    remaining = [doc for doc in ranked if doc in general or doc in eligible]
    return same + remaining
