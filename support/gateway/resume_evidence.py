"""Classify a recovered task for an explicit Resume, without launching it.

The decision table is conservative: only a task the store proves never
started, or a started task whose trusted, complete capture covers the whole
execution and whose every recorded process group is now verifiably gone, may
Resume without a confirmation. Everything else may Resume only after the user
accepts the recorded risk. An unreadable or unsupported check is unknown, not
evidence that a process stopped, and a missing card blocks rather than
falling back to the original objective.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, Literal

from localmelo.support.gateway.process_identity import gone
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.resume_tokens import ResumeBinding
from localmelo.support.gateway.task_store import TaskRecord, TaskStore, TaskStoreError

Kind = Literal["first_start", "free", "confirm"]
Process = tuple[int, int, str, str]

RESUME_WARNING: Final = (
    "A process from before the restart may still be running. Resuming may "
    "repeat effects. Confirmation does not prove the process stopped or "
    "effects happened only once."
)
MISSING_CARD_MESSAGE: Final = (
    "Resume unavailable: recovery information is missing. " "Start again as a new task."
)
INVALID_WORKSPACE_MESSAGE: Final = (
    "Resume unavailable: the task workspace is invalid. " "Start again as a new task."
)
STALE_IDENTITY_MESSAGE: Final = (
    "Resume unavailable: the recorded execution identity is stale."
)
UNRESOLVED_MESSAGE: Final = (
    "Resume unavailable: an earlier recovery operation has no settled receipt."
)


class ResumeRefusedError(TaskStoreError):
    """A decision-table row that blocks Resume; Start again stays available."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class RecoveryUnknownError(ResumeRefusedError):
    """An unresolved or foreign operation receipt blocks only Resume."""

    def __init__(self, message: str = UNRESOLVED_MESSAGE) -> None:
        super().__init__(message, reason="unresolved_operation")


class ConfirmationRequiredError(Exception):
    """The evidence cannot show the old execution is gone; ask the user."""

    def __init__(
        self,
        binding: ResumeBinding,
        *,
        token: str,
        warning: str = RESUME_WARNING,
    ) -> None:
        super().__init__(warning)
        self.binding = binding
        self.token = token
        self.warning = warning


class StaleConfirmationError(Exception):
    """The presented token no longer authorizes this transition."""


@dataclass(frozen=True)
class ResumeEvidence:
    task_id: str
    state: str
    first_start: str
    execution_id: str | None
    generation: int | None
    pending_operation_id: str | None
    card: str
    capture_complete: bool
    groups: tuple[Process, ...]
    groups_gone: bool
    evidence_key: str


@dataclass(frozen=True)
class ResumeDecision:
    kind: Kind
    evidence: ResumeEvidence


def _key(evidence: ResumeEvidence) -> str:
    raw = json.dumps(
        {
            "task": evidence.task_id,
            "state": evidence.state,
            "first_start": evidence.first_start,
            "execution": evidence.execution_id,
            "capture": evidence.capture_complete,
            "groups": [list(group) for group in evidence.groups],
            "gone": evidence.groups_gone,
            "card": evidence.card,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def collect_evidence(
    tasks: TaskStore,
    recovery: RecoveryStore,
    record: TaskRecord,
    *,
    verifier: Callable[[Process], bool] = gone,
) -> ResumeEvidence:
    """Read the evidence once; every read is conservative on failure."""
    task_id = record.task_id
    provenance = await recovery.provenance(task_id)
    generation, pending = await tasks.park_identity(task_id)
    active = await recovery.active_capture(task_id)
    if active is not None and provenance.execution_id != active:
        raise ResumeRefusedError(STALE_IDENTITY_MESSAGE, reason="stale_identity")
    if provenance.first_start == "started" and provenance.execution_id is None:
        raise ResumeRefusedError(STALE_IDENTITY_MESSAGE, reason="stale_identity")
    execution_id = provenance.execution_id
    complete = bool(
        execution_id
        and await recovery.capture_complete(task_id, execution_id=execution_id)
    )
    groups = (
        tuple(await recovery.processes(task_id, execution_id=execution_id))
        if execution_id
        else ()
    )
    try:
        groups_gone = all(verifier(group) for group in groups)
    except Exception:
        # A verifier that cannot answer is unknown, never "gone".
        groups_gone = False
    preliminary = ResumeEvidence(
        task_id=task_id,
        state=record.state,
        first_start=provenance.first_start,
        execution_id=execution_id,
        generation=generation,
        pending_operation_id=pending,
        card=record.card,
        capture_complete=complete,
        groups=groups,
        groups_gone=groups_gone,
        evidence_key="",
    )
    return replace(preliminary, evidence_key=_key(preliminary))


def decide(evidence: ResumeEvidence, *, workspace_valid: bool) -> ResumeDecision:
    """Apply the decision table to already-collected evidence."""
    if not workspace_valid:
        raise ResumeRefusedError(INVALID_WORKSPACE_MESSAGE, reason="invalid_workspace")
    if evidence.first_start == "never_started":
        return ResumeDecision("first_start", evidence)
    if not evidence.card.strip():
        raise ResumeRefusedError(MISSING_CARD_MESSAGE, reason="missing_card")
    if evidence.capture_complete and evidence.groups_gone:
        return ResumeDecision("free", evidence)
    return ResumeDecision("confirm", evidence)
