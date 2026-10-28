"""Typed payload dataclasses for Checker boundary validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from localmelo.melo.schema import ImagePart


@dataclass
class ValidationResult:
    """Actionable result from a boundary validator."""

    allowed: bool = True
    reason: str = ""
    sanitized_payload: Any = None


# ── Gateway boundary ──


@dataclass
class GatewayIngressPayload:
    """Payload arriving at the gateway ingress."""

    query: str = ""
    session_id: str | None = None


# ── Session boundary ──


@dataclass
class SessionTransition:
    """Proposed session state change."""

    from_status: str = ""
    to_status: str = ""
    session_id: str = ""


# ── Tool resolution boundary ──


@dataclass
class ToolResolutionResult:
    """Result of tool resolution (BM25 / hint-based)."""

    query: str = ""
    hints: list[str] = field(default_factory=list)
    resolved_tool_names: list[str] = field(default_factory=list)


# ── Executor boundaries ──


@dataclass
class ExecutorRequest:
    """Request entering the executor."""

    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_def_name: str | None = None  # None means unresolved


@dataclass
class ExecutorResultPayload:
    """Result coming back from the executor."""

    tool_name: str = ""
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    images: list[ImagePart] = field(default_factory=list)


# ── Memory boundary ──


@dataclass
class MemoryWritePayload:
    """Payload for a memory write operation."""

    text: str = ""
    role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[ImagePart] = field(default_factory=list)


@dataclass
class MemoryCandidatePayload:
    """One document offered to memory, before it reaches a prompt or a store.

    The reasoning-notes candidate boundary (R4): ``source`` says what kind
    of document it is, ``target`` where it is going, and ``kind`` the long
    store's entry kind when the target is ``long`` (R6-lite).
    """

    text: str = ""
    # reasoning_note | draft | thought | digest | reflection | user | recall | sleep
    source: str = ""
    # candidate | prompt | long
    target: str = ""
    # target "long" only: the kind Hippo assigned (R6-lite)
    kind: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
