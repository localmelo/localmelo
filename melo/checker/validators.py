"""Boundary validators for the Checker."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

from localmelo.melo.checker.payloads import (
    ExecutorRequest,
    ExecutorResultPayload,
    GatewayIngressPayload,
    MemoryCandidatePayload,
    MemoryWritePayload,
    SessionTransition,
    ToolResolutionResult,
    ValidationResult,
)
from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.sanitize import sanitize_assistant_text
from localmelo.melo.schema import ImagePart

# ── Constants ──

MAX_QUERY_LEN = 100_000
MAX_OUTPUT_LEN = 50_000
# Ends an output cut to fit MAX_OUTPUT_LEN; the one text allowed past it.
TRUNCATION_SUFFIX = "\n... [truncated]"
MAX_MEMORY_TEXT_LEN = 50_000
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

VALID_SESSION_STATES = {"idle", "running", "closed"}
LEGAL_SESSION_TRANSITIONS: set[tuple[str, str]] = {
    ("idle", "running"),
    ("running", "idle"),
    ("running", "closed"),
    ("idle", "closed"),
    ("running", "failed"),
}

VALID_MEMORY_ROLES = {"user", "assistant", "system", "tool"}

# Images a tool may hand back to the model.
ALLOWED_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # decoded size per image
MAX_IMAGES_PER_RESULT = 8
MAX_IMAGES_PER_MEMORY_WRITE = MAX_IMAGES_PER_RESULT

BLOCKED_COMMANDS = [
    r"\brm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{.*\}\s*;",
    r"\bshutdown\b",
    r"\breboot\b",
]
_BLOCKED_RE = [re.compile(p) for p in BLOCKED_COMMANDS]


# ── Gateway ingress ──


def validate_gateway_ingress(payload: GatewayIngressPayload) -> ValidationResult:
    if not isinstance(payload.query, str) or not payload.query.strip():
        return ValidationResult(
            allowed=False, reason="Query must be a non-empty string"
        )

    if len(payload.query) > MAX_QUERY_LEN:
        return ValidationResult(
            allowed=False,
            reason=f"Query too large: {len(payload.query)} chars (max {MAX_QUERY_LEN})",
        )

    if payload.session_id is not None and not SESSION_ID_PATTERN.match(
        payload.session_id
    ):
        return ValidationResult(
            allowed=False,
            reason=f"Invalid session_id format: {payload.session_id!r}",
        )

    # Sanitize: strip leading/trailing whitespace from query
    sanitized = GatewayIngressPayload(
        query=payload.query.strip(),
        session_id=payload.session_id,
    )
    return ValidationResult(allowed=True, sanitized_payload=sanitized)


# ── Session state transitions ──


def validate_session_transition(transition: SessionTransition) -> ValidationResult:
    if transition.from_status not in VALID_SESSION_STATES:
        return ValidationResult(
            allowed=False,
            reason=f"Unknown source state: {transition.from_status!r}",
        )

    if transition.to_status not in VALID_SESSION_STATES:
        return ValidationResult(
            allowed=False,
            reason=f"Unknown target state: {transition.to_status!r}",
        )

    pair = (transition.from_status, transition.to_status)
    if pair not in LEGAL_SESSION_TRANSITIONS:
        return ValidationResult(
            allowed=False,
            reason=f"Illegal transition: {transition.from_status} -> {transition.to_status}",
        )

    return ValidationResult(allowed=True)


# ── Tool resolution ──


def validate_tool_resolution(result: ToolResolutionResult) -> ValidationResult:
    if not result.query.strip():
        return ValidationResult(
            allowed=False, reason="Tool resolution query must not be empty"
        )

    for name in result.resolved_tool_names:
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                allowed=False,
                reason=f"Invalid resolved tool name: {name!r}",
            )

    seen: set[str] = set()
    for name in result.resolved_tool_names:
        if name in seen:
            return ValidationResult(
                allowed=False,
                reason=f"Duplicate resolved tool: {name!r}",
            )
        seen.add(name)

    return ValidationResult(allowed=True)


# ── Executor request ──


def validate_executor_request(request: ExecutorRequest) -> ValidationResult:
    if not request.tool_name or not request.tool_name.strip():
        return ValidationResult(
            allowed=False, reason="Executor request must specify a tool_name"
        )

    if request.tool_def_name is None:
        return ValidationResult(
            allowed=False,
            reason=f"Unknown tool: {request.tool_name} (no matching tool_def)",
        )

    if request.tool_name != request.tool_def_name:
        return ValidationResult(
            allowed=False,
            reason=f"Tool name mismatch: call={request.tool_name!r} vs def={request.tool_def_name!r}",
        )

    # Block dangerous shell commands
    if request.tool_name == "shell_exec":
        cmd = request.arguments.get("command", "")
        if not isinstance(cmd, str):
            return ValidationResult(
                allowed=False, reason="shell_exec command must be a string"
            )
        for pattern in _BLOCKED_RE:
            if pattern.search(cmd):
                return ValidationResult(
                    allowed=False,
                    reason=f"Blocked command: {cmd[:100]}",
                )

    return ValidationResult(allowed=True)


# ── Images ──


def image_problem(image: ImagePart) -> str:
    """Return why *image* cannot be passed to the model, or ``""``.

    The reason names a kind of problem, not the image, so that notes about
    dropped images can be counted per kind (see :func:`_sanitize_images`).
    """
    if image.mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_MIME_TYPES))
        return f"unsupported image type (allowed: {allowed})"
    if not image.data or image.size_bytes <= 0:
        return "image data is empty or not valid base64"
    if image.size_bytes > MAX_IMAGE_BYTES:
        return f"image too large (max {MAX_IMAGE_BYTES} bytes)"
    return ""


def _sanitize_images(images: list[ImagePart]) -> tuple[list[ImagePart], list[str]]:
    """Split *images* into the ones to keep and notes about dropped ones.

    The notes hold one line per kind of problem, with a count, e.g.
    ``[3 images dropped: more than 8 images in one result]``. However many
    images are dropped, that is at most four short lines.
    """
    kept: list[ImagePart] = []
    dropped: Counter[str] = Counter()
    for image in images:
        problem = image_problem(image)
        if not problem and len(kept) >= MAX_IMAGES_PER_RESULT:
            problem = f"more than {MAX_IMAGES_PER_RESULT} images in one result"
        if problem:
            dropped[problem] += 1
        else:
            kept.append(image)
    notes = [
        f"[{count} {'image' if count == 1 else 'images'} dropped: {problem}]"
        for problem, count in dropped.items()
    ]
    return kept, notes


# ── Executor result ──


def validate_executor_result(result: ExecutorResultPayload) -> ValidationResult:
    if result.duration_ms < 0:
        return ValidationResult(
            allowed=False,
            reason=f"Negative duration: {result.duration_ms}ms",
        )

    if not result.tool_name or not result.tool_name.strip():
        return ValidationResult(
            allowed=False, reason="Executor result must have a tool_name"
        )

    reasons: list[str] = []
    output, error = result.output, result.error

    # Drop images the model cannot (or should not) receive. The notes go
    # where the model will read them: the error text for failed calls,
    # after the output otherwise.
    kept, notes = _sanitize_images(result.images)
    note_text = ""
    if notes:
        if error:
            error = "\n".join((error, *notes))
        else:
            note_text = "\n".join(notes)
        reasons.append("Images dropped")

    # Truncate oversized output. Notes after it count toward MAX_OUTPUT_LEN
    # and are never cut off, so the output is cut to leave room for them
    # (and the newline before them): the text is never longer than
    # MAX_OUTPUT_LEN plus TRUNCATION_SUFFIX.
    room = MAX_OUTPUT_LEN - (len(note_text) + 1 if note_text else 0)
    if len(output) > room:
        output = output[: max(room, 0)] + TRUNCATION_SUFFIX
        reasons.append("Output truncated")
    if note_text:
        output = "\n".join(part for part in (output, note_text) if part)

    if reasons:
        return ValidationResult(
            allowed=True,
            reason="; ".join(reasons),
            sanitized_payload=replace(result, output=output, error=error, images=kept),
        )
    return ValidationResult(allowed=True)


# ── Memory write ──


def validate_memory_write(payload: MemoryWritePayload) -> ValidationResult:
    if not payload.text or not payload.text.strip():
        return ValidationResult(
            allowed=False, reason="Memory write text must not be empty"
        )

    if len(payload.text) > MAX_MEMORY_TEXT_LEN:
        return ValidationResult(
            allowed=False,
            reason=f"Memory write too large: {len(payload.text)} chars (max {MAX_MEMORY_TEXT_LEN})",
        )

    if payload.role and payload.role not in VALID_MEMORY_ROLES:
        return ValidationResult(
            allowed=False,
            reason=f"Invalid memory role: {payload.role!r}",
        )

    if len(payload.images) > MAX_IMAGES_PER_MEMORY_WRITE:
        return ValidationResult(
            allowed=False,
            reason=(
                f"Too many images in one memory write: {len(payload.images)} "
                f"(max {MAX_IMAGES_PER_MEMORY_WRITE})"
            ),
        )
    for image in payload.images:
        if not image.sha256:
            return ValidationResult(
                allowed=False, reason="Image in memory write has no sha256"
            )

    return ValidationResult(allowed=True)


# ── Memory candidate ──
#
# R4's boundary for the documents the reasoning-note loop puts into a
# prompt or a store: a note, a draft, a thought chunk, the attempt digest,
# a reflection, a stated fact, a recall block or a sleep-derived candidate.
# R4 uses the first five sources and the "candidate" target; R5 adds
# "prompt", R6-lite "long".

CANDIDATE_SOURCES: tuple[str, ...] = (
    "reasoning_note",
    "draft",
    "thought",
    "digest",
    "reflection",
    "user",
    "recall",
    "sleep",
)
CANDIDATE_TARGETS: tuple[str, ...] = ("candidate", "prompt", "long")
CANDIDATE_LIMITS: dict[str, int] = {
    "reasoning_note": 1_200,
    "draft": 8_000,
    "digest": 1_500,
    "reflection": 2_000,
    "thought": 1_500,
    "user": 300,
    "recall": 10_000,
    "sleep": 1_500,
}
# Ends a candidate cut to its source's limit; its length is inside the cap.
CANDIDATE_TRUNCATION_SUFFIX = "\n... [truncated]"


def _cut_candidate(text: str, limit: int) -> str:
    """*text* at most *limit* characters, ending with the truncation suffix.

    A cut never splits a surrogate pair: when the cut falls between the
    halves of one, it moves back by one character.
    """
    if len(text) <= limit:
        return text
    keep = max(limit - len(CANDIDATE_TRUNCATION_SUFFIX), 0)
    if (
        0 < keep < len(text)
        and "\ud800" <= text[keep - 1] <= "\udbff"
        and "\udc00" <= text[keep] <= "\udfff"
    ):
        keep -= 1
    return text[:keep] + CANDIDATE_TRUNCATION_SUFFIX


def validate_memory_candidate(payload: MemoryCandidatePayload) -> ValidationResult:
    """Check one memory candidate, in the order read, redact, sanitize, cap.

    Refusals: empty or whitespace-only text, an unknown ``source`` or
    ``target``, and a ``profile`` entry for the ``long`` target whose
    source is not ``user`` (R6-lite's rule, declared now). Otherwise
    :func:`~localmelo.melo.checker.redaction.redact_secrets` runs first,
    then :func:`~localmelo.melo.checker.sanitize.sanitize_assistant_text`,
    then the text is cut to its source's limit with
    :data:`CANDIDATE_TRUNCATION_SUFFIX`. The sanitized payload's metadata
    carries ``redactions``, the count the redaction pass returned.
    """
    if not payload.text or not payload.text.strip():
        return ValidationResult(
            allowed=False, reason="Memory candidate text must not be empty"
        )
    if payload.source not in CANDIDATE_SOURCES:
        return ValidationResult(
            allowed=False,
            reason=(
                f"Unknown memory candidate source: {payload.source!r}; "
                f"known sources: {CANDIDATE_SOURCES}"
            ),
        )
    if payload.target not in CANDIDATE_TARGETS:
        return ValidationResult(
            allowed=False,
            reason=(
                f"Unknown memory candidate target: {payload.target!r}; "
                f"known targets: {CANDIDATE_TARGETS}"
            ),
        )
    if (
        payload.target == "long"
        and payload.kind == "profile"
        and payload.source != "user"
    ):
        return ValidationResult(
            allowed=False,
            reason="A profile entry in long-term memory must come from the user",
        )
    text, redactions = redact_secrets(payload.text)
    text = sanitize_assistant_text(text)
    if not text.strip():
        return ValidationResult(
            allowed=False, reason="Memory candidate is empty after sanitizing"
        )
    text = _cut_candidate(text, CANDIDATE_LIMITS[payload.source])
    return ValidationResult(
        allowed=True,
        sanitized_payload=replace(
            payload,
            text=text,
            metadata={**payload.metadata, "redactions": redactions},
        ),
    )
