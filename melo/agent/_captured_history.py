"""Bounded model previews and separately redacted tool-result history."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.validators import MAX_MEMORY_TEXT_LEN
from localmelo.melo.schema import ToolResult, describe_images

PREVIEW_LIMIT = MAX_MEMORY_TEXT_LEN
CAPTURE_LIMIT = 200_000


def _bounded(text: str, limit: int) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    marker = f"\n... [truncated: {len(text)} characters]"
    if limit <= len(marker):
        return "", len(text)
    kept = limit - len(marker)
    while True:
        marker = f"\n... [truncated: {len(text) - kept} characters]"
        revised = min(kept, limit - len(marker))
        if revised == kept:
            return text[:kept] + marker, len(text) - kept
        kept = revised


def preview_result(
    result: ToolResult, notes: list[str] | None = None
) -> tuple[ToolResult, int]:
    """Bound the checked result and reserve room for image and error labels."""
    note = "\n".join(notes or [])
    room = max(0, PREVIEW_LIMIT - len(describe_images(result.images)) - len(note) - 33)
    error, error_cut = _bounded(result.error, room)
    output, output_cut = _bounded(result.output, max(0, room - len(error)))
    cut_before = (
        max(0, len(result.error) - len(error))
        if error_cut
        else max(0, len(result.output) - len(output)) if output_cut else 0
    )
    if note:
        if result.error:
            error = "\n".join(part for part in (error, note) if part)
        else:
            output = "\n".join(part for part in (output, note) if part)
    return (
        replace(result, output=output, error=error, images=list(result.images)),
        cut_before,
    )


def split_image_notes(text: str) -> tuple[str, str]:
    """Keep the checker's trailing image-drop notices separate during fitting."""
    lines = text.splitlines(keepends=True)
    count = 0
    while lines and count < 4:
        line = lines[-1].strip()
        if not (
            line.startswith("[")
            and " image" in line
            and " dropped: " in line
            and line.endswith("]")
            and len(line) <= 256
        ):
            break
        count += 1
        if count == 1:
            notes = [lines.pop().strip()]
        else:
            notes.append(lines.pop().strip())
    if not count:
        return text, ""
    return "".join(lines).rstrip("\n"), "\n".join(reversed(notes))


def capture_result(result: ToolResult) -> tuple[ToolResult, dict[str, Any]]:
    """Redact before limiting storage; tool-side acquisition is not observable."""
    output, output_redactions = redact_secrets(result.output)
    error, error_redactions = redact_secrets(result.error)
    kept_error = error[:CAPTURE_LIMIT]
    kept_output = output[: CAPTURE_LIMIT - len(kept_error)]
    return replace(
        result, output=kept_output, error=kept_error, images=list(result.images)
    ), {
        "output_omitted_chars": len(output) - len(kept_output),
        "error_omitted_chars": len(error) - len(kept_error),
        "acquisition_omitted_chars": None,
        "redactions": output_redactions + error_redactions,
    }
