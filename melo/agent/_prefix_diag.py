"""Prefix diagnostics: how much of a request repeats an earlier one.

A server's prompt cache can only reuse the leading part of a request that is
identical to an earlier one. :func:`fingerprint` records what a request sends
and :func:`best_match` says how far it matches the best of the run's earlier
requests, so a cache miss can be traced to the message that changed first.
Nothing here changes a request body.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from localmelo.melo.schema import Message, ToolDef

# Fingerprints kept per run; the oldest are dropped first.
MAX_FINGERPRINTS = 64

# The offered tools, then each message as (digest of its serialized form, role,
# content): the digest decides equality, the content (the message's own string,
# not a copy) gives the characters.
Fingerprint = tuple[str, list[tuple[str, str, str]]]


def fingerprint(
    messages: Sequence[Message], tools: Sequence[ToolDef] | None
) -> Fingerprint:
    """What *messages* and the offered *tools* send, in comparable form."""
    schema = json.dumps(
        [(t.name, t.description, t.parameters) for t in tools or []],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return schema, [
        (
            hashlib.blake2b(
                json.dumps(
                    [
                        message.role,
                        message.content,
                        message.thinking,
                        [(c.id, c.tool_name, c.arguments) for c in message.tool_calls],
                        message.tool_call_id,
                        [image.sha256 for image in message.images],
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode(),
                digest_size=16,
            ).hexdigest(),
            message.role,
            message.content,
        )
        for message in messages
    ]


def _shared(old: list[tuple[str, str, str]], new: list[tuple[str, str, str]]) -> int:
    """The number of leading messages that *old* and *new* have in common."""
    same = 0
    while same < min(len(old), len(new)) and old[same][0] == new[same][0]:
        same += 1
    return same


def _shared_chars(
    old: list[tuple[str, str, str]], new: list[tuple[str, str, str]], same: int
) -> int:
    """Characters of content in the *same* leading messages, plus the common
    start of the next message's content."""
    chars = sum(len(message[2]) for message in new[:same])
    if same < min(len(old), len(new)):
        a, b = old[same][2], new[same][2]
        shared = 0
        while shared < min(len(a), len(b)) and a[shared] == b[shared]:
            shared += 1
        chars += shared
    return chars


def best_match(
    earlier: Sequence[tuple[int, str, Fingerprint]], current: Fingerprint
) -> dict[str, Any] | None:
    """:func:`compare` of *current* with the best of the *earlier* requests.

    *earlier* holds ``(number, kind, fingerprint)`` of the run's earlier
    requests. The best is the one with the most identical leading messages,
    then the most characters, then the most recent. The result names it in
    ``against`` (its kind and 1-based number). ``None`` without any.
    """
    best: tuple[tuple[int, int], int, str, Fingerprint] | None = None
    for number, kind, fp in earlier:
        same = _shared(fp[1], current[1])
        score = (same, _shared_chars(fp[1], current[1], same))
        if best is None or score >= best[0]:
            best = (score, number, kind, fp)
    if best is None:
        return None
    _, number, kind, fp = best
    return {**compare(fp, current), "against": {"kind": kind, "request": number}}


def compare(previous: Fingerprint, current: Fingerprint) -> dict[str, Any]:
    """How far *current* repeats *previous*, as the ``prefix`` event object.

    ``messages`` and ``chars`` count the leading messages that are identical
    and their characters of content, plus the leading characters of the
    first message that differs when its content starts the same way.
    ``first_changed`` is the role of the first message that differs, or
    ``None`` when one request is a prefix of the other (a plain append).
    ``tools_changed`` is true when the offered tools differ.
    """
    old, new = previous[1], current[1]
    same = _shared(old, new)
    chars = _shared_chars(old, new, same)
    first_changed = new[same][1] if same < min(len(old), len(new)) else None
    return {
        "messages": same,
        "chars": chars,
        "previous_messages": len(old),
        "total_messages": len(new),
        "first_changed": first_changed,
        "tools_changed": previous[0] != current[0],
    }
