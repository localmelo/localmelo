"""Prefix diagnostics: how much of a request repeats the one before it.

A server's prompt cache can only reuse the leading part of a request that is
identical to an earlier one. :func:`fingerprint` records what a request sends
and :func:`compare` says how far it matches the previous request, so a cache
miss can be traced to the message that changed first. Nothing here changes a
request body.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from localmelo.melo.schema import Message, ToolDef

# The offered tools, then each message as (serialized form, role, content):
# the form decides equality, the content gives the characters.
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
            ),
            message.role,
            message.content,
        )
        for message in messages
    ]


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
    limit = min(len(old), len(new))
    same = chars = 0
    while same < limit and old[same][0] == new[same][0]:
        chars += len(new[same][2])
        same += 1
    first_changed = None
    if same < limit:
        first_changed = new[same][1]
        a, b = old[same][2], new[same][2]
        shared = 0
        while shared < min(len(a), len(b)) and a[shared] == b[shared]:
            shared += 1
        chars += shared
    return {
        "messages": same,
        "chars": chars,
        "previous_messages": len(old),
        "total_messages": len(new),
        "first_changed": first_changed,
        "tools_changed": previous[0] != current[0],
    }
