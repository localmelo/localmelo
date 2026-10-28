"""Read stored trailing attachment blocks; composing attachments is a later feature."""

from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK = re.compile(
    r'\n\n<attachment name="([^"\n]{1,200})" chars="([0-9]+)"'
    r'(?: cut="([0-9]+)")?>\n(.*?)\n</attachment>',
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAttachment:
    name: str
    chars: int
    kept: int
    text: str


def parse_user_text(user_text: str) -> tuple[str, list[ParsedAttachment]]:
    """Read a complete trailing suffix; malformed blocks remain ordinary typed text."""
    start = user_text.find('\n\n<attachment name="')
    if start < 0:
        return user_text, []
    position = start
    attachments: list[ParsedAttachment] = []
    while position < len(user_text):
        match = _BLOCK.match(user_text, position)
        if match is None:
            return user_text, []
        name, whole, cut, body = match.groups()
        body = body.replace("<\\/attachment", "</attachment")
        try:
            chars, kept = int(whole), int(cut) if cut is not None else len(body)
        except ValueError:
            return user_text, []
        if not 0 <= kept <= chars:
            return user_text, []
        attachments.append(ParsedAttachment(name, chars, kept, body))
        position = match.end()
    return user_text[:start], attachments
