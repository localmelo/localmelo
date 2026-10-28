"""Conservative incremental input filter for text shown to a user.

The existing redactor has unbounded token patterns, and an implicit thinking
block may end with a closing tag anywhere in the response. Consequently this
filter buffers the entire response: no display text is available until finish.
This trades streaming latency for consistent decisions across fragment
boundaries. A response above the character limit is refused.

Redaction only covers the patterns in :func:`redact_secrets`; it is not a
secrecy guarantee. Thinking is also sensitive even after redaction and should
only be displayed when the caller explicitly wants to show it.
"""

from __future__ import annotations

import re

from localmelo.melo.checker.redaction import redact_secrets

_TAG = re.compile(r"</?think>", re.IGNORECASE)
_TAGS = ("<think>", "</think>")


def _split_for_display(text: str) -> tuple[str, str]:
    """Separate all thinking blocks, conservatively handling incomplete tags.

    A closing tag without an opening tag moves preceding content into thinking,
    as for templates that open the block in the prompt. An unclosed opening
    tag anywhere keeps the remainder out of content. A trailing partial tag is
    withheld from content, since its intended meaning cannot be determined.
    This is intentionally more conservative than schema.split_thinking for
    malformed or multiple blocks.
    """
    content: list[str] = []
    thinking: list[str] = []
    depth = 0
    pos = 0
    for tag in _TAG.finditer(text):
        segment = text[pos : tag.start()]
        if depth:
            thinking.append(segment)
        else:
            content.append(segment)
        if tag.group().lower() == "<think>":
            depth += 1
        elif depth:
            depth -= 1
        else:
            thinking.extend(content)
            content.clear()
        pos = tag.end()

    remainder = text[pos:]
    if depth:
        thinking.append(remainder)
    else:
        # Withhold any final prefix of a tag, including a lone '<'.
        partial = 0
        lower = remainder.lower()
        for candidate in _TAGS:
            for length in range(1, len(candidate)):
                if lower.endswith(candidate[:length]):
                    partial = max(partial, length)
        if partial:
            content.append(remainder[:-partial])
            thinking.append(remainder[-partial:])
        else:
            content.append(remainder)
    return "".join(thinking).strip(), "".join(content)


class StreamFilter:
    """Collect fragments and return ``(thinking, content)`` display deltas.

    ``feed(fragment)`` always returns ``('', '')``. ``finish()`` returns the
    separately redacted thinking and content exactly once; subsequent feed or
    finish calls raise RuntimeError. Content whitespace is preserved, while
    thinking is stripped, matching split_thinking's strip_answer=False mode
    for ordinary single-block responses. Use a fresh instance per response.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self._finished = False

    def feed(self, fragment: str) -> tuple[str, str]:
        if self._finished:
            raise RuntimeError("stream filter already finished")
        if not isinstance(fragment, str):
            raise TypeError("fragment must be a string")
        self._length += len(fragment)
        if self._length > 1_048_576:
            self._parts.clear()
            self._finished = True
            raise ValueError("stream display exceeds size limit")
        self._parts.append(fragment)
        return "", ""

    def finish(self) -> tuple[str, str]:
        if self._finished:
            raise RuntimeError("stream filter already finished")
        self._finished = True
        text = "".join(self._parts)
        self._parts.clear()
        thinking, content = _split_for_display(redact_secrets(text)[0])
        return redact_secrets(thinking)[0], redact_secrets(content)[0]
