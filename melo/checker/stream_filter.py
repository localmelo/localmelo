"""Reasoning-aware display release with conservative line-based redaction.

A declared framing hint allows completed safe lines to be released. Without
one, the entire reply is buffered because a later closing tag can reclassify
all prior text. Redaction is pattern-based, not a guarantee against secrets.
"""

from __future__ import annotations

import re

from localmelo.melo.checker import redaction
from localmelo.melo.checker._stream_redaction import LineRedactor, snapshot_rules
from localmelo.melo.schema import (
    ReasoningTags,
    split_leading_thinking,
    split_thinking,
)

_TAG = re.compile(r"</?think>", re.IGNORECASE)
_TAGS = ("<think>", "</think>")
# What an explicit reply may begin with: a ``<think>`` or a ``<thinking>`` block.
_OPENINGS = ("<think>", "<thinking>")
_SPELLED_CLOSING = re.compile(r"</thinking>", re.IGNORECASE)


def _split_for_display(text: str) -> tuple[str, str]:
    """Separate all thinking blocks, conservatively handling incomplete tags.

    A closing tag without an opening tag moves preceding content into thinking,
    as for templates that open the block in the prompt. An unclosed opening
    tag anywhere keeps the remainder out of content. A trailing partial tag is
    withheld from content, since its intended meaning cannot be determined.
    This is intentionally more conservative than schema.split_thinking for
    malformed or multiple blocks. A closed ``<thinking>`` block that starts
    the reply is thinking, as in schema.split_thinking.
    """
    content: list[str] = []
    thinking: list[str] = []
    spelled = split_leading_thinking(text)
    if spelled is not None:
        thinking.append(spelled[0])
        text = spelled[1]
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


def split_framed(text: str, reasoning_tags: ReasoningTags) -> tuple[str, str]:
    """Split a whole reply according to a caller-declared model framing rule.

    In ``explicit`` only a leading opening tag starts reasoning. In
    ``closing_only`` everything before the first closing tag is reasoning.
    In ``none`` tags are ordinary content. These declarations are trusted;
    use the conservative no-hint filter for unknown model formats.
    """
    return split_thinking(text, strip_answer=False, reasoning_tags=reasoning_tags)


class StreamFilter:
    """Return safe ``(thinking_delta, content_delta)`` display fragments.

    Without a framing hint all output waits for finish. With a declared hint,
    redaction releases safe complete lines. Thinking remains buffered and is
    returned at finish; callers choose whether to display it. A hint is a
    trusted model-format declaration, not an observation of an earlier reply.
    Call finish only on completion; abort on failure to discard buffered text.
    """

    def __init__(
        self,
        reasoning_tags: ReasoningTags | None = None,
        *,
        separate_reasoning: bool = False,
    ) -> None:
        if reasoning_tags not in (None, "explicit", "closing_only", "none"):
            raise ValueError("unknown reasoning_tags")
        if separate_reasoning and reasoning_tags not in (None, "none"):
            raise ValueError("separate reasoning conflicts with inline tag framing")
        self._separate = separate_reasoning
        reasoning_tags = "none" if separate_reasoning else reasoning_tags
        self._hint = reasoning_tags
        self._parts: list[str] = []
        self._length = 0
        self._finished = False
        self._snapshot = snapshot_rules()
        self._redact = self._snapshot[0]
        self._raw = LineRedactor(self._snapshot)
        self._content = LineRedactor(self._snapshot)
        self._thinking: list[str] = []
        self._pending = ""
        self._looked = 0
        self._state = {
            None: "buffer",
            "none": "content",
            "explicit": "start",
            "closing_only": "thinking",
        }[reasoning_tags]

    def _accept(self, text: str, *, final: bool = False) -> str:
        self._pending += text
        if self._state == "start":
            stripped = self._pending.lstrip()
            lower = stripped.lower()
            if not final and (
                not stripped or any(tag.startswith(lower) for tag in _OPENINGS)
            ):
                return ""
            if lower.startswith("<think>"):
                self._pending = stripped[len("<think>") :]
                self._state = "thinking"
            elif lower.startswith("<thinking>"):
                self._state = "block"
            else:
                self._state = "content"
        if self._state == "block":
            # A spelled-out block is thinking only once it is closed: held
            # until then, and shown as it is if the reply ends without it.
            # Only the text since the last look is searched for the tag.
            closed = _SPELLED_CLOSING.search(self._pending, self._looked)
            spelled = split_leading_thinking(self._pending) if closed else None
            if spelled is not None:
                self._thinking.append(spelled[0])
                self._pending = spelled[1]
                self._state = "content"
            elif final:
                self._state = "content"
            else:
                self._looked = max(0, len(self._pending) - len("</thinking>"))
                return ""
        if self._state == "thinking":
            close = "</think>"
            match = re.search(re.escape(close), self._pending, re.IGNORECASE)
            if match:
                self._thinking.append(self._pending[: match.start()])
                self._pending = self._pending[match.end() :]
                self._state = "content"
            elif final:
                self._thinking.append(self._pending)
                self._pending = ""
            else:
                keep = max(
                    (
                        n
                        for n in range(1, len(close))
                        if self._pending.lower().endswith(close[:n])
                    ),
                    default=0,
                )
                if keep:
                    self._thinking.append(self._pending[:-keep])
                    self._pending = self._pending[-keep:]
                else:
                    self._thinking.append(self._pending)
                    self._pending = ""
                return ""
        if self._state == "content":
            text, self._pending = self._pending, ""
            return self._content.feed(text)
        return ""

    def _check_fragment(self, fragment: str) -> None:
        if self._finished:
            raise RuntimeError("stream filter already finished")
        if not isinstance(fragment, str):
            raise TypeError("fragment must be a string")
        self._length += len(fragment)
        if self._length > 1_048_576:
            self._parts.clear()
            self._thinking.clear()
            self._finished = True
            raise ValueError("stream display exceeds size limit")

    def feed_reasoning(self, fragment: str) -> tuple[str, str]:
        """Buffer a trusted separate reasoning field; never emit it as content."""
        if not self._separate:
            raise ValueError("separate reasoning was not declared")
        self._check_fragment(fragment)
        self._thinking.append(fragment)
        return "", ""

    def abort(self) -> None:
        """Discard buffered text; already released content cannot be recalled."""
        self._parts.clear()
        self._thinking.clear()
        self._pending = ""
        self._raw = LineRedactor(self._snapshot)
        self._content = LineRedactor(self._snapshot)
        self._finished = True

    def redact_for_storage(self, text: str) -> str:
        """Apply the stream snapshot, then the rules current at storage time.

        Call this separately for each completed display channel. Pattern-based
        redaction cannot guarantee detection of arbitrary secrets.
        """
        return redaction.redact_secrets(self._redact(text)[0])[0]

    def feed(self, fragment: str) -> tuple[str, str]:
        self._check_fragment(fragment)
        if self._hint is None:
            self._parts.append(fragment)
            return "", ""
        return "", self._accept(self._raw.feed(fragment))

    def finish(self) -> tuple[str, str]:
        if self._finished:
            raise RuntimeError("stream filter already finished")
        self._finished = True
        if self._hint is None:
            text = "".join(self._parts)
            self._parts.clear()
            thinking, content = _split_for_display(self._redact(text)[0])
            return self._redact(thinking)[0], self._redact(content)[0]
        content = self._accept(self._raw.finish(), final=True)
        content += self._content.finish()
        thinking = self._redact("".join(self._thinking).strip())[0]
        self._thinking.clear()
        return thinking, content
