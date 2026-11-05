"""Conservative line-granular release for the existing redaction rules."""

from __future__ import annotations

import re

from localmelo.melo.checker.redaction import redact_secrets

# Why releasing a complete line at a time is safe: key shapes start at
# sk-/gh.../github_pat_/AKIA/ASIA and end at their first non-token character;
# OTP codes/keywords are line-local and have a 20-character reach. Bearer
# starts at a word-boundary "bearer" and ends at the first non-token character
# after its whitespace and token. The bearer keyword and whitespace can span
# lines. A PEM block starts at a BEGIN marker for a PRIVATE KEY label,
# anywhere in a line, and ends at the next END marker, or at the end of the
# text; it spans lines. Every other rule stays within one line. Tests in
# tests/checker/test_stream_redaction.py fail the build when a rule can match
# across lines, when redaction.py matches with a regex they do not inspect, or
# when a streamed split gives other text than the whole redaction, so this
# list cannot go stale.

# A bearer keyword (and any white space) at the end of the text: its token is
# still to come, so its line is held back.
_OPEN_BEARER = re.compile(r"\bbearer\s*\Z", re.IGNORECASE)

# The same markers as redaction._PEM, split so an open block can be found.
_PEM_BEGIN = re.compile(
    r"-----BEGIN (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY(?: BLOCK)?-----"
)
_PEM_END = re.compile(r"-----END (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY(?: BLOCK)?-----")
# The longest END marker: 9 + 3 * 17 + 11 + 6 + 5 characters, with room to spare.
_PEM_END_MAX = 96

# The most characters held for one open PEM block, counted from its BEGIN
# marker. Past it the BEGIN line and the marker are released and the body is
# dropped until the END marker arrives, so a key never streams out and memory
# stays bounded. Whole-text redaction gives the same text.
PEM_HOLD_LIMIT = 16 * 1024


class LineRedactor:
    """Hold incomplete lines, any bearer prefix and any open PEM block.

    A complete newline is a conservative terminator: even when a token ends
    earlier, its line remains buffered. This lets the existing passes run on
    the same context as whole-response redaction. A PEM block that has begun
    but not ended is held from its BEGIN line until its END line, or until
    :data:`PEM_HOLD_LIMIT` characters, then its body is dropped as it arrives.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._finished = False
        self._dropping = False

    def feed(self, text: str) -> str:
        if self._finished:
            raise RuntimeError("redactor already finished")
        self._pending += text
        return self._step()

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("redactor already finished")
        self._finished = True
        out = self._drop_body() if self._dropping else ""
        text, self._pending = self._pending, ""
        return out + (redact_secrets(text)[0] if not self._dropping else "")

    def _step(self) -> str:
        out = self._drop_body() if self._dropping else ""
        return out if self._dropping else out + self._release()

    def _drop_body(self) -> str:
        """Skip a dropped key body; at its END marker, restore the newline."""
        stop = _PEM_END.search(self._pending)
        if stop is None:
            self._pending = self._pending[-(_PEM_END_MAX - 1) :]
            return ""
        self._pending = self._pending[stop.start() :]
        self._dropping = False
        return "\n"

    def _release(self) -> str:
        pending = self._pending
        end = pending.rfind("\n") + 1
        match = _OPEN_BEARER.search(pending[:end])
        if match:
            end = pending.rfind("\n", 0, match.start()) + 1
        open_block = self._open_block(end)
        if open_block is not None:
            begin, hold = open_block
            if len(pending) - begin.start() > PEM_HOLD_LIMIT:
                head, self._pending = pending[: begin.end()], pending[begin.end() :]
                self._dropping = True
                return redact_secrets(head)[0] + self._step()
            end = min(end, hold)
        if not end:
            return ""
        prefix, self._pending = pending[:end], pending[end:]
        return redact_secrets(prefix)[0]

    def _open_block(self, end: int) -> tuple[re.Match[str], int] | None:
        """The first BEGIN marker whose END marker is not in ``pending[:end]``.

        Also the start of the line to hold from: the BEGIN's line, or an earlier
        line when a finished block's END marker shares that line, so no block is
        ever cut in two.
        """
        pending = self._pending
        closed: list[tuple[int, int]] = []
        pos = 0
        while (begin := _PEM_BEGIN.search(pending, pos)) is not None:
            stop = _PEM_END.search(pending, begin.end())
            if stop is None or stop.end() > end:
                hold = pending.rfind("\n", 0, begin.start()) + 1
                for start, stop_end in reversed(closed):
                    if stop_end <= hold:
                        break
                    hold = pending.rfind("\n", 0, start) + 1
                return begin, hold
            closed.append((begin.start(), stop.end()))
            pos = stop.end()
        return None
