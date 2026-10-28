"""Conservative line-granular release for the existing redaction rules."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from localmelo.melo.checker import redaction
from localmelo.melo.checker.redaction import redact_secrets

# Any unreviewed change to the rules disables incremental release.
_RULES_SHA256 = "7f17ad11ae7be99bb8095a0cb398b8e5233d2937ee0171a724a1768070a1110f"


def _known_rules() -> bool:
    try:
        return (
            bool(redaction.__file__)
            and hashlib.sha256(Path(redaction.__file__).read_bytes()).hexdigest()
            == _RULES_SHA256
        )
    except OSError:
        return False


# Rule inventory: key shapes start at sk-/gh.../github_pat_/AKIA/ASIA and
# end at their first non-token character; OTP codes/keywords are line-local
# and have a 20-character reach. Bearer starts at a word-boundary "bearer"
# and ends at the first non-token character after its whitespace and token.
# The bearer keyword and whitespace can span lines. The source fingerprint
# below makes any rule change fail closed until these anchors are reviewed.
_OPEN_BEARER = re.compile(r"\bbearer\s*\Z", re.IGNORECASE)


class LineRedactor:
    """Hold incomplete lines and any bearer prefix that can cross a line.

    A complete newline is a conservative terminator: even when a token ends
    earlier, its line remains buffered. This lets the existing three passes
    run on the same context as whole-response redaction.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._finished = False
        self._known = _known_rules()

    def feed(self, text: str) -> str:
        if self._finished:
            raise RuntimeError("redactor already finished")
        self._pending += text
        if not self._known:
            return ""
        end = self._pending.rfind("\n") + 1
        if not end:
            return ""
        match = _OPEN_BEARER.search(self._pending[:end])
        if match:
            end = self._pending.rfind("\n", 0, match.start()) + 1
        prefix, self._pending = self._pending[:end], self._pending[end:]
        return redact_secrets(prefix)[0]

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("redactor already finished")
        self._finished = True
        text, self._pending = self._pending, ""
        return redact_secrets(text)[0]
