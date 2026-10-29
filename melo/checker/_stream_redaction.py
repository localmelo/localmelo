"""Conservative line-granular release for the existing redaction rules."""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

from localmelo.melo.checker import redaction

Redact = Callable[[str], tuple[str, int]]

# Any unreviewed change to the rules disables incremental release.
_RULES_SHA256 = "136eaa8013f7ce472d1cd8e1f45d8011746f78aaca98ecd227aa69a68c6fe8eb"

# Each active regex is reviewed with its start and end conditions. A newline
# closes every current match except a bearer prefix, held by _OPEN_BEARER, and
# a PEM block, held from its BEGIN line by LineRedactor until its END line.
_RULE_INVENTORY = {
    "_PEM": ("PEM private key BEGIN marker", "END marker or end of text; hold open"),
    "_KEY": ("key prefix", "non-token or newline"),
    "_BEARER": ("bearer plus whitespace", "non-token; hold open bearer"),
    "_CODE_KEYWORD": ("code keyword", "keyword end or newline"),
    "_CODE": ("digit boundary", "non-digit or newline"),
}
_MATCH_SITES = {
    ("_PEM", "sub"),
    ("_KEY", "subn"),
    ("_BEARER", "subn"),
    ("_CODE_KEYWORD", "finditer"),
    ("_CODE", "finditer"),
}
_MATCH_METHODS = {
    "match",
    "fullmatch",
    "search",
    "findall",
    "finditer",
    "sub",
    "subn",
    "split",
}


def _reviewed_inventory(source: bytes, bindings: dict[str, object]) -> bool:
    patterns = {
        name for name, value in bindings.items() if isinstance(value, re.Pattern)
    }
    if patterns != _RULE_INVENTORY.keys():
        return False
    sites: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _MATCH_METHODS:
            continue
        if not isinstance(node.func.value, ast.Name):
            return False
        sites.append((node.func.value.id, node.func.attr))
    return len(sites) == len(_MATCH_SITES) and set(sites) == _MATCH_SITES


def _known_rules(bindings: dict[str, object]) -> bool:
    try:
        source_file = bindings.get("__file__")
        if not isinstance(source_file, str):
            return False
        source = Path(source_file).read_bytes()
        if hashlib.sha256(source).hexdigest() != _RULES_SHA256:
            return False
        if not _reviewed_inventory(source, bindings):
            return False
        module_code = compile(source, source_file, "exec")
        expected = {
            code.co_name: code
            for code in module_code.co_consts
            if isinstance(code, CodeType)
        }
        rules = SimpleNamespace(**bindings)
        for name in (
            "redact_secrets",
            "_redact_pem",
            "_redact_codes",
            "_near_a_keyword",
        ):
            function = getattr(rules, name)
            if (
                not isinstance(function, FunctionType)
                or function.__code__ != expected[name]
            ):
                return False
        return bool(
            rules._PEM.pattern
            == r"(-----BEGIN (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY-----)"
            r"(.*?)"
            r"(-----END (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY-----|\Z)"
            and rules._PEM.flags == re.UNICODE | re.DOTALL
            and rules._KEY.pattern == r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"
            r"|(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"
            r"|(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"
            and rules._KEY.flags == re.UNICODE
            and rules._BEARER.pattern == r"(\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}"
            and rules._BEARER.flags == re.UNICODE | re.IGNORECASE
            and rules._CODE_KEYWORD.pattern
            == r"verification code|passcode|(?<![A-Za-z0-9_])otp(?![A-Za-z0-9_])|验证码"
            and rules._CODE_KEYWORD.flags == re.UNICODE | re.IGNORECASE
            and rules._CODE.pattern == r"(?<![A-Za-z0-9])[0-9]{4,8}(?![A-Za-z0-9])"
            and rules._CODE.flags == re.UNICODE
            and rules._CODE_REACH == 20
            and rules.REDACTED == "[redacted]"
        )
    except (OSError, KeyError, AttributeError, TypeError, SyntaxError):
        return False


def snapshot_rules() -> tuple[Redact, bool]:
    """Freeze the current redaction functions and their module bindings."""
    bindings = vars(redaction).copy()
    for name, value in list(bindings.items()):
        if isinstance(value, FunctionType) and value.__globals__ is vars(redaction):
            bindings[name] = FunctionType(
                value.__code__,
                bindings,
                value.__name__,
                value.__defaults__,
                value.__closure__,
            )
    return bindings["redact_secrets"], _known_rules(bindings)


# Rule inventory: key shapes start at sk-/gh.../github_pat_/AKIA/ASIA and
# end at their first non-token character; OTP codes/keywords are line-local
# and have a 20-character reach. Bearer starts at a word-boundary "bearer"
# and ends at the first non-token character after its whitespace and token.
# The bearer keyword and whitespace can span lines. A PEM block starts at a
# BEGIN marker for a PRIVATE KEY label, anywhere in a line, and ends at the
# next END marker, or at the end of the text; it spans lines. The source
# fingerprint above and semantic inventory make unreviewed changes fail closed.
_OPEN_BEARER = re.compile(r"\bbearer\s*\Z", re.IGNORECASE)

# The same markers as redaction._PEM, split so an open block can be found.
_PEM_BEGIN = re.compile(r"-----BEGIN (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY-----")
_PEM_END = re.compile(r"-----END (?:[A-Z0-9]{1,16} ){0,3}PRIVATE KEY-----")
# The longest END marker: 9 + 3 * 17 + 11 + 5 characters, with room to spare.
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

    def __init__(self, snapshot: tuple[Redact, bool] | None = None) -> None:
        self._pending = ""
        self._finished = False
        self._dropping = False
        self._redact, self._known = (
            snapshot if snapshot is not None else snapshot_rules()
        )

    def feed(self, text: str) -> str:
        if self._finished:
            raise RuntimeError("redactor already finished")
        self._pending += text
        if not self._known:
            return ""
        return self._step()

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("redactor already finished")
        self._finished = True
        out = self._drop_body() if self._dropping else ""
        text, self._pending = self._pending, ""
        return out + (self._redact(text)[0] if not self._dropping else "")

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
                return self._redact(head)[0] + self._step()
            end = min(end, hold)
        if not end:
            return ""
        prefix, self._pending = pending[:end], pending[end:]
        return self._redact(prefix)[0]

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
