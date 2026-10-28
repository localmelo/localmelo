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
_RULES_SHA256 = "7f17ad11ae7be99bb8095a0cb398b8e5233d2937ee0171a724a1768070a1110f"

# Each active regex is reviewed with its start and end conditions. A newline
# closes every current match except a bearer prefix, held by _OPEN_BEARER.
_RULE_INVENTORY = {
    "_KEY": ("key prefix", "non-token or newline"),
    "_BEARER": ("bearer plus whitespace", "non-token; hold open bearer"),
    "_CODE_KEYWORD": ("code keyword", "keyword end or newline"),
    "_CODE": ("digit boundary", "non-digit or newline"),
}
_MATCH_SITES = {
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
        for name in ("redact_secrets", "_redact_codes", "_near_a_keyword"):
            function = getattr(rules, name)
            if (
                not isinstance(function, FunctionType)
                or function.__code__ != expected[name]
            ):
                return False
        return bool(
            rules._KEY.pattern == r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"
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
# The bearer keyword and whitespace can span lines. The source fingerprint
# above and semantic inventory make unreviewed changes fail closed.
_OPEN_BEARER = re.compile(r"\bbearer\s*\Z", re.IGNORECASE)


class LineRedactor:
    """Hold incomplete lines and any bearer prefix that can cross a line.

    A complete newline is a conservative terminator: even when a token ends
    earlier, its line remains buffered. This lets the existing three passes
    run on the same context as whole-response redaction.
    """

    def __init__(self, snapshot: tuple[Redact, bool] | None = None) -> None:
        self._pending = ""
        self._finished = False
        self._redact, self._known = (
            snapshot if snapshot is not None else snapshot_rules()
        )

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
        return self._redact(prefix)[0]

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("redactor already finished")
        self._finished = True
        text, self._pending = self._pending, ""
        return self._redact(text)[0]
