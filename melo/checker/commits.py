"""A commit made from a step does not credit melo or an AI.

Checked before a ``shell_exec`` or ``python_exec`` call runs (see
:meth:`~localmelo.melo.checker.checker.Checker.pre_execute`): a call that
makes a git commit and has a line in it that credits melo or an AI is refused
with a reason, and the model commits again without that line. A trailer that
names a person is not touched.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_NAMES = (
    r"\b(?:local)?melo\b|\bclaude\b|\banthropic\b|\bopenai\b|\bchatgpt\b|\bgpt[\w.-]*"
    r"|\bcodex\b|\bcopilot\b|\bgemini\b|\bdeepseek\b|\bqwen\b|(?-i:\bAI\b)"
)
# The rest of a line, up to a quote or a backslash: in a shell command or in
# Python code the message is a string, and a "\n" ends its line.
_LINE = r"[^\n\"'\\]*?"
_CREDIT = re.compile(
    rf"co-authored-by:{_LINE}(?:{_NAMES})"
    rf"|generated (?:with|by)\b{_LINE}(?:{_NAMES})"
    r"|claude-session:"
    r"|claude\.ai/code/session|chatgpt\.com/(?:share|c)/|chat\.openai\.com/share/",
    re.IGNORECASE,
)
_GIT = re.compile(r"\bgit\b")
_COMMIT = re.compile(r"\bcommit\b")
_SHOWN = 80


def commit_credit_refusal(tool_name: str, arguments: Mapping[str, Any]) -> str | None:
    """Why the call is refused, or ``None`` when it is not a credited commit."""
    if tool_name not in ("shell_exec", "python_exec"):
        return None
    text = arguments.get("command" if tool_name == "shell_exec" else "code")
    if isinstance(text, list):
        text = " ".join(map(str, text))
    if not isinstance(text, str) or not (_GIT.search(text) and _COMMIT.search(text)):
        return None
    found = _CREDIT.search(text)
    if found is None:
        return None
    line = " ".join(found.group(0).split())[:_SHOWN]
    return (
        f'This commit message credits melo or an AI ("{line}"). Commit again '
        "without that line; a Co-Authored-By trailer that names a person is fine."
    )
