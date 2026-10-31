"""The closing line a task asks its answer to carry, found in the task's own text.

``AgentOptions.answer_on_stop`` uses this to notice an answer that lacks the line
(``FINAL ANSWER: ...``, ``Answer: ...``) and to ask the model for just that line.
The detection is conservative: a label counts only when the sentence around it says
it is a closing, final or starting line, so a few-shot ``Answer: 4`` or a plain
mention of "the answer:" asks for nothing.
"""

from __future__ import annotations

import re

# The labels that need no quoting: a closing-line label in the usual wording.
_KNOWN_LABEL = (
    r"(?:FINAL ANSWER|Final Answer|Final answer|ANSWER|Answer|最终答案|答案)\s*[:：]"
)
# A label the task puts in quotes or backticks (the closing quote must follow, which
# keeps an apostrophe in "don't" from opening one), or writes in capitals.
_QUOTED_LABEL = r"(?<!\w)[`'\"“‘]\s*[^\s`'\"”’:：][^`'\"”’\n:：]{0,39}[:：][^`'\"”’\n]{0,60}[`'\"”’]"
_CAPS_LABEL = r"[A-Z][A-Z _-]{2,30}[:：]"
_LABEL = re.compile(
    f"(?P<quoted>{_QUOTED_LABEL})|(?P<known>{_KNOWN_LABEL})|(?P<caps>{_CAPS_LABEL})"
)

# What marks a label as the line the answer must carry, in the same sentence.
_CUE = re.compile(
    r"\b(?:end|ends|ending|finish|finishes|finishing|conclude|concludes|close|closes)\b"
    r"[^\n]{0,60}?\bwith\b"
    r"|\b(?:final|last|closing|separate)\s+line\b"
    r"|\b(?:start|starts|starting|begin|begins|beginning)\s+with\b"
    r"|\b(?:reply|respond)\s+with\b"
    r"|\b(?:prefix|prefixed)\b"
    r"|\b(?:in|using)\s+the\s+(?:form|format)\b|\bline\s+(?:of|in)\s+the\s+form\b"
    r"|\banswer\s+format\b|\bformat\s*:"
    r"|最后一行|最后|末尾|结尾|结束|开头|格式|形式|单独一行|另起一行",
    re.IGNORECASE,
)
# A sentence ends at a newline, at 。！？, or at . ! ? followed by a space or the end.
_SENTENCE_END = re.compile(r"\n|[。！？]|[.!?](?=\s|$)")
_NOISE = " \t>*_`#-"  # Markdown a model puts before a line's label


def _tidy(label: str) -> str:
    """The label inside *label*, a match of ``_LABEL``: up to its colon, quotes dropped."""
    text = label.strip("`'\"“‘”’ ")
    colon = min((i for i in (text.find(":"), text.find("：")) if i != -1), default=-1)
    return text[: colon + 1].strip() if colon != -1 else ""


def _sentence(text: str, start: int, end: int) -> tuple[int, int]:
    """The span of the sentence of *text* around ``text[start:end]``, at most 240 characters."""
    before = 0
    for found in _SENTENCE_END.finditer(text, 0, start):
        before = found.end()
    after = _SENTENCE_END.search(text, end)
    stop = after.start() if after else len(text)
    return max(before, start - 120), min(stop, end + 120)


def requested_closing_label(task: str) -> str | None:
    """The label of the closing line *task* asks for (``"FINAL ANSWER:"``), or ``None``.

    The last label in the text whose sentence also says it is the final, last or
    starting line, or the answer's format. A label that is only in capitals
    (``RESULT:``) counts when the cue comes before it, so ``IMPORTANT: end with
    a summary`` asks for nothing. A task that names no such label gives
    ``None``: the format is never invented.
    """
    found: str | None = None
    for match in _LABEL.finditer(task):
        label = _tidy(match.group(0))
        if not label or not label.rstrip(":").strip():
            continue
        first, last = _sentence(task, match.start(), match.end())
        # The cue may follow a quoted or usual label, but must precede a bare one.
        stop = match.start() if match.group("caps") else last
        if _CUE.search(task, first, stop):
            found = label
    return found


def has_closing_line(answer: str, label: str) -> bool:
    """Whether a line of *answer* starts with *label* (Markdown marks before it aside)."""
    plain = label.replace("：", ":")
    for line in answer.replace("：", ":").splitlines():
        if line.lstrip(_NOISE).startswith(plain):
            return True
    return False


def closing_line_of(reply: str, label: str) -> str | None:
    """The last line of *reply* that starts with *label*, tidied; ``None`` if none does."""
    plain = label.replace("：", ":")
    for line in reversed(reply.splitlines()):
        stripped = line.strip().lstrip(_NOISE)
        if stripped.replace("：", ":").startswith(plain):
            return stripped.rstrip("*_` ")
    return None
