"""The main chat's rules in code: answer, delegate or change a task, before any model call.

Pure functions over the text the person typed and the open tasks:

- :func:`route` picks what the main chat does with a message: ``delegate``
  (``/task …`` or the composer's Run as a task), ``control`` (a control word
  and a task the person named), ``chore`` (a repeating phrase, left to the
  scheduled chores), ``answer`` (a greeting or a short question) or ``plan``
  (the main chat's own planning step decides).
- :func:`triage` gives a new task its level from the person's own words:
  Today for urgency words or a deadline today, Whenever for "no rush",
  Normal otherwise.
- :func:`allows` is the principal rule for the main chat's control tools: a
  change counts as the person's only when their own typed message has the
  control word and names the task. Page text, tool output and a task's
  report are never that message.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

# Levels in words, P0 to P5 (design 2.5); P-codes stay in developer mode.
LEVEL_WORDS: Final = ("Now", "Today", "Normal", "Whenever", "If idle", "Later")
FINISHED: Final = frozenset({"done", "failed", "cancelled", "expired"})

Route = Literal["delegate", "control", "chore", "answer", "plan"]
Action = Literal["stop", "pause", "resume", "first"]


def level_word(level: object) -> str:
    """``Normal`` for 2; ``Normal`` for anything that is not a level."""
    if isinstance(level, int) and not isinstance(level, bool) and 0 <= level <= 5:
        return LEVEL_WORDS[level]
    return LEVEL_WORDS[2]


def level_number(value: object) -> int | None:
    """0-5 from a level word (``today``, ``If idle``) or a P-code (``P1``); else None."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 0 <= value <= 5 else None
    text = " ".join(str(value).split()).casefold()
    if re.fullmatch(r"p[0-5]", text):
        return int(text[1])
    words = [word.casefold() for word in LEVEL_WORDS]
    return words.index(text) if text in words else None


@dataclass(frozen=True)
class TaskRef:
    """An open task as the rules see it."""

    task_id: str
    title: str
    state: str = "queued"
    level: int = 2


@dataclass(frozen=True)
class Routed:
    """What the main chat does with one message."""

    route: Route
    text: str = ""  # the work to delegate
    action: Action | None = None
    task: TaskRef | None = None


def norm(text: str) -> str:
    """NFKC, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _has_cjk(text: str) -> bool:
    return any("\u3400" <= ch <= "\u9fff" for ch in text)


def _find(pattern: str, text: str) -> re.Match[str] | None:
    """*pattern* on token boundaries for Latin words, anywhere for Chinese."""
    if _has_cjk(pattern):
        return re.search(re.escape(pattern), text)
    return re.search(r"(?<![\w])" + re.escape(pattern) + r"(?![\w])", text)


def _words(text: str) -> tuple[str, ...]:
    return tuple(text.split("|"))


# Control words by direction (scheduler design 2.5); longer phrases first.
_CONTROL: Final[dict[Action, tuple[str, ...]]] = {
    "pause": _words("put on hold|hold off|pause|暂停|先停"),
    "resume": _words("unpause|resume|continue|carry on|继续|恢复"),
    "stop": _words("cancel|stop|abort|kill|取消|停止|停掉|停"),
    "first": _words("top priority|prioritize|prioritise|first|最优先|先做|优先"),
}
_NEGATION: Final = re.compile(
    r"(?:\b(?:don['’]?t|do not|never|not|no need to|without)\s+(?:\w+\s+){0,2}$)"
    r"|(?:(?:不要|别|不用|不必|没|不)$)"
)
# Words that name no task, dropped before the rest is matched to a title.
_FILLER: Final = frozenset(
    _words(
        "the|a|an|my|our|this|that|task|job|please|now|pls|can|you|could|would|it|one|"
        "on|with|for|make|do|get|go|ahead|let's|lets|and|then|thanks"
    )
)
_FILLER_ZH: Final = _words("一下|这个|那个|任务|请|把|吧|了|的|先")


def control_action(typed: str) -> tuple[Action, str] | None:
    """The first control word in *typed* not under a negation, with its phrase."""
    text = norm(typed)
    for action, phrases in _CONTROL.items():
        for phrase in phrases:
            found = _find(phrase, text)
            if found is not None and not _NEGATION.search(text[: found.start()]):
                return action, phrase
    return None


def _rest(text: str, phrase: str) -> str:
    """*text* without the control phrase, filler and punctuation."""
    text = text.replace(phrase, " ", 1)
    for word in _FILLER_ZH:
        text = text.replace(word, " ")
    text = re.sub(r"[^\w\s:.-]+", " ", text)
    return " ".join(word for word in text.split() if word not in _FILLER)


def _title_tokens(title: str) -> set[str]:
    words = {w for w in re.findall(r"[^\W\d_]{4,}", norm(title)) if w not in _FILLER}
    runs = re.findall(r"[\u3400-\u9fff]{2,}", norm(title))
    return words | {run[i : i + 2] for run in runs for i in range(len(run) - 1)}


def named_task(
    typed: str, phrase: str, tasks: Sequence[TaskRef], *, strict: bool = True
) -> TaskRef | None:
    """The one open task *typed* names, besides its control *phrase*, or None.

    Strict (code's own route): what is left is the task's id, its title, or
    a part of exactly one title (3 characters at least, 2 for Chinese).
    Loose (a tool's principal check): also a word of the title, of 4
    letters at least, written in *typed*.
    """
    text = norm(typed)
    rest = _rest(text, phrase)
    open_tasks = [task for task in tasks if task.state not in FINISHED]
    for task in open_tasks:
        if task.task_id and re.search(rf"\b{re.escape(task.task_id)}\b", text):
            return task
    if rest and len(rest) >= (2 if _has_cjk(rest) else 3):
        exact = [task for task in open_tasks if norm(task.title) == rest]
        found = exact or [task for task in open_tasks if rest in norm(task.title)]
        if len(found) == 1:
            return found[0]
    if strict:
        return None
    words = set(re.findall(r"\w+", text))
    found = [
        task
        for task in open_tasks
        if any(
            token in words or (_has_cjk(token) and token in text)
            for token in _title_tokens(task.title)
        )
    ]
    return found[0] if len(found) == 1 else None


def allows(typed: str, action: Action, task: TaskRef) -> bool:
    """The principal rule: *typed* (the person's latest message) asks for *action* on *task*."""
    found = control_action(typed)
    if found is None or found[0] != action:
        return False
    named = named_task(typed, found[1], [task], strict=False)
    return named is not None and named.task_id == task.task_id


_LATER: Final = _words("later|not now|以后|稍后|晚点")


def allows_level(typed: str, raise_: bool, task: TaskRef) -> bool:
    """The principal rule for a level: *typed* raises (or lowers) *task* in its own words."""
    text = norm(typed)
    phrase = ""
    if raise_:
        found = control_action(typed)
        if found is not None and found[0] == "first":
            phrase = found[1]
        elif triage(typed)[0] != 1:
            return False
    elif triage(typed)[0] != 3 and not any(_find(word, text) for word in _LATER):
        return False
    return named_task(typed, phrase, [task], strict=False) is not None


# ── Delegation and its level ──

_URGENT: Final = _words(
    "as soon as possible|right away|right now|immediately|urgently|urgent|asap|"
    "end of day|end of the day|紧急|尽快|马上|立刻|立即|很急|着急|急用|今天之内|今晚之前|下班前"
)
_DEADLINE: Final = re.compile(
    r"\b(?:by|before|due)\s+(?:today|tonight|noon|eod|\d{1,2}(?::\d{2}\s*(?:am|pm)?|\s*(?:am|pm)))\b"
    r"|今[天晚][^。.!！?？]{0,8}前|\d{1,2}\s*点前"
)
_EASY: Final = _words(
    "no rush|not urgent|no hurry|whenever|low priority|when you have time|"
    "不急|不着急|不紧急|有空|慢慢来"
)


def triage(typed: str) -> tuple[int, str]:
    """A delegated task's level and reason, from the person's own words only.

    Ease words first, so "not urgent" is never urgency; then urgency words or a
    deadline today (Today); else Normal (design 2.5; S5's full table later).
    """
    text = norm(typed)
    for phrase in _EASY:
        if _find(phrase, text):
            return 3, f'you said "{phrase}"'
    for phrase in _URGENT:
        found = _find(phrase, text)
        if found and not _NEGATION.search(text[: found.start()]):
            return 1, f'you said "{phrase}"'
    deadline = _DEADLINE.search(text)
    if deadline:
        return 1, f'you said "{deadline.group(0).strip()}"'
    return 2, "you asked for it"


_QUESTION_START: Final = re.compile(
    r"^(?:what|why|how|when|where|who|whom|which|whose|is|are|am|was|were|can|could|"
    r"do|does|did|should|would|will|have|has|may)\b"
)
_QUESTION_ZH: Final = re.compile(r"(?:吗|呢|么|什么|怎么|为什么|哪|谁|几|是否|多少)")


def is_question(typed: str) -> bool:
    text = norm(typed).rstrip()
    return (
        text.endswith(("?", "？"))
        or bool(_QUESTION_START.match(text))
        or bool(_QUESTION_ZH.search(text[-6:]))
    )


_GREETING: Final = re.compile(
    r"^(?:hi|hello|hey|yo|thanks|thank you|thx|good (?:morning|afternoon|evening|night)|"
    r"你好|您好|嗨|哈喽|谢谢|早上好|早安|晚安|晚上好)(?:\s+(?:there|melo|all))?[\s!.,！。~]*$"
)
_NAMES_THING: Final = re.compile(
    r"https?://|www\.|[~/][\w.-]|\b[\w-]+\.[a-z0-9]{1,5}\b|\b(?:file|folder|directory)\b|文件|目录"
)
_ACTION_VERBS: Final = re.compile(
    r"\b(?:write|make|create|build|fix|refactor|summari[sz]e|research|find|search|download|"
    r"generate|draft|plan|translate|analy[sz]e|compare|review|edit|run|install|update|delete|"
    r"send|organi[sz]e|prepare|convert|collect|scrape|code|implement)\b"
    r"|写|做|生成|修|改|总结|查找|搜索|下载|整理|翻译|分析|比较|准备|制作|创建|运行|安装|删除|发送"
)
_REPEATS: Final = re.compile(
    r"\bevery\s+(?:day|weekday|morning|evening|night|week|month|hour|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|\d+\s+(?:minutes?|hours?|days?))\b"
    r"|\b(?:daily|weekly|monthly|hourly)\b|每天|每周|每晚|每早|每个工作日|每小时|每月|每隔"
)
# A message longer than this is not a quick question, and code never reads it as a control.
SHORT_CHARS: Final = 120
CONTROL_CHARS: Final = 80


def route(
    typed: str, tasks: Sequence[TaskRef] = (), *, as_task: bool = False
) -> Routed:
    """What the main chat does with *typed*, by code's rules (design 2.4)."""
    stripped = typed.strip()
    command = re.match(r"^/task(?:\s+|$)", stripped)
    if as_task or command:
        work = stripped[command.end() :] if command else stripped
        return Routed("delegate", text=work.strip())
    text = norm(stripped)
    if len(text) <= CONTROL_CHARS:
        found = control_action(stripped)
        if found is not None:
            task = named_task(stripped, found[1], tasks)
            if task is not None:
                return Routed("control", action=found[0], task=task)
    if _REPEATS.search(text):
        return Routed("chore")
    if _GREETING.match(text):
        return Routed("answer")
    if (
        len(text) <= SHORT_CHARS
        and is_question(text)
        and not _NAMES_THING.search(text)
        and not _ACTION_VERBS.search(text)
    ):
        return Routed("answer")
    return Routed("plan")


__all__ = [
    "LEVEL_WORDS",
    "Routed",
    "TaskRef",
    "allows",
    "allows_level",
    "control_action",
    "is_question",
    "level_number",
    "level_word",
    "named_task",
    "route",
    "triage",
]
