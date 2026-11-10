"""A chat's rules in code: the explicit words read before any model call.

Pure functions over the text the person typed and the open tasks:

- :func:`route` picks what a chat does with a message: ``delegate``
  (``/task …`` or the composer's Run as a task), ``control`` (a control word
  and a task the person named), ``follow`` (a follow-up word and a task the
  person named, finished ones too), ``continue`` ("continue the recovery card
  task": the tasks whose titles the words name, stopped ones continue),
  ``chore`` (a repeating phrase, left to the
  scheduled chores), ``reminder`` (a reminder cue: "remind me to … on Friday at
  9", "提醒我 …"; a one-time chore, or a question about when) or ``plan`` (the
  chat's own run: the model reads the message and does the work, #571). No list
  of words decides whether a chat acts.
- :func:`triage` gives a new task its level from the person's own words:
  P1 for urgency words, a deadline today or the bare word "today"
  (``今天``) anywhere, P3 for "no rush", P2 otherwise.
- :func:`allows` is the principal rule for a chat's control tools: a
  change counts as the person's only when their own typed message has the
  control word and names the task. Page text, tool output and a task's
  report are never that message.
- :func:`answered_task` picks the one task a plain reply answers without
  naming it: the one that waits for the person and asked last (#617);
  :func:`plain_yes` reads a plain yes (:data:`YES`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

from localmelo.melo.contracts.review import WORDS_CAN_ALLOW
from localmelo.support.gateway.reminder_time import parse_reminder
from localmelo.support.sessions.display import EASY_WORDS, GREETING, URGENT_WORDS

# Levels P0 to P5. People read the P-code, with its meaning where there is room
# (``P1 · today``). The words stay what people may type and what the API's
# ``level_word`` answers; the meanings are the one table to translate.
LEVEL_WORDS: Final = ("Now", "Today", "Normal", "Whenever", "If idle", "Later")
LEVEL_MEANINGS: Final = (
    "runs first; others wait",
    "today",
    "normal",
    "when there is time",
    "only when the computer is idle",
    "on hold until you start it",
)
FINISHED: Final = frozenset({"done", "failed", "cancelled", "expired"})

Route = Literal[
    "delegate",
    "control",
    "follow",
    "continue",
    "choose",
    "chore",
    "reminder",
    "plan",
]
Action = Literal["stop", "pause", "resume", "first"]


def _index(level: object) -> int:
    """0-5 for a level; 2 (normal) for anything that is not one."""
    if isinstance(level, int) and not isinstance(level, bool) and 0 <= level <= 5:
        return level
    return 2


def level_word(level: object) -> str:
    """``Normal`` for 2; ``Normal`` for anything that is not a level."""
    return LEVEL_WORDS[_index(level)]


def level_code(level: object) -> str:
    """``P2`` for 2; ``P2`` for anything that is not a level."""
    return f"P{_index(level)}"


def level_label(level: object) -> str:
    """The code and its meaning: ``P1 · today``."""
    return f"{level_code(level)} · {LEVEL_MEANINGS[_index(level)]}"


def level_menu() -> str:
    """Every level in one line, for a model's tool description."""
    return ", ".join(f"{level_code(n)} ({LEVEL_MEANINGS[n]})" for n in range(6))


def level_number(value: object) -> int | None:
    """0-5 from a P-code (``P1``, ``p1``, ``1``) or a level word (``today``, ``If idle``); else None."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 0 <= value <= 5 else None
    text = " ".join(str(value).split()).casefold()
    if re.fullmatch(r"p?[0-5]", text):
        return int(text[-1])
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
    matches: tuple[TaskRef, ...] = ()  # the tasks "continue the … task" names
    pick: str = ""  # "choose": the one-tap suggestion to offer for each match


def norm(text: str) -> str:
    """NFKC, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _has_cjk(text: str) -> bool:
    return any("\u3400" <= ch <= "\u9fff" for ch in text)


def is_chinese(text: str) -> bool:
    """Whether a sentence code writes for *text* is Chinese: it has Chinese characters."""
    return _has_cjk(text)


def reply_language(text: str) -> str:
    """The language the main chat answers *text* in, by its script."""
    return "Chinese" if is_chinese(text) else "English"


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


def named_tasks(
    typed: str,
    phrase: str,
    tasks: Sequence[TaskRef],
    *,
    strict: bool = True,
    finished: bool = False,
) -> list[TaskRef]:
    """Every open task *typed* names, besides its control *phrase*: one when it is clear, more when it is not.

    A task whose id is written is the one. Strict (code's own route): what
    is left names the tasks whose title holds it (3 characters at least, 2
    for Chinese; the exact ones first). Loose (a tool's principal check):
    when that does not single one out, the tasks with a word of the title,
    of 4 letters at least, written in *typed* do. With *finished*, a
    finished task can be named too.
    """
    text = norm(typed)
    rest = _rest(text, phrase)
    open_tasks = [task for task in tasks if finished or task.state not in FINISHED]
    for task in open_tasks:
        if task.task_id and re.search(rf"\b{re.escape(task.task_id)}\b", text):
            return [task]
    held: list[TaskRef] = []
    if rest and len(rest) >= (2 if _has_cjk(rest) else 3):
        # A title is read as the rest is: without filler ("Fix the e2e check"
        # is named by "fix e2e check").
        titles = {
            task.task_id: (norm(task.title), _rest(norm(task.title), ""))
            for task in open_tasks
        }
        exact = [task for task in open_tasks if rest in titles[task.task_id]]
        found = exact or [
            task
            for task in open_tasks
            if any(rest in title for title in titles[task.task_id])
        ]
        if len(found) == 1:
            return found
        held = found
    if strict:
        return held
    words = set(re.findall(r"\w+", text))
    loose = [
        task
        for task in open_tasks
        if any(
            token in words or (_has_cjk(token) and token in text)
            for token in _title_tokens(task.title)
        )
    ]
    return loose or held


def named_task(
    typed: str,
    phrase: str,
    tasks: Sequence[TaskRef],
    *,
    strict: bool = True,
    finished: bool = False,
) -> TaskRef | None:
    """The one open task *typed* names, besides its control *phrase*, or None.

    See :func:`named_tasks`: ``None`` when it names none, and when it names
    more than one.
    """
    found = named_tasks(typed, phrase, tasks, strict=strict, finished=finished)
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


# Words that carry on with a task: a follow-up, not a new piece of work.
_FOLLOW: Final = _words(
    "carry on|keep going|try again|go on|finish|continue|commit|push|redo|retry|"
    "继续|接着|完成|做完|提交|重试|再试"
)


def follow_cue(typed: str) -> str | None:
    """The first follow-up word in *typed* not under a negation, or None."""
    text = norm(typed)
    for phrase in _FOLLOW:
        found = _find(phrase, text)
        if found is not None and not _NEGATION.search(text[: found.start()]):
            return phrase
    return None


def plain_reply(typed: str) -> bool:
    """Whether *typed* reads as a reply: not a command, greeting, question, work request or control phrase."""
    text = typed.strip()
    return (
        bool(text)
        and not text.startswith("/")
        and not GREETING.match(norm(text))
        and not is_question(text)
        and not work_request(text)
        and control_action(text) is None
    )


# A plain yes: the whole message, once normalised, is one of these.
YES: Final = frozenset(
    _words("允许|可以|好|好的|同意|行|yes|ok|okay|allow|go ahead|do it")
)
# What may stand around a yes: spaces and a full stop or an exclamation mark.
_YES_MARKS: Final = " .!。！"


def plain_yes(typed: str) -> bool:
    """Whether *typed* is a plain yes: :data:`YES` exactly, with nothing else but :data:`_YES_MARKS`."""
    return norm(typed).strip(_YES_MARKS) in YES


@dataclass(frozen=True)
class Shown:
    """One thing a chat showed the person, as :func:`answered_task` reads it.

    ``task_id`` is the task it is about (``""`` for none); ``asks`` is true for
    that task's request to the person (the card of its end that needs the
    person, a step it had blocked, melo's follow-up on it); ``person`` for a
    message the person sent. For a card, ``block`` is the request id of the
    blocked step it offers to allow (its Allow this step), ``""`` when it
    offers none; ``None`` for anything that is not a card.
    """

    task_id: str = ""
    asks: bool = False
    person: bool = False
    block: str | None = None


def answered_task(
    typed: str,
    waiting: Sequence[TaskRef],
    takes: Collection[str],
    shown: Sequence[Shown],
    tasks: Sequence[TaskRef] = (),
) -> TaskRef | None:
    """The task *typed* (the person's new message) answers, or None.

    *waiting* are the tasks of the chat the stage table has waiting for the
    person, *takes* the ids of those a message continues, *shown* what the
    chat showed the person before this message, oldest first, and *tasks*
    every task the words could name. It answers one only when all hold:

    - *typed* is a plain yes (:func:`plain_yes`) or a plain reply
      (:func:`plain_reply`): never a command;
    - exactly one task waits for the person, and a message continues it;
    - the newest thing shown is that task's request, so neither another
      task's line nor a message of the person's came after it;
    - the words name no other task.

    Only the person's own typed words are *typed*; a page, a file or a task's
    report never is.
    """
    if not (plain_yes(typed) or plain_reply(typed)) or len(waiting) != 1:
        return None
    task = waiting[0]
    if task.task_id not in takes or not shown:
        return None
    newest = shown[-1]
    if newest.person or not newest.asks or newest.task_id != task.task_id:
        return None
    named = named_tasks(typed, "", tasks, strict=False, finished=True)
    if any(other.task_id != task.task_id for other in named):
        return None
    return task


def yes_step(
    typed: str, offered: str, blocks: Sequence[Mapping[str, object]]
) -> Mapping[str, object] | None:
    """The blocked step a plain yes allows once, or None: then *typed* is an ordinary answer.

    *offered* is the request id of the step the newest card of the task
    offers (``Shown.block``), and *blocks* the steps Auto blocked in it that
    Allow once may still run. It is one only when all hold: *typed* is a plain
    yes (:func:`plain_yes`); exactly one step is open; it is the step the card
    offers; and every kind of step its review named is one the person's words
    may allow (:data:`WORDS_CAN_ALLOW`), so a step that runs only after a
    click is never allowed by words.
    """
    if not plain_yes(typed) or not offered or len(blocks) != 1:
        return None
    block = blocks[0]
    review = block.get("review")
    kinds = review.get("categories") if isinstance(review, Mapping) else None
    if str(block.get("request_id") or "") != offered or not isinstance(kinds, list):
        return None
    return block if kinds and all(kind in WORDS_CAN_ALLOW for kind in kinds) else None


def allows_message(typed: str, task: TaskRef, *, newest: bool) -> bool:
    """The principal rule for a message to *task*: the person's own words point at it.

    *typed* names the task (its id, or a word of its title, finished tasks
    too), or has a follow-up word ("finish it", "继续那个任务") while *task*
    is the *newest* one the main chat showed. A page, a file or a task's
    report that names a task is never *typed*.
    """
    if not norm(typed):
        return False
    if named_task(typed, "", [task], strict=False, finished=True) is not None:
        return True
    return newest and follow_cue(typed) is not None


# "continue the recovery card task where it stopped": the words between the verb and
# "task" name the task. Latin phrases need "task" or "job"; Chinese needs 任务.
_CONTINUE_EN: Final = re.compile(
    r"^(?:(?:please|pls|kindly|ok|okay|so|then|just)[\s,]+)*"
    r"(?:(?:can|could|would) you (?:please )?)?"
    r"(?:continue|resume|carry on|keep going|go on|pick up)(?: with| on)?\s+"
    r"(?:(?:the|my|that|this|our)\s+)?(.+?)\s+(?:task|job)\b"
)
_CONTINUE_ZH: Final = re.compile(
    r"^(?:请|麻烦你?|帮我|你)?(?:继续|接着|恢复)(?:做|干|完成)?(?:一下)?"
    r"(?:那个|这个|上次的|刚才的|之前的)?(.+?)(?:那个|这个)?(?:的)?任务"
)


def continue_name(typed: str) -> str:
    """The words that name a task in "continue the recovery card task …", or ``""``."""
    text = norm(typed)
    for pattern in (_CONTINUE_EN, _CONTINUE_ZH):
        found = pattern.match(text)
        if found is not None:
            return found.group(1).strip(" ,:;，：；")
    return ""


def title_matches(name: str, task: TaskRef) -> bool:
    """Whether *name* (the words that name a task) is in the task's title or is its id."""
    wanted, title = _rest(norm(name), ""), norm(task.title)
    if not wanted:
        return False
    if task.task_id and norm(task.task_id) == wanted:
        return True
    words = wanted.split()
    return wanted in title or (len(words) > 1 and all(word in title for word in words))


# ── Delegation and its level ──

_DEADLINE: Final = re.compile(
    r"\b(?:by|before|due)\s+(?:today|tonight|noon|eod|\d{1,2}(?::\d{2}\s*(?:am|pm)?|\s*(?:am|pm)))\b"
    r"|今[天晚][^。.!！?？]{0,8}前|\d{1,2}\s*点前"
)

# The bare word for today, anywhere in the request, is a deadline of itself. A
# possessive ("today's news") only names a thing about today, so the Latin form
# asks for nothing after the word; Chinese has no such form.
_TODAY: Final = re.compile(r"(?<![\w])today(?![\w'’])")
_TODAY_ZH: Final = re.compile("今天")
# Chinese has no possessive, and its negation often follows the word instead of
# standing before it.
_NOT_AFTER_ZH: Final = re.compile(
    r"^(?:先|暂|暂时)?(?:不(?:用|要|必|需要|想|急|做|赶)|别|没)"
)
_NOT_BEFORE_ZH: Final = re.compile(r"(?:不是|并非|不|别|没|无)$")


def _says_today(text: str) -> str:
    """The bare ``today`` (``今天``) *text* says without negating it; else ``""``.

    A negation before the word negates it in both languages ("not today",
    "不是今天"); in Chinese one right after it does too ("今天不用"). A negated
    word is not the deadline the person asked for.
    """
    for word, pattern in (("today", _TODAY), ("今天", _TODAY_ZH)):
        for found in pattern.finditer(text):
            before = text[: found.start()]
            after = text[found.end() :]
            negated = bool(_NEGATION.search(before))
            if word == "今天":
                negated = negated or bool(_NOT_BEFORE_ZH.search(before))
                negated = negated or bool(_NOT_AFTER_ZH.match(after))
            if not negated:
                return word
    return ""


def triage(typed: str) -> tuple[int, str]:
    """A delegated task's level and reason, from the person's own words only.

    Ease words first, so "not urgent" is never urgency; then urgency words, a
    deadline today or the bare word for today (P1); else P2 (design 2.5;
    S5's full table later).
    """
    text = norm(typed)
    for phrase in EASY_WORDS:
        if _find(phrase, text):
            return 3, f'you said "{phrase}"'
    for phrase in URGENT_WORDS:
        found = _find(phrase, text)
        if found and not _NEGATION.search(text[: found.start()]):
            return 1, f'you said "{phrase}"'
    deadline = _DEADLINE.search(text)
    if deadline:
        return 1, f'you said "{deadline.group(0).strip()}"'
    word = _says_today(text)
    if word:
        return 1, f'you said "{word}"'
    return 2, "you asked for it"


_QUESTION_START: Final = re.compile(
    r"^(?:what|why|how|when|where|who|whom|which|whose|is|are|am|was|were|can|could|"
    r"do|does|did|should|would|will|have|has|may)\b"
)
_QUESTION_ZH: Final = re.compile(r"(?:吗|呢|么|什么|怎么|为什么|哪|谁|几|是否|多少)")
_QUICK: Final = re.compile(r"\bquick question\b|快问|请问|问一下")
_URL: Final = re.compile(r"https?://\S+|www\.\S+")


def is_question(typed: str) -> bool:
    """A question: a question mark anywhere (not in a link), "quick question", or a question's start."""
    text = norm(typed).rstrip()
    bare = _URL.sub(" ", text)
    return (
        "?" in bare
        or "？" in bare
        or bool(_QUICK.search(text))
        or bool(_QUESTION_START.match(text))
        or bool(_QUESTION_ZH.search(text[-6:]))
    )


# The rules read a long text by its opening: the person's intent is in the first
# few thousand characters, and a long paste behind it is for the task to read.
OPENING_CHARS: Final = 4_000


# ── Work named in the words: not an answer to a waiting task ──

_EXTENSIONS: Final = (
    "py|pyi|js|mjs|cjs|ts|tsx|jsx|md|rst|txt|json|toml|ya?ml|html?|css|scss|sh|zsh|"
    "rs|go|java|kt|swift|c|cc|cpp|h|hpp|rb|php|sql|cfg|ini|csv|ipynb|xml|lock"
)
# A part of a name or a path is read up to 200 characters, so a scan that starts
# at each character of one long token stays linear in the text.
_FILE_NAME: Final = re.compile(rf"\b[\w.-]{{0,200}}\w\.(?:{_EXTENSIONS})\b")
_PATH: Final = re.compile(
    r"(?:^|[\s`'\"(（：:，,])(?:~/|\.\.?/|/)[\w.-]"  # ~/x, ./x, ../x, /x
    r"|[\w.-]{1,200}/[\w.-]{1,200}/[\w./-]{0,200}"  # a/b/c
    r"|\b(?=[\w.-]{0,200}[_.])[\w.-]{1,200}/[\w.-]{1,200}"  # one slash, a _ or .
    r"|\b[\w.-]{1,200}/(?=[\w.-]{0,200}[_.])[\w.-]{1,200}"
    r"|\b[\w.-]{1,200}/(?:\s|$)"  # a folder: name/
)
# Verbs that change what is there; making something new ("make slides from
# notes.md") is left to the front's step.
_CHANGE_EN: Final = (
    "fix|change|edit|update|refactor|rename|replace|remove|delete|add|move|rewrite|"
    "implement|commit|patch|correct|format|migrate|bump|revert"
)
_LEAD_EN: Final = re.compile(
    r"^(?:(?:please|pls|kindly|now|also|and|then|ok|okay|so|just|quickly|go ahead and)[\s,]+)*"
    rf"(?:{_CHANGE_EN})\b"
)
_CHANGE_ZH: Final = re.compile(
    r"修复|修改|修一下|修好|改成|改为|改掉|改一下|换成|换掉|换一下|提交|删除|删掉|添加|加上|加到|"
    r"放到|移到|移动|重命名|替换|实现|重构|更新"
)
_LEAD_ZH: Final = re.compile(r"^(?:请|麻烦你?|帮我|帮忙|给我|你)?(?:把|将|在)?")
_LOCATION: Final = re.compile(r"^(?:in|for|at|on)\s+\S+\s*[,:;]\s*")
# A whole pipeline in one breath: change something, test it, commit it. The
# commit verb is left out of the change list, so "跑测试，过了就提交" (nothing
# to change) is not a request, and a bare 改 or 修 is not either ("改得更严格"
# changes how something reads, not code).
_EDIT_ZH: Final = re.compile(
    r"修复|修改|修一下|修好|改成|改为|改掉|改一下|换成|换掉|换一下|删除|删掉|添加|加上|加到|"
    r"放到|移到|移动|重命名|替换|实现|重构|更新"
)
_EDIT_EN: Final = re.compile(
    r"\b(?:fix|change|edit|update|refactor|rename|replace|remove|delete|add|move|rewrite|"
    r"implement|patch|correct|format|migrate|bump|revert)\w*\b",
    re.IGNORECASE,
)
_TEST_CUE: Final = re.compile(r"测试|运行测试|tests?\b|pytest", re.IGNORECASE)
_COMMIT_CUE: Final = re.compile(r"提交|push|commits?\b", re.IGNORECASE)
# The ask negated is not an ask: "今天不用改，跑测试再提交".
_PIPELINE_NO: Final = re.compile(
    r"(?:不用|不需要|无需|不必|不要|先不|暂时不|别|勿|禁止|不许|不得|不准|请勿|切勿)\s*(?:再|去|用)?\s*(?:改|修|测|提交|动手)"
    r"|\b(?:don['’]t|do\s+not|no\s+need\s+to|never|not\s+worth)\s+(?:even\s+)?(?:re-?)?"
    r"(?:fix|change|edit|update|refactor|rename|replace|remove|delete|add|test|commit|push)\w*",
    re.IGNORECASE,
)


# What is said in quotation marks is mentioned, not asked: "the log says “删除 a.md”".
# An ASCII single quote opens a span only after a non-letter and closes before one, so the
# apostrophes of "don't", "it's" and "users'" open nothing.
_QUOTED: Final = re.compile(
    r'"[^"]*"|“[^”]*”|`[^`]*`|「[^」]*」|『[^』]*』|«[^»]*»|‘[^’]*’(?=\W|$)'
    r"|(?<![A-Za-z0-9])'[^'\n]*'(?![A-Za-z0-9])"
)
# A prohibition before a Chinese change verb in the same clause: do not, stop, no, never.
_NO_ZH: Final = re.compile(
    r"(?<![分区特告辨差])别|不要|不用|不必|不需要|无需|勿|禁止|不许|不得|不准|不能|不可|不想|"
    r"先不|暂不|暂时不|不再|没有|没"
)
_CLAUSE_ZH: Final = re.compile(r"[，,、；;：:]")


def _forbidden_zh(first: str, at: int) -> bool:
    """Whether a prohibition governs the change verb at *at* in *first*.

    One in its own clause does. So does one earlier in an enumeration it
    belongs to: after 、 whatever follows, and after a comma a clause that
    opens with a verb when the clause before it holds one
    ("不要删除 a.md、修改 b.md", "不要删除 a.md，修改 b.md"); a clause with words of
    its own ("，请修复 b.md") or a ；/： ends it.
    """
    marks = [(found.start(), found.group()) for found in _CLAUSE_ZH.finditer(first)]
    end = at
    while True:
        before = [mark for mark in marks if mark[0] < end]
        start = before[-1][0] + 1 if before else 0
        clause = first[start:end]
        if _NO_ZH.search(clause):
            return True
        if not before:
            return False
        where, mark = before[-1]
        earlier = [m for m in marks if m[0] < where]
        previous = first[(earlier[-1][0] + 1 if earlier else 0) : where]
        if mark == "、" or (
            mark in "，," and not clause.strip() and _CHANGE_ZH.search(previous)
        ):
            end = where
            continue
        return False


def _asked_zh(first: str) -> bool:
    """Whether a change verb of *first* (one Chinese sentence) is asked for: one no prohibition governs."""
    return any(
        not _forbidden_zh(first, verb.start()) for verb in _CHANGE_ZH.finditer(first)
    )


def names_files(text: str) -> bool:
    """Whether *text* names a file or a path (a link is neither)."""
    bare = _URL.sub(" ", text)
    return bool(_FILE_NAME.search(bare)) or any(
        re.search(r"[^\W\d_]", found.group(0)) for found in _PATH.finditer(bare)
    )


def _pipeline(text: str) -> bool:
    """Whether *text* spells out the whole ask: change the code, test it, commit.

    "今天修一下：路由规则改掉，跑测试，过了就提交", "fix the router, run the
    tests and commit if they pass": a verb that changes something, a test cue
    and a commit cue, not said in a question and not under a negation. No file
    is named here, so the wording itself has to carry the request.
    """
    if _PIPELINE_NO.search(text):
        return False
    edit = _EDIT_ZH if _has_cjk(text) else _EDIT_EN
    return bool(
        edit.search(text) and _TEST_CUE.search(text) and _COMMIT_CUE.search(text)
    )


# What asks for a commit beyond the word: a push, a pull request, checking it in.
_ASKS_COMMIT: Final = re.compile(
    r"提交|推送|\bpush\b|\bcommits?\b|\bpull request\b|\bPR\b|\bcheck(?:ed)? (?:\w+ )?in\b",
    re.IGNORECASE,
)
_NO_PUSH: Final = re.compile(
    r"(?:不要|不用|别|先不)\s*推送|\b(?:don['’]t|do\s+not|never)\s+push\w*",
    re.IGNORECASE,
)


def asks_commit(text: str) -> bool:
    """Whether *text* asks for a commit, once quoted and forbidden spans are dropped.

    "fix the router, run the tests and commit", "push it", "open a PR",
    "check it in" and "don't push, just commit" ask; "don't commit yet",
    "先不要提交" and "the log says `git commit` failed" do not.
    """
    bare = _NO_PUSH.sub(" ", _PIPELINE_NO.sub(" ", _QUOTED.sub(" ", text or "")))
    return bool(_ASKS_COMMIT.search(bare))


# What asks for the tests to be run, beyond naming them: run, re-run, pytest, 跑测试.
_ASKS_TESTS: Final = re.compile(
    r"\b(?:run|re-?run|execute)\s+(?:\w+\s+){0,4}?(?:unit\s+)?tests?\b|\bpytest\b"
    r"|\bunit\s?tests?\b|(?:跑|运行|执行)[^，。；,;！？\n]{0,12}?测试|单元测试|单测"
    r"|\b(?:make\s+sure|ensure|check|verify|confirm)\s+(?:that\s+)?(?:all\s+)?(?:of\s+)?"
    r"(?:the\s+)?(?:unit\s+)?tests?\s+(?:all\s+)?(?:still\s+)?pass"
    r"|(?:确保|确认|保证)[^，。；,;！？\n]{0,6}?测试[^，。；,;！？\n]{0,4}?(?:通过|能过|都过)",
    re.IGNORECASE,
)
# A negated negation is an ask: "don't forget to run the tests", "make sure you
# don't skip the tests", "别忘了跑测试". It reads as "run" / "跑" before the
# negations go; in Chinese only before 跑, 运行 or 执行 ("别忘了测试一下这个网站"
# is trying the site out).
_KEEP_TESTS: Final = re.compile(
    r"\b(?:don['’]t|do\s+not|never)\s+forget\s+(?:to\s+)?(?:run\s+)?"
    r"|\b(?:make\s+sure|be\s+sure|ensure)\s+(?:that\s+)?(?:you\s+)?"
    r"(?:don['’]t|do\s+not|never)\s+(?:skip|forget)(?:ping)?\s+(?:to\s+)?(?:run\s+)?",
    re.IGNORECASE,
)
_KEEP_TESTS_ZH: Final = re.compile(
    r"(?:别|不要|不能|千万别)忘(?:了|记)?(?:要)?(?=跑|运行|执行)"
)
# The ask negated: "don't run the tests", "skip the tests", "不用跑测试".
_NO_TESTS: Final = re.compile(
    r"\b(?:don['’]t|do\s+not|no\s+need\s+to|never|without|skip(?:ping)?)\s+"
    r"(?:\w+\s+){0,2}?(?:run|re-?run|execut|test|pytest)\w*[^.;,!?\n]*"
    r"|(?:不用|不需要|无需|不必|不要|先不|别|跳过)\s*(?:再)?\s*(?:跑|运行|执行)?[^，。；,;！？\n]{0,6}?测试",
    re.IGNORECASE,
)


def asks_tests(text: str) -> bool:
    """Whether *text* asks for the tests to be run, once quoted and forbidden spans are dropped.

    "fix the parser and run the tests", "Finish it: run the tests that cover
    your changes", "改完跑一下测试", "make sure the tests pass", "don't forget
    to run the tests" and "别忘了跑测试" ask; "don't run the tests", "不用测试",
    "测试一下这个网站" (try the site out) and "the log says `pytest` failed" do
    not.
    """
    bare = _PIPELINE_NO.sub(" ", _QUOTED.sub(" ", text or ""))
    bare = _KEEP_TESTS_ZH.sub("", _KEEP_TESTS.sub(" run ", bare))
    return bool(_ASKS_TESTS.search(_NO_TESTS.sub(" ", bare)))


def work_request(typed: str) -> bool:
    """An imperative change to named files or paths: new work, never a waiting task's answer.

    "Please fix this today: in tests/cli/test_e2e_pty.py …", "把 docs/a.md 里的图换掉":
    the message starts with a change verb (after "please" and the like, or
    after a leading "in <path>:"), names a file or a path, and is not a
    question. With no file named, the same happens when the message asks for
    the whole pipeline at once (`_pipeline`: change it, run the tests, commit).
    Only the first :data:`OPENING_CHARS` characters are read.
    """
    typed = typed.lstrip()[:OPENING_CHARS]
    text = norm(typed)
    if not text or is_question(typed):
        return False
    # The verb counts only outside quotation marks and when nothing forbids it:
    # "不要删除 a.md，只解释它", "never delete a.md", "the log says “删除 a.md”".
    bare = _QUOTED.sub(" ", text).strip()
    if names_files(text):
        if _has_cjk(bare[:24]):
            lead = bare[_LEAD_ZH.match(bare).end() :]  # type: ignore[union-attr]
            first = re.split(r"[。！？；\n]", lead, maxsplit=1)[0]
            return _asked_zh(first)
        return bool(
            _LEAD_EN.match(bare) or _LEAD_EN.match(_LOCATION.sub("", bare, count=1))
        )
    return _pipeline(bare)


_REPEATS: Final = re.compile(
    r"\bevery\s+(?:day|weekday|morning|evening|night|week|month|hour|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|\d+\s+(?:minutes?|hours?|days?))\b"
    r"|\b(?:daily|weekly|monthly|hourly)\b|每天|每周|每晚|每早|每个工作日|每小时|每月|每隔"
)
# Any fixed moment: whether a message is a reminder does not depend on the time.
_ANY_TIME: Final = datetime(2026, 1, 5, 12, 0)
# A message longer than this is never read as a control.
CONTROL_CHARS: Final = 80


def _pool(
    typed: str, phrase: str, tasks: Sequence[TaskRef], fallback: Sequence[TaskRef]
) -> list[TaskRef]:
    """*tasks*, and *fallback* too only when the words name none of *tasks*.

    A chat's own tasks come first: words that name one of them, clearly or
    not, finished or not, never reach another chat's task.
    """
    if fallback and not named_tasks(typed, phrase, tasks, finished=True):
        return [*tasks, *fallback]
    return list(tasks)


def route(
    typed: str,
    tasks: Sequence[TaskRef] = (),
    *,
    as_task: bool = False,
    fallback: Sequence[TaskRef] = (),
) -> Routed:
    """What a chat does with *typed*, by code's rules (design 2.4).

    *tasks* are the chat's own tasks; *fallback* are the other chats' ones,
    which the words reach only when they name none of the chat's own. Words
    that name more than one task are a ``choose`` route: the chat asks which.
    The rules read the first :data:`OPENING_CHARS` characters; a ``delegate``
    route carries all of the text.
    """
    whole = typed.strip()
    command = re.match(r"^/task(?:\s+|$)", whole)
    if as_task or command:
        work = whole[command.end() :] if command else whole
        return Routed("delegate", text=work.strip())
    stripped = whole[:OPENING_CHARS]
    text = norm(stripped)
    if len(text) <= CONTROL_CHARS:
        found = control_action(stripped)
        if found is not None:
            pool = _pool(stripped, found[1], tasks, fallback)
            named = named_tasks(stripped, found[1], pool)
            if len(named) == 1:
                return Routed("control", action=found[0], task=named[0])
            if named:
                return Routed("choose", pick=found[0], matches=tuple(named))
    if _REPEATS.search(text):
        return Routed("chore")
    if parse_reminder(stripped, _ANY_TIME) is not None:
        return Routed("reminder")
    name = continue_name(stripped)
    if name and not is_question(stripped):
        mine = tuple(task for task in tasks if title_matches(name, task))
        found_tasks = mine or tuple(
            task for task in fallback if title_matches(name, task)
        )
        if found_tasks:
            one = found_tasks[0] if len(found_tasks) == 1 else None
            return Routed("continue", task=one, matches=found_tasks)
    cue = follow_cue(stripped)
    if len(text) <= CONTROL_CHARS and cue is not None and not is_question(stripped):
        pool = _pool(stripped, cue, tasks, fallback)
        named = named_tasks(stripped, cue, pool, finished=True)
        if len(named) == 1:
            return Routed("follow", task=named[0])
        if named:
            return Routed("choose", pick="send", matches=tuple(named))
    return Routed("plan")


__all__ = [
    "LEVEL_MEANINGS",
    "LEVEL_WORDS",
    "Routed",
    "Shown",
    "TaskRef",
    "YES",
    "allows",
    "allows_level",
    "allows_message",
    "answered_task",
    "asks_commit",
    "continue_name",
    "control_action",
    "follow_cue",
    "is_chinese",
    "is_question",
    "level_code",
    "level_label",
    "level_menu",
    "level_number",
    "level_word",
    "named_task",
    "named_tasks",
    "names_files",
    "plain_reply",
    "plain_yes",
    "reply_language",
    "route",
    "title_matches",
    "triage",
    "work_request",
    "yes_step",
]
