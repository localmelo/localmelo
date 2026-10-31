"""Records: what the user said about themselves or their work, one sentence each.

A record is written two ways, and neither asks the user to confirm. When the
user asks ("remember that…", "记住…", "别忘了…", "帮我记一下…"), code keeps the
fact the request carries (:func:`candidates`, the signal lexicon of
:mod:`localmelo.melo.memory.hippo.signals`), in the second person ("You
prefer short answers."). Otherwise the chat's model keeps a note with the
``remember`` tool when it judges something worth keeping; the note's content
words must come from what the user typed in that chat or from a finished
task's result shown there (:func:`unsupported`), never from a page or tool
output alone. With the developer setting ``[memory] keep_statements`` (off
by default, for comparison), code also keeps the clear cases of that
judgement: after the turn, a plain first-person statement of a lasting fact
that names a subject ("I live in …", "my advisor is …", "call me …",
"我的…是…") and that no note of the turn kept becomes one (:func:`statements`);
a question, a to-do, a hypothetical, a negation or someone else's opinion
never does. Whatever that setting says, when the turn kept no note, code also
keeps a plain statement that gives a new value to a subject an active note holds
(:func:`changes`: "My weekly sync is on Friday." after "Your weekly sync is on
Wednesday."). Each record has at most :data:`MAX_RECORD_CHARS` characters, its topic, the user's quote
and where it was said. Text that looks like a secret is refused, not stored
redacted. A new record of the same subject supersedes the old one ("call me
Ray" after "call me R.", "我的健身教练是小宇" after "…是阿凯"): the old one is
kept, not recalled, and comes back with an undo of the new one. The subject is
what a note is about without its value (:func:`subject`); with an embedding
model, a note close in meaning that differs from an old one in one short span
is the same subject too (:func:`value_apart`). A third party's fact names its
owner as well, so "Tom Briggs's office is in room 9C." never replaces "Your
manager is Tom Briggs.", whatever words they share. A new record lists the
turns whose facts it replaced (``provenance["replaces"]``), so recall can rank them
below it. Chat turns follow the same rule among themselves: of two turns whose
typed words state one subject with different values, the older is replaced by
the newer (:func:`newer_turns`), with or without a note.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Final

import numpy as np

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.memory.bm25 import tokenize
from localmelo.melo.memory.hippo.policy import (
    STOPWORDS,
    content_tokens,
    normalise,
    overlap,
)
from localmelo.melo.memory.hippo.signals import classify_user_text
from localmelo.melo.memory.long import StoredEntry
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.memory.notes import split_sentences
from localmelo.melo.schema import new_record_id
from localmelo.support.memory.recall import Item, stem, unit

TOPICS: Final = ("about", "preferences", "people", "projects", "decisions")
TOPIC_LABELS: Final = {
    "about": "About you",
    "preferences": "Preferences",
    "people": "People",
    "projects": "Projects",
    "decisions": "Decisions",
}
# The long-term kind each topic is stored as.
TOPIC_KINDS: Final = {
    "about": "profile",
    "preferences": "profile",
    "people": "semantic",
    "projects": "episodic",
    "decisions": "semantic",
}
MAX_RECORD_CHARS: Final = 300
# A message longer than this is a task or a paste, not a statement to keep.
MAX_TYPED_CHARS: Final = 2_000
# Content overlap at which a new record replaces an old one.
SUPERSEDE_OVERLAP: Final = 0.6
# With vectors: how close in meaning, and how long the one changed span (the
# value) may be, for a note to replace an old one of other subject words.
NEAR_COSINE: Final = 0.8
VALUE_TOKENS: Final = 8
NOT_KEPT: Final = "it looked like a password or key"
# What a list of notes says when it has none, and how memory starts.
EMPTY: Final = (
    "Nothing remembered yet. melo keeps a note when you tell it something about "
    "yourself in the web console or the app, such as “I prefer short answers”, "
    "or ask it to remember something."
)

_DECISION = re.compile(
    r"\b(we(?:'ve| have)? decided|let's go with|we(?:'ll| will) go with)\b"
    r"|我们决定|决定用|就用",
    re.IGNORECASE,
)
_PREFERENCE = re.compile(
    r"\b(prefer|like|love|hate|dislike|from now on)\b|喜欢|习惯|偏好|讨厌|以后",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(r"\b(I|I'm|my|me|mine|we|our|us)\b|我", re.IGNORECASE)
# Subjects a verb names, in the user's words, a record's or the model's
# ("I live in", "You live in", "The user lives in", "我住在", "用户住在").
_WHO = r"\b(?:i|you|user)(?: now| currently)?"
_SUBJECTS: Final = tuple(
    (key, re.compile(pattern))
    for key, pattern in (
        ("timezone", r"\btime ?zone\b|时区"),
        (
            "name",
            r"^(?:please )?call me\b|\bgo(?:es)? by [^ ]+(?: [^ ]+)?$"
            rf"|{_WHO}(?:'m| am|'re| are| is) (?:called|named)\b"
            r"|^(?:请)?叫我|希望被(?:叫作|称为)|(?:我|你|用户)的?名字",
        ),
        (
            "home",
            rf"{_WHO} (?:live|lives|reside|resides) (?:in|at|on|near)\b"
            r"|(?:我|你|用户)(?:现在)?住在"
            rf"|\b(?:i|you|we|user)(?:'ve| have)? moved to\b|(?:我|你|用户)搬(?:到|去)",
        ),
        (
            "job",
            rf"{_WHO} works? (?:at|for)\b"
            r"|(?:我|你|用户)(?:现在)?在[^，,。\s]{1,20}?(?:工作|上班)",
        ),
    )
)
# The subjects a verb names: a statement of one needs no lexicon marker.
_VERB_SUBJECTS: Final = frozenset(key for key, _ in _SUBJECTS)
# A verb key's value is a "unit" (a floor, a room, a number) or a "place"/"name";
# a unit never replaces a place ("Utrecht" and "3楼" are different things).
_UNIT = re.compile(
    r"\d|[楼层室号栋]|单元"
    r"|\b(?:floors?|apts?|apartments?|rooms?|units?|suites?|buildings?)\b"
)
# "my X is Y" / "the user's X is Y" / "我的X是Y": X is the subject, and how Y
# relates to it (:func:`_relation`) tells a name from a day or a description.
_ATTRIBUTE = r"((?:[\w'’-]+ ){0,5}?[\w'’-]+) (?:is|are|was|were)\b ?(.*)$"
_OWNED = re.compile(rf"^(?:my|your|our|(?:the )?user['’]s) {_ATTRIBUTE}")
_OWNED_ZH = re.compile(
    r"^(?:我|你|用户)的?([^\s，,。是叫姓]{1,12}?)(?:现在|也)?([是叫姓].*)$"
)
# "You have a weekly sync on Wednesdays" says what "Your weekly sync is on
# Wednesday" says: X is the subject and the value relates to it as "on".
_HAVE = re.compile(
    r"^(?:i|you|we|(?:the )?user)(?: now| currently)? (?:have|has|had) "
    r"(?:(?:a|an|my|your|our) )?((?:[\w'’-]+ ){0,4}?[\w'’-]+) "
    r"(?:on|at|in|every)\b"
)
# What follows a third party's "Name's" / "NAME的": "office is in room 9C",
# "工作室在中山路88号" (a place is "在", which the user's own form leaves alone).
_THIRD = re.compile(rf"^{_ATTRIBUTE}")
_THIRD_ZH = re.compile(r"^([^\s，,。是叫姓在]{1,12}?)(?:现在|也)?([是叫姓在].*)$")
_RELATIONS: Final = frozenset({"called", "named", "on", "at", "in", "from"})
_DESCRIBES: Final = frozenset(
    {"a", "an", "very", "so", "really", "quite", "too", "not", "always", "never"}
)
_STILL: Final = frozenset({"now", "still", "currently", "also"})
_ARTICLES: Final = frozenset({"a", "an", "the"})
_TOKEN = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
# Request phrases the shared lexicon lacks, read as "remember that":
# "帮我记一下：…", "keep in mind that …", "make a note that …".
_ALSO_ASKS = re.compile(
    r"^(?:please\s+)?(?:keep in mind|make a note)(?:\s+that)?\b[\s,:]*"
    r"|^(?:请)?(?:帮我)?记一下[\s,:：，]*",
    re.IGNORECASE,
)
# Words a note adds when it restates what was said ("The user prefers…",
# "You said: …", "You go by…") and function words a paraphrase swaps ("then",
# "each", "use"); they are never what has to be found in the user's words.
_FRAMING: Final = frozenset(
    [
        "user",
        "users",
        "prefer",
        "prefers",
        "like",
        "likes",
        "love",
        "loves",
        "want",
        "wants",
        "go",
        "goes",
        "call",
        "called",
        "name",
        "named",
        "remember",
        "note",
        "said",
        "say",
        "says",
        "told",
        "tell",
        "tells",
        "then",
        "each",
        "every",
        "use",
        "also",
        "always",
    ]
)
# Terms that are values: a change in one is a different fact. A word is a value
# when it has a digit, is a number word or a weekday, is capitalised inside a
# sentence or sits in a quoted span; every CJK character or pair is one too.
_VALUE_WORD = re.compile(
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty"
    r"|forty|fifty|hundred|thousand|(?:mon|tues|wednes|thurs|fri|satur|sun)day"
)
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”|「([^」]+)」")
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|\n")
# What share of a note's other content words the evidence must hold.
WORDS_FOUND: Final = 0.8
_CJK_FILLER: Final = frozenset("的了是在我你他她它们和与就也都吗呢吧啊么")
_CJK_FRAMING = re.compile("希望被叫作|希望被称为|用户")
_POSSESSIVE = re.compile(r"(?<=\w)['’]s\b")
_WORD = re.compile(r"[a-z0-9_]+")
_CASED = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
# The never-kept classes: passwords and codes, card and ID numbers. A lower-case
# "pin" is the verb ("Pin the versions") unless a code or digits follow it.
_SECRET_WORDS = re.compile(
    r"\b(passwords?|passcodes?|passwd|cvv|security code|one-time code|otp"
    r"|(?-i:PINs?)|pin\s?(?:codes?|numbers?)|pin(?=\s*(?:is\s*|[:=]\s*)?\d))\b"
    r"|密码|口令|验证码",
    re.IGNORECASE,
)
_ID_NUMBER = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4}|\d{17}[\dXx])(?![\dA-Za-z])")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

# A plain statement: the user's own words about themselves, at most this long
# once a lead-in ("By the way, …", "对了，…") is set aside.
STATEMENT_CHARS: Final = 200
_LEAD_IN = re.compile(
    r"^(?:(?:btw|by the way|fyi|also|actually|oh|ok|okay|so|and|well|hey|hi|"
    r"now|these days|currently|update|news|for the record|just so you know|"
    r"heads up)\b[\s,:;!.—–-]*)+"
    r"|^(?:对了|另外|顺便说一下|顺便|其实|告诉你|跟你说|现在)[\s，,：:]*",
    re.IGNORECASE,
)
# Where a request or a question joins a statement: "I live in Lisbon, can you
# find me a flat?" states "I live in Lisbon".
_TAIL = re.compile(
    r"\s*[,，;；:：]\s*(?:(?:can|could|would|will|please|so|but|and|then|find|"
    r"help|tell|show|give|make|write|search|look|check|book|send|what|where|"
    r"when|who|how|why|which|any|do you|is there)\b|请|帮|能|可以|麻烦|然后|所以|但是|你)",
    re.IGNORECASE,
)
_MINE = re.compile(r"^(?:i\b|my\b|(?:please )?call me\b|(?:请)?叫我|我)", re.IGNORECASE)
# Not a plain statement: a hypothetical, a negation, someone's opinion, a
# to-do, the past, or a question in Chinese words.
_NOT_PLAIN = re.compile(
    r"\b(?:if|unless|suppose|supposing|imagine|assuming|pretend|hypothetically"
    r"|wonder|would|could|might|maybe|perhaps|probably|wish|hope"
    r"|not|no|never|nobody|nothing|none|neither|nor|anymore"
    r"|thinks?|thought|says?|said|believes?|claims?|feels?|guess|suspects?"
    r"|according|need|needs|have to|has to|must|should|remind|todo|to-do"
    r"|was|were|used to|formerly|previously)\b|n't\b|'d\b"
    r"|如果|假如|要是|假设|万一|也许|可能|设想|想象|但愿|不|没|别"
    r"|觉得|认为|以为|听说|据说|相信|(?:他|她|他们|她们|别人|大家|有人)说"
    r"|需要|提醒|待办|以前|曾经|过去|吗|呢|什么|哪|谁|几|怎么|怎样|为什么|多少|是否",
    re.IGNORECASE,
)
# A value that is a passing state, not a lasting fact ("my build is failing").
_PASSING_WORDS: Final = (
    "broken down failing slow stuck missing wrong empty full ready done busy sick "
    "tired late gone away back out off fine ok okay here there over locked "
    "delayed cancelled canceled crashed frozen blocked finished"
)
_PASSING: Final = frozenset(_PASSING_WORDS.split())
# The "." of a title ("Mr."), "vs.", "No. 5", an "etc." that goes on and an
# initial ("J. Smith") ends no sentence: the joiner holds it to the word after.
_HELD = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Mx|Prof|St|Jr|Sr|Mt|vs)\.[^\S\n]+"
    r"|\betc\.[^\S\n]+(?=[a-z\d])|\bNo\.[^\S\n]+(?=\d)"
    r"|(?<![\w.'\u2019])[A-HJ-Z]\.[^\S\n]+(?=[A-Z][a-z])"
)
_JOINER: Final = "\u2060"
# A sentence that may carry a request to remember (:func:`said` reads its fact).
_MAY_ASK = re.compile(
    r"remember|forget|from now on|in memory|keep in mind|make a note"
    r"|记住|别忘|以后|记下|记一下",
    re.IGNORECASE,
)
# How many of the newest statements of no subject are compared by meaning,
# and how many close pairs of them at most, so recall stays quick.
PLAIN_TURNS: Final = 1_000
PAIR_CHECKS: Final = 5_000

# The first person in the user's words, in the second person in a record.
_SWAPS: Final = (
    (re.compile(r"^(?:please\s+)?call me\b", re.IGNORECASE), "you go by"),
    (re.compile(r"\b(?:I am|I'm)\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bI was\b", re.IGNORECASE), "you were"),
    (re.compile(r"\b(?:I have|I've)\b", re.IGNORECASE), "you have"),
    (re.compile(r"\b(?:I will|I'll)\b", re.IGNORECASE), "you will"),
    (re.compile(r"\b(?:we are|we're)\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bI\b"), "you"),
    (re.compile(r"\bmyself\b", re.IGNORECASE), "yourself"),
    (re.compile(r"\b(?:mine|ours)\b", re.IGNORECASE), "yours"),
    (re.compile(r"\b(?:my|our)\b", re.IGNORECASE), "your"),
    (re.compile(r"\b(?:me|we)\b", re.IGNORECASE), "you"),
    (re.compile(r"\bus\b"), "you"),
    (re.compile(r"^(?:请)?叫我"), "你希望被叫作"),
    (re.compile("我们"), "你们"),
    (re.compile("我"), "你"),
)


@dataclass(frozen=True)
class Candidate:
    """One record to write: its text, the user's quote, topic and signal."""

    text: str
    quote: str
    topic: str
    admitted_by: str

    @property
    def kind(self) -> str:
        return TOPIC_KINDS[self.topic]


@dataclass(frozen=True)
class Written:
    """A record a turn wrote; ``replaced`` is the id it superseded, if any."""

    id: int
    text: str
    topic: str
    replaced: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "topic": self.topic}


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char) * (2 if index % 2 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def sensitive(text: str) -> bool:
    """Whether *text* holds a secret: a key, a password or code, a card or ID number."""
    if redact_secrets(text)[1] or _SECRET_WORDS.search(text) or _ID_NUMBER.search(text):
        return True
    return any(_luhn(re.sub(r"\D", "", m.group(0))) for m in _CARD.finditer(text))


def second_person(sentence: str) -> str:
    """The user's *sentence* as a record: second person, one sentence, bounded."""
    text = " ".join(sentence.split()).strip(" -—:：,，")
    for pattern, replacement in _SWAPS:
        text = pattern.sub(replacement, text)
    return finished(text)


def finished(text: str) -> str:
    """*text* as a record reads: one line, a capital, an end mark, bounded."""
    text = " ".join(text.split()).strip(" -—:：,，")
    text = text[:1].upper() + text[1:]
    if len(text) > MAX_RECORD_CHARS:
        text = text[: MAX_RECORD_CHARS - 1].rstrip() + "…"
    elif text and text[-1].isascii() and text[-1].isalnum():
        text += "."
    elif text and not re.search(r"[.!。！…)）\"”]$", text):
        text += "。"
    return text


def topic_of(sentence: str, *, request: bool) -> str:
    """The topic of a fact the user stated (*request*: after "remember that")."""
    if _DECISION.search(sentence):
        return "decisions"
    if _PREFERENCE.search(sentence):
        return "preferences"
    if _FIRST_PERSON.search(sentence):
        return "about"
    return "people" if request else "about"


# How a reply says it kept a note (:func:`claimed`); group ``x`` starts the fact.
_AT = r"(?:^|(?<=[.!?。！？\n]))\s*"
_LEAD = r"\s*(?:[—–:：,]|--?)\s*"
_CLAIMS: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        _AT + r"Remembered[:：]\s*(?P<x>.+)",
        _AT + r"Noted" + _LEAD + r"(?P<x>.+)",
        _AT
        + r"Got it"
        + _LEAD
        + r"(?P<x>(?=\S)(?!I(?:'ve| have|'ll| will) (?:noted|remember)\b)"
        + r"(?=[\s\S]*\bnoted\b).+)",
        r"\bI(?:'ve| have) noted(?: that)?\s+(?P<x>.+)",
        r"\bI(?:'ll| will) remember(?: that)?\s+(?!to\b)(?P<x>.+)",
        r"(?:记下了|记住了)[：:，,]\s*(?P<x>.+)",
        r"已记住[：:]\s*(?P<x>.+)",
    )
)
_CLAIM_END = re.compile(r"[。！？]+|[.!?]+(?:\s|$)|\n|\s?[(（]")
_NOTED_TAIL = re.compile(r"[\s,;—–-]*\bnoted\b[.!\s]*$", re.IGNORECASE)
_TO_USER = re.compile(r"\b(?:you|your|yours)\b|[你您]", re.IGNORECASE)
# X reads as a statement of a fact when it has a verb or copula, or a possessive.
_STATEMENT = re.compile(
    r"\b(?:is|are|was|were|has|have|had|lives|works|starts|leaves|meets|teaches"
    r"|coaches)\b|['\u2019]s\b|[是在叫有住]",
    re.IGNORECASE,
)
CLAIM_TERMS: Final = 3


def claimed(answer: str) -> str | None:
    """The fact *answer* says it noted, as a note sentence: ``None`` when it does
    not claim a save, ``""`` when it does and names no statement.

    The first claim counts: ``Remembered: X``, ``Noted — X`` (or ``:``/``,``),
    ``I've noted (that) X``, ``I'll remember (that) X``, ``Got it — X`` with a
    ``noted`` after it, ``记下了：X``, ``已记住：X``, ``记住了：X``. X ends where its
    sentence does ("Prof." or "Mrs." does not end it, :data:`_HELD`), or at a
    ``(``. X must read as a statement: written to the user (``you``, ``your``,
    ``yours``, ``你``, ``您``) or with a verb, copula or possessive
    (:data:`_STATEMENT`: "Prof. Duarte's office is in the Atrium Wing."); the
    facts about another person are notes too. Otherwise X of at least
    :data:`CLAIM_TERMS` content words is a claim of nothing usable (``""``), and a
    shorter one ("Noted, thanks!") is no claim.
    """
    text = answer.replace("**", "").replace("__", "")
    found = [m for pattern in _CLAIMS if (m := pattern.search(text))]
    if not found:
        return None
    rest = _HELD.sub(
        lambda m: m.group().rstrip() + _JOINER,
        text[min(found, key=lambda m: m.start()).start("x") :],
    )
    fact = rest[: end.start()] if (end := _CLAIM_END.search(rest)) else rest
    note = finished(_release(_NOTED_TAIL.sub("", fact)))
    if _TO_USER.search(note):
        return note
    if len(note_terms(note)) < CLAIM_TERMS:
        return None
    return note if _STATEMENT.search(note) else ""


def candidates(typed: str) -> tuple[list[Candidate], bool]:
    """The records the user's *typed* text asks for, and whether one was refused.

    Only a request to remember counts ("remember that…", "记住…", "别忘了…",
    "帮我记一下…"); a plain statement is the model's to keep. A sentence that
    looks like a secret, a request's or a statement's, is never a candidate;
    it only sets the refusal flag, so the turn can say "Not remembered".
    """
    if not typed.strip() or len(typed) > MAX_TYPED_CHARS:
        return [], False
    # A ";" joins two statements as often as it joins two clauses.
    joined = "\n".join(
        _ALSO_ASKS.sub("remember that ", sentence, count=1)
        for sentence in _split(typed, hold=True)
    )
    signals = classify_user_text(joined)
    facts = [_release(f) for f in signals.facts]
    found = [_release(f) for f in signals.request_facts]
    # A request whose words after the marker did not read as a fact alone
    # ("Remember, I'm allergic to peanuts.") keeps the whole sentence.
    found += [f for f in facts if classify_user_text(f).remember]
    # A statement is not kept by rule, but one with a secret still says so.
    refused = any(sensitive(s) for s in facts)
    out: list[Candidate] = []
    for sentence in found:
        if sensitive(sentence):
            refused = True
            continue
        text = second_person(sentence)
        if len(normalise(text)) < 4 or any(c.text == text for c in out):
            continue
        topic = topic_of(sentence, request=True)
        out.append(Candidate(text, sentence[:MAX_RECORD_CHARS], topic, "user_request"))
    return out, refused


def _split(text: str, *, hold: bool = False) -> list[str]:
    """The sentences of *text*: a ";" ends one too, a title or abbreviation
    such as "Mr." or an initial such as "J." does not (:data:`_HELD`). With
    *hold* the joiner stays in them, for a reader that splits again."""
    joined = _HELD.sub(lambda m: m.group().rstrip() + _JOINER, text)
    joined = re.sub(r"\s*[;；]\s*", "\n", joined)
    return [s if hold else _release(s) for _, s in split_sentences(joined)]


def _release(sentence: str) -> str:
    """*sentence* with the joiner of :func:`_split` read back as a space."""
    return sentence.replace(_JOINER, " ")


def _sentences(typed: str) -> list[str]:
    """The sentences of the user's *typed* words (:func:`_split`); none for a
    message too long to be a statement."""
    if not typed.strip() or len(typed) > MAX_TYPED_CHARS:
        return []
    return _split(typed)


def _plain(sentence: str) -> str:
    """*sentence* without its lead-in or a request after it, when it is a plain
    first-person statement: not a question, a to-do, a hypothetical, a
    negation, the past or someone's opinion, and no secret; else ``""``."""
    text = _LEAD_IN.sub("", " ".join(sentence.split())).strip(" ,，")
    text = _TAIL.split(text, maxsplit=1)[0]
    if (
        not text
        or len(text) > STATEMENT_CHARS
        or text.endswith(("?", "？"))
        or not _MINE.match(text)
        or _NOT_PLAIN.search(text)
        or "`" in text
        or sensitive(text)
    ):
        return ""
    return text


def _lasting(text: str) -> bool:
    """Whether the value of a "my X is Y" statement lasts: not a passing state
    ("my build is failing", "my laptop is very slow") or a plan ("… is to …")."""
    found = _OWNED.match(normalise(text))
    if found is None:
        return True
    words = [w for w in _TOKEN.findall(found.group(2)) if w not in _STILL]
    first = words[0] if words else ""
    if first in _RELATIONS:
        return True
    return not (
        first == "to"
        or (len(first) > 4 and first.endswith("ing"))
        or _PASSING.intersection(words[:3])
    )


def subject_key(text: str) -> str:
    """:func:`subject` and how its value relates to it: two statements of one
    key and different values disagree. ``""`` when *text* names no subject."""
    mine, how = _subject(text)
    return f"{mine}|{how}" if mine else ""


def _keyed(text: str) -> str:
    """The :func:`subject_key` of the plain statement *text* when it states a
    lasting fact (:func:`_lasting`) that the signal lexicon reads as one (not
    a meta sentence such as "my task is …"); else ``""``."""
    key = subject_key(text)
    if not key or not _lasting(text):
        return ""
    if key.split("|")[0] not in _VERB_SUBJECTS and not classify_user_text(text).facts:
        return ""
    return key


def statements(typed: str) -> list[Candidate]:
    """The lasting facts the user's *typed* words state about themselves, as notes.

    Each is a plain first-person statement (no question, to-do, hypothetical,
    negation, past or someone else's opinion, no secret) that names a subject
    ("I live in …", "my advisor is …", "call me …", "我的…是…") with a lasting
    value, reads as a statement to the signal lexicon (a meta sentence such as
    "my task is …" does not) and asks nothing to be remembered
    (:func:`candidates` keeps those). With ``[memory] keep_statements``, code
    keeps after a turn the ones no note of the turn kept.
    """
    out: list[Candidate] = []
    for sentence in _sentences(typed):
        if _MAY_ASK.search(sentence) and classify_user_text(sentence).remember:
            continue
        text = _plain(sentence)
        if not text or not _keyed(text):
            continue
        note = second_person(text)
        if len(normalise(note)) < 4 or any(c.text == note for c in out):
            continue
        topic = topic_of(text, request=False)
        out.append(Candidate(note, text[:MAX_RECORD_CHARS], topic, "user_statement"))
    return out


def changes(typed: str, active: Iterable[StoredEntry]) -> list[Candidate]:
    """The plain statements of *typed* (:func:`statements`) that state a new value
    of a subject an *active* note holds ("My weekly sync is on Friday." beside
    "Your weekly sync is on Wednesday."): the user's change to an old note.
    A statement of a subject no note names is not one, nor is one a note says."""
    held: dict[str, set[str]] = {}
    for entry in active:
        if key := subject_key(entry.text):
            held.setdefault(key, set()).add(normalise(entry.text))
    return [
        cand
        for cand in statements(typed)
        if (key := subject_key(cand.text))
        and key in held
        and normalise(cand.text) not in held[key]
    ]


def covered(cand: Candidate, notes: Iterable[Written]) -> bool:
    """Whether one of *notes* (kept in the same turn) already says *cand*: the
    same subject, or most of the same content words."""
    key = subject_key(cand.text)
    tokens = content_tokens(cand.text)
    return any(
        (key and subject_key(note.text) == key)
        or overlap(tokens, content_tokens(note.text)) >= SUPERSEDE_OVERLAP
        for note in notes
    )


def said(typed: str) -> tuple[tuple[str, str], ...]:
    """What a chat turn's *typed* words state about the user, for
    :func:`newer_turns`: each plain statement, folded, with its
    :func:`subject_key` (``""`` for none). A request to remember counts by the
    fact it carries ("remember that I live in …")."""
    out: list[tuple[str, str]] = []
    for sentence in _sentences(typed):
        facts: Sequence[str] = (sentence,)
        if _MAY_ASK.search(sentence):
            facts = classify_user_text(sentence).request_facts or facts
        for fact in facts:
            if text := _plain(fact):
                out.append((_keyed(text), normalise(text)))
    return tuple(out)


def _stem(word: str) -> str:
    """*word* without its endings, twice, so "meetings" finds "meets"."""
    return stem(stem(word))


def note_terms(text: str) -> list[str]:
    """The content words a note must find in the evidence, in order."""
    terms: list[str] = []
    # "the user's name" restates; its "s" is not a word the user typed.
    text = _POSSESSIVE.sub("", _CJK_FRAMING.sub(" ", text))
    for term in tokenize(normalise(text)):
        if term.isascii():
            if term in STOPWORDS or term in _FRAMING:
                continue
            term = _stem(term)
        elif _CJK_FILLER.intersection(term):
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _evidence(texts: Sequence[str]) -> set[str]:
    """The words of *texts*, stemmed; two adjacent words joined ("3 pm"); and
    Chinese pairs with one character between them ("下周三要去" gives "三去")."""
    found: set[str] = set()
    for text in texts:
        folded = normalise(text)
        terms = tokenize(folded)
        found.update(_stem(t) if t.isascii() else t for t in terms)
        words = _WORD.findall(folded)
        found.update(a + b for a, b in zip(words, words[1:], strict=False))
        for run in _CJK_RUN.findall(folded):
            found.update(run[i] + run[i + 2] for i in range(len(run) - 2))
    return found


def _missing(note: str, evidence: Sequence[str]) -> list[str]:
    """The content words of *note* that none of the *evidence* texts holds.

    A word counts when the evidence has it after stemming, or a word at most
    two letters longer that starts with it ("named" finds "name"); the words a
    note adds to restate a fact ("The user prefers…") need not be found.
    """
    words = _evidence(evidence)
    ascii_words = [w for w in words if w.isascii()]
    return [
        term
        for term in note_terms(note)
        if term not in words
        and not (
            term.isascii()
            and len(term) >= 3
            and any(w.startswith(term) and len(w) <= len(term) + 2 for w in ascii_words)
        )
    ]


def _flagged(note: str) -> set[str]:
    """The stemmed words of *note* that are values by their form: every word of
    a quoted span, and a capitalised word that does not start its sentence."""
    words = [
        word
        for group in _QUOTED.findall(note)
        for span in group
        for word in tokenize(normalise(span))
    ]
    for sentence in _SENTENCE.split(note):
        words += [w.lower() for w in _CASED.findall(sentence)[1:] if w[0].isupper()]
    return {_stem(word) for word in words if word.isascii()}


def value_terms(note: str) -> set[str]:
    """The stemmed values whose exact presence grounding requires (not paraphrases)."""
    flagged = _flagged(note)
    return {
        term
        for term in note_terms(note)
        if not term.isascii()
        or term in flagged
        or _VALUE_WORD.fullmatch(term) is not None
        or any(char.isdigit() for char in term)
    }


def unsupported(note: str, evidence: Sequence[str]) -> list[str]:
    """The content words of *note* that the *evidence* texts lack, when that is
    too much: ``[]`` means grounded.

    The wording may vary, the values may not. Every value (a number or a word
    with a digit, a weekday, a capitalised word inside a sentence, a quoted
    span, a Chinese or Japanese character or pair) must be found, and so must
    at least :data:`WORDS_FOUND` of the other content words (:func:`_missing`
    says when one is found). The words listed are all the missing ones, values
    first.
    """
    gone = _missing(note, evidence)
    if not gone:
        return []
    valued = value_terms(note)
    values = [term for term in gone if term in valued]
    others = [term for term in note_terms(note) if term not in valued]
    lost = [term for term in gone if term not in values]
    if not values and len(others) - len(lost) >= WORDS_FOUND * len(others):
        return []
    return values + lost


def other_script(words: Sequence[str], evidence: Sequence[str]) -> bool:
    """Whether one of *words* is in a script none of the *evidence* texts uses
    (Chinese words in a note about an English message, or the other way round)."""
    cjk = any(_CJK_RUN.search(text) for text in evidence)
    latin = any(re.search("[A-Za-z]", text) for text in evidence)
    return any(
        (word.isascii() and word.isalpha() and not latin)
        or (not word.isascii() and not cjk)
        for word in words
    )


def quote_for(note: str, said: Sequence[str]) -> str:
    """The sentence of *said* that holds most of *note*'s words: the user's quote."""
    terms = set(note_terms(note))
    best, score = "", 0
    for text in said:
        for sentence in _split(text):
            hits = len(terms - set(_missing(note, [sentence])))
            if hits > score:
                best, score = sentence, hits
    return best[:MAX_RECORD_CHARS]


# Whose a sentence is, by its first words (:func:`wrong_owner`): the user's ("my",
# "your", "我的", "你的") or a named third party's ("Marco's …", "马可的 …"). A
# "Name's" before a determiner or an adverb is "Name is" ("John's my advisor").
_OWNER_EN = re.compile(
    r"(?:(?P<user>(?i:my|our|your|the user['’]s))\b"
    r"|(?P<name>(?:(?:Dr|Mr|Mrs|Ms|Mx|Prof|St|Mt)\.\s+)?[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)"
    r"['’]s(?!\s+(?i:my|our|your|his|her|their|the|an?|not|now|also|still|just|going"
    r"|been|got|so|here|there|one)\b))"
)
_OWNER_ZH = re.compile(
    r"(?:(?P<user>(?:我|咱|你|您)们?|用户)|(?P<name>[^\s的，,。！？]{1,6}?))的"
)
# What "Name's" is not a name: a contraction ("That's why …") or a time word.
_NOT_OWNERS = re.compile(
    r"it|that|there|here|what|who|where|when|why|how|let|this|he|she"
    r"|today|tomorrow|yesterday|tonight"
)
_NOT_NAME_ZH = re.compile(
    "[我你您他她它咱这那在是有和与把被让从到每]|^(?:今|明|昨|周末|现在)"
)


def _owner(sentence: str) -> str | None:
    """Whose *sentence* is: ``"user"``, a third party's name, or ``None`` when its
    first words say none (or not clearly: a contraction, a time word)."""
    text = sentence.strip()
    if found := _OWNER_EN.match(text):
        name = found["name"]
        if found["user"]:
            return "user"
        low = name.lower()
        bad = _NOT_OWNERS.fullmatch(low) or _VALUE_WORD.fullmatch(low)
        return None if bad else name
    if found := _OWNER_ZH.match(text):
        name = found["name"]
        return "user" if found["user"] else None if _NOT_NAME_ZH.search(name) else name
    return None


def wrong_owner(note: str, quote: str) -> str:
    """The refusal of *note* when it is about someone else than *quote*, the
    typed sentence it matches best, is; ``""`` when they agree or say nothing.

    A note about a person the user named ("马可的工作室…") that says "你的工作室…"
    would come back as a fact about the user, and the other way round. The name
    must be missing from the other sentence, so "Anna's car" for "My sister
    Anna's car" is no mix-up.
    """
    said, kept = _owner(quote), _owner(note)
    if not said or not kept or (said == "user") == (kept == "user"):
        return ""
    name = kept if said == "user" else said
    other = quote if said == "user" else note
    if name.split()[-1].lower() in other.lower():
        return ""
    own = "'s …" if name.isascii() else "的…"
    if said == "user":
        return (
            f"Not remembered: the user said that about themselves, not about {name}. "
            f"Write the note to the user: “{'Your …' if name.isascii() else '你的…'}”."
        )
    return (
        f"Not remembered: that is about {name}, not the user. "
        f"Write the note about them: “{name}{own}”."
    )


def _relation(value: str) -> str:
    """How the value after "is" (or 是, 叫, 姓) relates to its subject: a word
    such as ``on`` or ``called``, ``desc`` for a description ("very strict",
    "a doctor", "橙色的"), or ``""`` for a plain value ("Juno", "小宇")."""
    if value[:1] in ("叫", "姓"):
        return value[0]
    if value[:1] == "在":
        return "at"
    if value[:1] == "是":
        return "desc" if value.endswith("的") else ""
    words = [w for w in value.split() if w not in _STILL]
    first = words[0] if words else ""
    return first if first in _RELATIONS else "desc" if first in _DESCRIBES else ""


def _third_party(text: str) -> tuple[str, str] | None:
    """:func:`_subject` of "Tom Briggs's office is in room 9C" or "马可的工作室在
    中山路88号": the owner's name with the attribute, never the user's own
    subject. ``None`` unless *text* starts with a third party's possessive;
    ``("", "")`` when it says no "X is Y" of them."""
    owner = _owner(text)
    if owner is None or owner == "user":
        return None
    sentence = text.strip()
    found = _OWNER_EN.match(sentence)
    pattern, joint = _THIRD, "'s "
    if found is None:
        found = _OWNER_ZH.match(sentence)
        pattern, joint = _THIRD_ZH, "的"
    read = pattern.match(normalise(sentence[found.end() :])) if found else None
    if read is None:
        return "", ""
    attribute = read.group(1)
    if joint != "的":
        words = _TOKEN.findall(_POSSESSIVE.sub("", attribute))
        attribute = " ".join(stem(w) for w in words if w not in _ARTICLES)
    return normalise(owner) + joint + attribute, _relation(read.group(2))


def _subject(text: str) -> tuple[str, str]:
    """:func:`subject` and the relation of its value (:func:`_relation`)."""
    if (third := _third_party(text)) is not None:
        return third
    folded = normalise(text)
    for key, pattern in _SUBJECTS:
        if found := pattern.search(folded):
            return key, "unit" if _UNIT.search(folded[found.start() :]) else ""
    if found := _OWNED.match(folded):
        words = _TOKEN.findall(_POSSESSIVE.sub("", found.group(1)))
        stems = " ".join(stem(w) for w in words if w not in _ARTICLES)
        return stems, _relation(found.group(2))
    if found := _OWNED_ZH.match(folded):
        return found.group(1), _relation(found.group(2))
    if found := _HAVE.match(folded):
        words = _TOKEN.findall(_POSSESSIVE.sub("", found.group(1)))
        return " ".join(stem(w) for w in words if w not in _ARTICLES), "on"
    return "", ""


def subject(text: str) -> str:
    """What *text* is about without its value; ``""`` when it names none.

    "I live in Lisbon", "You live in Utrecht" and "The user lives in Utrecht"
    are ``home``; "my weekly sync is on Friday" is ``weekly sync``; "我的健身
    教练是小宇" is ``健身教练``. Look-alikes keep apart: "flight to porto" is
    not "flight to faro", "书法课" is not "游泳课". A third party's own fact
    names its owner too: "Tom Briggs's office is in room 9C" is ``tom briggs's
    office`` and "马可的工作室在中山路88号" is ``马可的工作室``, never the user's
    ``manager`` or ``工作室``.
    """
    return _subject(text)[0]


def _weight(tokens: Sequence[str]) -> float:
    """Words count 1, Chinese characters 1/2 (a word is about two)."""
    return sum(1.0 if t.isascii() else 0.5 for t in tokens)


def value_apart(text: str, other: str) -> bool:
    """Whether *text* says what *other* says but one value: one changed span of
    at most :data:`VALUE_TOKENS` words (or Chinese characters), in a frame of
    at least two content words they share that outweighs it ("You go to the
    gym on Mondays/Fridays"; not "你习惯在湖边公园跑步/早餐吃豆浆")."""
    left, right = _TOKEN.findall(normalise(text)), _TOKEN.findall(normalise(other))
    changed = [
        op
        for op in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    if len(changed) != 1:
        return False
    _, start, end, other_start, other_end = changed[0]
    frame = _weight(
        [
            t
            for t in left[:start] + left[end:]
            if t not in STOPWORDS and t not in _CJK_FILLER
        ]
    )
    value = max(_weight(left[start:end]), _weight(right[other_start:other_end]))
    longest = max(end - start, other_end - other_start)
    return longest <= VALUE_TOKENS and frame >= max(2.0, value)


def _lacks(subject_: str, text: str) -> bool:
    """Whether *text* lacks a word of the subject *subject_* names (an owner, an
    attribute). A subject a verb names ("home", "name") has no words of its own."""
    if not subject_ or subject_ in _VERB_SUBJECTS:
        return False
    tokens = _TOKEN.findall(normalise(text))
    words = {*tokens, *(stem(w) for w in tokens)}
    wanted = _TOKEN.findall(subject_.replace("'s ", " "))
    return any(w not in words for w in wanted if w not in _CJK_FILLER)


def _near(vector: Sequence[float] | None, model: str, entry: StoredEntry) -> bool:
    """Whether *vector* (of *model*) and *entry*'s are close in meaning."""
    mine, theirs = unit(vector), unit(entry.vector)
    if mine is None or theirs is None or not model or entry.embed_model != model:
        return False
    return mine.shape == theirs.shape and float(np.dot(mine, theirs)) >= NEAR_COSINE


def decide(
    text: str,
    active: Sequence[StoredEntry],
    *,
    kind: str = "",
    vector: Sequence[float] | None = None,
    embed_model: str = "",
) -> tuple[str, int | None]:
    """``("IGNORE", id)`` for a record kept already, ``("UPDATE", id)`` to supersede, or NEW.

    An old record of the same :func:`subject`, its value related the same way
    ("is on", "is called", a plain value), is replaced, whatever its topic; two
    records that name different subjects never replace each other, and the
    fallback to shared words refuses a record that names a subject the other's
    text does not hold ("Your manager is Tom Briggs." beside "Tom Briggs's
    office is in room 9C."). Else, with vectors, one close in meaning and
    :func:`value_apart`; else one of the same *kind* that shares most content
    words ("said again in other words").
    """
    folded = normalise(text)
    mine, how = _subject(text)
    tokens = content_tokens(text)
    best: StoredEntry | None = None
    best_score = 0.0
    for entry in active:
        if normalise(entry.text) == folded:
            return "IGNORE", entry.id
        theirs, their_how = _subject(entry.text)
        if mine == theirs and mine in _VERB_SUBJECTS and how != their_how:
            continue
        if mine and theirs and (mine != theirs or how == their_how):
            score = 1.0 if mine == theirs else 0.0
        elif _near(vector, embed_model, entry) and value_apart(text, entry.text):
            score = 1.0
        elif (not kind or entry.kind == kind) and not (
            _lacks(mine, entry.text) or _lacks(theirs, text)
        ):
            score = overlap(tokens, content_tokens(entry.text))
        else:
            score = 0.0
        if score >= SUPERSEDE_OVERLAP and score >= best_score:
            best, best_score = entry, score
    return ("UPDATE", best.id) if best is not None else ("NEW", None)


def newer_turns(items: Iterable[Item]) -> dict[tuple[str, int], Item]:
    """The chat turns a newer turn replaced, each (session id, turn) to that turn.

    Of the turns whose user stated one subject (:func:`subject_key`, in
    :attr:`Item.said`) with different values, each older one is replaced by
    the newest ("I live in Lisbon." by a later "I live in Utrecht."); a turn
    that said the newest value again is not. With an embedding model, two
    statements of no subject are one when their turns are close in meaning
    and they differ in one short value (:func:`value_apart`), among the
    :data:`PLAIN_TURNS` newest such statements (at most :data:`PAIR_CHECKS`
    close pairs). No model call.
    """
    turns = sorted(
        (item for item in items if not item.is_record and item.said),
        key=lambda item: (item.created_at, item.key),
        reverse=True,
    )
    out: dict[tuple[str, int], Item] = {}
    newest: dict[str, tuple[Item, str]] = {}
    plain: dict[tuple[str, int], list[tuple[Item, str]]] = {}
    compared = checks = 0
    for item in turns:
        where = (item.session_id, item.turn)
        for key, text in item.said:
            if not key:
                if item.vector is not None and compared < PLAIN_TURNS:
                    shape = (item.embed_model, int(item.vector.shape[0]))
                    plain.setdefault(shape, []).append((item, text))
                    compared += 1
                continue
            newer, value = newest.setdefault(key, (item, text))
            if newer is not item and value != text:
                out.setdefault(where, newer)
    for group in plain.values():
        vectors = np.stack(
            [item.vector for item, _ in group if item.vector is not None]
        )
        close = vectors @ vectors.T >= NEAR_COSINE
        for first, (newer, text) in enumerate(group):
            for other in np.flatnonzero(close[first, first + 1 :]) + first + 1:
                older, value = group[int(other)]
                where = (older.session_id, older.turn)
                if older is newer or where in out or value == text:
                    continue
                checks += 1
                if checks > PAIR_CHECKS:
                    return out
                if value_apart(text, value):
                    out[where] = newer
    return out


def _replaces(old: StoredEntry, source: dict[str, Any]) -> list[list[Any]]:
    """The turns whose facts a record replacing *old* replaces: *old*'s and
    those *old* replaced, but never the new record's own turn."""
    turns = [
        [str(t[0]), int(t[1])]
        for t in old.provenance.get("replaces") or []
        if isinstance(t, (list, tuple)) and len(t) == 2
    ]
    turns.append([old.session_id, int(old.provenance.get("turn") or 0)])
    own = [str(source.get("session_id", "")), int(source.get("turn") or 0)]
    return [t for n, t in enumerate(turns) if t != own and t not in turns[:n]]


async def write(
    store: SqliteLongTerm,
    found: Sequence[Candidate],
    vectors: Sequence[list[float] | None],
    *,
    source: dict[str, Any],
    embed_model: str,
    origin: str = "user",
    created_at: float | None = None,
) -> list[Written]:
    """Write each candidate as a record of *source* (session, turn, chat, surface).

    A project note also keeps the chat's folder (``source["workspace"]``), as an
    imported one does, so a task in that folder gets it. No other topic does.

    *origin* and *created_at* are what the store keeps as the row's source and
    time; sleep writes as ``sleep``, at the time of the turn it read.
    """
    written: list[Written] = []
    for cand, vector in zip(found, vectors, strict=True):
        active = await store.active_entries()
        action, target = decide(
            cand.text, active, kind=cand.kind, vector=vector, embed_model=embed_model
        )
        if action == "IGNORE":
            continue
        provenance = {
            **{k: v for k, v in source.items() if k != "workspace"},
            "topic": cand.topic,
            "quote": cand.quote,
            "embed_model": embed_model if vector is not None else "",
        }
        if cand.topic == "projects" and source.get("workspace"):
            provenance["workspace"] = source["workspace"]
        old = next((e for e in active if e.id == target), None)
        if old is not None and (replaces := _replaces(old, source)):
            provenance["replaces"] = replaces
        entry_id, actual = await store.apply(
            text=cand.text,
            kind=cand.kind,
            source=origin,
            admitted_by=cand.admitted_by,
            provenance=provenance,
            task_id="",
            session_id=str(source.get("session_id", "")),
            created_at=created_at or time.time(),
            embedding=vector,
            action=action,
            target_id=target,
            reason="said again in other words" if target else "new",
            source_ids=(),
            keep_text=False,
            candidate_id=new_record_id(),
            record={
                "topic": cand.topic,
                "quote": cand.quote,
                "embed_model": provenance["embed_model"],
            },
        )
        if entry_id is not None:
            replaced = target if actual == "UPDATE" else None
            written.append(Written(entry_id, cand.text, cand.topic, replaced))
    return written
