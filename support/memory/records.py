"""Records: what the user said about themselves or their work, one sentence each.

A record is written two ways, and neither asks the user to confirm. When the
user asks ("remember that…", "记住…", "别忘了…", "帮我记一下…"), code keeps the
fact the request carries (:func:`candidates`, the signal lexicon of
:mod:`localmelo.melo.memory.hippo.signals`), in the second person ("You
prefer short answers."). Otherwise the chat's model keeps a note with the
``remember`` tool when it judges something worth keeping; the note's content
words must come from what the user typed in that chat or from a finished
task's result shown there (:func:`unsupported`), never from a page or tool
output alone. A plain first-person statement of a lasting fact that names a
subject ("I live in …", "my advisor is …", "call me …", "我的…是…") is a
statement (:func:`statements`; a question, a to-do, a hypothetical, a negation or
someone else's opinion never is): a chat turn keeps none that the model did not
keep, but a turn written after the fact (``melo memory ingest``) does. When the
turn kept no note, code also keeps a plain statement that gives a new value to a
subject an active note holds (:func:`changes`: "My weekly sync is on Friday." after
"Your weekly sync is on Wednesday."). Each record has at most
:data:`MAX_RECORD_CHARS` characters, its topic, the user's quote and where it was
said. Text that looks like a secret is refused, not stored
redacted. A new record of the same subject supersedes the old one ("call me
Ray" after "call me R.", "我的健身教练是小宇" after "…是阿凯"): the old one is
kept, not recalled, and comes back with an undo of the new one; a note from a page, a
file or a result replaces only an older note of that kind, and both stand otherwise
(:func:`write`). The subject is
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

import dataclasses
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Final

import numpy as np

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.sign_in import SPENT, password_spans
from localmelo.melo.memory.bm25 import tokenize
from localmelo.melo.memory.hippo.policy import (
    STOPWORDS,
    content_tokens,
    normalise,
    overlap,
)
from localmelo.melo.memory.hippo.signals import (
    classify_user_text,
    request_clause,
    request_note,
)
from localmelo.melo.memory.long import StoredEntry
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.memory.notes import split_sentences
from localmelo.melo.schema import new_record_id
from localmelo.support.memory import people
from localmelo.support.memory.recall import (
    OLDER,
    OWN,
    REWORDED,
    TAKEN,
    Item,
    stem,
    unit,
)

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
# "You said: …", "You go by…"), the assistant's name a note uses for "I"
# ("You prefer that melo explains …"), and function words a paraphrase swaps
# ("then", "each", "use", "who"); they are never what has to be found in the
# user's words.
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
        "melo",
        "who",
        "whom",
        "whose",
        "which",
        "where",
        "when",
    ]
)
# Joining words: they state no relation, so a paraphrase may add or drop them
# ("so", "now", "will", "plus", "some", "about"; the "re", "ll" and "ve" a
# contraction leaves). A negation ("not", "no", "never") turns a fact round, and
# a word of relation ("over", "after", "before", "since", "off", "under") says
# how two things stand, so those stay content words. A joining word is skipped
# only where the note writes it in lower case: one written with a capital
# anywhere, first word of a sentence included, may be a given name ("Will Park",
# "Per Andersson") and stays a word that has to be found.
_JOINING: Final = frozenset(
    ["per", "now", "so", "will", "would", "could", "plus", "some", "currently"]
    + ["including", "about", "around", "re", "ll", "ve"]
)
# Past and participle forms that stemming cannot reduce, each with its base, so a
# note's "kept" finds the user's "keep" (the common irregular verbs only). A form
# that is also a name ("Ran") is compared by its base like any other.
_IRREGULAR: Final = {
    form: base
    for base, forms in {
        "be": "was were been",
        "begin": "began begun",
        "bring": "brought",
        "build": "built",
        "buy": "bought",
        "catch": "caught",
        "choose": "chose chosen",
        "come": "came",
        "do": "did done",
        "drive": "drove driven",
        "eat": "ate eaten",
        "feel": "felt",
        "find": "found",
        "fly": "flew flown",
        "forget": "forgot forgotten",
        "get": "got gotten",
        "give": "gave given",
        "go": "went gone",
        "grow": "grew grown",
        "have": "had",
        "hear": "heard",
        "hold": "held",
        "keep": "kept",
        "know": "knew known",
        "lead": "led",
        "leave": "left",
        "lose": "lost",
        "make": "made",
        "mean": "meant",
        "meet": "met",
        "pay": "paid",
        "ride": "rode ridden",
        "run": "ran",
        "see": "saw seen",
        "sell": "sold",
        "send": "sent",
        "sit": "sat",
        "sleep": "slept",
        "speak": "spoke spoken",
        "spend": "spent",
        "stand": "stood",
        "take": "took taken",
        "teach": "taught",
        "think": "thought",
        "understand": "understood",
        "wear": "wore worn",
        "write": "wrote written",
    }.items()
    for form in forms.split()
}
# Terms that are values: a change in one is a different fact. A word is a value
# when it has a digit, is a number word or a weekday, is capitalised inside a
# sentence or sits in a quoted span. A Chinese or Japanese character or pair is
# one when it sits in a quoted span, holds a number character, or holds a
# character that no evidence text has and that is no common wording
# (:data:`_CJK_WORDING`); the rest of the CJK terms are wording, like the other
# content words (:func:`value_terms`).
_VALUE_WORD = re.compile(
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty"
    r"|forty|fifty|hundred|thousand|(?:mon|tues|wednes|thurs|fri|satur|sun)day"
)
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”|「([^」]+)」|『([^』]+)』")
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|\n")
# What share of a note's other content words the evidence must hold (a word of
# another script, or a single Chinese or Japanese character: a pair of characters
# only counts when it is a value).
WORDS_FOUND: Final = 0.8
_CJK_FILLER: Final = frozenset("的了是在我你他她它们和与就也都吗呢吧啊么")
# Characters of common wording: a note that restates what the user typed may add
# or swap them without changing the fact (particles, pronouns, prepositions and
# conjunctions, adverbs, auxiliaries and light verbs, words of time and manner,
# the verbs of liking and planning, and the Japanese particles). Characters that
# make names and places (surnames, 州 山 江 京 市), the number characters and the
# negations are left out on purpose: those are details. This is in addition to
# :data:`_CJK_FILLER`, whose characters are dropped from a note's terms.
_CJK_WORDING: Final = frozenset(
    "着过地得呀啦嘛哦哈呗"  # particles
    "这那些个其该此自己咱您谁哪什怎样如某各本"  # pronouns
    "把被让给对向从到按照以为于由跟同往经通比离替靠"  # prepositions
    "并且或而但却则若果虽然因所及即既之"  # conjunctions
    "又再还才先刚已正将很太最更挺常总每只仅另全"  # adverbs
    "会能可要想应须需肯敢愿用做去来有说看使叫当作成出进回起上下开打拿放弄搞求请完"  # verbs
    "时候天次后前间内外里事情方式法种类点好物"  # time, manner, things
    "喜欢爱希望算计划准备决定觉认知道记住"  # liking, planning, knowing
    "习惯偏默般较尽"  # habit
    "のはがをにでともへやかねよなだてたるます"  # Japanese particles
)
# Number characters (the CJK values that hold a digit); 一 is left out because it
# is as often wording (一下, 一些, 一起) as a number.
_CJK_NUMBERS: Final = frozenset("〇零二三四五六七八九十百千万亿两")
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
# What a cut-out password's mark is read as: a value by its shape.
_STOOD_FOR: Final = "0000"
# In a title that names a password or a code, what may be its value
# (:func:`named_secret_spans`): a run of plain characters with a digit in it, and
# what follows the word and a link. A hyphen links only after a space
# (``password-protected`` names no value).
_PLAIN_RUN = re.compile(r"[!-~\d]+")
_DIGIT = re.compile(r"\d")
_RUN_TAIL: Final = ".,;:!?)"
_LINK = re.compile(
    r"\s*(?:[:：=\u2013\u2014]|(?:is|are|was|were|to)\b|[为是])\s*|\s+-\s*",
    re.IGNORECASE,
)
# An ID number as it is written: ``123-45-6789``, or 17 digits and a check character.
# The 18 do not start right after a digit and a dot (a decimal number's fraction).
_ID_NUMBER = re.compile(
    r"(?<!\d)(\d{3}-\d{2}-\d{4}|(?<!\d\.)\d{17}[\dXx])(?![\dA-Za-z])"
)
# A card number is 13 to 19 digits that pass the checksum. Read plainly, one space or
# hyphen may stand between two digits. A card does not start right after a digit and
# a dot: those are the fraction of a decimal number.
_CARD = re.compile(r"(?<!\d)(?<!\d\.)(?:\d[ -]?){12,18}\d(?!\d)")
# Read as it is written: between two digits at most one separator, a space, hyphen,
# dot or dash, one Unicode space or invisible character (no-break, thin, ideographic,
# zero-width ...), or a line break with a few spaces around it (a wrapped line). A
# tab or a run of spaces is not one: they set out a table. A dot or a dash (not the
# plain hyphen) stands between a card's digits only in the one form
# ``4111.1111.1111.1111`` (:func:`_card_like`): floats, prices, versions, addresses
# and year ranges are written with them too. A match may hold more than one line
# break; :func:`_card_like` allows one. Every count is bounded (12 to 18 digits, 8
# spaces around a break), so no repetition is nested unbounded and a match takes
# time in proportion to the text.
_SPACES: Final = 8
_FILL = r"\u00a0\u1680\u2000-\u200d\u202f\u205f\u2060\u3000\ufeff"
# The Unicode hyphens and dashes (hyphen, non-breaking hyphen, figure dash, en dash,
# em dash) and the minus sign.
_DASHES = r"\u2010-\u2014\u2212"
_GAP = rf"(?:[-.{_DASHES}{_FILL}]| )"
_BREAK = rf"[ {_FILL}]{{0,{_SPACES}}}(?:\r\n|[\r\n])[ {_FILL}]{{0,{_SPACES}}}"
_CARD_WRITTEN = re.compile(
    rf"(?<!\d)(?<!\d\.)(?:\d(?:{_GAP}|{_BREAK})?){{12,18}}\d(?!\d)"
)
_LINE_BREAK = re.compile(r"\r\n|[\r\n]")
# A separator that a card is written with only in one form (:func:`_card_like`).
_STRICT_GAP = re.compile(rf"[.{_DASHES}]")
# The end of a decimal number's whole part: a digit and a dot.
_WHOLE_PART = re.compile(r"\d\.\Z")
# A digit joined to a number through a separator or a line break, on either side.
_JOINED_BEFORE = re.compile(rf"\d(?:{_GAP}|{_BREAK})\Z")
_JOINED_AFTER = re.compile(rf"(?:{_GAP}|{_BREAK})\d")
# How far back a digit can be joined to a number: the digit and the longest break
# (CRLF with :data:`_SPACES` spaces on each side).
_JOIN_REACH: Final = 1 + 2 + 2 * _SPACES
# The digit groups of a card match that fails the checks, for the retry without its
# first or last group.
_DIGITS = re.compile(r"\d+")

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

# The first person in a note (:func:`first_person`): "I" only as typed, a capital.
_ME = re.compile(
    r"(?-i:\bI\b(?!-\d))|\b(?:me|my|mine|myself)\b|我(?!们)", re.IGNORECASE
)
# What a note keeps as it is when it is put in the second person: a quoted span
# and the numeral of "Phase I", "Type I diabetes", "World War I".
_KEPT = re.compile(
    r"\"[^\"]+\"|“[^”]+”|「[^」]+」|『[^』]+』"
    r"|\b(?:Phase|Type|War|Chapter|Volume|Appendix|Category|Tier|Edition) I\b(?!['’])"
)
# The fewest characters a sentence the user typed has to have for a note to quote it.
QUOTE_CHARS: Final = 8
# The user named in the third person (:func:`names_user`); "my user account" is no one.
_THIRD_USER = re.compile(
    r"(?<!\bmy )(?<!\byour )(?<!\bour )\buser(?:['’]s)?\b|\bthe person\b(?! I\b)|用户",
    re.IGNORECASE,
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
    """A record a turn wrote; ``replaced`` is the id it superseded, if any.

    ``source`` is where the note came from (``pointed``, ``result`` or ``inferred``)
    when the note gate says so, which it does for a note that rests on something
    other than the user's own words; ``""`` otherwise, and then the event has no
    ``source``.
    """

    id: int
    text: str
    topic: str
    replaced: int | None = None
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "text": self.text, "topic": self.topic}
        if self.source:
            data["source"] = self.source
        return data


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char) * (2 if index % 2 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def _card_like(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]``, a match of a card pattern, is a card number: a dot
    or a dash (not the plain hyphen) stands between its digits only when every
    separator is that one character, there are at least three, and every group of
    digits but the last has four (``4111.1111.1111.1111``); there is at most one line
    break, and with one the number is the whole run (no digit joined to it before or
    after, so the rows of a table are never one wrapped number); and the checksum
    passes."""
    part = text[start:end]
    separators = re.sub(r"\d", "", part)
    if _STRICT_GAP.search(separators) and (
        len(set(separators)) != 1
        or len(separators) < 3
        or any(len(group) != 4 for group in _DIGITS.findall(part)[:-1])
    ):
        return False
    breaks = len(_LINE_BREAK.findall(part))
    if breaks > 1:
        return False
    if breaks and (
        _JOINED_BEFORE.search(text[max(0, start - _JOIN_REACH) : start])
        or _JOINED_AFTER.match(text, end)
    ):
        return False
    return _luhn(re.sub(r"\D", "", part))


def number_spans(text: str) -> list[tuple[int, int]]:
    """Where *text* holds an ID number or a card number: the ``(start, end)`` of each,
    sorted, in *text* as given.

    An ID number is ``123-45-6789`` or 17 digits and a check character. A card number
    is 13 to 19 digits that pass the checksum, read both ways (:data:`_CARD`,
    :data:`_CARD_WRITTEN`) and checked by :func:`_card_like`. When a match fails the
    checks, it is read again without its first group of digits, without its last, and
    without both: an expiry date, a security code or an order number may stand next to
    a card. The part kept holds 13 to 19 digits and no line break, and with more than
    one group starts with a group of four or more digits. It is for a reader that
    takes text out by its place, as :func:`~localmelo.melo.checker.redaction.secret_spans`
    is: a number alone on a line, or one that crosses a line, belongs to no sentence.

    A dot or a dash separates a card's digits only in the form
    ``4111.1111.1111.1111``, and neither a card nor an 18-character ID number starts
    right after a digit and a dot (a decimal number's fraction): floats, prices,
    versions, addresses and coordinates are no cards. Four years joined in that form
    (``2019-2020-2021-2022``, with any one dash) are read as a card when they pass
    the checksum, about one time in ten.

    Not found: a card written in groups after a group of four or more digits on its
    line (written with dots, dashes or Unicode spaces, also when that group ends the
    line before it); a card with tabs or wide gaps between its groups; a card
    wrapped over two lines with another number joined to it on the line before or
    after (a line before it that is itself cut as a secret does not count: the
    cleaning of the index repeats, and finds the card when that line is gone); a
    hyphen followed by a line break; a card with one group per line; ``4111. 1111.``;
    a card right after a digit and a dot (``3.4111111111111111``); a card written
    with dots or dashes in groups other than four (``3782.822463.10005``,
    ``41111111.11111111``); a card whose groups a horizontal bar, a small or
    full-width hyphen or a soft hyphen separates; and, of the ID numbers, ``123 45 6789``, nine bare digits
    and the 15-digit form.
    """
    places = [match.span(1) for match in _ID_NUMBER.finditer(text)]
    for pattern in (_CARD, _CARD_WRITTEN):
        for match in pattern.finditer(text):
            if _card_like(text, *match.span()):
                places.append(match.span())
                continue
            groups = [
                (match.start() + group.start(), match.start() + group.end())
                for group in _DIGITS.finditer(match.group())
            ]
            last = len(groups) - 1
            all_dots = not re.sub(r"\d", "", match.group()).strip(".")
            for first, final in ((0, last - 1), (1, last), (1, last - 1)):
                if first > final:
                    continue
                start, end = groups[first][0], groups[final][1]
                digits = sum(b - a for a, b in groups[first : final + 1])
                if not 13 <= digits <= 19 or _LINE_BREAK.search(text[start:end]):
                    continue
                if first != final and groups[first][1] - groups[first][0] < 4:
                    continue
                # The kept part does not start in a decimal number's fraction, unless it
                # is dotted groups itself and so is the whole match
                # (``12.4111.1111.1111.1111``).
                dotted = all_dots and "." in text[start:end]
                if not dotted and _WHOLE_PART.search(text[max(0, start - 2) : start]):
                    continue
                if _card_like(text, start, end):
                    places.append((start, end))
    return sorted(set(places))


def holds_secret_value(text: str) -> bool:
    """Whether *text* holds the value of a secret: a key or token, a code next to its
    word, a private key, a card or ID number. The words alone (``password``) do not count.
    """
    return bool(redact_secrets(text)[1]) or bool(number_spans(text))


def named_secret_spans(text: str) -> list[tuple[int, int]]:
    """Where a short *text* that names a password or a code (:data:`_SECRET_WORDS`), a
    title, may give it: the ``(start, end)`` of each place, sorted.

    A place is every run of plain characters with a digit in it, anywhere in the
    text (``PIN 4821 for the bank``, ``4821 is my PIN``, a value on the next line),
    and, in a sentence that names one, everything after the word and a link (``is``,
    ``was``, ``to``, ``:``, ``=``, a dash) up to the sentence's closing punctuation
    (``my password is correct horse``). No digit of such a text is kept, so no part
    of a number that only a later cut would make readable is. It is for a reader
    that keeps the other words: a note and a turn's text lose the whole sentence
    (:func:`sensitive`).

    Not found: a password of letters only that no link comes before (``sunflower is
    my password``, ``password sunflower``, a value on the line after the link).
    """
    named = [
        (start, sentence, words)
        for start, sentence in split_sentences(text)
        if (words := list(_SECRET_WORDS.finditer(sentence)))
    ]
    if not named:
        return []
    places = [
        (run.start(), run.start() + len(run.group().rstrip(_RUN_TAIL)))
        for run in _PLAIN_RUN.finditer(text)
        if _DIGIT.search(run.group())
    ]
    for start, sentence, words in named:
        for word in words:
            link = _LINK.match(sentence, word.end())
            if link is None:
                continue
            # The sentence keeps its end, so the next one is not joined to it.
            rest = sentence[link.end() :].rstrip(_RUN_TAIL + "。！？")
            if re.search(r"\w", rest):
                places.append((start + link.end(), start + link.end() + len(rest)))
                break  # what a later word of the sentence links to lies inside
    return sorted(set(places))


def gives_password(text: str) -> bool:
    """Whether *text* gives a password as the sign-in contract reads one
    (:func:`~localmelo.melo.contracts.sign_in.password_spans`), or shows where one
    was cut out: the mark left there is read as a value, so ``pw: […]`` is one."""
    return bool(password_spans(text.replace(SPENT, _STOOD_FOR)))


def sensitive(text: str) -> bool:
    """Whether *text* holds a secret: a key, a password or code, a card or ID number.

    A password counts by its word alone, and wherever one is given
    (:func:`gives_password`): that adds the short words the sign-in contract
    knows (``pw: …``, ``pwd = …``), which are no secret by themselves (``pwd`` is
    also a command).
    """
    return (
        holds_secret_value(text)
        or _SECRET_WORDS.search(text) is not None
        or gives_password(text)
    )


def _swap(text: str) -> str:
    for pattern, replacement in _SWAPS:
        text = pattern.sub(replacement, text)
    return text


def _kept(keep: Sequence[str]) -> str:
    """The pattern of what a note leaves as typed: :data:`_KEPT` and each of the
    spans in *keep* (the sentences of :func:`typed_spans`), longest first."""
    spans = sorted(set(keep), key=len, reverse=True)
    return "|".join([_KEPT.pattern, *map(re.escape, spans)])


def typed_spans(note: str, said: Sequence[str]) -> list[str]:
    """The sentences the user typed that *note* copies word for word, which the
    first-person checks leave as typed (a form line such as "I confirm I have read
    the above." is the user's "I", not the note's).

    A whole sentence or line of the *said* texts (:func:`split_sentences`, its end
    mark left off) of at least :data:`QUOTE_CHARS` characters that the note holds
    exactly, white space aside. A phrase the note shares with the user's words
    (its own "I prefer that …" after the user's "I prefer short answers") is not
    one. None when the note says nothing besides such sentences: it only repeats
    the user's statement, which is put in the second person as ever.
    """
    flat = " ".join(note.split())
    spans = {
        span
        for text in said
        for _, sentence in split_sentences(text)
        if len(span := " ".join(sentence.split()).rstrip(" .!?。！？")) >= QUOTE_CHARS
        and span in flat
    }
    rest = flat
    for span in sorted(spans, key=len, reverse=True):
        rest = rest.replace(span, " ")
    return sorted(spans) if any(char.isalnum() for char in rest) else []


def second_person(
    sentence: str, *, protect: bool = False, keep: Sequence[str] = ()
) -> str:
    """The user's *sentence* as a record: second person, one sentence, bounded.

    With *protect*, a quoted span, a numeral "I" (:data:`_KEPT`) and each span of
    *keep* stay as typed.
    """
    text = " ".join(sentence.split()).strip(" -—:：,，")
    pieces = re.split(f"({_kept(keep)})", text) if protect else [text]
    text = "".join(_swap(p) if n % 2 == 0 else p for n, p in enumerate(pieces))
    return finished(text)


def first_person(note: str, *, keep: Sequence[str] = ()) -> bool:
    """Whether *note* speaks as "I": ``I`` (and ``I'm``, ``I've`` …), ``me``,
    ``my``, ``mine``, ``myself`` or ``我`` (not ``我们``, not "we"), a quoted span,
    a numeral "I" ("Phase I") and each span of *keep* aside."""
    return _ME.search(re.sub(_kept(keep), " ", note)) is not None


def addresses_user(note: str, *, keep: Sequence[str] = ()) -> bool:
    """Whether *note* says ``you``, ``your``, ``yours``, ``你`` or ``您``, a
    quoted span and each span of *keep* aside."""
    return _TO_USER.search(re.sub(_kept(keep), " ", note)) is not None


def names_user(note: str, *, keep: Sequence[str] = ()) -> bool:
    """Whether *note* names the user in the third person (``the user``, ``user``,
    ``the person``, ``用户``), a quoted span and each span of *keep* aside. Its "I"
    is then the assistant: "The user wants me to answer in Chinese."."""
    return _THIRD_USER.search(re.sub(_kept(keep), " ", note)) is not None


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
    """The topic of a fact the user stated (*request*: after "remember that").

    A request is filed under ``people`` only when its fact is about a person
    (:func:`_about_a_person`): "Marco's studio is on Main St", "Marco lives in
    Berlin". A rule or a fact about the work ("tests run with -n auto",
    "开发的代码一律用英文") is ``about``.
    """
    if _DECISION.search(sentence):
        return "decisions"
    if _PREFERENCE.search(sentence):
        return "preferences"
    if _FIRST_PERSON.search(sentence):
        return "about"
    return "people" if request and _about_a_person(sentence) else "about"


# What a fact starts with when a person is its subject: "Marco lives in Berlin",
# "马可住在杭州". The name must be followed by a verb, so a sentence's first
# capitalised word ("Tests run with -n auto") is not read as a name.
_SUBJECT_EN = re.compile(
    r"((?:(?:Dr|Mr|Mrs|Ms|Mx|Prof)\.\s+)?[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,2})"
    r"\s+(?:is|was|has|had|lives|lived|works|worked|studies|likes|loves|prefers|hates"
    r"|wants|needs|moved|moves|plays|teaches|coaches|manages|leads|speaks|uses|owns"
    r"|drives|goes|will|can|does|isn['’]t|doesn['’]t)\b"
)
_SUBJECT_ZH = re.compile("住|在|是|叫|姓|有|喜欢|爱|来自")
# A month is no subject even before a verb ("May is the deadline"), though some
# are names too.
_MONTH = re.compile(
    r"january|february|march|april|may|june|july|august|september|october"
    r"|november|december"
)


def _about_a_person(sentence: str) -> bool:
    """Whether the fact in *sentence* is about a person other than the user: its
    owner is a third party ("Marco's studio …", "妈妈的生日 …"), or a person is its
    subject ("Marco lives in Berlin", "马可住在杭州"). *sentence* may still start
    with the request's marker ("Remember that …", "记住：…"); the marker is not
    part of the fact."""
    fact = (request_clause(sentence) or sentence).strip()
    if (owner := _owner(fact)) is not None:
        return owner != "user"
    if found := _SUBJECT_EN.match(fact):
        name = found.group(1)
        low = name.split()[0].lower()
        if not (
            _NOT_OWNERS.fullmatch(low)
            or _VALUE_WORD.fullmatch(low)
            or _MONTH.fullmatch(low)
        ):
            return people.person(name)
    for size in range(1, 5):  # 马, 马可, 王小明: a Chinese name, then its verb
        name = fact[:size]
        if not re.fullmatch("[\u4e00-\u9fff]+", name):
            break
        if _SUBJECT_ZH.match(fact, size) and people.person(name):
            return True
    return False


# How a reply says it kept a note (:func:`claimed`); group ``x`` starts the fact.
_AT = r"(?:^|(?<=[.!?。！？\n]))\s*"
_LEAD = r"\s*(?:[—–:：,]|--?)\s*"
# Where a claim may start without a subject before it: the text, a sentence, a
# clause after "Got it," or a check mark.
_MARKS = "✅✔\ufe0f☑👍"
_LEFT = rf"(?:^|(?<=[.!?。！？\n,，、—–{_MARKS}])|(?<=\s-)|(?<=\s--))[ \t]*"
_ZH_AT = _LEFT + "(?:我)?"
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
        _ZH_AT
        + r"(?:已经?(?:记录|保存|记下)|已经记住|(?:记录|保存)(?:下来)?了"
        + r"|帮你记下(?:来)?了?)[：:，,]\s*(?P<x>.+)",
    )
)
# A claim to have saved or noted that names nothing: a sentence of its own ("Saved.",
# "Got it, noted…", "I've saved it.", "text✅ 已记录"), not a fact that follows it.
_MEMORY = r"(?: (?:to|in) (?:your |my |melo['’]s )?(?:memory|notes))?"
# What a claim may name and still name nothing: a pronoun or a generic noun.
_THING_EN = (
    r"(?:it|that|this|(?:your|the|that|this|these) "
    r"(?:preferences?|info|information|details?|requests?|settings?|choices?))"
)
_THING_ZH = (
    r"(?:(?:你|您)的)?(?:偏好|喜好|习惯|信息|要求|设置|内容|这件事|这些|这个|那个)"
)
_BARE_EN = (
    rf"(?:saved(?: {_THING_EN})?{_MEMORY}"
    rf"|noted(?: {_THING_EN})?"
    r"(?:[ \t]*[,—–-]+[ \t]*(?:thanks|thank you|thx)(?: a lot| so much| again)?)?"
    rf"|I(?:['’]ve| have) (?:saved|stored|recorded|noted)(?: {_THING_EN})?{_MEMORY}"
    rf"|I(?:['’]ll| will) remember {_THING_EN})"
    rf"(?=[ \t{_MARKS}]*(?:[.!…]+(?:\s|$)|\n|$))"
)
_BARE_ZH = (
    r"(?:我)?(?:已经?(?:记录|记住|保存|记下)了?|(?:记录|保存|记住|记下)(?:下来)?了"
    rf"|帮你记下(?:来)?了?)(?:{_THING_ZH})?(?:到记忆(?:里|中)?)?"
    rf"(?=[ \t{_MARKS}]*(?:[。！.!~～…，,、]|\n|$))"
)
# A fact that is only such a thing ("Your preferences.") is no fact.
_GENERIC = re.compile(rf"{_THING_EN}[.!]*|{_THING_ZH}[。！]*", re.IGNORECASE)
_BARE = re.compile(rf"{_LEFT}(?:{_BARE_EN}|{_BARE_ZH})", re.IGNORECASE)
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
    ``noted`` after it, ``记下了：X``, ``已记住：X``, ``记住了：X``, ``已记录：X``,
    ``已保存：X``. X ends where its sentence does ("Prof." or "Mrs." does not
    end it, :data:`_HELD`), or at a ``(``. X must read as a statement: written
    to the user (``you``, ``your``, ``yours``, ``你``, ``您``) or with a verb,
    copula or possessive (:data:`_STATEMENT`: "Prof. Duarte's office is in the
    Atrium Wing."); the facts about another person are notes too. Otherwise X
    of at least :data:`CLAIM_TERMS` content words is a claim of nothing usable
    (``""``).

    A claim with no fact after it is ``""`` too: a sentence of its own that
    says it saved or noted ("Saved.", "Noted.", "Noted, thanks!", "Got it,
    noted…", "I've saved it.", "I'll remember that.", "已记录。", "记住了！",
    "text✅ 已记录"). An acknowledgement that names no save ("Got it!") is no
    claim.
    """
    text = answer.replace("**", "").replace("__", "")
    found = [m for pattern in _CLAIMS if (m := pattern.search(text))]
    if found:
        rest = _HELD.sub(
            lambda m: m.group().rstrip() + _JOINER,
            text[min(found, key=lambda m: m.start()).start("x") :],
        )
        fact = rest[: end.start()] if (end := _CLAIM_END.search(rest)) else rest
        note = finished(_release(_NOTED_TAIL.sub("", fact)))
        if _GENERIC.fullmatch(note):
            return ""
        if _TO_USER.search(note):
            return note
        if len(note_terms(note, joining=True)) >= CLAIM_TERMS:
            return note if _STATEMENT.search(note) else ""
    return "" if _BARE.search(text) else None


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
    # ("Remember, I'm allergic to peanuts.") keeps the sentence without its marker.
    for fact in facts:
        if classify_user_text(fact).remember and (note := request_note(fact)):
            found.append(note)
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
    (:func:`candidates` keeps those).
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
    """*word* without its endings, twice, so "meetings" finds "meets"; an
    irregular verb form is its base ("kept" is "keep")."""
    return stem(stem(_IRREGULAR.get(word, word)))


def note_terms(text: str, *, joining: bool = False) -> list[str]:
    """The content words a note must find in the evidence, in order; with
    *joining*, the joining words (:data:`_JOINING`) are counted too."""
    terms: list[str] = []
    # "the user's name" restates; its "s" is not a word the user typed.
    text = _POSSESSIVE.sub("", _CJK_FRAMING.sub(" ", text))
    capitalised = {w.lower() for w in _CASED.findall(text) if w[0].isupper()}
    for term in tokenize(normalise(text)):
        if term.isascii():
            if (
                term in STOPWORDS
                or term in _FRAMING
                or (term in _JOINING and not joining and term not in capitalised)
            ):
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
    two letters longer that starts with it ("named" finds "name"), or, for an
    adverb, its base ("fluently" finds "fluent"); the words a note adds to
    restate a fact ("The user prefers…") need not be found.
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
        and not any(base in words for base in _adverb_bases(term))
    ]


def _adverb_bases(term: str) -> list[str]:
    """The bases an adverb in "-ly" can come from, stemmed like the evidence:
    "fluently" from "fluent", "happily" from "happy", "basically" from "basic".
    A base has at least four letters, so "apply" is not "app", and a word in
    "-ily" has only the base in "-y" ("family" is not "fami")."""
    if not (term.isascii() and term.endswith("ly")):
        return []
    if term.endswith("ily"):
        bases = [term[:-3] + "y"]
    else:
        bases = [term[:-2]]
        if term.endswith("ally"):
            bases.append(term[:-4])
    return [_stem(base) for base in bases if len(base) >= 4]


def missing_terms(note: str, evidence: Sequence[str]) -> list[str]:
    """The content words of *note* that none of the *evidence* texts holds."""
    return _missing(note, evidence)


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


def _quoted_cjk(note: str) -> set[str]:
    """The CJK terms of the quoted spans of *note*."""
    return {
        term
        for group in _QUOTED.findall(note)
        for span in group
        for term in tokenize(normalise(span))
        if not term.isascii()
    }


def value_terms(note: str, evidence: Sequence[str] | None = None) -> set[str]:
    """The stemmed values whose exact presence grounding requires (not paraphrases).

    A term of Latin script is a value when it is flagged (:func:`_flagged`), a
    number word or a weekday, or holds a digit. A CJK term is one when it sits in
    a quoted span, holds a number character, or, given the *evidence* texts, holds
    a character that none of them has and that is no common wording
    (:data:`_CJK_WORDING`): a name or a place is a detail the user must have
    typed, while a particle or a light verb is how the sentence is put. Without
    *evidence* every CJK term is a value (the sleep pass compares notes with
    no texts to look the characters up in).
    """
    flagged = _flagged(note)
    quoted = _quoted_cjk(note)
    seen = (
        None
        if evidence is None
        else {char for text in evidence for char in normalise(text)}
    )

    def cjk_value(term: str) -> bool:
        if seen is None or term in quoted:
            return True
        if any(char not in seen and char not in _CJK_WORDING for char in term):
            return True
        # A number with no wording beside it (周五, 五月, 十三) is typed as it is.
        return any(char in _CJK_NUMBERS for char in term) and not any(
            char in _CJK_WORDING for char in term
        )

    def is_value(term: str) -> bool:
        if not term.isascii():
            return cjk_value(term)
        return (
            term in flagged
            or _VALUE_WORD.fullmatch(term) is not None
            or any(char.isdigit() for char in term)
        )

    return {term for term in note_terms(note) if is_value(term)}


def unsupported(note: str, evidence: Sequence[str]) -> list[str]:
    """The content words of *note* that the *evidence* texts lack, when that is
    too much: ``[]`` means grounded.

    The wording may vary, the values may not. Every value (a number or a word
    with a digit, a weekday, a capitalised word inside a sentence, a quoted
    span, and in Chinese or Japanese a character that was never typed and is no
    common wording, :func:`value_terms`) must be found, and so must at least
    :data:`WORDS_FOUND` of the other content words (:func:`_missing` says when
    one is found; in Chinese or Japanese the share counts single characters, since
    one added or swapped character spoils two pairs of characters as well, and a
    pair only has to be found when it is a value). The words listed are all the
    missing ones, values first.
    """
    gone = _missing(note, evidence)
    if not gone:
        return []
    valued = value_terms(note, evidence)
    values = [term for term in gone if term in valued]
    others = [term for term in note_terms(note) if term not in valued]
    lost = [term for term in gone if term not in values]
    if not values and _counted(others) - _counted(lost) >= WORDS_FOUND * _counted(
        others
    ):
        return []
    return values + lost


def _counted(terms: Iterable[str]) -> int:
    """How many of *terms* count toward the share of words found: each word, and
    each CJK character; a CJK pair is only ever a value, never part of the share."""
    return sum(1 for term in terms if term.isascii() or len(term) == 1)


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
# What "Name's" is not a name: a contraction ("That's why …", "Everything's
# ready") or a time word.
_NOT_OWNERS = re.compile(
    r"it|that|there|here|what|who|where|when|why|how|let|this|he|she"
    r"|everything|everyone|everybody|nothing|nobody|something|someone|somebody"
    r"|anything|anyone|anybody|today|tomorrow|yesterday|tonight"
)
_NOT_NAME_ZH = re.compile(
    "[我你您他她它咱这那在是有和与把被让从到每]|^(?:今|明|昨|周末|现在)"
)


def _named(name: str) -> bool:
    """Whether *name*, before a possessive, is a person's: :func:`people.person`, or
    a Chinese nickname of one character twice ("思思", "豆豆")."""
    return people.person(name) or re.fullmatch("([\u4e00-\u9fff])\\1", name) is not None


def _owner(sentence: str) -> str | None:
    """Whose *sentence* is: ``"user"``, a third party's name, or ``None`` when its
    first words say none (or not clearly: a contraction, a time word, a word that is
    no person's name such as "Python's" or "开发的").

    A name before a possessive counts as a third party only when it is a person's
    (:func:`_named`). The trade-off: a Chinese given name without a listed surname,
    小, 老 or 阿, and that is no relative word and no reduplication ("嘉怡") is not
    seen as an owner, so a mix-up about that person is not refused by
    :func:`wrong_owner`. Losing the user's own rule to a false refusal ("开发的代码
    一律用英文" kept as "你的代码一律用英文") was the more common case.
    """
    text = sentence.strip()
    if found := _OWNER_EN.match(text):
        name = found["name"]
        if found["user"]:
            return "user"
        low = name.lower()
        bad = _NOT_OWNERS.fullmatch(low) or _VALUE_WORD.fullmatch(low)
        return None if bad or not _named(name) else name
    if found := _OWNER_ZH.match(text):
        name = found["name"]
        if found["user"]:
            return "user"
        return None if _NOT_NAME_ZH.search(name) or not _named(name) else name
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


# The sources the note gate tags a note with (``provenance["source"]``), most
# trusted first. A note the person's own words made has none.
TYPED: Final = "typed"
POINTED: Final = "pointed"
RESULT: Final = "result"
INFERRED: Final = "inferred"
# The sources of a note from material: a page or file, a finished task's result, or
# the model's own inference.
FROM_MATERIAL: Final = (POINTED, RESULT, INFERRED)

# What ``admitted_by`` is for a record the person's own words made: their message
# asked for it (``user_request``) or stated it (``user_statement``). Any other
# signal (a note from a page, a file or a result is ``outcome``) is not their words.
OWN_WORDS: Final = ("user_statement", "user_request")


def by_person(entry: StoredEntry) -> bool:
    """Whether the person's own words made *entry*: its signal is one of
    :data:`OWN_WORDS` and its provenance names no other source (a note kept from a
    page, a file or a result says so there, whatever signal it carries)."""
    return entry.admitted_by in OWN_WORDS and not entry.provenance.get("source")


def standing(entry: StoredEntry) -> int:
    """How far *entry* is the person's own, for what may replace it and for which
    note of a subject recall shows as the current one.

    :data:`TAKEN`: a note from a page, a file or a result as melo took it (its signal
    is ``outcome`` or its source is one of :data:`FROM_MATERIAL`; also one kept
    before sources were written). :data:`REWORDED`: such a note that the person
    edited by hand. :data:`OWN`: any other note: one they typed or asked for, one
    sleep wrote, an imported one, edited or not."""
    from_material = (
        entry.admitted_by == "outcome"
        or entry.provenance.get("source") in FROM_MATERIAL
    )
    if not from_material:
        return OWN
    return REWORDED if entry.provenance.get("edited") else TAKEN


def kept_from_replacement(entry: StoredEntry) -> bool:
    """Whether a note from a page, a file or a result must not replace *entry*: it
    stands above :data:`TAKEN` (:func:`standing`). An older note from a page, a file
    or a result that nobody edited is replaced."""
    return standing(entry) > TAKEN


def may_replace(admitted_by: str, old: StoredEntry) -> bool:
    """Whether a note admitted as *admitted_by* may replace *old*: not when it is
    from a page, a file or a result (``outcome``) and *old* is kept from replacement."""
    return admitted_by != "outcome" or not kept_from_replacement(old)


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


def notes_of_replaced(items: Iterable[Item]) -> dict[str, Item]:
    """The notes that say what a replaced memory item says, each note's key to
    that item.

    A note is of an item when the note's own turn (its session and turn, a
    turn of 1 or later) is one of the item's sources and most of the note's
    content words are the item's (:data:`SUPERSEDE_OVERLAP`, the test of
    :func:`covered`). A note of another fact of that turn, one of the same words
    from another turn, an import or a sleep result (no turn) is not. No model
    call.
    """
    items = list(items)
    notes: dict[tuple[str, int], list[Item]] = {}
    for item in items:
        if item.is_record and item.session_id and item.turn >= 1:
            notes.setdefault((item.session_id, item.turn), []).append(item)
    out: dict[str, Item] = {}
    for old in items:
        if not (old.is_memory and old.replaced_by):
            continue
        said = content_tokens(old.text)
        for source in old.sources:
            for note in notes.get(source, ()):
                if overlap(content_tokens(note.text), said) >= SUPERSEDE_OVERLAP:
                    out.setdefault(note.key, old)
    return out


def newer_values_first(items: Sequence[Item]) -> list[Item]:
    """*items* with the records of one subject (:func:`subject_key`) that state
    different values listed newest first, each older one marked
    :data:`~localmelo.support.memory.recall.OLDER` and kept, so a reader of the
    recall block can tell which value is the current one when the question asks
    about the past too.

    The records of a subject take the places they had, the newest in the first;
    a record that says the newest value again is not marked. A record that reads
    as older already (:func:`~localmelo.support.memory.recall.newer_first`) is
    left as it is. Other items stay where they were.

    A record is marked older only by a newer one that stands at least as high
    (``standing``: :func:`standing`), and the records come highest first. So a
    record from a page, a file or a result never marks the person's own older, nor
    one from a page that they reworded; a reworded one never marks their own; and
    the person's own newest marks every older record of the subject, as ever. Each
    record shows its source.
    """
    groups: dict[str, list[int]] = {}
    for at, item in enumerate(items):
        if item.is_record and not item.older and (key := subject_key(item.text)):
            groups.setdefault(key, []).append(at)
    out = list(items)
    for slots in groups.values():
        found = [items[at] for at in slots]
        values = {normalise(item.text) for item in found}
        if len(found) < 2 or len(values) < 2:
            continue
        ranked = sorted(
            found,
            key=lambda item: (item.standing, *_newness(item)),
            reverse=True,
        )
        # What replaces a record: the newest of those that stand at least as high.
        current = {
            level: normalise(
                max((i for i in found if i.standing >= level), key=_newness).text
            )
            for level in {item.standing for item in found}
        }
        for at, item in zip(slots, ranked, strict=True):
            same = normalise(item.text) == current[item.standing]
            out[at] = item if same else dataclasses.replace(item, older=OLDER)
    return out


def _newness(item: Item) -> tuple[float, int]:
    """What orders two records of a subject by age: the time, then the number."""
    return item.created_at, _record_number(item.key)


def _record_number(key: str) -> int:
    """The number in a record item's key (``r:12``); 0 for none."""
    return int(key[2:]) if key[2:].isdigit() else 0


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

    A candidate admitted as ``outcome`` (a note from a page, a file or a result)
    replaces only an older note that is itself from a page, a file or a result and
    was not reworded by hand (:func:`kept_from_replacement`); beside any other note
    (one the person typed or asked for, one sleep wrote, an imported one) it is
    written as a note of its own, each with its source. Any other candidate still
    replaces any older note of its subject.
    """
    written: list[Written] = []
    for cand, vector in zip(found, vectors, strict=True):
        active = await store.active_entries()
        action, target = decide(
            cand.text, active, kind=cand.kind, vector=vector, embed_model=embed_model
        )
        if action == "UPDATE" and any(
            e.id == target and not may_replace(cand.admitted_by, e) for e in active
        ):
            action, target = decide(
                cand.text,
                [e for e in active if may_replace(cand.admitted_by, e)],
                kind=cand.kind,
                vector=vector,
                embed_model=embed_model,
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
