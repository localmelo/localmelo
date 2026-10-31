"""A note an assistant wrote about the user, put to the user in the second person.

The notes in an assistant's own memory (its note files, its memory database) are
written by the assistant: "I" and "me" are the assistant, and "the user" is the
person. :func:`to_you` says the user as "you" where the user is the subject or the
owner (with its verb: "the user prefers" becomes "you prefer", "the user's repo"
becomes "your repo") and leaves "I" and "me" alone, since turning them into "you"
would claim the assistant's own doings for the person. "Ask the user first" keeps
its words: it is an instruction to the assistant. A leading "I" that only
introduces an instruction ("I should ask first") is dropped where what is left
still reads as one ("Ask first."). Files the user wrote keep
:func:`records.second_person`, where "I" is the user.
"""

from __future__ import annotations

import re

_ADVERBS = (
    r"(?:(?:also|always|never|usually|often|currently|mostly|generally|typically"
    r"|really|sometimes|still|only|just|already|rarely|strongly|explicitly)\s+)*"
)
# A subject follows the start of the text, a sentence or clause break, or a conjunction.
_SUBJECT = re.compile(
    r"(?P<lead>^|(?<=[.!?])\s+|[,;:]\s+|\b(?:that|and|but|when|whenever|if|because|while"
    r"|since|so|then|unless|although|where)\s+)"
    r"(?:the\s+)?user\b(?!['’]s)(?P<mid>\s+" + _ADVERBS + r")(?P<verb>[A-Za-z']+)",
    re.IGNORECASE,
)
_POSSESSIVE = re.compile(r"(?:\bthe\s+|^)user['’]s\b", re.IGNORECASE)
# 用户 at the start of a clause, or as a possessor ("用户的邮箱").
_ZH_USER = re.compile(
    "(?:^|(?<=[，,；;。！？]))用户(?=的|偏好|喜欢|希望|要求|习惯|使用|倾向|需要|想|也|都|只|还|更|不|会|是|有|已|正在|常|总是|通常)"
)
_CONJOINED = re.compile(
    r"(?P<conj>\b(?:and|but|or)\s+)(?P<mid>" + _ADVERBS + r")(?P<verb>[A-Za-z']+)",
    re.IGNORECASE,
)
_IRREGULAR = {
    "is": "are",
    "was": "were",
    "has": "have",
    "does": "do",
    "goes": "go",
    "doesn't": "don't",
    "isn't": "aren't",
    "wasn't": "weren't",
    "hasn't": "haven't",
}
_MODALS = re.compile(
    r"will|can|should|must|may|might|would|could|shall|cannot|can't|won't|wouldn't"
    r"|shouldn't|couldn't|mustn't|don't|aren't|weren't|haven't"
)
# Verbs an assistant's notes say of the user, in their "-s" form.
_VERBS = re.compile(
    r"prefers|wants|likes|dislikes|hates|loves|uses|works|runs|asks|tells|expects"
    r"|owns|lives|keeps|calls|needs|writes|reads|treats|gives|takes|makes|lets|hands"
    r"|opens|approves|decides|trusts|knows|thinks|believes|says|speaks"
    r"|studies|teaches|maintains|manages|wishes|hopes|avoids|accepts"
    r"|rejects|shares|sends|pushes|merges|deploys"
    r"|follows|enjoys|gets|starts|stops|tries|finds|sees|cares|insists"
    r"|requires"
)
_I_MODAL = re.compile(
    r"^I(?:\s+(?:should|must|shall|will|need to|have to|ought to)|'ll)\s+(?!not\b)(?=\S)"
)
_I_NOT = re.compile(r"^I\s+(?:do not|should not|must not|shall not|will not)\s+")
_I_NOT_SHORT = re.compile(r"^I\s+(?:don't|shouldn't|mustn't|shan't|won't)\s+")
_I_ADVERB = re.compile(
    r"^I\s+(?=(?:always|never|usually|first|also)\s+(?!(?:am|was|were|have|had|did"
    r"|think|believe|feel|know|guess|hope|wonder|mean|can|could|would|might|may)\b)"
    r"[a-z])"
)


def _base(verb: str) -> str | None:
    """The form of *verb* after "you", or ``None`` when *verb* is not one."""
    low = verb.lower()
    if low in _IRREGULAR:
        return _IRREGULAR[low]
    if _MODALS.fullmatch(low) or (low.endswith("ed") and len(low) > 3):
        return verb
    if not _VERBS.fullmatch(low):
        return None
    if low.endswith("ies"):
        return verb[:-3] + "y"
    if low.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
        return verb[:-2]
    return verb[:-1]


def _subject(found: re.Match[str]) -> str:
    verb = _base(found["verb"])
    if verb is None:
        return found[0]
    lead = found["lead"]
    you = "You" if not lead or lead.rstrip()[-1:] in ".!?" else "you"
    return f"{lead}{you}{found['mid']}{verb}"


def _conjoined(found: re.Match[str]) -> str:
    verb = _base(found["verb"])
    return found[0] if verb is None else f"{found['conj']}{found['mid']}{verb}"


def to_you(text: str) -> str:
    """*text*, written by an assistant about the user, as a note to the user."""
    text = _POSSESSIVE.sub("your", text)
    said = _SUBJECT.sub(_subject, text)
    if said != text:  # "the user prefers X and asks Y": both verbs follow "you"
        said = _CONJOINED.sub(_conjoined, said)
    text = said
    text = _ZH_USER.sub("你", text)
    for pattern, lead in (
        (_I_NOT, "Do not "),
        (_I_NOT_SHORT, "Don't "),
        (_I_MODAL, ""),
        (_I_ADVERB, ""),
    ):
        dropped = pattern.sub(lead, text, count=1)
        if dropped != text:
            return dropped
    return text
