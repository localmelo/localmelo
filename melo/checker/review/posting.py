"""What the person's own words say about posting, and what a post may hold (#547).

A step that posts or sends something in the person's name (a comment on a
forum, a mail, a pull request, a push) runs in Auto only when their own words
ask for it. The user's model reads those words (``user_asked``); these
functions are the part code can prove, from the person's messages and the text
the step sends and never from a page or a tool, so a model that a page talked
into saying "the user asked" still does not post:

- :func:`asks_to_send`: the words ask for a send (a verb of posting or sending
  in a request, not the noun "email" or "a post"), unless the person wrote the
  text themselves (:func:`written_by`). A page's instruction to post is in no
  message of the person's, so it has no such words to lean on.
- :func:`is_post`: which steps these checks apply to: a text melo typed that is
  a message rather than a field (:func:`is_message`), or a command that posts,
  whatever the review model called the step.
- :func:`personal_data`: the text holds a phone number, a street address, or the
  real name of this Mac's account. Only those three can be matched by code;
  anything else about the person that melo knows (its notes) is not read here.
- :func:`asks_for_review`: the words ask for a review or a rating, which speaks
  for the person and is public: ask, unless the person wrote the text
  (:func:`written_by`).
- :func:`repeats`: the task repeats ("every Monday"). melo keeps no store of
  what was approved once, so a repeating post asks each time.
- :func:`asks_to_share`: the words speak of sharing or making something public,
  the one thing that lets a pre-ticked "public" box stay ticked.
- :func:`asks_to_attach`: the words ask to send a file or an application (the
  verbs of a send, and "attach" or "apply"), the one thing that lets melo attach
  a file to a page's upload field without a card.
- :func:`asks_to_search`: the words ask to search or look something up, the one
  thing that lets a search box take a query of any length: a search is no post,
  whatever it holds, but only a box the person asked to search in is one.

The words are cues, matched on the person's text; they are a necessary
condition that fails closed, never a licence: the model's verdict still
decides. Standard library only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

__all__ = [
    "FIELD_CHARS",
    "POST",
    "account_names",
    "asks_for_review",
    "asks_to_attach",
    "asks_to_search",
    "asks_to_send",
    "asks_to_share",
    "asks_to_write",
    "file_bodies",
    "is_field",
    "is_message",
    "is_post",
    "personal_data",
    "post_stop",
    "repeats",
    "sent_texts",
    "written_by",
]

# The subject of a ``send_out`` finding of the rules that posts or sends a
# message in the person's name (``git push``, ``gh issue create``, ``mail``).
POST: Final = "post"

# Verbs of posting or sending, and "open an issue / a pull request". Only a
# word used as a request counts: "deal with this mail", "my email", "a post"
# and "the review" are nouns. A verb that is seldom a noun (``strong``) counts
# unless a determiner stands before it; one that often is (``weak``: "post",
# "comment", "email", "message") counts only where a verb stands, at the start
# of a sentence, after "please", "then", "to"..., or before its object ("post
# the report"); ``plain`` always counts.
_SEND = re.compile(
    r"\b(?:(?P<strong>send|sending|publish|publishing|tweet|tweeting|announce|announcing"
    r"|submit|submitting|upload|uploading|forward|forwarding|push|pushing|respond|responding"
    r"|reply|replying|dm|share|sharing)"
    r"|(?P<weak>post|posting|comment|commenting|e-?mail|e-?mailing|message|messaging))\b"
    r"|(?P<plain>\b(?:open|create|file|raise|make|write|log)\w*\s+(?:an?\s+|the\s+|this\s+|new\s+)*"
    r"(?:issue|pull request|pr)\b"
    r"|发布|发送|发帖|发邮件|发消息|发给|发表|发条|发个|回复|私信|转发|推送|提交|分享|上传|留言)",
    re.IGNORECASE,
)
# The words before a verb that make it a noun.
_DETERMINERS: Final = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "their",
        "our",
        "its",
        "new",
        "another",
        "each",
        "every",
        "any",
        "no",
        "some",
        "same",
        "of",
    ]
)
# The words that follow a verb of a request and make it a verb: "post the report".
_OBJECTS: Final = frozenset(
    [
        "the",
        "a",
        "an",
        "this",
        "that",
        "it",
        "them",
        "him",
        "her",
        "me",
        "us",
        "my",
        "your",
        "our",
        "his",
        "their",
    ]
)
# The words a verb of a request can follow (besides the start of a sentence).
_VERB_AFTER: Final = frozenset(
    [
        "please",
        "then",
        "and",
        "or",
        "to",
        "just",
        "also",
        "now",
        "can",
        "could",
        "will",
        "would",
        "should",
        "must",
        "you",
        "i",
        "we",
        "let's",
        "lets",
        "go",
        "ahead",
        "kindly",
        "first",
        "next",
        "finally",
        "too",
        "not",
        "don't",
        "never",
        "need",
        "want",
    ]
)
# Typing into a box is writing, not yet sending (a form is sent by its own
# question): for typed text the words may ask to write, fill in or type.
_WRITE = re.compile(
    r"\b(?P<strong>write|writing|type|typing|enter|entering|fill|filling|compose|composing"
    r"|draft|drafting|leave|leaving|put|putting|add|adding)\b|(?P<plain>填写|输入|写下|写个|写一)",
    re.IGNORECASE,
)
# Verbs of attaching a file or applying (an application is sent with its files).
# "apply" as a noun never stands alone; "an attachment" and "the attached file"
# are not requests: a determiner before the verb makes it a noun, as for :data:`_SEND`.
_ATTACH = re.compile(
    r"\b(?P<strong>attach|attaching|apply|applying)\b"
    r"|(?P<plain>附上|附加|附件|上传|投递|申请|报名)",
    re.IGNORECASE,
)
_REVIEW = re.compile(
    r"\breview(?:s)?\b|\brat(?:e|ing|ings)\b|\bfeedback\b|\btestimonial|\b\d[- ]?stars?\b"
    r"|评价|评分|点评|好评|差评|打分",
    re.IGNORECASE,
)
_WEEKDAY = r"(?:mon|tues|wednes|thurs|fri|satur|sun)days?"
_REPEAT = re.compile(
    r"\b(?:every|each)\s+(?:other\s+|single\s+)?"
    rf"(?:day|night|morning|evening|week|weekday|weekend|month|hour|{_WEEKDAY})\b"
    rf"|\bon\s+(?:mon|tues|wednes|thurs|fri|satur|sun)days\b"
    r"|\brecurring\b|\bregularly\b|\bon a schedule\b|\bevery time\b"
    r"|每(?:天|日|周|个?星期|月|小时)|定期|定时",
    re.IGNORECASE,
)
_SHARE = re.compile(
    r"\b(?:(?P<strong>publish\w*|tweet\w*|announce\w*|share|sharing)|(?P<weak>post|posting))\b"
    r"|(?P<plain>\bpublicly\b|\b(?:make|set|keep|leave|turn|mark)\b[^.!?\n]{0,30}\bpublic\b"
    r"|分享|公开|发布|发帖|朋友圈)",
    re.IGNORECASE,
)

# Verbs of searching. A word before it that makes it a noun ("the search box",
# "a find") does not count, as for :data:`_SEND`.
_SEARCH = re.compile(
    r"\b(?P<strong>search|searching|searches|google|googling|find|finding|lookup"
    r"|look(?:ing)?\s+(?:up|for)|browse|browsing)\b"
    r"|(?P<plain>搜|查一下|查一查|查查|查找|查询|找一下|找找|找一找)",
    re.IGNORECASE,
)

# A phone number is written in a shape (international, North American, or a
# Chinese mobile number), never as any run of digits: dates, ports and ids are
# not phone numbers.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\w)(?!\.\w)"
    r"|(?<![\w.])\+\d{8,14}(?!\w)(?!\.\w)"
    r"|(?<!\d)1[3-9]\d{9}(?!\d)"
)
_STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][\w'.-]*\s+){1,3}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pl|Place)\b"
    r"|\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"
    r"|[一-鿿]{2,}(?:省|市|区|县|镇|街道|路|街|巷|弄)[一-鿿\d]*\d+\s*(?:号|栋|幢|楼|室|单元)"
)
# The shortest piece of sent text that can prove the person wrote it.
MIN_PROOF: Final = 12
# The longest single line melo types into a box as "a field's worth": a search,
# a name, a title. A longer text, or one with a line break, is a message; so is
# a shorter one of this many words, which reads as a sentence.
FIELD_CHARS: Final = 80
MESSAGE_WORDS: Final = 5


def _squash(text: str) -> str:
    return " ".join(text.casefold().split())


def is_field(text: str) -> bool:
    """Whether *text* is a field's worth: one line of at most :data:`FIELD_CHARS` characters."""
    text = text.strip()
    return len(text) <= FIELD_CHARS and "\n" not in text and "\r" not in text


def is_message(texts: Iterable[str]) -> bool:
    """Whether a text is a message rather than a field: more than a field's worth, or a sentence of :data:`MESSAGE_WORDS` words or more."""
    return any(
        not is_field(text) or len(text.split()) >= MESSAGE_WORDS for text in texts
    )


def is_post(
    arguments: Mapping[str, Any], evidence: Iterable[tuple[str, str]] = ()
) -> bool:
    """Whether a step posts or sends a message in the person's name: text typed or a form sent from a page, or a command the rules mark so.

    A POST of data to a server, an upload or an API call is a send of data, not
    a post: the words it needs are the user's model's to read, as before.
    """
    return arguments.get("access") in ("type", "submit") or (
        "send_out",
        POST,
    ) in tuple(evidence)


def _word_before(text: str, start: int) -> str:
    """The word before ``text[start]`` in lower case, or ``""`` at the start of a sentence or clause."""
    head = text[:start].rstrip()
    if not head or head[-1] in ".!?;:,\n。！？；：，":
        return ""
    found = re.search(r"([\w'’-]+)$", head)
    return found.group(1).casefold() if found else ""


def _word_after(text: str, end: int) -> str:
    """The word after ``text[:end]`` in lower case, or ``""``."""
    found = re.match(r"\s+([\w'’-]+)", text[end:])
    return found.group(1).casefold() if found else ""


def _asks(pattern: re.Pattern[str], texts: Iterable[str]) -> bool:
    """Whether a text has a match of *pattern* used as a request (see :data:`_SEND`)."""
    for text in texts:
        for found in pattern.finditer(text):
            before = _word_before(text, found.start())
            groups = found.groupdict()
            if groups.get("plain") is not None:
                return True
            if groups.get("strong") is not None and before not in _DETERMINERS:
                return True
            if groups.get("weak") is not None and (
                not before
                or before in _VERB_AFTER
                or before not in _DETERMINERS
                and _word_after(text, found.end()) in _OBJECTS
            ):
                return True
    return False


def asks_to_send(texts: Iterable[str]) -> bool:
    """Whether the person's words ask to post, send, reply or open an issue or a pull request."""
    return _asks(_SEND, texts)


def asks_to_attach(texts: Iterable[str]) -> bool:
    """Whether the person's words ask to send a file to a site: to attach, upload or send it, or to apply."""
    texts = list(texts)
    return _asks(_ATTACH, texts) or asks_to_send(texts)


def asks_to_search(texts: Iterable[str]) -> bool:
    """Whether the person's words ask to search or look something up."""
    return _asks(_SEARCH, texts)


def asks_to_write(texts: Iterable[str]) -> bool:
    """Whether the person's words ask to write, type or fill something in."""
    return _asks(_WRITE, texts)


def asks_for_review(texts: Iterable[str]) -> bool:
    """Whether the person's words ask for a review or a rating."""
    return any(_REVIEW.search(text) for text in texts)


def repeats(texts: Iterable[str]) -> bool:
    """Whether the person's words make the task a repeating one ("every Monday", "on Mondays")."""
    return any(_REPEAT.search(text) for text in texts)


def asks_to_share(texts: Iterable[str]) -> bool:
    """Whether the person's words ask to share, publish or make something public."""
    return _asks(_SHARE, texts)


def account_names() -> tuple[str, ...]:
    """The real name of this Mac's account, as the system keeps it (empty when it has none)."""
    try:
        import pwd

        name = pwd.getpwuid(os.getuid()).pw_gecos.split(",")[0].strip()
    except (ImportError, KeyError, OSError):
        return ()
    wide = re.search(r"[\u4e00-\u9fff]", name) is not None
    return (name,) if len(name) >= 2 and (wide or " " in name) else ()


def personal_data(text: str, names: Sequence[str] = ()) -> tuple[str, ...]:
    """What *text* holds of the person that code can match: ``phone``, ``address`` and ``name``.

    A phone number or a street address by its written shape, and any of *names*
    (the account's real name) as a whole phrase. Nothing else melo knows of the
    person is matched, and a name or address in some other shape is not.
    """
    found: list[str] = []
    if _PHONE.search(text):
        found.append("phone number")
    if _STREET.search(text):
        found.append("address")
    squashed = f" {_squash(text)} "
    for name in names:
        wide = re.search(r"[\u4e00-\u9fff]", name) is not None
        said = _squash(name)
        if said and (said in squashed if wide else f" {said} " in squashed):
            found.append("name")
            break
    return tuple(found)


def written_by(sent: Sequence[str], texts: Iterable[str]) -> bool:
    """Whether the person wrote all the text a step sends: each piece is inside one of their messages.

    A piece shorter than :data:`MIN_PROOF` proves nothing, and no piece to show
    proves nothing either: then the text is not taken for the person's.
    """
    pieces = [_squash(piece) for piece in sent if len(piece.strip()) >= MIN_PROOF]
    wrote = [_squash(text) for text in texts]
    return bool(pieces) and all(
        any(piece in text for text in wrote) for piece in pieces
    )


# Options and a redirection that take a message from a file: ``--body-file``,
# ``--notes-file``, ``-F``, ``--input`` (``gh``), ``name=@file`` of a field,
# ``< file`` (``mail``).
_BODY_FILE = re.compile(
    r"""(?:^|\s)(?:--body-file|--notes-file|--input|-F)(?:\s+|=)(?!\w+=@)['"]?([^\s'"|;&<>]+)"""
    r"""|(?:^|\s)(?:-F|--field)\s+\w+=@['"]?([^\s'"|;&<>]+)"""
    r"""|(?:^|\s)<\s*['"]?([^\s'"|;&<>]+)"""
)
# The most of a message file that is read.
MAX_BODY_BYTES: Final = 65536


def file_bodies(command: str, workspace: str) -> tuple[str, ...]:
    """The text of the files a post command takes its message from, at most :data:`MAX_BODY_BYTES` each.

    A relative path is the workspace's. A file that is missing, not a regular
    file or unreadable adds nothing: the command fails on its own.
    """
    out: list[str] = []
    for found in _BODY_FILE.finditer(command):
        name = next(g for g in found.groups() if g)
        path = os.path.join(workspace, os.path.expanduser(name))
        try:
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as file:
                    out.append(file.read(MAX_BODY_BYTES))
        except OSError:
            continue
    return tuple(out)


def sent_texts(
    tool_name: str, arguments: Mapping[str, Any], workspace: str = ""
) -> tuple[str, ...]:
    """The text a step puts out: what a browser types or sends in a form, or a shell command.

    Only the arguments the tool itself chose (``text``; the values of
    ``fields``; ``command``): never a page's own words. With a *workspace*, the
    files a shell command takes its message from (:func:`file_bodies`) are read
    too, so a body in a file is checked like one on the command line.
    """
    out: list[str] = []
    text = arguments.get("text")
    if isinstance(text, str):
        out.append(text)
    fields = arguments.get("fields")
    if isinstance(fields, list):
        out += [
            str(row[1])
            for row in fields
            if isinstance(row, list | tuple)
            and len(row) > 1
            and str(row[1]) not in ("", "on", "off", "(hidden)")
        ]
    command = arguments.get("command")
    if tool_name == "shell_exec" and command:
        line = (
            " ".join(map(str, command)) if isinstance(command, list) else str(command)
        )
        out.append(line)
        if workspace:
            out += file_bodies(line, workspace)
    return tuple(out)


def post_stop(
    texts: Sequence[str],
    sent: Sequence[str],
    names: Sequence[str] = (),
    *,
    typing: bool = False,
) -> str:
    """Why a post the model let through must ask first, or ``""`` when nothing here stops it.

    In order: they asked for a review or a rating that they did not write; their
    words do not ask for a send (for text being *typed*, to write it) and they
    did not write the text, so whatever told melo to post was not them (a page, a
    tool, the model's own idea); the text holds their phone number, address or
    real name; the task repeats.
    """
    wrote = written_by(sent, texts)
    if asks_for_review(texts) and not wrote:
        return "Posts a review or a rating that speaks for you."
    asked = asks_to_send(texts) or typing and asks_to_write(texts)
    if not wrote and not asked:
        return "Posts or sends something your messages do not ask for."
    found = personal_data("\n".join(sent), names)
    if found:
        return f"Sends text that holds your {' and '.join(found)}."
    if repeats(texts):
        return "Posts for a task that repeats, which asks each time."
    return ""
