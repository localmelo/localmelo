"""What the person's own words say about posting, and what a post may hold (#547).

A step that posts or sends something in the person's name (a comment on a
forum, a mail, a pull request, a push) runs in Auto only when their own words
ask for it. The user's model reads those words (``user_asked``); these
functions are the part code can prove, from the person's messages and the text
the step sends and never from a page or a tool, so a model that a page talked
into saying "the user asked" still does not post:

- :func:`asks_to_send`: the words speak of posting or sending at all. A page's
  instruction to post is in no message of the person's, so it has no such words
  to lean on.
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
    "POST",
    "account_names",
    "asks_for_review",
    "asks_to_send",
    "asks_to_share",
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

# Verbs of posting or sending, and "open an issue / a pull request". A
# noun alone ("this mail") is no cue: "deal with this mail" does not say to send.
_SEND = re.compile(
    r"\b(post(?:ing|ed|s)?|publish\w*|repl(?:y|ies|ied|ying)|comment\w*|send\w*"
    r"|e-?mail\w*|messag(?:e|ing|ed)"
    r"|dm|tweet\w*|share|sharing|announce\w*|submit\w*|push\w*|upload\w*|review\w*"
    r"|rate|rating|forward\w*|respond\w*)\b"
    r"|\b(?:open|create|file|raise|make|write|log)\w*\s+(?:an?\s+|the\s+|this\s+|new\s+)*"
    r"(?:issue|pull request|pr)\b"
    r"|发布|发送|发帖|发邮件|发消息|发给|发表|发条|发个|回复|评论|留言|私信|转发|推送|提交|分享|点评|评价|上传|通知",
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
    r"\bshar(?:e|ed|es|ing)\b|\bpublic(?:ly)?\b|\bpublish\w*|\bpost(?:ing|ed)?\b"
    r"|\bsocial\b|分享|公开|发布|发帖|朋友圈",
    re.IGNORECASE,
)

# A phone number is written in a shape (international, North American, or a
# Chinese mobile number), never as any run of digits: dates, ports and ids are
# not phone numbers.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w.])"
    r"|(?<![\w.])\+\d{8,14}(?![\w.])"
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


def _squash(text: str) -> str:
    return " ".join(text.casefold().split())


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


def asks_to_send(texts: Iterable[str]) -> bool:
    """Whether the person's words speak of posting, sending, replying or opening an issue or a pull request."""
    return any(_SEND.search(text) for text in texts)


def asks_for_review(texts: Iterable[str]) -> bool:
    """Whether the person's words ask for a review or a rating."""
    return any(_REVIEW.search(text) for text in texts)


def repeats(texts: Iterable[str]) -> bool:
    """Whether the person's words make the task a repeating one ("every Monday", "on Mondays")."""
    return any(_REPEAT.search(text) for text in texts)


def asks_to_share(texts: Iterable[str]) -> bool:
    """Whether the person's words speak of sharing, publishing or making something public."""
    return any(_SHARE.search(text) for text in texts)


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


def sent_texts(tool_name: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """The text a step puts out: what a browser types or sends in a form, or a shell command.

    Only the arguments the tool itself chose (``text``; the values of
    ``fields``; ``command``): never a page's own words.
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
        out.append(
            " ".join(map(str, command)) if isinstance(command, list) else str(command)
        )
    return tuple(out)


def post_stop(
    texts: Sequence[str], sent: Sequence[str], names: Sequence[str] = ()
) -> str:
    """Why a post the model let through must ask first, or ``""`` when nothing here stops it.

    In order: the person's words never speak of posting or sending, so whatever
    told melo to post was not them (a page, a tool, the model's own idea); the
    text holds their phone number, address or real name; they asked for a
    review or a rating that they did not write; the task repeats.
    """
    if not asks_to_send(texts):
        return "Posts or sends something your messages do not ask for."
    found = personal_data("\n".join(sent), names)
    if found:
        return f"Sends text that holds your {' and '.join(found)}."
    if asks_for_review(texts) and not written_by(sent, texts):
        return "Posts a review or a rating that speaks for you."
    if repeats(texts):
        return "Posts for a task that repeats, which asks each time."
    return ""
