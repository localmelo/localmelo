"""One conservative repair of a reflection reply's JSON, and the decoder's complaint.

A reflection is JSON that a model wrote, and one stray character can make a
whole reply unreadable. :func:`repairs` lists the texts that a reply becomes
with one fix of five kinds, and :func:`error_line` is the line that a retry
adds to say what the decoder rejected. Neither decides what counts as a
reflection: the caller reads each repaired text as it reads any reply, so a
repair that gives no object with a valid ``recommended_action`` is no repair.
A fix deletes a character, escapes some, or inserts one backslash or one
closing bracket; nothing else is ever inserted or closed, and a reply that
needs two fixes is not repaired.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

# The repairs, in the order they are tried: a closing bracket or brace that
# no opener matches, a comma before a closer, raw newline, carriage return
# and tab characters inside strings, a backslash before a character that JSON
# has no escape for, and an array that the next key breaks off.
REFLECTION_REPAIRS = (
    "unmatched_closer",
    "trailing_comma",
    "control_char",
    "invalid_escape",
    "unclosed_array",
)

# Characters of the reply that a retry quotes on each side of the decoder's
# error.
ERROR_CONTEXT_CHARS = 30

_CLOSERS = ("]", "}")
_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)
# A string, a bracket, brace, comma or colon, or a word such as ``true`` or a number.
_TOKEN = re.compile(_STRING.pattern + r'|[\[\]{},:]|[^\s"\[\]{},:]+', re.DOTALL)
_ESCAPES = str.maketrans({"\n": "\\n", "\r": "\\r", "\t": "\\t"})


def rejected_objects(body: str) -> Iterator[tuple[int, json.JSONDecodeError]]:
    """Each ``{`` in *body* at which the decoder fails, with its error.

    They come in the order that ``_json_values`` (in ``agent.py``) meets
    them: after a ``{`` that does not start an object the search goes on
    from where the decoder gave up, and after an object, from its end.
    None when *body* is JSON as a whole, and none after a nesting or a
    number that the decoder cannot read.
    """
    try:
        json.loads(body)
        return
    except (ValueError, RecursionError):
        pass
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            _, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError as exc:
            yield start, exc
            start = body.find("{", max(exc.pos, start + 1))
        except (ValueError, RecursionError):
            return
        else:
            start = body.find("{", end)


def _comma_before_key(body: str, start: int, pos: int) -> int:
    """The comma before the key that an unclosed array swallowed, else ``-1``.

    The decoder read *body* from *start* as far as the colon at *pos*, so
    that text is valid JSON. The colon follows a key that an array swallowed
    when the innermost open value is an array whose last item is a string
    after a comma.
    """
    openers = ""
    before = last = ("", -1)  # the last two tokens, each with where it starts
    for match in _TOKEN.finditer(body, start, pos):
        token = match.group()
        if token in ("[", "{"):
            openers += token
        elif token in _CLOSERS:
            openers = openers[:-1]
        before, last = last, (token, match.start())
    if openers[-1:] == "[" and before[0] == "," and last[0][:1] == '"':
        return before[1]
    return -1


def repairs(
    body: str, start: int, error: json.JSONDecodeError
) -> Iterator[tuple[str, str]]:
    """The texts that *body* becomes with one repair, each with its name.

    *error* is the decoder's failure on the ``{`` at *start*. The repairs
    are those of :data:`REFLECTION_REPAIRS`, each only where the decoder
    stopped: ``unmatched_closer`` deletes the ``]`` or ``}`` at the error,
    ``trailing_comma`` the comma before a closer, ``control_char``
    escapes raw newlines, carriage returns and tabs in the strings from
    *start* on, when that is what the decoder met, ``invalid_escape`` doubles
    the backslash of an escape that JSON does not have (``\\;`` reads as the
    two characters ``\\;``), and ``unclosed_array`` inserts a ``]`` before the
    comma that precedes the string at which the decoder met a colon inside an
    array. Python before 3.13 reports a trailing comma at the closer after
    it, and 3.13 at the comma, so the character before the error is looked at
    too.
    """
    pos = error.pos
    for index in dict.fromkeys((pos, len(body[:pos].rstrip()) - 1)):
        char = body[index] if 0 <= index < len(body) else ""
        if char in _CLOSERS:
            yield "unmatched_closer", body[:index] + body[index + 1 :]
        elif char == "," and body[index + 1 :].lstrip()[:1] in _CLOSERS:
            yield "trailing_comma", body[:index] + body[index + 1 :]
    if error.msg.startswith("Invalid control character"):
        tail = _STRING.sub(lambda m: m.group().translate(_ESCAPES), body[start:])
        if tail != body[start:]:
            yield "control_char", body[:start] + tail
    elif error.msg.startswith("Invalid \\escape"):
        yield "invalid_escape", body[:pos] + "\\" + body[pos:]
    elif error.msg.startswith("Expecting ',' delimiter") and body[pos : pos + 1] == ":":
        comma = _comma_before_key(body, start, pos)
        if comma != -1:
            yield "unclosed_array", body[:comma] + "]" + body[comma:]


def error_line(body: str, error: json.JSONDecodeError, shift: int = 0) -> str:
    """The line a retry adds after its note: the decoder's error and its context.

    It names the error and its character, *shift* more than the decoder's
    position when *body* does not start the reply, and quotes up to
    ``2 * ERROR_CONTEXT_CHARS`` characters of *body* around it. Runs of
    whitespace in the quote become one space, so the line stays one line.
    """
    low = max(0, error.pos - ERROR_CONTEXT_CHARS)
    high = error.pos + ERROR_CONTEXT_CHARS
    quote = " ".join(body[low:high].split())
    where = f"{error.msg.removesuffix(' at')} at character {shift + error.pos}"
    return (
        f"Your reply was not valid JSON ({where}: {'...' if low else ''}{quote}"
        f"{'...' if high < len(body) else ''}). Reply again with one JSON object."
    )
