"""One conservative repair of a reflection reply's JSON, and the decoder's complaint.

A reflection is JSON that a model wrote, and one stray character can make a
whole reply unreadable. :func:`repairs` lists the texts that a reply becomes
with one fix of three kinds, and :func:`error_line` is the line that a retry
adds to say what the decoder rejected. Neither decides what counts as a
reflection: the caller reads each repaired text as it reads any reply, so a
repair that gives no object with a valid ``recommended_action`` is no repair.
Nothing is ever inserted or closed, and a reply that needs two fixes is not
repaired.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

# The repairs, in the order they are tried: a closing bracket or brace that
# no opener matches, a comma before a closer, and raw newline, carriage
# return and tab characters inside strings.
REFLECTION_REPAIRS = ("unmatched_closer", "trailing_comma", "control_char")

# Characters of the reply that a retry quotes on each side of the decoder's
# error.
ERROR_CONTEXT_CHARS = 30

_CLOSERS = ("]", "}")
_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)
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


def repairs(
    body: str, start: int, error: json.JSONDecodeError
) -> Iterator[tuple[str, str]]:
    """The texts that *body* becomes with one repair, each with its name.

    *error* is the decoder's failure on the ``{`` at *start*. The repairs
    are those of :data:`REFLECTION_REPAIRS`, each only where the decoder
    stopped: ``unmatched_closer`` deletes the ``]`` or ``}`` at the error,
    ``trailing_comma`` the comma before a closer, and ``control_char``
    escapes raw newlines, carriage returns and tabs in the strings from
    *start* on, when that is what the decoder met. Python before 3.13
    reports a trailing comma at the closer after it, and 3.13 at the comma,
    so the character before the error is looked at too.
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
