"""Make text safe to place in the assistant role of a chat template.

Some chat templates render an assistant message's content as it is: Flint's
``flint_text`` (line 3 of its ``chat_template.jinja``) removes NUL,
``<think>`` and ``</think>``, and escapes nothing else. Harness text put in
that role (a reasoning note, a recalled memory) could then carry a role
token or tool-call syntax that the model reads as structure, and a model
reply can contain such text. :func:`sanitize_assistant_text` removes or
escapes it, in the escape form of Flint's own template.
"""

from __future__ import annotations

import re

# Role tokens of Flint, then of Qwen, matched in exact case.
_ROLE_TOKEN_RE = re.compile(
    r"<\|(system|user|assistant|observation|developer|begin_of_text|end_of_text"
    r"|im_start|im_end)\|>"
)
# Tool-call and note tags, opening and closing, in any case.
_TAG_RE = re.compile(
    r"<(/?)(tool_call|tool_response|arg_key|arg_value|note)>", re.IGNORECASE
)
# The openings of a function or parameter element, and their closing tags.
_ELEMENT_OPEN_RE = re.compile(r"<(function|parameter)=", re.IGNORECASE)
_ELEMENT_CLOSE_RE = re.compile(r"</(function|parameter)>", re.IGNORECASE)

# What the kept text ends with when the next ">" completes a think tag.
_OPEN_THINK_HEAD = list("<think")
_CLOSE_THINK_HEAD = list("</think")


def sanitize_assistant_text(text: str) -> str:
    """*text* with nothing that a chat template could read as structure.

    In order:

    1. every NUL character is removed;
    2. every ``<think>`` and ``</think>`` (exact case) is removed, until
       none is left: removing one can join two halves into a new tag, as
       in ``<thi<think>nk>``;
    3. the role tokens ``<|system|>``, ``<|user|>``, ``<|assistant|>``,
       ``<|observation|>``, ``<|developer|>``, ``<|begin_of_text|>``,
       ``<|end_of_text|>``, ``<|im_start|>`` and ``<|im_end|>`` (exact
       case) are escaped, as ``&lt;|user|&gt;``;
    4. ``<tool_call>``, ``<tool_response>``, ``<arg_key>``,
       ``<arg_value>`` and ``<note>`` and their closing tags, in any case,
       are escaped, as ``&lt;tool_call&gt;`` and ``&lt;/tool_call&gt;``;
       so are the openings ``<function=`` and ``<parameter=``, as
       ``&lt;function=``, and ``</function>`` and ``</parameter>``, as
       ``&lt;/function&gt;``.

    Escaping replaces only ``<`` and ``>``, so a tag keeps its case.
    Steps 3 and 4 add no ``<`` or ``>``, so they cannot form a new token
    or tag, and the function is idempotent. Every other ``<``, ``>`` and
    ``&`` is left alone. Unlike ``flint_text``, which replaces each think
    tag once, step 2 also removes nested tags. The function is pure and
    runs in time linear in the length of *text*.
    """
    text = text.replace("\x00", "")
    text = _remove_think_tags(text)
    text = _ROLE_TOKEN_RE.sub(r"&lt;|\1|&gt;", text)
    text = _TAG_RE.sub(r"&lt;\1\2&gt;", text)
    text = _ELEMENT_OPEN_RE.sub(r"&lt;\1=", text)
    return _ELEMENT_CLOSE_RE.sub(r"&lt;/\1&gt;", text)


def _remove_think_tags(text: str) -> str:
    """*text* with ``<think>`` and ``</think>`` removed until none is left.

    Removals never overlap, so every order of them gives the same text.
    One pass from left to right finds it: the kept text holds no tag, so
    a new one can only end at the ``>`` being added, and the tag is
    dropped there instead. Text without a ``>`` is copied in bulk, so the
    pass costs one step per ``>``.
    """
    if "<think>" not in text and "</think>" not in text:
        return text
    kept: list[str] = []
    *segments, last = text.split(">")
    for segment in segments:
        kept.extend(segment)
        if kept[-7:] == _CLOSE_THINK_HEAD:
            del kept[-7:]
        elif kept[-6:] == _OPEN_THINK_HEAD:
            del kept[-6:]
        else:
            kept.append(">")
    kept.extend(last)
    return "".join(kept)
