"""Raw markup, read strictly from left to right, for the note gate (#442).

A file, or a body, that the chat read may carry markup: comments, elements a page
does not draw, style rules, attribute text. What a note may rest on is what a reader
of the page sees, so this module leaves only that, and it fails closed: a doubt is
read as hidden, since a note refused for a false "hidden" costs one note, while a
note supported by hidden text is the attack. (The page reader of ``web_fetch`` is a
separate, lenient parser that is not touched; it flattens a page to text before the
gate sees it.)

One scan (:func:`_tokens`) cuts the text into tokens, and nothing is read inside one:
a comment (``<!--`` to ``-->``, or to the end), a bogus comment (``<!...>``,
``<?...>``, ``</`` and a non-letter, each to the first ``>``), a raw text element
(``script``, ``style``, ``textarea``, ``title``, ``xmp``, ``noscript``, ``iframe``,
``noembed``, ``noframes`` with all it holds, to its end tag or the end; ``plaintext``
always to the end), a tag with its attributes, quoted or not. So a ``<!--`` in a
quoted attribute value or in a script is part of that token and starts no comment.
A Markdown autolink (``<https://x>``, ``<user@example.com>``), ``a < b`` and ``<3``
are no tags and stay text. Every other tag is replaced by a space after the cut.

An element is hidden (:func:`hides`) when its markup, its style or a rule of a
``<style>`` block says so (one reading for all of them: any hiding declaration
counts, nothing un-hides). Elements are tracked with counters, so the pass is linear
in the text, and a style value is read to its first :data:`MAX_VALUE_CHARS`
characters.
"""

from __future__ import annotations

import colorsys
import html
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import Final, NamedTuple

# The longest style declaration that is read, in characters; the rest is ignored. A
# style that is longer than this holds one declaration after another, each read.
MAX_VALUE_CHARS: Final = 2_000
# The longest selector, in characters, that is read.
MAX_SELECTOR_CHARS: Final = 500

# ── Sizes and colours ──

# Font sizes in pixels per unit; a number with no unit is a size only when it is 0. A
# relative unit is a share of 16 px. Zero is nothing in any unit, a viewport unit
# included.
_PIXELS: Final = {
    "px": 1.0,
    "pt": 4 / 3,
    "pc": 16.0,
    "q": 0.945,
    "em": 16.0,
    "rem": 16.0,
    "%": 0.16,
    "ex": 8.0,
    "ch": 8.0,
    "mm": 3.78,
    "cm": 37.8,
    "in": 96.0,
}
# A number with an optional sign and exponent. ``\d+(?:\.\d*)?`` cannot split a run of
# digits two ways, so a long run is read in one pass.
_NUMBER: Final = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_SIZE = re.compile(rf"({_NUMBER})\s*([a-z%]*)")
_OPACITY = re.compile(rf"{_NUMBER}%?")
_NUM = re.compile(_NUMBER)
_FUNCTION = re.compile(r"([a-z-]+)\( ?([^()]*?) ?\)")
_CALL = re.compile(r"(calc|min|max|clamp)\((.*)\)")
_PARTS = re.compile(r"[\s,/]+")
# The largest font size, in pixels, that counts as hidden, and the usual one.
TINY_PIXELS: Final = 2.0
_NORMAL_PIXELS: Final = 16.0
# A text colour with every channel at or above this is white; one whose alpha is at or
# below _CLEAR is no colour; text and background within _CLOSE of one another per
# channel cannot be told apart.
_WHITE_FROM: Final = 240
_CLEAR: Final = 0.05
_CLOSE: Final = 16
_NAMED: Final[dict[str, tuple[float, float, float, float]]] = {
    "white": (255, 255, 255, 1),
    "snow": (255, 250, 250, 1),
    "ivory": (255, 255, 240, 1),
    "azure": (240, 255, 255, 1),
    "ghostwhite": (248, 248, 255, 1),
    "floralwhite": (255, 250, 240, 1),
    "mintcream": (245, 255, 250, 1),
    "honeydew": (240, 255, 240, 1),
    "seashell": (255, 245, 238, 1),
    "aliceblue": (240, 248, 255, 1),
    "whitesmoke": (245, 245, 245, 1),
    "lavenderblush": (255, 240, 245, 1),
    "black": (0, 0, 0, 1),
    "transparent": (0, 0, 0, 0),
}
_CSS_ESCAPE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\n\r\f]?|([^\n]))")
_CSS_COMMENT = re.compile(r"/\*.*?(?:\*/|\Z)", re.DOTALL)

Colour = tuple[float, float, float, float]


def _unescape(text: str) -> str:
    """*text* with its CSS escapes undone (``\\6e`` is ``n``, ``\\:`` is ``:``)."""
    if "\\" not in text:
        return text

    def one(found: re.Match[str]) -> str:
        if found[2] is not None:
            return str(found[2])
        code = int(found[1], 16)
        return chr(code) if 0 < code <= 0x10FFFF else "\ufffd"

    return _CSS_ESCAPE.sub(one, text)


def _folded(value: str) -> str:
    return " ".join(value.lower().split())[:MAX_VALUE_CHARS]


def _amount(token: str, whole: float) -> float | None:
    """A number, or a percentage of *whole*; ``none`` is 0; else ``None``."""
    if token == "none":
        return 0.0
    percent = token.endswith("%")
    found = _NUM.fullmatch(token[:-1] if percent else token)
    if found is None:
        return None
    number = float(found.group())
    return number * whole / 100 if percent else number


def _angle(token: str) -> float | None:
    """A hue in degrees: a bare number, or one with ``deg``, ``grad``, ``rad`` or
    ``turn``."""
    scale = 1.0
    for unit, per in (("deg", 1.0), ("grad", 0.9), ("rad", 57.2958), ("turn", 360.0)):
        if token.endswith(unit):
            token, scale = token[: -len(unit)], per
            break
    found = _NUM.fullmatch(token)
    return float(found.group()) * scale if found else None


def _hex(value: str) -> Colour | None:
    digits = value[1:]
    if len(digits) not in (3, 4, 6, 8) or not re.fullmatch(r"[0-9a-f]+", digits):
        return None
    if len(digits) <= 4:
        digits = "".join(char * 2 for char in digits)
    red, green, blue = (int(digits[n : n + 2], 16) for n in (0, 2, 4))
    alpha = int(digits[6:8], 16) / 255 if len(digits) == 8 else 1.0
    return red, green, blue, alpha


def _alpha(parts: list[str], at: int) -> float | None:
    if len(parts) <= at:
        return 1.0
    value = _amount(parts[at], 1.0)
    return None if value is None else min(max(value, 0.0), 1.0)


def _colour(value: str) -> Colour | None:
    """A CSS colour as ``(red, green, blue, alpha)`` (channels 0 to 255, alpha 0 to
    1), or ``None`` when *value* is none this reads (``currentcolor``, ``var()``,
    ``color-mix()``). Colours of a perceptual space (``lab``, ``lch``, ``oklab``,
    ``oklch``) are read by their lightness alone, as a grey."""
    folded = _folded(value)
    if folded in _NAMED:
        return _NAMED[folded]
    if folded.startswith("#"):
        return _hex(folded)
    found = _FUNCTION.fullmatch(folded)
    if found is None:
        return None
    name, body = found.group(1), found.group(2)
    parts = [part for part in _PARTS.split(body.strip()) if part]
    if name in ("rgb", "rgba") and len(parts) >= 3:
        red, green, blue = (_amount(part, 255) for part in parts[:3])
        alpha = _alpha(parts, 3)
        if red is None or green is None or blue is None or alpha is None:
            return None
        return red, green, blue, alpha
    if name in ("hsl", "hsla") and len(parts) >= 3:
        hue, sat, light = (
            _angle(parts[0]),
            _amount(parts[1], 100),
            _amount(parts[2], 100),
        )
        alpha = _alpha(parts, 3)
        if hue is None or sat is None or light is None or alpha is None:
            return None
        red, green, blue = colorsys.hls_to_rgb(
            (hue / 360) % 1, min(max(light, 0), 100) / 100, min(max(sat, 0), 100) / 100
        )
        return red * 255, green * 255, blue * 255, alpha
    if name == "hwb" and len(parts) >= 3:
        hue, white, black = (
            _angle(parts[0]),
            _amount(parts[1], 100),
            _amount(parts[2], 100),
        )
        alpha = _alpha(parts, 3)
        if hue is None or white is None or black is None or alpha is None:
            return None
        white, black = min(max(white, 0), 100) / 100, min(max(black, 0), 100) / 100
        if white + black >= 1:
            grey = 255 * white / (white + black)
            return grey, grey, grey, alpha
        pure = colorsys.hls_to_rgb((hue / 360) % 1, 0.5, 1.0)
        mixed = [(c * (1 - white - black) + white) * 255 for c in pure]
        return mixed[0], mixed[1], mixed[2], alpha
    if name in ("lab", "lch", "oklab", "oklch") and len(parts) >= 3:
        whole = 1.0 if name.startswith("ok") else 100.0
        light = _amount(parts[0], whole)
        alpha = _alpha(parts, 3)
        if light is None or alpha is None:
            return None
        grey = min(max(light / whole, 0.0), 1.0) * 255
        return grey, grey, grey, alpha
    if name == "color" and len(parts) >= 4:
        red, green, blue = (_amount(part, 1.0) for part in parts[1:4])
        alpha = _alpha(parts, 4)
        if red is None or green is None or blue is None or alpha is None:
            return None
        return (
            min(max(red, 0.0), 1.0) * 255,
            min(max(green, 0.0), 1.0) * 255,
            min(max(blue, 0.0), 1.0) * 255,
            alpha,
        )
    return None


def _is_clear(colour: Colour | None) -> bool:
    return colour is not None and colour[3] <= _CLEAR


def _is_white(colour: Colour | None) -> bool:
    return (
        colour is not None
        and colour[3] > _CLEAR
        and min(colour[0], colour[1], colour[2]) >= _WHITE_FROM
    )


def _same(left: Colour | None, right: Colour | None) -> bool:
    """Whether text of colour *left* cannot be read on a background of *right*."""
    return (
        left is not None
        and right is not None
        and left[3] > _CLEAR
        and right[3] > 0.5
        and max(abs(left[n] - right[n]) for n in range(3)) <= _CLOSE
    )


_COLOUR_WORD = re.compile(r"[a-z-]+\([^()]*\)|#[0-9a-f]+|[a-z]+")


def _background_colour(value: str) -> Colour | None:
    """The colour a ``background`` or ``background-color`` value puts behind text: the
    first colour in it (the shorthand also holds an image, a position, a size...)."""
    folded = _folded(value)
    for found in _COLOUR_WORD.finditer(folded):
        colour = _colour(found.group())
        if colour is not None:
            return colour
    return None


# A zero of any unit is no size. A font size may also be a ``calc()``, ``min()``,
# ``max()`` or ``clamp()``: one of them is tiny when its arguments say so.
def _arguments(body: str) -> list[str]:
    out: list[str] = []
    depth = 0
    start = 0
    for at, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            out.append(body[start:at])
            start = at + 1
    out.append(body[start:])
    return out


def _size_px(value: str, base: float = _NORMAL_PIXELS) -> float | None:
    """A length in pixels, a relative unit taken against *base*; ``None`` when *value*
    is no length this reads (a keyword, a viewport unit that is not zero, a negative
    number a browser ignores)."""
    found = _SIZE.fullmatch(value.strip())
    if not found:
        return None
    number, unit = float(found.group(1)), found.group(2)
    if number == 0:
        return 0.0
    if not unit or unit not in _PIXELS or number < 0:
        return None
    return number * _PIXELS[unit]


def _tiny(value: str, depth: int = 0) -> bool:
    """Whether a font size (a length, or ``min()``, ``max()``, ``clamp()`` and
    ``calc()`` of them) is 2 px or less."""
    value = value.strip()
    call = _CALL.fullmatch(value) if depth < 4 else None
    if call is None:
        size = _size_px(value)
        return size is not None and size <= TINY_PIXELS
    name, body = call.group(1), call.group(2)
    args = _arguments(body)
    if name == "min":
        return any(_tiny(arg, depth + 1) for arg in args)
    if name == "max":
        return all(_tiny(arg, depth + 1) for arg in args)
    if name == "clamp" and len(args) == 3:
        return _tiny(args[0], depth + 1) and (
            _tiny(args[1], depth + 1) or _tiny(args[2], depth + 1)
        )
    terms = [t for t in re.split(r"[\s*/+-]+", body) if t]
    if any(_NUM.fullmatch(t) and float(t) == 0 for t in terms):
        return True
    lengths = [t for t in terms if _SIZE.fullmatch(t)]
    return bool(lengths) and all(_tiny(t, depth + 1) for t in lengths)


def _font_size_of(value: str) -> str | None:
    """The size a ``font`` shorthand names (``"0/0 serif"`` gives ``"0"``), if any."""
    for token in value.split():
        size = token.partition("/")[0]
        found = _SIZE.fullmatch(size)
        if found and (found.group(2) or float(found.group(1)) == 0):
            return size
    return None


def _unimportant(value: str) -> str:
    """*value* without a trailing ``!important`` (in any spacing)."""
    bang = value.rfind("!")
    if bang >= 0 and value[bang + 1 :].strip() == "important":
        return value[:bang].rstrip()
    return value


def _declarations(style: str) -> list[tuple[str, str]]:
    """The ``(property, value)`` pairs of a *style* in order, both in lower case, CSS
    escapes undone and comments dropped, the value without ``!important``; each is
    read to its first :data:`MAX_VALUE_CHARS` characters."""
    if not style:
        return []
    pairs = []
    for declaration in _unescape(_CSS_COMMENT.sub("", style)).split(";"):
        name, _, value = declaration[:MAX_VALUE_CHARS].partition(":")
        pairs.append((name.strip().lower(), _unimportant(value.strip().lower())))
    return pairs


_OPACITY_FILTER = re.compile(r"opacity\( ?([^()]*?) ?\)")
_SCALE_ZERO = re.compile(r"scale(?:x|y|3d)?\( ?0+(?:\.0*)?(?: ?, ?0+(?:\.0*)?)* ?\)")
_CLIP_ZERO = re.compile(r"rect\( ?0(?:px)?[ ,]+0(?:px)?[ ,]+0(?:px)?[ ,]+0(?:px)? ?\)")
_CLIP_ALL = re.compile(r"inset\( ?(?:100|50)% ?\)|circle\( ?0(?:px|%)? ?[^)]*\)")
_FAR = -500.0


def _zero(value: str) -> bool:
    found = _OPACITY.fullmatch(value)
    return found is not None and float(value.rstrip("%")) <= 0


def _far_off(value: str) -> bool:
    """Whether a length puts something at least 500 px off the page, to the left or up."""
    size = _SIZE.fullmatch(value.strip())
    if size is None or size.group(2) not in _PIXELS:
        return False
    return float(size.group(1)) * _PIXELS[size.group(2)] <= _FAR


def _hides_one(name: str, value: str) -> bool:
    """Whether one style declaration keeps the text of its element from being drawn."""
    if name == "display":
        return value == "none"
    if name == "visibility":
        return value in ("hidden", "collapse")
    if name == "content-visibility":
        return value == "hidden"
    if name == "opacity":
        return _zero(value)
    if name == "filter":
        found = _OPACITY_FILTER.search(value)
        return found is not None and _zero(found.group(1))
    if name == "font-size":
        return _tiny(value)
    if name == "font":
        size = _font_size_of(value)
        return size is not None and _tiny(size)
    if name in ("color", "-webkit-text-fill-color"):
        colour = _colour(value)
        return _is_white(colour) or _is_clear(colour)
    if name == "text-indent":
        return _far_off(value)
    if name in ("left", "top", "margin-left", "margin-top", "inset-inline-start"):
        return _far_off(value)
    if name == "transform":
        return _SCALE_ZERO.search(value) is not None
    if name == "clip":
        return _CLIP_ZERO.fullmatch(value) is not None
    if name == "clip-path":
        return _CLIP_ALL.fullmatch(value) is not None
    return False


def _hidden_pairs(pairs: list[tuple[str, str]]) -> bool:
    if any(_hides_one(name, value) for name, value in pairs):
        return True
    last = dict(pairs)
    # Text of the colour of its own background, and a box with no room that clips.
    colour = _colour(last.get("color", "")) if "color" in last else None
    for key in ("background-color", "background"):
        if key in last and _same(colour, _background_colour(last[key])):
            return True
    if last.get("overflow") in ("hidden", "clip") or last.get("overflow-y") in (
        "hidden",
        "clip",
    ):
        for key in ("height", "max-height", "width", "max-width"):
            if key in last and _tiny(last[key]):
                return True
    return False


def hidden_style(style: str) -> bool:
    """Whether *style* (an attribute value, character references already decoded, or the
    declarations of a rule) keeps the text of its element from being drawn:
    ``display: none``, ``visibility: hidden`` or ``collapse``, a font of 0 to
    :data:`TINY_PIXELS` pixels (a zero of any unit, sign or exponent; also in a
    ``font`` shorthand and inside ``calc()``, ``min()``, ``max()`` and ``clamp()``),
    ``opacity: 0`` or a ``filter: opacity(0)``, a text colour of white or near white
    (``white``, ``#fff``, ``rgb(255 255 255)``, ``rgb(100%, 100%, 100%)``,
    ``hsl(0, 0%, 100%)``, ``oklch(1 0 0)``) or clear (``transparent``, an alpha of 0),
    text of its own background's colour, a box of no size with ``overflow: hidden``,
    and the usual ways off the page (``text-indent`` or an offset of -500 px,
    ``scale(0)``, a zero ``clip``). Names and values are read in any case and spacing,
    with ``!important``, CSS comments and CSS escapes, each declaration to its first
    :data:`MAX_VALUE_CHARS` characters. Any one hiding declaration counts: nothing
    un-hides, and there is no cascade."""
    return _hidden_pairs(_declarations(style))


# ── Style sheets ──
#
# The ``<style>`` blocks of a text hide the elements their rules name: ``.class``,
# ``#id``, a tag name, an attribute selector, or these together (``p.x``), by the
# rightmost compound of a selector (``.a .b`` hides every ``.b``) with its pseudo
# classes dropped (``a:hover`` hides every ``a``); a rule for a pseudo element
# (``::before``) hides nothing. A rule counts in any at-rule (``@media print``): in
# doubt it hides.

_CSS_TOKEN = re.compile(
    r"""/\*.*?(?:\*/|\Z)|"(?:\\.|[^"\\])*+(?:"|\Z)|'(?:\\.|[^'\\])*+(?:'|\Z)"""
    r"|[{};]|[^{};\"'/]++|/",
    re.DOTALL,
)
_SELECTOR_TOKEN = re.compile(
    r"\\(?:[0-9a-fA-F]{1,6}[ \t\n\r\f]?|[^\n])|[\s>+~]++|[\[\]()]|[^\\\s>+~\[\]()]++",
    re.DOTALL,
)
_IDENT = re.compile(
    r"(?:\\(?:[0-9a-fA-F]{1,6}[ \t\n\r\f]?|[^\n])|[\w-]|[^\x00-\x7f])++"
)
_ATTRIBUTE_SELECTOR = re.compile(
    r"\s*([^\s=~|^$*\]]+)\s*(?:([~|^$*]?=)\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\]]*))"
    r"\s*[a-zA-Z]?)?\s*"
)


class _Compound(NamedTuple):
    tag: str
    ids: tuple[str, ...]
    classes: tuple[str, ...]
    attributes: tuple[tuple[str, str, str], ...]


def _rightmost(selector: str) -> str:
    """The compound after the last combinator of *selector* (one that ends it, or
    begins it, is none)."""
    depth = 0
    start = 0
    after = -1  # where the compound after a combinator just seen would begin
    for found in _SELECTOR_TOKEN.finditer(selector):
        token = found.group()
        if depth == 0 and token[0] in " \t\n\r\f>+~":
            after = found.end()
            continue
        if after >= 0:
            start, after = after, -1
        if token in ("[", "("):
            depth += 1
        elif token in ("]", ")"):
            depth = max(0, depth - 1)
    return selector[start:].strip()


def _compound(text: str) -> _Compound | None:
    """The tag, ids, classes and attribute conditions of a compound selector; ``None``
    when it is for a pseudo element or holds nothing to match."""
    text = text[:MAX_SELECTOR_CHARS]
    tag = ""
    ids: list[str] = []
    classes: list[str] = []
    attributes: list[tuple[str, str, str]] = []
    size = len(text)
    at = 0
    if text.startswith("*"):
        at = 1
    else:
        found = _IDENT.match(text)
        if found:
            tag, at = _unescape(found.group()).lower(), found.end()
    while at < size:
        char = text[at]
        if char in ".#":
            found = _IDENT.match(text, at + 1)
            if found is None:
                at += 1
                continue
            (ids if char == "#" else classes).append(_unescape(found.group()))
            at = found.end()
        elif char == "[":
            close = text.find("]", at)
            body = text[at + 1 : close if close >= 0 else size]
            condition = _ATTRIBUTE_SELECTOR.fullmatch(body)
            if condition is not None:
                value = next((g for g in condition.groups()[2:] if g is not None), "")
                attributes.append(
                    (
                        _unescape(condition.group(1)).lower(),
                        condition.group(2) or "",
                        _unescape(value).lower(),
                    )
                )
            at = size if close < 0 else close + 1
        elif char == ":":
            if text.startswith("::", at):
                return None
            found = _IDENT.match(text, at + 1)
            at = found.end() if found else at + 1
            if at < size and text[at] == "(":
                close = text.find(")", at)
                at = size if close < 0 else close + 1
        else:
            at += 1
    return _Compound(tag, tuple(ids), tuple(classes), tuple(attributes))


def _fits(
    rule: _Compound, name: str, attributes: Mapping[str, str], classes: set[str]
) -> bool:
    if rule.tag and rule.tag != name:
        return False
    if any(attributes.get("id") != wanted for wanted in rule.ids):
        return False
    if not all(wanted in classes for wanted in rule.classes):
        return False
    for key, operator, wanted in rule.attributes:
        value = attributes.get(key)
        if value is None:
            return False
        value = value.lower()
        if operator == "=" and value != wanted:
            return False
        if operator == "~=" and wanted not in value.split():
            return False
        if operator == "|=" and value != wanted and not value.startswith(wanted + "-"):
            return False
        if operator == "^=" and not value.startswith(wanted):
            return False
        if operator == "$=" and not value.endswith(wanted):
            return False
        if operator == "*=" and wanted not in value:
            return False
    return True


class Sheet:
    """The rules of the ``<style>`` blocks of a text that hide an element, indexed by
    the id, class, tag or attribute name each one needs, so an element is compared
    with the few rules that can name it."""

    def __init__(self) -> None:
        self._index: dict[str, list[_Compound]] = {}
        self.empty = True

    def _add(self, rule: _Compound) -> None:
        if rule.ids:
            key = "#" + rule.ids[0]
        elif rule.classes:
            key = "." + rule.classes[0]
        elif rule.tag:
            key = rule.tag
        elif rule.attributes:
            key = "[" + rule.attributes[0][0]
        else:
            key = "*"
        self._index.setdefault(key, []).append(rule)
        self.empty = False

    def add_rule(self, selectors: str) -> None:
        """Take a rule, for the comma list *selectors*, that hides what it matches."""
        for selector in selectors.split(","):
            last = _rightmost(selector)
            rule = _compound(last) if last else None
            if rule is not None:
                self._add(rule)

    def add_css(self, css: str) -> None:
        """Read the style sheet *css*: the hiding rules in it, at any depth."""
        stack: list[tuple[str, list[str]]] = []
        buffer: list[str] = []

        def close(prelude: str, declarations: list[str]) -> None:
            if (
                prelude
                and not prelude.startswith("@")
                and hidden_style(";".join(declarations))
            ):
                self.add_rule(prelude)

        for found in _CSS_TOKEN.finditer(css):
            token = found.group()
            if token == "{":
                stack.append(("".join(buffer).strip(), []))
                buffer.clear()
            elif token == ";":
                if stack:
                    stack[-1][1].append("".join(buffer))
                buffer.clear()
            elif token == "}":
                if stack:
                    prelude, declarations = stack.pop()
                    declarations.append("".join(buffer))
                    close(prelude, declarations)
                buffer.clear()
            elif token.startswith("/*"):
                continue
            else:
                buffer.append(token)
        while stack:
            prelude, declarations = stack.pop()
            declarations.append("".join(buffer))
            buffer.clear()
            close(prelude, declarations)

    def hides(self, name: str, attributes: Mapping[str, str]) -> bool:
        """Whether a rule hides an element of this tag *name* with these attributes."""
        if self.empty:
            return False
        classes = set((attributes.get("class") or "").split())
        keys = [name, "*"]
        if "id" in attributes:
            keys.append("#" + attributes["id"])
        keys.extend("." + wanted for wanted in classes)
        keys.extend("[" + key for key in attributes)
        return any(
            _fits(rule, name, attributes, classes)
            for key in keys
            for rule in self._index.get(key, ())
        )


# ── Tokens ──

_RAW: Final = frozenset(
    {
        "script",
        "style",
        "textarea",
        "title",
        "xmp",
        "noscript",
        "iframe",
        "noembed",
        "noframes",
        "plaintext",
    }
)
_RAW_END: Final = {
    name: re.compile(rf"</{name}(?=[\s/>])[^>]*+(?:>|\Z)", re.IGNORECASE)
    for name in _RAW
}
# A tag as a browser reads it: ``<``, ``/``, a name up to white space, ``/`` or ``>``,
# then attributes up to the first ``>`` that is not in a quoted value. A quote opens a
# value only after ``=``; elsewhere it is part of a name, and an unquoted value runs
# to white space or ``>``. A quote never closed, or a tag never ended, runs to the end
# of the text, as a browser drops it (a name that runs to the end is such a tag, and a
# NUL is part of a name). The pattern is possessive and a tag keeps every character it
# scans, so a run is never scanned twice. An autolink (``<https://x>``,
# ``<user@example.com>``) is tried first and is no tag.
_TOKEN = re.compile(
    r"(?P<comment><!--.*?(?:-->|\Z))"
    r"|(?P<bogus><[!?][^>]*+(?:>|\Z)|</(?![A-Za-z])[^>]*+(?:>|\Z))"
    r"|(?P<auto><(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\s<>]*+|[^\s<>@/]++@[^\s<>]++)>)"
    r"|(?P<tag><(?P<close>/?+)(?P<name>[A-Za-z][^\s/>]*+)"
    r"(?P<attrs>(?:(?:[^>=\"'/]|/(?!>))++"
    r"|=\s*+(?:\"[^\"]*+(?:\"|\Z)|'[^']*+(?:'|\Z)|[^\s>]*+)"
    r"|[\"'])*+)"
    r"(?P<self>/?+)(?:>|\Z))",
    re.DOTALL,
)
_ATTRIBUTE = re.compile(
    r"([^\s/>=][^\s/>=]*+)"
    r"(?:\s*+=\s*+(?:\"([^\"]*+)\"?+|'([^']*+)'?+|([^\s>]*+)))?+"
)
_VOID: Final = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
# Elements a browser never draws, or draws for a reader of another kind (fallback
# content, a dialog or popover not opened, a closed disclosure's body, ruby
# parentheses, the description and title of an SVG).
_NEVER_SHOWN: Final = frozenset(
    {
        "template",
        "canvas",
        "object",
        "datalist",
        "video",
        "audio",
        "rp",
        "desc",
        "meta",
        "link",
        "base",
    }
)
# Elements that show their content only when open.
_CLOSED: Final = frozenset({"dialog", "details"})
# Elements whose ``/>`` closes them (foreign content); on any other a ``/>`` is ignored.
_FOREIGN: Final = frozenset({"svg", "math"})
# Elements the next one of the same name closes.
_AUTO_CLOSED: Final = frozenset(
    {
        "p",
        "li",
        "dt",
        "dd",
        "tr",
        "td",
        "th",
        "option",
        "optgroup",
        "thead",
        "tbody",
        "tfoot",
    }
)
_TAG: Final = 1
_DROP: Final = 2
_STYLE: Final = 3


class _Token(NamedTuple):
    kind: int  # _TAG, or _DROP (a comment, a bogus comment, a raw text element)
    start: int
    end: int
    name: str  # lower case; of a raw text element too
    attrs: str
    close: bool
    self_closing: bool
    inner: tuple[int, int]  # of a raw text element, what it holds


_NO_INNER: Final = (0, 0)


def _tokens(text: str) -> list[_Token]:
    """The tokens of *text*, left to right. Nothing is read inside one: a ``<!--`` in a
    quoted attribute value or in a script starts no comment."""
    out: list[_Token] = []
    at = 0
    while True:
        found = _TOKEN.search(text, at)
        if found is None:
            return out
        at = found.end()
        kind = found.lastgroup
        if kind == "auto":
            continue
        if kind in ("comment", "bogus"):
            out.append(
                _Token(_DROP, found.start(), at, "", "", False, False, _NO_INNER)
            )
            continue
        name = found["name"].lower()
        close = bool(found["close"])
        if name in _RAW and not close:
            if name == "plaintext":
                end = len(text)
                inner = (at, end)
            else:
                closing = _RAW_END[name].search(text, at)
                end = closing.end() if closing else len(text)
                inner = (at, closing.start() if closing else end)
            out.append(
                _Token(
                    _STYLE if name == "style" else _DROP,
                    found.start(),
                    end,
                    name,
                    "",
                    False,
                    False,
                    inner,
                )
            )
            at = end
            continue
        out.append(
            _Token(
                _TAG,
                found.start(),
                at,
                name,
                found["attrs"],
                close,
                bool(found["self"]),
                _NO_INNER,
            )
        )


def _attributes(text: str) -> dict[str, str]:
    """The attributes of a tag's attribute text, names in lower case and values with
    their character references decoded; the first of a name wins, as in a browser."""
    values: dict[str, str] = {}
    for found in _ATTRIBUTE.finditer(text):
        value = next((g for g in found.groups()[1:] if g is not None), "")
        values.setdefault(found.group(1).lower(), html.unescape(value))
    return values


def hides(name: str, attributes: Mapping[str, str], sheet: Sheet | None = None) -> bool:
    """Whether an element of this tag *name* (lower case) and these attributes (names
    in lower case, values decoded) is not shown, by the strict reading: the element
    is one a browser does not draw (:data:`_NEVER_SHOWN`), a ``dialog`` or ``details``
    that is not ``open``, or it has the ``hidden`` or ``popover`` attribute,
    ``aria-hidden="true"``, a ``display``, ``visibility`` or ``opacity`` attribute
    that hides it (SVG), a style that hides it (:func:`hidden_style`), the old
    ``<font color=white>`` (or a clear one), or a rule of the page's *sheet*."""
    if name in _NEVER_SHOWN or (name in _CLOSED and "open" not in attributes):
        return True
    if "hidden" in attributes or "popover" in attributes:
        return True
    if (attributes.get("aria-hidden") or "").strip().lower() == "true":
        return True
    if (attributes.get("display") or "").strip().lower() == "none":
        return True
    if (attributes.get("visibility") or "").strip().lower() in ("hidden", "collapse"):
        return True
    if "opacity" in attributes and _zero(_folded(attributes["opacity"])):
        return True
    if name == "font":
        colour = _colour(attributes.get("color") or "")
        if _is_white(colour) or _is_clear(colour):
            return True
    if "style" in attributes and hidden_style(attributes["style"]):
        return True
    return sheet is not None and sheet.hides(name, attributes)


# ── Cutting ──


def cut(text: str, spans: Iterable[tuple[int, int, str]]) -> str:
    """*text* with each span ``(start, end, replacement)`` (in order, not overlapping)
    replaced. A line that held only what is removed (an empty replacement) goes with
    its line break, so the paragraph around it does not split in two. One pass: each
    span costs only the white space beside it."""
    kept: list[str] = []
    done = 0
    size = len(text)
    for start, end, replacement in spans:
        if start < done:
            continue
        kept.append(text[done:start])
        kept.append(replacement)
        done = end
        if replacement:
            continue
        before = start
        while before > 0 and text[before - 1] != "\n" and text[before - 1].isspace():
            before -= 1
        if before > 0 and text[before - 1] != "\n":
            continue
        after = end
        while after < size and text[after] != "\n" and text[after].isspace():
            after += 1
        if after < size and text[after] == "\n":
            done = after + 1
    kept.append(text[done:])
    return "".join(kept)


def _spans(
    text: str, tokens: list[_Token], sheet: Sheet
) -> Iterator[tuple[int, int, str]]:
    """What of *text* goes, in order: comments, raw text elements and hidden elements
    (with all they hold) for nothing, every other tag for a space."""
    open_names: list[str] = []
    count: Counter[str] = Counter()
    hidden_at = -1  # where in open_names the outermost hidden element is
    hidden_from = 0
    for token in tokens:
        if token.kind != _TAG:
            if hidden_at < 0:
                yield token.start, token.end, ""
            continue
        name = token.name
        if token.close:
            inside = hidden_at >= 0
            if count[name] > 0:
                while True:
                    popped = open_names.pop()
                    count[popped] -= 1
                    if popped == name:
                        break
                if inside and len(open_names) <= hidden_at:
                    yield hidden_from, token.end, ""
                    hidden_at = -1
                    continue
            if not inside:
                yield token.start, token.end, " "
            continue
        if name in _AUTO_CLOSED and open_names and open_names[-1] == name:
            open_names.pop()
            count[name] -= 1
            if hidden_at >= 0 and len(open_names) <= hidden_at:
                yield hidden_from, token.start, ""
                hidden_at = -1
        foreign = count["svg"] + count["math"] > 0
        ends_here = name in _VOID or (
            token.self_closing and (foreign or name in _FOREIGN)
        )
        if hidden_at < 0:
            attributes = _attributes(token.attrs) if token.attrs.strip() else {}
            if hides(name, attributes, sheet):
                if ends_here:
                    yield token.start, token.end, ""
                    continue
                hidden_from = token.start
                hidden_at = len(open_names)
            else:
                yield token.start, token.end, " "
        if not ends_here:
            open_names.append(name)
            count[name] += 1
    if hidden_at >= 0:
        yield hidden_from, len(text), ""


def without_markup(text: str) -> str:
    """*text* without its markup: comments, bogus comments, raw text elements
    (``script``, ``style``, ``textarea``, ``title``, ``noscript`` ...), the elements
    a page does not draw (:func:`hides`, with what they hold), and every other tag,
    which is no text (its attributes among it) and is replaced by a space. A comment,
    tag or element never closed runs to the end of the text. Text with no ``<`` comes
    back as it is."""
    if "<" not in text:
        return text
    tokens = _tokens(text)
    if not tokens:
        return text
    sheet = Sheet()
    for token in tokens:
        if token.kind == _STYLE:
            sheet.add_css(text[token.inner[0] : token.inner[1]])
    return cut(text, _spans(text, tokens, sheet))


def has_stray_closer(text: str) -> bool:
    """Whether *text*, a part of a page or file read without the part before it, holds
    a closer with no opener in it: a ``-->`` outside a comment, or an end tag of an
    element that is not open in the part. Such a part may begin inside a comment or a
    hidden element."""
    last = 0
    open_names: list[str] = []
    count: Counter[str] = Counter()
    for token in _tokens(text):
        if text.find("-->", last, token.start) >= 0:
            return True
        last = token.end
        if token.kind != _TAG:
            continue
        name = token.name
        if token.close:
            if count[name] <= 0:
                return True
            while True:
                popped = open_names.pop()
                count[popped] -= 1
                if popped == name:
                    break
        elif name not in _VOID and not (
            token.self_closing
            and (name in _FOREIGN or count["svg"] + count["math"] > 0)
        ):
            open_names.append(name)
            count[name] += 1
    return text.find("-->", last) >= 0
