#!/usr/bin/env python3
"""Make the menu-bar mark, Melo's head as a line drawing, from the pet's face picture; nothing is drawn by hand.

It reads support/gateway/static/moves/face-brand.png (Melo's brand head, ears up) with the classes and regions of
scripts/make_terminal_mark.py, and lays the head on the menu bar's 18 point canvas, fitted to the pixel grid at 1x
and 2x (18 and 36 px): every pixel is empty, a line in the bar's text colour (``w``) or a line in Melo's fur colour
(``y``). Only the standard library is used.

How the head becomes lines:
- The silhouette is the head's inside and outline above BOTTOM of the picture's height (the brand head is a bust; the
  cut is at the neck, under the mouth). Its bounding box is scaled to the canvas's width, centred, and
  each pixel of the canvas counts as head when half of it is. The picture is symmetric, so each pixel's shares are
  averaged with its mirror pixel's, which cancels the antialiasing's left-right noise.
- The outline is the silhouette's pixels within `stroke` pixels of what is not head (4-connected steps); with
  `closed`, the cut at the neck counts as not head, so the line closes under the chin.
- The features are the dark pixels inside the head (eyes, nose, mouth): a pixel is one when FEATURE_SHARE of it is,
  for the features the style keeps, and it is not beside the outline (a feature that touched it would read as part
  of it).
- An outline pixel is fur coloured when the head beside it is fur: each pixel of the picture takes the class of the
  nearest solid fur or cream (a pixel whose four neighbours share its class, so the antialiased rim between the
  outline and the cream does not count as fur), and an outline pixel is fur when more than half of what it covers
  is. Lines stay in the text colour below `fur_until` of the head's height, so the cheeks and the white mask between
  the eyes never turn fur coloured. Features are always in the text colour.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_terminal_mark  # type: ignore[import-not-found]  # noqa: E402

FACE = make_terminal_mark.FACE
POINTS = 18
# Where the head is cut, as a share of the picture's height: at the neck, just under the mouth.
BOTTOM = 0.58
FEATURE_SHARE = 0.5
FUR, CREAM = frozenset("ob"), frozenset("c")

Point = tuple[int, int]


@dataclass(frozen=True)
class Style:
    """How the head is drawn at one scale: the line's width in pixels, which features stay (``eyes``, ``nose``,
    ``mouth``), whether the line closes under the chin, and down to which share of the head's height fur coloured
    lines reach (0 keeps every line in the text colour)."""

    stroke: int = 1
    features: frozenset[str] = frozenset({"eyes", "nose"})
    closed: bool = True
    fur_until: float = 0.5
    # ``beside``: a line is fur coloured where the head beside it is fur; ``all``: every line above `fur_until` is.
    fur_rule: str = "beside"
    bottom: float = BOTTOM


def _features(head: make_terminal_mark.Head) -> dict[str, set[Point]]:
    """The dark pixels inside the head by feature: the eyes are the two blobs highest up, the nose the biggest blob
    below them, the mouth the rest below the eyes."""
    dark = {p for p in head.inside if head.kinds[p] == "k"}
    blobs: list[set[Point]] = []
    while dark:
        blob = make_terminal_mark._flood([next(iter(dark))], dark)
        blobs.append(blob)
        dark -= blob
    big = sorted((b for b in blobs if len(b) > 4), key=lambda b: min(y for _, y in b))
    eyes = big[:2]
    below = min(y for _, y in eyes[0] | eyes[1]) if len(eyes) == 2 else 0
    rest = [b for b in blobs if b not in eyes and min(y for _, y in b) > below]
    nose = max(rest, key=len) if rest else set()
    return {
        "eyes": set().union(*eyes),
        "nose": nose,
        "mouth": set().union(*(b for b in rest if b is not nose)),
    }


def _nearest_class(head: make_terminal_mark.Head) -> dict[Point, str]:
    """Each pixel of the picture: ``o`` when the nearest solid fur or cream pixel is fur, else ``c``."""

    def solid(p: Point) -> str | None:
        group = (
            FUR if head.kinds[p] in FUR else CREAM if head.kinds[p] in CREAM else None
        )
        near = make_terminal_mark._near(p)
        if group is None or not all(
            q in head.inside and head.kinds[q] in group for q in near
        ):
            return None
        return "o" if group is FUR else "c"

    found = {p: kind for p in head.inside if (kind := solid(p))}
    todo = list(found)
    while todo:
        later: list[Point] = []
        for p in todo:
            for q in make_terminal_mark._near(p):
                if (
                    q not in found
                    and 0 <= q[0] < head.width
                    and 0 <= q[1] < head.height
                ):
                    found[q] = found[p]
                    later.append(q)
        todo = later
    return found


def mark(scale: int, style: Style, face: Path = FACE) -> tuple[str, ...]:
    """The mark at `scale` (1 or 2) as rows of ``.`` (empty), ``w`` (text colour) and ``y`` (fur colour)."""
    size = POINTS * scale
    head = make_terminal_mark.head(face, style.bottom)
    shape = head.inside | head.outline
    x0, x1 = min(x for x, _ in shape), max(x for x, _ in shape) + 1
    y0 = min(y for _, y in shape)
    step = (x1 - x0) / size
    # The cut moves to the nearest edge between two rows of the canvas, so the line under the chin is whole.
    rows = round((style.bottom * head.height - y0) / step)
    top, cut = y0 - (size - rows) // 2 * step, y0 + rows * step
    head = make_terminal_mark.head(face, cut / head.height)
    shape = head.inside | head.outline
    features = set().union(
        *(pixels for name, pixels in _features(head).items() if name in style.features)
    )
    nearest = _nearest_class(head)
    y1 = cut

    def shares(i: int, j: int) -> tuple[float, float, float, float]:
        """Of canvas pixel (i, j): the share that is head, feature, below the cut, and the fur share of its head."""
        xa, xb = x0 + i * step, x0 + (i + 1) * step
        ya, yb = top + j * step, top + (j + 1) * step
        area = step * step
        got = [0.0, 0.0, 0.0, 0.0]
        for y in range(math.floor(ya), math.ceil(yb)):
            for x in range(math.floor(xa), math.ceil(xb)):
                share = (
                    (min(xb, x + 1) - max(xa, x)) * (min(yb, y + 1) - max(ya, y)) / area
                )
                got[0] += share if (x, y) in shape else 0
                got[1] += share if (x, y) in features else 0
                got[2] += share if y >= cut else 0
                got[3] += share if nearest.get((x, y)) == "o" else 0
        return got[0], got[1], got[2], got[3]

    raw = [[shares(i, j) for i in range(size)] for j in range(size)]
    cells = [
        [
            tuple((a + b) / 2 for a, b in zip(row[i], row[size - 1 - i], strict=True))
            for i in range(size)
        ]
        for row in raw
    ]
    inside = {(i, j) for j in range(size) for i in range(size) if cells[j][i][0] >= 0.5}
    # Steps from what is not head; an open neck does not count as an edge.
    seeds = [
        (i, j)
        for j in range(-1, size + 1)
        for i in range(-1, size + 1)
        if (i, j) not in inside
        and (
            style.closed
            or not (0 <= i < size and 0 <= j < size)
            or cells[j][i][2] < 0.5
        )
    ]
    distance = dict.fromkeys(seeds, 0)
    todo = seeds
    while todo:
        later = []
        for p in todo:
            for q in make_terminal_mark._near(p):
                if q in inside and q not in distance:
                    distance[q] = distance[p] + 1
                    later.append(q)
        todo = later
    reach = y0 + (y1 - y0) * style.fur_until

    def touches_line(i: int, j: int) -> bool:
        """Whether a line pixel is beside (i, j), 8-connected: a feature keeps a pixel's gap from the outline."""
        return any(
            (i + a, j + b) in inside
            and distance.get((i + a, j + b), size) <= style.stroke
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
        )

    lines = []
    for j in range(size):
        line = ""
        for i in range(size):
            _, feature, _, fur = cells[j][i]
            if (i, j) in inside and distance.get((i, j), size) <= style.stroke:
                coloured = (style.fur_rule == "all" or fur > 0.5) and top + (
                    j + 0.5
                ) * step < reach
                line += "y" if coloured else "w"
            elif (
                (i, j) in inside and feature >= FEATURE_SHARE and not touches_line(i, j)
            ):
                line += "w"
            else:
                line += "."
        lines.append(line)
    # A fur pixel with no fur pixel beside it (8-connected) is noise at the edge of the fur: it takes the text colour.
    lone = {
        (i, j)
        for j, line in enumerate(lines)
        for i, kind in enumerate(line)
        if kind == "y"
        and not any(
            0 <= j + b < size and 0 <= i + a < size and lines[j + b][i + a] == "y"
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            if a or b
        )
    }
    return tuple(
        "".join("w" if (i, j) in lone else kind for i, kind in enumerate(line))
        for j, line in enumerate(lines)
    )


# The ends of a light and of a dark menu bar (the system's own greys), which a fur line keeps 3:1 against.
BARS = {"light": ((255, 255, 255), (236, 236, 238)), "dark": ((0, 0, 0), (44, 44, 46))}
CONTRAST = 3.0


def _luminance(colour: tuple[int, int, int]) -> float:
    r, g, b = (
        c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4
        for c in colour
    )
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(one: tuple[int, int, int], other: tuple[int, int, int]) -> float:
    """The WCAG contrast ratio of two colours."""
    high, low = sorted((_luminance(one), _luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _srgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    lightness, a, b = lab
    long_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    mid = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    short = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = (
        4.0767416621 * long_ - 3.3077115913 * mid + 0.2309699292 * short,
        -1.2684380046 * long_ + 2.6097574011 * mid - 0.3413193965 * short,
        -0.0041960863 * long_ - 0.7034186147 * mid + 1.7076147010 * short,
    )

    def encode(c: float) -> int:
        c = min(max(c, 0.0), 1.0)
        return round(
            255 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)
        )

    r, g, b = (encode(c) for c in linear)
    return r, g, b


def fur_colours(face: Path = FACE) -> dict[str, tuple[int, int, int]]:
    """The fur line's colour on a light and on a dark bar: the picture's fur (the median of its solid fur pixels' OKLab
    values), made darker or lighter in steps of OKLab lightness until it has CONTRAST against both ends of the bar.
    """
    head = make_terminal_mark.head(face, BOTTOM)
    image = make_terminal_mark.read_png(face)
    fur = [
        p
        for p, kind in _nearest_class(head).items()
        if kind == "o" and p in head.inside and head.kinds[p] in FUR
    ]
    labs = sorted(make_terminal_mark._oklab(image[y][x][:3]) for x, y in fur)
    middle = labs[len(labs) // 2]
    chroma = [sorted(lab[i] for lab in labs)[len(labs) // 2] for i in (1, 2)]
    colours = {}
    for name, ends in BARS.items():
        lightness, step = middle[0], (-0.005 if name == "light" else 0.005)
        colour = _srgb((lightness, chroma[0], chroma[1]))
        while (
            min(contrast(colour, end) for end in ends) < CONTRAST and 0 < lightness < 1
        ):
            lightness += step
            colour = _srgb((lightness, chroma[0], chroma[1]))
        colours[name] = colour
    return colours


def main() -> int:
    for scale in (1, 2):
        print(*mark(scale, Style(stroke=scale)), sep="\n")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
