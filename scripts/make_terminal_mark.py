#!/usr/bin/env python3
"""Make the terminal welcome's mark, Melo's head in half-block pixels, from the pet's face picture. Run it when the
picture changes:

    uv run python scripts/make_terminal_mark.py

It reads macos/art/akita/face/idle@2x.png (Melo's idle face, ears up) and rewrites the PIXELS block of
support/cli/mark.py; nothing is drawn by hand. tests/cli/test_mark.py runs the same downsampling and fails when the
committed block and the picture disagree. Only the standard library is used (zlib for the PNG), so the test needs
nothing more.

How the head is found and sampled:
- Each pixel is sorted in OKLab: empty (alpha under half), dark (L < 0.45: the outline, eyes, nose and mouth),
  cream (L > 0.8, or chroma < 0.05: the muzzle, the brows and the halo), shade (L < 0.58: the inner ears) or fur.
- Outside is what the picture's edge reaches without crossing a dark pixel; the outline is the dark pixels joined to
  it. Inside is the rest above the collar (HEAD_BOTTOM of the picture's height), so the halo and the body drop out;
  the dark pixels inside are the eyes, nose and mouth. The mark covers the inside's bounding box.
- The box is cut into 12 x 10 cells, five rows of text (a cell's share of each pixel is its overlap); fewer rows
  lose the ears. The picture is symmetric, so each cell's counts are averaged with its mirror cell, which cancels
  the antialiasing's left-right noise.
- A cell is empty when under 40% of it is head (inside or outline); dark when 30% or more of its inside is dark
  (thin eyes would vanish in a plain average); else fur, cream or shade, whichever covers more.
"""

from __future__ import annotations

import math
import re
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACE = ROOT / "macos/art/akita/face/idle@2x.png"
MODULE = ROOT / "support/cli/mark.py"
# Pixels (columns, rows): two rows of pixels make one row of text.
COLUMNS, LINES = 12, 10
HEAD_SHARE, DARK_SHARE = 0.4, 0.3
# Where the collar begins, as a share of the picture's height.
HEAD_BOTTOM = 0.86

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]
Point = tuple[int, int]


def read_png(path: Path) -> list[list[RGBA]]:
    """The pixels of an 8-bit, non-interlaced RGB or RGBA PNG, row by row (RGB is opaque)."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    pos, idat = 8, b""
    width = height = channels = 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if depth != 8 or colour not in (2, 6) or interlace:
                raise ValueError(f"{path}: only 8-bit non-interlaced RGB(A) is read")
            channels = 3 if colour == 2 else 4
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw, stride = zlib.decompress(idat), width * channels
    rows: list[bytearray] = []
    previous = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        method, line = raw[start], bytearray(raw[start + 1 : start + 1 + stride])
        for x in range(stride):
            left = line[x - channels] if x >= channels else 0
            up, corner = previous[x], previous[x - channels] if x >= channels else 0
            if method == 1:
                line[x] = (line[x] + left) & 255
            elif method == 2:
                line[x] = (line[x] + up) & 255
            elif method == 3:
                line[x] = (line[x] + (left + up) // 2) & 255
            elif method == 4:
                guess = left + up - corner
                near = min(
                    (abs(guess - left), 0, left),
                    (abs(guess - up), 1, up),
                    (abs(guess - corner), 2, corner),
                )
                line[x] = (line[x] + near[2]) & 255
        rows.append(line)
        previous = line
    return [
        [
            (row[x], row[x + 1], row[x + 2], row[x + 3] if channels == 4 else 255)
            for x in range(0, stride, channels)
        ]
        for row in rows
    ]


def _oklab(colour: RGB) -> tuple[float, float, float]:
    def linear(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(value) for value in colour)
    long_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    mid = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    short = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * long_ + 0.7936177850 * mid - 0.0040720468 * short,
        1.9779984951 * long_ - 2.4285922050 * mid + 0.4505937099 * short,
        0.0259040371 * long_ + 0.7827717662 * mid - 0.8086757660 * short,
    )


def classify(colour: RGBA) -> str:
    """`` `` empty, ``k`` dark, ``c`` cream, ``b`` shade, ``o`` fur."""
    if colour[3] < 128:
        return " "
    lightness, a, b = _oklab(colour[:3])
    if lightness < 0.45:
        return "k"
    if lightness > 0.8 or math.hypot(a, b) < 0.05:
        return "c"
    return "b" if lightness < 0.58 else "o"


def _flood(starts: list[Point], keep: set[Point]) -> set[Point]:
    """The points of *keep* 4-connected to *starts*."""
    seen: set[Point] = set()
    todo = list(starts)
    while todo:
        x, y = point = todo.pop()
        if point in seen or point not in keep:
            continue
        seen.add(point)
        todo += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return seen


def _near(point: Point) -> list[Point]:
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def pixels(path: Path = FACE) -> tuple[str, ...]:
    """The mark's rows of ``.`` (empty), ``o`` fur, ``c`` cream, ``b`` shade, ``k`` dark."""
    image = read_png(path)
    height, width = len(image), len(image[0])
    kinds = {(x, y): classify(image[y][x]) for y in range(height) for x in range(width)}
    edge = [p for p in kinds if p[0] in (0, width - 1) or p[1] in (0, height - 1)]
    outside = _flood(edge, {p for p, kind in kinds.items() if kind != "k"})
    dark = {p for p, kind in kinds.items() if kind == "k"}
    touching = [p for p in dark if any(q in outside for q in _near(p))]
    bottom = HEAD_BOTTOM * height
    outline = {p for p in _flood(touching, dark) if p[1] < bottom}
    inside = {p for p in kinds if p not in outside and p not in outline}
    inside = {p for p in inside if p[1] < bottom}
    x0, x1 = min(x for x, _ in inside) - 1, max(x for x, _ in inside) + 2
    y0, y1 = min(y for _, y in inside) - 1, max(y for _, y in inside) + 2
    counts: list[list[dict[str, float]]] = []
    for j in range(LINES):
        row: list[dict[str, float]] = []
        for i in range(COLUMNS):
            xa, xb = x0 + (x1 - x0) * i / COLUMNS, x0 + (x1 - x0) * (i + 1) / COLUMNS
            ya, yb = y0 + (y1 - y0) * j / LINES, y0 + (y1 - y0) * (j + 1) / LINES
            cell = dict.fromkeys(("all", "head", "in", "k", "o", "c", "b"), 0.0)
            for y in range(math.floor(ya), math.ceil(yb)):
                for x in range(math.floor(xa), math.ceil(xb)):
                    share = (min(xb, x + 1) - max(xa, x)) * (
                        min(yb, y + 1) - max(ya, y)
                    )
                    cell["all"] += share
                    if (x, y) in inside or (x, y) in outline:
                        cell["head"] += share
                    if (x, y) in inside:
                        cell["in"] += share
                        cell[kinds[x, y]] += share
            row.append(cell)
        counts.append(row)
    rows: list[str] = []
    for row in counts:
        line = ""
        for i, cell in enumerate(row):
            both = {k: (cell[k] + row[COLUMNS - 1 - i][k]) / 2 for k in cell}
            if both["head"] < HEAD_SHARE * both["all"]:
                line += "."
            elif both["in"] and both["k"] >= DARK_SHARE * both["in"]:
                line += "k"
            else:
                line += max("ocb", key=lambda k: both[k])
        rows.append(line)
    return tuple(rows)


def block(rows: tuple[str, ...]) -> str:
    """The PIXELS block as support/cli/mark.py holds it."""
    body = "".join(f'    "{row}",\n' for row in rows)
    return f"PIXELS: Final = (\n{body})\n"


def main() -> int:
    source = MODULE.read_text(encoding="utf-8")
    pattern = re.compile(r"PIXELS: Final = \(\n(?:    \"[.ockb]+\",\n)+\)\n")
    if not pattern.search(source):
        print(f"no PIXELS block in {MODULE}", file=sys.stderr)
        return 1
    rows = pixels()
    MODULE.write_text(pattern.sub(lambda _: block(rows), source), encoding="utf-8")
    print(*rows, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
