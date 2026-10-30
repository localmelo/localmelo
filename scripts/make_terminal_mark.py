#!/usr/bin/env python3
"""Make the terminal welcome's mark, Melo's face in 10 x 6 pixels, from the logo. Run it when the logo changes:

    uv run python scripts/make_terminal_mark.py

It reads docs/assets/icon_small.png and rewrites the PIXELS block of support/cli/mark.py; nothing is drawn by hand.
tests/cli/test_mark.py runs the same downsampling and fails when the committed block and the logo disagree.
Only the standard library is used (zlib for the PNG), so the test needs nothing more.

How the face is found and sampled:
- Each pixel is sorted in OKLab: dark (L < 0.5: the eyes, nose, mouth and outlines), cream (L > 0.86, chroma
  < 0.07: the muzzle), fur (chroma > 0.11, hue 35-85 degrees: the orange coat), or other (the ring, the paper).
- The face is the region of fur and cream connected (4-neighbour) to the face-coloured pixel nearest the logo's
  centre, with the holes inside it (eyes, nose, mouth) filled. The mark covers its bounding box.
- The box is cut into 10 x 6 cells (a cell's share of each pixel is its overlap). The logo is symmetric, so each
  cell's counts are averaged with its mirror cell, which cancels the antialiasing's left-right noise.
- A cell is empty when under 40% of it is face; dark when 20% or more of its face is dark (thin eyes would vanish
  in a plain average); else fur or cream, whichever covers more.
"""

from __future__ import annotations

import math
import re
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "docs/assets/icon_small.png"
MODULE = ROOT / "support/cli/mark.py"
WIDTH, HEIGHT = 10, 6  # pixels: 10 columns, 3 rows of half blocks
FACE_SHARE, DARK_SHARE = 0.4, 0.2

RGB = tuple[int, int, int]


def read_png(path: Path) -> list[list[RGB]]:
    """The pixels of an 8-bit, non-interlaced RGB or RGBA PNG, row by row."""
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
        [(row[x], row[x + 1], row[x + 2]) for x in range(0, stride, channels)]
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


def classify(colour: RGB) -> str:
    """``k`` dark, ``c`` cream, ``o`` fur, ``x`` anything else."""
    lightness, a, b = _oklab(colour)
    chroma, hue = math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360
    if lightness < 0.5:
        return "k"
    if lightness > 0.86 and chroma < 0.07:
        return "c"
    return "o" if chroma > 0.11 and 35 < hue < 85 else "x"


def _flood(
    start: tuple[int, int], keep: set[tuple[int, int]], size: tuple[int, int]
) -> set[tuple[int, int]]:
    """The 4-connected region of *keep* that holds *start*."""
    width, height = size
    seen: set[tuple[int, int]] = set()
    todo = [start]
    while todo:
        x, y = todo.pop()
        if (
            (x, y) in seen
            or (x, y) not in keep
            or not (0 <= x < width and 0 <= y < height)
        ):
            continue
        seen.add((x, y))
        todo += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return seen


def pixels(path: Path = ICON) -> tuple[str, ...]:
    """The mark's rows of ``.`` (empty), ``o`` (fur), ``c`` (cream) and ``k`` (dark)."""
    image = read_png(path)
    height, width = len(image), len(image[0])
    kinds = {(x, y): classify(image[y][x]) for y in range(height) for x in range(width)}
    coat = {point for point, kind in kinds.items() if kind in "oc"}
    middle = (width // 2, height // 2)
    seed = min(coat, key=lambda p: (abs(p[0] - middle[0]) + abs(p[1] - middle[1]), p))
    face = _flood(seed, coat, (width, height))
    outside = _flood((0, 0), set(kinds) - face, (width, height))
    inside = set(kinds) - outside
    x0, x1 = min(x for x, _ in inside), max(x for x, _ in inside) + 1
    y0, y1 = min(y for _, y in inside), max(y for _, y in inside) + 1
    counts: list[list[dict[str, float]]] = []
    for j in range(HEIGHT):
        row: list[dict[str, float]] = []
        for i in range(WIDTH):
            xa, xb = x0 + (x1 - x0) * i / WIDTH, x0 + (x1 - x0) * (i + 1) / WIDTH
            ya, yb = y0 + (y1 - y0) * j / HEIGHT, y0 + (y1 - y0) * (j + 1) / HEIGHT
            cell = {"all": 0.0, "face": 0.0, "k": 0.0, "o": 0.0, "c": 0.0}
            for y in range(math.floor(ya), math.ceil(yb)):
                for x in range(math.floor(xa), math.ceil(xb)):
                    share = (min(xb, x + 1) - max(xa, x)) * (
                        min(yb, y + 1) - max(ya, y)
                    )
                    cell["all"] += share
                    if (x, y) in inside:
                        cell["face"] += share
                        if kinds[x, y] in "koc":
                            cell[kinds[x, y]] += share
            row.append(cell)
        counts.append(row)
    rows: list[str] = []
    for row in counts:
        line = ""
        for i, cell in enumerate(row):
            both = {key: (cell[key] + row[WIDTH - 1 - i][key]) / 2 for key in cell}
            if both["face"] < FACE_SHARE * both["all"]:
                line += "."
            elif both["k"] >= DARK_SHARE * both["face"]:
                line += "k"
            else:
                line += "o" if both["o"] >= both["c"] else "c"
        rows.append(line)
    return tuple(rows)


def block(rows: tuple[str, ...]) -> str:
    """The PIXELS block as support/cli/mark.py holds it."""
    body = "".join(f'    "{row}",\n' for row in rows)
    return f"PIXELS: Final = (\n{body})\n"


def main() -> int:
    source = MODULE.read_text(encoding="utf-8")
    pattern = re.compile(r"PIXELS: Final = \(\n(?:    \"[.ock]+\",\n)+\)\n")
    if not pattern.search(source):
        print(f"no PIXELS block in {MODULE}", file=sys.stderr)
        return 1
    MODULE.write_text(pattern.sub(block(pixels()), source), encoding="utf-8")
    print("\n".join(pixels()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
