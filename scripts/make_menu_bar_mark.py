#!/usr/bin/env python3
"""Make the menu-bar mark, Melo's head as a line drawing, from the pet's face picture; nothing is drawn by hand.
Run it when the picture changes:

    uv run python scripts/make_menu_bar_mark.py

It reads support/gateway/static/moves/face-brand.png (Melo's brand head, ears up) with the classes and regions of
scripts/make_terminal_mark.py and writes two masks for each scale of the menu bar's 18 point canvas (18 and 36 px):
macos/assets/MenuBarMark*.png, the lines in the bar's text colour, and MenuBarMarkFur*.png, the lines in Melo's fur
colour. Each is black with the mask in its alpha; the app colours them as it draws them (MenuBar.swift).
`scripts/art_pack.py apply` does the same after it replaces the moves, and tests/test_menu_bar_mark.py fails when the
committed masks and the picture disagree.

How the head becomes lines:
- The silhouette is the head's inside and outline above BOTTOM of the picture's height (the brand head is a bust; the
  cut is at the neck, just under the mouth). Its bounding box is scaled to the canvas's width and centred, with the
  cut on an edge between two rows of the canvas. The lines are drawn at OVERSAMPLE times the canvas's size and then
  reduced by area, so their edges carry partial alpha.
- The silhouette is smoothed (a Gaussian of SMOOTH picture pixels, which also rounds the ear tips as the picture has
  them) and averaged with its mirror image, since the picture is symmetric; its line is the band of `stroke` canvas
  pixels inside the smoothed contour. The cut counts as outside, so the line closes under the chin.
- The eyes and the nose are the picture's dark blobs inside the head, smoothed the same way. They keep a canvas pixel
  of room from the line and are always in the text colour. (The mouth is left out: at 36 px it joins the nose into a
  bar, and it has room above the chin only when the chin is cut lower.)
- The lines above `fur_until` of the head's height (the ears and the crown) are in the fur colour; the rest, the
  cheeks and the white mask between the eyes among them, in the text colour.
- With `crisp`, each canvas pixel is all or nothing (a pixel is line when half of it is): one pixel lines stay sharp at
  1x, where a smooth one would be grey.
- The fur colour is the picture's (the median OKLab of its solid fur), made darker or lighter until it has CONTRAST
  against both ends of a light and of a dark menu bar.
"""

from __future__ import annotations

import re
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_terminal_mark  # type: ignore[import-not-found]  # noqa: E402

ROOT = make_terminal_mark.ROOT
FACE = make_terminal_mark.FACE
ASSETS = ROOT / "macos/assets"
NAMES = (
    "MenuBarMark.png",
    "MenuBarMarkFur.png",
    "MenuBarMark@2x.png",
    "MenuBarMarkFur@2x.png",
)
# Where the app's palette holds the fur colours, as `fur = pair(light, dark)`.
PALETTE = "macos/LocalMeloApp/StatusView.swift"
POINTS = 18
OVERSAMPLE = 8
BOTTOM = 0.58
SMOOTH = 1.2
FUR, CREAM = frozenset("ob"), frozenset("c")
# The ends of a light and of a dark menu bar (the system's own greys), which a fur line keeps 3:1 against.
BARS = {"light": ((255, 255, 255), (236, 236, 238)), "dark": ((0, 0, 0), (44, 44, 46))}
CONTRAST = 3.0
# The bar's text colour as the app draws the lines (the system's label colour): black or white at 85%.
TEXT = {"light": ((0, 0, 0), 0.85), "dark": ((255, 255, 255), 0.85)}

Point = tuple[int, int]
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Style:
    """How the head is drawn at one scale: the line's width in canvas pixels, the features it keeps (``eyes``,
    ``nose``), down to which share of the head's height the lines are fur coloured, and whether each pixel
    is all or nothing."""

    stroke: float
    features: frozenset[str] = frozenset({"eyes", "nose"})
    fur_until: float = 0.3
    crisp: bool = False


# The mark the app shows: 1.5 point lines on Retina, one sharp pixel at 1x (a smooth 1 px line is grey there).
STYLES = {1: Style(stroke=1, crisp=True), 2: Style(stroke=3)}


def _features(head: make_terminal_mark.Head) -> dict[str, set[Point]]:
    """The dark pixels inside the head by feature: the eyes are the two biggest blobs highest up, the nose the biggest
    blob below them."""
    dark = {p for p in head.inside if head.kinds[p] == "k"}
    blobs: list[set[Point]] = []
    while dark:
        blob = make_terminal_mark._flood([next(iter(dark))], dark)
        blobs.append(blob)
        dark -= blob
    big = sorted((b for b in blobs if len(b) > 4), key=lambda b: min(y for _, y in b))
    eyes, rest = big[:2], big[2:]
    nose = max(rest, key=len) if rest else set()
    if len(eyes) < 2 or not nose:
        raise ValueError("the picture has no two eyes and a nose to draw")
    return {"eyes": eyes[0] | eyes[1], "nose": nose}


def _mask(points: set[Point], width: int, height: int) -> np.ndarray:
    array = np.zeros((height, width))
    for x, y in points:
        array[y, x] = 1
    return array


def _blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """`array` smoothed by a Gaussian of `sigma` pixels (zero beyond its edges)."""
    radius = int(3 * sigma) + 1
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    rows = np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="same"), 1, array
    )
    return np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="same"), 0, rows
    )


def _sample(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """`field` (one value per picture pixel, at the pixel's centre, and 0 beyond the picture) at the points (xs, ys),
    bilinearly."""
    field = np.pad(field, 1)
    height, width = field.shape
    x = np.clip(xs + 0.5, 0, width - 1)
    y = np.clip(ys + 0.5, 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0
    top = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    values: np.ndarray = top * (1 - fy) + bottom * fy
    return values


def _distance(inside: np.ndarray) -> np.ndarray:
    """For each pixel of `inside`, its distance in pixels to the nearest pixel outside it (the edges are outside)."""
    padded = np.pad(inside, 1)
    outside = ~padded
    edge = outside & (
        np.roll(padded, 1, 0)
        | np.roll(padded, -1, 0)
        | np.roll(padded, 1, 1)
        | np.roll(padded, -1, 1)
    )
    ey, ex = np.nonzero(edge)
    distance = np.zeros(inside.shape)
    py, px = np.nonzero(inside)
    for start in range(0, len(py), 4096):
        dy = py[start : start + 4096, None] + 1 - ey[None, :]
        dx = px[start : start + 4096, None] + 1 - ex[None, :]
        distance[py[start : start + 4096], px[start : start + 4096]] = np.sqrt(
            dx * dx + dy * dy
        ).min(axis=1)
    return distance


def mark(
    scale: int, style: Style | None = None, face: Path = FACE
) -> tuple[np.ndarray, np.ndarray]:
    """The mark at `scale` (1 or 2): the text-colour mask and the fur-colour mask, each (18 * scale) px square with
    values from 0 to 255."""
    style = style or STYLES[scale]
    size, k = POINTS * scale, OVERSAMPLE
    head = make_terminal_mark.head(face, BOTTOM)
    shape = head.inside | head.outline
    x0, x1 = min(x for x, _ in shape), max(x for x, _ in shape) + 1
    y0 = min(y for _, y in shape)
    step = (x1 - x0) / size
    rows = round((BOTTOM * head.height - y0) / step)
    top, cut = y0 - (size - rows) // 2 * step, y0 + rows * step
    head = make_terminal_mark.head(face, cut / head.height)
    shape = head.inside | head.outline
    # The centres of the fine pixels, in the picture's pixels.
    fine = (np.arange(size * k) + 0.5) / k * step
    xs, ys = np.meshgrid(x0 + fine, top + fine)

    def smooth(points: set[Point]) -> np.ndarray:
        field = _sample(_blur(_mask(points, head.width, head.height), SMOOTH), xs, ys)
        field = (field + field[:, ::-1]) / 2
        field[ys >= cut] = 0
        kept: np.ndarray = field >= 0.5
        return kept

    inside = smooth(shape)
    depth = _distance(inside) / k
    band = inside & (depth <= style.stroke)
    fur = band & (ys < y0 + (cut - y0) * style.fur_until)
    parts = _features(head)
    features = smooth(set().union(*(parts[name] for name in style.features)))
    features &= inside & (depth > style.stroke + 1)

    def reduce(fine_mask: np.ndarray) -> np.ndarray:
        share: np.ndarray = fine_mask.reshape(size, k, size, k).mean(axis=(1, 3))
        return share

    lines, furs = reduce((band & ~fur) | features), reduce(fur)
    if style.crisp:
        # A pixel that is half line is line, in the colour that covers more of it.
        whole = lines + furs >= 0.5
        lines, furs = (whole & (lines >= furs)).astype(float), (
            whole & (furs > lines)
        ).astype(float)
    return np.round(lines * 255).astype(np.uint8), np.round(furs * 255).astype(np.uint8)


def _png(alpha: np.ndarray) -> bytes:
    """An 8-bit RGBA PNG, black, with `alpha` as its alpha."""
    height, width = alpha.shape
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., 3] = alpha
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def files(face: Path = FACE) -> dict[str, bytes]:
    """The four mask files by name, as macos/assets holds them."""
    out = {}
    for scale, suffix in ((1, ""), (2, "@2x")):
        line, fur = mark(scale, face=face)
        out[f"MenuBarMark{suffix}.png"] = _png(line)
        out[f"MenuBarMarkFur{suffix}.png"] = _png(fur)
    return out


def write(assets: Path = ASSETS, face: Path = FACE) -> list[Path]:
    """Write the masks into `assets`; the paths written."""
    written = []
    for name, data in files(face).items():
        path = assets / name
        path.write_bytes(data)
        written.append(path)
    return written


def _luminance(colour: RGB) -> float:
    r, g, b = (
        c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4
        for c in colour
    )
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(one: RGB, other: RGB) -> float:
    """The WCAG contrast ratio of two colours."""
    high, low = sorted((_luminance(one), _luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _srgb(lab: tuple[float, float, float]) -> RGB:
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


def fur_colours(face: Path = FACE) -> dict[str, RGB]:
    """The fur line's colour on a light and on a dark bar: the median OKLab of the picture's solid fur (fur pixels whose
    four neighbours are fur), stepped darker or lighter in OKLab lightness until it has CONTRAST against both ends of
    that bar."""
    head = make_terminal_mark.head(face, BOTTOM)
    image = make_terminal_mark.read_png(face)
    fur = [
        p
        for p in head.inside
        if head.kinds[p] in FUR
        and all(
            q in head.inside and head.kinds[q] in FUR
            for q in make_terminal_mark._near(p)
        )
    ]
    labs = [make_terminal_mark._oklab(image[y][x][:3]) for x, y in fur]
    lightness, a, b = (sorted(lab[i] for lab in labs)[len(labs) // 2] for i in range(3))
    colours = {}
    for name, ends in BARS.items():
        level, step = lightness, (-0.005 if name == "light" else 0.005)
        colour = _srgb((level, a, b))
        while min(contrast(colour, end) for end in ends) < CONTRAST and 0 < level < 1:
            level += step
            colour = _srgb((level, a, b))
        colours[name] = colour
    return colours


def palette_fur(root: Path = ROOT) -> dict[str, RGB]:
    """The fur colours the app's palette holds, by bar."""
    found = re.search(
        r"static let fur = pair\(0x(\w{6}), 0x(\w{6})\)",
        (root / PALETTE).read_text("utf-8"),
    )
    if not found:
        raise ValueError(f"no fur pair in {PALETTE}")
    light, dark = (
        tuple(int(hex_[i : i + 2], 16) for i in (0, 2, 4)) for hex_ in found.groups()
    )
    return {
        "light": (light[0], light[1], light[2]),
        "dark": (dark[0], dark[1], dark[2]),
    }


def on_bars(scale: int, assets: Path = ASSETS, face: Path = FACE) -> np.ndarray:
    """The masks in `assets` at `scale` as the app draws them, on a light and a dark bar side by side (RGB, uint8),
    each with a margin of a quarter of the mark: for previews."""
    suffix = "@2x" if scale == 2 else ""
    masks = [
        np.asarray(
            make_terminal_mark.read_png(assets / f"{name}{suffix}.png"), dtype=float
        )[..., 3]
        / 255
        for name in ("MenuBarMark", "MenuBarMarkFur")
    ]
    size = POINTS * scale
    margin = size // 4
    fur = fur_colours(face)
    tiles = []
    for bar in ("light", "dark"):
        tile = np.tile(
            np.array(BARS[bar][1], dtype=float),
            (size + 2 * margin, size + 2 * margin, 1),
        )
        inner = tile[margin : margin + size, margin : margin + size]
        (text, strength), colour = TEXT[bar], fur[bar]
        for mask, rgb, alpha in ((masks[0], text, strength), (masks[1], colour, 1.0)):
            cover = (mask * alpha)[..., None]
            inner[:] = inner * (1 - cover) + np.array(rgb, dtype=float) * cover
        tiles.append(tile)
    out: np.ndarray = np.round(np.concatenate(tiles, axis=1)).astype(np.uint8)
    return out


def main() -> int:
    try:
        written = write()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(*(path.relative_to(ROOT) for path in written), sep="\n")
    print(
        "fur colour:",
        {name: "#" + bytes(colour).hex() for name, colour in fur_colours().items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
