#!/usr/bin/env python3
"""Make the macOS app's icon files from the logo art. This is the engine of `scripts/art_pack.py`, which hands it the
pack's brand pictures (docs/guides/art-pack.md); run it alone to make the files again from the committed pictures in
docs/assets, on a Mac:

    uv run --with pillow python scripts/make_app_icons.py

Pillow is not a dependency of localmelo: `--with` fetches it for this one run, and nothing is added to
pyproject.toml or uv.lock. The files it writes are committed, and scripts/build_macos_app.py only copies
them into the app.

- macos/assets/AppIcon.icns: Melo, the localmelo mascot, as the README shows the emblem (docs/assets/melo.png),
  on a rounded tile of the macOS icon grid: 824 of 1024 px, with 185 px corners. The emblem is scaled to the
  tile's height, and the tile is filled with the emblem's own paper colour (the mean of its four corners).
  The paper differs a little along each edge, so a flat tile shows a rectangle: the emblem fades into the tile
  over ICON_FEATHER px at its edges. Every size is resampled from the source with a Lanczos filter, so a source
  smaller than 1024 px is upscaled for the largest size. A picture with transparent pixels is laid on white first.
- macos/assets/MenuBarLogo.png and MenuBarLogo@2x.png: the round logo (docs/assets/icon_small.png) at 18 points,
  18 and 36 px, with rounded corners, for the menu-bar item.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "macos/assets"
ICONUTIL = Path("/usr/bin/iconutil")
ICON_GRID, ICON_TILE, ICON_RADIUS, ICON_FEATHER = (
    1024,
    824,
    185,
    40,
)  # px of the 1024 px grid
# The iconset names iconutil expects, with their pixel sizes.
ICONSET = {
    f"icon_{n}x{n}{suffix}.png": n * scale
    for n in (16, 32, 128, 256, 512)
    for suffix, scale in (("", 1), ("@2x", 2))
}
MENU_BAR_POINTS, MENU_BAR_RADIUS = 18, 4


def resize(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """(h, w, 3) pixels resampled to (height, width, 3) with a Lanczos filter, as floats."""
    image = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
    resized: np.ndarray = np.asarray(image, dtype=float)
    return resized


def coverage(size: int, side: float, radius: float) -> np.ndarray:
    """How much of each pixel of a size x size canvas a rounded square of `side` px, centred in it, covers."""
    inset = np.abs(np.mgrid[:size, :size] + 0.5 - size / 2) - (side / 2 - radius)
    outside = (
        np.hypot(*np.maximum(inset, 0)) + np.minimum(inset.max(axis=0), 0) - radius
    )
    covered: np.ndarray = np.clip(0.5 - outside, 0, 1)
    return covered


def save(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    """Write RGB colours and an alpha between 0 and 1 as an RGBA PNG that says it is sRGB."""
    pixels = np.rint(np.dstack([rgb, alpha * 255]).clip(0, 255)).astype(np.uint8)
    srgb = PngImagePlugin.PngInfo()
    srgb.add(b"sRGB", b"\0")
    Image.fromarray(pixels).save(path, pnginfo=srgb)


def draw_icon(emblem: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """The app icon at one size, as colours and alpha: the emblem on a rounded tile of its own paper."""
    scale = size / ICON_GRID
    side = ICON_TILE * scale
    corners = [emblem[:8, :8], emblem[:8, -8:], emblem[-8:, :8], emblem[-8:, -8:]]
    paper = np.mean(corners, axis=(0, 1, 2))
    height = round(side)
    width = round(height * emblem.shape[1] / emblem.shape[0])
    # The fade: 0 at the emblem's edges, rising smoothly to 1 ICON_FEATHER px inside them.
    ys, xs = np.arange(height) + 0.5, np.arange(width) + 0.5
    near = np.minimum(np.minimum(ys, height - ys)[:, None], np.minimum(xs, width - xs))
    fade = np.clip(near / (ICON_FEATHER * scale), 0, 1)
    fade = (fade * fade * (3 - 2 * fade))[..., None]
    top, left = (size - height) // 2, (size - width) // 2
    canvas = np.tile(paper, (size, size, 1))
    canvas[top : top + height, left : left + width] += (
        resize(emblem, height, width) - paper
    ) * fade
    return canvas, coverage(size, side, ICON_RADIUS * scale)


def draw_menu_bar_logo(logo: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """The round logo at 18 points, `scale` px to the point, as colours and alpha, with rounded corners."""
    size = MENU_BAR_POINTS * scale
    return resize(logo, size, size), coverage(size, size, MENU_BAR_RADIUS * scale)


def flatten(image: Image.Image) -> np.ndarray:
    """A picture as (h, w, 3) colours; where it is transparent it is laid on white first."""
    rgba = image.convert("RGBA")
    page = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    page.alpha_composite(rgba)
    flat: np.ndarray = np.asarray(page.convert("RGB"))
    return flat


def load(path: Path) -> np.ndarray:
    """A picture file as (h, w, 3) colours (see `flatten`)."""
    with Image.open(path) as image:
        return flatten(image)


def icon_image(emblem: np.ndarray, size: int) -> Image.Image:
    """The app icon at one size as a picture (what AppIcon.icns holds at that size)."""
    rgb, alpha = draw_icon(emblem, size)
    pixels = np.rint(np.dstack([rgb, alpha * 255]).clip(0, 255)).astype(np.uint8)
    return Image.fromarray(pixels)


def build_icons(
    out: Path, emblem: np.ndarray | None = None, logo: np.ndarray | None = None
) -> list[str]:
    """Write the icon from `emblem` and the menu-bar logo from `logo` into `out`; each is skipped when it is None.
    The names written. The icon needs macOS's iconutil."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if emblem is not None:
        with tempfile.TemporaryDirectory() as temp:
            iconset = Path(temp) / "AppIcon.iconset"
            iconset.mkdir()
            for name, size in ICONSET.items():
                save(iconset / name, *draw_icon(emblem, size))
            subprocess.run(
                [
                    str(ICONUTIL),
                    "-c",
                    "icns",
                    str(iconset),
                    "-o",
                    str(out / "AppIcon.icns"),
                ],
                check=True,
            )
        written.append("AppIcon.icns")
    if logo is not None:
        for scale, name in ((1, "MenuBarLogo.png"), (2, "MenuBarLogo@2x.png")):
            save(out / name, *draw_menu_bar_logo(logo, scale))
            written.append(name)
    return written


def main() -> None:
    written = build_icons(
        ASSETS,
        load(ROOT / "docs/assets/melo.png"),
        load(ROOT / "docs/assets/icon_small.png"),
    )
    print(f"Wrote {', '.join(written)} in {ASSETS}")


if __name__ == "__main__":
    main()
