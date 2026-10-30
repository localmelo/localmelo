#!/usr/bin/env python3
"""Make Melo's pictures for the macOS app from the chosen art masters. Run it by hand, on a Mac, when the art changes:

    uv run --with pillow python scripts/build_pet_assets.py --picks <picks.json>

Pillow is not a dependency of localmelo: `--with` fetches it for this one run. The files it writes in
macos/art/akita are committed, and scripts/build_macos_app.py only copies them into the app. The masters (about
600 to 700 px wide, cut out with a soft edge) stay outside the repository; `picks.json` lists them as
`{"root": DIR, "picks": {STATE: PATH below DIR}}` for the 14 states below. The same input gives the same bytes.

Per picture: the cut-out's soft dark fringe is cut away (an alpha ramp seven pixels wide becomes an edge two pixels
wide), the picture is trimmed, scaled with a Lanczos filter on premultiplied colour, and given a 1 pt cream outer rim
(`#f8f5e7` at 70%) so the navy outline, which vanishes on a dark desktop, still has an edge there.

Every family shares one canvas, with a transparent margin, so a state change swaps the picture without a jump: the
dogs rest on the same ground line and share one horizontal centre. One scale (SCALE points per master pixel) holds
for the whole set, so Melo is the same size sitting, lying and asleep.

- pet/<state>.png and @2x: the 9 states the desktop pet shows: lie-idle, lie-working, lie-waiting, lie-done, lie-error,
  rest-lazy (the default when nothing runs) and sleep-front, sleep-side, sleep-back (the sleeping poses).
- portrait/<state>.png and @2x: the 5 sitting states of the starting view and the empty chat.
- face/<name>.png and @2x: a square of the sitting head, 48 pt, for the small slots (idle, working, waiting, done,
  error); favicon/16.png, 32.png and 64.png: the idle face.
- anchors.json: per picture, its canvas (pt), `ground` (pt from the top: where the paws rest), `centre` (pt from the
  left) and `box` (x, y, width, height of the drawn dog on the canvas, pt, from the @2x file).

The web console shows the same pictures, so the same run copies the ones it needs, at 2x, into
support/gateway/static/akita (see CONSOLE_FILES); `--console-only` makes that copy again from the committed files in
macos/art/akita, without the masters.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "macos/art/akita"
CONSOLE = ROOT / "support/gateway/static/akita"

PET = (
    "lie-idle",
    "lie-working",
    "lie-waiting",
    "lie-done",
    "lie-error",
    "rest-lazy",
    "sleep-front",
    "sleep-side",
    "sleep-back",
)
PORTRAIT = ("sit-idle", "sit-working", "sit-waiting", "sit-done", "sit-error")
FAVICONS = (16, 32, 64)
FACE_NAMES = tuple(state.removeprefix("sit-") for state in PORTRAIT)

SCALE = 0.17  # points per master pixel: the lying dog is about 115 pt wide
# The generation drew a few poses a little larger or smaller; these factors match their eye distance to lie-idle's.
TRIM = {"rest-lazy": 1.03, "sleep-front": 0.95}
FACE_CENTRE, FACE_SIDE = (
    208,
    200,
), 470  # master pixels of the sitting pictures: the head, ears to collar
FACE_POINTS = 48
MARGIN = 3  # points of transparent canvas around the dogs: room for the rim
EDGE_GAIN = 4  # the cut-out's alpha is pulled to a hard edge: 0.5 stays, 0.375 and 0.625 become 0 and 1
RIM_POINTS, RIM_ALPHA, RIM_COLOUR = 1, 0.70, (0xF8, 0xF5, 0xE7)
SCALES = (1, 2)
# What the web console serves, as file below the art folder -> file below CONSOLE: the idle portrait of the empty
# chat, the five faces (the reply icons, the approval bar, the brand) and the three favicons. The portrait and the
# faces are the @2x files, shown at half their size, so one file looks sharp on any screen the console meets.
CONSOLE_FILES = {
    "portrait/sit-idle@2x.png": "sit-idle.png",
    **{f"face/{name}@2x.png": f"face-{name}.png" for name in FACE_NAMES},
    **{f"favicon/{size}.png": f"favicon-{size}.png" for size in FAVICONS},
}


def defringe(path: Path) -> Image.Image:
    """The master with the cut-out's soft alpha ramp pulled to a clean edge at its half-way line."""
    pixels = np.array(Image.open(path).convert("RGBA"))
    alpha = np.clip((pixels[..., 3] / 255 - 0.5) * EDGE_GAIN + 0.5, 0, 1)
    pixels[..., 3] = np.rint(alpha * 255)
    pixels[alpha == 0, :3] = 0
    return Image.fromarray(pixels)


def add_rim(image: Image.Image, radius: int) -> Image.Image:
    """The picture with a cream rim `radius` px wide round its drawn parts, as a soft-edged shape under it."""
    pixels = np.asarray(image, dtype=np.float32) / 255
    alpha = pixels[..., 3]
    height, width = alpha.shape
    padded = np.pad(alpha, radius)
    grown = alpha.copy()
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            if (dy - radius) ** 2 + (dx - radius) ** 2 <= radius * radius + 1:
                grown = np.maximum(grown, padded[dy : dy + height, dx : dx + width])
    padded = np.pad(
        grown, 1
    )  # one pixel of softening, so the rim's outer edge is not stepped
    soft = (
        sum(padded[y : y + height, x : x + width] for y in range(3) for x in range(3))
        / 9
    )
    rim = np.clip(soft - alpha, 0, 1) * RIM_ALPHA * (1 - alpha)
    covered = alpha + rim
    cream = np.array(RIM_COLOUR, np.float32) / 255
    colour = (pixels[..., :3] * alpha[..., None] + cream * rim[..., None]) / np.maximum(
        covered, 1e-6
    )[..., None]
    out = np.dstack([colour, covered])
    return Image.fromarray(np.rint(out * 255).clip(0, 255).astype(np.uint8))


def save(path: Path, image: Image.Image) -> None:
    """Write an RGBA PNG that says it is sRGB; the colour of a clear pixel is zero, so the bytes do not drift."""
    pixels = np.array(image)
    pixels[pixels[..., 3] == 0, :3] = 0
    srgb = PngImagePlugin.PngInfo()
    srgb.add(b"sRGB", b"\0")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, optimize=True, pnginfo=srgb)


def trimmed(master: Image.Image) -> Image.Image:
    """The master cut to the box of its drawn pixels."""
    box = master.getchannel("A").getbbox()
    assert box is not None, "an empty picture"
    return master.crop(box)


def family(
    masters: dict[str, Image.Image], states: Sequence[str], folder: str, out: Path
) -> dict[str, dict[str, object]]:
    """Export `states` on one shared canvas as `folder`/STATE.png and @2x; their anchors."""
    cut = {state: trimmed(masters[state]) for state in states}
    points = {
        state: (
            cut[state].width * SCALE * TRIM.get(state, 1),
            cut[state].height * SCALE * TRIM.get(state, 1),
        )
        for state in states
    }
    wide = math.ceil(max(w for w, _ in points.values())) + 2 * MARGIN
    tall = math.ceil(max(h for _, h in points.values())) + 2 * MARGIN
    anchors: dict[str, dict[str, object]] = {}
    for state in states:
        for scale in SCALES:
            size = (round(points[state][0] * scale), round(points[state][1] * scale))
            dog = cut[state].resize(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (wide * scale, tall * scale))
            left = (canvas.width - dog.width) // 2
            top = canvas.height - MARGIN * scale - dog.height
            canvas.alpha_composite(dog, (left, top))
            save(
                out / folder / f"{state}{'@2x' if scale == 2 else ''}.png",
                add_rim(canvas, RIM_POINTS * scale),
            )
        anchors[state] = {
            "canvas": [wide, tall],
            "ground": tall - MARGIN,
            "centre": wide / 2,
            "box": [
                left / 2,
                top / 2,
                dog.width / 2,
                dog.height / 2,
            ],  # from the @2x pass
        }
    return anchors


def faces(masters: dict[str, Image.Image], out: Path) -> None:
    """The sitting heads as small squares, and the idle one as the favicon sizes."""

    def render(state: str, pixels: int) -> Image.Image:
        (x, y), half = FACE_CENTRE, FACE_SIDE // 2
        crop = masters[state].crop((x - half, y - half, x + half, y + half))
        square = crop.resize((pixels, pixels), Image.Resampling.LANCZOS)
        return add_rim(square, max(1, round(RIM_POINTS * pixels / FACE_POINTS)))

    for state in PORTRAIT:
        name = state.removeprefix("sit-")
        for scale in SCALES:
            suffix = "@2x" if scale == 2 else ""
            save(
                out / "face" / f"{name}{suffix}.png", render(state, FACE_POINTS * scale)
            )
    for pixels in FAVICONS:
        save(out / "favicon" / f"{pixels}.png", render("sit-idle", pixels))


def copy_console(art: Path, out: Path) -> int:
    """Copy the pictures the console serves from `art` into `out`; the number of bytes."""
    out.mkdir(parents=True, exist_ok=True)
    for source, name in CONSOLE_FILES.items():
        shutil.copyfile(art / source, out / name)
    return sum((out / name).stat().st_size for name in CONSOLE_FILES.values())


def build(picks: Path, out: Path) -> int:
    """Write every file under `out`; the number of bytes of the pictures."""
    spec = json.loads(picks.read_text())
    root = Path(spec["root"])
    masters = {
        state: defringe(root / spec["picks"][state]) for state in (*PET, *PORTRAIT)
    }
    anchors = {
        **family(masters, PET, "pet", out),
        **family(masters, PORTRAIT, "portrait", out),
    }
    faces(masters, out)
    document = {"unit": "pt", "scale": SCALE, "pictures": anchors}
    (out / "anchors.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    return sum(path.stat().st_size for path in out.rglob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--picks", type=Path, help="picks.json of the masters")
    parser.add_argument(
        "--out", type=Path, default=ART, help=f"where to write (default {ART})"
    )
    parser.add_argument(
        "--console",
        type=Path,
        default=CONSOLE,
        help=f"where the web console's copies go (default {CONSOLE})",
    )
    parser.add_argument(
        "--console-only",
        action="store_true",
        help="only copy the console's pictures from --out; no masters needed",
    )
    args = parser.parse_args()
    if not args.console_only:
        if args.picks is None:
            parser.error("--picks is required unless --console-only is given")
        total = build(args.picks, args.out)
        print(f"Wrote {args.out}: {total / 1024:.0f} KiB of pictures and anchors.json")
    size = copy_console(args.out, args.console)
    print(
        f"Copied {len(CONSOLE_FILES)} pictures ({size / 1024:.0f} KiB) to {args.console}"
    )


if __name__ == "__main__":
    main()
