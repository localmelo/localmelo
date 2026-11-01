#!/usr/bin/env python3
"""Make Melo's pictures (the moves of the desktop pet and the console's portrait, their stills and Melo's face)
from an art pack.

    uv run --with pillow python scripts/build_moves.py --source <pack folder>

Pillow is not a dependency of localmelo: `--with` fetches it for this one run. The pack stays outside the
repository. Its folder holds `manifest.json` and `frames/<state>/frame_NN.png`: transparent frames, all on one
square canvas, for the six states below. The manifest gives each state its frames, a duration per frame
(`frame_durations_ms`), whether it plays once (`playback: "once"` or `loop_count: 1`) or loops, the frame that
stands for it when nothing may move (`poster_frame_index`), and the canvas with the point the paws rest on
(`canvas.anchor`).

What it writes in support/gateway/static/moves, committed, served by the console and copied into the app by
scripts/build_macos_app.py:

- <state>.webp: the state's frames scaled to CELL pixels (the whole canvas is scaled, never one frame cut on its
  own, so the paws stay on the anchor), in a COLUMNS x ROWS sheet, left to right and top to bottom. Lossy WebP with
  a lossless alpha: macOS 11 and every current browser read it.
- poster-<state>.webp: each state's poster frame alone, CELL pixels: the console's still pictures (the empty chat,
  and the panel's done, error and offline pictures) and the app's start window.
- face-<name>.png: Melo's head, cut from a state's poster frame by that name's box in FACES (pixels of a 512 px
  canvas; the box scales with the pack's canvas), FACE_PIXELS square: the reply icons, the approval bar, the
  brand, the phone's panel chip and the app's activity pill. idle and done are the sitting dog the wake-up ends on,
  working the dog at its laptop, waiting the raised paw, error the tilted head.
- favicon-16.png, favicon-32.png, favicon-64.png: the idle head cut closer (FAVICON_BOX), for the browser tab.
- manifest.json: the canvas, the anchor and the sheet's grid, and per state its file, its pose, once or loop, the
  poster frame and the frame durations; and the cross-fade times between states.

The same input gives the same bytes. The run stops if the moves would weigh more than MAX_BYTES.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "support/gateway/static/moves"

# The states, in the order the manifest lists them, and the pose each holds (a change between two states of one
# pose cross-fades a little faster). Startup ends sitting and holds that.
POSES = {
    "startup": "sit",
    "thinking": "sit",
    "working": "desk",
    "attention": "sit",
    "idle_rest": "lie",
    "idle_play": "stand",
}
# Pixels of a frame: twice the largest size a surface shows a frame at (the empty chat's 128 px portrait).
CELL = 256
COLUMNS, ROWS = 4, 4
QUALITY = 85
# Milliseconds of the cross-fade: between two states of one pose, between two poses, into "attention" (it
# interrupts), and any change while nothing may move.
FADE_MS = {"same_pose": 160, "pose_change": 220, "attention": 120, "reduced_motion": 80}
# Melo's head, by name: the state whose poster frame it is cut from and the box (x0, y0, x1, y1) in pixels of a
# 512 px canvas, square, the head centred with a margin round the ears and cheeks.
FACES = {
    "idle": ("startup", (146, 64, 354, 272)),
    "working": ("working", (148, 0, 388, 240)),
    "waiting": ("attention", (180, 0, 396, 216)),  # above the raised paw
    "done": ("startup", (146, 64, 354, 272)),
    "error": ("thinking", (180, 0, 400, 220)),
}
FACE_PIXELS = 96  # twice the largest size a face is shown at (48 pt)
# The browser tab's icon: the idle head, closer, so it still reads at 16 px.
FAVICON_BOX = (158, 74, 342, 258)
FAVICONS = (16, 32, 64)
MAX_BYTES = 2_000_000


class PackError(ValueError):
    """A pack that does not hold what the manifest says, worded for the person who ran the build."""


def read_pack(source: Path) -> dict[str, Any]:
    """The pack's manifest, checked: every state with its frames, durations, playback and poster, one canvas."""
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PackError(f"Cannot read {source / 'manifest.json'}: {error}") from error
    canvas = manifest.get("canvas") or {}
    width, height = canvas.get("width"), canvas.get("height")
    anchor = canvas.get("anchor") or {}
    if not (isinstance(width, int) and width == height and width > 0):
        raise PackError("The pack's canvas must be square")
    if not all(isinstance(anchor.get(key), int | float) for key in ("x", "y")):
        raise PackError("The pack's canvas has no anchor")
    states = manifest.get("states") or {}
    missing = [state for state in POSES if state not in states]
    if missing:
        raise PackError(f"The pack has no {', '.join(missing)}")
    for state in POSES:
        entry = states[state]
        frames, durations = (
            entry.get("frames") or [],
            entry.get("frame_durations_ms") or [],
        )
        if not 0 < len(frames) <= COLUMNS * ROWS:
            raise PackError(f"{state}: 1 to {COLUMNS * ROWS} frames, not {len(frames)}")
        if len(durations) != len(frames) or not all(
            isinstance(ms, int) and ms > 0 for ms in durations
        ):
            raise PackError(f"{state}: one positive duration (ms) per frame")
        poster = entry.get("poster_frame_index", 0)
        if not (isinstance(poster, int) and 0 <= poster < len(frames)):
            raise PackError(f"{state}: the poster is not one of its frames")
        for name in frames:
            path = (source / name).resolve()
            if not path.is_relative_to(source.resolve()) or not path.is_file():
                raise PackError(f"{state}: {name} is not a file of the pack")
    return dict(manifest)


def once(entry: dict[str, Any]) -> bool:
    return entry.get("playback") == "once" or entry.get("loop_count") == 1


def scaled(path: Path, canvas: int, side: int) -> Image.Image:
    """A frame on its `canvas` (px square) scaled to `side` pixels, on premultiplied colour (no dark halo at the
    edge)."""
    with Image.open(path) as image:
        if image.size != (canvas, canvas):
            raise PackError(f"{path.name} is not on the {canvas} x {canvas} canvas")
        frame = image.convert("RGBA")
    return (
        frame.convert("RGBa")
        .resize((side, side), Image.Resampling.LANCZOS)
        .convert("RGBA")
    )


def cut(frame: Image.Image, box: tuple[int, int, int, int], pixels: int) -> Image.Image:
    """`box` (pixels of a 512 px canvas) cut from a full-size frame and scaled to `pixels`, on premultiplied
    colour."""
    unit = frame.width / 512
    piece = frame.crop(tuple(round(edge * unit) for edge in box))
    return (
        piece.convert("RGBa")
        .resize((pixels, pixels), Image.Resampling.LANCZOS)
        .convert("RGBA")
    )


def save_png(path: Path, image: Image.Image) -> None:
    """An RGBA PNG; the colour of a clear pixel is zero, so the bytes do not drift."""
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            if pixels[x, y][3] == 0:
                pixels[x, y] = (0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)


def save(path: Path, image: Image.Image) -> None:
    """Lossy colour with a lossless alpha; the colour under a clear pixel is the encoder's to choose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "WEBP", quality=QUALITY, alpha_quality=100, method=6, exact=False)


def build(source: Path, out: Path) -> int:
    """Write the sheets, the posters, the faces, the favicons and the manifest under `out`; the bytes written."""
    pack = read_pack(source)
    side = pack["canvas"]["width"]
    scale = CELL / side
    states: dict[str, dict[str, Any]] = {}
    posters: dict[str, Image.Image] = {}
    for state, pose in POSES.items():
        entry = pack["states"][state]
        sheet = Image.new("RGBA", (COLUMNS * CELL, ROWS * CELL))
        frames = [scaled(source / name, side, CELL) for name in entry["frames"]]
        for index, frame in enumerate(frames):
            sheet.alpha_composite(
                frame, ((index % COLUMNS) * CELL, (index // COLUMNS) * CELL)
            )
        save(out / f"{state}.webp", sheet)
        poster = entry.get("poster_frame_index", 0)
        save(out / f"poster-{state}.webp", frames[poster])
        with Image.open(source / entry["frames"][poster]) as full:
            posters[state] = full.convert("RGBA")
        states[state] = {
            "file": f"{state}.webp",
            "pose": pose,
            "playback": "once" if once(entry) else "loop",
            "poster": poster,
            "durations_ms": list(entry["frame_durations_ms"]),
        }
    for name, (state, box) in FACES.items():
        save_png(out / f"face-{name}.png", cut(posters[state], box, FACE_PIXELS))
    for pixels in FAVICONS:
        save_png(
            out / f"favicon-{pixels}.png", cut(posters["startup"], FAVICON_BOX, pixels)
        )
    anchor = pack["canvas"]["anchor"]
    document = {
        "version": 1,
        "canvas": {
            "size": CELL,
            "anchor": [round(anchor["x"] * scale, 2), round(anchor["y"] * scale, 2)],
        },
        "sheet": {"columns": COLUMNS, "rows": ROWS},
        "fade_ms": FADE_MS,
        "states": states,
    }
    (out / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    total = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    if total > MAX_BYTES:
        raise PackError(
            f"The moves weigh {total / 1e6:.2f} MB, more than {MAX_BYTES / 1e6:.1f} MB: "
            "lower QUALITY or CELL"
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="The pack is read only; nothing in it is run. Its full-size frames stay out of the repository.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="the art pack's folder: manifest.json and frames/<state>/frame_NN.png",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT, help=f"where to write (default {OUT})"
    )
    args = parser.parse_args()
    try:
        total = build(args.source, args.out)
    except PackError as error:
        raise SystemExit(f"build_moves: {error}") from None
    seconds = {
        state: math.fsum(entry["frame_durations_ms"]) / 1000
        for state, entry in read_pack(args.source)["states"].items()
        if state in POSES
    }
    print(f"Wrote {args.out}: {total / 1024:.0f} KiB for {len(POSES)} states")
    for state, length in seconds.items():
        print(f"  {state}: {length:.1f} s")


if __name__ == "__main__":
    main()
