#!/usr/bin/env python3
"""Make Melo's pictures (the moves of the desktop pet and the console's portrait, their stills and Melo's face)
from an art pack. This is the engine of `scripts/art_pack.py`, the one command that checks a pack and replaces
every picture of Melo (see docs/guides/art-pack.md); run it alone to make only the moves:

    uv run --with pillow==13.0.0 python scripts/build_moves.py --source <pack folder> --stills <stills folder>

Pillow is not a dependency of localmelo: `--with` fetches it for this one run. The committed files were made with
Pillow 13.0.0, whose wheel bundles libwebp 1.6.0; another Pillow or libwebp may encode other bytes, so keep the pin
unless the files are made again on purpose. The pack stays outside the repository. Its folder holds `manifest.json` and `frames/<state>/frame_NN.png`: transparent frames, all on one
square canvas, for the six states below. The manifest gives each state its frames, a duration per frame
(`frame_durations_ms`), whether it plays once (`playback: "once"` or `loop_count: 1`) or loops, the frame that
stands for it when nothing may move (`poster_frame_index`), and the canvas with the point the paws rest on
(`canvas.anchor`).

The stills folder (optional) holds done.png, error.png and offline.png: one transparent picture each on the pack's
canvas, the paws on the pack's anchor line and centred like its frames. They are what Melo holds when nothing moves for
a reason: a run that ended well, a failure, and nothing known or no connection. It may also hold head-smile.png and
head-neutral.png: Melo's front-facing head on the same canvas (the bust fitted in HEAD_BOX), drawn for small sizes.
Their masters stay out of the repository too. Without the folder those three posters are not written and the faces
are cut from the pack's posters instead; without a head the face it gives is cut from the pack too. (`art_pack.py`
passes `fill=True`, which writes a missing still as a copy of the poster that stands in for it, as the app does.)

What it writes in support/gateway/static/moves, committed, served by the console and read by the app from the same
folder inside its bundled package (scripts/build_macos_app.py):

- <state>.webp: the state's frames scaled to CELL pixels (the whole canvas is scaled, never one frame cut on its
  own, so the paws stay on the anchor), in a COLUMNS x ROWS sheet, left to right and top to bottom. Lossy WebP with
  a lossless alpha: macOS 11 and every current browser read it.
- poster-<state>.webp: each state's poster frame alone, CELL pixels: the empty chat, the app's start window, and a
  state shown still (reduced motion, or a sheet that cannot be read).
- poster-done.webp, poster-error.webp, poster-offline.webp: the stills, CELL pixels, scaled as the frames are: done
  (eyes closed, happy, tail up), error (ears back, head low) and offline (curled up). The panel and the desktop pet
  show them for a run that just ended, a failure, and nothing known or no connection.
- face-<name>.png: Melo's head, cut from a state's poster frame by that name's box in FACES (pixels of a 512 px
  canvas; the box scales with the pack's canvas), FACE_PIXELS square: the reply icons, the approval bar, the
  brand, the phone's panel chip and the app's activity pill. idle is the sitting dog the wake-up ends on, working the
  dog at its laptop, waiting the raised paw; done and error are cut from those stills by STILL_FACES, or from the
  wake-up's end and the thinking poster when there are no stills. With the heads, idle is head-smile (the reply
  avatar, the Main chat, the phone chip), and face-brand.png is head-neutral (the sidebar and sign-in brand); without
  them face-brand is the idle face.
- favicon-16.png, favicon-32.png, favicon-64.png: the brand's head cut closer, for the browser tab: head-neutral by
  HEAD_FAVICON_BOX, or the wake-up's end by FAVICON_BOX without the heads.
- manifest.json: the canvas, the anchor and the sheet's grid, and per state its file, its pose, once or loop, the
  poster frame and the frame durations; and the cross-fade times between states.

The same input gives the same bytes. The run stops if the moves would weigh more than MAX_BYTES.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeGuard

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
# The faces cut from the stills instead, when they are given: the still and its box, as above.
STILL_FACES = {
    "done": ("done", (172, 58, 368, 254)),
    "error": ("error", (177, 52, 397, 272)),
}
# The stills, by the names their files have in the stills folder and their posters in the moves.
STILLS = ("done", "error", "offline")
# The poster that stands in for a still the pack does not have: the same the app shows (PetStill.standIn).
STILL_STAND_INS = {"done": "startup", "error": "thinking", "offline": "idle_rest"}
# The heads drawn for small sizes, when the stills folder has them: the face each gives, and the box the bust is fitted
# in (512 px canvas: 464 px square, bottom on the anchor line); the tab icon is cut closer, the neck left out.
HEADS = {"idle": "head-smile", "brand": "head-neutral"}
HEAD_BOX = (24, 24, 488, 488)
HEAD_FAVICON_BOX = (56, 24, 456, 424)
FACE_PIXELS = 96  # twice the largest size a face is shown at (48 pt)
# The browser tab's icon: the idle head, closer, so it still reads at 16 px.
FAVICON_BOX = (158, 74, 342, 258)
FAVICONS = (16, 32, 64)
MAX_BYTES = 2_000_000
# The largest side of any picture of a pack: a canvas, a frame, a still, a head or a brand picture.
MAX_SIDE = 4096

# Where each file of the moves came from: the pack pictures it is made of, and how.
Trace = dict[str, tuple[list[Path], str]]


class PackError(ValueError):
    """A pack that does not hold what the manifest says, worded for the person who ran the build."""


def whole(value: Any) -> TypeGuard[int]:
    """A whole number (a JSON integer; `true` and `false` are not numbers here)."""
    return isinstance(value, int) and not isinstance(value, bool)


def manifest_findings(source: Path, manifest: Any) -> tuple[list[str], list[str]]:
    """What is wrong with a pack's manifest (problems, which stop a build) and what looks wrong (warnings), each as
    "rule: what is wrong" with the rule in plain words. It looks at the manifest and at whether each frame is a file
    of the pack; the pictures themselves are looked at by art_pack.py."""
    problems: list[str] = []
    warnings: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest: manifest.json must hold one JSON object"], warnings
    canvas = manifest.get("canvas") or {}
    canvas = canvas if isinstance(canvas, dict) else {}
    width, height = canvas.get("width"), canvas.get("height")
    anchor = canvas.get("anchor") or {}
    anchor = anchor if isinstance(anchor, dict) else {}
    side: int = int(width) if whole(width) and width == height and width > 0 else 0
    if not side:
        problems.append(
            "square canvas: the pack's canvas must be square (canvas.width and canvas.height the same whole number)"
        )
    elif side > MAX_SIDE:
        problems.append(
            f"sane size: the canvas is {side} px a side, more than {MAX_SIDE}"
        )
    elif side < CELL:
        warnings.append(
            f"canvas size: the canvas is {side} px a side; the moves are made at {CELL} px, so the pictures "
            "are scaled up and look soft"
        )
    if not all(
        isinstance(anchor.get(key), int | float)
        and not isinstance(anchor.get(key), bool)
        for key in ("x", "y")
    ):
        problems.append(
            "anchor: the pack's canvas has no anchor (canvas.anchor with a number x and y)"
        )
    elif side and not (0 <= anchor["x"] <= side and 0 <= anchor["y"] <= side):
        problems.append(
            f"anchor: the anchor ({anchor['x']}, {anchor['y']}) is outside the {side} x {side} canvas"
        )
    states = manifest.get("states") or {}
    states = states if isinstance(states, dict) else {}
    missing = [state for state in POSES if state not in states]
    if missing:
        problems.append(f"six states: the pack has no {', '.join(missing)}")
    extra = [state for state in states if state not in POSES]
    if extra:
        warnings.append(
            f"six states: the manifest also lists {', '.join(map(str, extra))}; only the six are used"
        )
    root = source.resolve()
    for state in POSES:
        if state not in states:
            continue
        entry = states[state]
        if not isinstance(entry, dict):
            problems.append(f"six states: {state} must be an object")
            continue
        frames, durations = (
            entry.get("frames") or [],
            entry.get("frame_durations_ms") or [],
        )
        frames = frames if isinstance(frames, list) else []
        durations = durations if isinstance(durations, list) else []
        if not 0 < len(frames) <= COLUMNS * ROWS:
            problems.append(
                f"frame count: {state}: 1 to {COLUMNS * ROWS} frames, not {len(frames)}"
            )
        if len(durations) != len(frames) or not all(
            whole(ms) and ms > 0 for ms in durations
        ):
            problems.append(
                f"durations: {state}: one positive duration (ms) per frame, as whole numbers"
            )
        playback, loops = entry.get("playback"), entry.get("loop_count")
        if playback not in (None, "once", "loop") or not (
            loops is None or (whole(loops) and loops >= 0)
        ):
            problems.append(
                f'playback: {state}: playback is "once" or "loop", and loop_count 0 (loops) or 1 (once)'
            )
        else:
            if state == "startup" and not once(entry):
                warnings.append(
                    "playback: startup loops; the wake-up is meant to play once and hold its last frame"
                )
            if state not in ("startup", "idle_play") and once(entry):
                warnings.append(
                    f"playback: {state} plays once and stops on its last frame; it is meant to loop"
                )
        poster = entry.get("poster_frame_index", 0)
        if not (whole(poster) and 0 <= poster < len(frames)):
            problems.append(
                f"poster frame: {state}: the poster is not one of its frames"
            )
        for name in frames:
            if not isinstance(name, str):
                problems.append(f"frame files: {state}: a frame name must be text")
                continue
            path = (source / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                problems.append(
                    f"frame files: {state}: {name} is not a file of the pack"
                )
    return problems, warnings


def read_pack(source: Path) -> dict[str, Any]:
    """The pack's manifest, checked: every state with its frames, durations, playback and poster, one canvas."""
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PackError(f"Cannot read {source / 'manifest.json'}: {error}") from error
    problems, _ = manifest_findings(source, manifest)
    if problems:
        raise PackError(problems[0])
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


def read_stills(folder: Path, canvas: int) -> dict[str, Path]:
    """The stills folder, checked: done.png, error.png and offline.png, each a file on the pack's canvas, and the
    heads (HEADS) it may hold, on the same canvas."""
    found: dict[str, Path] = {}
    for name in (*STILLS, *HEADS.values()):
        path = folder / f"{name}.png"
        if not path.is_file():
            if name in HEADS.values():
                continue
            raise PackError(f"The stills folder has no {name}.png")
        found[name] = path
    return check_pictures(found, canvas)


def check_pictures(pictures: Mapping[str, Path], canvas: int) -> dict[str, Path]:
    """The stills and heads given by name (STILLS and HEADS), each a file on the pack's canvas."""
    for name, path in pictures.items():
        if name not in (*STILLS, *HEADS.values()):
            raise PackError(f"{name} is not a still or a head")
        with Image.open(path) as image:
            if image.size != (canvas, canvas):
                raise PackError(f"{name}.png is not on the {canvas} x {canvas} canvas")
    return dict(pictures)


def build(
    source: Path,
    out: Path,
    stills: Path | Mapping[str, Path] | None = None,
    trace: Trace | None = None,
    fill: bool = False,
) -> int:
    """Write the sheets, the posters, the stills, the faces, the favicons and the manifest under `out`; the bytes
    written.

    `stills` is the stills folder, or the stills and heads by name. `trace` is filled with where each file came from.
    `fill` writes the poster of a still the pack does not give as a copy of the poster that stands in for it
    (STILL_STAND_INS), so the folder always holds all nine posters.
    """
    pack = read_pack(source)
    side = pack["canvas"]["width"]
    if stills is None:
        pictures: dict[str, Path] = {}
    elif isinstance(stills, Path):
        pictures = read_stills(stills, side)
    else:
        pictures = check_pictures(stills, side)
    notes: Trace = trace if trace is not None else {}
    scale = CELL / side
    states: dict[str, dict[str, Any]] = {}
    posters: dict[str, Image.Image] = {}
    poster_files: dict[str, Path] = {}
    encodes: list[tuple[Path, Image.Image]] = []
    for state, pose in POSES.items():
        entry = pack["states"][state]
        sheet = Image.new("RGBA", (COLUMNS * CELL, ROWS * CELL))
        frames = [scaled(source / name, side, CELL) for name in entry["frames"]]
        for index, frame in enumerate(frames):
            sheet.alpha_composite(
                frame, ((index % COLUMNS) * CELL, (index // COLUMNS) * CELL)
            )
        encodes.append((out / f"{state}.webp", sheet))
        notes[f"{state}.webp"] = (
            [source / name for name in entry["frames"]],
            f"{len(frames)} frames",
        )
        poster = entry.get("poster_frame_index", 0)
        encodes.append((out / f"poster-{state}.webp", frames[poster]))
        poster_files[state] = source / entry["frames"][poster]
        notes[f"poster-{state}.webp"] = (
            [poster_files[state]],
            f"poster frame {poster}",
        )
        with Image.open(poster_files[state]) as full:
            posters[state] = full.convert("RGBA")
        states[state] = {
            "file": f"{state}.webp",
            "pose": pose,
            "playback": "once" if once(entry) else "loop",
            "poster": poster,
            "durations_ms": list(entry["frame_durations_ms"]),
        }
    held: dict[str, Image.Image] = {}
    for name, path in pictures.items():
        if name in STILLS:
            encodes.append((out / f"poster-{name}.webp", scaled(path, side, CELL)))
            notes[f"poster-{name}.webp"] = ([path], "still")
        with Image.open(path) as full:
            held[name] = full.convert("RGBA")
    faces: dict[str, Image.Image] = {}
    origins: dict[str, tuple[list[Path], str]] = {}
    for name, (state, box) in FACES.items():
        if HEADS.get(name, "") in held:
            head = HEADS[name]
            faces[name] = cut(held[head], HEAD_BOX, FACE_PIXELS)
            origins[name] = ([pictures[head]], "head fitted by HEAD_BOX")
        elif name in STILL_FACES and STILL_FACES[name][0] in held:
            still, box = STILL_FACES[name]
            faces[name] = cut(held[still], box, FACE_PIXELS)
            origins[name] = ([pictures[still]], "head cut from the still")
        else:
            faces[name] = cut(posters[state], box, FACE_PIXELS)
            origins[name] = (
                [poster_files[state]],
                f"head cut from the {state} poster",
            )
    brand = HEADS["brand"] in held
    faces["brand"] = (
        cut(held[HEADS["brand"]], HEAD_BOX, FACE_PIXELS) if brand else faces["idle"]
    )
    origins["brand"] = (
        ([pictures[HEADS["brand"]]], "head fitted by HEAD_BOX")
        if brand
        else origins["idle"]
    )
    for name, face in faces.items():
        save_png(out / f"face-{name}.png", face)
        notes[f"face-{name}.png"] = origins[name]
    for pixels in FAVICONS:
        icon = (
            cut(held[HEADS["brand"]], HEAD_FAVICON_BOX, pixels)
            if brand
            else cut(posters["startup"], FAVICON_BOX, pixels)
        )
        save_png(out / f"favicon-{pixels}.png", icon)
        notes[f"favicon-{pixels}.png"] = (
            (
                [pictures[HEADS["brand"]]],
                "head cut closer by HEAD_FAVICON_BOX",
            )
            if brand
            else (
                [poster_files["startup"]],
                "head cut from the startup poster by FAVICON_BOX",
            )
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
    out.mkdir(parents=True, exist_ok=True)
    # The WebP encoder is the slow part and works without the interpreter lock: encode the files side by side.
    # Each file is encoded alone, so the bytes do not depend on the order or the number of workers.
    workers = max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda job: save(*job), encodes))
    if fill:
        for name, state in STILL_STAND_INS.items():
            if name not in pictures:
                shutil.copyfile(
                    out / f"poster-{state}.webp", out / f"poster-{name}.webp"
                )
                notes[f"poster-{name}.webp"] = (
                    [poster_files[state]],
                    f"stand-in: a copy of the {state} poster",
                )
    (out / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    notes["manifest.json"] = ([source / "manifest.json"], "timelines")
    total = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    if total > MAX_BYTES:
        raise PackError(
            f"size budget: the moves weigh {total / 1e6:.2f} MB, more than {MAX_BYTES / 1e6:.1f} MB: "
            "lower QUALITY or CELL"
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="The pack is read only; nothing in it is run. Its full-size frames stay out of the repository. "
        "scripts/art_pack.py checks a pack first and replaces every picture of Melo.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="the art pack's folder: manifest.json and frames/<state>/frame_NN.png",
    )
    parser.add_argument(
        "--stills",
        type=Path,
        help="the stills' folder: done.png, error.png and offline.png on the pack's canvas, and maybe "
        "head-smile.png and head-neutral.png",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT, help=f"where to write (default {OUT})"
    )
    args = parser.parse_args()
    try:
        total = build(args.source, args.out, args.stills)
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
