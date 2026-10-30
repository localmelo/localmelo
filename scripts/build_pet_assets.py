#!/usr/bin/env python3
"""Make Melo's pictures for the macOS app from the chosen art masters. Run it by hand, on a Mac, when the art changes:

    uv run --with pillow python scripts/build_pet_assets.py --picks <picks.json>

Pillow is not a dependency of localmelo: `--with` fetches it for this one run. The files it writes in
macos/art/akita are committed, and scripts/build_macos_app.py only copies them into the app. The masters (about
600 to 700 px wide, cut out with a soft edge) stay outside the repository; `picks.json` lists them as
`{"root": DIR, "picks": {STATE: PATH below DIR}, "motion": {EDIT: PATH below DIR}}` for the 14 states below and the
three edits of the moves (`sit-idle-blink`, `lie-idle-blink`, `lie-done-notail`; the cut-outs sit next to a
cut-report.json that says where each one lies in the generated picture). The same input gives the same bytes.

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
- the moves' pieces, each on the canvas of the picture it belongs to (so a layer is placed with the picture's own
  frame): pet/lie-idle-blink, portrait/sit-idle-blink and face/idle-blink are eyes-closed patches (the edit's pixels
  where it differs from its base, nothing elsewhere); pet/lie-done-body and pet/lie-done-tail are the picture cut at
  the tail, which wags about its pivot. Every edit is checked against its base first (check_alignment).
- anchors.json: per picture, its canvas (pt), `ground` (pt from the top: where the paws rest), `centre` (pt from the
  left) and `box` (x, y, width, height of the drawn dog on the canvas, pt, from the @2x file); `eyes` (the eye band
  of a blink patch), `tail` (its box and pivot) and `z` (where a sleeping dog's "z" starts), all in pt.

The web console shows the same pictures, so the same run copies the ones it needs, at 2x, into
support/gateway/static/akita (see CONSOLE_FILES); `--console-only` makes that copy again from the committed files in
macos/art/akita, without the masters.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np
from PIL import (  # type: ignore[import-not-found]
    Image,
    ImageDraw,
    ImageFilter,
    PngImagePlugin,
)

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
# The moves. Marked by hand once on the masters, in pixels of the base picture's cut-out: where an edit may differ
# from its base (the eye band of each blink; the box round the tail and its pivot, where it joins the back).
BLINKS = {
    "sit-idle": ("sit-idle-blink", (103, 174, 309, 248)),
    "lie-idle": ("lie-idle-blink", (95, 164, 303, 232)),
}
TAIL = {"lie-done": ("lie-done-notail", (468, 162, 678, 366), (548, 290))}
# Where a sleeping dog's "z" starts, in points of the pet's canvas.
SLEEP_Z = {"sleep-front": (86, 24), "sleep-side": (62, 22), "sleep-back": (30, 40)}
OPEN = 7  # px: parts of the tail mask thinner than this are not the tail
EDIT_GAP = 48 / 255  # a colour gap this large means the edit changed the pixel
MAX_MISMATCH = 0.03  # share of pixels outside the band the edit may still differ in (it re-draws everything a little)
# What the web console serves, as file below the art folder -> file below CONSOLE: the idle portrait of the empty
# chat, the five faces (the reply icons, the approval bar, the brand) and the three favicons. The portrait and the
# faces are the @2x files, shown at half their size, so one file looks sharp on any screen the console meets.
CONSOLE_FILES = {
    "portrait/sit-idle@2x.png": "sit-idle.png",
    "portrait/sit-idle-blink@2x.png": "sit-idle-blink.png",
    **{f"face/{name}@2x.png": f"face-{name}.png" for name in FACE_NAMES},
    "face/idle-blink@2x.png": "face-idle-blink.png",
    **{f"favicon/{size}.png": f"favicon-{size}.png" for size in FAVICONS},
}


class AlignmentError(ValueError):
    """An edit that does not lie on its base, so a patch cut from it would not fit."""


def hardened(image: Image.Image) -> Image.Image:
    """The master with the cut-out's soft alpha ramp pulled to a clean edge at its half-way line."""
    pixels = np.array(image.convert("RGBA"))
    alpha = np.clip((pixels[..., 3] / 255 - 0.5) * EDGE_GAIN + 0.5, 0, 1)
    pixels[..., 3] = np.rint(alpha * 255)
    pixels[alpha == 0, :3] = 0
    return Image.fromarray(pixels)


def defringe(path: Path) -> Image.Image:
    return hardened(Image.open(path))


def origin(path: Path) -> tuple[int, int]:
    """Where the cut-out `path` lies in the generated picture (x, y of its top left), from its cut-report.json."""
    report = json.loads((path.parent / "cut-report.json").read_text())[path.name]
    return report["bbox"][0], report["bbox"][1]


def in_frame(
    edit: Image.Image,
    edit_at: tuple[int, int],
    base: Image.Image,
    base_at: tuple[int, int],
) -> Image.Image:
    """The cut-out `edit` laid on a transparent copy of `base`'s frame, both placed as the generated pictures were."""
    frame = Image.new("RGBA", base.size)
    frame.paste(edit, (edit_at[0] - base_at[0], edit_at[1] - base_at[1]))
    return frame


def gap(base: Image.Image, edit: Image.Image) -> np.ndarray:
    """Per pixel, 0 to 1: how far `edit` is from `base` (a pixel drawn in one and not the other, or another colour)."""
    one, two = (np.asarray(image, np.float32) / 255 for image in (base, edit))
    both = np.minimum(one[..., 3], two[..., 3])
    colour = np.abs(one[..., :3] - two[..., :3]).max(axis=-1) * both
    return np.asarray(np.maximum(np.abs(one[..., 3] - two[..., 3]), colour))


def changed_in(
    base: Image.Image, edit: Image.Image, box: tuple[int, int, int, int]
) -> np.ndarray:
    """A mask of the pixels inside `box` where `edit` differs from `base`."""
    x0, y0, x1, y1 = box
    mask = np.zeros((base.height, base.width), bool)
    mask[y0:y1, x0:x1] = gap(base, edit)[y0:y1, x0:x1] > EDIT_GAP
    return mask


def softened(mask: np.ndarray, grow: int, blur: float) -> np.ndarray:
    """`mask` grown by `grow` px and feathered by `blur` px, as 0 to 1."""
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    if grow:
        image = image.filter(ImageFilter.MaxFilter(2 * grow + 1))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255


def check_alignment(
    name: str, base: Image.Image, edit: Image.Image, box: tuple[int, int, int, int]
) -> None:
    """Stop unless `edit` is `base` drawn again in place outside `box`: it fits best with no shift (of up to two
    pixels), and few pixels differ. A model that moved, scaled or redrew the dog would leave a seam.
    """
    one, two = np.asarray(base), np.asarray(edit)
    drawn = (one[..., 3] == 255) & (two[..., 3] == 255)
    x0, y0, x1, y1 = box
    drawn[y0:y1, x0:x1] = False

    def mismatch(dx: int, dy: int) -> float:
        moved = np.roll(two[..., :3], (dy, dx), axis=(0, 1)).astype(np.int16)
        far = np.abs(one[..., :3].astype(np.int16) - moved).max(axis=-1) > 48
        return float(far[drawn].mean())

    shifts = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    best = min(shifts, key=lambda shift: mismatch(*shift))
    here = mismatch(0, 0)
    if best != (0, 0) or here > MAX_MISMATCH:
        raise AlignmentError(
            f"{name} does not lie on its base outside the marked box: best shift {best}, "
            f"{here:.1%} of the pixels differ (at most {MAX_MISMATCH:.0%}). Generate it again."
        )


def blink_patch(
    base: Image.Image, edit: Image.Image, box: tuple[int, int, int, int]
) -> Image.Image:
    """The eyes-closed patch on the base's frame: the edit's pixels where it differs from the base inside the eye
    band, grown a little and feathered; transparent everywhere else, so only the eyes ever change.
    """
    soft = softened(changed_in(base, edit, box), 3, 1.5)
    patch = np.array(edit)
    patch[..., 3] = np.rint(soft * np.asarray(base)[..., 3]).astype(np.uint8)
    return Image.fromarray(patch)


def tail_split(
    base: Image.Image, edit: Image.Image, box: tuple[int, int, int, int]
) -> tuple[Image.Image, Image.Image]:
    """The base cut at its tail, from the same picture drawn without it: (body, mask). The mask is where the edit
    lacks what the base has, inside `box`, holes filled (tail fur that is the colour of the back), feathered. The
    body is the base outside the mask and the edit inside it, so it shows the back where the tail was.
    """
    hard = Image.fromarray(
        changed_in(base, edit, box).astype(np.uint8) * 255
    ).copy()  # a writable one
    ImageDraw.floodfill(
        hard, (0, 0), 128
    )  # the outside; what it did not reach is the tail, holes included
    whole = Image.fromarray(((np.asarray(hard) != 128) * 255).astype(np.uint8))
    # An opening drops slivers of the back's outline that the edit drew a little differently where the tail joins it.
    opened = whole.filter(ImageFilter.MinFilter(OPEN)).filter(
        ImageFilter.MaxFilter(OPEN)
    )
    soft = softened(np.asarray(opened) > 0, 2, 1.5)[..., None]
    one, two = (np.asarray(image, np.float32) for image in (base, edit))
    alpha = (one[..., 3:] * (1 - soft) + two[..., 3:] * soft) / 255
    colour = (
        one[..., :3] * one[..., 3:] * (1 - soft) + two[..., :3] * two[..., 3:] * soft
    ) / 255
    colour = colour / np.maximum(alpha, 1e-6)
    body = np.dstack([colour, alpha * 255]).clip(0, 255)
    return Image.fromarray(np.rint(body).astype(np.uint8)), Image.fromarray(
        np.rint(soft[..., 0] * 255).astype(np.uint8)
    )


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


def trimmed_box(master: Image.Image) -> tuple[int, int, int, int]:
    """The box of the picture's drawn pixels (a mask: its non-zero ones)."""
    box = (master.getchannel("A") if "A" in master.getbands() else master).getbbox()
    assert box is not None, "an empty picture"
    return (box[0], box[1], box[2], box[3])


def trimmed(master: Image.Image) -> Image.Image:
    """The master cut to the box of its drawn pixels."""
    return master.crop(trimmed_box(master))


def split_tail(
    still: Image.Image, body: Image.Image, mask: Image.Image, radius: int
) -> tuple[Image.Image, Image.Image]:
    """The still picture cut at the tail, on its canvas: (body, tail). The body is the picture without the tail, with
    its own rim. The tail is the still where the mask is, plus the rim round it where no body is, so it carries its
    rim when it swings; the two laid together give back the still."""
    keep = np.asarray(mask, np.float32) / 255
    near = (
        np.asarray(mask.filter(ImageFilter.MaxFilter(2 * (radius + 2) + 1)), np.float32)
        / 255
    )
    bare = 1 - np.asarray(body.getchannel("A"), np.float32) / 255
    tail = np.array(still)
    tail[..., 3] = np.rint(tail[..., 3] * np.maximum(keep, near * bare))
    return add_rim(body, radius), Image.fromarray(tail)


def laid(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    canvas: tuple[int, int],
    at: tuple[int, int],
) -> Image.Image:
    """A picture on a master's frame, cut to the dog's box, scaled to `size` and laid on the canvas at `at`."""
    dog = frame.crop(box).resize(size, Image.Resampling.LANCZOS)
    sheet = Image.new(frame.mode, canvas)
    if frame.mode == "RGBA":
        sheet.alpha_composite(dog, at)
    else:
        sheet.paste(dog, at)
    return sheet


def family(
    masters: dict[str, Image.Image],
    states: Sequence[str],
    folder: str,
    out: Path,
    patches: dict[str, tuple[str, Image.Image]],
    splits: dict[str, tuple[Image.Image, Image.Image]],
) -> dict[str, dict[str, object]]:
    """Export `states` on one shared canvas as `folder`/STATE.png and @2x; their anchors. `patches` (state to the
    patch's name and picture) and `splits` (state to the body without the tail and the tail's mask) are on the
    masters' frames; they leave on the canvas of their state, so each lies exactly over its picture.
    """
    boxes = {state: trimmed_box(masters[state]) for state in states}
    cut = {state: masters[state].crop(boxes[state]) for state in states}
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
            suffix = "@2x" if scale == 2 else ""
            size = (round(points[state][0] * scale), round(points[state][1] * scale))
            left = (wide * scale - size[0]) // 2
            top = tall * scale - MARGIN * scale - size[1]
            lay = partial(
                laid,
                box=boxes[state],
                size=size,
                canvas=(wide * scale, tall * scale),
                at=(left, top),
            )
            rim = RIM_POINTS * scale
            still = add_rim(lay(masters[state]), rim)
            save(out / folder / f"{state}{suffix}.png", still)
            if state in patches:
                name, patch = patches[state]
                save(out / folder / f"{name}{suffix}.png", lay(patch))
            if state in splits:
                body, tail = split_tail(
                    still, lay(splits[state][0]), lay(splits[state][1]), rim
                )
                save(out / folder / f"{state}-body{suffix}.png", body)
                save(out / folder / f"{state}-tail{suffix}.png", tail)
        # Points per master pixel, and where the dog's box starts, from the @2x pass: marks go from the master's
        # frame to the canvas.
        unit = size[0] / 2 / cut[state].width, size[1] / 2 / cut[state].height
        start = left / 2, top / 2

        on_canvas = partial(to_canvas, crop=boxes[state], start=start, unit=unit)

        anchors[state] = {
            "canvas": [wide, tall],
            "ground": tall - MARGIN,
            "centre": wide / 2,
            "box": [left / 2, top / 2, size[0] / 2, size[1] / 2],  # from the @2x pass
        }
        if state in patches:
            anchors[state]["eyes"] = marked(trimmed_box(patches[state][1]), on_canvas)
        if state in splits:
            anchors[state]["tail"] = {
                "box": marked(trimmed_box(splits[state][1]), on_canvas),
                "pivot": on_canvas(*TAIL[state][2]),
            }
        if state in SLEEP_Z:
            anchors[state]["z"] = list(SLEEP_Z[state])
    return anchors


def to_canvas(
    x: float,
    y: float,
    crop: tuple[int, int, int, int],
    start: tuple[float, float],
    unit: tuple[float, float],
) -> list[float]:
    """A point on a master's frame as points on the canvas: `crop` is the dog's box on the frame, `start` where it
    lies on the canvas and `unit` the points a master pixel is."""
    return [
        round(start[0] + (x - crop[0]) * unit[0], 2),
        round(start[1] + (y - crop[1]) * unit[1], 2),
    ]


def marked(
    box: tuple[int, int, int, int], on_canvas: Callable[[float, float], list[float]]
) -> list[float]:
    """A box on a master's frame as x, y, width, height in points of the canvas."""
    x0, y0 = on_canvas(box[0], box[1])
    x1, y1 = on_canvas(box[2], box[3])
    return [x0, y0, round(x1 - x0, 2), round(y1 - y0, 2)]


def faces(masters: dict[str, Image.Image], blink: Image.Image, out: Path) -> None:
    """The sitting heads as small squares, the idle one's blink patch, and the idle head as the favicon sizes."""

    def render(picture: Image.Image, pixels: int, rimmed: bool = True) -> Image.Image:
        (x, y), half = FACE_CENTRE, FACE_SIDE // 2
        crop = picture.crop((x - half, y - half, x + half, y + half))
        square = crop.resize((pixels, pixels), Image.Resampling.LANCZOS)
        if not rimmed:
            return square
        return add_rim(square, max(1, round(RIM_POINTS * pixels / FACE_POINTS)))

    for state in PORTRAIT:
        name = state.removeprefix("sit-")
        for scale in SCALES:
            suffix = "@2x" if scale == 2 else ""
            save(
                out / "face" / f"{name}{suffix}.png",
                render(masters[state], FACE_POINTS * scale),
            )
    for scale in SCALES:
        suffix = "@2x" if scale == 2 else ""
        save(
            out / "face" / f"idle-blink{suffix}.png",
            render(blink, FACE_POINTS * scale, False),
        )
    for pixels in FAVICONS:
        save(out / "favicon" / f"{pixels}.png", render(masters["sit-idle"], pixels))


def copy_console(art: Path, out: Path) -> int:
    """Copy the pictures the console serves from `art` into `out`; the number of bytes."""
    out.mkdir(parents=True, exist_ok=True)
    for source, name in CONSOLE_FILES.items():
        shutil.copyfile(art / source, out / name)
    return sum((out / name).stat().st_size for name in CONSOLE_FILES.values())


def moves(
    root: Path, spec: dict[str, object], masters: dict[str, Image.Image]
) -> tuple[
    dict[str, tuple[str, Image.Image]], dict[str, tuple[Image.Image, Image.Image]]
]:
    """The blink patches and the tail split, each checked against its base: (patches, splits), by base state."""
    picks, edits = spec["picks"], spec["motion"]
    assert isinstance(picks, dict) and isinstance(edits, dict)

    def edit_of(state: str, name: str, box: tuple[int, int, int, int]) -> Image.Image:
        base_at = origin(root / picks[state])
        edit = in_frame(
            defringe(root / edits[name]),
            origin(root / edits[name]),
            masters[state],
            base_at,
        )
        check_alignment(name, masters[state], edit, box)
        return edit

    patches = {
        state: (name, blink_patch(masters[state], edit_of(state, name, box), box))
        for state, (name, box) in BLINKS.items()
    }
    splits = {
        state: tail_split(masters[state], edit_of(state, name, box), box)
        for state, (name, box, _) in TAIL.items()
    }
    return patches, splits


def build(picks: Path, out: Path) -> int:
    """Write every file under `out`; the number of bytes of the pictures."""
    spec = json.loads(picks.read_text())
    root = Path(spec["root"])
    masters = {
        state: defringe(root / spec["picks"][state]) for state in (*PET, *PORTRAIT)
    }
    patches, splits = moves(root, spec, masters)
    anchors = {
        **family(masters, PET, "pet", out, patches, splits),
        **family(masters, PORTRAIT, "portrait", out, patches, splits),
    }
    faces(masters, patches["sit-idle"][1], out)
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
