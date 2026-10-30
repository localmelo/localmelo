"""Melo's committed pictures (macos/art/akita, made by scripts/build_pet_assets.py): every state has its files, the
pixel sizes follow the anchors, the pictures have a real alpha, and the anchors lie inside their canvas. The web
console's copies (support/gateway/static/akita) are the same files, and the rim keeps them visible on its surfaces.
"""

from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "macos/art/akita"
CONSOLE = ROOT / "support/gateway/static/akita"
PET = [
    "lie-idle",
    "lie-working",
    "lie-waiting",
    "lie-done",
    "lie-error",
    "rest-lazy",
    "sleep-front",
    "sleep-side",
    "sleep-back",
]
PORTRAIT = ["sit-idle", "sit-working", "sit-waiting", "sit-done", "sit-error"]
FACES = ["idle", "working", "waiting", "done", "error"]
FAVICONS = [16, 32, 64]
FACE_POINTS = 48
MARGIN = 3  # points of clear canvas around the dogs, as the script leaves them
MAX_BYTES = 1_500_000
# What the console serves, as file below the art folder -> file below its own folder: one 2x portrait, the five
# faces and the three favicons (the same table as CONSOLE_FILES in the script, which Pillow keeps out of the tests).
CONSOLE_FILES = {
    "portrait/sit-idle@2x.png": "sit-idle.png",
    **{f"face/{name}@2x.png": f"face-{name}.png" for name in FACES},
    **{f"favicon/{size}.png": f"favicon-{size}.png" for size in FAVICONS},
}
MAX_CONSOLE_BYTES = 160_000  # the console's pictures weigh about 150 KB; the two logos they replaced, about 80 KB


def anchors() -> dict[str, dict[str, Any]]:
    document = json.loads((ART / "anchors.json").read_text())
    assert document["unit"] == "pt"
    pictures: dict[str, dict[str, Any]] = document["pictures"]
    return pictures


def header(path: Path) -> tuple[int, int, int, int]:
    """Width, height, bit depth and colour type of a PNG (colour type 6 is RGBA)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">IIBB", data[16:26])  # type: ignore[no-any-return]


def rgba_plane(path: Path) -> np.ndarray:
    """An 8-bit RGBA PNG as (height, width, 4) bytes, read without an imaging library."""
    data = path.read_bytes()
    width, height = struct.unpack(">II", data[16:24])
    pos, packed = 8, b""
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        if kind == b"IDAT":
            packed += data[pos + 8 : pos + 8 + length]
        pos += 12 + length
    raw, stride = zlib.decompress(packed), width * 4
    out = bytearray(height * stride)
    for y in range(height):
        kind, row, up = raw[y * (stride + 1)], y * stride, (y - 1) * stride
        for i in range(stride):
            a = out[row + i - 4] if i >= 4 else 0
            b = out[up + i] if y else 0
            c = out[up + i - 4] if y and i >= 4 else 0
            if kind == 0:
                guess = 0
            elif kind == 1:
                guess = a
            elif kind == 2:
                guess = b
            elif kind == 3:
                guess = (a + b) // 2
            else:  # Paeth
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                guess = a if pa <= pb and pa <= pc else b if pb <= pc else c
            out[row + i] = (raw[y * (stride + 1) + 1 + i] + guess) & 255
    return np.frombuffer(bytes(out), np.uint8).reshape(height, width, 4)


def alpha_plane(path: Path) -> np.ndarray:
    """The alpha of an 8-bit RGBA PNG as (height, width) bytes."""
    return rgba_plane(path)[..., 3]


def test_the_folder_holds_exactly_the_expected_files() -> None:
    expected = {"anchors.json"}
    for folder, names in (("pet", PET), ("portrait", PORTRAIT), ("face", FACES)):
        expected |= {f"{folder}/{n}{s}.png" for n in names for s in ("", "@2x")}
    expected |= {f"favicon/{n}.png" for n in FAVICONS}
    found = {str(p.relative_to(ART)) for p in ART.rglob("*") if p.is_file()}
    assert found == expected
    # The design asked for about 1.2 MB; a larger set has to be a deliberate change.
    assert sum(p.stat().st_size for p in ART.rglob("*.png")) <= MAX_BYTES


def test_every_picture_has_its_pixel_size_in_both_scales() -> None:
    pictures = anchors()
    assert sorted(pictures) == sorted(PET + PORTRAIT)
    for folder, names in (("pet", PET), ("portrait", PORTRAIT)):
        for name in names:
            width, height = pictures[name]["canvas"]
            for suffix, scale in (("", 1), ("@2x", 2)):
                path = ART / folder / f"{name}{suffix}.png"
                assert header(path) == (width * scale, height * scale, 8, 6), path
    for name in FACES:
        for suffix, scale in (("", 1), ("@2x", 2)):
            size = FACE_POINTS * scale
            assert header(ART / "face" / f"{name}{suffix}.png") == (size, size, 8, 6)
    for pixels in FAVICONS:
        assert header(ART / "favicon" / f"{pixels}.png") == (pixels, pixels, 8, 6)


def test_the_pictures_say_they_are_srgb() -> None:
    for path in ART.rglob("*.png"):
        assert b"sRGB" in path.read_bytes()[:64], path


def test_anchors_lie_inside_their_canvas_and_a_family_swaps_without_a_jump() -> None:
    pictures = anchors()
    for names in (PET, PORTRAIT):
        first = pictures[names[0]]
        for name in names:
            anchor = pictures[name]
            wide, tall = anchor["canvas"]
            x, y, width, height = anchor["box"]
            # One canvas, one ground line and one centre for the family: a swap moves nothing.
            assert anchor["canvas"] == first["canvas"], name
            assert anchor["ground"] == first["ground"] == tall - MARGIN
            assert anchor["centre"] == first["centre"] == wide / 2
            # The dog rests on the ground line, is centred, and keeps clear of the canvas edge.
            assert y + height == anchor["ground"], name
            assert x >= MARGIN and x + width <= wide - MARGIN and y >= MARGIN
            assert abs(x + width / 2 - anchor["centre"]) <= 0.5, name


@pytest.mark.parametrize(
    "folder, name",
    [
        ("pet", "lie-idle"),
        ("pet", "sleep-back"),
        ("portrait", "sit-error"),
        ("face", "done"),
    ],
)
def test_the_alpha_is_real_clear_round_the_dog_and_solid_inside(
    folder: str, name: str
) -> None:
    path = ART / folder / f"{name}@2x.png"
    alpha = alpha_plane(path)
    assert (alpha == 0).sum() > 0.1 * alpha.size  # a cut-out, not a rectangle
    assert (alpha == 255).mean() > 0.2  # and a solid dog inside it
    # The canvas margin holds the rim and then stays clear, so a window of the picture's size clips nothing.
    if folder != "face":
        # (The rim reaches 3 px out at @2x; the margin is 6.)
        assert (
            not alpha[:3].any() and not alpha[:, :3].any() and not alpha[:, -3:].any()
        )
        # The paws rest on the ground line: the last solid row is the one above it.
        ground = anchors()[name]["ground"] * 2
        solid_rows = np.nonzero((alpha >= 250).any(axis=1))[0]
        assert abs(int(solid_rows[-1]) + 1 - ground) <= 1
        # Below the paws only the rim shows, which is never fully opaque.
        assert alpha[int(solid_rows[-1]) + 1 :].max() < 250


def test_the_console_serves_exactly_the_pictures_it_needs_as_copies_of_the_art() -> (
    None
):
    found = {str(p.relative_to(CONSOLE)) for p in CONSOLE.rglob("*") if p.is_file()}
    assert found == set(CONSOLE_FILES.values())
    for source, name in CONSOLE_FILES.items():
        assert (CONSOLE / name).read_bytes() == (ART / source).read_bytes(), name
    assert sum(p.stat().st_size for p in CONSOLE.iterdir()) <= MAX_CONSOLE_BYTES
    # The portrait is the sitting canvas at 2x, the faces are 2x of 48 pt, the favicons are the three sizes.
    wide, tall = anchors()["sit-idle"]["canvas"]
    assert header(CONSOLE / "sit-idle.png") == (wide * 2, tall * 2, 8, 6)
    for name in FACES:
        assert header(CONSOLE / f"face-{name}.png") == (96, 96, 8, 6)
    for pixels in FAVICONS:
        assert header(CONSOLE / f"favicon-{pixels}.png") == (pixels, pixels, 8, 6)


def luminance(rgb: np.ndarray) -> float:
    """Relative luminance of an sRGB colour with channels in 0..1 (WCAG 2.1)."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def contrast(one: np.ndarray, other: np.ndarray) -> float:
    high, low = sorted((luminance(one), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def console_surfaces() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The console's surface tokens that Melo sits on, as (light, dark) colours."""
    css = (ROOT / "support/gateway/static/workspace.css").read_text()
    found = {}
    for name in ("canvas", "sidebar", "surface", "raised", "tint"):
        match = re.search(
            rf"--{name}: light-dark\((#[0-9a-f]{{6}}), (#[0-9a-f]{{6}})\)", css
        )
        assert match, name
        found[name] = tuple(
            np.array([int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)])
            for hex_ in match.groups()
        )
    return found  # type: ignore[return-value]


@pytest.mark.parametrize("name", ["sit-idle", "face-idle", "face-waiting"])
def test_the_picture_stays_visible_on_every_console_surface_in_both_themes(
    name: str,
) -> None:
    pixels = rgba_plane(CONSOLE / f"{name}.png").astype(float) / 255
    alpha, colour = pixels[..., 3], pixels[..., :3]
    shade = np.array([luminance(c) for c in colour.reshape(-1, 3)]).reshape(alpha.shape)
    # The navy outline is the solid dark layer; the rim is the cream layer round it, translucent.
    outline = np.median(colour[(alpha > 0.99) & (shade < 0.03)], axis=0)
    rim_pixels = (alpha > 0.5) & (alpha < 0.9) & (colour.min(axis=-1) > 0.85)
    assert rim_pixels.sum() > 50, "the rim is there"
    rim, rim_alpha = np.median(colour[rim_pixels], axis=0), np.median(alpha[rim_pixels])
    for surface, (light, dark) in console_surfaces().items():
        # Light: the navy outline sets the dog off (WCAG 1.4.11 asks 3:1 of a picture that carries meaning).
        assert contrast(outline, light) >= 7, (name, surface, "light outline")
        # Dark: the outline vanishes, so the rim, as it looks on the surface, has to.
        edge = rim * rim_alpha + dark * (1 - rim_alpha)
        assert contrast(edge, dark) >= 4.5, (name, surface, "dark rim")
