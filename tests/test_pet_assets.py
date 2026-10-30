"""Melo's committed pictures (macos/art/akita, made by scripts/build_pet_assets.py): every state has its files, the
pixel sizes follow the anchors, the pictures have a real alpha, and the anchors lie inside their canvas.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ART = Path(__file__).resolve().parents[1] / "macos/art/akita"
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


def alpha_plane(path: Path) -> np.ndarray:
    """The alpha of an 8-bit RGBA PNG as (height, width) bytes, read without an imaging library."""
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
    return np.frombuffer(bytes(out), np.uint8).reshape(height, width, 4)[..., 3]


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
