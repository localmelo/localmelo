"""A picture that goes to a model carries no metadata.

A photo's EXIF holds the camera, the time and often the GPS position; XMP, comments,
PNG text chunks and anything after the end of the image can hold more. Every picture
is rewritten to its image data (and the orientation and colour that decoding needs)
before it is kept or sent, also the ones that needed no conversion and the ones a
conversion made. No network and no model; the tests that run ``sips`` are skipped
where it is missing.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from localmelo.melo.agent._request_accounting import _image_dimensions
from localmelo.melo.schema import decode_base64
from localmelo.support.sessions import picture_input
from localmelo.support.sessions.picture_clean import (
    _orientation,
    strip,
    strip_jpeg,
    strip_png,
)
from localmelo.support.sessions.picture_input import (
    LONG_SIDE,
    PictureRefused,
    normalize,
)

from .picture_samples import (
    CAMERA,
    GPS,
    JPEG,
    NOTE,
    TRAILER,
    XMP,
    assert_clean,
    gif_picture,
    markers,
    photo,
    png_chunk,
    png_kinds,
    png_with_metadata,
    segment,
    wide,
)

needs_sips = pytest.mark.skipif(shutil.which("sips") is None, reason="macOS sips")

# ── JPEG ──


def test_a_jpeg_loses_its_gps_exif_xmp_comments_profile_and_trailer() -> None:
    raw = photo()
    for secret in (CAMERA, GPS, NOTE, b"xmpmeta", TRAILER):
        assert secret in raw, "the sample must hold what the test looks for"
    out = strip_jpeg(raw)
    assert out is not None
    assert_clean(out)
    assert _image_dimensions(out) == (8, 8)
    assert out.endswith(b"\xff\xd9")
    # Of the application segments only the orientation is left, and it holds nothing else.
    assert [m for m in markers(out) if 0xE0 <= m <= 0xEF or m == 0xFE] == [0xE1]
    assert markers(out)[0] == 0xE1
    exif = out[6 : 6 + 32]
    assert exif.startswith(b"Exif\x00\x00MM")
    assert _orientation(exif) == 6
    assert out[2:4] == b"\xff\xe1" and int.from_bytes(out[4:6], "big") == 34


def test_the_stripped_jpeg_keeps_the_image_segments_of_the_original() -> None:
    out = strip_jpeg(photo())
    assert out is not None
    original = [m for m in markers(JPEG) if m not in (0xE0, 0xE1, 0xED)]
    assert [m for m in markers(out) if m != 0xE1] == original
    # The scan is copied whole: tables, frame and entropy-coded data are the original's.
    assert out[out.index(b"\xff\xdb") :] == JPEG[JPEG.index(b"\xff\xdb") :]


def test_a_jpeg_with_the_default_orientation_keeps_no_exif_at_all() -> None:
    for out in (strip_jpeg(photo(orientation=1)), strip_jpeg(photo(orientation=None))):
        assert out is not None
        assert 0xE1 not in markers(out)
        assert_clean(out)


def test_the_adobe_colour_transform_is_kept_without_the_rest_of_the_segment() -> None:
    adobe = segment(
        0xEE, b"Adobe" + b"\x00\x65" + b"\x80\x00" + b"\xc0\x00" + b"\x01" + b"extra"
    )
    raw = JPEG[:2] + adobe + JPEG[2:]
    out = strip_jpeg(raw)
    assert out is not None
    assert segment(0xEE, b"Adobe\x00\x64\x00\x00\x00\x00\x01") in out
    assert b"extra" not in out


def test_a_progressive_jpeg_keeps_every_scan_and_a_restart_marker() -> None:
    sof = segment(0xC2, bytes([8, 0, 8, 0, 8, 1, 1, 0x11, 0]))
    dqt = segment(0xDB, b"\x00" + bytes(64))
    dht = segment(0xC4, b"\x00" + bytes(16))
    sos = segment(0xDA, bytes([1, 1, 0x00, 0, 0, 0]))
    # The first scan's data holds byte stuffing and a restart marker.
    first = sos + b"\x12\xff\x00\x34\xff\xd0\x56"
    kept = b"\xff\xd8" + dqt + sof + dht + first + dht + sos + b"\x78" + b"\xff\xd9"
    raw = b"\xff\xd8" + segment(0xE1, XMP) + kept[2:] + TRAILER
    assert strip_jpeg(raw) == kept


@pytest.mark.parametrize(
    "broken",
    [
        b"",
        b"\xff\xd8",  # no image
        JPEG[:-2],  # no end
        JPEG[: JPEG.index(b"\xff\xda") + 5],  # cut inside the scan header
        b"\xff\xd8" + segment(0xE1, b"x") + JPEG[-2:],  # no frame, no scan
        b"\xff\xd8\xff\xe1\x00\xff" + b"x" * 10,  # a length past the end
        b"\x89PNG\r\n\x1a\n",
    ],
)
def test_a_jpeg_whose_structure_does_not_hold_is_refused_not_passed_on(
    broken: bytes,
) -> None:
    assert strip_jpeg(broken) is None


# ── PNG ──


def test_a_png_loses_text_exif_time_resolution_profile_and_trailer() -> None:
    raw = png_with_metadata()
    for secret in (CAMERA, GPS, NOTE, b"xmpmeta", TRAILER):
        assert secret in raw, "the sample must hold what the test looks for"
    out = strip_png(raw)
    assert out is not None
    assert png_kinds(out) == [b"IHDR", b"gAMA", b"sRGB", b"tRNS", b"IDAT", b"IEND"]
    assert_clean(out)
    assert b"eXIf" not in out and b"tEXt" not in out and b"iTXt" not in out
    assert _image_dimensions(out) == (8, 8)
    # The pixels are the original's.
    assert (
        out[out.index(b"IDAT") - 4 :]
        == raw[raw.index(b"IDAT") - 4 : raw.index(b"IEND") - 4 + 12]
    )


def test_a_png_that_has_only_what_it_needs_is_returned_as_it_is() -> None:
    plain = strip_png(png_with_metadata())
    assert plain is not None
    assert strip_png(plain) == plain


@pytest.mark.parametrize("damage", ["checksum", "cut", "no_data", "not_first"])
def test_a_png_that_is_not_whole_is_refused_not_passed_on(damage: str) -> None:
    raw = png_with_metadata()
    if damage == "checksum":
        at = raw.index(b"IDAT") + 6
        raw = raw[:at] + bytes([raw[at] ^ 0xFF]) + raw[at + 1 :]
    elif damage == "cut":
        raw = raw[: raw.index(b"IEND")]
    elif damage == "no_data":
        raw = raw.replace(b"IDAT", b"idat")
    else:
        raw = raw[:8] + png_chunk(b"tEXt", b"a\x00b") + raw[8:]
    assert strip_png(raw) is None


def test_strip_knows_two_kinds_only() -> None:
    assert strip(png_with_metadata(), "png") is not None
    assert strip(photo(), "jpeg") is not None
    assert strip(b"GIF89a", "gif") is None


# ── Every route of normalize ends in a stripped picture ──


def test_a_small_photo_comes_out_of_normalize_without_its_metadata() -> None:
    picture = normalize(photo())
    data = decode_base64(picture.image.data) or b""
    assert picture.image.mime_type == "image/jpeg"
    assert picture.converted is True
    assert (picture.width, picture.height) == (8, 8)
    assert_clean(data)
    assert picture.image.size_bytes == len(data)


def test_a_small_png_comes_out_of_normalize_without_its_metadata() -> None:
    picture = normalize(png_with_metadata())
    data = decode_base64(picture.image.data) or b""
    assert picture.image.mime_type == "image/png"
    assert picture.converted is True
    assert_clean(data)
    assert png_kinds(data) == [b"IHDR", b"gAMA", b"sRGB", b"tRNS", b"IDAT", b"IEND"]


def test_a_picture_without_metadata_is_not_changed() -> None:
    bare = strip_png(png_with_metadata())
    assert bare is not None
    picture = normalize(bare)
    assert picture.converted is False
    assert decode_base64(picture.image.data) == bare


def test_a_conversion_is_stripped_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever a conversion tool writes, the picture that leaves has no metadata."""
    gif = gif_picture(8, 8, [(8, 8, 0)])
    outputs = {"png": png_with_metadata(), "jpeg": photo()}
    seen = []

    def convert(sips: str, raw: bytes, kind: str, target: str, scale: bool) -> bytes:
        seen.append((kind, target))
        return outputs[target]

    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    monkeypatch.setattr(picture_input, "_convert", convert)
    from_gif = normalize(gif)
    assert from_gif.image.mime_type == "image/png"
    assert_clean(decode_base64(from_gif.image.data) or b"")
    # A photo that is too large is scaled by the tool, which keeps its metadata: stripped after.
    scaled = normalize(wide(photo(), 2000, 100))
    assert scaled.image.mime_type == "image/jpeg"
    assert_clean(decode_base64(scaled.image.data) or b"")
    assert seen == [("gif", "png"), ("jpeg", "jpeg")]


def test_a_conversion_the_tool_cannot_make_whole_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    monkeypatch.setattr(
        picture_input, "_convert", lambda *args, **kwargs: b"\x89PNG\r\n\x1a\nnot whole"
    )
    with pytest.raises(PictureRefused) as refused:
        normalize(gif_picture(8, 8, [(8, 8, 0)]))
    assert refused.value.code == "picture_unreadable"


def test_without_sips_a_large_picture_is_refused_not_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(picture_input.shutil, "which", lambda name: None)
    for big in (wide(photo(), 2000, 100), png_with_metadata(LONG_SIDE + 1, 4)):
        with pytest.raises(PictureRefused) as refused:
            normalize(big)
        assert (refused.value.status, refused.value.code) == (413, "picture_scale")
        assert refused.value.params == {"px": LONG_SIDE}
    # A picture that fits needs no tool, and is stripped all the same.
    assert_clean(decode_base64(normalize(photo()).image.data) or b"")
    # A picture that needs converting is refused as before.
    with pytest.raises(PictureRefused) as refused:
        normalize(gif_picture(8, 8, [(8, 8, 0)]))
    assert refused.value.code == "picture_convert"


@needs_sips
def test_what_sips_writes_is_stripped_too(tmp_path: Path) -> None:
    """A real scale and a real conversion, with the metadata put in first."""
    big = tmp_path / "big.jpg"
    big.write_bytes(photo(trailing=False))
    subprocess.run(
        ["sips", "-z", "1600", "1500", str(big), "--out", str(big)],
        capture_output=True,
        check=True,
    )
    scaled = normalize(big.read_bytes())
    assert (scaled.width, scaled.height) == (round(1500 * LONG_SIDE / 1600), LONG_SIDE)
    assert_clean(decode_base64(scaled.image.data) or b"")
    source = tmp_path / "a.png"
    source.write_bytes(png_with_metadata(40, 30))
    gif = tmp_path / "a.gif"
    subprocess.run(
        ["sips", "-s", "format", "gif", str(source), "--out", str(gif)],
        capture_output=True,
        check=True,
    )
    frame = normalize(gif.read_bytes())
    kinds = png_kinds(decode_base64(frame.image.data) or b"")
    assert set(kinds) <= {
        b"IHDR",
        b"PLTE",
        b"IDAT",
        b"IEND",
        b"tRNS",
        b"gAMA",
        b"cHRM",
        b"sRGB",
        b"cICP",
    }


# ── A PNG whose chunks are not whole or not in their place is refused ──


def png_made(*chunks: bytes, colour_type: int = 2, bit_depth: int = 8) -> bytes:
    """A PNG of the framed *chunks* under an IHDR of the given colour type and depth."""
    header = struct.pack(">IIBBBBB", 8, 8, bit_depth, colour_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + b"".join(chunks)


DATA = png_chunk(b"IDAT", zlib.compress(b"\x00" * 200))
END = png_chunk(b"IEND", b"")
WHOLE = png_made(DATA, END)


BAD_PNGS = {
    "IEND with content": png_made(DATA, png_chunk(b"IEND", b"x")),
    "IHDR of the wrong length": b"\x89PNG\r\n\x1a\n"
    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0) + b"\0")
    + DATA
    + END,
    "IHDR twice": png_made(
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)), DATA, END
    ),
    "IHDR with a size of zero": b"\x89PNG\r\n\x1a\n"
    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 0, 8, 8, 2, 0, 0, 0))
    + DATA
    + END,
    "IHDR with a depth the colour type does not take": png_made(DATA, END, bit_depth=4),
    "IHDR with an unknown method": b"\x89PNG\r\n\x1a\n"
    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 1, 0, 0))
    + DATA
    + END,
    "gAMA too long": png_made(png_chunk(b"gAMA", b"\0" * 5000), DATA, END),
    "gAMA too short": png_made(png_chunk(b"gAMA", b"\0\0"), DATA, END),
    "cHRM of the wrong length": png_made(png_chunk(b"cHRM", b"\0" * 33), DATA, END),
    "sRGB of the wrong length": png_made(png_chunk(b"sRGB", b"\0\0"), DATA, END),
    "cICP of the wrong length": png_made(png_chunk(b"cICP", b"\0" * 5), DATA, END),
    "tRNS too long for the colour type": png_made(
        png_chunk(b"tRNS", b"\0" * 300), DATA, END
    ),
    "tRNS on a colour type that has alpha": png_made(
        png_chunk(b"tRNS", b"\0\0"), DATA, END, colour_type=6
    ),
    "a numeric chunk twice": png_made(
        png_chunk(b"gAMA", b"\0\0\xb1\x8f"),
        png_chunk(b"gAMA", b"\0\0\xb1\x8f"),
        DATA,
        END,
    ),
    "gAMA after the data": png_made(DATA, png_chunk(b"gAMA", b"\0\0\xb1\x8f"), END),
    "tRNS after the data": png_made(DATA, png_chunk(b"tRNS", b"\0" * 6), END),
    "gAMA after PLTE": png_made(
        png_chunk(b"PLTE", b"\0" * 6), png_chunk(b"gAMA", b"\0\0\xb1\x8f"), DATA, END
    ),
    "PLTE after the data": png_made(DATA, png_chunk(b"PLTE", b"\0" * 6), END),
    "PLTE twice": png_made(
        png_chunk(b"PLTE", b"\0" * 6), png_chunk(b"PLTE", b"\0" * 6), DATA, END
    ),
    "PLTE not a multiple of three": png_made(png_chunk(b"PLTE", b"\0" * 7), DATA, END),
    "PLTE too big": png_made(png_chunk(b"PLTE", b"\0" * 771), DATA, END),
    "PLTE in a greyscale picture": png_made(
        png_chunk(b"PLTE", b"\0" * 6), DATA, END, colour_type=0
    ),
    "a palette picture with no PLTE": png_made(DATA, END, colour_type=3),
    "tRNS longer than the palette": png_made(
        png_chunk(b"PLTE", b"\0" * 6),
        png_chunk(b"tRNS", b"\0" * 3),
        DATA,
        END,
        colour_type=3,
    ),
    "the data split by another chunk": png_made(
        png_chunk(b"IDAT", zlib.compress(b"\x00" * 100)),
        png_chunk(b"tEXt", b"Comment\0split"),
        png_chunk(b"IDAT", zlib.compress(b"\x00" * 100)),
        END,
    ),
    "a critical chunk nobody knows": png_made(png_chunk(b"XYZW", b"abc"), DATA, END),
}


def test_a_whole_png_in_order_is_kept() -> None:
    assert strip_png(WHOLE) == WHOLE


def test_every_chunk_of_a_palette_greyscale_and_numeric_png_is_kept_when_it_is_whole() -> (
    None
):
    palette = png_made(
        png_chunk(b"gAMA", b"\0\0\xb1\x8f"),
        png_chunk(b"sRGB", b"\x00"),
        png_chunk(b"PLTE", b"\0" * 6),
        png_chunk(b"tRNS", b"\xff\x80"),
        DATA,
        END,
        colour_type=3,
    )
    assert strip_png(palette) == palette
    grey = png_made(png_chunk(b"tRNS", b"\0\x10"), DATA, END, colour_type=0)
    assert strip_png(grey) == grey
    numeric = png_made(
        png_chunk(b"cHRM", b"\0" * 32), png_chunk(b"cICP", b"\1\x0d\0\1"), DATA, END
    )
    assert strip_png(numeric) == numeric


@pytest.mark.parametrize("raw", BAD_PNGS.values(), ids=BAD_PNGS.keys())
def test_a_png_whose_chunks_are_not_whole_or_not_in_their_place_is_refused(
    raw: bytes,
) -> None:
    assert strip_png(raw) is None
    with pytest.raises(PictureRefused) as refused:
        picture_input.normalize(raw)
    assert refused.value.code == "picture_unreadable"
