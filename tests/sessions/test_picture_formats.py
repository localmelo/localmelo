"""GIF, WebP and HEIC pictures: the pixel limit, the size limit and the metadata.

Each format has to show three things, with files that hold what the test looks
for:

* the pixel limit is checked from the file's own headers before any tool opens
  the file (a tool that is started fails the test);
* the picture that comes out fits in :data:`LONG_SIDE`, also when the tool does
  not scale it;
* nothing of the metadata that went in comes out.

The GIF and WebP files are built byte by byte (``picture_samples``): a GIF of
several frames with a comment and an XMP block, and a WebP with EXIF and XMP
chunks, one colour so they stay a few hundred bytes. ``camera.heic`` was made on
macOS 27.2 (build 26B123) from a 1600 x 40 gradient JPEG that holds the EXIF
block of ``picture_samples.exif_tiff`` (a camera name, the orientation and GPS
data)::

    sips -s format heic gradient_with_exif.jpg --out camera.heic

The tests that run ``sips`` for real are skipped where it is missing; the others
put a converter that behaves in the way under test in its place.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from localmelo.melo.schema import decode_base64
from localmelo.support.sessions import picture_input
from localmelo.support.sessions.picture_extent import claimed_sizes
from localmelo.support.sessions.picture_input import (
    LONG_SIDE,
    MAX_PIXELS,
    Picture,
    PictureRefused,
    normalize,
    sniff,
)

from .picture_samples import (
    CAMERA,
    GIF_COMMENT,
    GIF_XMP,
    GPS,
    ORANGE,
    assert_clean,
    box,
    gif_extras,
    gif_picture,
    heic_headers,
    markers,
    photo,
    png_kinds,
    png_with_metadata,
    riff,
    riff_chunk,
    vp8_header,
    vp8l,
    webp_animation,
    webp_picture,
)

needs_sips = pytest.mark.skipif(shutil.which("sips") is None, reason="macOS sips")

CAMERA_HEIC = (
    Path(__file__).parent.parent / "fixtures" / "pictures" / "camera.heic"
).read_bytes()
# The chunks a stripped PNG may hold, and the JPEG segments that carry metadata.
PNG_KEPT = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB"}
JPEG_METADATA = {0xE0, *range(0xE2, 0xEE), 0xEF, 0xFE}

GIF_3 = gif_picture(
    1600, 40, [(1600, 40, 0), (1600, 40, 1), (1600, 40, 2)], gif_extras()
)
GIF_TALL = gif_picture(40, 1600, [(40, 1600, 0), (40, 1600, 1)], gif_extras())
WEBP_META = webp_picture(1600, 40, metadata=True)
WEBP_ANIMATED = webp_animation(1600, 40, [(1600, 40), (1600, 40)])


def data_of(picture: Picture) -> bytes:
    return decode_base64(picture.image.data) or b""


# ── The sizes a header claims ──


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        pytest.param(GIF_3, "gif", [(1600, 40)] * 4, id="gif screen and frames"),
        pytest.param(WEBP_META, "webp", [(1600, 40)] * 2, id="webp canvas and image"),
        pytest.param(
            WEBP_ANIMATED,
            "webp",
            [(1600, 40)] * 5,
            id="webp canvas, frames, bitstreams",
        ),
        pytest.param(
            riff([riff_chunk(b"VP8 ", vp8_header(1600, 40))]),
            "webp",
            [(1600, 40)],
            id="webp lossy",
        ),
        pytest.param(
            heic_headers((320, 240), (4000, 3000)),
            "heic",
            [(320, 240), (4000, 3000)],
            id="heic images",
        ),
    ],
)
def test_the_sizes_in_a_header_are_read_without_decoding(
    raw: bytes, kind: str, expected: list[tuple[int, int]]
) -> None:
    assert sniff(raw) == kind
    assert claimed_sizes(raw, kind) == expected


def test_the_sizes_in_a_real_heic_are_read_from_its_boxes() -> None:
    sizes = claimed_sizes(CAMERA_HEIC, "heic")
    assert sizes is not None and (1600, 40) in sizes


@needs_sips
def test_the_size_read_from_a_header_is_the_size_the_system_reports(
    tmp_path: Path,
) -> None:
    for name, raw, kind in (
        ("a.gif", GIF_3, "gif"),
        ("a.webp", WEBP_META, "webp"),
        ("a.heic", CAMERA_HEIC, "heic"),
    ):
        path = tmp_path / name
        path.write_bytes(raw)
        out = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        reported = {
            line.split(":")[0].strip(): int(line.split(":")[1])
            for line in out.splitlines()[1:]
        }
        sizes = claimed_sizes(raw, kind) or []
        assert (reported["pixelWidth"], reported["pixelHeight"]) in sizes, name


# ── The limit on pixels is checked before any tool opens the file ──


@pytest.fixture
def no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sips`` looks present, and starting it (or any process) fails the test."""

    def started(*args: object, **kwargs: object) -> object:
        raise AssertionError("a tool was started before the pixel limit was checked")

    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    monkeypatch.setattr(picture_input, "_convert", started)
    monkeypatch.setattr(picture_input.subprocess, "run", started)


BOMBS = {
    "gif: logical screen": gif_picture(20000, 20000, [(8, 8, 0)]),
    "gif: frame larger than the screen": gif_picture(8, 8, [(20000, 20000, 0)]),
    "gif: only the second frame": gif_picture(8, 8, [(8, 8, 0), (20000, 20000, 1)]),
    "webp: lossless header": riff([riff_chunk(b"VP8L", vp8l(16384, 16384, ORANGE))]),
    "webp: lossy header": riff([riff_chunk(b"VP8 ", vp8_header(16383, 16383))]),
    "webp: canvas": webp_picture(8, 8, canvas=(30000, 30000)),
    "webp: animation frame": webp_animation(8, 8, [(8, 8), (16384, 16384)]),
    "heic: image": heic_headers((20000, 20000)),
    "heic: second image": heic_headers((320, 240), (20000, 20000)),
}


@pytest.mark.parametrize("raw", BOMBS.values(), ids=BOMBS.keys())
def test_too_many_pixels_are_refused_before_any_tool_opens_the_file(
    raw: bytes, no_tools: None
) -> None:
    with pytest.raises(PictureRefused) as refused:
        normalize(raw)
    assert (refused.value.status, refused.value.code) == (
        413,
        "picture_too_many_pixels",
    )
    assert refused.value.params == {"px": MAX_PIXELS // 1_000_000}


def test_the_limit_is_on_the_pixels_of_one_picture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly the limit goes on to the converter; one pixel more does not."""
    seen: list[bytes] = []

    def convert(sips: str, raw: bytes, kind: str, target: str, scale: bool) -> bytes:
        seen.append(raw)
        return png_with_metadata(64, 4)

    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    monkeypatch.setattr(picture_input, "_convert", convert)
    at_limit = gif_picture(12000, MAX_PIXELS // 12000, [(8, 8, 0)])
    assert normalize(at_limit).width == 64
    assert seen == [at_limit]
    with pytest.raises(PictureRefused) as refused:
        normalize(gif_picture(12000, MAX_PIXELS // 12000 + 1, [(8, 8, 0)]))
    assert refused.value.code == "picture_too_many_pixels"
    assert len(seen) == 1


def with_second_image_box(first: bytes, second: bytes) -> bytes:
    """A HEIC whose ``meta`` box holds two ``iprp`` boxes: *first* and *second* (each an ``ipco`` body)."""
    props = box(b"iprp", box(b"ipco", first)) + box(b"iprp", box(b"ipco", second))
    return box(b"ftyp", b"heic\0\0\0\0heicmif1") + box(b"meta", b"\0\0\0\0" + props)


SMALL_ISPE = box(b"ispe", struct.pack(">III", 0, 8, 8))
GOOD_WEBP = webp_picture(8, 8, metadata=True)
GOOD_HEIC = heic_headers((320, 240))
# A box that claims more than the file holds.
BROKEN_BOX = struct.pack(">I", 999) + b"ispe" + b"\0" * 12

MALFORMED = {
    "gif: no frame": gif_picture(8, 8, []),
    "gif: cut inside a frame": gif_picture(8, 8, [(8, 8, 0)])[:-30],
    "gif: a block that is not one": gif_picture(8, 8, [(8, 8, 0)]).replace(
        b"\x21\xf9", b"\x77\xf9", 1
    ),
    "gif: no trailer": gif_picture(8, 8, [(8, 8, 0)])[:-1],
    "webp: a chunk past the end": riff([b"VP8L" + (999).to_bytes(4, "little") + b"x"]),
    "webp: no signature": riff([riff_chunk(b"VP8L", b"\x00" * 10)]),
    "webp: no start code": riff([riff_chunk(b"VP8 ", b"\x10\x02\x00" + b"\x00" * 8)]),
    "webp: no size": riff([riff_chunk(b"EXIF", b"MM")]),
    "webp: the length of the file is cut short": GOOD_WEBP[:-30],
    "webp: a length longer than the file": GOOD_WEBP[:4]
    + struct.pack("<I", len(GOOD_WEBP) + 500)
    + GOOD_WEBP[8:],
    "webp: a partial chunk header at the end": riff(
        [riff_chunk(b"VP8L", vp8l(8, 8, ORANGE)), b"EXI"]
    ),
    "webp: a missing pad byte": riff(
        [
            riff_chunk(b"VP8L", vp8l(8, 8, ORANGE)),
            b"EXIF" + struct.pack("<I", 3) + b"abc",
        ]
    ),
    "webp: a pad byte missing after the last chunk": (
        lambda whole: whole[:4] + struct.pack("<I", len(whole) - 8 - 1) + whole[8:-1]
    )(riff([riff_chunk(b"VP8L", vp8l(8, 8, ORANGE)), riff_chunk(b"EXIF", b"abc")])),
    "heic: a box past the end": box(b"ftyp", b"heic\0\0\0\0")[:-2] + b"\0\0",
    "heic: no sizes": box(b"ftyp", b"heic\0\0\0\0heic") + box(b"meta", b"\0" * 4),
    "heic: a partial box header at the end": GOOD_HEIC + b"\0\0\0",
    "heic: a partial box header inside ipco": box(b"ftyp", b"heic\0\0\0\0heicmif1")
    + box(
        b"meta",
        b"\0\0\0\0" + box(b"iprp", box(b"ipco", SMALL_ISPE + b"\0\0\0")),
    ),
    "heic: a size box that is too short": box(b"ftyp", b"heic\0\0\0\0heicmif1")
    + box(
        b"meta",
        b"\0\0\0\0" + box(b"iprp", box(b"ipco", box(b"ispe", b"\0" * 6))),
    ),
    "heic: a short size box after a good one in the same ipco": box(
        b"ftyp", b"heic\0\0\0\0heicmif1"
    )
    + box(
        b"meta",
        b"\0\0\0\0" + box(b"iprp", box(b"ipco", SMALL_ISPE + box(b"ispe", b"\0" * 6))),
    ),
    "heic: a size box of a version nobody wrote": box(b"ftyp", b"heic\0\0\0\0heicmif1")
    + box(
        b"meta",
        b"\0\0\0\0"
        + box(
            b"iprp",
            box(b"ipco", box(b"ispe", struct.pack(">III", 1 << 24, 8, 8))),
        ),
    ),
    "heic: a meta box of a version nobody wrote": box(b"ftyp", b"heic\0\0\0\0heicmif1")
    + box(
        b"meta",
        b"\x01\0\0\0" + box(b"iprp", box(b"ipco", SMALL_ISPE)),
    ),
    "heic: a broken box after a good one in the same ipco": box(
        b"ftyp", b"heic\0\0\0\0heicmif1"
    )
    + box(
        b"meta",
        b"\0\0\0\0" + box(b"iprp", box(b"ipco", SMALL_ISPE + BROKEN_BOX)),
    ),
    "heic: a second ipco that does not parse, after a good one": with_second_image_box(
        SMALL_ISPE, BROKEN_BOX
    ),
    "heic: a second iprp whose children do not parse, after a good one": (
        box(b"ftyp", b"heic\0\0\0\0heicmif1")
        + box(
            b"meta",
            b"\0\0\0\0"
            + box(b"iprp", box(b"ipco", SMALL_ISPE))
            + box(b"iprp", BROKEN_BOX),
        )
    ),
    "heic: a broken box in meta after a good iprp": box(b"ftyp", b"heic\0\0\0\0heic")
    + box(
        b"meta",
        b"\0\0\0\0" + box(b"iprp", box(b"ipco", SMALL_ISPE)) + BROKEN_BOX,
    ),
    "heic: meta too short for its version": box(b"ftyp", b"heic\0\0\0\0heic")
    + box(b"meta", b"\0\0"),
}


@pytest.mark.parametrize("raw", MALFORMED.values(), ids=MALFORMED.keys())
def test_a_header_that_does_not_hold_together_is_refused_before_any_tool_opens_it(
    raw: bytes, no_tools: None
) -> None:
    with pytest.raises(PictureRefused) as refused:
        normalize(raw)
    assert (refused.value.status, refused.value.code) == (415, "picture_unreadable")


# ── The size that comes out ──


def conversion(
    monkeypatch: pytest.MonkeyPatch, made: bytes
) -> list[tuple[str, str, bool]]:
    """Put a converter that returns *made* in place of ``sips``; the list records its calls."""
    calls: list[tuple[str, str, bool]] = []

    def convert(sips: str, raw: bytes, kind: str, target: str, scale: bool) -> bytes:
        calls.append((kind, target, scale))
        return made

    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    monkeypatch.setattr(picture_input, "_convert", convert)
    return calls


@pytest.mark.parametrize(
    ("raw", "calls"),
    [
        (GIF_3, [("gif", "png", True)]),
        (gif_picture(64, 8, [(64, 8, 0)]), [("gif", "png", False)]),
        (WEBP_META, [("webp", "png", True)]),
        (webp_picture(64, 8), [("webp", "png", False)]),
        (CAMERA_HEIC, [("heic", "jpeg", True)]),
    ],
    ids=["gif large", "gif small", "webp large", "webp small", "heic large"],
)
def test_a_tool_is_told_to_scale_only_what_is_over_the_limit(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, calls: list[tuple[str, str, bool]]
) -> None:
    made = photo(trailing=False) if calls[0][1] == "jpeg" else png_with_metadata()
    seen = conversion(monkeypatch, made)
    normalize(raw)
    assert seen == calls


@pytest.mark.parametrize(
    "made",
    [png_with_metadata(LONG_SIDE + 1, 4), png_with_metadata(4, LONG_SIDE + 1)],
    ids=["too wide", "too tall"],
)
def test_a_tool_that_does_not_scale_is_not_believed(
    monkeypatch: pytest.MonkeyPatch, made: bytes
) -> None:
    conversion(monkeypatch, made)
    with pytest.raises(PictureRefused) as refused:
        normalize(GIF_3)
    assert (refused.value.status, refused.value.code) == (413, "picture_scale")
    assert refused.value.params == {"px": LONG_SIDE}


def test_a_picture_that_lies_about_its_size_is_measured_again_after_the_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GIF that says 8 x 8 and decodes to something large is caught by the second look."""
    conversion(monkeypatch, png_with_metadata(LONG_SIDE + 100, 4))
    with pytest.raises(PictureRefused) as refused:
        normalize(gif_picture(8, 8, [(8, 8, 0)]))
    assert refused.value.code == "picture_scale"


# ── The metadata that comes out (a converter that passes it on) ──


def png_kinds_of(picture: Picture) -> set[bytes]:
    return set(png_kinds(data_of(picture)))


@pytest.mark.parametrize(
    ("raw", "secrets"),
    [
        pytest.param(GIF_3, [GIF_COMMENT, GIF_XMP], id="gif"),
        pytest.param(WEBP_META, [CAMERA, GPS, b"SecretXmp-WebP"], id="webp"),
    ],
)
def test_a_png_made_from_a_gif_or_webp_has_no_metadata_even_if_the_tool_kept_it(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, secrets: list[bytes]
) -> None:
    for secret in secrets:
        assert secret in raw, "the file must hold what the test looks for"
    conversion(monkeypatch, png_with_metadata(128, 4))
    picture = normalize(raw)
    out = data_of(picture)
    assert_clean(out)
    for secret in secrets:
        assert secret not in out
    assert png_kinds_of(picture) <= PNG_KEPT
    assert picture.converted is True and picture.image.mime_type == "image/png"


def test_a_jpeg_made_from_a_heic_has_no_metadata_even_if_the_tool_kept_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert CAMERA in CAMERA_HEIC
    conversion(monkeypatch, photo())
    picture = normalize(CAMERA_HEIC)
    out = data_of(picture)
    assert_clean(out)
    assert not set(markers(out)) & JPEG_METADATA
    assert picture.converted is True and picture.image.mime_type == "image/jpeg"


# ── The same with the real tool ──


def first_pixel(png: bytes, folder: Path) -> tuple[int, int, int]:
    """The colour of the first pixel of PNG *png*, read through a BMP the system writes."""
    source = folder / "pixel.png"
    bmp = folder / "pixel.bmp"
    source.write_bytes(png)
    subprocess.run(
        ["sips", "-s", "format", "bmp", str(source), "--out", str(bmp)],
        capture_output=True,
        check=True,
    )
    raw = bmp.read_bytes()
    at = int.from_bytes(raw[10:14], "little")
    blue, green, red = raw[at : at + 3]
    return red, green, blue


REAL = [
    pytest.param(
        GIF_3, "image/png", (LONG_SIDE, 32), [GIF_COMMENT, GIF_XMP], id="gif 3 frames"
    ),
    pytest.param(
        GIF_TALL, "image/png", (32, LONG_SIDE), [GIF_COMMENT, GIF_XMP], id="gif tall"
    ),
    pytest.param(
        WEBP_META,
        "image/png",
        (LONG_SIDE, 32),
        [CAMERA, GPS, b"SecretXmp-WebP"],
        id="webp",
    ),
    pytest.param(WEBP_ANIMATED, "image/png", (LONG_SIDE, 32), [], id="webp animated"),
    pytest.param(CAMERA_HEIC, "image/jpeg", (LONG_SIDE, 32), [CAMERA], id="heic"),
]


@needs_sips
@pytest.mark.parametrize(("raw", "mime", "size", "secrets"), REAL)
def test_the_real_tool_scales_to_the_long_side_and_nothing_of_the_metadata_comes_out(
    raw: bytes, mime: str, size: tuple[int, int], secrets: list[bytes]
) -> None:
    for secret in secrets:
        assert secret in raw, "the file must hold what the test looks for"
    picture = normalize(raw)
    out = data_of(picture)
    assert picture.converted is True
    assert picture.image.mime_type == mime
    assert (picture.width, picture.height) == size
    assert max(picture.width, picture.height) <= LONG_SIDE
    assert_clean(out)
    for secret in secrets:
        assert secret not in out
    if mime == "image/png":
        assert set(png_kinds(out)) <= PNG_KEPT
    else:
        assert not set(markers(out)) & JPEG_METADATA


@needs_sips
def test_a_gif_gives_its_first_frame(tmp_path: Path) -> None:
    picture = normalize(GIF_3)
    red, green, blue = first_pixel(data_of(picture), tmp_path)
    assert red > 200 and green < 60 and blue < 60  # frame 1 is red; 2 green, 3 blue


@needs_sips
def test_a_webp_keeps_its_colour(tmp_path: Path) -> None:
    red, green, blue = first_pixel(data_of(normalize(WEBP_META)), tmp_path)
    assert red > 200 and 90 < green < 170 and blue < 60  # the orange of the sample


@needs_sips
@pytest.mark.parametrize(
    ("raw", "kind", "target"),
    [(GIF_3, "gif", "png"), (WEBP_META, "webp", "png"), (CAMERA_HEIC, "heic", "jpeg")],
)
def test_what_the_system_writes_on_its_own_carries_metadata_that_is_then_removed(
    raw: bytes, kind: str, target: str
) -> None:
    """Why the cleaning cannot be left to the tool: its output holds EXIF of its own."""
    sips = shutil.which("sips") or ""
    made = picture_input._convert(sips, raw, kind, target, scale=True)
    assert made
    if target == "png":
        assert b"eXIf" in png_kinds(made) or b"iTXt" in png_kinds(made)
    else:
        assert set(markers(made)) & JPEG_METADATA or b"Exif" in made
    cleaned = picture_input.strip(made, target)
    assert cleaned is not None and len(cleaned) < len(made)
    assert_clean(cleaned)


# ── A tool that fails ──


def test_a_converter_that_is_slow_or_broken_leaves_nothing_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    outcomes: list[Exception | subprocess.CompletedProcess[bytes]] = [
        subprocess.TimeoutExpired("sips", 20),
        OSError("no such tool"),
        subprocess.CompletedProcess([], 1, b"", b"broken"),
        subprocess.CompletedProcess([], 0, b"", b""),  # says it worked, wrote nothing
    ]

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(picture_input.subprocess, "run", run)
    for _ in range(4):
        with pytest.raises(PictureRefused) as refused:
            normalize(GIF_3)
        assert (refused.value.status, refused.value.code) == (415, "picture_unreadable")
    assert not outcomes
    assert list(scratch.iterdir()) == []


def test_the_tool_gets_fixed_file_names_and_no_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing the person typed (a file name) reaches the command line: only the bytes do."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        Path(argv[-1]).write_bytes(png_with_metadata(64, 4))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(picture_input.subprocess, "run", run)
    normalize(GIF_3)
    (argv, kwargs), *rest = calls
    assert not rest
    folder = os.path.dirname(argv[-1])
    assert folder.startswith(str(scratch / "melo-picture-"))
    assert argv == [
        "/usr/bin/sips",
        "-s",
        "format",
        "png",
        "-Z",
        str(LONG_SIDE),
        os.path.join(folder, "in.gif"),
        "--out",
        os.path.join(folder, "out.png"),
    ]
    assert not kwargs.get("shell")
    assert kwargs["timeout"] == picture_input.SIPS_TIMEOUT_S
    assert list(scratch.iterdir()) == []
