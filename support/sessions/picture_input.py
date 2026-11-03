"""Pictures the person adds to a message: sniffed, converted and scaled once.

The composer sends the bytes of a picture it was given (a drop, a paste or the
picker). This module decides what they are from the bytes themselves, not from
the name or the type the browser reported, and returns the picture the model
gets: a PNG or JPEG whose long side is at most :data:`LONG_SIDE` pixels, the
same limit the screenshots tools return use (``ToolImageStore``).

A PNG or JPEG that already fits keeps its pixels. A larger PNG or JPEG is scaled
and keeps its format. GIF (its first frame), WebP and HEIC are converted with
macOS ``sips``: HEIC to JPEG, GIF and WebP to PNG, which keeps transparency.
Without ``sips`` only the PNG and JPEG that fit can be taken; anything that needs
a conversion or a smaller size is refused with a plain reason instead of being
sent as it is.

The pixel limit is checked first, from the sizes the file's own headers claim
(:mod:`~localmelo.support.sessions.picture_extent`: every frame of a GIF, every
property of a HEIC), before ``sips`` or any decoder opens the file. What comes out
of a conversion is measured again and must fit in :data:`LONG_SIDE`.

Whatever the route, the picture that comes out has no metadata: no EXIF (the
camera, the time, the GPS position), XMP, comments or text chunks, and nothing
after the end of the image (see :mod:`~localmelo.support.sessions.picture_clean`).
Only the orientation and the colour that decoding needs are kept.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from localmelo.melo.agent._request_accounting import _image_dimensions
from localmelo.melo.checker.validators import MAX_IMAGE_BYTES
from localmelo.melo.schema import ImagePart
from localmelo.support.sessions.picture_clean import strip
from localmelo.support.sessions.picture_extent import claimed_sizes

# Pictures one message may carry. A picture costs a vision model roughly a
# thousand tokens at this size, and a local model's window is often 8k to 32k
# tokens; four fill a row of thumbnails and still leave room to talk. The
# checker's cap of eight is for one tool result, which nobody reads one by one.
MAX_PICTURES: Final = 4
# The long side of a picture sent to the model, as for the screenshots.
LONG_SIDE: Final = 1280
# The bytes of one upload; also the checker's limit for one image.
MAX_UPLOAD_BYTES: Final = MAX_IMAGE_BYTES
# Pixels a picture may have before it is decoded: a small file can be huge.
MAX_PIXELS: Final = 120_000_000
JPEG_QUALITY: Final = 85
SIPS_TIMEOUT_S: Final = 20.0

_HEIC_BRANDS: Final = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"heif"}
)


class PictureRefused(ValueError):  # noqa: N818
    """A picture that cannot be used: an HTTP status, a code, and the numbers its sentence needs.

    The console words each code in its catalog (``errors.picture…``).
    """

    def __init__(self, status: int, code: str, **params: object) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.params = params


@dataclass(frozen=True)
class Picture:
    """A picture ready for the model, and what the page needs to show it."""

    image: ImagePart
    width: int
    height: int
    # Whether the bytes differ from the ones that came in.
    converted: bool


def sniff(raw: bytes) -> str | None:
    """``png``, ``jpeg``, ``gif``, ``webp`` or ``heic`` by the first bytes, else None."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[4:8] == b"ftyp" and raw[8:12] in _HEIC_BRANDS:
        return "heic"
    return None


def _convert(
    sips: str, raw: bytes, kind: str, target: str, scale: bool
) -> bytes | None:
    """*raw* as a *target* (``png`` or ``jpeg``) file, scaled to :data:`LONG_SIDE` if asked."""
    with tempfile.TemporaryDirectory(prefix="melo-picture-") as folder:
        source = os.path.join(folder, f"in.{kind}")
        out = os.path.join(folder, f"out.{target}")
        Path(source).write_bytes(raw)
        argv = [sips, "-s", "format", target]
        if target == "jpeg":
            argv += ["-s", "formatOptions", str(JPEG_QUALITY)]
        if scale:
            argv += ["-Z", str(LONG_SIDE)]
        try:
            done = subprocess.run(
                [*argv, source, "--out", out],
                capture_output=True,
                timeout=SIPS_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0 or not os.path.isfile(out):
            return None
        return Path(out).read_bytes()


def normalize(raw: bytes) -> Picture:
    """The picture to send for the uploaded *raw* bytes, or :class:`PictureRefused`.

    Blocking: it runs ``sips`` for a picture that has to change, so a caller on
    the event loop runs it in a thread.
    """
    if len(raw) > MAX_UPLOAD_BYTES:
        raise PictureRefused(413, "picture_too_large", mb=MAX_UPLOAD_BYTES >> 20)
    kind = sniff(raw)
    if kind is None:
        raise PictureRefused(415, "picture_unsupported")
    sizes = claimed_sizes(raw, kind)
    if sizes is None:
        raise PictureRefused(415, "picture_unreadable")
    # Every picture the file mentions (each frame of a GIF, each property of a HEIC)
    # is held to the limit, not only the first.
    if any(w <= 0 or h <= 0 or w * h > MAX_PIXELS for w, h in sizes):
        raise PictureRefused(
            413,
            "picture_too_many_pixels",
            px=MAX_PIXELS // 1_000_000,
        )
    # The tool only enlarges a picture it is told to fit, so it is asked to scale only
    # when some size in the headers is over the limit; a grid of tiles can name a tile
    # larger than the picture, which is then scaled up to the limit at worst.
    fits = all(max(size) <= LONG_SIDE for size in sizes)
    if kind in ("png", "jpeg") and fits:
        return _clean(raw, kind, raw)
    sips = shutil.which("sips")
    if sips is None:
        if kind in ("png", "jpeg"):
            # Too large and no tool to scale it: not sent as it is.
            raise PictureRefused(413, "picture_scale", px=LONG_SIDE)
        raise PictureRefused(415, "picture_convert", kind=kind.upper())
    target = {"heic": "jpeg", "jpeg": "jpeg"}.get(kind, "png")
    converted = _convert(sips, raw, kind, target, scale=not fits)
    if not converted:
        raise PictureRefused(415, "picture_unreadable")
    return _clean(converted, target, raw)


def _clean(made: bytes, kind: str, original: bytes) -> Picture:
    """The picture for *made* (PNG or JPEG) with its metadata stripped, as ``Picture.converted`` against *original*."""
    cleaned = strip(made, kind)
    if cleaned is None:
        raise PictureRefused(415, "picture_unreadable")
    if len(cleaned) > MAX_IMAGE_BYTES:
        raise PictureRefused(413, "picture_too_large", mb=MAX_IMAGE_BYTES >> 20)
    size = _image_dimensions(cleaned)
    if size is None:
        raise PictureRefused(415, "picture_unreadable")
    if max(size) > LONG_SIDE:
        # A converter that did not scale: the picture is not sent as it is.
        raise PictureRefused(413, "picture_scale", px=LONG_SIDE)
    return Picture(
        ImagePart.from_bytes(cleaned, f"image/{kind}"), *size, cleaned != original
    )
