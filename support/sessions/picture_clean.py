"""Metadata stripped from every picture that goes to a model.

A photo carries more than pixels: EXIF (the camera, the time, often the GPS
position), XMP, IPTC, comments and a thumbnail, and a PNG can hold text chunks and
an EXIF block. A hosted model should get the pixels and nothing the person did not
mean to share, so every picture is rewritten here before it is kept or sent, also
the ones that needed no conversion.

Both functions walk the file's structure and copy only what a decoder needs; they
never decode pixels, and a file whose structure does not hold together comes back
as ``None`` (the caller refuses it) instead of being passed on.

* JPEG: the image segments (tables, frame, scans) are kept. Every APP segment and
  every comment is dropped, and so is anything after the end of the image (a
  trailing thumbnail or a video). Two things a decoder needs are written again
  from scratch with fixed contents: the orientation (a minimal EXIF block that
  holds that one number) and the Adobe colour transform byte.
* PNG: the critical chunks (``IHDR``, ``PLTE``, ``IDAT``, ``IEND``) and the small
  numeric ones that give the colour (``tRNS``, ``gAMA``, ``cHRM``, ``sRGB``,
  ``cICP``). Text, EXIF, time, resolution, background, embedded profile and
  animation chunks are dropped, and so is anything after ``IEND``.
"""

from __future__ import annotations

import zlib
from typing import Final, Literal

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_PNG_KEEP: Final = frozenset(
    {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"cICP"}
)
_PNG_MAX_CHUNK: Final = 0x7FFFFFFF
# The size of each fixed-size colour chunk, and the bit depths each colour type takes.
_PNG_SIZES: Final = {b"gAMA": 4, b"cHRM": 32, b"sRGB": 1, b"cICP": 4}
_PNG_DEPTHS: Final = {
    0: (1, 2, 4, 8, 16),
    2: (8, 16),
    3: (1, 2, 4, 8),
    4: (8, 16),
    6: (8, 16),
}

# JPEG markers.
_SOI, _EOI, _SOS = 0xD8, 0xD9, 0xDA
_APP1, _APP14 = 0xE1, 0xEE
# The frame headers (SOFn) and the tables a decoder needs: DQT, DHT, DAC, DRI and DNL.
_SOF: Final = frozenset(m for m in range(0xC0, 0xD0) if m not in (0xC4, 0xC8, 0xCC))
_JPEG_KEEP: Final = _SOF | {0xDB, 0xC4, 0xCC, 0xDD, 0xDC}
_RESTARTS: Final = frozenset(range(0xD0, 0xD8))


def strip(raw: bytes, kind: str) -> bytes | None:
    """*raw* (``png`` or ``jpeg``) without metadata, or None if its structure is broken."""
    if kind == "png":
        return strip_png(raw)
    if kind == "jpeg":
        return strip_jpeg(raw)
    return None


def strip_png(raw: bytes) -> bytes | None:
    """The critical and colour chunks of PNG *raw*; None if it is not whole and in order.

    A chunk is kept only when its checksum holds, its length is the one its kind has
    and it stands where the format puts it (header first, colour before the palette,
    the palette before the data, the data in one run, an empty end). Anything else
    is not a picture a decoder should get, so it is refused, not passed on.
    """
    if not raw.startswith(_PNG_SIGNATURE):
        return None
    out = bytearray(_PNG_SIGNATURE)
    pos = len(_PNG_SIGNATURE)
    order = _PngOrder()
    while pos + 12 <= len(raw):
        length = int.from_bytes(raw[pos : pos + 4], "big")
        kind = raw[pos + 4 : pos + 8]
        end = pos + 12 + length
        if length > _PNG_MAX_CHUNK or end > len(raw):
            return None
        body = raw[pos + 8 : pos + 8 + length]
        if kind in _PNG_KEEP:
            crc = int.from_bytes(raw[pos + 8 + length : end], "big")
            if zlib.crc32(kind + body) != crc:
                return None
        if not order.accepts(kind, body):
            return None
        if kind in _PNG_KEEP:
            out += raw[pos:end]
        pos = end
        if kind == b"IEND":
            return bytes(out)
    return None


class _PngOrder:
    """Where each chunk of a PNG may stand, and how long it may be, as the chunks come."""

    def __init__(self) -> None:
        self.colour = -1
        self.depth = 0
        self.seen: set[bytes] = set()
        self.palette = 0
        self.data = 0  # 0: no data yet, 1: inside the run of IDAT, 2: the run is over

    def accepts(self, kind: bytes, body: bytes) -> bool:
        if self.colour < 0:
            return kind == b"IHDR" and self._header(body)
        if self.data == 1 and kind != b"IDAT":
            self.data = 2
        if kind == b"IDAT":
            return self._data()
        if kind == b"IEND":
            return not body and self.data == 2
        if kind not in _PNG_KEEP:
            return bool(
                kind[0] & 0x20
            )  # an unknown critical chunk: not a picture to pass
        if kind in self.seen or self.data:
            return False
        self.seen.add(kind)
        if kind == b"PLTE":
            return self._palette(body)
        if kind == b"tRNS":
            return self._transparency(body)
        # gAMA, cHRM, sRGB, cICP: one fixed size, and ahead of the palette.
        return len(body) == _PNG_SIZES[kind] and b"PLTE" not in self.seen

    def _header(self, body: bytes) -> bool:
        if len(body) != 13:
            return False
        width = int.from_bytes(body[:4], "big")
        height = int.from_bytes(body[4:8], "big")
        depth, colour, method, filtering, interlace = body[8:13]
        if not (1 <= width <= _PNG_MAX_CHUNK and 1 <= height <= _PNG_MAX_CHUNK):
            return False
        if depth not in _PNG_DEPTHS.get(colour, ()) or (method, filtering) != (0, 0):
            return False
        if interlace > 1:
            return False
        self.colour, self.depth = colour, depth
        self.seen.add(b"IHDR")
        return True

    def _data(self) -> bool:
        if self.data == 2 or (self.colour == 3 and not self.palette):
            return False
        self.data = 1
        return True

    def _palette(self, body: bytes) -> bool:
        entries, extra = divmod(len(body), 3)
        if self.colour not in (2, 3, 6) or extra or not 1 <= entries <= 256:
            return False
        if self.colour == 3 and entries > 1 << self.depth:
            return False
        if b"tRNS" in self.seen:
            return False
        self.palette = entries
        return True

    def _transparency(self, body: bytes) -> bool:
        if self.colour == 3:
            return bool(self.palette) and 1 <= len(body) <= self.palette
        return len(body) == {0: 2, 2: 6}.get(self.colour, -1)


def _orientation(exif: bytes) -> int | None:
    """The orientation (1 to 8) in the body of an EXIF APP1 segment, or None."""
    if not exif.startswith(b"Exif\x00\x00"):
        return None
    tiff = exif[6:]
    orders: dict[bytes, Literal["little", "big"]] = {b"II": "little", b"MM": "big"}
    order = orders.get(tiff[:2])
    if order is None or len(tiff) < 8:
        return None
    if int.from_bytes(tiff[2:4], order) != 42:
        return None
    ifd = int.from_bytes(tiff[4:8], order)
    if ifd < 8 or ifd + 2 > len(tiff):
        return None
    count = int.from_bytes(tiff[ifd : ifd + 2], order)
    for index in range(min(count, 512)):
        at = ifd + 2 + index * 12
        if at + 12 > len(tiff):
            return None
        if int.from_bytes(tiff[at : at + 2], order) != 0x0112:
            continue
        kind = int.from_bytes(tiff[at + 2 : at + 4], order)
        items = int.from_bytes(tiff[at + 4 : at + 8], order)
        value = int.from_bytes(tiff[at + 8 : at + 10], order)
        return value if kind == 3 and items == 1 and 1 <= value <= 8 else None
    return None


def _orientation_segment(value: int) -> bytes:
    """An APP1 segment that holds the orientation and nothing else."""
    body = (
        b"Exif\x00\x00"
        + b"MM\x00\x2a\x00\x00\x00\x08"  # big-endian TIFF header, first IFD at 8
        + b"\x00\x01"  # one entry
        + b"\x01\x12\x00\x03\x00\x00\x00\x01"  # Orientation, SHORT, one value
        + value.to_bytes(2, "big")
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"  # no next IFD
    )
    return b"\xff" + bytes([_APP1]) + (len(body) + 2).to_bytes(2, "big") + body


def _adobe_segment(transform: int) -> bytes:
    """An APP14 segment that holds the Adobe colour transform and nothing else."""
    body = b"Adobe" + b"\x00\x64" + b"\x00\x00" + b"\x00\x00" + bytes([transform])
    return b"\xff" + bytes([_APP14]) + (len(body) + 2).to_bytes(2, "big") + body


def strip_jpeg(raw: bytes) -> bytes | None:
    """The image segments of JPEG *raw* up to its end, with orientation and colour transform; None if broken."""
    if raw[:2] != b"\xff\xd8":
        return None
    size = len(raw)
    body = bytearray()
    orientation: int | None = None
    transform: int | None = None
    framed = False
    scanned = False
    pos = 2
    while True:
        # A marker is 0xFF and a code; extra 0xFF bytes before it are fill.
        if pos >= size or raw[pos] != 0xFF:
            return None
        while pos < size and raw[pos] == 0xFF:
            pos += 1
        if pos >= size:
            return None
        marker = raw[pos]
        pos += 1
        if marker == _EOI:
            break
        if marker == 0x00 or marker == 0x01 or marker in _RESTARTS:
            continue  # stands alone, outside a scan
        if marker == _SOI:
            return None
        if pos + 2 > size:
            return None
        length = int.from_bytes(raw[pos : pos + 2], "big")
        end = pos + length
        if length < 2 or end > size:
            return None
        payload = raw[pos + 2 : end]
        if marker == _SOS:
            if not framed:
                return None
            # The scan's header, then its entropy-coded bytes up to the next real marker.
            at = end
            while True:
                found = raw.find(b"\xff", at)
                if found == -1 or found + 1 >= size:
                    return None
                after = raw[found + 1]
                if after == 0x00 or after in _RESTARTS:
                    at = found + 2
                elif after == 0xFF:
                    at = found + 1
                else:
                    break
            body += b"\xff" + bytes([marker]) + raw[pos:found]
            pos = found
            scanned = True
            continue
        if marker in _JPEG_KEEP:
            framed = framed or marker in _SOF
            body += b"\xff" + bytes([marker]) + raw[pos:end]
        elif marker == _APP1 and orientation is None:
            orientation = _orientation(payload)
        elif marker == _APP14 and transform is None:
            if payload.startswith(b"Adobe") and len(payload) >= 12:
                transform = payload[11]
        # Everything else (APP0 to APP15, comments, extensions) is dropped.
        pos = end
    if not (framed and scanned):
        return None
    head = b"\xff\xd8"
    if orientation is not None and orientation != 1:
        head += _orientation_segment(orientation)
    if transform is not None:
        head += _adobe_segment(transform)
    return head + bytes(body) + b"\xff\xd9"
