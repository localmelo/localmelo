"""Sample pictures for the picture tests: a JPEG and a PNG that carry the worst a phone adds, and GIF, WebP and HEIC built byte by byte.

The metadata samples hold strings that must not survive (a camera name, a GPS
position, a comment, XMP) so a test can look for them in what comes out. The GIF,
WebP and HEIC builders write only what a reader needs: they make a picture of one
colour in any size, so the files stay a few hundred bytes however large they claim
to be. The files are real: macOS reads them (see ``test_picture_formats``).
"""

from __future__ import annotations

import base64
import struct
import zlib

# An 8 x 8 JPEG that already holds a JFIF segment, an EXIF block and a Photoshop block.
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAASABIAAD/4QBMRXhpZgAATU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMA"
    "AAABAAEAAKACAAQAAAABAAAACKADAAQAAAABAAAACAAAAAD/7QA4UGhvdG9zaG9wIDMuMAA4QklNBAQAAAAA"
    "AAA4QklNBCUAAAAAABDUHYzZjwCyBOmACZjs+EJ+/8AAEQgACAAIAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEB"
    "AAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGh"
    "CCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3"
    "eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn"
    "6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQE"
    "AAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpD"
    "REVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2"
    "t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/bAEMABAQEBAQEBgQEBgkGBgYJDAkJ"
    "CQkMDwwMDAwMDxIPDw8PDw8SEhISEhISEhUVFRUVFRkZGRkZHBwcHBwcHBwcHP/bAEMBBAUFBwcHDAcHDB0U"
    "EBQdHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHf/dAAQAAf/aAAwD"
    "AQACEQMRAD8Ah+D/AIE/1H7v07V9J/8ACCf9M/0rh/g//wAsPwr6TrqOc//Z"
)

# Strings that must not survive, and where each is hidden.
CAMERA = b"SecretCam-9000"
GPS = b"GPS-LATLON-48.8584N-2.2945E"
NOTE = b"a comment the person never meant to share"
XMP = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>CreatorTool=Hidden</x:xmpmeta>"
TRAILER = b"TRAILING-VIDEO-BYTES"


def segment(marker: int, body: bytes) -> bytes:
    return b"\xff" + bytes([marker]) + (len(body) + 2).to_bytes(2, "big") + body


def exif_with_gps(orientation: int = 6) -> bytes:
    """An EXIF APP1 body: the camera, the orientation and a GPS block."""
    gps_ifd = (
        b"\x00\x01"
        + struct.pack(">HHI4s", 0x0001, 2, 2, b"N\x00\x00\x00")
        + struct.pack(">I", 0)
        + GPS
    )
    entries = [
        struct.pack(
            ">HHII", 0x010F, 2, len(CAMERA) + 1, 0
        ),  # Make: points at the string
        struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0),
        struct.pack(">HHII", 0x8825, 4, 1, 0),  # GPS IFD pointer
    ]
    ifd0 = struct.pack(">H", len(entries)) + b"".join(entries) + struct.pack(">I", 0)
    return b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08" + ifd0 + CAMERA + b"\x00" + gps_ifd


def photo(*, orientation: int | None = 6, trailing: bool = True) -> bytes:
    """The sample JPEG with the worst a phone adds: GPS EXIF, XMP, ICC, a comment, APP15 and a trailer."""
    extra = b""
    if orientation is not None:
        extra += segment(0xE1, exif_with_gps(orientation))
    extra += segment(0xE1, XMP)
    extra += segment(0xE2, b"ICC_PROFILE\x00\x01\x01" + b"profile-by-" + CAMERA)
    extra += segment(0xEF, NOTE)
    extra += segment(0xFE, NOTE)
    out = JPEG[:2] + extra + JPEG[2:]
    return out + TRAILER if trailing else out


def markers(jpeg: bytes) -> list[int]:
    """The markers of a JPEG before its first scan, in order."""
    found = []
    pos = 2
    while pos < len(jpeg) and jpeg[pos] == 0xFF:
        marker = jpeg[pos + 1]
        found.append(marker)
        if marker == 0xDA:
            break
        pos += 2 + int.from_bytes(jpeg[pos + 2 : pos + 4], "big")
    return found


def png_chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body))
    )


def png_with_metadata(width: int = 8, height: int = 8) -> bytes:
    rows = b"".join(
        b"\x00" + bytes((x * 7 + y) % 256 for x in range(width * 3))
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunks = [
        png_chunk(b"IHDR", header),
        png_chunk(b"tEXt", b"Comment\x00" + NOTE),
        png_chunk(b"eXIf", b"MM\x00\x2a" + CAMERA + GPS),
        png_chunk(b"gAMA", struct.pack(">I", 45455)),
        png_chunk(b"sRGB", b"\x00"),
        png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + XMP),
        png_chunk(b"zTXt", b"Note\x00\x00" + zlib.compress(NOTE)),
        png_chunk(b"tIME", struct.pack(">HBBBBB", 2026, 11, 3, 9, 30, 0)),
        png_chunk(b"pHYs", struct.pack(">IIB", 2835, 2835, 1)),
        png_chunk(b"tRNS", b"\x00\x00\x00\x00\x00\x00"),
        png_chunk(b"iCCP", b"profile\x00\x00" + zlib.compress(CAMERA)),
        png_chunk(b"IDAT", zlib.compress(rows)),
        png_chunk(b"IEND", b""),
    ]
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks) + TRAILER


def png_kinds(raw: bytes) -> list[bytes]:
    kinds, pos = [], 8
    while pos < len(raw):
        length = int.from_bytes(raw[pos : pos + 4], "big")
        kinds.append(raw[pos + 4 : pos + 8])
        pos += 12 + length
    return kinds


def assert_clean(raw: bytes) -> None:
    for secret in (
        CAMERA,
        GPS,
        NOTE,
        b"xmpmeta",
        b"Photoshop",
        b"JFIF",
        b"ICC_PROFILE",
        TRAILER,
    ):
        assert secret not in raw, secret


def wide(jpeg: bytes, width: int, height: int) -> bytes:
    """*jpeg* with the frame header of a *width* x *height* picture (its scan is not decoded here)."""
    at = jpeg.index(b"\xff\xc0") + 5
    return jpeg[:at] + struct.pack(">HH", height, width) + jpeg[at + 4 :]


# ── GIF ──


def sub_blocks(data: bytes) -> bytes:
    """*data* as GIF data sub-blocks, ending with the empty block."""
    out = b""
    for at in range(0, len(data), 255):
        part = data[at : at + 255]
        out += bytes([len(part)]) + part
    return out + b"\x00"


def lzw(indices: bytes) -> bytes:
    """GIF image data for *indices* (palette positions, 8 bits each), as an LZW stream.

    Codes stay 9 bits wide: the table is cleared before it would need a tenth bit,
    so a run of one colour packs well and any decoder reads the stream.
    """
    clear, end = 256, 257
    acc = bits = 0
    out = bytearray()

    def put(code: int) -> None:
        nonlocal acc, bits
        acc |= code << bits
        bits += 9
        while bits >= 8:
            out.append(acc & 255)
            acc >>= 8
            bits -= 8

    table = {bytes([i]): i for i in range(256)}
    free = end + 1
    put(clear)
    run = b""
    for index in indices:
        longer = run + bytes([index])
        if longer in table:
            run = longer
            continue
        put(table[run])
        table[longer] = free
        free += 1
        run = bytes([index])
        if free == 512:
            put(clear)
            table = {bytes([i]): i for i in range(256)}
            free = end + 1
    if run:
        put(table[run])
    put(end)
    if bits:
        out.append(acc & 255)
    return bytes([8]) + sub_blocks(bytes(out))


# 256 colours: red, green, blue and yellow first, then black.
GIF_PALETTE = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0)) + bytes(756)
GIF_COMMENT = b"SecretComment-GIF"
GIF_XMP = b"<x:xmpmeta SecretXmp-GIF/>"
# A frame that claims more pixels than this holds only this many (a header-only bomb).
GIF_PIXELS_HELD = 4096


def gif_extras() -> bytes:
    """A comment extension and an XMP application extension, as a camera app writes them."""
    comment = b"\x21\xfe" + sub_blocks(GIF_COMMENT)
    xmp = b"\x21\xff\x0bXMP DataXMP" + sub_blocks(GIF_XMP)
    return comment + xmp


def gif_picture(
    width: int,
    height: int,
    frames: list[tuple[int, int, int]],
    extras: bytes = b"",
) -> bytes:
    """A GIF whose logical screen is *width* x *height* and whose frames are ``(w, h, colour)``.

    A frame descriptor may claim any size; its pixels are the single colour index.
    A frame of over a million pixels holds only :data:`GIF_PIXELS_HELD` of them: it
    is a header for a reader of headers, not a picture.
    """
    out = b"GIF89a" + struct.pack("<HHBBB", width, height, 0xF7, 0, 0) + GIF_PALETTE
    out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00" + extras
    for w, h, colour in frames:
        out += b"\x21\xf9\x04\x00\x0a\x00\x00\x00"
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, w, h, 0)
        out += lzw(
            bytes([colour])
            * min(w * h, GIF_PIXELS_HELD if w * h > 1_000_000 else w * h)
        )
    return out + b"\x3b"


# ── WebP ──


class _Bits:
    """A little-endian bit writer, as the lossless WebP format reads."""

    def __init__(self) -> None:
        self.value = 0
        self.count = 0

    def put(self, value: int, count: int) -> None:
        self.value |= value << self.count
        self.count += count

    def data(self) -> bytes:
        return self.value.to_bytes((self.count + 7) // 8, "little")


def vp8l(width: int, height: int, argb: tuple[int, int, int, int]) -> bytes:
    """A lossless WebP bitstream of one colour: five one-symbol codes, no bits per pixel."""
    alpha, red, green, blue = argb
    bits = _Bits()
    bits.put(0x2F, 8)
    bits.put(width - 1, 14)
    bits.put(height - 1, 14)
    bits.put(1 if alpha != 255 else 0, 1)
    bits.put(0, 3)
    bits.put(0, 3)  # no transform, no colour cache, one prefix code group
    for symbol, wide in (
        (green, True),
        (red, True),
        (blue, True),
        (alpha, True),
        (0, False),
    ):
        bits.put(1, 1)  # a simple code
        bits.put(0, 1)  # of one symbol
        bits.put(1 if wide else 0, 1)  # whose value takes 8 bits or 1
        bits.put(symbol, 8 if wide else 1)
    return bits.data()


def riff_chunk(tag: bytes, body: bytes) -> bytes:
    return (
        tag + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) % 2 else b"")
    )


def riff(chunks: list[bytes]) -> bytes:
    body = b"WEBP" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


ORANGE = (255, 255, 128, 0)
WEBP_XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description Author="SecretXmp-WebP"/></x:xmpmeta>'


def exif_tiff(orientation: int = 6) -> bytes:
    """A well-formed big-endian TIFF block: the camera, the orientation and a GPS block."""
    make = CAMERA + b"\x00"
    method = GPS
    ifd0_size = 2 + 3 * 12 + 4
    make_at = 8 + ifd0_size
    gps_at = make_at + len(make)
    ifd0 = struct.pack(">H", 3)
    ifd0 += struct.pack(">HHII", 0x010F, 2, len(make), make_at)
    ifd0 += struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
    ifd0 += struct.pack(">HHII", 0x8825, 4, 1, gps_at)
    ifd0 += struct.pack(">I", 0)
    gps_ifd_size = 2 + 2 * 12 + 4
    gps = struct.pack(">H", 2)
    gps += struct.pack(">HHI4s", 0x0001, 2, 2, b"N\x00\x00\x00")
    gps += struct.pack(">HHII", 0x001B, 7, len(method), gps_at + gps_ifd_size)
    gps += struct.pack(">I", 0)
    return b"MM\x00\x2a" + struct.pack(">I", 8) + ifd0 + make + gps + method


def vp8_header(width: int, height: int) -> bytes:
    """The start of a lossy WebP bitstream: a frame tag, the start code and the size (no image data)."""
    return b"\x10\x02\x00\x9d\x01\x2a" + struct.pack("<HH", width, height)


def webp_picture(
    width: int,
    height: int,
    *,
    metadata: bool = False,
    canvas: tuple[int, int] | None = None,
) -> bytes:
    """A WebP of one colour; the extended form with EXIF and XMP chunks for *metadata*, or with a *canvas* of its own size."""
    image = riff_chunk(b"VP8L", vp8l(width, height, ORANGE))
    if not metadata and canvas is None:
        return riff([image])
    cw, ch = canvas or (width, height)
    flags = 0x08 | 0x04 if metadata else 0
    head = struct.pack("<B3x", flags) + (cw - 1).to_bytes(3, "little")
    head += (ch - 1).to_bytes(3, "little")
    chunks = [riff_chunk(b"VP8X", head), image]
    if metadata:
        chunks += [riff_chunk(b"EXIF", exif_tiff()), riff_chunk(b"XMP ", WEBP_XMP)]
    return riff(chunks)


def webp_animation(width: int, height: int, frames: list[tuple[int, int]]) -> bytes:
    """An animated WebP: a *width* x *height* canvas and ``(w, h)`` frames, each with a lossless bitstream."""
    canvas = struct.pack("<B3x", 0x02) + (width - 1).to_bytes(3, "little")
    canvas += (height - 1).to_bytes(3, "little")
    chunks = [
        riff_chunk(b"VP8X", canvas),
        riff_chunk(b"ANIM", b"\xff\xff\xff\xff\x00\x00"),
    ]
    for w, h in frames:
        head = (0).to_bytes(3, "little") * 2
        head += (w - 1).to_bytes(3, "little") + (h - 1).to_bytes(3, "little")
        head += (100).to_bytes(3, "little") + b"\x00"
        chunks.append(
            riff_chunk(b"ANMF", head + riff_chunk(b"VP8L", vp8l(w, h, ORANGE)))
        )
    return riff(chunks)


# ── HEIC ──


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def heic_headers(*sizes: tuple[int, int]) -> bytes:
    """The boxes of a HEIC that name only the sizes of its images, with no image data.

    Enough for a reader of headers, not for a decoder.
    """
    ispe = b"".join(box(b"ispe", struct.pack(">III", 0, w, h)) for w, h in sizes)
    meta = box(b"meta", b"\x00\x00\x00\x00" + box(b"iprp", box(b"ipco", ispe)))
    return box(b"ftyp", b"heic\x00\x00\x00\x00heicmif1") + meta
