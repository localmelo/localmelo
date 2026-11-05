"""The pixel size a picture's own headers claim, read without decoding anything.

A small file can hold a huge picture, so the size is read from the structure
(bytes of headers, never pixels) and checked against the pixel limit before any
decoder or converter sees the file. Each reader returns every size the file
mentions, so a picture cannot hide a large frame behind a small first one:

* PNG and JPEG: the header, through :func:`~localmelo.melo.agent._request_accounting._image_dimensions`.
* GIF: the logical screen and the descriptor of every frame.
* WebP: the canvas of an extended file, every animation frame, and the bitstream
  header of a lossy (``VP8``) or lossless (``VP8L``) image.
* HEIC: every ``ispe`` (image spatial extent) property of the file.

A file whose structure does not hold together gives ``None``; the caller refuses
it as unreadable. A size that is claimed is not a size that is proven: a decoder
can still disagree with a lying header, so the picture that comes out of a
conversion is measured again by the caller.
"""

from __future__ import annotations

from typing import Final

from localmelo.melo.agent._request_accounting import _image_dimensions

Size = tuple[int, int]

# The most boxes or chunks one file may list before it is called unreadable.
_MAX_PARTS: Final = 100_000


def claimed_sizes(raw: bytes, kind: str) -> list[Size] | None:
    """Every size *raw* (of *kind*, as ``sniff`` names it) claims, or None if it is not readable."""
    if kind in ("png", "jpeg"):
        size = _image_dimensions(raw)
        return [size] if size else None
    if kind == "gif":
        return _gif(raw)
    if kind == "webp":
        return _webp(raw)
    if kind == "heic":
        return _heic(raw)
    return None


# ── GIF ──


def _gif_blocks(raw: bytes, pos: int) -> int | None:
    """The offset after the data sub-blocks that start at *pos*, or None if they run off the end."""
    parts = 0
    while pos < len(raw):
        length = raw[pos]
        pos += 1 + length
        if length == 0:
            return pos
        parts += 1
        if parts > _MAX_PARTS:
            return None
    return None


def _gif(raw: bytes) -> list[Size] | None:
    """The logical screen and every frame of a GIF."""
    if len(raw) < 13 or raw[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    sizes = [(int.from_bytes(raw[6:8], "little"), int.from_bytes(raw[8:10], "little"))]
    pos = 13
    if raw[10] & 0x80:
        pos += 3 << ((raw[10] & 7) + 1)
    frames = 0
    ended = False
    while pos < len(raw):
        block = raw[pos]
        if block == 0x3B:  # trailer
            ended = True
            break
        if block == 0x21:  # extension: a label, then data sub-blocks
            after = _gif_blocks(raw, pos + 2)
        elif block == 0x2C:  # image descriptor
            if pos + 10 > len(raw):
                return None
            sizes.append(
                (
                    int.from_bytes(raw[pos + 5 : pos + 7], "little"),
                    int.from_bytes(raw[pos + 7 : pos + 9], "little"),
                )
            )
            frames += 1
            after = pos + 10
            if raw[pos + 9] & 0x80:
                after += 3 << ((raw[pos + 9] & 7) + 1)
            # The minimum LZW code size, then the data sub-blocks.
            after = _gif_blocks(raw, after + 1)
        else:
            return None
        if after is None:
            return None
        pos = after
    # A file that stops before its trailer was cut short.
    if frames == 0 or not ended:
        return None
    return sizes


# ── WebP ──


def _le(raw: bytes) -> int:
    return int.from_bytes(raw, "little")


def _webp_part(tag: bytes, body: bytes, sizes: list[Size], nested: bool) -> bool:
    """Add the size *body* (a chunk named *tag*) claims; False if the chunk is malformed."""
    if tag == b"VP8X":
        if len(body) < 10:
            return False
        sizes.append((1 + _le(body[4:7]), 1 + _le(body[7:10])))
    elif tag == b"VP8 ":
        if len(body) < 10 or body[3:6] != b"\x9d\x01\x2a":
            return False
        sizes.append((_le(body[6:8]) & 0x3FFF, _le(body[8:10]) & 0x3FFF))
    elif tag == b"VP8L":
        if len(body) < 5 or body[0] != 0x2F:
            return False
        bits = _le(body[1:5])
        sizes.append(((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
    elif tag == b"ANMF" and not nested:
        if len(body) < 16:
            return False
        sizes.append((1 + _le(body[6:9]), 1 + _le(body[9:12])))
        return _webp_chunks(body, 16, len(body), sizes, nested=True)
    return True


def _webp_chunks(
    raw: bytes, pos: int, end: int, sizes: list[Size], *, nested: bool
) -> bool:
    """Walk the chunks in ``raw[pos:end]``; False unless they fill it exactly, pad bytes included."""
    parts = 0
    while pos + 8 <= end:
        tag = raw[pos : pos + 4]
        length = _le(raw[pos + 4 : pos + 8])
        body_end = pos + 8 + length
        pos = body_end + (length & 1)
        if pos > end:  # the body, or the pad byte after an odd one, is not there
            return False
        if not _webp_part(tag, raw[body_end - length : body_end], sizes, nested):
            return False
        parts += 1
        if parts > _MAX_PARTS:
            return False
    return pos == end  # not a part of a header left over


def _webp(raw: bytes) -> list[Size] | None:
    """The canvas, every frame and the image header of a WebP."""
    if len(raw) < 20 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        return None
    end = 8 + _le(raw[4:8])
    if end > len(raw):  # the length the file gives is more than the file holds
        return None
    sizes: list[Size] = []
    if not _webp_chunks(raw, 12, end, sizes, nested=False):
        return None
    return sizes or None


# ── HEIC ──


def _boxes(raw: bytes, pos: int, end: int) -> list[tuple[bytes, int, int]] | None:
    """The boxes in ``raw[pos:end]`` as (type, body start, body end); None if one is malformed."""
    found: list[tuple[bytes, int, int]] = []
    while pos + 8 <= end:
        length = int.from_bytes(raw[pos : pos + 4], "big")
        kind = raw[pos + 4 : pos + 8]
        start = pos + 8
        if length == 1:
            if pos + 16 > end:
                return None
            length = int.from_bytes(raw[pos + 8 : pos + 16], "big")
            start = pos + 16
        elif length == 0:
            length = end - pos
        if length < start - pos or pos + length > end:
            return None
        found.append((kind, start, pos + length))
        pos += length
        if len(found) > _MAX_PARTS:
            return None
    # Fewer than eight bytes left over are the start of a box that is not there.
    return found if pos == end else None


def _heic(raw: bytes) -> list[Size] | None:
    """Every ``ispe`` property in ``meta`` > ``iprp`` > ``ipco`` of a HEIC; None if any box is broken."""
    top = _boxes(raw, 0, len(raw))
    if top is None:
        return None
    sizes: list[Size] = []
    for kind, start, end in top:
        if kind != b"meta":
            continue
        found = _heic_meta(raw, start, end)
        if found is None:
            return None
        sizes += found
    return sizes or None


def _heic_meta(raw: bytes, start: int, end: int) -> list[Size] | None:
    """The sizes in one ``meta`` box, a full box: a version of 0 and flags come first."""
    if end - start < 4 or raw[start] != 0:
        return None
    children = _boxes(raw, start + 4, end)
    if children is None:
        return None
    sizes: list[Size] = []
    for kind, begin, stop in children:
        if kind == b"iprp":
            found = _heic_iprp(raw, begin, stop)
            if found is None:
                return None
            sizes += found
    return sizes


def _heic_iprp(raw: bytes, start: int, end: int) -> list[Size] | None:
    """The sizes in the ``ipco`` boxes of one ``iprp`` box."""
    children = _boxes(raw, start, end)
    if children is None:
        return None
    sizes: list[Size] = []
    for kind, begin, stop in children:
        if kind != b"ipco":
            continue
        properties = _boxes(raw, begin, stop)
        if properties is None:
            return None
        for kind2, begin2, stop2 in properties:
            if kind2 != b"ispe":
                continue
            # A full box: a version of 0 and flags, then the width and the height.
            if stop2 - begin2 < 12 or raw[begin2] != 0:
                return None
            sizes.append(
                (
                    int.from_bytes(raw[begin2 + 4 : begin2 + 8], "big"),
                    int.from_bytes(raw[begin2 + 8 : begin2 + 12], "big"),
                )
            )
    return sizes
