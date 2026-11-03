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
    while pos < len(raw):
        block = raw[pos]
        if block == 0x3B:  # trailer
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
    if frames == 0:
        return None
    return sizes or None


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
    """Walk the chunks in ``raw[pos:end]``; False if the structure does not hold."""
    parts = 0
    while pos + 8 <= end:
        tag = raw[pos : pos + 4]
        length = _le(raw[pos + 4 : pos + 8])
        body_end = pos + 8 + length
        if body_end > end:
            return False
        if not _webp_part(tag, raw[pos + 8 : body_end], sizes, nested):
            return False
        pos = body_end + (length & 1)
        parts += 1
        if parts > _MAX_PARTS:
            return False
    return True


def _webp(raw: bytes) -> list[Size] | None:
    """The canvas, every frame and the image header of a WebP."""
    if len(raw) < 20 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        return None
    end = min(len(raw), 8 + _le(raw[4:8]))
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
    return found


def _heic(raw: bytes) -> list[Size] | None:
    """Every ``ispe`` property in ``meta`` > ``iprp`` > ``ipco`` of a HEIC."""
    top = _boxes(raw, 0, len(raw))
    if top is None:
        return None
    sizes: list[Size] = []
    for kind, start, end in top:
        if kind != b"meta":
            continue
        # ``meta`` is a full box: four bytes of version and flags come first.
        inside = _boxes(raw, start + 4, end)
        for kind2, start2, end2 in inside or []:
            if kind2 != b"iprp":
                continue
            for kind3, start3, end3 in _boxes(raw, start2, end2) or []:
                if kind3 != b"ipco":
                    continue
                for kind4, start4, end4 in _boxes(raw, start3, end3) or []:
                    if kind4 == b"ispe" and end4 - start4 >= 12:
                        sizes.append(
                            (
                                int.from_bytes(raw[start4 + 4 : start4 + 8], "big"),
                                int.from_bytes(raw[start4 + 8 : start4 + 12], "big"),
                            )
                        )
    return sizes or None
