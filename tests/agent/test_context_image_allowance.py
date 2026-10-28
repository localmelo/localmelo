from __future__ import annotations

import struct
import zlib

import pytest

from localmelo.melo.agent._request_accounting import (
    IMAGE_TOKENS,
    PROXY_VERSION,
    request_proxy,
)
from localmelo.melo.schema import ImagePart, Message


def test_image_formula_has_new_proxy_version() -> None:
    assert PROXY_VERSION == "utf8-bytes-v2"


def png(width: int, height: int) -> bytes:
    header = b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + header
        + struct.pack(">I", zlib.crc32(header))
    )


def allowance(*images: ImagePart, role: str = "user") -> int:
    return request_proxy([Message(role=role, images=list(images))], None, replay=False)[
        1
    ]


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1, 1, 1),
        (28, 28, 1),
        (29, 57, 6),
        (100000, 100000, IMAGE_TOKENS),
        (0, 28, IMAGE_TOKENS),
    ],
)
def test_png_dimensions(width: int, height: int, expected: int) -> None:
    assert allowance(ImagePart.from_bytes(png(width, height), "image/png")) == expected


def test_unknown_bad_and_reference_images() -> None:
    invalid = ImagePart(mime_type="image/png", data="not-base64!")
    corrupt = ImagePart.from_bytes(png(28, 28)[:-1], "image/png")
    unknown = ImagePart.from_bytes(b"not an image", "image/png")
    reference = unknown.as_reference()
    assert allowance(invalid, corrupt, unknown, reference) == 3 * IMAGE_TOKENS
    assert allowance(unknown, role="assistant") == 0
    assert allowance(unknown, role="tool") == IMAGE_TOKENS


def test_other_supported_headers() -> None:
    gif = b"GIF89a" + struct.pack("<HH", 29, 56) + b"\0\0\0"
    jpeg = b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x38\x00\x1d\x01\x01\x11\x00"
    webp = (
        b"RIFF"
        + struct.pack("<I", 22)
        + b"WEBPVP8X\x0a\0\0\0\0\0\0\0"
        + (28).to_bytes(3, "little")
        + (55).to_bytes(3, "little")
    )
    for raw, mime in ((gif, "image/gif"), (jpeg, "image/jpeg"), (webp, "image/webp")):
        assert allowance(ImagePart.from_bytes(raw, mime)) == 4


def test_payload_not_in_proxy_and_allowances_sum() -> None:
    image = ImagePart.from_bytes(png(29, 57), "image/png")
    text_only, _ = request_proxy([Message(role="user")], None, replay=False)
    combined, total = request_proxy(
        [Message(role="user", images=[image, image])], None, replay=False
    )
    assert total == 12
    assert combined == text_only + total
