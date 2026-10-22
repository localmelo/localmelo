"""Image parts in the schema — no network, no model."""

from __future__ import annotations

import base64
import dataclasses
import hashlib

import pytest

from localmelo.melo.schema import (
    ImagePart,
    Message,
    ToolResult,
    decode_base64,
    describe_images,
)

RAW = b"\x89PNG\r\n\x1a\n fake image bytes"
B64 = base64.b64encode(RAW).decode()


class TestImagePart:
    def test_digest_and_size_from_data(self) -> None:
        image = ImagePart(mime_type="image/png", data=B64)
        assert image.sha256 == hashlib.sha256(RAW).hexdigest()
        assert image.size_bytes == len(RAW)

    def test_from_bytes_matches(self) -> None:
        assert ImagePart.from_bytes(RAW, "image/png") == ImagePart(
            mime_type="image/png", data=B64
        )

    def test_whitespace_in_base64_is_removed(self) -> None:
        wrapped = "\n".join(B64[i : i + 8] for i in range(0, len(B64), 8))
        image = ImagePart(mime_type="image/png", data=f" {wrapped}\r\n\t")
        assert image.size_bytes == len(RAW)
        # What is sent is what the checker saw (see test_provider_vision.py
        # for the request bodies).
        assert image.data == B64
        assert image.data_url == f"data:image/png;base64,{B64}"

    def test_digest_and_size_always_come_from_data(self) -> None:
        real = (hashlib.sha256(RAW).hexdigest(), len(RAW))
        stale = ImagePart(
            mime_type="image/png", data=B64, sha256="0" * 64, size_bytes=1
        )
        assert (stale.sha256, stale.size_bytes) == real
        # A known digest without a size still gets the real size.
        digest_only = ImagePart(mime_type="image/png", data=B64, sha256=real[0])
        assert (digest_only.sha256, digest_only.size_bytes) == real

    def test_replacing_data_updates_digest_and_size(self) -> None:
        other = b"a different, larger fake image"
        image = dataclasses.replace(
            ImagePart.from_bytes(RAW, "image/png"),
            data=base64.b64encode(other).decode(),
        )
        assert image.sha256 == hashlib.sha256(other).hexdigest()
        assert image.size_bytes == len(other)

    def test_reference_keeps_given_digest_and_size(self) -> None:
        ref = ImagePart(mime_type="image/png", sha256="ab" * 32, size_bytes=7)
        assert (ref.data, ref.sha256, ref.size_bytes) == ("", "ab" * 32, 7)

    def test_invalid_base64_has_zero_size(self) -> None:
        image = ImagePart(mime_type="image/png", data="not base64!")
        assert image.size_bytes == 0
        assert len(image.sha256) == 64
        assert decode_base64("not base64!") is None

    def test_data_url(self) -> None:
        image = ImagePart(mime_type="image/jpeg", data=B64)
        assert image.data_url == f"data:image/jpeg;base64,{B64}"

    def test_reference_copy_drops_data(self) -> None:
        image = ImagePart(mime_type="image/png", data=B64)
        ref = image.as_reference()
        assert ref.data == ""
        assert (ref.sha256, ref.size_bytes, ref.mime_type) == (
            image.sha256,
            image.size_bytes,
            "image/png",
        )
        assert ref.ref == image.ref

    def test_repr_hides_base64(self) -> None:
        assert B64 not in repr(ImagePart(mime_type="image/png", data=B64))

    def test_frozen(self) -> None:
        image = ImagePart(mime_type="image/png", data=B64)
        with pytest.raises(dataclasses.FrozenInstanceError):
            image.data = ""  # type: ignore[misc]

    def test_describe_images(self) -> None:
        one = ImagePart(mime_type="image/png", data=B64)
        two = ImagePart.from_bytes(b"other", "image/jpeg")
        assert describe_images([]) == ""
        assert (
            describe_images([one]) == f"[1 image: sha256:{one.sha256[:12]} image/png]"
        )
        assert describe_images([one, two]).startswith("[2 images: sha256:")


class TestMessageImages:
    def test_default_empty(self) -> None:
        assert Message(role="user", content="hi").images == []
        assert ToolResult(tool_name="t").images == []

    def test_images_list_is_copied(self) -> None:
        images = [ImagePart(mime_type="image/png", data=B64)]
        msg = Message(role="tool", content="x", tool_call_id="c", images=images)
        images.append(ImagePart.from_bytes(b"more", "image/png"))
        assert len(msg.images) == 1

    def test_replace_keeps_images(self) -> None:
        msg = Message(
            role="user",
            content="look",
            images=[ImagePart(mime_type="image/png", data=B64)],
        )
        assert dataclasses.replace(msg, content="again").images == msg.images
