"""Adjacent user instructions keep every image when coalesced."""

from __future__ import annotations

from localmelo.melo.agent._chat import coalesce_user_messages
from localmelo.melo.schema import ImagePart, Message


def test_coalesce_user_messages_merges_images() -> None:
    first = ImagePart.from_bytes(b"first", "image/png")
    second = ImagePart.from_bytes(b"second", "image/jpeg")
    messages = [
        Message("user", "one", images=[first]),
        Message("user", "two", images=[second]),
        Message("assistant", "ok"),
        Message("user", "three"),
    ]
    merged = coalesce_user_messages(messages)
    assert [message.content for message in merged] == ["one\n\ntwo", "ok", "three"]
    assert merged[0].images == [first, second]
    assert merged[1].images == []
