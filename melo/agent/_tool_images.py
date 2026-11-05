"""Pictures that tools return, kept in RAM and sent to a model that can see.

History and working memory keep reference copies of tool-result images
(see :meth:`ImagePart.as_reference`). With ``AgentOptions.tool_images`` on,
:class:`ToolImageStore` keeps the payloads of the newest ones in RAM, and each
planning request carries their pixels: the providers put them in a ``user``
message after the tool results. The newest ``keep`` images are kept; when one
more arrives past ``keep + retire - 1``, the oldest ``retire`` lose their pixels
together, so the request prefix that a server caches changes once per batch,
not on every step. An image whose digest a kept image already has is not sent
again. A payload wider than :data:`MAX_IMAGE_WIDTH` is scaled to that width as
a JPEG once, when it is stored (with macOS ``sips``; elsewhere it is kept as
it is).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from localmelo.melo.agent._request_accounting import _image_dimensions
from localmelo.melo.schema import ImagePart, Message, ToolResult, decode_base64

MAX_IMAGE_WIDTH: Final = 1280
JPEG_QUALITY: Final = 75
SCALE_TIMEOUT_S: Final = 20.0
# Pictures the person added whose pixels stay in RAM; older ones lose them.
USER_KEEP: Final = 8
# Added to the tool message of an image whose pixels an earlier one carries.
UNCHANGED_NOTE: Final = "[same image as an earlier tool result; its pixels are above]"
# Ends the tool message of a picture that does not reach the model (images off, or the
# server does not say it reads them), so the model does not describe what it cannot see.
WITHHELD_NOTE: Final = (
    "You cannot see this picture (this model is not sent pictures); the person can. "
    "Do not describe it or guess what it shows."
)
# The tools that read a page as text; the note points to those the model has.
TEXT_READERS: Final = ("browser_read", "browser_find")


def withheld_note(readers: Sequence[str]) -> str:
    """The line that ends a tool message whose picture is withheld; *readers* are the text tools the model has."""
    if not readers:
        return WITHHELD_NOTE
    return f"{WITHHELD_NOTE} Read the page with {' or '.join(readers)}."


def _sips_jpeg(raw: bytes, width: int) -> bytes | None:
    """*raw* scaled to *width* pixels wide as a JPEG by ``sips``; None if it cannot."""
    sips = shutil.which("sips")
    if sips is None:
        return None
    with tempfile.TemporaryDirectory(prefix="melo-image-") as folder:
        source, target = os.path.join(folder, "in"), os.path.join(folder, "out.jpg")
        Path(source).write_bytes(raw)
        argv = [sips, "-s", "format", "jpeg", "-s", "formatOptions"]
        argv += [str(JPEG_QUALITY), "--resampleWidth", str(width), source]
        try:
            done = subprocess.run(
                [*argv, "--out", target],
                capture_output=True,
                timeout=SCALE_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0 or not os.path.isfile(target):
            return None
        return Path(target).read_bytes()


def shrink(image: ImagePart) -> ImagePart:
    """*image*, or a JPEG copy :data:`MAX_IMAGE_WIDTH` pixels wide when it is wider."""
    raw = decode_base64(image.data) if image.data else None
    size = _image_dimensions(raw) if raw else None
    if raw is None or size is None or size[0] <= MAX_IMAGE_WIDTH:
        return image
    scaled = _sips_jpeg(raw, MAX_IMAGE_WIDTH)
    if not scaled or _image_dimensions(scaled) is None:
        return image
    return ImagePart.from_bytes(scaled, "image/jpeg")


class ToolImageStore:
    """The payloads of the newest tool-result images, and which ones a request carries.

    ``_live`` holds the reference copies (by identity) on working-memory tool
    messages whose pixels go out, oldest first, and ``_repeats`` those that
    arrived with the digest of a live one; ``_payloads`` holds the payload for
    each live digest, keyed by the digest the tool's image had.
    """

    def __init__(self, keep: int = 3, retire: int = 2) -> None:
        self.keep_count = keep
        self.retire = retire
        self._payloads: dict[str, ImagePart] = {}
        self._live: list[ImagePart] = []
        self._repeats: list[ImagePart] = []
        # The pictures the person added: payloads by digest, oldest first.
        self._user: dict[str, ImagePart] = {}
        # Whether the messages last hydrated carry any pixels (a request the server
        # refused for its pictures is told apart from one it refused for another reason).
        self.sent_pixels = False

    @property
    def live(self) -> list[ImagePart]:
        return list(self._live)

    async def prepare(self, results: Sequence[ToolResult]) -> None:
        """Store each new payload of *results*, scaled once (off the event loop)."""
        for result in results:
            for image in result.images:
                if image.data and image.sha256 not in self._payloads:
                    self._payloads[image.sha256] = await asyncio.to_thread(
                        shrink, image
                    )

    def keep(self, messages: Sequence[Message]) -> None:
        """Make the images on *messages* (a turn's tool messages) live, then retire."""
        digests = {image.sha256 for image in self._live}
        for message in messages:
            for image in message.images if message.role == "tool" else ():
                if image.sha256 in digests:
                    self._repeats.append(image)  # its pixels are already sent
                elif image.sha256 in self._payloads:
                    digests.add(image.sha256)
                    self._live.append(image)
        while len(self._live) > self.keep_count + self.retire - 1:
            del self._live[: self.retire]
        digests = {image.sha256 for image in self._live}
        self._repeats = [image for image in self._repeats if image.sha256 in digests]
        self._payloads = {d: p for d, p in self._payloads.items() if d in digests}

    def keep_user(self, images: Sequence[ImagePart]) -> list[ImagePart]:
        """Keep the payloads of pictures the person added; the reference copies for their message.

        The newest :data:`USER_KEEP` stay. They are not scaled again: the
        gateway normalized them when they were added.
        """
        for image in images:
            if image.data:
                self._user.pop(image.sha256, None)
                self._user[image.sha256] = image
        while len(self._user) > USER_KEEP:
            del self._user[next(iter(self._user))]
        return [image.as_reference() for image in images]

    def _hydrate_user(self, message: Message) -> Message:
        """A person's message with the payloads that are kept, and a line for the others."""
        if not message.images or all(image.data for image in message.images):
            return message
        images = [self._user.get(image.sha256, image) for image in message.images]
        gone = [image for image in images if not image.data]
        content = message.content
        if gone:
            noun = "picture" if len(gone) == 1 else "pictures"
            content = f"{content}\n[{len(gone)} {noun} no longer attached: " + (
                ", ".join(image.ref for image in gone) + "]"
            )
        return dataclasses.replace(message, content=content, images=images)

    def hydrate(self, messages: list[Message]) -> list[Message]:
        """*messages* with the payloads of live images; the same list when none are."""
        if not self._live and not any(
            message.role == "user" and message.images for message in messages
        ):
            self.sent_pixels = False
            return messages
        live = {id(image) for image in self._live}
        repeats = {id(image) for image in self._repeats}
        out: list[Message] = []
        for message in messages:
            if message.role == "user":
                out.append(self._hydrate_user(message))
                continue
            ids = {id(image) for image in message.images}
            if message.role == "tool" and ids & (live | repeats):
                content = message.content
                if ids & repeats:
                    content = f"{content}\n{UNCHANGED_NOTE}"
                images = [
                    self._payloads[image.sha256] if id(image) in live else image
                    for image in message.images
                ]
                message = dataclasses.replace(message, content=content, images=images)
            out.append(message)
        self.sent_pixels = any(
            image.data for message in out for image in message.images
        )
        return out
