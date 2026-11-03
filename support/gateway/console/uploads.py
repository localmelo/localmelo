"""Pictures on their way to a message: ``POST /api/uploads`` and its stage.

A picture the person drops, pastes or picks is sent here as raw bytes (the one
``/api`` route that takes them; ``guard.py`` gives it its own size and types).
:func:`~localmelo.support.sessions.picture_input.normalize` decides what the
bytes are and returns the picture the model gets; the stage keeps that in memory
under an id, and the message that names the id takes it. Nothing is written to
disk, and a picture nobody sends is gone after :data:`TTL_S` seconds.

* ``POST /api/uploads?name=…`` answers 201 with ``{id, name, mime_type,
  size_bytes, width, height, model_reads}`` (``model_reads`` is whether the chat
  model reads pictures: true, false, or null when unknown); 413 ``picture_too_large``, 415
  ``picture_unsupported`` / ``picture_unreadable`` / ``picture_convert`` / ``picture_scale``.
* ``GET /api/uploads/{id}`` is the picture as it will be sent, for the thumbnail.
* ``DELETE /api/uploads/{id}`` drops it (the thumbnail's remove button).
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response

from localmelo.melo.schema import ImagePart, decode_base64
from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.picture_input import (
    Picture,
    PictureRefused,
    normalize,
)

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console

# Pictures kept at once, and their bytes; the oldest go first.
MAX_STAGED: Final = 24
MAX_STAGED_BYTES: Final = 64 * 1024 * 1024
# Seconds a picture nobody sent is kept.
TTL_S: Final = 1800.0
_ID: Final = re.compile(r"[0-9a-f]{32}")
_NAME_CHARS: Final = re.compile(r"[\x00-\x1f\x7f/\\]+")


@dataclass(frozen=True)
class Staged:
    """One picture waiting for its message."""

    id: str
    name: str
    picture: Picture
    at: float

    def json(self, reads: bool | None = None) -> dict[str, object]:
        """The page's view; *reads* is whether the chat model reads pictures (None: unknown)."""
        image = self.picture.image
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": image.mime_type,
            "size_bytes": image.size_bytes,
            "width": self.picture.width,
            "height": self.picture.height,
            "model_reads": reads,
        }


def refusal_text(refused: PictureRefused) -> str:
    """The sentence for a refused picture, in the language of the request."""
    params = refused.params
    match refused.code:
        case "picture_too_large":
            return t("errors.pictureTooLarge", **params)
        case "picture_too_many_pixels":
            return t("errors.pictureTooManyPixels", **params)
        case "picture_unsupported":
            return t("errors.pictureUnsupported")
        case "picture_convert":
            return t("errors.pictureConvert", **params)
        case "picture_scale":
            return t("errors.pictureScale", **params)
    return t("errors.pictureUnreadable")


def clean_name(name: str) -> str:
    """A file name safe to show: no path, no control characters, at most 120 characters."""
    base = _NAME_CHARS.sub(" ", name.replace("\\", "/").rsplit("/", 1)[-1]).strip()
    return base[:120] or "picture"


class PictureStage:
    """The pictures uploaded and not yet sent, in memory."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._items: OrderedDict[str, Staged] = OrderedDict()
        self._clock = clock

    def _expire(self) -> None:
        now = self._clock()
        for key in [k for k, v in self._items.items() if now - v.at > TTL_S]:
            del self._items[key]
        while len(self._items) > MAX_STAGED or (
            self._items
            and sum(v.picture.image.size_bytes for v in self._items.values())
            > MAX_STAGED_BYTES
        ):
            self._items.popitem(last=False)

    def put(self, picture: Picture, name: str) -> Staged:
        self._expire()
        staged = Staged(uuid.uuid4().hex, clean_name(name), picture, self._clock())
        self._items[staged.id] = staged
        self._expire()
        return staged

    def get(self, key: str) -> Staged | None:
        self._expire()
        return self._items.get(key)

    def resolve(self, keys: Sequence[str]) -> list[Staged]:
        """The staged pictures named by *keys*, in order; 400 for one that is gone."""
        found: list[Staged] = []
        for key in keys:
            staged = self.get(key) if _ID.fullmatch(key) else None
            if staged is None:
                raise ApiError(400, "picture_gone", t("errors.pictureGone"))
            found.append(staged)
        return found

    def release(self, keys: Sequence[str]) -> None:
        """Forget *keys*: their message started."""
        for key in keys:
            self._items.pop(key, None)

    def discard(self, key: str) -> bool:
        return self._items.pop(key, None) is not None


def images_of(staged: Sequence[Staged]) -> list[ImagePart]:
    """The model's pictures for *staged*, in order."""
    return [item.picture.image for item in staged]


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the upload routes on *app*."""

    @app.post("/api/uploads")
    async def upload(
        request: Request, name: str = Query("", max_length=500)
    ) -> JSONResponse:
        raw = await request.body()
        if not raw:
            raise ApiError(400, "invalid_request", t("errors.pictureEmpty"))
        try:
            picture = await asyncio.to_thread(normalize, raw)
        except PictureRefused as refused:
            raise ApiError(
                refused.status, refused.code, refusal_text(refused)
            ) from None
        staged = console.pictures.put(picture, name)
        # The composer warns at once when the model is known not to read pictures.
        return JSONResponse(
            status_code=201, content=staged.json(await console.vision.reads())
        )

    @app.get("/api/uploads/{upload_id}")
    async def staged_picture(upload_id: str) -> Response:
        staged = console.pictures.get(upload_id) if _ID.fullmatch(upload_id) else None
        if staged is None:
            raise ApiError(404, "not_found", t("errors.pictureGone"))
        image = staged.picture.image
        data = decode_base64(image.data) or b""
        return Response(
            content=data,
            media_type=image.mime_type,
            headers={
                "Content-Disposition": "inline",
                "Content-Security-Policy": "default-src 'none'; sandbox",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    @app.delete("/api/uploads/{upload_id}")
    async def drop_picture(upload_id: str) -> Response:
        console.pictures.discard(upload_id)
        return Response(status_code=204)
