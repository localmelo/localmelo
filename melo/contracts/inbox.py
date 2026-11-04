"""Run-scoped source of ordered worker instructions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from localmelo.melo.schema import ImagePart


class InboxError(ValueError):
    """The inbox cannot be read safely as an append-only file."""


class InboxSource(Protocol):
    """Snapshot messages and commit offsets only after context insertion.

    ``validate`` is called once before the run's first model call: it refuses
    an inbox the worker's file tools could write and a target that cannot be
    read as a regular file, so those failures are a clean ``setup_error``
    rather than a crash after a step has run.
    """

    def validate(self, workspace: str | None = None) -> None: ...

    def poll(self) -> list[tuple[int, str]]: ...

    def accept(self, offset: int) -> None: ...


class PictureInbox(InboxSource, Protocol):
    """An inbox whose messages can carry pictures (a chat's added words).

    ``pictures(offset)`` returns the pictures, with their data, of the message
    ``poll`` listed under *offset*; the run sends them with its words. An inbox
    without the method carries words only.
    """

    def pictures(self, offset: int) -> Sequence[ImagePart]: ...
