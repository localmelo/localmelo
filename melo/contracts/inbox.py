"""Run-scoped source of ordered worker instructions."""

from __future__ import annotations

from typing import Final, Protocol


class InboxError(ValueError):
    """The inbox cannot be read safely as an append-only file."""


# How many notes a turn takes, and the length of one: the gateway and the terminal both hold to these.
MAX_NOTES: Final = 20
MAX_NOTE_CHARS: Final = 8_000


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
