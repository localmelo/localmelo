"""Run-scoped source of ordered worker instructions."""

from __future__ import annotations

from typing import Protocol


class InboxSource(Protocol):
    """Snapshot messages and commit offsets only after context insertion."""

    def poll(self) -> list[tuple[int, str]]: ...

    def accept(self, offset: int) -> None: ...
