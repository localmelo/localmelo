"""Words the person adds to a chat's turn while it runs, as the run's inbox.

The Agent's run polls its inbox at the start of each step and accepts a
message once the request that carries it is built (``inbox.accepted``).
:class:`ChatNotes` is that inbox for a console chat: it reads the chat's notes
from the queued-message store (:class:`~.asks.OwnerAsks`), so a note the run
never read stays queued and runs as the chat's next turn. The model reads each
note as the person's message, marked as added while it worked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .asks import OwnerAsks

# The line before each note in the request: the person's words, not a tool's.
NOTE_MARK: Final = "[The user added this while you were working]"


class ChatNotes:
    """The run inbox of one chat (``melo.contracts.inbox.InboxSource``)."""

    def __init__(self, asks: OwnerAsks, chat_id: str) -> None:
        self._asks = asks
        self._chat_id = chat_id

    def validate(self, workspace: str | None = None) -> None:
        return None

    def poll(self) -> list[tuple[int, str]]:
        """The notes not read yet, oldest first, each marked as added mid-run."""
        return [
            (number, f"{NOTE_MARK}\n{text}")
            for number, text in self._asks.unread_notes(self._chat_id)
        ]

    def accept(self, offset: int) -> None:
        self._asks.note_read(self._chat_id, offset)
