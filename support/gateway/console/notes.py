"""Words the person adds to a chat's turn while it runs, as the run's inbox.

The Agent's run polls its inbox at the start of each step and accepts a
message once the request that carries it is built (``inbox.accepted``).
:class:`ChatNotes` is that inbox for a console chat: it reads the chat's notes
from the queued-message store (:class:`~.asks.OwnerAsks`), so a note the run
never read stays queued and runs as the chat's next turn. The model reads each
note as the person's message, marked as added while it worked. A note may carry
pictures: they go to the model with its words (``pictures``), unless the model
is known not to read pictures by then, when the words say so instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from localmelo.melo.schema import ImagePart

    from .asks import OwnerAsks

# The line before each note in the request: the person's words, not a tool's.
NOTE_MARK: Final = "[The user added this while you were working]"


def withheld_line(count: int) -> str:
    """The line that tells a model it was not sent a note's pictures."""
    which = ("this picture", "it") if count == 1 else ("these pictures", "them")
    return (
        f"[You cannot see {which[0]} (this model is not sent pictures); the person "
        f"can. Do not describe {which[1]}; ask the person if it matters.]"
    )


class ChatNotes:
    """The run inbox of one chat (``melo.contracts.inbox.InboxSource``)."""

    def __init__(self, asks: OwnerAsks, chat_id: str) -> None:
        self._asks = asks
        self._chat_id = chat_id

    def validate(self, workspace: str | None = None) -> None:
        return None

    def _withheld(self) -> bool:
        """Whether the model is known not to read pictures now."""
        return self._asks.console.vision.known() is False

    def poll(self) -> list[tuple[int, str]]:
        """The notes not read yet, oldest first, each marked as added mid-run."""
        withheld = self._withheld()
        notes = []
        for number, text in self._asks.unread_notes(self._chat_id):
            words = f"{NOTE_MARK}\n{text}"
            count = len(self._asks.note_images(self._chat_id, number))
            if count and withheld:
                words = f"{words}\n{withheld_line(count)}"
            notes.append((number, words))
        return notes

    def pictures(self, offset: int) -> Sequence[ImagePart]:
        """The pictures of note *offset*, with their data; none for a model that reads none."""
        if self._withheld():
            return []
        return self._asks.note_images(self._chat_id, offset)

    def accept(self, offset: int) -> None:
        self._asks.note_read(self._chat_id, offset)
