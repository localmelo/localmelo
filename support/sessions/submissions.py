"""Client request ids: their grammar, the submission errors and their rules.

A web or phone client names each message it sends with a
``client_request_id`` before the gateway accepts it. The store records an
accepted message in the transaction that creates its turn, and a message
withdrawn before acceptance as a withdrawal, so a client that lost a
response can ask what became of the message and retry or withdraw it
without running a turn twice (see :class:`SessionStore.begin_turn`).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from localmelo.support.sessions.store import SubmissionInfo, TurnInfo

# The key's grammar, matched with fullmatch: 1 to 128 characters, which
# covers crypto.randomUUID() and "req-" plus 32 hex.
CLIENT_REQUEST_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

# The error of a withdrawal: a message cancelled before it was accepted.
WITHDRAWN_ERROR: Final = "Request cancelled before acceptance."

_KEY_ERROR: Final = (
    "client_request_id must be 1 to 128 characters: letters, digits and "
    ". _ : -, starting with a letter or digit"
)


class SubmissionError(Exception):
    """A client request id that cannot start a turn.

    Not a ``SessionStoreError``: a surface maps these to their own answers
    (a repeat to the stored turn, the others to a conflict).
    """


class SubmissionRepeated(SubmissionError):  # noqa: N818 – the spec's name
    """The key was accepted already; ``turn`` is the turn it started."""

    def __init__(self, turn: TurnInfo) -> None:
        super().__init__(
            f"Request {turn.client_request_id} was accepted as turn {turn.turn} "
            f"of session {turn.session_id}"
        )
        self.turn = turn


class SubmissionWithdrawn(SubmissionError):  # noqa: N818 – the spec's name
    """The key was withdrawn before acceptance; ``submission`` is the withdrawal."""

    def __init__(self, submission: SubmissionInfo) -> None:
        super().__init__(
            f"Request {submission.client_request_id} was cancelled before it started"
        )
        self.submission = submission


class SubmissionConflict(SubmissionError):  # noqa: N818 – the spec's name
    """The key names another message; ``submission`` is the stored one."""

    def __init__(self, submission: SubmissionInfo) -> None:
        super().__init__(
            f"Request {submission.client_request_id} was already used for a "
            "different message"
        )
        self.submission = submission


def check_client_request_id(client_request_id: object) -> str:
    """Return *client_request_id*, or raise ``ValueError`` when it breaks the grammar."""
    if not isinstance(client_request_id, str) or not CLIENT_REQUEST_ID_RE.fullmatch(
        client_request_id
    ):
        raise ValueError(_KEY_ERROR)
    return client_request_id


def text_sha256(user_text: str) -> str:
    """The SHA-256 hex digest of *user_text* in UTF-8, as submissions store it.

    Lone surrogates, which JSON can carry, are encoded as they are, so any
    ``str`` has a digest.
    """
    return hashlib.sha256(user_text.encode("utf-8", "surrogatepass")).hexdigest()


def check_submission(info: SubmissionInfo | None, user_text: str) -> None:
    """Apply ``begin_turn``'s key rules to the stored submission *info*.

    ``None`` (an unknown key) returns. A stored text other than
    *user_text* raises :class:`SubmissionConflict`, for an accepted and for
    a withdrawn submission alike; then a withdrawal raises
    :class:`SubmissionWithdrawn`, and an accepted submission
    :class:`SubmissionRepeated` with its turn.
    """
    if info is None:
        return
    if info.text_sha256 != text_sha256(user_text):
        raise SubmissionConflict(info)
    if info.state == "withdrawn" or info.turn is None:
        raise SubmissionWithdrawn(info)
    raise SubmissionRepeated(info.turn)
