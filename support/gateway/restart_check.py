"""The note a task continued after a restart starts with, when one step may be half done.

A restart continues a task by itself when the only call that had not returned is
a file write or edit inside the task's folder, whose effect melo can look at
afterwards. The continued turn then starts with this note, as data: the step's
tool and a redacted summary of its arguments, and what to look at before
anything else. It never says whether the step happened.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from localmelo.melo.schema import RESTART_CHECK_HEADER
from localmelo.support.gateway.handover import fenced
from localmelo.support.gateway.park_card import _text
from localmelo.support.sessions.store import StoredEvent

# What the note asks for after the step: written by melo, never by the run.
CHECK_ASK: Final = (
    "localmelo stopped while this step ran, so its effect is unknown: it may have "
    "happened fully, partly or not at all. Before anything else, look at the file "
    "and `git status`, then decide whether the step still has to be done."
)
# The longest summary of the step's arguments the note keeps.
SUMMARY_CHARS: Final = 200


def check_note(tool: str, events: Iterable[StoredEvent], turn: int | None) -> str:
    """The note for the open call of *tool* in *turn*, the cut run's turn.

    The summary is the preview of that turn's one ``tool.started`` of *tool*
    with no ``tool.finished``: the arguments as every stored event has them
    (secrets redacted, a file's content only counted), redacted again and cut
    short. With no turn, or not exactly one such call in it, the note names the
    tool alone: it never describes another call.
    """
    name = _text(tool, 60) or "unknown tool"
    summary = _open_preview(tool, events, turn) if turn is not None else ""
    body = f"{name}: {summary}" if summary else name
    return fenced("step", RESTART_CHECK_HEADER, _text(body, SUMMARY_CHARS), CHECK_ASK)


def _open_preview(tool: str, events: Iterable[StoredEvent], turn: int) -> str:
    started: dict[tuple[str, str], str] = {}
    for stored in events:
        event = stored.event
        data = event.get("data") if isinstance(event, Mapping) else None
        if stored.turn != turn or not isinstance(data, Mapping):
            continue
        if data.get("name") != tool:
            continue
        key = (str(event.get("run_id") or ""), str(data.get("call_id") or ""))
        if event.get("type") == "tool.started":
            preview = data.get("preview")
            started[key] = preview if isinstance(preview, str) else ""
        elif event.get("type") == "tool.finished":
            started.pop(key, None)
    return (
        _text(next(iter(started.values())), SUMMARY_CHARS) if len(started) == 1 else ""
    )


__all__ = ["CHECK_ASK", "check_note"]
