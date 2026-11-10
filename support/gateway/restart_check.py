"""The note a task continued after a restart starts with, when one step may be half done.

A restart continues a task by itself when the only call that had not returned is
one whose effect stays where melo can look at it afterwards (a file in the task's
folder, a routine command on this Mac). The continued turn then starts with this
note, as data: the step's tool and a redacted summary of its arguments, and what
to look at before anything else. It never says whether the step happened.
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
    "happened fully, partly or not at all. Before anything else, look at the "
    "folder, `git status` and the history, and the file itself, then decide "
    "whether the step still has to be done."
)
# The longest summary of the step's arguments the note keeps.
SUMMARY_CHARS: Final = 200


def check_note(tool: str, events: Iterable[StoredEvent]) -> str:
    """The note for the open call of *tool*, its summary read from *events*.

    The summary is the preview of the newest ``tool.started`` of *tool* that
    has no ``tool.finished``: the arguments as every stored event has them
    (secrets redacted, a file's content only counted), redacted again and cut
    short. With none to read, the note names the tool alone.
    """
    name = _text(tool, 60) or "unknown tool"
    summary = _open_preview(tool, events)
    body = f"{name}: {summary}" if summary else name
    return fenced("step", RESTART_CHECK_HEADER, _text(body, SUMMARY_CHARS), CHECK_ASK)


def _open_preview(tool: str, events: Iterable[StoredEvent]) -> str:
    started: dict[tuple[str, str], str] = {}
    for stored in events:
        event = stored.event
        data = event.get("data") if isinstance(event, Mapping) else None
        if not isinstance(data, Mapping) or data.get("name") != tool:
            continue
        key = (str(event.get("run_id") or ""), str(data.get("call_id") or ""))
        if event.get("type") == "tool.started":
            preview = data.get("preview")
            started[key] = preview if isinstance(preview, str) else ""
        elif event.get("type") == "tool.finished":
            started.pop(key, None)
    return _text(list(started.values())[-1], SUMMARY_CHARS) if started else ""


__all__ = ["CHECK_ASK", "check_note"]
