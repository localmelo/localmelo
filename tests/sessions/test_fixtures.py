"""The recorded session streams: the envelope, pinned keys and clean strings.

``tests/fixtures/record_events.py`` writes these files (section 9 of
``.melo/u2_spec.md``); the renderer tests read them in the next task. This
check stays structural: one JSON object per line, the stored-event
envelope, the data keys EVENT_FIELDS or SESSION_EVENT_FIELDS pins, and no
path or secret of the machine that recorded them. What the streams look
like on screen is the goldens' business.

Every stream starts with ``turn.started`` and ends with
``turn.finished``: no scenario is exempt. ``stale`` has no ``run.finished``
between them because U1's recovery ends its turn, and its ending is the
``turn.finished`` the check expects.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.events import EVENT_FIELDS
from localmelo.support.sessions.store import SESSION_EVENT_FIELDS

EVENTS = Path(__file__).resolve().parents[1] / "fixtures" / "events"

# The scenarios section 9 records for U2a.2, and nothing else.
SCENARIOS = [
    "allowed",
    "answer",
    "attachments",
    "budget_none",
    "budget_reflect",
    "cancelled",
    "crashed",
    "denied",
    "empty_reply",
    "escapes",
    "failed",
    "file_write_overwrite",
    "gate_stopped",
    "history",
    "images",
    "long_output",
    "other_stop",
    "provider_errors",
    "retry_then_error",
    "stale",
    "thinking",
    "tools",
    "tools_again",
    "unreadable_reflection",
    "withdrawn",
    "yolo",
]

ENVELOPE_KEYS = frozenset({"turn", "event"})
EVENT_KEYS = frozenset({"v", "type", "run_id", "seq", "ts", "data"})
# A key, a bearer token or a private key: none belongs in a fixture.
SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9]{8,}|Bearer [A-Za-z0-9._-]{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
# An un-normalized uuid, hex or dashed; an image digest is 64 hex, so a
# 32-hex run inside a longer one is not an id.
ID_RE = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])|"
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
# A path of this machine, or of the recorder's temporary directory.
MACHINE_PATH_RE = re.compile(r"/Users/|/private/var/|/var/folders/")


def records(name: str) -> list[dict[str, Any]]:
    lines = (EVENTS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines, f"{name}.jsonl is empty"
    return [json.loads(line) for line in lines]


def test_the_streams_are_exactly_the_planned_scenarios() -> None:
    assert sorted(p.stem for p in EVENTS.glob("*.jsonl")) == SCENARIOS
    assert sorted(p.name for p in EVENTS.iterdir()) == [
        f"{name}.jsonl" for name in SCENARIOS
    ]


@pytest.mark.parametrize("name", SCENARIOS)
def test_each_line_is_one_envelope_with_the_pinned_keys(name: str) -> None:
    for record in records(name):
        assert set(record) == ENVELOPE_KEYS, record
        assert isinstance(record["turn"], int)
        event = record["event"]
        assert set(event) == EVENT_KEYS, event
        assert event["v"] == 1
        type_ = event["type"]
        if type_ not in EVENT_FIELDS and type_ not in SESSION_EVENT_FIELDS:
            pytest.fail(f"{name}: unknown event type {type_!r}")
        pinned = EVENT_FIELDS.get(type_)
        if pinned is None:
            pinned = SESSION_EVENT_FIELDS[type_]
        assert set(event["data"]) == pinned, (name, type_)


@pytest.mark.parametrize("name", SCENARIOS)
def test_each_stream_starts_and_ends_with_its_turn(name: str) -> None:
    stream = records(name)
    types = [record["event"]["type"] for record in stream]
    assert types[0] == "turn.started", name
    assert types[-1] == "turn.finished", name
    turns = [record["turn"] for record in stream]
    assert turns == sorted(turns), name
    for record in stream:
        event = record["event"]
        assert isinstance(event["run_id"], str)
        assert event["seq"] is None or (
            isinstance(event["seq"], int) and event["seq"] >= 1
        )
        assert isinstance(event["ts"], int | float)
    timestamps = [record["event"]["ts"] for record in stream]
    assert timestamps == sorted(timestamps), name
    assert len(set(timestamps)) == len(timestamps), name


@pytest.mark.parametrize("name", SCENARIOS)
def test_no_stream_holds_a_path_of_this_machine_or_a_secret(
    name: str, home: Path
) -> None:
    text = (EVENTS / f"{name}.jsonl").read_text(encoding="utf-8")
    assert "/Users/" not in text, name
    assert str(home) not in text, name
    assert not MACHINE_PATH_RE.search(text), name
    assert not SECRET_RE.search(text), name


@pytest.mark.parametrize("name", SCENARIOS)
def test_no_stream_holds_an_id_that_was_not_normalized(name: str) -> None:
    text = (EVENTS / f"{name}.jsonl").read_text(encoding="utf-8")
    assert not ID_RE.search(text), name
