"""Follow a task's session without changing the task.

Terminal observations are snapshots, not event-completeness guarantees: the
server has no final-event barrier or retention gap marker. With wait=True,
done/failed/cancelled-or-expired return 0/1/2; otherwise terminal returns 0.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any, TextIO
from urllib.parse import quote

import httpx


class FollowError(Exception):
    """The follow could not be completed; messages never include response bodies."""


_TERMINAL = {"done": 0, "failed": 1, "cancelled": 2, "expired": 2}
_TIMEOUT = httpx.Timeout(20.0, connect=5.0, write=5.0, pool=5.0)
_MAX_FRAME = 1_048_576
_MAX_ID = 2**63 - 1
_CAVEAT = "Event completeness is not guaranteed (retention and finalization races)."


def _checked(response: httpx.Response) -> None:
    if response.status_code != 200:
        raise FollowError(
            f"Gateway returned HTTP {response.status_code}; follow incomplete"
        )


def _detail(client: httpx.Client, path: str) -> str:
    response = client.get(path, timeout=_TIMEOUT, follow_redirects=False)
    _checked(response)
    try:
        task = response.json()["task"]
        state = task["state"]
        if not isinstance(state, str) or not state:
            raise ValueError
        return state
    except (ValueError, KeyError, TypeError):
        raise FollowError("Invalid task detail; follow incomplete") from None


def _emit(out: TextIO, record: dict[str, Any], json_mode: bool) -> None:
    # JSON escaping also renders C0/C1 and terminal escape sequences safe.
    if json_mode:
        line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
    elif "event" in record:
        event = json.dumps(record["event"], ensure_ascii=True, separators=(",", ":"))
        line = f"Event {record['id']}: {event}"
    else:
        line = f"Terminal state: {record['state']}. {_CAVEAT}"
    try:
        written = out.write(line + "\n")
        if written is not None and written != len(line) + 1:
            raise OSError("short write")
        out.flush()
    except (OSError, ValueError, UnicodeError):
        raise FollowError("Output failed; follow incomplete") from None


def _frames(response: httpx.Response, deadline: float) -> Iterator[list[bytes]]:
    """Parse complete SSE frames incrementally, accepting CR, LF and CRLF."""
    buffer = bytearray()
    lines: list[bytes] = []
    size = 0
    skip_lf = False
    for chunk in response.iter_bytes():
        if time.monotonic() > deadline:
            raise FollowError("SSE catch-up timed out; follow incomplete")
        for byte in chunk:
            if skip_lf:
                skip_lf = False
                if byte == 10:
                    continue
            if byte not in (10, 13):
                buffer.append(byte)
                if len(buffer) + size > _MAX_FRAME:
                    raise FollowError("SSE frame too large; follow incomplete")
                continue
            skip_lf = byte == 13
            line = bytes(buffer)
            buffer.clear()
            if line:
                size += len(line) + 1
                if size > _MAX_FRAME:
                    raise FollowError("SSE frame too large; follow incomplete")
                lines.append(line)
            else:
                if lines:
                    yield lines
                lines = []
                size = 0
    if buffer or lines:
        raise FollowError("Incomplete SSE frame; follow incomplete")


def _event(lines: list[bytes], task_id: str) -> tuple[int, dict[str, Any]] | None:
    ids: list[str] = []
    data: list[str] = []
    try:
        for raw in lines:
            line = raw.decode("utf-8", errors="strict")
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "id":
                ids.append(value)
            elif field == "data":
                data.append(value)
        if not ids and not data:
            return None
        if len(ids) != 1 or len(data) != 1 or not re.fullmatch(r"[0-9]{1,19}", ids[0]):
            raise ValueError
        event_id = int(ids[0])
        event = json.loads(data[0])
        if (
            not 0 < event_id <= _MAX_ID
            or not isinstance(event, dict)
            or type(event.get("id")) is not int
            or event["id"] != event_id
            or event.get("session_id") != task_id
        ):
            raise ValueError
        return event_id, event
    except (UnicodeError, ValueError, RecursionError):
        raise FollowError("Invalid SSE frame; follow incomplete") from None


def follow(
    client: httpx.Client,
    task_id: str,
    *,
    json_mode: bool,
    wait: bool,
    out: TextIO,
    after: int = 0,
) -> int:
    """Stream events and reconcile task detail; KeyboardInterrupt propagates."""
    if not re.fullmatch(r"[A-Za-z0-9_:-][A-Za-z0-9_.:-]{0,127}", task_id):
        raise FollowError("Invalid task id")
    encoded = quote(task_id, safe="")
    task_path = f"/api/tasks/{encoded}"
    events_path = f"/api/sessions/{encoded}/events"
    if not 0 <= after <= _MAX_ID:
        raise FollowError("Invalid event cursor")
    cursor = after
    failures = 0
    try:
        state = _detail(client, task_path)
        while True:
            catching_up = state in _TERMINAL
            deadline = time.monotonic() + 30 if catching_up else float("inf")
            last_poll = time.monotonic()
            advanced = False
            stream_floor = cursor
            try:
                with client.stream(
                    "GET",
                    events_path,
                    params={"follow": "false" if catching_up else "true"},
                    headers={
                        "Last-Event-ID": str(cursor),
                        "Accept": "text/event-stream",
                    },
                    timeout=_TIMEOUT,
                    follow_redirects=False,
                ) as response:
                    _checked(response)
                    if (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                        != "text/event-stream"
                    ):
                        raise FollowError("Invalid SSE content type; follow incomplete")
                    for frame in _frames(response, deadline):
                        if time.monotonic() > deadline:
                            raise FollowError(
                                "SSE catch-up timed out; follow incomplete"
                            )
                        item = _event(frame, task_id)
                        if item is not None:
                            event_id, event = item
                            if stream_floor < event_id < cursor:
                                raise FollowError(
                                    "Out-of-order SSE event; follow incomplete"
                                )
                            if event_id > cursor:
                                record = {
                                    "v": 1,
                                    "command": "follow",
                                    "event": event,
                                    "id": event_id,
                                }
                                _emit(out, record, json_mode)
                                cursor = event_id
                                advanced = True
                        if not catching_up and time.monotonic() - last_poll >= 5:
                            state = _detail(client, task_path)
                            last_poll = time.monotonic()
                            if state in _TERMINAL:
                                break
                    else:
                        if catching_up:
                            state = _detail(client, task_path)
                            if state in _TERMINAL:
                                record = {
                                    "v": 1,
                                    "command": "follow",
                                    "state": state,
                                    "event_completeness": _CAVEAT,
                                }
                                _emit(out, record, json_mode)
                                return _TERMINAL[state] if wait else 0
                            continue
                if state in _TERMINAL:
                    failures = 0
                    continue
            except httpx.ReadTimeout:
                if catching_up:
                    raise FollowError(
                        "SSE catch-up transport failed; follow incomplete"
                    ) from None
                # An idle stream is not a failed task; poll state and reconnect.
                failures = 0
                state = _detail(client, task_path)
                continue
            except httpx.TransportError:
                if catching_up:
                    raise FollowError(
                        "SSE catch-up transport failed; follow incomplete"
                    ) from None
            state = _detail(client, task_path)
            if state in _TERMINAL:
                failures = 0
                continue
            failures = 0 if advanced else failures + 1
            if failures >= 3:
                raise FollowError("SSE disconnected repeatedly; follow incomplete")
            time.sleep(min(0.25 * 2**failures, 2.0))
    except httpx.HTTPError:
        raise FollowError("Gateway transport failed; follow incomplete") from None
