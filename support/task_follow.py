"""Follow a task's session without changing the task.

Terminal observations are snapshots, not event-completeness guarantees: the
server has no final-event barrier or retention gap marker. With wait=True,
done/failed/cancelled-or-expired return 0/1/2; otherwise terminal returns 0.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterator, Mapping
from typing import Any, TextIO
from urllib.parse import quote

import httpx

from localmelo.melo.schema import broken_tool_call


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


_SUMMARY_CHARS = 120


def _clean(value: object, limit: int = _SUMMARY_CHARS) -> str:
    """One terminal-safe line from *value*, at most *limit* characters."""
    try:
        text = " ".join(str(value).split())
    except Exception:
        return ""
    text = "".join(c if c.isprintable() else "?" for c in text)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _field(data: Mapping[str, Any], name: str, limit: int = _SUMMARY_CHARS) -> str:
    value = data.get(name)
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return _clean(value, limit)
    return ""


def _clock(ts: object) -> str:
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        value = float(ts)
        if math.isfinite(value):
            try:
                return time.strftime("%H:%M:%S", time.localtime(value))
            except (OSError, OverflowError, ValueError):
                pass
    return "--:--:--"


def _summary(kind: str, data: Mapping[str, Any]) -> str:
    """A short, plain-text summary of one event's data."""
    if kind == "run.started":
        return _field(data, "query")
    if kind == "step.started":
        parts = []
        if step := _field(data, "step"):
            parts.append(f"step {step}")
        if attempt := _field(data, "attempt"):
            parts.append(f"attempt {attempt}")
        return ", ".join(parts)
    if kind == "assistant.message":
        said = data.get("content")
        if isinstance(said, str) and broken_tool_call(said):
            return "a tool call written as text, not run"
        if content := _field(data, "content"):
            return content
        if thinking := _field(data, "thinking"):
            return f"thinking: {thinking}"
        calls = data.get("tool_calls")
        if isinstance(calls, list) and calls:
            return f"{len(calls)} tool call" + ("s" if len(calls) != 1 else "")
        return ""
    if kind == "tool.started":
        name = _field(data, "name", 60)
        preview = _field(data, "preview", 80)
        return f"{name}: {preview}" if name and preview else name
    if kind == "tool.finished":
        base = " ".join(
            part
            for part in (
                _field(data, "name", 60),
                _field(data, "status", 40),
            )
            if part
        )
        duration = data.get("duration_s")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            base = f"{base} in {float(duration):.1f}s".strip()
        if error := _field(data, "error", 80):
            base = f"{base}: {error}".strip(": ")
        return base
    if kind == "note":
        return _field(data, "text") or _field(data, "message")
    if kind == "notice":
        return _field(data, "message") or _field(data, "kind", 60)
    if kind == "memory":
        return " ".join(
            part
            for part in (_field(data, "action", 40), _field(data, "kind", 40))
            if part
        )
    if kind == "recall":
        hits = data.get("hits")
        parts = []
        if isinstance(hits, list):
            parts.append(f"{len(hits)} hit" + ("" if len(hits) == 1 else "s"))
        elif count := _field(data, "hits"):
            parts.append(f"{count} hits")
        if ms := _field(data, "ms"):
            parts.append(f"{ms} ms")
        return ", ".join(parts)
    if kind == "reflection.started":
        attempt = _field(data, "attempt")
        return f"attempt {attempt}" if attempt else ""
    if kind == "reflection":
        return _field(data, "action", 60) or _field(data, "summary")
    if kind == "run.progress":
        return ", ".join(
            part
            for part in (
                _field(data, "phase", 40),
                _field(data, "state", 40),
                _field(data, "step", 20),
            )
            if part
        )
    if kind == "context_budget":
        return _field(data, "kind", 60)
    if kind == "run.finished":
        result = data.get("result")
        if isinstance(result, Mapping):
            status = _field(result, "status", 40)
            error = _field(result, "error", 80)
            return f"{status}: {error}" if status and error else status
        return ""
    if kind == "task.created":
        level = _field(data, "level")
        return f"P{level}" if level else ""
    if kind == "task.level":
        before, after = _field(data, "from", 40), _field(data, "to", 40)
        if before and after:
            return f"P{before} -> P{after}"
        return f"P{after}" if after else ""
    if kind == "task.state":
        before, after = _field(data, "from", 40), _field(data, "to", 40)
        return f"{before} -> {after}" if before and after else before or after
    if kind == "task.resume":
        level = _field(data, "level")
        return f"P{level}" if level else ""
    if kind == "approval.requested":
        return _field(data, "tool_name", 60) or _field(data, "preview", 100)
    if kind == "approval.resolved":
        tool = _field(data, "tool_name", 60)
        outcome = _field(data, "outcome", 40) or _field(data, "origin", 40)
        return f"{tool}: {outcome}" if tool and outcome else tool or outcome
    if kind == "turn.started":
        return _field(data, "user_text", 80) or (
            f"turn {_field(data, 'turn')}" if _field(data, "turn") else ""
        )
    if kind == "turn.finished":
        turn = _field(data, "turn")
        status = _field(data, "status", 40)
        return " ".join(
            part for part in (f"turn {turn}" if turn else "", status) if part
        )
    if kind == "session.title":
        return _field(data, "title", 80)
    before, after = _field(data, "from", 40), _field(data, "to", 40)
    if before or after:
        return f"{before} -> {after}" if before and after else before or after
    for key in ("message", "text", "summary", "reason", "name", "state", "status"):
        if value := _field(data, key):
            return value
    if level := _field(data, "level", 20):
        return f"P{level}"
    return ""


def _human_event(event: object) -> str:
    if not isinstance(event, Mapping):
        return "Event"
    kind = _clean(event.get("type") or "unknown", 60)
    data = event.get("data")
    summary = _summary(kind, data if isinstance(data, Mapping) else {})
    line = f"{_clock(event.get('ts'))}  {kind}"
    return f"{line}  {summary}" if summary else line


def _emit(out: TextIO, record: dict[str, Any], json_mode: bool) -> None:
    # JSON escaping also renders C0/C1 and terminal escape sequences safe.
    if json_mode:
        line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
    elif "event" in record:
        line = _human_event(record["event"])
    else:
        line = f"Terminal state: {record['state']}"
        if record.get("reconnected") or record.get("gap"):
            line += f". {_CAVEAT}"
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
    reconnected = False
    # An --after cursor skips the stored events before it; this output alone
    # is not a complete record, and the server exposes no retention floor.
    gap = after > 0
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
                                if not json_mode:
                                    record["reconnected"] = reconnected
                                    record["gap"] = gap
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
                reconnected = True
                failures = 0
                state = _detail(client, task_path)
                continue
            except httpx.TransportError:
                if catching_up:
                    raise FollowError(
                        "SSE catch-up transport failed; follow incomplete"
                    ) from None
            reconnected = True
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
