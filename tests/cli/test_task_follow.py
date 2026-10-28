from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Iterator

import httpx
import pytest

from localmelo.support.task_follow import FollowError, follow


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks


def frame(n: int, text: str = "hello") -> bytes:
    data = json.dumps({"id": n, "session_id": "abc", "data": text}, ensure_ascii=False)
    return f"id: {n}\r\ndata: {data}\r\n\r\n".encode()


def event_frame(
    n: int, kind: str, data: dict[str, object], ts: float = 1_700_000_000.0
) -> bytes:
    event = {
        "v": 1,
        "id": n,
        "session_id": "abc",
        "turn": 1,
        "type": kind,
        "run_id": "run",
        "seq": n,
        "ts": ts,
        "data": data,
    }
    payload = json.dumps(event, ensure_ascii=False)
    return f"id: {n}\r\ndata: {payload}\r\n\r\n".encode()


def clock(ts: float = 1_700_000_000.0) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "state,expected", [("done", 0), ("failed", 1), ("cancelled", 2), ("expired", 2)]
)
def test_terminal_catchup_wait_and_snapshot(state: str, expected: int) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/events"):
            data = frame(1, "é\x1b[31m")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Chunks([data[i : i + 1] for i in range(len(data))]),
            )
        return httpx.Response(200, json={"task": {"state": state}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=True, wait=True, out=out) == expected
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    assert records[0]["id"] == 1
    assert records[1]["state"] == state
    assert "not guaranteed" in records[1]["event_completeness"]
    assert requests[1].url.params["follow"] == "false"
    assert "\x1b" not in out.getvalue()


def test_reconnect_dedup_and_terminal_recheck(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("localmelo.support.task_follow.time.sleep", lambda _: None)
    streams = []
    details = iter(["running", "running", "done", "done"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            streams.append(request.headers["last-event-id"])
            payload = frame(1) if len(streams) == 1 else frame(1) + frame(2)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=payload
            )
        return httpx.Response(200, json={"task": {"state": next(details)}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=True, wait=False, out=out) == 0
    assert streams == ["0", "1", "2"]
    assert [
        r["id"] for r in map(json.loads, out.getvalue().splitlines()) if "id" in r
    ] == [1, 2]


@pytest.mark.parametrize(
    "body",
    [b"id: 1\ndata: {}\n\n", b"id: x\ndata: {}\n\n", frame(1)[:-2], b"data: \xff\n\n"],
)
def test_bad_frames_fail(body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body
            )
        return httpx.Response(200, json={"task": {"state": "done"}})

    with client(handler) as c, pytest.raises(FollowError):
        follow(c, "abc", json_mode=False, wait=False, out=io.StringIO())


def test_duplicate_event_within_stream_is_emitted_once() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=frame(1) + frame(1) + frame(2),
            )
        return httpx.Response(200, json={"task": {"state": "done"}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=True, wait=False, out=out) == 0
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [record["id"] for record in records if "id" in record] == [1, 2]


def test_human_lines_show_time_kind_and_short_summary() -> None:
    from localmelo.support.task_follow import _CAVEAT, _emit

    out = io.StringIO()
    _emit(
        out,
        {
            "event": {
                "type": "step.started",
                "ts": 1_700_000_000.0,
                "data": {"step": 1, "attempt": 2},
            }
        },
        False,
    )
    _emit(
        out,
        {
            "event": {
                "type": "tool.started",
                "ts": 1_700_000_000.0,
                "data": {"name": "x\x1b[2J", "preview": "echo hi"},
            }
        },
        False,
    )
    _emit(
        out,
        {
            "event": {
                "type": "tool.finished",
                "ts": 1_700_000_000.0,
                "data": {"name": "shell_exec", "status": "done", "duration_s": 0.4},
            }
        },
        False,
    )
    _emit(
        out,
        {
            "event": {
                "type": "mystery",
                "ts": 1_700_000_000.0,
                "data": {"message": "Caf\xe9\x07"},
            }
        },
        False,
    )
    _emit(out, {"state": "done"}, False)
    _emit(out, {"state": "done", "reconnected": True}, False)
    _emit(out, {"state": "done", "gap": True}, False)
    lines = out.getvalue().splitlines()
    assert lines[0] == f"{clock()}  step.started  step 1, attempt 2"
    assert lines[1] == f"{clock()}  tool.started  x?[2J: echo hi"
    assert lines[2] == f"{clock()}  tool.finished  shell_exec done in 0.4s"
    assert lines[3] == f"{clock()}  mystery  Caf\xe9?"
    assert lines[4] == "Terminal state: done"
    assert lines[5] == f"Terminal state: done. {_CAVEAT}"
    assert lines[6] == f"Terminal state: done. {_CAVEAT}"
    assert chr(27) not in out.getvalue()


def test_human_follow_is_readable_events_then_quiet_terminal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=event_frame(1, "step.started", {"step": 3, "attempt": 1})
                + event_frame(2, "run.finished", {"result": {"status": "completed"}}),
            )
        return httpx.Response(200, json={"task": {"state": "done"}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=False, wait=True, out=out) == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == f"{clock()}  step.started  step 3, attempt 1"
    assert lines[1] == f"{clock()}  run.finished  completed"
    assert lines[2] == "Terminal state: done"
    assert "not guaranteed" not in out.getvalue()
    assert "{" not in out.getvalue()


def test_human_terminal_caveat_after_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("localmelo.support.task_follow.time.sleep", lambda _: None)
    details = iter(["running", "done", "done"])
    streams = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal streams
        if request.url.path.endswith("/events"):
            streams += 1
            if streams == 1:
                raise httpx.ReadTimeout("idle", request=request)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b""
            )
        return httpx.Response(200, json={"task": {"state": next(details)}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=False, wait=True, out=out) == 0
    assert streams == 2
    assert out.getvalue().endswith(
        "Terminal state: done. Event completeness is not guaranteed "
        "(retention and finalization races).\n"
    )


def test_human_terminal_caveat_after_skipped_cursor() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b""
            )
        return httpx.Response(200, json={"task": {"state": "done"}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=False, wait=True, out=out, after=5) == 0
    assert out.getvalue().endswith(
        "Terminal state: done. Event completeness is not guaranteed "
        "(retention and finalization races).\n"
    )


def test_json_terminal_record_never_carries_human_flags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b""
            )
        return httpx.Response(200, json={"task": {"state": "done"}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=True, wait=True, out=out, after=5) == 0
    record = json.loads(out.getvalue())
    assert set(record) == {"v", "command", "state", "event_completeness"}
    assert "reconnected" not in record and "gap" not in record


def test_after_cursor_and_out_of_order_events() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=frame(4) + frame(3),
            )
        return httpx.Response(200, json={"task": {"state": "done"}})

    with client(handler) as c, pytest.raises(FollowError, match="Out-of-order"):
        follow(c, "abc", json_mode=True, wait=False, out=io.StringIO(), after=2)
    assert requests[1].headers["last-event-id"] == "2"


@pytest.mark.parametrize("status", [401, 404, 302, 503])
def test_http_errors_are_sanitized(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="secret payload")

    with client(handler) as c, pytest.raises(FollowError) as error:
        follow(c, "abc", json_mode=False, wait=False, out=io.StringIO())
    assert str(status) in str(error.value) and "secret" not in str(error.value)


def test_transport_failure_never_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("localmelo.support.task_follow.time.sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            raise httpx.ReadError("Bearer secret payload", request=request)
        return httpx.Response(200, json={"task": {"state": "running"}})

    with client(handler) as c, pytest.raises(FollowError) as error:
        follow(c, "abc", json_mode=False, wait=False, out=io.StringIO())
    assert "secret" not in str(error.value)


def test_idle_timeouts_keep_following_until_terminal() -> None:
    details = iter(["running"] * 5 + ["done", "done"])
    streams = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal streams
        if request.url.path.endswith("/events"):
            streams += 1
            if streams <= 5:
                raise httpx.ReadTimeout("idle", request=request)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b""
            )
        return httpx.Response(200, json={"task": {"state": next(details)}})

    out = io.StringIO()
    with client(handler) as c:
        assert follow(c, "abc", json_mode=True, wait=True, out=out) == 0
    assert streams == 6
    assert json.loads(out.getvalue())["state"] == "done"


def test_output_failure_and_interrupt_never_stop() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=frame(1)
            )
        return httpx.Response(200, json={"task": {"state": "done"}})

    class Broken(io.StringIO):
        def write(self, text: str) -> int:
            raise BrokenPipeError

    with client(handler) as c, pytest.raises(FollowError):
        follow(c, "abc", json_mode=True, wait=False, out=Broken())

    class Interrupted(io.StringIO):
        def write(self, text: str) -> int:
            raise KeyboardInterrupt

    with client(handler) as c, pytest.raises(KeyboardInterrupt):
        follow(c, "abc", json_mode=True, wait=False, out=Interrupted())
    assert set(requests) == {"GET"}
