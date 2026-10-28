from __future__ import annotations

import io
import json
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
