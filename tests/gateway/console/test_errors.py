"""The API v2 error envelope, and the legacy routes it leaves alone."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from localmelo.support.gateway.chore_store import (
    ChoreError,
    ChoreLimitError,
    ChoreNotFoundError,
    ChoreStaleError,
)
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.errors import (
    INTERNAL_ERROR,
    ApiError,
    ApiMiddleware,
)
from localmelo.support.gateway.task_store import (
    TaskConflictError,
    TaskNotFoundError,
    TaskNotPausedError,
    TaskStoreError,
)
from localmelo.support.sessions.store import (
    SessionBusyError,
    SessionStoreError,
    StoreOwnedError,
    UnknownSessionError,
)

from .conftest import open_session, seed_turn


def test_an_unknown_api_route_is_the_envelope(client: TestClient) -> None:
    resp = client.get("/api/no/such/route")
    assert resp.status_code == 404
    assert resp.headers["cache-control"] == "no-store"
    assert resp.json() == {
        "error": {"code": "not_found", "message": "Not found.", "details": {}}
    }


def test_the_api_root_is_the_envelope(client: TestClient) -> None:
    resp = client.get("/api")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_a_bad_query_parameter_is_a_400(client: TestClient) -> None:
    resp = client.get("/api/sessions?limit=999")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "limit" in resp.json()["error"]["message"]
    assert "999" not in resp.json()["error"]["message"]


def test_a_bad_kind_is_a_400(client: TestClient) -> None:
    resp = client.get("/api/sessions?kind=nope")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "kind" in resp.json()["error"]["message"]


def test_an_unhandled_api_error_is_the_500_envelope(console: Console) -> None:
    app = FastAPI()

    @app.get("/api/boom")
    async def boom() -> None:
        raise RuntimeError("a secret value")

    console.mount(app)
    client = TestClient(app, base_url="http://127.0.0.1:8401")
    resp = client.get("/api/boom")
    assert resp.status_code == 500
    assert resp.json() == {
        "error": {
            "code": "internal_error",
            "message": INTERNAL_ERROR,
            "details": {},
        }
    }
    assert "a secret value" not in resp.text


def test_a_known_path_with_the_wrong_method_is_the_envelope(client: TestClient) -> None:
    resp = client.post("/api/health")
    assert resp.status_code == 405
    assert resp.json() == {
        "error": {
            "code": "method_not_allowed",
            "message": "Method not allowed.",
            "details": {},
        }
    }
    assert "GET" in resp.headers["allow"]


@pytest.mark.parametrize("method", ["TRACE", "CONNECT"])
def test_an_unusual_method_on_a_known_path_is_the_envelope(
    client: TestClient, method: str
) -> None:
    resp = client.request(method, "/api/health")
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"
    assert "GET" in resp.headers["allow"]


def test_an_unknown_path_with_any_method_is_404(client: TestClient) -> None:
    assert client.post("/api/no/such/route").status_code == 404
    assert client.post("/api/no/such/route").json()["error"]["code"] == "not_found"


def test_a_legacy_404_keeps_fastapis_body(client: TestClient) -> None:
    resp = client.get("/no/such/route")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_a_legacy_422_keeps_fastapis_body(console: Console) -> None:
    from localmelo.support.gateway import GatewayServer
    from localmelo.support.gateway.console import ConsoleOptions

    from .conftest import make_cfg

    server = GatewayServer(
        make_cfg(),
        console_options=ConsoleOptions(
            store_path=console.store.path,
            builder=console.options.builder,
            probe=console.options.probe,
        ),
    )
    client = TestClient(
        server.app,
        base_url="http://127.0.0.1:8401",
        headers={"Authorization": "Bearer " + server.auth.read_token().decode("ascii")},
    )
    resp = client.post("/v1/agent/run", json=["not", "an", "object"])
    assert resp.status_code == 422
    assert "detail" in resp.json()
    assert "error" not in resp.json()


@pytest.mark.parametrize(
    "path",
    [
        "/api/sessions",
        "/api/sessions/0123456789ab",
        "/api/sessions/0123456789ab/turns",
        "/api/sessions/0123456789ab/events",
        "/api/sessions/0123456789ab/blobs/" + "0" * 64,
        "/api/sessions/0123456789ab/submissions/req-1",
        "/api/submissions/req-1",
    ],
)
async def test_every_store_route_is_503_when_the_store_failed(
    client: TestClient, console: Console, path: str
) -> None:
    console.store._failed = RuntimeError("The disk is gone")  # type: ignore[attr-defined]
    resp = client.get(path)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "store_unavailable"
    assert resp.json()["error"]["message"]


def _app_raising(console: Console, exc: Exception) -> TestClient:
    """The console's app plus two routes that raise *exc* with no wrapper of their own."""
    app = FastAPI()

    @app.get("/api/sessions/{session_id}/raises")
    async def in_a_session(session_id: str) -> None:
        raise exc

    @app.get("/api/raises")
    async def anywhere() -> None:
        raise exc

    console.mount(app)
    return TestClient(app, base_url="http://127.0.0.1:8401")


async def test_a_store_error_nobody_caught_is_the_503_envelope(
    console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    client = _app_raising(console, SessionStoreError("disk I/O error"))

    for path in (f"/api/sessions/{turn.session_id}/raises", "/api/raises"):
        resp = client.get(path)
        assert resp.status_code == 503, path
        assert resp.json()["error"]["code"] == "store_unavailable"
        assert resp.headers["cache-control"] == "no-store"


async def test_a_store_error_answer_never_carries_the_failure_text(
    console: Console, workspace: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A path under the home folder and the database's own words stay in the log."""
    turn = await seed_turn(console.store, workspace)
    detail = f"unable to open {Path.home()}/secret-folder/sessions.db: disk I/O error"
    client = _app_raising(console, SessionStoreError(detail))
    caplog.set_level(logging.WARNING)

    for path in (f"/api/sessions/{turn.session_id}/raises", "/api/raises"):
        resp = client.get(path)
        assert resp.status_code == 503, path
        message = resp.json()["error"]["message"]
        assert message == "The session store is not available.", path
        assert "secret-folder" not in resp.text and "disk I/O" not in resp.text
    assert "secret-folder" in caplog.text


async def test_a_store_error_in_a_session_that_is_gone_is_404(
    console: Console,
) -> None:
    client = _app_raising(console, SessionStoreError("disk I/O error"))

    resp = client.get("/api/sessions/0123456789ab/raises")

    assert resp.status_code == 404
    assert resp.json()["error"]["message"] == "This session was deleted."


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (UnknownSessionError("Unknown session: x"), 404, "not_found"),
        (TaskNotFoundError("No task with id t1"), 404, "not_found"),
        (TaskNotPausedError("task is not paused"), 409, "not_paused"),
        (TaskConflictError("request key reused"), 409, "invalid_transition"),
        (TaskStoreError("not allowed now"), 409, "invalid_transition"),
        (ChoreNotFoundError("No chore c1"), 404, "not_found"),
        (ChoreStaleError("changed since"), 409, "stale_revision"),
        (ChoreLimitError("too many chores"), 409, "chore_limit"),
        (ChoreError("chore trouble"), 503, "store_unavailable"),
        (StoreOwnedError("held by pid 1"), 503, "store_unavailable"),
    ],
)
async def test_a_store_error_nobody_caught_is_answered_by_its_kind(
    console: Console, workspace: str, exc: Exception, status: int, code: str
) -> None:
    """A task, chore or chat that is not there is not a broken store."""
    turn = await seed_turn(console.store, workspace)
    client = _app_raising(console, exc)

    for path in ("/api/raises", f"/api/sessions/{turn.session_id}/raises"):
        resp = client.get(path)
        assert (resp.status_code, resp.json()["error"]["code"]) == (status, code), path
        assert resp.headers["cache-control"] == "no-store"


async def test_a_task_error_says_what_the_task_store_said(console: Console) -> None:
    client = _app_raising(console, TaskNotFoundError("No task with id t1"))

    assert client.get("/api/raises").json()["error"]["message"] == "No task with id t1"


@pytest.mark.parametrize(
    "exc", [SessionStoreError("disk I/O error"), SessionBusyError("x")]
)
def test_a_legacy_route_keeps_its_unhandled_500(
    console: Console, exc: Exception
) -> None:
    """The two handlers answer ``/api``; a legacy route's error stays unhandled."""
    app = FastAPI()

    @app.get("/v1/raises")
    async def legacy() -> None:
        raise exc

    console.mount(app)
    client = TestClient(
        app, base_url="http://127.0.0.1:8401", raise_server_exceptions=False
    )

    resp = client.get("/v1/raises")

    assert resp.status_code == 500
    assert "error" not in resp.text and "session_busy" not in resp.text


async def test_a_busy_session_nobody_caught_is_409(console: Console) -> None:
    client = _app_raising(console, SessionBusyError("running"))

    resp = client.get("/api/sessions/0123456789ab/raises")

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "session_busy"
    assert (
        resp.json()["error"]["message"] == "A turn is already running in this session."
    )


async def test_require_session_returns_the_row_or_404(
    console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)

    assert (await console.require_session(turn.session_id)).id == turn.session_id
    with pytest.raises(ApiError) as gone:
        await console.require_session("0123456789ab")
    assert (gone.value.status_code, gone.value.code) == (404, "not_found")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"a": "x\ud800"}, "a: the text must be valid UTF-8"),
        ({"k\udc00": "x"}, "request: the text must be valid UTF-8"),
        ({"a": {"b\ud801": 1}}, "a: the text must be valid UTF-8"),
        ({"a": [1, [2, "\udfff"]]}, "a.1.1: the text must be valid UTF-8"),
    ],
)
def test_a_text_that_is_not_utf8_is_a_400_before_any_route(
    console: Console, body: dict[str, object], message: str
) -> None:
    app = FastAPI()

    @app.post("/api/echo")
    async def echo() -> None:
        raise AssertionError("the route was reached")

    console.mount(app)
    client = TestClient(app, base_url="http://127.0.0.1:8401")

    resp = client.post(
        "/api/echo",
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert resp.json()["error"]["message"] == message


@pytest.mark.parametrize(
    ("query", "body", "message"),
    [
        (b"q=%ff", b"", "Query string must be valid UTF-8."),
        (b"q=%c0%af", b"", "Query string must be valid UTF-8."),
        (b"q=%ed%a0%80", b"", "Query string must be valid UTF-8."),
        (b"", b'{"a":"\xff"}', "Request body must be valid UTF-8."),
        (b"", b"{nope", "Malformed JSON request body."),
    ],
)
def test_a_query_or_body_that_is_not_utf8_json_is_a_400_before_any_route(
    console: Console, query: bytes, body: bytes, message: str
) -> None:
    app = FastAPI()

    @app.post("/api/echo")
    async def echo() -> None:
        raise AssertionError("the route was reached")

    console.mount(app)
    client = TestClient(app, base_url="http://127.0.0.1:8401")

    resp = client.post(
        f"/api/echo?{query.decode('latin-1')}",
        content=body,
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 400
    assert resp.json()["error"] == {
        "code": "invalid_request",
        "message": message,
        "details": {},
    }


@pytest.mark.parametrize("depth", [sys.getrecursionlimit() + 10, 5_000, 100_000])
def test_a_body_nested_past_the_recursion_limit_is_a_400(
    console: Console, depth: int
) -> None:
    """The parser's or the walk's stack runs out; the answer is still the envelope."""
    app = FastAPI()

    @app.post("/api/echo")
    async def echo() -> None:
        raise AssertionError("the route was reached")

    console.mount(app)
    client = TestClient(
        app, base_url="http://127.0.0.1:8401", raise_server_exceptions=False
    )

    resp = client.post(
        "/api/echo",
        content=b"[" * depth + b"]" * depth,
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "Malformed JSON request body."
    assert resp.headers["cache-control"] == "no-store"


async def test_the_body_is_replayed_once_and_the_servers_disconnect_follows() -> None:
    """The route reads the whole body in one message, then hears the client go."""
    received: list[Message] = []

    async def reads_twice(scope: Scope, receive: Receive, send: Send) -> None:
        received.append(await receive())
        received.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    pieces = [b'{"a"', b":1}"]

    async def receive() -> Message:
        if pieces:
            return {
                "type": "http.request",
                "body": pieces.pop(0),
                "more_body": bool(pieces),
            }
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/echo",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    await ApiMiddleware(reads_twice)(scope, receive, send)

    assert received == [
        {"type": "http.request", "body": b'{"a":1}', "more_body": False},
        {"type": "http.disconnect"},
    ]


def test_a_picture_body_and_a_query_that_is_utf8_reach_the_route(
    console: Console,
) -> None:
    app = FastAPI()

    @app.post("/api/uploads")
    async def upload() -> dict[str, bool]:
        return {"reached": True}

    console.mount(app)
    client = TestClient(app, base_url="http://127.0.0.1:8401")

    resp = client.post(
        "/api/uploads?name=%E4%BD%A0.png",
        content=b"\xff\xfe not json",
        headers={"Content-Type": "image/png"},
    )

    assert resp.status_code == 200


def test_a_store_reason_names_the_home_folder_as_tilde_wherever_it_stands() -> None:
    """The reason a store could not open names its file in the middle of the text."""
    from localmelo.support.sessions import display

    home = str(Path.home())
    reason = f"Cannot open {home}/x/sessions.db: locked ({home})"
    assert display.home_labelled(reason) == "Cannot open ~/x/sessions.db: locked (~)"
    # A folder whose name only starts like the home folder is someone else's.
    assert display.home_labelled(f"{home}2/y") == f"{home}2/y"


async def test_a_chat_that_cannot_be_read_when_it_opens_says_the_fixed_message(
    console: Console, workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a chat's agent answers the fixed store message, never the failure text."""
    from localmelo.support.gateway.console.errors import ApiError

    turn = await seed_turn(console.store, workspace)
    detail = f"unable to open {Path.home()}/secret-folder/sessions.db: disk I/O error"

    async def broken(session_id: str) -> None:
        raise SessionStoreError(detail)

    monkeypatch.setattr(console.store, "get_session", broken)
    with pytest.raises(ApiError) as caught:
        await open_session(console.sessions, turn.session_id)
    assert caught.value.status_code == 503
    assert "secret-folder" not in str(caught.value.message)
    assert caught.value.message == "The session store is not available."
