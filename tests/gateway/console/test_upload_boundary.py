"""The edges of ``POST /api/uploads``: sign-in, body limits, the stage and the names.

The route takes raw bytes, so the gateway's guard gives it a body limit and image
types of its own. These tests drive the guard and the real routes with requests
built by hand (an ASGI call with the headers and body chunks a test names), so a
request can have no ``Content-Length``, a path spelled in an odd way or a foreign
``Host``. They need no model; the ones that run ``sips`` are skipped where it is
missing.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from localmelo.melo.schema import ImagePart
from localmelo.support.gateway.auth import GatewayAuth
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console import uploads as uploads_module
from localmelo.support.gateway.console.uploads import PictureStage, clean_name
from localmelo.support.gateway.guard import UPLOAD_BYTES_LIMIT, LoopbackGuard
from localmelo.support.sessions import picture_input
from localmelo.support.sessions.picture_input import Picture

from ...sessions.picture_samples import (
    CAMERA,
    GIF_COMMENT,
    assert_clean,
    gif_extras,
    gif_picture,
    heic_headers,
    webp_picture,
)
from .conftest import PORT
from .test_pictures_in import png

needs_sips = pytest.mark.skipif(shutil.which("sips") is None, reason="macOS sips")

MIB = 1024 * 1024
HOST = f"127.0.0.1:{PORT}"
PNG_8 = png(8, 8)


async def asgi(
    app: Any,
    method: str,
    path: str,
    *,
    headers: Sequence[tuple[str, str]] = (),
    chunks: Sequence[bytes] = (b"",),
    query: str = "",
) -> tuple[int, dict[str, str], bytes]:
    """Call *app* once with a request built by hand; the status, headers and body of its answer.

    *chunks* are the pieces of the body as the server would hand them over, and no
    ``Content-Length`` is added: a test that wants one names it.
    """
    pending = list(chunks)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode("latin-1")) for k, v in headers],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", PORT),
    }
    taken = 0

    async def receive() -> dict[str, Any]:
        nonlocal taken
        if pending:
            taken += 1
            body = pending.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    status = 0
    out_headers: dict[str, str] = {}
    out_body = b""

    async def send(message: dict[str, Any]) -> None:
        nonlocal status, out_body
        if message["type"] == "http.response.start":
            status = message["status"]
            out_headers.update(
                {k.decode().lower(): v.decode() for k, v in message["headers"]}
            )
        elif message["type"] == "http.response.body":
            out_body += message.get("body", b"")

    await app(scope, receive, send)
    out_headers["x-test-chunks-taken"] = str(taken)
    return status, out_headers, out_body


def image_headers(
    *extra: tuple[str, str], content_type: str = "image/png", host: str = HOST
) -> list[tuple[str, str]]:
    return [("host", host), ("content-type", content_type), *extra]


# ── Sign-in, Host, Origin and Fetch-Site ──


@pytest.fixture
def signed_in(app: FastAPI, tmp_path: Path) -> Any:
    """The console's app behind the guard with sign-in on, and the right credentials."""
    auth = GatewayAuth(tmp_path / "auth")
    guard = LoopbackGuard(app, port=PORT, auth=auth)
    bearer = ("authorization", "Bearer " + auth.read_token().decode())
    try:
        yield guard, bearer
    finally:
        auth.close()


async def test_every_way_into_the_upload_route_needs_the_sign_in(
    signed_in: Any, console: Console
) -> None:
    guard, bearer = signed_in
    # With the right credentials a picture is staged, shown and dropped.
    status, _, body = await asgi(
        guard, "POST", "/api/uploads", headers=image_headers(bearer), chunks=[PNG_8]
    )
    assert status == 201, body
    key = body.split(b'"id":"')[1].split(b'"')[0].decode()
    path = f"/api/uploads/{key}"
    assert (await asgi(guard, "GET", path, headers=[("host", HOST), bearer]))[0] == 200
    wrong = ("authorization", "Bearer " + "0" * 64)
    for credentials in ([], [wrong], [("authorization", "Basic abc")]):
        for method, target, chunks in (
            ("POST", "/api/uploads", [PNG_8]),
            ("GET", path, [b""]),
            ("DELETE", path, [b""]),
        ):
            headers = (
                image_headers(*credentials)
                if method == "POST"
                else [("host", HOST), *credentials]
            )
            status, out, _ = await asgi(
                guard, method, target, headers=headers, chunks=chunks
            )
            assert status == 401, (method, credentials)
            assert out.get("www-authenticate") == "Bearer"
    # Nothing was staged by a refused POST and nothing dropped by a refused DELETE.
    assert (await asgi(guard, "GET", path, headers=[("host", HOST), bearer]))[0] == 200
    assert len(console.pictures._items) == 1


@pytest.mark.parametrize(
    ("what", "headers", "status", "code"),
    [
        ("foreign host", [("host", "evil.example:8401")], 421, "misdirected_request"),
        (
            "foreign origin",
            [("host", HOST), ("origin", "http://evil.example")],
            403,
            "foreign_origin",
        ),
        (
            "origin of another local port",
            [("host", HOST), ("origin", "http://127.0.0.1:8402")],
            403,
            "foreign_origin",
        ),
        ("null origin", [("host", HOST), ("origin", "null")], 403, "foreign_origin"),
        (
            "cross-site fetch",
            [("host", HOST), ("sec-fetch-site", "cross-site")],
            403,
            "foreign_origin",
        ),
        (
            "same-site fetch",
            [("host", HOST), ("sec-fetch-site", "same-site")],
            403,
            "foreign_origin",
        ),
    ],
)
@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
async def test_a_foreign_host_origin_or_fetch_site_is_refused_also_with_the_right_credentials(
    signed_in: Any,
    console: Console,
    method: str,
    what: str,
    headers: list[tuple[str, str]],
    status: int,
    code: str,
) -> None:
    guard, bearer = signed_in
    path = "/api/uploads" if method == "POST" else "/api/uploads/" + "0" * 32
    given = [*headers, bearer]
    if method == "POST":
        given.append(("content-type", "image/png"))
    got, _, body = await asgi(guard, method, path, headers=given, chunks=[PNG_8])
    assert got == status, (what, body)
    assert f'"code":"{code}"'.encode() in body.replace(b" ", b"")
    assert console.pictures._items == {}


# Spellings of the path. The guard gives raw bytes and 20 MiB to a POST to
# ``/api/uploads`` and to the aliases that only differ by slashes at either end;
# nothing else gets that.
UPLOAD_ALIASES = ["/api/uploads", "/api/uploads/", "//api/uploads", "///api/uploads/"]
NOT_THE_UPLOAD_ROUTE = [
    "/API/uploads",
    "/api/Uploads",
    "/api/uploads/x",
    "/api/./uploads",
    "/api/uploads%2F",
    "/api/uploads\x00",
    "/api/uploads;x=1",
    "/api/x/../uploads",
    "/api//uploads",
    "/uploads",
    "/api/upload",
]


@pytest.mark.parametrize("path", UPLOAD_ALIASES + NOT_THE_UPLOAD_ROUTE)
async def test_no_spelling_of_the_path_reaches_a_route_without_the_sign_in(
    signed_in: Any, console: Console, path: str
) -> None:
    guard, bearer = signed_in
    status, _, _ = await asgi(
        guard, "POST", path, headers=image_headers(), chunks=[PNG_8]
    )
    assert status == 401
    assert console.pictures._items == {}


@pytest.mark.parametrize("path", UPLOAD_ALIASES)
async def test_only_the_exact_upload_path_takes_a_picture_and_its_aliases_stage_nothing(
    signed_in: Any, console: Console, path: str
) -> None:
    guard, bearer = signed_in
    status, _, body = await asgi(
        guard, "POST", path, headers=image_headers(bearer), chunks=[PNG_8]
    )
    # The guard lets the aliases through with the upload rules (size, image types); the
    # app has no route under their spelling, so they end in "not found".
    exact = path == "/api/uploads"
    assert status == (201 if exact else 404), body
    assert len(console.pictures._items) == (1 if exact else 0)


@pytest.mark.parametrize("path", NOT_THE_UPLOAD_ROUTE)
async def test_any_other_spelling_is_not_the_upload_route_and_stages_nothing(
    signed_in: Any, console: Console, path: str
) -> None:
    guard, bearer = signed_in
    status, _, body = await asgi(
        guard, "POST", path, headers=image_headers(bearer), chunks=[PNG_8]
    )
    # Under /api it gets the ordinary JSON-only rule; elsewhere no route takes a POST.
    assert status in (404, 415), (status, body)
    assert console.pictures._items == {}


# ── Body limits ──


async def test_a_body_without_a_length_is_stopped_at_the_limit_before_the_app_sees_it() -> (
    None
):
    reached: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        reached.append("app")

    guard = LoopbackGuard(app, port=PORT)
    chunk = b"\x00" * MIB
    more = UPLOAD_BYTES_LIMIT // MIB
    status, headers, body = await asgi(
        guard,
        "POST",
        "/api/uploads",
        headers=image_headers(),
        chunks=[chunk] * (more + 20),
    )
    assert status == 413
    assert b'"code":"too_large"' in body.replace(b" ", b"")
    assert reached == []
    # The guard stopped reading when the limit was crossed, not at the end.
    assert int(headers["x-test-chunks-taken"]) == more + 1


async def test_a_body_without_a_length_that_just_fits_goes_through_whole() -> None:
    seen: list[int] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        total = 0
        while True:
            message = await receive()
            total += len(message.get("body", b""))
            if not message.get("more_body"):
                break
        seen.append(total)
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guard = LoopbackGuard(app, port=PORT)
    chunks = [b"\x00" * MIB] * (UPLOAD_BYTES_LIMIT // MIB)
    status, _, _ = await asgi(
        guard, "POST", "/api/uploads", headers=image_headers(), chunks=chunks
    )
    assert status == 204 and seen == [UPLOAD_BYTES_LIMIT]


async def test_the_real_route_refuses_a_chunked_body_over_the_limit_and_stages_nothing(
    app: FastAPI, console: Console
) -> None:
    guard = LoopbackGuard(app, port=PORT)
    chunks = [b"\x89PNG\r\n\x1a\n" + b"\x00" * (MIB - 8)] + [b"\x00" * MIB] * (
        UPLOAD_BYTES_LIMIT // MIB
    )
    status, _, body = await asgi(
        guard, "POST", "/api/uploads", headers=image_headers(), chunks=chunks
    )
    assert status == 413 and b"too_large" in body
    assert console.pictures._items == {}
    # A chunked body of a small picture is taken as a sized one is.
    small = [PNG_8[:20], PNG_8[20:]]
    status, _, body = await asgi(
        guard, "POST", "/api/uploads", headers=image_headers(), chunks=small
    )
    assert status == 201, body


async def test_a_chunked_body_with_another_content_type_is_refused() -> None:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        raise AssertionError("the app was reached")

    status, _, body = await asgi(
        LoopbackGuard(app, port=PORT),
        "POST",
        "/api/uploads",
        headers=image_headers(content_type="application/json"),
        chunks=[PNG_8[:20], PNG_8[20:]],
    )
    assert status == 415 and b"unsupported_media_type" in body


# ── The stage ──


def staged_picture(size: int = 100) -> Picture:
    return Picture(
        ImagePart.from_bytes(b"\x89PNG" + b"\x00" * size, "image/png"), 8, 8, False
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_staged_picture_is_kept_for_its_time_and_then_gone() -> None:
    clock = FakeClock()
    stage = PictureStage(clock)
    staged = stage.put(staged_picture(), "a.png")
    clock.now += uploads_module.TTL_S
    assert stage.get(staged.id) is staged  # at the limit it is still there
    clock.now += 0.5
    assert stage.get(staged.id) is None
    with pytest.raises(Exception) as gone:
        stage.resolve([staged.id])
    assert (
        "picture_gone" in repr(gone.value)
        or getattr(gone.value, "code", "") == "picture_gone"
    )


def test_a_picture_that_is_asked_for_does_not_live_longer() -> None:
    """Showing a thumbnail does not renew the time: only the person's message uses the picture up."""
    clock = FakeClock()
    stage = PictureStage(clock)
    staged = stage.put(staged_picture(), "a.png")
    for _ in range(5):
        clock.now += uploads_module.TTL_S / 4
        stage.get(staged.id)
    assert stage.get(staged.id) is None


def test_the_oldest_pictures_go_first_when_there_are_too_many() -> None:
    clock = FakeClock()
    stage = PictureStage(clock)
    made = []
    for index in range(uploads_module.MAX_STAGED + 3):
        clock.now += 1
        made.append(stage.put(staged_picture(), f"{index}.png"))
    assert [stage.get(item.id) is None for item in made] == [True] * 3 + [False] * (
        uploads_module.MAX_STAGED
    )


def test_the_oldest_pictures_go_first_when_they_hold_too_many_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uploads_module, "MAX_STAGED_BYTES", 10_000)
    clock = FakeClock()
    stage = PictureStage(clock)
    made = []
    for _ in range(5):
        clock.now += 1
        made.append(stage.put(staged_picture(4000), "big.png"))
    # 4004 bytes each: two fit in 10 000, a third does not.
    kept = [item for item in made if stage.get(item.id) is not None]
    assert kept == made[-2:]


def test_ids_are_random_hex_and_a_name_that_is_not_one_is_gone() -> None:
    stage = PictureStage(FakeClock())
    ids = {stage.put(staged_picture(), "a.png").id for _ in range(20)}
    assert len(ids) == 20
    assert all(len(i) == 32 and set(i) <= set("0123456789abcdef") for i in ids)
    for bad in ("../x", "A" * 32, "0" * 31, "0" * 33, "", "0" * 32):
        with pytest.raises(Exception) as gone:
            stage.resolve([bad])
        assert getattr(gone.value, "code", "") == "picture_gone"


def test_a_message_that_starts_uses_its_pictures_up() -> None:
    stage = PictureStage(FakeClock())
    first = stage.put(staged_picture(), "a.png")
    second = stage.put(staged_picture(), "b.png")
    stage.release([first.id])
    assert stage.get(first.id) is None and stage.get(second.id) is second
    assert stage.discard(second.id) is True and stage.discard(second.id) is False


# ── Names ──


@pytest.mark.parametrize(
    ("given", "shown"),
    [
        ("holiday.png", "holiday.png"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\a.png", "a.png"),
        ("a/b/c.png", "c.png"),
        (
            "name\nwith\rbreaks\x00and\x1fcontrols.png",
            "name with breaks and controls.png",
        ),
        ("\x7f\x7f", "picture"),
        ("   ", "picture"),
        ("", "picture"),
        ("/", "picture"),
        ('"><img src=x onerror=alert(1)>.png', '"><img src=x onerror=alert(1)>.png'),
        ("a" * 400 + ".png", "a" * 120),
        ("‮gpj.exe", "‮gpj.exe"),
    ],
)
def test_a_name_is_shortened_to_a_file_name_without_a_path_or_control_characters(
    given: str, shown: str
) -> None:
    assert clean_name(given) == shown


@pytest.mark.parametrize(
    "name",
    [
        "../../etc/passwd",
        "a;rm -rf ~.png",
        "$(touch pwned).png",
        "`id`.png",
        '"><img src=x onerror=alert(1)>.png',
        "a\nb.png",
        "x" * 500,
    ],
)
async def test_a_hostile_name_is_shown_as_text_and_reaches_no_file_or_command(
    app: FastAPI,
    console: Console,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    ran: list[Any] = []
    monkeypatch.setattr(picture_input.subprocess, "run", lambda *a, **k: ran.append(a))
    guard = LoopbackGuard(app, port=PORT)
    from urllib.parse import quote

    status, headers, body = await asgi(
        guard,
        "POST",
        "/api/uploads",
        headers=image_headers(),
        chunks=[PNG_8],
        query="name=" + quote(name, safe=""),
    )
    assert status == 201, body
    import json

    data = json.loads(body)
    assert data["name"] == clean_name(name)
    assert "/" not in data["name"] and "\n" not in data["name"]
    assert data["mime_type"] == "image/png"
    # The picture is served as an image that cannot run, and the name is not in its headers.
    status, headers, shown = await asgi(
        guard, "GET", f"/api/uploads/{data['id']}", headers=[("host", HOST)]
    )
    assert status == 200 and shown == PNG_8
    assert headers["content-type"] == "image/png"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["content-disposition"] == "inline"
    assert data["name"] not in " ".join(headers.values())
    # No tool was run: a small picture needs none, and a name never reaches a command line.
    assert ran == []
    assert not any(tmp_path.glob("**/*pwned*"))


# ── GIF, WebP and HEIC through the route ──


async def upload(guard: Any, raw: bytes, content_type: str) -> tuple[int, Any]:
    import json

    status, _, body = await asgi(
        guard,
        "POST",
        "/api/uploads",
        headers=image_headers(content_type=content_type),
        chunks=[raw],
        query="name=a",
    )
    return status, json.loads(body)


@pytest.mark.parametrize(
    ("raw", "content_type"),
    [
        (gif_picture(20000, 20000, [(8, 8, 0)]), "image/gif"),
        (gif_picture(8, 8, [(8, 8, 0), (20000, 20000, 1)]), "image/gif"),
        (webp_picture(8, 8, canvas=(30000, 30000)), "image/webp"),
        (heic_headers((320, 240), (20000, 20000)), "image/heic"),
        (heic_headers((20000, 20000)), "application/octet-stream"),
    ],
    ids=[
        "gif screen",
        "gif second frame",
        "webp canvas",
        "heic second",
        "heic any type",
    ],
)
async def test_the_route_refuses_too_many_pixels_with_a_sentence_before_any_tool_runs(
    app: FastAPI,
    console: Console,
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    content_type: str,
) -> None:
    def started(*args: object, **kwargs: object) -> object:
        raise AssertionError("a tool was started")

    monkeypatch.setattr(picture_input.shutil, "which", lambda name: "/usr/bin/sips")
    monkeypatch.setattr(picture_input, "_convert", started)
    monkeypatch.setattr(picture_input.subprocess, "run", started)
    status, data = await upload(LoopbackGuard(app, port=PORT), raw, content_type)
    assert status == 413
    assert data["error"]["code"] == "picture_too_many_pixels"
    assert "120" in data["error"]["message"]
    assert console.pictures._items == {}


@needs_sips
@pytest.mark.parametrize(
    ("make", "content_type", "mime", "secret"),
    [
        (
            lambda: gif_picture(1600, 40, [(1600, 40, 0), (1600, 40, 1)], gif_extras()),
            "image/gif",
            "image/png",
            GIF_COMMENT,
        ),
        (
            lambda: webp_picture(1600, 40, metadata=True),
            "image/webp",
            "image/png",
            CAMERA,
        ),
        (
            lambda: (
                Path(__file__).parents[2] / "fixtures" / "pictures" / "camera.heic"
            ).read_bytes(),
            "image/heic",
            "image/jpeg",
            CAMERA,
        ),
    ],
    ids=["gif", "webp", "heic"],
)
async def test_a_gif_webp_or_heic_goes_through_the_route_scaled_and_without_metadata(
    app: FastAPI,
    console: Console,
    make: Callable[[], bytes],
    content_type: str,
    mime: str,
    secret: bytes,
) -> None:
    raw = make()
    assert secret in raw
    guard = LoopbackGuard(app, port=PORT)
    status, data = await upload(guard, raw, content_type)
    assert status == 201, data
    assert (data["mime_type"], data["width"], data["height"]) == (mime, 1280, 32)
    _, _, shown = await asgi(
        guard, "GET", f"/api/uploads/{data['id']}", headers=[("host", HOST)]
    )
    assert secret not in shown
    assert_clean(shown)
    assert len(shown) == data["size_bytes"]
